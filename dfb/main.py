import argparse
import torch
from torchvision import transforms as T
import yaml
import os
import sys
import cv2
import torch
from torch.nn import Softmax
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
sys.path.append("./../..")

# from utils import get_top_free_gpus
from preprocess import facecrop
from detectors import DETECTOR
from base import AbstractDetector
from pathlib import Path
from transformers import CLIPProcessor

class DFBDetector(AbstractDetector):
    def __init__(self, video_path=None):
        super().__init__()
        if video_path is not None:
            self.video_path = video_path

            self.face_crops, self.face_frames, self.img_frames = facecrop(self.video_path)
            self.face_crops += [self.face_crops[-1]] if len(self.face_crops) % 2 != 0 else []
            print(f'Extracted {len(self.face_crops)} face crops from the video.')

            self.transform_dfb = T.Compose([
                    T.ToTensor(),
                    T.Resize((256, 256)),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
            self.transform_vit = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        
    def load_model(self, device):
        if self.config['model_name'] in ['xception', 'efficientnetb4', 'ffd', 'core', 'ucf']:
            model_name = self.config['model_name']
            model_class = DETECTOR[model_name]
            self.model = model_class(self.config)
            ckpt = torch.load(f'{base_path}/pretrained/{model_name}_best.pth', map_location=device)

            if isinstance(ckpt, dict):
                # Regular state_dict
                self.model.load_state_dict(ckpt, strict=True)
                self.model.to(device)
            else:
                # TorchScript model - shouldn't happen here
                print(f"[Error] TorchScript model provided instead of state_dict for {self.config['model_name']}")
                return

        elif self.config['model_name'] == 'clip_vit':
            self.model = torch.jit.load(f'{base_path}/pretrained/clip_vit.torchscript', map_location=device)
        else:
            print(f"Invalid model! {self.config['model_name']}")

    def get_predictions(self, model_name, uploaded_file_name, device_id=1):
        device = torch.device('cuda', device_id)
        with open(f'{base_path}/config/detector/{model_name}.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

        self.load_model(device)
        self.model.eval()
        
        if model_name in ['xception', 'efficientnetb4', 'ffd', 'core']:
            batch_faces = torch.stack([self.transform_dfb(face) for face in self.face_crops])
            batch_faces = batch_faces.to(device)

            self.data_dict = {'image': batch_faces,'frame': self.face_frames}

            with torch.no_grad():
                output = self.model(self.data_dict)
                self.output = output['prob']
            
            self.avg_output = self.output.mean(dim=0)
  
        elif model_name == 'clip_vit':
            batch_faces = torch.stack([self.transform_vit(images=face, return_tensors="pt")["pixel_values"][0] for face in self.face_crops])
            self.data_dict = batch_faces.to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                output = self.model(self.data_dict)
                self.output = Softmax(dim=1)(output)

            self.avg_output = self.output.mean(dim=0)
        else:
            print('Invalid model!!!!!!!!!!!')
        self.pred = torch.max(self.avg_output, dim=0).indices        
        self.score = torch.max(self.output[:,self.pred]).item()
        prediction_label = 'fake' if self.pred == 1 else 'real'
        return {prediction_label: round(self.score,2)}

    def generate_outputs(self, pred, uploaded_file_name, model_name, device_id=0):
        device = torch.device('cuda', device_id)
        with open(f'{base_path}/config/detector/{model_name}.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

        # batch_faces = torch.stack([self.transform_dfb(face) for face in self.face_crops])
        # batch_faces = batch_faces.to(device)

        self.data_dict = {'frame': self.face_frames}
        
        frame = {}
        frame['orig'] = self.data_dict['frame']
        frame['conf'], frame['ind'] = torch.max(self.output, dim=1)

        best_index = max(
            (i for i, c in enumerate(frame['ind']) if c == pred),
            key=lambda i: frame['conf'][i],
            default=-1
            )
    
        keyframe = frame['orig'][best_index]

        if 'ffd' in self.config['model_name']:
            save_dir = f"{Path(base_path).parent}/outputs/{uploaded_file_name}"
            print(f'Saving video outptus at {save_dir}')
            os.makedirs(save_dir, exist_ok=True)

            from detectors.ffd_detector_grad import FFDDetector as FFDCam

            modelcam = FFDCam(self.config).to(device)

            model_name = self.config['model_name']
            ckpt = torch.load(f'{base_path}/pretrained/{model_name}_best.pth', map_location=device)
            modelcam.load_state_dict(ckpt, strict=True)

            key_frame_tensor = (torch.tensor(keyframe, dtype=torch.float32).permute(2, 0, 1) / 255.0).clone()
            target_layers=[]
            for name, layer in modelcam.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    target_layers.append(layer)
                    
            targets = [ClassifierOutputTarget(pred)]
            cam = GradCAMPlusPlus(model=modelcam, target_layers=target_layers[:-3])
            print("Shape of key_frame_tensor:",key_frame_tensor.shape)
            result=cam(input_tensor=key_frame_tensor.unsqueeze(0),targets=targets)
            saliance_map=result[0, :]
            saliance_map = 1 - saliance_map
            visualization = show_cam_on_image(keyframe/255.0, 
                                                saliance_map, use_rgb=True, image_weight=0.76,
                                                colormap=cv2.COLORMAP_JET)
            pred_label = 'real' if pred==0 else 'fake'
            cv2.imwrite(f'{save_dir}/gradcam_{pred_label}.png', visualization)
            cv2.imwrite(f'{save_dir}/keyframe_{pred_label}.png', keyframe)
            print("FFD Outputs generated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", "-v", default = '', type=str, help="File Path", required=True)
    args = parser.parse_args()

    dfb = DFBDetector(video_path=args.video_path)
    # gpu_ids = get_top_free_gpus(4)
    file_name = args.video_path.split('/')[-1].split('.mp4')[0]

    gpu_ids = [1, 2]
    results = {}
    results['ffd'] = dfb.get_predictions(model_name='ffd', uploaded_file_name=file_name)
    dfb.generate_outputs(1, file_name, 'ffd')

    results['clip_vit'] = dfb.get_predictions(model_name='clip_vit', uploaded_file_name=file_name, device_id=gpu_ids[0])
    # results['vit'] = dfb.pred

    results['efficientnetb4'] = dfb.get_predictions(model_name='efficientnetb4', uploaded_file_name=file_name, device_id=gpu_ids[1])
    # results['efficientnetb4'] = dfb.pred


    print(results)

if __name__ == '__main__':
    main()