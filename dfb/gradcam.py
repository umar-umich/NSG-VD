
import cv2
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def showcam(model, save_dir, keyframe, pred):
    
    key_frame_tensor = (torch.tensor(keyframe, dtype=torch.float32).permute(2, 0, 1) / 255.0).clone()
    target_layers=[]
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            target_layers.append(layer)
            
    targets = [ClassifierOutputTarget(pred)]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers[:-3])
    result=cam(input_tensor=key_frame_tensor.unsqueeze(0),targets=targets)
    saliance_map=result[0, :]
    saliance_map = 1 - saliance_map
    visualization = show_cam_on_image(keyframe/255.0, 
                                          saliance_map, use_rgb=True, image_weight=0.76,
                                          colormap=cv2.COLORMAP_JET)
    pred = 'real' if pred==0 else 'fake'
    cv2.imwrite(f'{save_dir}/gradcam_{pred}.png', visualization)
    cv2.imwrite(f'{save_dir}/keyframe_{pred}.png', keyframe)
    
    # return visualization
