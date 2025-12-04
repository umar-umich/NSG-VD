"""
NSG-VD (Neural Score Gradient - Velocity Distribution) Detector
Batch Evaluation for Real and Fake Video Folders - MEMORY EFFICIENT VERSION

Based on: https://github.com/ZSHsh98/NSG-VD
Paper: "Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection" (NeurIPS 2025)

EVALUATION MODE: Calculates only Accuracy and Confusion Matrix
MEMORY EFFICIENT: Uses chunked reference processing for large reference sets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
import sys
import yaml
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import json
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, ".."))

from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator, TALL_SWIN
from utils.train_utils import MMD_batch2


class NSGVDDetector:
    """
    NSG-VD Detector with memory-efficient chunked reference processing
    """
    
    def __init__(self, config_path=None, device_id=0):
        self.device = torch.device(
            f'cuda:{device_id}' if torch.cuda.is_available() and device_id >= 0 else 'cpu'
        )
        
        if config_path is None:
            config_path = './config/nsgvd_detector.yaml'
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.score_fn = None
        self.feature_ref = None
        self.ref_data = None
        
        # Memory management settings
        self.chunk_size = self.config.get('reference_chunk_size', 500)
        self.use_chunked_processing = self.config.get('use_chunked_processing', True)
        
        print(f"NSG-VD Detector initialized on {self.device}")
        print(f"Config num_frames: {self.config.get('num_frames', 'NOT SET')}")
        print(f"Chunked processing: {self.use_chunked_processing} (chunk_size={self.chunk_size})")
        
    def load_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.config.get('checkpoint_path')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from {checkpoint_path}")
        
        discriminator = TALL_SWIN(pretrained=True) # SingleSwinBlockDiscriminator(
        #     num_features=self.config.get('feature_dim', 300)
        # )
        
        self.model = deep_MMD(
            discriminator=discriminator,
            sigma=self.config.get('sigma', 1.0),
            sigma0=self.config.get('sigma0', 0.1),
            epsilon=self.config.get('epsilon', 1e-10),
            img_size=self.config.get('img_size', 224),
            is_yy_zero=self.config.get('is_yy_zero', True),
            is_smooth=self.config.get('is_smooth', True)
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully (MMD-{'MP' if self.model.is_yy_zero else 'D'})")
        
    def load_score_function(self):
        from data.utils import get_score_fn
        
        score_config_path = self.config.get('score_config_path', './libs/eps_ad/imagnet.yml')
        score_args_path = self.config.get('score_args_path', './libs/eps_ad/args.yml')
        
        print(f"Loading score function...")
        self.score_fn = get_score_fn(self.device, score_args_path, score_config_path)
        print("✓ Score function loaded")
        
    def load_reference_features(self, ref_features_path=None, ref_data_path=None):
        if ref_features_path is None:
            ref_features_path = self.config.get('ref_features_path')
        if ref_data_path is None:
            ref_data_path = self.config.get('ref_data_path')
        
        if not os.path.exists(ref_features_path):
            raise FileNotFoundError(f"Reference features not found: {ref_features_path}")
        if not os.path.exists(ref_data_path):
            raise FileNotFoundError(f"Reference data not found: {ref_data_path}")
        
        print(f"Loading reference features...")
        self.feature_ref = torch.load(ref_features_path, map_location='cpu', weights_only=False)
        self.ref_data = torch.load(ref_data_path, map_location='cpu', weights_only=False)
        
        expected_size = self.config.get('num_frames', 8) * 3 * 224 * 224
        actual_size = self.ref_data.shape[1]
        
        print(f"✓ Reference loaded: {self.feature_ref.shape[0]} samples")
        print(f"  Reference data shape: {self.ref_data.shape}")
        print(f"  Expected size per sample: {expected_size}")
        print(f"  Actual size per sample: {actual_size}")
        
        if self.use_chunked_processing:
            num_chunks = (self.feature_ref.shape[0] + self.chunk_size - 1) // self.chunk_size
            print(f"  Using chunked processing: {num_chunks} chunks of size {self.chunk_size}")
        
        if actual_size != expected_size:
            calculated_frames = actual_size // (3 * 224 * 224)
            print(f"  ⚠️  WARNING: Config num_frames={self.config.get('num_frames')} but reference has {calculated_frames} frames")
        
    @torch.no_grad()
    def extract_nsg_features(self, video_frames):
        if isinstance(video_frames, list):
            from torchvision import transforms as T
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            video_frames = torch.stack([transform(img) for img in video_frames])
        
        if video_frames.dim() == 5:
            video_frames = video_frames.squeeze(0)
        
        T_frames, C, H, W = video_frames.shape
        video_frames = video_frames.to(self.device)
        
        resolution_size = self.config.get('resolution_size', 224)
        if H != resolution_size or W != resolution_size:
            video_frames_resized = F.interpolate(
                video_frames, 
                size=(resolution_size, resolution_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            video_frames_resized = video_frames
        
        normalized_frames = 2 * video_frames_resized - 1
        
        diffuse_steps = self.config.get('diffuse_steps', 5)
        t_value = (diffuse_steps + 1) / 1000
        curr_t = torch.tensor(t_value, device=self.device).expand(T_frames)
        
        score = self.score_fn(normalized_frames, curr_t)
        score, _ = torch.split(score, score.shape[1] // 2, dim=1)
        
        if resolution_size != H or resolution_size != W:
            score = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=False)
        
        velocity = self._compute_nsg_velocity(score, video_frames)
        
        return score, velocity
    
    def _compute_nsg_velocity(self, score, frames, epsilon=1e-10):
        T, C, H, W = score.shape
        
        pixel_diff = torch.zeros_like(score)
        
        for t in range(1, T - 1):
            pixel_diff[t] = (frames[t + 1] - frames[t - 1]) / 2
        
        pixel_diff[0] = frames[1] - frames[0]
        pixel_diff[-1] = frames[-1] - frames[-2]
        
        denominator = (score * pixel_diff).sum(dim=(1, 2, 3)).view(T, 1, 1, 1) + epsilon
        velocity = score / denominator
        
        return velocity
    
    def _compute_mmd_chunked(self, feature_test, features_flat):
        """
        Compute MMD score using chunked reference processing to save memory
        
        Args:
            feature_test: High-level features from test sample (1, feature_dim)
            features_flat: Flattened raw features from test sample (1, raw_feature_dim)
        
        Returns:
            Average MMD score across all chunks
        """
        num_refs = self.feature_ref.shape[0]
        num_chunks = (num_refs + self.chunk_size - 1) // self.chunk_size
        
        mmd_scores = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, num_refs)
            
            # Load chunk to device
            feature_ref_chunk = self.feature_ref[start_idx:end_idx].to(self.device)
            ref_data_chunk = self.ref_data[start_idx:end_idx].to(self.device)
            
            # Compute MMD for this chunk
            combined_features = torch.cat([feature_ref_chunk, feature_test], dim=0)
            combined_data = torch.cat([ref_data_chunk, features_flat], dim=0)
            
            mmd_score = MMD_batch2(
                combined_features,
                feature_ref_chunk.shape[0],
                combined_data,
                self.model.sigma,
                self.model.sigma0_u,
                self.model.ep,
                is_smooth=self.model.is_smooth
            )
            
            mmd_scores.append(mmd_score.cpu().item())
            
            # Free GPU memory
            del feature_ref_chunk, ref_data_chunk, combined_features, combined_data
            torch.cuda.empty_cache()
        
        # Average MMD scores across chunks
        avg_mmd_score = np.mean(mmd_scores)
        
        return avg_mmd_score
    
    @torch.no_grad()
    def predict(self, video_path, feature_type='velocity'):
        from preprocess import extract_and_preprocess_frames
        
        num_frames = self.config.get('num_frames', 8)
        
        try:
            video_frames = extract_and_preprocess_frames(
                video_path, 
                num_frames=num_frames
            )
        except Exception as e:
            raise ValueError(f"Frame extraction failed: {str(e)}")
        
        if video_frames.shape[0] != num_frames:
            raise ValueError(
                f"Extracted {video_frames.shape[0]} frames but config expects {num_frames} frames. "
                f"Video may be too short or corrupted."
            )
        
        score, velocity = self.extract_nsg_features(video_frames)
        
        if feature_type == 'velocity':
            features = velocity
        elif feature_type == 'score':
            features = score
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        features = features.unsqueeze(0)
        
        features_flat = features.view(1, -1)
        expected_size = self.ref_data.shape[1]
        actual_size = features_flat.shape[1]
        
        if actual_size != expected_size:
            expected_frames = expected_size // (3 * 224 * 224)
            actual_frames = actual_size // (3 * 224 * 224)
            raise ValueError(
                f"Feature dimension mismatch:\n"
                f"  Video has {actual_frames} frames (size: {actual_size})\n"
                f"  Reference expects {expected_frames} frames (size: {expected_size})\n"
                f"  Config num_frames: {num_frames}\n"
                f"  Solution: Update config num_frames to {expected_frames} or regenerate reference with num_frames={num_frames}"
            )
        
        try:
            _, feature_test = self.model.net(features, out_feature=True)
        except RuntimeError as e:
            if "shape" in str(e).lower() or "size" in str(e).lower():
                raise ValueError(
                    f"Model forward pass failed with shape error:\n"
                    f"  Input shape: {features.shape}\n"
                    f"  Flattened size: {features_flat.shape[1]}\n"
                    f"  Expected size: {expected_size}\n"
                    f"  Original error: {str(e)}"
                )
            raise
        
        if feature_test.shape[1] != self.feature_ref.shape[1]:
            raise ValueError(
                f"High-level feature mismatch: got {feature_test.shape[1]} features "
                f"but reference has {self.feature_ref.shape[1]} features"
            )
        
        # Use chunked processing if enabled and reference is large
        if self.use_chunked_processing and self.feature_ref.shape[0] > self.chunk_size:
            mmd_score = self._compute_mmd_chunked(feature_test, features_flat)
        else:
            # Original single-pass computation
            feature_ref_device = self.feature_ref.to(self.device)
            ref_data_device = self.ref_data.to(self.device)
            
            mmd_score = MMD_batch2(
                torch.cat([feature_ref_device, feature_test], dim=0),
                feature_ref_device.shape[0],
                torch.cat([ref_data_device, features_flat], dim=0),
                self.model.sigma,
                self.model.sigma0_u,
                self.model.ep,
                is_smooth=self.model.is_smooth
            )
            
            mmd_score = mmd_score.cpu().item()
        
        threshold = self.config.get('threshold', 1.0)
        prediction = 'fake' if mmd_score > threshold else 'real'
        confidence = abs(mmd_score - threshold)
        
        return {
            'prediction': prediction,
            'mmd_score': round(mmd_score, 4),
            'confidence': round(confidence, 4)
        }
    
    def get_predictions(self, video_path, feature_type='velocity'):
        return self.predict(video_path, feature_type=feature_type)


def collect_videos_from_directory(directory: str, max_videos: int = None):
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.webm']
    video_paths = []
    
    dir_path = Path(directory)
    for ext in video_extensions:
        video_paths.extend(dir_path.rglob(ext))
    
    video_paths = sorted(video_paths)
    
    if max_videos is not None:
        video_paths = video_paths[:max_videos]
    
    return video_paths


def evaluate_detector(
    detector: NSGVDDetector,
    real_videos_dir: str,
    fake_videos_dir: str,
    output_prefix: str = 'nsgvd_evaluation',
    feature_type: str = 'velocity',
    max_videos_per_class: int = None
):
    print("="*70)
    print("NSG-VD BATCH EVALUATION (MEMORY EFFICIENT)")
    print("="*70)
    
    print(f"\nCollecting real videos from: {real_videos_dir}")
    real_videos = collect_videos_from_directory(real_videos_dir, max_videos_per_class)
    print(f"✓ Found {len(real_videos)} real videos")
    
    print(f"\nCollecting fake videos from: {fake_videos_dir}")
    fake_videos = collect_videos_from_directory(fake_videos_dir, max_videos_per_class)
    print(f"✓ Found {len(fake_videos)} fake videos")
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        raise ValueError("Need both real and fake videos for evaluation!")
    
    print(f"\nProcessing {len(real_videos) + len(fake_videos)} videos...")
    all_results = []
    skipped_videos = {
        'frame_mismatch': [],
        'extraction_failed': [],
        'other_errors': []
    }
    
    for video_path in tqdm(real_videos, desc="Real videos"):
        try:
            result = detector.predict(str(video_path), feature_type=feature_type)
            
            all_results.append({
                'video_name': video_path.name,
                'video_path': str(video_path),
                'true_label': 'real',
                'predicted_label': result['prediction'],
                'mmd_score': result['mmd_score'],
                'confidence': result['confidence'],
                'correct': result['prediction'] == 'real'
            })
        except ValueError as e:
            error_msg = str(e)
            if 'dimension mismatch' in error_msg.lower() or 'frame' in error_msg.lower():
                if 'extraction failed' in error_msg.lower():
                    skipped_videos['extraction_failed'].append({
                        'video': video_path.name,
                        'label': 'real',
                        'reason': error_msg
                    })
                else:
                    skipped_videos['frame_mismatch'].append({
                        'video': video_path.name,
                        'label': 'real',
                        'reason': error_msg
                    })
            else:
                skipped_videos['other_errors'].append({
                    'video': video_path.name,
                    'label': 'real',
                    'reason': error_msg
                })
        except Exception as e:
            skipped_videos['other_errors'].append({
                'video': video_path.name,
                'label': 'real',
                'reason': str(e)
            })
    
    for video_path in tqdm(fake_videos, desc="Fake videos"):
        try:
            result = detector.predict(str(video_path), feature_type=feature_type)
            
            all_results.append({
                'video_name': video_path.name,
                'video_path': str(video_path),
                'true_label': 'fake',
                'predicted_label': result['prediction'],
                'mmd_score': result['mmd_score'],
                'confidence': result['confidence'],
                'correct': result['prediction'] == 'fake'
            })
        except ValueError as e:
            error_msg = str(e)
            if 'dimension mismatch' in error_msg.lower() or 'frame' in error_msg.lower():
                if 'extraction failed' in error_msg.lower():
                    skipped_videos['extraction_failed'].append({
                        'video': video_path.name,
                        'label': 'fake',
                        'reason': error_msg
                    })
                else:
                    skipped_videos['frame_mismatch'].append({
                        'video': video_path.name,
                        'label': 'fake',
                        'reason': error_msg
                    })
            else:
                skipped_videos['other_errors'].append({
                    'video': video_path.name,
                    'label': 'fake',
                    'reason': error_msg
                })
        except Exception as e:
            skipped_videos['other_errors'].append({
                'video': video_path.name,
                'label': 'fake',
                'reason': str(e)
            })
    
    df = pd.DataFrame(all_results)
    
    output_dir = './evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f'{output_prefix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results to: {csv_path}")
    
    total_skipped = (len(skipped_videos['frame_mismatch']) + 
                     len(skipped_videos['extraction_failed']) + 
                     len(skipped_videos['other_errors']))
    
    if total_skipped > 0:
        skipped_path = os.path.join(output_dir, f'{output_prefix}_skipped.json')
        with open(skipped_path, 'w') as f:
            json.dump(skipped_videos, f, indent=2)
        print(f"✓ Saved skipped videos report to: {skipped_path}")
    
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    if total_skipped > 0:
        print(f"\n⚠️  SKIPPED VIDEOS SUMMARY:")
        print(f"  Frame/dimension mismatch: {len(skipped_videos['frame_mismatch'])} videos")
        print(f"  Extraction failures: {len(skipped_videos['extraction_failed'])} videos")
        print(f"  Other errors: {len(skipped_videos['other_errors'])} videos")
        print(f"  Total skipped: {total_skipped} videos")
    
    if len(df) == 0:
        print("\n❌ ERROR: No videos were successfully processed!")
        return {}
    
    valid_df = df
    
    if len(valid_df) == 0:
        print("ERROR: No valid predictions to evaluate!")
        return {}
    
    y_true = valid_df['true_label'].map({'real': 0, 'fake': 1}).values
    y_pred = valid_df['predicted_label'].map({'real': 0, 'fake': 1}).values
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
        },
        'confusion_matrix': {
            'True Negatives (TN)': int(tn),
            'False Positives (FP)': int(fp),
            'False Negatives (FN)': int(fn),
            'True Positives (TP)': int(tp),
            'matrix': cm.tolist()
        },
        'dataset_stats': {
            'total_videos': len(real_videos) + len(fake_videos),
            'successfully_processed': len(valid_df),
            'skipped_frame_mismatch': len(skipped_videos['frame_mismatch']),
            'skipped_extraction_failed': len(skipped_videos['extraction_failed']),
            'skipped_other_errors': len(skipped_videos['other_errors']),
            'real_videos': len(real_videos),
            'fake_videos': len(fake_videos),
            'real_processed': len([r for r in all_results if r['true_label'] == 'real']),
            'fake_processed': len([r for r in all_results if r['true_label'] == 'fake']),
        },
        'skipped_videos_summary': {
            'frame_mismatch': skipped_videos['frame_mismatch'],
            'extraction_failed': skipped_videos['extraction_failed'],
            'other_errors': skipped_videos['other_errors']
        }
    }
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n[OVERALL PERFORMANCE]")
    print(f"  Accuracy: {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
    
    print("\n[CONFUSION MATRIX]")
    print(f"| {'':<10} | {'Predicted Real':<15} | {'Predicted Fake':<15} |")
    print(f"|{'-'*11}|{'-'*17}|{'-'*17}|")
    print(f"| {'True Real':<10} | {tn:<15} | {fp:<15} |")
    print(f"| {'True Fake':<10} | {fn:<15} | {tp:<15} |")
    
    print(f"\n  True Negatives (TN):  {tn} (Real videos correctly classified as Real)")
    print(f"  False Positives (FP): {fp} (Real videos incorrectly classified as Fake)")
    print(f"  False Negatives (FN): {fn} (Fake videos incorrectly classified as Real)")
    print(f"  True Positives (TP):  {tp} (Fake videos correctly classified as Fake)")
    
    print("\n[DATASET STATISTICS]")
    print(f"  Total videos:                {metrics['dataset_stats']['total_videos']}")
    print(f"  Successfully processed:      {metrics['dataset_stats']['successfully_processed']}")
    print(f"  Skipped (frame mismatch):    {metrics['dataset_stats']['skipped_frame_mismatch']}")
    print(f"  Skipped (extraction failed): {metrics['dataset_stats']['skipped_extraction_failed']}")
    print(f"  Skipped (other errors):      {metrics['dataset_stats']['skipped_other_errors']}")
    
    metrics_path = os.path.join(output_dir, f'{output_prefix}_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"\n✓ Saved metrics to: {metrics_path}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NSG-VD Batch Evaluation (Memory Efficient)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--real_folder', '-r', type=str, required=True)
    parser.add_argument('--fake_folder', '-f', type=str, required=True)
    parser.add_argument('--config', '-c', type=str, default='./config/nsgvd_detector.yaml')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default=None)
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--feature_type', '-ft', type=str, default='velocity', choices=['velocity', 'score'])
    parser.add_argument('--output', '-o', type=str, default='nsgvd_evaluation')
    parser.add_argument('--max_videos', type=int, default=None)
    
    args = parser.parse_args()
    args.output = f"{Path(args.fake_folder).parent.name}_{Path(args.checkpoint).stem}"
    
    print("="*70)
    print("NSG-VD: Memory-Efficient Batch Evaluation")
    print("="*70)
    
    print("\n[1/3] Initializing detector...")
    detector = NSGVDDetector(config_path=args.config, device_id=args.device)
    
    print("\n[2/3] Loading model components...")
    detector.load_model(checkpoint_path=args.checkpoint)
    detector.load_score_function()
    detector.load_reference_features()
    
    print("\n[3/3] Running evaluation...")
    metrics = evaluate_detector(
        detector=detector,
        real_videos_dir=args.real_folder,
        fake_videos_dir=args.fake_folder,
        output_prefix=args.output,
        feature_type=args.feature_type,
        max_videos_per_class=args.max_videos
    )
    
    if metrics:
        print(f"\n✅ Evaluation complete! Results saved to: ./evaluation_results/")
    else:
        print(f"\n❌ Evaluation failed")


if __name__ == '__main__':
    main()