"""
NSG-VD (Neural Score Gradient - Velocity Distribution) Detector
Demo Integration for Single Video Inference

Based on: https://github.com/ZSHsh98/NSG-VD
Paper: "Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection" (NeurIPS 2025)
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

# Add project paths
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, ".."))

# Import from NSG-VD repository
from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator
from utils.train_utils import MMD_batch2


class NSGVDDetector:
    """
    NSG-VD Detector for single video inference
    Compatible with demo structure (similar to DFBDetector)
    """
    
    def __init__(self, config_path=None, device_id=0):
        """
        Args:
            config_path: Path to detector config YAML
            device_id: GPU device ID (use -1 for CPU)
        """
        self.device = torch.device(
            f'cuda:{device_id}' if torch.cuda.is_available() and device_id >= 0 else 'cpu'
        )
        
        # Load configuration
        if config_path is None:
            print("config:None")
            config_path = './config/nsgvd_detector.yaml'
            # config_path = os.path.join(base_path, 'config', 'nsgvd_detector.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model components (loaded later)
        self.model = None
        self.score_fn = None
        self.feature_ref = None
        self.ref_data = None
        
        print(f"NSG-VD Detector initialized on {self.device}")
        
    def load_model(self, checkpoint_path=None):
        """
        Load the NSG-VD model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.get('checkpoint_path')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from {checkpoint_path}")
        
        # Initialize discriminator (TALL architecture)
        discriminator = SingleSwinBlockDiscriminator(
            num_features=self.config.get('feature_dim', 300)
        )
        
        # Initialize Deep MMD model
        self.model = deep_MMD(
            discriminator=discriminator,
            sigma=self.config.get('sigma', 1.0),
            sigma0=self.config.get('sigma0', 0.1),
            epsilon=self.config.get('epsilon', 1e-10),
            img_size=self.config.get('img_size', 224),
            is_yy_zero=self.config.get('is_yy_zero', True),  # MMD-MP by default
            is_smooth=self.config.get('is_smooth', True)
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully (MMD-{'MP' if self.model.is_yy_zero else 'D'})")
        
    def load_score_function(self):
        """
        Load the pre-trained diffusion model for score estimation
        Requires: 256x256_diffusion_uncond.pt in ../Checkpoints/
        """
        from data.utils import get_score_fn
        
        score_config_path = self.config.get('score_config_path', './libs/eps_ad/imagnet.yml')
        score_args_path = self.config.get('score_args_path', './libs/eps_ad/args.yml')
        
        print(f"Loading score function...")
        self.score_fn = get_score_fn(self.device, score_args_path, score_config_path)
        print("✓ Score function loaded")
        
    def load_reference_features(self, ref_features_path=None, ref_data_path=None):
        """
        Load pre-computed reference features from real videos
        
        Args:
            ref_features_path: Path to reference features (.pt file)
            ref_data_path: Path to reference data (.pt file)
        """
        if ref_features_path is None:
            ref_features_path = self.config.get('ref_features_path')
        if ref_data_path is None:
            ref_data_path = self.config.get('ref_data_path')
        
        if not os.path.exists(ref_features_path):
            raise FileNotFoundError(f"Reference features not found: {ref_features_path}")
        if not os.path.exists(ref_data_path):
            raise FileNotFoundError(f"Reference data not found: {ref_data_path}")
        
        print(f"Loading reference features...")
        self.feature_ref = torch.load(ref_features_path, map_location=self.device)
        self.ref_data = torch.load(ref_data_path, map_location=self.device)
        
        print(f"✓ Reference loaded: {self.feature_ref.shape[0]} samples")
        
    @torch.no_grad()
    def extract_nsg_features(self, video_frames):
        """
        Extract NSG (Normalized Spatiotemporal Gradient) features
        
        Args:
            video_frames: Tensor of shape (T, C, H, W) or list of PIL Images
            
        Returns:
            score: (T, C, H, W) - spatial gradients
            velocity: (T, C, H, W) - NSG features
        """
        # Convert to tensor if needed
        if isinstance(video_frames, list):
            from torchvision import transforms as T
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            video_frames = torch.stack([transform(img) for img in video_frames])
        
        # Ensure correct shape: (T, C, H, W)
        if video_frames.dim() == 5:
            video_frames = video_frames.squeeze(0)
        
        T_frames, C, H, W = video_frames.shape
        video_frames = video_frames.to(self.device)
        
        # Prepare for score function
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
        
        # Normalize to [-1, 1] for diffusion model
        normalized_frames = 2 * video_frames_resized - 1
        
        # Get score features using diffusion model
        diffuse_steps = self.config.get('diffuse_steps', 5)
        t_value = (diffuse_steps + 1) / 1000
        curr_t = torch.tensor(t_value, device=self.device).expand(T_frames)
        
        # Compute spatial gradients (score)
        score = self.score_fn(normalized_frames, curr_t)
        score, _ = torch.split(score, score.shape[1] // 2, dim=1)
        
        # Resize back to original resolution
        if resolution_size != H or resolution_size != W:
            score = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=False)
        
        # Compute NSG (velocity) features
        velocity = self._compute_nsg_velocity(score, video_frames)
        
        return score, velocity
    
    def _compute_nsg_velocity(self, score, frames, epsilon=1e-10):
        """
        Compute NSG velocity: ratio of spatial gradients to temporal density changes
        
        Args:
            score: (T, C, H, W) - spatial probability gradients
            frames: (T, C, H, W) - original frames
            
        Returns:
            velocity: (T, C, H, W) - NSG features
        """
        T, C, H, W = score.shape
        
        # Compute temporal differences (pixel-level motion)
        pixel_diff = torch.zeros_like(score)
        
        # Central difference for middle frames
        for t in range(1, T - 1):
            pixel_diff[t] = (frames[t + 1] - frames[t - 1]) / 2
        
        # Forward/backward difference for boundaries
        pixel_diff[0] = frames[1] - frames[0]
        pixel_diff[-1] = frames[-1] - frames[-2]
        
        # Compute NSG: score / temporal_change
        # Aggregate temporal changes per frame
        denominator = (score * pixel_diff).sum(dim=(1, 2, 3)).view(T, 1, 1, 1) + epsilon
        velocity = score / denominator
        
        return velocity
    
    @torch.no_grad()
    def predict(self, video_path, feature_type='velocity'):
        """
        Predict whether a video is real or AI-generated
        
        Args:
            video_path: Path to video file or list of frame paths
            feature_type: 'velocity' (NSG) or 'score' (spatial gradients only)
            
        Returns:
            Dictionary with prediction results
        """
        # Import preprocessing utilities
        from preprocess import extract_and_preprocess_frames
        
        # Extract frames from video
        # print(f"Processing video: {video_path}")
        video_frames = extract_and_preprocess_frames(
            video_path, 
            num_frames=self.config.get('num_frames', 8)
        )
        
        # Extract NSG features
        # print("Extracting NSG features...")
        score, velocity = self.extract_nsg_features(video_frames)
        
        # Select feature type
        if feature_type == 'velocity':
            features = velocity
        elif feature_type == 'score':
            features = score
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Add batch dimension: (1, T, C, H, W)
        features = features.unsqueeze(0)
        
        # Extract high-level features using discriminator
        _, feature_test = self.model.net(features, out_feature=True)
        
        # Compute MMD distance to reference distribution
        # print("Computing MMD score...")
        mmd_score = MMD_batch2(
            torch.cat([self.feature_ref, feature_test], dim=0),
            self.feature_ref.shape[0],
            torch.cat([self.ref_data, features.view(1, -1)], dim=0),
            self.model.sigma,
            self.model.sigma0_u,
            self.model.ep,
            is_smooth=self.model.is_smooth
        )
        
        mmd_score = mmd_score.cpu().item()
        
        # Classification (threshold = 1.0)
        threshold = self.config.get('threshold', 1.0)
        prediction = 'fake' if mmd_score > threshold else 'real'
        confidence = abs(mmd_score - threshold)
        
        return {
            'prediction': prediction,
            'mmd_score': round(mmd_score, 4),
            'confidence': round(confidence, 4),
            prediction: round(min(confidence, 1.0), 2)  # Compatible with demo format
        }
    
    def get_predictions(self, video_path, feature_type='velocity', uploaded_file_name=None, device_id=None):
        """
        Main prediction interface (compatible with DFBDetector API)
        
        Args:
            video_path: Path to video file
            feature_type: 'velocity' or 'score'
            uploaded_file_name: Optional filename (for logging)
            device_id: Optional device override
            
        Returns:
            Dictionary with prediction results
        """
        # For compatibility with demo structure
        if uploaded_file_name:
            print(f"Processing: {uploaded_file_name}")
        
        return self.predict(video_path, feature_type=feature_type)


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NSG-VD: Physics-Driven AI-Generated Video Detection'
    )
    parser.add_argument(
        '--video_path', '-v',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file (default: config/nsgvd_detector.yaml)'
    )
    parser.add_argument(
        '--checkpoint', '-ckpt',
        type=str,
        default=None,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='GPU device ID (-1 for CPU)'
    )
    parser.add_argument(
        '--feature_type', '-f',
        type=str,
        default='velocity',
        choices=['velocity', 'score'],
        help='Feature type: velocity (NSG) or score'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("NSG-VD: AI-Generated Video Detection")
    print("="*60)
    
    # Initialize detector
    detector = NSGVDDetector(config_path=args.config, device_id=args.device)
    
    # Load components
    detector.load_model(checkpoint_path=args.checkpoint)
    detector.load_score_function()
    detector.load_reference_features()
    
    # Run prediction
    result = detector.get_predictions(args.video_path, feature_type=args.feature_type)
    
    # Display results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(f"Video: {args.video_path}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"MMD Score: {result['mmd_score']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Feature Type: {args.feature_type}")
    print("="*60)
    
    # Interpretation
    if result['prediction'] == 'fake':
        print("\n⚠️  This video is likely AI-GENERATED")
    else:
        print("\n✓ This video appears to be REAL")
    print(f"   (MMD score: {result['mmd_score']}, threshold: 1.0)")
    print("="*60)


if __name__ == '__main__':
    main()