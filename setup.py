"""
Setup script for NSG-VD Detector
Generates reference features from real videos for single-video inference

Based on NSG-VD repository: https://github.com/ZSHsh98/NSG-VD
"""

import torch
import torch.nn.functional as F
import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T

# Add project paths
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, ".."))

# Import NSG-VD components
from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator
from data.utils import get_score_fn
from preprocess import extract_and_preprocess_frames


def generate_reference_features(
    ref_video_dir,
    checkpoint_path,
    output_dir,
    num_videos=200,
    num_frames=8,
    device_id=0,
    diffuse_steps=5,
    resolution_size=224
):
    """
    Generate reference features from real videos
    
    Args:
        ref_video_dir: Directory containing real/reference videos
        checkpoint_path: Path to NSG-VD model checkpoint
        output_dir: Directory to save reference features
        num_videos: Number of reference videos to process
        num_frames: Frames per video
        device_id: GPU device ID
        diffuse_steps: Diffusion steps for NSG extraction
        resolution_size: Resolution for NSG computation
        
    Returns:
        feature_ref: Reference features tensor
        ref_data: Reference NSG features tensor
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("NSG-VD REFERENCE FEATURE GENERATION")
    print("="*70)
    print(f"Reference video directory: {ref_video_dir}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Number of videos: {num_videos}")
    print(f"Frames per video: {num_frames}")
    print(f"Diffusion steps: {diffuse_steps}")
    print("="*70)
    
    # Step 1: Load NSG-VD model
    print("\n[1/4] Loading NSG-VD model...")
    discriminator = SingleSwinBlockDiscriminator(num_features=300)
    model = deep_MMD(
        discriminator=discriminator,
        sigma=1.0,
        sigma0=0.1,
        epsilon=1e-10,
        img_size=224,
        is_yy_zero=True,  # MMD-MP
        is_smooth=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Step 2: Load score function (diffusion model)
    print("\n[2/4] Loading score function (diffusion model)...")
    score_config_path = './libs/eps_ad/imagnet.yml'
    score_args_path = './libs/eps_ad/args.yml'
    
    if not os.path.exists(score_config_path):
        print(f"Warning: Score config not found at {score_config_path}")
        print("Make sure diffusion model is set up correctly!")
    
    score_fn = get_score_fn(device, score_args_path, score_config_path)
    print("✓ Score function loaded")
    
    # Step 3: Collect video files
    print(f"\n[3/4] Collecting videos from {ref_video_dir}...")
    video_files = []
    for ext in ['**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv']:
        video_files.extend(Path(ref_video_dir).glob(ext))    
    video_files = sorted(video_files)[:num_videos]
    
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {ref_video_dir}")
    
    print(f"✓ Found {len(video_files)} videos (using {min(len(video_files), num_videos)})")
    
    # Step 4: Extract features from videos
    print(f"\n[4/4] Extracting NSG features from {len(video_files)} videos...")
    all_features = []
    all_nsg_data = []
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    
    with torch.no_grad():
        for video_idx, video_path in enumerate(tqdm(video_files, desc="Processing videos")):
            try:
                # Extract frames
                video_tensor = extract_and_preprocess_frames(
                    str(video_path), 
                    num_frames=num_frames
                )
                
                video_tensor = video_tensor.to(device)
                T_frames, C, H, W = video_tensor.shape
                
                # Extract NSG features
                # Resize for score computation
                if H != resolution_size or W != resolution_size:
                    video_resized = F.interpolate(
                        video_tensor,
                        size=(resolution_size, resolution_size),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    video_resized = video_tensor
                
                # Normalize to [-1, 1]
                normalized = 2 * video_resized - 1
                
                # Get scores
                t_value = (diffuse_steps + 1) / 1000
                curr_t = torch.tensor(t_value, device=device).expand(T_frames)
                
                score = score_fn(normalized, curr_t)
                score, _ = torch.split(score, score.shape[1] // 2, dim=1)
                
                # Resize back
                if resolution_size != H or resolution_size != W:
                    score = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=False)
                
                # Compute NSG velocity
                pixel_diff = torch.zeros_like(score)
                for t in range(1, T_frames - 1):
                    pixel_diff[t] = (video_tensor[t + 1] - video_tensor[t - 1]) / 2
                pixel_diff[0] = video_tensor[1] - video_tensor[0]
                pixel_diff[-1] = video_tensor[-1] - video_tensor[-2]
                
                epsilon = 1e-10
                denominator = (score * pixel_diff).sum(dim=(1, 2, 3)).view(T_frames, 1, 1, 1) + epsilon
                velocity = score / denominator
                
                # Add batch dimension and extract features
                velocity_batch = velocity.unsqueeze(0)  # (1, T, C, H, W)
                _, features = model.net(velocity_batch, out_feature=True)
                
                all_features.append(features)
                all_nsg_data.append(velocity_batch.view(1, -1))
                
            except Exception as e:
                print(f"\nWarning: Failed to process {video_path.name}: {e}")
                continue
    
    if len(all_features) == 0:
        raise RuntimeError("Failed to extract features from any video!")
    
    # Concatenate all features
    feature_ref = torch.cat(all_features, dim=0)
    ref_data = torch.cat(all_nsg_data, dim=0)
    
    print(f"\n✓ Extracted features from {len(all_features)} videos")
    print(f"  - feature_ref shape: {feature_ref.shape}")
    print(f"  - ref_data shape: {ref_data.shape}")
    
    # Save reference features
    print(f"\nSaving reference features to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(feature_ref, os.path.join(output_dir, 'feature_ref.pt'))
    torch.save(ref_data, os.path.join(output_dir, 'ref_data.pt'))
    
    print("✓ Reference features saved")
    
    return feature_ref, ref_data


def create_config_file(
    checkpoint_path,
    reference_dir,
    output_path='./config/nsgvd_detector.yaml'
):
    """
    Create configuration file for NSG-VD detector
    
    Args:
        checkpoint_path: Path to model checkpoint
        reference_dir: Directory containing reference features
        output_path: Where to save config file
    """
    config = {
        # Model configuration
        'model_name': 'NSG-VD',
        'feature_dim': 768,
        'feature_type': 'velocity',  # NSG features
        
        # MMD parameters
        'sigma': 1.0,
        'sigma0': 0.1,
        'epsilon': 1e-10,
        'is_yy_zero': True,  # MMD-MP (recommended for demo)
        'is_smooth': True,
        
        # Image processing
        'img_size': 224,
        'resolution_size': 224,
        'num_frames': 8,
        
        # NSG extraction
        'diffuse_steps': 5,
        'score_config_path': './libs/eps_ad/imagnet.yml',
        'score_args_path': './libs/eps_ad/args.yml',
        
        # Model checkpoint
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        
        # Reference features
        'ref_features_path': str(Path(reference_dir) / 'feature_ref.pt'),
        'ref_data_path': str(Path(reference_dir) / 'ref_data.pt'),
        
        # Detection
        'threshold': 1.0,
        
        # Device
        'device': 'cuda',
        'device_id': 0,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Config file created: {output_path}")


def verify_setup(config_path, test_video_path=None):
    """
    Verify setup by running a test inference
    
    Args:
        config_path: Path to config file
        test_video_path: Optional test video
    """
    from main import NSGVDDetector
    
    print("\n" + "="*70)
    print("VERIFYING SETUP")
    print("="*70)
    
    try:
        # Initialize detector
        print("\n[1/3] Initializing detector...")
        detector = NSGVDDetector(config_path=config_path)
        detector.load_model()
        detector.load_score_function()
        detector.load_reference_features()
        print("✓ Detector initialized successfully")
        
        # Test inference if video provided
        if test_video_path and os.path.exists(test_video_path):
            print(f"\n[2/3] Testing inference on {test_video_path}...")
            result = detector.get_predictions(test_video_path)
            print("✓ Inference successful!")
            print(f"\nTest Results:")
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  MMD Score: {result['mmd_score']}")
            print(f"  Confidence: {result['confidence']}")
        else:
            print("\n[2/3] Skipping test inference (no test video provided)")
        
        print("\n[3/3] Setup verification complete! ✓")
        
    except Exception as e:
        print(f"\n✗ Setup verification failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Setup NSG-VD Detector for Demo Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reference features from real videos
  python setup.py \\
    --ref_video_dir ../Data/GenVideo/video/real/Kinetics-400 \\
    --checkpoint ./ckpts/standard-Pika-mp.pth \\
    --output_dir ./reference \\
    --num_videos 200

  # Skip reference generation if already done
  python setup.py \\
    --checkpoint ./ckpts/standard-Pika-mp.pth \\
    --reference_dir ./reference \\
    --skip_reference \\
    --test_video test.mp4
        """
    )
    
    parser.add_argument(
        '--ref_video_dir',
        type=str,
        help='Directory containing reference (real) videos'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to NSG-VD model checkpoint (.pth)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./reference',
        help='Directory to save reference features'
    )
    parser.add_argument(
        '--reference_dir',
        type=str,
        help='Existing reference directory (use with --skip_reference)'
    )
    parser.add_argument(
        '--config_output',
        type=str,
        default='./config/nsgvd_detector.yaml',
        help='Output path for config file'
    )
    parser.add_argument(
        '--num_videos',
        type=int,
        default=200,
        help='Number of reference videos to process'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=8,
        help='Number of frames per video'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--test_video',
        type=str,
        help='Optional test video for verification'
    )
    parser.add_argument(
        '--skip_reference',
        action='store_true',
        help='Skip reference generation (use existing features)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_reference and not args.ref_video_dir:
        parser.error("--ref_video_dir required unless --skip_reference is set")
    
    if args.skip_reference and not args.reference_dir:
        args.reference_dir = args.output_dir
    
    # Step 1: Generate reference features
    if not args.skip_reference:
        feature_ref, ref_data = generate_reference_features(
            ref_video_dir=args.ref_video_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_videos=args.num_videos,
            num_frames=args.num_frames,
            device_id=args.device
        )
    else:
        print("\nSkipping reference feature generation...")
        print(f"Using existing features from {args.reference_dir}")
    
    # Step 2: Create config file
    print("\n" + "="*70)
    print("CREATING CONFIG FILE")
    print("="*70)
    create_config_file(
        checkpoint_path=args.checkpoint,
        reference_dir=args.reference_dir or args.output_dir,
        output_path=args.config_output
    )
    
    # Step 3: Verify setup
    verify_setup(
        config_path=args.config_output,
        test_video_path=args.test_video
    )
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nYou can now use NSG-VD detector in your demo:")
    print("\n  from main import NSGVDDetector")
    print(f"  detector = NSGVDDetector(config_path='{args.config_output}')")
    print("  detector.load_model()")
    print("  detector.load_score_function()")
    print("  detector.load_reference_features()")
    print("  result = detector.get_predictions('video.mp4')")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()