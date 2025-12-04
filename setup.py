"""
Setup script for NSG-VD Detector with Memory Management
Generates reference features from real videos for single-video inference

Based on NSG-VD repository: https://github.com/ZSHsh98/NSG-VD

UPDATED: Fixed num_frames mismatch and shape validation issues
"""

import torch
import torch.nn.functional as F
import os
import sys
import gc
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


def clear_gpu_memory(device):
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def collect_video_paths_by_category(root_dir, max_per_category):
    """
    Collect video paths from root directory with subdirectories as categories.
    Each subdirectory is treated as a category, and max_per_category videos
    are selected from each.
    
    Args:
        root_dir: Root directory containing category subdirectories
        max_per_category: Max videos per category (None = all)
    
    Returns:
        List of video paths
    """
    root = Path(root_dir)
    all_paths = []
    
    # Get immediate subdirectories (categories)
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    
    # If no subdirs, treat root itself as single category
    if not subdirs:
        subdirs = [root]
    
    for cat_dir in subdirs:
        cat_paths = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            cat_paths.extend(cat_dir.rglob(ext))
        cat_paths = sorted(cat_paths)
        
        if max_per_category is not None:
            cat_paths = cat_paths[:max_per_category]
        
        print(f"    Category: {cat_dir.relative_to(root)} -> {len(cat_paths)} videos")
        all_paths.extend(cat_paths)
    
    return all_paths


def generate_reference_features(
    ref_video_dir,
    checkpoint_path,
    output_dir,
    max_videos_per_category=None,
    num_frames=8,
    device_id=0,
    diffuse_steps=5,
    resolution_size=224,
    batch_save=50  # Save features in batches to avoid memory accumulation
):
    """
    Generate reference features from real videos with memory management
    
    Args:
        ref_video_dir: Directory containing real/reference videos
        checkpoint_path: Path to NSG-VD model checkpoint
        output_dir: Directory to save reference features
        max_videos_per_category: Max videos per category subfolder
        num_frames: Frames per video
        device_id: GPU device ID
        diffuse_steps: Diffusion steps for NSG extraction
        resolution_size: Resolution for NSG computation
        batch_save: Save features every N videos to avoid memory overflow
        
    Returns:
        feature_ref: Reference features tensor
        ref_data: Reference NSG features tensor
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("NSG-VD REFERENCE FEATURE GENERATION (WITH MEMORY MANAGEMENT)")
    print("="*70)
    print(f"Reference video directory: {ref_video_dir}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Max videos per category: {max_videos_per_category if max_videos_per_category else 'all'}")
    print(f"Frames per video: {num_frames}")
    print(f"Diffusion steps: {diffuse_steps}")
    print(f"Batch save interval: {batch_save} videos")
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
        is_yy_zero=True,
        is_smooth=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Step 2: Load score function
    print("\n[2/4] Loading score function (diffusion model)...")
    score_config_path = './libs/eps_ad/imagnet.yml'
    score_args_path = './libs/eps_ad/args.yml'
    
    if not os.path.exists(score_config_path):
        print(f"Warning: Score config not found at {score_config_path}")
        print("Make sure diffusion model is set up correctly!")
    
    score_fn = get_score_fn(device, score_args_path, score_config_path)
    print("✓ Score function loaded")
    
    # Step 3: Collect videos
    print(f"\n[3/4] Collecting videos from {ref_video_dir}...")
    video_files = collect_video_paths_by_category(ref_video_dir, max_videos_per_category)
    
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {ref_video_dir}")
    
    print(f"\n✓ Total videos collected: {len(video_files)}")
    
    # Step 4: Extract features with memory management
    print(f"\n[4/4] Extracting NSG features from {len(video_files)} videos...")
    all_features = []
    all_nsg_data = []
    batch_features = []
    batch_nsg_data = []
    failed_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    batch_counter = 0
    
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
                
                # Verify shape matches expected num_frames
                if T_frames != num_frames:
                    print(f"\n  ⚠ Warning: {video_path.name} has {T_frames} frames, expected {num_frames}")
                    continue
                
                # Extract NSG features
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
                
                # Add batch dimension and extract features (KEEP ON GPU)
                velocity_batch = velocity.unsqueeze(0)
                _, features = model.net(velocity_batch, out_feature=True)
                
                # Keep on GPU - don't move to CPU yet
                batch_features.append(features)
                batch_nsg_data.append(velocity_batch.view(1, -1))
                
                # Clear intermediate tensors but KEEP batch on GPU
                del video_tensor, video_resized, normalized, score, pixel_diff, denominator, velocity, velocity_batch
                
                # Save batch when limit reached
                if len(batch_features) >= batch_save:
                    # Move to CPU only when saving
                    all_features.extend([f.cpu() for f in batch_features])
                    all_nsg_data.extend([d.cpu() for d in batch_nsg_data])
                    batch_counter += 1
                    print(f"\n  ✓ Batch {batch_counter} saved ({len(batch_features)} videos, total: {video_idx + 1})")
                    batch_features = []
                    batch_nsg_data = []
                    clear_gpu_memory(device)  # Clear only when saving batch
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  ✗ CUDA OOM on {video_path.name}: {e}")
                    failed_count += 1
                    clear_gpu_memory(device)
                    # Skip this video and continue
                    continue
                else:
                    print(f"\n  ✗ Error processing {video_path.name}: {e}")
                    failed_count += 1
                    continue
            except Exception as e:
                print(f"\n  ✗ Failed to process {video_path.name}: {e}")
                failed_count += 1
                continue
    
    # Save remaining batch
    if len(batch_features) > 0:
        all_features.extend([f.cpu() for f in batch_features])
        all_nsg_data.extend([d.cpu() for d in batch_nsg_data])
        batch_counter += 1
        print(f"\n  ✓ Final batch {batch_counter} saved ({len(batch_features)} videos)")
    
    if len(all_features) == 0:
        raise RuntimeError("Failed to extract features from any video!")
    
    # Concatenate all features
    feature_ref = torch.cat(all_features, dim=0)
    ref_data = torch.cat(all_nsg_data, dim=0)
    
    print(f"\n✓ Extracted features from {len(all_features)} videos (failed: {failed_count})")
    print(f"  - feature_ref shape: {feature_ref.shape}")
    print(f"  - ref_data shape: {ref_data.shape}")
    
    # Save reference features
    print(f"\nSaving reference features to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(feature_ref, os.path.join(output_dir, 'feature_ref.pt'))
    torch.save(ref_data, os.path.join(output_dir, 'ref_data.pt'))
    
    # Save metadata
    metadata = {
        'num_frames': num_frames,
        'num_videos': len(all_features),
        'resolution_size': resolution_size,
        'diffuse_steps': diffuse_steps,
        'feature_dim': 300
    }
    torch.save(metadata, os.path.join(output_dir, 'metadata.pt'))
    
    print("✓ Reference features saved")
    clear_gpu_memory(device)
    
    return feature_ref, ref_data, num_frames


def create_config_file(
    checkpoint_path,
    reference_dir,
    num_frames,
    output_path='./config/nsgvd_detector.yaml'
):
    """
    Create configuration file for NSG-VD detector
    
    Args:
        checkpoint_path: Path to model checkpoint
        reference_dir: Directory containing reference features
        num_frames: Number of frames used during reference generation
        output_path: Path to save config file
    """
    config = {
        'model_name': 'NSG-VD',
        'feature_dim': 300,
        'feature_type': 'velocity',
        
        'sigma': 1.0,
        'sigma0': 0.1,
        'epsilon': 1e-10,
        'is_yy_zero': True,
        'is_smooth': True,
        
        'img_size': 224,
        'resolution_size': 224,
        'num_frames': num_frames,  # CRITICAL: Must match reference generation
        
        'diffuse_steps': 5,
        'score_config_path': './libs/eps_ad/imagnet.yml',
        'score_args_path': './libs/eps_ad/args.yml',
        
        'checkpoint_path': str(Path(checkpoint_path).resolve()),
        
        'ref_features_path': str(Path(reference_dir) / 'feature_ref.pt'),
        'ref_data_path': str(Path(reference_dir) / 'ref_data.pt'),
        
        'threshold': 1.0,
        
        'device': 'cuda',
        'device_id': 0,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Config file created: {output_path}")
    print(f"  - num_frames set to: {num_frames}")


def verify_setup(config_path, test_video_path=None):
    """
    Verify setup by running a test inference
    """
    from main import NSGVDDetector
    
    print("\n" + "="*70)
    print("VERIFYING SETUP")
    print("="*70)
    
    try:
        print("\n[1/3] Initializing detector...")
        detector = NSGVDDetector(config_path=config_path)
        detector.load_model()
        detector.load_score_function()
        detector.load_reference_features()
        print("✓ Detector initialized successfully")
        
        # Verify reference data shape
        ref_data = torch.load(detector.config['ref_data_path'])
        expected_size = detector.config['num_frames'] * 3 * 224 * 224
        actual_size = ref_data.shape[1]
        
        print(f"\n[2/3] Validating reference data...")
        print(f"  Expected size per sample: {expected_size}")
        print(f"  Actual size per sample: {actual_size}")
        print(f"  num_frames from config: {detector.config['num_frames']}")
        
        if actual_size != expected_size:
            print(f"\n✗ WARNING: Reference data size mismatch!")
            print(f"  This suggests num_frames in config doesn't match reference generation")
            print(f"  Expected: {expected_size}, Got: {actual_size}")
            # Calculate what num_frames should be
            correct_num_frames = actual_size // (3 * 224 * 224)
            print(f"  Reference data was likely generated with num_frames={correct_num_frames}")
            return False
        else:
            print("  ✓ Reference data shape is correct")
        
        if test_video_path and os.path.exists(test_video_path):
            print(f"\n[3/3] Testing inference on {test_video_path}...")
            result = detector.get_predictions(test_video_path)
            print("✓ Inference successful!")
            print(f"\nTest Results:")
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  MMD Score: {result['mmd_score']:.6f}")
            print(f"  Confidence: {result['confidence']:.4f}")
        else:
            print("\n[3/3] Skipping test inference (no test video provided)")
        
        print("\n" + "="*70)
        print("SETUP VERIFICATION COMPLETE! ✓")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Setup verification failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Setup NSG-VD Detector with Memory Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reference features with memory management
  python setup.py \\
    --ref_video_dir ../Data/GenVideo/video/real \\
    --checkpoint ./ckpts/standard-Pika-mp.pth \\
    --max_videos_per_category 50 \\
    --num_frames 8 \\
    --batch_save 30

  # With custom GPU and batch size
  python setup.py \\
    --ref_video_dir ../Data/GenVideo/video/real \\
    --checkpoint ./ckpts/standard-Pika-mp.pth \\
    --device 1 \\
    --num_frames 8 \\
    --batch_save 20
        """
    )
    
    parser.add_argument('--ref_video_dir', type=str, help='Reference videos directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./reference', help='Output directory')
    parser.add_argument('--reference_dir', type=str, help='Existing reference directory')
    parser.add_argument('--config_output', type=str, default='./config/nsgvd_detector.yaml')
    parser.add_argument('--max_videos_per_category', type=int, default=1000)
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames per video (CRITICAL)')
    parser.add_argument('--batch_save', type=int, default=50, help='Save features every N videos')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--test_video', type=str, help='Test video path for verification')
    parser.add_argument('--skip_reference', action='store_true', help='Skip reference generation')
    
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Auto-name reference dir and config file based on checkpoint
    # ------------------------------------------------------------------
    ckpt_name = Path(args.checkpoint).stem  # e.g. "unbalance-SEINE-mp"

    # if args.output_dir == './reference':
    args.reference_dir = f'./reference_{ckpt_name}_{args.max_videos_per_category}'

    # if args.config_output == './config/nsgvd_detector.yaml':
    args.config_output = f'./config/nsgvd_detector_{ckpt_name}_{args.max_videos_per_category}.yaml'

    print("\nResolved paths based on checkpoint:")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Reference dir:  {args.reference_dir}")
    print(f"  Config output:  {args.config_output}\n")

    
    if not args.skip_reference and not args.ref_video_dir:
        parser.error("--ref_video_dir required unless --skip_reference is set")
    
    
    num_frames_used = args.num_frames
    
    if not args.skip_reference:
        feature_ref, ref_data, num_frames_used = generate_reference_features(
            ref_video_dir=args.ref_video_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.reference_dir,
            max_videos_per_category=args.max_videos_per_category,
            num_frames=args.num_frames,
            device_id=args.device,
            batch_save=args.batch_save
        )
    else:
        print("\nSkipping reference feature generation...")
        print(f"Using existing features from {args.reference_dir}")
        
        # Load metadata to get num_frames
        metadata_path = os.path.join(args.reference_dir, 'metadata.pt')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            num_frames_used = metadata['num_frames']
            print(f"Loaded metadata: num_frames = {num_frames_used}")
        else:
            print(f"Warning: No metadata found at {metadata_path}")
            print(f"Using command-line num_frames = {args.num_frames}")
            num_frames_used = args.num_frames
    
    print("\n" + "="*70)
    print("CREATING CONFIG FILE")
    print("="*70)
    create_config_file(
        checkpoint_path=args.checkpoint,
        reference_dir=args.reference_dir or args.output_dir,
        num_frames=num_frames_used,
        output_path=args.config_output
    )
    
    success = verify_setup(config_path=args.config_output, test_video_path=args.test_video)
    
    if success:
        print("\n" + "="*70)
        print("SETUP COMPLETE!")
        print("="*70)
        print("\nYou can now use NSG-VD detector in your code:")
        print("\n  from main import NSGVDDetector")
        print(f"  detector = NSGVDDetector(config_path='{args.config_output}')")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SETUP FAILED - Please review errors above")
        print("="*70)
        sys.exit(1)
    

if __name__ == '__main__':
    main()