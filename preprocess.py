"""
Preprocessing utilities for NSG-VD detector
Frame extraction and video processing for single-video inference
Compatible with NSG-VD repository structure
"""

import cv2
import numpy as np
import os
import sys
import torch
import tempfile
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)


def get_video_frame_count(video_path):
    """Get total frame count of video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def extract_frames_from_video(video_path, num_frames=8, output_dir=None):
    """
    Extract frames uniformly from video using OpenCV
    
    Args:
        video_path: Path to input video
        num_frames: Number of frames to extract
        output_dir: Directory to save frames (None = temp directory)
        
    Returns:
        List of frame paths
    """
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='nsgvd_frames_')
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")
    
    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, requested {num_frames}")
        frame_indices = list(range(total_frames))
    else:
        # Uniform sampling
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frame_paths = []
    current_frame = 0
    extracted_count = 0
    
    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in frame_indices:
            frame_path = os.path.join(output_dir, f"frame{extracted_count + 1:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
            
            if extracted_count >= num_frames:
                break
        
        current_frame += 1
    
    cap.release()
    
    if len(frame_paths) == 0:
        raise ValueError(f"Failed to extract any frames from {video_path}")
    
    return frame_paths


def extract_frames_ffmpeg(video_path, num_frames=8, output_dir=None):
    """
    Extract frames using ffmpeg (faster, requires ffmpeg installed)
    
    Args:
        video_path: Path to input video
        num_frames: Number of frames to extract
        output_dir: Directory to save frames
        
    Returns:
        List of frame paths
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='nsgvd_frames_')
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get total frame count
        total_frames = get_video_frame_count(video_path)
        
        if total_frames == 0:
            raise ValueError(f"Video has 0 frames: {video_path}")
        
        # Calculate frame interval
        frame_interval = max(1, total_frames // num_frames)
        
        # FFmpeg command
        output_pattern = os.path.join(output_dir, "frame%04d.jpg")
        cmd = (
            f"ffmpeg -i {video_path} "
            f"-vf select='not(mod(n\\,{frame_interval}))',setpts=N/FRAME_RATE/TB "
            f"-vframes {num_frames} {output_pattern} "
            f"-loglevel quiet -y"
        )
        
        ret = os.system(cmd)
        
        if ret != 0:
            print("FFmpeg extraction failed, falling back to OpenCV")
            return extract_frames_from_video(video_path, num_frames, output_dir)
        
        # Get extracted frame paths
        frame_paths = sorted([
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.endswith('.jpg')
        ])
        
        if len(frame_paths) == 0:
            print("FFmpeg produced no frames, falling back to OpenCV")
            return extract_frames_from_video(video_path, num_frames, output_dir)
        
        return frame_paths
        
    except Exception as e:
        print(f"FFmpeg extraction error: {e}, falling back to OpenCV")
        return extract_frames_from_video(video_path, num_frames, output_dir)


def extract_and_preprocess_frames(video_path, num_frames=8, img_size=224):
    """
    Extract frames and convert to tensor for model input
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        img_size: Target image size (default 224x224)
        
    Returns:
        Tensor of shape (T, C, H, W)
    """
    # Try ffmpeg first, fallback to OpenCV
    try:
        frame_paths = extract_frames_ffmpeg(video_path, num_frames)
    except:
        frame_paths = extract_frames_from_video(video_path, num_frames)
    
    # Preprocessing transform
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    
    # Load and preprocess frames
    frames = []
    for frame_path in frame_paths:
        try:
            img = Image.open(frame_path).convert('RGB')
            frame_tensor = transform(img)
            frames.append(frame_tensor)
        except Exception as e:
            print(f"Warning: Failed to load frame {frame_path}: {e}")
            continue
    
    if len(frames) == 0:
        raise ValueError(f"Failed to load any frames from {video_path}")
    
    # Stack to (T, C, H, W)
    video_tensor = torch.stack(frames, dim=0)
    
    return video_tensor


def load_frames_from_directory(frame_dir, num_frames=8, img_size=224):
    """
    Load frames from a directory (for pre-extracted frames)
    
    Args:
        frame_dir: Directory containing frame images
        num_frames: Number of frames to load
        img_size: Target image size
        
    Returns:
        Tensor of shape (T, C, H, W)
    """
    # Get all image files
    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {frame_dir}")
    
    # Sample uniformly if we have more than needed
    if len(frame_paths) > num_frames:
        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
        frame_paths = [frame_paths[i] for i in indices]
    
    # Preprocessing transform
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    
    # Load frames
    frames = []
    for frame_path in frame_paths[:num_frames]:
        img = Image.open(frame_path).convert('RGB')
        frame_tensor = transform(img)
        frames.append(frame_tensor)
    
    video_tensor = torch.stack(frames, dim=0)
    
    return video_tensor


def batch_extract_frames(video_dir, output_base_dir, num_frames=8):
    """
    Extract frames from all videos in a directory
    
    Args:
        video_dir: Directory containing videos
        output_base_dir: Base directory for output
        num_frames: Number of frames per video
    """
    from tqdm import tqdm
    
    # Get all video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(Path(video_dir).glob(ext))
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = video_path.stem
        output_dir = os.path.join(output_base_dir, video_name)
        
        try:
            extract_frames_ffmpeg(str(video_path), num_frames, output_dir)
        except Exception as e:
            print(f"Failed to process {video_name}: {e}")


def preprocess_frame_from_path(frame_path, input_shape=(224, 224)):
    """
    Preprocess a single frame from file path
    Used in NSG-VD dataset processing
    
    Args:
        frame_path: Path to frame image
        input_shape: Target size (H, W)
        
    Returns:
        Preprocessed frame tensor (C, H, W)
    """
    frame_img = Image.open(frame_path).convert('RGB')
    frame_img = frame_img.resize(input_shape, Image.Resampling.LANCZOS)
    
    frame_tensor = torch.from_numpy(np.array(frame_img)).float()
    frame_tensor = frame_tensor / 255.0  # Normalize to [0, 1]
    frame_tensor = frame_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    return frame_tensor


def process_video_batch(video_paths, num_frames=8, img_size=224):
    """
    Process multiple videos into a batch tensor
    
    Args:
        video_paths: List of video file paths
        num_frames: Number of frames per video
        img_size: Target image size
        
    Returns:
        Batch tensor of shape (B, T, C, H, W)
    """
    batch = []
    
    for video_path in video_paths:
        try:
            video_tensor = extract_and_preprocess_frames(
                video_path, 
                num_frames=num_frames,
                img_size=img_size
            )
            batch.append(video_tensor)
        except Exception as e:
            print(f"Warning: Failed to process {video_path}: {e}")
            continue
    
    if len(batch) == 0:
        raise ValueError("Failed to process any videos")
    
    # Stack to (B, T, C, H, W)
    batch_tensor = torch.stack(batch, dim=0)
    
    return batch_tensor


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NSG-VD Frame Extraction Utilities')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['extract', 'batch_extract', 'load_test'],
        help='Operation mode'
    )
    parser.add_argument(
        '--video_path',
        type=str,
        help='Path to video file (for extract mode)'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        help='Directory containing videos (for batch_extract mode)'
    )
    parser.add_argument(
        '--frame_dir',
        type=str,
        help='Directory containing frames (for load_test mode)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=8,
        help='Number of frames to extract'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'extract':
        if not args.video_path:
            print("Error: --video_path required for extract mode")
            sys.exit(1)
        
        print(f"Extracting {args.num_frames} frames from {args.video_path}")
        frame_paths = extract_frames_from_video(
            args.video_path,
            num_frames=args.num_frames,
            output_dir=args.output_dir
        )
        print(f"Extracted {len(frame_paths)} frames:")
        for path in frame_paths:
            print(f"  {path}")
    
    elif args.mode == 'batch_extract':
        if not args.video_dir or not args.output_dir:
            print("Error: --video_dir and --output_dir required for batch_extract mode")
            sys.exit(1)
        
        print(f"Batch extracting frames from {args.video_dir}")
        batch_extract_frames(args.video_dir, args.output_dir, args.num_frames)
        print("Done!")
    
    elif args.mode == 'load_test':
        if not args.frame_dir:
            print("Error: --frame_dir required for load_test mode")
            sys.exit(1)
        
        print(f"Loading frames from {args.frame_dir}")
        video_tensor = load_frames_from_directory(args.frame_dir, args.num_frames)
        print(f"Loaded tensor shape: {video_tensor.shape}")