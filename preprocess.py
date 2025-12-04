"""
ULTRA-ROBUST NSG-VD Video Preprocessing
Strategy: Read ALL frames, keep valid ones, then sample uniformly
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
import tempfile
from pathlib import Path
from torchvision import transforms as T

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)


def get_video_frame_count(video_path):
    """Get total frame count of video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        frame_count = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            frame_count += 1
    
    cap.release()
    return frame_count


def extract_all_valid_frames(video_path, output_dir=None):
    """
    ULTRA-ROBUST: Read ALL frames sequentially, save only valid ones
    This avoids seek() failures and gets maximum valid frames
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        
    Returns:
        List of valid frame paths
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='nsgvd_frames_')
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_paths = []
    frame_idx = 0
    saved_count = 0
    
    print(f"  Reading all frames sequentially...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Validate frame
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            frame_idx += 1
            continue
        
        # Save valid frame
        frame_path = os.path.join(output_dir, f"frame{saved_count:05d}.jpg")
        success = cv2.imwrite(frame_path, frame)
        
        if success:
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    
    print(f"  Extracted {len(frame_paths)} valid frames from {frame_idx} total frames")
    
    return frame_paths


def sample_frames_uniformly(frame_paths, num_frames):
    """
    Sample num_frames uniformly from available valid frames
    
    Args:
        frame_paths: List of all valid frame paths
        num_frames: Number of frames to sample
        
    Returns:
        List of sampled frame paths (exactly num_frames)
    """
    total_valid = len(frame_paths)
    
    if total_valid < num_frames:
        # Not enough frames - return all and pad
        sampled = frame_paths.copy()
        print(f"  Warning: Only {total_valid} valid frames, padding to {num_frames}")
        return sampled
    
    # Uniform sampling
    indices = np.linspace(0, total_valid - 1, num_frames, dtype=int)
    sampled = [frame_paths[i] for i in indices]
    
    return sampled


def extract_frames_from_video(video_path, num_frames=8, output_dir=None):
    """
    ULTRA-ROBUST frame extraction:
    1. Read ALL frames sequentially (no seek issues)
    2. Keep only valid frames
    3. Sample uniformly from valid frames
    
    Args:
        video_path: Path to input video
        num_frames: Number of frames to extract
        output_dir: Directory to save frames
        
    Returns:
        List of frame paths (exactly num_frames)
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='nsgvd_frames_')
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract ALL valid frames
    all_valid_frames = extract_all_valid_frames(video_path, output_dir)
    
    if len(all_valid_frames) == 0:
        raise ValueError(f"No valid frames could be extracted from {video_path}")
    
    # Step 2: Sample uniformly
    sampled_frames = sample_frames_uniformly(all_valid_frames, num_frames)
    
    # Step 3: Pad if needed
    if len(sampled_frames) < num_frames:
        print(f"  Padding from {len(sampled_frames)} to {num_frames} frames")
        while len(sampled_frames) < num_frames:
            # Copy last valid frame
            last_frame_path = sampled_frames[-1]
            padded_idx = len(sampled_frames)
            padded_path = os.path.join(output_dir, f"padded{padded_idx:05d}.jpg")
            
            import shutil
            shutil.copy(last_frame_path, padded_path)
            sampled_frames.append(padded_path)
    
    # Step 4: Rename to sequential order
    final_frames = []
    for idx, frame_path in enumerate(sampled_frames[:num_frames]):
        final_path = os.path.join(output_dir, f"final{idx:04d}.jpg")
        import shutil
        shutil.copy(frame_path, final_path)
        final_frames.append(final_path)
    
    return final_frames


def extract_frames_ffmpeg(video_path, num_frames=8, output_dir=None):
    """
    Extract frames using ffmpeg - fallback to robust method if incomplete
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='nsgvd_frames_')
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        total_frames = get_video_frame_count(video_path)
        
        if total_frames == 0:
            raise ValueError(f"Video has 0 frames")
        
        if total_frames < num_frames:
            actual_num_frames = total_frames
        else:
            actual_num_frames = num_frames
        
        # FFmpeg extraction
        if total_frames == actual_num_frames:
            select_expr = "1"
        else:
            interval = total_frames / actual_num_frames
            select_expr = f"not(mod(n,{int(interval)}))"
        
        output_pattern = os.path.join(output_dir, "frame%04d.jpg")
        cmd = (
            f"ffmpeg -i '{video_path}' "
            f"-vf \"select='{select_expr}',setpts=N/FRAME_RATE/TB\" "
            f"-vsync vfr "
            f"-frames:v {actual_num_frames} "
            f"'{output_pattern}' "
            f"-loglevel error -y 2>&1"
        )
        
        ret = os.system(cmd)
        
        if ret != 0:
            raise Exception("FFmpeg failed")
        
        frame_paths = sorted([
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.endswith('.jpg')
        ])
        
        if len(frame_paths) == 0:
            raise Exception("No frames extracted")
        
        # If FFmpeg got less than 90% of requested frames, use robust method
        if len(frame_paths) < int(num_frames * 0.9):
            print(f"  FFmpeg only got {len(frame_paths)}/{num_frames}, using robust sequential read")
            return extract_frames_from_video(video_path, num_frames, output_dir)
        
        # FFmpeg got most frames - just pad the rest if needed
        if len(frame_paths) < num_frames:
            print(f"  FFmpeg got {len(frame_paths)}/{num_frames}, padding remainder")
            while len(frame_paths) < num_frames:
                import shutil
                last_frame = frame_paths[-1]
                padded_idx = len(frame_paths)
                padded_path = os.path.join(output_dir, f"frame{padded_idx:04d}.jpg")
                shutil.copy(last_frame, padded_path)
                frame_paths.append(padded_path)
        
        return frame_paths[:num_frames]
        
    except Exception as e:
        print(f"  FFmpeg failed ({str(e)[:40]}), using robust sequential read")
        return extract_frames_from_video(video_path, num_frames, output_dir)


def extract_and_preprocess_frames(video_path, num_frames=8, img_size=224):
    """
    Extract frames and convert to tensor - ULTRA ROBUST VERSION
    
    Strategy:
    1. Try FFmpeg first (fast)
    2. If < 90% success, read ALL frames sequentially
    3. Sample uniformly from valid frames
    4. Pad if needed
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        img_size: Target image size
        
    Returns:
        Tensor of shape (num_frames, 3, img_size, img_size) - GUARANTEED
    """
    temp_dir = None
    
    try:
        temp_dir = tempfile.mkdtemp(prefix='nsgvd_frames_')
        
        # Extract frames with robust fallback
        frame_paths = extract_frames_ffmpeg(video_path, num_frames, temp_dir)
        
        if len(frame_paths) != num_frames:
            raise ValueError(
                f"Frame extraction returned {len(frame_paths)} frames, expected {num_frames}"
            )
        
        # Preprocessing transform
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        
        # Load and preprocess frames
        frames = []
        failed_count = 0
        
        for idx, frame_path in enumerate(frame_paths):
            try:
                img = Image.open(frame_path).convert('RGB')
                frame_tensor = transform(img)
                
                if frame_tensor.shape != (3, img_size, img_size):
                    failed_count += 1
                    if len(frames) > 0:
                        frames.append(frames[-1].clone())
                    continue
                
                frames.append(frame_tensor)
                
            except Exception as e:
                failed_count += 1
                if len(frames) > 0:
                    frames.append(frames[-1].clone())
                continue
        
        # Pad if needed
        if len(frames) < num_frames:
            if len(frames) == 0:
                raise ValueError(f"Failed to load ANY frames from {video_path}")
            
            while len(frames) < num_frames:
                frames.append(frames[-1].clone())
        
        if failed_count > 0:
            print(f"  Note: {failed_count} frames had loading issues, used padding")
        
        # Stack to tensor
        video_tensor = torch.stack(frames[:num_frames], dim=0)
        
        # Final validation
        expected_shape = (num_frames, 3, img_size, img_size)
        if video_tensor.shape != expected_shape:
            raise ValueError(
                f"Output tensor has wrong shape: {video_tensor.shape}, expected {expected_shape}"
            )
        
        return video_tensor
        
    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass


def load_frames_from_directory(frame_dir, num_frames=8, img_size=224):
    """Load frames from a directory"""
    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {frame_dir}")
    
    if len(frame_paths) > num_frames:
        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
        frame_paths = [frame_paths[i] for i in indices]
    
    while len(frame_paths) < num_frames:
        frame_paths.append(frame_paths[-1])
    
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    
    frames = []
    for frame_path in frame_paths[:num_frames]:
        img = Image.open(frame_path).convert('RGB')
        frame_tensor = transform(img)
        frames.append(frame_tensor)
    
    video_tensor = torch.stack(frames, dim=0)
    return video_tensor


def batch_extract_frames(video_dir, output_base_dir, num_frames=8):
    """Extract frames from all videos in a directory"""
    from tqdm import tqdm
    
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
    """Preprocess a single frame"""
    frame_img = Image.open(frame_path).convert('RGB')
    frame_img = frame_img.resize(input_shape, Image.Resampling.LANCZOS)
    
    frame_tensor = torch.from_numpy(np.array(frame_img)).float()
    frame_tensor = frame_tensor / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)
    
    return frame_tensor


def process_video_batch(video_paths, num_frames=8, img_size=224):
    """Process multiple videos into a batch"""
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
    
    batch_tensor = torch.stack(batch, dim=0)
    return batch_tensor


# Test function
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <video_path> [num_frames]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    print("="*70)
    print(f"Testing ULTRA-ROBUST frame extraction")
    print(f"Video: {video_path}")
    print(f"Requested frames: {num_frames}")
    print("="*70)
    
    try:
        total = get_video_frame_count(video_path)
        print(f"\nVideo info: {total} total frames")
        
        print(f"\nExtracting {num_frames} frames with sequential read fallback...")
        video_tensor = extract_and_preprocess_frames(video_path, num_frames)
        
        print(f"\n✅ SUCCESS!")
        print(f"  Shape: {video_tensor.shape} (expected: ({num_frames}, 3, 224, 224))")
        print(f"  Range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
        print(f"  Match: {'✓' if video_tensor.shape == (num_frames, 3, 224, 224) else '✗'}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# def create_logger(log_path):
#     # Create logger object
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

#     # Create file handler and set the formatter
#     fh = logging.FileHandler(log_path)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)

#     # Add the file handler to the logger
#     logger.addHandler(fh)

#     # Add a stream handler to print to console
#     sh = logging.StreamHandler()
#     sh.setFormatter(formatter)
#     logger.addHandler(sh)

#     return logger

# def get_keypts(image, face, predictor, face_detector):
#     # detect the facial landmarks for the selected face
#     shape = predictor(image, face)
    
#     # select the key points for the eyes, nose, and mouth
#     leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
#     reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
#     nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
#     lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
#     rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
#     pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

#     return pts

# def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
#     def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
#         M = None
#         target_size = [112, 112]
#         dst = np.array([
#             [30.2946, 51.6963],
#             [65.5318, 51.5014],
#             [48.0252, 71.7366],
#             [33.5493, 92.3655],
#             [62.7299, 92.2041]], dtype=np.float32)

#         if target_size[1] == 112:
#             dst[:, 0] += 8.0

#         dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
#         dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

#         target_size = outsize

#         margin_rate = scale - 1
#         x_margin = target_size[0] * margin_rate / 2.
#         y_margin = target_size[1] * margin_rate / 2.

#         # move
#         dst[:, 0] += x_margin
#         dst[:, 1] += y_margin

#         # resize
#         dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
#         dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

#         src = landmark.astype(np.float32)

#         # use skimage tranformation
#         tform = trans.SimilarityTransform()
#         tform.estimate(src, dst)
#         M = tform.params[0:2, :]

#         img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

#         if outsize is not None:
#             img = cv2.resize(img, (outsize[1], outsize[0]))
        
#         if mask is not None:
#             mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
#             mask = cv2.resize(mask, (outsize[1], outsize[0]))
#             return img, mask
#         else:
#             return img, None

#     # Convert to rgb
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Detect with dlib
#     faces = face_detector(rgb, 1)
#     if len(faces):
#         # For now only take the biggest face
#         face = max(faces, key=lambda rect: rect.width() * rect.height())
        
#         # Get the landmarks/parts for the face in box d only with the five key points
#         landmarks = get_keypts(rgb, face, predictor, face_detector)

#         # Align and crop the face
#         cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
#         cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
#         # Extract the all landmarks from the aligned face
#         face_align = face_detector(cropped_face, 1)
#         if len(face_align) == 0:
#             return None, None, None
#         landmark = predictor(cropped_face, face_align[0])
#         landmark = face_utils.shape_to_np(landmark)

#         return cropped_face, landmark, mask_face
#     else:
#         return None, None, None

# def facecrop(video_path):
#     face_detector = dlib.get_frontal_face_detector()
#     predictor_path = f'{base_path}/dfb/shape_predictor_81_face_landmarks.dat'

#     face_predictor = dlib.shape_predictor(predictor_path)

#     # Open the video file
#     cap_org = cv2.VideoCapture(str(video_path))

#     # Get the number of frames in the video
#     frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
#     # Get the mode
#     frame_idxs = np.linspace(0, frame_count_org - 1, 32, endpoint=True, dtype=int)

#     face_frames = []
#     face_crops = []
#     img_frames = []
#     # Iterate through the frames
#     for cnt_frame in range(frame_count_org):
#         _, frame_org = cap_org.read()

#         # Check if the frame is one of the frames to extract
#         if cnt_frame not in frame_idxs:
#             continue

#         if frame_org is not None:
#             cropped_face, _, _ = extract_aligned_face_dlib(face_detector, face_predictor, frame_org)
#         else:
#             continue

#         if cropped_face is not None:
#             face_crops.append(cropped_face)
#             face_frames.append(frame_org)
#         else:
#             frame_org = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
#             img_frames.append(frame_org)
#             continue
        
#     # Release the video capture
#     cap_org.release()

#     return face_crops, face_frames, img_frames


# if __name__ == '__main__':
#     import argparse
    
#     parser = argparse.ArgumentParser(description='NSG-VD Frame Extraction Utilities')
#     parser.add_argument(
#         '--mode',
#         type=str,
#         required=True,
#         choices=['extract', 'batch_extract', 'load_test'],
#         help='Operation mode'
#     )
#     parser.add_argument(
#         '--video_path',
#         type=str,
#         help='Path to video file (for extract mode)'
#     )
#     parser.add_argument(
#         '--video_dir',
#         type=str,
#         help='Directory containing videos (for batch_extract mode)'
#     )
#     parser.add_argument(
#         '--frame_dir',
#         type=str,
#         help='Directory containing frames (for load_test mode)'
#     )
#     parser.add_argument(
#         '--output_dir',
#         type=str,
#         help='Output directory'
#     )
#     parser.add_argument(
#         '--num_frames',
#         type=int,
#         default=8,
#         help='Number of frames to extract'
#     )
    
#     args = parser.parse_args()
    
#     if args.mode == 'extract':
#         if not args.video_path:
#             print("Error: --video_path required for extract mode")
#             sys.exit(1)
        
#         print(f"Extracting {args.num_frames} frames from {args.video_path}")
#         frame_paths = extract_frames_from_video(
#             args.video_path,
#             num_frames=args.num_frames,
#             output_dir=args.output_dir
#         )
#         print(f"Extracted {len(frame_paths)} frames:")
#         for path in frame_paths:
#             print(f"  {path}")
    
#     elif args.mode == 'batch_extract':
#         if not args.video_dir or not args.output_dir:
#             print("Error: --video_dir and --output_dir required for batch_extract mode")
#             sys.exit(1)
        
#         print(f"Batch extracting frames from {args.video_dir}")
#         batch_extract_frames(args.video_dir, args.output_dir, args.num_frames)
#         print("Done!")
    
#     elif args.mode == 'load_test':
#         if not args.frame_dir:
#             print("Error: --frame_dir required for load_test mode")
#             sys.exit(1)
        
#         print(f"Loading frames from {args.frame_dir}")
#         video_tensor = load_frames_from_directory(args.frame_dir, args.num_frames)
#         print(f"Loaded tensor shape: {video_tensor.shape}")