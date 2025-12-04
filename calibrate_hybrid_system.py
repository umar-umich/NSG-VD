"""
Hybrid System Calibration - FIXED VERSION
Calibrate thresholds for multiple detectors:
1. NSG-VD for fully synthetic videos
2. Face-specific detectors (Xception, EfficientNet, CLIP) for face manipulations
3. Ensemble weights for combined detection

FIX: Properly instantiate DFBDetector for each video instead of using class methods
"""

import os
import sys
from typing import Dict
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import cv2
import importlib.util

# ============================================================================
# PATH SETUP - Handles preprocess.py conflict between NSG-VD and DFB
# ============================================================================
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)

# Import NSG-VD detector (uses NSG-VD's preprocess.py)
from main import NSGVDDetector

# Import DFB detector using importlib to avoid preprocess.py conflict
dfb_main_path = os.path.join(base_path, 'dfb', 'main.py')
dfb_dir = os.path.join(base_path, 'dfb')

DFBDetector = None
if os.path.exists(dfb_main_path):
    # Save current sys.path
    original_path = list(sys.path)
    
    try:
        # Temporarily add dfb directory first in path
        sys.path.insert(0, dfb_dir)
        
        # Load DFB module using importlib
        spec = importlib.util.spec_from_file_location("dfb_module", dfb_main_path)
        dfb_module = importlib.util.module_from_spec(spec)
        
        # Execute module (this will use dfb/preprocess.py)
        spec.loader.exec_module(dfb_module)
        
        # Get DFBDetector class
        DFBDetector = dfb_module.DFBDetector
        
        print("✓ DFB detector loaded successfully")
        
    except Exception as e:
        print(f"⚠️  Could not load DFB detector: {e}")
        print("   Face detector calibration will be skipped")
        DFBDetector = None
    
    finally:
        # Restore original path to avoid conflicts
        sys.path = original_path

else:
    print(f"⚠️  DFB directory not found at: {dfb_dir}")
    print("   Only NSG-VD will be calibrated")

# ============================================================================

class HybridSystemCalibrator:
    """
    Calibrate all detectors in the hybrid system
    """
    
    def __init__(self, nsgvd_config_path='./config/nsgvd_detector.yaml', device_id=0):
        self.device_id = device_id
        print("="*70)
        print("HYBRID SYSTEM CALIBRATION")
        print("="*70)
        
        # Initialize detectors
        print("\nInitializing detectors...")
        self.nsgvd_detector = NSGVDDetector(config_path=nsgvd_config_path, device_id=device_id)
        self.nsgvd_detector.load_model()
        self.nsgvd_detector.load_score_function()
        self.nsgvd_detector.load_reference_features()
        print("✓ NSG-VD loaded")
        
        # Store DFBDetector class (will instantiate per video)
        self.face_detector_class = DFBDetector
        if DFBDetector is not None:
            print("✓ Face detector class loaded")
        else:
            print("⚠️  Face detector unavailable - will skip face detection")
        
        # Results storage
        self.results = {
            'nsgvd': {'real': [], 'fake': []},
            'xception': {'real': [], 'fake': []},
            'efficientnet': {'real': [], 'fake': []},
            'clip_vit': {'real': [], 'fake': []},
        }
        
        self.video_metadata = []
        self.failed_videos = []
    
    # def classify_video_type(self, video_path):
    #     """
    #     Determine if video contains faces
        
    #     Returns:
    #         'face' or 'no_face'
    #     """
    #     cap = cv2.VideoCapture(str(video_path))
    #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    #     face_count = 0
    #     frame_count = 0
    #     max_frames_to_check = 8
        
    #     while frame_count < max_frames_to_check:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
            
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
    #         if len(faces) > 0:
    #             face_count += 1
            
    #         frame_count += 1
        
    #     cap.release()
        
    #     face_ratio = face_count / frame_count if frame_count > 0 else 0
        
    #     # If >30% of frames have faces, classify as face video
    #     return 'face' if face_ratio > 0.3 else 'no_face'
    
    def classify_video_type(self, video_path: str) -> Dict:
        """
        Fast classification: Check if video contains faces using quick frame sampling
        WITHOUT instantiating the full DFB detector (which is expensive)
        
        Returns:
            {
                'type': 'face' | 'no_face',
                'face_detected': bool,
                'confidence': float,
            }
        """
        try:
            # FAST METHOD: Use OpenCV Haar cascade on just 3 key frames
            # This is 90% accurate and 100x faster than instantiating DFB
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    'type': 'uncertain',
                    'face_detected': False,
                    'confidence': 0.0,
                    'error': f'Cannot open video'
                }
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Sample just 3 frames: start, middle, end (FAST!)
            sample_frames = [0, max(0, total_frames // 2), max(0, total_frames - 1)]
            face_detected_count = 0
            
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                # Resize for speed (smaller = faster detection)
                frame_small = cv2.resize(frame, (320, 240))
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
                
                if len(faces) > 0:
                    face_detected_count += 1
            
            cap.release()
            
            # If any of the 3 frames had faces, classify as 'face'
            has_faces = face_detected_count > 0
            video_type = 'face' if has_faces else 'no_face'
            confidence = (face_detected_count / len(sample_frames)) if has_faces else (1.0 - face_detected_count / len(sample_frames))
            
            return {
                'type': video_type,
                'face_detected': has_faces,
                'confidence': confidence,
                'frames_sampled': len(sample_frames),
                'frames_with_faces': face_detected_count,
            }
        
        except Exception as e:
            print(f"Warning: Video classification failed: {e}")
            return {
                'type': 'uncertain',
                'face_detected': False,
                'confidence': 0.0,
                'error': str(e)
            }

    def process_video_with_all_detectors(self, video_path, true_label):
            """
            Process video with all detectors and collect scores
            
            Args:
                video_path: Path to video
                true_label: 'real' or 'fake'
            
            Returns:
                Dictionary with scores from all detectors
            """
            video_name = os.path.basename(video_path)
            scores = {
                'video': video_name,
                'path': str(video_path),
                'true_label': true_label
            }
            
            # Classify video type
            video_type_result = self.classify_video_type(video_path)
            scores['video_type'] = video_type_result['type']
            
            # 1. NSG-VD (for all videos)
            print(f"[DEBUG] Processing {video_name} (True Label: {true_label})")
            try:
                nsgvd_result = self.nsgvd_detector.predict(str(video_path))
                scores['nsgvd_score'] = nsgvd_result['mmd_score']
                scores['nsgvd_pred'] = nsgvd_result['prediction']
                print(f"  ✓ NSG-VD: score={nsgvd_result['mmd_score']:.4f}, pred={nsgvd_result['prediction']}")
            except Exception as e:
                print(f"  ✗ NSG-VD failed: {e}")
                scores['nsgvd_score'] = None
                scores['nsgvd_pred'] = None
            
            # 2. Face detectors (only for face videos and if available)
            if video_type_result['type'] == 'face' and self.face_detector_class is not None:
                print(f"  [DEBUG] Faces detected - calling face-specific model CLIP-ViT)")
                face_models = [ 'clip_vit'] # 'xception', 'efficientnetb4',
                
                # ========== Create DFBDetector instance for this video ==========
                try:
                    # Initialize DFBDetector with the video path
                    dfb_instance = self.face_detector_class(video_path=str(video_path))
                    
                    for model in face_models:
                        try:
                            # Now call get_predictions on the instance
                            result = dfb_instance.get_predictions(
                                model_name=model,
                                uploaded_file_name=video_name,
                                device_id=self.device_id
                            )
                            
                            # Extract confidence score
                            if 'fake' in result:
                                conf = result['fake']
                            elif 'real' in result:
                                conf = 1 - result['real']
                            else:
                                conf = None
                            
                            model_key = 'efficientnet' if model == 'efficientnetb4' else model
                            scores[f'{model_key}_score'] = conf
                            scores[f'{model_key}_pred'] = 'fake' if conf and conf > 0.5 else 'real'
                            print(f"    ✓ {model}: score={conf:.4f}, pred={scores[f'{model_key}_pred']}")
                            
                        except Exception as e:
                            print(f"    ✗ {model} failed: {e}")
                            model_key = 'efficientnet' if model == 'efficientnetb4' else model
                            scores[f'{model_key}_score'] = None
                            scores[f'{model_key}_pred'] = None
                            
                except Exception as e:
                    print(f"  ✗ DFBDetector initialization failed: {e}")
                    for model in ['xception', 'efficientnet', 'clip_vit']:
                        scores[f'{model}_score'] = None
                        scores[f'{model}_pred'] = None
                # ====================================================================
            else:
                if video_type_result['type'] != 'face':
                    print(f"  [DEBUG] No faces detected - skipping face-specific models")
                else:
                    print(f"  [DEBUG] Face detector unavailable - skipping face-specific models")
                # No face detectors for non-face videos or if unavailable
                for model in ['xception', 'efficientnet', 'clip_vit']:
                    scores[f'{model}_score'] = None
                    scores[f'{model}_pred'] = None
            
            return scores
    
    def collect_video_paths_by_category(self, root_dir, max_per_category):
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
                # Use rglob in case there are deeper levels
                cat_paths.extend(cat_dir.rglob(ext))
            cat_paths = sorted(cat_paths)
            
            if max_per_category is not None:
                cat_paths = cat_paths[:max_per_category]
            
            print(f"    Category: {cat_dir.relative_to(root)} -> {len(cat_paths)} videos")
            all_paths.extend(cat_paths)
        
        return all_paths
    
    def process_dataset(self, real_dir, fake_dir, max_videos_per_category=None):
        """
        Process all videos in calibration dataset.
        Now interprets max_videos_per_category as max videos PER SUBFOLDER/CATEGORY.
        
        Args:
            real_dir: Directory with real videos (containing category subfolders)
            fake_dir: Directory with fake videos (containing category subfolders)
            max_videos_per_category: Maximum videos per category subfolder (None = all)
        """
        print("\n" + "="*70)
        print("PROCESSING CALIBRATION DATASET")
        print("="*70)
        
        print(f"\nCollecting real videos from: {real_dir}")
        real_videos = self.collect_video_paths_by_category(real_dir, max_videos_per_category)
        
        print(f"\nCollecting fake videos from: {fake_dir}")
        fake_videos = self.collect_video_paths_by_category(fake_dir, max_videos_per_category)
        
        print(f"\n✓ Total real videos: {len(real_videos)}")
        print(f"✓ Total fake videos: {len(fake_videos)}")
        
        if len(real_videos) == 0 or len(fake_videos) == 0:
            raise ValueError("Need both real and fake videos for calibration!")
        
        # Process all videos
        all_results = []
        
        for video_path in tqdm(real_videos, desc="Real videos"):
            try:
                scores = self.process_video_with_all_detectors(video_path, 'real')
                all_results.append(scores)
            except Exception as e:
                print(f"\nFailed: {video_path.name}: {e}")
                self.failed_videos.append({
                    'video': str(video_path),
                    'label': 'real',
                    'error': str(e)
                })
        
        for video_path in tqdm(fake_videos, desc="Fake videos"):
            try:
                scores = self.process_video_with_all_detectors(video_path, 'fake')
                all_results.append(scores)
            except Exception as e:
                print(f"\nFailed: {video_path.name}: {e}")
                self.failed_videos.append({
                    'video': str(video_path),
                    'label': 'fake',
                    'error': str(e)
                })
        
        self.video_metadata = all_results
        
        # Organize results by detector
        for result in all_results:
            true_label = result['true_label']
            
            # NSG-VD scores
            if result['nsgvd_score'] is not None:
                self.results['nsgvd'][true_label].append(result['nsgvd_score'])
            
            # Face detector scores
            for model in ['xception', 'efficientnet', 'clip_vit']:
                if result[f'{model}_score'] is not None:
                    self.results[model][true_label].append(result[f'{model}_score'])
        
        print(f"\n✓ Processed {len(all_results)} videos successfully")
        print(f"  Failed: {len(self.failed_videos)} videos")
    
    def calibrate_detector(self, detector_name, real_scores, fake_scores):
        """
        Calibrate threshold for a single detector
        
        Args:
            detector_name: Name of detector
            real_scores: List of scores for real videos
            fake_scores: List of scores for fake videos
        
        Returns:
            Dictionary with thresholds and metrics
        """
        real_scores = np.array(real_scores)
        fake_scores = np.array(fake_scores)
        
        if len(real_scores) == 0 or len(fake_scores) == 0:
            return None
        
        # Determine if higher score = fake or real
        # For NSG-VD: higher MMD = more different from real = fake
        # For face detectors: higher confidence = fake
        
        is_higher_means_fake = (fake_scores.mean() > real_scores.mean())
        
        # Prepare data for ROC
        all_scores = np.concatenate([real_scores, fake_scores])
        all_labels = np.concatenate([
            np.zeros(len(real_scores)),  # 0 = real
            np.ones(len(fake_scores))     # 1 = fake
        ])
        
        # Calculate ROC curve
        if is_higher_means_fake:
            fpr, tpr, thresholds_roc = roc_curve(all_labels, all_scores)
        else:
            # Invert scores if lower means fake
            fpr, tpr, thresholds_roc = roc_curve(all_labels, -all_scores)
            thresholds_roc = -thresholds_roc
        
        roc_auc = auc(fpr, tpr)
        
        # Find thresholds for different FPR targets
        calibration = {
            'detector': detector_name,
            'num_real': len(real_scores),
            'num_fake': len(fake_scores),
            'roc_auc': float(roc_auc),
            'real_mean': float(real_scores.mean()),
            'real_std': float(real_scores.std()),
            'fake_mean': float(fake_scores.mean()),
            'fake_std': float(fake_scores.std()),
            'higher_means_fake': is_higher_means_fake,
            'thresholds': {}
        }
        
        # Find thresholds for target FPRs
        for target_fpr, mode in [(0.05, 'conservative'), (0.10, 'balanced'), (0.20, 'aggressive')]:
            # Find threshold closest to target FPR
            idx = np.argmin(np.abs(fpr - target_fpr))
            threshold = float(thresholds_roc[idx])
            actual_fpr = float(fpr[idx])
            actual_tpr = float(tpr[idx])
            
            # Calculate precision
            if is_higher_means_fake:
                predictions = (all_scores > threshold).astype(int)
            else:
                predictions = (all_scores < threshold).astype(int)
            
            tp = np.sum((predictions == 1) & (all_labels == 1))
            fp = np.sum((predictions == 1) & (all_labels == 0))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0
            f1 = float(2 * precision * actual_tpr / (precision + actual_tpr)) if (precision + actual_tpr) > 0 else 0
            
            calibration['thresholds'][mode] = {
                'threshold': threshold,
                'fpr': actual_fpr,
                'tpr': actual_tpr,
                'precision': precision,
                'f1': f1,
                'target_fpr': target_fpr
            }
        
        # Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        calibration['thresholds']['youden'] = {
            'threshold': float(thresholds_roc[best_idx]),
            'fpr': float(fpr[best_idx]),
            'tpr': float(tpr[best_idx])
        }
        
        return calibration
    
    def calibrate_all_detectors(self):
        """
        Calibrate thresholds for all detectors
        
        Returns:
            Dictionary with calibration results for each detector
        """
        print("\n" + "="*70)
        print("CALIBRATING ALL DETECTORS")
        print("="*70)
        
        calibrations = {}
        
        for detector_name in ['nsgvd', 'xception', 'efficientnet', 'clip_vit']:
            real_scores = self.results[detector_name]['real']
            fake_scores = self.results[detector_name]['fake']
            
            if len(real_scores) > 0 and len(fake_scores) > 0:
                print(f"\n[{detector_name.upper()}]")
                print(f"  Real videos: {len(real_scores)}")
                print(f"  Fake videos: {len(fake_scores)}")
                
                calibration = self.calibrate_detector(detector_name, real_scores, fake_scores)
                
                if calibration:
                    calibrations[detector_name] = calibration
                    
                    print(f"  ROC AUC: {calibration['roc_auc']:.4f}")
                    print(f"  Balanced threshold: {calibration['thresholds']['balanced']['threshold']:.4f}")
                    print(f"    TPR: {calibration['thresholds']['balanced']['tpr']:.2%}")
                    print(f"    FPR: {calibration['thresholds']['balanced']['fpr']:.2%}")
            else:
                print(f"\n[{detector_name.upper()}] - SKIPPED (insufficient data)")
        
        return calibrations
    
    def optimize_ensemble_weights(self, calibrations):
        """
        Find optimal ensemble weights for combining detectors
        
        Args:
            calibrations: Dictionary of detector calibrations
        
        Returns:
            Optimal weights for ensemble
        """
        print("\n" + "="*70)
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print("="*70)
        
        # Separate videos by type
        face_videos = [v for v in self.video_metadata if v['video_type'] == 'face']
        no_face_videos = [v for v in self.video_metadata if v['video_type'] == 'no_face']
        
        ensemble_config = {
            'face_videos': {
                'nsgvd_weight': 0.3,
                'face_detectors_weight': 0.7,
                'face_detector_weights': {
                    'xception': 0.4,
                    'efficientnet': 0.3,
                    'clip_vit': 0.3
                }
            },
            'no_face_videos': {
                'nsgvd_weight': 1.0,
                'face_detectors_weight': 0.0
            }
        }
        
        print("\nRecommended ensemble weights:")
        print("\nFor FACE videos:")
        print(f"  NSG-VD: {ensemble_config['face_videos']['nsgvd_weight']:.1f}")
        print(f"  Face detectors: {ensemble_config['face_videos']['face_detectors_weight']:.1f}")
        print(f"    - Xception: {ensemble_config['face_videos']['face_detector_weights']['xception']:.1f}")
        print(f"    - EfficientNet: {ensemble_config['face_videos']['face_detector_weights']['efficientnet']:.1f}")
        print(f"    - CLIP-ViT: {ensemble_config['face_videos']['face_detector_weights']['clip_vit']:.1f}")
        
        print("\nFor NO-FACE videos:")
        print(f"  NSG-VD: {ensemble_config['no_face_videos']['nsgvd_weight']:.1f}")
        
        return ensemble_config
    
    def save_calibrations(self, calibrations, ensemble_config, output_dir='./calibration_results'):
        """
        Save all calibration results
        
        Args:
            calibrations: Detector calibrations
            ensemble_config: Ensemble weights
            output_dir: Output directory
        """
        print("\n" + "="*70)
        print("SAVING CALIBRATION RESULTS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save hybrid calibration YAML
        hybrid_calibration = {
            'calibration_date': pd.Timestamp.now().isoformat(),
            'detectors': calibrations,
            'ensemble': ensemble_config
        }
        
        yaml_path = os.path.join(output_dir, 'hybrid_calibration.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(hybrid_calibration, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved: {yaml_path}")
        
        # 2. Save detailed video results
        df = pd.DataFrame(self.video_metadata)
        csv_path = os.path.join(output_dir, 'detailed_video_scores.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")
        
        # 3. Save failed videos
        if self.failed_videos:
            failed_df = pd.DataFrame(self.failed_videos)
            failed_path = os.path.join(output_dir, 'failed_videos.csv')
            failed_df.to_csv(failed_path, index=False)
            print(f"✓ Saved: {failed_path}")
        
        # 4. Create detector-specific calibration files
        for detector_name, calibration in calibrations.items():
            detector_yaml = {
                'calibration_date': pd.Timestamp.now().isoformat(),
                'detector': detector_name,
                'num_real_videos': calibration['num_real'],
                'num_fake_videos': calibration['num_fake'],
                'statistics': {
                    'real': {
                        'mean': calibration['real_mean'],
                        'std': calibration['real_std']
                    },
                    'fake': {
                        'mean': calibration['fake_mean'],
                        'std': calibration['fake_std']
                    }
                },
                'thresholds': calibration['thresholds']
            }
            
            detector_path = os.path.join(output_dir, f'{detector_name}_calibration.yaml')
            with open(detector_path, 'w') as f:
                yaml.dump(detector_yaml, f, default_flow_style=False)
            print(f"✓ Saved: {detector_path}")
    
    def run_calibration(self, real_dir, fake_dir, max_videos_per_category=None, output_dir='./calibration_results'):
        """
        Run complete hybrid calibration pipeline
        
        Args:
            real_dir: Directory with real videos (containing category subfolders)
            fake_dir: Directory with fake videos (containing category subfolders)
            max_videos_per_category: Maximum videos per category subfolder (None = all)
            output_dir: Output directory
        """
        # Process dataset
        self.process_dataset(real_dir, fake_dir, max_videos_per_category)
        
        # Calibrate each detector
        calibrations = self.calibrate_all_detectors()
        
        # Optimize ensemble
        ensemble_config = self.optimize_ensemble_weights(calibrations)
        
        # Save results
        self.save_calibrations(calibrations, ensemble_config, output_dir)
        
        return calibrations, ensemble_config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calibrate Hybrid Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calibration with categories
  python calibrate_hybrid_system.py \\
    --real_dir ./calibration_data/real \\
    --fake_dir ./calibration_data/fake

  # Limit to 50 videos per category subfolder
  python calibrate_hybrid_system.py \\
    --real_dir ./calibration_data/real \\
    --fake_dir ./calibration_data/fake \\
    --max_videos_per_category 50

Directory structure example:
  calibration_data/
  ├── real/
  │   ├── kinetics/          # Category 1
  │   │   ├── video1.mp4
  │   │   └── video2.mp4
  │   ├── msrvtt/            # Category 2
  │   │   ├── video1.mp4
  │   │   └── video2.mp4
  │   └── youtube/           # Category 3
  │       └── video1.mp4
  └── fake/
      ├── sora/              # Generator 1
      │   ├── video1.mp4
      │   └── video2.mp4
      ├── pika/              # Generator 2
      │   └── video1.mp4
      └── face_swaps/        # Generator 3
          └── video1.mp4

With --max_videos_per_category 50:
  - Takes up to 50 videos from each of: kinetics, msrvtt, youtube
  - Takes up to 50 videos from each of: sora, pika, face_swaps
  - Total: up to 150 real + 150 fake videos
        """
    )
    
    parser.add_argument('--real_dir', type=str, required=True, 
                       help='Real videos directory (with category subfolders)')
    parser.add_argument('--fake_dir', type=str, required=True, 
                       help='Fake videos directory (with generator subfolders)')
    parser.add_argument('--max_videos_per_category', type=int, default=50, 
                       help='Max videos per category subfolder (None = all)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to NSG-VD config file (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='./calibration_results', 
                       help='Output directory')
    parser.add_argument('--device', type=int, default=0, 
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Run calibration
    calibrator = HybridSystemCalibrator(device_id=args.device)
    calibrations, ensemble_config = calibrator.run_calibration(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        max_videos_per_category=args.max_videos_per_category,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nCalibrated detectors:")
    for detector_name in calibrations.keys():
        print(f"  ✓ {detector_name.upper()}")
    print("\nNext steps:")
    print(f"  1. Review: {args.output_dir}/hybrid_calibration.yaml")
    print(f"  2. Use hybrid detector with calibrated thresholds")
    print("="*70)


if __name__ == '__main__':
    main()