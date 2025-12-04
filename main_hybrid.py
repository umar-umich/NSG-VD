"""
Hybrid Deepfake Detector
Combines NSG-VD (physics-based) with face-specific detectors
Handles: fully synthetic videos, face-swaps, and synthetic videos with faces

UPDATED: Added evaluation mode with CSV export and comprehensive metrics
"""

import torch
import numpy as np
import os
import sys
import yaml
import cv2
import pandas as pd
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# base_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(base_path)

import sys
import os
from pathlib import Path

base_path = os.path.dirname(os.path.abspath(__file__))

# CRITICAL: Import NSG-VD first (uses root preprocess.py)
# Add root to path TEMPORARILY for NSG-VD imports
sys.path.insert(0, base_path)

try:
    from preprocess import extract_frames_from_video
    from main import NSGVDDetector
    print("✓ NSG-VD components loaded (using root preprocess.py)")
except ImportError as e:
    print(f"✗ Failed to load NSG-VD: {e}")
    raise

# CRITICAL: Now handle DFB with isolated imports
dfb_dir = os.path.join(base_path, 'dfb')
dfb_main_path = os.path.join(dfb_dir, 'main.py')

DFBDetector = None

if os.path.exists(dfb_main_path):
    # Save original sys.path and modules
    original_path = sys.path.copy()
    original_modules = set(sys.modules.keys())
    
    try:
        # CRITICAL: Insert DFB dir at the VERY FRONT
        # This ensures dfb/preprocess.py is found BEFORE root preprocess.py
        sys.path.insert(0, dfb_dir)
        
        # Remove root from path temporarily to avoid conflicts
        if base_path in sys.path:
            sys.path.remove(base_path)
        
        # Use importlib to load DFB with isolated namespace
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dfb_module",
            dfb_main_path,
            submodule_search_locations=[dfb_dir]
        )
        dfb_module = importlib.util.module_from_spec(spec)
        
        # Register in sys.modules BEFORE executing
        sys.modules["dfb_module"] = dfb_module
        
        # Execute the module (now it will find dfb/preprocess.py first)
        spec.loader.exec_module(dfb_module)
        
        # Get the detector class
        DFBDetector = dfb_module.DFBDetector
        print("✓ DFB detector loaded successfully (using dfb/preprocess.py)")
        
    except ImportError as e:
        print(f"⚠️  Could not load DFB detector: {e}")
        print("   Face detector calibration will be skipped")
        print(f"   Error details: {e}")
        DFBDetector = None
        
    except Exception as e:
        print(f"⚠️  Unexpected error loading DFB: {e}")
        DFBDetector = None
        
    finally:
        # RESTORE sys.path to original state
        sys.path = original_path
        
        # Clean up any DFB-specific modules from sys.modules to avoid conflicts
        # Keep dfb_module but remove its internal imports
        modules_to_remove = [m for m in sys.modules.keys() 
                           if m not in original_modules and 'dfb' not in m]
        for mod in modules_to_remove:
            try:
                del sys.modules[mod]
            except KeyError:
                pass
else:
    print(f"⚠️  DFB directory not found at: {dfb_dir}")
    print("   Only NSG-VD will be used")



class HybridDeepfakeDetector:
    """
    Intelligent routing system for deepfake detection
    - Fully synthetic videos → NSG-VD (physics-based)
    - Face manipulations → Face-specific detectors
    - Mixed cases → Ensemble approach
    """
    
    def __init__(self, 
                 nsgvd_config_path='./config/nsgvd_detector.yaml',
                 calibration_config_path='./calibration_results/hybrid_calibration.yaml',
                 device_id=0):
        """
        Args:
            nsgvd_config_path: Path to NSG-VD config
            calibration_config_path: Path to hybrid calibration (from calibrate_hybrid_system.py)
            device_id: GPU device ID
        """
        self.device_id = device_id
        
        # Load NSG-VD detector (for fully synthetic)
        print("Loading NSG-VD detector...")
        self.nsgvd = NSGVDDetector(config_path=nsgvd_config_path, device_id=device_id)
        self.nsgvd.load_model()
        self.nsgvd.load_score_function()
        self.nsgvd.load_reference_features()
        print("✓ NSG-VD loaded")
        
        # Face detector class (lazy load per video)
        self.face_detector_class = DFBDetector
        if DFBDetector is not None:
            print("✓ Face detector class loaded")
        else:
            print("⚠️  Face detector unavailable - will only use NSG-VD")
        
        # Load calibrated thresholds and ensemble weights
        self.calibration = self._load_calibration(calibration_config_path)
        
        # Face detection model (for video classification)
        self.face_cascade = None  # Lazy load
        
    def _load_calibration(self, calibration_path):
        """Load hybrid calibration with detector-specific thresholds"""
        if os.path.exists(calibration_path):
            with open(calibration_path, 'r') as f:
                calibration = yaml.safe_load(f)
            print(f"✓ Loaded hybrid calibration from {calibration_path}")
            return calibration
        else:
            print(f"⚠️  No calibration found at {calibration_path}")
            print("   Using default thresholds (not recommended)")
            return {
                'detectors': {},
                'ensemble': {
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
            }
    
    def _get_face_cascade(self):
        """Lazy load OpenCV face cascade"""
        if self.face_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        return self.face_cascade
    
    def classify_video_type(self, video_path: str) -> Dict:
        """
        Classify video manipulation type
        
        Returns:
            {
                'type': 'fully_synthetic' | 'face_manipulation' | 'uncertain',
                'face_ratio': float,
                'confidence': float
            }
        """
        try:
            # Extract sample frames
            frame_paths = extract_frames_from_video(video_path, num_frames=8)
            
            face_cascade = self._get_face_cascade()
            face_detected_count = 0
            total_faces = 0
            
            for frame_path in frame_paths:
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    face_detected_count += 1
                    total_faces += len(faces)
            
            face_ratio = face_detected_count / len(frame_paths) if len(frame_paths) > 0 else 0
            
            # Classification logic
            if face_ratio < 0.25:
                video_type = 'fully_synthetic'
                confidence = 1.0 - face_ratio
            elif face_ratio > 0.6:
                video_type = 'face_manipulation'
                confidence = face_ratio
            else:
                video_type = 'uncertain'
                confidence = 0.5
            
            return {
                'type': video_type,
                'face_ratio': face_ratio,
                'confidence': confidence,
                'total_faces': total_faces,
                'frames_with_faces': face_detected_count,
            }
            
        except Exception as e:
            print(f"Warning: Video classification failed: {e}")
            return {
                'type': 'uncertain',
                'face_ratio': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def detect_with_nsgvd(self, video_path: str, mode: str = 'balanced') -> Dict:
        """
        Detect using NSG-VD with calibrated threshold
        
        Args:
            video_path: Path to video
            mode: 'conservative' | 'balanced' | 'aggressive'
        """
        # Get NSG-VD prediction
        result = self.nsgvd.get_predictions(video_path)
        mmd_score = result['mmd_score']
        
        # Get calibrated threshold for NSG-VD
        if 'nsgvd' in self.calibration.get('detectors', {}):
            nsgvd_cal = self.calibration['detectors']['nsgvd']
            if mode in nsgvd_cal.get('thresholds', {}):
                threshold = nsgvd_cal['thresholds'][mode]['threshold']
                is_higher_means_fake = nsgvd_cal.get('higher_means_fake', True)
            else:
                threshold = 1.0  # Default
                is_higher_means_fake = True
        else:
            # Fallback to default
            threshold = 1.0
            is_higher_means_fake = True
        
        # Apply threshold
        if is_higher_means_fake:
            prediction = 'fake' if mmd_score > threshold else 'real'
        else:
            prediction = 'fake' if mmd_score < threshold else 'real'
        
        confidence = abs(mmd_score - threshold) / threshold  # Normalized distance
        
        return {
            'prediction': prediction,
            'confidence': min(confidence, 1.0),
            'mmd_score': mmd_score,
            'threshold_used': threshold,
            'mode': mode,
            'detector': 'NSG-VD'
        }
    
    def detect_with_face_models(self, video_path: str, mode: str = 'balanced') -> Dict:
        """
        Detect using face-specific models with calibrated thresholds
        
        Returns:
            Aggregated prediction from multiple face detectors
        """
        if self.face_detector_class is None:
            return {
                'prediction': 'uncertain',
                'confidence': 0.0,
                'detector': 'Face-Models',
                'error': 'Face detector not available'
            }
        
        # Create instance for this video
        try:
            face_detector = self.face_detector_class(video_path=video_path)
        except Exception as e:
            return {
                'prediction': 'uncertain',
                'confidence': 0.0,
                'detector': 'Face-Models',
                'error': f'Failed to initialize face detector: {e}'
            }
        
        # Get predictions from multiple models
        models = {
            'xception': 'xception',
            'efficientnet': 'efficientnetb4',
            'clip_vit': 'clip_vit'
        }
        
        results = {}
        scores = {}
        
        for model_key, model_name in models.items():
            try:
                pred = face_detector.get_predictions(
                    model_name=model_name,
                    uploaded_file_name=os.path.basename(video_path),
                    device_id=self.device_id
                )
                
                # Extract confidence score (fake probability)
                if 'fake' in pred:
                    conf = pred['fake']
                elif 'real' in pred:
                    conf = 1 - pred['real']
                else:
                    conf = 0.5
                
                scores[model_key] = conf
                
                # Apply calibrated threshold
                if model_key in self.calibration.get('detectors', {}):
                    model_cal = self.calibration['detectors'][model_key]
                    if mode in model_cal.get('thresholds', {}):
                        threshold = model_cal['thresholds'][mode]['threshold']
                        is_higher_means_fake = model_cal.get('higher_means_fake', True)
                        
                        if is_higher_means_fake:
                            pred_label = 'fake' if conf > threshold else 'real'
                        else:
                            pred_label = 'fake' if conf < threshold else 'real'
                    else:
                        pred_label = 'fake' if conf > 0.5 else 'real'
                else:
                    pred_label = 'fake' if conf > 0.5 else 'real'
                
                results[model_key] = pred_label
                
            except Exception as e:
                print(f"Warning: {model_name} failed: {e}")
                results[model_key] = None
                scores[model_key] = None
        
        # Aggregate with calibrated weights
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return {
                'prediction': 'uncertain',
                'confidence': 0.0,
                'detector': 'Face-Models',
                'error': 'All face models failed'
            }
        
        # Use ensemble weights
        ensemble_weights = self.calibration.get('ensemble', {}).get('face_videos', {}).get('face_detector_weights', {
            'xception': 0.4,
            'efficientnet': 0.3,
            'clip_vit': 0.3
        })
        
        # Weighted average of scores
        weighted_score = 0
        total_weight = 0
        for model_key, score in scores.items():
            if score is not None:
                weight = ensemble_weights.get(model_key, 0.33)
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        prediction = 'fake' if weighted_score > 0.5 else 'real'
        
        return {
            'prediction': prediction,
            'confidence': abs(weighted_score - 0.5) * 2,  # 0-1 scale
            'weighted_score': weighted_score,
            'detector': 'Face-Models',
            'individual_results': results,
            'individual_scores': scores
        }
    
    def detect_ensemble(self, video_path: str, mode: str = 'balanced') -> Dict:
        """
        Ensemble detection: combine NSG-VD and face models
        
        Args:
            video_path: Path to video
            mode: Detection mode
        """
        # Get predictions from both detectors
        nsgvd_result = self.detect_with_nsgvd(video_path, mode=mode)
        face_result = self.detect_with_face_models(video_path, mode=mode)
        
        # Get ensemble weights
        ensemble_config = self.calibration.get('ensemble', {}).get('face_videos', {})
        weight_nsgvd = ensemble_config.get('nsgvd_weight', 0.3)
        weight_face = ensemble_config.get('face_detectors_weight', 0.7)
        
        nsgvd_score = 1.0 if nsgvd_result['prediction'] == 'fake' else 0.0
        face_score = 1.0 if face_result['prediction'] == 'fake' else 0.0
        
        # Weighted average
        combined_score = (nsgvd_score * weight_nsgvd + 
                         face_score * weight_face)
        
        prediction = 'fake' if combined_score >= 0.5 else 'real'
        
        # Combined confidence (average of individual confidences)
        combined_confidence = (nsgvd_result['confidence'] * weight_nsgvd + 
                              face_result['confidence'] * weight_face)
        
        return {
            'prediction': prediction,
            'confidence': combined_confidence,
            'combined_score': combined_score,
            'detector': 'Ensemble',
            'nsgvd_result': nsgvd_result,
            'face_result': face_result,
            'weights': {'nsgvd': weight_nsgvd, 'face': weight_face}
        }
    
    def predict(self, video_path: str, mode: str = 'balanced', 
                force_detector: str = None, verbose: bool = True) -> Dict:
        """
        Main prediction interface with intelligent routing
        
        Args:
            video_path: Path to video file
            mode: 'conservative' | 'balanced' | 'aggressive'
            force_detector: Override automatic routing ('nsgvd', 'face', 'ensemble')
            verbose: Print progress messages
        
        Returns:
            {
                'prediction': 'real' | 'fake' | 'uncertain',
                'confidence': float,
                'video_type': str,
                'detector_used': str,
                'details': dict,
                'flag': str (optional)
            }
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(video_path)}")
            print(f"Mode: {mode}")
        
        # Step 1: Classify video type (unless forced)
        if force_detector is None:
            if verbose:
                print("Classifying video type...")
            video_classification = self.classify_video_type(video_path)
            video_type = video_classification['type']
            if verbose:
                print(f"Video type: {video_type} (confidence: {video_classification['confidence']:.2f})")
        else:
            video_type = 'forced'
            video_classification = {'type': 'forced', 'confidence': 1.0}
        
        # Step 2: Route to appropriate detector
        if force_detector == 'nsgvd' or video_type == 'fully_synthetic':
            if verbose:
                print("Using NSG-VD detector...")
            result = self.detect_with_nsgvd(video_path, mode=mode)
            
        elif force_detector == 'face' or video_type == 'face_manipulation':
            if verbose:
                print("Using face-specific detectors...")
            result = self.detect_with_face_models(video_path, mode=mode)
            
        else:  # uncertain or ensemble
            if verbose:
                print("Using ensemble approach...")
            result = self.detect_ensemble(video_path, mode=mode)
        
        # Step 3: Add metadata
        result['video_type'] = video_type
        result['video_classification'] = video_classification
        result['mode'] = mode
        
        # Step 4: Flag uncertain predictions
        if result['confidence'] < 0.3:
            result['flag'] = 'MANUAL_REVIEW'
            result['flag_reason'] = 'Low confidence prediction'
            if verbose:
                print("⚠️  LOW CONFIDENCE - Manual review recommended")
        
        # Step 5: Print summary
        if verbose:
            print(f"{'='*60}")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Detector: {result.get('detector', 'Unknown')}")
            print(f"{'='*60}\n")
        
        return result


def collect_videos_from_directory(directory: str, max_videos: int = None) -> List[Path]:
    """
    Collect video files from directory
    
    Args:
        directory: Directory path
        max_videos: Maximum number of videos to collect
    
    Returns:
        List of video paths
    """
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_paths = []
    
    dir_path = Path(directory)
    for ext in video_extensions:
        video_paths.extend(dir_path.rglob(ext))
    
    video_paths = sorted(video_paths)
    
    if max_videos is not None:
        video_paths = video_paths[:max_videos]
    
    return video_paths


# def evaluate_detector(
#     detector: HybridDeepfakeDetector,
#     real_videos_dir: str,
#     fake_videos_dir: str,
#     output_dir: str = './evaluation_results',
#     mode: str = 'balanced',
#     max_videos_per_class: int = None,
#     force_detector: str = None
# ) -> Dict:
#     """
#     Evaluate detector on real and fake videos, generate CSV and metrics
    
#     Args:
#         detector: HybridDeepfakeDetector instance
#         real_videos_dir: Directory containing real videos
#         fake_videos_dir: Directory containing fake videos
#         output_dir: Output directory for results
#         mode: Detection mode
#         max_videos_per_class: Max videos per class (None = all)
#         force_detector: Force specific detector
    
#     Returns:
#         Dictionary with comprehensive metrics
#     """
#     print("="*70)
#     print("HYBRID DETECTOR EVALUATION")
#     print("="*70)
    
#     # Collect videos
#     print(f"\nCollecting real videos from: {real_videos_dir}")
#     real_videos = collect_videos_from_directory(real_videos_dir, max_videos_per_class)
#     print(f"✓ Found {len(real_videos)} real videos")
    
#     print(f"\nCollecting fake videos from: {fake_videos_dir}")
#     fake_videos = collect_videos_from_directory(fake_videos_dir, max_videos_per_class)
#     print(f"✓ Found {len(fake_videos)} fake videos")
    
#     if len(real_videos) == 0 or len(fake_videos) == 0:
#         raise ValueError("Need both real and fake videos for evaluation!")
    
#     # Process all videos
#     print(f"\nProcessing {len(real_videos) + len(fake_videos)} videos...")
#     all_results = []
    
#     # Process real videos
#     for video_path in tqdm(real_videos, desc="Real videos"):
#         try:
#             result = detector.predict(
#                 video_path=str(video_path),
#                 mode=mode,
#                 force_detector=force_detector,
#                 verbose=False
#             )
            
#             all_results.append({
#                 'video_name': video_path.name,
#                 'video_path': str(video_path),
#                 'true_label': 'real',
#                 'predicted_label': result['prediction'],
#                 'confidence': result['confidence'],
#                 'video_type': result['video_type'],
#                 'detector_used': result.get('detector', 'Unknown'),
#                 'mode': mode,
#                 'correct': result['prediction'] == 'real',
#                 'flag': result.get('flag', ''),
#                 'mmd_score': result.get('mmd_score', result.get('nsgvd_result', {}).get('mmd_score', None)),
#                 'face_ratio': result.get('video_classification', {}).get('face_ratio', None),
#             })
#         except Exception as e:
#             print(f"\nFailed to process {video_path.name}: {e}")
#             all_results.append({
#                 'video_name': video_path.name,
#                 'video_path': str(video_path),
#                 'true_label': 'real',
#                 'predicted_label': 'error',
#                 'confidence': 0.0,
#                 'video_type': 'error',
#                 'detector_used': 'error',
#                 'mode': mode,
#                 'correct': False,
#                 'flag': 'ERROR',
#                 'error': str(e)
#             })
    
#     # Process fake videos
#     for video_path in tqdm(fake_videos, desc="Fake videos"):
#         try:
#             result = detector.predict(
#                 video_path=str(video_path),
#                 mode=mode,
#                 force_detector=force_detector,
#                 verbose=False
#             )
            
#             all_results.append({
#                 'video_name': video_path.name,
#                 'video_path': str(video_path),
#                 'true_label': 'fake',
#                 'predicted_label': result['prediction'],
#                 'confidence': result['confidence'],
#                 'video_type': result['video_type'],
#                 'detector_used': result.get('detector', 'Unknown'),
#                 'mode': mode,
#                 'correct': result['prediction'] == 'fake',
#                 'flag': result.get('flag', ''),
#                 'mmd_score': result.get('mmd_score', result.get('nsgvd_result', {}).get('mmd_score', None)),
#                 'face_ratio': result.get('video_classification', {}).get('face_ratio', None),
#             })
#         except Exception as e:
#             print(f"\nFailed to process {video_path.name}: {e}")
#             all_results.append({
#                 'video_name': video_path.name,
#                 'video_path': str(video_path),
#                 'true_label': 'fake',
#                 'predicted_label': 'error',
#                 'confidence': 0.0,
#                 'video_type': 'error',
#                 'detector_used': 'error',
#                 'mode': mode,
#                 'correct': False,
#                 'flag': 'ERROR',
#                 'error': str(e)
#             })
    
#     # Create DataFrame
#     df = pd.DataFrame(all_results)
    
#     # Save detailed results to CSV
#     os.makedirs(output_dir, exist_ok=True)
#     csv_path = os.path.join(output_dir, f'evaluation_results_{mode}.csv')
#     df.to_csv(csv_path, index=False)
#     print(f"\n✓ Saved detailed results to: {csv_path}")
    
#     # Calculate metrics
#     print("\n" + "="*70)
#     print("CALCULATING METRICS")
#     print("="*70)
    
#     # Filter out errors
#     valid_df = df[df['predicted_label'] != 'error']
    
#     if len(valid_df) == 0:
#         print("ERROR: No valid predictions to evaluate!")
#         return {}
    
#     # Prepare labels for sklearn
#     y_true = valid_df['true_label'].map({'real': 0, 'fake': 1}).values
#     y_pred = valid_df['predicted_label'].map({'real': 0, 'fake': 1, 'uncertain': 1}).values
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     recall = recall_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
    
#     # Confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = cm.ravel()
    
#     # Additional metrics
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
#     fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
#     # Video type breakdown
#     video_type_accuracy = {}
#     for vtype in valid_df['video_type'].unique():
#         vtype_df = valid_df[valid_df['video_type'] == vtype]
#         if len(vtype_df) > 0:
#             vtype_acc = vtype_df['correct'].mean()
#             video_type_accuracy[vtype] = vtype_acc
    
#     # Detector usage breakdown
#     detector_accuracy = {}
#     for detector in valid_df['detector_used'].unique():
#         det_df = valid_df[valid_df['detector_used'] == detector]
#         if len(det_df) > 0:
#             det_acc = det_df['correct'].mean()
#             detector_accuracy[detector] = det_acc
    
#     metrics = {
#         'overall': {
#             'accuracy': float(accuracy),
#             'precision': float(precision),
#             'recall': float(recall),
#             'f1_score': float(f1),
#             'specificity': float(specificity),
#             'fpr': float(fpr),
#             'fnr': float(fnr),
#         },
#         'confusion_matrix': {
#             'true_negatives': int(tn),
#             'false_positives': int(fp),
#             'false_negatives': int(fn),
#             'true_positives': int(tp),
#         },
#         'by_video_type': video_type_accuracy,
#         'by_detector': detector_accuracy,
#         'dataset_stats': {
#             'total_videos': len(df),
#             'valid_predictions': len(valid_df),
#             'errors': len(df) - len(valid_df),
#             'real_videos': len(real_videos),
#             'fake_videos': len(fake_videos),
#         }
#     }
    
#     # Print metrics
#     print("\n" + "="*70)
#     print("EVALUATION METRICS")
#     print("="*70)
    
#     print("\n[OVERALL PERFORMANCE]")
#     print(f"  Accuracy:    {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
#     print(f"  Precision:   {metrics['overall']['precision']:.4f}")
#     print(f"  Recall:      {metrics['overall']['recall']:.4f}")
#     print(f"  F1-Score:    {metrics['overall']['f1_score']:.4f}")
#     print(f"  Specificity: {metrics['overall']['specificity']:.4f}")
#     print(f"  FPR:         {metrics['overall']['fpr']:.4f}")
#     print(f"  FNR:         {metrics['overall']['fnr']:.4f}")
    
#     print("\n[CONFUSION MATRIX]")
#     print(f"  True Negatives:  {metrics['confusion_matrix']['true_negatives']}")
#     print(f"  False Positives: {metrics['confusion_matrix']['false_positives']}")
#     print(f"  False Negatives: {metrics['confusion_matrix']['false_negatives']}")
#     print(f"  True Positives:  {metrics['confusion_matrix']['true_positives']}")
    
#     print("\n[PERFORMANCE BY VIDEO TYPE]")
#     for vtype, acc in metrics['by_video_type'].items():
#         print(f"  {vtype}: {acc:.4f} ({acc*100:.2f}%)")
    
#     print("\n[PERFORMANCE BY DETECTOR]")
#     for detector, acc in metrics['by_detector'].items():
#         print(f"  {detector}: {acc:.4f} ({acc*100:.2f}%)")
    
#     print("\n[DATASET STATISTICS]")
#     print(f"  Total videos:        {metrics['dataset_stats']['total_videos']}")
#     print(f"  Valid predictions:   {metrics['dataset_stats']['valid_predictions']}")
#     print(f"  Errors:              {metrics['dataset_stats']['errors']}")
#     print(f"  Real videos:         {metrics['dataset_stats']['real_videos']}")
#     print(f"  Fake videos:         {metrics['dataset_stats']['fake_videos']}")
    
#     # Save metrics to YAML
#     metrics_path = os.path.join(output_dir, f'metrics_{mode}.yaml')
#     with open(metrics_path, 'w') as f:
#         yaml.dump(metrics, f, default_flow_style=False)
#     print(f"\n✓ Saved metrics to: {metrics_path}")
    
#     # Generate classification report
#     print("\n[DETAILED CLASSIFICATION REPORT]")
#     target_names = ['Real', 'Fake']
#     report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
#     print(report)
    
#     report_path = os.path.join(output_dir, f'classification_report_{mode}.txt')
#     with open(report_path, 'w') as f:
#         f.write("HYBRID DETECTOR EVALUATION - CLASSIFICATION REPORT\n")
#         f.write("="*70 + "\n\n")
#         f.write(report)
#     print(f"✓ Saved classification report to: {report_path}")
    
#     print("\n" + "="*70)
#     print("EVALUATION COMPLETE!")
#     print("="*70)
    
#     return metrics


# gem
def evaluate_detector(
    detector: HybridDeepfakeDetector,
    real_videos_dir: str,
    fake_videos_dir: str,
    output_dir: str = './evaluation_results',
    mode: str = 'balanced',
    max_videos_per_class: int = None,
    force_detector: str = None
) -> Dict:
    """
    Evaluate detector on real and fake videos, generate CSV and only calculate
    Accuracy and Confusion Matrix.
    
    Args:
        detector: HybridDeepfakeDetector instance
        real_videos_dir: Directory containing real videos
        fake_videos_dir: Directory containing fake videos
        output_dir: Output directory for results
        mode: Detection mode
        max_videos_per_class: Max videos per class (None = all)
        force_detector: Force specific detector
    
    Returns:
        Dictionary with only Accuracy and Confusion Matrix results.
    """
    print("="*70)
    print("HYBRID DETECTOR EVALUATION (ACCURACY & CONFUSION MATRIX ONLY)")
    print("="*70)
    
    # Collect videos
    print(f"\nCollecting real videos from: {real_videos_dir}")
    real_videos = collect_videos_from_directory(real_videos_dir, max_videos_per_class)
    print(f"✓ Found {len(real_videos)} real videos")
    
    print(f"\nCollecting fake videos from: {fake_videos_dir}")
    fake_videos = collect_videos_from_directory(fake_videos_dir, max_videos_per_class)
    print(f"✓ Found {len(fake_videos)} fake videos")
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        raise ValueError("Need both real and fake videos for evaluation!")
    
    # Process all videos
    print(f"\nProcessing {len(real_videos) + len(fake_videos)} videos...")
    all_results = []
    
    # Process real videos (TRUNCATED FOR BREVITY - LOGIC UNCHANGED)
    for video_path in tqdm(real_videos, desc="Real videos"):
        try:
            result = detector.predict(
                video_path=str(video_path),
                mode=mode,
                force_detector=force_detector,
                verbose=False
            )
            
            all_results.append({
                'video_name': video_path.name,
                'video_path': str(video_path),
                'true_label': 'real',
                'predicted_label': result['prediction'],
                'confidence': result['confidence'],
                'video_type': result['video_type'],
                'detector_used': result.get('detector', 'Unknown'),
                'mode': mode,
                'correct': result['prediction'] == 'real',
                'flag': result.get('flag', ''),
                'mmd_score': result.get('mmd_score', result.get('nsgvd_result', {}).get('mmd_score', None)),
                'face_ratio': result.get('video_classification', {}).get('face_ratio', None),
            })
        except Exception as e:
            print(f"\nFailed to process {video_path.name}: {e}")
            all_results.append({
                'video_name': video_path.name,
                'video_path': str(video_path),
                'true_label': 'real',
                'predicted_label': 'error',
                'confidence': 0.0,
                'video_type': 'error',
                'detector_used': 'error',
                'mode': mode,
                'correct': False,
                'flag': 'ERROR',
                'error': str(e)
            })
    
    # Process fake videos (TRUNCATED FOR BREVITY - LOGIC UNCHANGED)
    for video_path in tqdm(fake_videos, desc="Fake videos"):
        try:
            result = detector.predict(
                video_path=str(video_path),
                mode=mode,
                force_detector=force_detector,
                verbose=False
            )
            
            all_results.append({
                'video_name': video_path.name,
                'video_path': str(video_path),
                'true_label': 'fake',
                'predicted_label': result['prediction'],
                'confidence': result['confidence'],
                'video_type': result['video_type'],
                'detector_used': result.get('detector', 'Unknown'),
                'mode': mode,
                'correct': result['prediction'] == 'fake',
                'flag': result.get('flag', ''),
                'mmd_score': result.get('mmd_score', result.get('nsgvd_result', {}).get('mmd_score', None)),
                'face_ratio': result.get('video_classification', {}).get('face_ratio', None),
            })
        except Exception as e:
            print(f"\nFailed to process {video_path.name}: {e}")
            all_results.append({
                'video_name': video_path.name,
                'video_path': str(video_path),
                'true_label': 'fake',
                'predicted_label': 'error',
                'confidence': 0.0,
                'video_type': 'error',
                'detector_used': 'error',
                'mode': mode,
                'correct': False,
                'flag': 'ERROR',
                'error': str(e)
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save detailed results to CSV (KEPT FOR DATA EXPORT)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'evaluation_results_{mode}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results to: {csv_path}")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("CALCULATING REQUIRED METRICS")
    print("="*70)
    
    # Filter out errors
    valid_df = df[df['predicted_label'] != 'error']
    
    if len(valid_df) == 0:
        print("ERROR: No valid predictions to evaluate!")
        return {}
    
    # Prepare labels for sklearn
    y_true = valid_df['true_label'].map({'real': 0, 'fake': 1}).values
    y_pred = valid_df['predicted_label'].map({'real': 0, 'fake': 1, 'uncertain': 1}).values
    
    # Calculate required metrics: Accuracy and Confusion Matrix
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Removed: precision, recall, f1, specificity, fpr, fnr, roc_auc
    
    # Removed: video_type_accuracy, detector_accuracy calculations
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            # Removed other overall metrics
        },
        'confusion_matrix': {
            'True Negatives (TN)': int(tn),
            'False Positives (FP)': int(fp),
            'False Negatives (FN)': int(fn),
            'True Positives (TP)': int(tp),
            'matrix_array': cm.tolist()
        },
        'dataset_stats': {
            'total_videos': len(df),
            'valid_predictions': len(valid_df),
            'errors': len(df) - len(valid_df),
            'real_videos': len(real_videos),
            'fake_videos': len(fake_videos),
        }
    }
    
    # Print metrics
    print("\n" + "="*70)
    print("RESULTS (ACCURACY & CONFUSION MATRIX)")
    print("="*70)
    
    print("\n[OVERALL PERFORMANCE]")
    print(f"  Accuracy: {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
    
    print("\n[CONFUSION MATRIX]")
    # Print the 2x2 matrix
    print(f"| {'':<10} | {'Predicted Real':<15} | {'Predicted Fake':<15} |")
    print(f"|{'-'*11}|{'-'*17}|{'-'*17}|")
    print(f"| {'True Real':<10} | {metrics['confusion_matrix']['True Negatives (TN)']:<15} | {metrics['confusion_matrix']['False Positives (FP)']:<15} |")
    print(f"| {'True Fake':<10} | {metrics['confusion_matrix']['False Negatives (FN)']:<15} | {metrics['confusion_matrix']['True Positives (TP)']:<15} |")
    
    print(f"\n  True Negatives (TN): {metrics['confusion_matrix']['True Negatives (TN)']}")
    print(f"  False Positives (FP): {metrics['confusion_matrix']['False Positives (FP)']}")
    print(f"  False Negatives (FN): {metrics['confusion_matrix']['False Negatives (FN)']}")
    print(f"  True Positives (TP): {metrics['confusion_matrix']['True Positives (TP)']}")

    
    print("\n[DATASET STATISTICS]")
    print(f"  Total videos:        {metrics['dataset_stats']['total_videos']}")
    print(f"  Valid predictions:   {metrics['dataset_stats']['valid_predictions']}")
    print(f"  Errors:              {metrics['dataset_stats']['errors']}")
    print(f"  Real videos:         {metrics['dataset_stats']['real_videos']}")
    print(f"  Fake videos:         {metrics['dataset_stats']['fake_videos']}")
    
    # Save metrics to YAML (only containing the required fields)
    metrics_path = os.path.join(output_dir, f'metrics_{mode}_minimal.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"\n✓ Saved minimal metrics to: {metrics_path}")
    
    # Removed: Detailed Classification Report generation
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    
    return metrics

def main():
    """Command-line interface for hybrid detector"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hybrid Deepfake Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video detection
  python hybrid_detector.py --video_path test.mp4
  
  # Force specific detector
  python hybrid_detector.py --video_path test.mp4 --force nsgvd
  
  # Conservative mode (low false positives)
  python hybrid_detector.py --video_path test.mp4 --mode conservative
  
  # Evaluate on dataset
  python hybrid_detector.py --evaluate \\
    --real_dir ./real_videos \\
    --fake_dir ./fake_videos \\
    --output_dir ./evaluation_results
  
  # Evaluate with limits
  python hybrid_detector.py --evaluate \\
    --real_dir ./real_videos \\
    --fake_dir ./fake_videos \\
    --max_videos 100 \\
    --mode balanced
        """
    )
    
    parser.add_argument('--video_path', type=str, help='Path to video file')
    parser.add_argument('--mode', type=str, default='balanced',
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='Detection mode')
    parser.add_argument('--force', type=str, choices=['nsgvd', 'face', 'ensemble'],
                       help='Force specific detector (override automatic routing)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation mode')
    parser.add_argument('--real_dir', type=str,
                       help='Directory with real videos (for evaluation)')
    parser.add_argument('--fake_dir', type=str,
                       help='Directory with fake videos (for evaluation)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Max videos per class for evaluation')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--nsgvd_config', type=str, 
                       default='./config/nsgvd_detector.yaml',
                       help='Path to NSG-VD config')
    parser.add_argument('--calibration', type=str,
                       default='./calibration_results/hybrid_calibration.yaml',
                       help='Path to calibration config')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HybridDeepfakeDetector(
        nsgvd_config_path=args.nsgvd_config,
        calibration_config_path=args.calibration,
        device_id=args.device
    )
    
    if args.evaluate:
        # Evaluation mode
        if not args.real_dir or not args.fake_dir:
            print("Error: --real_dir and --fake_dir required for evaluation")
            return
        
        metrics = evaluate_detector(
            detector=detector,
            real_videos_dir=args.real_dir,
            fake_videos_dir=args.fake_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            max_videos_per_class=args.max_videos,
            force_detector=args.force
        )
    
    elif args.video_path:
        # Single video detection mode
        result = detector.predict(
            video_path=args.video_path,
            mode=args.mode,
            force_detector=args.force
        )
        
        # Print detailed results
        print("\nDETAILED RESULTS:")
        print(f"  Video Type: {result['video_type']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Detector Used: {result.get('detector', 'Unknown')}")
        if 'flag' in result:
            print(f"  ⚠️  Flag: {result['flag']} - {result['flag_reason']}")    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()