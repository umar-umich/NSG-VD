"""
Complete Threshold Calibration Script for NSG-VD
Uses reference real and fake videos to optimize detection threshold
Reduces false positives while maintaining high detection rate
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report
)

# Add NSG-VD to path
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

from main import NSGVDDetector


class ThresholdCalibrator:
    """
    Calibrate NSG-VD detection threshold using labeled validation data
    """
    
    def __init__(self, config_path='./config/nsgvd_detector.yaml', device_id=0):
        """
        Args:
            config_path: Path to NSG-VD config
            device_id: GPU device ID
        """
        print("="*70)
        print("NSG-VD THRESHOLD CALIBRATION")
        print("="*70)
        
        self.config_path = config_path
        self.device_id = device_id
        
        # Initialize detector
        print("\nInitializing NSG-VD detector...")
        self.detector = NSGVDDetector(config_path=config_path, device_id=device_id)
        self.detector.load_model()
        self.detector.load_score_function()
        self.detector.load_reference_features()
        print("✓ Detector loaded successfully")
        
        # Results storage
        self.real_scores = []
        self.fake_scores = []
        self.real_videos = []
        self.fake_videos = []
        self.failed_videos = []
    
    def collect_video_paths(self, real_dir, fake_dir, max_videos_per_category=None):
        """
        Collect video paths from directories.

        Now interprets max_videos_per_category as:
          - max videos PER SUBFOLDER inside real_dir and fake_dir.
        If there are no subfolders, it treats the root dir as a single category.

        Args:
            real_dir: Directory with real videos (containing subfolders for categories)
            fake_dir: Directory with fake videos (containing subfolders for categories)
            max_videos_per_category: Limit number of videos per subfolder/category (None = all)
        
        Returns:
            real_videos, fake_videos: Lists of video paths
        """
        print(f"\nCollecting videos...")
        print(f"  Real videos from: {real_dir}")
        print(f"  Fake videos from: {fake_dir}")

        def collect_from_root(root_dir, max_per_category):
            root = Path(root_dir)
            all_paths = []

            # Immediate subfolders = categories
            subdirs = [p for p in root.iterdir() if p.is_dir()]

            # If no subdirs, treat root itself as a single category
            if not subdirs:
                subdirs = [root]

            for cat_dir in subdirs:
                cat_paths = []
                for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                    # Use rglob in case there are deeper levels inside category
                    cat_paths.extend(cat_dir.rglob(ext))
                cat_paths = sorted(cat_paths)

                if max_per_category is not None:
                    cat_paths = cat_paths[:max_per_category]

                print(f"    Category: {cat_dir.relative_to(root)} -> {len(cat_paths)} videos")
                all_paths.extend(cat_paths)

            return all_paths

        # Collect real videos per category
        real_videos = collect_from_root(real_dir, max_videos_per_category)
        # Collect fake videos per category
        fake_videos = collect_from_root(fake_dir, max_videos_per_category)

        print(f"\n✓ Total real videos used: {len(real_videos)}")
        print(f"✓ Total fake videos used: {len(fake_videos)}")

        if len(real_videos) == 0 or len(fake_videos) == 0:
            raise ValueError("Need both real and fake videos for calibration!")

        return real_videos, fake_videos

    # def collect_video_paths(self, real_dir, fake_dir, max_videos_per_class=None):
    #     """
    #     Collect video paths from directories
        
    #     Args:
    #         real_dir: Directory with real videos
    #         fake_dir: Directory with fake videos
    #         max_videos_per_class: Limit number of videos per class (None = all)
        
    #     Returns:
    #         real_videos, fake_videos: Lists of video paths
    #     """
    #     print(f"\nCollecting videos...")
    #     print(f"  Real videos from: {real_dir}")
    #     print(f"  Fake videos from: {fake_dir}")
        
    #     # Collect real videos
    #     real_videos = []
    #     for ext in ['**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv']:
    #         real_videos.extend(Path(real_dir).glob(ext))
    #     real_videos = sorted(real_videos)
        
    #     # Collect fake videos
    #     fake_videos = []
    #     for ext in ['**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv']:
    #         fake_videos.extend(Path(fake_dir).glob(ext))
    #     fake_videos = sorted(fake_videos)
        
    #     # Limit if specified
    #     if max_videos_per_class:
    #         real_videos = real_videos[:max_videos_per_class]
    #         fake_videos = fake_videos[:max_videos_per_class]
        
    #     print(f"\n✓ Found {len(real_videos)} real videos")
    #     print(f"✓ Found {len(fake_videos)} fake videos")
        
    #     if len(real_videos) == 0 or len(fake_videos) == 0:
    #         raise ValueError("Need both real and fake videos for calibration!")
        
    #     return real_videos, fake_videos
    
    def compute_scores(self, real_videos, fake_videos):
        """
        Compute MMD scores for all videos
        
        Args:
            real_videos: List of real video paths
            fake_videos: List of fake video paths
        """
        print("\n" + "="*70)
        print("COMPUTING MMD SCORES")
        print("="*70)
        
        # Process real videos
        print(f"\n[1/2] Processing {len(real_videos)} real videos...")
        self.real_scores = []
        self.real_videos = []
        
        for video_path in tqdm(real_videos, desc="Real videos"):
            try:
                result = self.detector.predict(str(video_path), feature_type='velocity')
                self.real_scores.append(result['mmd_score'])
                self.real_videos.append(str(video_path))
            except Exception as e:
                print(f"\n  Warning: Failed to process {video_path.name}: {e}")
                self.failed_videos.append({
                    'video': str(video_path),
                    'label': 'real',
                    'error': str(e)
                })
        
        print(f"✓ Successfully processed {len(self.real_scores)} real videos")
        
        # Process fake videos
        print(f"\n[2/2] Processing {len(fake_videos)} fake videos...")
        self.fake_scores = []
        self.fake_videos = []
        
        for video_path in tqdm(fake_videos, desc="Fake videos"):
            try:
                result = self.detector.predict(str(video_path), feature_type='velocity')
                self.fake_scores.append(result['mmd_score'])
                self.fake_videos.append(str(video_path))
            except Exception as e:
                print(f"\n  Warning: Failed to process {video_path.name}: {e}")
                self.failed_videos.append({
                    'video': str(video_path),
                    'label': 'fake',
                    'error': str(e)
                })
        
        print(f"✓ Successfully processed {len(self.fake_scores)} fake videos")
        
        if len(self.failed_videos) > 0:
            print(f"\n⚠️  {len(self.failed_videos)} videos failed to process")
    
    def calculate_statistics(self):
        """Calculate statistics on MMD scores"""
        real_scores = np.array(self.real_scores)
        fake_scores = np.array(self.fake_scores)
        
        stats = {
            'real': {
                'mean': float(real_scores.mean()),
                'std': float(real_scores.std()),
                'min': float(real_scores.min()),
                'max': float(real_scores.max()),
                'median': float(np.median(real_scores)),
                'q25': float(np.percentile(real_scores, 25)),
                'q75': float(np.percentile(real_scores, 75)),
            },
            'fake': {
                'mean': float(fake_scores.mean()),
                'std': float(fake_scores.std()),
                'min': float(fake_scores.min()),
                'max': float(fake_scores.max()),
                'median': float(np.median(fake_scores)),
                'q25': float(np.percentile(fake_scores, 25)),
                'q75': float(np.percentile(fake_scores, 75)),
            }
        }
        
        return stats
    
    def find_optimal_thresholds(self):
        """
        Find optimal thresholds for different operating points
        
        Returns:
            Dictionary with thresholds for different FPR targets
        """
        print("\n" + "="*70)
        print("CALCULATING OPTIMAL THRESHOLDS")
        print("="*70)
        
        real_scores = np.array(self.real_scores)
        fake_scores = np.array(self.fake_scores)
        
        # Calculate statistics
        stats = self.calculate_statistics()
        
        print("\nScore Statistics:")
        print(f"\nReal videos (n={len(real_scores)}):")
        print(f"  Mean:   {stats['real']['mean']:.4f}")
        print(f"  Std:    {stats['real']['std']:.4f}")
        print(f"  Median: {stats['real']['median']:.4f}")
        print(f"  Range:  [{stats['real']['min']:.4f}, {stats['real']['max']:.4f}]")
        
        print(f"\nFake videos (n={len(fake_scores)}):")
        print(f"  Mean:   {stats['fake']['mean']:.4f}")
        print(f"  Std:    {stats['fake']['std']:.4f}")
        print(f"  Median: {stats['fake']['median']:.4f}")
        print(f"  Range:  [{stats['fake']['min']:.4f}, {stats['fake']['max']:.4f}]")
        
        # Calculate percentile-based thresholds
        thresholds = {
            'conservative': {
                'threshold': float(np.percentile(real_scores, 95)),
                'target_fpr': 0.05,
                'description': 'Conservative (5% FPR) - For demos/public use'
            },
            'balanced': {
                'threshold': float(np.percentile(real_scores, 90)),
                'target_fpr': 0.10,
                'description': 'Balanced (10% FPR) - Default for most use cases'
            },
            'aggressive': {
                'threshold': float(np.percentile(real_scores, 80)),
                'target_fpr': 0.20,
                'description': 'Aggressive (20% FPR) - For security applications'
            },
            'paper_default': {
                'threshold': 1.0,
                'target_fpr': None,
                'description': 'Original paper threshold'
            }
        }
        
        # Calculate actual FPR and TPR for each threshold
        all_scores = np.concatenate([real_scores, fake_scores])
        all_labels = np.concatenate([
            np.zeros(len(real_scores)),  # 0 = real
            np.ones(len(fake_scores))     # 1 = fake
        ])
        
        print("\n" + "-"*70)
        print("Threshold Options:")
        print("-"*70)
        
        for mode, config in thresholds.items():
            threshold = config['threshold']
            
            # Calculate metrics
            predictions = (all_scores > threshold).astype(int)
            
            # True Positives: Correctly identified fakes
            tp = np.sum((predictions == 1) & (all_labels == 1))
            # False Positives: Real videos incorrectly marked as fake
            fp = np.sum((predictions == 1) & (all_labels == 0))
            # True Negatives: Correctly identified real
            tn = np.sum((predictions == 0) & (all_labels == 0))
            # False Negatives: Fake videos missed
            fn = np.sum((predictions == 0) & (all_labels == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
            
            config['actual_fpr'] = float(fpr)
            config['tpr'] = float(tpr)
            config['precision'] = float(precision)
            config['f1'] = float(f1)
            config['accuracy'] = float((tp + tn) / len(all_scores))
            
            print(f"\n{mode.upper()}: {config['description']}")
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Accuracy:  {config['accuracy']:.2%}")
            print(f"  TPR (Recall): {tpr:.2%} ({tp}/{tp+fn} fakes detected)")
            print(f"  FPR:       {fpr:.2%} ({fp}/{fp+tn} false alarms)")
            print(f"  Precision: {precision:.2%}")
            print(f"  F1 Score:  {f1:.4f}")
        
        # Find Youden's J statistic optimal threshold
        fpr_vals, tpr_vals, thresholds_roc = roc_curve(all_labels, all_scores)
        j_scores = tpr_vals - fpr_vals
        best_idx = np.argmax(j_scores)
        
        thresholds['youden'] = {
            'threshold': float(thresholds_roc[best_idx]),
            'tpr': float(tpr_vals[best_idx]),
            'fpr': float(fpr_vals[best_idx]),
            'description': "Youden's J statistic (optimal TPR-FPR)",
        }
        
        print(f"\nYOUDEN (Optimal): {thresholds['youden']['description']}")
        print(f"  Threshold: {thresholds['youden']['threshold']:.4f}")
        print(f"  TPR: {thresholds['youden']['tpr']:.2%}")
        print(f"  FPR: {thresholds['youden']['fpr']:.2%}")
        
        return thresholds, stats
    
    def plot_results(self, thresholds, stats, output_dir='./calibration_results'):
        """
        Generate visualization plots
        
        Args:
            thresholds: Dictionary of threshold configurations
            stats: Score statistics
            output_dir: Directory to save plots
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        real_scores = np.array(self.real_scores)
        fake_scores = np.array(self.fake_scores)
        
        # Prepare data for ROC curve
        all_scores = np.concatenate([real_scores, fake_scores])
        all_labels = np.concatenate([
            np.zeros(len(real_scores)),
            np.ones(len(fake_scores))
        ])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Score Distribution Histogram
        ax = axes[0, 0]
        bins = np.linspace(
            min(real_scores.min(), fake_scores.min()),
            max(real_scores.max(), fake_scores.max()),
            50
        )
        ax.hist(real_scores, bins=bins, alpha=0.6, label='Real', color='green', edgecolor='black')
        ax.hist(fake_scores, bins=bins, alpha=0.6, label='Fake', color='red', edgecolor='black')
        
        # Add threshold lines
        for mode in ['conservative', 'balanced', 'aggressive', 'paper_default']:
            if mode in thresholds:
                ax.axvline(
                    thresholds[mode]['threshold'], 
                    linestyle='--', 
                    label=f"{mode} ({thresholds[mode]['threshold']:.2f})",
                    linewidth=2
                )
        
        ax.set_xlabel('MMD Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('MMD Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. ROC Curve
        ax = axes[0, 1]
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        
        # Mark operating points
        for mode, color in [('conservative', 'green'), ('balanced', 'orange'), ('aggressive', 'red')]:
            if mode in thresholds and 'actual_fpr' in thresholds[mode]:
                ax.plot(
                    thresholds[mode]['actual_fpr'],
                    thresholds[mode]['tpr'],
                    marker='o',
                    markersize=10,
                    color=color,
                    label=f"{mode}"
                )
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax = axes[1, 0]
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color='darkblue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Mark operating points
        for mode, color in [('conservative', 'green'), ('balanced', 'orange'), ('aggressive', 'red')]:
            if mode in thresholds and 'precision' in thresholds[mode]:
                ax.plot(
                    thresholds[mode]['tpr'],
                    thresholds[mode]['precision'],
                    marker='o',
                    markersize=10,
                    color=color,
                    label=f"{mode}"
                )
        
        ax.set_xlabel('Recall (TPR)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        # 4. Box Plot Comparison
        ax = axes[1, 1]
        box_data = [real_scores, fake_scores]
        bp = ax.boxplot(box_data, labels=['Real', 'Fake'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        # Add threshold lines
        for mode, linestyle in [('conservative', '--'), ('balanced', '-.'), ('aggressive', ':')]:
            if mode in thresholds:
                ax.axhline(
                    thresholds[mode]['threshold'],
                    color='blue',
                    linestyle=linestyle,
                    label=f"{mode}",
                    linewidth=2
                )
        
        ax.set_ylabel('MMD Score', fontsize=12)
        ax.set_title('Score Distribution by Class', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'calibration_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {plot_path}")
        
        plt.close()
    
    def save_results(self, thresholds, stats, output_dir='./calibration_results'):
        """
        Save calibration results to files
        
        Args:
            thresholds: Dictionary of threshold configurations
            stats: Score statistics
            output_dir: Directory to save results
        """
        print("\n" + "="*70)
        print("SAVING CALIBRATION RESULTS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save calibration YAML
        calibration_yaml = {
            'calibration_date': pd.Timestamp.now().isoformat(),
            'num_real_videos': len(self.real_scores),
            'num_fake_videos': len(self.fake_scores),
            'statistics': stats,
            'thresholds': {
                k: {
                    'threshold': v['threshold'],
                    'description': v['description'],
                    'tpr': v.get('tpr'),
                    'fpr': v.get('actual_fpr'),
                    'precision': v.get('precision'),
                    'f1': v.get('f1'),
                    'accuracy': v.get('accuracy'),
                }
                for k, v in thresholds.items()
            }
        }
        
        yaml_path = os.path.join(output_dir, 'calibration.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved calibration config: {yaml_path}")
        
        # 2. Save detailed scores CSV
        detailed_results = []
        
        for video, score in zip(self.real_videos, self.real_scores):
            detailed_results.append({
                'video': os.path.basename(video),
                'full_path': video,
                'true_label': 'real',
                'mmd_score': score,
            })
        
        for video, score in zip(self.fake_videos, self.fake_scores):
            detailed_results.append({
                'video': os.path.basename(video),
                'full_path': video,
                'true_label': 'fake',
                'mmd_score': score,
            })
        
        # Add predictions for each threshold
        for mode in ['conservative', 'balanced', 'aggressive']:
            if mode in thresholds:
                threshold = thresholds[mode]['threshold']
                for result in detailed_results:
                    predicted = 'fake' if result['mmd_score'] > threshold else 'real'
                    result[f'pred_{mode}'] = predicted
                    result[f'correct_{mode}'] = (predicted == result['true_label'])
        
        df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(output_dir, 'detailed_scores.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved detailed scores: {csv_path}")
        
        # 3. Save failed videos list
        if self.failed_videos:
            failed_df = pd.DataFrame(self.failed_videos)
            failed_path = os.path.join(output_dir, 'failed_videos.csv')
            failed_df.to_csv(failed_path, index=False)
            print(f"✓ Saved failed videos list: {failed_path}")
        
        # 4. Update NSG-VD config with recommended threshold
        self.update_config(thresholds['balanced']['threshold'])
    
    def update_config(self, threshold):
        """
        Update NSG-VD config file with calibrated threshold
        
        Args:
            threshold: New threshold value
        """
        print(f"\n✓ Recommended threshold for config: {threshold:.4f}")
        print(f"  Update {self.config_path}")
        print(f"  Set: threshold: {threshold:.4f}")
        
        # Optionally auto-update (commented out for safety)
        # with open(self.config_path, 'r') as f:
        #     config = yaml.safe_load(f)
        # config['threshold'] = float(threshold)
        # with open(self.config_path, 'w') as f:
        #     yaml.dump(config, f, default_flow_style=False)
    
    def run_calibration(self, real_dir, fake_dir, max_videos=None, output_dir='./calibration_results'):
        """
        Run complete calibration pipeline
        
        Args:
            real_dir: Directory with real videos
            fake_dir: Directory with fake videos
            max_videos: Maximum videos per class (None = all)
            output_dir: Directory to save results
        
        Returns:
            Dictionary with thresholds and statistics
        """
        # Collect videos
        real_videos, fake_videos = self.collect_video_paths(real_dir, fake_dir, max_videos)
        
        # Compute MMD scores
        self.compute_scores(real_videos, fake_videos)
        
        # Find optimal thresholds
        thresholds, stats = self.find_optimal_thresholds()
        
        # Generate visualizations
        self.plot_results(thresholds, stats, output_dir)
        
        # Save results
        self.save_results(thresholds, stats, output_dir)
        
        return thresholds, stats


def main():
    """Command-line interface for threshold calibration"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calibrate NSG-VD Detection Threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calibration
  python calibrate_threshold.py \\
    --real_dir ./calibration_data/real \\
    --fake_dir ./calibration_data/fake

  # Limit number of videos (faster for testing)
  python calibrate_threshold.py \\
    --real_dir ./calibration_data/real \\
    --fake_dir ./calibration_data/fake \\
    --max_videos 50

  # Specify output directory
  python calibrate_threshold.py \\
    --real_dir ./calibration_data/real \\
    --fake_dir ./calibration_data/fake \\
    --output_dir ./my_calibration_results
        """
    )
    
    parser.add_argument(
        '--real_dir',
        type=str,
        required=True,
        help='Directory containing known real videos (not used in reference!)'
    )
    parser.add_argument(
        '--fake_dir',
        type=str,
        required=True,
        help='Directory containing known fake videos'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config/nsgvd_detector.yaml',
        help='Path to NSG-VD config file'
    )
    parser.add_argument(
        '--max_videos',
        type=int,
        default=None,
        help='Maximum videos per class (None = use all)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./calibration_results',
        help='Directory to save calibration results'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.real_dir):
        print(f"Error: Real videos directory not found: {args.real_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.fake_dir):
        print(f"Error: Fake videos directory not found: {args.fake_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run calibration
    calibrator = ThresholdCalibrator(config_path=args.config, device_id=args.device)
    
    thresholds, stats = calibrator.run_calibration(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        max_videos=args.max_videos,
        output_dir=args.output_dir
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nRecommended thresholds:")
    print(f"  Conservative (demos):  {thresholds['conservative']['threshold']:.4f}")
    print(f"  Balanced (default):    {thresholds['balanced']['threshold']:.4f}")
    print(f"  Aggressive (security): {thresholds['aggressive']['threshold']:.4f}")
    print("\nNext steps:")
    print(f"  1. Review plots: {args.output_dir}/calibration_analysis.png")
    print(f"  2. Check detailed scores: {args.output_dir}/detailed_scores.csv")
    print(f"  3. Update config: {args.config}")
    print(f"     Set 'threshold: {thresholds['balanced']['threshold']:.4f}' for balanced mode")
    print("="*70)


if __name__ == '__main__':
    main()