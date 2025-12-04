import argparse
import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from main import DFBDetector  # make sure main.py is in the same folder or adjust import path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", "-d", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_file", "-o", type=str, default="results.txt", help="File to save predictions")
    args = parser.parse_args()

    video_dir = args.video_dir
    output_file = args.output_file

    # Collect all .mp4 files in directory
    # video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4", ".mov")]
    # video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".mov"))]
    video_files = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".mov")):
                video_files.append(os.path.join(root, f))
    
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video(s) in '{video_dir}'")

    y_true = []  # All are fake (ground truth)
    y_pred = []

    with open(output_file, "w") as f:
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"\nProcessing {video_path} ...")

            try:
                dfb = DFBDetector(video_path=video_path)
                file_name = os.path.splitext(video_file)[0]

                # Get predictions using your chosen model
                result = dfb.get_predictions(model_name='ffd', uploaded_file_name=file_name)

                # Extract prediction and score
                pred_label = list(result.keys())[0]
                score = result[pred_label]

                # Write prediction result
                f.write(f"{video_file}: {pred_label} ({score})\n")
                print(f"→ Prediction: {pred_label} (score={score})")

                # Metrics data
                y_true.append("fake")  # treating all as fake
                y_pred.append(pred_label)

            except Exception as e:
                print(f"⚠️ Error processing {video_file}: {e}")
                f.write(f"{video_file}: error ({e})\n")

        # --- Compute metrics ---
        acc = accuracy_score(y_true, y_pred)
        labels = ["real", "fake"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Write summary
        f.write("\n\n--- Summary ---\n")
        f.write(f"Accuracy: {acc:.2f}\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(str(cm) + "\n")

    print("\nDone!")
    print(f"Results saved to: {output_file}")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()

