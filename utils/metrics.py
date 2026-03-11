import os
import unicodedata
from jiwer import wer
import Levenshtein


# 🔹 Normalize text based on project notes
def normalize_text(text):
    text = text.lower()

    # Replace ç with z
    text = text.replace("ç", "z")

    # Treat v as u
    text = text.replace("v", "u")

    # Treat f as s (long s issue)
    text = text.replace("f", "s")

    # Remove accents except ñ
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn' or c == 'ñ'
    )

    # Remove extra spaces
    text = " ".join(text.split())

    return text


# 🔹 Compute Character Error Rate
def compute_cer(gt, pred):
    distance = Levenshtein.distance(gt, pred)
    return distance / len(gt)


# 🔹 Evaluate and Save Results
def evaluate(ground_truth_folder, prediction_folder):
    gt_files = sorted(os.listdir(ground_truth_folder))

    total_cer = 0
    total_wer = 0
    count = 0

    results_lines = []

    for file_name in gt_files:
        if file_name.endswith(".txt"):
            gt_path = os.path.join(ground_truth_folder, file_name)
            pred_path = os.path.join(prediction_folder, file_name)

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_text = f.read()

            with open(pred_path, "r", encoding="utf-8") as f:
                pred_text = f.read()

            gt_norm = normalize_text(gt_text)
            pred_norm = normalize_text(pred_text)

            cer = compute_cer(gt_norm, pred_norm)
            w = wer(gt_norm, pred_norm)

            line = f"{file_name} → CER: {cer:.4f}, WER: {w:.4f}"
            print(line)

            results_lines.append(line)

            total_cer += cer
            total_wer += w
            count += 1

    avg_cer = total_cer / count
    avg_wer = total_wer / count

    print("\nAverage CER:", avg_cer)
    print("Average WER:", avg_wer)

    results_lines.append(f"\nAverage CER: {avg_cer}")
    results_lines.append(f"Average WER: {avg_wer}")

    # 🔹 Save results to file
    os.makedirs("results/baseline", exist_ok=True)

    with open("results/baseline/metrics.txt", "w", encoding="utf-8") as f:
        for line in results_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    evaluate(
        ground_truth_folder="data/ground_truth",
        prediction_folder="data/predictions/baseline_clean"
    )