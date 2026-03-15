import os
import re

input_file = "results/tesseract_predictions.txt"
output_file = "results/page_1_tesseract.txt"

predictions = []

os.makedirs("results", exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:

        parts = line.split("→")

        if len(parts) < 2:
            continue

        filename = parts[0].strip()
        text = parts[1].strip()

        if "page_1" in filename:

            match = re.search(r"line_(\d+)", filename)

            if match:
                line_number = int(match.group(1))
                predictions.append((line_number, text))

print("Lines found:", len(predictions))

predictions.sort(key=lambda x: x[0])

merged_text = " ".join([text for _, text in predictions])

with open(output_file, "w", encoding="utf-8") as f:
    f.write(merged_text)

print("Saved to:", output_file)