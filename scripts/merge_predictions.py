import re

input_file = "results/crnn_predictions.txt"
output_file = "results/page_1_predicted.txt"

predictions = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if "page_1" in line:
            parts = line.split("→")
            filename = parts[0].strip()
            text = parts[1].strip()

            if text:
                # extract line number
                match = re.search(r"line_(\d+)", filename)
                if match:
                    line_number = int(match.group(1))
                    predictions.append((line_number, text))

# sort by line number
predictions.sort(key=lambda x: x[0])

# merge all lines into paragraph
merged_text = " ".join([text for _, text in predictions])

# save merged paragraph
with open(output_file, "w", encoding="utf-8") as f:
    f.write(merged_text)

print("Merged page prediction saved to:", output_file)