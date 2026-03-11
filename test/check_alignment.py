import os

# Change these according to your page
page_id = "page12"   # example

line_image_folder = "data/lines_test"
pseudo_label_file = "data/ground_truth/train_pseudo/page_12.txt"

# ---- Count segmented line images ----
segmented_lines = [
    f for f in os.listdir(line_image_folder)
    if f.startswith(page_id) and f.endswith(".png")
]

segmented_count = len(segmented_lines)

# ---- Count text lines ----
with open(pseudo_label_file, "r", encoding="utf-8") as f:
    text = f.read()

# Split and remove empty lines
text_lines = [line.strip() for line in text.split("\n") if line.strip() != ""]

text_count = len(text_lines)

print("Segmented line images:", segmented_count)
print("Text lines (non-empty):", text_count)
print("Difference:", abs(segmented_count - text_count))