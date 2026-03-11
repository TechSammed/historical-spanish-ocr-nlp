import os

print("Images:", len(os.listdir("data/train_lines")))
print("Labels:", len(os.listdir("data/train_line_labels")))