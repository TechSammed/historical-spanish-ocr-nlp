import os
import json

label_dir = "data/train_line_labels"

chars = set()

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
            text = f.read().strip()
            chars.update(list(text))

chars = sorted(chars)

# 0 reserved for CTC blank
vocab = {c: i+1 for i, c in enumerate(chars)}

os.makedirs("models", exist_ok=True)

with open("models/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("Vocabulary size:", len(vocab))