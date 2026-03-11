import os
import json

def collect_characters(folder_paths):
    charset = set()

    for folder in folder_paths:
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                    text = f.read()
                    for ch in text:
                        charset.add(ch)

    return sorted(list(charset))


if __name__ == "__main__":
    folders = [
        "data/ground_truth/train_pseudo",
        "data/ground_truth"
    ]

    characters = collect_characters(folders)

    vocab = {"<blank>": 0}

    for i, ch in enumerate(characters, start=1):
        vocab[ch] = i

    os.makedirs("models", exist_ok=True)

    with open("models/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("Vocabulary size:", len(vocab))