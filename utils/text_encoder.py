import json

class TextEncoder:

    def __init__(self, vocab_path):

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.char_to_idx = json.load(f)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

    def encode(self, text):

        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):

        return "".join([self.idx_to_char[i] for i in indices if i in self.idx_to_char])