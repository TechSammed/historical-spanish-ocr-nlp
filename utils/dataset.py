import os
import cv2
import torch
from torch.utils.data import Dataset


class LineDataset(Dataset):

    def __init__(self, img_dir, label_dir, encoder, max_samples=1000):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.encoder = encoder

        files = sorted(os.listdir(img_dir))

        # limit dataset size for low RAM systems
        self.files = files[:max_samples]

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        img_name = self.files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".png", ".txt"))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # smaller image size for faster training
        image = cv2.resize(image, (192, 48))

        image = image / 255.0
        image = torch.tensor(image).unsqueeze(0).float()

        with open(label_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        encoded = self.encoder.encode(text)

        return image, torch.tensor(encoded), len(encoded)