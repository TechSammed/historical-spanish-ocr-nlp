from torch.utils.data import DataLoader
from utils.dataset import LineDataset
from utils.text_encoder import TextEncoder
import torch


def collate_fn(batch):

    images = []
    labels = []
    label_lengths = []

    for img, label, length in batch:

        images.append(img)
        labels.extend(label.tolist())
        label_lengths.append(length)

    images = torch.stack(images)

    labels = torch.tensor(labels)

    label_lengths = torch.tensor(label_lengths)

    return images, labels, label_lengths


encoder = TextEncoder("models/vocab.json")

dataset = LineDataset(
    "data/train_lines",
    "data/train_line_labels",
    encoder
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for images, labels, lengths in loader:

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Lengths:", lengths)

    break