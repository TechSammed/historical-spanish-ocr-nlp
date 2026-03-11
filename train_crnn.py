import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.crnn import CRNN
from utils.dataset import LineDataset
from utils.text_encoder import TextEncoder

print("train_crnn.py started")


def collate_fn(batch):

    images = []
    labels = []
    label_lengths = []

    for img, label, length in batch:
        images.append(img)
        labels.extend(label.tolist())
        label_lengths.append(length)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, labels, label_lengths


# Load encoder
encoder = TextEncoder("models/vocab.json")


# Dataset
dataset = LineDataset(
    "data/train_lines",
    "data/train_line_labels",
    encoder,
    max_samples=3000
)

print("Dataset loaded")
print("Dataset size:", len(dataset))


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

print("DataLoader ready")


# Model
num_classes = len(encoder.char_to_idx) + 1
model = CRNN(num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Loss
criterion = nn.CTCLoss(blank=0, zero_infinity=True)


# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


# Training epochs
epochs = 50


for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels, label_lengths in loader:

        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        logits = model(images)

        log_probs = logits.log_softmax(2)

        # sequence length for each sample
        input_lengths = torch.full(
            size=(logits.size(0),),
            fill_value=logits.size(1),
            dtype=torch.long
        )

        loss = criterion(
            log_probs.permute(1, 0, 2),
            labels,
            input_lengths,
            label_lengths
        )

        loss.backward()

        # prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    # Save checkpoint each epoch
    torch.save(model.state_dict(), f"models/crnn_epoch_{epoch+1}.pth")


# Save final model
torch.save(model.state_dict(), "models/crnn_model1.pth")

print("Training completed. Model saved.")