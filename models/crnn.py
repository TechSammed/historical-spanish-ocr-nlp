import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # H/2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # H/4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # reduce height only

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # reduce height

            nn.AdaptiveAvgPool2d((1, None))
        )

        # BiLSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)

        features = self.cnn(x)
        # shape: (B, C, H', W')

        b, c, h, w = features.size()


        features = features.squeeze(2)  # (B, C, W)
        features = features.permute(0, 2, 1)  # (B, W, C)

        rnn_out, _ = self.rnn(features)  # (B, W, 512)

        logits = self.fc(rnn_out)  # (B, W, num_classes)

        return logits