import torch
import torch.nn as nn


class CNNBiGRUCTC(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Convert CNN features into sequence
        self.sequence_projection = nn.Linear(512 * 8, 256)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # CTC classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)

        # x: [batch, channels, height, width]
        batch, channels, height, width = x.size()

        # Width becomes sequence length
        x = x.permute(0, 3, 1, 2)

        # Flatten channels + height
        x = x.reshape(batch, width, channels * height)

        x = self.sequence_projection(x)

        x, _ = self.gru(x)

        x = self.classifier(x)

        # CTC expects [sequence, batch, classes]
        x = x.permute(1, 0, 2)

        return x
