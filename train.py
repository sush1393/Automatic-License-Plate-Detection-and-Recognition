import sys
sys.path.insert(0, r"C:\Users\sasik\License_Plate_Project")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from recognition.dataset import LicensePlateDataset
from recognition.model import CNNBiGRUCTC


BASE = r"C:\Users\sasik\License_Plate_Project"

train_dataset = LicensePlateDataset(
    BASE + r"\recognition_dataset\train\images",
    BASE + r"\recognition_dataset\train\labels",
    train=True
)

val_dataset = LicensePlateDataset(
    BASE + r"\recognition_dataset\val\images",
    BASE + r"\recognition_dataset\val\labels",
    train=False
)


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    target_lengths = torch.tensor(
        [len(item[1]) for item in batch],
        dtype=torch.long
    )
    texts = [item[2] for item in batch]

    return images, targets, target_lengths, texts


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


model = CNNBiGRUCTC(num_classes=36).to(device)

criterion = nn.CTCLoss(
    blank=35,
    zero_infinity=True
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)


def calculate_loss(loader, training=False):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(training):

        for images, targets, target_lengths, texts in loader:

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            input_lengths = torch.full(
                (images.size(0),),
                outputs.size(0),
                dtype=torch.long
            )

            loss = criterion(
                outputs.log_softmax(2),
                targets,
                input_lengths,
                target_lengths
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


best_val_loss = float("inf")

epochs = 30

for epoch in range(epochs):

    train_loss = calculate_loss(
        train_loader,
        training=True
    )

    val_loss = calculate_loss(
        val_loader,
        training=False
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BASE + r"\recognition\best_model.pth"
        )

        print("Best model saved.")


print("Training completed.")
print("Best validation loss:", best_val_loss)
