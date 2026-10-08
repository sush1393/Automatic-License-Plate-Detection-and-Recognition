import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms


CHARS = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {char: i for i, char in enumerate(CHARS)}
BLANK_IDX = len(CHARS)


class LicensePlateDataset(Dataset):

    def __init__(self, image_dir, label_dir, train=False):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.train = train

        self.images = sorted(self.image_dir.glob("*.jpg"))

        # Training transformations
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((64, 160)),

                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2
                    )
                ], p=0.5),

                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 1.0)
                    )
                ], p=0.2),

                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

        # Validation transformations
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64, 160)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        text = label_path.read_text(
            encoding="utf-8"
        ).strip().upper()

        encoded = torch.tensor(
            [CHAR_TO_IDX[c] for c in text],
            dtype=torch.long
        )

        return image, encoded, text