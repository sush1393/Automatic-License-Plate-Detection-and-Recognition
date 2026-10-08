import sys
sys.path.insert(0, r"C:\Users\sasik\License_Plate_Project")

import torch
from torch.utils.data import DataLoader
from recognition.dataset import LicensePlateDataset
from recognition.model import CNNBiGRUCTC

train_dataset = LicensePlateDataset(
    r"C:\Users\sasik\License_Plate_Project\recognition_dataset\train\images",
    r"C:\Users\sasik\License_Plate_Project\recognition_dataset\train\labels"
)

loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),
        torch.cat([item[1] for item in batch]),
        [item[1].numel() for item in batch],
        [item[2] for item in batch]
    )
)

images, targets, target_lengths, texts = next(iter(loader))

model = CNNBiGRUCTC(num_classes=36)

outputs = model(images)

input_lengths = torch.full(
    (images.size(0),),
    outputs.size(0),
    dtype=torch.long
)

target_lengths = torch.tensor(
    target_lengths,
    dtype=torch.long
)

ctc_loss = torch.nn.CTCLoss(
    blank=35,
    zero_infinity=True
)

loss = ctc_loss(
    outputs.log_softmax(2),
    targets,
    input_lengths,
    target_lengths
)

print("Images:", images.shape)
print("CTC outputs:", outputs.shape)
print("Targets:", targets.shape)
print("Input lengths:", input_lengths.tolist())
print("Target lengths:", target_lengths.tolist())
print("Sample texts:", texts)
print("CTC loss:", loss.item())
print("CTC pipeline test: SUCCESS")
