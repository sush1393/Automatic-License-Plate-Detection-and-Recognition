import sys
sys.path.insert(0, r"C:\Users\sasik\License_Plate_Project")

import torch
from torch.utils.data import DataLoader
from recognition.dataset import LicensePlateDataset
from recognition.model import CNNBiGRUCTC

BASE = r"C:\Users\sasik\License_Plate_Project"

CHARS = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"
BLANK = len(CHARS)

idx_to_char = {i: c for i, c in enumerate(CHARS)}

dataset = LicensePlateDataset(
    BASE + r"\recognition_dataset\val\images",
    BASE + r"\recognition_dataset\val\labels"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=lambda batch: (
        torch.stack([x[0] for x in batch]),
        [x[2] for x in batch]
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNBiGRUCTC(num_classes=36).to(device)

model.load_state_dict(
    torch.load(
        BASE + r"\recognition\best_model.pth",
        map_location=device
    )
)

model.eval()


def ctc_decode(output):
    """
    Greedy CTC decoding:
    output shape: [T, B, C]
    Convert to [B, T] before decoding.
    """

    prediction = output.argmax(dim=2).permute(1, 0)

    results = []

    for seq in prediction:
        text = []
        previous = BLANK

        for index in seq.tolist():

            if index != BLANK and index != previous:
                text.append(idx_to_char[index])

            previous = index

        results.append("".join(text))

    return results


total = 0
exact_correct = 0
total_characters = 0
correct_characters = 0

all_predictions = []
all_actual = []

with torch.no_grad():

    for images, actual_texts in loader:

        images = images.to(device)

        outputs = model(images)

        predictions = ctc_decode(outputs)

        for actual, predicted in zip(actual_texts, predictions):

            total += 1

            if actual == predicted:
                exact_correct += 1

            # Character-level accuracy
            max_len = max(len(actual), len(predicted))

            for i in range(max_len):
                total_characters += 1

                if i < len(actual) and i < len(predicted):
                    if actual[i] == predicted[i]:
                        correct_characters += 1

            all_actual.append(actual)
            all_predictions.append(predicted)


accuracy = (exact_correct / total) * 100
character_accuracy = (correct_characters / total_characters) * 100

print()
print("========================================")
print("CNN + BiGRU + CTC VALIDATION RESULTS")
print("========================================")
print(f"Total validation samples : {total}")
print(f"Exact plate matches      : {exact_correct}")
print(f"Exact plate accuracy     : {accuracy:.2f}%")
print(f"Character-level accuracy : {character_accuracy:.2f}%")
print("========================================")
print()

print("Sample predictions:")
print("----------------------------------------")

for actual, predicted in list(zip(all_actual, all_predictions))[:20]:
    status = "CORRECT" if actual == predicted else "WRONG"
    print(f"Actual: {actual:<12} Predicted: {predicted:<12} [{status}]")
