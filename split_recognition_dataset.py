import random
import shutil
from pathlib import Path

base = Path(r"C:\Users\sasik\License_Plate_Project\recognition_dataset")

images = base / "images"
labels = base / "labels"

train_images = base / "train" / "images"
train_labels = base / "train" / "labels"

val_images = base / "val" / "images"
val_labels = base / "val" / "labels"

pairs = []

for img in images.glob("*.jpg"):
    label = labels / f"{img.stem}.txt"

    if label.exists():
        pairs.append((img, label))

random.seed(42)
random.shuffle(pairs)

split = int(len(pairs) * 0.8)

train_pairs = pairs[:split]
val_pairs = pairs[split:]

for img, label in train_pairs:
    shutil.copy2(img, train_images / img.name)
    shutil.copy2(label, train_labels / label.name)

for img, label in val_pairs:
    shutil.copy2(img, val_images / img.name)
    shutil.copy2(label, val_labels / label.name)

print(f"Total pairs: {len(pairs)}")
print(f"Training pairs: {len(train_pairs)}")
print(f"Validation pairs: {len(val_pairs)}")
