from pathlib import Path

CHARS = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"

char_to_idx = {char: i for i, char in enumerate(CHARS)}

label_file = Path(
    r"C:\Users\sasik\License_Plate_Project\recognition_dataset\labels\video_15.txt"
)

text = label_file.read_text(encoding="utf-8").strip().upper()

encoded = [char_to_idx[c] for c in text]

print("Plate text :", text)
print("Encoded    :", encoded)
print("Length     :", len(encoded))
print("Vocabulary :", len(CHARS))
print("CTC blank  : 35")
