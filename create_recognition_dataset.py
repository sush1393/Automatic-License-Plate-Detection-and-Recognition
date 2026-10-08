import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

src = Path(r"C:\Users\sasik\License_Plate_Project\dataset_raw\video_images")
out = Path(r"C:\Users\sasik\License_Plate_Project\recognition_dataset")

n = 0
fail = 0

for xml_file in src.glob("*.xml"):
    try:
        root = ET.parse(xml_file).getroot()

        obj = root.find("object")
        bbox = obj.find("bndbox")
        filename = root.find("filename").text
        label = obj.find("name").text.strip()

        image = cv2.imread(str(src / filename))

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        crop = image[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            fail += 1
            continue

        stem = Path(filename).stem

        cv2.imwrite(
            str(out / "images" / f"{stem}.jpg"),
            crop
        )

        (out / "labels" / f"{stem}.txt").write_text(
            label,
            encoding="utf-8"
        )

        n += 1

    except Exception as e:
        print(f"Failed: {xml_file.name} -> {e}")
        fail += 1

print(f"Crops generated: {n}")
print(f"Failed: {fail}")
