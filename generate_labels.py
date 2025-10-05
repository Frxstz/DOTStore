from PIL import Image
import easyocr
import os
import glob
import numpy as np

# Initialize EasyOCR
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'])

# Find all cropped images starting with "cropped_"
cropped_folder = ".cropped"
all_files = glob.glob(os.path.join(cropped_folder, "cropped_*.jpg")) + glob.glob(os.path.join(cropped_folder, "cropped_*.png"))

# Sort by the index number in the filename
def get_index(filename):
    basename = os.path.basename(filename)
    # Extract number after "cropped_"
    try:
        return int(basename.split('_')[1])
    except:
        return 999999

image_files = sorted(all_files, key=get_index)

print(f"\nFound {len(image_files)} cropped images\n")
print("Generating labels...\n")
print("="*70)

labels = []

for idx, img_path in enumerate(image_files):
    # Read the cropped image
    img = Image.open(img_path)
    img_array = np.array(img)

    # Perform OCR
    results = reader.readtext(img_array)

    # Get the first line of text (usually the item name)
    if results:
        first_line = results[0][1]
        labels.append(first_line)

        print(f"{idx}: {os.path.basename(img_path)}")
        print(f"   Detected: '{first_line}'")
        if len(results) > 1:
            print(f"   (Other text: {[r[1] for r in results[1:]]})")
    else:
        labels.append("UNKNOWN")
        print(f"{idx}: {os.path.basename(img_path)}")
        print(f"   NO TEXT DETECTED")

    print()

print("="*70)
print("\nCopy this labels_text list into train_ocr_simple.py:\n")
print("labels_text = [")
for label in labels:
    # Escape quotes in the label
    escaped_label = label.replace("'", "\\'")
    print(f"    '{escaped_label}',")
print("]")
print(f"\nTotal: {len(labels)} labels")
