import numpy as np
from PIL import Image
import easyocr
import os
import glob

# Define the ROI (Region of Interest) for 2560x1440 resolution
ROI_BASE = {
    "top": 122,
    "left": 1462,
    "width": 320,
    "height": 170,
}
BASE_RESOLUTION = (2560, 1440)

def scale_roi(roi, original_size, base_size):
    """Scale ROI coordinates based on image resolution"""
    width_ratio = original_size[0] / base_size[0]
    height_ratio = original_size[1] / base_size[1]

    scaled_roi = {
        "left": int(roi["left"] * width_ratio),
        "top": int(roi["top"] * height_ratio),
        "width": int(roi["width"] * width_ratio),
        "height": int(roi["height"] * height_ratio),
    }
    return scaled_roi

def extract_roi(image_path, roi):
    image = Image.open(image_path)
    cropped = image.crop((
        roi["left"],
        roi["top"],
        roi["left"] + roi["width"],
        roi["top"] + roi["height"]
    ))
    return cropped

# Load EasyOCR
print("Loading OCR model...")
reader = easyocr.Reader(['en'])

# Test on images from .data folder (or change to .rawtest)
data_folder = ".data"
image_files = glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png"))

print(f"\nExtracting text from {len(image_files)} images using EasyOCR:\n")

for idx, img_path in enumerate(image_files):
    # Get image size and scale ROI
    img = Image.open(img_path)
    img_size = img.size
    img.close()

    roi = scale_roi(ROI_BASE, img_size, BASE_RESOLUTION)

    # Extract ROI
    cropped = extract_roi(img_path, roi)
    cropped_array = np.array(cropped)

    # OCR
    results = reader.readtext(cropped_array)

    print(f"Image {idx}: {os.path.basename(img_path)}")
    if results:
        # Get first line of text (usually the item name)
        first_line = results[0][1] if results else "No text detected"
        print(f"  Detected text: '{first_line}'")
        if len(results) > 1:
            print(f"  Additional lines: {[r[1] for r in results[1:]]}")
    else:
        print(f"  No text detected")
    print()
