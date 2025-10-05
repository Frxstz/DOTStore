from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
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

print("Loading fine-tuned TrOCR model...")
processor = TrOCRProcessor.from_pretrained("./trocr_finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./trocr_finetuned")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()

# Test on images from .rawtest folder
data_folder = ".rawtest"
image_files = glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png"))

print(f"\nExtracting text from {len(image_files)} images using fine-tuned TrOCR:\n")

for idx, img_path in enumerate(image_files):
    # Load image
    image = Image.open(img_path)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_size = image.size

    # Scale ROI
    roi = scale_roi(ROI_BASE, img_size, BASE_RESOLUTION)

    # Crop to ROI
    cropped = image.crop((
        roi["left"],
        roi["top"],
        roi["left"] + roi["width"],
        roi["top"] + roi["height"]
    ))

    # Process and predict
    pixel_values = processor(cropped, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Image {idx}: {os.path.basename(img_path)}")
    print(f"  Detected text: '{generated_text}'")
    print()
