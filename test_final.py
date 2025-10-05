import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import glob
import string

# Define the ROI (Region of Interest) for 2560x1440 resolution
ROI_BASE = {
    "top": 122,
    "left": 1462,
    "width": 320,
    "height": 170,
}
BASE_RESOLUTION = (2560, 1440)

# Character set (must match training)
characters = string.ascii_letters + string.digits + ' .-_()/:"\'-&""?*+[]<>,'
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for idx, char in enumerate(characters)}
max_length = 64

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

def load_and_preprocess_image(image_path, roi):
    image = Image.open(image_path)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    cropped = image.crop((
        roi["left"],
        roi["top"],
        roi["left"] + roi["width"],
        roi["top"] + roi["height"]
    ))

    img_array = np.array(cropped) / 255.0
    return cropped, img_array

def decode_predictions(pred):
    """Decode numerical predictions back to text"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0].numpy()

    output_text = []
    for result in results:
        text = ""
        for idx in result:
            if idx > 0 and idx <= len(characters):
                text += idx_to_char[idx - 1]
        output_text.append(text)
    return output_text

# Load the trained model
print("Loading trained OCR model...")
model = keras.models.load_model('ocr_model_final.h5')

# Test on images from .rawtest folder
data_folder = ".rawtest"
image_files = sorted(glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png")))

print(f"\nExtracting text from {len(image_files)} images using custom trained OCR:\n")

for idx, img_path in enumerate(image_files):
    # Get image size and scale ROI
    img = Image.open(img_path)
    img_size = img.size
    img.close()

    roi = scale_roi(ROI_BASE, img_size, BASE_RESOLUTION)

    # Load and preprocess
    cropped, preprocessed = load_and_preprocess_image(img_path, roi)

    # Make prediction
    pred = model.predict(np.expand_dims(preprocessed, axis=0), verbose=0)
    pred_text = decode_predictions(pred)[0]

    print(f"Image {idx}: {os.path.basename(img_path)}")
    print(f"  Detected text: '{pred_text}'")
    print()
