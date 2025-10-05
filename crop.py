from PIL import Image
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

# Create .cropped folder if it doesn't exist
cropped_folder = ".cropped"
os.makedirs(cropped_folder, exist_ok=True)

# Find all images in .data folder
data_folder = ".data"
image_files = sorted(glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png")))

print(f"Found {len(image_files)} images in {data_folder} folder\n")

# Process each image
for idx, img_path in enumerate(image_files):
    # Load image
    image = Image.open(img_path)
    img_size = image.size

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Scale ROI based on image size
    roi = scale_roi(ROI_BASE, img_size, BASE_RESOLUTION)

    # Crop to ROI
    cropped = image.crop((
        roi["left"],
        roi["top"],
        roi["left"] + roi["width"],
        roi["top"] + roi["height"]
    ))

    # Save cropped image
    filename = os.path.basename(img_path)
    output_path = os.path.join(cropped_folder, f"cropped_{idx}_{filename}")
    cropped.save(output_path)

    print(f"Cropped {filename}")
    print(f"  Original size: {img_size[0]}x{img_size[1]}")
    print(f"  ROI: top={roi['top']}, left={roi['left']}, width={roi['width']}, height={roi['height']}")
    print(f"  Saved to: {output_path}\n")

print(f"Done! Cropped {len(image_files)} images to {cropped_folder}/")
