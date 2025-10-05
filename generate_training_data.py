from PIL import Image, ImageDraw, ImageFont
import random
import os
import csv

# Create output folder
output_folder = ".generated"
os.makedirs(output_folder, exist_ok=True)

# Read item names from CSV
labels_text = []
print("Reading item names from Marketplace_Data.csv...")
with open('Marketplace_Data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        item_name = row['item_name']
        if item_name and item_name.strip():  # Only add non-empty names
            # Clean up HTML entities
            item_name = item_name.replace('&quot;', '"')
            item_name = item_name.replace('&amp;', '&')
            item_name = item_name.replace('&lt;', '<')
            item_name = item_name.replace('&gt;', '>')
            labels_text.append(item_name)

print(f"Found {len(labels_text)} unique item names")

# Image dimensions (based on ROI)
width = 320
height = 170

# Try to load Eurostile font, fallback to default
try:
    # Try common paths for Eurostile
    font_paths = [
        "C:/Windows/Fonts/Eurostile.ttf",
        "C:/Windows/Fonts/eurostile.ttf",
        "C:/Windows/Fonts/EUROSTIL.TTF",
        "/usr/share/fonts/truetype/eurostile/Eurostile.ttf",
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            font = ImageFont.truetype(path, 16)
            print(f"Using font: {path}")
            break

    if font is None:
        # Try Arial as fallback
        font = ImageFont.truetype("arial.ttf", 16)
        print("Eurostile not found, using Arial")
except:
    print("Using default font (Eurostile not found)")
    font = ImageFont.load_default()

# Generate variations for each label
samples_per_label = 20  # Reduced from 50 since we have many more labels
total_images = 0

print(f"\nGenerating {samples_per_label} variations for each of {len(labels_text)} labels...")
print(f"Total images to generate: {samples_per_label * len(labels_text)}\n")

for label_idx, text in enumerate(labels_text):
    for variation in range(samples_per_label):
        # Create image with dark background
        bg_color = random.randint(10, 30)  # Dark background variation
        img = Image.new('RGB', (width, height), (bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(img)

        # Text color (white with slight variation)
        text_color = random.randint(240, 255)
        color = (text_color, text_color, text_color)

        # Draw text at slightly varying positions
        x_offset = random.randint(15, 20)
        y_offset = random.randint(18, 22)

        draw.text((x_offset, y_offset), text, fill=color, font=font)

        # Optional: Add slight noise
        if random.random() < 0.3:
            # Add some noise pixels
            pixels = img.load()
            for _ in range(random.randint(50, 150)):
                px = random.randint(0, width-1)
                py = random.randint(0, height-1)
                noise_val = random.randint(20, 60)
                pixels[px, py] = (noise_val, noise_val, noise_val)

        # Save image
        filename = f"gen_{label_idx:04d}_{variation:03d}.png"
        filepath = os.path.join(output_folder, filename)
        img.save(filepath)

        total_images += 1

    if (label_idx + 1) % 100 == 0:
        print(f"Generated {(label_idx + 1) * samples_per_label} images...")

print(f"\n✓ Done! Generated {total_images} images in {output_folder}/")
print(f"\nNow create a labels file for training...")

# Create labels file
labels_file = os.path.join(output_folder, "labels.txt")
with open(labels_file, 'w', encoding='utf-8') as f:
    for label_idx, text in enumerate(labels_text):
        for variation in range(samples_per_label):
            filename = f"gen_{label_idx:04d}_{variation:03d}.png"
            f.write(f"{filename}\t{text}\n")

print(f"✓ Labels file saved to {labels_file}")
print(f"\nSummary:")
print(f"  - {len(labels_text)} unique item names")
print(f"  - {samples_per_label} variations per item")
print(f"  - {total_images} total training images")
