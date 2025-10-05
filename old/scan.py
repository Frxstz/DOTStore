import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Build CNN model for text recognition
def create_text_recognition_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Load and preprocess image
def load_and_preprocess_image(image_path, roi):
    image = Image.open(image_path)

    # Crop to ROI
    cropped = image.crop((
        roi["left"],
        roi["top"],
        roi["left"] + roi["width"],
        roi["top"] + roi["height"]
    ))

    # Resize to base ROI dimensions for consistency
    cropped_resized = cropped.resize((ROI_BASE["width"], ROI_BASE["height"]))

    # Convert to numpy array and normalize
    img_array = np.array(cropped_resized) / 255.0

    return cropped, img_array

# Main execution
# Find all screenshots in .data folder
data_folder = ".data"
cropped_folder = ".cropped"

# Create cropped folder if it doesn't exist
os.makedirs(cropped_folder, exist_ok=True)

image_files = glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png"))

print(f"Found {len(image_files)} images in {data_folder} folder:")
for img_path in image_files:
    print(f"  - {img_path}")

# Process all images and extract ROIs
X_data = []
for idx, img_path in enumerate(image_files):
    # Get image size and scale ROI
    img = Image.open(img_path)
    img_size = img.size  # (width, height)
    img.close()

    roi = scale_roi(ROI_BASE, img_size, BASE_RESOLUTION)
    print(f"\nImage {idx}: {img_size[0]}x{img_size[1]}")
    print(f"  Scaled ROI: top={roi['top']}, left={roi['left']}, width={roi['width']}, height={roi['height']}")

    cropped_img, preprocessed = load_and_preprocess_image(img_path, roi)

    # Save each cropped region to .cropped folder
    output_name = os.path.join(cropped_folder, f"roi_cropped_{idx}.jpg")
    cropped_img.save(output_name)
    print(f"  - ROI saved to {output_name}")
    print(f"  - Shape: {preprocessed.shape}")

    X_data.append(preprocessed)

# Convert to numpy array
X_data = np.array(X_data)
print(f"\nTotal data shape: {X_data.shape}")

# Now you need to label your data
print("\n" + "="*60)
print("NEXT STEP: Label your data")
print("="*60)
print("Check the roi_cropped_*.jpg files and identify what text is in each.")
print("Then update the labels list below in the code:")
print("Example: labels = [0, 1, 0] if images 0 and 2 have same text\n")

# TODO: Update these labels based on what you see in the cropped ROIs
# Each label should correspond to the image at the same index
labels = [0, 1, 2]  # All 3 images have different text

if labels is not None and len(labels) == len(X_data):
    y_train = np.array(labels)

    # Shuffle the data
    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    X_data = X_data[indices]
    y_train = y_train[indices]

    # Create and train model
    num_classes = len(set(labels))
    print(f"\nTraining model with {len(X_data)} samples and {num_classes} classes...")

    # Use the base ROI dimensions for the model input shape
    model = create_text_recognition_model(
        input_shape=(ROI_BASE["height"], ROI_BASE["width"], 3),
        num_classes=num_classes
    )

    print(model.summary())

    # With only 3 samples, skip validation split and train on all data
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )

    # Train the model without validation (too few samples)
    history = model.fit(
        datagen.flow(X_data, y_train, batch_size=1),
        epochs=100,
        verbose=1
    )

    # Save the model
    model.save('text_recognition_model.h5')
    print("\nModel saved to text_recognition_model.h5")

    # Make predictions on all samples
    print("\nPredictions on training data:")
    for i, img_data in enumerate(X_data):
        predictions = model.predict(np.expand_dims(img_data, axis=0), verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        print(f"Image {i}: Predicted class {predicted_class}, Confidence: {confidence:.2f}")
else:
    print("Set the 'labels' variable to train the model!")
