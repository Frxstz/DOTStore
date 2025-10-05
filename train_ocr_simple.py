import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

# Character set for OCR (alphanumeric + common symbols)
characters = string.ascii_letters + string.digits + ' .-_()/:"\''
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for idx, char in enumerate(characters)}
num_classes = len(characters) + 1  # +1 for blank

max_length = 40  # Maximum text length

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
    cropped = image.crop((
        roi["left"],
        roi["top"],
        roi["left"] + roi["width"],
        roi["top"] + roi["height"]
    ))
    cropped_resized = cropped.resize((ROI_BASE["width"], ROI_BASE["height"]))
    img_array = np.array(cropped_resized) / 255.0
    return cropped, img_array

def encode_text(text):
    """Encode text to numbers"""
    text = text.lower()
    encoded = []
    for char in text:
        if char in char_to_idx:
            encoded.append(char_to_idx[char] + 1)  # +1 because 0 is blank for CTC
        else:
            print(f"Warning: Character '{char}' not in vocabulary, skipping")
    # Pad to max_length
    encoded = encoded[:max_length]
    encoded += [0] * (max_length - len(encoded))
    return np.array(encoded)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_ocr_model():
    """Build CTC-based OCR model"""
    # Input layer
    input_img = layers.Input(shape=(ROI_BASE["height"], ROI_BASE["width"], 3), name="image")

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN - use width as time steps
    # After 3 pooling layers: width //8 = 320//8 = 40, height //8 = 170//8 = 21
    new_shape = ((ROI_BASE["width"] // 8), (ROI_BASE["height"] // 8) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    y_pred = layers.Dense(num_classes, activation="softmax", name="output")(x)

    # Model for prediction
    model_pred = keras.models.Model(inputs=input_img, outputs=y_pred)

    # Inputs for CTC loss
    labels = layers.Input(name="label", shape=(max_length,), dtype="float32")
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")

    # CTC loss
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    # Model for training
    model = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )

    model.compile(optimizer=keras.optimizers.Adam(), loss={"ctc": lambda y_true, y_pred: y_pred})

    return model, model_pred

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

# Labels for your 3 training images
labels_text = [
    'Geist Armor Core ASD Edition',  # roi_cropped_0
    'Fresnel "Rockfall" Energy LMG',  # roi_cropped_1
    'Corbel Legs Halcyon',           # roi_cropped_2
]

print(f"Character vocabulary: {characters}")
print(f"Total characters: {num_classes}")

# Load training data
data_folder = ".data"
image_files = sorted(glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png")))

print(f"\nLoading {len(image_files)} training images...")

X_train = []
y_train = []
input_lengths = []
label_lengths = []

for idx, (img_path, label_text) in enumerate(zip(image_files, labels_text)):
    img = Image.open(img_path)
    img_size = img.size
    img.close()

    roi = scale_roi(ROI_BASE, img_size, BASE_RESOLUTION)
    _, preprocessed = load_and_preprocess_image(img_path, roi)

    # Encode the label
    encoded_label = encode_text(label_text)

    X_train.append(preprocessed)
    y_train.append(encoded_label)

    # CTC requires input/label lengths
    input_lengths.append([40])  # Width after convolutions: 320 // 8 = 40
    label_lengths.append([len(label_text)])

    print(f"  {idx}: {os.path.basename(img_path)} -> '{label_text}' (len: {len(label_text)})")

X_train = np.array(X_train)
y_train = np.array(y_train)
input_lengths = np.array(input_lengths)
label_lengths = np.array(label_lengths)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")

# Build and train model
print("\nBuilding OCR model...")
model, model_pred = build_ocr_model()
print(model.summary())

print("\nTraining model...")
# Dummy output for CTC loss
outputs = np.zeros([len(X_train)])

# Train
epochs = 50
history = model.fit(
    [X_train, y_train, input_lengths, label_lengths],
    outputs,
    batch_size=1,
    epochs=epochs,
    verbose=1
)

# Save the prediction model
model_pred.save('ocr_prediction_model.h5')
print("\nPrediction model saved to ocr_prediction_model.h5")

# Test predictions on training data
print("\nTesting on training data:")
preds = model_pred.predict(X_train)
pred_texts = decode_predictions(preds)

for i, (original, predicted) in enumerate(zip(labels_text, pred_texts)):
    print(f"Image {i}:")
    print(f"  Original:  '{original}'")
    print(f"  Predicted: '{predicted}'")
    print()
