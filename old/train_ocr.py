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
characters = string.ascii_letters + string.digits + " .-_()/:µ"
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

max_length = 50  # Maximum text length

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
    encoded = char_to_num(tf.strings.unicode_split(text, input_encoding="UTF-8"))
    # Pad to max_length
    length = tf.shape(encoded)[0]
    pad_amount = max_length - length
    return tf.pad(encoded, [[0, pad_amount]], constant_values=0)

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len,), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def decode_predictions(pred):
    """Decode numerical predictions back to text"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def build_ocr_model():
    """Build CTC-based OCR model"""
    # Input layer
    input_img = layers.Input(shape=(ROI_BASE["height"], ROI_BASE["width"], 3), name="image")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN
    new_shape = ((ROI_BASE["height"] // 8), (ROI_BASE["width"] // 8) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    # CTC loss layer
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam())

    return model

# Labels for your 3 training images (UPDATE THESE!)
# Based on the cropped images you have
labels_text = [
    "Geist Armor Core ASD Edition",  # roi_cropped_0
    'Fresnel "Rockfall" Energy LMG',  # roi_cropped_1
    "Corbel Legs Halcyon",           # roi_cropped_2
]

print(f"Character vocabulary: {char_to_num.get_vocabulary()}")
print(f"Total characters: {len(char_to_num.get_vocabulary())}")

# Load training data
data_folder = ".data"
image_files = glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png"))
image_files.sort()  # Ensure consistent ordering

print(f"\nLoading {len(image_files)} training images...")

X_train = []
y_train = []

for idx, (img_path, label_text) in enumerate(zip(image_files, labels_text)):
    img = Image.open(img_path)
    img_size = img.size
    img.close()

    roi = scale_roi(ROI_BASE, img_size, BASE_RESOLUTION)
    _, preprocessed = load_and_preprocess_image(img_path, roi)

    # Encode the label
    encoded_label = encode_text(label_text)

    X_train.append(preprocessed)
    y_train.append(encoded_label.numpy())

    print(f"  {idx}: {os.path.basename(img_path)} -> '{label_text}'")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")

# Build and train model
print("\nBuilding OCR model...")
model = build_ocr_model()
print(model.summary())

print("\nTraining model...")
# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(1).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Train
epochs = 200
history = model.fit(train_dataset, epochs=epochs, verbose=1)

# Save the model
model.save('ocr_model.h5')
print("\nModel saved to ocr_model.h5")

# Create prediction model (without CTC layer)
prediction_model = keras.models.Model(
    inputs=model.get_layer(name="image").input,
    outputs=model.get_layer(name="dense2").output
)
prediction_model.save('ocr_prediction_model.h5')
print("Prediction model saved to ocr_prediction_model.h5")

# Test predictions on training data
print("\nTesting on training data:")
preds = prediction_model.predict(X_train)
pred_texts = decode_predictions(preds)

for i, (original, predicted) in enumerate(zip(labels_text, pred_texts)):
    print(f"Image {i}:")
    print(f"  Original:  '{original}'")
    print(f"  Predicted: '{predicted}'")
    print()
