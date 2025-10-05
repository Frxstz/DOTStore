import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import string

# Define image dimensions
IMG_WIDTH = 320
IMG_HEIGHT = 170

# Character set for OCR (alphanumeric + common symbols)
characters = string.ascii_letters + string.digits + ' .-_()/:"\'-&""?*+[]<>,'
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for idx, char in enumerate(characters)}
num_classes = len(characters) + 1  # +1 for blank

max_length = 64  # Maximum text length

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    return img_array

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
    input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="image")

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN - use width as time steps
    # After 3 pooling layers: width //8 = 320//8 = 40, height //8 = 170//8 = 21
    new_shape = ((IMG_WIDTH // 8), (IMG_HEIGHT // 8) * 128)
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

print(f"Character vocabulary: {characters}")
print(f"Total characters: {num_classes}")

# Load generated training data
generated_folder = ".generated"
labels_file = os.path.join(generated_folder, "labels.txt")

print(f"\nLoading training data from {labels_file}...")

image_files = []
labels_text = []

with open(labels_file, 'r', encoding='utf-8') as f:
    for line in f:
        filename, label = line.strip().split('\t')
        image_path = os.path.join(generated_folder, filename)
        if os.path.exists(image_path):
            image_files.append(image_path)
            labels_text.append(label)

print(f"Loaded {len(image_files)} training images")

# Create generator function
def data_generator(image_files, labels_text, batch_size):
    num_samples = len(image_files)
    indexes = np.arange(num_samples)

    while True:
        np.random.shuffle(indexes)

        for start_idx in range(0, num_samples, batch_size):
            batch_indexes = indexes[start_idx:start_idx + batch_size]

            batch_images = []
            batch_labels = []
            batch_input_lengths = []
            batch_label_lengths = []

            for idx in batch_indexes:
                img_path = image_files[idx]
                label_text = labels_text[idx]

                # Skip labels that are too long for CTC (need some margin)
                if len(label_text) > 38:
                    continue

                preprocessed = load_and_preprocess_image(img_path)
                encoded_label = encode_text(label_text)

                batch_images.append(preprocessed)
                batch_labels.append(encoded_label)
                batch_input_lengths.append([40])
                batch_label_lengths.append([len(label_text)])

            # Skip empty batches
            if len(batch_images) == 0:
                continue

            yield (
                (np.array(batch_images, dtype=np.float32),
                 np.array(batch_labels, dtype=np.float32),
                 np.array(batch_input_lengths, dtype=np.int64),
                 np.array(batch_label_lengths, dtype=np.int64)),
                np.zeros(len(batch_images), dtype=np.float32)
            )

print(f"\nTotal training samples: {len(image_files)}")

# Split into train/validation
split_idx = int(0.9 * len(image_files))
train_files = image_files[:split_idx]
train_labels = labels_text[:split_idx]
val_files = image_files[split_idx:]
val_labels = labels_text[split_idx:]

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# Build and train model
print("\nBuilding OCR model...")
model, model_pred = build_ocr_model()
print(model.summary())

print("\nTraining model...")
# Train
epochs = 20
batch_size = 32

steps_per_epoch = len(train_files) // batch_size
validation_steps = len(val_files) // batch_size

# Create TensorFlow datasets from the generator
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_files, train_labels, batch_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, max_length), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_files, val_labels, batch_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, max_length), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    epochs=epochs,
    verbose=1
)

# Save the prediction model
model_pred.save('ocr_model_final.h5')
print("\nPrediction model saved to ocr_model_final.h5")

# Test predictions on a few samples
print("\nTesting on random validation samples:")
test_indices = np.random.choice(len(val_files), min(10, len(val_files)), replace=False)

for idx in test_indices:
    preprocessed = load_and_preprocess_image(val_files[idx])
    pred = model_pred.predict(np.expand_dims(preprocessed, axis=0), verbose=0)
    pred_text = decode_predictions(pred)[0]

    print(f"\nImage: {os.path.basename(val_files[idx])}")
    print(f"  Original:  '{val_labels[idx]}'")
    print(f"  Predicted: '{pred_text}'")
