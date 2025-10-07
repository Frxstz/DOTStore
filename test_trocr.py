from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import glob
import json
from collections import defaultdict

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

def calculate_metrics(predictions, ground_truths):
    """Calculate accuracy metrics for predictions"""
    if len(predictions) != len(ground_truths):
        print("Warning: Number of predictions doesn't match ground truths")

    exact_matches = 0
    char_correct = 0
    total_chars = 0

    for pred, truth in zip(predictions, ground_truths):
        # Exact match
        if pred.strip() == truth.strip():
            exact_matches += 1

        # Character-level accuracy
        truth_chars = list(truth.strip())
        pred_chars = list(pred.strip())

        # Calculate edit distance for character accuracy
        for tc, pc in zip(truth_chars, pred_chars):
            if tc == pc:
                char_correct += 1
        total_chars += len(truth_chars)

    accuracy = exact_matches / len(predictions) * 100 if predictions else 0
    char_accuracy = char_correct / total_chars * 100 if total_chars > 0 else 0

    return {
        "exact_match_accuracy": accuracy,
        "character_accuracy": char_accuracy,
        "total_samples": len(predictions),
        "correct_predictions": exact_matches
    }

if __name__ == '__main__':
    print("Loading fine-tuned TrOCR model...")
    processor = TrOCRProcessor.from_pretrained("./trocr_finetuned")
    model = VisionEncoderDecoderModel.from_pretrained("./trocr_finetuned")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Test on validation data from .generated folder
    print("\n" + "="*60)
    print("Testing on Validation Set")
    print("="*60)

    generated_folder = ".generated"
    labels_file = os.path.join(generated_folder, "labels.txt")

    if os.path.exists(labels_file):
        # Load validation data (last 10% of dataset)
        val_predictions = []
        val_ground_truths = []

        with open(labels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Use last 10% as validation (matching train script)
            val_start_idx = int(0.9 * len(lines))
            val_lines = lines[val_start_idx:]

            # Sample diverse labels for testing
            import random
            label_groups = defaultdict(list)
            for i, line in enumerate(val_lines):
                label = line.strip().split('\t')[1]
                label_groups[label].append((i, line))

            # Take up to 2 samples per unique label (ensures diversity)
            sampled_lines = []
            for label, samples in label_groups.items():
                sampled_lines.extend(random.sample(samples, min(2, len(samples))))

            # Limit to 100 samples total and sort by index
            random.shuffle(sampled_lines)
            sampled_lines = sorted(sampled_lines[:100], key=lambda x: x[0])

            print(f"\nTesting on {len(sampled_lines)} diverse validation samples ({len(label_groups)} unique labels)...")

            for idx, (_, line) in enumerate(sampled_lines):
                filename, label = line.strip().split('\t')
                image_path = os.path.join(generated_folder, filename)

                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=64,
                            num_beams=4,
                            early_stopping=True,
                            no_repeat_ngram_size=3,
                            repetition_penalty=2.0
                        )
                    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    val_predictions.append(predicted_text)
                    val_ground_truths.append(label)

                    # Show all examples
                    match = "✓" if predicted_text.strip() == label.strip() else "✗"
                    print(f"{match} [{idx+1:3d}] Expected: '{label}' | Got: '{predicted_text}'")

        # Calculate and display metrics
        if val_predictions:
            metrics = calculate_metrics(val_predictions, val_ground_truths)
            print("\n" + "-"*60)
            print("Validation Results:")
            print("-"*60)
            print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
            print(f"Character Accuracy:   {metrics['character_accuracy']:.2f}%")
            print(f"Correct Predictions:  {metrics['correct_predictions']}/{metrics['total_samples']}")

    # Test on raw images from .rawtest folder
    print("\n" + "="*60)
    print("Testing on Raw Images (.rawtest)")
    print("="*60)

    data_folder = ".rawtest"
    if os.path.exists(data_folder):
        image_files = glob.glob(os.path.join(data_folder, "*.jpg")) + glob.glob(os.path.join(data_folder, "*.png"))

        if image_files:
            print(f"\nProcessing {len(image_files)} raw images:\n")

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
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=64,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=2.0
                    )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                print(f"[{idx+1}] {os.path.basename(img_path)}: '{generated_text}'")
        else:
            print(f"No images found in {data_folder}")
    else:
        print(f"Directory {data_folder} not found")

    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)
