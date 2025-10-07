from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import os
import time

class OCRDataset(Dataset):
    def __init__(self, image_files, labels, processor):
        self.image_files = image_files
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image directly (already cropped)
        image = Image.open(self.image_files[idx])

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Process label
        labels = self.processor.tokenizer(self.labels[idx],
                                         padding="max_length",
                                         max_length=64,
                                         truncation=True).input_ids

        # Replace padding token id with -100 for loss calculation
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

if __name__ == '__main__':
    # Load labels from generated synthetic data
    print("Loading labels from .generated folder...")
    generated_folder = ".generated"
    labels_file = os.path.join(generated_folder, "labels.txt")

    image_files = []
    labels_text = []

    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            filename, label = line.strip().split('\t')
            image_path = os.path.join(generated_folder, filename)
            if os.path.exists(image_path):
                # Only use labels that are reasonable length
                if len(label) <= 50:
                    image_files.append(image_path)
                    labels_text.append(label)

    print(f"Loaded {len(image_files)} training samples")

    # Uncomment below to reduce dataset size for faster CPU training
    # max_samples = 10000
    # if len(image_files) > max_samples:
    #     image_files = image_files[:max_samples]
    #     labels_text = labels_text[:max_samples]
    #     print(f"Reduced to {len(image_files)} samples for faster CPU training")

    print("Loading pre-trained TrOCR model...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    # Set special tokens correctly for TrOCR
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search params
    model.config.num_beams = 4
    model.config.max_length = 64
    model.config.early_stopping = True

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA not available. Install CUDA-enabled PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    model.to(device)

    # Split into train/val
    split_idx = int(0.9 * len(image_files))
    train_files = image_files[:split_idx]
    train_labels = labels_text[:split_idx]
    val_files = image_files[split_idx:]
    val_labels = labels_text[split_idx:]

    print(f"\nTraining samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

    # Create datasets
    train_dataset = OCRDataset(train_files, train_labels, processor)
    val_dataset = OCRDataset(val_files, val_labels, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Mixed precision training setup
    scaler = GradScaler('cuda')

    # Fine-tune
    epochs = 10
    print(f"\nFine-tuning for {epochs} epochs...")

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        epoch_start_time = time.time()
        batch_timer_start = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Mixed precision training
            with autocast('cuda'):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                batch_time = time.time() - batch_timer_start
                avg_time_per_batch = batch_time / 100
                print(f"  Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Avg Time/Batch: {avg_time_per_batch:.3f}s, Last 100 batches: {batch_time:.1f}s")
                batch_timer_start = time.time()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.1f}s")

        # Quick sanity check - test on a few training samples
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:  # Every 2 epochs and last epoch
            print(f"\n  Quick sanity check (epoch {epoch+1}):")
            model.eval()
            with torch.no_grad():
                for i in range(3):  # Test 3 random samples
                    test_img = Image.open(train_files[i])
                    if test_img.mode != 'RGB':
                        test_img = test_img.convert('RGB')
                    test_pixels = processor(test_img, return_tensors="pt").pixel_values.to(device)
                    gen_ids = model.generate(test_pixels, max_length=64, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=2.0)
                    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                    print(f"    Expected: '{train_labels[i]}' | Got: '{gen_text}'")
            print()

    # Save the fine-tuned model
    output_dir = "./trocr_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}/")

    # Test on validation samples
    print("\nTesting on random validation samples:")
    model.eval()

    import random
    test_indices = random.sample(range(len(val_files)), min(10, len(val_files)))

    for idx in test_indices:
        image = Image.open(val_files[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"\nImage: {os.path.basename(val_files[idx])}")
        print(f"  Original:  '{val_labels[idx]}'")
        print(f"  Predicted: '{generated_text}'")
