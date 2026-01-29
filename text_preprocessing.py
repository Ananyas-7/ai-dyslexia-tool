import os
import zipfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils.readability_utils import calculate_readability_scores

MODEL_ZIP = "./models/t5_readability.zip"
MODEL_DIR = "./models/t5_readability"

# --- Automatically extract the model if not already extracted ---
if not os.path.exists(MODEL_DIR):
    print("Extracting T5 model...")
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("Model extracted successfully!")

# --- Load model and tokenizer ---
print("Loading T5 readability model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
print("Model loaded successfully!")

class ReadabilityEnhancer:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def enhance_text(self, text):
        """
        Uses the fine-tuned T5 model to simplify and correct text.
        Returns both processed text and readability metrics.
        """
        if not text.strip():
            return "Please enter some text.", None, None

        # Prepare model input
        input_text = "simplify: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute readability before and after
        original_score = calculate_readability_scores(text)
        simplified_score = calculate_readability_scores(simplified)

        # Save for teacher feedback
        os.makedirs("output", exist_ok=True)
        with open("output/processed_text.txt", "w", encoding="utf-8") as f:
            f.write("=== Original Text ===\n")
            f.write(text.strip() + "\n\n")
            f.write("=== Simplified Text ===\n")
            f.write(simplified.strip() + "\n\n")
            f.write("=== Readability Scores ===\n")
            f.write(f"Before: {original_score}\nAfter: {simplified_score}\n")

        # ✅ Return three values now
        return simplified, original_score, simplified_score

