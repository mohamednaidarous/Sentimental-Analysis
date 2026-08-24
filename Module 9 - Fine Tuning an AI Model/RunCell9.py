import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

output_dir = "./fine_tuned_model"

if os.path.exists(output_dir):
    print(f"Found existing fine-tuned model at {output_dir}, loading it...")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
else:
    print("No saved model found — please run the training script (RunCell8.py) first.")
    raise SystemExit(1)

print("="*70)
print("FINE-TUNED MODEL SAVED LOCALLY")
print("="*70)

print(f"\nModel location: {os.path.abspath(output_dir)}")
print(f"\nSaved files:")
for file in os.listdir(output_dir):
    file_size = os.path.getsize(os.path.join(output_dir, file)) / (1024*1024)
    print(f"   - {file} ({file_size:.2f} MB)")

print("\nYou can load it later with:")
print("   model = AutoModelForSequenceClassification.from_pretrained('fine_tuned_model')")
print("   tokenizer = AutoTokenizer.from_pretrained('fine_tuned_model')")
print("\n" + "="*70)