print("="*70)
print("SAVING FINE-TUNED MODEL")
print("="*70)

# Save model and tokenizer
output_dir = "./fine_tuned_sentiment_model"

print(f"\n💾 Saving model to: {output_dir}")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("   ✅ Model saved!")
print("   ✅ Tokenizer saved!")

print(f"\n🔍 Saved files:")
import os
for file in os.listdir(output_dir):
    file_size = os.path.getsize(os.path.join(output_dir, file)) / (1024*1024)  # Size in MB
    print(f"   - {file} ({file_size:.2f} MB)")

# Create a zip file
print(f"\n🗜️ Creating zip file...")
!zip -r fine_tuned_sentiment_model.zip {output_dir}

print("   ✅ Zip file created!")

# Download the zip file
print(f"\n📥 Downloading model...")
from google.colab import files
files.download('fine_tuned_sentiment_model.zip')

print("   ✅ Download started!")
print("\n💡 The zip file should download to your computer's Downloads folder.")
print("   You can extract it and use it later by loading it with:")
print("   model = AutoModelForSequenceClassification.from_pretrained('path/to/extracted/folder')")
print("   tokenizer = AutoTokenizer.from_pretrained('path/to/extracted/folder')")

print("\n" + "="*70)