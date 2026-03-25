print("="*70)
print("LOADING PRE-TRAINED MODEL")
print("="*70)

# Model name
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"\n🔄 Loading model: {model_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("   ✅ Tokenizer loaded")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification: negative (0) and positive (1)
)
print("   ✅ Model loaded")

print(f"\n📊 Model details:")
print(f"   Number of parameters: {model.num_parameters():,}")
print(f"   Number of labels: 2 (Negative, Positive)")

print("\n" + "="*70)