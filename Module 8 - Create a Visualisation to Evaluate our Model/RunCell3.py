from transformers import pipeline
from datasets import load_dataset
from collections import Counter

# Load the model

print("Loading model...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("✅ Model loaded successfully!")

# Load IMDB dataset (we'll use a small subset for speed)

print("Loading dataset...")
dataset = load_dataset("imdb", split="test")
dataset = dataset.shuffle(seed=42).select(range(1000))
print(f"✅ Dataset loaded {len(dataset)} reviews ready for evaluation!")

# Verify balance

label_counts = Counter(dataset["label"])
print(f"...Negative reviews: {label_counts[0]}")
print(f"...Positive reviews: {label_counts[1]}")