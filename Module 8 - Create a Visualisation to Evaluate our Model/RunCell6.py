import numpy as np
from sklearn.metrics import confusion_matrix
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

# Load IMDB dataset
print("Loading dataset...")
dataset = load_dataset("imdb", split="test")
dataset = dataset.shuffle(seed=42).select(range(1000))
print(f"✅ Dataset loaded {len(dataset)} reviews ready for evaluation!")

# Predict sentiment
def predict_sentiment(texts, batch_size=32):
    predictions = []
    totalbatches = (len(texts) + batch_size - 1) // batch_size
    print(f"Processing {len(texts)} reviews in {totalbatches} batches...")
    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True, max_length=512)
        predictions.extend(results)
        if batch_num % 10 == 0 or batch_num == totalbatches:

# Calculate confusion matrix
cm = confusion_matrix(true_labels, pred_labels)