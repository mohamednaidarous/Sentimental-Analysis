import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import pipeline
from datasets import load_dataset
from collections import Counter

print("Loading model...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded successfully!")

print("Loading dataset...")
dataset = load_dataset("imdb", split="test")
dataset = dataset.shuffle(seed=42).select(range(1000))
print(f"Dataset loaded {len(dataset)} reviews ready for evaluation!")

def predict_sentiment(texts, batch_size=32):
    predictions = []
    totalbatches = (len(texts) + batch_size - 1) // batch_size
    print(f"Processing {len(texts)} reviews in {totalbatches} batches...")
    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True, max_length=512)
        predictions.extend(results)
        if batch_num % 10 == 0 or batch_num == totalbatches:
            print(f"Processed batch {batch_num}/{totalbatches}")
    return predictions

print("Making predictions on 1000 reviews...")
texts = dataset["text"]
predictions = predict_sentiment(texts)
pred_labels = [1 if p["label"] == "POSITIVE" else 0 for p in predictions]
true_labels = list(dataset["label"])
print(f"Converted {len(pred_labels)} predictions to binary labels.")

cm = confusion_matrix(true_labels, pred_labels)

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()
print("="*70)
print("CONFUSION MATRIX BREAKDOWN")
print("="*70)
print(f"\nTrue Negatives (TN): {tn}")
print(f" Correctly identified as NEGATIVE: {tn} reviews")
print(f"\nFalse Positives (FP): {fp}")
print(f" Incorrectly identified as POSITIVE: {fp} reviews")
print(f"\nFalse Negatives (FN): {fn}")
print(f" Incorrectly identified as NEGATIVE: {fn} reviews")
print(f"\nTrue Positives (TP): {tp}")
print(f" Correctly identified as POSITIVE: {tp} reviews")
print("\n")
print("KEY INSIGHTS:")
print("=" * 70)
false_positive_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
print(f"\n False Positive Rate: {false_positive_rate:.2f}%")
print(f"  (Model incorrectly calls negative reviews positive)")
print(f"\n False Negative Rate: {false_negative_rate:.2f}%")
print(f"  (Model incorrectly calls positive reviews negative)")
if false_positive_rate > false_negative_rate:
    print("\n The model tends to be OPTIMISTIC (over predicted positive).")
elif false_negative_rate > false_positive_rate:
    print("\n The model tends to be PESSIMISTIC (over predicted negative).")
else:
    print("\n The model has a BALANCED error rate.")