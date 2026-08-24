import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Sentiment Analysis")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()