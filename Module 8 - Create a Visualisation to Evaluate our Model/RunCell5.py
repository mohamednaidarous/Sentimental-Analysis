import numpy as np
from sklearn.metrics import accuracy_score, classification_report
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
            print(f"Processed batch {batch_num}/{totalbatches} ")
    return predictions

print("Making predictions on 1000 reviews...")
texts = dataset['text']
predictions = predict_sentiment(texts)
pred_labels = [1 if p['label'] == 'POSITIVE' else 0 for p in predictions]
true_labels = list(dataset['label'])
print(f"\n✅ Converted {len(pred_labels)} predictions to binary labels.")
# Determine accuracy and evaluation results
accuracy = accuracy_score(true_labels, pred_labels)

print("="*70)
print("EVALUATION RESULTS")
print("="*70)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correct Predictions: {sum(np.array(true_labels) == np.array(pred_labels))} / {len(true_labels)}")

print("\nDetailed Classification Report:")
print(classification_report(true_labels, pred_labels,
                            target_names=['Negative', 'Positive']))
