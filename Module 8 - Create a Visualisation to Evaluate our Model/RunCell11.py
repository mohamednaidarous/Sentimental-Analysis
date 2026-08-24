import numpy as np
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

print("="*70)
print("EXAMPLE MISCLASSIFICATIONS")
print("="*70)
errors = []
for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
    if true != pred:
        errors.append({
            'index': i,
            'text':texts[i],
            'true':'Positive' if true == 1 else 'Negative',
            'predicted':'Positive' if pred == 1 else 'Negative',
            'confidence': predictions[i]['score']
        })
print(f"\nFound {len(errors)} misclassified reviews out of {len(true_labels)}")
print(f"Error rate: {len(errors)/len(true_labels)*100:.2f}%")
high_conf_errors = [e for e in errors if e['confidence'] > 0.9]
print(f"\nHigh-confidence errors (>90% confidence): {len(high_conf_errors)}")
print("\n" + "-"*70)
print("TOP 5 MISCLASSIFIED (Most confidence mistakes)")
print("-"*70)
for i,error in enumerate(sorted(high_conf_errors,
                              key=lambda x: x['confidence'],
                              reverse=True)[:5],1):
    print(f"\n{i}. Review (first 200 chars):")
    print(f"    {error['text'][:200]}...")
    print(f"    Actual: {error['true']}")
    print(f"    Predicted: {error['predicted']} (confidence: {error['confidence']:.2%})")