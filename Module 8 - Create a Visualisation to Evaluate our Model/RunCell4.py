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

# Function to predict on batch
def predict_sentiment(texts, batch_size=32):
    """
    Predict sentiment for a list of texts in batches.
    """

    predictions = []
    totalbatches = (len(texts) + batch_size - 1) // batch_size

    print(f"Processing {len(texts)} reviews in {totalbatches}) batches...")

    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True, max_length=512)
        predictions.extend(results)

        #Progress indicator
        if batch_num % 10 == 0 or batch_num == totalbatches:
            print(f"Processed batch {batch_num}/{totalbatches} ")
        
    return predictions

# Get predictions
print("Making predictions on 1000 reviews...   ")
texts = dataset['text']
predictions = predict_sentiment(texts)

#Convert predictions to binary lables
pred_labels = [1 if p['label'] == 'POSITIVE' else 0 for p in predictions]
true_labels = list(dataset['label'])

print(f"\n✅ Converted {len(pred_labels)} predictions to binary labels.")