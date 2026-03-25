from transformers import pipeline

# Load a pre-trained sentiment analysis model
print("Loading...")
classifier = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english")

# Test it with a simple example
text = "This movie was absolutely fantastic! I loved every minute."
result = classifier(text)
print(f"Text: {text}")
print(f"Result: {result}")