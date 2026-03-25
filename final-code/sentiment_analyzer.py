from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the model
print("Loading model...")
classifier = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english")

print ("Model loaded successfully!\n")

#Interactive testing
def analyze_custom_text():
    print("\n" + "="*50)
    print("Custom Sentiment Analysis")
    print("="*50)
    print("Enter movie reviews to analyze (or 'quit' to exit)\n")
    
    while True:
        user_input = input("Enter review: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if user_input:
            result = classifier(user_input)[0]
            print(f"Sentiment: {result['label']}")
            print(f"Confidence: {result['score']:.2%}\n")


    
    # Then allow custom input
    analyze_custom_text()