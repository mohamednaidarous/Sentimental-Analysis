from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def analyze_custom_text(classifier):
    """
    Runs an interactive loop that lets the user enter movie reviews
    and see the model's predicted sentiment and confidence score.
    """
    print("\n" + "="*50)
    print("Custom Sentiment Analysis")
    print("="*50)
    print("Enter movie reviews to analyze (or 'quit' to exit)\n")
    while True:
        user_input = input("Enter review: ").strip()
        if user_input.lower() == 'quit':
            print("\nThanks for using the sentiment analyzer!")
            break
        if user_input:
            result = classifier(user_input)[0]
            print(f"Sentiment: {result['label']}")
            print(f"Confidence: {result['score']:.2%}\n")
        else:
            print("Please enter some text.\n")


def main():
    print("Loading model...")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded successfully!\n")

    analyze_custom_text(classifier)


if __name__ == "__main__":
    main()