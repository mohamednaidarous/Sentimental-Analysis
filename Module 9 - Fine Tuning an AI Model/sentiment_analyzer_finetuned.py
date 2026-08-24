from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "./fine_tuned_model"

print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

classifier = pipeline("sentiment-analysis",
                      model=model,
                      tokenizer=tokenizer)
print("Fine-tuned model loaded successfully!\n")

def analyse_custom_text():
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
            sentiment = "POSITIVE" if result['label'] == 'LABEL_1' else "NEGATIVE"
            print(f"-> Sentiment:  {sentiment}")
            print(f"-> Confidence: {result['score']:.2%}\n")
        else:
            print("Please enter some text.\n")

analyse_custom_text()