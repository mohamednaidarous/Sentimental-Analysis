import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# --- Setup from earlier cells ---
df = pd.read_csv("slang_reviews.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- This cell ---
print("="*70)
print("TOKENIZING DATASET")
print("="*70)

def tokenize_function(examples):
    """
    Converts text to tokens (numbers) that the model understands.
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("\nTokenizing training set...")
tokenized_train = dataset['train'].map(tokenize_function, batched=True)
print("Tokenizing test set...")
tokenized_test = dataset['test'].map(tokenize_function, batched=True)
print("\nTokenization complete!")

print(f"\nExample tokenization:")
print(f"   Original text: {dataset['train'][0]['text']}")
print(f"   Label: {dataset['train'][0]['label']}")
print(f"   Tokenized (first 20 tokens): {tokenized_train[0]['input_ids'][:20]}")
print(f"   Total tokens: {len(tokenized_train[0]['input_ids'])}")
print("\n" + "="*70)