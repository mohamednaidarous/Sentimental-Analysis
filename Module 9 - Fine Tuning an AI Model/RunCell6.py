print("="*70)
print("TOKENIZING DATASET")
print("="*70)


def tokenize_function(examples):
    """
    Converts text to tokens (numbers) that the model understands.
    """
    return tokenizer(
        examples["text"],
        padding="max_length",  # Pad shorter texts to max length
        truncation=True,       # Cut longer texts at max length
        max_length=512         # Maximum token length
    )

print("\n🔄 Tokenizing training set...")
tokenized_train = dataset['train'].map(tokenize_function, batched=True)

print("🔄 Tokenizing test set...")
tokenized_test = dataset['test'].map(tokenize_function, batched=True)

print("\n✅ Tokenization complete!")

# Show example
print(f"\n🔍 Example tokenization:")
print(f"   Original text: {dataset['train'][0]['text']}")
print(f"   Label: {dataset['train'][0]['label']}")
print(f"   Tokenized (first 20 tokens): {tokenized_train[0]['input_ids'][:20]}")
print(f"   Total tokens: {len(tokenized_train[0]['input_ids'])}")

print("\n" + "="*70)