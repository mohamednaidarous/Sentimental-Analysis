print("="*70)
print("PREPARING DATASET FOR TRAINING")
print("="*70)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split into train (80%) and test (20%)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"\n✅ Dataset split:")
print(f"   Training examples: {len(dataset['train'])}")
print(f"   Testing examples: {len(dataset['test'])}")

# Verify the splits
train_labels = pd.Series(dataset['train']['label'])
test_labels  = pd.Series(dataset['test']['label'])

print(f"\n📊 Training set balance:")
print(f"   Negative: {(train_labels == 0).sum()}")
print(f"   Positive: {(train_labels == 1).sum()}")

print(f"\n📊 Test set balance:")
print(f"   Negative: {(test_labels == 0).sum()}")
print(f"   Positive: {(test_labels == 1).sum()}")

print("\n" + "="*70)