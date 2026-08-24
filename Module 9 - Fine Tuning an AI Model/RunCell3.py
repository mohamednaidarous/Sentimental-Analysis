import pandas as pd

print("="*70)
print("LOADING CUSTOM DATASET")
print("="*70)

df = pd.read_csv("slang_reviews.csv")
print("\nDataset loaded successfully!")
print(f"     Total examples: {len(df)}")
print(f"     Columns: {df.columns.tolist()}")

label_counts = df['label'].value_counts()
print("\nDataset balance:")
print(f"    Negative: {label_counts[0]} ({label_counts[0]/len(df)*100:.2f}%)")
print(f"    Positive: {label_counts[1]} ({label_counts[1]/len(df)*100:.2f}%)")

print("\nFirst 5 examples:")
print(df.head())
print("\n" + "="*70)