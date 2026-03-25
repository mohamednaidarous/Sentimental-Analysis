print("="*70)
print("UPLOADING AND LOADING CUSTOM DATASET")
print("="*70)

from google.colab import files

print("\n📁 Please upload your slang_reviews.csv file...")
uploaded = files.upload()

# Load the CSV
df = pd.read_csv("slang_reviews.csv")

print("\n✅ Dataset loaded successfully!")
print(f"     Total examples: {len(df)}")
print(f"     Columns: {df.columns.tolist()}")

# Check balance
label_counts = df['label'].value_counts()
print("\n📊 Dataset balance:")
print(f"    Negative: {label_counts[0]} ({label_counts[0]/len(df)*100:.2f}%)")
print(f"    Positive: {label_counts[1]} ({label_counts[1]/len(df)*100:.2f}%)")

# Show first few examples
print("\n 📝 First 5 examples:")
print(df.head())

print("\n" + "="*70)

