import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# --- Load and prepare dataset ---
print("="*70)
print("LOADING CUSTOM DATASET")
print("="*70)
df = pd.read_csv("slang_reviews.csv")
print(f"Total examples: {len(df)}")

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Training examples: {len(dataset['train'])}")
print(f"Testing examples: {len(dataset['test'])}")

# --- Load model and tokenizer ---
print("\n" + "="*70)
print("LOADING PRE-TRAINED MODEL")
print("="*70)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("Model and tokenizer loaded")

# --- Tokenize ---
print("\n" + "="*70)
print("TOKENIZING DATASET")
print("="*70)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_train = dataset['train'].map(tokenize_function, batched=True)
tokenized_test = dataset['test'].map(tokenize_function, batched=True)
print("Tokenization complete!")

# --- Training arguments ---
print("\n" + "="*70)
print("CONFIGURING TRAINING PARAMETERS")
print("="*70)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    seed=42,
)
print("Training configuration ready")

# --- Fine-tuning ---
print("\n" + "="*70)
print("STARTING FINE-TUNING")
print("="*70)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("\nBeginning fine-tuning...")
print("   This will take a few minutes depending on your CPU/GPU.")
print("   You'll see progress updates below:\n")

trainer.train()

print("\n" + "="*70)
print("FINE-TUNING COMPLETE!")
print("="*70)

# Save the fine-tuned model so later cells can reuse it without retraining
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("\nFine-tuned model saved to ./fine_tuned_model")