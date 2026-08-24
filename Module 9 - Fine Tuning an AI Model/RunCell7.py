import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments

# --- Setup from earlier cells ---
df = pd.read_csv("slang_reviews.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_train = dataset['train'].map(tokenize_function, batched=True)
tokenized_test = dataset['test'].map(tokenize_function, batched=True)

# --- This cell ---
print("="*70)
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
print("\nTraining configuration:")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Evaluation: After each epoch")
print("\nTraining will involve:")
train_steps = len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs
print(f"   Approximately {train_steps} training steps")
print(f"   {len(tokenized_train) // training_args.per_device_train_batch_size} steps per epoch")
print("\n" + "="*70)