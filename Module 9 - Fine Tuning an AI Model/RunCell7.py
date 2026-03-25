print("="*70)
print("CONFIGURING TRAINING PARAMETERS")
print("="*70)

training_args = TrainingArguments(
    output_dir="./results",                    # Where to save checkpoints
    num_train_epochs=3,                        # Train for 3 complete passes
    per_device_train_batch_size=8,             # Process 8 examples at once
    per_device_eval_batch_size=8,              # Evaluate 8 examples at once
    learning_rate=2e-5,                        # How aggressively to update weights
    weight_decay=0.01,                         # Regularization to prevent overfitting
    eval_strategy="epoch",                     # Evaluate after each epoch
    save_strategy="epoch",                     # Save model after each epoch
    load_best_model_at_end=True,               # Load best performing model at end
    logging_dir='./logs',                      # Where to save training logs
    logging_steps=10,                          # Log every 10 steps
    seed=42,                                   # For reproducibility
)

print("\n✅ Training configuration:")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Evaluation: After each epoch")

print("\n📊 Training will involve:")
train_steps = len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs
print(f"   Approximately {train_steps} training steps")
print(f"   {len(tokenized_train) // training_args.per_device_train_batch_size} steps per epoch")

print("\n" + "="*70)