print("="*70)
print("STARTING FINE-TUNING")
print("="*70)

# Define evaluation metric
def compute_metrics(eval_pred):
    """
    Calculates accuracy during training.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("\n🔥 Beginning fine-tuning...")
print("   This will take 2-5 minutes depending on GPU allocation.")
print("   You'll see progress updates below:\n")

# Start training!
trainer.train()

print("\n" + "="*70)
print("✅ FINE-TUNING COMPLETE!")
print("="*70)