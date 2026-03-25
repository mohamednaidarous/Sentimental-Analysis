# Determine accuracy and evaluation results
accuracy = accuracy_score(true_labels, pred_labels)

print("="*70)
print("EVALUATION RESULTS")
print("="*70)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correct Predictions: {sum(np.array(true_labels) == np.array(pred_labels))} / {len(true_labels)}")

print("\nDetailed Classification Report:")
print(classification_report(true_labels, pred_labels,
                            target_names=['Negative', 'Positive']))
