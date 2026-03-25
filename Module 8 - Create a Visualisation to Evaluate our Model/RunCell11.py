print("="*70)
print("EXAMPLE MISCLASSIFICATIONS")
print("="*70)

#Find misclassified examples
errors = []
for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
    if true != pred:
        errors.append({
            'index': i,
            'text':texts[i],
            'true':'Positive' if true == 1 else 'Negative',
            'predicted':'Positive' if pred == 1 else 'Negative',
            'confidence': predictions[i]['score']            
        })

print(f"\nFound {len(errors)} misclassified reviews out of {len(true_labels)}")
print(f"Error rate: {len(errors)/len(true_labels)*100:.2f}%")

#Show high-confidence errors (most interesting)
high_conf_errors = [e for e in errors if e['confidence'] > 0.9]
print(f"\nHigh-confidence errors (>90% confidence): {len(high_conf_errors)}")

print("\n" + "-"*70)
print("TOP 5 MISCLASSIFIED (Most confidence mistakes)")
print("-"*70)

for i,error in enumerate(sorted(high_conf_errors,
                              key=lambda x: x['confidence'], 
                              reverse=True)[:5],1):
    print(f"\n[i]. Review (first 200 chars):")
    print(f"    {error['text'][:200]}...")
    print(f"    ✓ Actual: {error['true']}")
    print(f"    ✘ Predicted: {error['predicted']} (confidence: {error['confidence']:.2%})")    
    