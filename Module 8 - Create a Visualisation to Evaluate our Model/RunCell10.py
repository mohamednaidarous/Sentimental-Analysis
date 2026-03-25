# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()

print("="*70)
print("CONFUSION MATRIX BREAKDOWN")
print("="*70)
print(f"\nTrue Negatives (TN): {tn}")
print(f" ✅ Correctly identified as NEGATIVE: {tn} reviews")

print(f"\nFalse Positives (FP): {fp}")
print(f" ❌ Incorrectly identified as POSITIVE: {fp} reviews")

print(f"\nFalse Negatives (FN): {fn}")
print(f" ❌ Incorrectly identified as NEGATIVE: {fn} reviews")

print(f"\nTrue Positives (TP): {tp}")
print(f" ✅ Correctly identified as POSITIVE: {tp} reviews")

print("\n")
print("KEY INSIGHTS:")
print("=" * 70)

# Calculate error rates
false_positive_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

print(f"\n False Positive Rate: {false_positive_rate:.2f}%")
print(f"  (Model incorreclty calls negative reviews positive)")

print(f"\n False Negative Rate: {false_negative_rate:.2f}%")
print(f"  (Model incorrectly calls positive reviews negative)")

if false_positive_rate > false_negative_rate:
    print("\n⚠️ The model tends to be OPTIMISTIC (over predicted postive).")
elif false_negative_rate > false_positive_rate:
    print("\n⚠️ The model tends to be PESSIMISTIC (over predicted negative).")  
else:
    print("\n✅ The model has a BALANCED in its errors.")