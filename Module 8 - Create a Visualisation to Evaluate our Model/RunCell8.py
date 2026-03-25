sns.heatmap(cm, 
           annot=True, 
           fmt='d', 
           cmap='Blues', 
           xticklabels=['Negative', 'Positive'], 
           yticklabels=['Negative', 'Positive'],
           cbar_kws={'label': 'Number of Reviews'},
           annot_kws={'size': 16,'weight': 'bold'})

plt.xlabel('Predicted Sentiment', fontsize=12, weight='bold')
plt.ylabel('Actual Sentiment', fontsize=12, weight='bold')
plt.title('Confusion Matrix - Sentiment Analysis Results\n' +
          f'Accuracy: {accuracy:.2%}', 
          fontsize=14, weight='bold',pad=20)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300,bbox_inches='tight')
plt.show()

print("✅ Confusion matrix saved as 'confusion_matrix.png'!")