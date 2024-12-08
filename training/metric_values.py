import torch
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import label_binarize

# Load saved model
model_path = './Sequence_classification_saved_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load processed test data
with open('processed_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract the test split (X_test and y_test) from the loaded data
if isinstance(data, tuple) and len(data) == 5:
    _, X_test, _, y_test, labels = data
else:
    print("Unexpected data structure in 'processed_data.pkl'.")
    exit()

# Tokenize the test data
inputs = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')

# Make predictions
with torch.no_grad():
    model.eval()
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).numpy()

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Print metrics only
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to avoid it displaying

# Multi-class ROC AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
roc_auc = roc_auc_score(y_test_bin, outputs.logits.detach().numpy(), average='macro', multi_class='ovr')
print(f'Multi-class ROC AUC: {roc_auc:.4f}')

# Print message that the graph has been saved
print("Graphs have been saved as 'confusion_matrix.png'.")
