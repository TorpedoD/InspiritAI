import torch
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import label_binarize

# Load saved model
model_path = './saved_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load processed test data
with open('processed_data.pkl', 'rb') as file:
    data = pickle.load(file)

# If data is a tuple, unpack it
texts, labels = data  # Assuming the tuple contains texts and labels in that order

# Tokenize texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Make predictions
with torch.no_grad():
    model.eval()
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).numpy()

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
f1 = f1_score(labels, predictions, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(labels, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Multi-class ROC AUC
labels_bin = label_binarize(labels, classes=np.unique(labels))
roc_auc = roc_auc_score(labels_bin, outputs.logits.detach().numpy(), average='macro', multi_class='ovr')
print(f'Multi-class ROC AUC: {roc_auc:.4f}')
