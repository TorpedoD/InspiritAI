import torch
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load saved model and tokenizer
model_path = './Sequence_classification_saved_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load processed test data
with open('processed_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract the test split (X_test and y_test) from the loaded data
if isinstance(data, tuple) and len(data) == 5:
    _, X_test, _, y_test, labels = data
else:
    raise ValueError("Unexpected data structure in 'processed_data.pkl'.")

# Ensure labels are consistent
if isinstance(y_test[0], str):
    y_test = [str(label) for label in y_test]
elif isinstance(y_test[0], int):
    y_test = [int(label) for label in y_test]
y_test = np.array(y_test)

# Increase batch size for speed (adjust based on GPU capacity)
batch_size = 64

# Make predictions
model.eval()
all_predictions = []
all_logits = []

with torch.no_grad():
    for i in tqdm(range(0, len(X_test), batch_size), desc="Processing", unit="batch"):
        batch_inputs = tokenizer(
            X_test[i:i + batch_size],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        # Perform inference
        with torch.amp.autocast('cuda'):
            outputs = model(**batch_inputs)

        # Collect predictions and logits
        logits = outputs.logits.cpu().numpy()
        predictions = np.argmax(logits, axis=-1)

        all_predictions.extend(predictions)
        all_logits.extend(logits)

# Convert to numpy arrays
predictions = np.array(all_predictions)
logits = np.array(all_logits)

# Debugging: Print sample logits and predictions
print(f"Sample logits: {logits[:5]}")
print(f"Sample predictions: {predictions[:5]}")
print(f"Unique predictions: {np.unique(predictions)}")
print(f"Unique ground truth labels: {np.unique(y_test)}")

# Ensure label alignment
if set(np.unique(predictions)) - set(np.unique(y_test)):
    print("Warning: Predicted labels do not match ground truth labels.")

# Metrics calculation
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted', zero_division=1)
recall = recall_score(y_test, predictions, average='weighted', zero_division=1)
f1 = f1_score(y_test, predictions, average='weighted')

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

# Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot and save confusion matrices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Multi-class ROC AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
roc_auc = roc_auc_score(y_test_bin, logits, average='macro', multi_class='ovr')
print(f'Multi-class ROC AUC: {roc_auc:.4f}')

# Final confirmation
print("Metrics calculated and graphs saved as 'confusion_matrices.png'.")
