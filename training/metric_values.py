import torch
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from collections import Counter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load saved model and tokenizer
model_path = './Sequence_classification_saved_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load processed data (both training and test data)
with open('processed_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract training and test splits (X_train, y_train, X_test, y_test)
if isinstance(data, tuple) and len(data) == 5:
    X_train, X_test, y_train, y_test, labels = data
else:
    raise ValueError("Unexpected data structure in 'processed_data.pkl'.")

# Ensure labels are consistent (convert to string or integer if necessary)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Convert labels to integers if they are strings
if isinstance(y_train[0], str):
    unique_labels = np.unique(y_train)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_mapping[label] for label in y_train])
    y_test = np.array([label_mapping[label] for label in y_test])

# Increase batch size for speed (adjust based on GPU capacity)
batch_size = 64

# Define training function to fine-tune the model
def train_model(model, X_train, y_train, tokenizer, batch_size=64, epochs=3):
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i in tqdm(range(0, len(X_train), batch_size), desc="Training", unit="batch"):
            batch_inputs = tokenizer(
                X_train[i:i + batch_size],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            labels = torch.tensor(y_train[i:i + batch_size]).to(device)

            # Forward pass
            outputs = model(**batch_inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log loss every 100 batches
            if i % 100 == 0:
                print(f"Batch {i}/{len(X_train)} - Loss: {loss.item():.4f}")
    
    print("Training complete.")

# Fine-tune the model on the training data (ensure this is done for a sufficient number of epochs)
train_model(model, X_train, y_train, tokenizer, batch_size, epochs=3)

# Make predictions using the fine-tuned model
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

# Metrics calculation
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Print label distributions
print("True label distribution:", Counter(y_test))
print("Predicted label distribution:", Counter(predictions))

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
