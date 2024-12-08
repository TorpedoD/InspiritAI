import torch
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import label_binarize
from tqdm import tqdm  # For progress bar

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load saved model and move it to the appropriate device
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
    print("Unexpected data structure in 'processed_data.pkl'.")
    exit()

# Batch processing parameters
batch_size = 16  # Reduce batch size to fit in GPU memory
num_batches = len(X_test) // batch_size + (1 if len(X_test) % batch_size != 0 else 0)

# Initialize storage for predictions
all_predictions = []
all_logits = []

# Make predictions batch by batch with progress bar
model.eval()
with torch.no_grad():
    with tqdm(total=num_batches, desc="Processing", unit="batch") as pbar:
        for i in range(0, len(X_test), batch_size):
            # Prepare batch
            batch_inputs = tokenizer(
                X_test[i:i + batch_size],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                outputs = model(**batch_inputs)

            # Collect predictions and logits
            all_predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
            all_logits.extend(outputs.logits.cpu().numpy())

            # Update progress bar
            pbar.update(1)

# Convert predictions and logits to numpy arrays
predictions = np.array(all_predictions)
logits = np.array(all_logits)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Print metrics
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
roc_auc = roc_auc_score(y_test_bin, logits, average='macro', multi_class='ovr')
print(f'Multi-class ROC AUC: {roc_auc:.4f}')

# Print confirmation of graph save
print("Graphs have been saved as 'confusion_matrix.png'.")
