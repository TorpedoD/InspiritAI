import pickle
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to saved model, tokenizer, and data
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_dir = "llama_model"
saved_model_dir = "./save_model"
data_path = "processed_data.pkl"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = AutoModelForCausalLM.from_pretrained(saved_model_dir).to(device)

# Add a padding token if not defined
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

# Load data
with open(data_path, "rb") as f:
    X_train, X_test, y_train, y_test, labels = pickle.load(f)

# Prepare label encoder
with open(f"{saved_model_dir}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

y_test_encoded = label_encoder.transform(y_test)
num_labels = len(set(labels))

print(f"Total test samples: {len(X_test)}")
print(f"Number of unique labels: {num_labels}")

# Tokenize test data
test_dataset = Dataset.from_dict({"text": X_test, "labels": y_test_encoded})
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# DataLoader for test data
batch_size = 8
dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size)

# Function to compute predictions
def compute_predictions(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = F.softmax(logits, dim=-1).cpu().numpy()

            all_predictions.append(predictions)
            all_labels.extend(labels.cpu().numpy())

    return np.vstack(all_predictions), np.array(all_labels)

# Compute predictions
predicted_probs, true_labels = compute_predictions(model, dataloader)
predicted_labels = np.argmax(predicted_probs, axis=1)

# Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC Curve for each class
true_labels_one_hot = np.eye(num_labels)[true_labels]

plt.figure(figsize=(10, 8))
for i in range(num_labels):
    fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predicted_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", lw=2)
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
