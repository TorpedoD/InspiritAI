import pickle
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your saved model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Change to your model name
model_dir = "llama_model"  # Directory where model is saved
saved_model_dir = './save_model'  # Directory where the fine-tuned model is saved
data_path = 'processed_data.pkl'  # Path to the pickle file with your data

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = AutoModelForCausalLM.from_pretrained(saved_model_dir).to(device)

# Add a padding token if not already defined
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Ensure the model recognizes the new token
model.resize_token_embeddings(len(tokenizer))

# Load the label encoder
with open(f"{saved_model_dir}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the data
with open(data_path, 'rb') as f:
    X_train, X_test, y_train, y_test, labels = pickle.load(f)

# Ensure the number of labels is correct
num_labels = len(set(labels))
y_test_encoded = label_encoder.transform(y_test)

print(f"Total number of test samples: {len(X_test)}")
print(f"Total number of unique labels: {num_labels}")

# Prepare the dataset for tokenization
test_dataset = Dataset.from_dict({'text': X_test, 'labels': y_test_encoded})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Tokenizing the dataset
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Get predictions
model.eval()
all_predictions = []
all_labels = []

for i in range(len(tokenized_test_dataset)):
    input_ids = tokenized_test_dataset[i]['input_ids'].unsqueeze(0).to(device)
    attention_mask = tokenized_test_dataset[i]['attention_mask'].unsqueeze(0).to(device)
    label = tokenized_test_dataset[i]['labels']

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    predictions = F.softmax(logits, dim=-1).cpu().numpy()
    all_predictions.append(predictions)
    all_labels.append(label)

predicted_probs = np.vstack(all_predictions)
predicted_labels = np.argmax(predicted_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Compute ROC curve for each class (multi-class ROC curve)
true_labels_one_hot = np.eye(num_labels)[y_test_encoded]

plt.figure(figsize=(8, 6))
for i in range(num_labels):
    fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predicted_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2)
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
