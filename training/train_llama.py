import os
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from difflib import get_close_matches
from sklearn.metrics import accuracy_score, classification_report

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    processed_data, labels, file_names = pickle.load(f)

print(f"Total number of samples in the dataset: {len(processed_data)}")
unique_labels = set(labels)
print(f"Total number of unique labels: {len(unique_labels)}")

# Split data into training and testing (80% training, 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
print(f"Training Dataset Samples: {len(X_train)}")
print(f"Testing Dataset Samples: {len(X_test)}")

# Load the model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(load_in_8bit=torch.cuda.is_available())
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if not os.path.exists("llama_model"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.save_pretrained("llama_model")
else:
    model = AutoModelForCausalLM.from_pretrained("llama_model")

model.to(device)
print("Model loaded.")

categories = list(set(labels))
categories_str = ', '.join(categories)

def create_prompt(text):
    max_input_length = tokenizer.model_max_length
    truncated_text = text[:max_input_length]
    return (
        f"Given the following text:\n\n"
        f"\"{truncated_text}\"\n\n"
        f"Predict the category it belongs to from the following options: {categories_str}.\n"
        f"Answer with only the category name."
    )

# Use batching for prediction
batch_size = 8  # Adjust batch size based on available GPU memory
predictions = []
print("Predicting categories for the test set...")

for idx in range(0, len(X_test), batch_size):
    batch_texts = X_test[idx:idx + batch_size]
    batch_prompts = [create_prompt(text) for text in batch_texts]
    inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        predicted_category = generated_text[len(batch_prompts[i]):].strip().split("\n")[0]
        predicted_category = predicted_category.lower().strip('".\'')
        match = get_close_matches(predicted_category, [cat.lower() for cat in categories], n=1, cutoff=0.0)
        if match:
            predictions.append(categories[[cat.lower() for cat in categories].index(match[0])])
        else:
            predictions.append("Unknown")
    print(f"Batch {idx // batch_size + 1}/{len(X_test) // batch_size} processed.")

valid_indices = [i for i, p in enumerate(predictions) if p != "Unknown"]
filtered_predictions = [predictions[i] for i in valid_indices]
filtered_y_test = [y_test[i] for i in valid_indices]

if filtered_predictions:
    accuracy = accuracy_score(filtered_y_test, filtered_predictions)
    print(f"\nAccuracy on the test set (excluding 'Unknown'): {accuracy:.2f}")
else:
    print("No valid predictions made.")

print("\nClassification Report:")
print(classification_report(y_test, predictions, labels=categories))