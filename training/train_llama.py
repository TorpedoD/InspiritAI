import pickle
from difflib import get_close_matches
from sklearn.metrics import accuracy_score, classification_report
from load_model import load_model  # Import the function from the model loading script

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, labels = pickle.load(f)

print(f"Total number of samples in the dataset: {len(X_train) + len(X_test)}")
unique_labels = set(labels)
print(f"Total number of unique labels: {len(unique_labels)}")

# Load the model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_dir = "llama_model"  # Directory to save/load the model

# Load model using the function from load_model.py
model, tokenizer, device = load_model(model_name=model_name, model_dir=model_dir)

# Set the pad_token if it's not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token if not set

# Set padding_side to 'left' for decoder-only architecture
tokenizer.padding_side = 'left'

categories = list(set(labels))
categories_str = ', '.join(categories)

# Create prompt for model input
def create_prompt(text):
    max_input_length = tokenizer.model_max_length
    truncated_text = text[:max_input_length]
    return (
        f"Given the following text:\n\n"
        f"\"{truncated_text}\"\n\n"
        f"Predict the category it belongs to from the following options: {categories_str}.\n"
        f"Answer with only the category name."
    )

# Use smaller batch size to avoid overload on CPU
batch_size = 2  # Reduced batch size for CPU
predictions = []
print("Predicting categories for the test set...")

for idx in range(0, len(X_test), batch_size):
    batch_texts = X_test[idx:idx + batch_size]
    batch_prompts = [create_prompt(text) for text in batch_texts]
    
    # Ensure padding and truncation work without errors
    inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(device)

    # Generate outputs using the model with limited max_new_tokens and adjusted batch size
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,  # Limit token generation to 20 to speed up processing
            do_sample=False,    # This ensures deterministic generation
            eos_token_id=tokenizer.eos_token_id
        )
    except KeyboardInterrupt:
        print("Inference interrupted due to timeout or processing delay.")
        break

    # Decode and process predictions
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        predicted_category = generated_text[len(batch_prompts[i]):].strip().split("\n")[0]
        predicted_category = predicted_category.lower().strip('".\'')
        
        # Find best match with available categories
        match = get_close_matches(predicted_category, [cat.lower() for cat in categories], n=1, cutoff=0.0)
        if match:
            predictions.append(categories[[cat.lower() for cat in categories].index(match[0])])
        else:
            predictions.append("Unknown")
    
    print(f"Batch {idx // batch_size + 1}/{len(X_test) // batch_size} processed.")

# Filter out 'Unknown' predictions
valid_indices = [i for i, p in enumerate(predictions) if p != "Unknown"]
filtered_predictions = [predictions[i] for i in valid_indices]
filtered_y_test = [y_test[i] for i in valid_indices]

# Calculate accuracy if valid predictions exist
if filtered_predictions:
    accuracy = accuracy_score(filtered_y_test, filtered_predictions)
    print(f"\nAccuracy on the test set (excluding 'Unknown'): {accuracy:.2f}")
else:
    print("No valid predictions made.")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions, labels=categories))
