import os
import pickle
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer

# Main function to execute the workflow
if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data_path = "processed_data.pkl"  # Path to preprocessed data
    if not os.path.exists(preprocessed_data_path):
        print(f"Error: The file '{preprocessed_data_path}' does not exist.")
        exit(1)

    with open(preprocessed_data_path, "rb") as file:
        data = pickle.load(file)

    processed_data = data['processed_data']
    labels = data['labels']

    total_samples = len(processed_data)
    print(f"Total number of samples in the dataset: {total_samples}")
    unique_labels = set(labels)
    num_labels = len(unique_labels)
    print(f"Total number of unique labels: {num_labels}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, labels, test_size=0.2, random_state=42
    )
    print(f"Training Dataset Samples: {len(X_train)}")
    print(f"Testing Dataset Samples: {len(X_test)}")

    # Assuming the model is already loaded in Colab
    print("Using the preloaded model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Adjust as per the model's actual name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    categories = list(unique_labels)
    categories_str = ', '.join(categories)

    # Define the prompt creation function
    def create_prompt(text):
        max_input_length = tokenizer.model_max_length
        truncated_text = text[:max_input_length]
        return (
            f"Given the following text:\n\n"
            f"\"{truncated_text}\"\n\n"
            f"Predict the category it belongs to from the following options: {categories_str}.\n"
            f"Answer with only the category name."
        )

    predictions = []
    print("Predicting categories for the test set...")
    for idx, text in enumerate(X_test):
        prompt = create_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # Adjust device if using GPU
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_category = generated_text[len(prompt):].strip().split("\n")[0]
        predicted_category = predicted_category.lower().strip('".\'')
        match = get_close_matches(predicted_category, [cat.lower() for cat in categories], n=1, cutoff=0.0)
        if match:
            predictions.append(categories[[cat.lower() for cat in categories].index(match[0])])
        else:
            predictions.append("Unknown")
        print(f"Sample {idx + 1}/{len(X_test)} - Predicted: {predictions[-1]}")

    valid_indices = [i for i, p in enumerate(predictions) if p != "Unknown"]
    filtered_predictions = [predictions[i] for i in valid_indices]
    filtered_y_test = [y_test[i] for i in valid_indices]

    # Evaluate and report results
    if filtered_predictions:
        accuracy = accuracy_score(filtered_y_test, filtered_predictions)
        print(f"\nAccuracy on the test set (excluding 'Unknown'): {accuracy:.2f}")
    else:
        print("No valid predictions made.")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=categories))
