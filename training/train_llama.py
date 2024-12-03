import os
import pandas as pd
import zipfile
import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from transformers import BitsAndBytesConfig
from difflib import get_close_matches
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Define a function to read .txt files from subfolders
def read_txt_files_from_folder(folder_path):
    data = []  # Stores content of .txt files
    labels = []  # Stores folder names as labels
    file_names = []  # Stores file names

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return None, None, None

    # Traverse the folder structure
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(file_path))  # Get folder name (label)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        data.append(content)
                        labels.append(folder_name)  # Use folder name as label
                        file_names.append(file)  # Save file name
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return data, labels, file_names

# Step 2: Preprocess the text data (tokenization, stopword removal, lemmatization)
def preprocess_text(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_data = []
    for text in data:
        tokens = word_tokenize(text)
        filtered_tokens = [
            lemmatizer.lemmatize(word.lower())
            for word in tokens
            if word.isalpha() and word.lower() not in stop_words
        ]
        processed_data.append(" ".join(filtered_tokens))
    return processed_data

# Unzip the dataset
def unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"Error: The zip file '{zip_path}' does not exist.")
        return
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files unzipped to: {extract_to}")

# Main function to execute the workflow
if __name__ == "__main__":
    zip_file_path = "data/txt.zip"  # Replace with the path to your zip file
    extract_to = "data/txt"  # Path to extract files
    pull_from = "data/txt/txt"  # Adjust this path if needed

    # Unzip the file
    unzip_file(zip_file_path, extract_to)

    # Read .txt files and handle subfolders
    data, labels, file_names = read_txt_files_from_folder(pull_from)
    if data is None or labels is None or file_names is None:
        exit(1)

    total_samples = len(data)
    print(f"Total number of samples in the dataset: {total_samples}")

    unique_labels = set(labels)
    num_labels = len(unique_labels)
    print(f"Total number of unique labels: {num_labels}")

    processed_data = preprocess_text(data)

    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, labels, test_size=0.2, random_state=42
    )
    print(f"Training Dataset Samples: {len(X_train)}")
    print(f"Testing Dataset Samples: {len(X_test)}")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(load_in_8bit=torch.cuda.is_available())
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
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

    predictions = []
    print("Predicting categories for the test set...")
    for idx, text in enumerate(X_test):
        prompt = create_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
        print(f"Sample {idx+1}/{len(X_test)} - Predicted: {predictions[-1]}")

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
