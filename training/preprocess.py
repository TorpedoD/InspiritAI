import os
import zipfile
import pandas as pd
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Data Augmentation (Synonym Replacement)
def synonym_replacement(text, word_map):
    tokens = word_tokenize(text)
    augmented_text = [word_map.get(word, word) for word in tokens]
    return " ".join(augmented_text)

def generate_word_map():
    # Replace these with actual synonyms or an external library
    return {"good": "excellent", "bad": "poor", "happy": "joyful"}

# Step 1: Define a function to read .txt files from subfolders
def read_txt_files_from_folder(folder_path):
    data = []  # Stores content of .txt files
    labels = []  # Stores folder names as labels
    file_count = 0  # Count of processed files

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return None, None

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(file_path)).strip().lower()

                if folder_name not in {'gpt_txt', 'theo_txt', 'damian_txt', 'misia_txt'}:
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        data.append(content)
                        labels.append(folder_name)
                        file_count += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    print(f"Processed {file_count} files with labels: {set(labels)}")
    return data, labels

# Step 2: Preprocess the text data (tokenization, stopword removal, lemmatization)
def preprocess_text(data, augment=False):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    word_map = generate_word_map() if augment else {}

    processed_data = []
    for text in data:
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [
                lemmatizer.lemmatize(word.lower())
                for word in tokens
                if word.isalpha() and word.lower() not in stop_words
            ]
            text = " ".join(filtered_tokens)
            if augment:
                text = synonym_replacement(text, word_map)
            processed_data.append(text)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            processed_data.append("")
    return processed_data

# Step 3: Unzip the dataset
def unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"Error: The zip file '{zip_path}' does not exist.")
        return
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files unzipped to: {extract_to}")


# Main preprocessing script
if __name__ == "__main__":
    zip_file_path = "data/txt.zip"
    extract_to = "data/txt"
    pull_from = "data/txt/txt"

    unzip_file(zip_file_path, extract_to)
    data, labels = read_txt_files_from_folder(pull_from)
    if data is None or labels is None:
        exit(1)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    print("Preprocessing data...")
    X_train_processed = preprocess_text(X_train, augment=True)
    X_test_processed = preprocess_text(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train + y_test)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    with open('processed_data.pkl', 'wb') as f:
        pickle.dump((X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, label_encoder.classes_), f)
    print("Preprocessing complete.")
