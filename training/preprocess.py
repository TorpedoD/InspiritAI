import os
import zipfile
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download required NLTK resources (only needed once)
nltk.download('punkt')  # Ensure punkt tokenizer is available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Add this line to download the missing resource
nltk.download('omw-1.4')

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

# Main function to execute preprocessing
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
    
    print(f"Total number of samples in the dataset: {len(data)}")
    print(f"Total number of unique labels: {len(set(labels))}")
    
    # Preprocess the data
    processed_data = preprocess_text(data)

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
    print(f"Training Dataset Samples: {len(X_train)}")
    print(f"Testing Dataset Samples: {len(X_test)}")

    # Save the preprocessed data and splits for the training script
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test, labels), f)

    print("Preprocessing and data saving complete.")
