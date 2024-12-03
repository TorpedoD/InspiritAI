import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
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
# Main function for preprocessing
if __name__ == "__main__":
    folder_path = "data/txt/txt"  # Folder containing .txt files (change to your path)
    
    # Read .txt files and handle subfolders
    data, labels, file_names = read_txt_files_from_folder(folder_path)
    if data is None or labels is None or file_names is None:
        exit(1)
    # Preprocess the text
    processed_data = preprocess_text(data)
    # Save processed data, labels, and file names
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump((processed_data, labels, file_names), f)
    
    print("Preprocessing complete and data saved as 'processed_data.pkl'.")
