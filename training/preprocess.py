import os
import zipfile
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk

# Download required NLTK resources (ensure correct installation)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Define a function to read .txt files from subfolders
def read_txt_files_from_folder(folder_path):
    data = []  # Stores content of .txt files
    labels = []  # Stores folder names as labels
    file_count = 0  # Count of processed files

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return None, None

    # Traverse the folder structure
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(file_path)).strip().lower()

                # Skip unexpected or invalid labels
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

    # Log a summary of the file processing
    print(f"Processed {file_count} files with labels: {set(labels)}")
    print(f"Label distribution: {pd.Series(labels).value_counts()}")

    return data, labels

# Step 2: Preprocess the text data (tokenization, stopword removal, lemmatization)
def preprocess_text(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_data = []

    for text in data:
        try:
            # Tokenize text
            tokens = word_tokenize(text, language='english')
            # Remove stopwords and lemmatize
            filtered_tokens = [
                lemmatizer.lemmatize(word.lower())
                for word in tokens
                if word.isalpha() and word.lower() not in stop_words
            ]
            # Join tokens back into a single string
            processed_data.append(" ".join(filtered_tokens))
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            processed_data.append("")  # Append empty string if an error occurs

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
    zip_file_path = "data/txt.zip"  # Path to your zip file
    extract_to = "data/txt"  # Path to extract files
    pull_from = "data/txt/txt"  # Adjust this path if needed

    # Step 1: Unzip the file
    unzip_file(zip_file_path, extract_to)

    # Step 2: Read .txt files and handle subfolders
    data, labels = read_txt_files_from_folder(pull_from)
    if data is None or labels is None:
        exit(1)

    # Step 3: Log unique labels before splitting
    print(f"Unique labels in the dataset: {set(labels)}")
    print(f"Label distribution: {pd.Series(labels).value_counts()}")

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Training labels: {set(y_train)}")
    print(f"Testing labels: {set(y_test)}")

    # Step 5: Preprocess the training and testing data separately
    print("Preprocessing training data...")
    X_train_processed = preprocess_text(X_train)
    print("Preprocessing testing data...")
    X_test_processed = preprocess_text(X_test)

    # Step 6: Combine all labels and fit LabelEncoder
    all_labels = y_train + y_test
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Encode labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Log the encoder classes
    print(f"LabelEncoder classes: {label_encoder.classes_}")

    # Step 7: Save the processed data and splits
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump((X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, label_encoder.classes_), f)

    print("Preprocessing and data saving complete.")
