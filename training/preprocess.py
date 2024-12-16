import os
import zipfile
import re
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk
from collections import Counter

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Data Augmentation Functions (Only used for under-represented classes)
def random_insertion(text, n=2):
    tokens = word_tokenize(text)
    for _ in range(n):
        if tokens:
            random_word = random.choice(tokens)
            random_position = random.randint(0, len(tokens) - 1)
            tokens.insert(random_position, random_word)
    return " ".join(tokens)

def random_deletion(text, p=0.2):
    tokens = word_tokenize(text)
    if len(tokens) == 1:
        return text
    return " ".join([word for word in tokens if random.random() > p])

def random_swap(text, n=2):
    tokens = word_tokenize(text)
    for _ in range(n):
        if len(tokens) > 1:
            idx1, idx2 = random.sample(range(len(tokens)), 2)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
    return " ".join(tokens)

# Step 2: Read .txt files
def read_txt_files(folder_path, allowed_labels=None):
    data, labels = [], []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                folder_name = os.path.basename(root).strip().lower()
                if allowed_labels and folder_name not in allowed_labels:
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        data.append(f.read())
                        labels.append(folder_name)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return data, labels

# Step 3: Preprocess the text
def preprocess_text(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_data = []
    for text in data:
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-ASCII characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        tokens = word_tokenize(text)
        filtered_tokens = [
            lemmatizer.lemmatize(word.lower())
            for word in tokens
            if word.isalpha() and word.lower() not in stop_words
        ]
        processed_data.append(" ".join(filtered_tokens))
    return processed_data

# Step 4: Balance Dataset
def balance_dataset(data, labels):
    label_counter = Counter(labels)
    max_count = max(label_counter.values())
    new_data, new_labels = [], []

    for label in label_counter.keys():
        class_data = [d for d, l in zip(data, labels) if l == label]
        if len(class_data) < max_count:
            additional_samples = max_count - len(class_data)
            print(f"Augmenting class '{label}' with {additional_samples} samples...")
            for _ in range(additional_samples):
                new_data.append(random.choice(class_data))
                new_labels.append(label)
        new_data.extend(class_data)
        new_labels.extend([label] * len(class_data))

    return new_data, new_labels

# Step 5: Unzip the dataset
def unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Error: The zip file '{zip_path}' does not exist.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files unzipped to: {extract_to}")

# Main Script
if __name__ == "__main__":
    zip_file_path = "data/txt.zip"
    extract_to = "data/txt"
    folder_path = "data/txt/txt"
    allowed_labels = {'gpt_txt', 'theo_txt', 'damian_txt', 'misia_txt'}

    # Unzip and read data
    unzip_file(zip_file_path, extract_to)
    data, labels = read_txt_files(folder_path, allowed_labels)
    print(f"Original dataset: {len(data)} samples.")

    # Balance dataset (limited to original sample sizes)
    data, labels = balance_dataset(data, labels)
    print(f"Balanced dataset: {len(data)} samples.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Preprocess training and testing data
    print("Preprocessing training and testing data...")
    X_train_processed = preprocess_text(X_train)
    X_test_processed = preprocess_text(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Save processed data
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump((X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, label_encoder.classes_), f)

    print("Preprocessing complete.")
