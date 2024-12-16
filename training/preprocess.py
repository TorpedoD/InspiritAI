import os
import zipfile
import re
import random
import numpy as np
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

# Mixup function
def mixup(data, labels, alpha=0.2):
    new_data, new_labels = [], []
    for i in range(len(data)):
        j = random.randint(0, len(data) - 1)
        lam = np.random.beta(alpha, alpha)
        mix_text = f"{data[i]} {data[j]}"[: int(len(data[i]) * lam)]
        new_data.append(mix_text)
        new_labels.append(labels[i])
    return new_data, new_labels

# Cutmix function
def cutmix(data, labels, alpha=0.2):
    new_data, new_labels = [], []
    for i in range(len(data)):
        j = random.randint(0, len(data) - 1)
        lam = np.random.beta(alpha, alpha)
        cut_position = int(len(data[i]) * lam)
        mix_text = data[i][:cut_position] + data[j][cut_position:]
        new_data.append(mix_text)
        new_labels.append(labels[i])
    return new_data, new_labels

# Balance Dataset with Oversampling and Augmentation
def balance_dataset(data, labels):
    label_counter = Counter(labels)
    max_count = max(label_counter.values())
    new_data, new_labels = [], []

    for label in label_counter.keys():
        class_data = [d for d, l in zip(data, labels) if l == label]
        if len(class_data) < max_count:
            additional_samples = max_count - len(class_data)
            for _ in range(additional_samples):
                augmented = random.choice(class_data)
                new_data.append(augmented)  # Random oversampling
                new_labels.append(label)
        new_data.extend(class_data)
        new_labels.extend([label] * len(class_data))

    # Apply Mixup and Cutmix
    mixed_data, mixed_labels = mixup(new_data, new_labels)
    cut_data, cut_labels = cutmix(new_data, new_labels)

    return mixed_data + cut_data + new_data, mixed_labels + cut_labels + new_labels

# Main Preprocessing Script
if __name__ == "__main__":
    # Paths and labels
    zip_file_path = "data/txt.zip"
    extract_to = "data/txt"
    folder_path = "data/txt/txt"
    allowed_labels = {'gpt_txt', 'theo_txt', 'damian_txt', 'misia_txt'}

    # Unzip data
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Read data
    data, labels = [], []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                folder_name = os.path.basename(root)
                if folder_name in allowed_labels:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data.append(f.read())
                        labels.append(folder_name)

    # Balance dataset
    balanced_data, balanced_labels = balance_dataset(data, labels)

   # Calculate a valid test size
    num_samples = len(balanced_labels)
    num_classes = len(set(balanced_labels))

    # Ensure test_size is a valid float between 0 and 1
    test_size = min(0.2, max(2 * num_classes / num_samples, 0.1))

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_data, balanced_labels, test_size=test_size, stratify=balanced_labels, random_state=42
    )


    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Save processed data
    with open("processed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train_encoded, y_test_encoded, label_encoder.classes_), f)

    print("Preprocessing complete. Data saved to 'processed_data.pkl'.")
