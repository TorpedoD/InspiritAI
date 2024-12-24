import os
import zipfile
import random
import numpy as np
from collections import Counter
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# NLTK downloads
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

# Balance Dataset
def balance_dataset(data, labels, target_count=None):
    label_counter = Counter(labels)
    max_count = target_count or max(label_counter.values())
    balanced_data, balanced_labels = [], []

    for label in label_counter.keys():
        label_data = [d for d, l in zip(data, labels) if l == label]
        if len(label_data) < max_count:
            oversampled = resample(
                label_data,
                replace=True,
                n_samples=max_count - len(label_data),
                random_state=42,
            )
            balanced_data.extend(oversampled)
            balanced_labels.extend([label] * len(oversampled))
        balanced_data.extend(label_data)
        balanced_labels.extend([label] * len(label_data))

    print(f"Balanced dataset: {Counter(balanced_labels)}")
    return balanced_data, balanced_labels

# Preprocessing
if __name__ == "__main__":
    zip_file_path = "data/txt.zip"
    extract_to = "data/txt"
    folder_path = "data/txt/txt"
    allowed_labels = {'GPT_txt', 'Theo_txt', 'damian_txt', 'misia_txt'}

    # Unzip data
    print("Unzipping data...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Files unzipped to: {extract_to}")

    # Read and clean data
    data, labels = [], []
    for root, _, files in os.walk(folder_path):
        label = os.path.basename(root)
        if label in allowed_labels:
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        data.append(content)
                        labels.append(label)
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                content = f.read()
                            data.append(content)
                            labels.append(label)
                        except Exception as e:
                            print(f"Skipping file {file_path}: {e}")

    # Filter empty files and truncate long texts
    MAX_TEXT_LENGTH = 1000
    cleaned_data, cleaned_labels = [], []
    for text, label in zip(data, labels):
        if len(text.split()) > 0:
            truncated_text = ' '.join(text.split()[:MAX_TEXT_LENGTH])
            cleaned_data.append(truncated_text)
            cleaned_labels.append(label)

    print(f"Initial dataset size: {len(cleaned_data)}")
    print(f"Initial label distribution: {Counter(cleaned_labels)}")

    # Balance dataset
    balanced_data, balanced_labels = balance_dataset(cleaned_data, cleaned_labels)

    # Train-test split
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(balanced_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        balanced_data, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Training label distribution: {Counter(y_train)}")
    print(f"Test label distribution: {Counter(y_test)}")

    # Save processed data
    with open("processed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test, label_encoder.classes_), f)

    print("Preprocessing complete. Data saved to 'processed_data.pkl'.")
