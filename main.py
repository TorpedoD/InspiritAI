import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import zipfile

# Download required NLTK resources (only needed once)
nltk.download('all') 

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
                    # Open the file with a fallback encoding
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
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords and punctuation
        filtered_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word not in stop_words]
        processed_data.append(" ".join(filtered_tokens))
    return processed_data


# Step 3: Generate word clouds for each category
def generate_word_clouds(data, labels):
    category_texts = {}
    for text, label in zip(data, labels):
        if label not in category_texts:
            category_texts[label] = []
        category_texts[label].append(text)
    
    # Generate a word cloud for each category
    for category, texts in category_texts.items():
        combined_text = " ".join(texts)  # Combine all text for the category
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Word Cloud for Category: {category}")
        plt.axis("off")
        plt.show()


# Step 4: Save data into a DataFrame
def save_to_dataframe(data, labels, file_names, output_file='prepared_data.csv'):
    df = pd.DataFrame({'Content': data, 'Label': labels, 'File_Name': file_names})
    df.to_csv(output_file, index=False)
    print(f"Data saved as '{output_file}'")
    return df


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
    # Path to the zip file
    zip_file_path = "data/txt_test.zip"  # Replace with the path to your zip file
    extract_to = "data/txt_test"  # Path to extract files

    # Unzip the file
    unzip_file(zip_file_path, extract_to)

    # Read .txt files and handle subfolders
    data, labels, file_names = read_txt_files_from_folder(extract_to)
    if data is None or labels is None or file_names is None:
        exit(1)  # Exit if folder path is invalid

    # Count the total number of samples
    total_samples = len(data)
    print(f"Total number of samples in the dataset (including all subfolders): {total_samples}")

    # Count the number of unique labels (subfolders)
    unique_labels = set(labels)  # Use set to get unique folder names
    num_labels = len(unique_labels)
    print(f"Total number of unique labels (subfolders): {num_labels}")

    # Step 4: Preprocess the text data
    processed_data = preprocess_text(data)

    # Step 5: Generate word clouds for visualization
    generate_word_clouds(processed_data, labels)

    # Step 6: Split the dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
    print(f"Training Dataset # of Samples: {len(X_train)}")
    print(f"Testing Dataset # of Samples: {len(X_test)}")

    # Step 7: Vectorize the text data
    vectorizer = TfidfVectorizer()  # You can use CountVectorizer() or TfidfVectorizer() here
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Step 8: Output the names of the X matrix and y vector
    print("Name of the X matrix: X_train_vec (Training Data Vectorized), X_test_vec (Testing Data Vectorized)")
    print("Name of the y vector: y_train (Training Labels), y_test (Testing Labels)")

    # Output dimensions and shapes
    print(f"Training data vectorized shape: {X_train_vec.shape}")
    print(f"Testing data vectorized shape: {X_test_vec.shape}")

    # Save data to a DataFrame and CSV
    df = save_to_dataframe(processed_data, labels, file_names)

    # Optional: Preview the first few rows
    print("First few samples from the dataset:")
    print(df.head())
