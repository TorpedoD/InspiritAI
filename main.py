import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt_tab')

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
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    data.append(content)
                    labels.append(folder_name)  # Use folder name as label
                    file_names.append(file)  # Save file name
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

# Step 3: Save data into a DataFrame
def save_to_dataframe(data, labels, file_names, output_file='prepared_data.csv'):
    df = pd.DataFrame({'Content': data, 'Label': labels, 'File_Name': file_names})
    df.to_csv(output_file, index=False)
    print(f"Data saved as '{output_file}'")
    return df

# Main function to execute the workflow
if __name__ == "__main__":
    # Hardcode the folder path here
    folder_path = "/Users/damianwong/Downloads/txt_test"  # Replace this with your folder path

    # Read .txt files and handle subfolders
    data, labels, file_names = read_txt_files_from_folder(folder_path)
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

    # Step 5: Split the dataset into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(processed_data, labels, test_size=0.2, random_state=42)
    print(f"Training Dataset # of Samples: {len(X_train)}")
    print(f"Testing Dataset # of Samples: {len(X_test)}")

    # Step 6: Vectorize the text data
    vectorizer = TfidfVectorizer()  # You can use CountVectorizer() or TfidfVectorizer() here
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Step 7: Output the names of the X matrix and y vector
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
