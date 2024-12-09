import pickle
import train_llama

# Define file paths
model_dir = './saved_model'  # Path to the saved model directory
data_path = 'processed_data.pkl'  # Path to your processed data file

# Load the dataset and labels
with open(data_path, 'rb') as f:
    _, _, _, _, labels = pickle.load(f)

# Instantiate the TextClassifier
num_labels = len(set(labels))
classifier = TextClassifier(model_name="meta-llama/Llama-3.2-1B-Instruct", model_dir=model_dir, num_labels=num_labels)

# Load the model and data
classifier.load_data(data_path)
classifier.encode_labels()
classifier.prepare_datasets()

# Evaluate the model and print metrics
classifier.evaluate()
