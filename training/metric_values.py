import pickle
import torch
from transformers import AutoModelForCausalLM  # Import from Hugging Face library
from train_llama import TextClassifier  # Ensure this is importing the correct class from your original script

def evaluate_model(model_path, data_path):
    # Load the data
    with open(data_path, 'rb') as f:
        X_train, X_test, y_train, y_test, labels = pickle.load(f)

    # Initialize the classifier (no need to train)
    classifier = TextClassifier(model_name="meta-llama/Llama-3.2-1B-Instruct", model_dir=model_path, num_labels=len(set(labels)))
    
    # Load the fine-tuned model directly from the saved path
    classifier.model = AutoModelForCausalLM.from_pretrained(model_path).to(classifier.device)
    
    # No need to load data, encode labels, or prepare datasets again if they were already preprocessed
    classifier.X_test = X_test
    classifier.y_test = y_test
    classifier.labels = labels
    classifier.encode_labels()
    classifier.prepare_datasets()

    # Evaluate the model and print metrics
    classifier.evaluate()

if __name__ == "__main__":
    model_path = './saved_model'  # Path where the trained model is saved
    data_path = 'processed_data.pkl'  # Path to your dataset
    evaluate_model(model_path, data_path)
