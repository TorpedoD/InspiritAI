import pickle
import torch
from transformers import AutoModelForCausalLM
from train_llama import TextClassifier  # Ensure this is importing the correct class from your original script

def evaluate_model(model_path, data_path):
    # Load the data
    with open(data_path, 'rb') as f:
        X_train, X_test, y_train, y_test, labels = pickle.load(f)

    # Initialize the classifier
    classifier = TextClassifier(model_name="meta-llama/Llama-3.2-1B-Instruct", model_dir=model_path, num_labels=len(set(labels)))
    
    # Load the saved model manually from Hugging Face
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Reinitialize your custom classifier with the loaded model
    classifier.model = TextClassifier.LlamaForSequenceClassification(base_model, num_labels=len(set(labels)))
    classifier.model.eval()

    # Evaluate the model and print metrics
    classifier.evaluate()

if __name__ == "__main__":
    model_path = './saved_model'  # Path where the trained model is saved
    data_path = 'processed_data.pkl'  # Path to your dataset
    evaluate_model(model_path, data_path)
