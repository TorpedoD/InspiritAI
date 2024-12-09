import pickle
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from train_llama import TextClassifier  # Replace with the correct file name where the class is located

def evaluate_saved_model(model_dir, data_path):
    # Load the saved model and tokenizer
    with open(data_path, 'rb') as f:
        X_train, X_test, y_train, y_test, labels = pickle.load(f)

    classifier = TextClassifier(model_name="meta-llama/Llama-3.2-1B-Instruct", model_dir=model_dir, num_labels=len(set(labels)))
    
    # Load data and prepare datasets
    classifier.load_data(data_path)
    classifier.encode_labels()
    classifier.prepare_datasets()

    # Evaluate the model
    print("Evaluating the saved model...")
    classifier.evaluate()

if __name__ == "__main__":
    # Path to the saved model and data
    model_dir = './saved_model'  # Replace with the directory where your model is saved
    data_path = 'processed_data.pkl'  # Replace with your actual data file path

    evaluate_saved_model(model_dir, data_path)
