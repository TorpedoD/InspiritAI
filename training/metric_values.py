import pickle
from train_llama import TextClassifier  # Make sure this imports your original class

def evaluate_model(model_path, data_path):
    # Load the data
    with open(data_path, 'rb') as f:
        X_train, X_test, y_train, y_test, labels = pickle.load(f)

    # Initialize the classifier
    classifier = TextClassifier(model_name="meta-llama/Llama-3.2-1B-Instruct", model_dir=model_path, num_labels=len(set(labels)))
    classifier.load_data(data_path)
    classifier.encode_labels()
    classifier.prepare_datasets()

    # Load the trained model
    classifier.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    classifier.model.eval()

    # Evaluate the model and print metrics
    classifier.evaluate()

if __name__ == "__main__":
    model_path = './results'  # Path where the trained model is saved
    data_path = 'processed_data.pkl'  # Path to your dataset
    evaluate_model(model_path, data_path)
