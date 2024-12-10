import pickle
from text_classifier import TextClassifier  # Assuming the class is saved in a file named 'text_classifier.py'

def main():
    # Define the necessary paths
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_dir = "save_model"  # Path to the pre-trained model directory
    data_path = 'processed_data.pkl'  # Path to your data file (pickle file with training/testing data)
    
    # Initialize the TextClassifier
    classifier = TextClassifier(model_name, model_dir, num_labels=10)  # Assuming 10 classes in your dataset
    
    # Load the data
    classifier.load_data(data_path)
    
    # Encode the labels
    classifier.encode_labels()
    
    # Prepare the datasets
    classifier.prepare_datasets()
    
    # Optionally, you can train the model here (uncomment if you want to train)
    # classifier.train()
    
    # Load the trained model if you have already saved it
    classifier.load_trained_model('./save_model')  # Assuming you saved the model to './save_model'
    
    # Evaluate the model and generate visualizations
    classifier.evaluate()  # This will generate and save the confusion matrix and ROC curve as PNG files

if __name__ == "__main__":
    main()
