import os
import pickle
from train_llama import OptimizedTextClassifier  # Import the class from the appropriate file
from datasets import Dataset

# Load the preprocessed data (same as the training script)
data_path = 'processed_data.pkl'

with open(data_path, 'rb') as f:
    X_train, X_test, y_train, y_test, _ = pickle.load(f)

# Initialize the classifier
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_dir = "llama_model"
classifier = OptimizedTextClassifier(model_name, model_dir, num_labels=10)

# Prepare the datasets
classifier.prepare_datasets(X_train, y_train, X_test, y_test)

# Train the model
classifier.train()

# Save the model after training
saved_model_path = './save_model'
classifier.save_model(output_dir=saved_model_path)

# Plot the ROC curve and confusion matrix using the saved model
output_dir = './plots'  # Folder where the plots will be saved

# Generate and save the ROC curve and confusion matrix plots
classifier.plot_roc_curve(saved_model_path, classifier.test_dataset, output_dir)
classifier.plot_confusion_matrix(saved_model_path, classifier.test_dataset, output_dir)
