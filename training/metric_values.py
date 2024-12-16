from train_llama import TextClassifier  # Import TextClassifier from train_llama.py
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold

if __name__ == "__main__":
    # Fixed best parameters from hyperparameter tuning
    best_params = {'learning_rate': 6.044169858061599e-05, 'batch_size': 8, 'num_train_epochs': 4}

    # Load the preprocessed data
    with open("processed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test, label_classes = pickle.load(f)

    # Generate K-folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = [
        {
            "train_data": [X_train[i] for i in train_idx],
            "train_labels": [y_train[i] for i in train_idx],
            "test_data": [X_train[i] for i in val_idx],
            "test_labels": [y_train[i] for i in val_idx],
        }
        for train_idx, val_idx in kf.split(X_train)
    ]

    # Use a single fold (the first one) for evaluation
    fold = folds[0]

    # Compute class weights
    class_counts = Counter(fold["train_labels"])
    class_weights = [1.0 / class_counts[i] for i in range(len(label_classes))]

    # Initialize classifier
    classifier = TextClassifier(
        model_name="bert-base-uncased",  # Replace with your desired model name
        num_labels=len(label_classes),
        class_weights=class_weights
    )

    # Tokenize data
    train_dataset = classifier.tokenize_data(fold["train_data"], fold["train_labels"])
    test_dataset = classifier.tokenize_data(fold["test_data"], fold["test_labels"])

    # Train and evaluate
    eval_loss, trainer = classifier.train_and_evaluate(
        train_dataset,
        test_dataset,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
    )

    # Predict and print metrics
    preds = trainer.predict(test_dataset).predictions.argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(fold["test_labels"], preds, target_names=label_classes))

'''
{'learning_rate': 5.699270651832986e-05, 'batch_size': 16, 'num_train_epochs': 8}
{'learning_rate': 6.044169858061599e-05, 'batch_size': 8, 'num_train_epochs': 4}
Classification Report:
              precision    recall  f1-score   support

           0       0.74      0.73      0.73        73
           1       0.99      1.00      0.99        85
           2       0.78      0.73      0.75        77
           3       0.92      0.98      0.95        82

    accuracy                           0.86       317
   macro avg       0.86      0.86      0.86       317
weighted avg       0.86      0.86      0.86       317
'''