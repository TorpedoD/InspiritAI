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
Classification Report:
              precision    recall  f1-score   support

     GPT_txt       0.94      0.91      0.92       221
    Theo_txt       0.76      0.92      0.83       199
  damian_txt       0.83      0.71      0.77       218
   misia_txt       0.80      0.78      0.79       218

    accuracy                           0.83       856
   macro avg       0.83      0.83      0.83       856
weighted avg       0.83      0.83      0.83       856

{'learning_rate': 6.044169858061599e-05, 'batch_size': 8, 'num_train_epochs': 4}
Classification Report:
              precision    recall  f1-score   support

     GPT_txt       0.93      0.91      0.92       221
    Theo_txt       0.75      0.92      0.83       199
  damian_txt       0.76      0.69      0.73       218
   misia_txt       0.80      0.72      0.76       218

    accuracy                           0.81       856
   macro avg       0.81      0.81      0.81       856
weighted avg       0.81      0.81      0.81       856
'''
