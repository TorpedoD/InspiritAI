from train_llama import TextClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Best parameters from hyperparameter tuning
    best_params = {'learning_rate': 3.360440024365462e-05, 'batch_size': 32, 'num_train_epochs': 5}

    # Initialize classifier
    classifier = TextClassifier("bert-base-uncased", num_labels=4)

    # Load and tokenize data
    X_train, X_test, y_train, y_test, _ = classifier.load_data("processed_data.pkl")
    train_dataset, test_dataset = classifier.tokenize_data(X_train, X_test, y_train, y_test)

    # Train final model
    eval_loss, trainer = classifier.train_and_evaluate(
        train_dataset,
        test_dataset,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
    )

    # Predict and evaluate
    preds = classifier.predict(test_dataset, trainer)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

'''
              precision    recall  f1-score   support

           0       0.44      0.62      0.51        52
           1       0.99      0.99      0.99        85
           2       0.46      0.48      0.47        54
           3       0.00      0.00      0.00        24

    accuracy                           0.66       215
   macro avg       0.47      0.52      0.49       215
weighted avg       0.61      0.66      0.63       215

'''