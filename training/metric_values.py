from train_llama import TextClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    best_params = {'learning_rate': 6.044169858061599e-05, 'batch_size': 8, 'num_train_epochs': 4}

    classifier = TextClassifier("roberta-base", num_labels=4)
    X_train, X_test, y_train, y_test, _ = classifier.load_data("processed_data.pkl")
    train_dataset, test_dataset = classifier.tokenize_data(X_train, X_test, y_train, y_test)

    eval_loss, trainer = classifier.train_and_evaluate(
        train_dataset,
        test_dataset,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
    )

    preds = trainer.predict(test_dataset).predictions.argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

'''
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