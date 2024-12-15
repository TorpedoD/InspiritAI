from train_llama import TextClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    best_params = {'learning_rate': 2e-5, 'batch_size': 16, 'num_train_epochs': 5}

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
