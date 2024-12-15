import optuna
import torch
import pickle
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

    def load_data(self, data_path):
        with open(data_path, "rb") as f:
            X_train, X_test, y_train, y_test, label_classes = pickle.load(f)
        return X_train, X_test, y_train, y_test, label_classes

    def tokenize_data(self, X_train, X_test, y_train, y_test):
        train_encodings = self.tokenizer(
            X_train, truncation=True, padding=True, max_length=512
        )
        test_encodings = self.tokenizer(
            X_test, truncation=True, padding=True, max_length=512
        )

        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': y_train,
        })

        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': y_test,
        })

        return train_dataset, test_dataset

    def train_and_evaluate(self, train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_loss"], trainer

    def predict(self, test_dataset, trainer):
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        return preds

# Objective function for Optuna
def objective(trial):
    # Create classifier
    classifier = TextClassifier("bert-base-uncased", num_labels=4)

    # Load and tokenize data
    X_train, X_test, y_train, y_test, _ = classifier.load_data("processed_data.pkl")
    train_dataset, test_dataset = classifier.tokenize_data(X_train, X_test, y_train, y_test)

    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 10)

    # Train and evaluate model
    eval_loss, _ = classifier.train_and_evaluate(
        train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs
    )
    return eval_loss

if __name__ == "__main__":
    # Hyperparameter tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
