import optuna
import torch
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

    def load_data(self, data_path):
        with open(data_path, "rb") as f:
            X_train, X_test, y_train, y_test, label_classes = pickle.load(f)
        return X_train, X_test, y_train, y_test, label_classes

    def tokenize_data(self, X_train, X_test, y_train, y_test):
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=512)
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=512)

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
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()
        return trainer.evaluate()["eval_loss"]

def objective(trial):
    classifier = TextClassifier("roberta-base", num_labels=4)
    X_train, X_test, y_train, y_test, _ = classifier.load_data("processed_data.pkl")
    train_dataset, test_dataset = classifier.tokenize_data(X_train, X_test, y_train, y_test)

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)

    return classifier.train_and_evaluate(train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best hyperparameters:", study.best_params)
