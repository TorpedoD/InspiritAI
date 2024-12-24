import pickle
import random
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import optuna
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

class TextClassifier:
    def __init__(self, model_name, num_labels, class_weights=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float).to(self.device)
            if class_weights is not None
            else None
        )

    def tokenize_data(self, X, y):
        encodings = self.tokenizer(X, truncation=True, padding='max_length', max_length=256)
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': y,
        })

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=1)
        metrics = classification_report(
            labels, preds, output_dict=True, zero_division=0
        )
        print("Confusion Matrix:\n", confusion_matrix(labels, preds))
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro avg"]["f1-score"],
            "weighted_f1": metrics["weighted avg"]["f1-score"],
        }

    def create_trainer(self, train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            label_smoothing_factor=0.1,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            fp16=True,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(3)],
        )

    def train_and_evaluate(self, train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs):
        trainer = self.create_trainer(
            train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs
        )
        trainer.train()
        eval_results = trainer.evaluate(test_dataset)
        return eval_results["eval_loss"], trainer

def objective(trial):
    with open("processed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test, label_classes = pickle.load(f)

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

    fold = random.choice(folds)

    class_counts = Counter(fold["train_labels"])
    class_weights = [
        1.0 / class_counts.get(i, 1) for i in range(len(label_classes))
    ]  # Default to 1 if class is missing in fold

    classifier = TextClassifier(
        model_name="bert-base-uncased",
        num_labels=len(label_classes),
        class_weights=class_weights
    )

    train_dataset = classifier.tokenize_data(fold["train_data"], fold["train_labels"])
    test_dataset = classifier.tokenize_data(fold["test_data"], fold["test_labels"])

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)

    eval_loss, trainer = classifier.train_and_evaluate(
        train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs
    )

    # Log trial information
    print(f"Trial {trial.number}: LR={learning_rate}, BS={batch_size}, Epochs={num_train_epochs}, Loss={eval_loss}")
    return eval_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best hyperparameters:", study.best_params)
