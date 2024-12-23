import optuna
import random  # Missing import added here
import torch
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter
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
        # Handle NoneType class weights
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float).to(self.device)
            if class_weights is not None
            else None
        )

    def compute_loss(self, outputs, labels):
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(outputs, labels)


    def tokenize_data(self, X, y):
        encodings = self.tokenizer(
            X, truncation=True, padding=True, max_length=256
        )
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': y,
        })
        return dataset

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

    def train_and_evaluate(self, train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs):
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
            fp16=True,  # Mixed-precision training
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(3)],
        )

        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate(test_dataset)
        eval_loss = eval_results['eval_loss']  # Extract evaluation loss

        return eval_loss, trainer

def objective(trial):
    # Load the preprocessed data
    with open("processed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test, label_classes = pickle.load(f)

    # Generate K-folds dynamically
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

    # Randomly select one fold for this trial
    fold = random.choice(folds)

    # Compute class weights
    class_counts = Counter(fold["train_labels"])
    class_weights = [1.0 / class_counts[i] for i in range(len(label_classes))]

    # Initialize classifier
    classifier = TextClassifier(
        model_name="bert-base-uncased",
        num_labels=len(label_classes),
        class_weights=class_weights
    )

    # Tokenize data
    train_dataset = classifier.tokenize_data(fold["train_data"], fold["train_labels"])
    test_dataset = classifier.tokenize_data(fold["test_data"], fold["test_labels"])

    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)

    # Train and evaluate
    trainer = classifier.train_and_evaluate(
        train_dataset, test_dataset, learning_rate, batch_size, num_train_epochs
    )

    return trainer.evaluate()["eval_loss"]


if __name__ == "__main__":
    # Optuna study for hyperparameter tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best hyperparameters:", study.best_params)
