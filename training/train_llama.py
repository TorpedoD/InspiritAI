import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os
from transformers import AutoModelForSequenceClassification  # Changed model type to SequenceClassification

class OptimizedTextClassifier:
    def __init__(self, model_name=None, model_dir=None, num_labels=None, model_path=None):
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("CUDA not available, using CPU.")
        
        if model_path:
            # Load saved model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
        else:
            # Load model and tokenizer from pretrained
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'

            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_dir, num_labels=num_labels)
            self.model.to(self.device)

    def prepare_datasets(self, X_train, y_train, X_test, y_test):
        def tokenize_function(examples):
            # Debugging line to ensure 'text' field contains list of strings
            print(f"Examples: {examples['text'][:5]}")  # Print first 5 examples for debugging
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        
        # Ensure that X_train and X_test are lists of strings
        train_data = Dataset.from_dict({'text': X_train, 'labels': y_train})
        test_data = Dataset.from_dict({'text': X_test, 'labels': y_test})
        
        # Ensure input format is correct before mapping
        print(f"Train data first 5 texts: {X_train[:5]}")  # Check format of X_train
        self.train_dataset = train_data.map(tokenize_function, batched=True)
        self.test_dataset = test_data.map(tokenize_function, batched=True)
        
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    def train(self, output_dir='./results', num_epochs=3, batch_size=4, lr=5e-5):
        scaler = GradScaler()  # For mixed precision training

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Reduce memory overhead
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_dir='./logs',
            fp16=True,  # Mixed precision
            learning_rate=lr,
            dataloader_num_workers=4,  # Optimize data loading
            report_to="tensorboard",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def save_model(self, output_dir='./save_model'):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def plot_roc_curve(self, model_path, test_dataset, output_dir='./plots'):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load saved model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
        # Get predictions from the model
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer
        )
        pred = trainer.predict(test_dataset)
    
        # Get true labels and predictions
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Binarize the labels for multi-class ROC curve
        lb = LabelBinarizer()
        lb.fit(labels)
        y_true_bin = lb.transform(labels)
        y_pred_bin = lb.transform(preds)
    
        # Compute ROC curve and AUC for each class
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
        roc_auc = auc(fpr, tpr)
    
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
    
        # Save the plot to a file
        output_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(output_path)
        print(f"ROC curve saved to {output_path}")
        plt.close()  # Close the plot to avoid overlapping with future plots

    def plot_confusion_matrix(self, model_path, test_dataset, output_dir='./plots'):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
        # Load saved model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
        # Get predictions from the model
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer
        )
        pred = trainer.predict(test_dataset)
    
        # Get true labels and predictions
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
    
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)
    
        # Plot confusion matrix as heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
    
        # Save the plot to a file
        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path)
        print(f"Confusion matrix saved to {output_path}")
        plt.close()  # Close the plot to avoid overlapping with future plots

# Usage
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_dir = "llama_model"
    data_path = 'processed_data.pkl'

    with open(data_path, 'rb') as f:
        X_train, X_test, y_train, y_test, _ = pickle.load(f)

    classifier = OptimizedTextClassifier(model_name, model_dir, num_labels=10)
    classifier.prepare_datasets(X_train, y_train, X_test, y_test)
    classifier.train()
    classifier.save_model()
