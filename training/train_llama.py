import pickle
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TextClassifier:
    def __init__(self, model_name, model_dir, num_labels):
        # Check if CUDA is available and if it's working
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("CUDA is not available. Using CPU.")

        self.model_name = model_name
        self.model_dir = model_dir
        self.num_labels = num_labels

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_dir)

        # Set the pad_token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set padding_side to 'left' for decoder-only architecture
        self.tokenizer.padding_side = 'left'

        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_dir)

        # Define the custom model with classification head
        self.model = self.LlamaForSequenceClassification(base_model, self.num_labels).to(self.device)

        # Initialize Trainer (move this to __init__)
        self.trainer = None  # Initialize it to None to handle cases before training

    class LlamaForSequenceClassification(nn.Module):
        def __init__(self, base_model, num_labels):
            super(TextClassifier.LlamaForSequenceClassification, self).__init__()
            self.base_model = base_model
            self.num_labels = num_labels
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            # Ensure the base model outputs hidden states
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            
            # Get the last hidden state
            hidden_states = outputs.hidden_states[-1]
            
            # Use attention mask to get the last valid token for each sequence
            if attention_mask is not None:
                # Get the last valid token index for each sequence
                last_token_indices = attention_mask.sum(dim=1) - 1
                # Use gather to select the last valid token's hidden state for each sequence
                batch_indices = torch.arange(input_ids.size(0)).to(input_ids.device)
                pooled_output = hidden_states[batch_indices, last_token_indices, :]
            else:
                # If no attention mask, use the last token's hidden state
                pooled_output = hidden_states[:, -1, :]
            
            # Apply dropout and classification
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            return {'loss': loss, 'logits': logits}

    def load_data(self, data_path):
        with open(data_path, 'rb') as f:
            X_train, X_test, y_train, y_test, label_classes = pickle.load(f)
        print(f"Total number of samples in the dataset: {len(X_train) + len(X_test)}")
        print(f"Total number of unique labels: {len(label_classes)}")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.labels = label_classes

    def encode_labels(self):
        self.label_encoder = LabelEncoder()
        
        # Combine training and testing labels correctly
        all_labels = list(self.y_train) + list(self.y_test)  # Convert to lists and concatenate
        self.label_encoder.fit(all_labels)  # Fit on the combined set of labels

        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        print("Labels encoded successfully.")


    def prepare_datasets(self):
        train_dataset = Dataset.from_dict({'text': self.X_train, 'labels': self.y_train_encoded})
        test_dataset = Dataset.from_dict({'text': self.X_test, 'labels': self.y_test_encoded})

        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

        self.tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        self.tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        print("Datasets prepared and tokenized.")

    def train(self, output_dir='./results', num_train_epochs=3, batch_size=2, learning_rate=5e-5):
        assert self.num_labels == len(self.label_encoder.classes_), "Mismatch in num_labels and actual label classes."

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            save_strategy='no',  # Disable saving checkpoints
            report_to="tensorboard",
            learning_rate=learning_rate,  # Added learning rate parameter
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        print("Starting training...")
        self.trainer.train()
        print("Training completed.")

    def save_model(self):
        """Save the model, tokenizer, and label encoder to './save_model'."""
        output_dir = './save_model'  # Set the save location to './save_model'
    
        # Save the base model
        self.model.base_model.save_pretrained(output_dir, safe_serialization=True)  # Safe serialization for transformers models
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save the label encoder
        with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Model saved successfully to {output_dir}")

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def predict(self, texts):
        encoding = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs['logits']
            preds = logits.argmax(-1).cpu().numpy()

        predicted_labels = self.label_encoder.inverse_transform(preds)
        return predicted_labels

    def load_trained_model(self, model_dir):
        """Load a trained model, tokenizer, and label encoder from a saved directory."""
        print("Loading trained model...")

        # Load the base model (Llama) architecture
        base_model = AutoModelForCausalLM.from_pretrained(model_dir, cache_dir=model_dir)

        # Recreate the custom classification model
        self.model = self.LlamaForSequenceClassification(base_model, self.num_labels).to(self.device)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load the label encoder
        with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        print(f"Trained model loaded from {model_dir}")

    def evaluate(self):
        # Debugging log for num_labels and label encoder classes
        print(f"Expected num_labels: {self.num_labels}")
        print(f"Actual classes in LabelEncoder: {len(self.label_encoder.classes_)}")
        print(f"LabelEncoder classes: {self.label_encoder.classes_}")

        # Validate `num_labels`
        assert self.num_labels == len(self.label_encoder.classes_), "Mismatch in num_labels and actual label classes."

        # Initialize the Trainer if not already done
        if self.trainer is None:
            print("Initializing Trainer for evaluation...")
            training_args = TrainingArguments(
                output_dir='./results',
                per_device_eval_batch_size=16,
                evaluation_strategy="epoch",
                logging_dir='./logs',
                logging_steps=10,
                save_strategy='no',
            )
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=self.tokenized_test_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )

        print("Evaluating the model...")
        eval_results = self.trainer.evaluate()
        print(f"\nEvaluation results:\n{eval_results}")

        # Get predictions
        predictions = self.trainer.predict(self.tokenized_test_dataset)
        preds = predictions.predictions.argmax(-1)

        # Decode predictions and labels
        true_labels = self.label_encoder.inverse_transform(self.y_test_encoded)
        predicted_labels = self.label_encoder.inverse_transform(preds)

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels, labels=self.label_encoder.classes_))

        # Generate confusion matrix and ROC curve
        self.generate_visualizations(true_labels, predicted_labels)

    def generate_visualizations(self, true_labels, predicted_labels):
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # ROC Curve for valid indices
        true_labels_one_hot = np.array([self.label_encoder.transform([label])[0] for label in true_labels])
        predicted_probs = self.trainer.predict(self.tokenized_test_dataset).predictions
        predicted_probs = torch.softmax(torch.tensor(predicted_probs), dim=-1).numpy()

        plt.figure(figsize=(8, 6))
        for i in range(len(self.label_encoder.classes_)):
            if i >= predicted_probs.shape[1]:
                print(f"Skipping invalid index {i} for ROC Curve")
                continue
            fpr, tpr, _ = roc_curve(true_labels_one_hot == i, predicted_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {self.label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')
        plt.close()


# Usage example
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_dir = "./save_model"
    data_path = 'processed_data.pkl'

    classifier = TextClassifier(model_name, model_dir, num_labels=4)  # Correct number of labels
    classifier.load_data(data_path)
    classifier.encode_labels()
    classifier.prepare_datasets()
    classifier.train()
    classifier.save_model()  # This will save everything in './save_model'