import pickle
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import os

# Set the CUDA_VISIBLE_DEVICES environment variable if needed to avoid conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU

class TextClassifier:
    def __init__(self, model_name, model_dir, num_labels, device='cuda'):
        self.device = device
        self.model_name = model_name
        self.model_dir = model_dir
        self.num_labels = num_labels

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_dir)

        # Set the pad_token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use left padding for decoder-only models
        self.tokenizer.padding_side = 'left'

        # Load the base model in mixed precision (half-precision)
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_dir).half()

        # Define the custom model with classification head
        self.model = self.LlamaForSequenceClassification(base_model, self.num_labels).to(self.device)

    class LlamaForSequenceClassification(nn.Module):
        def __init__(self, base_model, num_labels):
            super(TextClassifier.LlamaForSequenceClassification, self).__init__()
            self.base_model = base_model
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            pooled_output = last_hidden_state[:, -1, :]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

            return {'loss': loss, 'logits': logits}

    def load_data(self, data_path):
        # Load processed data
        with open(data_path, 'rb') as f:
            X_train, X_test, y_train, y_test, labels = pickle.load(f)

        print(f"Total number of samples in the dataset: {len(X_train) + len(X_test)}")
        unique_labels = set(labels)
        print(f"Total number of unique labels: {len(unique_labels)}")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.labels = labels

    def encode_labels(self):
        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        print("Labels encoded successfully.")

    def prepare_datasets(self):
        # Create datasets
        train_dataset = Dataset.from_dict({'text': self.X_train, 'labels': self.y_train_encoded})
        test_dataset = Dataset.from_dict({'text': self.X_test, 'labels': self.y_test_encoded})

        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True)

        # Tokenize datasets
        self.tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Set format for PyTorch with dynamic padding
        self.tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        print("Datasets prepared and tokenized.")

    def train(self, output_dir='./results', num_train_epochs=3, batch_size=2):
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Use gradient accumulation
            fp16=True,  # Enable mixed precision
            evaluation_strategy="epoch",
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=10,
            device=self.device  # Make sure device is set properly
        )

        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        print("Starting training...")
        self.trainer.train()
        print("Training completed.")

    def evaluate(self):
        # Evaluate the model
        print("Evaluating the model...")
        eval_results = self.trainer.evaluate()
        print(f"\nEvaluation results:\n{eval_results}")

        # Get predictions
        predictions = self.trainer.predict(self.tokenized_test_dataset)
        preds = predictions.predictions.argmax(-1)

        # Decode labels
        true_labels = self.label_encoder.inverse_transform(self.y_test_encoded)
        predicted_labels = self.label_encoder.inverse_transform(preds)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels, labels=self.label_encoder.classes_))

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
        # Tokenize texts
        encoding = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoding = encoding.to(self.device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)
            return self.label_encoder.inverse_transform(preds.cpu().numpy())

# Example Usage:
# Initialize the classifier
model_name = 'path_to_pretrained_model_or_model_name'
output_dir = './results'
num_labels = len(set(y_train))  # Define number of labels

classifier = TextClassifier(model_name=model_name, model_dir=output_dir, num_labels=num_labels)

# Load and process data
classifier.load_data('path_to_pickled_data.pkl')

# Encode labels
classifier.encode_labels()

# Prepare datasets
classifier.prepare_datasets()

# Train the model
classifier.train()

# Evaluate the model
classifier.evaluate()
