import pickle
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

class TextClassifier:
    def __init__(self, model_name, model_dir, num_labels):
        self.model_name = model_name
        self.model_dir = model_dir
        self.num_labels = num_labels

        # Load the tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {self.model_dir}: {e}")

        # Set pad_token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = 'left'

        # Load the base model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_dir}: {e}")

        # Define custom model with classification head
        self.model = self.LlamaForSequenceClassification(base_model, self.num_labels)

        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return {'loss': loss, 'logits': logits}

    def load_data(self, data_path):
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
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        print("Labels encoded successfully.")

    def prepare_datasets(self):
        train_dataset = Dataset.from_dict({'text': self.X_train, 'labels': self.y_train_encoded})
        test_dataset = Dataset.from_dict({'text': self.X_test, 'labels': self.y_test_encoded})

        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)

        self.tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        self.tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        print("Datasets prepared and tokenized.")

    def train(self, output_dir='./results', num_train_epochs=3, batch_size=2):
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
            fp16=True,
            gradient_accumulation_steps=4,
            load_best_model_at_end=True,
            max_grad_norm=1.0,
            dataloader_num_workers=2,
            report_to="none",
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

    def evaluate(self):
        print("Evaluating the model...")
        eval_results = self.trainer.evaluate()
        print(f"\nEvaluation results:\n{eval_results}")

        predictions = self.trainer.predict(self.tokenized_test_dataset)
        preds = predictions.predictions.argmax(-1)

        true_labels = self.label_encoder.inverse_transform(self.y_test_encoded)
        predicted_labels = self.label_encoder.inverse_transform(preds)

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
        encoding = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        encoding = {k: v.to(self.model.base_model.device) for k, v in encoding.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs['logits']
            preds = logits.argmax(-1).cpu().numpy()

        predicted_labels = self.label_encoder.inverse_transform(preds)
        return predicted_labels

if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7B-hf"
    model_dir = "llama2_model"
    data_path = 'processed_data.pkl'

    with open(data_path, 'rb') as f:
        _, _, _, _, labels = pickle.load(f)
    num_labels = len(set(labels))

    classifier = TextClassifier(model_name, model_dir, num_labels)

    classifier.load_data(data_path)
    classifier.encode_labels()
    classifier.prepare_datasets()

    classifier.train(output_dir='./results', num_train_epochs=3, batch_size=2)

    classifier.evaluate()

    new_texts = [
        "Sample text for classification.",
        "Another example text to classify."
    ]
    predictions = classifier.predict(new_texts)
    print("\nPredictions on new texts:")
    for text, label in zip(new_texts, predictions):
        print(f"Text: {text}\nPredicted Label: {label}\n")
