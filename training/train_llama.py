import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from torch.cuda.amp import GradScaler, autocast

class OptimizedTextClassifier:
    def __init__(self, model_name, model_dir, num_labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("CUDA not available, using CPU.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        # Model with classification head
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir)
        self.num_labels = num_labels
        self.model.to(self.device)

    def prepare_datasets(self, X_train, y_train, X_test, y_test):
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        
        train_data = Dataset.from_dict({'text': X_train, 'labels': y_train})
        test_data = Dataset.from_dict({'text': X_test, 'labels': y_test})
        
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
