import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

class TextClassifier:
    def __init__(self, model_name, model_dir, num_labels):
        self.model_name = model_name
        self.model_dir = model_dir
        self.num_labels = num_labels

        # Move model to GPU (if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_dir)
        self.model.to(self.device)

    def train(self, output_dir='./results', num_train_epochs=3, batch_size=2):
        # Ensure datasets are loaded
        if not hasattr(self, 'tokenized_train_dataset') or not hasattr(self, 'tokenized_test_dataset'):
            raise ValueError("Datasets must be tokenized and assigned before training.")

        print(f"Training with {len(self.tokenized_train_dataset)} training samples.")

        # Set up optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        num_train_steps = len(self.tokenized_train_dataset) * num_train_epochs // batch_size
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_train_steps)

        # Use gradient accumulation
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Gradient accumulation
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            report_to="tensorboard",  # Enables TensorBoard logging
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, lr_scheduler),
            compute_metrics=self.compute_metrics
        )

        print("Starting training...")
        # Train with mixed precision
        scaler = GradScaler()
        trainer.train()  # Trainer handles batch processing and mixed precision for you
        print("Training completed.")

    def evaluate(self):
        print("Evaluating the model...")
        self.model.eval()
        # Perform evaluation (with torch.cuda.amp autocast if desired)
        with torch.no_grad():
            eval_results = self.trainer.evaluate()
            print("Evaluation Results:", eval_results)

    # Additional functions like `predict` and others can also use similar optimizations

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(logits, axis=-1)
        accuracy = (predictions == labels).float().mean()
        return {"accuracy": accuracy.item()}
