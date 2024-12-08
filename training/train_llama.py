import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from safetensors.torch import save_file, load_file

class TextClassifier:
    def __init__(self, model_name, model_dir, num_labels, device=None):
        """
        Initialize the text classifier.
        
        Args:
            model_name (str): Pretrained model name
            model_dir (str): Directory to cache/save models
            num_labels (int): Number of unique labels
            device (str, optional): Device to use (cuda/cpu)
        """
        # Set up device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store initialization parameters
        self.model_name = model_name
        self.model_dir = model_dir
        self.num_labels = num_labels

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.model_dir,
            padding_side='left',
            use_fast=True
        )

        # Set pad token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=self.model_dir,
            device_map=self.device
        )

        # Create classification model
        self.model = self.LlamaForSequenceClassification(base_model, num_labels)
        self.model.to(self.device)

    class LlamaForSequenceClassification(nn.Module):
        def __init__(self, base_model, num_labels):
            """
            Custom classification model on top of a base language model.
            
            Args:
                base_model: Pretrained language model
                num_labels (int): Number of unique labels
            """
            super().__init__()
            self.base_model = base_model
            self.num_labels = num_labels
            
            # Freeze base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            # Classification head
            self.dropout = nn.Dropout(0.3)  # Increased dropout for regularization
            self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
            
            # Initialize classifier weights
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            """
            Forward pass for classification.
            
            Args:
                input_ids (torch.Tensor): Input token IDs
                attention_mask (torch.Tensor): Attention mask
                labels (torch.Tensor, optional): Ground truth labels
            
            Returns:
                dict: Model outputs including loss and logits
            """
            # Get base model outputs
            outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            
            # Extract hidden states
            hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            
            # Use last layer's hidden state
            last_hidden_state = hidden_states[-1]
            
            # Pool by taking the last token's representation
            pooled_output = last_hidden_state[:, -1, :]
            
            # Apply dropout and classification
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return {
                'loss': loss, 
                'logits': logits
            }

    def load_data(self, data_path):
        """
        Load processed data from pickle file.
        
        Args:
            data_path (str): Path to pickled data file
        """
        try:
            with open(data_path, 'rb') as f:
                X_train, X_test, y_train, y_test, labels = pickle.load(f)

            print(f"Total samples: {len(X_train) + len(X_test)}")
            print(f"Unique labels: {len(set(labels))}")

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.labels = labels
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def encode_labels(self):
        """
        Encode labels to integer representation.
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        print("Labels encoded successfully.")

    def prepare_datasets(self):
        """
        Prepare and tokenize datasets for training.
        """
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': self.X_train, 
            'labels': self.y_train_encoded
        })
        test_dataset = Dataset.from_dict({
            'text': self.X_test, 
            'labels': self.y_test_encoded
        })

        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                padding='max_length', 
                truncation=True, 
                max_length=512
            )

        # Tokenize and prepare datasets
        self.tokenized_train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        self.tokenized_test_dataset = test_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=test_dataset.column_names
        )

        # Add the 'labels' column to the datasets explicitly
        self.tokenized_train_dataset = self.tokenized_train_dataset.add_column('labels', self.y_train_encoded)
        self.tokenized_test_dataset = self.tokenized_test_dataset.add_column('labels', self.y_test_encoded)

        # Set format for PyTorch
        self.tokenized_train_dataset.set_format(
            'torch', 
            columns=['input_ids', 'attention_mask', 'labels']
        )
        self.tokenized_test_dataset.set_format(
            'torch', 
            columns=['input_ids', 'attention_mask', 'labels']
        )
        
        print("Datasets prepared and tokenized.")

    def compute_metrics(self, eval_pred):
        """
        Compute accuracy and other evaluation metrics.
        
        Args:
            eval_pred: Prediction results from the model evaluation
        
        Returns:
            dict: Dictionary with accuracy and other metrics
        """
        logits, labels = eval_pred
        logits = torch.tensor(logits)  # Ensure logits is a tensor

        preds = torch.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy
        }

    def train(self, output_dir='./results', num_train_epochs=3, batch_size=2):
        """
        Train the model with optimized training arguments.
        
        Args:
            output_dir (str): Directory to save results
            num_train_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Training arguments with improved configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            save_total_limit=2,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            report_to="tensorboard",
            gradient_accumulation_steps=1,
            fp16=torch.cuda.is_available(),
        )

        # Initialize Trainer with early stopping
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train the model
        print("Starting training...")
        training_result = self.trainer.train()
        
        # Save the model
        self.save_model(output_dir)
        
        return training_result

    def save_model(self, output_dir):
        """
        Save model and tokenizer safely.
        
        Args:
            output_dir (str): Directory to save model
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state dict
        state_dict = {k: v.clone().detach().cpu() 
                      for k, v in self.model.state_dict().items()}
        
        # Save using safetensors
        save_file(state_dict, os.path.join(output_dir, 'model.safetensors'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")

    def load_saved_model(self, output_dir):
        """
        Load a saved model.
        
        Args:
            output_dir (str): Directory with saved model
        
        Returns:
            TextClassifier: Loaded model instance
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        # Recreate model structure
        loaded_model = self.LlamaForSequenceClassification(
            base_model, 
            self.num_labels
        )
        
        # Load state dict
        state_dict = load_file(os.path.join(output_dir, 'model.safetensors'))
        loaded_model.load_state_dict(state_dict)
        
        # Return loaded model
        return loaded_model

    def evaluate(self):
        """
        Evaluate the trained model on test data.
        """
        results = self.trainer.evaluate()
        print(f"Evaluation results: {results}")
        return results


# Main method to run the script
def main():
    model_name = 'your-pretrained-model-name'  # Replace with the correct model name
    model_dir = './model_cache'  # Directory to save cached models
    num_labels = 2  # Adjust according to your number of labels

    # Create an instance of the TextClassifier
    classifier = TextClassifier(model_name, model_dir, num_labels)

    # Load data
    data_path = 'path_to_your_data.pkl'  # Replace with your data path
    classifier.load_data(data_path)

    # Encode labels
    classifier.encode_labels()

    # Prepare datasets
    classifier.prepare_datasets()

    # Train the model
    output_dir = './results'
    classifier.train(output_dir=output_dir)

    # Save the trained model
    classifier.save_model(output_dir)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
