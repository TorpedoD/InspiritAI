import pickle
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

class TextStyleGenerator:
    def __init__(self, model_name, model_dir):
        self.model_name = model_name
        self.model_dir = model_dir

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_dir)

        # Set the pad_token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad_token to eos_token if not set

        # Set padding_side to 'left' for decoder-only architecture
        self.tokenizer.padding_side = 'left'

        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_dir)

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
        self.unique_labels = unique_labels

    def prepare_datasets(self):
        # Combine labels and texts for conditional generation
        def combine_label_and_text(texts, labels):
            return [f"<LABEL>{label}</LABEL> {text}" for label, text in zip(labels, texts)]

        # Encode labels if necessary
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.label_encoder.classes_)}

        # Prepare training and test data
        train_texts = combine_label_and_text(self.X_train, self.y_train)
        test_texts = combine_label_and_text(self.X_test, self.y_test)

        # Create datasets
        train_dataset = Dataset.from_dict({'text': train_texts})
        test_dataset = Dataset.from_dict({'text': test_texts})

        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=512)

        # Tokenize datasets
        self.tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Remove unused columns
        self.tokenized_train_dataset = self.tokenized_train_dataset.remove_columns(['text'])
        self.tokenized_test_dataset = self.tokenized_test_dataset.remove_columns(['text'])

        # Set format for PyTorch
        self.tokenized_train_dataset.set_format('torch')
        self.tokenized_test_dataset.set_format('torch')

        print("Datasets prepared and tokenized for conditional generation.")

    def train(self, output_dir='./results', num_train_epochs=3, batch_size=2):
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            prediction_loss_only=True,
            evaluation_strategy="epoch",
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        print("Starting training...")
        self.trainer.train()
        print("Training completed.")

    def generate_text(self, label, prompt="", max_length=50, num_return_sequences=1):
        # Ensure the label is valid
        if label not in self.label_encoder.classes_:
            raise ValueError(f"Label '{label}' not found in label encoder classes.")

        # Encode the label and prompt
        label_text = f"<LABEL>{label}</LABEL> {prompt}"
        input_ids = self.tokenizer.encode(label_text, return_tensors='pt').to(self.model.device)

        # Generate text
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode generated texts
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Remove the label and prompt from the generated text
        generated_texts = [text.replace(label_text, '').strip() for text in generated_texts]

        return generated_texts

# Usage example
if __name__ == "__main__":
    # Initialize the generator
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_dir = "llama_model"  # Directory to save/load the model
    data_path = 'processed_data.pkl'

    generator = TextStyleGenerator(model_name, model_dir)

    # Load and preprocess data
    generator.load_data(data_path)
    generator.prepare_datasets()

    # Train the model
    generator.train(output_dir='./results', num_train_epochs=3, batch_size=2)

    # Generate text in the style of a label
    label = "YourLabel"  # Replace with an actual label from your dataset
    prompt = "Once upon a time"
    generated_texts = generator.generate_text(label, prompt, max_length=100, num_return_sequences=3)
    for idx, text in enumerate(generated_texts):
        print(f"Generated Text {idx+1}:\n{text}\n")
