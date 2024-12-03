import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct", model_dir="llama_model"):
    """
    Loads the model and tokenizer from the Hugging Face Hub or from a saved directory.
    If the model is not saved locally, it will be downloaded and saved.

    Args:
        model_name (str): The Hugging Face model name or path.
        model_dir (str): The directory to save the model.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        device: The device (CPU or GPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(load_in_8bit=torch.cuda.is_available())
    
    # Check if model is saved locally, otherwise load it from Hugging Face
    if not os.path.exists(model_dir):
        print("Loading model from Hugging Face...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        # Save model to local directory
        model.save_pretrained(model_dir)
        print(f"Model saved at {model_dir}")
    else:
        print("Loading model from local directory...")
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer, device

if __name__ == "__main__":
    model, tokenizer, device = load_model()
