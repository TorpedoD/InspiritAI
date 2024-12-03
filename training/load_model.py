import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct", model_dir="llama_model", reload_model=False):
    """
    Loads the model and tokenizer from the Hugging Face Hub or from a saved directory.
    If the model is not saved locally, it will be downloaded and saved.
    Optionally, you can delete the local model and reload it from Hugging Face.

    Args:
        model_name (str): The Hugging Face model name or path.
        model_dir (str): The directory to save the model.
        reload_model (bool): If True, the model will be reloaded from Hugging Face even if it exists locally.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        device: The device (CPU or GPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(load_in_8bit=torch.cuda.is_available())
    
    # Check if the model needs to be reloaded or not
    if reload_model or not os.path.exists(model_dir):
        # Delete the local model directory if it exists
        if os.path.exists(model_dir):
            print(f"Deleting the existing model directory at {model_dir}...")
            for root, dirs, files in os.walk(model_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(model_dir)
        
        # Load model from Hugging Face and save it locally
        print("Loading model from Hugging Face...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.save_pretrained(model_dir)
        print(f"Model saved at {model_dir}")
    else:
        # Load model from local directory
        print("Loading model from local directory...")
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer, device

if __name__ == "__main__":
    model, tokenizer, device = load_model(reload_model=True)
