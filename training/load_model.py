import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def load_model(model_name="meta-llama/Llama-2-7b-chat-hf", model_dir="llama2_model", reload_model=False):
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
    # Set device to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device set to {device}.")

    # Initialize BitsAndBytes config for 8-bit quantization if CUDA is available
    bnb_config = BitsAndBytesConfig(load_in_8bit=torch.cuda.is_available())

    # Check if the model needs to be reloaded
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
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else None,  # Automatically handle GPU if available
                quantization_config=bnb_config if device == "cuda" else None,  # Use 8-bit if on GPU
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Set dtype for GPU or CPU
            )
            model.save_pretrained(model_dir)
            print(f"Model saved at {model_dir}")
            
            # Save the tokenizer along with the model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            print(f"Tokenizer saved at {model_dir}")
            
        except Exception as e:
            print(f"Error loading model from Hugging Face: {e}")
            return None, None, None
    else:
        # Load model from local directory
        print("Loading model from local directory...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir)
        except Exception as e:
            print(f"Error loading model from local directory: {e}")
            return None, None, None

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None, None

    # No need to move model to GPU manually as it's already handled by device_map
    print(f"Model is already on {device}.")
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer, device

if __name__ == "__main__":
    model, tokenizer, device = load_model(reload_model=True)
    if model is None:
        print("Failed to load the model. Exiting.")
