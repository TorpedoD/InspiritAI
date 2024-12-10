
# InspiritAI Project

This project includes scripts and dependencies required for training and deploying AI models. Below is an overview of the commands executed and functionalities involved:

## Key Commands

1. **Cloning Repository**:
   ```
   git clone https://github.com/TorpedoD/InspiritAI.git
   cd InspiritAI
   ```

2. **Installing Dependencies**:
   - System-level installations:
     ```
     apt-get update
     apt-get install -y libnvinfer8 libnvinfer-dev libnvinfer-plugin8
     ```
   - Python packages:
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     pip install transformers bitsandbytes --extra-index-url https://huggingface.co/transformers/bitsandbytes/
     pip install -r requirements.txt
     pip install datasets
     ```

3. **Model Preprocessing and Training**:
   - Preprocessing data:
     ```
     python3 training/preprocess.py
     ```
   - Loading the model:
     ```
     python3 training/load_model.py
     ```
   - Training the LLaMA model:
     ```
     python3 training/train_llama.py
     ```

## Dependencies

The project utilizes the following key libraries:
- PyTorch
- Hugging Face Transformers
- NVIDIA TensorRT libraries
- Datasets for handling data
- Bitsandbytes for efficient model training

## Hardware Requirements

- GPU: T4 or higher
- CUDA-enabled environment
- Minimum 8 GB GPU memory

## Notes

- Ensure you have a Hugging Face token for authentication during model training.
- Make sure all required dependencies are installed before running the scripts.
- The project assumes a CUDA-enabled environment for optimal performance.

## Troubleshooting

- If you encounter dependency issues, use `pip` to reinstall the conflicting packages.
- For hardware-related errors, ensure that your GPU drivers and CUDA toolkit are up to date.

