import llama

def generate_text(prompt, max_length=200, temperature=0.7, top_p=0.9):
    """
    Generate text completion using Llama 3.2-3B-Instruct.

    Args:
        prompt (str): The input prompt to the model.
        max_length (int): Maximum length of the generated text.
        temperature (float): Sampling temperature (lower is more deterministic).
        top_p (float): Top-p (nucleus) sampling parameter.

    Returns:
        str: The generated text.
    """
    # Load the model
    model = llama.LlamaModel.from_pretrained(
        model_path="/root/.llama/checkpoints/Llama3.2-3B-Instruct",
        model_id="Llama3.2-3B-Instruct"
    )

    # Generate text
    response = model.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p
    )

    return response["text"]


if __name__ == "__main__":
    # Define a prompt
    input_prompt = "Explain the benefits of machine learning in healthcare:"

    # Generate a completion
    completion = generate_text(prompt=input_prompt)

    print("Generated Text:")
    print(completion)
