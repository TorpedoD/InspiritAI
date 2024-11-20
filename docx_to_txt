import os
from docx import Document

def convert_docx_to_txt(input_folder, output_folder):
    """
    Converts all .docx files in the input folder to .txt files and saves them in the output folder.
    
    Args:
        input_folder (str): Path to the folder containing .docx files.
        output_folder (str): Path to the folder where .txt files will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".docx"):
            file_path = os.path.join(input_folder, filename)
            output_file = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_file)

            try:
                # Extract text from .docx
                doc = Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

                # Write the extracted text to a .txt file
                with open(output_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(text)
                
                print(f"Converted: {filename} -> {output_file}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
