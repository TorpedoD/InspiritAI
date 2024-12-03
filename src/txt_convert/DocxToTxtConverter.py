from docx_to_txt import convert_docx_to_txt

class DocxToTxtConverter:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def convert(self):
        convert_docx_to_txt(self.input_folder, self.output_folder)

def main():
    input_folder = '/content/Input/Docx'  # Use Colab's file system path
    output_folder = '/content/Output'  # Use Colab's file system path
    
    # Create an instance of the DocxToTxtConverter class
    converter = DocxToTxtConverter(input_folder, output_folder)
    
    # Call the convert method to perform the conversion
    converter.convert()

if __name__ == '__main__':
    main()
