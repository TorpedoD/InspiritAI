from docx_to_txt import convert_docx_to_txt

def main():
    input_folder = r"/Users/damianwong/Desktop/Input"  # Replace with the path to the folder containing .docx files
    output_folder = r"/Users/damianwong/Desktop/Output"  # Replace with the path to save .txt files
    
    convert_docx_to_txt(input_folder, output_folder)

if __name__ == '__main__':
  main()
