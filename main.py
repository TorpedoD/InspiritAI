from docx_to_txt import convert_docx_to_txt

def main():
    input_folder = '/Users/damianwong/Desktop/Input'  # Use Colab's file system path
    output_folder = '/Users/damianwong/Desktop/Output'  # Use Colab's file system path
    
    convert_docx_to_txt(input_folder, output_folder)

if __name__ == '__main__':
  main()
