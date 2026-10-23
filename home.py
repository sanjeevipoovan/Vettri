import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extracts and prints text from all pages of a given PDF file.
    """
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return
    
    try:
        # Create a reader object
        reader = PdfReader(pdf_path)
        
        # Get total number of pages
        num_pages = len(reader.pages)
        print(f"Successfully opened '{pdf_path}'. Total pages: {num_pages}\n")
        
        full_text = []
        
        # Loop through all the pages and extract text
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"--- Page {index + 1} ---\n{text}\n")
            else:
                full_text.append(f"--- Page {index + 1} ---\n[No text found on this page]\n")
                
        # Join all text together
        complete_text = "\n".join(full_text)
        
        # Print the extracted text (or save it to a file)
        print(complete_text)
        
        # Optional: Save the extracted text to a .txt file
        output_txt_path = "extracted1_output.txt"
        with open(output_txt_path, "w", encoding="utf-8") as text_file:
            text_file.write(complete_text)
        print(f"\n[Success] Extracted text has also been saved to '{output_txt_path}'.")

    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")

if __name__ == "__main__":
    # Replace with the path to your PDF file
    pdf_file_path = r"E:\Sanjeevi_kumar.pdf"
    
    extract_text_from_pdf(pdf_file_path)