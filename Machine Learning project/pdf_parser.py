import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extracts the raw text from the current page
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n" # Add extra newlines for page separation
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    return full_text

# Example Usage
pdf_file_path = "Course_document.pdf"  # Replace with your PDF file path
raw_text = extract_text_from_pdf(pdf_file_path)

if raw_text:
    save_path = "extracted_text.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"Extracted text saved to {save_path}")