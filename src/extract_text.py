import pdfplumber
import os
from config import PDF_DIR, TEXT_DIR

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            return all_text
    except Exception as e:
        print(f"Error extracting {pdf_path}: {str(e)}")
        return ""

def save_text_to_file(text, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_pdfs():
    os.makedirs(TEXT_DIR, exist_ok=True)
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            if text:
                text_path = os.path.join(TEXT_DIR, "book.txt")
                save_text_to_file(text, text_path)
                print(f"Extracted {pdf_file} to {text_path}")

if __name__ == "__main__":
    process_pdfs()
