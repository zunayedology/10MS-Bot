import pdfplumber
import os
from config import PDF_DIR, TEXT_DIR

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text

def save_text_to_file(text, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_pdfs():
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            grade = pdf_file.split("_")[0]  # e.g., "HSC26" for naming
            text = extract_text_from_pdf(pdf_path)
            text_path = os.path.join(TEXT_DIR, f"book_text_{grade}.txt")
            save_text_to_file(text, text_path)

if __name__ == "__main__":
    process_pdfs()