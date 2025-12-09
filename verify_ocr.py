import os
from PyPDF2 import PdfReader

def verify_pdf_extraction(pdf_path):
    print(f"--- Verifying OCR (Text Extraction) for: {pdf_path} ---")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    try:
        reader = PdfReader(pdf_path)
        number_of_pages = len(reader.pages)
        print(f"Number of Pages: {number_of_pages}")
        
        full_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += text
                print(f"\n[Page {i+1} Concent Start]...")
                print(text[:300] + "..." if len(text) > 300 else text)
                print(f"...[Page {i+1} Content End]\n")
            else:
                print(f"[Page {i+1}] No text could be extracted. It might be a scanned image.")
        
        if len(full_text.strip()) == 0:
            print("\nFAILURE: No text extracted from the entire document.")
            print("The PDF might be a scanned image (requires true OCR like Tesseract) or empty.")
        else:
            print(f"\nSUCCESS: Extracted {len(full_text)} characters.")

    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    # Test with one of the known files
    # Correct path based on exploration: "BillExtraction/Bills/Alzheimer.pdf"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(base_dir, "BillExtraction", "Bills", "Alzheimer.pdf")
    
    verify_pdf_extraction(test_file)
