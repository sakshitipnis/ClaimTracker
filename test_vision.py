import os
from main_BUPA import get_pdf_text_via_vision

def test_vision():
    # Use the known path for Alzheimer.pdf
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_pdf = os.path.join(base_dir, "BillExtraction", "Bills", "Alzheimer.pdf")
    
    if not os.path.exists(sample_pdf):
        print(f"Error: Sample file not found at {sample_pdf}")
        return

    print(f"--- Testing Vision OCR on {sample_pdf} ---")
    print("Reading file bytes...")
    with open(sample_pdf, "rb") as f:
        file_bytes = f.read()

    print(f"Sending {len(file_bytes)} bytes to Vision OCR (PyMuPDF + GPT-4o)...")
    try:
        text = get_pdf_text_via_vision(file_bytes)
        print("\n--- EXTRACTED TEXT START ---")
        print(text[:500] + "..." if len(text) > 500 else text)
        print("--- EXTRACTED TEXT END ---\n")
        
        if len(text.strip()) > 10 and "Failed" not in text:
            print("SUCCESS: Vision OCR returned text.")
        else:
            print("FAILURE: Vision OCR returned empty or error.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_vision()
