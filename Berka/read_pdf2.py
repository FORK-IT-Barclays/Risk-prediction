import sys
import os

try:
    import fitz # PyMuPDF
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install pymupdf")
    import fitz

def read_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        text = ""
        for i, page in enumerate(doc):
            text += f"\n--- PAGE {i+1} ---\n"
            text += page.get_text()
        
        out_path = "presentation_text.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"SUCCESS: Extracted {len(text)} characters to {out_path}")
    except Exception as e:
        print(f"Failed to extract: {e}")

if __name__ == "__main__":
    read_pdf(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "barclaysppt.pdf"))
