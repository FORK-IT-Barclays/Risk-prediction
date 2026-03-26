import sys
import os

try:
    import PyPDF2
except ImportError:
    os.system("pip install PyPDF2")
    import PyPDF2

def read_pdf(filepath):
    try:
        reader = PyPDF2.PdfReader(filepath)
        text = []
        for i, p in enumerate(reader.pages):
            text.append(f"--- PAGE {i+1} ---")
            text.append(p.extract_text())
        print("\n".join(text))
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    read_pdf(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "barclayspptx.pdf"))
