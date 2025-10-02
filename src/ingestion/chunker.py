import os
import re
import json
from typing import List, Dict
import pdfplumber
from transformers import AutoTokenizer

# Load tokenizer (match your embedding model)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Common keywords often used as section titles in policies
SECTION_KEYWORDS = [
    "coverage", "exclusion", "claim", "definition", 
    "eligibility", "benefit", "policy", "general conditions"
]

def is_probable_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if len(line.split()) <= 12 and not line.endswith("."):
        return True
    if re.match(r"^(\d+[\.\)]|[IVX]+\.)", line):
        return True
    if line.isupper():
        return True
    if line.istitle() and len(line.split()) <= 8:
        return True
    if any(kw in line.lower() for kw in SECTION_KEYWORDS):
        return True
    return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber"""
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_content.append(page.extract_text() or "")
    return "\n".join(text_content)


def chunk_document(text: str, doc_name: str,
                   max_tokens: int = 512, overlap: float = 0.15) -> List[Dict]:

    lines = text.split("\n")
    paragraphs = []
    current_section = "General"
    buffer = []

    # Group paragraphs under headings
    for line in lines:
        if is_probable_heading(line):
            if buffer:
                paragraphs.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            current_section = line.strip()
        elif line.strip() == "":
            if buffer:
                paragraphs.append((current_section, "\n".join(buffer).strip()))
                buffer = []
        else:
            buffer.append(line.strip())
    if buffer:
        paragraphs.append((current_section, "\n".join(buffer).strip()))

    # Tokenize and chunk with overlap
    chunks = []
    chunk_id = 0
    step_size = int(max_tokens * (1 - overlap))

    for section_heading, para in paragraphs:
        tokens = tokenizer.encode(para, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            chunks.append({
                "chunk_id": f"{doc_name}_{chunk_id}",
                "text": para,
                "metadata": {
                    "document_name": doc_name,
                    "section_heading": section_heading
                }
            })
            chunk_id += 1
        else:
            for i in range(0, len(tokens), step_size):
                window = tokens[i:i + max_tokens]
                if not window:
                    continue
                sub_text = tokenizer.decode(window)
                chunks.append({
                    "chunk_id": f"{doc_name}_{chunk_id}",
                    "text": sub_text,
                    "metadata": {
                        "document_name": doc_name,
                        "section_heading": section_heading
                    }
                })
                chunk_id += 1

    return chunks



def chunk_pdfs(input_folder: str, output_file: str = None) -> List[Dict]:
    """
    Process all PDFs in a folder, chunk them, and return a list of chunks.
    Optionally save them to JSON.
    """
    all_chunks = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(input_folder, file_name)
            print(f"📄 Extracting text from {file_name}...")
            text = extract_text_from_pdf(file_path)
            
            chunks = chunk_document(text, doc_name=file_name)
            all_chunks.extend(chunks)
            print(f"✅ Processed {file_name} → {len(chunks)} chunks")

    # Optional: save all chunks to a JSON file
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"📂 Saved all chunks to {output_file}")

    return all_chunks


# # -------------------
# # Example usage
# # -------------------
# if __name__ == "__main__":
#     input_folder = "data/pdfs"    # folder containing your PDFs
#     all_chunks = chunk_pdfs(input_folder, output_file="data/processed/chunks.json")

#     print(f"🔥 Total chunks created: {len(all_chunks)}")
