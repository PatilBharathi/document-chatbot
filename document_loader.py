import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ---------- CONFIG ----------
PDF_PATH = "data/user_manual.pdf"
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 350   # smaller chunks for better recall
CHUNK_OVERLAP = 50
# ----------------------------

def load_pdf(pdf_path):
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide with overlap
    return chunks

def build_faiss_index(chunks, model):
    """Create FAISS index from text chunks."""
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main():
    print("Loading PDF...")
    text = load_pdf(PDF_PATH)

    print("Splitting into chunks...")
    chunks = chunk_text(text)

    print("Generating embeddings...")
    model = SentenceTransformer(EMBED_MODEL)
    index = build_faiss_index(chunks, model)

    # Save FAISS index and chunks
    print("Saving FAISS index and chunks...")
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done! Created {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
