# document_loader.py
import os
import re
import pickle
from typing import List, Dict

import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# -------- CONFIG --------
PDF_PATH = "data/user_manual.pdf"          # your PDF
CHUNKS_PATH = "data/chunks.pkl"            # list[dict], each: {"text", "section", "page_start", "page_end"}
INDEX_PATH = "data/faiss_index.bin"        # FAISS L2 index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 200                          # ~200 words per chunk
OVERLAP_WORDS = 50                         # ~20% overlap
# ------------------------

HEADER_FOOTER_PAT = re.compile(
    r"(Ampd\s+Enertainer.*Rev\s*\d+\.\d+)|(^\s*Page\s*\d+\s*$)", re.IGNORECASE
)

HEADING_PAT = re.compile(r"^\s*\d+(?:\.\d+)*\s+[^\n]+", re.MULTILINE)  # e.g., 5, 5.2, 5.2.1 Title


def read_pdf_text(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        # normalize hyphen line-breaks and whitespace
        t = t.replace("-\n", "")          # join hyphenated breaks
        t = t.replace("\n", " ")
        t = HEADER_FOOTER_PAT.sub("", t)  # drop repeated headers/footers
        t = re.sub(r"\s{2,}", " ", t).strip()
        pages.append(t)
    return pages


def split_by_headings(full_text: str) -> List[Dict]:
    """Return sections with their heading title and span in text."""
    sections = []
    positions = [(m.start(), m.group(0).strip()) for m in HEADING_PAT.finditer(full_text)]
    if not positions:
        return [{"title": "Document", "text": full_text}]

    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        body = full_text[start:end].strip()
        sections.append({"title": title, "text": body})
    return sections


def chunk_section(text: str, title: str, page_span: tuple, words: int, overlap: int) -> List[Dict]:
    """Split section text into sentence-aware chunks with overlap."""
    sentences = sent_tokenize(text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        tokens = sent.split()
        if current_len + len(tokens) > words:
            piece = " ".join(current).strip()
            if len(piece.split()) >= 25:
                chunks.append({
                    "text": piece,
                    "section": title,
                    "page_start": page_span[0],
                    "page_end": page_span[1]
                })
            # apply overlap (reuse last N tokens for continuity)
            overlap_tokens = " ".join(current[-overlap:])
            current = overlap_tokens.split()
            current_len = len(current)

        current.append(sent)
        current_len += len(tokens)

    # last chunk
    if current:
        piece = " ".join(current).strip()
        if len(piece.split()) >= 25:
            chunks.append({
                "text": piece,
                "section": title,
                "page_start": page_span[0],
                "page_end": page_span[1]
            })

    return chunks


def map_pages_to_ranges(pages: List[str], full_text: str):
    """Roughly map character offsets back to page ranges for metadata."""
    ranges = []
    cursor = 0
    bounds = []
    for p in pages:
        length = len(p)
        bounds.append((cursor, cursor + length))
        cursor += length

    def page_of(pos: int) -> int:
        for i, (a, b) in enumerate(bounds):
            if a <= pos < b:
                return i + 1    # 1-based
        return len(bounds)

    return bounds, page_of


def build_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors.astype("float32"))
    return index


def main():
    os.makedirs("data", exist_ok=True)
    print("Reading PDF...")
    pages = read_pdf_text(PDF_PATH)
    full_text = " ".join(pages)

    print("Splitting by headings...")
    sections = split_by_headings(full_text)

    print("Mapping pages (approx.)...")
    bounds, page_of = map_pages_to_ranges(pages, full_text)

    print("Creating chunks...")
    all_chunks: List[Dict] = []
    offset = 0
    for sec in sections:
        # approximate page span using offsets
        start_pos = full_text.find(sec["text"], offset)
        if start_pos < 0:
            start_pos = offset
        end_pos = start_pos + len(sec["text"])
        offset = end_pos
        ps = page_of(start_pos)
        pe = page_of(end_pos - 1)

        # chunk the section with sentence alignment
        chunks = chunk_section(sec["text"], sec["title"], (ps, pe), CHUNK_WORDS, OVERLAP_WORDS)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("Embedding & building FAISS index...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    index = build_index(embs)
    faiss.write_index(index, INDEX_PATH)

    print(f"Saved: {INDEX_PATH} and {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
