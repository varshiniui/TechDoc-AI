import fitz
import faiss
import pickle
import os
import re
from collections import Counter
from sentence_transformers import SentenceTransformer


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text, max_chars=1000, overlap_sentences=2):
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def deduplicate_chunks(chunks, similarity_threshold=0.75):
    """
    Drop chunks that are near-duplicates of a chunk already kept.
    Uses containment ratio (overlap / smaller chunk's word count).
    """
    kept = []
    for chunk in chunks:
        words = set(chunk.lower().split())
        is_duplicate = False

        for existing in kept:
            existing_words = set(existing.lower().split())
            if not words or not existing_words:
                continue
            overlap = len(words & existing_words)
            smaller = min(len(words), len(existing_words))
            containment = overlap / smaller
            if containment > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(chunk)

    return kept


def extract_pages(pdf_path):
    """Return a list of raw page texts."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def strip_running_headers_footers(pages, min_page_fraction=0.3):
    """
    Detect lines that repeat across many pages (running headers, footers,
    page numbers, module labels) and remove them from every page before
    chunking. A line that appears on a large fraction of pages is
    structural boilerplate, not real content.
    """
    line_counts = Counter()

    for page_text in pages:
        # Use a set so a line repeated twice on the same page only counts once
        lines_on_page = set(line.strip() for line in page_text.split("\n") if line.strip())
        for line in lines_on_page:
            line_counts[line] += 1

    total_pages = len(pages)
    threshold = max(2, int(total_pages * min_page_fraction))

    boilerplate_lines = {
        line for line, count in line_counts.items()
        if count >= threshold
    }

    cleaned_pages = []
    for page_text in pages:
        kept_lines = [
            line for line in page_text.split("\n")
            if line.strip() not in boilerplate_lines
        ]
        cleaned_pages.append("\n".join(kept_lines))

    return cleaned_pages, boilerplate_lines


def process_single_pdf(pdf_path):
    """
    Run one PDF through the same extraction -> header/footer stripping
    -> chunking -> dedup pipeline used for full ingestion, and return
    just its chunks. Used both by full ingest and single-file uploads.
    """
    pages = extract_pages(pdf_path)
    cleaned_pages, removed_lines = strip_running_headers_footers(pages)
    full_text = "\n".join(cleaned_pages)

    file_chunks = chunk_text(full_text, max_chars=1000, overlap_sentences=2)
    file_chunks = deduplicate_chunks(file_chunks)

    return file_chunks


def build_index_from_data_dir(data_dir="data"):
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    all_chunks = []
    chunk_sources = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        file_chunks = process_single_pdf(pdf_path)

        print(f"  {pdf_file}: {len(file_chunks)} chunks after dedup")

        all_chunks.extend(file_chunks)
        chunk_sources.extend([pdf_file] * len(file_chunks))

    print("\nTotal chunks across all PDFs:", len(all_chunks))

    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(all_chunks, show_progress_bar=True)

    print("Embeddings created!")
    print("Embedding shape:", embeddings.shape)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("FAISS index created!")
    print("Vectors stored:", index.ntotal)

    faiss.write_index(index, "vector.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": chunk_sources}, f)

    print("\nSaved:")
    print("- vector.index")
    print("- chunks.pkl")


if __name__ == "__main__":
    build_index_from_data_dir()