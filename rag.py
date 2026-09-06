import re
import faiss
import pickle
import os
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq
from ingest import process_single_pdf

# -----------------------------
# Setup
# -----------------------------

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Used to get embeddings from Hugging Face's hosted API instead of
# loading the model locally — keeps memory usage low enough for
# Render's free tier (loading sentence-transformers/torch locally
# was causing out-of-memory crashes).
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load the FAISS index
index = faiss.read_index("vector.index")

# Load the chunks + which PDF each one came from
with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    chunk_sources = data["sources"]


def embed_texts(texts):
    """
    Get embeddings for a list of texts via Hugging Face's hosted
    Inference API in one batched call. Used for both querying (via
    embed_query) and indexing new uploaded chunks, so all vectors
    stay comparable in the same FAISS index.
    """
    embeddings = hf_client.feature_extraction(texts, model=EMBEDDING_MODEL)
    return np.array(embeddings, dtype="float32")


def embed_query(text):
    """
    Get the embedding for a single piece of text.
    """
    return embed_texts([text])



def search_documents(query, top_k=6, candidate_pool=20, distance_margin=0.35, source_filter=None):
    """
    Retrieve candidates via FAISS, optionally scoped to a single source
    PDF, then filter out results that are much farther from the query
    than the closest match. FAISS's L2 distance is free (already
    computed by the search) — this gives some quality filtering
    without needing a reranker model.
    """
    query_embedding = embed_query(query)
    distances, indices = index.search(query_embedding, candidate_pool)

    candidates = []
    for dist, i in zip(distances[0], indices[0]):
        i = int(i)
        if i >= len(chunks):
            continue
        if source_filter and chunk_sources[i] != source_filter:
            continue
        candidates.append({
            "chunk_id": i,
            "text": chunks[i],
            "source_pdf": chunk_sources[i],
            "distance": float(dist)
        })

    if not candidates:
        return []

    best_distance = candidates[0]["distance"]
    filtered = [c for c in candidates if c["distance"] <= best_distance + distance_margin]

    return filtered[:top_k]
     


def list_available_documents():
    """
    Return the sorted, deduplicated list of source PDFs currently
    indexed, so the UI can offer a document-scope filter.
    """
    return sorted(set(chunk_sources))

def clean_citations(text):
    """
    Merge adjacent citations like [Source 3][Source 7] into [Source 3, 7].
    """
    def merge(match):
        numbers = re.findall(r'\d+', match.group(0))
        return f"[Source {', '.join(numbers)}]"

    return re.sub(r'(\[Source \d+\])+', merge, text)


def generate_answer(question, context_chunks):
    """
    Send the question + retrieved chunks to Groq and get a final answer.
    context_chunks is a list of dicts: {"chunk_id": int, "text": str, "source_pdf": str}
    """

    context = "\n\n".join(
        f"[Source {c['chunk_id']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""You are a helpful technical assistant. Explain things in plain, simple language, as if teaching a student who is new to the topic. Avoid unnecessary jargon; when you must use a technical term, briefly explain it in everyday words.

Answer the question using ONLY the context below.

Citation format rules (follow exactly):
- Always write citations as [Source N] or [Source N, M] using plain square brackets and the word "Source".
- Never drop the word "Source". Never use full-width brackets 【 】.

If the answer isn't in the context, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    return clean_citations(answer)


def add_pdf_to_index(pdf_path, filename):
    """
    Process a newly uploaded PDF with the same pipeline as ingest.py,
    embed its chunks via the Hugging Face Inference API, and append
    them to the already-loaded FAISS index and chunk store, then
    persist both to disk.
    """
    new_chunks = process_single_pdf(pdf_path)

    if not new_chunks:
        return 0

    new_embeddings = embed_texts(new_chunks)
    index.add(new_embeddings)

    chunks.extend(new_chunks)
    chunk_sources.extend([filename] * len(new_chunks))

    faiss.write_index(index, "vector.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "sources": chunk_sources}, f)

    return len(new_chunks)


if __name__ == "__main__":

    print("=" * 60)
    print("TechDoc-AI — Ask questions about your PDFs")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        results = search_documents(question)

        if not results:
            print("\n⚠️  No relevant information found for that question.")
            continue

        print(f"\nRetrieved {len(results)} chunk(s):")
        for r in results:
            preview = r["text"][:100].replace("\n", " ")
            print(f"  [Source {r['chunk_id']}] ({r['source_pdf']}): {preview}...")

        print("\nGenerating answer...\n")
        answer = generate_answer(question, results)

        print("--- ANSWER ---")
        print(answer)