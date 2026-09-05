import re
import faiss
import pickle
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# -----------------------------
# Setup
# -----------------------------

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the FAISS index
index = faiss.read_index("vector.index")

# Load the chunks + which PDF each one came from
with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    chunk_sources = data["sources"]

# Load the embedding model + reranker
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def search_documents(query, top_k=3, candidate_pool=10, relative_margin=6.0):
    """
    Retrieve a wider candidate pool with FAISS, then rerank with a
    cross-encoder. Instead of an absolute score cutoff (which doesn't
    generalize across different queries/topics), we keep only chunks
    within `relative_margin` points of the top-scoring chunk for THIS
    query. This adapts to each query's own score distribution.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, candidate_pool)

    candidates = []
    for i in indices[0]:
        i = int(i)
        if i < len(chunks):
            candidates.append({
                "chunk_id": i,
                "text": chunks[i],
                "source_pdf": chunk_sources[i]
            })

    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    top_score = reranked[0][1]
    filtered = [(c, s) for c, s in reranked if s >= top_score - relative_margin]

    return [c for c, s in filtered[:top_k]]


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

    # Build context with labeled sources so the model can reference them
    context = "\n\n".join(
        f"[Source {c['chunk_id']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""You are a helpful technical assistant. Answer the question using ONLY the context below.
When you cite sources, group them in a single bracket using plain ASCII brackets like [Source 3, 7] — never use full-width brackets 【 】.
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