import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Define test questions
# -----------------------------
# IMPORTANT: pick questions you did NOT use to generate training_pairs.jsonl,
# ideally about content near the END of the documents (training_pairs.jsonl
# only used the first 150 chunks, so later content is a fair, unseen test).
# expected_source should be the PDF filename this question's answer comes from.

TEST_QUESTIONS = [
    # cuda.pdf
    {"question": "What is CUDA?", "expected_source": "cuda.pdf"},
    {"question": "What is the difference between a CUDA block and a grid?", "expected_source": "cuda.pdf"},
    {"question": "What is a CUDA kernel function?", "expected_source": "cuda.pdf"},
    {"question": "What is shared memory in CUDA?", "expected_source": "cuda.pdf"},
    {"question": "What is a warp in CUDA programming?", "expected_source": "cuda.pdf"},
    {"question": "What is the role of a streaming multiprocessor?", "expected_source": "cuda.pdf"},
    {"question": "How does CUDA manage thread synchronization?", "expected_source": "cuda.pdf"},
    {"question": "What is global memory in CUDA architecture?", "expected_source": "cuda.pdf"},

    # gpu.pdf
    {"question": "What is a GPU used for?", "expected_source": "gpu.pdf"},
    {"question": "What is the difference between a GPU and a CPU?", "expected_source": "gpu.pdf"},
    {"question": "What are tensor cores?", "expected_source": "gpu.pdf"},
    {"question": "Why are GPUs suited for parallel processing?", "expected_source": "gpu.pdf"},
    {"question": "What is VRAM in a GPU?", "expected_source": "gpu.pdf"},
    {"question": "How is a GPU used in deep learning training?", "expected_source": "gpu.pdf"},
    {"question": "What is the difference between integrated and discrete GPUs?", "expected_source": "gpu.pdf"},

    # machine_learning.pdf (confirm exact filename)
    {"question": "What is the difference between supervised and unsupervised learning?", "expected_source": "machine_learning.pdf"},
    {"question": "What is overfitting in machine learning?", "expected_source": "machine_learning.pdf"},
    {"question": "What is the bias-variance tradeoff?", "expected_source": "machine_learning.pdf"},
    {"question": "What is gradient descent?", "expected_source": "machine_learning.pdf"},
    {"question": "What is regularization used for in machine learning?", "expected_source": "machine_learning.pdf"},
    {"question": "What is a confusion matrix?", "expected_source": "machine_learning.pdf"},
    {"question": "What is the purpose of a validation set?", "expected_source": "machine_learning.pdf"},
    {"question": "What is feature engineering?", "expected_source": "machine_learning.pdf"},

    # DBMS_Notes.pdf
    {"question": "What is normalization in DBMS?", "expected_source": "DBMS_Notes.pdf"},
    {"question": "What are the ACID properties in a database transaction?", "expected_source": "DBMS_Notes.pdf"},
    {"question": "What is the difference between a primary key and a foreign key?", "expected_source": "DBMS_Notes.pdf"},
    {"question": "What is the difference between 2NF and 3NF?", "expected_source": "DBMS_Notes.pdf"},
    {"question": "What is a functional dependency in DBMS?", "expected_source": "DBMS_Notes.pdf"},
    {"question": "What is indexing used for in a database?", "expected_source": "DBMS_Notes.pdf"},
    {"question": "What is the difference between an inner join and an outer join?", "expected_source": "DBMS_Notes.pdf"},
    {"question": "What is concurrency control in DBMS?", "expected_source": "DBMS_Notes.pdf"},
]

TOP_K = 3  # counts as a "hit" if the correct source appears in the top 3 results


# -----------------------------
# 2. Load chunks and both models
# -----------------------------

with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    sources = data["sources"]

print("Loading base model...")
base_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading fine-tuned model...")
finetuned_model = SentenceTransformer("finetuned-embedding-model")


# -----------------------------
# 3. Build embeddings for all chunks, with BOTH models
# -----------------------------

print("Embedding all chunks with both models (one-time cost)...")
base_chunk_embeddings = base_model.encode(chunks, show_progress_bar=True)
finetuned_chunk_embeddings = finetuned_model.encode(chunks, show_progress_bar=True)


def top_k_sources(query_embedding, chunk_embeddings, k):
    """Return the source filenames of the k closest chunks by cosine similarity."""
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    normalized = chunk_embeddings / norms
    scores = normalized @ query_embedding
    top_indices = np.argsort(scores)[::-1][:k]
    return [sources[i] for i in top_indices]


# -----------------------------
# 4. Run the comparison
# -----------------------------

base_hits = 0
finetuned_hits = 0

print(f"\n{'Question':<50} {'Base hit?':<12} {'Fine-tuned hit?':<15}")
print("-" * 80)

for item in TEST_QUESTIONS:
    question = item["question"]
    expected = item["expected_source"]

    base_q_embedding = base_model.encode([question])[0]
    finetuned_q_embedding = finetuned_model.encode([question])[0]

    base_top = top_k_sources(base_q_embedding, base_chunk_embeddings, TOP_K)
    finetuned_top = top_k_sources(finetuned_q_embedding, finetuned_chunk_embeddings, TOP_K)

    base_hit = expected in base_top
    finetuned_hit = expected in finetuned_top

    base_hits += base_hit
    finetuned_hits += finetuned_hit

    print(f"{question[:48]:<50} {str(base_hit):<12} {str(finetuned_hit):<15}")

total = len(TEST_QUESTIONS)
print("-" * 80)
print(f"\nBase model:       {base_hits}/{total} correct ({100*base_hits/total:.0f}%)")
print(f"Fine-tuned model: {finetuned_hits}/{total} correct ({100*finetuned_hits/total:.0f}%)")