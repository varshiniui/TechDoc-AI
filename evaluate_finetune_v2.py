"""
Improved evaluation: measures the RANK of the correct chunk, not just whether
the correct PDF appears in the top-3. This avoids the ceiling effect of the
previous test (4 very distinct topics made source-level discrimination too
easy for both models to show a difference).

For each question, you now specify the exact expected_chunk_id (from
chunks.pkl) instead of just a source filename. The script finds where that
specific chunk ranks in the full similarity-sorted list, for both models.

Lower average rank = better. This gives a continuous, sensitive signal
instead of a saturated True/False one.
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load chunks
# -----------------------------

with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    sources = data["sources"]

# -----------------------------
# 2. Find candidate chunk_ids for your test questions
# -----------------------------
# Run this first to print a few chunks per source, so you can pick the
# EXACT chunk_id that answers each of your test questions.

PRINT_PREVIEW = False

if PRINT_PREVIEW:
    print("=== Chunk previews by source (use these chunk_ids below) ===\n")
    seen_sources = set()
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        if source not in seen_sources:
            seen_sources.add(source)
        preview = chunk[:120].replace("\n", " ")
        print(f"[{i}] ({source}) {preview}...")
    print("\n=== Copy the chunk_ids that match your test questions below, then set PRINT_PREVIEW = False and rerun ===")
    exit()

# -----------------------------
# 3. Test questions with EXACT expected chunk_id
# -----------------------------
# Fill these in after reading the preview above. Pick chunks from LATER
# in each document where possible (fairer test of generalization).

TEST_QUESTIONS = [
    # {"question": "What is a warp in CUDA programming?", "expected_chunk_id": 12},
    # {"question": "What is the bias-variance tradeoff?", "expected_chunk_id": 47},
    # ... fill in ~15-20 of these
]

# -----------------------------
# 4. Load both models
# -----------------------------

print("Loading base model...")
base_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading fine-tuned model...")
finetuned_model = SentenceTransformer("finetuned-embedding-model")

print("Embedding all chunks with both models...")
base_chunk_embeddings = base_model.encode(chunks, show_progress_bar=True)
finetuned_chunk_embeddings = finetuned_model.encode(chunks, show_progress_bar=True)


def rank_of_chunk(query_embedding, chunk_embeddings, target_chunk_id):
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    normalized = chunk_embeddings / norms
    scores = normalized @ query_embedding
    ranked_indices = np.argsort(scores)[::-1]
    rank = int(np.where(ranked_indices == target_chunk_id)[0][0]) + 1  # 1-indexed
    return rank


# -----------------------------
# 5. Run comparison
# -----------------------------

base_ranks = []
finetuned_ranks = []

print(f"\n{'Question':<50} {'Base rank':<12} {'Fine-tuned rank':<15}")
print("-" * 80)

for item in TEST_QUESTIONS:
    question = item["question"]
    target_id = item["expected_chunk_id"]

    base_q_embedding = base_model.encode([question])[0]
    finetuned_q_embedding = finetuned_model.encode([question])[0]

    base_rank = rank_of_chunk(base_q_embedding, base_chunk_embeddings, target_id)
    finetuned_rank = rank_of_chunk(finetuned_q_embedding, finetuned_chunk_embeddings, target_id)

    base_ranks.append(base_rank)
    finetuned_ranks.append(finetuned_rank)

    print(f"{question[:48]:<50} {base_rank:<12} {finetuned_rank:<15}")

print("-" * 80)
print(f"\nAverage rank — Base model:       {np.mean(base_ranks):.2f}")
print(f"Average rank — Fine-tuned model: {np.mean(finetuned_ranks):.2f}")
print("(Lower is better — 1 means the correct chunk was ranked first)")

# Mean Reciprocal Rank (MRR) — standard IR metric, also more sensitive than raw rank
base_mrr = np.mean([1 / r for r in base_ranks])
finetuned_mrr = np.mean([1 / r for r in finetuned_ranks])
print(f"\nMRR — Base model:       {base_mrr:.3f}")
print(f"MRR — Fine-tuned model: {finetuned_mrr:.3f}")
print("(Higher is better — 1.0 is a perfect score)")