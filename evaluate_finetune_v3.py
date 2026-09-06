"""
Final evaluation step — no manual work. Loads eval_pairs.jsonl (auto-generated
by generate_eval_set.py, with chunk_ids already known) and compares the base
vs fine-tuned model on how well each ranks the correct chunk.
"""

import pickle
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Pass a filename to evaluate a different eval set, e.g.:
#   python evaluate_finetune_v3.py ambiguous_eval_pairs.jsonl
eval_file = sys.argv[1] if len(sys.argv) > 1 else "eval_pairs.jsonl"

with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]

eval_pairs = []
with open(eval_file, "r", encoding="utf-8") as f:
    for line in f:
        eval_pairs.append(json.loads(line))

print(f"Evaluating on: {eval_file}")

print(f"Loaded {len(eval_pairs)} eval questions.\n")

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
    rank = int(np.where(ranked_indices == target_chunk_id)[0][0]) + 1
    return rank


base_ranks = []
finetuned_ranks = []

print(f"\n{'Question':<55} {'Base rank':<12} {'Fine-tuned rank':<15}")
print("-" * 85)

for item in eval_pairs:
    question = item["question"]
    target_id = item["expected_chunk_id"]

    base_q_embedding = base_model.encode([question])[0]
    finetuned_q_embedding = finetuned_model.encode([question])[0]

    base_rank = rank_of_chunk(base_q_embedding, base_chunk_embeddings, target_id)
    finetuned_rank = rank_of_chunk(finetuned_q_embedding, finetuned_chunk_embeddings, target_id)

    base_ranks.append(base_rank)
    finetuned_ranks.append(finetuned_rank)

    print(f"{question[:53]:<55} {base_rank:<12} {finetuned_rank:<15}")

print("-" * 85)
print(f"\nAverage rank — Base model:       {np.mean(base_ranks):.2f}")
print(f"Average rank — Fine-tuned model: {np.mean(finetuned_ranks):.2f}")
print("(Lower is better — 1 means the correct chunk was ranked first)")

base_mrr = np.mean([1 / r for r in base_ranks])
finetuned_mrr = np.mean([1 / r for r in finetuned_ranks])
print(f"\nMRR — Base model:       {base_mrr:.3f}")
print(f"MRR — Fine-tuned model: {finetuned_mrr:.3f}")
print("(Higher is better — 1.0 is a perfect score)")