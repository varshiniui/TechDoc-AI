"""
Finds terms that appear across MULTIPLE different source PDFs (the exact
condition that caused your real "normalization" ambiguity bug), then
auto-generates a targeted, specific question for one representative chunk
per source per term. Unlike the previous eval, this deliberately tests the
failure mode you already found in production, instead of easy single-topic
questions.

Output: ambiguous_eval_pairs.jsonl — same format as before, ready to
evaluate with evaluate_finetune_v3.py (just point it at this file).
"""

import pickle
import json
import os
import re
import time
from collections import defaultdict, Counter
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "it", "its", "we", "you", "they", "he", "she", "his", "her",
    "their", "our", "not", "can", "will", "would", "should", "could",
    "may", "might", "must", "have", "has", "had", "do", "does", "did",
    "which", "what", "when", "where", "how", "why", "who", "also", "into",
    "such", "each", "other", "some", "more", "most", "so", "no", "there",
}

# -----------------------------
# 1. Load chunks
# -----------------------------

with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    sources = data["sources"]

TRAIN_CUTOFF = 150  # avoid reusing chunks the model already trained on

# -----------------------------
# 2. Find words that appear across MULTIPLE distinct source files
# -----------------------------

word_sources = defaultdict(set)          # word -> set of source files it appears in
word_best_chunk = {}                     # (word, source) -> chunk_id with highest count of that word

for i, (chunk, source) in enumerate(zip(chunks, sources)):
    if i < TRAIN_CUTOFF:
        continue  # keep this a held-out test too
    words = re.findall(r"[a-z]{4,}", chunk.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    for word, count in counts.items():
        word_sources[word].add(source)
        key = (word, source)
        if key not in word_best_chunk or count > word_best_chunk[key][1]:
            word_best_chunk[key] = (i, count)

# Keep only words appearing in 2+ distinct source files — these are the
# ambiguous, cross-document terms that could confuse retrieval.
shared_terms = {
    word: sources_set
    for word, sources_set in word_sources.items()
    if len(sources_set) >= 2
}

print(f"Found {len(shared_terms)} term(s) shared across multiple documents.")

# Take the top 10 by how many chunks they appear in (roughly, by frequency)
sorted_terms = sorted(shared_terms.items(), key=lambda x: -len(x[1]))[:10]

for word, srcs in sorted_terms:
    print(f"  '{word}' appears in: {', '.join(srcs)}")

# -----------------------------
# 3. Generate a specific, disambiguating question per (term, source)
# -----------------------------


def generate_specific_question(chunk_text, shared_word):
    prompt = f"""Read this text. Write ONE specific question that this text answers,
that MUST include the exact word "{shared_word}" and enough surrounding context
(mention the subject/domain) that someone could only answer it correctly using
THIS text, even if the word "{shared_word}" also appears in a completely
different document on a different topic.
Respond with ONLY the question.

Text:
{chunk_text}

Question:"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


eval_pairs = []

for word, srcs in sorted_terms:
    for source in srcs:
        chunk_id, count = word_best_chunk[(word, source)]
        chunk_text = chunks[chunk_id]
        try:
            question = generate_specific_question(chunk_text, word)
            eval_pairs.append({
                "question": question,
                "expected_chunk_id": chunk_id,
                "source": source,
                "shared_term": word,
            })
            print(f"[{word} / {source} / chunk {chunk_id}] {question}")
        except Exception as e:
            print(f"[{word} / {source}] Skipped: {e}")
        time.sleep(0.5)

with open("ambiguous_eval_pairs.jsonl", "w", encoding="utf-8") as f:
    for pair in eval_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"\nSaved {len(eval_pairs)} targeted eval pairs to ambiguous_eval_pairs.jsonl")