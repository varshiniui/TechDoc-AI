import pickle
import json
import os
import time
import random
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    sources = data["sources"]

TRAIN_CUTOFF = 150   # must match MAX_CHUNKS in generate_training_data.py
NUM_EVAL_QUESTIONS = 20

held_out_indices = list(range(TRAIN_CUTOFF, len(chunks)))

if len(held_out_indices) < NUM_EVAL_QUESTIONS:
    print(f"Only {len(held_out_indices)} held-out chunks available — using all of them.")
    sample_indices = held_out_indices
else:
    random.seed(42)
    sample_indices = random.sample(held_out_indices, NUM_EVAL_QUESTIONS)


def generate_question(chunk_text):
    prompt = f"""Read this text and write ONE natural question that this text directly answers.
Respond with ONLY the question, nothing else.

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

for idx in sample_indices:
    chunk = chunks[idx]
    try:
        question = generate_question(chunk)
        eval_pairs.append({
            "question": question,
            "expected_chunk_id": idx,
            "source": sources[idx],
        })
        print(f"[chunk {idx}] {question}")
    except Exception as e:
        print(f"[chunk {idx}] Skipped due to error: {e}")
    time.sleep(0.5)

with open("eval_pairs.jsonl", "w", encoding="utf-8") as f:
    for pair in eval_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"\nSaved {len(eval_pairs)} eval pairs to eval_pairs.jsonl — no manual matching needed.")