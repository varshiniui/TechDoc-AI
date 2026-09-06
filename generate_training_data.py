import pickle
import json
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 1. Load your existing chunks
# -----------------------------

with open("chunks.pkl", "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    sources = data["sources"]

print(f"Loaded {len(chunks)} chunks.")

# Optional: cap how many pairs you generate, to control time/API usage.
# Start small to test the pipeline works, then increase.
MAX_CHUNKS = 150  # adjust up later if this works well and you want more data

if len(chunks) > MAX_CHUNKS:
    print(f"Using a sample of {MAX_CHUNKS} chunks (out of {len(chunks)}).")
    chunks = chunks[:MAX_CHUNKS]
    sources = sources[:MAX_CHUNKS]


# -----------------------------
# 2. Generate one question per chunk
# -----------------------------

def generate_question(chunk_text):
    prompt = f"""Read this text and write ONE natural question that this text directly answers.
The question should sound like something a real user would type into a search box.
Respond with ONLY the question, nothing else — no quotes, no numbering, no explanation.

Text:
{chunk_text}

Question:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


pairs = []

for i, chunk in enumerate(chunks):
    try:
        question = generate_question(chunk)
        pairs.append({
            "question": question,
            "chunk": chunk,
            "source": sources[i],
        })
        print(f"[{i+1}/{len(chunks)}] {question}")
    except Exception as e:
        print(f"[{i+1}/{len(chunks)}] Skipped due to error: {e}")

    # Small delay to stay comfortably within Groq's free-tier rate limits
    time.sleep(0.5)


# -----------------------------
# 3. Save as JSONL
# -----------------------------

with open("training_pairs.jsonl", "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"\nSaved {len(pairs)} training pairs to training_pairs.jsonl")