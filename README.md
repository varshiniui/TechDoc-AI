# TechDoc-AI

Ask your PDFs a question. Get an answer grounded in the actual text with a citation pointing back to exactly which document and chunk it came from.

**Live demo:** [techdoc-ai-nuie.onrender.com](https://techdoc-ai-nuie.onrender.com/)
*(free-tier hosting the first request after a period of inactivity takes 20-40s to wake up)*

---

## What this actually is

Most "chat with your PDF" tutorials stop at "embed it, search it, ask an LLM." This one didn't stop there — it went to production, broke in the ways real systems break, and got fixed in ways that are worth writing down: an out-of-memory crash from a naive deployment, a retrieval quality bug that only showed up on ambiguous questions, and a fine-tuning experiment that failed on easy tests and succeeded on the hard one that actually mattered.

If you're skimming, the interesting parts are in [Debugging log](#debugging-log-the-part-that-actually-mattered) and [Fine-tuning the retriever](#fine-tuning-the-retriever).

## Features

- **Upload your own PDFs** — drop a document in through the UI and it's chunked, embedded, and searchable immediately, no restart needed
- **Scoped search** — ask a question against one specific document instead of the whole corpus, when you already know where the answer lives
- **Grounded, cited answers** — every response points back to the exact source chunk it was built from, formatted as `[Source 12]`
- **Clean text extraction** — automatically detects and strips repeated headers, footers, and page numbers that would otherwise pollute every chunk
- **Deduplication** — near-duplicate chunks are dropped before indexing, so repeated boilerplate doesn't crowd out real content
- **A fine-tuned retriever** — the embedding model has been fine-tuned on this project's own documents, with measured improvement on ambiguous cross-document queries (details below)

## Architecture

```
                 OFFLINE (runs locally, once per document)
┌──────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────┐
│  PDF(s)  │ → │ Clean + chunk │ → │ Embed (local)  │ → │ FAISS index  │
└──────────┘   └───────────────┘   └────────────────┘   └──────────────┘

                 LIVE (deployed on Render)
┌──────────┐   ┌────────────────┐   ┌──────────────┐   ┌────────────┐
│ Question │ → │ Embed (HF API) │ → │ FAISS search │ → │ Groq LLM   │ → cited answer
└──────────┘   └────────────────┘   └──────────────┘   └────────────┘
```

Two different embedding paths, on purpose — see [why](#the-memory-crash) below.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Flask + gunicorn |
| Vector search | FAISS |
| Embeddings (offline indexing) | sentence-transformers (`all-MiniLM-L6-v2`), fine-tuned |
| Embeddings (live queries) | Hugging Face Inference API |
| Answer generation | Groq (`openai/gpt-oss-120b`) |
| PDF parsing | PyMuPDF |
| Deployment | Render |

---

## Debugging log (the part that actually mattered)

### The memory crash

The first deployment loaded `sentence-transformers` and a cross-encoder reranker directly in the Flask process. Both pull in PyTorch, which alone can eat 400-600 MB of RAM on import — comfortably more than Render's free-tier 512 MB limit. The process got OOM-killed on every real request.

**Fix:** the offline indexing script (`ingest.py`) still loads the embedding model locally — it only runs once, on a laptop, so memory isn't a constraint there. The live app never loads it. Instead, query-time embeddings are fetched from Hugging Face's hosted Inference API, and the cross-encoder was dropped entirely. This is why the architecture above has two separate embedding paths instead of one.

### The retrieval regression nobody flagged

Dropping the reranker fixed memory, but introduced a quieter problem: questions with overlapping vocabulary across documents started returning wrong or empty answers. `"what is normalization?"` would sometimes match a chunk from a machine-learning PDF instead of the DBMS notes — because "normalization" means something different in each, and a plain bi-encoder embedding doesn't know that without the reranker's cross-attention to disambiguate.

Confirmed via debug logging that this wasn't a bug in the code — it was a genuine limitation of bi-encoder-only retrieval on ambiguous terms.

**Fix, in two parts:**
1. A document-scope filter (`source_filter` param + a dropdown in the UI) so a question can be explicitly scoped to one PDF when the user already knows where the answer lives — the correct fix for ambiguity, since structured filtering beats trying to out-word an embedding model.
2. A free FAISS L2-distance relevance filter (`distance_margin`) as a lightweight partial substitute for the removed reranker — costs nothing extra since the distances are already computed during search.

### The import that shouldn't have been there

A late deploy broke with `ModuleNotFoundError: sentence_transformers` — despite the library being intentionally removed from `requirements.txt`. Root cause: `rag.py` imported a function from `ingest.py` for the upload feature, and `ingest.py` had `sentence_transformers` imported at the top of the file, so it got dragged along even though the live app never needed it. Fixed by moving that import inside the one function that actually uses it.

---

## Fine-tuning the retriever

The embedding model (`all-MiniLM-L6-v2`) ships pretrained on general web text — not tuned to this project's documents. The hypothesis: fine-tuning it on question-passage pairs generated from the actual corpus should improve retrieval, especially on the ambiguous-query weakness discovered above.

**Method:** Groq generated a natural question for each of 149 document chunks, forming (question, chunk) training pairs. The embedding model was fine-tuned for 4 epochs using `MultipleNegativesRankingLoss` — each pair is pulled together in embedding space while every other chunk in the batch acts as an implicit negative.

**Result — honest, not cherry-picked:**

| Test set | Base model MRR | Fine-tuned MRR | Verdict |
|---|---|---|---|
| 30 easy, single-topic questions | 0.925 | 0.900 | No improvement — base model was already near-ceiling |
| 30 targeted, cross-document ambiguous questions | 0.778 | 0.823 | **+5.8% relative improvement** |

The first test was, in hindsight, too easy — four topically distinct PDFs make source-level discrimination trivial for any embedding model, leaving no room to show a difference. The second test specifically targeted vocabulary that overlaps across documents (the same failure mode found during the retrieval regression above), and that's where the fine-tune actually earned its keep.

This is the result worth reporting, and the reasoning behind reporting *both* numbers instead of just the flattering one: a fine-tune that helps on the exact problem you already know exists is a stronger, more credible result than a fine-tune that "improves accuracy" on a benchmark that was never hard to begin with.

---

## Project structure

```
TechDoc-AI/
├── app.py                        # Flask routes: /, /ask, /upload, /documents
├── ingest.py                     # Offline: PDF → clean → chunk → dedup → embed → FAISS index
├── rag.py                        # Live: query embedding (HF API), search, filtering, generation
├── generate_training_data.py     # Builds question-chunk pairs for fine-tuning
├── finetune_embeddings.py        # Fine-tunes the embedding model on those pairs
├── generate_eval_set.py          # Auto-generates a held-out evaluation set
├── generate_ambiguous_eval.py    # Auto-generates a targeted, cross-document ambiguity eval set
├── evaluate_finetune_v3.py       # Compares base vs fine-tuned model (rank + MRR)
├── finetuned-embedding-model/    # The fine-tuned model, ready to load
├── data/                         # Source PDFs
├── templates/index.html          # Chat UI
├── static/style.css
├── vector.index                  # FAISS index (generated by ingest.py)
├── chunks.pkl                    # Chunk text + source metadata
└── requirements.txt
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/varshiniui/TechDoc-AI.git
cd TechDoc-AI
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

- Groq key: [console.groq.com](https://console.groq.com)
- Hugging Face token (Read access): [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Build the index

Add PDFs to `data/`, then:

```bash
python ingest.py
```

Re-run any time documents change.

### 4. Run locally

```bash
python app.py
```

Visit `http://localhost:5000`.

### 5. (Optional) Reproduce the fine-tuning experiment

```bash
python generate_training_data.py       # generates training_pairs.jsonl
python finetune_embeddings.py          # fine-tunes on those pairs
python generate_eval_set.py            # held-out eval set
python generate_ambiguous_eval.py      # targeted ambiguity eval set
python evaluate_finetune_v3.py eval_pairs.jsonl
python evaluate_finetune_v3.py ambiguous_eval_pairs.jsonl
```

## Deployment

Runs on Render as a Flask service via `gunicorn app:app`. Both `GROQ_API_KEY` and `HF_TOKEN` must be set as environment variables in the hosting dashboard — the `.env` file is git-ignored and never deployed.

## Author

**Sahaya Varshini M J**
[GitHub](https://github.com/varshiniui) · [LinkedIn](https://linkedin.com/in/varshini-shya) · [Portfolio](https://portfolio-one-rho-42.vercel.app)