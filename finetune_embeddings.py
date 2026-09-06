import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -----------------------------
# 1. Load the training pairs
# -----------------------------

pairs = []
with open("training_pairs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))

print(f"Loaded {len(pairs)} training pairs.")

train_examples = [
    InputExample(texts=[pair["question"], pair["chunk"]])
    for pair in pairs
]

# -----------------------------
# 2. Load the base model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 3. Set up training
# -----------------------------

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# -----------------------------
# 4. Fine-tune
# -----------------------------

print("Starting fine-tuning...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=int(len(train_dataloader) * 0.1),
    show_progress_bar=True,
)

# -----------------------------
# 5. Save the fine-tuned model
# -----------------------------

model.save("finetuned-embedding-model")
print("\nSaved fine-tuned model to ./finetuned-embedding-model")