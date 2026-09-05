from flask import Flask, render_template, request, jsonify
from rag import search_documents, generate_answer

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    results = search_documents(question)

    if not results:
        return jsonify({
            "answer": None,
            "sources": [],
            "message": "No relevant information found for that question."
        })

    answer = generate_answer(question, results)

    sources = [
        {
            "chunk_id": r["chunk_id"],
            "source_pdf": r["source_pdf"],
            "preview": r["text"][:200]
        }
        for r in results
    ]

    return jsonify({
        "answer": answer,
        "sources": sources,
        "message": None
    })


import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)