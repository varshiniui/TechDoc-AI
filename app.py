from flask import Flask, render_template, request, jsonify
from rag import search_documents, generate_answer, add_pdf_to_index, list_available_documents
import os
from werkzeug.utils import secure_filename

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/documents")
def documents():
    return jsonify({"documents": list_available_documents()})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    source = data.get("source", "").strip() or None

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    results = search_documents(question, source_filter=source)

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


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    added = add_pdf_to_index(save_path, filename)

    if added == 0:
        return jsonify({"error": "Couldn't extract any usable text from that PDF."}), 400

    return jsonify({
        "message": f"Added {added} chunk(s) from {filename}. You can ask questions about it now."
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)