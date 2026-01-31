import os
import sqlite3
import math
from typing import List

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ================= CONFIG =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_PATH = "embeddings.db"
DOCS_PATH = "docs"
TOP_K = 3

# ================= MODELS =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = pipeline("text2text-generation", model="google/flan-t5-small")

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB
        )
        """
    )
    conn.commit()
    conn.close()


def store_chunk(text: str, embedding: List[float]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chunks (text, embedding) VALUES (?, ?)",
        (text, sqlite3.Binary(float_list_to_bytes(embedding))),
    )
    conn.commit()
    conn.close()


def load_chunks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT text, embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()
    return [(text, bytes_to_float_list(blob)) for text, blob in rows]

# ================= UTILITIES =================
def chunk_text(text: str, size: int = 200):
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(current) >= size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)


def float_list_to_bytes(arr: List[float]) -> bytes:
    import struct
    return b"".join(struct.pack("f", x) for x in arr)


def bytes_to_float_list(b: bytes) -> List[float]:
    import struct
    return [struct.unpack("f", b[i:i+4])[0] for i in range(0, len(b), 4)]

# ================= INGEST DOCUMENTS =================
def ingest_docs():
    if not os.path.exists(DOCS_PATH):
        return
    for file in os.listdir(DOCS_PATH):
        if not file.endswith((".txt", ".md")):
            continue
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_text(text):
            embedding = embedder.encode(chunk).tolist()
            store_chunk(chunk, embedding)

# ================= RAG =================
def retrieve_context(query: str):
    query_emb = embedder.encode(query).tolist()
    chunks = load_chunks()
    scored = [(cosine_similarity(query_emb, emb), text) for text, emb in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:TOP_K]]


def generate_answer(query: str, context: List[str]) -> str:
    prompt = (
        "Answer the question using the context below. "
        "If the answer is not present, say you don't know.\n\n"
        f"Context:\n{' '.join(context)}\n\n"
        f"Question: {query}\nAnswer:"
    )
    result = llm(prompt, max_new_tokens=200)[0]["generated_text"]
    return result.strip()

# ================= BOT COMMANDS =================
user_history = {}

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/ask <question> - Ask a question from documents\n"
        "/help - Show help"
    )


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Please provide a question.")
        return

    ctx = retrieve_context(query)
    answer = generate_answer(query, ctx)

    uid = update.effective_user.id
    user_history.setdefault(uid, []).append((query, answer))
    user_history[uid] = user_history[uid][-3:]

    await update.message.reply_text(answer)

# ================= MAIN =================
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set")

    init_db()
    if not load_chunks():
        ingest_docs()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))

    print("Telegram Mini-RAG Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
