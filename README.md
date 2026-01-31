# Telegram Mini-RAG Bot

This project implements a lightweight **GenAI Telegram bot** using a
**Mini Retrieval-Augmented Generation (RAG)** pipeline.

The bot answers user questions based on a small local document knowledge base.

---

## 🎯 Features

- Telegram bot using `python-telegram-bot`
- Mini-RAG pipeline:
  - Text chunking
  - Local embeddings (`all-MiniLM-L6-v2`)
  - SQLite-based vector storage
  - Cosine similarity retrieval
- Local open-source LLM (`flan-t5-small`)
- `/ask` and `/help` commands
- Keeps last 3 interactions per user
- Fully local (no OpenAI API)

---

## 📁 Project Structure

telegram-mini-rag-bot/
│── app.py
│── requirements.txt
│── README.md
│── embeddings.db (auto-created)
│
└── docs/
├── faq.txt
├── policy.md
└── notes.txt