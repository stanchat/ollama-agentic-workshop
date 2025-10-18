# Ollama Agentic Patterns — Python Only

**What you’ll learn**
- Run local LLMs with Ollama (no cloud keys).
- Use `/api/generate` and `/api/chat` from Python.
- Implement **tool-calling (agentic)** with a minimal loop.
- Add a tiny **RAG** example using local text.
- Summarize local **PDF documents** on-device with Ollama.

---

## Demo Index
- **01 — Generate**: Single-shot via `/api/generate`
- **02 — Chat (basic)**: Multi-turn via `/api/chat`
- **03 — Agentic Tool-Calling**: Function/tools loop
- **04 — Tiny Local RAG**: Embeddings + retrieval + context stuffing
- **05 — Local Doc Summarizer**: Summarize PDFs entirely on-device

---

## Prereqs
- Ollama installed and running (`ollama -v`), default host `http://localhost:11434`.
- Python 3.10+
- Install deps:
```bash
pip install -r requirements.txt
