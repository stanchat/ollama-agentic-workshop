# Ollama Agentic Patterns — Python Only

**What you’ll learn**
- Run local LLMs with Ollama (no cloud keys).
- Use `/api/generate` and `/api/chat` from Python.
- Implement **tool-calling (agentic)** with a minimal loop.
- Add a tiny **RAG** example using local text.

## Prereqs
- Ollama installed and running (`ollama -v`), default host `http://localhost:11434`.
- Python 3.10+
- Install deps:
```bash
pip install -r requirements.txt
```

## Quickstart
```bash
python examples/01_generate.py
python examples/02_chat_basic.py
python examples/03_chat_tools_agent.py
python examples/04_rag_minimal.py
```

## Models
Pull one or more before running examples:
```bash
ollama pull llama3.2:latest
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral
ollama pull nomic-embed-text
```
