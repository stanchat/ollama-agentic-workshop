"""
04_rag_minimal.py — Tiny local RAG with Ollama (robust version)

Prereqs (PowerShell):
  ollama pull nomic-embed-text
  ollama pull llama3.2:latest
  pip install requests numpy faiss-cpu==1.8.0.post1  # (FAISS optional; script falls back if missing)

Run:
  python examples/04_rag_minimal.py
"""

import os
import re
import sys
import json
import math
import time
import numpy as np
import requests

# ----- Try FAISS; fall back to NumPy cosine if unavailable -----
USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    USE_FAISS = False

EMBED_URL   = "http://localhost:11434/api/embeddings"
CHAT_URL    = "http://localhost:11434/api/chat"
EMBED_MODEL = "nomic-embed-text"      # ensure pulled: ollama pull nomic-embed-text
GEN_MODEL   = "llama3.2:latest"       # ensure pulled: ollama pull llama3.2:latest

# ---------------------- Helpers ---------------------- #

def ensure_ollama_up():
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        if r.status_code != 200:
            print(f"[!] Ollama responded with {r.status_code}. Is it running correctly?")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("[!] Cannot reach Ollama at http://localhost:11434 — start the Ollama app/service.")
        sys.exit(1)

def ensure_embeddings_ready():
    """Probe embeddings endpoint & model; give clear guidance if not ready."""
    try:
        # Try single-text schema first (most compatible)
        probe = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": "ping"}, timeout=10)
        if probe.status_code == 404:
            print("[!] /api/embeddings returned 404. Update Ollama to a build that supports embeddings and restart.")
            sys.exit(1)
        if probe.status_code == 400 and "model" in (probe.text or "").lower():
            print("[!] Embedding model not found. Pull it:\n    ollama pull nomic-embed-text")
            sys.exit(1)
        probe.raise_for_status()
        # Try parsing once
        _ = parse_embeddings(probe.json())
    except requests.exceptions.ConnectionError:
        print("[!] Cannot reach Ollama at http://localhost:11434 — is it running?")
        sys.exit(1)
    except ValueError as e:
        print(f"[!] Unexpected embeddings schema: {e}")
        sys.exit(1)

def parse_embeddings(j):
    """
    Normalize different possible response schemas to: List[List[float]]
    Common shapes seen:
      - {"embedding":[...]}                      # single
      - {"data":[{"embedding":[...]}, ...]}     # batch
      - {"embeddings":[[...], [...]]}           # batch
    """
    if not isinstance(j, dict):
        raise ValueError("Embeddings response is not a dict")

    if "data" in j and isinstance(j["data"], list):
        return [item["embedding"] for item in j["data"] if "embedding" in item]

    if "embedding" in j:
        return [j["embedding"]]

    if "embeddings" in j and isinstance(j["embeddings"], list):
        return j["embeddings"]

    raise ValueError(f"Unknown embeddings schema keys: {list(j.keys())}")

def embed_texts(texts):
    """
    Robust embeddings: send one text at a time with 'prompt' (works across builds).
    Returns np.array shape (N, D).
    """
    vecs = []
    for t in texts:
        r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": t}, timeout=60)
        r.raise_for_status()
        parsed = parse_embeddings(r.json())
        if not parsed or not parsed[0]:
            raise ValueError(f"Empty embedding for text starting: {t[:80]!r}")
        vecs.append(parsed[0])
    return np.array(vecs, dtype="float32")

def chunk_text(text, size=700):
    """Simple chunker by character length, preserving word boundaries."""
    words = re.findall(r"\S+\s*", text)
    buff, cur = [], 0
    for w in words:
        buff.append(w)
        cur += len(w)
        if cur >= size:
            yield "".join(buff).strip()
            buff, cur = [], 0
    if buff:
        yield "".join(buff).strip()

def build_index(vectors):
    """
    Build FAISS inner-product index if available; else return None and we’ll cosine with NumPy.
    We normalize for cosine similarity (FAISS inner product with normalized vectors == cosine).
    """
    V = np.array(vectors, dtype="float32")
    # Normalize for cosine
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    Vn = V / norms

    if USE_FAISS:
        idx = faiss.IndexFlatIP(Vn.shape[1])
        idx.add(Vn)
        return idx, Vn
    else:
        return None, Vn

def search_top_k(index, matrix_normed, query_vec, k=3):
    """
    Return (scores, indices) of top-k using FAISS if available; else NumPy cosine.
    """
    q = np.array(query_vec, dtype="float32")
    qn = q / (np.linalg.norm(q) + 1e-12)

    if index is not None:  # FAISS path
        scores, idxs = index.search(qn.reshape(1, -1), k)
        return scores[0], idxs[0]
    else:
        # cosine with NumPy
        scores = matrix_normed @ qn
        idxs = np.argsort(scores)[::-1][:k]
        return scores[idxs], idxs

def chat(messages):
    r = requests.post(
        CHAT_URL,
        json={"model": GEN_MODEL, "messages": messages, "stream": False},
        timeout=60
    )
    r.raise_for_status()
    j = r.json()
    return (j.get("message") or {}).get("content", "").strip()

# ---------------------- Main ---------------------- #

if __name__ == "__main__":
    ensure_ollama_up()
    ensure_embeddings_ready()

    # Load a small local text file as our "corpus"
    # (You can replace this path or put your own document here.)
    data_path = os.path.join(os.path.dirname(__file__), "data", "sample.txt")
    if not os.path.exists(data_path):
        print(f"[!] Missing {data_path}. Create it with a few paragraphs of text and re-run.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = list(chunk_text(text, size=700))
    if not docs:
        print("[!] sample.txt is empty; add some content and re-run.")
        sys.exit(1)

    # Embed corpus
    print(f"Embedding {len(docs)} chunks...")
    vecs = embed_texts(docs)

    # Build index
    idx, vecs_normed = build_index(vecs)

    # Query
    query = "What are the key steps to find and track federal grants?"
    print(f"Embedding query: {query!r}")
    qvec = embed_texts([query])[0]

    # Retrieve
    k = 3
    scores, top_idx = search_top_k(idx, vecs_normed, qvec, k=k)
    context = "\n\n".join(docs[i] for i in top_idx)

    # Ask the model with stuffed context
    messages = [
        {"role": "system", "content": "Use the provided context to answer concisely. If missing, say you don't know."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"},
    ]
    print("Asking the model with retrieved context...")
    answer = chat(messages)

    print("\n--- RAG Answer ---\n")
    print(answer or "(No answer returned.)")

    # Optional: show which chunks were used
    print("\n--- Top Chunks (scores) ---")
    for rank, i in enumerate(top_idx, 1):
        sc = scores[rank - 1] if isinstance(scores, np.ndarray) else scores[rank - 1]
        print(f"[{rank}] score={float(sc):.4f}\n{docs[i][:200].strip()}...\n")
