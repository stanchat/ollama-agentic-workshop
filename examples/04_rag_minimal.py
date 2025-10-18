"""
Super-minimal RAG:
- Build an index from a local text file (examples/data/sample.txt)
- Embed chunks with a small local embedding model served by Ollama (e.g., nomic-embed-text)
- Retrieve top chunks and stuff into the prompt.
This is intentionally compact for workshop use.
"""

import os, re, json, numpy as np, requests
import faiss

EMBED = "http://localhost:11434/api/embeddings"
CHAT = "http://localhost:11434/api/chat"
EMBED_MODEL = "nomic-embed-text"      # pull with: ollama pull nomic-embed-text
GEN_MODEL   = "llama3.2:latest"

def chunks(text, size=600):
    words = re.findall(r"\S+\s*", text)
    buff, cur = [], 0
    for w in words:
        buff.append(w); cur += len(w)
        if cur >= size:
            yield "".join(buff).strip(); buff, cur = [], 0
    if buff: yield "".join(buff).strip()

def embed(texts):
    r = requests.post(EMBED, json={"model": EMBED_MODEL, "input": texts}, timeout=60)
    r.raise_for_status()
    return np.array([d["embedding"] for d in r.json()["data"]], dtype="float32")

# Build corpus
with open(os.path.join(os.path.dirname(__file__), "data", "sample.txt"), "r", encoding="utf-8") as f:
    docs = list(chunks(f.read(), 700))

vecs = embed(docs)
index = faiss.IndexFlatIP(vecs.shape[1])
# normalize for cosine
faiss.normalize_L2(vecs)
index.add(vecs)

# Query
query = "What are the key steps to find and track federal grants?"
qv = embed([query]); faiss.normalize_L2(qv)
dist, idx = index.search(qv, 3)
context = "\n\n".join(docs[i] for i in idx[0])

messages = [
    {"role":"system","content":"Use the provided context to answer concisely. If missing, say you don't know."},
    {"role":"user","content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
]
r = requests.post(CHAT, json={"model": GEN_MODEL, "messages": messages}, timeout=60)
r.raise_for_status()
print(r.json()["message"]["content"].strip())
