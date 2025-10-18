
"""
05_local_summarize.py — Summarize local PDFs with Ollama (no cloud)

Usage:
  python 05_local_summarize.py path\to\doc1.pdf path\to\doc2.pdf --model llama3.2:latest --max-chars 1800 --overlap 0.12

Requires:
  pip install requests pypdf

Notes:
  - Runs entirely against local Ollama at http://localhost:11434
  - Strategy: extract → chunk → per‑chunk notes → final synthesis
"""

import os, sys, json, argparse
import requests
from pypdf import PdfReader
import numpy as np

OLLAMA_CHAT = "http://localhost:11434/api/chat"

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts).strip()

def chunk_text(text: str, max_chars: int = 1800, overlap: float = 0.12):
    if not text:
        return []
    step = int(max_chars * (1 - overlap))
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        if i + max_chars < len(text):
            j = chunk.rfind(" ")
            if j > 0:
                chunk = chunk[:j]
        chunks.append(chunk.strip())
        i += step
    return [c for c in chunks if c]

def chat(model: str, messages: list, temperature=0.2, seed=42):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed
        }
    }
    r = requests.post(OLLAMA_CHAT, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    return (j.get("message") or {}).get("content", "").strip()

def summarize_chunks(model: str, chunks: list) -> list:
    notes = []
    for idx, ch in enumerate(chunks, 1):
        prompt = f"""You are an expert technical summarizer. Read the following excerpt and produce:
- 3-6 concise bullet points capturing the most important facts and claims
- Any dates, identifiers, or entities that matter
- If it's boilerplate, say 'LOW-SIGNAL' and keep bullets short

EXCERPT ({idx}/{len(chunks)}):
{ch}
"""
        msg = [{"role":"user","content": prompt}]
        note = chat(model, msg)
        notes.append(note)
    return notes

def synthesize(model: str, doc_summaries: list, title: str) -> str:
    joined = "\n\n---\n\n".join(doc_summaries)
    prompt = f"""Create an executive briefing based on chunk summaries below.

Document: {title}

Instructions:
- Start with a 4-7 sentence narrative summary (no fluff).
- Then provide 6-10 bullets of key points.
- Then list any action items/next steps in 3-6 bullets.
- If material contains legal/USPTO context, keep terms precise and avoid speculation.

CHUNK SUMMARIES:
{joined}
"""
    msg = [{"role":"user","content": prompt}]
    return chat(model, msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="PDF file paths")
    ap.add_argument("--model", default="llama3.2:latest", help="Ollama model name")
    ap.add_argument("--max-chars", type=int, default=1800, help="Chunk size in characters")
    ap.add_argument("--overlap", type=float, default=0.12, help="Chunk overlap ratio (0-0.3)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    final_reports = []
    for p in args.paths:
        if not os.path.exists(p):
            print(f"[!] File not found: {p}", file=sys.stderr)
            continue
        print(f"[+] Reading {p} ...")
        text = read_pdf(p)

        if not text.strip():
            print(f"[!] No extractable text in {p} (scanned PDF?)", file=sys.stderr)
            continue

        chunks = chunk_text(text, max_chars=args.max_chars, overlap=args.overlap)
        print(f"[+] {len(chunks)} chunks")

        notes = summarize_chunks(args.model, chunks)
        title = os.path.basename(p)
        briefing = synthesize(args.model, notes, title)
        final_reports.append((title, briefing))

    print("\n================ EXECUTIVE BRIEFINGS ================\n")
    for title, briefing in final_reports:
        print(f"# {title}\n")
        print(briefing)
        print("\n-----------------------------------------------------\n")

if __name__ == "__main__":
    main()
