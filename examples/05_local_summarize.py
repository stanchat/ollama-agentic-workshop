# examples/05_local_summarize.py
"""
Demo 05 — Local Summarizer (no NLTK; hardcoded PDFs)

- Reads two PDFs from ./examples/data
- Splits into sentences via regex (no external downloads)
- Summarizes chunks with local Ollama and stitches a final brief
"""

from pathlib import Path
import re
import requests
from pypdf import PdfReader

# ---- Config ----
OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2:latest"  # try "mistral:latest" if outputs are too thin
CHUNK_TARGET_CHARS = 4800
OVERLAP_CHARS = 200
NUM_CTX = 8192
TEMPERATURE = 0.2



# Hardcoded PDF paths relative to repo root (…/ollama-agentic-workshop)
BASE = Path(__file__).resolve().parents[1]
PDFS = [
    BASE / "examples" / "data" / "PROVISIONAL PATENT APPLICATION fonts changed.pdf",
    BASE / "examples" / "data" / "Submission Receipt - Submissions - Patent Center - USPTO.pdf",
]

# ---- Utilities ----
def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        parts.append(t)
    text = "\n".join(parts)
    # basic cleanup
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

_ABBR = r"(?:Mr|Ms|Mrs|Dr|Prof|Inc|Ltd|Co|No|Jr|Sr|vs|U\.S|U\.S\.P\.T\.O)"
def sentence_split(text: str) -> list[str]:
    """
    Simple, fast sentence splitter:
    - splits on ., !, ? followed by space & capital
    - avoids splitting on common abbreviations
    """
    # Protect periods in common abbreviations by temporary token
    safe = re.sub(fr"\b({_ABBR})\.", r"\1<PERIOD>", text)
    # Split on end punctuation followed by space+capital or end of line
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", safe)
    # Restore periods
    parts = [p.replace("<PERIOD>", ".").strip() for p in parts if p.strip()]
    return parts

def chunk_by_sentences(text: str, target_chars=CHUNK_TARGET_CHARS, overlap=OVERLAP_CHARS) -> list[str]:
    sents = sentence_split(text)
    chunks, buf, count = [], [], 0
    for s in sents:
        buf.append(s)
        count += len(s) + 1
        if count >= target_chars:
            chunk = " ".join(buf).strip()
            chunks.append(chunk)
            # overlap: keep tail
            tail, tlen = [], 0
            for rs in reversed(buf):
                tail.append(rs)
                tlen += len(rs) + 1
                if tlen >= overlap:
                    break
            buf = list(reversed(tail))
            count = sum(len(x) + 1 for x in buf)
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks

def ollama_generate(prompt: str, model: str = MODEL) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": TEMPERATURE,
                # "top_p": 0.9, "repeat_penalty": 1.05,  # optional knobs
            },
            "stream": False,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()

# ---- Prompts ----
CHUNK_PROMPT = """You are a precise summarizer. Read the CHUNK and return 3–7 factual bullet points.

Rules:
- Keep key facts, dates, entities, definitions.
- Be terse and specific. No filler. No speculation.

CHUNK:
{chunk}
"""

FINAL_PROMPT = """You will receive several bullet lists, one per chunk. Combine them into ONE cohesive brief.

Output format:
**Overview (2–4 sentences)**
- Key Entities & Roles: (3–5 bullets)
- Important Dates / Numbers: (3–5 bullets)
- Core Concepts / Definitions: (3–6 bullets)
- Notable Procedural Details: (3–6 bullets)

Do not repeat. Do not invent facts.

BULLETS:
{bullets}
"""

# ---- Main ----
def main():
    print("📄 Loading PDFs...")
    texts = []
    for p in PDFS:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        print(f" - {p.name}")
        texts.append(read_pdf_text(p))

    combined = "\n\n".join(texts)
    print(f"Total characters: {len(combined)}")

    chunks = chunk_by_sentences(combined)
    print(f"🔍 Splitting into {len(chunks)} chunks...")

    chunk_summaries = []
    for i, ch in enumerate(chunks, 1):
        print(f"→ Summarizing chunk {i}/{len(chunks)}...")
        prompt = CHUNK_PROMPT.format(chunk=ch[:12000])  # safety cap
        s = ollama_generate(prompt)
        if not s:
            # Retry once with slightly different instruction if empty
            s = ollama_generate("Summarize the following in 3–6 factual bullet points:\n\n" + ch[:12000])
        if not s:
            s = "- (No summary returned for this chunk.)"
        chunk_summaries.append(s)

    bullets_blob = "\n\n".join(chunk_summaries)
    final = ollama_generate(FINAL_PROMPT.format(bullets=bullets_blob))

    print("\n=== FINAL SUMMARY ===\n")
    print(final if final else "(Model returned empty text. Try MODEL='mistral:latest' or reduce CHUNK_TARGET_CHARS.)")

if __name__ == "__main__":
    main()
