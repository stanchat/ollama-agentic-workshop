import requests

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"

payload = {
    "model": MODEL,
    "prompt": "One-line summary of zero-shot vs. few-shot.",
    # "options": {"temperature": 0.2, "num_predict": 128}
}

r = requests.post(OLLAMA, json=payload, timeout=60)
r.raise_for_status()
print(r.json().get("response", "").strip())
