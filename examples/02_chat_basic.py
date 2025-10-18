import requests

CHAT = "http://localhost:11434/api/chat"
MODEL = "llama3.2:latest"

messages = [
    {"role": "system", "content": "You are a helpful AI for software demos."},
    {"role": "user", "content": "Explain retrieval-augmented generation in 2 sentences."},
]

r = requests.post(CHAT, json={"model": MODEL, "messages": messages}, timeout=60)
r.raise_for_status()
print(r.json().get("message", {}).get("content", "").strip())
