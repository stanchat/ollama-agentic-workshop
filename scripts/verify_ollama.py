import requests
base = "http://localhost:11434"
try:
    r = requests.get(base, timeout=5)
    print("Ollama OK:", r.status_code, "at", base)
except Exception as e:
    print("Could not reach Ollama at", base, "-", e)
