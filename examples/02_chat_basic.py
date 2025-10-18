import requests
import json

OLLAMA = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3.2:latest",
    "messages": [
        {"role": "user", "content": "Explain zero-shot vs few-shot learning"}
    ],
    "tools": []
}

r = requests.post(OLLAMA, json=payload)

# Parse streamed JSON lines response
responses = []

for line in r.text.strip().split('\n'):
    data = json.loads(line)
    # Extract message content from the nested structure or fallback to empty string
    message_content = data.get("message", {}).get("content", "")
    responses.append(message_content)

full_response = "".join(responses)
print(full_response.strip())
