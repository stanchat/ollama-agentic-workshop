import requests
import json

# Replace with your actual Ollama API endpoint
OLLAMA = "http://localhost:11434/api/generate"

# Define your payload (example)
payload = {
    "model": "llama3.2:latest",
    "prompt": "Explain zero-shot vs few-shot learning concisely",
    "temperature": 0.2,
    "max_tokens": 128
}

# Make the POST request
r = requests.post(OLLAMA, json=payload)

# Parse the streaming JSON lines response
responses = []

for line in r.text.strip().split('\n'):
    data = json.loads(line)
    responses.append(data.get("response", ""))

# Combine all parts into the full response string
full_response = "".join(responses)

# Print the complete generated text
print(full_response.strip())
