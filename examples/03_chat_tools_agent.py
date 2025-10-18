"""
Minimal agent loop:
1) Ask model with a tool schema.
2) If model requests the tool, run the Python function.
3) Send tool result back as a 'tool' message.
4) Print model's final answer.
"""

import json, requests

CHAT = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-q4_K_M"  # compact instruct model

def llm(payload: dict) -> dict:
    r = requests.post(CHAT, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def search_grants(keyword: str):
    # TODO: swap in your real Grants.gov Search2 proxy call
    return [{
        "title": "DOE Education Innovation 2025",
        "deadline": "2025-09-30",
        "link": "https://www.grants.gov/..."
    }]

messages = [{"role": "user", "content": "Find a federal education grant and summarize it."}]

tools = [{
    "type": "function",
    "function": {
        "name": "search_grants",
        "description": "Search federal grants by keyword",
        "parameters": {
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"]
        }
    }
}]

# Step 1 — ask with tools declared
res = llm({"model": MODEL, "messages": messages, "tools": tools})

# Step 2 — if the model calls our tool, run it
call = (res.get("message") or {}).get("tool_calls", [{}])[0]
if (call.get("function") or {}).get("name") == "search_grants":
    args = json.loads(call["function"].get("arguments") or "{}")
    result = search_grants(args.get("keyword", "education"))

    # Step 3 — feed result back as a tool message
    messages.append({"role": "tool", "name": "search_grants", "content": json.dumps(result)})

    final = llm({"model": MODEL, "messages": messages})
    print(final.get("message", {}).get("content", "").strip())
else:
    # No tool call; model answered directly
    print(res.get("message", {}).get("content", "").strip())
