import json
import requests

CHAT = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-q4_K_M"  # tools-capable instruct model

def llm(payload: dict) -> dict:
    # Force non-streaming for simpler JSON handling
    payload = {**payload, "stream": False}
    r = requests.post(CHAT, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def coerce_args(args_field):
    """
    Ollama may return function.arguments as a JSON string or as a dict.
    Normalize to dict.
    """
    if args_field is None:
        return {}
    if isinstance(args_field, dict):
        return args_field
    if isinstance(args_field, (bytes, bytearray)):
        args_field = args_field.decode("utf-8", errors="ignore")
    if isinstance(args_field, str):
        args_field = args_field.strip() or "{}"
        try:
            return json.loads(args_field)
        except Exception:
            # Fall back to empty on malformed string
            return {}
    # Unknown type
    return {}

def get_tool_calls(resp: dict):
    """
    Some Ollama builds nest tool calls under resp['message']['tool_calls'],
    others may surface differently. Normalize to a list.
    """
    m = resp.get("message") or {}
    calls = m.get("tool_calls")
    if isinstance(calls, list):
        return calls
    # Fallback: try top-level or alternative keys if your version differs
    return resp.get("tool_calls") or []

def search_grants(keyword: str):
    # TODO: swap in your real Grants.gov Search2 proxy call
    return [{
        "title": "DOE Education Innovation 2025",
        "deadline": "2025-09-30",
        "link": "https://www.grants.gov/..."
    }]

messages = [
    {
        "role": "system",
        "content": (
            "You are an assistant that MUST use available tools when the user asks to find real data. "
            "When asked to find a federal education grant, CALL the function `search_grants` "
            "with a relevant keyword; do not fabricate results."
        ),
    },
    {"role": "user", "content": "Find a federal education grant and summarize it."},
]

tools = [{
    "type": "function",
    "function": {
        "name": "search_grants",
        "description": "Search federal grants by keyword",
        "parameters": {
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"],
        },
    },
}]

print("Calling model with initial message and tools...")
res = llm({"model": MODEL, "messages": messages, "tools": tools})

assistant_msg = (res.get("message") or {}).get("content", "")
if assistant_msg:
    print("Assistant (pre-tool):", assistant_msg)

tool_calls = get_tool_calls(res)

if tool_calls:
    call = tool_calls[0]
    func = (call.get("function") or {})
    fname = func.get("name")
    args = coerce_args(func.get("arguments"))

    if fname == "search_grants":
        keyword = args.get("keyword") or "education"
        result = search_grants(keyword)

        # Send tool result back
        messages.append({
            "role": "tool",
            "name": "search_grants",
            "content": json.dumps(result)
        })

        print(f"Sent tool result for keyword='{keyword}' back to model...")
        final = llm({"model": MODEL, "messages": messages})
        print("\nFinal model response:\n", (final.get("message") or {}).get("content", "").strip())
    else:
        print(f"Model requested unknown tool: {fname}")
else:
    # Model skipped tools and answered directly (or returned nothing).
    msg = (res.get("message") or {}).get("content", "")
    if msg:
        print("\nModel answered directly (no tool call):\n", msg.strip())
    else:
        # Helpful debug if you see a blank response
        print("\nNo tool calls and empty content. Raw response for debugging:")
        print(json.dumps(res, indent=2)[:1200], "...")
