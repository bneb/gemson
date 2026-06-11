import json
import requests

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "gemma4:12b", "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    resp = requests.post(url, json=payload, timeout=60)
    return resp.json().get("response", "")

with open("data/eval_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        user_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), None)
        print(f"\n================ SAMPLE {i+1} ================")
        print("PROMPT:", user_msg[:100], "...")
        print("BASELINE RAW OUTPUT:")
        print(query_ollama(user_msg))

