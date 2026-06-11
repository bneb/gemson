import json
import requests

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    system_instruction = (
        "You are a strict data extraction assistant. "
        "You must extract the bug data and return it as perfectly valid, parseable JSON. "
        "Do not include any conversational text or markdown code blocks like ```json. "
        "Your output must exactly match this schema: "
        "{\"user_name\": string|null, \"os_version\": string|null, \"device_model\": string|null, "
        "\"issue_type\": \"crash\"|\"UI_glitch\"|\"performance\"|\"Unknown\", \"reproduction_steps\": list[string]|null}. "
        "Here is the transcript:\n\n"
    )
    full_prompt = system_instruction + prompt
    payload = {"model": "gemma4:12b", "prompt": full_prompt, "stream": False, "options": {"temperature": 0.1}}
    resp = requests.post(url, json=payload, timeout=60)
    return resp.json().get("response", "")

with open("data/eval_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        data = json.loads(line)
        user_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), None)
        print(f"\n================ SAMPLE {i+1} ================")
        print("BASELINE RAW OUTPUT (WITH EXPLICIT PROMPT):")
        print(query_ollama(user_msg))

