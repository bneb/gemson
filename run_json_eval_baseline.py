import json
import requests
from tqdm import tqdm
import time

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    system_instruction = (
        "You are a strict data extraction assistant. "
        "You must extract the bug data and return it as perfectly valid, parseable JSON. "
        "Your output must exactly match this schema: "
        "{\"user_name\": string|null, \"os_version\": string|null, \"device_model\": string|null, "
        "\"issue_type\": \"crash\"|\"UI_glitch\"|\"performance\"|\"Unknown\", \"reproduction_steps\": list[string]|null}. "
        "Here is the transcript:\n\n"
    )
    payload = {
        "model": "gemma4:12b", 
        "prompt": system_instruction + prompt, 
        "stream": False, 
        "format": "json",
        "options": {"temperature": 0.0}
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        return resp.json().get("response", "")
    except Exception:
        return "{}"

samples = []
with open("data/eval_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 10: break # Only test 10 samples to be fast
        samples.append(json.loads(line))

total_score = 0
for i, data in enumerate(tqdm(samples)):
    user_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), None)
    expected_str = next((m["content"] for m in data["messages"] if m["role"] == "assistant"), None)
    
    raw_output = query_ollama(user_msg)
    try:
        actual = json.loads(raw_output)
        expected = json.loads(expected_str)
        
        score = 0
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        
        if expected_keys == actual_keys: score += 20
        if actual.get("issue_type") == expected.get("issue_type"): score += 30
        
        actual_steps = actual.get("reproduction_steps", [])
        expected_steps = expected.get("reproduction_steps", [])
        if isinstance(actual_steps, list) and len(actual_steps) > 0:
            if len(actual_steps) == 1 and len(expected_steps) > 1: score += 10
            else: score += 30
            
        match_count = 0
        for key in ["user_name", "os_version", "device_model"]:
            if str(actual.get(key)).lower() == str(expected.get(key)).lower(): match_count += 1
        score += int((match_count / 3) * 20)
        
        total_score += score
    except Exception as e:
        pass
        
print(f"\nAccuracy on 10 samples with STRICT JSON PROMPT + JSON FLAG: {total_score / 10}%")
