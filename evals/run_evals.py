import json
import argparse
import requests
from pydantic import BaseModel, ValidationError, Field
from typing import List
from tqdm import tqdm

class BugReport(BaseModel):
    user_name: str = Field(min_length=2)
    os_version: str = Field(min_length=3)
    device_model: str = Field(min_length=1)
    issue_type: str
    reproduction_steps: List[str]

def load_test_data(filepath, max_samples=50):
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            user_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), None)
            expected_json = next((m["content"] for m in data["messages"] if m["role"] == "assistant"), None)
            if user_msg and expected_json:
                samples.append((user_msg, expected_json))
                if len(samples) >= max_samples:
                    break
    return samples

def query_ollama(prompt, model="gemma4:12b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception:
        pass
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma4:12b", help="Ollama model to evaluate")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--data", type=str, default="../data/eval_data.jsonl", help="Path to JSONL data")
    args = parser.parse_args()

    print(f"Loading {args.samples} samples from {args.data}...")
    try:
        samples = load_test_data(args.data, max_samples=args.samples)
    except FileNotFoundError:
        print("Data file not found. Ensure the data generator has run.")
        return
    
    if not samples:
        print("No test data found.")
        return

    print(f"Evaluating {args.model} on {len(samples)} samples...")
    success = 0
    failures = 0
    
    for user_prompt, _ in tqdm(samples):
        # The user_prompt already contains "Extract bug data from this transcript:\n\n..."
        raw_output = query_ollama(user_prompt, model=args.model)
        
        try:
            parsed = json.loads(raw_output)
            BugReport(**parsed)
            if parsed.get("issue_type") not in ["crash", "UI_glitch", "latency", "feature_request"]:
                raise ValueError("Invalid issue_type enum")
            success += 1
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            failures += 1

    total = success + failures
    accuracy = (success / total) * 100
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Model:      {args.model}")
    print(f"Total:      {total}")
    print(f"Success:    {success}")
    print(f"Failures:   {failures}")
    print(f"Accuracy:   {accuracy:.2f}%")
    print("="*40)
    
if __name__ == "__main__":
    main()
