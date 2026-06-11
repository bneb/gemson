import json
import requests
from tqdm import tqdm
import re

def load_data(filepath, max_samples=50):
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
    
    # Give the baseline model a fair chance by explicitly asking for JSON
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
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print(f"Error querying Ollama: {e}")
    return ""

def clean_json_output(raw_text):
    # Strip markdown codeblocks
    cleaned = re.sub(r'```json\n', '', raw_text)
    cleaned = re.sub(r'```', '', cleaned).strip()
    # Strip leading/trailing conversational filler if any
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]
    return cleaned

def score_json(actual_json_str, expected_json_str):
    try:
        actual = json.loads(actual_json_str)
        expected = json.loads(expected_json_str)
    except json.JSONDecodeError:
        return 0, "Invalid JSON after cleanup"

    score = 0
    feedback = []
    
    # 1. Check exact key matches (no hallucinations)
    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())
    
    if expected_keys == actual_keys:
        score += 20
    else:
        extra = actual_keys - expected_keys
        missing = expected_keys - actual_keys
        if extra: feedback.append(f"Hallucinated keys: {extra}")
        if missing: feedback.append(f"Missing keys: {missing}")
        
    # 2. Check issue_type constraint
    if actual.get("issue_type") == expected.get("issue_type"):
        score += 30
    else:
        feedback.append(f"issue_type mismatch: got {actual.get('issue_type')}, expected {expected.get('issue_type')}")
        
    # 3. Check reproduction steps quality
    actual_steps = actual.get("reproduction_steps", [])
    expected_steps = expected.get("reproduction_steps", [])
    
    if isinstance(actual_steps, list) and len(actual_steps) > 0:
        if len(actual_steps) == 1 and len(expected_steps) > 1:
            score += 10
            feedback.append("Steps compressed into a single string (lazy extraction)")
        else:
            score += 30
    else:
        feedback.append("Missing or invalid reproduction_steps")
        
    # 4. Data Extraction Accuracy (device/os/user)
    match_count = 0
    for key in ["user_name", "os_version", "device_model"]:
        if str(actual.get(key)).lower() == str(expected.get(key)).lower():
            match_count += 1
    score += int((match_count / 3) * 20)
    
    return score, ", ".join(feedback) if feedback else "Perfect match!"

def main():
    samples = load_data("data/eval_data.jsonl", 50)
    print(f"Loaded {len(samples)} samples. Running qualitative analysis on gemma4:12b...")
    
    total_score = 0
    detailed_logs = []
    
    for i, (prompt, expected_str) in enumerate(tqdm(samples)):
        raw_output = query_ollama(prompt)
        cleaned_output = clean_json_output(raw_output)
        
        score, feedback = score_json(cleaned_output, expected_str)
        total_score += score
        
        # Save detailed logs for a few samples to review
        if i < 5:
            detailed_logs.append({
                "sample": i+1,
                "expected": json.loads(expected_str),
                "actual_raw": raw_output,
                "actual_cleaned": cleaned_output,
                "score": score,
                "feedback": feedback
            })
            
        # Append to persistent output file
        with open("evals/baseline_text_outputs.jsonl", "a") as out_f:
            out_f.write(json.dumps({
                "sample_index": i,
                "prompt": prompt,
                "expected": json.loads(expected_str),
                "actual_raw": raw_output,
                "actual_cleaned": cleaned_output,
                "score": score,
                "feedback": feedback
            }) + "\n")
            
    avg_score = total_score / len(samples)
    print(f"\n========================================")
    print(f"QUALITATIVE ANALYSIS RESULTS")
    print(f"========================================")
    print(f"Average Qualitative Score: {avg_score:.2f} / 100")
    print(f"========================================\n")
    
    print("TOP 3 QUALITATIVE DIFFERENCES (First 3 Samples):")
    for log in detailed_logs[:3]:
        print(f"\n--- Sample {log['sample']} ---")
        print(f"Score: {log['score']}/100 | Feedback: {log['feedback']}")
        try:
            actual = json.loads(log['actual_cleaned'])
            print(f"  Baseline `issue_type`: {actual.get('issue_type')} | Expected: {log['expected'].get('issue_type')}")
            print(f"  Baseline `steps` len: {len(actual.get('reproduction_steps', []))} | Expected `steps` len: {len(log['expected'].get('reproduction_steps', []))}")
            if len(actual.get('reproduction_steps', [])) > 0:
                print(f"  Baseline Step 1: {actual.get('reproduction_steps', [])[0]}")
        except:
            print("  Failed to parse JSON.")

if __name__ == "__main__":
    main()
