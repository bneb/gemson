import os
import json
import base64
import requests
import subprocess
import time
import signal

IMAGES = {
    "404_error": "data/images/404_error.png",
    "bsod_crash": "data/images/bsod_crash.png",
    "android_crash": "data/images/android_crash.png"
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_gemma(image_path):
    url = "http://localhost:11434/v1/chat/completions"
    base64_image = encode_image(image_path)
    payload = {
        "model": "gemma4:12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract bug data from this image. Format strictly as JSON with: user_name, os_version, device_model, issue_type (crash|UI_glitch|latency|feature_request), reproduction_steps."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1
    }
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def unload_ollama():
    print("Unloading Ollama model to free memory...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma4:12b",
        "keep_alive": 0
    }
    try:
        requests.post(url, json=payload)
        time.sleep(2)
    except:
        pass

def query_gemson(image_path):
    url = "http://localhost:8080/v1/chat/completions"
    base64_image = encode_image(image_path)
    payload = {
        "model": "gemson-12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract bug data from this image. Format strictly as JSON with: user_name, os_version, device_model, issue_type (crash|UI_glitch|latency|feature_request), reproduction_steps."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1
    }
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

print("\n========================================")
print("SEQUENTIAL A/B TEST: IMAGE-TO-JSON")
print("========================================")

# Step 1: Run Baseline
print("\n[PHASE 1: RUNNING BASELINE (OLLAMA)]")
baseline_results = {}
for name, path in IMAGES.items():
    print(f"Querying Gemma-4-12B for {name}...")
    baseline_results[name] = query_gemma(path)

# Step 2: Unload Ollama memory
unload_ollama()

# Step 3: Start llama-server
print("\n[PHASE 2: STARTING LLAMA-SERVER]")
server_process = subprocess.Popen(
    ["llama-server", "-m", "../outputs/gemson-12b-lora.gguf", "--mmproj", "../outputs/gemson-12b-lora-mmproj.gguf", "--port", "8080"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd="serve"
)

# Wait for server to be ready
print("Waiting for llama-server to load into RAM (this takes a moment)...")
for _ in range(30):
    try:
        r = requests.get("http://localhost:8080/health")
        if r.status_code == 200:
            break
    except:
        pass
    time.sleep(2)

print("llama-server is ready!")

# Step 4: Run Fine-tune
print("\n[PHASE 3: RUNNING FINE-TUNE (LLAMA-SERVER)]")
finetune_results = {}
for name, path in IMAGES.items():
    print(f"Querying Gemson-12B for {name}...")
    finetune_results[name] = query_gemson(path)

# Step 5: Kill llama-server
print("Killing llama-server to free memory...")
os.kill(server_process.pid, signal.SIGTERM)
server_process.wait()

print("\n========================================")
print("RESULTS SUMMARY")
print("========================================")

for name in IMAGES.keys():
    print(f"\n--- Evaluating Image: {name}.png ---")
    print(f"\n[Baseline Model: Gemma-4-12B]")
    print(baseline_results[name])
    print(f"\n[Fine-Tuned Model: Gemson-12B]")
    print(finetune_results[name])
    print("\n" + "="*40)
