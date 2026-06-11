import requests
import json
import base64
import time

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

images = {
    "BSOD": "data/images/bsod_crash.png",
    "404": "data/images/404_error.png",
    "Android": "data/images/android_crash.png"
}

prompt = "Extract bug data. Format strictly as JSON with: user_name, os_version, device_model, issue_type (crash|UI_glitch|latency|feature_request), reproduction_steps."

def test_ollama(img_b64):
    payload = {
        "model": "gemma4:12b",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "keep_alive": 0
    }
    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        return resp.json().get('response', 'Error')
    except Exception as e:
        return f"Error: {e}"

def test_gemson(img_b64):
    payload = {
        "prompt": prompt,
        "image_data": [{"data": img_b64, "id": 12}], # llama.cpp expects specific format for images sometimes, or we can use the OAI compatible endpoint?
        # Actually, let's use the standard completion endpoint that llama.cpp provides.
    }
    # Wait, the OpenAI vision format for llama.cpp is better:
    payload_oai = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ]
    }
    try:
        resp = requests.post("http://localhost:8080/v1/chat/completions", json=payload_oai, timeout=120)
        choices = resp.json().get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', 'Error')
        return str(resp.json())
    except Exception as e:
        return f"Error: {e}"

def main():
    print("Waiting for servers to boot...")
    time.sleep(5) # Give llama.cpp a sec to load weights
    
    for name, path in images.items():
        print(f"\n=====================================")
        print(f"Testing Image: {name} ({path})")
        print(f"=====================================")
        img_b64 = encode_image(path)
        
        print("\n--- Baseline (Gemma-4-12B via Ollama) ---")
        baseline_res = test_ollama(img_b64)
        print(baseline_res)
        
        print("\n--- Fine-Tuned (Gemson-12B via llama.cpp) ---")
        gemson_res = test_gemson(img_b64)
        print(gemson_res)

if __name__ == "__main__":
    main()
