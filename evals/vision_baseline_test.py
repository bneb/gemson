import base64
import requests
import json
import time

IMAGES = ["data/images/404_error.png", "data/images/bsod_crash.png", "data/images/android_crash.png"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_vision():
    url = "http://localhost:11434/api/generate"
    
    print("========================================")
    print("BASELINE VISION TEST (Ollama Native API)")
    print("========================================")
    
    for img_path in IMAGES:
        print(f"\nQuerying Gemma-4-12B for {img_path}...")
        base64_image = encode_image(img_path)
        
        payload = {
            "model": "gemma4:12b",
            "prompt": "Extract bug data from this image. Format strictly as JSON with: user_name, os_version, device_model, issue_type (crash|UI_glitch|latency|feature_request), reproduction_steps.",
            "images": [base64_image],
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        try:
            # high timeout because we are queuing behind the 50-sample qualitative test
            response = requests.post(url, json=payload, timeout=6000)
            if response.status_code == 200:
                print(f"[Baseline Output for {img_path}]\n{response.json().get('response', '')}\n")
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print("ERROR:", e)

if __name__ == "__main__":
    test_vision()
