import os
import json
import base64
import requests

IMAGES = {
    "404_error": "data/images/404_error.png",
    "bsod_crash": "data/images/bsod_crash.png",
    "android_crash": "data/images/android_crash.png"
}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

print("\n========================================")
print("MULTIMODAL A/B TEST: IMAGE-TO-JSON")
print("========================================")

for name, path in IMAGES.items():
    print(f"\n--- Evaluating Image: {name}.png ---")
    
    print("\n[Baseline Model: Gemma-4-12B]")
    baseline = query_gemma(path)
    print(baseline)
    
    print("\n[Fine-Tuned Model: Gemson-12B]")
    finetune = query_gemson(path)
    print(finetune)
    print("\n" + "="*40)
