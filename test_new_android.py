import requests
import json
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = "http://127.0.0.1:8080/api/generate"
payload = {
    "model": "gemson-12b",
    "prompt": "Extract bug data. Format strictly as JSON with: user_name, os_version, device_model, issue_type (crash|UI_glitch|latency|feature_request), reproduction_steps.",
    "images": [encode_image("data/images/android_crash.png")],
    "stream": False,
    "keep_alive": 0
}

response = requests.post(url, json=payload)
print(response.json().get('response', 'Error'))
