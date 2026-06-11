import os
import requests
import json
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def demo():
    print("==============================================")
    print("🚀 GEMSON-12B DEMO: Automated Bug Triage")
    print("==============================================\n")
    
    url = "http://127.0.0.1:8000/v1/extract"
    api_key = os.getenv("GEMSON_API_KEY", "dev")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Example 1: Text Transcript Extraction
    print("--- [Test 1: Messy Text Transcript] ---")
    transcript = """
    User: hey my app is broken. I opened it and tried to go to the settings page, but it just completely froze and then closed. 
    It's an iPhone 15 Pro on iOS 17.2. Please fix it!
    """
    print(f"Input: {transcript.strip()}")
    
    payload_text = {
        "messages": [
            {"role": "system", "content": "Extract bug data. Format strictly as JSON with: user_name, os_version, device_model, issue_type (crash|UI_glitch|latency|feature_request), reproduction_steps."},
            {"role": "user", "content": transcript}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload_text, timeout=30)
        if response.status_code == 200:
            print(f"Gemson Extracted Output:\n{json.dumps(response.json(), indent=2)}\n")
        else:
            print(f"Error: Server returned {response.status_code} - {response.text}\n")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the Rust gateway. Did you run `make serve-gateway` and `make serve-model`?\n")

if __name__ == "__main__":
    demo()
