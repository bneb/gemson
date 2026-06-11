import json
import requests

def query_ollama(prompt, model="gemson-12b:latest"):
    url = "http://localhost:11434/api/generate"
    
    system_instruction = (
        "You are a recipe extraction assistant. "
        "Extract the recipe into the following JSON schema: "
        "{\"recipe_name\": string, \"ingredients\": list[string], \"instructions\": list[string]}\n\n"
    )
    
    payload = {
        "model": model,
        "prompt": system_instruction + prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json().get("response", "")
    except Exception as e:
        return f"Error: {e}"

recipe_transcript = """
User: Hey, I want to make chocolate chip cookies.
Support: Sure! You'll need 1 cup butter, 1 cup sugar, 2 eggs, and 2 cups of chocolate chips.
User: Awesome. And how do I bake it?
Support: First, preheat your oven to 350 degrees. Then mix all the wet ingredients together. After that, slowly stir in the dry ingredients and finally fold in the chocolate chips. Bake for 10-12 minutes!
"""

print("Testing Gemson with a Cooking Recipe Transcript...")
print(f"Transcript:\n{recipe_transcript}")
print("-" * 40)

result = query_ollama(recipe_transcript)
print(f"Output:\n{result}")
