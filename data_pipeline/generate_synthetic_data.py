import json
import os
from openai import OpenAI
from schema import BugReport

# You would point this to Gemini 1.5 Pro or GPT-4o
client = OpenAI(api_key=os.environ.get("TEACHER_MODEL_API_KEY"))

PROMPT = """
You are a data generator. Generate a realistic, highly conversational, and messy 
customer support transcript where a user is complaining about a software bug. 
Then, output the strictly formatted JSON extracting the data according to the schema.

Format your output exactly like this:
---TRANSCRIPT---
[Messy conversational text here]
---JSON---
[Strict JSON here]
"""

def generate_example():
    response = client.chat.completions.create(
        model="gpt-4o", # Or gemini-1.5-pro
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.9 # High temperature for diverse, weird transcripts
    )
    
    # Parse the response into our required chat format
    raw_text = response.choices[0].message.content
    transcript, json_data = raw_text.split("---JSON---")
    transcript = transcript.replace("---TRANSCRIPT---", "").strip()
    
    # Format exactly as Gemma expects it
    return {
        "messages": [
            {"role": "user", "content": f"Extract bug data from this transcript:\n\n{transcript}"},
            {"role": "assistant", "content": json_data.strip()}
        ]
    }

if __name__ == "__main__":
    print("Generating synthetic data...")
    # Generate 5 examples (Loop this to 500-1000 in production)
    dataset = [generate_example() for _ in range(5)]
    
    with open("training_data.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print("Dataset generation complete!")
