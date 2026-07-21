import os
from google import genai
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("LLM_API_KEY")
client = genai.Client(api_key=api_key)

with open("prompts/sudan_dialect_system.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# Models to test
models_to_try = [
    'gemini-3.5-flash-lite',
    'gemini-2.0-flash-lite',
    'gemini-3.6-flash',
    'gemini-1.5-flash',
]

test_q = "السلام عليكم، داير أسأل من الأسعار"

print("🔍 Testing models locally...")

for model_id in models_to_try:
    print(f"\n--- Testing {model_id} ---")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=f"{system_prompt}\n\nCustomer: {test_q}\nReply:"
        )
        print(f"✅ Success!")
        print(f"🇸🇩 {response.text[:100]}...")
        # If we find a working one, we can stop or keep checking
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    time.sleep(1) # Small delay to be polite to the API
