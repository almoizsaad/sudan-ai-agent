import os
from google import genai
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
client = genai.Client(api_key=api_key)

# Using gemini-3.5-flash-lite as the stable standard for 2026
model_id = 'gemini-3.5-flash-lite'

with open("prompts/sudan_dialect_system.txt", encoding="utf-8") as f:
    system_prompt = f.read()

test_questions = [
    "يا زول، السلام عليكم، داير أسأل من الأسعار",
    "ممكن أشوف المنتج قبل ما أشتريه؟",
    "في ضمان على الجهاز؟",
]

def generate_with_retry(prompt, model_id, retries=3, delay=5):
    for i in range(retries):
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return response.text
        except Exception as e:
            if "503" in str(e) or "High Demand" in str(e):
                print(f"⚠️ Model busy (503), retrying in {delay}s... ({i+1}/{retries})")
                time.sleep(delay)
                delay *= 2 
            else:
                raise e
    raise Exception("Max retries exceeded for model.")

for q in test_questions:
    prompt = f"{system_prompt}\n\nCustomer: {q}\nReply:"
    try:
        reply = generate_with_retry(prompt, model_id)
        print(f"Question: {q}")
        print(f"Reply: {reply}")
        print("-" * 40)
    except Exception as e:
        print(f"Error testing LLM: {str(e)}")
