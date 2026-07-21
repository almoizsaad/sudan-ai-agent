import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
genai.configure(api_key=api_key)

# Using gemini-2.0-flash for testing
model = genai.GenerativeModel('gemini-2.0-flash')

with open("prompts/sudan_dialect_system.txt", encoding="utf-8") as f:
    system_prompt = f.read()

test_questions = [
    "يا زول، السلام عليكم، داير أسأل من الأسعار",
]

for q in test_questions:
    prompt = f"{system_prompt}\n\nCustomer: {q}\nReply:"
    response = model.generate_content(prompt)
    
    print(f"Question: {q}")
    print(f"Reply: {response.text}")
    print("-" * 40)
