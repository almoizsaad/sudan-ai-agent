from fastapi import FastAPI, Request
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure Gemini
api_key = os.getenv("LLM_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

with open("prompts/sudan_dialect_system.txt", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")

@app.get("/webhook")
async def verify(request: Request):
    # Meta verification
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == "sudan_agent_token":
        return int(params.get("hub.challenge", 0))
    return "Verification failed"

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        from_number = msg["from"]
        
        if "text" in msg:
            text = msg["text"]["body"]
            print(f"Received text: {text}")
            
            prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {text}\nReply:"
            response = model.generate_content(prompt)
            reply = response.text
            
            # Send reply back via WhatsApp API
            url = f"https://graph.facebook.com/v19.0/{PHONE_ID}/messages"
            headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
            json_data = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "text": {"body": reply}
            }
            requests.post(url, headers=headers, json=json_data)
            
        elif "audio" in msg:
            # Handle audio messages (integration with pipeline.py)
            pass
            
    except (KeyError, IndexError) as e:
        print(f"Error processing message: {e}")
        
    return {"status": "ok"}
