# --- SUDANESE AI AGENT: COLAB ALL-IN-ONE SETUP & TEST ---
# This script is designed to be run in a Google Colab notebook cell.

import os
import subprocess
import sys

def setup_colab():
    # 1. INSTALL DEPENDENCIES
    print("🚀 Installing libraries (Whisper, Google GenAI, etc.)...")
    packages = ["openai-whisper", "google-genai", "python-dotenv", "anthropic"]
    
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + packages)
        import importlib
        importlib.invalidate_caches()
    else:
        print("Note: Not running in Colab environment. Ensure dependencies are installed manually.")

    try:
        from google import genai
    except ImportError:
        print("❌ Failed to import 'google-genai'. Please restart the session and try again.")
        return

    import whisper
    
    # 2. SETUP API KEYS
    llm_api_key = os.environ.get("LLM_API_KEY")
    
    if not llm_api_key:
        if is_colab:
            from google.colab import userdata
            try:
                llm_api_key = userdata.get('LLM_API_KEY')
                print("✅ Found LLM_API_KEY in Colab Secrets.")
            except:
                from getpass import getpass
                llm_api_key = getpass("🔑 Please enter your Gemini API Key: ")
        else:
            from getpass import getpass
            llm_api_key = getpass("🔑 Please enter your Gemini API Key: ")

    os.environ["LLM_API_KEY"] = llm_api_key

    # 3. CREATE DIRECTORY STRUCTURE
    print("📁 Creating project folders...")
    folders = ["prompts", "src", "tests/audio_samples", "logs"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # 4. WRITE SUDANESE DIALECT SYSTEM PROMPT
    print("📝 Writing system prompt...")
    sudanese_prompt = """
You are a Sudanese customer service representative speaking in the colloquial Sudanese Arabic of Khartoum, in a friendly and direct manner.

Follow these examples exactly:

Example 1:
Customer: كم يوم بياخد التوصيل؟
Reply: التوصيل داخل الخرطوم بياخد يوم لليومين بالكتير. برا الخرطوم بياخد زمن أطول شوية.

Example 2:
Customer: السعر ده شامل التوصيل؟
Reply: أيوه شامل، مافي أي رسوم إضافية. السعر ده نهائي.

Example 3:
Customer: في نظام تقسيط؟
Reply: للأسف حالياً ما متوفر، لكن شغالين عليهو وقريب إن شاء الله حيكون متاح.

Rules:
- Avoid using overly formal Arabic. Keep your speech natural, like a normal conversation.
- If you don't have an answer to a question, be honest and say: "والله حالياً ما عندي معلومة أكيدة، خليني أتأكد وأرجع ليك."
- Keep your answers short and direct.
"""

    with open("prompts/sudan_dialect_system.txt", "w", encoding="utf-8") as f:
        f.write(sudanese_prompt)

    # 5. WRITE .ENV FILE
    with open(".env", "w") as f:
        f.write(f"LLM_API_KEY={llm_api_key}\n")

    # 6. RUN PHASE 1: LLM TEST
    print("\n--- 🤖 TESTING SUDANESE LLM ---")
    client = genai.Client(api_key=llm_api_key)
    # Using gemini-3.5-flash as the stable standard for 2026
    model_id = 'gemini-3.5-flash'

    test_questions = [
        "السلام عليكم، الأسعار عندكم كم؟",
        "عندكم توصيل لولايات تانية؟",
        "ممكن أرجع المنتج لو ما عجبني؟"
    ]

    for q in test_questions:
        prompt = f"{sudanese_prompt}\n\nCustomer: {q}\nReply:"
        try:
            # Explicitly setting the model ID
            response = client.models.generate_content(model=model_id, contents=prompt)
            print(f"❓ Q: {q}")
            print(f"🇸🇩 A: {response.text}")
            print("-" * 30)
        except Exception as e:
            print(f"❌ Error testing LLM ({model_id}): {str(e)}")

    # 7. RUN PHASE 2: WHISPER ASR TEST
    print("\n--- 🎤 TESTING WHISPER ASR ---")
    print("Loading Whisper model (this may take a minute)...")
    asr_model = whisper.load_model("medium")

    # Check for audio sample
    audio_path = "tests/audio_samples/sample1.mp3"
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        print(f"Transcribing {audio_path}...")
        result = asr_model.transcribe(audio_path, language="ar")
        print("\n📜 Transcribed Text:")
        print(result["text"])
    else:
        print("⚠️ No audio sample found in 'tests/audio_samples/sample1.mp3'.")
        print("💡 Upload an Arabic audio file to that folder and re-run this setup to test ASR.")

    print("\n✨ Setup Complete! Your project is ready for development.")

if __name__ == "__main__":
    setup_colab()
