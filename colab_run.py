# --- SUDANESE AI AGENT: COLAB ALL-IN-ONE SETUP & TEST ---
# This script is designed to be run in a Google Colab notebook cell.

import os
import subprocess
import sys

def setup_colab():
    # 1. INSTALL DEPENDENCIES
    print("🚀 Installing libraries (Whisper, Google GenAI, etc.)...")
    packages = ["openai-whisper", "google-genai", "python-dotenv", "anthropic"]
    # ... (check if is_colab)
    if is_colab:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + packages)
    else:
        print("Note: Not running in Colab environment. Ensure dependencies are installed manually.")

    from genai import Client
    import whisper
    
    # ... (setup API keys)
    
    # 6. RUN PHASE 1: LLM TEST
    print("\n--- 🤖 TESTING SUDANESE LLM ---")
    client = Client(api_key=llm_api_key)
    # Using gemini-1.5-flash for better free-tier availability
    model_id = 'gemini-1.5-flash'

    test_questions = [
        "السلام عليكم، الأسعار عندكم كم؟",
        "عندكم توصيل لولايات تانية؟",
        "ممكن أرجع المنتج لو ما عجبني؟"
    ]

    for q in test_questions:
        prompt = f"{sudanese_prompt}\n\nCustomer: {q}\nReply:"
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            print(f"❓ Q: {q}")
            print(f"🇸🇩 A: {response.text}")
            print("-" * 30)
        except Exception as e:
            print(f"❌ Error testing LLM: {str(e)}")

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
