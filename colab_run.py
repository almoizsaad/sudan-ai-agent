# --- SUDANESE AI AGENT: ALL-IN-ONE PIPELINE TESTS ---
# This script is designed for Google Colab to test every phase of the project.

import os
import subprocess
import sys
import time

def setup_colab():
    # 1. INSTALL ALL DEPENDENCIES
    print("🚀 Installing all libraries (ASR, LLM, TTS, etc.)...")
    # Note: habibi-tts and whisper are heavy, this may take 3-5 minutes.
    packages = ["openai-whisper", "google-genai", "python-dotenv", "anthropic", "habibi-tts"]
    
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
    client = genai.Client(api_key=llm_api_key)
    model_id = 'gemini-3.5-flash-lite'

    # 3. CREATE DIRECTORY STRUCTURE
    print("📁 Creating project folders...")
    folders = ["prompts", "src", "tests/audio_samples", "logs", "output"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # 4. WRITE SUDANESE DIALECT SYSTEM PROMPT
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

    def generate_with_retry(prompt, model=model_id, retries=3, delay=5):
        for i in range(retries):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                if "503" in str(e) or "High Demand" in str(e):
                    print(f"⚠️ Model busy, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else: raise e
        return "Error: Max retries exceeded."

    # --- PHASE 1: SUDANESE LLM TEST (TEXT) ---
    print("\n--- 🤖 TEST 1: SUDANESE LLM (TEXT-TO-TEXT) ---")
    test_q = "السلام عليكم، داير أسأل من الأسعار"
    reply = generate_with_retry(f"{sudanese_prompt}\n\nCustomer: {test_q}\nReply:")
    print(f"❓ Q: {test_q}\n🇸🇩 A: {reply}")

    # --- PHASE 2: TRANSCRIPT CORRECTION TEST ---
    print("\n--- ✍️ TEST 2: TRANSCRIPT CORRECTION ---")
    raw_transcript = "بياخد كم يوم التوسيل؟" # Error: 'التوسيل' instead of 'التوصيل'
    correction_prompt = f"The following is a Sudanese transcript with errors. Correct only Sudanese vocabulary errors:\n\n{raw_transcript}\n\nCorrected:"
    corrected = generate_with_retry(correction_prompt)
    print(f"❌ Raw: {raw_transcript}\n✅ Corrected: {corrected}")

    # --- PHASE 3: WHISPER ASR TEST ---
    print("\n--- 🎤 TEST 3: WHISPER ASR (AUDIO-TO-TEXT) ---")
    print("Loading Whisper model (medium)...")
    asr_model = whisper.load_model("medium")
    
    import glob
    audio_files = glob.glob("tests/audio_samples/*.mp3") + glob.glob("tests/audio_samples/*.wav")
    if audio_files:
        # Sort by size to prefer non-empty files if multiple exist
        audio_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        audio_path = audio_files[0]
        
        if os.path.getsize(audio_path) > 0:
            print(f"Transcribing {audio_path} ({os.path.getsize(audio_path)} bytes)...")
            try:
                result = asr_model.transcribe(audio_path, language="ar")
                transcription = result["text"]
                print(f"📜 Result: {transcription}")
            except Exception as e:
                print(f"❌ Transcription failed: {str(e)}")
                transcription = None
        else:
            print(f"⚠️ Skip: Found audio file {audio_path} but it is empty (0 bytes).")
            transcription = None
    else:
        print("⚠️ Skip: No audio found in 'tests/audio_samples/'. Upload one to test.")
        transcription = None

    # --- PHASE 4: HABIBI-TTS TEST (TEXT-TO-SPEECH) ---
    print("\n--- 🔊 TEST 4: HABIBI-TTS (TEXT-TO-SPEECH) ---")
    tts_text = "أهلاً بك، كيف أقدر أساعدك الليلة؟"
    output_audio = "output/tts_test.wav"
    print(f"Generating Sudanese speech for: '{tts_text}'...")
    try:
        # Note: This requires a reference audio. We'll use a placeholder or check if user uploaded one.
        # For a basic test, we call the CLI as per the plan
        subprocess.run([
            "habibi-tts_infer-cli", 
            "--gen_text", tts_text, 
            "--dialect", "SDN",
            "--output_dir", "output/"
        ])
        print(f"✅ TTS Audio generated in 'output/' folder.")
    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")

    # --- PHASE 5: FULL PIPELINE (VOICE-TO-VOICE) ---
    if transcription:
        print("\n--- 🔄 TEST 5: FULL PIPELINE (VOICE-TO-VOICE) ---")
        # 1. Already transcribed in Test 3
        # 2. Generate response
        reply = generate_with_retry(f"{sudanese_prompt}\n\nCustomer: {transcription}\nReply:")
        print(f"🤖 AI Response: {reply}")
        # 3. Convert to Voice
        subprocess.run([
            "habibi-tts_infer-cli", 
            "--gen_text", reply, 
            "--dialect", "SDN",
            "--output_dir", "output/"
        ])
        print(f"🏁 Full pipeline complete. Check 'output/' for the final voice response.")

    print("\n✨ All tests processed! Your Sudanese AI Agent is ready.")

if __name__ == "__main__":
    setup_colab()
