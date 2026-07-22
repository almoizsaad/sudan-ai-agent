# --- SUDANESE AI AGENT: ALL-IN-ONE PIPELINE TESTS ---
# This script is designed for Google Colab to test every phase of the project.

import os
import subprocess
import sys
import time

def setup_colab():
    # 0. DIRECTORY FIX: Ensure we are in the right place
    if os.path.exists("/content/sudan-ai-agent"):
        os.chdir("/content/sudan-ai-agent")
        
    # 1. INSTALL ALL DEPENDENCIES
    print("🚀 Installing all libraries (ASR, LLM, TTS, etc.)...")
    
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if is_colab:
        # 0.5 CLEANUP: Remove zero-byte placeholders to prevent Test 3 skips
        placeholder = "tests/audio_samples/sample1.mp3"
        if os.path.exists(placeholder) and os.path.getsize(placeholder) == 0:
            os.remove(placeholder)

        # HARD FIX: Pin versions to 2.4.1 stack using the official stable index
        print("⚙️ Installing verified audio/torch stack (2.4.1)...")
        # Added --no-warn-conflicts and >/dev/null to clean up output
        subprocess.run(
            f"{sys.executable} -m pip install -q -U --force-reinstall --no-warn-conflicts "
            "torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 "
            "--index-url https://download.pytorch.org/whl/cu121 >/dev/null 2>&1",
            shell=True
        )
        
        print("🚀 Installing project libraries...")
        packages = ["openai-whisper", "google-genai", "python-dotenv", "anthropic", "habibi-tts"]
        subprocess.run(
            f"{sys.executable} -m pip install -q -U --no-warn-conflicts {' '.join(packages)} >/dev/null 2>&1",
            shell=True
        )
        
        import importlib
        importlib.invalidate_caches()
        
        # Verify versions
        try:
            import torch
            import torchaudio
            print(f"✅ Torch: {torch.__version__} | Audio: {torchaudio.__version__}")
            if torch.cuda.is_available():
                print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ WARNING: No GPU detected.")
        except Exception as e:
            print(f"⚠️ Version check failed: {e}")
    else:
        print("Note: Not running in Colab environment.")

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
You are a Sudanese customer service representative speaking in the colloquial Sudanese Arabic of Khartoum. 

Rules:
- Speak in a friendly, direct, and helpful manner.
- ALWAYS use natural colloquial Sudanese (Khartoum dialect). 
- AVOID formal Arabic grammar and formal marks (like hamzas on 'alif' in verbs, e.g., use 'بياخد' not 'بيأخد').
- Keep responses short and avoid being overly wordy.
- If you are correcting a transcript, keep it in the natural spoken dialect of Sudan.

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
    tts_text = "حبابك عشرة، كيف أقدر أساعدك الليلة؟"
    output_audio = "output/tts_test.wav"
    print(f"Generating Sudanese speech for: '{tts_text}'...")
    try:
        # Try to find a reference audio
        import glob
        ref_audio = None
        ref_text = None
        if os.path.exists("tests/audio_samples/voice.wav"):
            ref_audio = "tests/audio_samples/voice.wav"
        else:
            samples = glob.glob("tests/audio_samples/*.wav") + glob.glob("tests/audio_samples/*.mp3")
            for s in samples:
                if os.path.getsize(s) > 0:
                    ref_audio = s
                    break

        if ref_audio:
            txt_path = os.path.splitext(ref_audio)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()

        cmd = [
            "habibi-tts_infer-cli", 
            "--gen_text", tts_text, 
            "--dialect", "SDN",
            "--output_dir", "output/"
        ]
        if ref_audio:
            print(f"Using reference audio: {ref_audio}")
            cmd.extend(["--ref_audio", ref_audio])
            if ref_text:
                print(f"Using reference text: {ref_text}")
                cmd.extend(["--ref_text", ref_text])
            else:
                print("⚠️ Warning: No reference text found. Habibi-TTS works best with --ref_text.")

        subprocess.run(cmd)
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
        cmd = [
            "habibi-tts_infer-cli", 
            "--gen_text", reply, 
            "--dialect", "SDN",
            "--output_dir", "output/"
        ]
        if ref_audio:
            cmd.extend(["--ref_audio", ref_audio])
            if ref_text:
                cmd.extend(["--ref_text", ref_text])
            
        subprocess.run(cmd)
        print(f"🏁 Full pipeline complete. Check 'output/' for the final voice response.")

    print("\n✨ All tests processed! Your Sudanese AI Agent is ready.")

if __name__ == "__main__":
    setup_colab()
