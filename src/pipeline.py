import os
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
import subprocess

load_dotenv()

# Configure Gemini
api_key = os.getenv("LLM_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

# Load Whisper (using base for speed/memory on Cloud Shell)
whisper_model = whisper.load_model("base")

with open("prompts/sudan_dialect_system.txt", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

def process_voice_message(audio_path):
    # 1. Audio Transcription
    print(f"Transcribing {audio_path}...")
    transcript = whisper_model.transcribe(audio_path, language="ar")["text"]
    print(f"Transcript: {transcript}")

    # 2. Response Generation
    print("Generating response...")
    prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {transcript}\nReply:"
    response = model.generate_content(prompt)
    reply_text = response.text
    print(f"Reply: {reply_text}")

    # 3. Convert Reply to Voice (using habibi-tts if installed)
    # Note: This part assumes habibi-tts is in the PATH
    output_audio = "logs/reply.wav"
    # subprocess.run(["habibi-tts_infer-cli", "--gen_text", reply_text, "--dialect", "SDN", "--out_path", output_audio])

    return {"input_transcript": transcript, "reply_text": reply_text, "output_audio": output_audio}

if __name__ == "__main__":
    sample_audio = "tests/audio_samples/sample1.wav"
    if os.path.exists(sample_audio):
        result = process_voice_message(sample_audio)
        print(result)
    else:
        print(f"Sample audio {sample_audio} not found.")
