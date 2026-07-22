import os
from google import genai
import whisper
import subprocess
from dotenv import load_dotenv

load_dotenv()

class SudaneseAIPipeline:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("LLM_API_KEY"))
        self.model_id = 'gemini-3.5-flash-lite'
        self.whisper_model = None
        
        with open("prompts/sudan_dialect_system.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def transcribe(self, audio_path):
        if self.whisper_model is None:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("medium")
        
        print(f"Transcribing {audio_path}...")
        result = self.whisper_model.transcribe(audio_path, language="ar")
        return result["text"]

    def generate_response(self, text):
        prompt = f"{self.system_prompt}\n\nCustomer: {text}\nReply:"
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt
        )
        return response.text

    def text_to_speech(self, text, output_path="output/response.wav"):
        print(f"Generating speech: {text}")
        
        # Try to find a reference audio in tests/audio_samples/
        import glob
        ref_audio = None
        ref_text = None
        
        # Prefer 'voice.wav' if recorded by user
        if os.path.exists("tests/audio_samples/voice.wav"):
            ref_audio = "tests/audio_samples/voice.wav"
        else:
            samples = glob.glob("tests/audio_samples/*.wav") + glob.glob("tests/audio_samples/*.mp3")
            for s in samples:
                if os.path.getsize(s) > 0:
                    ref_audio = s
                    break
        
        if ref_audio:
            # Look for a matching .txt file for ref_text
            txt_path = os.path.splitext(ref_audio)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
            
        cmd = [
            "habibi-tts_infer-cli",
            "--gen_text", text,
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
        return output_path

    def process_voice(self, input_audio):
        # 1. Audio -> Text
        transcript = self.transcribe(input_audio)
        print(f"Transcript: {transcript}")
        
        # 2. Text -> Sudanese AI Response
        reply_text = self.generate_response(transcript)
        print(f"AI Reply: {reply_text}")
        
        # 3. Reply -> Audio
        output_audio = self.text_to_speech(reply_text)
        
        return {
            "transcript": transcript,
            "reply_text": reply_text,
            "output_audio": output_audio
        }

if __name__ == "__main__":
    # Test locally if file exists
    pipeline = SudaneseAIPipeline()
    import glob
    samples = glob.glob("tests/audio_samples/*.wav") + glob.glob("tests/audio_samples/*.mp3")
    if samples:
        res = pipeline.process_voice(samples[0])
        print("Success:", res)
    else:
        print("No audio sample found for local pipeline test.")
