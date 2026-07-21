import whisper
import os

def test_asr():
    model_type = "base" # Using base for faster loading in Cloud Shell
    print(f"Loading Whisper {model_type} model...")
    model = whisper.load_model(model_type)
    
    audio_path = "tests/audio_samples/sample1.mp3"
    
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found. Please provide a sample audio file.")
        return

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path, language="ar")
    print("Extracted text:")
    print(result["text"])

if __name__ == "__main__":
    test_asr()
