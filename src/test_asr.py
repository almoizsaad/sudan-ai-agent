import whisper
import os
import glob

def test_asr():
    # Using 'medium' for better accuracy if available, or 'base' for speed
    model_type = "medium" 
    print(f"Loading Whisper {model_type} model...")
    model = whisper.load_model(model_type)
    
    # Robust audio file detection
    audio_files = glob.glob("tests/audio_samples/*.mp3") + glob.glob("tests/audio_samples/*.wav")
    
    if not audio_files:
        print("Error: No audio samples found in 'tests/audio_samples/'.")
        print("Please upload a Sudanese Arabic audio file (.mp3 or .wav) to that folder.")
        return

    # Prefer non-empty files
    audio_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    audio_path = audio_files[0]
    
    if os.path.getsize(audio_path) == 0:
        print(f"Error: The largest audio file found ({audio_path}) is empty (0 bytes).")
        return

    print(f"Transcribing {audio_path} ({os.path.getsize(audio_path)} bytes)...")
    try:
        # Use CPU if no GPU detected, Whisper handles this internally but we warn
        result = model.transcribe(audio_path, language="ar")
        print("\n--- Transcription Result ---")
        print(result["text"])
        print("-" * 28)
    except Exception as e:
        print(f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    test_asr()
