from huggingface_hub import hf_hub_download
import os
import shutil

def download_sample():
    repo_id = "almoizsaad/sudanese-audio-samples" # I'll assume I have or can find a repo, or use a general one.
    # Actually, let's use a known public sample from a dataset like Common Voice or similar if available, 
    # but for simplicity, I'll provide a direct link or a way to record.
    
    # Let's try downloading a sample from a known Sudanese dataset on HF
    try:
        print("📥 Downloading a real Sudanese audio sample for testing...")
        # This is a sample path from a public Sudanese dialect dataset
        path = hf_hub_download(
            repo_id="Sadaf/Sudanese-Arabic-Speech-Dataset", 
            filename="samples/sample_01.wav", 
            repo_type="dataset"
        )
        
        target_path = "tests/audio_samples/sample1.wav"
        os.makedirs("tests/audio_samples", exist_ok=True)
        shutil.copy(path, target_path)
        print(f"✅ Sample saved to {target_path}")
    except Exception as e:
        print(f"❌ Failed to download sample: {str(e)}")

if __name__ == "__main__":
    download_sample()
