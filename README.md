# Sudanese AI Agent 🇸🇩

An interactive AI agent designed to understand and speak the Sudanese Arabic dialect (Khartoum colloquial). 

This project integrates:
1.  **ASR (Whisper):** Voice-to-Text transcription.
2.  **LLM (Gemini 3.5 Flash Lite):** Sudanese dialect processing and customer service intelligence.
3.  **TTS (Habibi-TTS):** Text-to-Speech specifically for Arabic dialects.

---

## 🚀 Quick Start on Google Colab

The easiest way to run this agent is using the **T4 GPU** on Google Colab.

### 1. Initial Setup
```python
%cd /content/
!git clone https://github.com/almoizsaad/sudan-ai-agent.git
%cd sudan-ai-agent
```

### 2. Record Your Voice (Microphone)
Run this cell in Colab to record yourself speaking Sudanese:
```python
from google.colab import output
from IPython.display import HTML, display
from base64 import b64decode

def record_audio(filename='tests/audio_samples/voice.wav'):
  js = """
    async def recordAudio() {
      const div = document.createElement('div');
      const button = document.createElement('button');
      button.textContent = '🔴 Click to Record Sudanese Arabic';
      button.style.padding = '10px'; button.style.fontSize = '20px';
      document.body.appendChild(div); div.appendChild(button);
      const stream = await navigator.mediaDevices.getUserMedia({audio:true});
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.start();
      await new Promise(resolve => button.onclick = resolve);
      recorder.stop(); button.textContent = '✅ Recording Saved';
      await new Promise(resolve => recorder.onstop = resolve);
      const blob = new Blob(chunks);
      const url = URL.createObjectURL(blob);
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      await new Promise(resolve => reader.onloadend = resolve);
      return reader.result;
    }
  """
  display(HTML(f'<script>{js}</script>'))
  data = output.eval_js('recordAudio()')
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  print(f"✅ Saved recording to {filename}")

record_audio()
```

### 3. Run Full Pipeline
Ensure you have added your `LLM_API_KEY` to Colab Secrets.
```python
!python colab_run.py
```

---

## 📂 Project Structure

- `prompts/`: Contains the Sudanese dialect system instructions.
- `src/`: Core logic and individual test scripts.
    - `pipeline.py`: The complete end-to-end class.
    - `test_llm.py`: Tests Sudanese text generation.
    - `test_asr.py`: Tests Whisper audio transcription.
- `tests/audio_samples/`: Folder where you upload/record audio.
- `output/`: Folder where generated Sudanese speech (.wav) is saved.

---

## 🛠 Troubleshooting
- **OSError (Torch):** Fixed by force-installing Torch 2.4.1. The `colab_run.py` script handles this automatically.
- **Empty Transcription:** Ensure you have uploaded a non-zero byte audio file to `tests/audio_samples/`.
- **Quota Limit:** Using `gemini-3.5-flash-lite` for optimal free-tier availability.
