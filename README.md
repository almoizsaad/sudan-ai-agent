# Sudanese AI Agent

An automated customer service agent specialized in the Sudanese Arabic dialect (Khartoum colloquial). This agent supports both text and voice interactions, integrating speech-to-text (Whisper) and generative AI (Gemini).

## Features
- **Sudanese Dialect Support:** Fine-tuned system prompts for natural Sudanese Arabic.
- **ASR (Speech-to-Text):** Integrated with OpenAI Whisper for voice message transcription.
- **FastAPI Webhook:** Ready to be connected to the WhatsApp Business API.
- **Unified Pipeline:** Seamlessly connects voice input to dialect-aware LLM responses.

## Structure
- `src/`: Core logic and integration scripts.
- `prompts/`: System instructions and dialect examples.
- `tests/`: Directory for audio samples and unit tests.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python3 -m venv venv`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Configure `.env` with your `LLM_API_KEY` (Gemini).

## Usage
- Test LLM: `python src/test_llm.py`
- Test ASR: `python src/test_asr.py`
- Run Webhook: `uvicorn src.whatsapp_webhook:app --reload`
