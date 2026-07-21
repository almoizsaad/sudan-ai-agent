import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
genai.configure(api_key=api_key)

def correct_sudanese_transcript(raw_text):
    # Using gemini-2.0-flash which might have separate quota
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""The following text is automatically transcribed from Sudanese colloquial Arabic and may contain errors. 
    Correct only obvious errors in Sudanese vocabulary or grammar without changing the meaning or dialect:

    Text: {raw_text}

    Corrected Text:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during correction: {str(e)}"

if __name__ == "__main__":
    raw = "كيف حالك يا زول؟ بياخد كم يوم التوسيل؟" # "التوسيل" instead of "التوصيل"
    print(f"Raw: {raw}")
    print(f"Corrected: {correct_sudanese_transcript(raw)}")
