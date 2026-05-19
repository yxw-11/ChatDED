import os

import requests
from google import genai


def _get_openai_api_key() -> str:
    """
    Read OpenAI API key from environment variables.
    Preferred: OPENAI_API_KEY
    Backward-compatible fallback: API_KEY
    """
    return os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""


def request_gpt(prompt, api_key, model="gpt-5-mini", temperature=1):
    if not api_key:
        return ""

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an ophthalmology expert specializing in orthokeratology (OK) lens fitting and follow-up."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=100)

        if response.status_code == 200:
            return response.json()
        else:
            return response.text
    except:
        return ""


def dp_answer(prompt):
    api_key = _get_openai_api_key()
    response = request_gpt(prompt, api_key, "gpt-5")
    try:
        raw_text = response['choices'][0]['message']['content']
    except:
        raw_text = ""
    return raw_text


def request_gemini(content):
    client = genai.Client(api_key="Your_Gemini_Key")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=content
        )
        return(response.text)
    except:
        return "no result"
