import os
from typing import Any, Dict, Optional, List, Tuple

from dotenv import load_dotenv
import requests

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("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")
DEEPSEEK_ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _call_deepseek(messages: List[Dict[str, str]]) -> str:
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API key not configured. Set DEEPSEEK_API_KEY in your .env file."

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
    }
    try:
        response = requests.post(DEEPSEEK_ENDPOINT, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"DeepSeek request failed: {exc}"


def explain_patient_results(input_data: Dict[str, Any], prediction: str, risk_score: Optional[float] = None) -> str:
    prompt = f"""
    You are a medical assistant. Explain this patient's prediction in simple but correct medical terms.

    PATIENT DATA:
    {input_data}

    MODEL PREDICTION:
    {prediction}

    RISK SCORE:
    {risk_score if risk_score is not None else "N/A"}

    Provide:
    - What this risk means
    - Which features contributed the most
    - What the patient should monitor
    - What lifestyle or medical checks are recommended
    """

    return _call_deepseek([{"role": "user", "content": prompt}])


def chat_with_ai(history: List[Tuple[str, str]], user_message: str) -> str:
    messages = [{"role": "system", "content": "You are a supportive medical assistant. Avoid diagnoses; focus on education and guidance."}]
    for role, msg in history:
        messages.append({"role": role, "content": msg})
    messages.append({"role": "user", "content": user_message})

    return _call_deepseek(messages)
