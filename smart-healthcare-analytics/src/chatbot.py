import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from dotenv import load_dotenv
import requests

# Ensure .env is loaded from the project root, regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _local_fallback_response(user_text: str) -> str:
    """Return an on-device educational response when the API is unavailable."""
    user_text = user_text or ""
    concerns = []
    lowered = user_text.lower()
    if any(word in lowered for word in ["diet", "food", "nutrition"]):
        concerns.append("Keep meals balanced with vegetables, lean protein, and whole grains.")
    if any(word in lowered for word in ["exercise", "workout", "activity"]):
        concerns.append("Aim for 150 minutes of moderate activity per week and include strength work twice weekly.")
    if any(word in lowered for word in ["blood", "pressure", "hypertension"]):
        concerns.append("Monitor blood pressure regularly and limit salt intake to under 5g per day.")
    if any(word in lowered for word in ["glucose", "sugar", "diabetes"]):
        concerns.append("Track fasting glucose, stay hydrated, and space carbohydrates across meals.")
    if not concerns:
        concerns.append("Maintain regular checkups, follow medication guidance, and contact a clinician if symptoms worsen.")

    return (
        "I'm providing a local educational response. "
        + " ".join(concerns)
        + " This assistant cannot deliver medical diagnoses; please consult your healthcare professional for personal advice."
    )


def _call_deepseek(messages: List[Dict[str, str]]) -> str:
    user_text = next((msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"), "")

    if not DEEPSEEK_API_KEY:
        return _local_fallback_response(user_text)

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
    except Exception:
        return _local_fallback_response(user_text)


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
