import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from dotenv import load_dotenv
import requests

try:  # Streamlit is only available in the app runtime
    import streamlit as st
except Exception:  # pragma: no cover - non-Streamlit contexts
    st = None

# Ensure .env is loaded from the project root, regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _get_deepseek_config() -> tuple[Optional[str], str, str]:
    """
    Resolve API config from environment and, when available, Streamlit secrets.
    Streamlit Cloud users should define these in the app's Secrets / Environment.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    endpoint = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if st is not None:
        try:
            secrets = st.secrets  # type: ignore[attr-defined]
        except Exception:
            secrets = None
        if secrets:
            api_key = secrets.get("DEEPSEEK_API_KEY", api_key)
            endpoint = secrets.get("DEEPSEEK_ENDPOINT", endpoint)
            model = secrets.get("DEEPSEEK_MODEL", model)

    return api_key, endpoint, model


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

    api_key, endpoint, model = _get_deepseek_config()

    if not api_key:
        return _local_fallback_response(user_text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Smart Healthcare Analytics",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        status = getattr(http_err.response, "status_code", "unknown")
        body = ""
        try:
            body = http_err.response.text
        except Exception:
            body = ""
        debug_msg = f" [API error {status}: {body}]"
        return _local_fallback_response(user_text) + debug_msg
    except Exception as exc:
        debug_msg = f" [Network or other error: {exc}]"
        return _local_fallback_response(user_text) + debug_msg


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
