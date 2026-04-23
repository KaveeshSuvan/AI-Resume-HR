"""
AI engine — calls Ollama (local LLM) with timeout and error handling.
"""

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b-cloud"
TIMEOUT_SECONDS = 120          # generous for a 20B model


def ask_ai(prompt: str) -> str:
    """Send a prompt to Ollama and return the text response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"AI model timed out after {TIMEOUT_SECONDS}s")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Is it running on localhost:11434?")
    except Exception as e:
        raise RuntimeError(f"AI engine error: {e}")


def ask_ai_json(prompt: str) -> dict:
    """Ask AI and try to parse the response as JSON."""
    raw = ask_ai(prompt)
    # Try to find JSON in the response
    try:
        # Sometimes the model wraps JSON in markdown code blocks
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, IndexError):
        return {"raw": raw}