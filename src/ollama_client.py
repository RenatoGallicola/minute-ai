"""
ollama_client.py
----------------
Thin HTTP client for a local Ollama server. Shared by cleanup.py and summarize.py.
"""

import sys
import requests
from src.logger import get_logger


def check_ollama(host: str, model: str) -> bool:
    """Checks that Ollama is running and the model is available."""
    log = get_logger()
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code != 200:
            return False
        models = [m["name"] for m in response.json().get("models", [])]
        return any(model in m for m in models)
    except requests.exceptions.RequestException:
        log.warning(f"Cannot connect to Ollama at {host}")
        return False


def call_ollama(host: str, model: str, prompt: str) -> str:
    """Calls Ollama and returns the response."""
    log = get_logger()
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300
        )
        if response.status_code != 200:
            log.error(f"Ollama responded with status {response.status_code}")
            sys.exit(1)
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama. Make sure it is running.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        log.error("Ollama timed out. Try a lighter model.")
        sys.exit(1)
