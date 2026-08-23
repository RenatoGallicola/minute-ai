"""
ollama_client.py
----------------
Thin HTTP client for a local Ollama server. Shared by cleanup.py and summarize.py.
"""

import requests

from src.errors import OllamaError
from src.logger import get_logger

DEFAULT_TIMEOUT = 600
PROBE_TIMEOUT = 5


def probe(host: str, timeout: int = PROBE_TIMEOUT) -> tuple[bool, list[str]]:
    """Asks the Ollama server what it has.

    Returns (reachable, models). The two are reported separately because a
    freshly installed Ollama is up but has no models yet, and calling that
    "offline" would send the user to check the service instead of pulling a
    model.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        if response.status_code != 200:
            return False, []
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return False, []

    models = payload.get("models") or []
    tags = sorted(
        m["model"] if "model" in m else m.get("name", "")
        for m in models if isinstance(m, dict)
    )
    return True, [t for t in tags if t]


def list_models(host: str, timeout: int = PROBE_TIMEOUT) -> list[str]:
    """Returns the model tags installed on the Ollama server, or [] if unreachable."""
    return probe(host, timeout)[1]


def model_available(installed: list[str], model: str) -> bool:
    """Checks whether `model` is among the installed tags.

    Ollama reports fully qualified tags ('llama3.1:latest'), while users
    normally type the bare name ('llama3.1'). Match on the whole tag or on the
    part before ':', never on a loose substring, which used to let a typo like
    'llama' pass and then fail at generation time.
    """
    wanted = (model or "").strip().lower()
    if not wanted:
        return False
    for tag in installed:
        tag = tag.lower()
        if wanted == tag or wanted == tag.split(":", 1)[0]:
            return True
    return False


def check_ollama(host: str, model: str) -> bool:
    """Checks that Ollama is running and the model is available."""
    log = get_logger()
    reachable, installed = probe(host)
    if not reachable:
        log.warning(f"Cannot connect to Ollama at {host}")
        return False
    if not installed:
        log.warning(f"Ollama is running at {host} but has no models. Run: ollama pull {model}")
        return False
    return model_available(installed, model)


def call_ollama(host: str, model: str, prompt: str, num_ctx: int = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Calls Ollama and returns the generated text.

    Args:
        host:    Ollama server URL
        model:   Model tag to generate with
        prompt:  Full prompt
        num_ctx: Context window to request. Ollama defaults to a small window
                 and silently truncates anything longer, which would quietly
                 drop most of a long meeting transcript.
        timeout: Seconds to wait for the response

    Raises:
        OllamaError: on any connection, timeout or server-side failure
    """
    log = get_logger()
    payload = {"model": model, "prompt": prompt, "stream": False}
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}

    try:
        response = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(f"Cannot connect to Ollama at {host}. Make sure it is running.") from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(f"Ollama timed out after {timeout}s. Try a lighter model or a shorter audio file.") from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Request to Ollama failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text.strip()[:200]
        raise OllamaError(f"Ollama responded with status {response.status_code}: {detail}")

    try:
        body = response.json()
    except ValueError as exc:
        raise OllamaError("Ollama returned a response that is not valid JSON.") from exc

    if body.get("error"):
        raise OllamaError(f"Ollama reported an error: {body['error']}")

    text = (body.get("response") or "").strip()
    if not text:
        log.warning("Ollama returned an empty response.")
    return text
