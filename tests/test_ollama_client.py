import requests

from src import ollama_client


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


class TestCheckOllama:
    def test_model_found(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: FakeResponse(200, {"models": [{"name": "llama3.1:latest"}]})
        )
        assert ollama_client.check_ollama("http://localhost:11434", "llama3.1") is True

    def test_model_not_found(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: FakeResponse(200, {"models": [{"name": "mistral:latest"}]})
        )
        assert ollama_client.check_ollama("http://localhost:11434", "llama3.1") is False

    def test_non_200_status_returns_false(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(500))
        assert ollama_client.check_ollama("http://localhost:11434", "llama3.1") is False

    def test_connection_error_returns_false(self, monkeypatch):
        def raise_connection_error(*a, **k):
            raise requests.exceptions.ConnectionError("refused")
        monkeypatch.setattr(requests, "get", raise_connection_error)
        assert ollama_client.check_ollama("http://localhost:11434", "llama3.1") is False

    def test_timeout_returns_false_instead_of_raising(self, monkeypatch):
        # Regression test: check_ollama used to only catch ConnectionError,
        # so a slow/hanging Ollama server would crash the whole pipeline
        # instead of degrading gracefully like call_ollama does.
        def raise_timeout(*a, **k):
            raise requests.exceptions.Timeout("timed out")
        monkeypatch.setattr(requests, "get", raise_timeout)
        assert ollama_client.check_ollama("http://localhost:11434", "llama3.1") is False


class TestCallOllama:
    def test_returns_response_text(self, monkeypatch):
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(200, {"response": "cleaned text"})
        )
        result = ollama_client.call_ollama("http://localhost:11434", "llama3.1", "prompt")
        assert result == "cleaned text"

    def test_connection_error_exits(self, monkeypatch):
        def raise_connection_error(*a, **k):
            raise requests.exceptions.ConnectionError("refused")
        monkeypatch.setattr(requests, "post", raise_connection_error)

        import pytest
        with pytest.raises(SystemExit):
            ollama_client.call_ollama("http://localhost:11434", "llama3.1", "prompt")

    def test_timeout_exits(self, monkeypatch):
        def raise_timeout(*a, **k):
            raise requests.exceptions.Timeout("timed out")
        monkeypatch.setattr(requests, "post", raise_timeout)

        import pytest
        with pytest.raises(SystemExit):
            ollama_client.call_ollama("http://localhost:11434", "llama3.1", "prompt")
