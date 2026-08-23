import pytest
import requests

from src import ollama_client
from src.errors import OllamaError


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = ""

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


class TestModelAvailable:
    def test_exact_tag(self):
        assert ollama_client.model_available(["llama3.1:latest"], "llama3.1:latest") is True

    def test_bare_name_matches_tag(self):
        assert ollama_client.model_available(["llama3.1:latest"], "llama3.1") is True

    def test_partial_name_does_not_match(self):
        # Regression: a loose substring match let a typo like "llama" pass the
        # pre-flight check and only fail later, mid-generation.
        assert ollama_client.model_available(["llama3.1:latest"], "llama") is False

    def test_empty_name_is_false(self):
        assert ollama_client.model_available(["llama3.1:latest"], "") is False


class TestListModels:
    def test_returns_sorted_tags(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: FakeResponse(200, {"models": [{"name": "mistral"}, {"name": "llama3.1"}]})
        )
        assert ollama_client.list_models("http://localhost:11434") == ["llama3.1", "mistral"]

    def test_unreachable_returns_empty(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("refused")
        monkeypatch.setattr(requests, "get", boom)
        assert ollama_client.list_models("http://localhost:11434") == []

    def test_invalid_json_returns_empty(self, monkeypatch):
        class Broken(FakeResponse):
            def json(self):
                raise ValueError("not json")
        monkeypatch.setattr(requests, "get", lambda *a, **k: Broken(200))
        assert ollama_client.list_models("http://localhost:11434") == []


class TestCallOllama:
    def test_returns_response_text(self, monkeypatch):
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(200, {"response": "cleaned text"})
        )
        result = ollama_client.call_ollama("http://localhost:11434", "llama3.1", "prompt")
        assert result == "cleaned text"

    def test_num_ctx_is_sent_when_given(self, monkeypatch):
        # Ollama silently truncates prompts longer than its default context,
        # which would quietly drop most of a long meeting transcript.
        seen = {}

        def capture(url, json=None, timeout=None):
            seen.update(json)
            return FakeResponse(200, {"response": "ok"})

        monkeypatch.setattr(requests, "post", capture)
        ollama_client.call_ollama("http://h", "m", "prompt", num_ctx=8192)
        assert seen["options"] == {"num_ctx": 8192}

    def test_connection_error_raises_ollama_error(self, monkeypatch):
        def raise_connection_error(*a, **k):
            raise requests.exceptions.ConnectionError("refused")
        monkeypatch.setattr(requests, "post", raise_connection_error)

        with pytest.raises(OllamaError):
            ollama_client.call_ollama("http://localhost:11434", "llama3.1", "prompt")

    def test_timeout_raises_ollama_error(self, monkeypatch):
        def raise_timeout(*a, **k):
            raise requests.exceptions.Timeout("timed out")
        monkeypatch.setattr(requests, "post", raise_timeout)

        with pytest.raises(OllamaError):
            ollama_client.call_ollama("http://localhost:11434", "llama3.1", "prompt")

    def test_error_status_raises_instead_of_killing_the_process(self, monkeypatch):
        # Regression: this used to sys.exit(1) from inside a library module,
        # which raised SystemExit past run_batch's `except Exception` and took
        # the whole batch down.
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(500))
        with pytest.raises(OllamaError):
            ollama_client.call_ollama("http://localhost:11434", "llama3.1", "prompt")

    def test_error_field_in_body_raises(self, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, {"error": "model not found"}))
        with pytest.raises(OllamaError):
            ollama_client.call_ollama("http://localhost:11434", "nope", "prompt")


class TestProbe:
    def test_reachable_with_models(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: FakeResponse(200, {"models": [{"name": "llama3.2:3b"}]})
        )
        assert ollama_client.probe("http://h") == (True, ["llama3.2:3b"])

    def test_reachable_but_empty_is_not_unreachable(self, monkeypatch):
        # A freshly installed Ollama answers fine but has nothing pulled yet;
        # reporting that as "cannot connect" points at the wrong problem.
        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, {"models": []}))
        assert ollama_client.probe("http://h") == (True, [])

    def test_unreachable(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("refused")
        monkeypatch.setattr(requests, "get", boom)
        assert ollama_client.probe("http://h") == (False, [])

    def test_check_ollama_says_so_when_no_models_are_installed(self, monkeypatch, caplog):
        import logging
        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, {"models": []}))
        with caplog.at_level(logging.WARNING, logger="minute-ai"):
            assert ollama_client.check_ollama("http://h", "llama3.2:3b") is False
        assert any("ollama pull" in r.message for r in caplog.records)
