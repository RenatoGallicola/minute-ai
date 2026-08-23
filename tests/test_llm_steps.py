"""Cleanup and summary behaviour that does not depend on a running Ollama."""

import pytest

from src import cleanup, summarize
from src.errors import OllamaError


@pytest.fixture
def ollama_up(monkeypatch):
    monkeypatch.setattr(cleanup, "check_ollama", lambda host, model: True)
    monkeypatch.setattr(summarize, "check_ollama", lambda host, model: True)


class TestCleanup:
    def test_returns_the_original_when_ollama_is_down(self, monkeypatch):
        monkeypatch.setattr(cleanup, "check_ollama", lambda host, model: False)
        original = "Alice: hello"
        assert cleanup.cleanup_transcript(original, "m", "http://h", "en") == original

    def test_empty_transcript_short_circuits(self, monkeypatch):
        called = []
        monkeypatch.setattr(cleanup, "check_ollama", lambda host, model: called.append(1) or True)
        assert cleanup.cleanup_transcript("   ", "m", "http://h", "en") == "   "
        assert called == []

    def test_short_transcript_is_one_request(self, ollama_up, monkeypatch):
        calls = []
        monkeypatch.setattr(cleanup, "call_ollama",
                            lambda host, model, prompt, **kw: calls.append(prompt) or "cleaned")
        assert cleanup.cleanup_transcript("Alice: hi", "m", "http://h", "en") == "cleaned"
        assert len(calls) == 1

    def test_long_transcript_is_cleaned_in_chunks(self, ollama_up, monkeypatch):
        # Regression: the whole transcript went out in one prompt, and Ollama
        # silently truncated everything past its context window.
        calls = []
        monkeypatch.setattr(cleanup, "call_ollama",
                            lambda host, model, prompt, **kw: calls.append(prompt) or f"part{len(calls)}")

        transcript = "\n\n".join(f"S{i}: " + "word " * 60 for i in range(10))
        result = cleanup.cleanup_transcript(transcript, "m", "http://h", "en", chunk_chars=500)

        assert len(calls) > 1
        assert "part1" in result and f"part{len(calls)}" in result
        assert all("part 1 of" not in c or "clean only what is given" in c for c in calls)

    def test_num_ctx_is_forwarded(self, ollama_up, monkeypatch):
        seen = {}

        def capture(host, model, prompt, num_ctx=None, timeout=None):
            seen["num_ctx"] = num_ctx
            return "cleaned"

        monkeypatch.setattr(cleanup, "call_ollama", capture)
        cleanup.cleanup_transcript("Alice: hi", "m", "http://h", "en", num_ctx=8192)
        assert seen["num_ctx"] == 8192

    def test_ollama_failure_keeps_the_raw_transcript(self, ollama_up, monkeypatch):
        def boom(*a, **k):
            raise OllamaError("server exploded")

        monkeypatch.setattr(cleanup, "call_ollama", boom)
        assert cleanup.cleanup_transcript("Alice: hi", "m", "http://h", "en") == "Alice: hi"


class TestSummarize:
    def test_returns_empty_when_ollama_is_down(self, monkeypatch):
        monkeypatch.setattr(summarize, "check_ollama", lambda host, model: False)
        assert summarize.summarize_transcript("Alice: hi", "m", "http://h", "en", "same") == ""

    def test_short_transcript_is_a_single_pass(self, ollama_up, monkeypatch):
        calls = []
        monkeypatch.setattr(summarize, "call_ollama",
                            lambda host, model, prompt, **kw: calls.append(prompt) or "## Topics")
        assert summarize.summarize_transcript("Alice: hi", "m", "http://h", "en", "same") == "## Topics"
        assert len(calls) == 1

    def test_long_transcript_maps_then_merges(self, ollama_up, monkeypatch):
        calls = []

        def capture(host, model, prompt, **kw):
            calls.append(prompt)
            return "merged" if "NOTES FROM PART" in prompt else "notes"

        monkeypatch.setattr(summarize, "call_ollama", capture)
        transcript = "\n\n".join(f"S{i}: " + "word " * 60 for i in range(10))
        result = summarize.summarize_transcript(
            transcript, "m", "http://h", "en", "same", chunk_chars=500
        )

        assert result == "merged"
        assert calls[-1].count("NOTES FROM PART") == len(calls) - 1

    def test_failure_degrades_to_no_summary(self, ollama_up, monkeypatch):
        def boom(*a, **k):
            raise OllamaError("nope")

        monkeypatch.setattr(summarize, "call_ollama", boom)
        assert summarize.summarize_transcript("Alice: hi", "m", "http://h", "en", "same") == ""


class TestOutputLanguage:
    def test_same_uses_the_transcript_language(self):
        assert summarize.resolve_output_language("it", "same") == "Italian"

    def test_explicit_language_wins(self):
        assert summarize.resolve_output_language("it", "en") == "English"

    def test_unknown_code_is_passed_through(self):
        assert summarize.resolve_output_language("xx", "same") == "xx"

    def test_missing_language_falls_back_to_english(self):
        assert summarize.resolve_output_language("", "same") == "English"
