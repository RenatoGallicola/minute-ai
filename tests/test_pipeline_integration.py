"""End-to-end pipeline runs with whisperX and Ollama stubbed out.

These cover the seams the unit tests can't: process_single wiring the steps
together, batch mode, and the CLI's exit codes.
"""

import argparse
from pathlib import Path

import pytest

import main
from src import cleanup, summarize, transcribe as transcribe_module
from src.errors import TranscriptionError


SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "Hello everyone.", "speaker": "SPEAKER_01"},
    {"start": 2.0, "end": 4.0, "text": "Hi there.", "speaker": "SPEAKER_00"},
    {"start": 4.0, "end": 7.0, "text": "Let's ship on Friday.", "speaker": "SPEAKER_01"},
]


@pytest.fixture
def stub_pipeline(monkeypatch):
    """whisperX returns fixed segments; Ollama is reachable and echoes back."""
    def fake_transcribe(**kwargs):
        return transcribe_module.TranscriptionResult(
            segments=[dict(s) for s in SEGMENTS],
            language="en",
            duration_seconds=420.0,
            diarized=kwargs.get("diarize", True),
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )

    monkeypatch.setattr(main, "transcribe", fake_transcribe)
    monkeypatch.setattr(cleanup, "check_ollama", lambda host, model: True)
    monkeypatch.setattr(summarize, "check_ollama", lambda host, model: True)
    monkeypatch.setattr(cleanup, "call_ollama", lambda host, model, prompt, **kw: "Ana: Hello everyone. Let's ship on Friday.\n\nBen: Hi there.")
    monkeypatch.setattr(summarize, "call_ollama", lambda host, model, prompt, **kw: "## Decisions Made\n- Ship on Friday")


def make_args(tmp_path, **overrides):
    defaults = dict(
        audio=[], language="auto", speakers=None, speaker_names=None, model="base",
        no_diarize=False, meeting_name=None, mode="full", cleanup_model="llama3.1",
        summary_model="llama3.1", summary_language="same", summary_preset="meeting", summary_prompt=None,
        summary_prompt_file=None, output_dir=str(tmp_path),
        format="md", export_content="full", parallel=False, parallel_workers=2,
        force=False, recursive=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestProcessSingle:
    def test_full_mode_produces_everything(self, tmp_path, stub_pipeline):
        result = main.process_single("inputs/q3_kickoff.wav", make_args(tmp_path), is_batch=False)

        assert result.meeting_name == "Q3 Kickoff"
        assert result.language == "en"
        assert result.duration_seconds == 420.0
        assert result.summary.startswith("## Decisions Made")
        assert "Ana:" in result.transcript
        assert len(result.output_files) == 1

        content = open(result.output_files[0], encoding="utf-8").read()
        assert "# Q3 Kickoff" in content
        assert "Decisions Made" in content
        assert "Full Transcript" in content
        assert "Duration: 7m 00s" in content

    def test_transcript_mode_skips_both_llm_steps(self, tmp_path, stub_pipeline, monkeypatch):
        monkeypatch.setattr(cleanup, "call_ollama", lambda *a, **k: pytest.fail("cleanup should not run"))
        monkeypatch.setattr(summarize, "call_ollama", lambda *a, **k: pytest.fail("summary should not run"))

        result = main.process_single("a.wav", make_args(tmp_path, mode="transcript"), is_batch=False)
        assert result.summary == ""
        assert "SPEAKER_01: Hello everyone." in result.transcript

    def test_speaker_names_follow_order_of_appearance(self, tmp_path, stub_pipeline, monkeypatch):
        monkeypatch.setattr(cleanup, "check_ollama", lambda host, model: False)
        monkeypatch.setattr(summarize, "check_ollama", lambda host, model: False)

        args = make_args(tmp_path, speaker_names="Ana,Ben", mode="transcript")
        result = main.process_single("a.wav", args, is_batch=False)
        # SPEAKER_01 speaks first in the fixture, so it gets the first name.
        assert result.transcript.startswith("Ana: Hello everyone.")
        assert "Ben: Hi there." in result.transcript

    def test_batch_ignores_speaker_names(self, tmp_path, stub_pipeline):
        args = make_args(tmp_path, speaker_names="Ana,Ben", mode="transcript")
        result = main.process_single("a.wav", args, is_batch=True)
        assert "SPEAKER_01:" in result.transcript

    def test_no_diarize_produces_a_flat_transcript(self, tmp_path, stub_pipeline):
        args = make_args(tmp_path, no_diarize=True, mode="transcript")
        result = main.process_single("a.wav", args, is_batch=False)
        assert ":" not in result.transcript
        assert result.transcript.startswith("Hello everyone.")

    def test_all_formats_writes_every_file_including_srt(self, tmp_path, stub_pipeline):
        result = main.process_single("a.wav", make_args(tmp_path, format="all"), is_batch=False)
        suffixes = sorted(f.rsplit(".", 1)[-1] for f in result.output_files)
        assert suffixes == ["docx", "md", "pdf", "srt", "txt"]

    def test_ollama_down_still_yields_a_transcript_file(self, tmp_path, stub_pipeline, monkeypatch):
        monkeypatch.setattr(cleanup, "check_ollama", lambda host, model: False)
        monkeypatch.setattr(summarize, "check_ollama", lambda host, model: False)

        result = main.process_single("a.wav", make_args(tmp_path), is_batch=False)
        assert result.summary == ""
        assert len(result.output_files) == 1
        assert "Hello everyone." in open(result.output_files[0], encoding="utf-8").read()

    def test_meeting_name_with_illegal_characters_is_written_safely(self, tmp_path, stub_pipeline):
        args = make_args(tmp_path, meeting_name="Q3: Kickoff <draft>")
        result = main.process_single("a.wav", args, is_batch=False)
        written = [p.name for p in tmp_path.iterdir() if p.is_file()]
        assert written == [Path(result.output_files[0]).name]
        assert ":" not in written[0] and "<" not in written[0]


class TestCliRun:
    def test_missing_file_exits_non_zero(self, tmp_path, stub_pipeline):
        assert main.main(["does-not-exist.wav", "-o", str(tmp_path), "--no-diarize"]) == 1

    def test_single_file_run_returns_zero(self, tmp_path, stub_pipeline, monkeypatch):
        audio = tmp_path / "team_sync.wav"
        audio.touch()
        out = tmp_path / "out"
        assert main.main([str(audio), "-o", str(out), "--no-diarize"]) == 0
        assert list(out.glob("*.md"))

    def test_batch_keeps_going_after_a_failure(self, tmp_path, stub_pipeline, monkeypatch):
        good = tmp_path / "good.wav"
        bad = tmp_path / "bad.wav"
        good.touch()
        bad.touch()
        out = tmp_path / "out"

        original = main.transcribe

        def flaky(**kwargs):
            if "bad" in kwargs["audio_path"]:
                raise TranscriptionError("cannot read audio")
            return original(**kwargs)

        monkeypatch.setattr(main, "transcribe", flaky)

        # One file fails, the other still gets exported, and the exit code says so.
        assert main.main([str(good), str(bad), "-o", str(out), "--no-diarize"]) == 1
        assert len(list(out.glob("*.md"))) == 1

    def test_batch_skips_already_processed_then_force_reprocesses(self, tmp_path, stub_pipeline):
        a = tmp_path / "one.wav"
        b = tmp_path / "two.wav"
        a.touch()
        b.touch()
        out = tmp_path / "out"

        main.main([str(a), str(b), "-o", str(out), "--no-diarize"])
        assert len(list(out.glob("*.md"))) == 2

        # Second run: both are recognised as done and nothing new is written.
        main.main([str(a), str(b), "-o", str(out), "--no-diarize"])
        assert len(list(out.glob("*.md"))) == 2

        main.main([str(a), str(b), "-o", str(out), "--no-diarize", "--force"])
        assert len(list(out.glob("*.md"))) == 4
