import io
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

import gui


@pytest.fixture(autouse=True)
def reset_current_job():
    gui._current_job = None
    yield
    gui._current_job = None


@pytest.fixture
def client():
    return TestClient(gui.app)


def _wait_until_done(timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if gui._current_job and gui._current_job.done:
            return
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


class TestIndex:
    def test_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Minute AI" in r.text
        assert "Generate" in r.text


class TestBuildArgs:
    def test_single_file_keeps_speaker_names_and_meeting_name(self):
        args = gui._build_args(
            "en", "medium", "full", True, "2", "Marco,Sara", "Q3 Kickoff",
            "md", "full", "llama3.1", "llama3.1", "same", "outputs", is_batch=False,
        )
        assert args.speaker_names == "Marco,Sara"
        assert args.meeting_name == "Q3 Kickoff"
        assert args.speakers == 2
        assert args.no_diarize is False

    def test_batch_drops_speaker_names_and_meeting_name(self):
        args = gui._build_args(
            "en", "medium", "full", True, "auto", "Marco,Sara", "Q3 Kickoff",
            "md", "full", "llama3.1", "llama3.1", "same", "outputs", is_batch=True,
        )
        assert args.speaker_names is None
        assert args.meeting_name is None

    def test_auto_speakers_becomes_none(self):
        args = gui._build_args(
            "auto", "auto", "full", True, "auto", "", "",
            "md", "full", "llama3.1", "llama3.1", "same", "outputs", is_batch=False,
        )
        assert args.speakers is None

    def test_no_diarize_when_diarize_false(self):
        args = gui._build_args(
            "auto", "auto", "full", False, "auto", "", "",
            "md", "full", "llama3.1", "llama3.1", "same", "outputs", is_batch=False,
        )
        assert args.no_diarize is True


class TestRunValidation:
    def test_missing_hf_token_with_diarize_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(gui.config, "HF_TOKEN", "hf_XXXXXXXXXX")
        r = client.post(
            "/run",
            files={"audio_files": ("a.wav", b"fake", "audio/wav")},
            data={"diarize": "true"},
        )
        assert "HF_TOKEN" in r.text
        assert gui._current_job is None

    def test_summary_only_with_transcript_mode_is_rejected(self, client):
        r = client.post(
            "/run",
            files={"audio_files": ("a.wav", b"fake", "audio/wav")},
            data={"mode": "transcript", "export_content": "summary"},
        )
        assert "summary" in r.text.lower()
        assert gui._current_job is None


class TestRunHappyPath:
    def _patch_pipeline(self, monkeypatch, output_files=None, model="base"):
        output_files = output_files or []
        monkeypatch.setattr(gui.pipeline, "process_single", lambda audio_path, args, is_batch: list(output_files))
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: model)

    def test_creates_job_and_processes_in_background(self, client, monkeypatch, tmp_path):
        output_file = tmp_path / "result.md"
        output_file.write_text("hello", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(output_file)])

        r = client.post("/run", files={"audio_files": ("a.wav", b"fake-audio", "audio/wav")}, data={})
        assert r.status_code == 200

        _wait_until_done()
        assert gui._current_job.output_files == ["result.md"]
        assert gui._current_job.errors == []

    def test_status_endpoint_reflects_running_job(self, client, monkeypatch, tmp_path):
        self._patch_pipeline(monkeypatch, output_files=[])
        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job_id = gui._current_job.id

        r = client.get(f"/jobs/{job_id}/status")
        assert r.status_code == 200

    def test_status_endpoint_unknown_job_shows_message(self, client):
        r = client.get("/jobs/does-not-exist/status")
        assert r.status_code == 200
        assert "no matching job" in r.text.lower()

    def test_download_serves_generated_file(self, client, monkeypatch, tmp_path):
        output_file = tmp_path / "result.md"
        output_file.write_text("hello world", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(output_file)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        _wait_until_done()
        job_id = gui._current_job.id

        r = client.get(f"/download/{job_id}/result.md")
        assert r.status_code == 200
        assert r.content == b"hello world"

    def test_download_rejects_unknown_filename(self, client, monkeypatch, tmp_path):
        output_file = tmp_path / "result.md"
        output_file.write_text("hello", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(output_file)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        _wait_until_done()
        job_id = gui._current_job.id

        r = client.get(f"/download/{job_id}/../../config.py")
        assert r.status_code == 404

    def test_failed_file_is_recorded_as_error_not_crash(self, client, monkeypatch, tmp_path):
        def raising_process_single(audio_path, args, is_batch):
            raise RuntimeError("boom")

        monkeypatch.setattr(gui.pipeline, "process_single", raising_process_single)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        _wait_until_done()

        assert gui._current_job.output_files == []
        assert len(gui._current_job.errors) == 1
        assert gui._current_job.errors[0][1] == "boom"

    def test_second_run_rejected_while_first_in_progress(self, client, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def slow_process_single(audio_path, args, is_batch):
            started.set()
            release.wait(timeout=5)
            return []

        monkeypatch.setattr(gui.pipeline, "process_single", slow_process_single)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        assert started.wait(timeout=2)

        r2 = client.post("/run", files={"audio_files": ("b.wav", b"y", "audio/wav")}, data={})
        assert "already running" in r2.text

        release.set()
        _wait_until_done()

    def test_download_all_returns_zip_with_every_file(self, client, monkeypatch, tmp_path):
        f1 = tmp_path / "result.md"
        f1.write_text("markdown content", encoding="utf-8")
        f2 = tmp_path / "result.pdf"
        f2.write_text("pdf content", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(f1), str(f2)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        _wait_until_done()
        job_id = gui._current_job.id

        r = client.get(f"/download/{job_id}/all")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            assert sorted(zf.namelist()) == ["result.md", "result.pdf"]
            assert zf.read("result.md") == b"markdown content"
            assert zf.read("result.pdf") == b"pdf content"

    def test_download_all_404_when_no_job(self, client):
        r = client.get("/download/does-not-exist/all")
        assert r.status_code == 404

    def test_status_page_shows_download_all_only_for_multiple_files(self, client, monkeypatch, tmp_path):
        f1 = tmp_path / "result.md"
        f1.write_text("x", encoding="utf-8")
        f2 = tmp_path / "result.pdf"
        f2.write_text("x", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(f1), str(f2)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        _wait_until_done()
        job_id = gui._current_job.id

        r = client.get(f"/jobs/{job_id}/status")
        assert "Download all" in r.text

    def test_status_page_hides_download_all_for_single_file(self, client, monkeypatch, tmp_path):
        f1 = tmp_path / "result.md"
        f1.write_text("x", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(f1)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        _wait_until_done()
        job_id = gui._current_job.id

        r = client.get(f"/jobs/{job_id}/status")
        assert "Download all" not in r.text


class TestCleanLogLine:
    def test_strips_whitespace(self):
        assert gui._clean_log_line("      Detected language: en") == "Detected language: en"

    def test_filters_out_model_debug_line(self):
        assert gui._clean_log_line("      Model: medium | Device: cpu | Language: auto-detect") is None

    def test_simplifies_exported_line_to_basename(self):
        result = gui._clean_log_line("      Exported: outputs/2026-07-05_10-00_Team_Sync.md")
        assert result == "Exported 2026-07-05_10-00_Team_Sync.md"

    def test_leaves_other_lines_unchanged(self):
        assert gui._clean_log_line("[1/4] Transcribing audio: meeting.wav") == "[1/4] Transcribing audio: meeting.wav"

    def test_simplifies_transcribing_line_to_basename(self):
        message = r"[1/4] Transcribing audio: C:\Users\renat\AppData\Local\Temp\minute-ai-xyz\meeting.wav"
        assert gui._clean_log_line(message) == "[1/4] Transcribing audio: meeting.wav"
