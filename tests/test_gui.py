import contextlib
import io
import threading
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gui
from main import PipelineResult


@pytest.fixture(autouse=True)
def reset_jobs():
    gui._jobs.clear()
    gui._active_job_id = None
    gui._starting = False
    yield
    gui._jobs.clear()
    gui._active_job_id = None
    gui._starting = False


@pytest.fixture
def client():
    return TestClient(gui.app)


def _current():
    return next(reversed(gui._jobs.values())) if gui._jobs else None


def _wait_until_done(timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _current()
        if job and job.done:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def _result(output_files=(), summary="", transcript=""):
    return PipelineResult(
        audio_path="a.wav",
        meeting_name="A",
        output_files=list(output_files),
        summary=summary,
        transcript=transcript,
        language="en",
        duration_seconds=61.0,
    )


class TestIndex:
    def test_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Minute AI" in r.text
        assert "Generate notes" in r.text

    def test_shows_idle_placeholder_when_no_job(self, client):
        r = client.get("/")
        assert "Nothing running yet" in r.text

    def test_resumes_polling_a_running_job_on_refresh(self, client):
        job = gui.Job("job-123", ["a.wav"])
        job.add_log("Transcribing...", "INFO")
        gui._jobs[job.id] = job
        gui._active_job_id = job.id

        r = client.get("/")
        assert f"/jobs/{job.id}/status" in r.text
        assert "Transcribing..." in r.text

    def test_shows_results_of_a_finished_job_on_refresh(self, client):
        job = gui.Job("job-456", ["a.wav"])
        job.done = True
        job.status = "done"
        entry = gui.FileResult("a.wav")
        entry.status = "done"
        entry.files = ["result.md"]
        job.results.append(entry)
        job.output_files = ["result.md"]
        job.output_paths = {"result.md": Path("result.md")}
        gui._jobs[job.id] = job

        r = client.get("/")
        assert "All done" in r.text
        assert f"/download/{job.id}/result.md" in r.text


class TestHealth:
    def test_reports_environment(self, client, monkeypatch):
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (True, ["llama3.1:latest"]))
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ollama_online"] is True
        assert body["ollama_models"] == ["llama3.1:latest"]
        assert body["device"] in ("CPU", "GPU")

    def test_offline_ollama_is_reported_not_raised(self, client, monkeypatch):
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (False, []))
        assert client.get("/api/health").json()["ollama_online"] is False

    def test_running_but_empty_is_not_reported_as_offline(self, client, monkeypatch):
        # A freshly installed Ollama is up with no models pulled. Calling that
        # "offline" would send the user to check the service instead of
        # pulling a model.
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (True, []))
        body = client.get("/api/health").json()
        assert body["ollama_online"] is True
        assert body["ollama_models"] == []


class TestModelFields:
    def test_starts_disabled_so_the_page_never_waits_on_ollama(self, client):
        # The page must not block on a possibly-unreachable Ollama, so the
        # picker ships disabled and is filled in client-side once health answers.
        page = client.get("/").text
        assert "data-model-field" in page
        assert 'name="cleanup_model"' in page
        assert 'data-configured="llama3.1"' in page

    def test_health_preselects_the_configured_model_when_installed(self, client, monkeypatch):
        monkeypatch.setattr(gui.config, "DEFAULT_CLEANUP_MODEL", "llama3.2:3b")
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (True, ["llama3.2:3b", "qwen2.5:7b"]))
        assert client.get("/api/health").json()["cleanup_model"] == "llama3.2:3b"

    def test_health_falls_back_when_the_configured_model_is_missing(self, client, monkeypatch):
        # The list only ever offers installed models, so config pointing at
        # something absent has to resolve to something runnable.
        monkeypatch.setattr(gui.config, "DEFAULT_CLEANUP_MODEL", "llama3.1")
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (True, ["qwen2.5:7b"]))
        body = client.get("/api/health").json()
        assert body["cleanup_model"] == "qwen2.5:7b"
        assert body["configured_cleanup_model"] == "llama3.1"

    def test_resolve_keeps_the_configured_name_when_nothing_is_installed(self):
        assert gui._resolve_ollama_model("llama3.1", []) == "llama3.1"

    def test_resolve_matches_a_bare_name_against_its_tag(self):
        assert gui._resolve_ollama_model("llama3.2", ["llama3.2:3b"]) == "llama3.2"


class TestOllamaUsable:
    def test_usable_needs_both_a_server_and_a_model(self, client, monkeypatch):
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (True, ["llama3.2:3b"]))
        assert client.get("/api/health").json()["ollama_usable"] is True

    def test_running_but_empty_is_not_usable(self, client, monkeypatch):
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (True, []))
        body = client.get("/api/health").json()
        assert body["ollama_online"] is True
        assert body["ollama_usable"] is False

    def test_offline_is_not_usable(self, client, monkeypatch):
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (False, []))
        assert client.get("/api/health").json()["ollama_usable"] is False


class TestWhisperModelAvailability:
    def test_health_reports_which_models_are_downloaded(self, client, monkeypatch):
        monkeypatch.setattr(
            gui, "system_info",
            lambda: {"device": "CPU", "gpu": "", "vram_gb": 0.0, "ram_gb": 8.0,
                     "whisper_installed": ["small"]},
        )
        body = client.get("/api/health").json()
        assert body["whisper_installed"] == ["small"]
        # Sizes let the UI say what an uninstalled model would cost.
        assert body["whisper_download_mb"]["large-v3"] > 1000


class TestResolveWorkers:
    def test_a_single_file_is_never_parallel(self):
        assert gui.resolve_workers(4, is_batch=False) == 1

    def test_clamped_to_the_maximum(self):
        assert gui.resolve_workers(99, is_batch=True) == gui.MAX_PARALLEL_WORKERS

    def test_at_least_one(self):
        assert gui.resolve_workers(0, is_batch=True) == 1
        assert gui.resolve_workers(-3, is_batch=True) == 1

    def test_garbage_falls_back_to_one(self):
        assert gui.resolve_workers("abc", is_batch=True) == 1

    def test_passes_a_sane_value_through(self):
        assert gui.resolve_workers("3", is_batch=True) == 3

    def test_build_args_marks_parallel_only_above_one_worker(self):
        args = gui._build_args(
            "auto", "auto", "full", True, "auto", "", "",
            "md", "full", "m", "m", "same", "outputs", is_batch=True, workers=3,
        )
        assert args.parallel is True
        assert args.parallel_workers == 3

        single = gui._build_args(
            "auto", "auto", "full", True, "auto", "", "",
            "md", "full", "m", "m", "same", "outputs", is_batch=True, workers=1,
        )
        assert single.parallel is False

    def test_health_exposes_the_list_the_picker_needs(self, client, monkeypatch):
        monkeypatch.setattr(gui, "probe_ollama", lambda host: (True, ["llama3.2:3b", "qwen2.5:7b"]))
        body = client.get("/api/health").json()
        assert body["ollama_models"] == ["llama3.2:3b", "qwen2.5:7b"]
        assert body["ollama_online"] is True

    def test_a_model_that_is_not_installed_is_still_accepted_by_run(self, client, monkeypatch, tmp_path):
        # Picking a missing model must not 400: the pipeline warns and skips
        # the LLM steps rather than failing the whole run.
        out = tmp_path / "result.md"
        out.write_text("x", encoding="utf-8")
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")
        seen = {}

        def capture(audio_path, args, is_batch):
            seen["cleanup"] = args.cleanup_model
            return _result([str(out)])

        monkeypatch.setattr(gui.pipeline, "process_single", capture)
        client.post(
            "/run",
            files={"audio_files": ("a.wav", b"x", "audio/wav")},
            data={"cleanup_model": "not-installed"},
        )
        _wait_until_done()
        assert seen["cleanup"] == "not-installed"


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

    def test_speaker_names_dropped_when_diarization_is_off(self):
        # The names field stays in the DOM when the switch is off, so the form
        # still submits it; passing it through would silently do nothing.
        args = gui._build_args(
            "auto", "auto", "full", False, "auto", "Marco,Sara", "",
            "md", "full", "llama3.1", "llama3.1", "same", "outputs", is_batch=False,
        )
        assert args.speaker_names is None


class TestRunValidation:
    def test_missing_hf_token_with_diarize_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr(gui.config, "HF_TOKEN", "hf_INSERT_HERE")
        r = client.post(
            "/run",
            files={"audio_files": ("a.wav", b"fake", "audio/wav")},
            data={"diarize": "true"},
        )
        assert "HF_TOKEN" in r.text
        assert gui._active_job_id is None

    def test_example_token_placeholder_is_recognised(self, client, monkeypatch):
        # Regression: the check only knew "hf_XXXXXXXXXX" while config.example.py
        # shipped "hf_INSERT_HERE", so a fresh copy sailed past validation.
        monkeypatch.setattr(gui.config, "HF_TOKEN", "hf_XXXXXXXXXX")
        r = client.post(
            "/run",
            files={"audio_files": ("a.wav", b"fake", "audio/wav")},
            data={"diarize": "true"},
        )
        assert "HF_TOKEN" in r.text

    def test_summary_only_with_transcript_mode_is_rejected(self, client):
        r = client.post(
            "/run",
            files={"audio_files": ("a.wav", b"fake", "audio/wav")},
            data={"mode": "transcript", "export_content": "summary"},
        )
        assert "summary" in r.text.lower()
        assert gui._active_job_id is None

    def test_unsupported_file_type_is_rejected_before_starting(self, client):
        r = client.post(
            "/run",
            files={"audio_files": ("notes.txt", b"not audio", "text/plain")},
            data={},
        )
        assert "Unsupported file type" in r.text
        assert gui._active_job_id is None

    def test_srt_with_summary_only_is_rejected(self, client):
        r = client.post(
            "/run",
            files={"audio_files": ("a.wav", b"x", "audio/wav")},
            data={"format": "srt", "export_content": "summary"},
        )
        assert "Subtitles" in r.text

    def test_validation_error_keeps_the_previous_job_visible(self, client, monkeypatch):
        # Regression: a rejected submission used to wipe _current_job, so the
        # results (and download links) of the previous run vanished.
        finished = gui.Job("earlier", ["a.wav"])
        finished.done = True
        finished.status = "done"
        gui._jobs[finished.id] = finished

        client.post(
            "/run",
            files={"audio_files": ("notes.txt", b"x", "text/plain")},
            data={},
        )
        assert "earlier" in gui._jobs


class TestRunHappyPath:
    def _patch_pipeline(self, monkeypatch, output_files=None, model="base", **kwargs):
        result = _result(output_files or [], **kwargs)
        monkeypatch.setattr(gui.pipeline, "process_single", lambda audio_path, args, is_batch: result)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: model)

    def test_creates_job_and_processes_in_background(self, client, monkeypatch, tmp_path):
        output_file = tmp_path / "result.md"
        output_file.write_text("hello", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(output_file)])

        r = client.post("/run", files={"audio_files": ("a.wav", b"fake-audio", "audio/wav")}, data={})
        assert r.status_code == 200

        job = _wait_until_done()
        assert job.output_files == ["result.md"]
        assert job.errors == []
        assert job.percent == 100

    def test_active_slot_is_released_when_the_job_finishes(self, client, monkeypatch):
        self._patch_pipeline(monkeypatch, output_files=[])
        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        _wait_until_done()
        assert gui._active_job_id is None

    def test_status_endpoint_reflects_running_job(self, client, monkeypatch):
        self._patch_pipeline(monkeypatch, output_files=[])
        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job_id = _current().id

        r = client.get(f"/jobs/{job_id}/status")
        assert r.status_code == 200

    def test_status_endpoint_unknown_job_shows_message(self, client):
        r = client.get("/jobs/does-not-exist/status")
        assert r.status_code == 200
        assert "no longer available" in r.text.lower()

    def test_download_serves_generated_file(self, client, monkeypatch, tmp_path):
        output_file = tmp_path / "result.md"
        output_file.write_text("hello world", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(output_file)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job_id = _wait_until_done().id

        r = client.get(f"/download/{job_id}/result.md")
        assert r.status_code == 200
        assert r.content == b"hello world"

    def test_download_still_works_after_a_newer_run_started(self, client, monkeypatch, tmp_path):
        # Regression: only one job was kept, so finishing a second run made the
        # first run's download links 404.
        first = tmp_path / "first.md"
        first.write_text("one", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(first)])
        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        first_id = _wait_until_done().id

        second = tmp_path / "second.md"
        second.write_text("two", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(second)])
        client.post("/run", files={"audio_files": ("b.wav", b"y", "audio/wav")}, data={})
        _wait_until_done()

        assert client.get(f"/download/{first_id}/first.md").content == b"one"

    def test_download_rejects_unknown_filename(self, client, monkeypatch, tmp_path):
        output_file = tmp_path / "result.md"
        output_file.write_text("hello", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(output_file)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job_id = _wait_until_done().id

        r = client.get(f"/download/{job_id}/../../config.py")
        assert r.status_code == 404

    def test_failed_file_is_recorded_as_error_not_crash(self, client, monkeypatch):
        def raising_process_single(audio_path, args, is_batch):
            raise RuntimeError("boom")

        monkeypatch.setattr(gui.pipeline, "process_single", raising_process_single)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job = _wait_until_done()

        assert job.output_files == []
        assert len(job.errors) == 1
        assert job.errors[0][1] == "boom"

    def test_failure_while_starting_does_not_wedge_the_app(self, client, monkeypatch):
        # Regression: an exception between claiming the slot and starting the
        # worker left _current_job non-done forever, so every later run was
        # rejected with "already running".
        def boom(args):
            raise RuntimeError("no model")

        monkeypatch.setattr(gui.pipeline, "resolve_model", boom)
        r = client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        assert "Could not start the run" in r.text
        assert gui._active_job_id is None
        assert gui._starting is False

        self._patch_pipeline(monkeypatch, output_files=[])
        r2 = client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        assert "already in progress" not in r2.text
        _wait_until_done()

    def test_second_run_rejected_while_first_in_progress(self, client, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def slow_process_single(audio_path, args, is_batch):
            started.set()
            release.wait(timeout=5)
            return _result()

        monkeypatch.setattr(gui.pipeline, "process_single", slow_process_single)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        assert started.wait(timeout=2)

        r2 = client.post("/run", files={"audio_files": ("b.wav", b"y", "audio/wav")}, data={})
        assert "already in progress" in r2.text

        release.set()
        _wait_until_done()

    def test_cancel_stops_the_remaining_files(self, client, monkeypatch):
        seen = []
        first_started = threading.Event()
        release = threading.Event()

        def process(audio_path, args, is_batch):
            seen.append(Path(audio_path).name)
            if len(seen) == 1:
                first_started.set()
                release.wait(timeout=5)
            return _result()

        monkeypatch.setattr(gui.pipeline, "process_single", process)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files=[
            ("audio_files", ("a.wav", b"x", "audio/wav")),
            ("audio_files", ("b.wav", b"y", "audio/wav")),
        ], data={})
        assert first_started.wait(timeout=2)

        job_id = _current().id
        client.post(f"/jobs/{job_id}/cancel")
        release.set()

        job = _wait_until_done()
        assert seen == ["a.wav"]
        assert job.status == "cancelled"

    def test_download_all_returns_zip_with_every_file(self, client, monkeypatch, tmp_path):
        f1 = tmp_path / "result.md"
        f1.write_text("markdown content", encoding="utf-8")
        f2 = tmp_path / "result.pdf"
        f2.write_text("pdf content", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(f1), str(f2)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job_id = _wait_until_done().id

        r = client.get(f"/download/{job_id}/all")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            assert sorted(zf.namelist()) == ["result.md", "result.pdf"]
            assert zf.read("result.md") == b"markdown content"

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
        job_id = _wait_until_done().id

        assert "Download all" in client.get(f"/jobs/{job_id}/status").text

    def test_status_page_hides_download_all_for_single_file(self, client, monkeypatch, tmp_path):
        f1 = tmp_path / "result.md"
        f1.write_text("x", encoding="utf-8")
        self._patch_pipeline(monkeypatch, output_files=[str(f1)])

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job_id = _wait_until_done().id

        assert "Download all" not in client.get(f"/jobs/{job_id}/status").text

    def test_summary_is_rendered_as_html_in_the_results_panel(self, client, monkeypatch, tmp_path):
        f1 = tmp_path / "result.md"
        f1.write_text("x", encoding="utf-8")
        self._patch_pipeline(
            monkeypatch, output_files=[str(f1)],
            summary="## Decisions Made\n- Ship on **Friday**",
            transcript="Alice: Hello there.",
        )

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")}, data={})
        job_id = _wait_until_done().id

        page = client.get(f"/jobs/{job_id}/status").text
        assert "<h2>Decisions Made</h2>" in page
        assert "<strong>Friday</strong>" in page
        assert "Alice" in page


class TestJobProgress:
    def test_percent_tracks_stage_and_file(self):
        job = gui.Job("j", ["a.wav", "b.wav"])
        assert job.percent == 0
        job.stage = 2
        assert job.percent == 25            # 2 of 8 stage-slots
        job.current_index = 1
        job.stage = 4
        assert job.percent == 99            # capped until actually done
        job.done = True
        assert job.percent == 100

    def test_log_is_capped(self):
        job = gui.Job("j", ["a.wav"])
        for i in range(gui.MAX_LOG_LINES + 50):
            job.add_log(f"line {i}", "INFO")
        assert len(job.log_lines) == gui.MAX_LOG_LINES
        assert job.log_lines[-1] == f"line {gui.MAX_LOG_LINES + 49}"

    def test_only_recent_jobs_are_remembered(self):
        for i in range(gui.MAX_REMEMBERED_JOBS + 5):
            gui._remember(gui.Job(f"job-{i}", ["a.wav"]))
        assert len(gui._jobs) == gui.MAX_REMEMBERED_JOBS
        assert "job-0" not in gui._jobs


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


class TestJobLogHandler:
    def test_stage_marker_advances_the_progress_stage(self):
        import logging

        job = gui.Job("j", ["a.wav"])
        handler = gui._JobLogHandler(job)
        handler.emit(logging.LogRecord("minute-ai", logging.INFO, "", 0, "[3/4] Generating summary...", None, None))
        assert job.stage == 3

    def test_warnings_are_collected_separately(self):
        import logging

        job = gui.Job("j", ["a.wav"])
        handler = gui._JobLogHandler(job)
        handler.emit(logging.LogRecord("minute-ai", logging.WARNING, "", 0, "Ollama not reachable", None, None))
        assert job.warnings == ["⚠ Ollama not reachable"]


class TestParallelRun:
    def test_files_overlap_when_more_than_one_worker(self, client, monkeypatch, tmp_path):
        import threading as _t

        concurrent = []
        peak = [0]
        lock = _t.Lock()
        gate = _t.Barrier(2, timeout=5)

        def process(audio_path, args, is_batch):
            with lock:
                concurrent.append(1)
                peak[0] = max(peak[0], len(concurrent))
            # Only clears if two workers really overlap; a timeout means they did not.
            with contextlib.suppress(Exception):
                gate.wait()
            with lock:
                concurrent.pop()
            return _result()

        monkeypatch.setattr(gui.pipeline, "process_single", process)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files=[
            ("audio_files", ("a.wav", b"x", "audio/wav")),
            ("audio_files", ("b.wav", b"y", "audio/wav")),
        ], data={"parallel_workers": "2"})

        job = _wait_until_done(timeout=8)
        assert peak[0] == 2
        assert job.workers == 2
        assert job.is_parallel is True

    def test_one_worker_keeps_files_sequential(self, client, monkeypatch):
        order = []

        def process(audio_path, args, is_batch):
            order.append(("start", Path(audio_path).name))
            time.sleep(0.05)
            order.append(("end", Path(audio_path).name))
            return _result()

        monkeypatch.setattr(gui.pipeline, "process_single", process)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files=[
            ("audio_files", ("a.wav", b"x", "audio/wav")),
            ("audio_files", ("b.wav", b"y", "audio/wav")),
        ], data={"parallel_workers": "1"})
        job = _wait_until_done(timeout=8)

        assert order == [("start", "a.wav"), ("end", "a.wav"), ("start", "b.wav"), ("end", "b.wav")]
        assert job.is_parallel is False

    def test_results_keep_upload_order_even_when_they_finish_out_of_order(self, client, monkeypatch):
        def process(audio_path, args, is_batch):
            # "a" is slow, so "b" finishes first.
            time.sleep(0.15 if Path(audio_path).name == "a.wav" else 0.0)
            return _result()

        monkeypatch.setattr(gui.pipeline, "process_single", process)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")

        client.post("/run", files=[
            ("audio_files", ("a.wav", b"x", "audio/wav")),
            ("audio_files", ("b.wav", b"y", "audio/wav")),
        ], data={"parallel_workers": "2"})
        job = _wait_until_done(timeout=8)

        assert [r.name for r in job.results] == ["a.wav", "b.wav"]

    def test_parallel_progress_counts_whole_files(self):
        job = gui.Job("j", ["a.wav", "b.wav", "c.wav", "d.wav"], workers=2)
        assert job.percent == 0
        job.stage = 3                     # meaningless while workers overlap
        assert job.percent == 0
        job.completed = 2
        assert job.percent == 50
        assert job.stage_label == "Processing"

    def test_a_single_file_is_never_treated_as_parallel(self):
        assert gui.Job("j", ["only.wav"], workers=4).is_parallel is False


class TestResolveSpeakers:
    def test_a_listed_number_passes_through(self):
        assert gui.resolve_speakers("4", "") == "4"

    def test_auto_passes_through(self):
        assert gui.resolve_speakers("auto", "") == "auto"

    def test_custom_uses_the_typed_number(self):
        assert gui.resolve_speakers("custom", "32") == "32"

    def test_custom_trims_whitespace(self):
        assert gui.resolve_speakers("custom", "  17 ") == "17"

    def test_custom_left_blank_is_rejected_downstream(self):
        value = gui.resolve_speakers("custom", "")
        assert gui._validate_submission(True, "full", "full", "md", value) is not None

    def test_custom_nonsense_is_rejected_downstream(self):
        value = gui.resolve_speakers("custom", "many")
        assert gui._validate_submission(True, "full", "full", "md", value) is not None

    def test_a_large_custom_count_is_accepted(self):
        # The dropdown stops at 10, but nothing downstream caps the number.
        assert gui._validate_submission(True, "full", "full", "md", "32") is None

    def test_the_form_accepts_a_custom_count_end_to_end(self, client, monkeypatch, tmp_path):
        out = tmp_path / "r.md"
        out.write_text("x", encoding="utf-8")
        seen = {}

        def capture(audio_path, args, is_batch):
            seen["speakers"] = args.speakers
            return _result([str(out)])

        monkeypatch.setattr(gui.pipeline, "process_single", capture)
        monkeypatch.setattr(gui.pipeline, "resolve_model", lambda args: "base")
        monkeypatch.setattr(gui.config, "HF_TOKEN", "hf_realtoken123")

        client.post("/run", files={"audio_files": ("a.wav", b"x", "audio/wav")},
                    data={"diarize": "true", "speakers": "custom", "speakers_custom": "32"})
        _wait_until_done()
        assert seen["speakers"] == 32
