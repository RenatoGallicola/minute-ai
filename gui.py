#!/usr/bin/env python3
"""
gui.py
------
Local web GUI for minute-ai: FastAPI + Jinja2 + htmx, styled with a hand-written
CSS design system (static/app.css). No Node, no build step.

Wraps the same pipeline used by main.py (main.process_single / main.resolve_model);
no pipeline logic is duplicated here.

Run with:
    python gui.py
"""

import argparse
import io
import logging
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import main as pipeline
from src import languages, prompts
from src.batch import SUPPORTED_EXTENSIONS
from src.errors import MinuteAIError
from src.hardware import MODEL_DOWNLOAD_MB, system_info
from src.logger import get_logger, setup_logger
from src.markdown_lite import render as render_markdown
from src.markdown_lite import render_transcript
from src.ollama_client import model_available
from src.ollama_client import probe as probe_ollama
from src.transcribe import format_duration

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="minute-ai")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# The logger is set up at import time, not just under __main__, so running the
# app through `uvicorn gui:app` still writes to logs/ and still feeds the UI.
setup_logger()


LANGUAGES = languages.audio_options()
MODELS = [
    ("auto", "Automatic (picked from your GPU/RAM)"),
    ("tiny", "Tiny (fastest, roughest)"),
    ("base", "Base (quick draft quality)"),
    ("small", "Small (balanced)"),
    ("medium", "Medium (good accuracy)"),
    ("large-v3", "Large v3 (most accurate, slowest)"),
]
# The list stops at a comfortable 10; anything beyond that is typed in.
LISTED_SPEAKERS = 10
SPEAKER_COUNTS = (
    [("auto", "Auto")]
    + [(str(n), str(n)) for n in range(1, LISTED_SPEAKERS + 1)]
    + [("custom", f"More than {LISTED_SPEAKERS}…")]
)
MODES = [
    ("full", "Full", "Transcript, cleanup and summary"),
    ("clean", "Cleaned", "Transcript and cleanup, no summary"),
    ("summary", "Summary", "Transcript and summary, no cleanup"),
    ("transcript", "Raw", "Transcript only, no LLM"),
]
FORMATS = [
    ("md", "Markdown", ".md"), ("txt", "Text", ".txt"), ("docx", "Word", ".docx"),
    ("pdf", "PDF", ".pdf"), ("srt", "Subtitles", ".srt"), ("all", "All formats", "every file"),
]
EXPORT_CONTENTS = [
    ("full", "Summary + transcript", ""),
    ("summary", "Summary only", ""),
]
SUMMARY_LANGUAGES = languages.summary_options()
SUMMARY_PRESETS = prompts.options()

STAGE_LABELS = ["Waiting", "Transcribing", "Cleaning up", "Summarizing", "Exporting"]
TOTAL_STAGES = 4
MAX_LOG_LINES = 2000
MAX_REMEMBERED_JOBS = 20
# Each worker loads its own Whisper model, so this stays deliberately small.
MAX_PARALLEL_WORKERS = 4


def _index_context() -> dict:
    return {
        "languages": LANGUAGES,
        "models": MODELS,
        "default_model": config.DEFAULT_WHISPER_MODEL,
        "modes": MODES,
        "default_mode": getattr(config, "DEFAULT_MODE", "full"),
        "speaker_counts": SPEAKER_COUNTS,
        "listed_speakers": LISTED_SPEAKERS,
        "formats": FORMATS,
        "default_format": config.DEFAULT_FORMAT,
        "export_contents": EXPORT_CONTENTS,
        "default_export_content": config.DEFAULT_EXPORT_CONTENT,
        "summary_languages": SUMMARY_LANGUAGES,
        "summary_presets": SUMMARY_PRESETS,
        "default_summary_preset": getattr(config, "DEFAULT_SUMMARY_PRESET", prompts.DEFAULT_PRESET),
        "default_language": getattr(config, "DEFAULT_LANGUAGE", "auto"),
        "default_summary_language": getattr(config, "DEFAULT_SUMMARY_LANGUAGE", "same"),
        "default_cleanup_model": config.DEFAULT_CLEANUP_MODEL,
        "default_summary_model": config.DEFAULT_SUMMARY_MODEL,
        "max_workers": MAX_PARALLEL_WORKERS,
        "accepted_extensions": ",".join(sorted(SUPPORTED_EXTENSIONS)),
        "extension_list": " · ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS)),
    }


class FileResult:
    """One processed audio file, as the results panel needs it."""

    def __init__(self, name: str):
        self.name = name
        self.meeting_name = name
        self.status = "running"          # running | done | failed | skipped
        self.message = ""
        self.files: list[str] = []
        self.summary_html = ""
        self.transcript_html = ""
        self.summary_text = ""
        self.transcript_text = ""
        self.duration = ""
        self.language = ""
        self.speakers: list[str] = []


class Job:
    def __init__(self, job_id: str, file_names: list[str] = None, workers: int = 1):
        self.id = job_id
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.status = "running"          # running | done | cancelled
        self.done = False
        self.stage = 0
        self.file_names = list(file_names or [])
        self.workers = max(workers, 1)
        self.completed = 0                # finished files, the only reliable
                                          # progress signal once workers overlap
        self.current_index = 0
        self.log_lines: list[str] = []
        self.results: list[FileResult] = []
        self.output_files: list[str] = []       # basenames, for display + download lookup
        self.output_paths: dict[str, Path] = {}
        self.errors: list[tuple[str, str]] = []
        self.warnings: list[str] = []
        self.cancel_requested = threading.Event()

    # ── progress ───────────────────────────────────────────────
    @property
    def total_files(self) -> int:
        return max(len(self.file_names), 1)

    @property
    def is_parallel(self) -> bool:
        return self.workers > 1 and len(self.file_names) > 1

    @property
    def percent(self) -> int:
        if self.done:
            return 100
        if self.is_parallel:
            # Several files are mid-flight at once, so per-stage progress is
            # meaningless; count whole files instead.
            return min(int(self.completed / self.total_files * 100), 99)
        completed = self.current_index * TOTAL_STAGES + min(self.stage, TOTAL_STAGES)
        return min(int(completed / (self.total_files * TOTAL_STAGES) * 100), 99)

    @property
    def next_percent(self) -> int:
        """The next checkpoint, so the bar can creep towards it between polls.

        Progress is only known at stage boundaries, which makes the bar jump.
        Advancing towards this value (without reaching it) keeps it moving
        without ever claiming more than the pipeline has actually reported.
        """
        if self.done:
            return 100
        span = self.total_files if self.is_parallel else self.total_files * TOTAL_STAGES
        return min(self.percent + int(100 / max(span, 1)), 99)

    @property
    def stage_label(self) -> str:
        if self.status == "cancelled":
            return "Stopped"
        if self.done:
            return "Finished"
        if self.is_parallel:
            return "Processing"
        return STAGE_LABELS[min(self.stage, TOTAL_STAGES)]

    @property
    def current_file(self) -> str:
        if self.is_parallel:
            return ""
        if self.current_index < len(self.file_names):
            return self.file_names[self.current_index]
        return ""

    @property
    def elapsed(self) -> str:
        end = self.finished_at or time.time()
        return format_duration(end - self.created_at)

    @property
    def log_text(self) -> str:
        return "\n".join(self.log_lines)

    @property
    def succeeded(self) -> list[FileResult]:
        return [r for r in self.results if r.status == "done"]

    def note_stage(self, stage: int):
        self.stage = stage

    def add_log(self, line: str, level: str):
        self.log_lines.append(line)
        if len(self.log_lines) > MAX_LOG_LINES:
            del self.log_lines[: len(self.log_lines) - MAX_LOG_LINES]
        if level == "WARNING":
            self.warnings.append(line)


_NOISY_LINE_PREFIXES = ("Model:",)
_EXPORTED_LINE = re.compile(r"^Exported:\s*(.+)$")
_TRANSCRIBING_LINE = re.compile(r"^(\[1/4\] Transcribing audio:)\s*(.+)$")
_STAGE_LINE = re.compile(r"^\[(\d)/4\]")


def _basename(path: str) -> str:
    """Last segment of a path, whichever separator it uses.

    PureWindowsPath understands both '/' and '\\', while PurePosixPath treats a
    backslash as an ordinary character. Using the plain Path here made this
    display helper behave differently depending on the host OS.
    """
    return PureWindowsPath(path).name or path


def _clean_log_line(message: str) -> str | None:
    """Trims and simplifies a raw pipeline log line for display in the web UI.

    The CLI log format includes terminal-oriented details (indentation, a
    settings dump, full server-side paths) that don't read well in a browser.
    """
    text = message.strip()
    if text.startswith(_NOISY_LINE_PREFIXES):
        return None
    exported = _EXPORTED_LINE.match(text)
    if exported:
        return f"Exported {_basename(exported.group(1))}"
    transcribing = _TRANSCRIBING_LINE.match(text)
    if transcribing:
        return f"{transcribing.group(1)} {_basename(transcribing.group(2))}"
    return text


class _JobLogHandler(logging.Handler):
    """Forwards log records into a Job's log so the UI can poll and display them."""

    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord):
        try:
            raw = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken log call must not kill a run
            return
        cleaned = _clean_log_line(raw)
        if cleaned is None:
            return
        stage = _STAGE_LINE.match(cleaned)
        if stage:
            self.job.note_stage(int(stage.group(1)))
        prefix = {"WARNING": "⚠ ", "ERROR": "✖ "}.get(record.levelname, "")
        self.job.add_log(f"{prefix}{cleaned}", record.levelname)


_lock = threading.Lock()
_jobs: "OrderedDict[str, Job]" = OrderedDict()
_active_job_id: str | None = None
# Held between "this request won the slot" and "the worker thread is running",
# so two submissions arriving together cannot both get past the busy check.
_starting = False


def _remember(job: Job):
    """Keeps a bounded history of jobs so old download links stay alive."""
    _jobs[job.id] = job
    while len(_jobs) > MAX_REMEMBERED_JOBS:
        _jobs.popitem(last=False)


def _get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def _latest_job() -> Job | None:
    with _lock:
        return next(reversed(_jobs.values())) if _jobs else None


def _running_job() -> Job | None:
    with _lock:
        job = _jobs.get(_active_job_id) if _active_job_id else None
    return job if job and not job.done else None


def _build_args(
    language, model, mode, diarize, speakers, speaker_names, meeting_name,
    fmt, export_content, cleanup_model, summary_model, summary_language, output_dir,
    file_count: int, workers: int = 1,
    summary_preset: str = prompts.DEFAULT_PRESET, summary_prompt: str = "",
):
    """Builds the plain object main.process_single()/resolve_model() expect."""
    is_batch = file_count > 1
    workers = resolve_workers(workers, file_count)
    return argparse.Namespace(
        language=language,
        model=model,
        mode=mode,
        no_diarize=not diarize,
        speakers=None if speakers == "auto" else int(speakers),
        speaker_names=(speaker_names or None) if (not is_batch and diarize) else None,
        meeting_name=(meeting_name or None) if not is_batch else None,
        format=fmt,
        export_content=export_content,
        cleanup_model=cleanup_model,
        summary_model=summary_model,
        summary_language=summary_language,
        summary_preset=summary_preset,
        summary_prompt=summary_prompt or None,
        summary_prompt_file=None,
        output_dir=output_dir,
        # main.resolve_model divides the RAM budget by this, so it has to be
        # the real worker count and not just a flag.
        parallel=workers > 1,
        parallel_workers=workers,
    )


def resolve_workers(requested, file_count: int) -> int:
    """Clamps the requested worker count to something that can actually run.

    Capped by the number of files as well as by MAX_PARALLEL_WORKERS: asking for
    four workers to process two files leaves two idle, and main.resolve_model
    would still divide the memory budget by four and pick a smaller Whisper
    model than the machine can afford.
    """
    try:
        workers = int(requested)
    except (TypeError, ValueError):
        workers = 1
    return max(1, min(workers, MAX_PARALLEL_WORKERS, max(file_count, 1)))


def resolve_speakers(speakers: str, custom: str) -> str:
    """Turns the speaker dropdown plus its 'More than 10' box into one value.

    Returns 'auto' or a digit string, so everything downstream keeps seeing the
    same shape it always did.
    """
    if speakers != "custom":
        return speakers or "auto"
    return (custom or "").strip()


def _validate_submission(diarize: bool, mode: str, export_content: str, fmt: str, speakers: str,
                         language: str = "auto", summary_language: str = "same",
                         summary_preset: str = prompts.DEFAULT_PRESET,
                         summary_prompt: str = "") -> str | None:
    """Mirrors main.validate_args for the combinations the form can produce."""
    if diarize and pipeline.hf_token_is_placeholder(config.HF_TOKEN):
        return (
            "HF_TOKEN is missing from config.py, which is required for speaker identification. "
            "Add it, or turn off “Identify speakers”."
        )
    if export_content == "summary" and mode in pipeline.MODES_WITHOUT_SUMMARY:
        return "“Summary only” needs a mode that produces a summary. Pick Full or Summary."
    if fmt == "srt" and export_content == "summary":
        return "Subtitles only carry the timestamped transcript, so they cannot hold a summary alone."
    if speakers != "auto" and (not speakers.isdigit() or int(speakers) < 1):
        return (
            "Enter how many speakers to expect as a whole number of 1 or more, or leave it on Auto."
            if not speakers else f"“{speakers}” is not a valid number of speakers."
        )
    if not languages.is_supported(language):
        return f"“{language}” is not a language Whisper can transcribe."
    if summary_language != "same" and not languages.is_supported(summary_language):
        return f"“{summary_language}” is not a language the summary can be written in."
    if summary_preset not in prompts.PRESET_KEYS:
        return f"Unknown summary preset “{summary_preset}”."
    if summary_preset == prompts.CUSTOM_KEY and not summary_prompt.strip():
        return "Pick a summary preset, or write your own instructions in the box."
    if mode not in pipeline.MODE_CHOICES:
        return f"Unknown pipeline mode “{mode}”."
    if fmt not in pipeline.FORMAT_CHOICES:
        return f"Unknown export format “{fmt}”."
    return None


def _status_response(request: Request, job: Job | None, error: str | None = None) -> HTMLResponse:
    return templates.TemplateResponse(request, "_status.html", {"job": job, "error": error})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    job = _running_job() or _latest_job()
    return templates.TemplateResponse(
        request, "index.html", {**_index_context(), "job": job, "error": None}
    )


@app.get("/api/health")
def health():
    """Everything the header chips report: hardware, Ollama, HF token, ffmpeg."""
    info = system_info()
    reachable, models = probe_ollama(config.OLLAMA_HOST)
    installed_whisper = info["whisper_installed"]

    return JSONResponse({
        "device": info["device"],
        "hardware": (
            f"{info['gpu']} · {info['vram_gb']} GB VRAM" if info["gpu"]
            else f"CPU · {info['ram_gb']} GB free RAM"
        ),
        "ollama_online": reachable,
        # The LLM steps need a reachable server *and* something to run on it.
        "ollama_usable": reachable and bool(models),
        "ollama_host": config.OLLAMA_HOST,
        "ollama_models": models,
        "cleanup_model": _resolve_ollama_model(config.DEFAULT_CLEANUP_MODEL, models),
        "summary_model": _resolve_ollama_model(config.DEFAULT_SUMMARY_MODEL, models),
        "configured_cleanup_model": config.DEFAULT_CLEANUP_MODEL,
        "configured_summary_model": config.DEFAULT_SUMMARY_MODEL,
        "whisper_installed": installed_whisper,
        "whisper_download_mb": MODEL_DOWNLOAD_MB,
        "diarization_ready": not pipeline.hf_token_is_placeholder(config.HF_TOKEN),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "max_workers": MAX_PARALLEL_WORKERS,
    })


def _resolve_ollama_model(configured: str, installed: list[str]) -> str:
    """Which model the form should preselect.

    The configured one when it is actually installed, otherwise the first thing
    Ollama does have. Listing a model that isn't there only invites picking it.
    """
    if not installed:
        return configured
    return configured if model_available(installed, configured) else installed[0]


@app.post("/run", response_class=HTMLResponse)
def run(
    request: Request,
    audio_files: list[UploadFile] = File(...),
    language: str = Form("auto"),
    model: str = Form("auto"),
    mode: str = Form("full"),
    diarize: str | None = Form(None),
    speakers: str = Form("auto"),
    speakers_custom: str = Form(""),
    speaker_names: str = Form(""),
    meeting_name: str = Form(""),
    format: str = Form("md"),
    export_content: str = Form("full"),
    cleanup_model: str = Form("llama3.1"),
    summary_model: str = Form("llama3.1"),
    summary_language: str = Form("same"),
    summary_preset: str = Form(prompts.DEFAULT_PRESET),
    summary_prompt: str = Form(""),
    parallel_workers: str = Form("1"),
):
    global _active_job_id, _starting

    with _lock:
        active = _jobs.get(_active_job_id) if _active_job_id else None
        if _starting or (active is not None and not active.done):
            return _status_response(request, active, "A run is already in progress.")
        # Claimed here and released in the finally below, so nothing between
        # this point and the worker thread starting can leave the app wedged
        # on a run that never actually began.
        _starting = True

    try:
        return _start_run(
            request, audio_files, language, model, mode, bool(diarize),
            resolve_speakers(speakers, speakers_custom),
            speaker_names, meeting_name, format, export_content,
            cleanup_model, summary_model, summary_language, parallel_workers,
            summary_preset, summary_prompt,
        )
    finally:
        with _lock:
            _starting = False


def _start_run(
    request, audio_files, language, model, mode, wants_diarization, speakers,
    speaker_names, meeting_name, fmt, export_content,
    cleanup_model, summary_model, summary_language, parallel_workers="1",
    summary_preset=prompts.DEFAULT_PRESET, summary_prompt="",
) -> HTMLResponse:
    """Validates the submission, saves the uploads, and kicks off the worker."""
    global _active_job_id

    problem = _validate_submission(wants_diarization, mode, export_content, fmt, speakers,
                                   language, summary_language, summary_preset, summary_prompt)
    if problem:
        return _status_response(request, _latest_job(), problem)

    uploads = [uf for uf in audio_files if uf.filename]
    if not uploads:
        return _status_response(request, _latest_job(), "Choose at least one audio file first.")

    unsupported = [
        Path(uf.filename).name for uf in uploads
        if Path(uf.filename).suffix.lower() not in SUPPORTED_EXTENSIONS
    ]
    if unsupported:
        return _status_response(
            request, _latest_job(),
            f"Unsupported file type: {', '.join(unsupported)}. Accepted: "
            f"{', '.join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))}.",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="minute-ai-"))
    try:
        audio_paths = []
        for uf in uploads:
            dest = tmp_dir / Path(uf.filename).name
            with open(dest, "wb") as out:
                shutil.copyfileobj(uf.file, out)
            audio_paths.append(str(dest))

        is_batch = len(audio_paths) > 1
        args = _build_args(
            language, model, mode, wants_diarization, speakers, speaker_names, meeting_name,
            fmt, export_content, cleanup_model, summary_model, summary_language,
            config.DEFAULT_OUTPUT_DIR,
            len(audio_paths), workers=parallel_workers,
            summary_preset=summary_preset, summary_prompt=summary_prompt,
        )
        args.model = pipeline.resolve_model(args)
        job = Job(str(uuid.uuid4()), [Path(p).name for p in audio_paths], workers=args.parallel_workers)
    except Exception as exc:  # noqa: BLE001 - never leave the app stuck on a half-started run
        shutil.rmtree(tmp_dir, ignore_errors=True)
        get_logger().exception("Could not start the run")
        return _status_response(request, _latest_job(), f"Could not start the run: {exc}")

    handler = _JobLogHandler(job)
    logger = get_logger()

    def process_one(index: int, audio_path: str, entry: FileResult):
        if not job.is_parallel:
            job.current_index = index
            job.stage = 0

        if job.cancel_requested.is_set():
            entry.status = "skipped"
            entry.message = "Stopped before this file started."
            return

        try:
            result = pipeline.process_single(audio_path, args, is_batch)
            _record_success(job, entry, result)
        except MinuteAIError as exc:
            _record_failure(job, entry, str(exc))
        except Exception as exc:  # noqa: BLE001 - surface whisperX/Ollama failures, don't crash the app
            _record_failure(job, entry, str(exc) or exc.__class__.__name__)
        finally:
            with _results_lock:
                job.completed += 1

    def worker():
        global _active_job_id
        try:
            # One FileResult per upload, created up front so the results panel
            # keeps the order you dropped the files in even when they finish
            # out of order.
            entries = [FileResult(Path(p).name) for p in audio_paths]
            job.results.extend(entries)

            if job.is_parallel:
                logger.info(f"Processing {len(audio_paths)} files, {job.workers} at a time.")
                with ThreadPoolExecutor(max_workers=job.workers) as pool:
                    futures = [
                        pool.submit(process_one, i, path, entry)
                        for i, (path, entry) in enumerate(zip(audio_paths, entries, strict=True))
                    ]
                    for future in futures:
                        future.result()  # process_one swallows its own failures
            else:
                for i, (path, entry) in enumerate(zip(audio_paths, entries, strict=True)):
                    process_one(i, path, entry)

            job.current_index = len(audio_paths)
        finally:
            logger.removeHandler(handler)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            job.status = "cancelled" if job.cancel_requested.is_set() else "done"
            job.stage = TOTAL_STAGES
            job.finished_at = time.time()
            job.done = True
            with _lock:
                if _active_job_id == job.id:
                    _active_job_id = None

    with _lock:
        _remember(job)
        _active_job_id = job.id
    logger.addHandler(handler)

    try:
        threading.Thread(target=worker, daemon=True, name=f"minute-ai-job-{job.id[:8]}").start()
    except Exception as exc:  # noqa: BLE001 - out of threads: undo the claim rather than hang
        logger.removeHandler(handler)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        job.done = True
        job.status = "done"
        job.finished_at = time.time()
        with _lock:
            _active_job_id = None
        return _status_response(request, _latest_job(), f"Could not start the run: {exc}")

    return _status_response(request, job, None)


_results_lock = threading.Lock()


def _record_success(job: Job, entry: FileResult, result):
    entry.status = "done"
    entry.meeting_name = result.meeting_name
    entry.language = result.language
    entry.speakers = result.speakers
    entry.duration = format_duration(result.duration_seconds) if result.duration_seconds else ""
    entry.summary_text = result.summary
    entry.transcript_text = result.transcript
    entry.summary_html = render_markdown(result.summary)
    entry.transcript_html = render_transcript(result.transcript)
    # Several workers can land here at once, and these are the job-wide lists
    # the status page reads on every poll.
    with _results_lock:
        for out_path in result.output_files:
            p = Path(out_path)
            entry.files.append(p.name)
            job.output_files.append(p.name)
            job.output_paths[p.name] = p


def _record_failure(job: Job, entry: FileResult, message: str):
    entry.status = "failed"
    entry.message = message
    with _results_lock:
        job.errors.append((entry.name, message))


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    job = _get_job(job_id)
    error = None if job else "That run is no longer available. Start a new one."
    return _status_response(request, job, error)


@app.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
def cancel_job(request: Request, job_id: str):
    job = _get_job(job_id)
    if not job:
        return _status_response(request, None, "That run is no longer available.")
    if not job.done:
        job.cancel_requested.set()
        job.add_log("⚠ Stop requested. Finishing the current file, then stopping.", "WARNING")
    return _status_response(request, job, None)


@app.get("/download/{job_id}/all")
def download_all(job_id: str):
    job = _get_job(job_id)
    if not job or not job.output_paths:
        raise HTTPException(status_code=404, detail="No files to download.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in job.output_paths.items():
            if path.exists():
                zf.write(path, arcname=name)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=minute-ai-export.zip"},
    )


@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    job = _get_job(job_id)
    if not job or filename not in job.output_paths:
        raise HTTPException(status_code=404, detail="File not found.")
    path = job.output_paths[filename]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
