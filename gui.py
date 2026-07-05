#!/usr/bin/env python3
"""
gui.py
------
Local web GUI for minute-ai: FastAPI + Jinja2 + htmx, styled with precompiled Tailwind CSS.
Wraps the same pipeline used by main.py (main.process_single / main.resolve_model) —
no pipeline logic is duplicated here.

Run with:
    python gui.py
"""

import argparse
import logging
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import main as pipeline
from src.logger import get_logger, setup_logger


BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="minute-ai")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


LANGUAGES = [
    ("auto", "Auto-detect"), ("it", "Italian"), ("en", "English"),
    ("fr", "French"), ("de", "German"), ("es", "Spanish"), ("pt", "Portuguese"),
]
MODELS = [
    ("auto", "Automatic (based on available RAM)"), ("tiny", "Tiny — fastest"),
    ("base", "Base"), ("small", "Small"), ("medium", "Medium"), ("large-v3", "Large-v3 — most accurate"),
]
SPEAKER_COUNTS = [("auto", "Automatic")] + [(str(n), str(n)) for n in range(1, 7)]
MODES = [
    ("full", "Full — transcript + cleanup + summary"), ("transcript", "Transcript only"),
    ("clean", "Cleaned transcript, no summary"), ("summary", "Summary only"),
]
FORMATS = [
    ("md", "Markdown (.md)"), ("txt", "Plain text (.txt)"),
    ("docx", "Word (.docx)"), ("pdf", "PDF"), ("all", "All formats"),
]
EXPORT_CONTENTS = [("full", "Summary + full transcript"), ("summary", "Summary only")]
SUMMARY_LANGUAGES = [("same", "Same as transcript")] + LANGUAGES[1:]

INDEX_CONTEXT = {
    "languages": LANGUAGES,
    "models": MODELS,
    "default_model": config.DEFAULT_WHISPER_MODEL,
    "modes": MODES,
    "speaker_counts": SPEAKER_COUNTS,
    "formats": FORMATS,
    "default_format": config.DEFAULT_FORMAT,
    "export_contents": EXPORT_CONTENTS,
    "default_export_content": config.DEFAULT_EXPORT_CONTENT,
    "summary_languages": SUMMARY_LANGUAGES,
    "default_cleanup_model": config.DEFAULT_CLEANUP_MODEL,
    "default_summary_model": config.DEFAULT_SUMMARY_MODEL,
    "default_output_dir": config.DEFAULT_OUTPUT_DIR,
}


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.done = False
        self.log_lines: list[str] = []
        self.output_files: list[str] = []  # basenames, for display + download lookup
        self.output_paths: dict[str, Path] = {}
        self.errors: list[tuple[str, str]] = []

    @property
    def log_text(self) -> str:
        return "\n".join(self.log_lines)


class _JobLogHandler(logging.Handler):
    """Forwards log records into a Job's log so the UI can poll and display them."""

    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord):
        prefix = {"WARNING": "⚠ ", "ERROR": "✖ "}.get(record.levelname, "")
        self.job.log_lines.append(f"{prefix}{record.getMessage()}")


_lock = threading.Lock()
_current_job: Optional[Job] = None


def _build_args(
    language, model, mode, diarize, speakers, speaker_names, meeting_name,
    fmt, export_content, cleanup_model, summary_model, summary_language, output_dir,
    is_batch: bool,
):
    """Builds the plain object main.process_single()/resolve_model() expect."""
    return argparse.Namespace(
        language=language,
        model=model,
        mode=mode,
        no_diarize=not diarize,
        speakers=None if speakers == "auto" else int(speakers),
        speaker_names=(speaker_names or None) if not is_batch else None,
        meeting_name=(meeting_name or None) if not is_batch else None,
        format=fmt,
        export_content=export_content,
        cleanup_model=cleanup_model,
        summary_model=summary_model,
        summary_language=summary_language,
        output_dir=output_dir,
        parallel=False,
        parallel_workers=1,
    )


def _status_response(request: Request, job: Optional[Job], error: Optional[str] = None) -> HTMLResponse:
    return templates.TemplateResponse(request, "_status.html", {"job": job, "error": error})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {**INDEX_CONTEXT, "job": None, "error": None})


@app.post("/run", response_class=HTMLResponse)
def run(
    request: Request,
    audio_files: list[UploadFile] = File(...),
    language: str = Form("auto"),
    model: str = Form("auto"),
    mode: str = Form("full"),
    diarize: Optional[str] = Form(None),
    speakers: str = Form("auto"),
    speaker_names: str = Form(""),
    meeting_name: str = Form(""),
    format: str = Form("md"),
    export_content: str = Form("full"),
    cleanup_model: str = Form("llama3.1"),
    summary_model: str = Form("llama3.1"),
    summary_language: str = Form("same"),
    output_dir: str = Form("outputs"),
):
    global _current_job

    with _lock:
        if _current_job is not None and not _current_job.done:
            return _status_response(request, None, "A job is already running. Please wait for it to finish.")
        job = Job(str(uuid.uuid4()))
        _current_job = job

    if bool(diarize) and config.HF_TOKEN == "hf_XXXXXXXXXX":
        with _lock:
            _current_job = None
        return _status_response(
            request, None,
            "HF_TOKEN is missing from config.py, which is required for diarization. "
            "Add it, or turn off 'Identify speakers'.",
        )

    if export_content == "summary" and mode in ("transcript", "clean"):
        with _lock:
            _current_job = None
        return _status_response(
            request, None,
            "'Summary only' requires a mode that generates a summary "
            "('Full' or 'Summary only').",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="minute-ai-"))
    audio_paths = []
    for uf in audio_files:
        if not uf.filename:
            continue
        dest = tmp_dir / Path(uf.filename).name
        with open(dest, "wb") as out:
            shutil.copyfileobj(uf.file, out)
        audio_paths.append(str(dest))

    if not audio_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        with _lock:
            _current_job = None
        return _status_response(request, None, "Please upload at least one audio file.")

    is_batch = len(audio_paths) > 1
    args = _build_args(
        language, model, mode, bool(diarize), speakers, speaker_names, meeting_name,
        format, export_content, cleanup_model, summary_model, summary_language, output_dir,
        is_batch,
    )
    args.model = pipeline.resolve_model(args)

    handler = _JobLogHandler(job)
    logger = get_logger()
    logger.addHandler(handler)

    def worker():
        try:
            for audio_path in audio_paths:
                try:
                    for out_path in pipeline.process_single(audio_path, args, is_batch):
                        p = Path(out_path)
                        job.output_files.append(p.name)
                        job.output_paths[p.name] = p
                except BaseException as exc:  # noqa: BLE001 - surface Ollama/whisperX failures, don't crash the app
                    message = str(exc) or exc.__class__.__name__
                    job.errors.append((Path(audio_path).name, message))
        finally:
            logger.removeHandler(handler)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            job.done = True

    threading.Thread(target=worker, daemon=True).start()
    return _status_response(request, job, None)


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    with _lock:
        job = _current_job if _current_job and _current_job.id == job_id else None
    error = None if job else "No matching job found (a new run may have started since)."
    return _status_response(request, job, error)


@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    with _lock:
        job = _current_job if _current_job and _current_job.id == job_id else None
    if not job or filename not in job.output_paths:
        raise HTTPException(status_code=404, detail="File not found.")
    path = job.output_paths[filename]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename=filename)


if __name__ == "__main__":
    setup_logger()
    uvicorn.run(app, host="127.0.0.1", port=7860)
