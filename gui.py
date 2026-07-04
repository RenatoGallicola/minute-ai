#!/usr/bin/env python3
"""
gui.py
------
Gradio web interface for minute-ai. Wraps the same pipeline used by main.py
(src/transcribe.py, cleanup.py, summarize.py, export.py) — no logic is duplicated.

Run with:
    python gui.py
"""

import logging
import queue
import threading
import time
from pathlib import Path

import gradio as gr

import config
import main as pipeline
from src.logger import setup_logger, get_logger


LANGUAGE_CHOICES = [
    ("Rilevamento automatico", "auto"),
    ("Italiano", "it"),
    ("Inglese", "en"),
    ("Francese", "fr"),
    ("Tedesco", "de"),
    ("Spagnolo", "es"),
    ("Portoghese", "pt"),
]

MODEL_CHOICES = [
    ("Automatico (in base alla RAM disponibile)", "auto"),
    ("Tiny — più veloce", "tiny"),
    ("Base", "base"),
    ("Small", "small"),
    ("Medium", "medium"),
    ("Large-v3 — più accurato", "large-v3"),
]

SPEAKERS_CHOICES = [("Automatico", "auto")] + [(str(n), str(n)) for n in range(1, 7)]

MODE_CHOICES = [
    ("Completa — trascrizione + pulizia + riassunto", "full"),
    ("Solo trascrizione", "transcript"),
    ("Trascrizione pulita, senza riassunto", "clean"),
    ("Solo riassunto", "summary"),
]

FORMAT_CHOICES = [
    ("Markdown (.md)", "md"),
    ("Testo semplice (.txt)", "txt"),
    ("Word (.docx)", "docx"),
    ("PDF", "pdf"),
    ("Tutti i formati", "all"),
]

EXPORT_CONTENT_CHOICES = [
    ("Riassunto + trascrizione completa", "full"),
    ("Solo riassunto", "summary"),
]

SUMMARY_LANGUAGE_CHOICES = [("Stessa lingua della trascrizione", "same")] + LANGUAGE_CHOICES[1:]


class _QueueLogHandler(logging.Handler):
    """Forwards log records to a queue so the UI can display them as they happen."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        prefix = {"WARNING": "⚠ ", "ERROR": "✖ "}.get(record.levelname, "")
        self.log_queue.put(f"{prefix}{record.getMessage()}")


def _build_args(
    language, model, mode, diarize, speakers, speaker_names, meeting_name,
    fmt, export_content, cleanup_model, summary_model, summary_language, output_dir,
    is_batch: bool,
):
    """Builds the plain object main.process_single()/resolve_model() expect."""
    import argparse
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


def run_pipeline(
    audio_files, language, model, mode, diarize, speakers, speaker_names, meeting_name,
    fmt, export_content, cleanup_model, summary_model, summary_language, output_dir,
):
    if not audio_files:
        raise gr.Error("Carica almeno un file audio.")

    if diarize and config.HF_TOKEN == "hf_XXXXXXXXXX":
        raise gr.Error(
            "Manca l'HF_TOKEN in config.py, necessario per la diarizzazione. "
            "Aggiungilo oppure disattiva 'Identifica i parlanti'."
        )

    if export_content == "summary" and mode in ("transcript", "clean"):
        raise gr.Error(
            "'Solo riassunto' richiede una modalità che generi un riassunto "
            "('Completa' o 'Solo riassunto')."
        )

    audio_paths = [f if isinstance(f, str) else f.name for f in audio_files]
    is_batch = len(audio_paths) > 1
    args = _build_args(
        language, model, mode, diarize, speakers, speaker_names, meeting_name,
        fmt, export_content, cleanup_model, summary_model, summary_language, output_dir,
        is_batch,
    )
    args.model = pipeline.resolve_model(args)

    log_queue = queue.Queue()
    handler = _QueueLogHandler(log_queue)
    logger = get_logger()
    logger.addHandler(handler)

    result = {}

    def worker():
        output_files = []
        errors = []
        for audio_path in audio_paths:
            try:
                output_files.extend(pipeline.process_single(audio_path, args, is_batch))
            except BaseException as exc:  # noqa: BLE001 - surface Ollama/whisperX failures, don't crash the app
                errors.append((Path(audio_path).name, exc))
        result["files"] = output_files
        result["errors"] = errors

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    lines = [f"▶ Avvio elaborazione di {len(audio_paths)} file..."]
    yield "\n".join(lines), None, gr.update(interactive=False)

    while thread.is_alive():
        drained = False
        while not log_queue.empty():
            lines.append(log_queue.get_nowait())
            drained = True
        if drained:
            yield "\n".join(lines), None, gr.update(interactive=False)
        time.sleep(0.3)

    while not log_queue.empty():
        lines.append(log_queue.get_nowait())
    logger.removeHandler(handler)

    output_files = result.get("files", [])
    errors = result.get("errors", [])

    if errors:
        lines.append(f"\n✖ {len(errors)} file non completati:")
        for name, exc in errors:
            lines.append(f"   - {name}: {exc}")
    if output_files:
        lines.append(f"\n✔ Fatto — {len(output_files)} file generati.")
    elif not errors:
        lines.append("\n✖ Nessun file generato.")

    yield "\n".join(lines), (output_files or None), gr.update(interactive=True)


def _toggle_speaker_fields(diarize: bool):
    return gr.update(visible=diarize), gr.update(visible=diarize)


CSS = """
.gradio-container {max-width: 1040px !important; margin: 0 auto !important;}
#header {text-align: center; padding: 6px 0 2px 0;}
#header h1 {margin-bottom: 2px; font-size: 1.9em;}
#header p {color: var(--body-text-color-subdued); margin-top: 0;}
.log-box textarea {
    font-family: 'Consolas', 'SFMono-Regular', Menlo, monospace !important;
    font-size: 12.5px !important;
    background: #0f172a !important;
    color: #d6dee8 !important;
    border-radius: 10px !important;
}
#footer {text-align: center; opacity: 0.55; font-size: 12px; margin-top: 10px;}
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="minute-ai") as demo:
        gr.HTML(
            """
            <div id="header">
                <h1>🎙️ minute-ai</h1>
                <p>Trascrizione e riassunto riunioni — 100% locale, nessun dato lascia il tuo PC</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    audio_files = gr.File(
                        label="Audio (uno o più file)",
                        file_count="multiple",
                        file_types=["audio"],
                    )

                with gr.Group():
                    language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="Lingua audio", allow_custom_value=True)
                    model = gr.Dropdown(MODEL_CHOICES, value=config.DEFAULT_WHISPER_MODEL, label="Modello Whisper")
                    mode = gr.Radio(MODE_CHOICES, value=config.DEFAULT_MODE, label="Modalità pipeline")

                with gr.Group():
                    diarize = gr.Checkbox(value=True, label="Identifica i parlanti (diarizzazione)")
                    speakers = gr.Dropdown(SPEAKERS_CHOICES, value="auto", label="Numero di parlanti", visible=True)
                    speaker_names = gr.Textbox(
                        label="Nomi dei parlanti (in ordine, solo file singolo)",
                        placeholder="Marco,Sara",
                        visible=True,
                    )
                    meeting_name = gr.Textbox(
                        label="Nome riunione (solo file singolo)",
                        placeholder="Es. Q3 Kickoff — se vuoto, usa il nome del file",
                    )

                with gr.Group():
                    fmt = gr.Dropdown(FORMAT_CHOICES, value=config.DEFAULT_FORMAT, label="Formato di esportazione")
                    export_content = gr.Radio(
                        EXPORT_CONTENT_CHOICES, value=config.DEFAULT_EXPORT_CONTENT, label="Contenuto esportato"
                    )

                with gr.Accordion("Impostazioni avanzate", open=False):
                    cleanup_model = gr.Textbox(value=config.DEFAULT_CLEANUP_MODEL, label="Modello Ollama per la pulizia")
                    summary_model = gr.Textbox(value=config.DEFAULT_SUMMARY_MODEL, label="Modello Ollama per il riassunto")
                    summary_language = gr.Dropdown(
                        SUMMARY_LANGUAGE_CHOICES, value=config.DEFAULT_SUMMARY_LANGUAGE, label="Lingua del riassunto"
                    )
                    output_dir = gr.Textbox(value=config.DEFAULT_OUTPUT_DIR, label="Cartella di output")

                run_btn = gr.Button("▶ Genera", variant="primary", size="lg")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("**Log**")
                    log_box = gr.Textbox(
                        label=None, show_label=False, lines=20, interactive=False,
                        elem_classes=["log-box"], placeholder="I log dell'elaborazione appariranno qui...",
                    )
                with gr.Group():
                    gr.Markdown("**File generati**")
                    files_box = gr.File(label=None, show_label=False, file_count="multiple", interactive=False)

        gr.HTML('<div id="footer">minute-ai · whisperX + Ollama, in esecuzione interamente sul tuo PC</div>')

        diarize.change(_toggle_speaker_fields, inputs=diarize, outputs=[speakers, speaker_names])

        run_btn.click(
            run_pipeline,
            inputs=[
                audio_files, language, model, mode, diarize, speakers, speaker_names, meeting_name,
                fmt, export_content, cleanup_model, summary_model, summary_language, output_dir,
            ],
            outputs=[log_box, files_box, run_btn],
        )

    return demo


if __name__ == "__main__":
    setup_logger()
    demo = build_demo()
    demo.queue().launch(theme=THEME, css=CSS)
