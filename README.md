# minute-ai 🎙️

Local pipeline for transcription, cleanup, and summarization of meeting audio.  
Everything runs on your PC — no data is sent to external servers.

## Stack

- **[whisperX](https://github.com/m-bain/whisperX)** — transcription + speaker diarization
- **[Ollama](https://ollama.com)** — local LLM for cleanup and summarization
- Output as **Markdown**, **txt**, **docx**, or **PDF** ready for Notion
- CLI or a local **FastAPI + htmx** web GUI, your choice

---

## Requirements

- Python **3.10–3.12** (Python 3.13+ is not supported by whisperX)
- [Ollama](https://ollama.com) installed and running
- [HuggingFace](https://huggingface.co) account and token (free, required for diarization)
  - Not required if using `--no-diarize`

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/RenatoGallicola/minute-ai.git
cd minute-ai
```

### 2. Create and activate the virtual environment

```bash
py -3.12 -m venv venv
venv\Scripts\activate        # Windows cmd
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install --upgrade pip

# Install PyTorch first (CPU version)
pip install torch torchvision torchaudio
# For GPU (CUDA) see: https://pytorch.org/get-started/locally/

# Install all other dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` contains all pinned transitive dependencies for reproducibility.  
> `requirements.in` lists only the direct dependencies for reference.

### 4. Configure

```bash
copy config.example.py config.py   # Windows
# cp config.example.py config.py   # Mac/Linux
```

Open `config.py` and insert your `HF_TOKEN` from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  
Not needed if you always use `--no-diarize`.

### 5. Download the LLM model

```bash
ollama pull llama3.1
```

### 6. First run — download Whisper and diarization models

On first run, whisperX will automatically download the transcription and diarization models (~2 GB total) to `~/.cache/huggingface/hub/`.

```bash
python main.py inputs/your_audio.m4a --mode transcript
```

> **Corporate network / restricted environment:**  
> If your network blocks HuggingFace or Ollama registries, download the models on an unrestricted machine and copy the cache folders manually:
> - Whisper + diarization: `C:\Users\username\.cache\huggingface\hub\` → same path on target machine
> - Ollama models: `C:\Users\username\.ollama\models\` → same path on target machine

---

## Usage

```bash
# Activate the venv (every time you open a new terminal)
venv\Scripts\activate

# Single file — full pipeline, markdown output
python main.py inputs/meeting.m4a

# Single file — transcript only, no diarization (single speaker)
python main.py inputs/meeting.m4a --no-diarize --mode transcript

# Single file — full pipeline, no diarization
python main.py inputs/meeting.m4a --no-diarize

# Single file — summary only exported as PDF
python main.py inputs/meeting.m4a --mode full --format pdf --export-content summary

# Single file — all formats
python main.py inputs/meeting.m4a --format all

# Entire folder, sequential
python main.py inputs/

# Entire folder, parallel
python main.py inputs/ --parallel

# Force reprocess already-processed files
python main.py inputs/ --force

# Full example
python main.py inputs/meeting.m4a \
    --language en \
    --speakers 2 \
    --speaker-names "Marco,Sara" \
    --meeting-name "Q3 Kickoff" \
    --model medium \
    --mode full \
    --summary-language it \
    --format docx \
    --export-content full
```

---

## GUI

Prefer a graphical interface over the CLI? minute-ai also ships a small local web app
(FastAPI + Jinja2 + [htmx](https://htmx.org), styled with precompiled [Tailwind CSS](https://tailwindcss.com))
that wraps the same pipeline:

```bash
venv\Scripts\activate
python gui.py
```

Open `http://127.0.0.1:7860` in your browser — everything still runs locally, the browser is just
the interface (no telemetry, no CDN calls: htmx and the compiled Tailwind CSS are vendored in `static/`).
Upload one or more audio files, pick your options, and download the generated files when done.
Pipeline logs stream to the page in near-real time (polled via htmx every second) while processing runs
in a background thread. Only one job runs at a time.

**If you edit `templates/*.html`** and use Tailwind classes that aren't in the compiled stylesheet yet,
regenerate it (requires Node.js, only for this one-off build step — not a runtime dependency):
```bash
npm install --no-save tailwindcss @tailwindcss/cli
./node_modules/.bin/tailwindcss -i static/src.css -o static/tailwind.css --minify
rm -rf node_modules package.json package-lock.json
```

---

## Pipeline modes

The `--mode` parameter controls which steps of the pipeline are executed:

| Mode | Transcribe | Cleanup | Summary |
|---|---|---|---|
| `full` *(default)* | ✓ | ✓ | ✓ |
| `transcript` | ✓ | — | — |
| `clean` | ✓ | ✓ | — |
| `summary` | ✓ | — | ✓ |

- **full** — best quality: transcript is cleaned before summarization
- **transcript** — fastest, raw transcription with speaker labels only
- **clean** — cleaned transcript without summary (useful for manual review)
- **summary** — quick summary without cleanup (slightly lower quality)

---

## Automatic model selection

By default, `--model auto` picks a whisper model based on available system RAM (via `psutil`),
so you don't have to know which model your machine can handle:

| Available RAM | Model |
|---|---|
| ≥ 16 GB | `large-v3` |
| ≥ 8 GB | `medium` |
| ≥ 4 GB | `small` |
| ≥ 2 GB | `base` |
| < 2 GB | `tiny` |

With `--parallel`, available RAM is divided across `--parallel-workers` before picking a model,
since each worker loads its own model instance.

To pin a specific model instead, pass `--model` explicitly (e.g. `--model medium`).

---

## Speaker diarization

By default, minute-ai identifies who said what using speaker diarization.  
Use `--no-diarize` to skip this step entirely:

- Faster processing
- No HuggingFace token required
- Output is plain text without speaker labels
- Ideal for single-speaker recordings (interviews, voice memos, lectures)

> **Note:** `--no-diarize` is incompatible with `--speakers` and `--speaker-names`.

---

## Export content

The `--export-content` parameter controls what is included in the output file:

| Value | Includes |
|---|---|
| `full` *(default)* | Summary + full transcript |
| `summary` | Summary only |

> **Note:** `--export-content summary` requires a mode that generates a summary (`full` or `summary`).  
> Using it with `--mode transcript` or `--mode clean` will raise an error.

---

## All parameters

| Parameter | Default | Description |
|---|---|---|
| `--language` | `auto` | Audio language: `it`, `en`, `auto`, etc. |
| `--speakers` | `auto` | Number of speakers or `auto` (requires diarization) |
| `--speaker-names` | — | Speaker names: `"Marco,Sara"` (single file only, requires diarization) |
| `--meeting-name` | filename | Meeting name (single file only) |
| `--model` | `auto` | Whisper model: `auto` `tiny` `base` `small` `medium` `large-v3` |
| `--no-diarize` | — | Disable speaker diarization (single-speaker audio) |
| `--mode` | `full` | Pipeline mode: `full` `transcript` `clean` `summary` |
| `--cleanup-model` | `llama3.1` | Ollama model for cleanup |
| `--summary-model` | `llama3.1` | Ollama model for summary |
| `--summary-language` | `same` | Summary language: `same`, `it`, `en` |
| `--output-dir` | `outputs/` | Output folder |
| `--format` | `md` | Output format: `md`, `txt`, `docx`, `pdf`, `all` |
| `--export-content` | `full` | Export content: `full`, `summary` |
| `--parallel` | — | Process multiple files in parallel |
| `--parallel-workers` | `2` | Number of parallel workers |
| `--force` | — | Reprocess files even if output already exists |

---

## Project structure

```
minute-ai/
├── src/
│   ├── transcribe.py     # whisperX: transcription + diarization
│   ├── cleanup.py        # Ollama: transcript cleanup
│   ├── summarize.py      # Ollama: structured summary
│   ├── ollama_client.py  # Shared HTTP client for Ollama
│   ├── export.py         # md / txt / docx / pdf export
│   ├── batch.py          # Batch processing logic
│   ├── hardware.py       # RAM-based auto-selection of the whisper model
│   └── logger.py         # Centralized logging
├── templates/            # Jinja2 templates for the web GUI
├── static/               # Vendored htmx + precompiled Tailwind CSS for the GUI
├── tests/               # pytest unit tests
├── inputs/              # Audio files (git-ignored)
├── outputs/             # Generated files (git-ignored)
├── logs/                # Log files (git-ignored)
├── main.py              # CLI entry point
├── gui.py               # FastAPI web GUI (wraps the same pipeline)
├── config.py            # Local config with tokens (do not commit!)
├── config.example.py    # Config template (safe to commit)
├── requirements.in      # Direct dependencies (human-maintained)
└── requirements.txt     # All pinned dependencies (auto-generated)
```

---

## Running tests

```bash
venv\Scripts\activate
pytest
```

Tests don't require Ollama, whisperX models, or a real `config.py` — external calls are mocked and a stub config is injected automatically.

---

## Dependency management

This project uses a two-file approach for dependencies, managed with [pip-tools](https://github.com/jazzband/pip-tools):

- **`requirements.in`** — lists only the packages you directly depend on, without version pins. Edit this file when adding or removing dependencies.
- **`requirements.txt`** — generated by `pip-compile` from `requirements.in`. Contains all transitive dependencies with exact pinned versions for full reproducibility. Never edit this file by hand.

One-time setup:
```bash
pip install pip-tools
```

To add, remove, or update a dependency:
```bash
# 1. Edit requirements.in (add/remove the package name)
# 2. Regenerate requirements.txt deterministically:
pip-compile requirements.in --output-file requirements.txt

# 3. Install the updated lockfile in your venv:
pip install -r requirements.txt

# 4. Commit both files together
git add requirements.in requirements.txt
git commit -m "chore: add <new-package>"
```

There is no automation (hook or CI) that regenerates `requirements.txt` — running `pip-compile` is a manual step you must remember before committing a dependency change.

---

## Known warnings (safe to ignore)

- **torchcodec not installed** — whisperX uses ffmpeg directly as fallback, no action needed
- **Lightning checkpoint upgrade** — cosmetic warning from pyannote, does not affect results
- **symlinks warning on Windows** — HuggingFace cache works in degraded mode, files are duplicated but functional. To fix, enable Windows Developer Mode.
