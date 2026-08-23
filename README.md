# minute-ai

Local pipeline for transcription, cleanup, and summarization of meeting audio.  
Everything runs on your PC — no data is sent to external servers.

## Stack

- **[whisperX](https://github.com/m-bain/whisperX)** — transcription + speaker diarization
- **[Ollama](https://ollama.com)** — local LLM for cleanup and summarization
- Output as **Markdown**, **txt**, **docx**, **PDF**, or **SRT** subtitles
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
ollama pull llama3.1        # 8B, ~4.9 GB — best quality, wants a GPU or plenty of free RAM
# ollama pull llama3.2:3b   # 3B, ~2.0 GB — much faster on a CPU-only machine
```

Whichever you pull, set it as `DEFAULT_CLEANUP_MODEL` / `DEFAULT_SUMMARY_MODEL` in `config.py`.
On a CPU-only machine prefer the 3B model: the cleanup step sends the whole transcript through the
LLM, so an 8B model turns a five-minute recording into a long wait. The GUI's *Advanced* section
lists exactly what Ollama has installed, so you can switch per run without editing config.

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
(FastAPI + Jinja2 + [htmx](https://htmx.org)) that wraps the same pipeline:

```bash
venv\Scripts\activate
python gui.py
```

Open `http://127.0.0.1:7860` in your browser — everything still runs locally, the browser is just
the interface. No telemetry and no CDN calls: htmx is vendored in `static/`, and the stylesheet
(`static/app.css`) is hand-written, so **there is no build step and no Node.js involved** — edit the
CSS or the templates and reload the page.

What the GUI adds on top of the CLI:

- **Status chips** in the header: detected hardware, whether Ollama is up and which models it has,
  whether diarization is configured, and whether ffmpeg is on PATH (hover any chip for the detail)
- **Model picker** under *Advanced*: a dropdown of the models Ollama actually has installed —
  nothing else is offered, because nothing else can run. If `config.py` names a model that isn't
  installed, the picker falls back to one that is and says so
- **Options switch themselves off when they can't work**: with Ollama down or empty, the Full,
  Cleaned and Summary modes, the "summary only" export and the whole model section are disabled,
  the raw transcript is selected instead, and a note explains why
- **Whisper models say what they cost**: each entry in the model list is marked *ready* when it is
  already downloaded, or shows the download size (up to 3 GB) when it is not
- **Run several files at once**: an optional worker count, shown as soon as you pick more than one
  file
- **Drag & drop** upload, with per-file removal before you start
- **Live progress**: a four-stage tracker (transcribe -> clean up -> summarize -> export), a progress
  bar, elapsed time, and the pipeline log streamed to the page (polled once a second)
- **Results in the browser**: the summary rendered as formatted text, the transcript with speaker
  labels, one-click copy, plus per-file and zipped downloads
- **Light and dark themes**, following your OS by default
- **Stop after the current file** when processing a batch

Processing runs in a background thread and only one job runs at a time. Download links from earlier
runs keep working (the last 20 runs are remembered for the lifetime of the process).

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

By default, `--model auto` picks a whisper model based on the hardware it finds, so you don't have
to know which model your machine can handle. A CUDA GPU is preferred when present (the model is
loaded into VRAM); otherwise the choice is made from free system RAM (via `psutil`):

| Free RAM (CPU) | GPU VRAM | Model |
|---|---|---|
| ≥ 16 GB | ≥ 7 GB | `large-v3` |
| ≥ 8 GB | ≥ 4 GB | `medium` |
| ≥ 4 GB | ≥ 2 GB | `small` |
| ≥ 2 GB | ≥ 1 GB | `base` |
| < 2 GB | < 1 GB | `tiny` |

With `--parallel`, the budget is divided across `--parallel-workers` before picking a model,
since each worker loads its own model instance.

`DEFAULT_COMPUTE_TYPE = "auto"` in `config.py` picks the matching precision too: `float16` on a
GPU, `int8` on CPU.

Among the models that fit, one already downloaded always wins. `auto` is meant to just work, and
silently starting a multi-gigabyte download — which fails outright on a restricted network — is the
opposite of that. If nothing is downloaded yet, the hardware pick is used and the log says it will
be fetched on first use.

To pin a specific model instead, pass `--model` explicitly (e.g. `--model medium`). Whichever you
pick, whisperX downloads it on first use if it isn't in `~/.cache/huggingface/hub` already.

---

## Transcript formatting

Consecutive segments from the same speaker are joined into one paragraph, which is broken whenever
they pause for more than 2 seconds or the paragraph passes ~700 characters. Only the first paragraph
of a turn carries the speaker label, so continuation paragraphs read as the same person still
talking. Without this a single-speaker recording — a lecture, an interview, a voice memo — came out
as one unbroken wall of text.

---

## CLI and GUI parity

Both front ends drive the same pipeline and expose the same options — including all 100 Whisper
languages, any speaker count, and parallel processing. `tests/test_cli_gui_parity.py` compares the
two automatically and fails if an option ever lands on one side only.

Three CLI options have no GUI equivalent, on purpose:

| CLI option | Why the GUI has no equivalent |
|---|---|
| `--recursive` | The GUI works on browser uploads, not server-side paths — there is no folder to descend into |
| `--force` | Uploading a file *is* the request to process it, so the GUI always behaves as if `--force` were set |
| `--output-dir` | Writing to an arbitrary server path from a web form is a footgun; the GUI uses `DEFAULT_OUTPUT_DIR` and hands the files back as downloads |

---

## Processing several files at once

`--parallel` (CLI) and the *Files at a time* control under *Advanced* (GUI) run several recordings
concurrently. Each worker loads its own Whisper model, so memory use scales with the worker count —
which is why `--model auto` divides the RAM budget by `--parallel-workers` before choosing a tier,
and why the GUI caps the count at 4.

Note that Ollama serialises generation requests by default, so on a single machine the cleanup and
summary steps queue up regardless. The win is in the transcription stage.

---

## Long recordings

Local models have a limited context window, and Ollama silently truncates anything longer rather
than erroring — which would quietly leave most of a long meeting out of the summary. minute-ai
avoids that:

- The context window is requested explicitly (`OLLAMA_NUM_CTX` in `config.py`, default `8192`)
- Transcripts longer than `LLM_CHUNK_CHARS` (default `6000`) are split on speaker boundaries,
  cleaned chunk by chunk, and summarized with a map-then-merge pass

Raise `OLLAMA_NUM_CTX` and `LLM_CHUNK_CHARS` together if your machine has memory to spare — fewer,
larger passes generally give a better summary.

---

## Speaker diarization

By default, minute-ai identifies who said what using speaker diarization.  
Use `--no-diarize` to skip this step entirely:

- Faster processing
- No HuggingFace token required
- Output is plain text without speaker labels
- Ideal for single-speaker recordings (interviews, voice memos, lectures)

> **Note:** `--no-diarize` is incompatible with `--speakers` and `--speaker-names`.

`--speakers 1` skips diarization altogether: there is nothing to tell apart, and running pyannote
anyway costs minutes to confirm what you already said. The transcript still carries the label (or the
name you gave), so `--speaker-names "Marco"` works as expected.

`--speaker-names` is matched against the speakers in order of first appearance. If you give more
names than there are speakers the extras are ignored; if you give fewer, the remaining speakers keep
their `SPEAKER_xx` labels. Either way a warning says so, rather than leaving you to spot it in the
export.

---

## Export content

The `--export-content` parameter controls what is included in the output file:

| Value | Includes |
|---|---|
| `full` *(default)* | Summary + full transcript |
| `summary` | Summary only |

> **Note:** `--export-content summary` requires a mode that generates a summary (`full` or `summary`).  
> Using it with `--mode transcript` or `--mode clean` will raise an error.

`--format srt` writes timestamped subtitles built from the aligned whisperX segments (with speaker
labels when diarization ran). Subtitles carry the transcript and nothing else, so `srt` cannot be
combined with `--export-content summary`. `--format all` writes every format, `.srt` included.

---

## All parameters

| Parameter | Default | Description |
|---|---|---|
| `--recursive` / `-r` | — | When a folder is given, also look inside sub-folders |
| `--language` | `auto` | Audio language: `auto` or any of the 100 codes Whisper supports (`it`, `en`, `tr`, `yue`, …) |
| `--speakers` | `auto` | Number of speakers or `auto` (requires diarization) |
| `--speaker-names` | — | Speaker names in order of first appearance: `"Marco,Sara"` (single file only, requires diarization) |
| `--meeting-name` | filename | Meeting name (single file only) |
| `--model` | `auto` | Whisper model: `auto` `tiny` `base` `small` `medium` `large-v3` |
| `--no-diarize` | — | Disable speaker diarization (single-speaker audio) |
| `--mode` | `full` | Pipeline mode: `full` `transcript` `clean` `summary` |
| `--cleanup-model` | `llama3.1` | Ollama model for cleanup |
| `--summary-model` | `llama3.1` | Ollama model for summary |
| `--summary-language` | `same` | Summary language: `same`, `it`, `en` |
| `--output-dir` | `outputs/` | Output folder |
| `--format` | `md` | Output format: `md`, `txt`, `docx`, `pdf`, `srt`, `all` |
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
│   ├── chunking.py       # Splits long transcripts to fit the LLM context window
│   ├── export.py         # md / txt / docx / pdf / srt export
│   ├── naming.py         # Portable output file names, shared by export + batch
│   ├── batch.py          # Batch processing logic
│   ├── hardware.py       # GPU/RAM-based auto-selection of the whisper model
│   ├── markdown_lite.py  # Small Markdown -> HTML renderer for the GUI preview
│   ├── errors.py         # Pipeline exception types
│   └── logger.py         # Centralized logging
├── templates/            # Jinja2 templates for the web GUI
├── static/               # Vendored htmx + hand-written app.css for the GUI
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

## What happens when something goes wrong

The pipeline degrades instead of throwing away work you have already paid for:

| Problem | What happens |
|---|---|
| Ollama not running, or the model isn't pulled | Cleanup and summary are skipped with a warning; the transcript is still exported |
| Diarization fails (bad or ungated HF token) | The transcript is kept and exported without speaker labels |
| No alignment model for the detected language | Timestamps stay unaligned; everything else continues |
| One file fails in a batch | The remaining files still run; the closing summary lists what failed and why |
| Audio can't be read (missing ffmpeg, corrupt file) | That file fails with a clear message, and the CLI exits non-zero |

---

## Known warnings (safe to ignore)

- **torchcodec not installed** — whisperX uses ffmpeg directly as fallback, no action needed
- **Lightning checkpoint upgrade** — cosmetic warning from pyannote, does not affect results
- **symlinks warning on Windows** — HuggingFace cache works in degraded mode, files are duplicated but functional. To fix, enable Windows Developer Mode.
