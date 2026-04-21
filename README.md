# minute-ai 🎙️

Local pipeline for transcription, cleanup, and summarization of meeting audio.  
Everything runs on your PC — no data is sent to external servers.

## Stack

- **[whisperX](https://github.com/m-bain/whisperX)** — transcription + speaker diarization
- **[Ollama](https://ollama.com)** — local LLM for cleanup and summarization
- Output as **Markdown**, **txt**, **docx**, or **PDF** ready for Notion

---

## Requirements

- Python **3.10–3.12** (Python 3.13+ is not supported by whisperX)
- [Ollama](https://ollama.com) installed and running
- [HuggingFace](https://huggingface.co) account and token (free, required for diarization)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourname/minute-ai.git
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

# Single file — transcript only
python main.py inputs/meeting.m4a --mode transcript

# Single file — summary exported as PDF
python main.py inputs/meeting.m4a --mode full --format pdf --export-content summary

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
| `--speakers` | `auto` | Number of speakers or `auto` |
| `--speaker-names` | — | Speaker names: `"Marco,Sara"` (single file only) |
| `--meeting-name` | filename | Meeting name (single file only) |
| `--model` | `medium` | Whisper model: `tiny` `small` `medium` `large-v3` |
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
│   ├── transcribe.py   # whisperX: transcription + diarization
│   ├── cleanup.py      # Ollama: transcript cleanup
│   ├── summarize.py    # Ollama: structured summary
│   ├── export.py       # md / txt / docx / pdf export
│   ├── batch.py        # Batch processing logic
│   └── logger.py       # Centralized logging
├── inputs/             # Audio files (git-ignored)
├── outputs/            # Generated files (git-ignored)
├── logs/               # Log files (git-ignored)
├── main.py             # Entry point
├── config.py           # Local config with tokens (do not commit!)
├── config.example.py   # Config template (safe to commit)
├── requirements.in     # Direct dependencies (human-maintained)
└── requirements.txt    # All pinned dependencies (auto-generated)
```

---

## Dependency management

This project uses a two-file approach for dependencies:

- **`requirements.in`** — lists only the packages you directly depend on, without version pins. Edit this file when adding or removing dependencies.
- **`requirements.txt`** — generated automatically by the pre-commit hook via `pip freeze`. Contains all transitive dependencies with exact versions for full reproducibility.

To update dependencies after adding a new package:
```bash
pip install <new-package>
git add .
git commit -m "chore: add <new-package>"  # pre-commit hook updates requirements.txt automatically
```

---

## Known warnings (safe to ignore)

- **torchcodec not installed** — whisperX uses ffmpeg directly as fallback, no action needed
- **Lightning checkpoint upgrade** — cosmetic warning from pyannote, does not affect results
- **symlinks warning on Windows** — HuggingFace cache works in degraded mode, files are duplicated but functional. To fix, enable Windows Developer Mode.

---

## Roadmap

- [ ] Graphical user interface (GUI)
- [ ] Direct Notion API integration
- [ ] Automatic model selection based on available RAM
