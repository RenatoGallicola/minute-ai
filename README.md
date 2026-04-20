# minute-ai 🎙️

Local pipeline for transcription, cleanup, and summarization of meeting audio.  
Everything runs on your PC — no data is sent to external servers.

## Stack

- **[whisperX](https://github.com/m-bain/whisperX)** — transcription + speaker diarization
- **[Ollama](https://ollama.com)** — local LLM for cleanup and summarization
- Output as **Markdown**, **txt**, **docx**, or **PDF** ready for Notion

---

## Installation

### 1. Prerequisites

- Python 3.10–3.12
- [Ollama](https://ollama.com) installed and running
- [HuggingFace](https://huggingface.co) token (free, required for diarization)

### 2. Clone the repo and create the venv

```bash
git clone https://github.com/yourname/minute-ai.git
cd minute-ai
py -3.12 -m venv venv
venv\Scripts\activate        # Windows cmd
# source venv/bin/activate   # Mac/Linux
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install whisperx requests python-docx reportlab
```

### 3. Configure

```bash
copy config.example.py config.py   # Windows
# cp config.example.py config.py   # Mac/Linux
```

Open `config.py` and insert your `HF_TOKEN`.

### 4. Download the LLM model

```bash
ollama pull llama3.1
```

---

## Usage

```bash
# Activate the venv (every time you open a new terminal)
venv\Scripts\activate

# Single file — full pipeline, markdown output
python main.py inputs/meeting.m4a

# Single file — transcript only, plain text
python main.py inputs/meeting.m4a --mode transcript --format txt

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

# Full usage example
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

- **full** — best quality output: transcript is cleaned before summarization
- **transcript** — fastest, just raw transcription with speaker labels
- **clean** — cleaned transcript without summary (e.g. for manual review)
- **summary** — quick summary without cleanup step (slightly lower quality)

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
├── config.py           # Local config (do not commit!)
├── config.example.py   # Config template
└── requirements.txt
```

---

## Roadmap

- [ ] Graphical user interface (GUI)
- [ ] Direct Notion API integration
- [ ] Automatic model selection based on available RAM
