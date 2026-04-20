# minute-ai 🎙️

Local pipeline for transcription, cleanup, and summarization of meeting audio.  
Everything runs on your PC — no data is sent to external servers.

## Stack

- **[whisperX](https://github.com/m-bain/whisperX)** — transcription + speaker diarization
- **[Ollama](https://ollama.com)** — local LLM for cleanup and summarization
- Output as **Markdown** ready for Notion

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
pip install whisperx requests
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

# Basic usage
python main.py inputs/meeting.m4a

# Full usage
python main.py inputs/meeting.m4a \
    --language en \
    --speakers 2 \
    --speaker-names "Marco,Sara" \
    --meeting-name "Q3 Kickoff" \
    --model medium \
    --summary-language it \
    --format md
```

### All parameters

| Parameter | Default | Description |
|---|---|---|
| `--language` | `auto` | Audio language: `it`, `en`, `auto`, etc. |
| `--speakers` | `auto` | Number of speakers or `auto` |
| `--speaker-names` | — | Speaker names in order: `"Marco,Sara"` |
| `--meeting-name` | filename | Human-readable meeting name |
| `--model` | `medium` | Whisper model: `tiny` `small` `medium` `large-v3` |
| `--no-cleanup` | — | Disable transcript cleanup |
| `--cleanup-model` | `llama3.1` | Ollama model for cleanup |
| `--no-summary` | — | Disable summary generation |
| `--summary-model` | `llama3.1` | Ollama model for summary |
| `--summary-language` | `same` | Summary language: `same`, `it`, `en` |
| `--output-dir` | `outputs/` | Output folder |
| `--format` | `md` | Output format: `md`, `txt`, `all` |

---

## Project structure

```
minute-ai/
├── src/
│   ├── transcribe.py   # whisperX: transcription + diarization
│   ├── cleanup.py      # Ollama: transcript cleanup
│   ├── summarize.py    # Ollama: structured summary
│   └── export.py       # Markdown/txt export
├── inputs/             # Audio files (git-ignored)
├── outputs/            # Generated files (git-ignored)
├── main.py             # Entry point
├── config.py           # Local config (do not commit!)
├── config.example.py   # Config template
└── requirements.txt
```

---

## Roadmap

- [ ] Graphical user interface (GUI)
- [ ] Direct Notion API integration
- [ ] Batch support (multiple files at once)
- [ ] Export to additional formats (docx, pdf)
- [ ] Automatic model selection based on available RAM
