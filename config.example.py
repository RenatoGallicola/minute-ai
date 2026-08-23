# ─────────────────────────────────────────────
# minute-ai — Configuration (EXAMPLE)
# ─────────────────────────────────────────────
# Copy this file to config.py and insert your tokens.
# NEVER commit config.py to Git if it contains real tokens.

# Whisper
DEFAULT_WHISPER_MODEL = "auto"        # auto | tiny | base | small | medium | large-v3
                                      # 'auto' picks a model based on GPU VRAM, or free RAM on CPU
DEFAULT_LANGUAGE = "auto"             # auto | it | en | fr | de | etc.
DEFAULT_SPEAKERS = "auto"             # auto | integer (e.g. 2, 3)
DEFAULT_COMPUTE_TYPE = "auto"         # auto (recommended) | int8 (CPU) | float16 (Nvidia GPU)

# HuggingFace (for diarization)
# Get a token at https://huggingface.co/settings/tokens and accept the terms of
# pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0.
HF_TOKEN = "hf_INSERT_HERE"

# Ollama
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_CLEANUP_MODEL = "llama3.1"
DEFAULT_SUMMARY_MODEL = "llama3.1"
DEFAULT_SUMMARY_LANGUAGE = "same"     # same | it | en | fr | etc.
DEFAULT_SUMMARY_PRESET = "meeting"    # meeting | lecture | interview | one-on-one | custom
                                      # 'custom' needs --summary-prompt (CLI) or the box in the GUI
OLLAMA_NUM_CTX = 8192                 # Context window requested from Ollama. Ollama's own
                                      # default is small and silently truncates long prompts.
OLLAMA_TIMEOUT = 600                  # Seconds to wait for one generation
LLM_CHUNK_CHARS = 6000                # Transcript characters sent per request; longer
                                      # transcripts are cleaned/summarized in several passes

# Pipeline
DEFAULT_MODE = "full"                 # full | transcript | clean | summary

# Export
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_FORMAT = "md"                 # md | txt | docx | pdf | srt | all
DEFAULT_EXPORT_CONTENT = "full"       # full | summary
