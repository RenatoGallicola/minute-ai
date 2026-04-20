# ─────────────────────────────────────────────
# minute-ai — Configuration (EXAMPLE)
# ─────────────────────────────────────────────
# Copy this file to config.py and insert your tokens.
# NEVER commit config.py to Git if it contains real tokens.

# Whisper
DEFAULT_WHISPER_MODEL = "medium"       # tiny | base | small | medium | large-v3
DEFAULT_LANGUAGE = "auto"              # auto | it | en | fr | de | etc.
DEFAULT_SPEAKERS = "auto"             # auto | integer (e.g. 2, 3)
DEFAULT_COMPUTE_TYPE = "int8"         # int8 (CPU) | float16 (Nvidia GPU)

# HuggingFace (for diarization)
HF_TOKEN = "hf_INSERT_HERE"

# Ollama
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_CLEANUP_MODEL = "llama3.1"
DEFAULT_SUMMARY_MODEL = "llama3.1"
DEFAULT_SUMMARY_LANGUAGE = "same"     # same | it | en | fr | etc.

# Export
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_FORMAT = "md"                 # md | txt | all

# Cleanup and summary
DEFAULT_CLEANUP = True
DEFAULT_SUMMARY = True
