"""
Shared test fixtures.

config.py is git-ignored (holds per-developer secrets), so a fresh clone of the
repo has no config.py at all. We inject a stub `config` module into sys.modules
before anything imports it, so tests don't depend on a local config.py existing.
"""

import sys
import types


def _install_stub_config():
    if "config" in sys.modules:
        return

    stub = types.ModuleType("config")
    stub.DEFAULT_WHISPER_MODEL = "medium"
    stub.DEFAULT_LANGUAGE = "auto"
    stub.DEFAULT_SPEAKERS = "auto"
    stub.DEFAULT_COMPUTE_TYPE = "int8"
    stub.HF_TOKEN = "hf_test_token"
    stub.OLLAMA_HOST = "http://localhost:11434"
    stub.DEFAULT_CLEANUP_MODEL = "llama3.1"
    stub.DEFAULT_SUMMARY_MODEL = "llama3.1"
    stub.DEFAULT_SUMMARY_LANGUAGE = "same"
    stub.DEFAULT_MODE = "full"
    stub.DEFAULT_OUTPUT_DIR = "outputs"
    stub.DEFAULT_FORMAT = "md"
    stub.DEFAULT_EXPORT_CONTENT = "full"
    sys.modules["config"] = stub


_install_stub_config()
