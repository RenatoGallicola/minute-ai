"""
errors.py
---------
Exception types raised by the pipeline.

Library modules under src/ must never call sys.exit(): the CLI, the batch
runner and the web GUI all need to catch failures and keep going (or report
them cleanly). They raise MinuteAIError instead, and main.py turns it into a
non-zero exit code at the top level.
"""


class MinuteAIError(Exception):
    """Base class for every expected, user-facing pipeline failure."""


class DependencyMissingError(MinuteAIError):
    """A required third-party package is not installed."""


class OllamaError(MinuteAIError):
    """The Ollama server could not be reached or returned an error."""


class TranscriptionError(MinuteAIError):
    """whisperX failed to transcribe or diarize the audio."""


class ExportError(MinuteAIError):
    """A file could not be written."""
