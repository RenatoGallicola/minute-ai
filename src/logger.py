"""
logger.py
---------
Centralized logging setup for minute-ai.
Logs to both terminal (stdout) and a persistent log file.
"""

import contextlib
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = "logs"
LOG_FILE = "minute-ai.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def _utf8_stream(stream):
    """Makes stdout tolerate the arrows and em dashes used in log messages.

    When stdout is redirected to a file or a pipe on Windows it falls back to
    the ANSI code page, and logging a '→' raises UnicodeEncodeError.
    """
    with contextlib.suppress(AttributeError, ValueError, OSError):
        stream.reconfigure(encoding="utf-8", errors="replace")
    return stream


def setup_logger(log_dir: str = LOG_DIR) -> logging.Logger:
    """
    Sets up the application logger.
    Outputs to both terminal and a persistent, size-capped log file.

    Args:
        log_dir: Directory where the log file will be stored

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("minute-ai")
    logger.setLevel(logging.DEBUG)
    # Keep our records out of the root logger: uvicorn installs its own root
    # handlers, which would otherwise print every pipeline line twice.
    logger.propagate = False

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, LOG_FILE)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler: full detail, rotates so the log cannot grow without bound
    file_handler = RotatingFileHandler(
        log_path, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Terminal handler: same detail
    stream_handler = logging.StreamHandler(_utf8_stream(sys.stdout))
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def get_logger() -> logging.Logger:
    """Returns the existing logger instance (must call setup_logger first)."""
    return logging.getLogger("minute-ai")
