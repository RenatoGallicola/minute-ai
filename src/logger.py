"""
logger.py
---------
Centralized logging setup for minute-ai.
Logs to both terminal (stdout) and a persistent log file.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


LOG_DIR = "logs"
LOG_FILE = "minute-ai.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(log_dir: str = LOG_DIR) -> logging.Logger:
    """
    Sets up the application logger.
    Outputs to both terminal and a persistent log file.

    Args:
        log_dir: Directory where the log file will be stored

    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, LOG_FILE)

    logger = logging.getLogger("minute-ai")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler — full detail, appends to existing log
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Terminal handler — same detail
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def get_logger() -> logging.Logger:
    """Returns the existing logger instance (must call setup_logger first)."""
    return logging.getLogger("minute-ai")
