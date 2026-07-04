"""
hardware.py
-----------
Detects available system resources to pick sensible defaults automatically.
"""

from src.logger import get_logger


# (min_available_gb, model) — checked from largest requirement to smallest.
# Rough RAM headroom needed to comfortably load + run each whisper model on CPU,
# leaving room for the OS, Ollama, and other apps running alongside it.
MODEL_RAM_THRESHOLDS = [
    (16, "large-v3"),
    (8, "medium"),
    (4, "small"),
    (2, "base"),
    (0, "tiny"),
]


def select_whisper_model(available_ram_gb: float) -> str:
    """Picks the largest whisper model that comfortably fits in the given available RAM (GB)."""
    for min_gb, model in MODEL_RAM_THRESHOLDS:
        if available_ram_gb >= min_gb:
            return model
    return "tiny"


def auto_select_model(parallel_workers: int = 1) -> str:
    """
    Detects available system RAM and returns the best-fitting whisper model.

    In parallel mode each worker loads its own model instance, so the available
    RAM is divided across workers before picking a model tier.

    Args:
        parallel_workers: Number of files processed concurrently (1 if sequential)

    Returns:
        Whisper model name: 'tiny', 'base', 'small', 'medium', or 'large-v3'
    """
    log = get_logger()
    try:
        import psutil
    except ImportError:
        log.warning("psutil is not installed; cannot auto-detect RAM. Falling back to 'small'.")
        return "small"

    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    effective_gb = available_gb / max(parallel_workers, 1)

    model = select_whisper_model(effective_gb)

    workers_note = f" / {parallel_workers} parallel workers" if parallel_workers > 1 else ""
    log.info(f"Auto-selected Whisper model '{model}' (available RAM: {available_gb:.1f} GB{workers_note})")
    return model
