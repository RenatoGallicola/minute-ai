"""
hardware.py
-----------
Detects available system resources to pick sensible defaults automatically.
"""

import os
from pathlib import Path

from src.logger import get_logger


# Smallest to largest. Used to compare tiers when a model has to be downgraded.
MODEL_ORDER = ["tiny", "base", "small", "medium", "large-v3"]

# whisperX pulls its models from these HuggingFace repos.
WHISPER_REPO = "Systran/faster-whisper-{model}"

# Rough download sizes, so the UI can warn before a model is fetched.
MODEL_DOWNLOAD_MB = {
    "tiny": 75,
    "base": 145,
    "small": 484,
    "medium": 1530,
    "large-v3": 3090,
}


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

# On a GPU the model lives in VRAM instead, and float16 halves the footprint.
MODEL_VRAM_THRESHOLDS = [
    (7, "large-v3"),
    (4, "medium"),
    (2, "small"),
    (1, "base"),
    (0, "tiny"),
]


def _select(thresholds, available_gb: float) -> str:
    for min_gb, model in thresholds:
        if available_gb >= min_gb:
            return model
    return "tiny"


def select_whisper_model(available_ram_gb: float) -> str:
    """Picks the largest whisper model that comfortably fits in the given available RAM (GB)."""
    return _select(MODEL_RAM_THRESHOLDS, available_ram_gb)


def select_whisper_model_for_vram(available_vram_gb: float) -> str:
    """Picks the largest whisper model that fits in the given GPU memory (GB)."""
    return _select(MODEL_VRAM_THRESHOLDS, available_vram_gb)


def detect_gpu() -> tuple[str, float]:
    """Returns (gpu_name, total_vram_gb), or ("", 0.0) when no CUDA device is usable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "", 0.0
        properties = torch.cuda.get_device_properties(0)
        return properties.name, properties.total_memory / (1024 ** 3)
    except Exception:
        return "", 0.0


def available_ram_gb() -> float:
    """Returns free system RAM in GB, or 0.0 if psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return 0.0
    return psutil.virtual_memory().available / (1024 ** 3)


def _hub_cache_dir() -> Path:
    """Where huggingface_hub keeps downloaded models, honouring the usual env vars."""
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_whisper_model_installed(model: str, cache_dir: Path = None) -> bool:
    """True when a whisper model is already in the HuggingFace cache.

    Checks for the actual weights, not just the folder: an interrupted download
    leaves the directory behind with nothing usable in it.
    """
    repo = WHISPER_REPO.format(model=model).replace("/", "--")
    root = (cache_dir or _hub_cache_dir()) / f"models--{repo}" / "snapshots"
    if not root.is_dir():
        return False
    return any(snapshot.joinpath("model.bin").exists() for snapshot in root.iterdir() if snapshot.is_dir())


def installed_whisper_models(cache_dir: Path = None) -> list[str]:
    """The whisper models already downloaded, smallest first."""
    return [m for m in MODEL_ORDER if is_whisper_model_installed(m, cache_dir)]


def auto_select_model(parallel_workers: int = 1, installed: list[str] = None) -> str:
    """
    Detects available system resources and returns the best-fitting whisper model.

    A CUDA GPU is preferred when present, since the model is loaded into VRAM.
    Otherwise the decision falls back to free system RAM. In parallel mode each
    worker loads its own model instance, so the budget is divided across workers
    before picking a tier.

    Among the models that fit, one already on disk always wins: 'auto' is meant
    to just work, and silently starting a multi-gigabyte download (which fails
    outright on a restricted network) is the opposite of that.

    Args:
        parallel_workers: Number of files processed concurrently (1 if sequential)
        installed:        Models already downloaded; detected when not given

    Returns:
        Whisper model name: 'tiny', 'base', 'small', 'medium', or 'large-v3'
    """
    log = get_logger()
    workers = max(parallel_workers, 1)
    workers_note = f" / {workers} parallel workers" if workers > 1 else ""

    gpu_name, vram_gb = detect_gpu()
    if vram_gb:
        fits = select_whisper_model_for_vram(vram_gb / workers)
        budget = f"{gpu_name}, {vram_gb:.1f} GB VRAM{workers_note}"
    else:
        ram_gb = available_ram_gb()
        if not ram_gb:
            log.warning("psutil is not installed; cannot auto-detect RAM. Falling back to 'small'.")
            fits = "small"
            budget = "no RAM detection"
        else:
            fits = select_whisper_model(ram_gb / workers)
            budget = f"available RAM: {ram_gb:.1f} GB{workers_note}"

    on_disk = installed if installed is not None else installed_whisper_models()
    model = _prefer_installed(fits, on_disk)

    if model != fits:
        log.info(
            f"Auto-selected Whisper model '{model}' ({budget}); "
            f"'{fits}' would also fit but is not downloaded yet."
        )
    else:
        note = "" if not on_disk or model in on_disk else " — it will be downloaded on first use"
        log.info(f"Auto-selected Whisper model '{model}' ({budget}){note}")
    return model


def _prefer_installed(fits: str, installed: list[str]) -> str:
    """Largest already-downloaded model no bigger than `fits`, else `fits` itself."""
    if not installed or fits in installed:
        return fits
    ceiling = MODEL_ORDER.index(fits)
    candidates = [m for m in installed if m in MODEL_ORDER and MODEL_ORDER.index(m) <= ceiling]
    return max(candidates, key=MODEL_ORDER.index) if candidates else fits


def system_info() -> dict:
    """A snapshot of the host, for the GUI status panel."""
    gpu_name, vram_gb = detect_gpu()
    return {
        "ram_gb": round(available_ram_gb(), 1),
        "gpu": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "device": "GPU" if gpu_name else "CPU",
        "whisper_installed": installed_whisper_models(),
    }
