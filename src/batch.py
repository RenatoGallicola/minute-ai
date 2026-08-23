"""
batch.py
--------
Batch processing logic for multiple audio files.
Handles sequential and parallel execution, and skip-if-already-processed logic.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.export import EXTENSIONS, formats_for
from src.logger import get_logger
from src.naming import meeting_name_from_path, output_stem_pattern

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".wma", ".aac", ".mp4", ".mkv", ".webm", ".opus"}


def collect_audio_files(inputs: list[str], recursive: bool = False) -> list[str]:
    """
    Collects all audio files from a list of paths (files or folders).

    Args:
        inputs:    List of file paths or folder paths
        recursive: Also descend into sub-folders

    Returns:
        Deduplicated list of audio file paths
    """
    log = get_logger()
    collected = []

    for input_path in inputs:
        path = Path(input_path)

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                collected.append(str(path))
            else:
                log.warning(f"Skipping unsupported file format: {input_path}")

        elif path.is_dir():
            # iterdir/rglob rather than glob(): a folder whose name contains
            # '[' or '*' would otherwise silently match nothing.
            entries = path.rglob("*") if recursive else path.iterdir()
            found = sorted(
                str(f) for f in entries
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not found:
                log.warning(f"No audio files found in folder: {input_path}")
            collected.extend(found)

        else:
            log.warning(f"Path not found: {input_path}")

    # Deduplicate while preserving order
    seen = set()
    result = []
    for f in collected:
        resolved = str(Path(f).resolve()).lower()
        if resolved not in seen:
            seen.add(resolved)
            result.append(f)

    return result


def already_processed(audio_path: str, output_dir: str, fmt: str) -> bool:
    """
    Checks if an audio file has already been processed by looking for
    an existing output file with a matching name.

    The match is anchored on the exact '<timestamp>_<slug>' shape export.py
    produces, so 'test.wav' is no longer considered done just because an
    unrelated 'Test_Two' export happens to sit in the same folder.

    Args:
        audio_path: Path to the audio file
        output_dir: Output directory to check
        fmt:        Expected format ('md', 'txt', 'docx', 'pdf', 'srt', 'all')

    Returns:
        True if output already exists
    """
    directory = Path(output_dir)
    if not directory.exists():
        return False

    extensions = {EXTENSIONS[name] for name in formats_for(fmt) if name in EXTENSIONS}
    if not extensions:
        return False

    pattern = output_stem_pattern(meeting_name_from_path(audio_path))
    return any(
        f.suffix.lower() in extensions and pattern.match(f.stem)
        for f in directory.iterdir()
        if f.is_file()
    )


def run_batch(
    audio_files: list[str],
    process_fn,
    parallel: bool,
    force: bool,
    output_dir: str,
    fmt: str,
    max_workers: int = 2,
) -> dict:
    """
    Runs the processing pipeline on multiple audio files.

    Args:
        audio_files: List of audio file paths
        process_fn:  Function to call for each file: process_fn(audio_path)
        parallel:    Whether to run in parallel
        force:       Whether to skip already-processed check
        output_dir:  Output directory
        fmt:         Output format
        max_workers: Max parallel workers (default: 2)

    Returns:
        Dict with 'success', 'skipped', 'failed' lists ('failed' holds
        (path, message) pairs)
    """
    log = get_logger()
    results = {"success": [], "skipped": [], "failed": []}

    # Filter out already-processed files unless --force
    to_process = []
    for audio in audio_files:
        if not force and already_processed(audio, output_dir, fmt):
            log.info(f"[SKIP] Already processed: {Path(audio).name} (use --force to reprocess)")
            results["skipped"].append(audio)
        else:
            to_process.append(audio)

    if not to_process:
        log.info("All files already processed. Use --force to reprocess.")
        return results

    total = len(to_process)
    workers = max(1, min(max_workers, total))
    log.info(f"Processing {total} file(s) {f'in parallel ({workers} workers)' if parallel else 'sequentially'}...")

    if parallel and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_fn, audio): audio for audio in to_process}
            for i, future in enumerate(as_completed(futures), 1):
                audio = futures[future]
                name = Path(audio).name
                try:
                    future.result()
                    log.info(f"[{i}/{total}] Done: {name}")
                    results["success"].append(audio)
                except Exception as e:
                    log.error(f"[{i}/{total}] Failed: {name} — {e}")
                    results["failed"].append((audio, str(e)))
    else:
        for i, audio in enumerate(to_process, 1):
            name = Path(audio).name
            log.info(f"File {i}/{total}: {name}")
            try:
                process_fn(audio)
                results["success"].append(audio)
            except Exception as e:
                log.error(f"Failed to process {name}: {e}")
                results["failed"].append((audio, str(e)))

    return results


def print_batch_summary(results: dict):
    """Prints and logs a summary of batch processing results."""
    log = get_logger()
    total = len(results["success"]) + len(results["skipped"]) + len(results["failed"])

    log.info("=" * 55)
    log.info(f"BATCH SUMMARY ({total} file(s))")
    log.info(f"  Processed : {len(results['success'])}")
    log.info(f"  Skipped   : {len(results['skipped'])}")
    log.info(f"  Failed    : {len(results['failed'])}")

    if results["failed"]:
        log.error("Failed files:")
        for entry in results["failed"]:
            path, message = entry if isinstance(entry, tuple) else (entry, "")
            log.error(f"  - {Path(path).name}{f' — {message}' if message else ''}")

    log.info("=" * 55)
