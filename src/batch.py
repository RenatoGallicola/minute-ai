"""
batch.py
--------
Batch processing logic for multiple audio files.
Handles sequential and parallel execution, and skip-if-already-processed logic.
"""

import os
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".wma", ".aac"}


def collect_audio_files(inputs: list[str]) -> list[str]:
    """
    Collects all audio files from a list of paths (files or folders).

    Args:
        inputs: List of file paths or folder paths

    Returns:
        Deduplicated list of audio file paths
    """
    collected = []

    for input_path in inputs:
        path = Path(input_path)

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                collected.append(str(path))
            else:
                print(f"[WARNING] Skipping unsupported file format: {input_path}")

        elif path.is_dir():
            found = []
            for ext in SUPPORTED_EXTENSIONS:
                found.extend(glob.glob(str(path / f"*{ext}")))
                found.extend(glob.glob(str(path / f"*{ext.upper()}")))
            if not found:
                print(f"[WARNING] No audio files found in folder: {input_path}")
            collected.extend(found)

        else:
            print(f"[WARNING] Path not found: {input_path}")

    # Deduplicate while preserving order
    seen = set()
    result = []
    for f in collected:
        resolved = str(Path(f).resolve())
        if resolved not in seen:
            seen.add(resolved)
            result.append(f)

    return result


def already_processed(audio_path: str, output_dir: str, fmt: str) -> bool:
    """
    Checks if an audio file has already been processed by looking for
    an existing output file with a matching name.

    Args:
        audio_path: Path to the audio file
        output_dir: Output directory to check
        fmt:        Expected format ('md', 'txt', 'all')

    Returns:
        True if output already exists
    """
    stem = Path(audio_path).stem.replace(" ", "_").replace("-", "_").lower()
    extensions = []

    if fmt in ("md", "all"):
        extensions.append(".md")
    if fmt in ("txt", "all"):
        extensions.append(".txt")

    for f in Path(output_dir).glob("*"):
        file_stem = f.stem.lower()
        # Output files are named: YYYY-MM-DD_HH-MM_MeetingName
        # Check if the stem of the audio file appears in the output filename
        if stem in file_stem and f.suffix in extensions:
            return True

    return False


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
        process_fn:  Function to call for each file: process_fn(audio_path) -> list[str]
        parallel:    Whether to run in parallel
        force:       Whether to skip already-processed check
        output_dir:  Output directory
        fmt:         Output format
        max_workers: Max parallel workers (default: 2)

    Returns:
        Dict with 'success', 'skipped', 'failed' lists
    """
    results = {"success": [], "skipped": [], "failed": []}

    # Filter out already-processed files unless --force
    to_process = []
    for audio in audio_files:
        if not force and already_processed(audio, output_dir, fmt):
            print(f"[SKIP] Already processed: {Path(audio).name} (use --force to reprocess)")
            results["skipped"].append(audio)
        else:
            to_process.append(audio)

    if not to_process:
        print("\nAll files already processed. Use --force to reprocess.")
        return results

    total = len(to_process)
    print(f"\nProcessing {total} file(s){'in parallel' if parallel else ' sequentially'}...\n")

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_fn, audio): audio for audio in to_process}
            for i, future in enumerate(as_completed(futures), 1):
                audio = futures[future]
                name = Path(audio).name
                try:
                    output_files = future.result()
                    print(f"[{i}/{total}] ✓ {name}")
                    results["success"].append(audio)
                except Exception as e:
                    print(f"[{i}/{total}] ✗ {name} — {e}")
                    results["failed"].append(audio)
    else:
        for i, audio in enumerate(to_process, 1):
            name = Path(audio).name
            print(f"\n{'=' * 55}")
            print(f"  File {i}/{total}: {name}")
            print(f"{'=' * 55}")
            try:
                process_fn(audio)
                results["success"].append(audio)
            except Exception as e:
                print(f"\n[ERROR] Failed to process {name}: {e}")
                results["failed"].append(audio)

    return results


def print_batch_summary(results: dict):
    """Prints a summary of batch processing results."""
    total = len(results["success"]) + len(results["skipped"]) + len(results["failed"])
    print(f"\n{'=' * 55}")
    print(f"  BATCH SUMMARY ({total} file(s))")
    print(f"{'=' * 55}")
    print(f"  ✓ Processed:  {len(results['success'])}")
    print(f"  → Skipped:    {len(results['skipped'])}")
    print(f"  ✗ Failed:     {len(results['failed'])}")

    if results["failed"]:
        print("\n  Failed files:")
        for f in results["failed"]:
            print(f"    - {Path(f).name}")

    print(f"{'=' * 55}")
