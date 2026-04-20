"""
export.py
---------
Exports the transcript and summary to the requested formats.
"""

import os
from datetime import datetime
from pathlib import Path
from src.logger import get_logger


def export(
    audio_path: str,
    meeting_name: str,
    transcript: str,
    summary: str,
    language: str,
    output_dir: str,
    fmt: str,
) -> list[str]:
    """
    Exports results to the specified formats.

    Args:
        audio_path:   Path to the original audio file
        meeting_name: Human-readable meeting name
        transcript:   Full transcript
        summary:      Structured summary
        language:     Detected language
        output_dir:   Output folder
        fmt:          Format: 'md', 'txt', or 'all'

    Returns:
        List of paths of created files
    """
    log = get_logger()
    os.makedirs(output_dir, exist_ok=True)

    audio_stem = Path(audio_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_name = meeting_name.replace(" ", "_").replace("/", "-")
    base_filename = f"{timestamp}_{safe_name}"

    created_files = []

    if fmt in ("md", "all"):
        md_path = os.path.join(output_dir, f"{base_filename}.md")
        _write_markdown(md_path, meeting_name, audio_stem, language, summary, transcript)
        created_files.append(md_path)
        log.info(f"      Exported: {md_path}")

    if fmt in ("txt", "all"):
        txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        _write_txt(txt_path, meeting_name, summary, transcript)
        created_files.append(txt_path)
        log.info(f"      Exported: {txt_path}")

    return created_files


def _write_markdown(path, meeting_name, audio_stem, language, summary, transcript):
    """Writes a Markdown file formatted for Notion."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    content = f"""# {meeting_name}

> **Date:** {date_str} | **Language:** {language} | **Source:** {audio_stem}

---

"""
    if summary:
        content += summary
        content += "\n\n---\n\n"

    content += "## Full Transcript\n\n"
    content += transcript
    content += "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_txt(path, meeting_name, summary, transcript):
    """Writes a plain text file."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"MEETING: {meeting_name}",
        f"DATE: {date_str}",
        "=" * 60,
        "",
    ]

    if summary:
        clean_summary = summary.replace("## ", "").replace("**", "")
        lines += ["SUMMARY", "-" * 40, clean_summary, "", "=" * 60, ""]

    lines += ["TRANSCRIPT", "-" * 40, transcript]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
