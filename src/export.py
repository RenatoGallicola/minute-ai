"""
export.py
---------
Exports transcript and summary to md, txt, docx, and pdf formats.
"""

import os
import re
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
    export_content: str,
) -> list[str]:
    """
    Exports results to the specified formats.

    Args:
        audio_path:     Path to the original audio file
        meeting_name:   Human-readable meeting name
        transcript:     Full transcript
        summary:        Structured summary (may be empty)
        language:       Detected language
        output_dir:     Output folder
        fmt:            Format: 'md', 'txt', 'docx', 'pdf', 'all'
        export_content: Content to include: 'full' or 'summary'

    Returns:
        List of paths of created files
    """
    log = get_logger()
    os.makedirs(output_dir, exist_ok=True)

    audio_stem = Path(audio_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_name = meeting_name.replace(" ", "_").replace("/", "-")
    base_filename = f"{timestamp}_{safe_name}"

    # Determine what content to include
    include_transcript = export_content == "full"
    include_summary = bool(summary)

    created_files = []

    if fmt in ("md", "all"):
        path = os.path.join(output_dir, f"{base_filename}.md")
        _write_markdown(path, meeting_name, audio_stem, language, summary, transcript, include_transcript, include_summary)
        created_files.append(path)
        log.info(f"      Exported: {path}")

    if fmt in ("txt", "all"):
        path = os.path.join(output_dir, f"{base_filename}.txt")
        _write_txt(path, meeting_name, summary, transcript, include_transcript, include_summary)
        created_files.append(path)
        log.info(f"      Exported: {path}")

    if fmt in ("docx", "all"):
        path = os.path.join(output_dir, f"{base_filename}.docx")
        _write_docx(path, meeting_name, audio_stem, language, summary, transcript, include_transcript, include_summary)
        created_files.append(path)
        log.info(f"      Exported: {path}")

    if fmt in ("pdf", "all"):
        path = os.path.join(output_dir, f"{base_filename}.pdf")
        _write_pdf(path, meeting_name, audio_stem, language, summary, transcript, include_transcript, include_summary)
        created_files.append(path)
        log.info(f"      Exported: {path}")

    return created_files


# ─────────────────────────────────────────────
# Markdown
# ─────────────────────────────────────────────

def _write_markdown(path, meeting_name, audio_stem, language, summary, transcript, include_transcript, include_summary):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = f"# {meeting_name}\n\n"
    content += f"> **Date:** {date_str} | **Language:** {language} | **Source:** {audio_stem}\n\n---\n\n"

    if include_summary and summary:
        content += summary + "\n\n---\n\n"

    if include_transcript:
        content += "## Full Transcript\n\n" + transcript + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ─────────────────────────────────────────────
# Plain text
# ─────────────────────────────────────────────

def _write_txt(path, meeting_name, summary, transcript, include_transcript, include_summary):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"MEETING: {meeting_name}", f"DATE: {date_str}", "=" * 60, ""]

    if include_summary and summary:
        clean = summary.replace("## ", "").replace("**", "")
        lines += ["SUMMARY", "-" * 40, clean, "", "=" * 60, ""]

    if include_transcript:
        lines += ["TRANSCRIPT", "-" * 40, transcript]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────
# DOCX
# ─────────────────────────────────────────────

def _write_docx(path, meeting_name, audio_stem, language, summary, transcript, include_transcript, include_summary):
    log = get_logger()
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        log.error("python-docx is not installed. Run: pip install python-docx")
        return

    doc = Document()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Title
    title = doc.add_heading(meeting_name, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Metadata
    meta = doc.add_paragraph()
    meta.add_run(f"Date: {date_str}  |  Language: {language}  |  Source: {audio_stem}").italic = True

    doc.add_paragraph()  # spacer

    # Summary
    if include_summary and summary:
        _docx_add_markdown(doc, summary)
        doc.add_page_break()

    # Transcript
    if include_transcript:
        doc.add_heading("Full Transcript", level=1)
        for line in transcript.split("\n\n"):
            if line.strip():
                p = doc.add_paragraph()
                parts = line.split(":", 1)
                if len(parts) == 2:
                    run_name = p.add_run(parts[0] + ": ")
                    run_name.bold = True
                    p.add_run(parts[1].strip())
                else:
                    p.add_run(line.strip())

    doc.save(path)


def _docx_add_markdown(doc, text):
    """Converts basic Markdown to python-docx elements."""
    from docx.shared import Pt

    for line in text.split("\n"):
        line = line.rstrip()

        if line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            _docx_add_inline(p, line[2:])
        elif line == "---":
            doc.add_paragraph("─" * 40)
        elif line.strip() == "":
            doc.add_paragraph()
        else:
            p = doc.add_paragraph()
            _docx_add_inline(p, line)


def _docx_add_inline(paragraph, text):
    """Handles **bold** inline markdown in a paragraph."""
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        else:
            paragraph.add_run(part)


# ─────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────

def _write_pdf(path, meeting_name, audio_stem, language, summary, transcript, include_transcript, include_summary):
    log = get_logger()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
    except ImportError:
        log.error("reportlab is not installed. Run: pip install reportlab")
        return

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=6)
    style_meta = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=12)
    style_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceBefore=12, spaceAfter=6)
    style_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    style_body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14)
    style_speaker = ParagraphStyle("Speaker", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14)

    story = []

    # Title
    story.append(Paragraph(meeting_name, style_title))
    story.append(Paragraph(f"Date: {date_str} &nbsp;&nbsp;|&nbsp;&nbsp; Language: {language} &nbsp;&nbsp;|&nbsp;&nbsp; Source: {audio_stem}", style_meta))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 12))

    # Summary
    if include_summary and summary:
        for line in summary.split("\n"):
            line = line.rstrip()
            if line.startswith("## "):
                story.append(Paragraph(line[3:], style_h2))
            elif line.startswith("# "):
                story.append(Paragraph(line[2:], style_h1))
            elif line.startswith("- "):
                content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line[2:])
                story.append(Paragraph(f"• {content}", style_body))
            elif line == "---":
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            elif line.strip():
                content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
                story.append(Paragraph(content, style_body))
            else:
                story.append(Spacer(1, 6))

        if include_transcript:
            story.append(PageBreak())

    # Transcript
    if include_transcript:
        story.append(Paragraph("Full Transcript", style_h1))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 8))

        for block in transcript.split("\n\n"):
            if not block.strip():
                continue
            parts = block.split(":", 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                text = parts[1].strip()
                content = f"<b>{speaker}:</b> {text}"
            else:
                content = block.strip()
            story.append(Paragraph(content, style_speaker))

    doc.build(story)
