"""
naming.py
---------
Filename rules shared by the exporter and the batch runner.

Both modules need to agree on how a meeting name becomes a file name:
export.py writes "<timestamp>_<slug>.<ext>", and batch.py decides whether a
file was already processed by looking for exactly that shape.
"""

import re
from datetime import datetime
from pathlib import Path

# Characters Windows forbids in a file name. On NTFS a ':' is especially nasty:
# open() silently writes to an alternate data stream instead of failing, so the
# exported file never shows up in Explorer.
_ILLEGAL = r'<>:"/\|?*'
_ILLEGAL_RE = re.compile(f"[{re.escape(_ILLEGAL)}\x00-\x1f]")
_SEPARATOR_RE = re.compile(r"[\s]+")
_COLLAPSE_RE = re.compile(r"_{2,}")

# Reserved DOS device names — a file called "con.md" cannot be created on Windows.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

MAX_SLUG_LENGTH = 80
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M"
_TIMESTAMP_RE = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}"


def slugify(name: str) -> str:
    """Turns a meeting name into a portable file name component.

    Whitespace becomes '_', characters Windows forbids are dropped, and the
    result is trimmed to a sane length. Always returns a non-empty string.
    """
    text = _SEPARATOR_RE.sub("_", (name or "").strip())
    text = _ILLEGAL_RE.sub("", text)
    text = _COLLAPSE_RE.sub("_", text).strip("._ ")
    text = text[:MAX_SLUG_LENGTH].strip("._ ")

    if not text or text.lower() in _RESERVED:
        text = f"meeting_{text}".strip("_")
    return text


def meeting_name_from_path(audio_path: str) -> str:
    """Derives a human-readable meeting name from an audio file name."""
    return Path(audio_path).stem.replace("_", " ").replace("-", " ").strip().title() or "Meeting"


def output_stem(meeting_name: str, when: datetime = None) -> str:
    """Builds the '<timestamp>_<slug>' stem every exported file shares."""
    timestamp = (when or datetime.now()).strftime(TIMESTAMP_FORMAT)
    return f"{timestamp}_{slugify(meeting_name)}"


def output_stem_pattern(meeting_name: str) -> re.Pattern:
    """Matches file stems previously produced by output_stem() for this meeting.

    Anchored on both ends so 'Test' no longer matches an earlier 'Test_Two'
    export, and tolerant of the ' (2)' suffix unique_path() may have added.
    """
    slug = re.escape(slugify(meeting_name))
    return re.compile(rf"^{_TIMESTAMP_RE}_{slug}(?: \(\d+\))?$", re.IGNORECASE)


def unique_path(path: str) -> str:
    """Returns `path`, or 'name (2).ext' etc. if something is already there.

    Two source files can collapse onto the same output name (meeting.mp3 and
    meeting.wav in the same folder, exported within the same minute); without
    this the second run would silently overwrite the first.
    """
    candidate = Path(path)
    if not candidate.exists():
        return str(candidate)

    for counter in range(2, 1000):
        alternative = candidate.with_name(f"{candidate.stem} ({counter}){candidate.suffix}")
        if not alternative.exists():
            return str(alternative)
    return str(candidate)
