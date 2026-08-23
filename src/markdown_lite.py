"""
markdown_lite.py
----------------
Renders the small Markdown subset the summarizer produces (headings, bullets,
bold/italic, rules) as HTML for the web GUI's preview pane.

Deliberately tiny and escape-first: the text comes from a local LLM, and it is
injected into the page with |safe, so everything is escaped before any markup
is re-introduced.
"""

import re
from html import escape

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*([^*\n]+?)\*(?!\*)")
_CODE = re.compile(r"`([^`\n]+?)`")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


def _inline(text: str) -> str:
    out = escape(text)
    out = _CODE.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)
    return out


def render(text: str) -> str:
    """Converts a Markdown summary to a safe HTML fragment."""
    if not text or not text.strip():
        return ""

    html: list[str] = []
    list_tag = None

    def close_list():
        nonlocal list_tag
        if list_tag:
            html.append(f"</{list_tag}>")
            list_tag = None

    def open_list(tag):
        nonlocal list_tag
        if list_tag != tag:
            close_list()
            html.append(f"<{tag}>")
            list_tag = tag

    for raw in text.splitlines():
        line = raw.rstrip()

        if not line.strip():
            close_list()
            continue

        if set(line.strip()) <= {"-", "_", "*"} and len(line.strip()) >= 3:
            close_list()
            html.append("<hr>")
            continue

        heading = _HEADING.match(line)
        if heading:
            close_list()
            # The page itself owns <h1>, and the summarizer writes its sections
            # as '##', so '#' and '##' both land on <h2>.
            level = min(max(len(heading.group(1)), 2), 5)
            html.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue

        bullet = _BULLET.match(line)
        if bullet:
            open_list("ul")
            html.append(f"<li>{_inline(bullet.group(1))}</li>")
            continue

        numbered = _NUMBERED.match(line)
        if numbered:
            open_list("ol")
            html.append(f"<li>{_inline(numbered.group(1))}</li>")
            continue

        close_list()
        html.append(f"<p>{_inline(line.strip())}</p>")

    close_list()
    return "\n".join(html)


def render_transcript(text: str) -> str:
    """Renders a speaker-labelled transcript as HTML blocks."""
    if not text or not text.strip():
        return ""

    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        head, sep, tail = block.partition(":")
        if sep and "\n" not in head and 0 < len(head.strip()) <= 60:
            blocks.append(
                f'<p class="line"><span class="who">{escape(head.strip())}</span>'
                f'<span class="said">{escape(tail.strip())}</span></p>'
            )
        else:
            blocks.append(f'<p class="line"><span class="said">{escape(block)}</span></p>')
    return "\n".join(blocks)
