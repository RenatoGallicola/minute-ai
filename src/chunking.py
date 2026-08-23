"""
chunking.py
-----------
Splits a long transcript into LLM-sized pieces.

A one-hour meeting easily runs past any local model's context window. Sending
it in one shot does not error. Ollama simply truncates the prompt, so the
summary silently describes only part of the meeting. These helpers cut the
transcript on speaker-block boundaries instead.
"""


def split_blocks(text: str, max_chars: int) -> list[str]:
    """Splits `text` into chunks of at most `max_chars`, preferring blank-line breaks.

    Speaker blocks are kept whole whenever they fit; a single oversized block
    is split on sentence ends, and failing that on a hard character boundary.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for block in text.split("\n\n"):
        if not block.strip():
            continue

        for piece in _split_oversized(block, max_chars):
            piece_len = len(piece) + 2
            if current and current_len + piece_len > max_chars:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += piece_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_oversized(block: str, max_chars: int) -> list[str]:
    """Breaks a single block that is longer than max_chars into sentence-sized pieces."""
    if len(block) <= max_chars:
        return [block]

    pieces: list[str] = []
    remaining = block
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if cut < max_chars // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        else:
            cut += 1
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining.strip():
        pieces.append(remaining.strip())
    return pieces
