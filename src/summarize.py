"""
summarize.py
------------
Generates a structured summary using a local LLM via Ollama.
"""

from src.chunking import split_blocks
from src.errors import OllamaError
from src.languages import name_for
from src.logger import get_logger
from src.ollama_client import DEFAULT_TIMEOUT, call_ollama, check_ollama
from src.prompts import DEFAULT_PRESET, fill, is_full_template, resolve_instructions

DEFAULT_CHUNK_CHARS = 6000


def resolve_output_language(transcript_language: str, summary_language: str) -> str:
    """Picks the language name the summary should be written in."""
    code = transcript_language if summary_language == "same" else summary_language
    return name_for(code) if code else "English"


def summarize_transcript(
    transcript: str,
    model: str,
    host: str,
    transcript_language: str,
    summary_language: str,
    num_ctx: int = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
    preset: str = DEFAULT_PRESET,
    custom_prompt: str = "",
) -> str:
    """
    Generates a structured summary of the transcript.

    Transcripts longer than `chunk_chars` are summarized chunk by chunk and the
    partial summaries are then merged, so a long meeting is not silently
    truncated to whatever fits in the model's context window.

    Args:
        transcript:          Transcript (cleaned or raw)
        model:               Ollama model to use
        host:                Ollama server URL
        transcript_language: Language of the transcript
        summary_language:    Language of the summary ('same', 'it', 'en', etc.)
        num_ctx:             Context window to request from Ollama
        chunk_chars:         Maximum transcript characters sent in one request
        timeout:             Per-request timeout in seconds
        preset:              Which summary shape to produce (see src/prompts.py)
        custom_prompt:       Instructions replacing the preset, when given

    Returns:
        Structured summary in Markdown (empty string if it could not be produced)
    """
    log = get_logger()
    log.info(f"[3/4] Generating summary with {model}...")

    if not transcript.strip():
        log.warning("Transcript is empty. Skipping summary.")
        return ""

    if not check_ollama(host, model):
        log.warning(f"Ollama not reachable or model '{model}' not found. Skipping summary.")
        return ""

    output_lang = resolve_output_language(transcript_language, summary_language)
    instructions = resolve_instructions(preset, custom_prompt)
    chunks = split_blocks(transcript, chunk_chars)

    try:
        if len(chunks) <= 1:
            summary = call_ollama(
                host, model, _final_prompt(transcript, output_lang, instructions),
                num_ctx=num_ctx, timeout=timeout,
            )
        else:
            log.info(f"      Transcript is long, summarizing in {len(chunks)} passes, then merging.")
            partials = []
            for index, chunk in enumerate(chunks, 1):
                log.info(f"      Summarizing part {index}/{len(chunks)}...")
                partials.append(call_ollama(
                    host, model,
                    _partial_prompt(chunk, output_lang, instructions, index, len(chunks)),
                    num_ctx=num_ctx, timeout=timeout,
                ))
            log.info("      Merging partial summaries...")
            summary = call_ollama(
                host, model, _merge_prompt(partials, output_lang, instructions),
                num_ctx=num_ctx, timeout=timeout,
            )
    except OllamaError as exc:
        log.error(f"Summary failed ({exc}). Continuing without a summary.")
        return ""

    log.info("      Summary generated.")
    return summary


def _final_prompt(transcript: str, output_lang: str, instructions: str) -> str:
    if is_full_template(instructions):
        return fill(instructions, transcript, output_lang)

    return f"""You are an expert analyst. Read this transcript and produce a structured summary.

Write the summary in {output_lang}.

Use Markdown formatting with ## headers for each section.

Produce exactly this:
{instructions}

Be concise but thorough. Do not invent anything that is not in the transcript.

TRANSCRIPT:
{transcript}"""


def _partial_prompt(chunk: str, output_lang: str, instructions: str, index: int, total: int) -> str:
    """First map-reduce pass.

    The extraction is driven by the same instructions as the final summary.
    A fixed 'decisions and action items' pass would throw away exactly the
    material a lecture or interview summary needs, before the merge ever sees it.
    """
    return f"""You are an expert analyst. This is part {index} of {total} of a longer transcript.

Write in {output_lang}.

Later on, the notes from every part will be combined into a summary shaped like this:

{instructions}

From THIS part only, extract as concise Markdown bullet points every detail that
summary will need. Keep names, figures, quotes and commitments verbatim where they matter.
Do not invent anything, and do not write a conclusion, because this is only one part.

TRANSCRIPT PART {index}/{total}:
{chunk}"""


def _merge_prompt(partials: list[str], output_lang: str, instructions: str) -> str:
    joined = "\n\n".join(f"--- NOTES FROM PART {i} ---\n{p}" for i, p in enumerate(partials, 1))
    return f"""You are an expert analyst. Below are notes taken from consecutive parts of one recording.

Merge them into a single summary of the whole recording, written in {output_lang}.
Remove duplicates, keep every distinct point, and do not invent anything.

Use Markdown formatting with ## headers for each section.

Produce exactly this:
{instructions}

NOTES:
{joined}"""
