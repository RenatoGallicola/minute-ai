"""
summarize.py
------------
Generates a structured meeting summary using a local LLM via Ollama.
"""

from src.chunking import split_blocks
from src.languages import name_for
from src.errors import OllamaError
from src.logger import get_logger
from src.ollama_client import DEFAULT_TIMEOUT, call_ollama, check_ollama


DEFAULT_CHUNK_CHARS = 6000



SECTIONS = """## Participants
List all speakers. Use their real names if available, otherwise use the speaker labels. Include any inferred role or context.

## Topics Discussed
Main topics and themes covered during the meeting.

## Decisions Made
Concrete decisions that were agreed upon. If none, write "None".

## Action Items
Specific tasks to be done. For each item include:
- What needs to be done
- Who is responsible (if mentioned)
- Deadline (if mentioned)
If none, write "None".

## Open Points
Unresolved questions or topics to revisit in future meetings. If none, write "None".

## Additional Notes
Any other relevant information, context, or observations."""


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
    chunks = split_blocks(transcript, chunk_chars)

    try:
        if len(chunks) <= 1:
            summary = call_ollama(host, model, _final_prompt(transcript, output_lang), num_ctx=num_ctx, timeout=timeout)
        else:
            log.info(f"      Transcript is long — summarizing in {len(chunks)} passes, then merging.")
            partials = []
            for index, chunk in enumerate(chunks, 1):
                log.info(f"      Summarizing part {index}/{len(chunks)}...")
                partials.append(call_ollama(
                    host, model, _partial_prompt(chunk, output_lang, index, len(chunks)),
                    num_ctx=num_ctx, timeout=timeout,
                ))
            log.info("      Merging partial summaries...")
            summary = call_ollama(
                host, model, _merge_prompt(partials, output_lang), num_ctx=num_ctx, timeout=timeout
            )
    except OllamaError as exc:
        log.error(f"Summary failed ({exc}). Continuing without a summary.")
        return ""

    log.info("      Summary generated.")
    return summary


def _final_prompt(transcript: str, output_lang: str) -> str:
    return f"""You are an expert meeting analyst. Analyze this meeting transcript and produce a structured summary.

Write the summary in {output_lang}.

Use Markdown formatting with ## headers for each section.

Include these sections:
{SECTIONS}

Be concise but thorough. Focus on actionable content.

TRANSCRIPT:
{transcript}"""


def _partial_prompt(chunk: str, output_lang: str, index: int, total: int) -> str:
    return f"""You are an expert meeting analyst. This is part {index} of {total} of a longer meeting transcript.

Write in {output_lang}.

Extract, as concise Markdown bullet points, only what this part actually contains:
- Speakers who appear and any role you can infer
- Topics discussed
- Decisions made
- Action items (task, owner, deadline)
- Open questions

Do not invent anything and do not add a conclusion — this is only one part of the meeting.

TRANSCRIPT PART {index}/{total}:
{chunk}"""


def _merge_prompt(partials: list[str], output_lang: str) -> str:
    joined = "\n\n".join(f"--- NOTES FROM PART {i} ---\n{p}" for i, p in enumerate(partials, 1))
    return f"""You are an expert meeting analyst. Below are notes taken from consecutive parts of a single meeting.

Merge them into one structured summary of the whole meeting, written in {output_lang}.
Remove duplicates, keep every distinct decision and action item, and do not invent anything.

Use Markdown formatting with ## headers for each section.

Include these sections:
{SECTIONS}

NOTES:
{joined}"""
