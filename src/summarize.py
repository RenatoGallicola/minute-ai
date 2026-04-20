"""
summarize.py
------------
Generates a structured meeting summary using a local LLM via Ollama.
"""

from src.cleanup import call_ollama, check_ollama


LANGUAGE_NAMES = {
    "it": "Italian",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
}


def summarize_transcript(
    transcript: str,
    model: str,
    host: str,
    transcript_language: str,
    summary_language: str,
) -> str:
    """
    Generates a structured summary of the transcript.

    Args:
        transcript:          Transcript (cleaned or raw)
        model:               Ollama model to use
        host:                Ollama server URL
        transcript_language: Language of the transcript
        summary_language:    Language of the summary ('same', 'it', 'en', etc.)

    Returns:
        Structured summary in Markdown
    """
    print(f"\n[3/4] Generating summary with {model}...")

    if not check_ollama(host, model):
        print(f"[WARNING] Ollama not reachable or model '{model}' not found.")
        print("          Skipping summary step.")
        return ""

    # Determine summary output language
    if summary_language == "same":
        output_lang = LANGUAGE_NAMES.get(transcript_language, transcript_language)
    else:
        output_lang = LANGUAGE_NAMES.get(summary_language, summary_language)

    prompt = f"""You are an expert meeting analyst. Analyze this meeting transcript and produce a structured summary.

Write the summary in {output_lang}.

Use Markdown formatting with ## headers for each section.

Include these sections:
## Participants
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
Any other relevant information, context, or observations.

Be concise but thorough. Focus on actionable content.

TRANSCRIPT:
{transcript}"""

    summary = call_ollama(host, model, prompt)
    print("      Summary generated.")
    return summary
