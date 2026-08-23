"""
prompts.py
----------
Summary templates.

The default six-section layout is shaped like a work meeting, which is wrong
for a lecture (no decisions, no action items) or an interview. A preset picks
what the summary should contain; a custom template replaces it entirely.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryPreset:
    key: str
    label: str
    note: str
    instructions: str


MEETING = SummaryPreset(
    key="meeting",
    label="Meeting",
    note="Decisions and action items",
    instructions="""## Participants
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
Any other relevant information, context, or observations.""",
)

LECTURE = SummaryPreset(
    key="lecture",
    label="Lecture",
    note="Concepts, definitions, examples",
    instructions="""## Overview
What this lecture covered, in two or three sentences.

## Key Concepts
The main ideas introduced, each with a short explanation in your own words.

## Definitions
Terms that were defined, with their definitions. If none, write "None".

## Examples and Demonstrations
Worked examples, case studies or demonstrations used to illustrate the concepts.

## Practical Notes
Anything about assessment, deadlines, required reading, tools or logistics. If none, write "None".

## Open Questions
Points the speaker left unresolved, or flagged as coming later in the course.""",
)

INTERVIEW = SummaryPreset(
    key="interview",
    label="Interview",
    note="Positions, quotes, follow-ups",
    instructions="""## Participants
Who took part, and their role in the conversation (interviewer, interviewee, and any inferred background).

## Topics Covered
The subjects discussed, in the order they came up.

## Key Points
The substance of what was said on each topic: positions, claims, and the reasoning behind them.

## Notable Quotes
Direct quotes worth keeping, attributed to the speaker. Quote verbatim; do not paraphrase.

## Follow-ups
Questions left unanswered, or threads worth returning to.""",
)

ONE_ON_ONE = SummaryPreset(
    key="one-on-one",
    label="One-on-one",
    note="Updates, blockers, next steps",
    instructions="""## Updates
What has happened since the last conversation.

## Blockers
Anything getting in the way, and who or what it depends on. If none, write "None".

## Feedback
Feedback given or received, in either direction. If none, write "None".

## Agreed Next Steps
What each person committed to, with owner and timing where mentioned. If none, write "None".""",
)

CUSTOM_KEY = "custom"

PRESETS = {p.key: p for p in (MEETING, LECTURE, INTERVIEW, ONE_ON_ONE)}
PRESET_KEYS = list(PRESETS) + [CUSTOM_KEY]
DEFAULT_PRESET = MEETING.key

TRANSCRIPT_PLACEHOLDER = "{transcript}"
LANGUAGE_PLACEHOLDER = "{language}"


def resolve_instructions(preset: str, custom: str = "") -> str:
    """The instruction body to summarize with.

    A custom template wins whenever it has content, so a preset left at its
    default never silently overrides what the user typed.
    """
    text = (custom or "").strip()
    if preset == CUSTOM_KEY or text:
        return text or PRESETS[DEFAULT_PRESET].instructions
    return PRESETS.get(preset, PRESETS[DEFAULT_PRESET]).instructions


def is_full_template(instructions: str) -> bool:
    """True when the text is a complete prompt rather than an instruction body.

    Writing {transcript} yourself is the opt-in for full control: minute-ai then
    stops wrapping the text and uses it verbatim.
    """
    return TRANSCRIPT_PLACEHOLDER in (instructions or "")


def fill(template: str, transcript: str, language: str) -> str:
    """Substitutes the two supported placeholders.

    Uses plain replacement rather than str.format so that stray braces in a
    hand-written template cannot raise KeyError mid-run.
    """
    return (template
            .replace(TRANSCRIPT_PLACEHOLDER, transcript)
            .replace(LANGUAGE_PLACEHOLDER, language))


def options() -> list[tuple[str, str, str]]:
    """(value, label, note) triples for the GUI picker."""
    return [(p.key, p.label, p.note) for p in PRESETS.values()] + [
        (CUSTOM_KEY, "Custom", "Write your own"),
    ]
