"""
transcribe.py
-------------
Audio transcription and optional speaker diarization using whisperX.
"""

from dataclasses import dataclass, field

from src.errors import DependencyMissingError, TranscriptionError
from src.logger import get_logger


WHISPER_SAMPLE_RATE = 16000
SINGLE_SPEAKER_LABEL = "SPEAKER_00"


@dataclass
class TranscriptionResult:
    """Everything the transcription step produces about one audio file."""

    segments: list[dict]
    language: str
    duration_seconds: float = 0.0
    diarized: bool = False
    speakers: list[str] = field(default_factory=list)


def resolve_compute_type(configured: str, device: str) -> str:
    """Picks a compute type valid for the current device.

    CTranslate2 cannot run float16 on CPU (it warns and silently falls back),
    and int8 on a GPU throws away most of the speed-up. 'auto' resolves to the
    right one for whatever hardware is actually present.
    """
    value = (configured or "auto").strip().lower()
    if value in ("", "auto"):
        return "float16" if device == "cuda" else "int8"
    if value == "float16" and device != "cuda":
        get_logger().warning("compute_type 'float16' needs a GPU — falling back to 'int8' on CPU.")
        return "int8"
    return value


def transcribe(
    audio_path: str,
    language: str,
    num_speakers,
    model_name: str,
    compute_type: str,
    hf_token: str,
    diarize: bool = True,
) -> TranscriptionResult:
    """
    Transcribes an audio file and optionally identifies speakers.

    Args:
        audio_path:   Path to the audio file
        language:     Language code ('it', 'en', etc.) or None for auto-detect
        num_speakers: Number of speakers (int) or None for auto-detect
        model_name:   Whisper model to use
        compute_type: Computation type ('auto', 'int8' for CPU, 'float16' for GPU)
        hf_token:     HuggingFace token for the diarization model
        diarize:      Whether to run speaker diarization (default: True)

    Returns:
        TranscriptionResult

    Raises:
        DependencyMissingError: whisperX or torch is not installed
        TranscriptionError:     the audio could not be loaded or transcribed
    """
    log = get_logger()

    try:
        import torch
        import whisperx
    except ImportError as exc:
        raise DependencyMissingError(
            f"whisperx/torch is not installed ({exc}). Run: pip install -r requirements.txt"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = resolve_compute_type(compute_type, device)
    lang = language if language and language != "auto" else None

    log.info(f"[1/4] Transcribing audio: {audio_path}")
    log.info(
        f"      Model: {model_name} | Device: {device} | Compute: {compute_type} | "
        f"Language: {lang or 'auto-detect'} | Speakers: {num_speakers or 'auto'} | Diarize: {diarize}"
    )

    try:
        audio = whisperx.load_audio(audio_path)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not read the audio file ({exc}). Is ffmpeg installed and the file a valid recording?"
        ) from exc

    duration = len(audio) / WHISPER_SAMPLE_RATE
    log.info(f"      Audio duration: {format_duration(duration)}")

    try:
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=lang)
        result = model.transcribe(audio, batch_size=8)
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed ({exc}).") from exc

    detected_language = result.get("language") or lang or "en"
    log.info(f"      Detected language: {detected_language}")

    if not result.get("segments"):
        log.warning("      No speech detected in this file.")
        return TranscriptionResult([], detected_language, duration, diarized=False)

    # Temporal alignment — best-effort: whisperX ships no alignment model for
    # every language, and a missing one must not cost us the transcript.
    log.info("      Aligning timestamps...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False
        )
    except Exception as exc:
        log.warning(f"      Alignment unavailable for '{detected_language}' ({exc}). Continuing unaligned.")

    diarized = False
    if diarize and num_speakers == 1:
        # Nothing to tell apart. Running pyannote anyway costs minutes and can
        # only ever return the answer we were already given.
        log.info("      One speaker expected — skipping diarization.")
        for seg in result["segments"]:
            seg["speaker"] = SINGLE_SPEAKER_LABEL
        diarized = True
    elif diarize:
        log.info("      Diarizing speakers...")
        diarize_kwargs = {}
        if num_speakers:
            diarize_kwargs["min_speakers"] = num_speakers
            diarize_kwargs["max_speakers"] = num_speakers
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.diarize.assign_word_speakers(diarize_segments, result)
            diarized = True
        except Exception as exc:
            # Usually a bad or ungated HF token. Falling back to a speaker-less
            # transcript beats throwing away a transcription that already ran.
            log.error(
                f"      Diarization failed ({exc}). Continuing without speaker labels — "
                f"check that HF_TOKEN is valid and that you accepted the pyannote model terms."
            )
    else:
        log.info("      Diarization: skipped.")

    segments = result["segments"]
    speakers = sorted({seg["speaker"] for seg in segments if seg.get("speaker")}) if diarized else []
    log.info(f"      Done — {len(segments)} segments" + (f", {len(speakers)} speaker(s)" if speakers else ""))
    return TranscriptionResult(segments, detected_language, duration, diarized, speakers)


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds as '1h 05m' or '4m 12s'."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {secs:02d}s"


# A speaker's turn is broken into a new paragraph when they pause for this
# long, or when the paragraph simply gets too long to read comfortably.
# Without this, a lecture or a voice memo (one speaker, no turn changes) comes
# out as a single unbroken wall of text.
PARAGRAPH_GAP_SECONDS = 2.0
PARAGRAPH_MAX_CHARS = 700


def format_transcript(
    segments: list[dict],
    speaker_names: dict = None,
    diarize: bool = True,
    gap_seconds: float = PARAGRAPH_GAP_SECONDS,
    max_chars: int = PARAGRAPH_MAX_CHARS,
) -> str:
    """
    Formats segments into a readable transcript.

    Consecutive segments from the same speaker are joined into one paragraph,
    which is broken whenever they pause for `gap_seconds` or the paragraph
    passes `max_chars`. Only the first paragraph of a turn carries the speaker
    label, so the continuation paragraphs read as the same person talking.

    Args:
        segments:      whisperX segment list
        speaker_names: Optional dict {SPEAKER_00: "Marco", SPEAKER_01: "Sara"}
        diarize:       Whether speaker labels are present in segments
        gap_seconds:   Silence that starts a new paragraph (0 disables)
        max_chars:     Longest paragraph before forcing a break (0 disables)

    Returns:
        Formatted transcript string
    """
    paragraphs = []
    current_speaker = None
    current_text = []
    label_pending = False
    previous_end = None

    def flush():
        nonlocal current_text, label_pending
        if not current_text:
            return
        body = " ".join(current_text)
        paragraphs.append(f"{current_speaker}: {body}" if label_pending else body)
        current_text = []
        label_pending = False

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        if diarize:
            speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
            display_name = speaker_names.get(speaker, speaker) if speaker_names else speaker
        else:
            display_name = None

        start = seg.get("start")
        gap = (
            start - previous_end
            if gap_seconds and start is not None and previous_end is not None
            else 0
        )
        too_long = max_chars and sum(len(t) + 1 for t in current_text) >= max_chars

        if display_name != current_speaker:
            flush()
            current_speaker = display_name
            label_pending = diarize
        elif gap >= gap_seconds > 0 or too_long:
            flush()

        current_text.append(text)
        if seg.get("end") is not None:
            previous_end = seg["end"]

    flush()
    return "\n\n".join(paragraphs)


def build_speaker_map(segments: list[dict], speaker_names_str: str) -> dict:
    """
    Builds a {SPEAKER_00: "Name"} dictionary from a comma-separated name string.

    Names are matched in order of first appearance in the recording — the only
    order a user can know without opening the output first. (This used to sort
    the labels instead, which silently swapped names whenever pyannote did not
    hand SPEAKER_00 to whoever spoke first.)

    Args:
        segments:          whisperX segments
        speaker_names_str: String "Marco,Sara" in order of first appearance

    Returns:
        Dict {SPEAKER_00: "Marco", SPEAKER_01: "Sara"}
    """
    if not speaker_names_str:
        return {}

    log = get_logger()
    seen = []
    for seg in segments:
        sp = seg.get("speaker")
        if sp and sp not in seen:
            seen.append(sp)

    names = [n.strip() for n in speaker_names_str.split(",") if n.strip()]

    # Mismatches used to pass silently, so you only found out by reading the
    # export and spotting a stray SPEAKER_02 or a name that never appears.
    if len(names) > len(seen):
        log.warning(
            f"{len(names)} speaker names given but only {len(seen)} speaker(s) were detected — "
            f"ignoring: {', '.join(names[len(seen):])}"
        )
    elif len(names) < len(seen):
        log.warning(
            f"Only {len(names)} speaker name(s) given for {len(seen)} detected speaker(s) — "
            f"the rest keep their {seen[len(names)]}-style labels."
        )

    return {sp: names[i] for i, sp in enumerate(seen) if i < len(names)}
