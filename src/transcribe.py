"""
transcribe.py
-------------
Audio transcription and speaker diarization using whisperX.
"""

import sys
from src.logger import get_logger


def transcribe(
    audio_path: str,
    language: str,
    num_speakers,
    model_name: str,
    compute_type: str,
    hf_token: str,
) -> tuple[list[dict], str]:
    """
    Transcribes an audio file and identifies speakers.

    Args:
        audio_path:   Path to the audio file
        language:     Language code ('it', 'en', etc.) or None for auto-detect
        num_speakers: Number of speakers (int) or None for auto-detect
        model_name:   Whisper model to use
        compute_type: Computation type ('int8' for CPU, 'float16' for GPU)
        hf_token:     HuggingFace token for the diarization model

    Returns:
        (segments, detected_language)
    """
    log = get_logger()

    try:
        import whisperx
    except ImportError:
        log.error("whisperx is not installed. Run: pip install whisperx")
        sys.exit(1)

    device = "cpu"
    lang = language if language and language != "auto" else None

    log.info(f"[1/4] Transcribing audio: {audio_path}")
    log.info(f"      Model: {model_name} | Language: {lang or 'auto-detect'} | Speakers: {num_speakers or 'auto'}")

    # Load model and transcribe
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=lang)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=8)
    detected_language = result.get("language", lang or "en")
    log.info(f"      Detected language: {detected_language}")

    # Temporal alignment
    log.info("      Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False
    )

    # Diarization
    log.info("      Diarizing speakers...")
    diarize_kwargs = {}
    if num_speakers:
        diarize_kwargs["min_speakers"] = num_speakers
        diarize_kwargs["max_speakers"] = num_speakers

    diarize_model = whisperx.DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        device=device
    )
    diarize_segments = diarize_model(audio, **diarize_kwargs)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = result["segments"]
    log.info(f"      Done — {len(segments)} segments")
    return segments, detected_language


def format_transcript(segments: list[dict], speaker_names: dict = None) -> str:
    """
    Formats segments into a readable transcript with speaker labels.

    Args:
        segments:      whisperX segment list
        speaker_names: Optional dict {SPEAKER_00: "Marco", SPEAKER_01: "Sara"}

    Returns:
        Formatted string with speakers and text
    """
    lines = []
    current_speaker = None
    current_text = []

    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        text = seg.get("text", "").strip()

        if not text:
            continue

        display_name = speaker
        if speaker_names:
            display_name = speaker_names.get(speaker, speaker)

        if display_name != current_speaker:
            if current_speaker and current_text:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = display_name
            current_text = [text]
        else:
            current_text.append(text)

    if current_speaker and current_text:
        lines.append(f"{current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(lines)


def build_speaker_map(segments: list[dict], speaker_names_str: str) -> dict:
    """
    Builds a {SPEAKER_00: "Name"} dictionary from a comma-separated name string.

    Args:
        segments:          whisperX segments
        speaker_names_str: String "Marco,Sara" in order of appearance

    Returns:
        Dict {SPEAKER_00: "Marco", SPEAKER_01: "Sara"}
    """
    if not speaker_names_str:
        return {}

    seen = []
    for seg in segments:
        sp = seg.get("speaker")
        if sp and sp not in seen:
            seen.append(sp)
    seen.sort()

    names = [n.strip() for n in speaker_names_str.split(",")]
    return {sp: names[i] for i, sp in enumerate(seen) if i < len(names)}
