#!/usr/bin/env python3
"""
minute-ai
---------
Local pipeline for transcription, cleanup, and summarization of meeting audio.

Basic usage:
    python main.py inputs/meeting.m4a

Full usage:
    python main.py inputs/meeting.m4a \
        --language en \
        --speakers 2 \
        --speaker-names "Marco,Sara" \
        --meeting-name "Q3 Kickoff" \
        --model medium \
        --summary-language it \
        --format md
"""

import argparse
import sys
import os
from pathlib import Path

import config
from src.transcribe import transcribe, format_transcript, build_speaker_map
from src.cleanup import cleanup_transcript
from src.summarize import summarize_transcript
from src.export import export


def parse_args():
    parser = argparse.ArgumentParser(
        prog="minute-ai",
        description="Local transcription and summarization pipeline for meeting audio.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Input
    parser.add_argument(
        "audio",
        help="Path to the audio file (mp3, m4a, wav, flac, etc.)"
    )

    # Transcription
    parser.add_argument(
        "--language", "-l",
        default=config.DEFAULT_LANGUAGE,
        help=f"Audio language: 'it', 'en', 'auto', etc. (default: {config.DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--speakers", "-s",
        default=config.DEFAULT_SPEAKERS,
        help=f"Number of speakers: integer or 'auto' (default: {config.DEFAULT_SPEAKERS})"
    )
    parser.add_argument(
        "--speaker-names",
        default=None,
        help="Speaker names in order, comma-separated: e.g. 'Marco,Sara'"
    )
    parser.add_argument(
        "--model", "-m",
        default=config.DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help=f"Whisper model to use (default: {config.DEFAULT_WHISPER_MODEL})"
    )

    # Meeting
    parser.add_argument(
        "--meeting-name", "-n",
        default=None,
        help="Human-readable meeting name (default: audio filename)"
    )

    # Cleanup
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable transcript cleanup with LLM"
    )
    parser.add_argument(
        "--cleanup-model",
        default=config.DEFAULT_CLEANUP_MODEL,
        help=f"Ollama model for cleanup (default: {config.DEFAULT_CLEANUP_MODEL})"
    )

    # Summary
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summary generation"
    )
    parser.add_argument(
        "--summary-model",
        default=config.DEFAULT_SUMMARY_MODEL,
        help=f"Ollama model for summary (default: {config.DEFAULT_SUMMARY_MODEL})"
    )
    parser.add_argument(
        "--summary-language",
        default=config.DEFAULT_SUMMARY_LANGUAGE,
        help=f"Summary language: 'same', 'it', 'en', etc. (default: {config.DEFAULT_SUMMARY_LANGUAGE})"
    )

    # Export
    parser.add_argument(
        "--output-dir", "-o",
        default=config.DEFAULT_OUTPUT_DIR,
        help=f"Output folder (default: {config.DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--format", "-f",
        default=config.DEFAULT_FORMAT,
        choices=["md", "txt", "all"],
        help=f"Output format (default: {config.DEFAULT_FORMAT})"
    )

    return parser.parse_args()


def validate_args(args):
    """Validates arguments before starting."""
    if not os.path.exists(args.audio):
        print(f"[ERROR] File not found: {args.audio}")
        sys.exit(1)

    if config.HF_TOKEN == "hf_XXXXXXXXXX":
        print("[ERROR] Please insert your HF_TOKEN in config.py")
        sys.exit(1)

    # Convert speakers to int if not 'auto'
    if args.speakers != "auto":
        try:
            args.speakers = int(args.speakers)
        except ValueError:
            print(f"[ERROR] --speakers must be an integer or 'auto', got '{args.speakers}'")
            sys.exit(1)
    else:
        args.speakers = None

    # Default meeting name from filename
    if not args.meeting_name:
        args.meeting_name = Path(args.audio).stem.replace("_", " ").replace("-", " ").title()

    return args


def main():
    args = parse_args()
    args = validate_args(args)

    print("=" * 55)
    print("  minute-ai")
    print(f"  {args.meeting_name}")
    print("=" * 55)

    # ── Step 1: Transcription + diarization ──
    segments, detected_language = transcribe(
        audio_path=args.audio,
        language=args.language if args.language != "auto" else None,
        num_speakers=args.speakers,
        model_name=args.model,
        compute_type=config.DEFAULT_COMPUTE_TYPE,
        hf_token=config.HF_TOKEN,
    )

    # Map speaker labels → real names
    speaker_map = build_speaker_map(segments, args.speaker_names)

    # Format transcript
    transcript = format_transcript(segments, speaker_map)

    # ── Step 2: Transcript cleanup ──
    if not args.no_cleanup:
        transcript = cleanup_transcript(
            transcript=transcript,
            model=args.cleanup_model,
            host=config.OLLAMA_HOST,
            language=detected_language,
        )
    else:
        print("\n[2/4] Transcript cleanup: skipped.")

    # ── Step 3: Summary ──
    summary = ""
    if not args.no_summary:
        summary = summarize_transcript(
            transcript=transcript,
            model=args.summary_model,
            host=config.OLLAMA_HOST,
            transcript_language=detected_language,
            summary_language=args.summary_language,
        )
    else:
        print("\n[3/4] Summary: skipped.")

    # ── Step 4: Export ──
    print(f"\n[4/4] Exporting files...")
    output_files = export(
        audio_path=args.audio,
        meeting_name=args.meeting_name,
        transcript=transcript,
        summary=summary,
        language=detected_language,
        output_dir=args.output_dir,
        fmt=args.format,
    )

    print("\n" + "=" * 55)
    print("  DONE")
    for f in output_files:
        print(f"  → {f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
