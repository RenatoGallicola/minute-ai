#!/usr/bin/env python3
"""
minute-ai
---------
Local pipeline for transcription, cleanup, and summarization of meeting audio.

Single file:
    python main.py inputs/meeting.m4a

Multiple files:
    python main.py inputs/meeting1.m4a inputs/meeting2.m4a

Entire folder:
    python main.py inputs/

Folder in parallel:
    python main.py inputs/ --parallel

Force reprocess already-processed files:
    python main.py inputs/ --force

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
from src.batch import collect_audio_files, run_batch, print_batch_summary


def parse_args():
    parser = argparse.ArgumentParser(
        prog="minute-ai",
        description="Local transcription and summarization pipeline for meeting audio.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Input — one or more files or folders
    parser.add_argument(
        "audio",
        nargs="+",
        help="Path(s) to audio file(s) or folder(s) (mp3, m4a, wav, flac, etc.)"
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
        help="Speaker names in order, comma-separated: e.g. 'Marco,Sara'\n(only used when processing a single file)"
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
        help="Human-readable meeting name (default: audio filename)\n(only used when processing a single file)"
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

    # Batch
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple files in parallel (faster but uses more RAM)"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2, only used with --parallel)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if output already exists"
    )

    return parser.parse_args()


def validate_args(args):
    """Validates arguments before starting."""
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

    return args


def process_single(audio_path: str, args) -> list[str]:
    """
    Runs the full pipeline on a single audio file.

    Args:
        audio_path: Path to the audio file
        args:       Parsed CLI arguments

    Returns:
        List of output file paths
    """
    # Derive meeting name from filename if not specified
    meeting_name = args.meeting_name or \
        Path(audio_path).stem.replace("_", " ").replace("-", " ").title()

    # Step 1: Transcription + diarization
    segments, detected_language = transcribe(
        audio_path=audio_path,
        language=args.language if args.language != "auto" else None,
        num_speakers=args.speakers,
        model_name=args.model,
        compute_type=config.DEFAULT_COMPUTE_TYPE,
        hf_token=config.HF_TOKEN,
    )

    # Map speaker labels to real names (single file only)
    speaker_names = args.speaker_names if len(args.audio) == 1 else None
    speaker_map = build_speaker_map(segments, speaker_names)
    transcript = format_transcript(segments, speaker_map)

    # Step 2: Transcript cleanup
    if not args.no_cleanup:
        transcript = cleanup_transcript(
            transcript=transcript,
            model=args.cleanup_model,
            host=config.OLLAMA_HOST,
            language=detected_language,
        )
    else:
        print("\n[2/4] Transcript cleanup: skipped.")

    # Step 3: Summary
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

    # Step 4: Export
    print(f"\n[4/4] Exporting files...")
    output_files = export(
        audio_path=audio_path,
        meeting_name=meeting_name,
        transcript=transcript,
        summary=summary,
        language=detected_language,
        output_dir=args.output_dir,
        fmt=args.format,
    )

    return output_files


def main():
    args = parse_args()
    args = validate_args(args)

    # Collect all audio files from inputs
    audio_files = collect_audio_files(args.audio)

    if not audio_files:
        print("[ERROR] No valid audio files found.")
        sys.exit(1)

    is_batch = len(audio_files) > 1

    # Single file mode
    if not is_batch:
        audio_path = audio_files[0]
        meeting_name = args.meeting_name or \
            Path(audio_path).stem.replace("_", " ").replace("-", " ").title()

        print("=" * 55)
        print("  minute-ai")
        print(f"  {meeting_name}")
        print("=" * 55)

        output_files = process_single(audio_path, args)

        print("\n" + "=" * 55)
        print("  DONE")
        for f in output_files:
            print(f"  → {f}")
        print("=" * 55)

    # Batch mode
    else:
        print("=" * 55)
        print("  minute-ai — BATCH MODE")
        print(f"  Found {len(audio_files)} file(s)")
        if args.parallel:
            print(f"  Mode: parallel ({args.parallel_workers} workers)")
        else:
            print("  Mode: sequential")
        print("=" * 55)

        if args.meeting_name:
            print("[WARNING] --meeting-name is ignored in batch mode (filename is used instead)")
        if args.speaker_names:
            print("[WARNING] --speaker-names is ignored in batch mode")

        results = run_batch(
            audio_files=audio_files,
            process_fn=lambda audio: process_single(audio, args),
            parallel=args.parallel,
            force=args.force,
            output_dir=args.output_dir,
            fmt=args.format,
            max_workers=args.parallel_workers,
        )

        print_batch_summary(results)


if __name__ == "__main__":
    main()
