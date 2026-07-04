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

Single speaker (no diarization):
    python main.py inputs/meeting.m4a --no-diarize

Full usage:
    python main.py inputs/meeting.m4a \
        --language en \
        --speakers 2 \
        --speaker-names "Marco,Sara" \
        --meeting-name "Q3 Kickoff" \
        --model medium \
        --mode full \
        --summary-language it \
        --format pdf \
        --export-content summary
"""

import argparse
import sys
from pathlib import Path

import config
from src.logger import setup_logger, get_logger
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

    # Input
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
        help="Speaker names in order, comma-separated: e.g. 'Marco,Sara'\n(single file only, requires diarization)"
    )
    parser.add_argument(
        "--model", "-m",
        default=config.DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help=f"Whisper model to use (default: {config.DEFAULT_WHISPER_MODEL})"
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization (use for single-speaker audio)"
    )

    # Meeting
    parser.add_argument(
        "--meeting-name", "-n",
        default=None,
        help="Human-readable meeting name (default: audio filename)\n(single file only)"
    )

    # Pipeline mode
    parser.add_argument(
        "--mode",
        default=config.DEFAULT_MODE,
        choices=["full", "transcript", "clean", "summary"],
        help=(
            "Pipeline mode (default: full)\n"
            "  full       — transcribe + cleanup + summary\n"
            "  transcript — transcribe only\n"
            "  clean      — transcribe + cleanup, no summary\n"
            "  summary    — transcribe + summary, no cleanup"
        )
    )

    # LLM models
    parser.add_argument(
        "--cleanup-model",
        default=config.DEFAULT_CLEANUP_MODEL,
        help=f"Ollama model for cleanup (default: {config.DEFAULT_CLEANUP_MODEL})"
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
        choices=["md", "txt", "docx", "pdf", "all"],
        help=f"Output format (default: {config.DEFAULT_FORMAT})"
    )
    parser.add_argument(
        "--export-content",
        default=config.DEFAULT_EXPORT_CONTENT,
        choices=["full", "summary"],
        help=(
            "What to include in the exported file (default: full)\n"
            "  full    — summary + full transcript\n"
            "  summary — summary only"
        )
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
    """Validates arguments and catches invalid combinations."""
    log = get_logger()

    if config.HF_TOKEN == "hf_XXXXXXXXXX" and not args.no_diarize:
        log.error("Please insert your HF_TOKEN in config.py (or use --no-diarize to skip diarization)")
        sys.exit(1)

    # Convert speakers to int if not 'auto'
    if args.speakers != "auto":
        try:
            args.speakers = int(args.speakers)
        except ValueError:
            log.error(f"--speakers must be an integer or 'auto', got '{args.speakers}'")
            sys.exit(1)
    else:
        args.speakers = None

    # --speaker-names requires diarization
    if args.speaker_names and args.no_diarize:
        log.error("--speaker-names requires diarization. Remove --no-diarize or remove --speaker-names.")
        sys.exit(1)

    # --speakers requires diarization
    if args.speakers and args.no_diarize:
        log.error("--speakers requires diarization. Remove --no-diarize or remove --speakers.")
        sys.exit(1)

    # --export-content summary requires a mode that generates a summary
    modes_without_summary = ("transcript", "clean")
    if args.export_content == "summary" and args.mode in modes_without_summary:
        log.error(
            f"--export-content summary requires a summary, "
            f"but --mode {args.mode} does not generate one.\n"
            f"Use --mode full or --mode summary to generate a summary."
        )
        sys.exit(1)

    return args


def resolve_meeting_name(audio_path: str, args) -> str:
    """Determines the meeting name: explicit --meeting-name, or derived from the filename."""
    return args.meeting_name or \
        Path(audio_path).stem.replace("_", " ").replace("-", " ").title()


def process_single(audio_path: str, args, is_batch: bool) -> list[str]:
    """Runs the full pipeline on a single audio file."""
    log = get_logger()

    meeting_name = resolve_meeting_name(audio_path, args)

    do_cleanup = args.mode in ("full", "clean")
    do_summary = args.mode in ("full", "summary")
    do_diarize = not args.no_diarize

    # Step 1: Transcription (+ optional diarization)
    segments, detected_language = transcribe(
        audio_path=audio_path,
        language=args.language if args.language != "auto" else None,
        num_speakers=args.speakers,
        model_name=args.model,
        compute_type=config.DEFAULT_COMPUTE_TYPE,
        hf_token=config.HF_TOKEN,
        diarize=do_diarize,
    )

    speaker_names = args.speaker_names if not is_batch else None
    speaker_map = build_speaker_map(segments, speaker_names) if do_diarize else {}
    transcript = format_transcript(segments, speaker_map, diarize=do_diarize)

    # Step 2: Cleanup
    if do_cleanup:
        transcript = cleanup_transcript(
            transcript=transcript,
            model=args.cleanup_model,
            host=config.OLLAMA_HOST,
            language=detected_language,
        )
    else:
        log.info("[2/4] Transcript cleanup: skipped.")

    # Step 3: Summary
    summary = ""
    if do_summary:
        summary = summarize_transcript(
            transcript=transcript,
            model=args.summary_model,
            host=config.OLLAMA_HOST,
            transcript_language=detected_language,
            summary_language=args.summary_language,
        )
    else:
        log.info("[3/4] Summary: skipped.")

    # Step 4: Export
    log.info("[4/4] Exporting files...")
    output_files = export(
        audio_path=audio_path,
        meeting_name=meeting_name,
        transcript=transcript,
        summary=summary,
        language=detected_language,
        output_dir=args.output_dir,
        fmt=args.format,
        export_content=args.export_content,
    )

    return output_files


def main():
    args = parse_args()

    # Initialize logger before anything else
    log = setup_logger()
    log.info("=" * 55)
    log.info("minute-ai started")

    args = validate_args(args)

    # Collect all audio files
    audio_files = collect_audio_files(args.audio)

    if not audio_files:
        log.error("No valid audio files found.")
        sys.exit(1)

    is_batch = len(audio_files) > 1

    # Single file mode
    if not is_batch:
        audio_path = audio_files[0]
        meeting_name = resolve_meeting_name(audio_path, args)

        log.info("=" * 55)
        log.info(f"minute-ai — {meeting_name}")
        log.info(f"Mode: {args.mode} | Format: {args.format} | Content: {args.export_content} | Diarize: {not args.no_diarize}")
        log.info("=" * 55)

        output_files = process_single(audio_path, args, is_batch=False)

        log.info("=" * 55)
        log.info("DONE")
        for f in output_files:
            log.info(f"  → {f}")
        log.info("=" * 55)

    # Batch mode
    else:
        log.info("=" * 55)
        log.info("minute-ai — BATCH MODE")
        log.info(f"Found {len(audio_files)} file(s)")
        log.info(f"Mode: {args.mode} | Format: {args.format} | Content: {args.export_content} | Diarize: {not args.no_diarize}")
        log.info(f"Parallel: {'yes' if args.parallel else 'no'}")
        log.info("=" * 55)

        if args.meeting_name:
            log.warning("--meeting-name is ignored in batch mode (filename is used instead)")
        if args.speaker_names:
            log.warning("--speaker-names is ignored in batch mode")

        results = run_batch(
            audio_files=audio_files,
            process_fn=lambda audio: process_single(audio, args, is_batch=True),
            parallel=args.parallel,
            force=args.force,
            output_dir=args.output_dir,
            fmt=args.format,
            max_workers=args.parallel_workers,
        )

        print_batch_summary(results)


if __name__ == "__main__":
    main()
