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
from dataclasses import dataclass, field

import config
from src.batch import collect_audio_files, print_batch_summary, run_batch
from src.cleanup import cleanup_transcript
from src.errors import MinuteAIError
from src.export import export
from src.hardware import MODEL_DOWNLOAD_MB, auto_select_model, is_whisper_model_installed
from src.languages import is_supported as is_supported_language
from src.logger import get_logger, setup_logger
from src.naming import meeting_name_from_path
from src.summarize import summarize_transcript
from src.transcribe import build_speaker_map, format_transcript, transcribe

# Values shipped in config.example.py — a config.py still carrying one of these
# means the user never filled in their own token.
HF_TOKEN_PLACEHOLDERS = {"", "hf_insert_here", "hf_xxxxxxxxxx", "hf_your_token_here", "none"}

MODE_CHOICES = ["full", "transcript", "clean", "summary"]
FORMAT_CHOICES = ["md", "txt", "docx", "pdf", "srt", "all"]
MODEL_CHOICES = ["auto", "tiny", "base", "small", "medium", "large-v3"]
MODES_WITHOUT_SUMMARY = ("transcript", "clean")


@dataclass
class PipelineResult:
    """What one processed audio file produced."""

    audio_path: str
    meeting_name: str
    output_files: list[str] = field(default_factory=list)
    transcript: str = ""
    summary: str = ""
    language: str = ""
    duration_seconds: float = 0.0
    speakers: list[str] = field(default_factory=list)


def hf_token_is_placeholder(token: str) -> bool:
    """True when config.py still holds the example token instead of a real one."""
    value = (token or "").strip().lower()
    return value in HF_TOKEN_PLACEHOLDERS or not value.startswith("hf_")


def parse_args(argv=None):
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
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="When a folder is given, also look inside sub-folders"
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
        help="Speaker names in order of first appearance, comma-separated: e.g. 'Marco,Sara'\n(single file only, requires diarization)"
    )
    parser.add_argument(
        "--model", "-m",
        default=config.DEFAULT_WHISPER_MODEL,
        choices=MODEL_CHOICES,
        help=(
            f"Whisper model to use, or 'auto' to pick one based on available GPU/RAM\n"
            f"(default: {config.DEFAULT_WHISPER_MODEL})"
        )
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
        choices=MODE_CHOICES,
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
        choices=FORMAT_CHOICES,
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

    return parser.parse_args(argv)


def validate_args(args):
    """Validates arguments and catches invalid combinations."""
    log = get_logger()

    if not args.no_diarize and hf_token_is_placeholder(config.HF_TOKEN):
        log.error(
            "HF_TOKEN in config.py is missing or still set to the example value. "
            "Add your HuggingFace token, or use --no-diarize to skip diarization."
        )
        sys.exit(1)

    # Convert speakers to int if not 'auto'
    if args.speakers != "auto":
        try:
            args.speakers = int(args.speakers)
        except (TypeError, ValueError):
            log.error(f"--speakers must be an integer or 'auto', got '{args.speakers}'")
            sys.exit(1)
        if args.speakers < 1:
            log.error(f"--speakers must be at least 1, got {args.speakers}")
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
    if args.export_content == "summary" and args.mode in MODES_WITHOUT_SUMMARY:
        log.error(
            f"--export-content summary requires a summary, "
            f"but --mode {args.mode} does not generate one.\n"
            f"Use --mode full or --mode summary to generate a summary."
        )
        sys.exit(1)

    if not is_supported_language(args.language):
        log.error(f"--language '{args.language}' is not a language Whisper can transcribe.")
        sys.exit(1)

    if args.summary_language != "same" and not is_supported_language(args.summary_language):
        log.error(f"--summary-language '{args.summary_language}' is not a supported language.")
        sys.exit(1)

    if args.parallel_workers < 1:
        log.error(f"--parallel-workers must be at least 1, got {args.parallel_workers}")
        sys.exit(1)

    if args.format == "srt" and args.export_content == "summary":
        log.error("--format srt only carries the timestamped transcript; it cannot hold a summary alone.")
        sys.exit(1)

    return args


def resolve_meeting_name(audio_path: str, args) -> str:
    """Determines the meeting name: explicit --meeting-name, or derived from the filename."""
    return args.meeting_name or meeting_name_from_path(audio_path)


def resolve_model(args) -> str:
    """Resolves '--model auto' to a concrete whisper model based on available hardware."""
    if args.model != "auto":
        return args.model
    workers = args.parallel_workers if args.parallel else 1
    return auto_select_model(parallel_workers=workers)


def warn_if_model_needs_download(model: str):
    """Says so before whisperX quietly fetches a few gigabytes."""
    if model == "auto" or is_whisper_model_installed(model):
        return
    size = MODEL_DOWNLOAD_MB.get(model)
    detail = f" (~{size / 1024:.1f} GB)" if size and size >= 1024 else (f" (~{size} MB)" if size else "")
    get_logger().warning(
        f"Whisper model '{model}' is not downloaded yet{detail}. "
        f"It will be fetched on first use — this needs network access."
    )


def _llm_settings() -> dict:
    """Optional Ollama tuning read from config.py, with safe fallbacks.

    getattr keeps an older config.py (written before these keys existed) working.
    """
    return {
        "num_ctx": getattr(config, "OLLAMA_NUM_CTX", 8192),
        "chunk_chars": getattr(config, "LLM_CHUNK_CHARS", 6000),
        "timeout": getattr(config, "OLLAMA_TIMEOUT", 600),
    }


def process_single(audio_path: str, args, is_batch: bool) -> PipelineResult:
    """Runs the full pipeline on a single audio file."""
    log = get_logger()

    meeting_name = resolve_meeting_name(audio_path, args)

    do_cleanup = args.mode in ("full", "clean")
    do_summary = args.mode in ("full", "summary")
    do_diarize = not args.no_diarize
    llm = _llm_settings()

    # Step 1: Transcription (+ optional diarization)
    result = transcribe(
        audio_path=audio_path,
        language=args.language if args.language != "auto" else None,
        num_speakers=args.speakers,
        model_name=args.model,
        compute_type=getattr(config, "DEFAULT_COMPUTE_TYPE", "auto"),
        hf_token=config.HF_TOKEN,
        diarize=do_diarize,
    )

    speaker_names = args.speaker_names if not is_batch else None
    speaker_map = build_speaker_map(result.segments, speaker_names) if result.diarized else {}
    transcript = format_transcript(result.segments, speaker_map, diarize=result.diarized)

    # Step 2: Cleanup
    if do_cleanup:
        transcript = cleanup_transcript(
            transcript=transcript,
            model=args.cleanup_model,
            host=config.OLLAMA_HOST,
            language=result.language,
            **llm,
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
            transcript_language=result.language,
            summary_language=args.summary_language,
            **llm,
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
        language=result.language,
        output_dir=args.output_dir,
        fmt=args.format,
        export_content=args.export_content,
        duration_seconds=result.duration_seconds,
        segments=result.segments,
    )

    return PipelineResult(
        audio_path=audio_path,
        meeting_name=meeting_name,
        output_files=output_files,
        transcript=transcript,
        summary=summary,
        language=result.language,
        duration_seconds=result.duration_seconds,
        speakers=[speaker_map.get(s, s) for s in result.speakers],
    )


def warn_about_ignored_flags(args, is_batch: bool):
    """Points out flags that have no effect for the run about to start."""
    log = get_logger()
    if is_batch:
        if args.meeting_name:
            log.warning("--meeting-name is ignored in batch mode (filename is used instead)")
        if args.speaker_names:
            log.warning("--speaker-names is ignored in batch mode")
    else:
        if args.parallel:
            log.warning("--parallel has no effect on a single file")
        if args.force:
            log.warning("--force has no effect on a single file (single files are never skipped)")


def main(argv=None) -> int:
    args = parse_args(argv)

    # Initialize logger before anything else
    log = setup_logger()
    log.info("=" * 55)
    log.info("minute-ai started")

    args = validate_args(args)
    warn_if_model_needs_download(args.model)
    args.model = resolve_model(args)

    # Collect all audio files
    audio_files = collect_audio_files(args.audio, recursive=args.recursive)

    if not audio_files:
        log.error("No valid audio files found.")
        return 1

    is_batch = len(audio_files) > 1
    warn_about_ignored_flags(args, is_batch)

    # Single file mode
    if not is_batch:
        audio_path = audio_files[0]
        meeting_name = resolve_meeting_name(audio_path, args)

        log.info("=" * 55)
        log.info(f"minute-ai — {meeting_name}")
        log.info(f"Mode: {args.mode} | Format: {args.format} | Content: {args.export_content} | Diarize: {not args.no_diarize}")
        log.info("=" * 55)

        try:
            result = process_single(audio_path, args, is_batch=False)
        except MinuteAIError as exc:
            log.error(str(exc))
            return 1

        log.info("=" * 55)
        log.info("DONE")
        for f in result.output_files:
            log.info(f"  -> {f}")
        log.info("=" * 55)
        return 0

    # Batch mode
    log.info("=" * 55)
    log.info("minute-ai — BATCH MODE")
    log.info(f"Found {len(audio_files)} file(s)")
    log.info(f"Mode: {args.mode} | Format: {args.format} | Content: {args.export_content} | Diarize: {not args.no_diarize}")
    log.info(f"Parallel: {'yes' if args.parallel else 'no'}")
    log.info("=" * 55)

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
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        get_logger().warning("Interrupted by user.")
        sys.exit(130)
    except MinuteAIError as exc:
        get_logger().error(str(exc))
        sys.exit(1)
