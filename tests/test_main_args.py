import argparse

import pytest

import main


def make_args(**overrides):
    defaults = dict(
        audio=["meeting.wav"],
        language="auto",
        speakers="auto",
        speaker_names=None,
        model="medium",
        no_diarize=False,
        meeting_name=None,
        mode="full",
        cleanup_model="llama3.1",
        summary_model="llama3.1",
        summary_language="same",
        output_dir="outputs",
        format="md",
        export_content="full",
        parallel=False,
        parallel_workers=2,
        force=False,
        recursive=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestValidateArgs:
    def test_valid_defaults_pass_through(self):
        args = make_args()
        result = main.validate_args(args)
        assert result.speakers is None  # "auto" -> None

    def test_speakers_integer_string_is_converted(self):
        args = make_args(speakers="3")
        result = main.validate_args(args)
        assert result.speakers == 3

    def test_invalid_speakers_value_exits(self):
        args = make_args(speakers="not-a-number")
        with pytest.raises(SystemExit):
            main.validate_args(args)

    def test_speaker_names_with_no_diarize_exits(self):
        args = make_args(no_diarize=True, speaker_names="Alice,Bob")
        with pytest.raises(SystemExit):
            main.validate_args(args)

    def test_explicit_speakers_with_no_diarize_exits(self):
        args = make_args(no_diarize=True, speakers="2")
        with pytest.raises(SystemExit):
            main.validate_args(args)

    def test_no_diarize_alone_is_valid(self):
        args = make_args(no_diarize=True)
        result = main.validate_args(args)
        assert result.speakers is None

    def test_export_content_summary_requires_summary_mode(self):
        args = make_args(mode="transcript", export_content="summary")
        with pytest.raises(SystemExit):
            main.validate_args(args)

        args = make_args(mode="clean", export_content="summary")
        with pytest.raises(SystemExit):
            main.validate_args(args)

    def test_export_content_summary_allowed_with_full_or_summary_mode(self):
        for mode in ("full", "summary"):
            args = make_args(mode=mode, export_content="summary")
            main.validate_args(args)  # should not raise

    def test_missing_hf_token_without_no_diarize_exits(self, monkeypatch):
        monkeypatch.setattr(main.config, "HF_TOKEN", "hf_XXXXXXXXXX")
        args = make_args(no_diarize=False)
        with pytest.raises(SystemExit):
            main.validate_args(args)

    def test_missing_hf_token_with_no_diarize_is_fine(self, monkeypatch):
        monkeypatch.setattr(main.config, "HF_TOKEN", "hf_XXXXXXXXXX")
        args = make_args(no_diarize=True)
        main.validate_args(args)  # should not raise


class TestResolveModel:
    def test_explicit_model_is_returned_unchanged(self):
        args = make_args(model="small")
        assert main.resolve_model(args) == "small"

    def test_auto_delegates_to_hardware_detection(self, monkeypatch):
        monkeypatch.setattr(main, "auto_select_model", lambda parallel_workers: "medium")
        args = make_args(model="auto", parallel=False)
        assert main.resolve_model(args) == "medium"

    def test_auto_passes_worker_count_only_when_parallel(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            main, "auto_select_model",
            lambda parallel_workers: seen.setdefault("workers", parallel_workers) or "medium"
        )
        args = make_args(model="auto", parallel=True, parallel_workers=4)
        main.resolve_model(args)
        assert seen["workers"] == 4

    def test_auto_uses_single_worker_when_not_parallel(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            main, "auto_select_model",
            lambda parallel_workers: seen.setdefault("workers", parallel_workers) or "medium"
        )
        args = make_args(model="auto", parallel=False, parallel_workers=4)
        main.resolve_model(args)
        assert seen["workers"] == 1


class TestResolveMeetingName:
    def test_explicit_meeting_name_wins(self):
        args = make_args(meeting_name="Q3 Kickoff")
        assert main.resolve_meeting_name("inputs/whatever.wav", args) == "Q3 Kickoff"

    def test_falls_back_to_filename(self):
        args = make_args(meeting_name=None)
        assert main.resolve_meeting_name("inputs/team_sync-notes.wav", args) == "Team Sync Notes"


class TestValidateArgsExtra:
    def test_example_placeholder_token_is_caught(self, monkeypatch):
        # Regression: the check only knew "hf_XXXXXXXXXX", while
        # config.example.py shipped "hf_INSERT_HERE" — so a fresh copy of the
        # example passed validation and failed deep inside pyannote instead.
        monkeypatch.setattr(main.config, "HF_TOKEN", "hf_INSERT_HERE")
        with pytest.raises(SystemExit):
            main.validate_args(make_args())

    def test_empty_token_is_caught(self, monkeypatch):
        monkeypatch.setattr(main.config, "HF_TOKEN", "")
        with pytest.raises(SystemExit):
            main.validate_args(make_args())

    def test_real_looking_token_passes(self, monkeypatch):
        monkeypatch.setattr(main.config, "HF_TOKEN", "hf_aRealLookingToken123")
        main.validate_args(make_args())

    def test_zero_speakers_is_rejected(self):
        # Regression: 0 is falsy, so it slipped past the diarization check and
        # was handed to pyannote as min_speakers=0.
        with pytest.raises(SystemExit):
            main.validate_args(make_args(speakers="0"))

    def test_negative_speakers_is_rejected(self):
        with pytest.raises(SystemExit):
            main.validate_args(make_args(speakers="-2"))

    def test_zero_parallel_workers_is_rejected(self):
        # ThreadPoolExecutor(max_workers=0) raises ValueError deep in the batch.
        with pytest.raises(SystemExit):
            main.validate_args(make_args(parallel_workers=0))

    def test_srt_cannot_carry_a_summary_alone(self):
        with pytest.raises(SystemExit):
            main.validate_args(make_args(format="srt", export_content="summary"))

    def test_srt_with_full_content_is_fine(self):
        main.validate_args(make_args(format="srt", export_content="full"))


class TestHfTokenPlaceholder:
    def test_known_placeholders(self):
        for value in ("", "  ", "hf_INSERT_HERE", "hf_XXXXXXXXXX", "none", "your-token"):
            assert main.hf_token_is_placeholder(value) is True

    def test_real_token(self):
        assert main.hf_token_is_placeholder("hf_abcDEF123456") is False


class TestWarnAboutIgnoredFlags:
    def test_parallel_on_a_single_file_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="minute-ai"):
            main.warn_about_ignored_flags(make_args(parallel=True), is_batch=False)
        assert any("--parallel" in r.message for r in caplog.records)

    def test_meeting_name_in_batch_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="minute-ai"):
            main.warn_about_ignored_flags(make_args(meeting_name="Q3"), is_batch=True)
        assert any("--meeting-name" in r.message for r in caplog.records)


class TestParseArgs:
    def test_defaults(self):
        args = main.parse_args(["inputs/a.wav"])
        assert args.audio == ["inputs/a.wav"]
        assert args.recursive is False

    def test_srt_is_an_accepted_format(self):
        assert main.parse_args(["a.wav", "--format", "srt"]).format == "srt"

    def test_recursive_flag(self):
        assert main.parse_args(["inputs/", "-r"]).recursive is True
