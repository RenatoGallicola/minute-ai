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


class TestResolveMeetingName:
    def test_explicit_meeting_name_wins(self):
        args = make_args(meeting_name="Q3 Kickoff")
        assert main.resolve_meeting_name("inputs/whatever.wav", args) == "Q3 Kickoff"

    def test_falls_back_to_filename(self):
        args = make_args(meeting_name=None)
        assert main.resolve_meeting_name("inputs/team_sync-notes.wav", args) == "Team Sync Notes"
