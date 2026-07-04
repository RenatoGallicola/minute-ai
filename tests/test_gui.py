import gradio as gr
import pytest

import gui


class TestBuildArgs:
    def test_single_file_keeps_speaker_names_and_meeting_name(self):
        args = gui._build_args(
            language="en", model="medium", mode="full", diarize=True,
            speakers="2", speaker_names="Marco,Sara", meeting_name="Q3 Kickoff",
            fmt="md", export_content="full", cleanup_model="llama3.1",
            summary_model="llama3.1", summary_language="same", output_dir="outputs",
            is_batch=False,
        )
        assert args.speaker_names == "Marco,Sara"
        assert args.meeting_name == "Q3 Kickoff"
        assert args.speakers == 2
        assert args.no_diarize is False

    def test_batch_drops_speaker_names_and_meeting_name(self):
        args = gui._build_args(
            language="en", model="medium", mode="full", diarize=True,
            speakers="auto", speaker_names="Marco,Sara", meeting_name="Q3 Kickoff",
            fmt="md", export_content="full", cleanup_model="llama3.1",
            summary_model="llama3.1", summary_language="same", output_dir="outputs",
            is_batch=True,
        )
        assert args.speaker_names is None
        assert args.meeting_name is None

    def test_auto_speakers_becomes_none(self):
        args = gui._build_args(
            language="auto", model="auto", mode="full", diarize=True,
            speakers="auto", speaker_names="", meeting_name="",
            fmt="md", export_content="full", cleanup_model="llama3.1",
            summary_model="llama3.1", summary_language="same", output_dir="outputs",
            is_batch=False,
        )
        assert args.speakers is None

    def test_no_diarize_when_diarize_false(self):
        args = gui._build_args(
            language="auto", model="auto", mode="full", diarize=False,
            speakers="auto", speaker_names="", meeting_name="",
            fmt="md", export_content="full", cleanup_model="llama3.1",
            summary_model="llama3.1", summary_language="same", output_dir="outputs",
            is_batch=False,
        )
        assert args.no_diarize is True


class TestToggleSpeakerFields:
    def test_diarize_true_shows_fields(self):
        speakers_update, names_update = gui._toggle_speaker_fields(True)
        assert speakers_update["visible"] is True
        assert names_update["visible"] is True

    def test_diarize_false_hides_fields(self):
        speakers_update, names_update = gui._toggle_speaker_fields(False)
        assert speakers_update["visible"] is False
        assert names_update["visible"] is False


class TestRunPipelineValidation:
    def test_no_audio_files_raises(self):
        with pytest.raises(gr.Error):
            list(gui.run_pipeline(
                None, "auto", "auto", "full", True, "auto", "", "",
                "md", "full", "llama3.1", "llama3.1", "same", "outputs",
            ))

    def test_summary_only_with_transcript_mode_raises(self):
        with pytest.raises(gr.Error):
            list(gui.run_pipeline(
                ["fake.wav"], "auto", "auto", "transcript", True, "auto", "", "",
                "md", "summary", "llama3.1", "llama3.1", "same", "outputs",
            ))

    def test_missing_hf_token_with_diarize_raises(self, monkeypatch):
        monkeypatch.setattr(gui.config, "HF_TOKEN", "hf_XXXXXXXXXX")
        with pytest.raises(gr.Error):
            list(gui.run_pipeline(
                ["fake.wav"], "auto", "auto", "full", True, "auto", "", "",
                "md", "full", "llama3.1", "llama3.1", "same", "outputs",
            ))
