"""Keeps the CLI and the web GUI offering the same thing.

The two front ends drifted apart once already (the GUI listed 12 languages
while the CLI accepted all 100, and it had no parallel option at all). These
tests fail when a new option lands on one side only.
"""

import inspect
import re

import gui
import main
from src import languages


def cli_flags() -> set[str]:
    """Every --long-option main.parse_args defines."""
    return set(re.findall(r'"--([a-z0-9-]+)"', inspect.getsource(main.parse_args)))


def gui_form_fields() -> set[str]:
    """Every form field the /run endpoint accepts."""
    return {name for name in inspect.signature(gui.run).parameters if name != "request"}


# CLI options the GUI deliberately does not have, and why. Anything not listed
# here must exist on both sides.
GUI_EXEMPT = {
    # The GUI works on browser uploads, not server-side paths, so there is no
    # folder to descend into.
    "recursive",
    # Uploading a file *is* the request to process it; there is nothing to skip.
    # The GUI always behaves as if --force were set.
    "force",
    # Writing anywhere on the server from a web form is a footgun; the GUI uses
    # config.DEFAULT_OUTPUT_DIR and hands the files back as downloads.
    "output-dir",
}

# Form fields with no direct CLI twin, and why.
CLI_EXEMPT = {
    "audio_files",      # the CLI takes paths as positional arguments
    "diarize",          # the inverse of --no-diarize
    "speakers_custom",  # part of the speakers control; --speakers takes any int
    "parallel_workers", # covers both --parallel and --parallel-workers
}


class TestOptionParity:
    def test_every_cli_flag_is_reachable_from_the_gui(self):
        aliases = {"no-diarize": "diarize", "parallel": "parallel_workers",
                   "parallel-workers": "parallel_workers"}
        fields = gui_form_fields()
        missing = []
        for flag in cli_flags() - GUI_EXEMPT:
            candidate = aliases.get(flag, flag.replace("-", "_"))
            if candidate not in fields:
                missing.append(flag)
        assert not missing, f"CLI options with no GUI equivalent: {missing}"

    def test_every_gui_field_is_reachable_from_the_cli(self):
        flags = {f.replace("-", "_") for f in cli_flags()}
        extra = [f for f in gui_form_fields() - CLI_EXEMPT if f not in flags]
        assert not extra, f"GUI fields with no CLI equivalent: {extra}"

    def test_exemptions_still_refer_to_real_options(self):
        # Stops the allowlist rotting into a lie after a rename.
        assert GUI_EXEMPT <= cli_flags()
        assert CLI_EXEMPT <= gui_form_fields()


class TestChoiceParity:
    def _values(self, options):
        return {entry[0] for entry in options}

    def test_modes_match(self):
        assert self._values(gui.MODES) == set(main.MODE_CHOICES)

    def test_formats_match(self):
        assert self._values(gui.FORMATS) == set(main.FORMAT_CHOICES)

    def test_whisper_models_match(self):
        assert self._values(gui.MODELS) == set(main.MODEL_CHOICES)

    def test_export_contents_match(self):
        assert self._values(gui.EXPORT_CONTENTS) == {"full", "summary"}

    def test_gui_offers_every_language_the_cli_accepts(self):
        # Regression: the dropdown had 12 hand-picked languages, so Turkish
        # audio could be transcribed from the terminal but not the browser.
        offered = self._values(gui.LANGUAGES) - {"auto"}
        assert offered == set(languages.LANGUAGE_NAMES)

    def test_summary_languages_match_too(self):
        offered = self._values(gui.SUMMARY_LANGUAGES) - {"same"}
        assert offered == set(languages.LANGUAGE_NAMES)


class TestValidationParity:
    """The same combination must be refused by both front ends."""

    def _cli_rejects(self, **overrides):
        import argparse

        import pytest

        defaults = dict(
            audio=["a.wav"], language="auto", speakers="auto", speaker_names=None,
            model="medium", no_diarize=False, meeting_name=None, mode="full",
            cleanup_model="m", summary_model="m", summary_language="same",
            output_dir="outputs", format="md", export_content="full",
            parallel=False, parallel_workers=2, force=False, recursive=False,
        )
        defaults.update(overrides)
        try:
            main.validate_args(argparse.Namespace(**defaults))
            return False
        except SystemExit:
            return True

    def test_summary_only_with_a_transcript_mode(self):
        assert self._cli_rejects(mode="transcript", export_content="summary")
        assert gui._validate_submission(False, "transcript", "summary", "md", "auto")

    def test_srt_cannot_carry_a_summary_alone(self):
        assert self._cli_rejects(format="srt", export_content="summary")
        assert gui._validate_submission(False, "full", "summary", "srt", "auto")

    def test_zero_speakers(self):
        assert self._cli_rejects(speakers="0")
        assert gui._validate_submission(True, "full", "full", "md", "0")

    def test_unknown_language(self):
        assert self._cli_rejects(language="xx")
        assert gui._validate_submission(False, "full", "full", "md", "auto", language="xx")

    def test_a_valid_combination_passes_both(self):
        assert not self._cli_rejects(language="tr", speakers="3", no_diarize=False)
        assert gui._validate_submission(False, "full", "full", "md", "3", language="tr") is None
