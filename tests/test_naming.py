from datetime import datetime
from pathlib import Path

from src.naming import (
    meeting_name_from_path,
    output_stem,
    output_stem_pattern,
    slugify,
    unique_path,
)


class TestSlugify:
    def test_spaces_become_underscores(self):
        assert slugify("Q3 Kickoff") == "Q3_Kickoff"

    def test_windows_illegal_characters_are_removed(self):
        # Regression: a ':' survived into the file name, and on NTFS open()
        # then wrote to an alternate data stream — the .md never appeared in
        # Explorer and the content was effectively lost.
        assert ":" not in slugify("Q3: Kickoff")
        assert slugify('a<b>c:d"e/f\\g|h?i*j') == "abcdefghij"

    def test_control_characters_are_removed(self):
        assert slugify("bad\nname\ttab") == "bad_name_tab"

    def test_reserved_device_names_are_escaped(self):
        assert slugify("CON") == "meeting_CON"

    def test_never_returns_an_empty_string(self):
        assert slugify("///") == "meeting"
        assert slugify("") == "meeting"

    def test_is_length_capped(self):
        assert len(slugify("x" * 300)) <= 80


class TestMeetingNameFromPath:
    def test_underscores_and_dashes_become_words(self):
        assert meeting_name_from_path("inputs/team_sync-notes.wav") == "Team Sync Notes"

    def test_blank_stem_falls_back(self):
        assert meeting_name_from_path("inputs/___.wav") == "Meeting"


class TestOutputStem:
    def test_shape_is_timestamp_then_slug(self):
        stem = output_stem("Q3 Kickoff", datetime(2026, 8, 23, 14, 5))
        assert stem == "2026-08-23_14-05_Q3_Kickoff"

    def test_pattern_matches_its_own_output(self):
        stem = output_stem("Q3 Kickoff", datetime(2026, 8, 23, 14, 5))
        assert output_stem_pattern("Q3 Kickoff").match(stem)

    def test_pattern_is_anchored_at_both_ends(self):
        pattern = output_stem_pattern("Test")
        assert pattern.match("2026-01-01_10-00_Test")
        assert not pattern.match("2026-01-01_10-00_Test_Two")
        assert not pattern.match("Test")

    def test_pattern_tolerates_the_duplicate_suffix(self):
        assert output_stem_pattern("Test").match("2026-01-01_10-00_Test (2)")


class TestUniquePath:
    def test_free_path_is_returned_as_is(self, tmp_path):
        target = tmp_path / "notes.md"
        assert unique_path(str(target)) == str(target)

    def test_existing_path_gets_a_counter(self, tmp_path):
        target = tmp_path / "notes.md"
        target.touch()
        assert Path(unique_path(str(target))).name == "notes (2).md"

    def test_counter_keeps_climbing(self, tmp_path):
        (tmp_path / "notes.md").touch()
        (tmp_path / "notes (2).md").touch()
        assert Path(unique_path(str(tmp_path / "notes.md"))).name == "notes (3).md"
