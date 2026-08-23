from pathlib import Path

from src.export import export


def _run_export(tmp_path, fmt, export_content="full", summary="## Participants\n- Alice\n"):
    return export(
        audio_path="meeting.wav",
        meeting_name="Test Meeting",
        transcript="Alice: Hello there.\n\nBob: Hi Alice.",
        summary=summary,
        language="en",
        output_dir=str(tmp_path),
        fmt=fmt,
        export_content=export_content,
    )


class TestMarkdownExport:
    def test_creates_md_file(self, tmp_path):
        files = _run_export(tmp_path, "md")
        assert len(files) == 1
        assert files[0].endswith(".md")
        assert Path(files[0]).exists()

    def test_full_content_includes_summary_and_transcript(self, tmp_path):
        files = _run_export(tmp_path, "md", export_content="full")
        content = Path(files[0]).read_text(encoding="utf-8")
        assert "Participants" in content
        assert "Full Transcript" in content
        assert "Alice: Hello there." in content

    def test_summary_only_excludes_transcript(self, tmp_path):
        files = _run_export(tmp_path, "md", export_content="summary")
        content = Path(files[0]).read_text(encoding="utf-8")
        assert "Participants" in content
        assert "Full Transcript" not in content

    def test_missing_summary_is_omitted_even_in_full_mode(self, tmp_path):
        files = _run_export(tmp_path, "md", export_content="full", summary="")
        content = Path(files[0]).read_text(encoding="utf-8")
        assert "Full Transcript" in content
        assert "Participants" not in content


class TestTxtExport:
    def test_creates_txt_file_with_expected_sections(self, tmp_path):
        files = _run_export(tmp_path, "txt")
        content = Path(files[0]).read_text(encoding="utf-8")
        assert "SUMMARY" in content
        assert "TRANSCRIPT" in content
        assert "Alice: Hello there." in content


class TestDocxExport:
    def test_creates_docx_file(self, tmp_path):
        files = _run_export(tmp_path, "docx")
        assert len(files) == 1
        assert Path(files[0]).exists()
        assert Path(files[0]).stat().st_size > 0


class TestPdfExport:
    def test_creates_pdf_file(self, tmp_path):
        files = _run_export(tmp_path, "pdf")
        assert len(files) == 1
        assert Path(files[0]).exists()
        assert Path(files[0]).stat().st_size > 0


class TestAllFormats:
    def test_all_creates_four_files(self, tmp_path):
        files = _run_export(tmp_path, "all")
        extensions = sorted(Path(f).suffix for f in files)
        assert extensions == [".docx", ".md", ".pdf", ".txt"]
        assert all(Path(f).exists() for f in files)


class TestPdfExport:
    def test_creates_pdf_file(self, tmp_path):
        files = _run_export(tmp_path, "pdf")
        assert len(files) == 1
        assert Path(files[0]).stat().st_size > 0

    def test_ampersands_and_angle_brackets_do_not_break_the_pdf(self, tmp_path):
        # Regression: reportlab parses Paragraph text as mini-HTML, so a bare
        # '&' (as in "R&D") aborted the whole PDF export with a parse error.
        files = export(
            audio_path="meeting.wav",
            meeting_name="R&D <Q3> Review",
            transcript="Alice: We discussed R&D and 5 < 7.\n\nBob: Deal & done.",
            summary="## Topics\n- R&D budget <draft>",
            language="en",
            output_dir=str(tmp_path),
            fmt="pdf",
            export_content="full",
        )
        assert Path(files[0]).exists()
        assert Path(files[0]).stat().st_size > 0


class TestSrtExport:
    def _segments(self):
        return [
            {"start": 0.0, "end": 2.5, "text": "Hello there.", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 4.25, "text": "Hi Alice.", "speaker": "SPEAKER_01"},
            {"start": 4.25, "end": 5.0, "text": "   "},
        ]

    def test_writes_numbered_timestamped_blocks(self, tmp_path):
        files = export(
            audio_path="meeting.wav", meeting_name="Test", transcript="x", summary="",
            language="en", output_dir=str(tmp_path), fmt="srt", export_content="full",
            segments=self._segments(),
        )
        content = Path(files[0]).read_text(encoding="utf-8")
        assert content.startswith("1\n00:00:00,000 --> 00:00:02,500\n[SPEAKER_00] Hello there.")
        assert "2\n00:00:02,500 --> 00:00:04,250" in content
        assert "3\n" not in content  # the blank segment is dropped

    def test_skipped_when_there_are_no_segments(self, tmp_path):
        files = export(
            audio_path="meeting.wav", meeting_name="Test", transcript="x", summary="",
            language="en", output_dir=str(tmp_path), fmt="srt", export_content="full",
            segments=None,
        )
        assert files == []

    def test_all_format_includes_srt_when_segments_exist(self, tmp_path):
        files = export(
            audio_path="meeting.wav", meeting_name="Test", transcript="Alice: hi", summary="",
            language="en", output_dir=str(tmp_path), fmt="all", export_content="full",
            segments=self._segments(),
        )
        assert sorted(Path(f).suffix for f in files) == [".docx", ".md", ".pdf", ".srt", ".txt"]


class TestFileNames:
    def test_illegal_characters_never_reach_the_filesystem(self, tmp_path):
        files = export(
            audio_path="meeting.wav", meeting_name="Q3: Kickoff", transcript="Alice: hi",
            summary="", language="en", output_dir=str(tmp_path), fmt="md", export_content="full",
        )
        name = Path(files[0]).name
        assert ":" not in name
        # The file really is on disk under the name we reported.
        assert name in [p.name for p in tmp_path.iterdir()]

    def test_a_second_export_does_not_overwrite_the_first(self, tmp_path):
        first = _run_export(tmp_path, "md")
        second = _run_export(tmp_path, "md")
        assert first[0] != second[0]
        assert Path(first[0]).exists() and Path(second[0]).exists()


class TestExportContent:
    def test_summary_only_without_a_summary_still_writes_a_file(self, tmp_path):
        files = _run_export(tmp_path, "md", export_content="summary", summary="")
        assert Path(files[0]).exists()
        assert "Full Transcript" not in Path(files[0]).read_text(encoding="utf-8")
