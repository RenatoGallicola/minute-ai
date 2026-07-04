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
