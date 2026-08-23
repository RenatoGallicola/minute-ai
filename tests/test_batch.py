
from src.batch import already_processed, collect_audio_files, run_batch


class TestCollectAudioFiles:
    def test_single_file(self, tmp_path):
        audio = tmp_path / "meeting.wav"
        audio.touch()
        assert collect_audio_files([str(audio)]) == [str(audio)]

    def test_unsupported_extension_is_skipped(self, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.touch()
        assert collect_audio_files([str(doc)]) == []

    def test_folder_collects_all_supported_files(self, tmp_path):
        (tmp_path / "a.wav").touch()
        (tmp_path / "b.mp3").touch()
        (tmp_path / "ignore.txt").touch()
        result = collect_audio_files([str(tmp_path)])
        assert len(result) == 2

    def test_missing_path_is_skipped(self):
        assert collect_audio_files(["does/not/exist.wav"]) == []

    def test_folder_with_glob_characters_in_its_name(self, tmp_path):
        # Regression: folder scanning used glob(), so a directory called
        # "notes [2026]" matched nothing at all.
        folder = tmp_path / "notes [2026]"
        folder.mkdir()
        (folder / "a.wav").touch()
        assert len(collect_audio_files([str(folder)])) == 1

    def test_subfolders_only_with_recursive(self, tmp_path):
        (tmp_path / "top.wav").touch()
        nested = tmp_path / "week-1"
        nested.mkdir()
        (nested / "deep.wav").touch()
        assert len(collect_audio_files([str(tmp_path)])) == 1
        assert len(collect_audio_files([str(tmp_path)], recursive=True)) == 2

    def test_deduplicates_same_file(self, tmp_path):
        audio = tmp_path / "meeting.wav"
        audio.touch()
        result = collect_audio_files([str(audio), str(audio)])
        assert len(result) == 1


class TestAlreadyProcessed:
    def test_false_when_output_dir_missing(self, tmp_path):
        missing_dir = tmp_path / "outputs"
        assert already_processed("meeting.wav", str(missing_dir), "md") is False

    def test_true_when_matching_md_exists(self, tmp_path):
        (tmp_path / "2026-01-01_10-00_Meeting.md").touch()
        assert already_processed("meeting.wav", str(tmp_path), "md") is True

    def test_false_when_no_match(self, tmp_path):
        (tmp_path / "2026-01-01_10-00_Other.md").touch()
        assert already_processed("meeting.wav", str(tmp_path), "md") is False

    def test_longer_name_is_not_a_match(self, tmp_path):
        # Regression: the check was a substring match, so "test.wav" counted as
        # already processed the moment an unrelated "Test_Two" export existed.
        (tmp_path / "2026-01-01_10-00_Test_Two.md").touch()
        assert already_processed("test.wav", str(tmp_path), "md") is False

    def test_numbered_duplicate_still_counts_as_processed(self, tmp_path):
        (tmp_path / "2026-01-01_10-00_Meeting (2).md").touch()
        assert already_processed("meeting.wav", str(tmp_path), "md") is True

    def test_a_plain_file_without_a_timestamp_is_not_a_match(self, tmp_path):
        (tmp_path / "Meeting.md").touch()
        assert already_processed("meeting.wav", str(tmp_path), "md") is False

    def test_docx_format_is_detected(self, tmp_path):
        # Regression test: already_processed used to ignore docx/pdf entirely,
        # so --format docx always reprocessed even with existing output.
        (tmp_path / "2026-01-01_10-00_Meeting.docx").touch()
        assert already_processed("meeting.wav", str(tmp_path), "docx") is True

    def test_pdf_format_is_detected(self, tmp_path):
        (tmp_path / "2026-01-01_10-00_Meeting.pdf").touch()
        assert already_processed("meeting.wav", str(tmp_path), "pdf") is True

    def test_docx_not_confused_with_missing_pdf(self, tmp_path):
        (tmp_path / "2026-01-01_10-00_Meeting.docx").touch()
        assert already_processed("meeting.wav", str(tmp_path), "pdf") is False

    def test_all_format_matches_any_known_extension(self, tmp_path):
        (tmp_path / "2026-01-01_10-00_Meeting.pdf").touch()
        assert already_processed("meeting.wav", str(tmp_path), "all") is True


class TestRunBatch:
    def test_skips_already_processed_unless_forced(self, tmp_path):
        done = tmp_path / "2026-01-01_10-00_Done.md"
        done.touch()
        audio_files = [str(tmp_path / "done.wav"), str(tmp_path / "new.wav")]
        calls = []

        results = run_batch(
            audio_files=audio_files,
            process_fn=lambda audio: calls.append(audio),
            parallel=False,
            force=False,
            output_dir=str(tmp_path),
            fmt="md",
        )

        assert len(calls) == 1
        assert results["skipped"] == [str(tmp_path / "done.wav")]
        assert results["success"] == [str(tmp_path / "new.wav")]

    def test_force_reprocesses_everything(self, tmp_path):
        done = tmp_path / "2026-01-01_10-00_Done.md"
        done.touch()
        audio_files = [str(tmp_path / "done.wav")]
        calls = []

        results = run_batch(
            audio_files=audio_files,
            process_fn=lambda audio: calls.append(audio),
            parallel=False,
            force=True,
            output_dir=str(tmp_path),
            fmt="md",
        )

        assert len(calls) == 1
        assert results["skipped"] == []
        assert results["success"] == audio_files

    def test_failed_file_does_not_abort_batch(self, tmp_path):
        audio_files = [str(tmp_path / "bad.wav"), str(tmp_path / "good.wav")]

        def process_fn(audio):
            if "bad" in audio:
                raise RuntimeError("boom")

        results = run_batch(
            audio_files=audio_files,
            process_fn=process_fn,
            parallel=False,
            force=True,
            output_dir=str(tmp_path),
            fmt="md",
        )

        assert [path for path, _ in results["failed"]] == [str(tmp_path / "bad.wav")]
        assert results["success"] == [str(tmp_path / "good.wav")]

    def test_parallel_mode_processes_all_files(self, tmp_path):
        audio_files = [str(tmp_path / f"f{i}.wav") for i in range(4)]

        results = run_batch(
            audio_files=audio_files,
            process_fn=lambda audio: None,
            parallel=True,
            force=True,
            output_dir=str(tmp_path),
            fmt="md",
            max_workers=2,
        )

        assert sorted(results["success"]) == sorted(audio_files)
        assert results["failed"] == []
