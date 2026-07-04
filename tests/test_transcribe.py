from src.transcribe import format_transcript, build_speaker_map


class TestBuildSpeakerMap:
    def test_maps_names_to_sorted_speaker_labels(self):
        # Speaker labels are sorted before zipping with names, regardless of
        # the order segments first appear in.
        segments = [
            {"speaker": "SPEAKER_01"},
            {"speaker": "SPEAKER_00"},
            {"speaker": "SPEAKER_01"},
        ]
        result = build_speaker_map(segments, "Bob,Alice")
        assert result == {"SPEAKER_00": "Bob", "SPEAKER_01": "Alice"}

    def test_no_names_returns_empty_dict(self):
        segments = [{"speaker": "SPEAKER_00"}]
        assert build_speaker_map(segments, None) == {}

    def test_more_speakers_than_names(self):
        segments = [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}]
        result = build_speaker_map(segments, "Alice")
        assert result == {"SPEAKER_00": "Alice"}


class TestFormatTranscript:
    def test_groups_consecutive_segments_by_speaker(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_00", "text": "there"},
            {"speaker": "SPEAKER_01", "text": "Hi"},
        ]
        result = format_transcript(segments, diarize=True)
        assert result == "SPEAKER_00: Hello there\n\nSPEAKER_01: Hi"

    def test_applies_speaker_names(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        result = format_transcript(segments, {"SPEAKER_00": "Alice"}, diarize=True)
        assert result == "Alice: Hello"

    def test_no_diarize_concatenates_without_labels(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello"},
            {"speaker": "SPEAKER_01", "text": "there"},
        ]
        result = format_transcript(segments, diarize=False)
        assert result == "Hello there"

    def test_empty_segments_are_skipped(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "  "},
            {"speaker": "SPEAKER_00", "text": "Hello"},
        ]
        result = format_transcript(segments, diarize=True)
        assert result == "SPEAKER_00: Hello"
