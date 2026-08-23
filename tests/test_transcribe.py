from src.transcribe import build_speaker_map, format_duration, format_transcript, resolve_compute_type


class TestBuildSpeakerMap:
    def test_maps_names_in_order_of_first_appearance(self):
        # Regression: the labels used to be sorted before zipping with the
        # names, so whenever pyannote did not hand SPEAKER_00 to whoever spoke
        # first, every name landed on the wrong person.
        segments = [
            {"speaker": "SPEAKER_01"},
            {"speaker": "SPEAKER_00"},
            {"speaker": "SPEAKER_01"},
        ]
        result = build_speaker_map(segments, "Bob,Alice")
        assert result == {"SPEAKER_01": "Bob", "SPEAKER_00": "Alice"}

    def test_blank_names_are_ignored(self):
        segments = [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}]
        assert build_speaker_map(segments, "Alice, ,Bob") == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}

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


class TestResolveComputeType:
    def test_auto_picks_int8_on_cpu(self):
        assert resolve_compute_type("auto", "cpu") == "int8"

    def test_auto_picks_float16_on_gpu(self):
        assert resolve_compute_type("auto", "cuda") == "float16"

    def test_float16_downgrades_on_cpu(self):
        # CTranslate2 cannot run float16 on CPU; it would warn and fall back
        # anyway, so resolve it up front instead.
        assert resolve_compute_type("float16", "cpu") == "int8"

    def test_explicit_value_is_kept(self):
        assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"

    def test_missing_value_behaves_like_auto(self):
        assert resolve_compute_type("", "cpu") == "int8"


class TestFormatDuration:
    def test_minutes_and_seconds(self):
        assert format_duration(252) == "4m 12s"

    def test_hours(self):
        assert format_duration(3900) == "1h 05m"


class TestTranscriptParagraphs:
    def _segs(self, *spans):
        # (start, end, text, speaker)
        return [{"start": s, "end": e, "text": t, "speaker": sp} for s, e, t, sp in spans]

    def test_a_long_pause_starts_a_new_paragraph(self):
        # Regression: a single-speaker recording (a lecture, a voice memo) came
        # out as one unbroken wall of text, because paragraphs were only ever
        # broken on a change of speaker.
        segments = self._segs(
            (0.0, 2.0, "First thought.", "SPEAKER_00"),
            (2.5, 4.0, "Still the same one.", "SPEAKER_00"),
            (9.0, 11.0, "New thought after a pause.", "SPEAKER_00"),
        )
        result = format_transcript(segments, diarize=True)
        assert result == (
            "SPEAKER_00: First thought. Still the same one.\n\n"
            "New thought after a pause."
        )

    def test_only_the_first_paragraph_of_a_turn_is_labelled(self):
        segments = self._segs(
            (0.0, 1.0, "One.", "SPEAKER_00"),
            (8.0, 9.0, "Two.", "SPEAKER_00"),
        )
        assert format_transcript(segments, diarize=True).count("SPEAKER_00:") == 1

    def test_a_change_of_speaker_always_relabels(self):
        segments = self._segs(
            (0.0, 1.0, "Hello", "SPEAKER_00"),
            (1.0, 2.0, "Hi", "SPEAKER_01"),
            (2.0, 3.0, "Bye", "SPEAKER_00"),
        )
        result = format_transcript(segments, diarize=True)
        assert result == "SPEAKER_00: Hello\n\nSPEAKER_01: Hi\n\nSPEAKER_00: Bye"

    def test_an_overlong_paragraph_is_broken_even_without_a_pause(self):
        segments = self._segs(*[(i * 0.5, i * 0.5 + 0.5, "word " * 20, "SPEAKER_00") for i in range(12)])
        result = format_transcript(segments, diarize=True, max_chars=300)
        assert len(result.split("\n\n")) > 1

    def test_gap_breaking_can_be_switched_off(self):
        segments = self._segs(
            (0.0, 1.0, "One.", "SPEAKER_00"),
            (60.0, 61.0, "Two.", "SPEAKER_00"),
        )
        assert format_transcript(segments, diarize=True, gap_seconds=0, max_chars=0) == "SPEAKER_00: One. Two."

    def test_undiarized_audio_is_still_paragraphed(self):
        segments = self._segs(
            (0.0, 2.0, "First thought.", None),
            (9.0, 11.0, "Second thought.", None),
        )
        result = format_transcript(segments, diarize=False)
        assert result == "First thought.\n\nSecond thought."
        assert ":" not in result

    def test_segments_without_timings_do_not_crash(self):
        segments = [{"text": "Hello", "speaker": "SPEAKER_00"}, {"text": "there", "speaker": "SPEAKER_00"}]
        assert format_transcript(segments, diarize=True) == "SPEAKER_00: Hello there"


class TestSpeakerNameCountMismatch:
    def _segments(self, *speakers):
        return [{"speaker": s, "text": "hi"} for s in speakers]

    def test_extra_names_are_reported(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="minute-ai"):
            result = build_speaker_map(self._segments("SPEAKER_00"), "Ana,Ben,Cleo")
        assert result == {"SPEAKER_00": "Ana"}
        assert any("Ben, Cleo" in r.message for r in caplog.records)

    def test_missing_names_are_reported(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="minute-ai"):
            result = build_speaker_map(self._segments("SPEAKER_00", "SPEAKER_01"), "Ana")
        assert result == {"SPEAKER_00": "Ana"}
        assert any("Only 1 speaker name" in r.message for r in caplog.records)

    def test_matching_counts_say_nothing(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="minute-ai"):
            build_speaker_map(self._segments("SPEAKER_00", "SPEAKER_01"), "Ana,Ben")
        assert [r for r in caplog.records if r.levelname == "WARNING"] == []


class TestSingleSpeakerSkipsDiarization:
    """`--speakers 1` should not pay for pyannote."""

    def _run(self, monkeypatch, num_speakers, diarize=True):
        import sys, types
        from src import transcribe as tr

        calls = {"diarize": 0}
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        fake_whisperx = types.ModuleType("whisperx")
        fake_whisperx.load_audio = lambda path: [0.0] * 16000
        fake_whisperx.load_model = lambda *a, **k: types.SimpleNamespace(
            transcribe=lambda audio, batch_size=8: {"language": "en", "segments": segments}
        )
        fake_whisperx.load_align_model = lambda **k: (None, None)
        fake_whisperx.align = lambda segs, *a, **k: {"segments": segs}

        def fake_pipeline(*a, **k):
            calls["diarize"] += 1
            return lambda audio, **kw: None

        fake_whisperx.diarize = types.SimpleNamespace(
            DiarizationPipeline=fake_pipeline,
            assign_word_speakers=lambda d, r: r,
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        result = tr.transcribe(
            audio_path="a.wav", language="en", num_speakers=num_speakers,
            model_name="small", compute_type="int8", hf_token="hf_x", diarize=diarize,
        )
        return result, calls

    def test_one_speaker_does_not_load_pyannote(self, monkeypatch):
        result, calls = self._run(monkeypatch, num_speakers=1)
        assert calls["diarize"] == 0
        # Still reported as diarized: we know exactly who is speaking.
        assert result.diarized is True
        assert result.speakers == ["SPEAKER_00"]

    def test_two_speakers_still_runs_pyannote(self, monkeypatch):
        _, calls = self._run(monkeypatch, num_speakers=2)
        assert calls["diarize"] == 1

    def test_auto_still_runs_pyannote(self, monkeypatch):
        _, calls = self._run(monkeypatch, num_speakers=None)
        assert calls["diarize"] == 1

    def test_one_speaker_name_is_applied(self, monkeypatch):
        result, _ = self._run(monkeypatch, num_speakers=1)
        transcript = format_transcript(result.segments, build_speaker_map(result.segments, "Marco"))
        assert transcript == "Marco: Hello"
