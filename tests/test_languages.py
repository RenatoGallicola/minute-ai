from src import languages


class TestSupport:
    def test_auto_is_supported(self):
        assert languages.is_supported("auto")

    def test_known_codes(self):
        for code in ("it", "en", "tr", "yue", "haw"):
            assert languages.is_supported(code), code

    def test_case_and_padding_do_not_matter(self):
        assert languages.is_supported("  IT ")

    def test_unknown_code(self):
        assert not languages.is_supported("xx")

    def test_empty(self):
        assert not languages.is_supported("")
        assert not languages.is_supported(None)


class TestNames:
    def test_known_code(self):
        assert languages.name_for("tr") == "Turkish"

    def test_unknown_code_is_returned_as_is(self):
        assert languages.name_for("zz") == "zz"


class TestOptions:
    def test_audio_starts_with_auto_then_the_common_ones(self):
        options = languages.audio_options()
        assert options[0] == ("auto", "Auto-detect")
        assert [code for code, _ in options[1:7]] == languages.COMMON

    def test_every_language_is_offered_exactly_once(self):
        codes = [code for code, _ in languages.audio_options()[1:]]
        assert len(codes) == len(set(codes)) == len(languages.LANGUAGE_NAMES)

    def test_the_tail_is_alphabetical_by_name(self):
        names = [name for _, name in languages.audio_options()[7:]]
        assert names == sorted(names)

    def test_summary_options_swap_auto_for_same(self):
        options = languages.summary_options()
        assert options[0] == ("same", "Same as the audio")
        assert "auto" not in {code for code, _ in options}
