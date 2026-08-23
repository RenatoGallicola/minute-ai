from src import prompts


class TestResolveInstructions:
    def test_a_preset_returns_its_own_instructions(self):
        assert prompts.resolve_instructions("lecture") == prompts.LECTURE.instructions

    def test_default_when_the_preset_is_unknown(self):
        assert prompts.resolve_instructions("nonsense") == prompts.MEETING.instructions

    def test_custom_text_wins_over_the_preset(self):
        # A preset left at its default must never quietly override what the
        # user actually typed.
        assert prompts.resolve_instructions("meeting", "## Risks") == "## Risks"

    def test_custom_preset_uses_the_custom_text(self):
        assert prompts.resolve_instructions("custom", "## Risks") == "## Risks"

    def test_custom_preset_with_no_text_falls_back(self):
        assert prompts.resolve_instructions("custom", "   ") == prompts.MEETING.instructions

    def test_whitespace_is_trimmed(self):
        assert prompts.resolve_instructions("custom", "\n  ## Risks  \n") == "## Risks"


class TestFullTemplate:
    def test_transcript_placeholder_opts_into_full_control(self):
        assert prompts.is_full_template("Summarise this: {transcript}")

    def test_plain_instructions_are_not_a_full_template(self):
        assert not prompts.is_full_template("## Risks")

    def test_empty(self):
        assert not prompts.is_full_template("")


class TestFill:
    def test_substitutes_both_placeholders(self):
        out = prompts.fill("In {language}: {transcript}", "hello", "Italian")
        assert out == "In Italian: hello"

    def test_stray_braces_do_not_raise(self):
        # A hand-written template is not a format string; an unknown {thing}
        # must survive rather than blow up mid-run.
        out = prompts.fill("Keep {thing} as is: {transcript}", "hi", "English")
        assert out == "Keep {thing} as is: hi"

    def test_repeated_placeholders(self):
        assert prompts.fill("{language}/{language}", "x", "Greek") == "Greek/Greek"


class TestCatalogue:
    def test_every_preset_is_listed(self):
        assert {key for key, _, _ in prompts.options()} == set(prompts.PRESET_KEYS)

    def test_custom_is_last(self):
        assert prompts.options()[-1][0] == prompts.CUSTOM_KEY

    def test_presets_use_markdown_headings(self):
        for preset in prompts.PRESETS.values():
            assert preset.instructions.startswith("## "), preset.key

    def test_default_preset_exists(self):
        assert prompts.DEFAULT_PRESET in prompts.PRESETS
