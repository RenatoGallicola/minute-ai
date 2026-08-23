from src.markdown_lite import render, render_transcript


class TestRender:
    def test_empty_input(self):
        assert render("") == ""

    def test_headings_become_h2(self):
        assert render("## Decisions Made") == "<h2>Decisions Made</h2>"

    def test_bullets_become_a_list(self):
        html = render("- one\n- two")
        assert html == "<ul>\n<li>one</li>\n<li>two</li>\n</ul>"

    def test_numbered_lists(self):
        assert "<ol>" in render("1. first\n2. second")

    def test_bold_and_italic(self):
        assert render("**bold** and *soft*") == "<p><strong>bold</strong> and <em>soft</em></p>"

    def test_horizontal_rule(self):
        assert render("---") == "<hr>"

    def test_html_in_the_model_output_is_escaped(self):
        # The text comes from a local LLM and is injected with |safe, so
        # everything must be escaped before markup is re-introduced.
        html = render("<script>alert(1)</script>")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_ampersand_is_escaped(self):
        assert "R&amp;D" in render("R&D budget")

    def test_list_is_closed_before_a_heading(self):
        html = render("- one\n## Next")
        assert html.index("</ul>") < html.index("<h2>")


class TestRenderTranscript:
    def test_speaker_line_is_split(self):
        html = render_transcript("Alice: Hello there.")
        assert '<span class="who">Alice</span>' in html
        assert '<span class="said">Hello there.</span>' in html

    def test_line_without_a_label_has_no_speaker(self):
        html = render_transcript("Just some narration without a label.")
        assert 'class="who"' not in html

    def test_a_long_prefix_is_not_treated_as_a_speaker(self):
        long_prefix = "x" * 80
        assert 'class="who"' not in render_transcript(f"{long_prefix}: tail")

    def test_escapes_markup(self):
        assert "&lt;b&gt;" in render_transcript("Alice: <b>hi</b>")
