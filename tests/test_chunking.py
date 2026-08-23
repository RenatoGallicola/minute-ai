from src.chunking import split_blocks


class TestSplitBlocks:
    def test_short_text_is_one_chunk(self):
        assert split_blocks("Alice: hi", 100) == ["Alice: hi"]

    def test_empty_text_yields_nothing(self):
        assert split_blocks("   ", 100) == []

    def test_splits_on_blank_line_boundaries(self):
        text = "\n\n".join(["Alice: " + "a" * 40, "Bob: " + "b" * 40, "Alice: " + "c" * 40])
        chunks = split_blocks(text, 100)
        assert len(chunks) > 1
        assert all(len(c) <= 110 for c in chunks)
        assert all(c.startswith(("Alice:", "Bob:")) for c in chunks)

    def test_no_content_is_lost(self):
        blocks = [f"S{i}: " + "word " * 30 for i in range(12)]
        text = "\n\n".join(blocks)
        rejoined = "\n\n".join(split_blocks(text, 500))
        for block in blocks:
            assert block.strip() in rejoined

    def test_single_oversized_block_is_split_on_sentences(self):
        block = "Alice: " + "This is a sentence. " * 40
        chunks = split_blocks(block, 200)
        assert len(chunks) > 1
        assert all(len(c) <= 210 for c in chunks)

    def test_a_block_with_no_spaces_is_still_split(self):
        chunks = split_blocks("x" * 500, 100)
        assert len(chunks) == 5

    def test_non_positive_limit_returns_the_whole_text(self):
        assert split_blocks("hello", 0) == ["hello"]
