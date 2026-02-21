from kb.chunking import chunk_markdown


def test_chunking_short_text():
    md = "Short paragraph."
    chunks = chunk_markdown(md, source_path="test.md", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short paragraph."
    assert chunks[0]["meta"]["source_path"] == "test.md"


def test_chunking_headings():
    md = """
# Heading 1
Section 1 content.

## Heading 2
Section 2 content.
"""
    chunks = chunk_markdown(md, source_path="test.md", chunk_size=1000, overlap=0)
    assert len(chunks) == 2
    assert "# Heading 1" in chunks[0]["text"]
    assert "Section 1 content" in chunks[0]["text"]
    assert "## Heading 2" in chunks[1]["text"]


def test_page_markers_split_blocks():
    md = """
<!-- kb_page: 1 -->
Page 1 content.
<!-- kb_page: 2 -->
Page 2 content.
"""
    chunks = chunk_markdown(md, source_path="test.md")
    assert chunks[0]["meta"]["page_start"] == 1
    assert chunks[0]["meta"]["page_end"] == 2

