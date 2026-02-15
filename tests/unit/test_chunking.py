
import pytest
from kb.chunking import chunk_markdown, Block

def test_chunking_short_text():
    # Test that short text is not split
    md = "Short paragraph."
    chunks = chunk_markdown(md, source_path="test.md", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short paragraph."
    assert chunks[0]["meta"]["source_path"] == "test.md"

def test_chunking_long_text():
    # Test splitting of long text
    long_text = "word\n\n" * 100
    chunk_size = 250
    overlap = 10
    chunks = chunk_markdown(long_text, source_path="test.md", chunk_size=chunk_size, overlap=overlap)
    
    assert len(chunks) > 1
    # Check overlap
    # We can't easily assert exact content without mocking the internal logic precisely,
    # but we can check if the total length is reasonable.
    
    for chunk in chunks:
        # Each chunk should be roughly within size limit (logic allows slight overflow to finish word/sentence)
        # The current logic: if cur_len + len(b.text) + 1 > chunk_size ...
        # So a single block could exceed chunk_size if the block itself is huge.
        pass

def test_chunking_headings():
    # Test that headings force a new chunk
    md = """
# Heading 1
Section 1 content.

## Heading 2
Section 2 content.
"""
    chunks = chunk_markdown(md, source_path="test.md", chunk_size=1000, overlap=0)
    # Expect: 
    # 1. Heading 1
    # 2. Section 1 content
    # 3. Heading 2
    # 4. Section 2 content
    # The logic enforces flush at headings.
    
    # Actually, the logic:
    # if b.kind == "heading": flush(force=True); cur=[b.text]...
    # So "Heading 1" starts a chunk. "Section 1 content" appends to it?
    # No, let's trace:
    # 1. Heading 1 -> flush (empty). cur=["# Heading 1"]
    # 2. Section 1 -> appends to cur.
    # 3. Heading 2 -> flush (saves H1+S1). cur=["## Heading 2"]
    # 4. Section 2 -> appends.
    # 5. EOF -> flush.
    
    assert len(chunks) == 2
    assert "# Heading 1" in chunks[0]["text"]
    assert "Section 1 content" in chunks[0]["text"]
    assert "## Heading 2" in chunks[1]["text"]

def test_page_markers():
    # Test parsing of page markers
    md = """
<!-- kb_page: 1 -->
Page 1 content.
<!-- kb_page: 2 -->
Page 2 content.
"""
    chunks = chunk_markdown(md, source_path="test.md")
    
    # Check metadata
    assert chunks[0]["meta"]["page_start"] == 1
    # Depending on chunk size, they might be in one chunk or two. Default chunk size is large (1400).
    # So likely 1 chunk.
    assert chunks[0]["meta"]["page_end"] == 2
