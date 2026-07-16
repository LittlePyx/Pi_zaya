from kb.chunking import _semantic_overlap_tail, chunk_markdown


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


def test_overlap_tail_does_not_start_mid_word():
    text = (
        "Compressed sensing can recover signals from limited measurements. "
        "A person can be described uniquely with a few targeted questions, "
        "which is closely related to sparsity in imaging systems."
    )

    tail = _semantic_overlap_tail(text, overlap=75)

    assert not tail.startswith("rson")
    assert tail.startswith("questions") or tail.startswith("which") or tail.startswith("A person")


def test_chunk_overlap_prefers_sentence_boundary():
    first = (
        "Single-pixel imaging uses structured illumination to acquire measurements. "
        "The overlap region should start from a complete sentence for evidence cards. "
        "This sentence carries the retrieval context without a broken leading word."
    )
    second = "The next paragraph should be appended after the semantic overlap."
    md = f"# Intro\n\n{first}\n\n{second}"

    chunks = chunk_markdown(md, source_path="test.md", chunk_size=260, overlap=115)

    assert len(chunks) >= 2
    assert chunks[1]["text"].startswith("This sentence carries")


def test_equation_image_retry_marker_becomes_safe_searchable_locator():
    md = """# Method

<!-- kb_page: 4 -->

![Equation](./assets/page_4_eq_1.png)
<!-- kb:conversion_retry kind=equation page=4 asset=page_4_eq_1.png number=3 -->
<!-- kb:conversion_retry kind=math_text page=4 -->
"""

    chunks = chunk_markdown(md, source_path="paper.md", overlap=0)

    assert len(chunks) == 1
    assert "Equation (3) is preserved as a source image on page 4" in chunks[0]["text"]
    assert "kb:conversion_retry" not in chunks[0]["text"]
    assert chunks[0]["meta"]["conversion_fallback_kinds"] == ["equation_image"]
    assert chunks[0]["meta"]["equation_numbers"] == [3]
    assert chunks[0]["meta"]["equation_assets"] == ["page_4_eq_1.png"]
