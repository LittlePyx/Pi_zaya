from __future__ import annotations

from api import reference_summary_text as summary_text


def test_clean_summary_line_removes_markup_citations_and_heading() -> None:
    raw = (
        "<jats:p>Abstract: We propose an adaptive imaging method [12] "
        "that improves reconstruction under low light.</jats:p>"
    )

    assert summary_text._clean_summary_line(raw) == (
        "We propose an adaptive imaging method that improves reconstruction under low light."
    )


def test_summary_excerpt_respects_sentence_and_length_limits() -> None:
    raw = (
        "We propose an adaptive imaging method for low-light reconstruction. "
        "Experiments show improved fidelity under limited measurements. "
        "The final sentence should not be included."
    )

    assert summary_text._summary_excerpt(raw, max_sentences=2, max_len=140) == (
        "We propose an adaptive imaging method for low-light reconstruction. "
        "Experiments show improved fidelity under limited measurements."
    )


def test_openalex_abstract_text_reconstructs_inverted_index() -> None:
    work = {
        "abstract_inverted_index": {
            "We": [0],
            "improve": [2],
            "imaging": [1],
            "quality.": [3],
        }
    }

    assert summary_text._openalex_abstract_text(work) == "We imaging improve quality."


def test_html_meta_and_jsonld_description_extraction() -> None:
    page = """
    <html>
      <head>
        <meta name="citation_abstract" content="A concise abstract from metadata.">
        <script type="application/ld+json">
          {"description": "A fallback JSON-LD description."}
        </script>
      </head>
    </html>
    """

    assert summary_text._html_meta_content(page, ("citation_abstract",)) == "A concise abstract from metadata."
    assert summary_text._jsonld_description_from_html(page) == "A fallback JSON-LD description."


def test_looks_like_title_echo_detects_title_only_summary() -> None:
    title = "Adaptive sampling for single pixel imaging reconstruction"

    assert summary_text._looks_like_title_echo(title, title) is True
    assert summary_text._looks_like_title_echo(
        "The paper develops a sampling policy and evaluates reconstruction quality under low light.",
        title,
    ) is False


def test_script_detection_helpers() -> None:
    assert summary_text._has_cjk_text("\u672c\u6587\u63d0\u51fa\u65b9\u6cd5") is True
    assert summary_text._has_latin_text("We propose a method") is True
