from __future__ import annotations

from api import reference_summary_fallbacks as fallbacks


def test_metadata_summary_line_names_author_and_publication_context() -> None:
    out = fallbacks._metadata_summary_line(
        {
            "title": "Adaptive sampling for single-pixel imaging",
            "authors": "Gehm M, Brady D, Extended Collaborator Name",
            "venue": "Optics Express Journal of Imaging",
            "year": "2007",
        }
    )

    assert "Gehm M" in out
    assert "Optics Express Journal of Imaging" in out
    assert "2007" in out
    assert "缺少可用摘要文本" in out


def test_metadata_summary_line_handles_sparse_metadata() -> None:
    out = fallbacks._metadata_summary_line({})

    assert "当前仅检索到有限元数据" in out
    assert "建议通过 DOI 查看原文摘要与正文" in out


def test_contextual_summary_line_uses_claim_location_and_context() -> None:
    out = fallbacks._contextual_summary_line(
        {
            "citation_context": (
                "The current paper cites this method when explaining how adaptive sampling "
                "reduces measurements in single-pixel imaging."
            ),
            "answer_claim": "Adaptive sampling can reduce the measurement budget.",
            "location_label": "Introduction / prior work",
        }
    )

    assert "Adaptive sampling can reduce the measurement budget" in out
    assert "Introduction / prior work" in out
    assert "adaptive sampling reduces measurements" in out


def test_contextual_summary_line_returns_empty_without_context() -> None:
    assert fallbacks._contextual_summary_line({"answer_claim": "No context"}) == ""
