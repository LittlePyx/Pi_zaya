from __future__ import annotations

from kb.citation_context import extract_inpaper_reference_context


def test_extract_inpaper_reference_context_skips_reference_list(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "<!-- kb_page: 2 -->",
                "## 2. Related Work",
                "Most existing methods employ ADMM [4] for optimization in compressive imaging.",
                "",
                "## References",
                "[4] Boyd S. Distributed Optimization and Statistical Learning via ADMM. 2011.",
            ]
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(
        str(md),
        4,
        answer_context="ADMM is prior optimization machinery.",
    )

    assert out["citation_context_source"] == "source_markdown"
    assert "Most existing methods employ ADMM [4]" in out["citation_context"]
    assert "Boyd S" not in out["citation_context"]
    assert out["heading_path"].endswith("2. Related Work")
    assert out["page_start"] == 2
    assert "p. 2" in out["location_label"]


def test_extract_inpaper_reference_context_expands_ranges(tmp_path) -> None:
    md = tmp_path / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Example Paper",
                "## Method",
                "Deep reconstruction networks follow earlier compressive sensing systems [11-13].",
                "",
                "## References",
                "[12] Example reference entry that must not be selected.",
            ]
        ),
        encoding="utf-8",
    )

    out = extract_inpaper_reference_context(str(md), 12)

    assert out["citation_context_source"] == "source_markdown"
    assert "earlier compressive sensing systems [11-13]" in out["citation_context"]
    assert "Example reference entry" not in out["citation_context"]
    assert out["heading_path"].endswith("Method")
