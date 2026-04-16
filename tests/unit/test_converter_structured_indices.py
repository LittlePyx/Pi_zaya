from pathlib import Path

import json

from kb.converter.structured_indices import rebuild_structured_indices_for_markdown


def test_rebuild_structured_indices_emits_anchor_equation_reference_indices(tmp_path: Path):
    md = """# Demo Paper

## 1. Method

We define the objective below.

$$
E = mc^2
\\tag{1}
$$

## References

[1] A. Author. Demo reference entry. 2020.
"""
    md_path = tmp_path / "output.md"
    assets_dir = tmp_path / "assets"

    out = rebuild_structured_indices_for_markdown(md_path, md_text=md, assets_dir=assets_dir)

    assert (assets_dir / "anchor_index.json").exists()
    assert (assets_dir / "equation_index.json").exists()
    assert (assets_dir / "reference_index.json").exists()

    anchor_payload = out.get("anchor_index") or {}
    equation_payload = out.get("equation_index") or {}
    ref_payload = out.get("reference_index") or {}

    assert int(anchor_payload.get("anchor_count") or 0) > 0
    assert int(equation_payload.get("equation_count") or 0) >= 1
    assert int(ref_payload.get("ref_count") or 0) >= 1


def test_rebuild_structured_indices_emits_figure_index_without_preexisting_figure_rows(tmp_path: Path):
    md = """# Demo

![Figure 1](./assets/page_1_fig_1.png)

**Figure 1.** A simple caption for the demo figure.
"""
    md_path = tmp_path / "output.md"
    assets_dir = tmp_path / "assets"

    out = rebuild_structured_indices_for_markdown(md_path, md_text=md, assets_dir=assets_dir)

    fig_payload = out.get("figure_index") or {}
    # When a figure exists in markdown blocks, we should still emit a minimal figure index.
    assert isinstance(fig_payload, dict)
    assert (assets_dir / "figure_index.json").exists()
    figures = fig_payload.get("figures") or []
    assert isinstance(figures, list) and len(figures) >= 1


def test_rebuild_structured_indices_writes_empty_figure_index_when_no_figures(tmp_path: Path):
    md = """# Demo

No figures here.
"""
    md_path = tmp_path / "output.md"
    assets_dir = tmp_path / "assets"

    out = rebuild_structured_indices_for_markdown(md_path, md_text=md, assets_dir=assets_dir)

    fig_payload = out.get("figure_index") or {}
    assert isinstance(fig_payload, dict)
    assert (assets_dir / "figure_index.json").exists()
    payload = json.loads((assets_dir / "figure_index.json").read_text(encoding="utf-8"))
    assert payload.get("figures") == []


def test_rebuild_structured_indices_enriches_reference_index_from_catalog_metadata(tmp_path: Path):
    md = """# Demo Paper

## References

[1] A. Demo and B. Example. Stable grounding for figures. Journal of Grounding, 2024. doi:10.1000/demo-grounding
"""
    md_path = tmp_path / "output.md"
    assets_dir = tmp_path / "assets"
    (tmp_path / "reference_catalog.json").write_text(
        json.dumps(
            {
                "version": 1,
                "ref_count": 1,
                "tail_continuity_status": "continuous",
                "missing_numbers": [],
                "refs": [
                    {
                        "reference_number": 1,
                        "reference_text": "[1] A. Demo and B. Example. Stable grounding for figures. Journal of Grounding, 2024. doi:10.1000/demo-grounding",
                        "reference_entry_id": "ref_0001",
                        "parse_confidence": 0.93,
                        "title": "Stable grounding for figures",
                        "authors": "A. Demo and B. Example",
                        "venue": "Journal of Grounding",
                        "year": "2024",
                        "volume": "12",
                        "issue": "3",
                        "pages": "101-109",
                        "doi": "10.1000/demo-grounding",
                        "match_method": "source_work_reference",
                        "crossref_ok": True,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out = rebuild_structured_indices_for_markdown(md_path, md_text=md, assets_dir=assets_dir)

    refs = ((out.get("reference_index") or {}).get("references") or [])
    assert len(refs) == 1
    ref = refs[0]
    assert ref["title"] == "Stable grounding for figures"
    assert ref["authors"] == "A. Demo and B. Example"
    assert ref["venue"] == "Journal of Grounding"
    assert ref["volume"] == "12"
    assert ref["issue"] == "3"
    assert ref["pages"] == "101-109"
    assert ref["match_method"] == "source_work_reference"
    assert ref["crossref_ok"] is True


def test_rebuild_structured_indices_emits_page_and_panel_clauses_for_fallback_figure_index(tmp_path: Path):
    md = """# Demo

![Figure 1](./assets/page_1_fig_1.png)

**Figure 1.** Overview of the demo setup.

a Input acquisition stage with the coded mask. b Reconstruction stage with the decoder.
"""
    md_path = tmp_path / "output.md"
    assets_dir = tmp_path / "assets"

    out = rebuild_structured_indices_for_markdown(md_path, md_text=md, assets_dir=assets_dir)

    figures = ((out.get("figure_index") or {}).get("figures") or [])
    assert len(figures) == 1
    fig = figures[0]
    assert fig["page"] == 1
    assert "Input acquisition stage" in fig["caption_continuation"]
    clauses = fig["panel_clauses"]
    assert clauses == [
        {"panel_letter": "a", "clause": "a Input acquisition stage with the coded mask"},
        {"panel_letter": "b", "clause": "b Reconstruction stage with the decoder"},
    ]
