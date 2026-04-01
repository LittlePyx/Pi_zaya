from pathlib import Path

from kb.paper_guide_citation_surfacing import (
    _collect_paper_guide_candidate_refs_by_source,
    _drop_paper_guide_locate_only_line_citations,
    _inject_paper_guide_card_citations,
    _inject_paper_guide_fallback_citations,
    _inject_paper_guide_focus_citations,
    _promote_paper_guide_numeric_reference_citations,
)


def test_collect_paper_guide_candidate_refs_by_source_merges_focus_excerpt_refs():
    out = _collect_paper_guide_candidate_refs_by_source(
        [
            {
                "source_path": r"db\doc\paper.en.md",
                "candidate_refs": [],
                "snippet": "APR aligns the interferometric stack.",
            }
        ],
        focus_source_path=r"db\doc\paper.en.md",
        special_focus_block="Paper-guide method focus:\n- Focus snippet:\nAPR was performed using image registration based on phase correlation [35].",
        extract_special_focus_excerpt=lambda block: str(block or "").split("Focus snippet:\n", 1)[-1].strip(),
    )
    assert out == {r"db\doc\paper.en.md": [35]}


def test_inject_paper_guide_fallback_citations_adds_marker_for_cue_overlap():
    out = _inject_paper_guide_fallback_citations(
        "Conclusion: APR is the key method step.\n\nEvidence:\n- Uses phase correlation based image registration between each off-axis raw image and the central image.",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [24],
                "cue": "APR was performed using image registration based on phase correlation [24].",
            }
        ],
        prompt_family="method",
    )
    assert "[[CITE:s50f9c165:24]]" in out


def test_inject_paper_guide_card_citations_uses_heading_and_short_tokens():
    out = _inject_paper_guide_card_citations(
        "Conclusion: APR improves CNR.\n\nEvidence:\n- APR improves CNR in the reassigned image.",
        cards=[
            {
                "sid": "s50f9c165",
                "candidate_refs": [24],
                "heading": "Adaptive pixel-reassignment (APR)",
                "snippet": "APR improves CNR in the reassigned image.",
            }
        ],
        prompt_family="method",
    )
    assert "[[CITE:s50f9c165:24]]" in out


def test_inject_paper_guide_focus_citations_rewrites_inline_focus_ref():
    out = _inject_paper_guide_focus_citations(
        "Conclusion:\nAPR improves alignment.\n\nEvidence:\n- Implementation detail: APR was performed using image registration based on phase correlation [35].",
        special_focus_block="Paper-guide method focus:\n- Focus snippet:\nAPR was performed using image registration based on phase correlation [35].",
        source_path=r"db\doc\paper.en.md",
        prompt_family="method",
        cite_source_id=lambda _src: "s50f9c165",
        extract_special_focus_excerpt=lambda block: str(block or "").split("Focus snippet:\n", 1)[-1].strip(),
    )
    assert "[35]" not in out
    assert "[[CITE:s50f9c165:35]]" in out


def test_drop_paper_guide_locate_only_line_citations_strips_structured_cite():
    out = _drop_paper_guide_locate_only_line_citations(
        "Caption anchor: Figure 1. [[CITE:s50f9c165:35]]",
        support_resolution=[
            {
                "line_index": 0,
                "cite_policy": "locate_only",
                "resolved_ref_num": 0,
            }
        ],
    )
    assert "[[CITE:" not in out


def test_promote_paper_guide_numeric_reference_citations_rewrites_reference_label():
    out = _promote_paper_guide_numeric_reference_citations(
        "Consult reference [35] for the implementation details.",
        locked_source={"sid": "s50f9c165"},
    )
    assert out == "Consult reference [[CITE:s50f9c165:35]] for the implementation details."
