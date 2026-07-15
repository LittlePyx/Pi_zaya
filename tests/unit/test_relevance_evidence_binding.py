from __future__ import annotations

from api import reference_ui
from kb import retrieval_engine


_DEGRADATION_PROMPT = "这篇论文建模了哪些真实退化？请只根据本文用三点回答，并给出对应引用。"


def test_supporting_citation_output_constraint_does_not_navigate_to_references() -> None:
    assert retrieval_engine._wants_reference_navigation(_DEGRADATION_PROMPT) is False
    assert retrieval_engine._should_skip_reference_like_snippet(
        "## References\n[1] Edgar et al. Principles of single-pixel imaging.",
        heading_path="References",
        question=_DEGRADATION_PROMPT,
    ) is True

    assert retrieval_engine._wants_reference_navigation("请打开参考文献章节") is True
    assert retrieval_engine._wants_reference_navigation("Which prior work introduced ADMM?") is True


def test_reference_card_copy_stays_bound_to_body_primary_evidence_and_rejects_wrong_llm() -> None:
    body_evidence = (
        "The degradation model covers illumination scattering and defocus, mechanical jitter, "
        "and photon shot noise together with electronic noise."
    )
    hit = {
        "text": "## References\n[1] Edgar et al. Principles and prospects for single-pixel imaging.",
        "meta": {
            "heading_path": "References",
            "ref_best_heading_path": "References",
            "ref_show_snippets": [
                "Edgar et al. define several noise sources in a single-pixel imaging review."
            ],
        },
    }
    ui_meta = {
        "display_name": "Comprehensive compensation of real-world degradations.pdf",
        "heading_path": "References",
        "location_label": "References P.10",
        "summary_line": "本文建立并分析了真实退化的综合物理模型。",
        "why_line": (
            "在“References”中可找到 Edgar 等人关于单像素成像原理与前景的文献，"
            "该文献定义了多种噪声源。"
        ),
        "primary_evidence": {
            "source_path": r"db\LSA-2025\LSA-2025.en.md",
            "source_name": "Comprehensive compensation of real-world degradations.pdf",
            "heading_path": "Abstract / Structured degradation modeling",
            "snippet": body_evidence,
            "highlight_snippet": body_evidence,
            "page_start": 1,
            "block_id": "blk-degradation-model",
            "selection_reason": "prompt_aligned_block",
            "strict_locate": True,
        },
        "reader_open": {
            "headingPath": "References",
            "snippet": str(hit["text"]),
        },
    }

    prepared = reference_ui._prepare_ref_hit_card_llm_grounding(
        prompt=_DEGRADATION_PROMPT,
        hit=hit,
        ui_meta=ui_meta,
    )

    assert prepared["heading_path"] == "Abstract / Structured degradation modeling"
    assert body_evidence in prepared["candidate_payload"]
    assert "Edgar" not in prepared["candidate_payload"]
    assert "References" not in prepared["candidate_payload"]
    prepared_ui = dict(prepared["ui_meta"])
    assert prepared_ui["heading_path"] == "Abstract / Structured degradation modeling"
    assert prepared_ui["location_label"] == "Abstract / Structured degradation modeling P.1"
    assert dict(prepared_ui["reader_open"])["headingPath"] == prepared_ui["heading_path"]
    assert "Edgar" not in str(prepared_ui.get("why_line") or "")

    out = reference_ui._apply_llm_grounded_ref_hit_card_copy(
        prompt=_DEGRADATION_PROMPT,
        prepared=prepared,
        polished_summary="Edgar 等人的综述定义了多种噪声来源。",
        polished_why=(
            "在“References”中可找到 Edgar 等人的文献，因此它能验证本文退化模型。"
        ),
    )

    rendered_copy = " ".join(
        str(out.get(key) or "") for key in ("summary_line", "why_line", "heading_path", "location_label")
    )
    assert "Edgar" not in rendered_copy
    assert "References" not in rendered_copy
    assert out.get("summary_generation") != "llm_grounded"
    assert out.get("why_generation") != "llm_grounded"

    unsupported_author = reference_ui._apply_llm_grounded_ref_hit_card_copy(
        prompt=_DEGRADATION_PROMPT,
        prepared=prepared,
        polished_summary="",
        polished_why="Edgar 等人的综述定义了多种噪声源，因此能够验证本文退化模型的全面性。",
    )
    assert "Edgar" not in str(unsupported_author.get("why_line") or "")
    assert unsupported_author.get("why_generation") != "llm_grounded"
