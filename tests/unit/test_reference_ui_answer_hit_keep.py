from __future__ import annotations

from api.reference_ui import _filter_refs_hits_by_prompt_focus, _sort_refs_hits_for_display


def _hit(source_path: str, text: str, *, answer_hit: bool = False) -> dict:
    meta = {
        "source_path": source_path,
        "ref_pack_state": "ready",
        "ref_rank": {"bm25": 7.0, "llm": 70.0, "score": 8.0},
        "ref_best_heading_path": "1. Introduction",
    }
    if answer_hit:
        meta["ref_display_reason"] = "answer_hit_top"
    return {"text": text, "meta": meta}


def test_answer_hit_top_survives_focus_filter_and_sorts_first() -> None:
    prompt = "深度学习给单像素成像带来的好处和坑分别是什么？"
    answer_source = _hit(
        "db/dl-spi-review.en.md",
        "The encoder samples images into low-dimensional measurements; data-driven strategies still face generalization challenges.",
        answer_hit=True,
    )
    lexical_source = _hit(
        "db/ilnet.en.md",
        "Keywords: Single-pixel imaging Information extraction network Deep learning.",
    )

    filtered = _filter_refs_hits_by_prompt_focus(prompt, [answer_source, lexical_source])
    ordered = _sort_refs_hits_for_display(prompt=prompt, hits=filtered)

    assert answer_source in filtered
    assert ordered[0] is answer_source
