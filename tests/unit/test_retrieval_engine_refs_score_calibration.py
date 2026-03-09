from __future__ import annotations

from kb.retrieval_engine import _calibrate_refs_pack_score


def test_calibrate_refs_pack_score_penalizes_weak_evidence_high_raw_score():
    meta = {
        "ref_rank": {
            "bm25": 0.6,
            "deep": 0.0,
            "term_bonus": -0.8,
            "semantic_score": 0.0,
        },
        "explicit_doc_match_score": 0.0,
    }
    score = _calibrate_refs_pack_score(raw_score=97.6, meta=meta, section="")
    assert float(score) <= 60.0


def test_calibrate_refs_pack_score_keeps_anchor_grounded_item_high():
    meta = {
        "ref_rank": {
            "bm25": 0.5,
            "deep": 0.0,
            "term_bonus": -0.5,
            "semantic_score": 0.0,
        },
        "anchor_target_kind": "equation",
        "anchor_match_score": 13.5,
        "explicit_doc_match_score": 0.0,
    }
    score = _calibrate_refs_pack_score(raw_score=52.0, meta=meta, section="")
    assert float(score) >= 85.0
