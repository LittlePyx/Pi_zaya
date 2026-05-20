from __future__ import annotations

import pytest
from fastapi import HTTPException

import api.routers.references as references_router


class _FakeStore:
    def __init__(self, refs: dict) -> None:
        self._refs = refs

    def get_conversation(self, conv_id: str):
        return {}

    def list_message_refs(self, conv_id: str, timeout_s: float = 10.0):
        if conv_id == "not-found":
            return None
        return self._refs


def _make_no_candidate_pack() -> dict:
    """A pack with no raw hits — triggers no_candidate_hits suppression."""
    return {
        "prompt": "What is the GPU model used in the paper?",
        "hits": [],
        "pipeline_debug": {
            "raw_hit_count": 0,
            "post_score_gate_hit_count": 0,
            "post_focus_filter_hit_count": 0,
            "post_llm_filter_hit_count": 0,
            "final_hit_count": 0,
        },
    }


def _make_score_gate_suppressed_pack() -> dict:
    """A pack where all hits were removed by the score gate."""
    return {
        "prompt": "What reconstruction method does the paper propose?",
        "hits": [],
        "pipeline_debug": {
            "raw_hit_count": 5,
            "post_score_gate_hit_count": 0,
            "post_focus_filter_hit_count": 0,
            "post_llm_filter_hit_count": 0,
            "final_hit_count": 0,
        },
    }


def _make_focus_filter_suppressed_pack() -> dict:
    """A pack where hits passed score gate but were removed by focus filter."""
    return {
        "prompt": "Compare the PSNR results of method A and method B.",
        "hits": [],
        "pipeline_debug": {
            "raw_hit_count": 8,
            "post_score_gate_hit_count": 6,
            "post_focus_filter_hit_count": 0,
            "post_llm_filter_hit_count": 0,
            "final_hit_count": 0,
        },
    }


def _make_llm_filter_suppressed_pack() -> dict:
    """A pack where hits passed focus filter but all were removed by LLM filter."""
    return {
        "prompt": "What does Figure 3 show?",
        "hits": [],
        "pipeline_debug": {
            "raw_hit_count": 10,
            "post_score_gate_hit_count": 8,
            "post_focus_filter_hit_count": 5,
            "post_llm_filter_hit_count": 0,
            "final_hit_count": 0,
        },
    }


def test_diagnose_endpoint_not_found(monkeypatch):
    store = _FakeStore(None)
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    with pytest.raises(HTTPException) as exc:
        references_router.get_refs_diagnose("not-found")
    assert exc.value.status_code == 404


def test_diagnose_empty_refs(monkeypatch):
    store = _FakeStore({})
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    report = references_router.get_refs_diagnose("conv-empty")
    assert report["total_packs"] == 0
    assert report["empty_packs"] == 0
    assert report["suppressed_packs"] == 0
    assert report["packs"] == {}


def test_diagnose_no_candidate_hits(monkeypatch):
    refs = {1: _make_no_candidate_pack()}
    store = _FakeStore(refs)
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    report = references_router.get_refs_diagnose("conv-1")
    assert report["total_packs"] == 1
    assert report["empty_packs"] == 1
    pack = report["packs"].get(1, {})
    assert pack["display_state"] == "empty"
    assert pack["suppression_reason"] == "no_candidate_hits"
    assert pack["suggestion"]
    assert pack["used_query"] == "What is the GPU model used in the paper?"
    assert pack["used_translation"] is False


def test_diagnose_score_gate_suppressed(monkeypatch):
    refs = {1: _make_score_gate_suppressed_pack()}
    store = _FakeStore(refs)
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    report = references_router.get_refs_diagnose("conv-sg")
    assert report["total_packs"] == 1
    assert report["suppressed_packs"] == 1
    pack = report["packs"].get(1, {})
    assert pack["display_state"] == "suppressed"
    assert pack["suppression_reason"] == "score_gate_removed_all"
    assert pack["suggestion"]


def test_diagnose_focus_filter_suppressed(monkeypatch):
    refs = {1: _make_focus_filter_suppressed_pack()}
    store = _FakeStore(refs)
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    report = references_router.get_refs_diagnose("conv-ff")
    pack = report["packs"].get(1, {})
    assert pack["suppression_reason"] == "focus_filter_removed_all"


def test_diagnose_llm_filter_suppressed(monkeypatch):
    refs = {1: _make_llm_filter_suppressed_pack()}
    store = _FakeStore(refs)
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    report = references_router.get_refs_diagnose("conv-lf")
    pack = report["packs"].get(1, {})
    assert pack["suppression_reason"] == "llm_filter_removed_all"


def test_diagnose_mixed_packs(monkeypatch):
    refs = {
        1: _make_no_candidate_pack(),
        2: _make_score_gate_suppressed_pack(),
        3: {
            "prompt": "What is the paper about?",
            "hits": [{"score": 12.5, "meta": {"source_path": "db/paper/paper.en.md", "source_name": "paper"}}],
            "pipeline_debug": {
                "raw_hit_count": 1,
                "post_score_gate_hit_count": 1,
                "post_focus_filter_hit_count": 1,
                "post_llm_filter_hit_count": 1,
                "final_hit_count": 1,
            },
        },
    }
    store = _FakeStore(refs)
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    report = references_router.get_refs_diagnose("conv-mixed")
    assert report["total_packs"] == 3
    assert report["empty_packs"] == 1
    assert report["suppressed_packs"] == 1
    # Pack 3 should be "ready"
    pack3 = report["packs"].get(3, {})
    assert pack3["display_state"] == "ready"


def test_diagnose_top_scores(monkeypatch):
    refs = {
        1: {
            "prompt": "What method?",
            "hits": [
                {"score": 15.0, "meta": {"source_path": "db/a/a.en.md", "source_name": "paper-a"}},
                {"score": 12.5, "meta": {"source_path": "db/b/b.en.md", "source_name": "paper-b", "heading_path": "Method"}},
            ],
            "pipeline_debug": {
                "raw_hit_count": 2,
                "post_score_gate_hit_count": 2,
                "post_focus_filter_hit_count": 2,
                "post_llm_filter_hit_count": 2,
                "final_hit_count": 2,
            },
        }
    }
    store = _FakeStore(refs)
    monkeypatch.setattr(references_router, "get_chat_store", lambda: store)
    report = references_router.get_refs_diagnose("conv-scores")
    pack = report["packs"].get(1, {})
    scores = pack["top_scores"]
    assert len(scores) == 2
    assert scores[0]["score"] == 15.0
    assert scores[0]["doc_name"] == "paper-a"
    assert scores[1]["doc_name"] == "paper-b"
    assert scores[1]["heading_path"] == "Method"


def test_diagnose_suggestion_all_reasons():
    """Every suppression_reason maps to a non-empty suggestion."""
    reasons = [
        "no_candidate_hits",
        "score_gate_removed_all",
        "focus_filter_removed_all",
        "llm_filter_removed_all",
        "guide_self_source_only",
        "render_failed",
        "pending_enrichment",
        "no_renderable_hits",
        "unknown_reason",
    ]
    for reason in reasons:
        suggestion = references_router._compute_diagnose_suggestion(reason)
        assert suggestion, f"Empty suggestion for {reason}"
