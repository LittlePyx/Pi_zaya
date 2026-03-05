from __future__ import annotations

import api.reference_ui as reference_ui
from api.reference_ui import _effective_ui_score, build_hit_ui_meta, enrich_refs_payload, ensure_source_citation_meta


def test_effective_ui_score_penalizes_weak_evidence_high_llm_score():
    hit = {
        "meta": {
            "ref_pack_state": "ready",
            "ref_rank": {
                "llm": 85.0,
                "bm25": 0.6,
                "deep": 0.0,
                "term_bonus": 0.0,
                "semantic_score": 8.5,
            },
            "ref_loc_quality": "low",
        }
    }
    score, pending = _effective_ui_score(hit)
    assert pending is False
    assert score is not None
    assert score < 6.0


def test_effective_ui_score_keeps_high_score_for_strong_evidence():
    hit = {
        "meta": {
            "ref_pack_state": "ready",
            "ref_rank": {
                "llm": 91.0,
                "bm25": 6.2,
                "deep": 2.8,
                "term_bonus": 2.2,
                "semantic_score": 9.1,
            },
            "ref_section": "Method",
            "ref_loc_quality": "high",
        }
    }
    score, pending = _effective_ui_score(hit)
    assert pending is False
    assert score is not None
    assert score >= 8.5


def test_enrich_refs_payload_keeps_anchor_grounded_hit_even_when_score_is_low():
    refs = {
        11: {
            "prompt": "What does equation 8 describe in this paper?",
            "hits": [
                {
                    "text": "Equation (8) defines the total-curvature objective.",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "explicit_doc_match_score": 8.4,
                        "anchor_target_kind": "equation",
                        "anchor_target_number": 8,
                        "anchor_match_score": 13.0,
                        "ref_rank": {
                            "llm": 56.0,
                            "bm25": 0.8,
                            "deep": 0.0,
                            "term_bonus": 0.0,
                            "semantic_score": 5.6,
                        },
                    },
                }
            ],
        }
    }

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(11) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = hits[0].get("ui_meta", {}) or {}
    assert str(ui_meta.get("display_name") or "").endswith(".pdf")
    badges = list(ui_meta.get("semantic_badges") or [])
    assert badges
    assert str(badges[0].get("text") or "").strip()


def test_enrich_refs_payload_prefetches_citation_meta(monkeypatch):
    refs = {
        7: {
            "prompt": "What is SCIGS?",
            "hits": [
                {
                    "text": "SCIGS recovers 3D Gaussian splats from a snapshot compressive image.",
                    "meta": {
                        "source_path": r"db\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 72.0, "bm25": 3.0, "deep": 1.0, "term_bonus": 1.0, "semantic_score": 7.2},
                        "ref_section": "Related Work",
                        "ref_loc_quality": "high",
                    },
                }
            ],
        }
    }

    def _fake_ensure_source_citation_meta(**kwargs):
        return {
            "title": "SCIGS: 3D Gaussians Splatting from A Snapshot Compressive Image",
            "venue": "ICIP",
            "year": "2025",
        }

    monkeypatch.setattr(reference_ui, "ensure_source_citation_meta", _fake_ensure_source_citation_meta)
    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    ui_meta = ((out.get(7) or {}).get("hits") or [])[0].get("ui_meta", {})

    citation_meta = ui_meta.get("citation_meta", {}) or {}
    assert citation_meta.get("venue") == "ICIP"
    assert citation_meta.get("year") == "2025"


def test_ensure_source_citation_meta_seeds_filename_fields_when_lookup_is_empty(monkeypatch):
    monkeypatch.setattr(reference_ui, "fetch_crossref_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_enrich_bibliometrics", lambda meta: meta)

    meta = ensure_source_citation_meta(
        source_path=r"db\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.en.md",
        pdf_root=None,
        md_root=None,
        lib_store=None,
    )

    assert meta.get("venue") == "ICIP"
    assert meta.get("year") == "2025"
    assert "SCIGS" in str(meta.get("title") or "")


def test_build_hit_ui_meta_exposes_anchor_semantic_fields():
    hit = {
        "meta": {
            "source_path": r"db\LPR-2025\LPR-2025.en.md",
            "ref_pack_state": "ready",
            "anchor_target_kind": "figure",
            "anchor_target_number": 3,
            "anchor_match_score": 11.8,
            "explicit_doc_match_score": 8.2,
            "ref_rank": {
                "llm": 81.0,
                "bm25": 4.2,
                "deep": 1.8,
                "term_bonus": 0.6,
                "semantic_score": 7.9,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="What does figure 3 show?",
        pdf_root=None,
        lib_store=None,
    )

    assert ui_meta.get("anchor_target_kind") == "figure"
    assert int(ui_meta.get("anchor_target_number") or 0) == 3
    assert float(ui_meta.get("anchor_match_score") or 0.0) == 11.8
    assert float(ui_meta.get("explicit_doc_match_score") or 0.0) == 8.2
    badges = list(ui_meta.get("semantic_badges") or [])
    assert len(badges) == 1
    assert "图示语义命中" in str(badges[0].get("text") or "")
    assert float(badges[0].get("score") or 0.0) == 11.8


def test_build_hit_ui_meta_adds_doc_semantic_badge_without_anchor():
    hit = {
        "meta": {
            "source_path": r"db\LPR-2025\LPR-2025.en.md",
            "ref_pack_state": "ready",
            "anchor_target_kind": "",
            "anchor_target_number": 0,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 7.1,
            "ref_rank": {
                "llm": 74.0,
                "bm25": 3.0,
                "deep": 0.9,
                "term_bonus": 0.0,
                "semantic_score": 6.8,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="Please summarize this paper.",
        pdf_root=None,
        lib_store=None,
    )

    badges = list(ui_meta.get("semantic_badges") or [])
    assert len(badges) == 1
    assert str(badges[0].get("text") or "") == "文档语义直连"
    assert float(badges[0].get("score") or 0.0) == 7.1
