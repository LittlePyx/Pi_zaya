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


def test_build_hit_ui_meta_falls_back_to_snippet_summary_when_ref_pack_missing(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    hit = {
        "meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "ref_best_heading_path": "3. Method / 3.1. Background on NeRF",
            "ref_section": "3. Method",
            "ref_subsection": "3.1. Background on NeRF",
            "ref_show_snippets": [
                "In this paper, we present SCINeRF, a novel approach for 3D scene representation learning from a single snapshot compressed image."
            ],
            "ref_overview_snippets": [],
            "ref_rank": {
                "llm": 0.0,
                "bm25": 6.0,
                "deep": 0.0,
                "term_bonus": 0.0,
                "semantic_score": 0.0,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="SCINeRF 是什么？",
        pdf_root=None,
        lib_store=None,
    )

    assert "SCINeRF" in str(ui_meta.get("summary_line") or "")
    assert str(ui_meta.get("why_line") or "").strip()


def test_build_hit_ui_meta_falls_back_to_citation_summary_when_ref_pack_missing(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    hit = {
        "meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "ref_best_heading_path": "3. Method / 3.1. Background on NeRF",
            "ref_section": "3. Method",
            "ref_subsection": "3.1. Background on NeRF",
            "ref_show_snippets": [],
            "ref_overview_snippets": [],
            "ref_rank": {
                "llm": 0.0,
                "bm25": 4.2,
                "deep": 0.0,
                "term_bonus": 0.0,
                "semantic_score": 0.0,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="这个方法和当前问题有什么关系？",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            r"db\SCINeRF\SCINeRF.en.md": {
                "title": "SCINeRF",
                "summary_line": "当前仅检索到文献元数据：该工作发表于 2024。由于缺少可用摘要文本，暂无法可靠提炼其方法细节与实验结论，建议通过 DOI 查看原文摘要与正文。",
            }
        },
    )

    assert "当前仅检索到文献元数据" in str(ui_meta.get("summary_line") or "")
    assert str(ui_meta.get("why_line") or "").strip()


def test_effective_ui_score_uses_evidence_spread_for_same_llm_value():
    hit_strong = {
        "meta": {
            "source_path": r"db\A\A.en.md",
            "ref_pack_state": "ready",
            "ref_rank": {
                "llm": 97.6,
                "bm25": 9.8,
                "deep": 4.4,
                "term_bonus": 2.6,
                "semantic_score": 9.7,
            },
            "ref_section": "Method",
            "ref_loc_quality": "high",
        }
    }
    hit_weak = {
        "meta": {
            "source_path": r"db\B\B.en.md",
            "ref_pack_state": "ready",
            "ref_rank": {
                "llm": 97.6,
                "bm25": 1.1,
                "deep": 0.0,
                "term_bonus": 0.0,
                "semantic_score": 7.0,
            },
            "ref_section": "",
            "ref_loc_quality": "low",
        }
    }

    strong, pending_strong = _effective_ui_score(hit_strong)
    weak, pending_weak = _effective_ui_score(hit_weak)

    assert pending_strong is False
    assert pending_weak is False
    assert strong is not None and weak is not None
    assert strong > weak
    assert (strong - weak) >= 0.6


def test_effective_ui_score_breaks_identical_decimal_tails_with_stable_jitter():
    base_meta = {
        "ref_pack_state": "ready",
        "ref_rank": {
            "llm": 87.6,
            "bm25": 5.0,
            "deep": 2.0,
            "term_bonus": 1.0,
            "semantic_score": 8.6,
        },
        "ref_section": "Results",
        "ref_loc_quality": "high",
    }
    hit_a = {"meta": dict(base_meta, source_path=r"db\X\doc_x.en.md")}
    hit_b = {"meta": dict(base_meta, source_path=r"db\Y\doc_y.en.md")}

    score_a, pending_a = _effective_ui_score(hit_a)
    score_b, pending_b = _effective_ui_score(hit_b)

    assert pending_a is False
    assert pending_b is False
    assert score_a is not None and score_b is not None
    assert abs(score_a - score_b) >= 0.005
    assert abs(score_a - score_b) <= 0.09


def test_enrich_refs_payload_hides_bound_paper_for_paper_guide_mode():
    refs = {
        21: {
            "prompt": "Explain the bound paper only.",
            "hits": [
                {
                    "text": "Bound paper evidence.",
                    "meta": {
                        "source_path": r"db\SCINeRF\SCINeRF.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {
                            "llm": 88.0,
                            "bm25": 6.0,
                            "deep": 2.0,
                            "term_bonus": 1.0,
                            "semantic_score": 8.6,
                        },
                        "ref_section": "Method",
                        "ref_loc_quality": "high",
                    },
                }
            ],
        }
    }

    out = enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        guide_mode=True,
        guide_source_path=r"F:\papers\SCINeRF.pdf",
        guide_source_name="SCINeRF.pdf",
    )

    entry = out.get(21) or {}
    assert list(entry.get("hits") or []) == []
    guide_filter = entry.get("guide_filter") or {}
    assert guide_filter.get("hidden_self_source") is True
    assert int(guide_filter.get("filtered_hit_count") or 0) == 1
    assert str(guide_filter.get("guide_source_name") or "") == "SCINeRF.pdf"


def test_enrich_refs_payload_keeps_non_bound_paper_hits_in_paper_guide_mode():
    refs = {
        22: {
            "prompt": "Find related external papers.",
            "hits": [
                {
                    "text": "External evidence.",
                    "meta": {
                        "source_path": r"db\SCIGS\SCIGS.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {
                            "llm": 84.0,
                            "bm25": 5.8,
                            "deep": 1.9,
                            "term_bonus": 0.8,
                            "semantic_score": 8.1,
                        },
                        "ref_section": "Results",
                        "ref_loc_quality": "high",
                    },
                }
            ],
        }
    }

    out = enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        guide_mode=True,
        guide_source_path=r"F:\papers\SCINeRF.pdf",
        guide_source_name="SCINeRF.pdf",
    )

    entry = out.get(22) or {}
    hits = list(entry.get("hits") or [])
    assert len(hits) == 1
    guide_filter = entry.get("guide_filter") or {}
    assert guide_filter.get("hidden_self_source") is False
    assert int(guide_filter.get("filtered_hit_count") or 0) == 0
