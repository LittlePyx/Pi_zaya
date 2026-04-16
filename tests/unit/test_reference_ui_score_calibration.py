from __future__ import annotations

from pathlib import Path

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


def test_enrich_refs_payload_filters_bound_source_by_guide_name_without_bound_path():
    refs = {
        5: {
            "prompt": "Summarize Figure 1.",
            "hits": [
                {
                    "text": "Figure 1 shows the SCI pipeline.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {
                            "llm": 88.0,
                            "bm25": 5.8,
                            "deep": 1.6,
                            "term_bonus": 0.8,
                            "semantic_score": 8.1,
                        },
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
        guide_source_path="",
        guide_source_name="2024 IEEE-CVF Conference on Computer Vision and Pattern Recognition (CVPR)-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf",
    )

    entry = out.get(5) or {}
    assert list(entry.get("hits") or []) == []
    guide_filter = entry.get("guide_filter", {}) or {}
    assert guide_filter.get("active") is True
    assert guide_filter.get("hidden_self_source") is True
    assert int(guide_filter.get("filtered_hit_count") or 0) == 1
    pipeline_debug = entry.get("pipeline_debug", {}) or {}
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 0
    assert int(pipeline_debug.get("filtered_self_hit_count") or 0) == 1
    assert int(pipeline_debug.get("final_hit_count") or 0) == 0


def test_enrich_refs_payload_keeps_external_hits_while_filtering_bound_source(monkeypatch):
    refs = {
        6: {
            "prompt": "What other papers are relevant?",
            "hits": [
                {
                    "text": "SCINeRF paper hit.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 81.0, "bm25": 4.2, "deep": 1.0, "term_bonus": 0.4, "semantic_score": 7.4},
                    },
                },
                {
                    "text": "Another paper remains visible.",
                    "meta": {
                        "source_path": r"db\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 79.0, "bm25": 4.0, "deep": 1.2, "term_bonus": 0.6, "semantic_score": 7.1},
                        "ref_section": "Related Work",
                        "ref_loc_quality": "high",
                    },
                },
            ],
        }
    }

    with monkeypatch.context() as m:
        m.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
        m.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))
        out = enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        guide_mode=True,
        guide_source_path=r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md",
        guide_source_name="CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf",
        )

    entry = out.get(6) or {}
    hits = list(entry.get("hits") or [])
    assert len(hits) == 1
    kept_path = str(((hits[0].get("meta") or {}).get("source_path") or "")).strip()
    assert "SCIGS" in kept_path
    guide_filter = entry.get("guide_filter", {}) or {}
    assert guide_filter.get("hidden_self_source") is True
    assert int(guide_filter.get("filtered_hit_count") or 0) == 1
    pipeline_debug = entry.get("pipeline_debug", {}) or {}
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 1
    assert int(pipeline_debug.get("filtered_self_hit_count") or 0) == 1
    assert int(pipeline_debug.get("post_score_gate_hit_count") or 0) == 1
    assert int(pipeline_debug.get("final_hit_count") or 0) == 1


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


def test_effective_ui_score_keeps_failed_hit_when_evidence_surface_is_strong():
    hit = {
        "meta": {
            "ref_pack_state": "failed",
            "ref_rank": {
                "bm25": 8.4,
                "deep": 18.0,
                "term_bonus": 1.8,
                "semantic_score": 7.6,
                "score": 26.0,
            },
            "ref_best_heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "ref_show_snippets": [
                "This section explicitly defines dynamic supersampling by shifting pixel boundaries frame by frame.",
            ],
        }
    }

    score, pending = _effective_ui_score(hit)
    assert pending is False
    assert score is not None
    assert score >= 6.0


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


def test_enrich_refs_payload_skips_citation_prefetch_while_hits_are_pending(monkeypatch):
    refs = {
        71: {
            "prompt": "Which paper in my library most directly defines dynamic supersampling?",
            "hits": [
                {
                    "text": "Spatially variant digital supersampling shifts pixel boundaries frame by frame.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "pending",
                        "ref_rank": {"llm": 0.0, "bm25": 7.0, "deep": 4.0, "term_bonus": 0.5, "semantic_score": 0.0},
                    },
                }
            ],
        }
    }

    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pending refs should not prefetch citation meta")))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(71) or {}).get("hits") or [])

    assert len(hits) == 1


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
    reader_open = ui_meta.get("reader_open") or {}
    assert str(reader_open.get("sourcePath") or "") == r"db\SCINeRF\SCINeRF.en.md"
    assert str(reader_open.get("headingPath") or "") == "3. Method / 3.1. Background on NeRF"
    assert "SCINeRF" in str(reader_open.get("snippet") or "")
    assert str(reader_open.get("highlightSnippet") or "") == str(reader_open.get("snippet") or "")


def test_build_hit_ui_meta_builds_reader_open_candidates_from_refs_signals(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    hit = {
        "meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "ref_best_heading_path": "3. Method / 3.1. Background on NeRF",
            "ref_section": "3. Method",
            "ref_subsection": "3.1. Background on NeRF",
            "anchor_target_kind": "equation",
            "anchor_target_number": 8,
            "ref_show_snippets": [
                "SCINeRF introduces a NeRF-oriented reconstruction pipeline from a single compressed snapshot.",
                "The method uses structured priors to stabilize training and improve scene recovery.",
            ],
            "ref_snippets": [
                "Equation (8) is used to balance fidelity and regularization during optimization."
            ],
            "ref_overview_snippets": [
                "The paper reports stronger reconstruction quality than prior snapshot baselines."
            ],
            "ref_locs": [
                {
                    "heading_path": "4. Experiments / 4.2 Quantitative Results",
                    "score": 8.6,
                    "quality": "high",
                },
                {
                    "heading_path": "5. Discussion",
                    "text": "This section discusses tradeoffs between compression ratio and fidelity.",
                    "score": 7.1,
                    "quality": "high",
                },
            ],
            "ref_rank": {
                "llm": 76.0,
                "bm25": 4.8,
                "deep": 1.7,
                "term_bonus": 0.6,
                "semantic_score": 7.5,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="Equation 8 is used in which part of the paper?",
        pdf_root=None,
        lib_store=None,
    )

    reader_open = ui_meta.get("reader_open") or {}
    assert str(reader_open.get("headingPath") or "") == "3. Method / 3.1. Background on NeRF"
    assert str(reader_open.get("anchorKind") or "") == "equation"
    assert int(reader_open.get("anchorNumber") or 0) == 8
    assert reader_open.get("initialAltIndex") == 0

    alternatives = list(reader_open.get("alternatives") or [])
    visible = list(reader_open.get("visibleAlternatives") or [])
    evidence = list(reader_open.get("evidenceAlternatives") or [])
    assert len(alternatives) >= 2
    assert len(visible) >= 3
    assert len(evidence) >= 3
    assert str(visible[0].get("headingPath") or "") == "3. Method / 3.1. Background on NeRF"
    assert int(visible[0].get("anchorNumber") or 0) == 8
    assert any("4. Experiments / 4.2 Quantitative Results" in str(item.get("headingPath") or "") for item in alternatives)
    assert any("Equation (8)" in str(item.get("snippet") or "") for item in alternatives)
    assert any("tradeoffs between compression ratio and fidelity" in str(item.get("snippet") or "") for item in alternatives)


def test_build_hit_ui_meta_prefers_prompt_aligned_summary_snippet(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    hit = {
        "meta": {
            "source_path": r"db\Demo\Demo.en.md",
            "ref_best_heading_path": "2. Related Work",
            "ref_section": "2. Related Work",
            "ref_show_snippets": [
                "This paper surveys reconstruction strategies for compressive image formation and compares broad families of methods.",
            ],
            "ref_snippets": [
                "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4] for iterative optimization.",
            ],
            "ref_rank": {
                "llm": 75.0,
                "bm25": 4.6,
                "deep": 1.2,
                "term_bonus": 0.4,
                "semantic_score": 7.2,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="Where is ADMM discussed in this paper?",
        pdf_root=None,
        lib_store=None,
    )

    assert "ADMM" in str(ui_meta.get("summary_line") or "")
    reader_open = ui_meta.get("reader_open") or {}
    assert "ADMM" in str(reader_open.get("snippet") or "")


def test_build_hit_ui_meta_prefers_anchor_number_aligned_summary(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    hit = {
        "meta": {
            "source_path": r"db\Demo\Demo.en.md",
            "ref_best_heading_path": "4. Results",
            "ref_section": "4. Results",
            "anchor_target_kind": "figure",
            "anchor_target_number": 3,
            "ref_show_snippets": [
                "The results section reports stronger reconstruction quality than prior methods.",
            ],
            "ref_snippets": [
                "Figure 3 compares reconstruction fidelity across different compression ratios.",
            ],
            "ref_rank": {
                "llm": 79.0,
                "bm25": 5.0,
                "deep": 1.5,
                "term_bonus": 0.6,
                "semantic_score": 7.8,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="What does figure 3 show?",
        pdf_root=None,
        lib_store=None,
    )

    assert "Figure 3" in str(ui_meta.get("summary_line") or "")
    reader_open = ui_meta.get("reader_open") or {}
    assert "Figure 3" in str(reader_open.get("snippet") or "")


def test_build_hit_ui_meta_resolves_exact_reader_open_identity_from_source_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    md_path = tmp_path / "fixture.en.md"
    md_path.write_text(
        "# Fixture Paper\n\n"
        "## 2. Method\n\n"
        "Equation (1) defines the rendering loss used for scene reconstruction.\n\n"
        "$$\nL = ||x-y||_2^2 \\tag{1}\n$$\n\n"
        "## 4. Experiments\n\n"
        "Experimental analysis reuses the same rendering loss for ablation studies.\n",
        encoding="utf-8",
    )

    hit = {
        "meta": {
            "source_path": str(md_path),
            "ref_best_heading_path": "2. Method",
            "ref_section": "2. Method",
            "anchor_target_kind": "equation",
            "anchor_target_number": 1,
            "ref_show_snippets": [
                "Equation (1) defines the rendering loss used for scene reconstruction.",
            ],
            "ref_snippets": [
                "Experimental analysis reuses the same rendering loss for ablation studies.",
            ],
            "ref_rank": {
                "llm": 79.0,
                "bm25": 4.8,
                "deep": 1.4,
                "term_bonus": 0.5,
                "semantic_score": 7.7,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="Where is equation 1 introduced?",
        pdf_root=None,
        lib_store=None,
    )

    reader_open = ui_meta.get("reader_open") or {}
    locate_target = reader_open.get("locateTarget") or {}
    primary_evidence = ui_meta.get("primary_evidence") or {}
    reader_primary = reader_open.get("primaryEvidence") or {}
    assert reader_open.get("strictLocate") is True
    assert str(reader_open.get("blockId") or "").strip()
    assert str(reader_open.get("anchorId") or "").strip()
    assert str(locate_target.get("blockId") or "") == str(reader_open.get("blockId") or "")
    assert str(locate_target.get("anchorId") or "") == str(reader_open.get("anchorId") or "")
    assert str(locate_target.get("hitLevel") or "") == "block"
    assert str(primary_evidence.get("block_id") or "") == str(reader_open.get("blockId") or "")
    assert str(primary_evidence.get("anchor_id") or "") == str(reader_open.get("anchorId") or "")
    assert str(primary_evidence.get("heading_path") or "") == str(reader_open.get("headingPath") or "")
    assert str(primary_evidence.get("selection_reason") or "").strip()
    assert primary_evidence == reader_primary
    related_block_ids = list(reader_open.get("relatedBlockIds") or [])
    assert related_block_ids
    visible = list(reader_open.get("visibleAlternatives") or [])
    assert visible
    assert any(str(item.get("blockId") or "").strip() for item in visible)


def test_build_hit_ui_meta_skips_exact_reader_open_resolution_while_pending(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(
        reference_ui,
        "_resolve_refs_exact_candidates",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pending refs should not resolve exact locate blocks")),
    )

    hit = {
        "meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "ref_pack_state": "pending",
            "ref_best_heading_path": "3. Method / 3.1. Background on NeRF",
            "ref_section": "3. Method",
            "anchor_target_kind": "equation",
            "anchor_target_number": 8,
            "ref_show_snippets": [
                "Equation (8) defines the NeRF rendering objective for snapshot reconstruction.",
            ],
            "ref_snippets": [
                "The method overview summarizes the rendering objective used by SCINeRF.",
            ],
            "ref_rank": {
                "llm": 0.0,
                "bm25": 4.7,
                "deep": 0.0,
                "term_bonus": 0.0,
                "semantic_score": 0.0,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt="Where is equation 8 introduced?",
        pdf_root=None,
        lib_store=None,
    )

    reader_open = ui_meta.get("reader_open") or {}
    assert reader_open.get("strictLocate") is False
    assert str(reader_open.get("headingPath") or "") == "3. Method / 3.1. Background on NeRF"
    assert not str(reader_open.get("blockId") or "").strip()
    assert not str(reader_open.get("anchorId") or "").strip()
    assert list(reader_open.get("visibleAlternatives") or [])


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


def test_resolve_refs_exact_candidates_llm_can_reorder_ambiguous_block_choice(tmp_path, monkeypatch):
    md_path = tmp_path / "ambiguous.en.md"
    md_path.write_text("# Demo\n", encoding="utf-8")

    blocks = [
        {
            "block_id": "blk_method",
            "anchor_id": "a_method",
            "heading_path": "Method",
            "kind": "paragraph",
            "text": "ADMM is only mentioned here as a generic optimization family.",
        },
        {
            "block_id": "blk_related",
            "anchor_id": "a_related",
            "heading_path": "Related Work",
            "kind": "blockquote",
            "text": "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
        },
    ]

    def fake_load_source_blocks(_path):
        return blocks

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del snippet, kwargs
        if heading_path == "Method":
            return [
                {"score": 0.81, "block": blocks[0]},
                {"score": 0.77, "block": blocks[1]},
            ]
        return [
            {"score": 0.79, "block": blocks[1]},
            {"score": 0.76, "block": blocks[0]},
        ]

    monkeypatch.setattr(reference_ui, "load_source_blocks", fake_load_source_blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)
    monkeypatch.setattr(reference_ui, "_refs_locate_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_should_try_refs_locate_llm", lambda rows: True)
    monkeypatch.setattr(reference_ui, "_llm_pick_refs_exact_candidate_index", lambda **kwargs: 2)

    out = reference_ui._resolve_refs_exact_candidates(
        prompt="Where is ADMM discussed in this paper?",
        source_path=str(md_path),
        anchor_target_kind="",
        anchor_target_number=0,
        primary_candidate={
            "headingPath": "Method",
            "snippet": "ADMM is discussed in this paper.",
            "highlightSnippet": "ADMM is discussed in this paper.",
        },
        secondary_candidates=[
            {
                "headingPath": "Related Work",
                "snippet": "alternating direction method of multipliers (ADMM) [4],",
                "highlightSnippet": "alternating direction method of multipliers (ADMM) [4],",
            }
        ],
    )

    assert out
    assert "Related Work" in str(out[0].get("headingPath") or "")
    assert "ADMM" in str(out[0].get("snippet") or "")


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


def test_enrich_refs_payload_sorts_hits_by_ui_score_for_display(monkeypatch):
    refs = {
        31: {
            "prompt": "Which papers are most relevant?",
            "hits": [
                {
                    "text": "Lower scoring hit.",
                    "meta": {
                        "source_path": r"db\A\A.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 78.0, "bm25": 4.0, "deep": 1.1, "term_bonus": 0.2, "semantic_score": 7.3},
                    },
                },
                {
                    "text": "Higher scoring hit.",
                    "meta": {
                        "source_path": r"db\B\B.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 86.0, "bm25": 5.0, "deep": 1.5, "term_bonus": 0.6, "semantic_score": 8.0},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        score = 7.2 if source_path.endswith(r"A\A.en.md") else 8.7
        return {
            "display_name": source_path,
            "heading_path": "Method",
            "summary_line": str(hit.get("text") or ""),
            "why_line": "relevant",
            "score": score,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(31) or {}).get("hits") or [])

    assert len(hits) == 2
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"B\B.en.md")
    assert str((((hits[1].get("meta") if isinstance(hits[1].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"A\A.en.md")


def test_enrich_refs_payload_llm_can_rerank_ambiguous_top_hits(monkeypatch):
    refs = {
        32: {
            "prompt": "Which paper in my library most directly discusses ADMM?",
            "hits": [
                {
                    "text": "ADMM is mentioned here only as a generic optimization family.",
                    "meta": {
                        "source_path": r"db\MethodPaper\MethodPaper.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 84.0, "bm25": 5.0, "deep": 1.5, "term_bonus": 0.4, "semantic_score": 7.9},
                    },
                },
                {
                    "text": "Explicit ADMM citation in related work.",
                    "meta": {
                        "source_path": r"db\RelatedWorkPaper\RelatedWorkPaper.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 83.5, "bm25": 4.8, "deep": 1.4, "term_bonus": 0.4, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "ADMM appears in background discussion but without a direct explanatory citation.",
                    "meta": {
                        "source_path": r"db\BackgroundPaper\BackgroundPaper.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 80.0, "bm25": 4.0, "deep": 1.0, "term_bonus": 0.2, "semantic_score": 7.2},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        if source_path.endswith(r"MethodPaper\MethodPaper.en.md"):
            score = 8.42
            heading = "Method"
        elif source_path.endswith(r"RelatedWorkPaper\RelatedWorkPaper.en.md"):
            score = 8.36
            heading = "Related Work"
        else:
            score = 7.61
            heading = "Background"
        return {
            "display_name": source_path,
            "heading_path": heading,
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": score,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "citation_meta": {"title": Path(source_path).stem},
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_should_try_refs_hit_rerank", lambda prompt, hits: True)
    monkeypatch.setattr(reference_ui, "_llm_rerank_refs_hit_order", lambda **kwargs: (2, 1, 3))
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(32) or {}).get("hits") or [])

    assert len(hits) == 3
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"RelatedWorkPaper\RelatedWorkPaper.en.md")
    assert str((((hits[1].get("meta") if isinstance(hits[1].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"MethodPaper\MethodPaper.en.md")


def test_enrich_refs_payload_filters_irrelevant_hits_for_explicit_term_prompt(monkeypatch):
    refs = {
        33: {
            "prompt": "Which paper in my library most directly discusses ADMM? Please point me to the source section.",
            "hits": [
                {
                    "text": "Generic optimization discussion without the requested term.",
                    "meta": {
                        "source_path": r"db\MethodPaper\MethodPaper.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 85.0, "bm25": 5.0, "deep": 1.5, "term_bonus": 0.4, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4].",
                    "meta": {
                        "source_path": r"db\RelatedWorkPaper\RelatedWorkPaper.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 84.0, "bm25": 4.9, "deep": 1.4, "term_bonus": 0.4, "semantic_score": 7.7},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        summary = str(hit.get("text") or "")
        return {
            "display_name": source_path,
            "heading_path": "Related Work" if "RelatedWorkPaper" in source_path else "Method",
            "summary_line": summary,
            "why_line": "candidate",
            "score": 8.5 if "MethodPaper" in source_path else 8.2,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(33) or {}).get("hits") or [])

    assert len(hits) == 1
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"RelatedWorkPaper\RelatedWorkPaper.en.md")


def test_enrich_refs_payload_prefers_prompt_named_source_even_if_raw_score_is_lower(monkeypatch):
    refs = {
        34: {
            "prompt": "In the SCINeRF paper, where is ADMM discussed? Please point me to the source section.",
            "hits": [
                {
                    "text": "Background optimization mention.",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 85.0, "bm25": 5.2, "deep": 1.6, "term_bonus": 0.4, "semantic_score": 7.9},
                    },
                },
                {
                    "text": "ADMM is not mentioned in the SCINeRF snippets we retrieved.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 82.0, "bm25": 4.6, "deep": 1.2, "term_bonus": 0.2, "semantic_score": 7.4},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        if "SCINeRF" in source_path:
            score = 8.1
            display_name = "SCINeRF.pdf"
            heading = "2. Related Work"
        else:
            score = 8.8
            display_name = "NatPhoton.pdf"
            heading = "Abstract"
        return {
            "display_name": display_name,
            "heading_path": heading,
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": score,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(34) or {}).get("hits") or [])

    assert len(hits) == 1
    assert "SCINeRF" in str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or ""))


def test_enrich_refs_payload_drops_single_negative_reason_hit_for_explicit_term_prompt(monkeypatch):
    refs = {
        34: {
            "prompt": "Which paper in my library most directly discusses ADMM? Please point me to the source section.",
            "hits": [
                {
                    "text": "This paper proposes NeRF-based SCI reconstruction from a snapshot measurement.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 32.0, "bm25": 19.0, "deep": 32.0, "term_bonus": 0.0, "semantic_score": 8.3},
                        "ref_pack": {"why": "The paper does not mention ADMM at all and instead focuses on NeRF-based SCI reconstruction."},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": "SCINeRF.pdf",
            "heading_path": "3. Method / 3.1. Background on NeRF",
            "summary_line": "This paper proposes NeRF-based SCI reconstruction from a snapshot measurement.",
            "why_line": "The paper does not mention ADMM at all and instead focuses on NeRF-based SCI reconstruction.",
            "score": 3.2,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(34) or {}).get("hits") or [])

    assert hits == []


def test_enrich_refs_payload_filters_pending_hits_by_prompt_focus_too(monkeypatch):
    refs = {
        35: {
            "prompt": "Which paper in my library most directly discusses ADMM? Please point me to the source section.",
            "hits": [
                {
                    "text": "Generic pending hit without the requested term.",
                    "meta": {
                        "source_path": r"db\PendingA\PendingA.en.md",
                        "ref_pack_state": "pending",
                        "ref_rank": {"llm": 0.0, "bm25": 8.0, "deep": 6.0, "term_bonus": 0.0, "semantic_score": 0.0},
                    },
                },
                {
                    "text": "Pending hit mentioning alternating direction method of multipliers (ADMM).",
                    "meta": {
                        "source_path": r"db\PendingB\PendingB.en.md",
                        "ref_pack_state": "pending",
                        "ref_rank": {"llm": 0.0, "bm25": 7.9, "deep": 5.8, "term_bonus": 0.0, "semantic_score": 0.0},
                    },
                },
            ],
        }
    }

    observed_sources: list[str] = []

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        observed_sources.append(source_path)
        return {
            "display_name": source_path,
            "heading_path": "Method",
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": None,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(35) or {}).get("hits") or [])

    assert len(hits) == 1
    assert observed_sources == [r"db\PendingB\PendingB.en.md"]
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"PendingB\PendingB.en.md")


def test_enrich_refs_payload_named_source_prompt_still_requires_non_source_focus_term(monkeypatch):
    refs = {
        36: {
            "prompt": "In the SCINeRF paper, where is ADMM discussed? Please point me to the source section.",
            "hits": [
                {
                    "text": "SCINeRF method overview without the requested optimization term.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "pending",
                        "ref_rank": {"llm": 0.0, "bm25": 8.0, "deep": 6.1, "term_bonus": 0.0, "semantic_score": 0.0},
                    },
                },
                {
                    "text": "NatPhoton background discussion without the requested optimization method.",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md",
                        "ref_pack_state": "pending",
                        "ref_rank": {"llm": 0.0, "bm25": 7.9, "deep": 6.0, "term_bonus": 0.0, "semantic_score": 0.0},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        display = "SCINeRF.pdf" if "SCINeRF" in source_path else "NatPhoton.pdf"
        return {
            "display_name": display,
            "heading_path": "Method",
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": None,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(36) or {}).get("hits") or [])

    assert hits == []


def test_refs_prompt_focus_terms_extracts_descriptive_phrase_for_library_query():
    terms = reference_ui._refs_prompt_focus_terms(
        "Which paper in my library most directly discusses dynamic supersampling? Please point me to the source section."
    )

    assert "dynamic supersampling" in terms


def test_refs_prompt_focus_terms_extracts_definition_phrase_for_library_query():
    terms = reference_ui._refs_prompt_focus_terms(
        "Which paper in my library most directly defines dynamic supersampling?"
    )

    assert "dynamic supersampling" in terms


def test_refs_prompt_focus_terms_extracts_compare_phrase_for_library_query():
    terms = reference_ui._refs_prompt_focus_terms(
        "Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?"
    )

    assert "hadamard single pixel imaging and fourier single pixel imaging" in terms
    assert "hadamard single pixel imaging" in terms
    assert "fourier single pixel imaging" in terms


def test_basis_meta_auto_uses_existing_card_language_even_for_english_prompt(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_ui, "_refs_card_ui_locale_pref", lambda: "")

    why_meta = reference_ui._build_ref_why_basis_meta(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        why_generation="deterministic_grounded",
        why_line="这条命中直接比较了 Hadamard 与 Fourier 两种 single-pixel imaging 方案。",
    )
    summary_meta = reference_ui._build_ref_summary_basis_meta(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        summary_kind="guide",
        summary_generation="deterministic_grounded",
        summary_line="这篇论文系统比较了 Hadamard single-pixel imaging 和 Fourier single-pixel imaging。",
    )

    assert "focus-term alignment" not in str(why_meta.get("why_basis") or "")
    assert "matched section evidence" not in str(summary_meta.get("summary_basis") or "")
    assert str(why_meta.get("why_basis") or "").strip()
    assert str(summary_meta.get("summary_basis") or "").strip()


def test_basis_meta_auto_uses_ui_locale_when_no_card_language_signal(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_ui, "_refs_card_ui_locale_pref", lambda: "zh")

    why_meta = reference_ui._build_ref_why_basis_meta(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        why_generation="deterministic_grounded",
        why_line="This hit directly compares Hadamard and Fourier single-pixel imaging.",
    )

    assert "focus-term alignment" not in str(why_meta.get("why_basis") or "")
    assert str(why_meta.get("why_basis") or "").strip()


def test_enrich_refs_payload_filters_to_phrase_matched_hit_for_descriptive_query(monkeypatch):
    refs = {
        37: {
            "prompt": "Which paper in my library most directly discusses dynamic supersampling? Please point me to the source section.",
            "hits": [
                {
                    "text": "Generic visual perception background without the requested concept.",
                    "meta": {
                        "source_path": r"db\Psychological Review-1954\Psychological Review-1954.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 86.0, "bm25": 5.2, "deep": 1.8, "term_bonus": 0.4, "semantic_score": 7.9},
                    },
                },
                {
                    "text": "The pipeline of the proposed method is shown in Fig. 1.",
                    "meta": {
                        "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 84.5, "bm25": 5.1, "deep": 1.6, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "Spatially variant digital supersampling is introduced as a dynamic supersampling strategy for adaptive single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 83.0, "bm25": 4.8, "deep": 1.5, "term_bonus": 0.3, "semantic_score": 7.7},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        text = str(hit.get("text") or "")
        return {
            "display_name": source_path,
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling" if "SciAdv-2017" in source_path else "Background",
            "summary_line": text,
            "why_line": "candidate",
            "score": 8.8 if "Psychological Review-1954" in source_path else (8.4 if "ICIP-2025-SCIGS" in source_path else 8.1),
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(37) or {}).get("hits") or [])

    assert len(hits) == 1
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"SciAdv-2017\SciAdv-2017.en.md")


def test_enrich_refs_payload_filters_to_definition_matched_hit(monkeypatch):
    refs = {
        371: {
            "prompt": "Which paper in my library most directly defines dynamic supersampling?",
            "hits": [
                {
                    "text": "This conclusion summarizes dynamic scene recovery from a snapshot compressive image.",
                    "meta": {
                        "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 84.0, "bm25": 5.1, "deep": 1.7, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "Dynamic supersampling is defined by shifting pixel boundaries frame by frame so that each frame samples a different subset of spatial information.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.5, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": source_path,
            "heading_path": "3. Spatially variant digital supersampling" if "SciAdv-2017" in source_path else "5. Conclusion",
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": 8.7 if "ICIP-2025-SCIGS" in source_path else 8.2,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(371) or {}).get("hits") or [])

    assert len(hits) == 1
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"SciAdv-2017\SciAdv-2017.en.md")


def test_enrich_refs_payload_definition_prompt_drops_dynamic_only_false_positive(monkeypatch):
    refs = {
        372: {
            "prompt": "Which paper in my library most directly defines dynamic supersampling?",
            "hits": [
                {
                    "text": "While the closed pinhole provides improved lateral resolution, it suffers from increased noise, which typically restricts its use in dynamic imaging at low incident laser powers.",
                    "meta": {
                        "source_path": r"db\LSA-2026\LSA-2026.en.md",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "Data analysis / Noise Equivalent Contrast",
                        "ref_show_snippets": [
                            "## Data analysis / Noise Equivalent Contrast\nWhile the closed pinhole provides improved lateral resolution, it suffers from increased noise, which typically restricts its use in dynamic imaging at low incident laser powers."
                        ],
                        "ref_rank": {"llm": 84.0, "bm25": 5.1, "deep": 1.7, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "If the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "INTRODUCTION / Spatially variant digital supersampling",
                        "ref_show_snippets": [
                            "## Spatially variant digital supersampling\nIf the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene."
                        ],
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.5, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                },
            ],
        }
    }

    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(372) or {}).get("hits") or [])

    assert len(hits) == 1
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(r"SciAdv-2017\SciAdv-2017.en.md")


def test_enrich_refs_payload_llm_relevance_gate_keeps_only_direct_compare_hit(monkeypatch):
    refs = {
        372: {
            "prompt": "Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
            "hits": [
                {
                    "text": "This work uses Fourier single-pixel imaging as a background example but does not compare it against Hadamard sampling.",
                    "meta": {
                        "source_path": r"db\BackgroundPaper\BackgroundPaper.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 84.0, "bm25": 5.1, "deep": 1.7, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "The paper directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging in numerical simulations and experiments.",
                    "meta": {
                        "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.5, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        text = str(hit.get("text") or "")
        return {
            "display_name": source_path,
            "heading_path": "3. Comparison of experiment" if "Fourier single-pixel imaging" in source_path else "2. Related work",
            "summary_line": text,
            "why_line": "candidate",
            "score": 8.4 if "BackgroundPaper" in source_path else 8.2,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_refs_hit_relevance_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_llm_filter_refs_hit_indices", lambda **kwargs: (1,))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(372) or {}).get("hits") or [])

    assert len(hits) == 1
    assert "Fourier single-pixel imaging" in str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or ""))


def test_enrich_refs_payload_llm_relevance_gate_can_hide_all_false_positive_hits(monkeypatch):
    refs = {
        373: {
            "prompt": "Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
            "hits": [
                {
                    "text": "This work mentions Fourier single-pixel imaging in passing.",
                    "meta": {
                        "source_path": r"db\BackgroundPaper\BackgroundPaper.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 84.0, "bm25": 5.1, "deep": 1.7, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                }
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": source_path,
            "heading_path": "2. Related work",
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": 8.3,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_hit_relevance_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_llm_filter_refs_hit_indices", lambda **kwargs: ())

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(373) or {}).get("hits") or [])

    assert hits == []


def test_enrich_refs_payload_keeps_title_aligned_hit_for_specific_topic_query(monkeypatch):
    refs = {
        38: {
            "prompt": "Which paper in my library most directly discusses Fourier single-pixel imaging? Please point me to the source section.",
            "hits": [
                {
                    "text": "Comparison results of single-pixel photography.",
                    "meta": {
                        "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 84.0, "bm25": 5.0, "deep": 1.4, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "Adaptive supersampling for single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.3, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        if "Fourier single-pixel imaging" in source_path:
            display_name = "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf"
            heading = "3. Comparison of experiment / 3.1 Numerical simulations"
            score = 8.2
        else:
            display_name = "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf"
            heading = "INTRODUCTION"
            score = 8.5
        return {
            "display_name": display_name,
            "heading_path": heading,
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": score,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(38) or {}).get("hits") or [])

    assert len(hits) == 1
    assert "Fourier single-pixel imaging" in str((((hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}).get("display_name") or ""))


def test_enrich_refs_payload_polishes_top_hit_card_copy_with_llm(monkeypatch):
    refs = {
        39: {
            "prompt": "Which paper in my library most directly discusses Fourier single-pixel imaging? Please point me to the source section.",
            "hits": [
                {
                    "text": "Fig. 12 compares reconstruction quality for Hadamard and Fourier single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "ref_show_snippets": [
                            "Fig. 12 compares reconstruction quality for Hadamard and Fourier single-pixel imaging.",
                            "The paper analyzes efficiency, robustness, and reconstruction fidelity for both sampling strategies.",
                        ],
                        "ref_rank": {"llm": 84.0, "bm25": 5.0, "deep": 1.4, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                }
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "summary_line": "$$ C(\\mathbf{r}) = \\int ... $$",
            "why_line": "该文在“3.1 Numerical simulations”给出了与“Which paper in my library most...”直接相关的定义、方法或结果信息。",
            "score": 8.4,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_pick_ref_card_summary_fallback", lambda **kwargs: "")
    monkeypatch.setattr(reference_ui, "_llm_ground_ref_why_line", lambda **kwargs: "")
    monkeypatch.setattr(
        reference_ui,
        "_llm_polish_ref_card_copy_v2",
        lambda **kwargs: (
            "该文系统比较了 Hadamard 与 Fourier 单像素成像在重建质量和效率上的差异。",
            "标题与当前问题中的 Fourier single-pixel imaging 直接对齐，且该小节给出了对应比较。",
        ),
    )

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(39) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "Fourier 单像素成像" in str(ui_meta.get("summary_line") or "")
    assert "直接对齐" in str(ui_meta.get("why_line") or "")


def test_enrich_refs_payload_upgrades_generic_why_line_deterministically_without_llm(monkeypatch):
    refs = {
        40: {
            "prompt": "Which paper in my library most directly discusses dynamic supersampling? Please point me to the source section.",
            "hits": [
                {
                    "text": "Spatially variant digital supersampling is introduced for adaptive single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "ready",
                        "ref_show_snippets": [
                            "Spatially variant digital supersampling is introduced for adaptive single-pixel imaging."
                        ],
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.3, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                }
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "summary_line": "Spatially variant digital supersampling is introduced for adaptive single-pixel imaging.",
            "why_line": "该文内容与“Which paper in my library most...”主题一致，可作为当前问题的直接参考依据。",
            "score": 8.2,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: False)

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(40) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "dynamic supersampling" in str(ui_meta.get("why_line") or "").lower()


def test_build_prompt_aligned_ref_why_line_v3_makes_compare_requests_specific(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_ui, "_refs_card_ui_locale_pref", lambda: "")
    out = reference_ui._build_prompt_aligned_ref_why_line_v3(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        display_name="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        heading_path="3. Comparison of experiment / 3.1 Numerical simulations",
        summary_line="The paper compares Hadamard single-pixel imaging and Fourier single-pixel imaging in simulations and experiments.",
        why_line="",
    )

    out_low = str(out or "").lower()
    assert "compare" in out_low
    assert "hadamard" in out_low
    assert "fourier" in out_low


def test_build_prompt_aligned_ref_why_line_v3_keeps_english_for_strong_english_prompt(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_ui, "_refs_card_ui_locale_pref", lambda: "")
    out = reference_ui._build_prompt_aligned_ref_why_line_v3(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        display_name="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        heading_path="3. Comparison of experiment / 3.1 Numerical simulations",
        summary_line="The paper compares Hadamard single-pixel imaging and Fourier single-pixel imaging in simulations and experiments.",
        why_line="",
    )

    assert "This hit directly compares" in str(out or "")


def test_build_hit_ui_meta_prefers_prompt_aligned_why_when_navigation_why_omits_focus_term(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(
        reference_ui,
        "_build_ref_navigation",
        lambda meta, prompt, heading_fallback: {
            "summary_line": "",
            "why": "This section provides a useful entry point for the current question.",
            "find": ["frame", "scene"],
        },
    )

    hit = {
        "text": "## Spatially variant digital supersampling\nIf the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene.",
        "meta": {
            "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
            "ref_pack_state": "ready",
            "ref_best_heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "ref_show_snippets": [
                "## Spatially variant digital supersampling\nIf the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene."
            ],
            "ref_rank": {"bm25": 7.2, "deep": 15.4, "term_bonus": 1.7, "semantic_score": 7.4, "score": 24.8},
        },
    }

    ui = build_hit_ui_meta(
        hit,
        prompt="Which paper in my library most directly defines dynamic supersampling?",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            r"db\SciAdv-2017\SciAdv-2017.en.md": {
                "title": "Adaptive foveated single-pixel imaging with dynamic supersampling",
            }
        },
    )

    why_line = str(ui.get("why_line") or "").lower()
    assert "dynamic supersampling" in why_line


def test_build_hit_ui_meta_prefers_prompt_aligned_why_when_definition_why_only_mentions_section_heading(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)
    monkeypatch.setattr(
        reference_ui,
        "_build_ref_navigation",
        lambda meta, prompt, heading_fallback: {
            "summary_line": "",
            "why": "This section in 'Spatially variant digital supersampling' explains the frame-to-frame scene sampling strategy.",
            "find": ["frame", "scene"],
        },
    )

    hit = {
        "text": "## Spatially variant digital supersampling\nIf the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene.",
        "meta": {
            "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
            "ref_pack_state": "ready",
            "ref_best_heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "ref_show_snippets": [
                "## Spatially variant digital supersampling\nIf the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene."
            ],
            "ref_rank": {"bm25": 7.2, "deep": 15.4, "term_bonus": 1.7, "semantic_score": 7.4, "score": 24.8},
        },
    }

    ui = build_hit_ui_meta(
        hit,
        prompt="Which paper in my library most directly defines dynamic supersampling?",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            r"db\SciAdv-2017\SciAdv-2017.en.md": {
                "title": "Adaptive foveated single-pixel imaging with dynamic supersampling",
            }
        },
    )

    why_line = str(ui.get("why_line") or "").lower()
    assert "dynamic supersampling" in why_line
    assert ("defines or explains" in why_line) or ("定义或解释" in why_line)


def test_build_hit_ui_meta_overrides_title_like_summary_with_prompt_aligned_snippet(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    hit = {
        "text": "### 2.4 Efficiency\nThis section compares Hadamard and Fourier single-pixel imaging in efficiency and sampling trade-offs.",
        "meta": {
            "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
            "ref_pack_state": "failed",
            "ref_rank": {
                "bm25": 7.2,
                "deep": 15.4,
                "term_bonus": 1.7,
                "semantic_score": 7.4,
                "score": 24.8,
            },
            "ref_best_heading_path": "2. Comparison of theory / 2.4 Efficiency",
            "ref_show_snippets": [
                "### 2.2 Basis patterns generation\nFigure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
            ],
        },
    }

    ui = build_hit_ui_meta(
        hit,
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        preloaded_citation_meta={
            r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md": {
                "title": "Hadamard single-pixel imaging versus Fourier single-pixel imaging ZIBANG ZHANG OCIS codes",
                "summary_line": "Hadamard single-pixel imaging versus Fourier single-pixel imaging ZIBANG ZHANG OCIS codes",
                "summary_source": "abstract",
            }
        },
    )

    summary_line = str(ui.get("summary_line") or "")
    assert "comparison between the Hadamard and Fourier basis patterns" in summary_line
    assert "OCIS codes" not in summary_line
    assert str(ui.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    reader_open = ui.get("reader_open") or {}
    assert str(reader_open.get("headingPath") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_build_hit_ui_meta_rebinds_summary_heading_to_exact_loc_path(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    hit = {
        "text": "This paper compares Hadamard and Fourier single-pixel imaging.",
        "meta": {
            "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
            "ref_pack_state": "ready",
            "ref_best_heading_path": "2. Comparison of theory / 2.4 Efficiency",
            "ref_show_snippets": [
                "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
            ],
            "ref_locs": [
                {
                    "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    "snippet": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
                }
            ],
            "ref_rank": {
                "llm": 81.0,
                "bm25": 6.8,
                "deep": 13.4,
                "term_bonus": 2.1,
                "semantic_score": 8.1,
            },
        },
    }

    ui = build_hit_ui_meta(
        hit,
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
    )

    assert str(ui.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert str(ui.get("section_label") or "") == "2. Comparison of theory"
    assert str(ui.get("subsection_label") or "") == "2.2 Basis patterns generation"
    primary_evidence = ui.get("primary_evidence") or {}
    assert str(primary_evidence.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert str(primary_evidence.get("selection_reason") or "") == "prompt_aligned"
    reader_open = ui.get("reader_open") or {}
    assert str(reader_open.get("headingPath") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert (reader_open.get("primaryEvidence") or {}) == primary_evidence


def test_select_primary_ref_evidence_prefers_prompt_aligned_heading_over_fallback_compare_summary(monkeypatch):
    monkeypatch.setattr(
        reference_ui,
        "_build_ref_navigation",
        lambda *args, **kwargs: {
            "what": "",
            "summary_line": "",
            "why": "",
            "find": [],
        },
    )
    monkeypatch.setattr(
        reference_ui,
        "_fallback_ref_ui_summary_line",
        lambda *args, **kwargs: (
            "该研究比较了哈达玛（Hadamard）与傅里叶（Fourier）基函数在单像素成像中的特性："
            "哈达玛基仅包含水平与垂直方向特征，而傅里叶基兼具水平、垂直及斜向特征。"
        ),
    )
    monkeypatch.setattr(
        reference_ui,
        "_choose_prompt_aligned_ref_summary_candidate",
        lambda *args, **kwargs: {
            "summary": "本节介绍了基模式的生成方法，通过图1对比了哈达玛（Hadamard）与傅里叶（Fourier）基模式的特性差异。",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
        },
    )

    out = reference_ui._select_primary_ref_evidence(
        meta={
            "ref_best_heading_path": "2. Comparison of theory / 2.4 Efficiency",
            "top_heading": "2. Comparison of theory",
        },
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        source_path=r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
        display_name="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        citation_meta={"title": "Hadamard single-pixel imaging versus Fourier single-pixel imaging"},
        heading_context={
            "heading_path": "2. Comparison of theory / 2.4 Efficiency",
            "heading": "2. Comparison of theory",
            "section_label": "2. Comparison of theory",
            "subsection_label": "2.4 Efficiency",
        },
        anchor_target_kind="",
        anchor_target_number=0,
        allow_exact_locate=False,
    )

    assert str(out.get("summary_source") or "") == "prompt_aligned"
    assert bool(out.get("used_prompt_aligned_summary")) is True
    assert str(out.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert str(out.get("subsection_label") or "") == "2.2 Basis patterns generation"


def test_build_hit_ui_meta_infers_heading_from_source_blocks_for_body_only_compare_snippet(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    md_path = tmp_path / "compare_fixture.en.md"
    md_path.write_text(
        "# Compare Fixture\n\n"
        "## 2. Comparison of theory\n\n"
        "### 2.2 Basis patterns generation\n\n"
        "The difference can be summarized as follows: 1) Hadamard basis patterns are binary while Fourier basis patterns are grayscale; "
        "2) Hadamard basis patterns only have horizontal and vertical features while Fourier basis patterns have horizontal, vertical, and oblique features.\n\n"
        "### 2.4 Efficiency\n\n"
        "We refer efficient single-pixel imaging to a technique that allows one to reconstruct a sharp image with a small number of measurements.\n",
        encoding="utf-8",
    )

    hit = {
        "text": "This paper compares Hadamard and Fourier single-pixel imaging.",
        "meta": {
            "source_path": str(md_path),
            "ref_pack_state": "ready",
            "ref_best_heading_path": "2. Comparison of theory / 2.4 Efficiency",
            "ref_show_snippets": [
                "### 2.4 Efficiency\nWe refer efficient single-pixel imaging to a technique that allows one to reconstruct a sharp image with a small number of measurements.",
                "The difference can be summarized as follows: 1) Hadamard basis patterns are binary while Fourier basis patterns are grayscale; 2) Hadamard basis patterns only have horizontal and vertical features while Fourier basis patterns have horizontal, vertical, and oblique features.",
            ],
            "ref_rank": {
                "llm": 84.0,
                "bm25": 6.9,
                "deep": 14.1,
                "term_bonus": 2.0,
                "semantic_score": 8.3,
            },
        },
    }

    ui = build_hit_ui_meta(
        hit,
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
    )

    assert str(ui.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    reader_open = ui.get("reader_open") or {}
    assert str(reader_open.get("headingPath") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_choose_prompt_aligned_ref_summary_candidate_ignores_front_matter_boilerplate(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    meta = {
        "ref_best_heading_path": "2. Comparison of theory / 2.4 Efficiency",
        "ref_show_snippets": [
            "horizontal and vertical features while Fourier basis patterns have horizontal, vertical, and oblique features;",
        ],
        "ref_snippets": [
            "### 2.2 Basis patterns generation\nFigure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
        ],
        "ref_overview_snippets": [
            "Hadamard single-pixel imaging versus Fourier single-pixel imaging **ZIBANG ZHANG**,$^{1}$ **XUEYING WANG**,$^{1}$ © 2017 Optical Society of America **OCIS codes**: (110.1758) Computational imaging;",
        ],
    }

    out = reference_ui._choose_prompt_aligned_ref_summary_candidate(
        meta,
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        source_path=r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
        citation_meta={},
    )

    assert "OCIS codes" not in str(out.get("summary") or "")
    assert "comparison between the Hadamard and Fourier basis patterns" in str(out.get("summary") or "")
    assert str(out.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_resolve_refs_exact_candidates_prefers_primary_heading_match_when_scores_are_close(tmp_path, monkeypatch):
    md_path = tmp_path / "compare_exact.en.md"
    md_path.write_text("# Compare\n", encoding="utf-8")

    blocks = [
        {
            "block_id": "blk_21",
            "anchor_id": "a_21",
            "heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
            "kind": "paragraph",
            "text": "Hadamard and Fourier single-pixel imaging are introduced in principle terms.",
        },
        {
            "block_id": "blk_22",
            "anchor_id": "a_22",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
            "kind": "paragraph",
            "text": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
        },
    ]

    def fake_load_source_blocks(_path):
        return blocks

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del _blocks, snippet, kwargs
        if "2.2 Basis patterns generation" in str(heading_path or ""):
            return [
                {"score": 0.89, "block": blocks[0]},
                {"score": 0.84, "block": blocks[1]},
            ]
        return [
            {"score": 0.88, "block": blocks[0]},
            {"score": 0.80, "block": blocks[1]},
        ]

    monkeypatch.setattr(reference_ui, "load_source_blocks", fake_load_source_blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)

    out = reference_ui._resolve_refs_exact_candidates(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        source_path=str(md_path),
        anchor_target_kind="",
        anchor_target_number=0,
        primary_candidate={
            "headingPath": "2. Comparison of theory / 2.2 Basis patterns generation",
            "snippet": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
            "highlightSnippet": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
        },
        secondary_candidates=[
            {
                "headingPath": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                "snippet": "Hadamard and Fourier single-pixel imaging are introduced in principle terms.",
                "highlightSnippet": "Hadamard and Fourier single-pixel imaging are introduced in principle terms.",
            }
        ],
        allow_llm_disambiguation=False,
    )

    assert out
    assert str(out[0].get("headingPath") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert str(out[0].get("blockId") or "") == "blk_22"


def test_choose_prompt_aligned_ref_summary_prefers_definition_heading_with_explanatory_sentence(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    meta = {
        "ref_show_snippets": [
            "## Spatially variant digital supersampling\nIf the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene.",
            "we could reconstruct pairs of subframes with identical pixel footprints and look for changes between these to track motion. However, this strategy would reduce the supersampling rate by a factor of 2.",
        ]
    }

    out = reference_ui._choose_prompt_aligned_ref_summary(
        meta,
        prompt="Which paper in my library most directly defines dynamic supersampling?",
        source_path=r"db\SciAdv-2017\SciAdv-2017.en.md",
        citation_meta={"title": "Adaptive foveated single-pixel imaging with dynamic supersampling"},
    )

    out_low = str(out or "").lower()
    assert "dynamic supersampling" in out_low
    assert ("pixel boundaries" in out_low) or ("samples a different subset" in out_low)


def test_choose_prompt_aligned_ref_summary_candidate_skips_partial_dynamic_match_for_definition_prompt(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    meta = {
        "ref_show_snippets": [
            "## Data analysis / Noise Equivalent Contrast\nWhile the closed pinhole provides improved lateral resolution, it suffers from increased noise, which typically restricts its use in dynamic imaging at low incident laser powers.",
        ]
    }

    out = reference_ui._choose_prompt_aligned_ref_summary_candidate(
        meta,
        prompt="Which paper in my library most directly defines dynamic supersampling?",
        source_path=r"db\LSA-2026\LSA-2026.en.md",
        citation_meta={"title": "Interferometric Image Scanning Microscopy for label-free imaging at 120 nm lateral resolution inside live cells"},
    )

    assert out == {}


def test_choose_prompt_aligned_ref_summary_prefers_fourier_specific_sentence(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    meta = {
        "ref_show_snippets": [
            "### 2.4 Efficiency\nWe refer efficient single-pixel imaging to a technique that allows one to reconstruct a sharp image with a small number of measurements. Additionally, highly efficient single-pixel imaging enables time-lapse imaging.",
            "horizontal and vertical features while Fourier basis patterns have horizontal, vertical, and oblique features; 3) Fourier basis patterns are strictly periodical while Hadamard basis patterns are not.",
        ],
        "ref_snippets": [
            "### 2.2 Basis patterns generation\nFigure 1 shows the comparison between the Hadamard and Fourier basis patterns. The difference can be summarized as follows: 1) Hadamard basis patterns are binary while Fourier basis patterns are grayscale.",
        ],
    }

    out = reference_ui._choose_prompt_aligned_ref_summary(
        meta,
        prompt="Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?",
        source_path=r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
        citation_meta={"title": "Hadamard single-pixel imaging versus Fourier single-pixel imaging"},
    )

    out_low = str(out or "").lower()
    assert "fourier" in out_low
    assert ("hadamard" in out_low) or ("comparison" in out_low)


def test_enrich_refs_payload_prefers_descriptive_summary_candidate_without_llm(monkeypatch):
    refs = {
        41: {
            "prompt": "Which paper in my library most directly discusses Fourier single-pixel imaging? Please point me to the source section.",
            "hits": [
                {
                    "text": "Fig. 12 compares reconstruction quality for Hadamard and Fourier single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "ref_show_snippets": [
                            "Fig. 12 compares reconstruction quality for Hadamard and Fourier single-pixel imaging.",
                            "The paper analyzes efficiency, robustness, and reconstruction fidelity for Hadamard and Fourier single-pixel imaging.",
                        ],
                        "ref_rank": {"llm": 84.0, "bm25": 5.0, "deep": 1.4, "term_bonus": 0.3, "semantic_score": 7.8},
                    },
                }
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "summary_line": "$$ C(\\mathbf{r}) = \\int ... $$",
            "why_line": "candidate",
            "score": 8.4,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: False)

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(41) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert str(ui_meta.get("summary_line") or "").startswith("The paper analyzes efficiency")


def test_enrich_refs_payload_fast_ready_skips_translation_and_citation_prefetch(monkeypatch):
    refs = {
        99: {
            "prompt": "Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
            "hits": [
                {
                    "text": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
                    "meta": {
                        "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        "ref_show_snippets": [
                            "### 2.2 Basis patterns generation\nFigure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
                        ],
                        "ref_rank": {
                            "llm": 82.0,
                            "bm25": 6.5,
                            "deep": 13.1,
                            "term_bonus": 2.0,
                            "semantic_score": 8.0,
                        },
                    },
                }
            ],
        }
    }

    monkeypatch.setattr(reference_ui, "_prefer_zh_ref_card_locale", lambda *args: True)

    def fail_translate(_text):
        raise AssertionError("fast ready path should not call _translate_summary_to_zh")

    def fail_prefetch(*args, **kwargs):
        raise AssertionError("fast ready path should not call _prefetch_refs_citation_meta")

    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", fail_translate)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", fail_prefetch)
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: False)

    out = enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        allow_expensive_llm_for_ready=False,
        allow_exact_locate=False,
    )

    hits = list((out.get(99) or {}).get("hits") or [])
    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "Hadamard" in str(ui_meta.get("summary_line") or "")
    assert str(ui_meta.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_enrich_refs_payload_bounded_full_skips_heavy_refine_but_keeps_exact_locate(monkeypatch):
    refs = {
        101: {
            "prompt": "Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
            "hits": [
                {
                    "text": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
                    "meta": {
                        "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {
                            "llm": 82.0,
                            "bm25": 6.5,
                            "deep": 13.1,
                            "term_bonus": 2.0,
                            "semantic_score": 8.0,
                        },
                    },
                }
            ],
        }
    }
    calls: dict[str, object] = {}

    def fail_prefetch(*args, **kwargs):
        raise AssertionError("bounded_full should not prefetch citation meta")

    def fail_rerank(**kwargs):
        raise AssertionError("bounded_full should not call _maybe_llm_rerank_refs_hits")

    def fail_filter(**kwargs):
        raise AssertionError("bounded_full should not call _maybe_llm_filter_refs_hits")

    def fail_polish(**kwargs):
        raise AssertionError("bounded_full should not call _maybe_polish_refs_card_copy")

    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", fail_prefetch)
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", fail_rerank)
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", fail_filter)
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", fail_polish)

    def fake_build_hit_ui_meta(*args, **kwargs):
        del args
        calls["allow_expensive_llm"] = kwargs.get("allow_expensive_llm")
        calls["allow_exact_locate"] = kwargs.get("allow_exact_locate")
        return {
            "summary_line": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)

    out = enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        render_variant="bounded_full",
    )

    hits = list((out.get(101) or {}).get("hits") or [])
    assert len(hits) == 1
    assert calls == {
        "allow_expensive_llm": False,
        "allow_exact_locate": True,
    }
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert str(ui_meta.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_filter_refs_hits_by_prompt_focus_compare_prefers_explicit_versus_title_match():
    prompt = "Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?"
    hits = [
        {
            "meta": {
                "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
            },
            "ui_meta": {
                "display_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
                "heading_path": "2. Comparison of theory / 2.4 Efficiency",
                "summary_line": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
                "why_line": "This hit directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging.",
            },
        },
        {
            "meta": {
                "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
            },
            "ui_meta": {
                "display_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
                "heading_path": "Abstract / Acquisition and image reconstruction strategies.",
                "summary_line": "In the case of Fourier single-pixel imaging, it is possible to employ three-step phase-shifting. When using the Hadamard basis, one typically requires a stable differential measurement.",
                "why_line": "This hit mentions both Hadamard and Fourier methods.",
            },
        },
    ]

    filtered = reference_ui._filter_refs_hits_by_prompt_focus(prompt, hits)

    assert len(filtered) == 1
    ui_meta = (filtered[0].get("ui_meta") if isinstance(filtered[0].get("ui_meta"), dict) else {}) or {}
    assert "versus" in str(ui_meta.get("display_name") or "").lower()


def test_filter_refs_hits_by_prompt_focus_drops_single_generic_non_matching_hit():
    prompt = "Which paper in my library most directly discusses ADMM? Please point me to the source section."
    hits = [
        {
            "meta": {
                "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            },
            "ui_meta": {
                "display_name": "CVPR-2024-SCINeRF.pdf",
                "heading_path": "2. Related Work",
                "summary_line": "This paper studies snapshot compressive imaging and neural radiance field reconstruction.",
                "why_line": "This hit is directly relevant because the related work section discusses prior NeRF reconstruction methods.",
            },
        }
    ]

    filtered = reference_ui._filter_refs_hits_by_prompt_focus(prompt, hits)

    assert filtered == []


def test_filter_refs_hits_by_prompt_focus_drops_focus_term_that_only_appears_negated():
    prompt = "Besides this paper, what other papers in my library discuss ADMM?"
    hits = [
        {
            "meta": {
                "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            },
            "ui_meta": {
                "display_name": "ICIP-2025-SCIGS.pdf",
                "heading_path": "2. Related Work",
                "summary_line": "This paper proposes a reconstruction method for snapshot compressive imaging without relying on ADMM.",
                "why_line": "This hit is directly relevant because it mentions ADMM in the related work discussion.",
            },
        }
    ]

    filtered = reference_ui._filter_refs_hits_by_prompt_focus(prompt, hits)

    assert filtered == []


def test_enrich_refs_payload_applies_focus_filter_even_for_single_ready_hit(monkeypatch):
    refs = {
        102: {
            "prompt": "Which paper in my library most directly discusses ADMM? Please point me to the source section.",
            "hits": [
                {
                    "text": "Volumetric rendering details for SCI-NeRF.",
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {
                            "llm": 52.4,
                            "bm25": 4.2,
                            "deep": 9.1,
                            "term_bonus": 0.2,
                            "semantic_score": 5.5,
                        },
                    },
                }
            ],
        }
    }

    def fake_build_hit_ui_meta(*args, **kwargs):
        del args, kwargs
        return {
            "display_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "Abstract",
            "summary_line": "In this paper, we explore snapshot compressive imaging for neural radiance fields.",
            "why_line": "该文在“Abstract”给出了与“Which paper in my library most...”直接相关的定义、方法或结果信息。",
            "score": 5.2,
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})

    out = enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        render_variant="bounded_full",
    )

    assert list((out.get(102) or {}).get("hits") or []) == []


def test_summary_line_needs_polish_for_surface_like_caption_and_raw_heading():
    prompt = "Which paper in my library most directly discusses Fourier single-pixel imaging? Please point me to the source section."
    title = "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf"

    assert reference_ui._summary_line_needs_polish(
        prompt=prompt,
        title=title,
        summary_line="Optics EXPRESS Fig. Comparison results of single-pixel photography.",
    )
    assert reference_ui._summary_line_needs_polish(
        prompt=prompt,
        title=title,
        summary_line="## Spatially variant digital supersampling If the positions of the pixel boundaries are modified from one frame to the next, then each frame samples a different subset of the spatial information in the scene.",
    )


def test_generic_ref_why_line_detects_prompt_echo_template():
    assert reference_ui._looks_generic_ref_why_line(
        "该文在“3.1 Numerical simulations”给出了与“Which paper in my library most...”直接相关的定义、方法或结果信息。"
    )


def test_enrich_refs_payload_can_polish_from_hit_text_without_extra_snippets(monkeypatch):
    refs = {
        42: {
            "prompt": "Which paper in my library most directly discusses dynamic supersampling? Please point me to the source section.",
            "hits": [
                {
                    "text": "Spatially variant digital supersampling is introduced for adaptive single-pixel imaging so each frame captures a different subset of spatial information and progressively refines local resolution where needed.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.3, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                }
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "summary_line": "## Spatially variant digital supersampling If the positions of the pixel boundaries are modified from one frame to the next...",
            "why_line": "该文在“Spatially variant digital supersampling”给出了与“Which paper in my library most...”直接相关的定义、方法或结果信息。",
            "score": 8.2,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_pick_ref_card_summary_fallback", lambda **kwargs: "")
    monkeypatch.setattr(reference_ui, "_llm_ground_ref_why_line", lambda **kwargs: "")
    monkeypatch.setattr(
        reference_ui,
        "_llm_polish_ref_card_copy_v2",
        lambda **kwargs: (
            "The paper explicitly defines dynamic supersampling as shifting pixel boundaries frame-by-frame to capture complementary spatial information.",
            "This is directly relevant because the section names and explains dynamic supersampling itself rather than only mentioning it in passing.",
        ),
    )

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(42) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "defines dynamic supersampling" in str(ui_meta.get("summary_line") or "")
    assert "directly relevant" in str(ui_meta.get("why_line") or "").lower()


def test_enrich_refs_payload_skips_expensive_llm_refine_while_hits_are_pending(monkeypatch):
    refs = {
        43: {
            "prompt": "Which paper in my library most directly discusses dynamic supersampling? Please point me to the source section.",
            "hits": [
                {
                    "text": "Spatially variant digital supersampling is introduced for adaptive single-pixel imaging so each frame captures a different subset of spatial information and progressively refines local resolution where needed.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "pending",
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.3, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                }
            ],
        }
    }

    observed_allow_flags: list[bool] = []

    def fake_build_hit_ui_meta(hit, **kwargs):
        observed_allow_flags.append(bool(kwargs.get("allow_expensive_llm")))
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "summary_line": "## Spatially variant digital supersampling If the positions of the pixel boundaries are modified from one frame to the next...",
            "why_line": "该文在“Spatially variant digital supersampling”给出了与“Which paper in my library most...”直接相关的定义、方法或结果信息。",
            "score": 8.2,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: (_ for _ in ()).throw(AssertionError("pending refs should not run llm filter")))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: (_ for _ in ()).throw(AssertionError("pending refs should not run llm polish")))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(43) or {}).get("hits") or [])

    assert len(hits) == 1
    assert observed_allow_flags == [False]
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "Spatially variant digital supersampling" in str(ui_meta.get("summary_line") or "")
    assert "candidate" not in str(ui_meta.get("why_line") or "").lower()


def test_build_ref_summary_surface_meta_uses_guide_label_for_non_abstract_cards(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "zh")
    out = reference_ui._build_ref_summary_surface_meta(
        prompt="这篇文章里 dynamic supersampling 是怎么定义的？",
        summary_kind="guide",
    )
    assert out["summary_label"] == "导读"
    assert "命中章节" in out["summary_title"]


def test_build_ref_summary_surface_meta_auto_uses_summary_language(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "auto")
    out = reference_ui._build_ref_summary_surface_meta(
        prompt="Which paper in my library discusses dynamic supersampling?",
        summary_kind="guide",
        summary_line="该研究提出了一种空间可变的数字超采样方法。",
    )
    assert out["summary_label"] == "导读"


def test_metadata_summary_line_for_ref_card_explains_missing_abstract(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "en")
    out = reference_ui._metadata_summary_line_for_ref_card(
        {
            "title": "A Paper Without Abstract",
            "venue": "CVPR",
            "year": "2024",
            "authors": "Jane Doe, John Smith",
        },
        prompt="Summarize this reference card.",
    )
    assert "No abstract is available" in out
    assert "metadata only" in out


def test_build_ref_summary_basis_meta_describes_llm_abstract(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "zh")
    out = reference_ui._build_ref_summary_basis_meta(
        prompt="请总结这篇论文。",
        summary_kind="abstract",
        summary_generation="llm_abstract",
        summary_line="该研究提出了一种新的成像方法。",
    )
    assert out["summary_generation"] == "llm_abstract"
    assert "LLM" in str(out["summary_basis"] or "")
    assert "abstract" in str(out["summary_basis"] or "")


def test_build_ref_why_basis_meta_describes_llm_grounded_reason(monkeypatch):
    monkeypatch.setattr(reference_ui, "_refs_card_locale_pref", lambda: "zh")
    out = reference_ui._build_ref_why_basis_meta(
        prompt="为什么这篇文献和我的问题相关？",
        why_generation="llm_grounded",
        why_line="这条命中直接解释了 dynamic supersampling 的定义和用途。",
    )
    assert out["why_generation"] == "llm_grounded"
    assert "LLM" in str(out["why_basis"] or "")
    assert "相关性说明" in str(out["why_basis"] or "")


def test_finalize_abstract_summary_line_prefers_llm_summary(monkeypatch):
    monkeypatch.setattr(
        reference_ui,
        "_llm_summarize_abstract_zh",
        lambda **kwargs: "这是一段基于摘要的 LLM 提炼总结。",
    )
    out, generation = reference_ui._finalize_abstract_summary_line(
        title="Test Paper",
        abstract_text="We propose a new imaging method and show improved reconstruction quality.",
    )
    assert out == "这是一段基于摘要的 LLM 提炼总结。"
    assert generation == "llm_abstract"


def test_ensure_summary_line_marks_existing_abstract_as_llm_distilled(monkeypatch):
    monkeypatch.setattr(
        reference_ui,
        "_llm_summarize_abstract_zh",
        lambda **kwargs: "这是一段被重新提炼过的摘要。",
    )
    out = reference_ui._ensure_summary_line(
        {
            "title": "Test Paper",
            "summary_line": "We propose a new imaging method and evaluate it on microscopy data.",
            "summary_source": "abstract",
        },
        allow_crossref_abstract=True,
    )
    assert out["summary_source"] == "abstract"
    assert out["summary_generation"] == "llm_abstract"
    assert out["summary_line"] == "这是一段被重新提炼过的摘要。"


def test_enrich_refs_payload_uses_grounded_llm_for_why_basis_when_hits_are_ready(monkeypatch):
    refs = {
        43: {
            "prompt": "Which paper in my library most directly defines dynamic supersampling?",
            "hits": [
                {
                    "text": "Spatially variant digital supersampling shifts pixel boundaries frame by frame to capture complementary spatial samples.",
                    "meta": {
                        "source_path": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
                        "ref_pack_state": "ready",
                        "ref_show_snippets": [
                            "Spatially variant digital supersampling shifts pixel boundaries frame by frame to capture complementary spatial samples."
                        ],
                        "ref_best_heading_path": "3. Spatially variant digital supersampling",
                        "ref_rank": {"llm": 83.5, "bm25": 4.9, "deep": 1.3, "term_bonus": 0.2, "semantic_score": 7.7},
                    },
                    "ui_meta": {
                        "display_name": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
                        "heading_path": "3. Spatially variant digital supersampling",
                        "summary_line": "This section defines dynamic supersampling by shifting pixel boundaries frame by frame.",
                        "summary_kind": "guide",
                        "summary_label": "Guide",
                        "summary_title": "What This Matched Section Covers",
                        "why_line": "candidate",
                    },
                }
            ],
        }
    }
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(
        reference_ui,
        "_llm_ground_ref_why_line",
        lambda **kwargs: "This hit is directly relevant because the section explicitly defines dynamic supersampling rather than merely citing it.",
    )
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))

    out = reference_ui.enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(43) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "explicitly defines dynamic supersampling" in str(ui_meta.get("why_line") or "")
    assert str(ui_meta.get("why_generation") or "") == "llm_grounded"
    assert "LLM" in str(ui_meta.get("why_basis") or "")
