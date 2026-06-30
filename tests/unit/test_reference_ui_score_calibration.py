from __future__ import annotations

from pathlib import Path

import pytest

import api.reference_card_locale as reference_card_locale
import api.reference_ui as reference_ui
from api.reference_ui import _effective_ui_score, build_hit_ui_meta, enrich_refs_payload, ensure_source_citation_meta


@pytest.fixture(autouse=True)
def _disable_remote_summary_translation(monkeypatch):
    # Keep this suite deterministic and prevent accidental live translation calls from dragging test time.
    cache_clear = getattr(reference_ui._translate_summary_to_zh, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    monkeypatch.setenv("KB_REFS_CARD_LOCALE", "auto")
    monkeypatch.setenv("KB_CITE_SUMMARY_TRANSLATE_ZH", "0")
    yield
    cache_clear = getattr(reference_ui._translate_summary_to_zh, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


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


def test_heading_sanitizer_drops_pdf_shell_noise_without_hiding_real_abstract():
    source_path = r"db\Demo\Demo.en.md"

    assert reference_ui._sanitize_heading_path_ui(
        "A B S T R A C T / 2. Method",
        prompt="How does the method work?",
        source_path=source_path,
    ) == "2. Method"
    assert reference_ui._sanitize_heading_path_ui(
        "ARTICLE INFO / 3. Results",
        prompt="Where are the results?",
        source_path=source_path,
    ) == "3. Results"
    assert reference_ui._sanitize_heading_path_ui(
        "Abstract",
        prompt="Which papers discuss deep learning?",
        source_path=source_path,
    ) == "Abstract"


def test_reader_open_exact_anchor_prefers_requested_caption_over_later_reference(tmp_path: Path):
    md = tmp_path / "CVPR-2024-SCINeRF.en.md"
    md.write_text(
        "\n".join(
            [
                "# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "",
                "## 4. Experiments",
                "### 4.1. Experimental Setup",
                "![Figure 3](./assets/fig3.png)",
                "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera, primary and relay lens, and a DMD.",
                "",
                "### 4.2. Additional Study",
                "![Figure 5](./assets/fig5.png)",
                "**Figure 5.** Qualitative evaluations on the real dataset captured by our system in Fig. 3.",
            ]
        ),
        encoding="utf-8",
    )

    reader_open = reference_ui._build_refs_reader_open_payload(
        meta={
            "ref_show_snippets": [
                "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera, primary and relay lens, and a DMD.",
            ],
            "ref_locs": [
                {"heading_path": "4. Experiments / 4.1. Experimental Setup"},
            ],
        },
        prompt="SCINeRF hardware setup Figure 3",
        source_path=str(md),
        display_name="SCINeRF",
        heading_path="4. Experiments / 4.1. Experimental Setup",
        heading="4.1. Experimental Setup",
        summary_line="This card discusses SCINeRF in Experimental Setup.",
        why_line="",
        anchor_target_kind="figure",
        anchor_target_number=3,
        allow_llm_disambiguation=False,
        allow_exact_locate=True,
    )

    assert "Figure 3" in str(reader_open.get("headingPath") or "")
    assert "Figure 5" not in str(reader_open.get("headingPath") or "")


def test_build_hit_ui_meta_anchor_heading_not_rebound_to_unrelated_summary(monkeypatch, tmp_path: Path):
    md = tmp_path / "CVPR-2024-SCINeRF.en.md"
    md.write_text(
        "\n".join(
            [
                "# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "",
                "## 4. Experiments",
                "### 4.1. Experimental Setup",
                "![Figure 3](./assets/fig3.png)",
                "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera, primary and relay lens, and a DMD.",
                "",
                "## 5. Conclusion",
                "SCINeRF is a novel approach for 3D scene representation learning from a snapshot compressed image.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reference_ui,
        "_choose_prompt_aligned_ref_summary_candidate",
        lambda *args, **kwargs: {
            "summary": "SCINeRF is a novel approach for 3D scene representation learning.",
            "heading_path": "5. Conclusion",
        },
    )

    ui_meta = build_hit_ui_meta(
        {
            "text": "**Figure 3.** Experimental setup for real dataset collection.",
            "score": 28.0,
            "meta": {
                "source_path": str(md),
                "ref_best_heading_path": "4. Experiments / 4.1. Experimental Setup",
                "top_heading": "4.1. Experimental Setup",
                "anchor_target_kind": "figure",
                "anchor_target_number": 3,
                "anchor_match_score": 28.0,
                "explicit_doc_match_score": 8.0,
                "ref_show_snippets": [
                    "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera, primary and relay lens, and a DMD.",
                ],
                "ref_rank": {"bm25": 8.0, "deep": 2.0, "llm": 0.0, "term_bonus": 0.0, "semantic_score": 0.0},
            },
        },
        prompt="SCINeRF hardware setup Figure 3",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )

    assert str(ui_meta["heading_path"]).startswith("4. Experiments / 4.1. Experimental Setup")
    assert "5. Conclusion" not in str(ui_meta["heading_path"])


def test_build_hit_ui_meta_infers_prompt_figure_anchor_for_stale_hit(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    md = tmp_path / "CVPR-2024-SCINeRF.en.md"
    md.write_text(
        "\n".join(
            [
                "# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "",
                "## 4. Experiments",
                "### 4.1. Experimental Setup",
                "Real-world datasets. The setup consists of an iRAYPLE A5402MU90 camera and a FLDISCOVERY F4110 DMD. Fig. 3 shows the experimental setup we used to collect real dataset.",
                "",
                "![Figure 3](./assets/fig3.png)",
                "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera, primary and relay lens, and a DMD.",
                "",
                "### 4.2. Additional Study",
                "**Figure 5.** Qualitative evaluations on the real dataset captured by our system in Fig. 3.",
            ]
        ),
        encoding="utf-8",
    )
    hit = {
        "meta": {
            "source_path": str(md),
            "ref_best_heading_path": "4. Experiments / 4.1. Experimental Setup",
            "ref_section": "4. Experiments",
            "ref_subsection": "4.1. Experimental Setup",
            "ref_show_snippets": [
                "High compression ratio We study the performance of our SCINeRF under different compression ratios.",
            ],
            "ref_rank": {
                "llm": 78.0,
                "bm25": 4.7,
                "deep": 1.2,
                "term_bonus": 0.5,
                "semantic_score": 7.4,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt=(
            "SCINeRF\u7684\u771f\u5b9e\u786c\u4ef6\u5b9e\u9a8c\u88c5\u7f6e\u5305\u542b"
            "\u54ea\u4e9b\u90e8\u4ef6\uff1f\u8bf7\u5bf9\u5e94\u5230\u539f\u6587\u56fe3"
            "\u6216\u5b9e\u9a8c\u8bbe\u7f6e\u3002"
        ),
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=True,
    )

    summary = str(ui_meta.get("summary_line") or "")
    reader_open = ui_meta.get("reader_open") or {}
    assert "iRAYPLE A5402MU90" in summary or "CCD camera" in summary
    assert "Figure 5" not in summary
    assert str(reader_open.get("anchorKind") or "") == "figure"
    assert int(reader_open.get("anchorNumber") or 0) == 3
    assert "Figure 3" in str(reader_open.get("headingPath") or "")


def test_single_source_prompt_filters_other_papers_that_only_mention_source():
    prompt = (
        "SCINeRF\u7684\u771f\u5b9e\u786c\u4ef6\u5b9e\u9a8c\u88c5\u7f6e\u5305\u542b"
        "\u54ea\u4e9b\u90e8\u4ef6\uff1f\u8bf7\u5bf9\u5e94\u5230\u539f\u6587\u56fe3"
        "\u6216\u5b9e\u9a8c\u8bbe\u7f6e\u3002"
    )
    scinerf_hit = {
        "text": "Figure 3 describes the real-world SCI setup.",
        "meta": {
            "source_path": (
                r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image"
                r"\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
            ),
        },
        "ui_meta": {
            "display_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "summary_line": "This SCI imaging system contains a CCD camera, primary and relay lens, and a DMD.",
        },
    }
    scigs_hit = {
        "text": "SCIGS compares its reconstruction quality against SCINeRF on static datasets.",
        "meta": {
            "source_path": (
                r"db\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image"
                r"\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.en.md"
            ),
        },
        "ui_meta": {
            "display_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
            "summary_line": "The existing SOTA SCI image decoding methods and SCINeRF are compared.",
        },
    }

    filtered = reference_ui._filter_refs_hits_by_prompt_focus(prompt, [scigs_hit, scinerf_hit])

    assert filtered == [scinerf_hit]


def test_cross_paper_prompt_does_not_apply_single_source_binding():
    prompt = "Which papers discuss SCINeRF?"
    scinerf_hit = {
        "text": "SCINeRF proposes neural radiance fields from a snapshot compressed image.",
        "meta": {
            "source_path": (
                r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image"
                r"\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
            ),
        },
        "ui_meta": {"display_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf"},
    }
    scigs_hit = {
        "text": "SCIGS compares against SCINeRF.",
        "meta": {
            "source_path": (
                r"db\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image"
                r"\ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.en.md"
            ),
        },
        "ui_meta": {"display_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf"},
    }

    assert reference_ui._prompt_explicitly_binds_single_source(prompt, [scinerf_hit, scigs_hit]) == []


def test_enrich_refs_payload_keeps_bound_source_for_guide_evidence_location():
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
    assert len(list(entry.get("hits") or [])) == 1
    guide_filter = entry.get("guide_filter", {}) or {}
    assert guide_filter.get("active") is True
    assert guide_filter.get("hidden_self_source") is False
    assert int(guide_filter.get("filtered_hit_count") or 0) == 0
    assert str(entry.get("display_state") or "") == "ready"
    assert str(entry.get("suppression_reason") or "") == ""
    pipeline_debug = entry.get("pipeline_debug", {}) or {}
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 1
    assert int(pipeline_debug.get("filtered_self_hit_count") or 0) == 0
    assert int(pipeline_debug.get("final_hit_count") or 0) == 1


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


def test_sort_refs_hits_prefers_decisive_raw_retrieval_leader_for_general_prompt():
    prompt = "结构化探测在激光扫描显微中主要解决什么矛盾？与传统ISM或共聚焦的权衡有什么不同？"
    target = {
        "text": "Structured detection overcomes the trade-off between optical sectioning and super-resolution.",
        "meta": {
            "source_path": r"db\NatPhoton-2025-Structured detection\NatPhoton-2025.en.md",
            "ref_rank": {"display_score": 26.3, "score": 26.3},
        },
        "ui_meta": {
            "display_name": "NatPhoton-2025-Structured detection.pdf",
            "summary_line": "Structured detection for laser scanning microscopy.",
            "score": 5.72,
            "source_path": r"db\NatPhoton-2025-Structured detection\NatPhoton-2025.en.md",
        },
    }
    loose_ism = {
        "text": "This ISM microscope enables fluorescence detection.",
        "meta": {
            "source_path": r"db\LSA-2026-Interferometric Image Scanning\LSA-2026.en.md",
            "ref_rank": {"display_score": 21.9, "score": 21.9},
        },
        "ui_meta": {
            "display_name": "LSA-2026-Interferometric Image Scanning.pdf",
            "summary_line": "An ISM microscope setup.",
            "score": 6.1,
            "source_path": r"db\LSA-2026-Interferometric Image Scanning\LSA-2026.en.md",
        },
    }

    ordered = reference_ui._sort_refs_hits_for_display(prompt=prompt, hits=[loose_ism, target])

    assert ordered[0] is target
    assert reference_ui._refs_has_decisive_raw_retrieval_leader(prompt, ordered) is True


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


def test_chinese_prompt_focus_aliases_select_deep_learning_evidence(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    prompt = (
        "\u54ea\u4e9b\u6587\u732e\u8ba8\u8bba\u4e86\u5355\u50cf\u7d20\u6210\u50cf\u4e2d\u7684"
        "\u6df1\u5ea6\u5b66\u4e60\uff1f"
    )
    assert "deep learning" in reference_ui._refs_prompt_focus_terms(prompt)

    hit = {
        "meta": {
            "source_path": r"db\ILNet\ILNet.en.md",
            "ref_best_heading_path": "Abstract",
            "ref_section": "Abstract",
            "ref_show_snippets": [
                "The self-supervised image-loop neural network (ILNet) uses deep learning for single-pixel imaging at an ultra-low sampling rate without ground-truth images.",
            ],
            "ref_snippets": [
                "The method addresses low sampling rates and the lack of labeled training data in single-pixel imaging.",
            ],
            "ref_rank": {
                "llm": 78.0,
                "bm25": 5.1,
                "deep": 1.4,
                "term_bonus": 0.4,
                "semantic_score": 7.7,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt=prompt,
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )

    summary = str(ui_meta.get("summary_line") or "")
    assert "\u76f8\u5173\u5185\u5bb9\u4f4d\u4e8e" not in summary
    assert "The relevant discussion appears" not in summary
    assert "deep learning" in summary
    assert "low sampling" in summary or "ground-truth" in summary


def test_chinese_prompt_focus_aliases_keep_structured_detection_evidence(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    prompt = (
        "\u7ed3\u6784\u5316\u63a2\u6d4b\u5728\u6fc0\u5149\u626b\u63cf\u663e\u5fae\u4e2d"
        "\u4e3b\u8981\u89e3\u51b3\u4ec0\u4e48\u77db\u76fe\uff1f"
        "\u4e0e\u4f20\u7edfISM\u6216\u5171\u805a\u7126\u7684\u6743\u8861\u6709\u4ec0\u4e48\u4e0d\u540c\uff1f"
    )
    focus_terms = reference_ui._refs_prompt_focus_terms(prompt)
    assert "structured detection" in focus_terms
    assert any("confocal" in term for term in focus_terms)

    hit = {
        "meta": {
            "source_path": r"db\NatPhoton-2025\s2ISM.en.md",
            "ref_best_heading_path": "Experimental validation of s2ISM",
            "ref_section": "Experimental validation of s2ISM",
            "ref_show_snippets": [
                "Structured detection in laser scanning microscopy resolves the trade-off between optical sectioning and signal-to-noise ratio compared with conventional ISM or confocal microscopy.",
            ],
            "ref_snippets": [
                "The s2ISM implementation preserves optical sectioning while improving SNR over standard image scanning microscopy.",
            ],
            "ref_rank": {
                "llm": 82.0,
                "bm25": 5.7,
                "deep": 1.8,
                "term_bonus": 0.7,
                "semantic_score": 8.4,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt=prompt,
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )

    summary = str(ui_meta.get("summary_line") or "")
    assert "\u76f8\u5173\u5185\u5bb9\u4f4d\u4e8e" not in summary
    assert "The relevant discussion appears" not in summary
    assert "Structured detection" in summary
    assert "trade-off" in summary or "optical sectioning" in summary


def test_chinese_multi_paper_focus_filter_requires_primary_concept():
    prompt = (
        "\u54ea\u4e9b\u6587\u732e\u8ba8\u8bba\u4e86\u5355\u50cf\u7d20\u6210\u50cf\u4e2d\u7684"
        "\u6df1\u5ea6\u5b66\u4e60\uff1f"
    )
    deep_learning_hit = {
        "text": "The paper discusses deep learning for single-pixel imaging.",
        "meta": {"source_path": r"db\LPR-2025\LPR-2025.en.md"},
        "ui_meta": {
            "display_name": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
            "summary_line": "Deep learning is used to improve single-pixel imaging reconstruction.",
        },
    }
    single_pixel_only_hit = {
        "text": "A single-pixel camera measures scenes with spatial patterns.",
        "meta": {"source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md"},
        "ui_meta": {
            "display_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
            "summary_line": "The paper explains how single-pixel cameras work.",
        },
    }

    filtered = reference_ui._filter_refs_hits_by_prompt_focus(
        prompt,
        [single_pixel_only_hit, deep_learning_hit],
    )

    assert filtered == [deep_learning_hit]


def test_prompt_aligned_summary_can_use_late_focus_sentence(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    prompt = (
        "\u54ea\u4e9b\u6587\u732e\u8ba8\u8bba\u4e86\u5355\u50cf\u7d20\u6210\u50cf\u4e2d\u7684"
        "\u6df1\u5ea6\u5b66\u4e60\uff1f"
    )
    filler_sentences = [
        f"The system discussion sentence {idx} describes detector memory, laser power, and holographic alignment."
        for idx in range(10)
    ]
    late_focus_sentence = (
        "Real-time imaging remains difficult, but this constraint can be alleviated through the assistance of deep learning and compressive sensing."
    )
    hit = {
        "meta": {
            "source_path": r"db\NatCommun-2021\NatCommun-2021.en.md",
            "ref_best_heading_path": "ARTICLE / Discussion",
            "ref_section": "Discussion",
            "ref_snippets": ["## Discussion\n" + " ".join(filler_sentences + [late_focus_sentence])],
            "ref_rank": {
                "llm": 74.0,
                "bm25": 4.9,
                "deep": 1.1,
                "term_bonus": 0.3,
                "semantic_score": 7.0,
            },
        }
    }

    ui_meta = build_hit_ui_meta(
        hit,
        prompt=prompt,
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )

    summary = str(ui_meta.get("summary_line") or "")
    assert "deep learning" in summary
    assert "detector memory" not in summary


def test_compare_focus_filter_drops_one_term_tradeoff_noise():
    prompt = (
        "\u7ed3\u6784\u5316\u63a2\u6d4b\u5728\u6fc0\u5149\u626b\u63cf\u663e\u5fae\u4e2d"
        "\u4e3b\u8981\u89e3\u51b3\u4ec0\u4e48\u77db\u76fe\uff1f"
        "\u4e0e\u4f20\u7edfISM\u6216\u5171\u805a\u7126\u7684\u6743\u8861\u6709\u4ec0\u4e48\u4e0d\u540c\uff1f"
    )
    direct_hit = {
        "text": "Structured detection in laser scanning microscopy overcomes the trade-off between SNR and optical sectioning in confocal microscopy.",
        "meta": {"source_path": r"db\NatPhoton-2025\s2ISM.en.md"},
        "ui_meta": {
            "display_name": "NatPhoton-2025-Structured detection for laser scanning microscopy.pdf",
            "summary_line": "Structured detection compares ISM and confocal microscopy trade-offs.",
        },
    }
    loose_hit = {
        "text": "A broad comparison of scanning modalities shows a trade-off with detector dynamic range.",
        "meta": {"source_path": r"db\NatPhoton-2019\single-pixel.en.md"},
        "ui_meta": {
            "display_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
            "summary_line": "The paper mentions a trade-off in scanning modalities.",
        },
    }

    filtered = reference_ui._filter_refs_hits_by_prompt_focus(prompt, [loose_hit, direct_hit])

    assert filtered == [direct_hit]


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


def _legacy_mojibake_build_hit_ui_meta_falls_back_to_citation_summary_when_ref_pack_missing(monkeypatch):
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


def test_build_hit_ui_meta_falls_back_to_citation_summary_when_ref_pack_missing_utf8_safe(monkeypatch):
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
        prompt="\u8fd9\u4e2a\u65b9\u6cd5\u548c\u5f53\u524d\u95ee\u9898\u6709\u4ec0\u4e48\u5173\u7cfb\uff1f",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            r"db\SCINeRF\SCINeRF.en.md": {
                "title": "SCINeRF",
                "summary_line": (
                    "\u5f53\u524d\u4ec5\u68c0\u7d22\u5230\u6587\u732e\u5143\u6570\u636e\uff1a"
                    "\u8be5\u5de5\u4f5c\u53d1\u8868\u4e8e 2024\u3002"
                    "\u7531\u4e8e\u7f3a\u5c11\u53ef\u7528\u6458\u8981\u6587\u672c\uff0c"
                    "\u6682\u65e0\u6cd5\u53ef\u9760\u63d0\u70bc\u5176\u65b9\u6cd5\u7ec6\u8282"
                    "\u4e0e\u5b9e\u9a8c\u7ed3\u8bba\uff0c\u5efa\u8bae\u901a\u8fc7 DOI "
                    "\u67e5\u770b\u539f\u6587\u6458\u8981\u4e0e\u6b63\u6587\u3002"
                ),
            }
        },
    )

    assert "\u5f53\u524d\u4ec5\u68c0\u7d22\u5230\u6587\u732e\u5143\u6570\u636e" in str(ui_meta.get("summary_line") or "")
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


def test_enrich_refs_payload_keeps_bound_paper_for_paper_guide_evidence_mode():
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
    assert len(list(entry.get("hits") or [])) == 1
    guide_filter = entry.get("guide_filter") or {}
    assert guide_filter.get("hidden_self_source") is False
    assert int(guide_filter.get("filtered_hit_count") or 0) == 0
    assert str(guide_filter.get("guide_source_name") or "") == "SCINeRF.pdf"
    assert str(entry.get("display_state") or "") == "ready"


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


def test_enrich_refs_payload_drops_ambiguous_admm_hits_before_llm_rerank(monkeypatch):
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
    rerank_calls: list[int] = []

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
    monkeypatch.setattr(
        reference_ui,
        "_llm_rerank_refs_hit_order",
        lambda **kwargs: rerank_calls.append(len(list(kwargs.get("hits") or []))) or (2, 1, 3),
    )
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(32) or {}).get("hits") or [])
    entry = out.get(32) or {}

    assert hits == []
    assert rerank_calls == []
    assert str(entry.get("display_state") or "") == "suppressed"
    assert str(entry.get("suppression_reason") or "") == "focus_filter_removed_all"


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
    entry = out.get(33) or {}

    assert hits == []
    assert str(entry.get("display_state") or "") == "suppressed"
    assert str(entry.get("suppression_reason") or "") == "focus_filter_removed_all"


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
    entry = out.get(34) or {}

    assert hits == []
    assert str(entry.get("display_state") or "") == "suppressed"
    assert str(entry.get("suppression_reason") or "") == "score_gate_removed_all"
    pipeline_debug = entry.get("pipeline_debug", {}) or {}
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 1
    assert int(pipeline_debug.get("post_score_gate_hit_count") or 0) == 0


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


def test_filter_pending_refs_hits_by_prompt_focus_drops_related_work_only_admm_hit():
    prompt = "Which paper in my library most directly discusses ADMM? Please point me to the source section."
    hits = [
        {
            "text": "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4].",
            "meta": {
                "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                "ref_best_heading_path": "2. Related Work",
                "ref_section": "2. Related Work",
                "ref_show_snippets": [
                    "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4].",
                ],
            },
        }
    ]

    filtered = reference_ui._filter_pending_refs_hits_by_prompt_focus(prompt, hits)

    assert filtered == []


def test_filter_pending_refs_hits_by_prompt_focus_compare_prefers_explicit_versus_paper():
    prompt = "Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?"
    hits = [
        {
            "text": (
                "Instead of using random patterns, basis-scanning single-pixel imaging techniques use deterministic basis "
                "patterns for illumination. Figure 1 shows the comparison between the Hadamard and Fourier basis patterns."
            ),
            "meta": {
                "source_path": (
                    r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
                ),
                "ref_best_heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                "ref_section": "2. Comparison of theory",
                "ref_show_snippets": [
                    "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns."
                ],
            },
        },
        {
            "text": (
                "In the case of Fourier single-pixel imaging, it is possible to employ three-step phase-shifting. "
                "When using the Hadamard basis, one typically requires differential measurements."
            ),
            "meta": {
                "source_path": (
                    r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging"
                    r"\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
                ),
                "ref_best_heading_path": "Abstract / Acquisition and image reconstruction strategies.",
                "ref_section": "Acquisition and image reconstruction strategies",
                "ref_show_snippets": [
                    "In the case of Fourier single-pixel imaging, it is possible to employ three-step phase-shifting."
                ],
            },
        },
    ]

    filtered = reference_ui._filter_pending_refs_hits_by_prompt_focus(prompt, hits)

    assert len(filtered) == 1
    source_path = str((((filtered[0].get("meta") if isinstance(filtered[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
    assert "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging" in source_path


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


def test_focus_term_matches_surface_requires_compound_phrase_not_scattered_tokens():
    assert not reference_ui._focus_term_matches_surface(
        "physics informed deep learning",
        "Developing more precise forward physics models is a promising approach for deep learning in SPI.",
    )
    assert reference_ui._focus_term_matches_surface(
        "deep learning for single pixel imaging",
        "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
    )


def test_enrich_refs_payload_direct_focus_query_drops_scattered_token_false_positive(monkeypatch):
    refs = {
        381: {
            "prompt": "Which paper in my library most directly discusses physics-informed deep learning?",
            "hits": [
                {
                    "text": "We introduce physics-informed deep learning into SPAD imaging to model multiple physical noise sources.",
                    "meta": {
                        "source_path": r"db\NatCommun-2023\NatCommun-2023.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 83.0, "bm25": 4.7, "deep": 1.6, "term_bonus": 0.4, "semantic_score": 7.8},
                    },
                },
                {
                    "text": "Developing more precise forward physics models is a promising approach for deep learning in SPI.",
                    "meta": {
                        "source_path": r"db\LPR-2025\LPR-2025.en.md",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "4.1.2. Model-Driven Strategy",
                        "ref_rank": {"llm": 86.0, "bm25": 5.0, "deep": 1.8, "term_bonus": 0.5, "semantic_score": 8.0},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        heading_path = (
            "Abstract"
            if "NatCommun-2023" in source_path
            else "4.1.2. Model-Driven Strategy"
        )
        return {
            "display_name": source_path,
            "heading_path": heading_path,
            "summary_line": str(hit.get("text") or "").strip(),
            "why_line": "candidate",
            "score": 8.8 if "LPR-2025" in source_path else 8.2,
            "anchor_match_score": 0.0,
            "explicit_doc_match_score": 0.0,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})

    out = enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        allow_expensive_llm_for_ready=False,
    )
    hits = list((out.get(381) or {}).get("hits") or [])

    assert len(hits) == 1
    assert str((((hits[0].get("meta") if isinstance(hits[0].get("meta"), dict) else {}) or {}).get("source_path") or "")).endswith(
        r"NatCommun-2023\NatCommun-2023.en.md"
    )


def test_basis_meta_auto_prefers_prompt_language_over_existing_card_language(monkeypatch):
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_card_locale, "_refs_card_ui_locale_pref", lambda: "")

    why_meta = reference_ui._build_ref_why_basis_meta(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        why_generation="deterministic_grounded",
        why_line="该文比较了 Hadamard 与 Fourier 单像素成像。",
    )
    summary_meta = reference_ui._build_ref_summary_basis_meta(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        summary_kind="guide",
        summary_generation="deterministic_grounded",
        summary_line="该文讨论了 Hadamard single-pixel imaging 与 Fourier single-pixel imaging。",
    )

    assert "focus-term alignment" in str(why_meta.get("why_basis") or "")
    assert "matched section evidence" in str(summary_meta.get("summary_basis") or "")
    assert str(why_meta.get("why_basis") or "").strip()
    assert str(summary_meta.get("summary_basis") or "").strip()


def test_basis_meta_auto_uses_ui_locale_when_no_card_language_signal(monkeypatch):
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_card_locale, "_refs_card_ui_locale_pref", lambda: "zh")

    why_meta = reference_ui._build_ref_why_basis_meta(
        prompt="",
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
    assert "Fourier" in str(ui_meta.get("summary_line") or "")
    assert "单像素" in str(ui_meta.get("summary_line") or "")
    assert "直接对齐" in str(ui_meta.get("why_line") or "")
    assert str(ui_meta.get("summary_generation") or "") == "llm_grounded"
    assert str(ui_meta.get("why_generation") or "") == "llm_grounded"


def test_enrich_refs_payload_prefers_llm_grounded_guide_summary_over_english_surface_copy(monkeypatch):
    refs = {
        39: {
            "prompt": "Give me a guide-style summary of what the SCINeRF paper is about.",
            "hits": [
                {
                    "text": (
                        "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single "
                        "compressed snapshot can recover the underlying 3D scene representation."
                    ),
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "ready",
                        "ref_show_snippets": [
                            "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single compressed snapshot can recover the underlying 3D scene representation.",
                            "The method models the scene with neural radiance fields and optimizes reconstruction from one coded measurement.",
                        ],
                        "ref_rank": {"llm": 85.0, "bm25": 5.2, "deep": 1.6, "term_bonus": 0.4, "semantic_score": 7.9},
                    },
                }
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        return {
            "display_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "Abstract",
            "summary_line": (
                "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single "
                "compressed snapshot can recover the underlying 3D scene representation."
            ),
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "The abstract is a useful reading entry for understanding the paper.",
            "why_generation": "deterministic_grounded",
            "score": 8.5,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_prefer_zh_ref_card_locale", lambda *args: True)
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(
        reference_ui,
        "_pick_ref_card_summary_fallback",
        lambda **kwargs: (
            "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single "
            "compressed snapshot can recover the underlying 3D scene representation."
        ),
    )
    monkeypatch.setattr(reference_ui, "_llm_ground_ref_why_line", lambda **kwargs: "")
    monkeypatch.setattr(
        reference_ui,
        "_llm_polish_ref_card_copy_v2",
        lambda **kwargs: (
            "SCINeRF treats snapshot compressive imaging as a 3D scene reconstruction problem and solves it with a NeRF-based scene model learned from one coded measurement.",
            "The abstract is relevant because it states the paper's core task and NeRF-based reconstruction strategy directly.",
        ),
    )

    out = enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(39) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "NeRF" in str(ui_meta.get("summary_line") or "")
    assert "core task" in str(ui_meta.get("why_line") or "")
    assert str(ui_meta.get("summary_generation") or "") == "llm_grounded"
    assert str(ui_meta.get("why_generation") or "") == "llm_grounded"
    assert "LLM" in str(ui_meta.get("summary_basis") or "")
    assert "This paper proposes" not in str(ui_meta.get("summary_line") or "")
def test_maybe_polish_single_ref_hit_card_strict_mode_uses_llm_output_without_rule_fallback(monkeypatch):
    hit = {
        "text": (
            "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single "
            "compressed snapshot can recover the underlying 3D scene representation."
        ),
        "meta": {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "ref_show_snippets": [
                (
                    "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single "
                    "compressed snapshot can recover the underlying 3D scene representation."
                )
            ],
        },
    }
    ui_meta = {
        "display_name": "CVPR-2024-SCINeRF.pdf",
        "heading_path": "Abstract",
        "summary_line": (
            "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single "
            "compressed snapshot can recover the underlying 3D scene representation."
        ),
        "summary_kind": "guide",
        "summary_generation": "section_grounded",
        "why_line": "The abstract is a useful reading entry for understanding the paper.",
        "why_generation": "deterministic_grounded",
    }

    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(
        reference_ui,
        "_llm_polish_ref_card_copy_v2",
        lambda **kwargs: (
            "This paper proposes a NeRF-based SCI reconstruction pipeline and shows that a single compressed snapshot can recover the underlying 3D scene representation.",
            "This hit is directly relevant because it is a good entry point for the user's question.",
        ),
    )

    out = reference_ui._maybe_polish_single_ref_hit_card(
        prompt="Give me a guide-style summary of what the SCINeRF paper is about.",
        hit=hit,
        ui_meta=ui_meta,
        allow_expensive_llm=True,
    )

    assert str(out.get("summary_line") or "") == str(ui_meta.get("summary_line") or "")
    assert str(out.get("why_line") or "") == "This hit is directly relevant because it is a good entry point for the user's question."
    assert str(out.get("summary_generation") or "") == "llm_grounded"
    assert str(out.get("why_generation") or "") == "llm_grounded"


def test_maybe_polish_single_ref_hit_card_falls_back_to_real_snippet_when_llm_empty(monkeypatch):
    hit = {
        "text": "Table 1. Quantitative comparison results for USAF 1951 test chart",
        "meta": {
            "source_path": r"db\OE-2017-Hadamard\OE-2017-Hadamard.en.md",
            "ref_show_snippets": [
                (
                    "### 2.2 Basis patterns generation\n"
                    "The core of single-pixel imaging is to employ active illumination to acquire the spatial information of a target object. "
                    "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns. "
                    "Hadamard basis patterns are binary (black-and-white), which makes HSI naturally suitable for single-pixel imaging systems based on a digital micro-mirror device (DMD). "
                    "As DMD is a binary device, HSI can benefit from the high-speed binary illumination ability given by a DMD."
                )
            ],
        },
    }
    ui_meta = {
        "display_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
        "summary_line": "该文在“Hadamard single-pixel imaging versus Fourier single-pixel imaging / 3. Comparison of experiment / 3.1 Numerical simulations”讨论了“single pixel imaging”。",
        "summary_kind": "guide",
        "summary_generation": "section_grounded",
        "why_line": "该文在“3.1 Numerical simulations”给出了与“single pixel imaging”直接相关的定义、方法或结果信息。",
        "why_generation": "deterministic_grounded",
    }

    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(reference_ui, "_llm_polish_ref_card_copy_v2", lambda **kwargs: ("", ""))
    monkeypatch.setattr(reference_ui, "_llm_ground_ref_why_line", lambda **kwargs: "")
    monkeypatch.setattr(reference_ui, "_llm_select_best_evidence_candidate", lambda **kwargs: "")

    out = reference_ui._maybe_polish_single_ref_hit_card(
        prompt="我做单像素实验，Hadamard 和 Fourier 到底该怎么选？",
        hit=hit,
        ui_meta=ui_meta,
        allow_expensive_llm=True,
    )

    summary = str(out.get("summary_line") or "")
    assert "该文在" not in summary
    assert "DMD" in summary or "binary" in summary
    assert str(out.get("summary_generation") or "") == "deterministic_grounded"


def test_doc_list_primary_evidence_replaces_synthetic_section_rescue_with_alternative():
    synthetic = {
        "source_path": r"db\OE-2017-Hadamard\OE-2017-Hadamard.en.md",
        "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
        "snippet": "该文在“Hadamard single-pixel imaging versus Fourier single-pixel imaging / 3. Comparison of experiment / 3.1 Numerical simulations”讨论了“single pixel imaging”。",
        "highlight_snippet": "该文在“Hadamard single-pixel imaging versus Fourier single-pixel imaging / 3. Comparison of experiment / 3.1 Numerical simulations”讨论了“single pixel imaging”。",
        "selection_reason": "section_intent_rescue",
        "strict_locate": False,
        "alternatives": [
            {
                "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                "snippet": "Hadamard basis patterns are binary (black-and-white), which makes HSI naturally suitable for single-pixel imaging systems based on a digital micro-mirror device (DMD).",
                "highlight_snippet": "Hadamard basis patterns are binary (black-and-white), which makes HSI naturally suitable for single-pixel imaging systems based on a digital micro-mirror device (DMD).",
            }
        ],
    }

    primary, source = reference_ui._select_doc_list_effective_primary_evidence(
        prompt="我做单像素实验，Hadamard 和 Fourier 到底该怎么选？",
        display_name="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        authoritative_primary_evidence=synthetic,
        synthesized_primary_evidence={},
    )

    assert source == "authoritative"
    assert str(primary.get("selection_reason") or "") == "alternative_rescue"
    assert "DMD" in str(primary.get("snippet") or "")
    assert "该文在" not in str(primary.get("snippet") or "")


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
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_card_locale, "_refs_card_ui_locale_pref", lambda: "")
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
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_card_locale, "_refs_card_ui_locale_pref", lambda: "")
    out = reference_ui._build_prompt_aligned_ref_why_line_v3(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        display_name="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        heading_path="3. Comparison of experiment / 3.1 Numerical simulations",
        summary_line="The paper compares Hadamard single-pixel imaging and Fourier single-pixel imaging in simulations and experiments.",
        why_line="",
    )

    assert "compares" in str(out or "")
    assert "directly relevant" not in str(out or "").lower()
    assert "good entry point" not in str(out or "").lower()


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
    assert str(primary_evidence.get("selection_reason") or "") in {"prompt_aligned", "prompt_aligned_block"}
    reader_open = ui.get("reader_open") or {}
    assert str(reader_open.get("headingPath") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert (reader_open.get("primaryEvidence") or {}) == primary_evidence


def test_build_hit_ui_meta_prefers_authoritative_source_block_for_reader_open(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    prompt = "Which paper in my library most directly defines dynamic supersampling? Please point me to the source section."
    source_path = (
        r"db\SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling"
        r"\SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.en.md"
    )
    authoritative_candidate = {
        "summary": (
            "The paper defines dynamic supersampling by explaining that because the pixel geometry of each frame "
            "in the single-pixel imaging system is defined by the masking patterns applied to the DMD and used to "
            "measure the image, it is possible to perform digital supersampling."
        ),
        "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
        "raw_focus_surface": (
            "INTRODUCTION / Spatially variant digital supersampling "
            "If the positions of the pixel boundaries are modified from one frame to the next, then each frame "
            "samples a different subset of the spatial information in the scene."
        ),
        "source_kind": "source_block",
        "block_id": "blk_dynamic_define",
        "anchor_id": "a_dynamic_define",
        "block_kind": "paragraph",
        "block_number": 0,
        "block_text": (
            "If the positions of the pixel boundaries are modified from one frame to the next, then each frame "
            "samples a different subset of the spatial information in the scene. Consequently, successive frames "
            "capture complementary spatial information."
        ),
    }

    monkeypatch.setattr(
        reference_ui,
        "_build_ref_navigation",
        lambda *args, **kwargs: {"what": "", "summary_line": "", "why": "", "find": []},
    )
    monkeypatch.setattr(
        reference_ui,
        "_fallback_ref_ui_summary_line",
        lambda *args, **kwargs: (
            "dynamic supersampling: Because the pixel geometry of each frame in our single-pixel imaging system "
            "is defined by the masking patterns applied to the DMD and used to measure the image, it is possible "
            "to perform digital supersampling."
        ),
    )
    monkeypatch.setattr(reference_ui, "_choose_prompt_aligned_ref_summary_candidate", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        reference_ui,
        "_choose_prompt_aligned_ref_summary_candidate_from_source_blocks",
        lambda **kwargs: dict(authoritative_candidate),
    )
    monkeypatch.setattr(
        reference_ui,
        "_pick_best_prompt_aligned_ref_summary_candidate",
        lambda candidates, **kwargs: next(
            (dict(item) for item in candidates if isinstance(item, dict) and item),
            {},
        ),
    )
    monkeypatch.setattr(
        reference_ui,
        "_resolve_refs_exact_candidates",
        lambda **kwargs: [
            {
                "headingPath": "INTRODUCTION / Foveated single-pixel imaging",
                "snippet": "Foveated single-pixel imaging adapts spatial resolution over the field of view.",
                "highlightSnippet": "Foveated single-pixel imaging adapts spatial resolution over the field of view.",
                "blockId": "blk_foveated",
                "anchorId": "a_foveated",
                "anchorKind": "paragraph",
            }
        ],
    )

    ui = build_hit_ui_meta(
        {
            "meta": {
                "source_path": source_path,
                "ref_pack_state": "ready",
                "ref_best_heading_path": "INTRODUCTION / Foveated single-pixel imaging",
                "ref_section": "INTRODUCTION",
                "ref_rank": {
                    "llm": 81.0,
                    "bm25": 5.2,
                    "deep": 12.4,
                    "term_bonus": 1.0,
                    "semantic_score": 8.0,
                },
            }
        },
        prompt=prompt,
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
    )

    assert str(ui.get("heading_path") or "") == "INTRODUCTION / Spatially variant digital supersampling"
    assert str(ui.get("summary_line") or "").startswith("The paper defines dynamic supersampling")

    reader_open = dict(ui.get("reader_open") or {})
    reader_primary = dict(reader_open.get("primaryEvidence") or {})
    assert str(reader_open.get("blockId") or "") == "blk_dynamic_define"
    assert str(reader_open.get("headingPath") or "") == "INTRODUCTION / Spatially variant digital supersampling"
    assert str(reader_primary.get("block_id") or "") == "blk_dynamic_define"
    assert str(reader_primary.get("heading_path") or "") == "INTRODUCTION / Spatially variant digital supersampling"

    alternatives = list(reader_open.get("alternatives") or [])
    assert any(str(item.get("blockId") or "") == "blk_foveated" for item in alternatives)


def test_build_refs_reader_open_payload_prefers_exact_candidate_related_to_card_heading(monkeypatch):
    prompt = "Which paper in my library most directly defines dynamic supersampling? Please point me to the source section."
    source_path = (
        r"db\SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling"
        r"\SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.en.md"
    )

    monkeypatch.setattr(
        reference_ui,
        "_resolve_refs_exact_candidates",
        lambda **kwargs: [
            {
                "headingPath": "INTRODUCTION / Foveated single-pixel imaging",
                "snippet": "The paper defines dynamic supersampling by explaining that because the pixel geometry of each frame is defined by the masking patterns, it is possible to perform digital supersampling.",
                "highlightSnippet": "The paper defines dynamic supersampling by explaining that because the pixel geometry of each frame is defined by the masking patterns, it is possible to perform digital supersampling.",
                "blockId": "blk_foveated",
                "anchorId": "a_foveated",
                "anchorKind": "paragraph",
            },
            {
                "headingPath": "INTRODUCTION / Spatially variant digital supersampling",
                "snippet": "The paper defines dynamic supersampling by explaining that because the pixel geometry of each frame is defined by the masking patterns, it is possible to perform digital supersampling.",
                "highlightSnippet": "The paper defines dynamic supersampling by explaining that because the pixel geometry of each frame is defined by the masking patterns, it is possible to perform digital supersampling.",
                "blockId": "blk_dynamic_define",
                "anchorId": "a_dynamic_define",
                "anchorKind": "paragraph",
            },
        ],
    )

    out = reference_ui._build_refs_reader_open_payload(
        meta={"ref_pack_state": "ready"},
        prompt=prompt,
        source_path=source_path,
        display_name="SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
        heading_path="INTRODUCTION / Spatially variant digital supersampling",
        heading="INTRODUCTION",
        summary_line=(
            "The paper defines dynamic supersampling by explaining that because the pixel geometry of each frame "
            "is defined by the masking patterns, it is possible to perform digital supersampling."
        ),
        why_line="",
        anchor_target_kind="",
        anchor_target_number=0,
        allow_llm_disambiguation=False,
        allow_exact_locate=True,
    )

    assert out.get("strictLocate") is True
    assert str(out.get("blockId") or "") == "blk_dynamic_define"
    assert str(out.get("headingPath") or "") == "INTRODUCTION / Spatially variant digital supersampling"
    assert str(((out.get("locateTarget") or {}).get("headingPath")) or "") == "INTRODUCTION / Spatially variant digital supersampling"


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
            "summary": "本节介绍了基模式的生成方法，并对比了哈达玛（Hadamard）与傅里叶（Fourier）基模式的特性差异。",
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


def test_build_hit_ui_meta_recovers_prompt_aligned_block_summary_when_meta_has_no_snippets(tmp_path, monkeypatch):
    md_path = tmp_path / "frontiers_fixture.en.md"
    md_path.write_text("# Demo\n", encoding="utf-8")

    blocks = [
        {
            "block_id": "blk_qc_ref",
            "anchor_id": "a_qc_ref",
            "heading_path": "5 Application / 5.3 Quantum communication",
            "kind": "paragraph",
            "text": "Hong, C. Yu, J. Zhang, Q. Zhang, C. Z. Peng, F. Xu, and J. W. Pan, Single-photon imaging over 200 km, Optica 8(3), 344 (2021)",
        },
        {
            "block_id": "blk_optical",
            "anchor_id": "a_optical",
            "heading_path": "5 Application / 5.1 Optical imaging",
            "kind": "paragraph",
            "text": (
                "A photon is the smallest energy unit of light that can be detected. Traditional cameras realize object imaging "
                "by detecting light intensity at different positions, while single-photon imaging can reconstruct the image of "
                "the object by detecting the three-dimensional space position and time information of each photon."
            ),
        },
        {
            "block_id": "blk_qc_body",
            "anchor_id": "a_qc_body",
            "heading_path": "5 Application / 5.3 Quantum communication",
            "kind": "paragraph",
            "text": (
                "Single-photon ranging and detection both require single-photon sensitivity and picosecond timing resolution "
                "for ultra-long distance 3D imaging."
            ),
        },
    ]

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del snippet, heading_path, kwargs
        return [
            {"score": 0.94, "block": blocks[1]},
            {"score": 0.62, "block": blocks[2]},
        ]

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
    monkeypatch.setattr(reference_ui, "_choose_prompt_aligned_ref_summary_candidate", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_fallback_ref_ui_summary_line", lambda *args, **kwargs: "")
    monkeypatch.setattr(reference_ui, "load_source_blocks", lambda _path: blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)

    ui = build_hit_ui_meta(
        {
            "text": "",
            "meta": {
                "source_path": str(md_path),
                "ref_pack_state": "ready",
                "ref_best_heading_path": "5 Application / 5.3 Quantum communication",
                "ref_section": "5 Application",
                "ref_subsection": "5.3 Quantum communication",
                "ref_rank": {"llm": 84.0, "bm25": 4.7, "deep": 1.5, "term_bonus": 0.4, "semantic_score": 7.8},
            },
        },
        prompt="Which papers in my library discuss single-photon imaging?",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            str(md_path): {
                "title": "Emerging single-photon detection technique for high-performance photodetector",
            }
        },
        allow_expensive_llm=False,
        allow_exact_locate=True,
    )

    assert str(ui.get("heading_path") or "") == "5 Application / 5.1 Optical imaging"
    assert "single-photon imaging" in str(ui.get("summary_line") or "").lower()
    assert "over 200 km" not in str(ui.get("summary_line") or "").lower()
    primary_evidence = ui.get("primary_evidence") or {}
    assert str(primary_evidence.get("heading_path") or "") == "5 Application / 5.1 Optical imaging"
    assert str(primary_evidence.get("selection_reason") or "") == "prompt_aligned_block"


def test_build_hit_ui_meta_pending_can_rescue_prompt_aligned_block_summary_without_strict_locate(tmp_path, monkeypatch):
    md_path = tmp_path / "frontiers_pending_fixture.en.md"
    md_path.write_text("# Demo\n", encoding="utf-8")

    blocks = [
        {
            "block_id": "blk_qc_ref",
            "anchor_id": "a_qc_ref",
            "heading_path": "5 Application / 5.3 Quantum communication",
            "kind": "paragraph",
            "text": "Hong, C. Yu, J. Zhang, Q. Zhang, C. Z. Peng, F. Xu, and J. W. Pan, Single-photon imaging over 200 km, Optica 8(3), 344 (2021)",
        },
        {
            "block_id": "blk_optical",
            "anchor_id": "a_optical",
            "heading_path": "5 Application / 5.1 Optical imaging",
            "kind": "paragraph",
            "text": (
                "Traditional cameras realize object imaging by detecting light intensity at different positions, while "
                "single-photon imaging can reconstruct the image of the object by detecting the three-dimensional space "
                "position and time information of each photon."
            ),
        },
    ]

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del snippet, heading_path, kwargs
        return [{"score": 0.94, "block": blocks[1]}]

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
    monkeypatch.setattr(reference_ui, "_choose_prompt_aligned_ref_summary_candidate", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_fallback_ref_ui_summary_line", lambda *args, **kwargs: "")
    monkeypatch.setattr(reference_ui, "load_source_blocks", lambda _path: blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)

    ui = build_hit_ui_meta(
        {
            "text": "",
            "meta": {
                "source_path": str(md_path),
                "ref_pack_state": "pending",
                "ref_best_heading_path": "5 Application / 5.3 Quantum communication",
                "ref_section": "5 Application",
                "ref_subsection": "5.3 Quantum communication",
                "ref_rank": {"llm": 0.0, "bm25": 4.7, "deep": 1.5, "term_bonus": 0.4, "semantic_score": 7.8},
            },
        },
        prompt="Which papers in my library discuss single-photon imaging?",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            str(md_path): {
                "title": "Emerging single-photon detection technique for high-performance photodetector",
            }
        },
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )

    assert str(ui.get("heading_path") or "") == "5 Application / 5.1 Optical imaging"
    assert "single-photon imaging" in str(ui.get("summary_line") or "").lower()
    primary_evidence = ui.get("primary_evidence") or {}
    assert str(primary_evidence.get("heading_path") or "") == "5 Application / 5.1 Optical imaging"
    assert str(primary_evidence.get("selection_reason") or "") == "prompt_aligned_block"
    reader_open = dict(ui.get("reader_open") or {})
    assert str(reader_open.get("headingPath") or "") == "5 Application / 5.1 Optical imaging"
    assert reader_open.get("strictLocate") is False


def test_build_hit_ui_meta_prefers_block_summary_over_prefixed_abstract_shell(tmp_path, monkeypatch):
    md_path = tmp_path / "frontiers_prefixed_shell_fixture.en.md"
    md_path.write_text("# Demo\n", encoding="utf-8")

    prefixed_shell = (
        "single-photon imaging: ABSTRACT Single-photon detections (SPDs) represent a highly sensitive light "
        "detection technique capable of detecting individual photons at extremely low light intensity levels."
    )
    assert reference_ui._is_ref_card_summary_acceptable(
        prompt="Which papers in my library discuss single-photon imaging?",
        title="Emerging single-photon detection technique for high-performance photodetector",
        summary_line=prefixed_shell,
    ) is False

    blocks = [
        {
            "block_id": "blk_qc",
            "anchor_id": "a_qc",
            "heading_path": "5 Application / 5.3 Quantum communication",
            "kind": "paragraph",
            "text": (
                "Single-photon ranging and detection both require single-photon sensitivity and picosecond timing resolution "
                "for ultra-long distance 3D imaging."
            ),
        },
        {
            "block_id": "blk_optical",
            "anchor_id": "a_optical",
            "heading_path": "5 Application / 5.1 Optical imaging",
            "kind": "paragraph",
            "text": (
                "Traditional cameras realize object imaging by detecting light intensity at different positions, while "
                "single-photon imaging can reconstruct the image of the object by detecting the three-dimensional space "
                "position and time information of each photon."
            ),
        },
    ]

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del snippet, heading_path, kwargs
        return [{"score": 0.94, "block": blocks[1]}]

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
        lambda *args, **kwargs: prefixed_shell,
    )
    monkeypatch.setattr(reference_ui, "load_source_blocks", lambda _path: blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)

    ui = build_hit_ui_meta(
        {
            "text": "",
            "meta": {
                "source_path": str(md_path),
                "ref_pack_state": "ready",
                "ref_best_heading_path": "5 Application / 5.3 Quantum communication",
                "ref_section": "5 Application",
                "ref_subsection": "5.3 Quantum communication",
                "ref_overview_snippets": [
                    "ABSTRACT Single-photon detections (SPDs) represent a highly sensitive light detection technique capable of detecting individual photons at extremely low light intensity levels."
                ],
                "ref_rank": {"llm": 84.0, "bm25": 4.7, "deep": 1.5, "term_bonus": 0.4, "semantic_score": 7.8},
            },
        },
        prompt="Which papers in my library discuss single-photon imaging?",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            str(md_path): {
                "title": "Emerging single-photon detection technique for high-performance photodetector",
            }
        },
        allow_expensive_llm=False,
        allow_exact_locate=True,
    )

    assert str(ui.get("heading_path") or "") == "5 Application / 5.1 Optical imaging"
    summary_line = str(ui.get("summary_line") or "")
    assert summary_line.startswith("Traditional cameras realize object imaging")
    assert not summary_line.lower().startswith("single-photon imaging:")
    primary_evidence = ui.get("primary_evidence") or {}
    assert str(primary_evidence.get("heading_path") or "") == "5 Application / 5.1 Optical imaging"
    assert str(primary_evidence.get("selection_reason") or "") == "prompt_aligned_block"


def test_build_hit_ui_meta_focus_prefixed_fallback_still_triggers_block_rescue(tmp_path, monkeypatch):
    md_path = tmp_path / "frontiers_focus_prefix_fixture.en.md"
    md_path.write_text("# Demo\n", encoding="utf-8")

    fallback_summary = (
        "single-photon imaging: This technology mainly relies on the mainstream SPDs, such as photomultiplier tubes "
        "(PMTs), avalanche photodiodes (SAPD), superconducting nanowire single-photon detectors (SNSPDs)."
    )
    assert reference_ui._looks_focus_prefixed_ref_summary(
        "Which papers in my library discuss single-photon imaging?",
        fallback_summary,
    ) is True

    blocks = [
        {
            "block_id": "blk_qc",
            "anchor_id": "a_qc",
            "heading_path": "5 Application / 5.3 Quantum communication",
            "kind": "paragraph",
            "text": (
                "Single-photon ranging and detection both require single-photon sensitivity and picosecond timing resolution "
                "for ultra-long distance 3D imaging."
            ),
        },
        {
            "block_id": "blk_optical",
            "anchor_id": "a_optical",
            "heading_path": "5 Application / 5.1 Optical imaging",
            "kind": "paragraph",
            "text": (
                "Traditional cameras realize object imaging by detecting light intensity at different positions, while "
                "single-photon imaging can reconstruct the image of the object by detecting the three-dimensional space "
                "position and time information of each photon."
            ),
        },
    ]

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del snippet, heading_path, kwargs
        return [{"score": 0.94, "block": blocks[1]}]

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
        lambda *args, **kwargs: fallback_summary,
    )
    monkeypatch.setattr(reference_ui, "load_source_blocks", lambda _path: blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)

    ui = build_hit_ui_meta(
        {
            "text": "",
            "meta": {
                "source_path": str(md_path),
                "ref_pack_state": "ready",
                "ref_best_heading_path": "5 Application / 5.3 Quantum communication",
                "ref_section": "5 Application",
                "ref_subsection": "5.3 Quantum communication",
                "ref_overview_snippets": [
                    "ABSTRACT Single-photon detections (SPDs) represent a highly sensitive light detection technique capable of detecting individual photons at extremely low light intensity levels."
                ],
                "ref_rank": {"llm": 84.0, "bm25": 4.7, "deep": 1.5, "term_bonus": 0.4, "semantic_score": 7.8},
            },
        },
        prompt="Which papers in my library discuss single-photon imaging?",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={
            str(md_path): {
                "title": "Emerging single-photon detection technique for high-performance photodetector",
            }
        },
        allow_expensive_llm=False,
        allow_exact_locate=True,
    )

    assert str(ui.get("heading_path") or "") == "5 Application / 5.1 Optical imaging"
    assert str(ui.get("summary_line") or "").startswith("Traditional cameras realize object imaging")
    primary_evidence = ui.get("primary_evidence") or {}
    assert str(primary_evidence.get("selection_reason") or "") == "prompt_aligned_block"


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


def test_resolve_refs_exact_candidates_demotes_title_echo_heading_block(tmp_path, monkeypatch):
    md_path = tmp_path / "compare_exact_title_echo.en.md"
    md_path.write_text("# Compare\n", encoding="utf-8")

    blocks = [
        {
            "block_id": "blk_bad_title",
            "anchor_id": "a_bad_title",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "kind": "heading",
            "text": "Hadamard single-pixel imaging versus Fourier single-pixel imaging ZIBANG ZHANG OCIS codes",
        },
        {
            "block_id": "blk_bad_caption",
            "anchor_id": "a_bad_caption",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "kind": "paragraph",
            "text": "(A) Numerical simulations comparing Hadamard and Fourier basis patterns.",
        },
        {
            "block_id": "blk_good_compare",
            "anchor_id": "a_good_compare",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
            "kind": "paragraph",
            "text": (
                "Instead of using random patterns, basis-scanning single-pixel imaging techniques use deterministic basis "
                "patterns for illumination. Figure 1 shows the comparison between the Hadamard and Fourier basis patterns."
            ),
        },
    ]

    def fake_load_source_blocks(_path):
        return blocks

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del _blocks, snippet, kwargs
        if "3.1 Numerical simulations" in str(heading_path or ""):
            return [
                {"score": 0.96, "block": blocks[0]},
                {"score": 0.93, "block": blocks[1]},
                {"score": 0.91, "block": blocks[2]},
            ]
        return [
            {"score": 0.92, "block": blocks[2]},
            {"score": 0.89, "block": blocks[0]},
            {"score": 0.87, "block": blocks[1]},
        ]

    monkeypatch.setattr(reference_ui, "load_source_blocks", fake_load_source_blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)

    out = reference_ui._resolve_refs_exact_candidates(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        source_path=str(md_path),
        display_name="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        anchor_target_kind="",
        anchor_target_number=0,
        primary_candidate={
            "headingPath": "3. Comparison of experiment / 3.1 Numerical simulations",
            "snippet": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
            "highlightSnippet": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
        },
        secondary_candidates=[
            {
                "headingPath": "2. Comparison of theory / 2.2 Basis patterns generation",
                "snippet": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
                "highlightSnippet": "Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
            }
        ],
        allow_llm_disambiguation=False,
    )

    assert out
    assert str(out[0].get("blockId") or "") == "blk_good_compare"
    assert str(out[0].get("headingPath") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert "comparison between the Hadamard and Fourier basis patterns" in str(out[0].get("snippet") or "")
    assert "OCIS codes" not in str(out[0].get("snippet") or "")


def test_resolve_refs_exact_candidates_prefers_definition_paragraph_over_caption_fragment(tmp_path, monkeypatch):
    md_path = tmp_path / "dynamic_exact_caption.en.md"
    md_path.write_text("# Dynamic\n", encoding="utf-8")

    blocks = [
        {
            "block_id": "blk_dynamic_heading",
            "anchor_id": "a_dynamic_heading",
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "kind": "heading",
            "text": "Spatially variant digital supersampling",
        },
        {
            "block_id": "blk_dynamic_define",
            "anchor_id": "a_dynamic_define",
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "kind": "paragraph",
            "text": (
                "If the positions of the pixel boundaries are modified from one frame to the next, then each frame samples "
                "a different subset of the spatial information in the scene. Consequently, successive frames capture "
                "complementary spatial information."
            ),
        },
        {
            "block_id": "blk_dynamic_caption",
            "anchor_id": "a_dynamic_caption",
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "kind": "paragraph",
            "text": "(A) Four subframes, each with the foveal cells shifted by half a cell in x and/or y with respect to one another.",
        },
    ]

    def fake_load_source_blocks(_path):
        return blocks

    def fake_match_source_blocks(_blocks, *, snippet="", heading_path="", **kwargs):
        del _blocks, snippet, heading_path, kwargs
        return [
            {"score": 0.95, "block": blocks[2]},
            {"score": 0.92, "block": blocks[1]},
            {"score": 0.88, "block": blocks[0]},
        ]

    monkeypatch.setattr(reference_ui, "load_source_blocks", fake_load_source_blocks)
    monkeypatch.setattr(reference_ui, "match_source_blocks", fake_match_source_blocks)

    out = reference_ui._resolve_refs_exact_candidates(
        prompt="Which paper in my library most directly defines dynamic supersampling? Please point me to the source section.",
        source_path=str(md_path),
        display_name="SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
        anchor_target_kind="",
        anchor_target_number=0,
        primary_candidate={
            "headingPath": "INTRODUCTION / Spatially variant digital supersampling",
            "snippet": "dynamic supersampling: Because the pixel geometry of each frame in our single-pixel imaging system is defined by the masking patterns applied to the DMD and used to measure the image, it is possible to perform digital supersampling.",
            "highlightSnippet": "dynamic supersampling: Because the pixel geometry of each frame in our single-pixel imaging system is defined by the masking patterns applied to the DMD and used to measure the image, it is possible to perform digital supersampling.",
        },
        secondary_candidates=[],
        allow_llm_disambiguation=False,
    )

    assert out
    assert str(out[0].get("blockId") or "") == "blk_dynamic_define"
    assert str(out[0].get("headingPath") or "") == "INTRODUCTION / Spatially variant digital supersampling"
    assert str(out[0].get("snippet") or "").startswith("If the positions of the pixel boundaries are modified")
    assert not str(out[0].get("snippet") or "").startswith("(A)")


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


def test_choose_prompt_aligned_ref_summary_rewrites_long_definition_style_snippet(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    meta = {
        "ref_show_snippets": [
            (
                "## Spatially variant digital supersampling\n"
                "If the positions of the pixel boundaries are modified from one frame to the next, "
                "then each frame samples a different subset of the spatial information in the scene. "
                "Consequently, successive frames are capturing not only information about the temporal "
                "variation of the scene but also additional complementary information about the spatial "
                "structure of the scene."
            ),
        ]
    }

    out = reference_ui._choose_prompt_aligned_ref_summary(
        meta,
        prompt="Which paper in my library most directly defines dynamic supersampling? Please point me to the source section.",
        source_path=r"db\SciAdv-2017\SciAdv-2017.en.md",
        citation_meta={"title": "Adaptive foveated single-pixel imaging with dynamic supersampling"},
    )

    assert str(out or "").startswith("The paper defines dynamic supersampling")
    assert "when the positions of the pixel boundaries are modified" in str(out or "")


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


def test_pick_ref_card_summary_fallback_rewrites_long_definition_excerpt():
    raw = (
        "## Spatially variant digital supersampling\n"
        "If the positions of the pixel boundaries are modified from one frame to the next, "
        "then each frame samples a different subset of the spatial information in the scene. "
        "Consequently, successive frames are capturing not only information about the temporal "
        "variation of the scene but also additional complementary information about the spatial "
        "structure of the scene."
    )

    out = reference_ui._pick_ref_card_summary_fallback(
        prompt="Which paper in my library most directly defines dynamic supersampling? Please point me to the source section.",
        title="SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
        candidates=[raw],
    )

    assert str(out or "").startswith("The paper defines dynamic supersampling")
    assert "Spatially variant digital supersampling:" not in str(out or "")


def test_expand_ref_summary_candidates_does_not_prefix_physics_informed_focus_without_informative_hit(monkeypatch):
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: text)

    out = reference_ui._expand_ref_summary_candidates(
        "### 4.1.2. Model-Driven Strategy\nAdvances and Challenges of Single-Pixel Imaging Based on Deep Learning. However, the limited image quality and lengthy computational times for iterative reconstruction still hinder its practical application.",
        prompt="Which paper in my library most directly discusses physics-informed deep learning? Please point me to the source section.",
        title="Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
        prefer_zh=False,
    )

    lowered = [str(item or "").lower() for item in out]
    assert not any(item.startswith("physics-informed deep learning:") for item in lowered)
    assert not any(item.startswith("physics informed deep learning:") for item in lowered)


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
        allow_expensive_llm_for_ready=False,
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
        {
            "meta": {
                "source_path": r"db\Journal of Optics-2016-3D single-pixel video\Journal of Optics-2016-3D single-pixel video.en.md",
            },
            "ui_meta": {
                "display_name": "Journal of Optics-2016-3D single-pixel video.pdf",
                "heading_path": "Results",
                "summary_line": "As used in other work with single-pixel cameras, the Hadamard basis yields better quality results compared to raster scanning techniques that suffer from poorer signal-to-noise.",
                "why_line": "This hit compares Hadamard measurements with a different scanning baseline rather than directly comparing Hadamard and Fourier single-pixel imaging.",
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


def test_filter_refs_hits_by_prompt_focus_keeps_multiple_hits_for_multi_paper_list_query():
    prompt = "有哪几篇文章提到了SCI（单次曝光压缩成像）？"
    hits = [
        {
            "meta": {
                "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            },
            "ui_meta": {
                "display_name": "ICIP-2025-SCIGS.pdf",
                "heading_path": "Introduction",
                "summary_line": "The paper explicitly introduces Snapshot Compressive Imaging (SCI) and builds on that setting.",
                "why_line": "This hit directly discusses Snapshot Compressive Imaging (SCI).",
            },
        },
        {
            "meta": {
                "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            },
            "ui_meta": {
                "display_name": "CVPR-2024-SCINeRF.pdf",
                "heading_path": "Abstract",
                "summary_line": "The paper repeatedly mentions Snapshot Compressive Imaging (SCI) in the abstract and introduction.",
                "why_line": "This hit directly discusses Snapshot Compressive Imaging (SCI).",
            },
        },
        {
            "meta": {
                "source_path": r"db\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.en.md",
            },
            "ui_meta": {
                "display_name": "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
                "heading_path": "5. Conclusions",
                "summary_line": "This early single-shot compressive spectral imaging paper is treated as an SCI predecessor in the retrieved evidence.",
                "why_line": "This hit is directly relevant to the SCI question in the library-wide list query.",
            },
        },
    ]

    filtered = reference_ui._filter_refs_hits_by_prompt_focus(prompt, hits)

    assert len(filtered) == 3


def test_should_try_refs_hit_relevance_gate_skips_llm_for_multi_paper_list_query():
    prompt = "Which papers in my library mention SCI?"
    hits = [
        {"meta": {"source_path": "doc1.md"}},
        {"meta": {"source_path": "doc2.md"}},
    ]

    assert reference_ui._should_try_refs_hit_relevance_gate(prompt, hits, guide_mode=False) is False


def test_enrich_refs_payload_keeps_multiple_hits_for_multi_paper_list_despite_large_score_gap(monkeypatch):
    refs = {
        314: {
            "prompt": "有哪几篇文章提到了SCI（单次曝光压缩成像）",
            "hits": [
                {
                    "text": "Snapshot Compressive Imaging (SCI) is introduced in the abstract.",
                    "score": 8.9,
                    "meta": {
                        "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                        "ref_pack_state": "ready",
                    },
                },
                {
                    "text": "The paper repeatedly mentions Snapshot Compressive Imaging (SCI).",
                    "score": 3.2,
                    "meta": {
                        "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                        "ref_pack_state": "ready",
                    },
                },
                {
                    "text": "This early single-shot compressive spectral imaging paper is treated as an SCI predecessor.",
                    "score": 3.2,
                    "meta": {
                        "source_path": r"db\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.en.md",
                        "ref_pack_state": "ready",
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source = str(((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")
        name = source.split("\\")[-1].replace(".en.md", ".pdf")
        return {
            "display_name": name,
            "heading_path": "Abstract",
            "summary_line": str(hit.get("text") or ""),
            "why_line": "This hit directly discusses SCI.",
            "score": float(hit.get("score") or 0.0),
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

    pack = dict(out.get(314) or {})
    hits = list(pack.get("hits") or [])
    assert len(hits) == 3
    debug = dict(pack.get("pipeline_debug") or {})
    assert debug.get("prompt_explicitly_requests_multi_paper_list") is True


def test_build_doc_list_refs_payload_uses_lightweight_authoritative_seed_when_doc_list_evidence_is_strong(monkeypatch):
    pack = {
        "prompt": "Which papers in my library mention SCI (Snapshot Compressive Imaging)?",
    }
    doc_list = [
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS.pdf",
            "heading_path": "1. Introduction",
            "summary_line": "The paper introduces Snapshot Compressive Imaging (SCI) in the introduction.",
            "primary_evidence": {
                "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                "source_name": "ICIP-2025-SCIGS.pdf",
                "heading_path": "1. Introduction",
                "snippet": "Snapshot Compressive Imaging (SCI) is introduced for recovering dynamic scene information.",
                "highlight_snippet": "Snapshot Compressive Imaging (SCI) is introduced for recovering dynamic scene information.",
                "block_id": "blk-scigs-intro",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
        }
    ]

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call heavy build_hit_ui_meta")))
    monkeypatch.setattr(
        reference_ui,
        "_maybe_polish_single_ref_hit_card",
        lambda **kwargs: dict(kwargs["ui_meta"]),
    )

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=314,
        pack=pack,
        doc_list=doc_list,
    )

    hit = list(out.get("hits") or [])[0]
    ui = dict(hit.get("ui_meta") or {})
    assert ui.get("heading_path") == "1. Introduction"
    assert "Snapshot Compressive Imaging" in str(ui.get("summary_line") or "")
    assert "sci" in str(ui.get("why_line") or "").lower()
    assert ui.get("summary_generation") in {"section_grounded", "deterministic_grounded"}
    assert ui.get("why_generation") == "deterministic_grounded"
    assert "direct library matches" not in str(ui.get("why_line") or "")


def test_build_doc_list_refs_payload_prefers_stronger_synthesized_primary_over_weak_answer_hit_top(monkeypatch):
    pack = {
        "prompt": "Which papers in my library mention single-photon imaging?",
    }
    doc_list = [
        {
            "source_path": r"db\Frontiers-2024\Frontiers-2024.en.md",
            "source_name": "Frontiers-2024-single-photon.pdf",
            "heading_path": "5 Application / 5.3 Quantum communication",
            "summary_line": "",
            "primary_evidence": {
                "source_path": r"db\Frontiers-2024\Frontiers-2024.en.md",
                "source_name": "Frontiers-2024-single-photon.pdf",
                "heading_path": "5 Application / 5.3 Quantum communication",
                "selection_reason": "answer_hit_top",
            },
        }
    ]
    calls: dict[str, object] = {}

    def fake_build_hit_ui_meta(hit, **kwargs):
        calls["allow_exact_locate"] = kwargs.get("allow_exact_locate")
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        snippet = (
            "This optical imaging section explains how single-photon imaging reconstructs object "
            "images under extremely low-light conditions."
        )
        return {
            "display_name": str(meta.get("source_name") or "Reference"),
            "heading_path": "5 Application / 5.1 Optical imaging",
            "summary_line": snippet,
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "This hit directly discusses single-photon imaging in an optical imaging application section.",
            "why_generation": "deterministic_grounded",
            "score": 8.7,
            "score_pending": False,
            "score_tier": "high",
            "primary_evidence": {
                "source_path": str(meta.get("source_path") or ""),
                "source_name": str(meta.get("source_name") or ""),
                "heading_path": "5 Application / 5.1 Optical imaging",
                "snippet": snippet,
                "highlight_snippet": snippet,
                "block_id": "blk-optical-imaging",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
            "reader_open": {
                "sourcePath": str(meta.get("source_path") or ""),
                "sourceName": str(meta.get("source_name") or ""),
                "headingPath": "5 Application / 5.1 Optical imaging",
                "snippet": snippet,
                "highlightSnippet": snippet,
                "blockId": "blk-optical-imaging",
                "strictLocate": True,
            },
            "primary_evidence_source": "prompt_aligned_block",
            "source_path": str(meta.get("source_path") or ""),
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", lambda **kwargs: dict(kwargs["ui_meta"]))

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=315,
        pack=pack,
        doc_list=doc_list,
    )

    hit = list(out.get("hits") or [])[0]
    ui = dict(hit.get("ui_meta") or {})
    reader_primary = dict(((ui.get("reader_open") or {}).get("primaryEvidence")) or {})

    assert calls.get("allow_exact_locate") is True
    assert ui.get("heading_path") == "5 Application / 5.1 Optical imaging"
    assert "single-photon imaging" in str(ui.get("summary_line") or "")
    assert str((ui.get("primary_evidence") or {}).get("selection_reason") or "") == "prompt_aligned_block"
    assert str((ui.get("primary_evidence") or {}).get("block_id") or "") == "blk-optical-imaging"
    assert str((ui.get("authoritative_primary_evidence") or {}).get("selection_reason") or "") == "answer_hit_top"
    assert str(((ui.get("reader_open") or {}).get("headingPath")) or "") == "5 Application / 5.1 Optical imaging"
    assert str(reader_primary.get("selection_reason") or "") == "prompt_aligned_block"


def test_build_doc_list_refs_payload_keeps_strong_authoritative_primary_when_synthesized_points_elsewhere(monkeypatch):
    pack = {
        "prompt": "Which papers in my library mention single-photon imaging?",
    }
    authoritative_snippet = (
        "This section explains how single-photon imaging reconstructs object images from photon timing signals "
        "in a quantum communication setting."
    )
    doc_list = [
        {
            "source_path": r"db\Frontiers-2024\Frontiers-2024.en.md",
            "source_name": "Frontiers-2024-single-photon.pdf",
            "heading_path": "5 Application / 5.3 Quantum communication",
            "summary_line": authoritative_snippet,
            "primary_evidence": {
                "source_path": r"db\Frontiers-2024\Frontiers-2024.en.md",
                "source_name": "Frontiers-2024-single-photon.pdf",
                "heading_path": "5 Application / 5.3 Quantum communication",
                "snippet": authoritative_snippet,
                "highlight_snippet": authoritative_snippet,
                "block_id": "blk-quantum-communication",
                "selection_reason": "shared_refs_pack",
                "strict_locate": True,
            },
        }
    ]

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        snippet = (
            "This optical imaging section describes low-light image reconstruction for biological sensing."
        )
        return {
            "display_name": str(meta.get("source_name") or "Reference"),
            "heading_path": "5 Application / 5.1 Optical imaging",
            "summary_line": snippet,
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "This hit directly discusses single-photon imaging in an optical imaging application section.",
            "why_generation": "deterministic_grounded",
            "score": 8.7,
            "score_pending": False,
            "score_tier": "high",
            "primary_evidence": {
                "source_path": str(meta.get("source_path") or ""),
                "source_name": str(meta.get("source_name") or ""),
                "heading_path": "5 Application / 5.1 Optical imaging",
                "snippet": snippet,
                "highlight_snippet": snippet,
                "block_id": "blk-optical-imaging",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
            "reader_open": {
                "sourcePath": str(meta.get("source_path") or ""),
                "sourceName": str(meta.get("source_name") or ""),
                "headingPath": "5 Application / 5.1 Optical imaging",
                "snippet": snippet,
                "highlightSnippet": snippet,
                "blockId": "blk-optical-imaging",
                "strictLocate": True,
            },
            "primary_evidence_source": "prompt_aligned_block",
            "source_path": str(meta.get("source_path") or ""),
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", lambda **kwargs: dict(kwargs["ui_meta"]))

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=316,
        pack=pack,
        doc_list=doc_list,
    )

    hit = list(out.get("hits") or [])[0]
    ui = dict(hit.get("ui_meta") or {})
    reader_open = dict(ui.get("reader_open") or {})
    reader_primary = dict(reader_open.get("primaryEvidence") or {})

    assert ui.get("heading_path") == "5 Application / 5.3 Quantum communication"
    assert "photon timing signals" in str(ui.get("summary_line") or "")
    assert str((ui.get("primary_evidence") or {}).get("selection_reason") or "") == "shared_refs_pack"
    assert str((ui.get("primary_evidence") or {}).get("block_id") or "") == "blk-quantum-communication"
    assert str((ui.get("authoritative_primary_evidence") or {}).get("selection_reason") or "") == "shared_refs_pack"
    assert str(reader_open.get("headingPath") or "") == "5 Application / 5.3 Quantum communication"
    assert str(reader_primary.get("selection_reason") or "") == "shared_refs_pack"
    assert str(reader_primary.get("block_id") or "") == "blk-quantum-communication"


def test_build_doc_list_refs_payload_polishes_each_doc_list_hit(monkeypatch):
    pack = {
        "prompt": "Which papers in my library mention SCI (Snapshot Compressive Imaging)?",
    }
    doc_list = [
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS.pdf",
            "heading_path": "1. Introduction",
            "summary_line": "The paper introduces Snapshot Compressive Imaging (SCI).",
        },
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "The paper discusses Snapshot Compressive Imaging (SCI) for 3D reconstruction.",
        },
        {
            "source_path": r"db\OE-2007\OE-2007.en.md",
            "source_name": "OE-2007.pdf",
            "heading_path": "5. Conclusions",
            "summary_line": "The paper presents a single-shot compressive spectral imaging approach.",
        },
    ]
    polished_titles: list[str] = []
    allow_flags: list[bool] = []

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        title = str(meta.get("display_name") or meta.get("source_name") or "Reference").strip()
        heading_path = str(meta.get("heading_path") or "").strip()
        text = str(hit.get("text") or "").strip()
        return {
            "display_name": title,
            "heading_path": heading_path,
            "summary_line": text,
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": f"generic::{title}",
            "why_generation": "deterministic_grounded",
            "score": 8.8,
            "score_pending": False,
            "score_tier": "high",
            "reader_open": {"sourcePath": str(meta.get("source_path") or "")},
            "source_path": str(meta.get("source_path") or ""),
        }

    def fake_polish_single_ref_hit_card(*, prompt, hit, ui_meta, allow_expensive_llm):
        del prompt, hit
        ui = dict(ui_meta or {})
        title = str(ui.get("display_name") or "").strip()
        allow_flags.append(bool(allow_expensive_llm))
        polished_titles.append(title)
        ui["why_line"] = f"polished::{title}"
        return ui

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", fake_polish_single_ref_hit_card)

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=2718,
        pack=pack,
        doc_list=doc_list,
    )

    hits = [hit for hit in list(out.get("hits") or []) if isinstance(hit, dict)]
    titles = [str(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("display_name") or "").strip() for hit in hits]
    why_lines = [str(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("why_line") or "").strip() for hit in hits]
    assert sorted(polished_titles) == sorted(titles)
    assert allow_flags == [False, False, False]
    assert why_lines == [f"polished::{title}" for title in titles]


def test_build_doc_list_refs_payload_reuses_llm_pack_copy_without_repolish(monkeypatch):
    pack = {
        "prompt": "Which papers in my library mention SCI (Snapshot Compressive Imaging)?",
    }
    source_path = r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md"
    doc_list = [
        {
            "source_path": source_path,
            "source_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "Abstract",
            "summary_line": "The work uses Snapshot Compressive Imaging (SCI) as the sensing setup and recovers a 3D scene representation from a single compressed observation.",
            "summary_generation": "llm_pack",
            "why_line": "Because the abstract explicitly names Snapshot Compressive Imaging (SCI), it is a direct match for papers that mention SCI.",
            "why_generation": "llm_pack",
            "primary_evidence": {
                "source_path": source_path,
                "source_name": "CVPR-2024-SCINeRF.pdf",
                "heading_path": "Abstract",
                "snippet": "We explore Snapshot Compressive Imaging for recovering the underlying 3D scene representation from a single temporal compressed image.",
                "highlight_snippet": "We explore Snapshot Compressive Imaging for recovering the underlying 3D scene representation from a single temporal compressed image.",
                "block_id": "blk-scinerf-abstract",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
        }
    ]

    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)

    def fail_force_llm(**kwargs):
        raise AssertionError(f"unexpected repolish for {kwargs.get('ui_meta')}")

    monkeypatch.setattr(reference_ui, "_force_llm_ground_ref_hit_card_copy", fail_force_llm)

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=2719,
        pack=pack,
        doc_list=doc_list,
        allow_expensive_llm=True,
        apply_copy_polish=True,
    )

    hits = [hit for hit in list(out.get("hits") or []) if isinstance(hit, dict)]
    assert len(hits) == 1
    ui = dict(hits[0].get("ui_meta") or {})
    assert ui.get("summary_generation") == "llm_pack"
    assert ui.get("why_generation") == "llm_pack"
    assert "Snapshot Compressive Imaging (SCI)" in str(ui.get("summary_line") or "")
    assert "Snapshot Compressive Imaging (SCI)" in str(ui.get("why_line") or "")


def test_maybe_polish_refs_card_copy_parallel_preserves_hit_order(monkeypatch):
    hits = [
        {
            "ui_meta": {
                "display_name": "Paper A.pdf",
                "summary_line": "Summary A",
                "summary_kind": "guide",
                "why_line": "Why A",
            }
        },
        {
            "ui_meta": {
                "display_name": "Paper B.pdf",
                "summary_line": "Summary B",
                "summary_kind": "guide",
                "why_line": "Why B",
            }
        },
        {
            "ui_meta": {
                "display_name": "Paper C.pdf",
                "summary_line": "Summary C",
                "summary_kind": "guide",
                "why_line": "Why C",
            }
        },
    ]
    observed: dict[str, object] = {"submitted": []}

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            submitted = observed.get("submitted")
            if isinstance(submitted, list):
                submitted.append(args[0])
            return FakeFuture(fn(*args))

    def fake_polish_single_ref_hit_card(*, prompt, hit, ui_meta, allow_expensive_llm):
        del prompt, hit, allow_expensive_llm
        ui = dict(ui_meta or {})
        ui["why_line"] = f"polished::{ui.get('display_name')}"
        return ui

    monkeypatch.setenv("KB_REFS_CARD_POLISH_TOP_N", "4")
    monkeypatch.setenv("KB_REFS_CARD_POLISH_MAX_WORKERS", "3")
    monkeypatch.setattr(reference_ui, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(reference_ui, "as_completed", lambda futs: list(reversed(list(futs))))
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", fake_polish_single_ref_hit_card)

    out = reference_ui._maybe_polish_refs_card_copy(
        prompt="Which papers discuss SCI?",
        hits=hits,
        guide_mode=False,
    )

    titles = [str(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("display_name") or "").strip() for hit in out]
    why_lines = [str(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("why_line") or "").strip() for hit in out]
    assert observed.get("max_workers") == 3
    assert observed.get("submitted") == [0, 1, 2]
    assert titles == ["Paper A.pdf", "Paper B.pdf", "Paper C.pdf"]
    assert why_lines == [f"polished::{title}" for title in titles]


def test_build_doc_list_refs_payload_batches_authoritative_card_polish(monkeypatch):
    pack = {
        "prompt": "How does the library evidence around Snapshot Compressive Imaging (SCI) line up?",
    }
    doc_list = [
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "Abstract",
            "summary_line": "Snapshot Compressive Imaging (SCI) is used for 3D scene recovery.",
            "primary_evidence": {
                "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
                "source_name": "CVPR-2024-SCINeRF.pdf",
                "heading_path": "Abstract",
                "snippet": "Snapshot Compressive Imaging (SCI) is used for 3D scene recovery.",
                "highlight_snippet": "Snapshot Compressive Imaging (SCI) is used for 3D scene recovery.",
                "block_id": "blk-scinerf-abstract",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
        },
        {
            "source_path": r"db\OE-2007\OE-2007.en.md",
            "source_name": "OE-2007.pdf",
            "heading_path": "5. Conclusions",
            "summary_line": "The paper presents single-shot compressive spectral imaging.",
            "primary_evidence": {
                "source_path": r"db\OE-2007\OE-2007.en.md",
                "source_name": "OE-2007.pdf",
                "heading_path": "5. Conclusions",
                "snippet": "The paper presents single-shot compressive spectral imaging.",
                "highlight_snippet": "The paper presents single-shot compressive spectral imaging.",
                "block_id": "blk-oe-conclusion",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
        },
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS.pdf",
            "heading_path": "Abstract",
            "summary_line": "SCI is framed as a way to capture dynamic scenes efficiently.",
            "primary_evidence": {
                "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
                "source_name": "ICIP-2025-SCIGS.pdf",
                "heading_path": "Abstract",
                "snippet": "SCI is framed as a way to capture dynamic scenes efficiently.",
                "highlight_snippet": "SCI is framed as a way to capture dynamic scenes efficiently.",
                "block_id": "blk-scigs-abstract",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
        },
    ]
    observed: dict[str, object] = {}
    fallback_calls: list[str] = []

    def fake_batch_polish(*, prompt, cards_payload, card_count):
        observed["prompt"] = prompt
        observed["card_count"] = card_count
        observed["cards_payload"] = cards_payload
        return (
            (1, "The paper uses SCI to recover 3D scenes with a NeRF formulation.", "It explicitly names SCI in the abstract."),
            (2, "The paper presents single-shot compressive spectral imaging as an SCI-related predecessor.", "It is relevant as an early compressive imaging precursor to SCI."),
            (3, "The paper frames SCI as an efficient way to capture dynamic scenes before reconstruction.", "It directly discusses SCI in the abstract."),
        )

    monkeypatch.setattr(reference_ui, "_llm_batch_polish_ref_card_copy_v1", fake_batch_polish)

    def fake_single_polish(**kwargs):
        ui = dict(kwargs["ui_meta"])
        title = str(ui.get("display_name") or "")
        fallback_calls.append(title)
        ui["summary_line"] = f"Single LLM summary::{title}"
        ui["summary_generation"] = "llm_grounded"
        ui["why_line"] = f"Single LLM why::{title}"
        ui["why_generation"] = "llm_grounded"
        return ui

    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", fake_single_polish)

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=4010,
        pack=pack,
        doc_list=doc_list,
        allow_expensive_llm=True,
        apply_copy_polish=True,
    )

    hits = [hit for hit in list(out.get("hits") or []) if isinstance(hit, dict)]
    assert observed.get("card_count") == 3
    assert "Card 1" in str(observed.get("cards_payload") or "")
    assert len(hits) == 3
    assert len(fallback_calls) < 3
    assert all(str(((hit.get("ui_meta") or {}).get("summary_generation")) or "") == "llm_grounded" for hit in hits)
    assert all(str(((hit.get("ui_meta") or {}).get("why_generation")) or "") == "llm_grounded" for hit in hits)


def test_build_doc_list_refs_payload_keeps_sci_predecessor_why_line_honest_after_polish(monkeypatch):
    pack = {
        "prompt": "Which papers in my library mention SCI (Snapshot Compressive Imaging)?",
    }
    doc_list = [
        {
            "source_path": r"db\OE-2007\OE-2007.en.md",
            "source_name": "OE-2007.pdf",
            "heading_path": "5. Conclusions",
            "summary_line": "The paper presents a single-shot compressive spectral imaging approach.",
            "topic_match_kind": "sci_related_predecessor",
        }
    ]

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        return {
            "display_name": "OE-2007.pdf",
            "heading_path": str(meta.get("heading_path") or "5. Conclusions"),
            "summary_line": str(hit.get("text") or "").strip(),
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "This hit directly discusses Snapshot Compressive Imaging (SCI).",
            "why_generation": "deterministic_grounded",
            "score": 8.6,
            "score_pending": False,
            "score_tier": "high",
            "reader_open": {"sourcePath": str(meta.get("source_path") or "")},
            "source_path": str(meta.get("source_path") or ""),
        }

    def fake_polish_single_ref_hit_card(*, prompt, hit, ui_meta, allow_expensive_llm):
        del prompt, hit, allow_expensive_llm
        ui = dict(ui_meta or {})
        ui["why_line"] = "This hit directly discusses Snapshot Compressive Imaging (SCI)."
        return ui

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", fake_polish_single_ref_hit_card)

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=2719,
        pack=pack,
        doc_list=doc_list,
    )

    hit = list(out.get("hits") or [])[0]
    ui = dict(hit.get("ui_meta") or {})
    summary_line = str(ui.get("summary_line") or "")
    why_line = str(ui.get("why_line") or "")
    assert "single-shot" in summary_line.lower()
    assert not summary_line.lower().startswith("snapshot compressive imaging:")
    assert "single-shot compressive spectral imaging" in why_line
    assert ("exact SCI term match" in why_line) or ("SCI 术语命中" in why_line)
    assert "directly discusses Snapshot Compressive Imaging (SCI)" not in why_line


def test_refs_prompt_focus_terms_detects_sci_inside_chinese_prompt():
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"

    terms = reference_ui._refs_prompt_focus_terms(prompt)

    assert any("sci" in term for term in terms)
    assert any("snapshot compressive imaging" in term for term in terms)


def test_apply_doc_list_topic_match_hints_upgrades_generic_sci_why_line():
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"

    out = reference_ui._apply_doc_list_topic_match_hints(
        prompt=prompt,
        raw_item={
            "topic_match_kind": "explicit_sci_mention",
            "heading_path": "2. Related Work",
            "source_name": "CVPR-2024-SCINeRF.pdf",
        },
        ui_meta={
            "display_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) technique for recovering the underlying 3D scene representation.",
            "why_line": "\u8fd9\u6761\u547d\u4e2d\u843d\u5728\u201c2. Related Work\u201d\uff0c\u80fd\u76f4\u63a5\u63d0\u4f9b\u548c\u5f53\u524d\u95ee\u9898\u76f8\u5173\u7684\u5b9a\u4e49\u3001\u65b9\u6cd5\u6216\u7ed3\u679c\u8bc1\u636e\u3002",
        },
    )

    why_line = str(out.get("why_line") or "")
    assert "Snapshot Compressive Imaging" in why_line
    assert "SCI" in why_line


def test_expand_ref_summary_candidates_does_not_prefix_sci_predecessor_sentence():
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"

    candidates = reference_ui._expand_ref_summary_candidates(
        "This paper describes a single-shot spectral imaging approach based on the concept of compressive sensing.",
        prompt=prompt,
        title="OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
        prefer_zh=True,
        allow_llm_translate=False,
    )

    assert candidates
    assert all(not cand.lower().startswith("snapshot compressive imaging:") for cand in candidates)


def test_pick_best_prompt_aligned_ref_summary_candidate_prefers_reader_friendly_sci_copy():
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"
    title = "ICIP-2025-SCIGS.pdf"
    source_path = r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md"

    chosen = reference_ui._pick_best_prompt_aligned_ref_summary_candidate(
        [
            {
                "summary": "Video Snapshot Compressive Imaging (SCI) technology has been developed. A SCI system usually has two components: a hardware encoder and a software decoder.",
                "heading_path": "1. Introduction",
            },
            {
                "summary": "snapshot compressive imaging: In the process of capturing compressed images in the SCI system, an exposure time is divided into $B$ time intervals by the corresponding $B$ encoding masks.",
                "heading_path": "3. Method / 3.2. Snapshot Compressive Imaging Model",
                "raw_focus_surface": "3. Method / 3.2. Snapshot Compressive Imaging Model In the process of capturing compressed images in the SCI system.",
                "source_kind": "source_block",
            },
        ],
        prompt=prompt,
        source_path=source_path,
        title=title,
        anchor_target_kind="",
        anchor_target_number=0,
    )

    assert str(chosen.get("heading_path") or "") == "1. Introduction"
    assert str(chosen.get("summary") or "").startswith("Video Snapshot Compressive Imaging (SCI) technology has been developed.")


def test_pick_best_prompt_aligned_ref_summary_candidate_skips_fragmentary_focus_prefixed_copy():
    prompt = "Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?"
    title = "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf"
    source_path = (
        r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
        r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    )

    chosen = reference_ui._pick_best_prompt_aligned_ref_summary_candidate(
        [
            {
                "summary": "Fourier single-pixel imaging: of Fourier coefficients.",
                "heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                "raw_focus_surface": "2.1 Principle of HSI and FSI of Fourier coefficients.",
                "source_kind": "source_block",
            },
            {
                "summary": "The paper compares Hadamard and Fourier single-pixel imaging and explains how Fourier coefficients are sampled in the reconstruction process.",
                "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                "raw_focus_surface": "2.2 Basis patterns generation The paper compares Hadamard and Fourier single-pixel imaging and explains how Fourier coefficients are sampled.",
                "source_kind": "source_block",
            },
        ],
        prompt=prompt,
        source_path=source_path,
        title=title,
        anchor_target_kind="",
        anchor_target_number=0,
    )

    assert str(chosen.get("heading_path") or "") == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert "compares Hadamard and Fourier" in str(chosen.get("summary") or "")


def test_pick_best_prompt_aligned_ref_summary_candidate_skips_define_style_focus_prefixed_copy():
    prompt = "Which paper in my library most directly defines dynamic supersampling? Please point me to the source section."
    title = "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf"
    source_path = (
        r"db\SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling"
        r"\SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.en.md"
    )

    chosen = reference_ui._pick_best_prompt_aligned_ref_summary_candidate(
        [
            {
                "summary": "dynamic supersampling: Because the pixel geometry of each frame in our single-pixel imaging system is defined by the masking patterns applied to the DMD and used to measure the image, it is possible to perform digital supersampling.",
                "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
                "raw_focus_surface": "INTRODUCTION / Spatially variant digital supersampling dynamic supersampling: Because the pixel geometry of each frame...",
            },
            {
                "summary": "The paper defines dynamic supersampling by shifting the effective pixel boundaries between frames so complementary spatial information can be fused for higher-resolution reconstruction.",
                "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
                "raw_focus_surface": "INTRODUCTION / Spatially variant digital supersampling The paper defines dynamic supersampling by shifting the effective pixel boundaries between frames.",
                "source_kind": "source_block",
            },
        ],
        prompt=prompt,
        source_path=source_path,
        title=title,
        anchor_target_kind="",
        anchor_target_number=0,
    )

    assert str(chosen.get("summary") or "").startswith("The paper defines dynamic supersampling")


def test_choose_prompt_aligned_ref_summary_candidate_from_source_blocks_skips_title_like_block(monkeypatch):
    prompt = "\u6709\u54ea\u51e0\u7bc7\u6587\u7ae0\u63d0\u5230\u4e86SCI\uff08\u5355\u6b21\u66dd\u5149\u538b\u7f29\u6210\u50cf\uff09"
    title = "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf"

    monkeypatch.setattr(reference_ui, "_resolve_source_md_path", lambda source_path: Path(source_path))
    monkeypatch.setattr(
        reference_ui,
        "load_source_blocks",
        lambda _md_path: [
            {
                "text": "SCIGS: 3D Gaussians Splatting from A Snapshot Compressive Image",
                "heading_path": "",
                "kind": "heading",
            },
            {
                "text": "Snapshot Compressive Imaging (SCI) offers a possibility for capturing information in high-speed dynamic scenes, requiring efficient reconstruction method to recover scene information.",
                "heading_path": "Abstract",
                "kind": "paragraph",
            },
            {
                "text": "In the process of capturing compressed images in the SCI system, an exposure time is divided into $B$ time intervals by the corresponding $B$ encoding masks.",
                "heading_path": "3. Method / 3.2. Snapshot Compressive Imaging Model",
                "kind": "paragraph",
            },
        ],
    )

    chosen = reference_ui._choose_prompt_aligned_ref_summary_candidate_from_source_blocks(
        prompt=prompt,
        source_path=r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
        title=title,
        allow_llm_translate=False,
    )

    assert str(chosen.get("heading_path") or "") == "Abstract"
    assert str(chosen.get("summary") or "").startswith("Snapshot Compressive Imaging (SCI) offers a possibility")
    assert not str(chosen.get("summary") or "").lower().startswith("snapshot compressive imaging:")


def test_build_doc_list_refs_payload_repairs_mixed_quote_artifacts_after_polish(monkeypatch):
    pack = {
        "prompt": "有哪几篇文章提到了SCI（单次曝光压缩成像）",
    }
    doc_list = [
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "The paper explicitly mentions Snapshot Compressive Imaging (SCI).",
            "topic_match_kind": "explicit_sci_mention",
        }
    ]

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        return {
            "display_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": str(meta.get("heading_path") or "2. Related Work"),
            "summary_line": str(hit.get("text") or "").strip(),
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "这条命中直接讨论了 SCI。",
            "why_generation": "deterministic_grounded",
            "score": 8.9,
            "score_pending": False,
            "score_tier": "high",
            "reader_open": {"sourcePath": str(meta.get("source_path") or "")},
            "source_path": str(meta.get("source_path") or ""),
        }

    def fake_polish_single_ref_hit_card(*, prompt, hit, ui_meta, allow_expensive_llm):
        del prompt, hit, allow_expensive_llm
        ui = dict(ui_meta or {})
        ui["why_line"] = "Related Work’中明确提及Snapshot Compressive Imaging（SCI），直接回应用户查询。"
        return ui

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", fake_polish_single_ref_hit_card)

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=2720,
        pack=pack,
        doc_list=doc_list,
    )

    hit = list(out.get("hits") or [])[0]
    ui = dict(hit.get("ui_meta") or {})
    why_line = str(ui.get("why_line") or "")
    assert "“Related Work”中明确提及" in why_line
    assert "Related Work’中" not in why_line


def test_normalize_ref_copy_text_keeps_balanced_heading_quotes():
    text = "“Related Work”中明确提及 Snapshot Compressive Imaging（SCI）。"

    out = reference_ui._normalize_ref_copy_text(text)

    assert out == text


def test_build_doc_list_refs_payload_filters_bound_source_in_guide_mode(monkeypatch):
    pack = {
        "prompt": "Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?",
    }
    doc_list = [
        {
            "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
            "source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
            "heading_path": "Acquisition and image reconstruction strategies",
            "summary_line": "The bound paper reviews single-pixel imaging and briefly mentions Fourier patterns.",
        },
        {
            "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
            "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "2.2 Basis patterns generation",
            "summary_line": "The paper directly compares Hadamard and Fourier single-pixel imaging.",
        },
    ]

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        return {
            "display_name": str(meta.get("source_name") or "Reference"),
            "heading_path": str(meta.get("heading_path") or ""),
            "summary_line": str(hit.get("text") or "").strip(),
            "why_line": "polished external paper",
            "score": 8.8,
            "score_pending": False,
            "score_tier": "high",
            "reader_open": {"sourcePath": str(meta.get("source_path") or "")},
            "source_path": str(meta.get("source_path") or ""),
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", lambda **kwargs: dict(kwargs["ui_meta"]))

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=3001,
        pack=pack,
        doc_list=doc_list,
        guide_mode=True,
        guide_source_path=r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
        guide_source_name="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
    )

    hits = list(out.get("hits") or [])
    assert len(hits) == 1
    ui = dict((hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {})
    assert str(ui.get("display_name") or "") == "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf"
    guide_filter = dict(out.get("guide_filter") or {})
    assert guide_filter.get("active") is True
    assert guide_filter.get("hidden_self_source") is True
    assert int(guide_filter.get("filtered_hit_count") or 0) == 1
    pipeline_debug = dict(out.get("pipeline_debug") or {})
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 1
    assert int(pipeline_debug.get("filtered_self_hit_count") or 0) == 1
    assert str(out.get("display_state") or "") == "ready"


def test_build_doc_list_refs_payload_repairs_title_echo_summary_from_primary_evidence(monkeypatch):
    pack = {
        "prompt": "Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?",
    }
    doc_list = [
        {
            "source_path": (
                r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
            ),
            "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "summary_line": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
            "primary_evidence": {
                "source_path": (
                    r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
                ),
                "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
                "snippet": (
                    "The paper analyzes reconstruction quality, efficiency, and robustness for "
                    "Hadamard and Fourier single-pixel imaging in numerical simulations."
                ),
                "selection_reason": "answer_hit_top",
            },
        }
    ]

    def fake_build_hit_ui_meta(hit, **kwargs):
        del hit, kwargs
        return {
            "display_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "summary_line": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
            "why_line": "placeholder",
            "score": 8.8,
            "score_pending": False,
            "score_tier": "high",
            "reader_open": {
                "sourcePath": (
                    r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
                )
            },
            "source_path": (
                r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
            ),
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", lambda **kwargs: dict(kwargs["ui_meta"]))

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=3003,
        pack=pack,
        doc_list=doc_list,
        guide_mode=True,
        guide_source_path=r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
        guide_source_name="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
    )

    hits = list(out.get("hits") or [])
    assert len(hits) == 1
    ui = dict((hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {})
    summary_line = str(ui.get("summary_line") or "")
    assert "reconstruction quality" in summary_line
    assert "Hadamard single-pixel imaging versus Fourier single-pixel imaging" != summary_line


def test_build_doc_list_refs_payload_replaces_why_like_summary_with_prompt_aligned_fallback(monkeypatch):
    pack = {
        "prompt": "Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?",
    }
    doc_list = [
        {
            "source_path": (
                r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
            ),
            "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "summary_line": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
            "primary_evidence": {
                "source_path": (
                    r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
                ),
                "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
                "snippet": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "selection_reason": "prompt_aligned_block",
                "strict_locate": True,
            },
        }
    ]

    why_like_summary = (
        "This hit directly covers 'Fourier single-pixel imaging' in "
        "'3. Comparison of experiment / 3.1 Numerical simulations', so it is a good entry point."
    )

    def fake_build_hit_ui_meta(hit, **kwargs):
        del hit, kwargs
        return {
            "display_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "summary_line": why_like_summary,
            "why_line": why_like_summary,
            "score": 8.8,
            "score_pending": False,
            "score_tier": "high",
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "reader_open": {
                "sourcePath": (
                    r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                    r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
                )
            },
            "source_path": (
                r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging"
                r"\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
            ),
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", lambda **kwargs: dict(kwargs["ui_meta"]))

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=3004,
        pack=pack,
        doc_list=doc_list,
        guide_mode=True,
        guide_source_path=r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
        guide_source_name="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
    )

    hits = list(out.get("hits") or [])
    assert len(hits) == 1
    ui = dict((hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {})
    summary_line = str(ui.get("summary_line") or "")
    assert "good entry point" not in summary_line.lower()
    assert ("The paper discusses" in summary_line) or ("讨论了" in summary_line)


def test_build_doc_list_refs_payload_hides_self_only_guide_doc_list(monkeypatch):
    pack = {
        "prompt": "Besides this paper, what other papers in my library discuss ADMM?",
    }
    doc_list = [
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "This paper does not discuss ADMM.",
        }
    ]

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", lambda **kwargs: dict(kwargs["ui_meta"]))

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=3002,
        pack=pack,
        doc_list=doc_list,
        guide_mode=True,
        guide_source_path=r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
        guide_source_name="CVPR-2024-SCINeRF.pdf",
    )

    assert list(out.get("hits") or []) == []
    guide_filter = dict(out.get("guide_filter") or {})
    assert guide_filter.get("active") is True
    assert guide_filter.get("hidden_self_source") is True
    assert int(guide_filter.get("filtered_hit_count") or 0) == 1
    pipeline_debug = dict(out.get("pipeline_debug") or {})
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 0
    assert int(pipeline_debug.get("filtered_self_hit_count") or 0) == 1
    assert str(out.get("display_state") or "") == "hidden_by_guide"
    assert str(out.get("suppression_reason") or "") == "guide_self_source_only"


def test_build_doc_list_refs_payload_keeps_bound_source_for_guide_location(monkeypatch):
    pack = {
        "prompt": "Summarize Figure 1 in this paper and point me to the source section.",
    }
    doc_list = [
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF.pdf",
            "heading_path": "3. Method",
            "summary_line": "Figure 1 explains the SCINeRF pipeline.",
        }
    ]

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_polish_single_ref_hit_card", lambda **kwargs: dict(kwargs["ui_meta"]))

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=3004,
        pack=pack,
        doc_list=doc_list,
        guide_mode=True,
        guide_source_path=r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
        guide_source_name="CVPR-2024-SCINeRF.pdf",
    )

    assert len(list(out.get("hits") or [])) == 1
    guide_filter = dict(out.get("guide_filter") or {})
    assert guide_filter.get("active") is True
    assert guide_filter.get("hidden_self_source") is False
    assert int(guide_filter.get("filtered_hit_count") or 0) == 0
    pipeline_debug = dict(out.get("pipeline_debug") or {})
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 1
    assert int(pipeline_debug.get("filtered_self_hit_count") or 0) == 0
    assert str(out.get("display_state") or "") == "ready"
    assert str(out.get("suppression_reason") or "") == ""


def test_build_doc_list_refs_payload_marks_empty_authoritative_guide_doc_list():
    pack = {
        "prompt": "Besides this paper, what other papers in my library discuss ADMM?",
    }

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=3003,
        pack=pack,
        doc_list=[],
        guide_mode=True,
        guide_source_path=r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
        guide_source_name="CVPR-2024-SCINeRF.pdf",
    )

    assert list(out.get("hits") or []) == []
    guide_filter = dict(out.get("guide_filter") or {})
    assert guide_filter.get("active") is True
    assert guide_filter.get("hidden_self_source") is True
    assert int(guide_filter.get("filtered_hit_count") or 0) == 0
    pipeline_debug = dict(out.get("pipeline_debug") or {})
    assert pipeline_debug.get("doc_list_authoritative") is True
    assert pipeline_debug.get("guide_active") is True
    assert int(pipeline_debug.get("raw_hit_count") or 0) == 0
    assert int(pipeline_debug.get("filtered_self_hit_count") or 0) == 0
    assert str(out.get("display_state") or "") == "hidden_by_guide"
    assert str(out.get("suppression_reason") or "") == "guide_self_source_only"


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


def test_summary_line_needs_polish_for_fragmentary_focus_prefixed_clause():
    assert reference_ui._summary_line_needs_polish(
        prompt="Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?",
        title="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        summary_line="Fourier single-pixel imaging: of Fourier coefficients.",
    )


def test_summary_line_needs_polish_for_compare_style_focus_prefixed_copy():
    assert reference_ui._summary_line_needs_polish(
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        title="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        summary_line="Hadamard single-pixel imaging and Fourier single-pixel imaging: Figure 1 shows the comparison between the Hadamard and Fourier basis patterns.",
    )


def test_summary_line_needs_polish_for_define_style_focus_prefixed_copy():
    assert reference_ui._summary_line_needs_polish(
        prompt="Which paper in my library most directly defines dynamic supersampling? Please point me to the source section.",
        title="SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
        summary_line="dynamic supersampling: Because the pixel geometry of each frame in our single-pixel imaging system is defined by the masking patterns applied to the DMD and used to measure the image, it is possible to perform digital supersampling.",
    )


def test_summary_line_allows_complete_lowercase_technical_sentence():
    assert not reference_ui._summary_line_needs_polish(
        prompt="Which papers in my library mention single-photon imaging?",
        title="Frontiers-2024-Emerging single-photon detection technique for high-performance photodetector.pdf",
        summary_line=(
            "single-photon imaging can reconstruct object images from photon timing signals in the optical imaging section."
        ),
    )


def test_summary_line_needs_polish_for_why_like_copy():
    assert reference_ui._summary_line_needs_polish(
        prompt="Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?",
        title="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        summary_line=(
            "This hit directly covers 'Fourier single-pixel imaging' in "
            "'3. Comparison of experiment / 3.1 Numerical simulations', so it is a good entry point."
        ),
    )


def test_summary_line_needs_polish_for_synthetic_location_discussion_copy():
    assert reference_ui._summary_line_needs_polish(
        prompt="我做单像素实验，Hadamard 和 Fourier 到底该怎么选？",
        title="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        summary_line="该文在“Hadamard single-pixel imaging versus Fourier single-pixel imaging / 3. Comparison of experiment / 3.1 Numerical simulations”讨论了“single pixel imaging”。",
    )


def test_summary_line_needs_polish_for_missing_subject_template_tail():
    assert reference_ui._summary_line_needs_polish(
        prompt="Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?",
        title="OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        summary_line="Comparison of experiment / 3.1 Numerical simulations”讨论了“Fourier single-pixel imaging”。",
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


def test_enrich_refs_payload_polishes_explicit_multi_paper_list(monkeypatch):
    refs = {
        45: {
            "prompt": "Which papers should I read first for single-pixel imaging? Please list several papers and what each one is useful for.",
            "hits": [
                {
                    "text": "This review introduces single-pixel imaging principles and explains compressed sensing trade-offs.",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 81.0, "bm25": 5.0, "semantic_score": 8.0},
                    },
                },
                {
                    "text": "This paper proposes adaptive foveated single-pixel imaging with dynamic supersampling.",
                    "meta": {
                        "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
                        "ref_pack_state": "ready",
                        "ref_rank": {"llm": 79.0, "bm25": 4.8, "semantic_score": 7.8},
                    },
                },
            ],
        }
    }
    polish_calls: list[list[str]] = []

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        source_path = str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("source_path") or "")).strip()
        title = source_path.rsplit("\\", 1)[-1].replace(".en.md", ".pdf")
        return {
            "display_name": title,
            "source_path": source_path,
            "heading_path": "Abstract",
            "summary_line": str(hit.get("text") or "").strip(),
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "This hit is directly relevant because it answers the user's question.",
            "why_generation": "deterministic_grounded",
            "score": 8.0,
            "score_pending": False,
            "reader_open": {"sourcePath": source_path},
        }

    def fake_polish_refs_card_copy(*, prompt, hits, guide_mode):
        del prompt, guide_mode
        titles = [
            str(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("display_name") or "")
            for hit in hits
        ]
        polish_calls.append(titles)
        out = []
        for hit in hits:
            hit2 = dict(hit)
            ui = dict(hit2.get("ui_meta") or {})
            ui["summary_line"] = f"LLM summary::{ui.get('display_name')}"
            ui["summary_generation"] = "llm_grounded"
            ui["why_line"] = f"LLM why::{ui.get('display_name')}"
            ui["why_generation"] = "llm_grounded"
            hit2["ui_meta"] = ui
            out.append(hit2)
        return out

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", fake_polish_refs_card_copy)

    out = reference_ui.enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    pack = out.get(45) or {}
    hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]

    assert pack.get("pipeline_debug", {}).get("prompt_explicitly_requests_multi_paper_list") is True
    assert pack.get("pipeline_debug", {}).get("llm_polish_allowed") is True
    assert polish_calls and len(polish_calls[0]) == 2
    assert len(hits) == 2
    assert all(str(((hit.get("ui_meta") or {}).get("summary_generation")) or "") == "llm_grounded" for hit in hits)
    assert all(str(((hit.get("ui_meta") or {}).get("why_generation")) or "") == "llm_grounded" for hit in hits)


def test_dedupe_refs_hits_merges_same_section_duplicate_and_prefers_precise_locate():
    prompt = "Which paper explains ADMM reconstruction?"
    loose_hit = {
        "text": "Most existing methods employ ADMM-based optimization for reconstruction.",
        "meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "ref_best_heading_path": "2. Related Work",
        },
        "ui_meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "display_name": "SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "Most existing methods employ ADMM-based optimization for reconstruction.",
            "why_line": "This section mentions ADMM reconstruction.",
            "score": 9.2,
        },
    }
    precise_hit = {
        "text": "Most existing methods employ ADMM-based optimization for reconstruction.",
        "meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "ref_best_heading_path": "2. Related Work",
        },
        "ui_meta": {
            "source_path": r"db\SCINeRF\SCINeRF.en.md",
            "display_name": "SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "Most existing methods employ ADMM-based optimization for reconstruction.",
            "why_line": "This section mentions ADMM reconstruction.",
            "score": 7.1,
            "reader_open": {
                "sourcePath": r"db\SCINeRF\SCINeRF.en.md",
                "headingPath": "2. Related Work",
                "snippet": "Most existing methods employ ADMM-based optimization for reconstruction.",
                "strictLocate": True,
                "blockId": "blk-related-admm",
                "anchorId": "sent-admm",
                "anchorKind": "sentence",
            },
        },
    }

    hits, removed = reference_ui._dedupe_refs_hits_for_display(
        prompt=prompt,
        hits=[loose_hit, precise_hit],
    )

    assert removed == 1
    assert len(hits) == 1
    ui = dict(hits[0].get("ui_meta") or {})
    assert ui.get("merged_duplicate_count") == 1
    assert str((ui.get("reader_open") or {}).get("blockId") or "") == "blk-related-admm"


def test_sort_refs_hits_prefers_precise_llm_card_over_higher_raw_score():
    prompt = "Which paper explains ADMM reconstruction?"
    high_score_loose = {
        "text": "ADMM appears in related work.",
        "meta": {
            "source_path": "paper-a.en.md",
            "ref_best_heading_path": "Related Work",
            "ref_rank": {"display_score": 98.0},
        },
        "ui_meta": {
            "display_name": "Paper A.pdf",
            "heading_path": "Related Work",
            "summary_line": "ADMM appears in related work.",
            "summary_generation": "section_grounded",
            "why_line": "ADMM appears here.",
            "why_generation": "deterministic_grounded",
            "score": 9.8,
        },
    }
    lower_score_precise = {
        "text": "ADMM is used as reconstruction machinery.",
        "meta": {
            "source_path": "paper-b.en.md",
            "ref_best_heading_path": "2. Related Work",
            "ref_rank": {"display_score": 80.0},
        },
        "ui_meta": {
            "display_name": "Paper B.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "The section explains ADMM as reconstruction machinery.",
            "summary_generation": "llm_grounded",
            "why_line": "The matched sentence shows how ADMM is used for reconstruction.",
            "why_generation": "llm_grounded",
            "score": 7.0,
            "reader_open": {
                "sourcePath": "paper-b.en.md",
                "headingPath": "2. Related Work",
                "strictLocate": True,
                "blockId": "blk-admm",
                "anchorId": "sent-admm",
            },
        },
    }

    out = reference_ui._sort_refs_hits_for_display(
        prompt=prompt,
        hits=[high_score_loose, lower_score_precise],
    )

    assert str(((out[0].get("ui_meta") or {}).get("display_name")) or "") == "Paper B.pdf"


def test_enrich_refs_payload_records_deduped_duplicate_count(monkeypatch):
    refs = {
        46: {
            "prompt": "Which paper explains ADMM reconstruction?",
            "hits": [
                {
                    "text": "Most existing methods employ ADMM-based optimization for reconstruction.",
                    "meta": {
                        "source_path": r"db\SCINeRF\SCINeRF.en.md",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "2. Related Work",
                        "explicit_doc_match_score": 8.0,
                        "ref_rank": {"display_score": 9.2, "bm25": 6.0, "deep": 12.0},
                    },
                },
                {
                    "text": "Most existing methods employ ADMM-based optimization for reconstruction.",
                    "meta": {
                        "source_path": r"db\SCINeRF\SCINeRF.en.md",
                        "ref_pack_state": "ready",
                        "ref_best_heading_path": "2. Related Work",
                        "explicit_doc_match_score": 8.0,
                        "ref_rank": {"display_score": 8.8, "bm25": 5.8, "deep": 11.5},
                    },
                },
            ],
        }
    }

    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        return {
            "display_name": "SCINeRF.pdf",
            "source_path": str(meta.get("source_path") or ""),
            "heading_path": str(meta.get("ref_best_heading_path") or ""),
            "summary_line": str(hit.get("text") or ""),
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "This hit is directly relevant because it answers the user's question.",
            "why_generation": "deterministic_grounded",
            "score": 8.0,
            "reader_open": {"sourcePath": str(meta.get("source_path") or ""), "headingPath": str(meta.get("ref_best_heading_path") or "")},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: False)

    out = reference_ui.enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    pack = out.get(46) or {}
    hits = [hit for hit in list(pack.get("hits") or []) if isinstance(hit, dict)]
    ui = dict(hits[0].get("ui_meta") or {}) if hits else {}

    assert len(hits) == 1
    assert ui.get("merged_duplicate_count") == 1
    assert pack.get("pipeline_debug", {}).get("deduped_duplicate_hit_count") == 1


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
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "zh")
    out = reference_ui._build_ref_summary_surface_meta(
        prompt="这篇文章里 dynamic supersampling 是怎么定义的？",
        summary_kind="guide",
    )
    assert out["summary_label"] == "导读"
    assert out["summary_title"] == "这条证据说明什么"


def test_build_ref_summary_surface_meta_auto_prefers_prompt_language(monkeypatch):
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_card_locale, "_refs_card_ui_locale_pref", lambda: "")
    out = reference_ui._build_ref_summary_surface_meta(
        prompt="Which paper in my library discusses dynamic supersampling?",
        summary_kind="guide",
        summary_line="该研究提出了一种空间可变的数字超采样方法。",
    )
    assert out["summary_label"] == "Guide"


def test_align_ref_card_copy_to_user_locale_prefers_chinese_prompt(monkeypatch):
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "auto")
    monkeypatch.setattr(reference_card_locale, "_refs_card_ui_locale_pref", lambda: "")
    monkeypatch.setattr(reference_ui, "_translate_summary_to_zh", lambda text: f"中文概括：{text}")

    summary_line, why_line = reference_ui._align_ref_card_copy_to_user_locale(
        prompt="哪篇文章最直接定义了 dynamic supersampling？",
        display_name="SciAdv-2017.pdf",
        heading_path="INTRODUCTION / Spatially variant digital supersampling",
        summary_line="The paper defines dynamic supersampling by shifting the effective pixel boundaries between frames.",
        why_line="This hit directly defines dynamic supersampling in this section.",
        summary_kind="guide",
        allow_llm_translate=True,
    )

    assert "中文概括" in str(summary_line or "")
    assert reference_ui._has_cjk_text(str(summary_line or ""))
    assert reference_ui._has_cjk_text(str(why_line or ""))


def test_metadata_summary_line_for_ref_card_explains_missing_abstract(monkeypatch):
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "en")
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
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "zh")
    out = reference_ui._build_ref_summary_basis_meta(
        prompt="请总结这篇论文？",
        summary_kind="abstract",
        summary_generation="llm_abstract",
        summary_line="该研究提出了一种新的成像方法。",
    )
    assert out["summary_generation"] == "llm_abstract"
    assert "LLM" in str(out["summary_basis"] or "")
    assert "abstract" in str(out["summary_basis"] or "")


def _legacy_mojibake_build_ref_why_basis_meta_describes_llm_grounded_reason(monkeypatch):
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "zh")
    out = reference_ui._build_ref_why_basis_meta(
        prompt="为什么这篇文献和我的问题相关？",
        why_generation="llm_grounded",
        why_line="这条命中直接解释了 dynamic supersampling 的定义和用途。",
    )
    assert out["why_generation"] == "llm_grounded"
    assert "LLM" in str(out["why_basis"] or "")
    assert "相关性说明" in str(out["why_basis"] or "")


def test_build_ref_why_basis_meta_describes_llm_grounded_reason_utf8_safe(monkeypatch):
    monkeypatch.setattr(reference_card_locale, "_refs_card_locale_pref", lambda: "zh")
    out = reference_ui._build_ref_why_basis_meta(
        prompt="\u4e3a\u4ec0\u4e48\u8fd9\u7bc7\u6587\u732e\u548c\u6211\u7684\u95ee\u9898\u76f8\u5173\uff1f",
        why_generation="llm_grounded",
        why_line="\u8fd9\u6761\u547d\u4e2d\u76f4\u63a5\u89e3\u91ca\u4e86 dynamic supersampling \u7684\u5b9a\u4e49\u548c\u7528\u9014\u3002",
    )
    assert out["why_generation"] == "llm_grounded"
    assert "LLM" in str(out["why_basis"] or "")
    assert "\u76f8\u5173\u6027\u8bf4\u660e" in str(out["why_basis"] or "")


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
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(
        reference_ui,
        "_llm_polish_ref_card_copy_v2",
        lambda **kwargs: (
            "The section defines dynamic supersampling as shifting pixel boundaries across frames to accumulate complementary spatial samples.",
            "This section is relevant because it explains the definition of dynamic supersampling itself rather than mentioning it in passing.",
        ),
    )
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))

    out = reference_ui.enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(43) or {}).get("hits") or [])

    assert len(hits) == 1
    ui_meta = (hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {}
    assert "definition of dynamic supersampling" in str(ui_meta.get("why_line") or "")
    assert str(ui_meta.get("why_generation") or "") == "llm_grounded"
    assert "LLM" in str(ui_meta.get("why_basis") or "")


def test_enrich_refs_payload_skips_llm_filter_for_single_ready_hit(monkeypatch):
    refs = {
        44: {
            "prompt": "Which paper in my library most directly defines dynamic supersampling?",
            "hits": [
                {
                    "text": "Spatially variant digital supersampling shifts pixel boundaries frame by frame to capture complementary spatial samples.",
                    "meta": {
                        "source_path": r"db\\SciAdv-2017\\SciAdv-2017.en.md",
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
            "heading_path": "3. Spatially variant digital supersampling",
            "summary_line": "The section defines dynamic supersampling through frame-varying pixel boundaries.",
            "summary_kind": "guide",
            "summary_generation": "section_grounded",
            "why_line": "This section explains the definition of dynamic supersampling.",
            "why_generation": "deterministic_grounded",
            "score": 8.5,
            "reader_open": {"sourcePath": source_path},
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_refs_card_polish_llm_enabled", lambda: True)
    monkeypatch.setattr(
        reference_ui,
        "_maybe_llm_filter_refs_hits",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("single ready hit should not call llm filter")),
    )
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))

    out = reference_ui.enrich_refs_payload(refs, pdf_root=None, md_root=None, lib_store=None)
    hits = list((out.get(44) or {}).get("hits") or [])

    assert len(hits) == 1


def test_primary_ref_evidence_payload_cleans_markdown_heading_and_keeps_body_sentence():
    reader_open = {
        "sourcePath": r"db\\SciAdv-2017\\paper.en.md",
        "sourceName": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
        "headingPath": "INTRODUCTION / Foveated single-pixel imaging",
        "snippet": (
            "## Foveated single-pixel imaging\n"
            "Single-pixel imaging is based on the measurement of the level of correlation between the scene and a series of patterns."
        ),
        "highlightSnippet": (
            "## Foveated single-pixel imaging\n"
            "Single-pixel imaging is based on structured illumination and detector measurements."
        ),
        "blockId": "blk_intro",
        "anchorId": "p_001",
        "strictLocate": True,
    }

    out = reference_ui._build_primary_ref_evidence_payload(
        source_path=r"db\\SciAdv-2017\\paper.en.md",
        display_name="SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
        reader_open=reader_open,
        selection_reason="prompt_aligned_block",
        score=9.1,
        prompt="我刚开始看单像素成像，先看什么？",
    )

    assert out["snippet"].startswith("Single-pixel imaging is based")
    assert out["highlight_snippet"].startswith("Single-pixel imaging is based")
    assert "##" not in out["snippet"]
    assert "Foveated single-pixel imaging Single-pixel imaging" not in out["snippet"]
    assert out["block_id"] == "blk_intro"
    assert out["strict_locate"] is True


def test_source_block_answer_primary_evidence_strips_title_author_prefix():
    block = {
        "block_id": "blk_lpr_abs",
        "anchor_id": "p_abs",
        "kind": "paragraph",
        "heading_path": "Abstract",
        "text": (
            "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning "
            "Kai Song, Yaoxing Bian,\\ Dong Wang, Runrui Li, Ku Wu, Hongrui Liu, "
            "Chengbing Qin, Jianyong Hu,\\ and Liantuan Xiao* "
            "Single-pixel imaging technology can capture images at wavelengths outside the reach of conventional focal plane array detectors. "
            "However, the limited image quality and lengthy computational times still hinder practical application."
        ),
    }

    out = reference_ui._source_block_to_answer_primary_evidence(
        block=block,
        prompt="深度学习为什么会用于单像素成像？",
        source_path=r"db\\LPR-2025\\paper.en.md",
        display_name="LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
        terms=["single-pixel imaging", "deep learning"],
    )

    text = str(out.get("highlight_snippet") or out.get("snippet") or "")
    assert text.startswith("Single-pixel imaging technology can capture")
    assert "Kai Song" not in text
    assert "Yaoxing" not in text
    assert "\\" not in text
    assert "Advances and Challenges" not in text
