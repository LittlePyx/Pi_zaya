from __future__ import annotations

import api.reference_ui as reference_ui


def _ready_hit(source_path: str, text: str, *, llm: float = 84.0, bm25: float = 5.0) -> dict:
    return {
        "text": text,
        "meta": {
            "source_path": source_path,
            "ref_pack_state": "ready",
            "ref_rank": {
                "llm": llm,
                "bm25": bm25,
                "deep": 1.4,
                "term_bonus": 0.4,
                "semantic_score": 7.8,
            },
        },
    }


def _install_deterministic_cards(monkeypatch, cards_by_source_token: dict[str, dict]) -> None:
    def fake_build_hit_ui_meta(hit, **kwargs):
        del kwargs
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or "").strip()
        for token, ui in cards_by_source_token.items():
            if token in source_path:
                out = dict(ui)
                out.setdefault("source_path", source_path)
                out.setdefault("reader_open", {"sourcePath": source_path})
                return out
        return {
            "display_name": source_path,
            "heading_path": "Abstract",
            "summary_line": str(hit.get("text") or ""),
            "why_line": "candidate",
            "score": 7.2,
            "reader_open": {"sourcePath": source_path},
            "source_path": source_path,
        }

    monkeypatch.setattr(reference_ui, "build_hit_ui_meta", fake_build_hit_ui_meta)
    monkeypatch.setattr(reference_ui, "_prefetch_refs_citation_meta", lambda *args, **kwargs: {})
    monkeypatch.setattr(reference_ui, "_maybe_llm_rerank_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_llm_filter_refs_hits", lambda **kwargs: list(kwargs.get("hits") or []))
    monkeypatch.setattr(reference_ui, "_maybe_polish_refs_card_copy", lambda **kwargs: list(kwargs.get("hits") or []))


def test_golden_scinerf_figure3_prompt_binds_to_named_source_and_evidence(monkeypatch):
    prompt = (
        "SCINeRF\u7684\u771f\u5b9e\u786c\u4ef6\u5b9e\u9a8c\u88c5\u7f6e"
        "\u5305\u542b\u54ea\u4e9b\u90e8\u4ef6\uff1f\u8bf7\u5bf9\u5e94\u5230"
        "\u539f\u6587\u56fe3\u6216\u5b9e\u9a8c\u8bbe\u7f6e\u3002"
    )
    scinerf_path = r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md"
    nat_path = r"db\NatPhoton-2019-Principles\NatPhoton-2019.en.md"
    refs = {
        9001: {
            "prompt": prompt,
            "hits": [
                _ready_hit(nat_path, "Single-pixel imaging review mentions acquisition strategies.", llm=85.0),
                _ready_hit(
                    scinerf_path,
                    "Figure 3. Experimental setup for real dataset collection with CCD camera, relay lens, and DMD.",
                    llm=83.0,
                ),
            ],
        }
    }
    _install_deterministic_cards(
        monkeypatch,
        {
            "SCINeRF": {
                "display_name": "CVPR-2024-SCINeRF.pdf",
                "heading_path": "4. Experiments / 4.1. Experimental Setup / Figure 3",
                "summary_line": "Figure 3. Experimental setup for real dataset collection contains a CCD camera, primary and relay lens, and a DMD.",
                "why_line": "The card is grounded in the Figure 3 experimental setup evidence.",
                "score": 8.3,
            },
            "NatPhoton": {
                "display_name": "NatPhoton-2019.pdf",
                "heading_path": "Acquisition strategies",
                "summary_line": "This review discusses single-pixel imaging acquisition strategies.",
                "why_line": "candidate",
                "score": 8.7,
            },
        },
    )

    out = reference_ui.enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        allow_expensive_llm_for_ready=False,
        render_variant="bounded_full",
    )

    pack = out.get(9001) or {}
    hits = list(pack.get("hits") or [])
    assert len(hits) == 1
    ui = dict((hits[0].get("ui_meta") if isinstance(hits[0].get("ui_meta"), dict) else {}) or {})
    assert "SCINeRF" in str(ui.get("display_name") or "")
    assert "Figure 3" in str(ui.get("heading_path") or "")
    assert "CCD camera" in str(ui.get("summary_line") or "")
    assert "DMD" in str(ui.get("summary_line") or "")
    pipeline = dict(pack.get("pipeline_debug") or {})
    assert int(pipeline.get("raw_hit_count") or 0) == 2
    assert int(pipeline.get("final_hit_count") or 0) == 1


def test_golden_structured_detection_compare_keeps_two_evidence_matched_papers(monkeypatch):
    prompt = (
        "\u54ea\u4e9b\u6587\u732e\u6bd4\u8f83\u4e86 structured detection "
        "\u548c confocal/open pinhole \u7684\u6743\u8861\uff1f"
    )
    nat_path = r"db\NatPhoton-2025-Structured detection\NatPhoton-2025.en.md"
    lsa_path = r"db\LSA-2026-Interferometric Image Scanning\LSA-2026.en.md"
    weak_path = r"db\Generic-Laser-Scanning\Generic.en.md"
    refs = {
        9002: {
            "prompt": prompt,
            "hits": [
                _ready_hit(
                    nat_path,
                    "Structured detection compares open pinhole and closed pinhole trade-offs in laser scanning microscopy.",
                    llm=86.0,
                ),
                _ready_hit(
                    lsa_path,
                    "Interferometric image scanning discusses structured detection, confocal detection, and optical sectioning trade-offs.",
                    llm=85.5,
                ),
                _ready_hit(
                    weak_path,
                    "Laser scanning microscopy uses a detector and discusses resolution.",
                    llm=85.0,
                ),
            ],
        }
    }
    _install_deterministic_cards(
        monkeypatch,
        {
            "NatPhoton-2025": {
                "display_name": "NatPhoton-2025-Structured detection.pdf",
                "heading_path": "Structured detection / Open and closed pinhole views",
                "summary_line": "Structured detection compares open pinhole and closed pinhole measurements and explains the sectioning versus signal trade-off.",
                "why_line": "This evidence directly compares structured detection with pinhole-based detection choices.",
                "score": 8.8,
            },
            "LSA-2026": {
                "display_name": "LSA-2026-Interferometric Image Scanning.pdf",
                "heading_path": "Principle / Confocal comparison",
                "summary_line": "The paper discusses structured detection alongside confocal detection and optical sectioning trade-offs.",
                "why_line": "This evidence links structured detection to confocal-style sectioning trade-offs.",
                "score": 8.7,
            },
            "Generic-Laser-Scanning": {
                "display_name": "Generic laser scanning microscopy.pdf",
                "heading_path": "Background",
                "summary_line": "Laser scanning microscopy uses detectors to collect optical signals.",
                "why_line": "candidate",
                "score": 8.6,
            },
        },
    )

    out = reference_ui.enrich_refs_payload(
        refs,
        pdf_root=None,
        md_root=None,
        lib_store=None,
        allow_expensive_llm_for_ready=False,
        render_variant="bounded_full",
    )

    pack = out.get(9002) or {}
    titles = [
        str(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("display_name") or "")
        for hit in list(pack.get("hits") or [])
    ]
    assert titles == [
        "NatPhoton-2025-Structured detection.pdf",
        "LSA-2026-Interferometric Image Scanning.pdf",
    ]
    summaries = " ".join(
        str(((hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}) or {}).get("summary_line") or "")
        for hit in list(pack.get("hits") or [])
    )
    assert "structured detection" in summaries.lower()
    assert "trade-off" in summaries.lower()
    pipeline = dict(pack.get("pipeline_debug") or {})
    assert int(pipeline.get("raw_hit_count") or 0) == 3
    assert int(pipeline.get("final_hit_count") or 0) == 2
