from __future__ import annotations

from api.chat_render import _primary_evidence_from_ref_hit


def test_primary_evidence_selection_uses_card_claim() -> None:
    hit = {
        "meta": {"source_path": "db/3d-single-pixel-video.md"},
        "ui_meta": {
            "summary_line": "The dynamic-link library API simplifies system control.",
            "primary_evidence": {
                "heading_path": "Results",
                "snippet": "The corresponding frame rates are 8.7 Hz, 2.4 Hz, and 0.5 Hz.",
            },
            "reader_open": {
                "alternatives": [
                    {
                        "heading_path": "Custom single-pixel system design",
                        "snippet": "The application programming interface is written as a dynamic-link library file.",
                    }
                ]
            },
        },
    }

    primary = _primary_evidence_from_ref_hit(hit)

    assert "dynamic-link library" in str(primary.get("snippet") or "")
