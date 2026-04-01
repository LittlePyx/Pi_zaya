import json
from pathlib import Path

from kb.converter.page_figure_metadata import reconcile_figure_metadata_from_markdown


def test_reconcile_figure_metadata_from_markdown_updates_sidecars_and_doc_index(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "page_5_fig_1.png").write_bytes(b"x" * 300)

    payload = {
        "page": 5,
        "figures": [
            {
                "page": 5,
                "index": 1,
                "asset_name": "page_5_fig_1.png",
                "fig_no": 3,
                "fig_ident": "3",
                "caption": "Figure 3. Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and rely lens, and a DMD to modulate input frames.",
            }
        ],
    }
    (assets_dir / "page_5_fig_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = "\n".join(
        [
            "![Figure 3](./assets/page_5_fig_1.png)",
            "",
            "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.",
        ]
    )

    updated = reconcile_figure_metadata_from_markdown(md=md, assets_dir=assets_dir)

    assert "page_5_fig_1.png" in updated
    assert "relay lens" in str(updated["page_5_fig_1.png"].get("caption") or "")
    assert "**" not in str(updated["page_5_fig_1.png"].get("caption") or "")

    sidecar = json.loads((assets_dir / "page_5_fig_1.meta.json").read_text(encoding="utf-8"))
    assert "relay lens" in str(sidecar.get("caption") or "")

    doc_index = json.loads((assets_dir / "figure_index.json").read_text(encoding="utf-8"))
    figures = list(doc_index.get("figures") or [])
    assert len(figures) == 1
    assert "relay lens" in str(figures[0].get("caption") or "")


def test_reconcile_figure_metadata_from_markdown_matches_alias_asset_names(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "page_8_fig_1.png").write_bytes(b"x" * 300)

    payload = {
        "page": 8,
        "figures": [
            {
                "page": 8,
                "index": 1,
                "asset_name": "page_8_fig_1.png",
                "fig_no": 5,
                "fig_ident": "5",
                "caption": "Figure 5. Qualitative evaluations on the real dataset capptured by our system in Fig. 3.",
            }
        ],
    }
    (assets_dir / "page_8_fig_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = "\n".join(
        [
            "![Figure 5](./assets/fig_5.png)",
            "",
            "**Figure 5.** Qualitative evaluations on the real dataset captured by our system in Fig. 3.",
        ]
    )

    updated = reconcile_figure_metadata_from_markdown(md=md, assets_dir=assets_dir)

    assert "page_8_fig_1.png" in updated
    assert "captured by our system" in str(updated["page_8_fig_1.png"].get("caption") or "")
