from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb.converter import figure_assets
from kb.converter.figure_assets import figure_asset_needs_refresh, scan_figure_asset_quality


Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def _write_png(path: Path, *, size: tuple[int, int] = (320, 240), variant: int = 1) -> None:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle([max(2, w // 8), max(2, h // 8), max(4, w - w // 8), max(4, h - h // 8)], outline="black", width=3)
    draw.line([0, h // 2, w, h // 2], fill="black", width=2)
    draw.text((max(4, w // 5), max(4, h // 3)), f"fig {variant}", fill="black")
    img.save(path)


def _write_top_clipped_png(path: Path, *, size: tuple[int, int] = (320, 240)) -> None:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle([0, 0, w, 5], fill="black")
    draw.rectangle([w // 8, h // 5, w - w // 8, h - h // 8], outline="black", width=3)
    img.save(path)


def _write_markdown(md_path: Path, image_names: list[str]) -> None:
    md_path.write_text(
        "\n".join([f"![Figure](./assets/{name})" for name in image_names]),
        encoding="utf-8",
    )


def _write_figure_index(assets_dir: Path, figures: list[dict]) -> None:
    (assets_dir / "figure_index.json").write_text(
        json.dumps({"figures": figures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_scan_figure_asset_quality_flags_low_resolution_and_missing_assets(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    md_path = tmp_path / "paper.en.md"
    _write_markdown(md_path, ["page_1_fig_1.png", "page_1_fig_2.png"])
    _write_png(assets / "page_1_fig_1.png", size=(160, 160))
    _write_figure_index(
        assets,
        [
            {
                "page": 1,
                "index": 1,
                "asset_name": "page_1_fig_1.png",
                "crop_bbox": [0, 0, 72, 72],
                "bbox": [0, 0, 72, 72],
            },
            {
                "page": 1,
                "index": 2,
                "asset_name": "page_1_fig_2.png",
                "crop_bbox": [0, 0, 72, 72],
                "bbox": [0, 0, 72, 72],
            },
        ],
    )

    report = scan_figure_asset_quality(md_path, target_dpi=320)

    assert report["status"] == "error"
    assert report["refresh_recommended"] is True
    assert report["issue_counts"]["low_resolution"] == 1
    assert report["issue_counts"]["missing_asset"] == 1
    low = next(issue for issue in report["issues"] if issue["code"] == "low_resolution")
    assert low["actual_width"] == 160
    assert low["expected_width"] == 320


def test_scan_figure_asset_quality_flags_duplicate_and_suspicious_crop(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    md_path = tmp_path / "paper.en.md"
    image_names = ["page_1_fig_1.png", "page_1_fig_2.png", "page_1_fig_3.png"]
    _write_markdown(md_path, image_names)
    _write_png(assets / "page_1_fig_1.png", size=(320, 320), variant=1)
    (assets / "page_1_fig_2.png").write_bytes((assets / "page_1_fig_1.png").read_bytes())
    _write_png(assets / "page_1_fig_3.png", size=(180, 120), variant=3)
    _write_figure_index(
        assets,
        [
            {
                "page": 1,
                "index": 1,
                "asset_name": "page_1_fig_1.png",
                "crop_bbox": [0, 0, 72, 72],
                "bbox": [0, 0, 72, 72],
            },
            {
                "page": 1,
                "index": 2,
                "asset_name": "page_1_fig_2.png",
                "crop_bbox": [80, 0, 152, 72],
                "bbox": [80, 0, 152, 72],
            },
            {
                "page": 1,
                "index": 3,
                "asset_name": "page_1_fig_3.png",
                "crop_bbox": [0, 0, 40, 30],
                "bbox": [0, 0, 200, 100],
            },
        ],
    )

    report = scan_figure_asset_quality(md_path, target_dpi=320)

    assert report["status"] == "warning"
    assert report["issue_counts"]["duplicate_asset"] == 2
    assert report["issue_counts"]["suspicious_crop"] == 1
    suspicious = next(issue for issue in report["issues"] if issue["code"] == "suspicious_crop")
    assert suspicious["asset_name"] == "page_1_fig_3.png"
    assert "smaller than" in suspicious["message"]


def test_scan_figure_asset_quality_flags_tight_top_edge_crop(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    md_path = tmp_path / "paper.en.md"
    _write_markdown(md_path, ["page_1_fig_1.png"])
    _write_top_clipped_png(assets / "page_1_fig_1.png", size=(320, 240))
    _write_figure_index(
        assets,
        [
            {
                "page": 1,
                "index": 1,
                "asset_name": "page_1_fig_1.png",
                "crop_bbox": [0, 4, 72, 58],
                "bbox": [0, 8, 72, 58],
            }
        ],
    )

    report = scan_figure_asset_quality(md_path, target_dpi=320)

    assert report["status"] == "warning"
    assert report["refresh_recommended"] is True
    assert report["issue_counts"]["top_edge_clipped"] == 1
    issue = next(issue for issue in report["issues"] if issue["code"] == "top_edge_clipped")
    assert issue["asset_name"] == "page_1_fig_1.png"
    assert issue["top_padding_pt"] == 4.0


@pytest.mark.skipif(figure_assets.fitz is None, reason="PyMuPDF not available")
def test_figure_asset_needs_refresh_catches_small_crop_expansion(tmp_path: Path):
    path = tmp_path / "page_1_fig_1.png"
    _write_png(path, size=(436, 436))

    assert figure_asset_needs_refresh(
        path,
        clip_rect=figure_assets.fitz.Rect(0, 0, 100.2, 100.2),
        dpi=320,
    )
