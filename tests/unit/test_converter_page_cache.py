from __future__ import annotations

import json
from pathlib import Path

from kb.converter.config import ConvertConfig, LlmConfig
from kb.converter.page_cache import PageConversionCache, page_markdown_is_reusable


def _config(tmp_path: Path, *, detect_tables: bool = True, model: str = "qwen-vl-test") -> ConvertConfig:
    return ConvertConfig(
        pdf_path=tmp_path / "paper.pdf",
        out_dir=tmp_path / "out",
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=LlmConfig(
            api_key="secret-must-not-be-persisted",
            base_url="https://private.example.test/v1",
            model=model,
        ),
        detect_tables=detect_tables,
        speed_mode="normal",
    )


def _source(tmp_path: Path, content: bytes = b"stable-pdf-content") -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)
    return path


def test_page_cache_restores_markdown_and_page_assets_without_indexable_fragments(tmp_path: Path) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    asset = assets_dir / "page_1_fig_1.png"
    asset.write_bytes(b"verified-image-bytes")
    markdown = "# Method\n\n![Figure 1](./assets/page_1_fig_1.png)\n\nStable page text."

    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=2)
    assert first.store_page(0, markdown, assets_dir=assets_dir) is True
    asset.unlink()

    second = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=2)
    restored = second.load_page(0, assets_dir=assets_dir)
    second.finish()

    assert restored == markdown
    assert asset.read_bytes() == b"verified-image-bytes"
    assert not list((save_dir / ".conversion_cache").rglob("*.md"))
    manifest_text = (save_dir / ".conversion_cache" / "manifest.json").read_text(encoding="utf-8")
    assert "secret-must-not-be-persisted" not in manifest_text
    assert "private.example.test" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["hits"] == [1]


def test_page_cache_invalidates_when_conversion_config_or_model_changes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    original = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=_config(tmp_path), total_pages=1)
    assert original.store_page(0, "Converted page text.", assets_dir=assets_dir) is True

    changed_tables = PageConversionCache(
        save_dir=save_dir,
        pdf_path=source,
        cfg=_config(tmp_path, detect_tables=False),
        total_pages=1,
    )
    changed_model = PageConversionCache(
        save_dir=save_dir,
        pdf_path=source,
        cfg=_config(tmp_path, model="qwen-vl-new"),
        total_pages=1,
    )

    assert changed_tables.load_page(0, assets_dir=assets_dir) is None
    assert changed_model.load_page(0, assets_dir=assets_dir) is None


def test_page_cache_invalidates_when_pdf_content_changes(tmp_path: Path) -> None:
    source = _source(tmp_path, b"pdf-version-one")
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Version one page.", assets_dir=assets_dir) is True

    source.write_bytes(b"pdf-version-two")
    changed = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert changed.load_page(0, assets_dir=assets_dir) is None


def test_page_cache_keeps_completed_pages_and_retries_incomplete_pages(tmp_path: Path) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    interrupted = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=2)

    assert interrupted.store_page(0, "Completed first page.", assets_dir=assets_dir) is True
    assert interrupted.store_page(1, "[Page 2 conversion incomplete]", assets_dir=assets_dir) is False

    resumed = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=2)
    assert resumed.load_page(0, assets_dir=assets_dir) == "Completed first page."
    assert resumed.load_page(1, assets_dir=assets_dir) is None


def test_page_cache_refresh_forces_reconversion(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    monkeypatch.setenv("KB_PDF_PAGE_CACHE_REFRESH", "1")
    refreshed = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert refreshed.load_page(0, assets_dir=assets_dir) is None


def test_page_markdown_reuse_gate_rejects_explicit_failures() -> None:
    assert page_markdown_is_reusable("# Valid page\n\nEvidence.") is True
    assert page_markdown_is_reusable("<!-- kb_page: 1 -->") is False
    assert page_markdown_is_reusable("[Page 1 conversion incomplete]") is False
