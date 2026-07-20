from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kb.converter.config import ConvertConfig, LlmConfig
from kb.converter import page_cache as page_cache_module
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


def _config_with_extra(tmp_path: Path, **updates):
    values = dict(vars(_config(tmp_path)))
    values.update(updates)
    return SimpleNamespace(**values)


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
    assert manifest["schema_version"] == 2
    assert manifest["page_output_algorithm_version"] == page_cache_module.PAGE_OUTPUT_ALGORITHM_VERSION
    assert manifest["page_output_fingerprint"]
    assert manifest["config_fingerprint"]
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


def test_page_cache_reuses_equivalent_speed_mode_aliases(tmp_path: Path) -> None:
    source = _source(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    balanced = replace(_config(tmp_path), speed_mode="balanced")
    normal = replace(_config(tmp_path), speed_mode="normal")
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=balanced, total_pages=1)
    assert first.store_page(0, "Converted page text.", assets_dir=assets_dir) is True

    second = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=normal, total_pages=1)

    assert second.load_page(0, assets_dir=assets_dir) == "Converted page text."


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


def test_page_cache_invalidates_when_vision_page_budget_changes(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    monkeypatch.setenv("KB_PDF_VISION_PAGE_BUDGET_S", "120")
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    monkeypatch.setenv("KB_PDF_VISION_PAGE_BUDGET_S", "90")
    changed = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert changed.load_page(0, assets_dir=assets_dir) is None


@pytest.mark.parametrize(
    ("field", "original_value", "changed_value"),
    [
        ("dpi", 200, 240),
        ("llm_render_max_tokens", 0, 2048),
    ],
)
def test_page_cache_invalidates_when_page_output_config_changes(
    tmp_path: Path,
    field: str,
    original_value: int,
    changed_value: int,
) -> None:
    source = _source(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(
        save_dir=save_dir,
        pdf_path=source,
        cfg=_config_with_extra(tmp_path, **{field: original_value}),
        total_pages=1,
    )
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    changed = PageConversionCache(
        save_dir=save_dir,
        pdf_path=source,
        cfg=_config_with_extra(tmp_path, **{field: changed_value}),
        total_pages=1,
    )

    assert changed.config_fingerprint != first.config_fingerprint
    assert changed.page_output_fingerprint == first.page_output_fingerprint
    assert changed.load_page(0, assets_dir=assets_dir) is None


def test_page_cache_reuses_pages_when_only_classify_batch_size_changes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(
        save_dir=save_dir,
        pdf_path=source,
        cfg=_config_with_extra(tmp_path, classify_batch_size=40),
        total_pages=1,
    )
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    changed = PageConversionCache(
        save_dir=save_dir,
        pdf_path=source,
        cfg=_config_with_extra(tmp_path, classify_batch_size=80),
        total_pages=1,
    )

    assert changed.config_fingerprint == first.config_fingerprint
    assert changed.load_page(0, assets_dir=assets_dir) == "Completed page."


def test_page_cache_accepts_and_migrates_matching_legacy_config_fingerprint(tmp_path: Path) -> None:
    source = _source(tmp_path)
    cfg = _config_with_extra(tmp_path, classify_batch_size=80)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True
    entry_path = save_dir / ".conversion_cache" / "pages" / "00001" / "entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    legacy_fingerprint = page_cache_module._stable_json_hash(
        {"config": page_cache_module._legacy_config_payload(cfg)}
    )
    assert legacy_fingerprint != first.config_fingerprint
    entry["config_fingerprint"] = legacy_fingerprint
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    current = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert current.load_page(0, assets_dir=assets_dir) == "Completed page."
    migrated = json.loads(entry_path.read_text(encoding="utf-8"))
    assert migrated["config_fingerprint"] == current.config_fingerprint


def test_page_cache_accepts_legacy_fingerprint_when_batch_size_changed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    legacy_cfg = _config_with_extra(tmp_path, classify_batch_size=40)
    current_cfg = _config_with_extra(tmp_path, classify_batch_size=80)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=legacy_cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True
    entry_path = save_dir / ".conversion_cache" / "pages" / "00001" / "entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["config_fingerprint"] = page_cache_module._stable_json_hash(
        {
            "config": page_cache_module._legacy_config_payload(
                legacy_cfg,
                classify_batch_size=40,
            )
        }
    )
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    current = PageConversionCache(
        save_dir=save_dir,
        pdf_path=source,
        cfg=current_cfg,
        total_pages=1,
    )

    assert current.load_page(0, assets_dir=assets_dir) == "Completed page."
    migrated = json.loads(entry_path.read_text(encoding="utf-8"))
    assert migrated["config_fingerprint"] == current.config_fingerprint


@pytest.mark.parametrize(
    "env_name",
    [
        "KB_LLM_HARD_TIMEOUT_S",
        "KB_PDF_FIGURE_DPI",
        "KB_PDF_VISION_REFS_LOCAL_MIN_ENTRIES",
    ],
)
def test_page_cache_invalidates_when_page_output_environment_changes(
    tmp_path: Path,
    monkeypatch,
    env_name: str,
) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    monkeypatch.setenv(env_name, "11")
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    monkeypatch.setenv(env_name, "17")
    changed = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert changed.config_fingerprint != first.config_fingerprint
    assert changed.page_output_fingerprint == first.page_output_fingerprint
    assert changed.load_page(0, assets_dir=assets_dir) is None


def test_page_cache_invalidates_when_page_output_algorithm_version_changes(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    monkeypatch.setattr(
        page_cache_module,
        "PAGE_OUTPUT_ALGORITHM_VERSION",
        page_cache_module.PAGE_OUTPUT_ALGORITHM_VERSION + 1,
    )
    changed = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert changed.config_fingerprint == first.config_fingerprint
    assert changed.page_output_fingerprint != first.page_output_fingerprint
    assert changed.load_page(0, assets_dir=assets_dir) is None


def test_page_cache_invalidates_when_direct_page_output_dependency_changes(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    original_sha256_file = page_cache_module._sha256_file

    def changed_dependency_hash(path: Path) -> str:
        if Path(path).name == "llm_worker.py":
            return "f" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(page_cache_module, "_sha256_file", changed_dependency_hash)
    changed = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert changed.config_fingerprint == first.config_fingerprint
    assert changed.page_output_fingerprint != first.page_output_fingerprint
    assert changed.load_page(0, assets_dir=assets_dir) is None


def test_page_cache_ignores_document_finalization_module_changes(tmp_path: Path, monkeypatch) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True

    original_sha256_file = page_cache_module._sha256_file
    hashed_components: list[str] = []

    def finalization_only_change(path: Path) -> str:
        name = Path(path).name
        hashed_components.append(name)
        if name == "post_processing.py":
            return "e" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(page_cache_module, "_sha256_file", finalization_only_change)
    unchanged = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert unchanged.load_page(0, assets_dir=assets_dir) == "Completed page."
    assert "post_processing.py" not in page_cache_module._PAGE_OUTPUT_COMPONENTS
    assert "pipeline.py" not in page_cache_module._PAGE_OUTPUT_COMPONENTS
    assert "post_processing.py" not in hashed_components


def test_page_output_dependency_list_covers_both_conversion_paths() -> None:
    expected = {
        "llm_worker.py",
        "page_layout_crops.py",
        "page_local_pipeline.py",
        "page_vision_direct_page.py",
        "page_vision_guardrails.py",
        "pipeline_render_markdown.py",
        "pipeline_vision_direct.py",
        "page_text_blocks.py",
        "page_image_markdown.py",
        "page_figure_metadata.py",
        "page_table_fallback.py",
        "figure_assets.py",
        "layout_analysis.py",
        "post_heading_rules.py",
        "post_math_rules.py",
        "post_references.py",
        "tables.py",
        "text_utils.py",
    }

    assert expected <= set(page_cache_module._PAGE_OUTPUT_COMPONENTS)


def test_page_cache_rejects_previous_schema_entries(tmp_path: Path) -> None:
    source = _source(tmp_path)
    cfg = _config(tmp_path)
    save_dir = tmp_path / "out"
    assets_dir = save_dir / "assets"
    assets_dir.mkdir(parents=True)
    first = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)
    assert first.store_page(0, "Completed page.", assets_dir=assets_dir) is True
    entry_path = save_dir / ".conversion_cache" / "pages" / "00001" / "entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["schema_version"] = 1
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    current = PageConversionCache(save_dir=save_dir, pdf_path=source, cfg=cfg, total_pages=1)

    assert current.load_page(0, assets_dir=assets_dir) is None


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
