from __future__ import annotations

from pathlib import Path


def test_reference_rendering_adapter_preserves_filename_meta():
    from api import reference_rendering
    from ui import refs_renderer

    sample = r"F:\kb\pdfs\CVPR-2024-SCINeRF- Neural Radiance Fields.pdf"

    assert reference_rendering._parse_filename_meta(sample) == refs_renderer._parse_filename_meta(sample)


def test_api_reference_modules_use_backend_adapter():
    import api.chat_render as chat_render
    import api.reference_rendering as reference_rendering
    import api.reference_source_identity as source_identity

    assert chat_render._source_cite_id is reference_rendering._source_cite_id
    assert source_identity._parse_filename_meta is reference_rendering._parse_filename_meta


def test_production_api_modules_do_not_import_refs_renderer_directly():
    root = Path(__file__).resolve().parents[2]
    allowed = {root / "api" / "reference_rendering.py"}
    offenders: list[str] = []
    for path in (root / "api").rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "ui.refs_renderer" in text:
            offenders.append(str(path.relative_to(root)).replace("\\", "/"))

    assert offenders == []
