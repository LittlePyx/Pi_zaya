from __future__ import annotations

from pathlib import Path

from api.reference_source_display import _display_source_name, _hit_matches_guide_source
from kb.file_naming import KB_DISPLAY_FULL_NAME_KEY


def test_hit_matches_guide_source_by_path_and_title():
    meta = {
        "source_path": r"F:\kb\db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
        "source_name": "CVPR-2024-SCINeRF.pdf",
        "display_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
    }

    assert _hit_matches_guide_source(
        meta,
        guide_source_path=r"F:\kb\db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.pdf",
        guide_source_name="",
    ) is True
    assert _hit_matches_guide_source(
        meta,
        guide_source_path="",
        guide_source_name="SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
    ) is True
    assert _hit_matches_guide_source(
        meta,
        guide_source_path="",
        guide_source_name="A Completely Different Paper",
    ) is False


def test_hit_matches_guide_source_does_not_collapse_same_markdown_name_in_other_directory():
    meta = {
        "source_path": r"F:\kb\collection-a\Paper.en.md",
        "source_name": "Paper.pdf",
    }

    assert _hit_matches_guide_source(
        meta,
        guide_source_path=r"F:\kb\collection-b\Paper.en.md",
        guide_source_name="",
    ) is False


def test_hit_matches_guide_source_does_not_collapse_cross_format_namesake():
    meta = {
        "source_path": r"F:\kb\collection-a\Paper.en.md",
        "source_name": "Paper.pdf",
    }

    assert _hit_matches_guide_source(
        meta,
        guide_source_path=r"F:\kb\collection-b\Paper.pdf",
        guide_source_name="Paper.pdf",
    ) is False


def test_display_source_name_prefers_library_citation_meta(tmp_path: Path):
    pdf_path = tmp_path / "stored.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Store:
        def get_citation_meta(self, path: Path) -> dict:
            assert path == pdf_path
            return {KB_DISPLAY_FULL_NAME_KEY: "CVPR-2024-SCINeRF- Neural Radiance Fields.pdf"}

    assert _display_source_name("fallback.en.md", pdf_path, _Store()) == "CVPR-2024-SCINeRF- Neural Radiance Fields.pdf"


def test_display_source_name_falls_back_to_source_filename():
    assert _display_source_name(r"F:\kb\db\Paper\Paper.en.md", None, None) == "Paper.pdf"
    assert _display_source_name(r"F:\kb\db\Paper\notes.md", None, None) == "notes.pdf"
    assert _display_source_name("", None, None) == "unknown.pdf"


def test_display_source_name_reports_debug_on_store_error(tmp_path: Path):
    messages: list[str] = []

    class _Store:
        def get_citation_meta(self, _path: Path) -> dict:
            raise RuntimeError("boom")

    out = _display_source_name("broken.en.md", tmp_path / "broken.pdf", _Store(), debug_log=messages.append)

    assert out == "broken.pdf"
    assert messages and "boom" in messages[0]
