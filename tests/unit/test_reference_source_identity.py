from api.reference_source_identity import (
    _normalize_title_identity,
    _same_source_identity,
    _same_source_title_identity,
    _source_filename,
    _source_identity_keys,
    _title_identity_keys,
)


def test_source_filename_accepts_windows_and_posix_separators():
    assert _source_filename(r"C:\papers\2025 - Neural Rendering.en.md") == "2025 - Neural Rendering.en.md"
    assert _source_filename("/papers/2025 - Neural Rendering.pdf") == "2025 - Neural Rendering.pdf"


def test_source_identity_keys_bridge_markdown_and_pdf_names():
    keys = _source_identity_keys("2025 - Neural Rendering.en.md")

    assert "2025 - neural rendering.en.md" in keys
    assert "2025 - neural rendering.pdf" in keys
    assert "2025 - neural rendering" in keys
    assert _same_source_identity("2025 - Neural Rendering.en.md", "2025 - Neural Rendering.pdf") is True


def test_title_identity_normalizes_extensions_and_punctuation():
    assert _normalize_title_identity("2025_Neural-Rendering.en.md") == "2025 neural rendering"
    assert _normalize_title_identity("Neural Rendering.pdf") == "neural rendering"


def test_title_identity_keys_include_filename_parsed_title():
    keys = _title_identity_keys("CVPR - 2025 - Neural Rendering with Sparse Views.pdf")

    assert "cvpr 2025 neural rendering with sparse views" in keys
    assert "neural rendering with sparse views" in keys
    assert _same_source_title_identity(
        "CVPR - 2025 - Neural Rendering with Sparse Views.pdf",
        "Neural Rendering with Sparse Views",
    ) is True
