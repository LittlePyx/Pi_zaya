
import json
import pytest
import fitz
import os
import re
from pathlib import Path
from types import SimpleNamespace
from kb.converter.pipeline import PDFConverter
from kb.converter.config import ConvertConfig
from kb.converter.post_processing import postprocess_markdown
from kb.converter.page_figure_metadata import persist_page_figure_metadata
from kb.converter.page_image_markdown import repair_broken_image_links
from kb.converter.models import TextBlock
from kb.source_blocks import build_source_blocks

@pytest.fixture
def sample_pdf(tmp_path):
    """Generates a simple PDF for testing."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    
    # Insert Title
    page.insert_text((50, 50), "Test Title", fontsize=20, fontname="Helvetica-Bold")
    
    # Insert Paragraph
    page.insert_text((50, 100), "This is a test paragraph with some content.", fontsize=12)
    
    # Insert Table-like structure (text)
    page.insert_text((50, 150), "Column1    Column2", fontsize=10)
    page.insert_text((50, 165), "Value1     Value2", fontsize=10)
    
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d

def test_convert_pipeline_fast_mode(sample_pdf, output_dir):
    """Test basic conversion pipeline in fast mode."""
    cfg = ConvertConfig(
        pdf_path=sample_pdf,
        out_dir=output_dir,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None
    )
    converter = PDFConverter(cfg)
    
    converter.convert(str(sample_pdf), str(output_dir))
    
    out_file = output_dir / "output.md"
    assert out_file.exists()
    
    content = out_file.read_text(encoding="utf-8")
    
    # Check for content presence
    assert "Test Title" in content
    assert "This is a test paragraph with some content." in content
    assert content.lstrip().startswith("<!-- kb_page: 1 -->")
    
    # Check simple structure based on heuristics (fontsize)
    # The pipeline might not perfectly identify H1 without LLM, but let's see.
    # Note: Default heuristic might treat larger font as Heading.
    # We assert presence first. Structure assertion depends on specific heuristic tuning.
    
    # Check that assets dir was created
    assert (output_dir / "assets").exists()
    assert (output_dir / "assets" / "anchor_index.json").exists()
    assert (output_dir / "assets" / "equation_index.json").exists()
    assert (output_dir / "assets" / "reference_index.json").exists()


def test_ensure_page_marker_inserts_and_normalizes_marker():
    assert PDFConverter._ensure_page_marker("# Page body", 2).startswith("<!-- kb_page: 3 -->")
    assert PDFConverter._ensure_page_marker("<!-- kb_page: 99 -->\n\n# Page body", 2).startswith("<!-- kb_page: 3 -->")
    assert PDFConverter._ensure_page_marker("", 0) == "<!-- kb_page: 1 -->"


def test_title_fallback_from_filename_preserves_numbered_outline_depth(tmp_path):
    title = "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning"
    cfg = ConvertConfig(
        pdf_path=tmp_path / f"LPR-2025-{title}.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)
    md = """
<!-- kb_page: 1 -->

# Abstract

This review summarizes the field.

## 1. Introduction

Intro text.

## 4. Strategy and Advantages of Single-Pixel Imaging via Deep Learning

## 4.1. Strategy of Single-Pixel Imaging via Deep Learning

Strategy overview text.

### 4.1.1. Data-Driven Strategy

Body text.
"""

    injected = converter._inject_title_from_pdf_metadata(md, SimpleNamespace(metadata={}))
    fixed = postprocess_markdown(injected)

    assert f"# {title}" in fixed
    assert "## Abstract" in fixed
    assert "## 4. Strategy and Advantages of Single-Pixel Imaging via Deep Learning" in fixed
    assert "### 4.1. Strategy of Single-Pixel Imaging via Deep Learning" in fixed
    assert "#### 4.1.1. Data-Driven Strategy" in fixed

    heading_paths = {
        str(block.get("text") or ""): str(block.get("heading_path") or "")
        for block in build_source_blocks(fixed, doc_id="doc")
        if str(block.get("kind") or "") == "heading"
    }
    assert heading_paths["4. Strategy and Advantages of Single-Pixel Imaging via Deep Learning"] == (
        f"{title} / 4. Strategy and Advantages of Single-Pixel Imaging via Deep Learning"
    )
    assert heading_paths["4.1. Strategy of Single-Pixel Imaging via Deep Learning"] == (
        f"{title} / 4. Strategy and Advantages of Single-Pixel Imaging via Deep Learning / "
        "4.1. Strategy of Single-Pixel Imaging via Deep Learning"
    )
    assert heading_paths["4.1.1. Data-Driven Strategy"] == (
        f"{title} / 4. Strategy and Advantages of Single-Pixel Imaging via Deep Learning / "
        "4.1. Strategy of Single-Pixel Imaging via Deep Learning / 4.1.1. Data-Driven Strategy"
    )


def test_existing_title_heading_is_promoted_to_document_root(tmp_path):
    title = "Imaging biological tissue with high-throughput single-pixel compressive holography"
    cfg = ConvertConfig(
        pdf_path=tmp_path / f"NatCommun-2021-{title}.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)
    md = f"""
<!-- kb_page: 1 -->

## {title}

## Abstract

Body text.
"""

    fixed = converter._inject_title_from_pdf_metadata(md, SimpleNamespace(metadata={}))

    assert f"# {title}" in fixed
    assert f"## {title}" not in fixed


def test_repair_broken_image_links_uses_page_marker_and_filename_hint(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "page_1_fig_1.png").write_bytes(b"page1")
    (assets_dir / "page_2_fig_1.png").write_bytes(b"page2")
    md = "\n".join(
        [
            "<!-- kb_page: 1 -->",
            "Text on page one.",
            "",
            "<!-- kb_page: 2 -->",
            "Supplement figure:",
            "![Figure](./assets/image_auto_01_p002_r2.png)",
        ]
    )

    out = repair_broken_image_links(md, save_dir=tmp_path, assets_dir=assets_dir)

    assert "![Figure](./assets/page_2_fig_1.png)" in out
    assert "image_auto_01_p002_r2.png" not in out


def test_recover_references_from_pdf_text_layer_when_markdown_missing_refs(tmp_path, monkeypatch):
    import kb.converter.pipeline as pipeline_module

    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)

    class _RefPage:
        def get_text(self, mode: str):
            assert mode == "text"
            return "\n".join(
                [
                    "References",
                    "[1] Ada Lovelace. Example reference. Journal of Testing, 2024.",
                    "[2] Alan Turing. Another reference. Proceedings of Tests, 2025.",
                ]
            )

    class _Doc:
        def __len__(self):
            return 1

        def load_page(self, page_index: int):
            assert page_index == 0
            return _RefPage()

    monkeypatch.setattr(pipeline_module, "_page_has_references_heading", lambda page: True)
    monkeypatch.setattr(pipeline_module, "_page_looks_like_references_content", lambda page: False)

    fixed, repair = converter._recover_references_from_pdf_if_needed(
        "# Demo Paper\n\n## Abstract\n\nThis paper cites prior work [1].",
        _Doc(),
    )

    assert repair["changed"] is True
    assert repair["applied"] == ["pdf_reference_backfill"]
    assert "# References" in fixed
    assert "[1] Ada Lovelace" in fixed
    assert "[2] Alan Turing" in fixed


def test_recover_references_replaces_truncated_index_from_pdf_text_layer(tmp_path, monkeypatch):
    import kb.converter.pipeline as pipeline_module

    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)

    class _RefPage:
        def get_text(self, mode: str):
            assert mode == "text"
            return "\n".join(
                [
                    "Final body text should not become a reference.",
                    "REFERENCES",
                    "1. ALPHA, A. First recovered reference. Journal, 1950, 1, 1-2.",
                    "2. BETA, B. Second recovered reference. Journal, 1951, 2, 3-4.",
                    "3. GAMMA, C. Third recovered reference. Journal, 1952, 3, 5-6.",
                    "4. DELTA, D. Fourth recovered reference. Journal, 1953, 4, 7-8.",
                    "5. EPSILON, E. Fifth recovered reference. Journal, 1954, 5, 9-10.",
                    "6. ZETA, F. Sixth recovered reference. Journal, 1955, 6, 11-12.",
                    "7. ETA, G. Seventh recovered reference. Journal, 1956, 7, 13-14.",
                    "8. THETA, H. Eighth recovered reference. Journal, 1957, 8, 15-16.",
                ]
            )

    class _Doc:
        def __len__(self):
            return 1

        def load_page(self, page_index: int):
            assert page_index == 0
            return _RefPage()

    monkeypatch.setattr(pipeline_module, "_page_has_references_heading", lambda page: True)
    monkeypatch.setattr(pipeline_module, "_page_looks_like_references_content", lambda page: False)

    broken = "\n".join(
        [
            "# Demo Paper",
            "",
            "## References",
            "",
            "[1] ALPHA, A. First broken reference. Journal, 1950, 1, 1-2.",
            "[2] BETA, B. Second broken reference. Journal, 1951, 2, 3-4.",
            "[3] GAMMA, C. Third broken reference. Journal, 1952, 3, 5-6.",
            "### INFORMATIONAL ASPECTS OF VISUAL PERCEPTION",
            "[4] This fake entry came from a running header and should be replaced.",
            "[5] EPSILON, E. Fifth broken reference. Journal, 1954, 5, 9-10.",
            "[6] ZETA, F. Sixth broken reference. Journal, 1955, 6, 11-12.",
            "[7] ETA, G. Seventh broken reference. Journal, 1956, 7, 13-14.",
            "[8] THETA, H. Eighth broken reference. Journal, 1957, 8, 15-16.",
        ]
    )

    fixed, repair = converter._recover_references_from_pdf_if_needed(broken, _Doc())

    assert repair["changed"] is True
    assert repair["applied"] == ["pdf_reference_backfill"]
    assert "reference_index_truncated" in repair["issue_codes_before"]
    assert "Final body text should not become a reference" not in fixed
    assert "INFORMATIONAL ASPECTS" not in fixed
    assert "[8] THETA" in fixed


def test_recover_references_replaces_gapped_index_from_pdf_text_layer(tmp_path, monkeypatch):
    import kb.converter.pipeline as pipeline_module

    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)

    class _RefPage:
        def get_text(self, mode: str):
            assert mode == "text"
            return "\n".join(
                [
                    "References",
                    *[
                        f"{idx}. REF{idx}, A. Recovered reference {idx}. Journal of Tests {idx}, {idx}-{idx + 1} (20{idx:02d})."
                        for idx in range(1, 11)
                    ],
                ]
            )

    class _Doc:
        def __len__(self):
            return 1

        def load_page(self, page_index: int):
            assert page_index == 0
            return _RefPage()

    monkeypatch.setattr(pipeline_module, "_page_has_references_heading", lambda page: True)
    monkeypatch.setattr(pipeline_module, "_page_looks_like_references_content", lambda page: False)

    broken = "\n".join(
        [
            "# Demo Paper",
            "## Abstract",
            "This paper cites a dense range [1-10].",
            "## References",
            *[
                f"[{idx}] REF{idx}, A. Broken reference {idx}. Journal of Tests {idx}, {idx}-{idx + 1} (20{idx:02d})."
                for idx in [1, 2, 4, 5, 6, 7, 8, 9, 10]
            ],
        ]
    )

    fixed, repair = converter._recover_references_from_pdf_if_needed(broken, _Doc())

    assert repair["changed"] is True
    assert repair["applied"] == ["pdf_reference_backfill"]
    assert "reference_index_truncated" in repair["issue_codes_before"]
    assert "[3] REF3, A. Recovered reference 3." in fixed
    assert "Broken reference" not in fixed


def test_recover_references_replaces_short_truncated_entries_from_pdf_text_layer(tmp_path, monkeypatch):
    import kb.converter.pipeline as pipeline_module

    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)

    class _RefPage:
        def get_text(self, mode: str):
            assert mode == "text"
            return "\n".join(
                [
                    "References",
                    *[
                        f"{idx}. REF{idx}, A. Recovered reference {idx}. Journal of Tests {idx}, {idx}-{idx + 1} (20{idx:02d})."
                        for idx in range(1, 10)
                    ],
                    "10. Wu, D. et al. Final recovered source data record. Zenodo, 2021.",
                ]
            )

    class _Doc:
        def __len__(self):
            return 1

        def load_page(self, page_index: int):
            assert page_index == 0
            return _RefPage()

    monkeypatch.setattr(pipeline_module, "_page_has_references_heading", lambda page: True)
    monkeypatch.setattr(pipeline_module, "_page_looks_like_references_content", lambda page: False)

    broken = "\n".join(
        [
            "# Demo Paper",
            "## References",
            *[
                f"[{idx}] REF{idx}, A. Broken reference {idx}. Journal of Tests {idx}, {idx}-{idx + 1} (20{idx:02d})."
                for idx in range(1, 10)
            ],
            "[10] Wu, D. et al.",
        ]
    )

    fixed, repair = converter._recover_references_from_pdf_if_needed(broken, _Doc())

    assert repair["changed"] is True
    assert "reference_index_truncated" in repair["issue_codes_before"]
    assert "[10] Wu, D. et al. Final recovered source data record. Zenodo, 2021." in fixed
    assert "\n[10] Wu, D. et al.\n" not in f"\n{fixed}\n"


def test_extract_pdf_reference_markdown_keeps_continuation_pages_and_stops_at_body(tmp_path):
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)

    class _Page:
        rect = SimpleNamespace(height=800.0)

        def __init__(self, text: str):
            self._text = text

        def get_text(self, mode: str):
            if mode == "text":
                return self._text
            if mode == "dict":
                return {"blocks": []}
            return ""

    class _Doc:
        def __init__(self):
            self.pages = [
                _Page(
                    "\n".join(
                        [
                            "Title and abstract text.",
                            "References and links",
                            "1. ALPHA, A. First source. Journal of Tests 1, 1-2 (2001).",
                            "2. BETA, B. Second source. Journal of Tests 2, 3-4 (2002).",
                            "3. GAMMA, G. Third source begins. Journal of Tests 3,",
                        ]
                    )
                ),
                _Page(
                    "\n".join(
                        [
                            "5-6 (2003).",
                            "4.",
                            "DELTA, D., EPSILON, E., and ZETA, F. Fourth source. Journal of Tests 4, 7-8 (2004).",
                            "5. ETA, H. Fifth source. Proceedings of Tests 5, 9-10 (2005).",
                            "6. THETA, I. Sixth source. IEEE Tests 6, 11-12 (2006).",
                            "1. Introduction",
                            "This resumed body text must not be emitted as a reference.",
                        ]
                    )
                ),
                _Page("2. Method\nRegular body text with a numbered heading."),
            ]

        def __len__(self):
            return len(self.pages)

        def load_page(self, page_index: int):
            return self.pages[page_index]

    references_md, entry_count = converter._extract_pdf_reference_markdown(_Doc())

    assert entry_count == 6
    assert "[3] GAMMA, G. Third source begins. Journal of Tests 3, 5-6 (2003)." in references_md
    assert "[4] DELTA, D., EPSILON, E., and ZETA, F. Fourth source." in references_md
    assert "This resumed body text must not be emitted" not in references_md


def test_recover_references_replaces_inflated_tail_from_pdf_text_layer(tmp_path):
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)

    class _Page:
        rect = SimpleNamespace(height=800.0)

        def get_text(self, mode: str):
            if mode == "text":
                return "\n".join(
                    [
                        "References and links",
                        *[
                            f"{idx}. REF{idx}, A. Recovered source {idx}. Journal of Tests {idx}, {idx}-{idx + 1} (20{idx:02d})."
                            for idx in range(1, 11)
                        ],
                        "1. Introduction",
                        "Body text after references.",
                    ]
                )
            if mode == "dict":
                return {"blocks": []}
            return ""

    class _Doc:
        def __len__(self):
            return 1

        def load_page(self, page_index: int):
            assert page_index == 0
            return _Page()

    inflated = "\n".join(
        [
            "# Demo Paper",
            "## References",
            *[
                f"[{idx}] REF{idx}, A. Existing source {idx}. Journal of Tests {idx}, {idx}-{idx + 1} (20{idx:02d})."
                for idx in range(1, 11)
            ],
            *[
                f"[{idx}] EXTRA{idx}, X. Inflated source {idx}. Journal of Hallucinated Tests {idx}, {idx}-{idx + 1} (20{idx:02d})."
                for idx in range(11, 19)
            ],
        ]
    )

    fixed, repair = converter._recover_references_from_pdf_if_needed(inflated, _Doc())

    assert repair["changed"] is True
    assert repair["issue_codes_before"] == ["reference_index_inflated"]
    assert "[10] REF10, A. Recovered source 10." in fixed
    assert "Inflated source" not in fixed


def test_convert_pipeline_missing_file(output_dir):
    """Test error handling for missing file."""
    cfg = ConvertConfig(
        pdf_path=Path("non_existent.pdf"),
        out_dir=output_dir,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None
    )
    converter = PDFConverter(cfg)
    
    with pytest.raises(Exception): # fitz.open might raise exception or pipeline checks
        converter.convert("non_existent.pdf", str(output_dir))


def test_collect_non_body_metadata_rects_uses_line_boxes_so_title_is_not_masked(tmp_path):
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)

    class _DummyPage:
        rect = SimpleNamespace(width=612.0, height=792.0)

        def get_text(self, mode: str):
            assert mode == "dict"
            return {
                "blocks": [
                    {
                        "bbox": (35.9, 59.8, 556.1, 148.6),
                        "lines": [
                            {
                                "bbox": (35.9, 59.8, 556.1, 73.5),
                                "spans": [
                                    {
                                        "text": "A P P L I E D O P T I C S 2017 © The Authors, some rights reserved.",
                                        "size": 9.0,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "bbox": (35.9, 77.3, 385.9, 115.2),
                        "lines": [
                            {
                                "bbox": (35.9, 77.3, 385.9, 115.2),
                                "spans": [
                                    {
                                        "text": "Adaptive foveated single-pixel imaging with dynamic supersampling",
                                        "size": 18.0,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "bbox": (35.9, 120.8, 386.6, 135.3),
                        "lines": [
                            {
                                "bbox": (35.9, 120.8, 386.6, 135.3),
                                "spans": [
                                    {
                                        "text": "David B. Phillips, 1 * Ming-Jie Sun, 1,2 * Jonathan M. Taylor, 1 Matthew P. Edgar, 1",
                                        "size": 10.0,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "bbox": (35.9, 132.8, 303.7, 147.2),
                        "lines": [
                            {
                                "bbox": (35.9, 132.8, 303.7, 147.2),
                                "spans": [
                                    {
                                        "text": "Stephen M. Barnett, 1 Graham M. Gibson, 1 Miles J. Padgett 1",
                                        "size": 10.0,
                                    }
                                ],
                            }
                        ],
                    },
                ]
            }

    rects = converter._collect_non_body_metadata_rects(
        _DummyPage(),
        page_index=0,
        is_references_page=False,
    )

    title_rect = fitz.Rect(35.9, 77.3, 385.9, 115.2)
    assert len(rects) >= 1
    assert all(not fitz.Rect(r).intersects(title_rect) for r in rects)
    assert any(fitz.Rect(r).y0 >= 120.0 for r in rects)


def test_cleanup_unreferenced_assets_removes_unused_files_and_rewrites_index(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    used_png = assets_dir / "page_12_fig_1.png"
    used_meta = assets_dir / "page_12_fig_1.meta.json"
    unused_png = assets_dir / "page_12_fig_2.png"
    unused_meta = assets_dir / "page_12_fig_2.meta.json"
    index_path = assets_dir / "page_12_fig_index.json"

    used_png.write_bytes(b"png")
    used_meta.write_text('{"asset_name":"page_12_fig_1.png"}', encoding="utf-8")
    unused_png.write_bytes(b"png")
    unused_meta.write_text('{"asset_name":"page_12_fig_2.png"}', encoding="utf-8")
    index_path.write_text(
        """
{
  "page": 12,
  "figures": [
    {"asset_name": "page_12_fig_1.png"},
    {"asset_name": "page_12_fig_2.png"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None
    )
    converter = PDFConverter(cfg)
    md = "![Figure 5](./assets/page_12_fig_1.png)"
    converter._cleanup_unreferenced_assets(md, assets_dir=assets_dir)

    assert used_png.exists()
    assert used_meta.exists()
    assert not unused_png.exists()
    assert not unused_meta.exists()
    index_text = index_path.read_text(encoding="utf-8")
    assert "page_12_fig_1.png" in index_text
    assert "page_12_fig_2.png" not in index_text


def test_persist_page_figure_metadata_writes_document_index_and_alias(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    raw_asset = assets_dir / "page_7_fig_1.png"
    raw_asset.write_bytes(b"png")

    out = persist_page_figure_metadata(
        assets_dir=assets_dir,
        page_index=6,
        figure_entries=[
            {
                "asset_name": "page_7_fig_1.png",
                "fig_no": 4,
                "fig_ident": "4",
                "caption": "Figure 4. Demo caption.",
                "bbox": [1, 2, 3, 4],
                "crop_bbox": [1, 2, 3, 4],
                "caption_bbox": [5, 6, 7, 8],
            }
        ],
    )

    assert "page_7_fig_1.png" in out
    rec = out["page_7_fig_1.png"]
    assert rec["figure_id"] == "fig_004"
    assert rec["paper_figure_number"] == 4
    assert rec["asset_name_alias"] == "fig_4.png"
    assert (assets_dir / "fig_4.png").exists()

    doc_index = json.loads((assets_dir / "figure_index.json").read_text(encoding="utf-8"))
    assert doc_index["figures"][0]["figure_id"] == "fig_004"
    assert doc_index["figures"][0]["asset_name_alias"] == "fig_4.png"


def test_inject_missing_page_image_links_places_assets_before_matching_caption():
    md = "\n".join(
        [
            "**Figure 8.** Noise-robustness comparison using the 'Lena' image.",
            "",
            "**Figure 9.** Noise-robustness comparison using the 'Cameraman' image.",
        ]
    )
    out = PDFConverter._inject_missing_page_image_links(
        md,
        page_index=14,
        image_names=["page_15_fig_1.png", "page_15_fig_2.png"],
        figure_meta_by_asset={
            "page_15_fig_1.png": {"fig_no": 8},
            "page_15_fig_2.png": {"fig_no": 9},
        },
        is_references_page=False,
    )
    assert out.index("![Figure 8](./assets/page_15_fig_1.png)") < out.index("**Figure 8.**")
    assert out.index("![Figure 9](./assets/page_15_fig_2.png)") < out.index("**Figure 9.**")


def test_inject_page_image_captions_from_meta_when_missing():
    md = "![Figure 6](./assets/page_13_fig_1.png)"
    out = PDFConverter._inject_page_image_captions_from_meta(
        md,
        page_index=12,
        figure_meta_by_asset={
            "page_13_fig_1.png": {
                "fig_no": 6,
                "caption": "Fig. 6. Statistical comparison results for all four different kinds of images.",
            }
        },
    )
    assert "Figure 6. Statistical comparison results for all four different kinds of images." in out


def test_normalize_page_image_caption_order_moves_matching_caption_below_image():
    md = "\n".join(
        [
            "**Figure 17.** The partial enlargement of the images shown in Fig. 16.",
            "",
            "![Figure 17](./assets/page_20_fig_2.png)",
        ]
    )
    out = PDFConverter._normalize_page_image_caption_order(
        md,
        page_index=19,
        figure_meta_by_asset={
            "page_20_fig_2.png": {
                "fig_no": 17,
                "caption": "Fig. 17. The partial enlargement of the images shown in Fig. 16.",
            }
        },
    )
    assert out.index("![Figure 17](./assets/page_20_fig_2.png)") < out.index("**Figure 17.** The partial enlargement of the images shown in Fig. 16.")
    assert out.count("**Figure 17.** The partial enlargement of the images shown in Fig. 16.") == 1


def test_normalize_page_image_caption_order_moves_image_above_matching_caption_with_short_body_gap():
    md = "\n".join(
        [
            "**Figure 4.** Qualitative evaluations on the synthetic dataset.",
            "",
            "High compression ratio We study the performance of our model under different compression ratios.",
            "",
            "![Figure 4](./assets/page_7_fig_1.png)",
        ]
    )
    out = PDFConverter._normalize_page_image_caption_order(
        md,
        page_index=6,
        figure_meta_by_asset={
            "page_7_fig_1.png": {
                "fig_no": 4,
                "caption": "Fig. 4. Qualitative evaluations on the synthetic dataset.",
            }
        },
    )
    assert out.index("![Figure 4](./assets/page_7_fig_1.png)") < out.index("**Figure 4.** Qualitative evaluations on the synthetic dataset.")
    assert "High compression ratio We study the performance of our model under different compression ratios." in out


def test_merge_adjacent_math_fragments_does_not_swallow_long_prose_with_math_tokens(tmp_path):
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)
    blocks = [
        TextBlock(bbox=(0, 0, 30, 10), text="L =", is_math=True),
        TextBlock(
            bbox=(0, 12, 200, 42),
            text="where R denotes the set of sampled rays r, Y(r) is pixel value of the real captured image corresponding to r, and M(r, i) is the mask value.",
            is_math=False,
        ),
        TextBlock(bbox=(0, 44, 120, 54), text="More prose follows.", is_math=False),
    ]

    merged = converter._merge_adjacent_math_fragments(blocks, page_wh=(220, 300))

    assert len(merged) == 3
    assert merged[0].text == "L ="
    assert merged[1].text.startswith("where R denotes the set of sampled rays")


def test_auto_repair_final_markdown_applies_safe_quality_repairs(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "page_1_fig_1.png").write_bytes(b"png")
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "asset_name": "page_1_fig_1.png",
                        "caption": "Figure 1. Optical layout with the detector and modulation mask.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)
    md = "\n".join(
        [
            "# Demo Paper",
            "",
            "## Abstract",
            "",
            "This converted paper contains a readable abstract for retrieval.",
            "",
            "![Figure 1](./assets/page_1_fig_1.png)",
            "",
            "$$",
            "x = y",
        ]
    )

    out, repair_result = converter._auto_repair_final_markdown(md, out_file=tmp_path / "output.md")

    assert out.lstrip().startswith("<!-- kb_page: 1 -->")
    assert "**Figure 1.** Optical layout with the detector and modulation mask." in out
    assert out.rstrip().endswith("$$")
    assert repair_result["changed"] is True


def test_auto_repair_final_markdown_recovers_missing_pdf_pages_on_first_conversion(tmp_path):
    pdf_path = tmp_path / "nature-layout.pdf"
    page_texts = {
        page: f"page{page:02d} " + " ".join(f"token{page:02d}{idx:03d}" for idx in range(90))
        for page in range(1, 14)
    }
    doc = fitz.open()
    for page in range(1, 14):
        pdf_page = doc.new_page()
        pdf_page.insert_textbox(fitz.Rect(40, 60, 560, 760), page_texts[page], fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    cfg = ConvertConfig(
        pdf_path=pdf_path,
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    converter = PDFConverter(cfg)
    md_parts = []
    for page in range(1, 8):
        md_parts.extend([f"<!-- kb_page: {page} -->", page_texts[page]])
    md_parts.extend(
        [
            "## References",
            f"[1] Reference line one, Journal, 2024. {page_texts[9]} <!-- kb_page: 9 -->",
            f"[2] Reference line two, Journal, 2025. {page_texts[10]} <!-- kb_page: 10 -->",
            "<!-- kb_page: 13 -->",
            page_texts[13],
        ]
    )
    md = "\n\n".join(md_parts)

    out, repair_result = converter._auto_repair_final_markdown(md, out_file=tmp_path / "output.md")

    markers = [int(match.group(1)) for match in re.finditer(r"<!--\s*kb_page:\s*(\d+)\s*-->", out)]
    assert repair_result["changed"] is True
    assert "recover_missing_source_pages" in repair_result["applied"]
    assert markers == list(range(1, 14))
    assert "page08 token08000" in out
    assert "page11 token11000" in out
    assert "page12 token12000" in out
    assert "missing_source_pages" not in repair_result["remaining_issue_codes"]
    assert "page_marker_gaps" not in repair_result["remaining_issue_codes"]
