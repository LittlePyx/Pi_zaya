
import json
import pytest
import fitz
import os
from pathlib import Path
from kb.converter.pipeline import PDFConverter
from kb.converter.config import ConvertConfig
from kb.converter.page_figure_metadata import persist_page_figure_metadata
from kb.converter.models import TextBlock

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
    
    # Check simple structure based on heuristics (fontsize)
    # The pipeline might not perfectly identify H1 without LLM, but let's see.
    # Note: Default heuristic might treat larger font as Heading.
    # We assert presence first. Structure assertion depends on specific heuristic tuning.
    
    # Check that assets dir was created
    assert (output_dir / "assets").exists()

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
