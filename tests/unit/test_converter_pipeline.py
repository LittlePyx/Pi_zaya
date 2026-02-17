
import pytest
import fitz
import os
from pathlib import Path
from kb.converter.pipeline import PDFConverter
from kb.converter.config import ConvertConfig

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

