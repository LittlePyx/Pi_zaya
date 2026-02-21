#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced test script for PDF to Markdown converter
Tests single-column, double-column, and full-width images
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from kb.converter import PDFConverter, ConvertConfig, LlmConfig

def analyze_output(output_dir: Path):
    """Analyze the converted markdown output."""
    import re
    
    md_file = output_dir / "output.md"
    if not md_file.exists():
        print(f"Error: Output file not found: {md_file}")
        return
    
    content = md_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    print("\n" + "="*60)
    print("OUTPUT ANALYSIS")
    print("="*60)
    
    # Count headings
    headings = [l for l in lines if l.strip().startswith("#")]
    print(f"\nHeadings found: {len(headings)}")
    for h in headings[:10]:  # Show first 10
        try:
            print(f"  {h[:80]}")
        except UnicodeEncodeError:
            print(f"  {h[:80].encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')}")
    
    # Count formulas
    in_formula = False
    formula_count = 0
    for line in lines:
        if line.strip().startswith("$$"):
            if not in_formula:
                formula_count += 1
            in_formula = not in_formula
    
    print(f"\nFormulas ($$ blocks): {formula_count}")
    
    # Count tables
    in_table = False
    table_count = 0
    for line in lines:
        if "|" in line and not line.strip().startswith("```"):
            if not in_table:
                table_count += 1
                in_table = True
        elif in_table and not "|" in line and line.strip():
            in_table = False
    
    print(f"Tables: {table_count}")
    
    # Count images
    image_count = sum(1 for l in lines if "![Figure" in l or "![Image" in l)
    print(f"Images: {image_count}")
    
    # Check for references
    ref_section = False
    for i, line in enumerate(lines):
        if re.match(r'^#+\s*REFERENCES?\s*$', line, re.IGNORECASE):
            ref_section = True
            print(f"\nReferences section found at line {i+1}")
            # Count potential references
            ref_lines = 0
            for j in range(i+1, min(i+50, len(lines))):
                l = lines[j]
                if re.search(r'\b(?:19|20)\d{2}\b', l) and (',' in l or 'doi' in l.lower() or 'http' in l.lower()):
                    ref_lines += 1
            print(f"  Potential reference entries: {ref_lines}")
            break
    
    if not ref_section:
        print("\nNo REFERENCES section detected")
    
    # Check for noise
    noise_patterns = ['ACM Trans', 'Publication date', 'PDF Download', 'Total Citations']
    noise_found = []
    for line in lines:
        for pattern in noise_patterns:
            if pattern in line:
                noise_found.append(f"{pattern}: {line[:60]}")
                break
    
    if noise_found:
        print(f"\nPotential noise detected ({len(noise_found)}):")
        for n in noise_found[:5]:
            print(f"  {n}")
    else:
        print("\nNo obvious noise detected")
    
    # Show quality report if exists
    report_file = output_dir / "quality_report.md"
    if report_file.exists():
        print("\n" + "="*60)
        print("QUALITY REPORT SUMMARY")
        print("="*60)
        report = report_file.read_text(encoding="utf-8")
        # Extract issue counts
        import re
        issues = re.findall(r'## (\w+) Issues \((\d+)\)', report)
        if issues:
            for category, count in issues:
                print(f"{category}: {count} issues")
        else:
            print("No issues found in report")
    
    print("\n" + "="*60)

def test_convert(pdf_path: str, output_dir: str = None, use_llm: bool = False):
    """Test PDF conversion with detailed analysis."""
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    if output_dir is None:
        output_dir = f"test_output_{pdf_path.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing conversion of: {pdf_path.name}")
    print(f"Output directory: {output_dir}")
    
    try:
        llm_config = None
        if use_llm:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("Warning: OPENAI_API_KEY not set, disabling LLM")
                use_llm = False
            else:
                llm_config = LlmConfig(
                    api_key=api_key,
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    model="gpt-4o"
                )
        
        cfg = ConvertConfig(
            pdf_path=pdf_path,
            out_dir=output_dir,
            translate_zh=False,
            start_page=0,
            end_page=-1,
            skip_existing=False,
            keep_debug=True,
            llm=llm_config
        )
        
        converter = PDFConverter(cfg)
        converter.dpi = 200
        converter.analyze_quality = True
        
        print("\nStarting conversion...")
        converter.convert(str(pdf_path), str(output_dir))
        
        print("\n[OK] Conversion completed!")
        print(f"  Output: {output_dir / 'output.md'}")
        
        # Analyze output
        analyze_output(output_dir)
        
        return True
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import re
    
    if len(sys.argv) < 2:
        print("Usage: python test_converter_advanced.py <pdf_path> [--llm]")
        print("\nExample:")
        print("  python test_converter_advanced.py tmp/smoke_test.pdf")
        print("  python test_converter_advanced.py tmp/smoke_test.pdf --llm")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    use_llm = "--llm" in sys.argv
    
    test_convert(pdf_file, use_llm=use_llm)
