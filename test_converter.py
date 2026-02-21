#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for PDF to Markdown converter
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from kb.converter import PDFConverter, ConvertConfig, LlmConfig
import os

def test_convert(pdf_path: str, output_dir: str = "test_output", use_llm: bool = False, speed_mode: str = "balanced"):
    """Test PDF conversion with the new converter."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    # Note: ConvertConfig is a frozen dataclass, so we need all fields
    try:
        llm_config = None
        if use_llm:
            # Check both DEEPSEEK_API_KEY and OPENAI_API_KEY (same as kb/config.py)
            api_key = (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
            # Strip quotes if present (same as kb/config.py)
            if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
                api_key = api_key[1:-1].strip()
            if not api_key:
                print("Warning: DEEPSEEK_API_KEY or OPENAI_API_KEY not set, disabling LLM")
                use_llm = False
            else:
                base_url = (os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1").strip().rstrip("/")
                # Be forgiving: add /v1 if missing
                if "api.deepseek.com" in base_url and not base_url.endswith("/v1"):
                    base_url = base_url + "/v1"
                model = os.environ.get("DEEPSEEK_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat"))
                llm_config = LlmConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
        
        # Create config with all required fields
        cfg = ConvertConfig(
            pdf_path=pdf_path,
            out_dir=output_dir,
            translate_zh=False,
            start_page=0,
            end_page=-1,  # -1 means all pages
            skip_existing=False,
            keep_debug=True,
            llm=llm_config,
            speed_mode=speed_mode
        )
        
        # Set optional attributes via __dict__ (since it's frozen)
        # Actually, we modified pipeline to read from instance attributes
        converter = PDFConverter(cfg)
        # Set optional config after creation
        converter.dpi = 200
        converter.analyze_quality = True
        print(f"Converting {pdf_path.name}...")
        converter.convert(str(pdf_path), str(output_dir))
        
        print(f"\n[OK] Conversion completed!")
        print(f"  Output: {output_dir / 'output.md'}")
        if (output_dir / "quality_report.md").exists():
            print(f"  Quality report: {output_dir / 'quality_report.md'}")
        
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import os
    
    # Speed mode options
    SPEED_MODES = {
        '1': ('full_llm', 'Full LLM - Maximum quality, no time limit'),
        '2': ('balanced', 'Balanced - ~25s per PDF, good quality'),
        '3': ('fast', 'Fast - ~10s per PDF, acceptable quality'),
        '4': ('ultra_fast', 'Ultra Fast - ~5s per PDF, basic quality'),
    }
    
    # Find a test PDF
    test_pdfs = [
        "tmp/smoke_test.pdf",
        "tmp/smoke_table_ruled.pdf",
        "tmp/smoke_figure_split.pdf",
    ]
    
    pdf_file = None
    for p in test_pdfs:
        if Path(p).exists():
            pdf_file = p
            break
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    
    if not pdf_file or not Path(pdf_file).exists():
        print("Usage: python test_converter.py [pdf_path] [--speed-mode 1|2|3|4]")
        print("\nSpeed Modes:")
        for key, (mode, desc) in SPEED_MODES.items():
            print(f"  {key}: {desc}")
        print(f"\nLooking for test PDFs: {test_pdfs}")
        if Path("tmp").exists():
            pdfs = list(Path("tmp").glob("*.pdf"))
            if pdfs:
                print(f"\nFound PDFs in tmp/:")
                for p in pdfs[:5]:
                    print(f"  - {p}")
                pdf_file = str(pdfs[0])
                print(f"\nUsing: {pdf_file}")
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Parse speed mode
    speed_mode = 'balanced'  # default
    if '--speed-mode' in sys.argv:
        idx = sys.argv.index('--speed-mode')
        if idx + 1 < len(sys.argv):
            mode_key = sys.argv[idx + 1]
            if mode_key in SPEED_MODES:
                speed_mode = SPEED_MODES[mode_key][0]
            else:
                print(f"Invalid speed mode: {mode_key}. Using default: balanced")
    elif '--full-llm' in sys.argv:
        speed_mode = 'full_llm'
    elif '--fast' in sys.argv:
        speed_mode = 'fast'
    elif '--ultra-fast' in sys.argv:
        speed_mode = 'ultra_fast'
    
    use_llm = "--llm" in sys.argv or speed_mode != 'ultra_fast'
    mode_name, mode_desc = SPEED_MODES.get([k for k, v in SPEED_MODES.items() if v[0] == speed_mode][0] if speed_mode in [v[0] for v in SPEED_MODES.values()] else '2', SPEED_MODES['2'])
    print(f"\nUsing speed mode: {mode_name} - {mode_desc}")
    test_convert(pdf_file, output_dir=f"test_output_{Path(pdf_file).stem}", use_llm=use_llm, speed_mode=speed_mode)
