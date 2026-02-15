import argparse
import sys
from .config import ConvertConfig
from .pipeline import PDFConverter

def main():
    parser = argparse.ArgumentParser(description="Reliable PDF to Markdown Converter")
    
    # Input/Output
    parser.add_argument("pdf_path", help="Path to input PDF file")
    parser.add_argument("--save_dir", "-o", default="output", help="Directory to save output (default: output)")
    
    # Config overrides
    parser.add_argument("--dpi", type=float, default=192, help="DPI for image extraction")
    parser.add_argument("--noise_threshold", type=float, default=600, help="Noise threshold for small text")
    
    # LLM Options
    parser.add_argument("--llm", action="store_true", help="Enable LLM processing")
    parser.add_argument("--llm_api_key", help="OpenAI API Key (or set OPENAI_API_KEY env)")
    parser.add_argument("--llm_model", default="gpt-4o", help="LLM Model to use")
    parser.add_argument("--llm_workers", type=int, default=4, help="Parallel workers for LLM")

    args = parser.parse_args()
    
    # Build config
    cfg = ConvertConfig()
    cfg.dpi = args.dpi
    # ... map other args ...
    
    if args.llm:
        cfg.llm = cfg.LLMConfig() # Ensure sub-config exists
        if args.llm_api_key:
            cfg.llm.api_key = args.llm_api_key
        if args.llm_model:
            cfg.llm.model = args.llm_model
        cfg.llm_workers = args.llm_workers
    else:
        cfg.llm = None
        
    converter = PDFConverter(cfg)
    try:
        converter.convert(args.pdf_path, args.save_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
