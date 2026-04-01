from __future__ import annotations

import re


def _ensure_llm_client(converter, context: str) -> bool:
    if not converter.cfg.llm:
        return False

    if hasattr(converter.llm_worker, "_client") and converter.llm_worker._client:
        return True

    try:
        converter.llm_worker._client = converter.llm_worker._ensure_openai_class()(
            api_key=converter.cfg.llm.api_key,
            base_url=converter.cfg.llm.base_url,
        )
        return True
    except Exception as e:
        print(f"Failed to initialize LLM client for {context}: {e}")
        return False


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text


def llm_light_cleanup(converter, md: str) -> str:
    """Light LLM cleanup - only fix remaining mojibake."""
    if not converter.cfg.llm:
        return md
    if not _ensure_llm_client(converter, "light cleanup"):
        return md

    mojibake_patterns = ["�", "ˆ", "鈥", "閳", "鑶", "鑴", "鑺", "钘"]
    if not any(pattern in md for pattern in mojibake_patterns):
        return md

    max_chunk_size = 12000
    if len(md) <= max_chunk_size:
        return llm_cleanup_chunk(converter, md)

    lines = md.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    repaired_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"LLM cleanup chunk {i+1}/{len(chunks)}...", end="\r")
        repaired_chunks.append(llm_cleanup_chunk(converter, chunk))
    print()

    return "\n\n".join(repaired_chunks)


def llm_cleanup_chunk(converter, md_chunk: str) -> str:
    """Light cleanup of a chunk - fix mojibake and improve inline formulas."""
    prompt = f"""Fix issues in this Markdown chunk:

1. FIX MOJIBAKE (garbled Unicode):
   - Fix broken ligatures so words become readable (fi/fl/ffi/ffl, etc.)
   - Fix broken Greek/math OCR tokens into the intended symbol
   - Fix broken punctuation and dash encodings into the intended punctuation
   - Fix broken accent markers in formulas (for example, hat accents)

2. FIX INLINE FORMULAS (single $...$):
   - Fix garbled math tokens into clean LaTeX (example: broken hat notation -> "\\hat{{C}}(r) - C(r)^2")
   - Remove extra spaces: "C ( r )" -> "C(r)"
   - Fix subscripts/superscripts: "x 2" -> "x^2" or "x_2"
   - Use proper LaTeX commands for accents and symbols (example: hat accent -> "\\hat", alpha -> "\\alpha")

3. PRESERVE:
   - All headings (do NOT change)
   - All display math blocks ($$...$$)
   - All tables, images, code blocks
   - Document structure

Return ONLY the fixed Markdown, no explanations.

INPUT:
{md_chunk}
"""

    try:
        resp = converter.llm_worker._llm_create(
            messages=[
                {"role": "system", "content": "You are a mojibake fixer. Only fix garbled characters."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=min(16384, converter.cfg.llm.max_tokens),
        )
        result = resp.choices[0].message.content or md_chunk
        return _strip_code_fences(result).strip()
    except Exception as e:
        print(f"LLM cleanup failed: {e}")
        return md_chunk


def llm_postprocess_markdown(converter, md: str) -> str:
    """Use LLM to fix mojibake, formulas, and structure in the final markdown."""
    if not converter.cfg.llm:
        return md
    if not _ensure_llm_client(converter, "post-processing"):
        return md

    max_chunk_size = 8000
    if len(md) <= max_chunk_size:
        return llm_repair_markdown_chunk(converter, md)

    lines = md.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    repaired_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"LLM post-processing chunk {i+1}/{len(chunks)}...")
        repaired_chunks.append(llm_repair_markdown_chunk(converter, chunk))

    return "\n\n".join(repaired_chunks)


def llm_repair_markdown_chunk(converter, md_chunk: str) -> str:
    """Repair a chunk of markdown using LLM."""
    prompt = f"""You are fixing a Markdown document converted from a PDF. The conversion has many errors that need to be fixed.

CRITICAL TASKS:

1. FIX ALL MOJIBAKE/GARBLED TEXT (This is the MOST IMPORTANT):
   - Fix broken ligatures like fi/fl/ffi/ffl when words are visibly garbled
   - Fix broken Greek letters in math context (alpha, beta, gamma, delta, sigma, tau, etc.)
   - Fix broken operators and relations (<=, >=, !=, in, not in, times, sum, integral, infinity, etc.)
   - Fix broken punctuation encodings (em dash, en dash, quotes)
   - Fix ALL garbled characters systematically

2. FIX ALL FORMULAS (CRITICAL):
   - Convert ALL Unicode math symbols to LaTeX:
     * Greek letters -> \\alpha, \\beta, \\gamma, \\delta, \\epsilon, \\theta, \\lambda, \\mu, \\pi, \\sigma, \\tau, \\phi, \\omega
     * Relations -> \\leq, \\geq, \\neq, \\in, \\notin
     * Operators -> \\sum, \\prod, \\int, \\infty
   - Fix subscripts: "\\tau 1" -> "\\tau_1", "x i" -> "x_i", "I j" -> "I_j"
   - Fix superscripts: "x 2" -> "x^2" when appropriate
   - Display math (block): use $$...$$ for equations that should be centered
   - Inline math: use $...$ for formulas within text
   - Merge split formulas that belong together
   - Fix spacing: "log k" -> "\\log k", "exp" -> "\\exp", etc.
   - Remove equation numbers from inside formulas

3. FIX HEADING STRUCTURE (CRITICAL):
   - "I. INTRODUCTION" -> "## I. INTRODUCTION" (H2, not H1)
   - "II. MAIN RESULT" -> "## II. MAIN RESULT"
   - "III. SEQUENTIAL..." -> "## III. SEQUENTIAL..."
   - "A. Structured..." -> "### A. Structured..."
   - "B. Sensing..." -> "### B. Sensing..."
   - "IV. PROOF..." -> "## IV. PROOF..."
   - "V. ACKNOWLEDGEMENTS" -> "## V. ACKNOWLEDGEMENTS"
   - "APPENDIX" -> "## APPENDIX" (H2)
   - "A. Proof of..." -> "### A. Proof of..." (H3)
   - "REFERENCES" -> "## REFERENCES" (H2)
   - Remove author names from headings (they should be plain text)
   - Remove duplicate headings completely (if you see the same heading twice, keep only the first occurrence)
   - Remove any heading that appears in the middle of a paragraph (headings should be on their own line)
   - Ensure proper hierarchy: H1 for title only, H2 for main sections (I, II, III, IV, V, APPENDIX, REFERENCES), H3 for subsections (A, B, C, etc.)

4. FIX TEXT CONTENT:
   - Fix misencoded dashes and punctuation in text
   - Fix all ligatures and special characters
   - Preserve mathematical notation in text

5. PRESERVE:
   - All tables (keep markdown table format)
   - All images (keep ![alt](path) format)
   - All code blocks (keep ``` format)
   - All citations [1], [2], etc.
   - All references

6. OUTPUT REQUIREMENTS:
   - Return ONLY the fixed Markdown
   - NO explanations, NO comments, NO code fences
   - Maintain original paragraph structure
   - Keep all content, just fix the errors
   - Remove duplicate headings (if the same heading appears twice, keep only the first occurrence)
   - Ensure headings appear in logical order (I, II, III, IV, V, then APPENDIX, then REFERENCES)

INPUT MARKDOWN (fix all errors):
{md_chunk}
"""

    try:
        resp = converter.llm_worker._llm_create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at fixing PDF-to-Markdown conversion errors, especially mojibake, formula formatting, and document structure.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=converter.cfg.llm.max_tokens,
        )
        result = resp.choices[0].message.content or md_chunk
        return _strip_code_fences(result).strip()
    except Exception as e:
        print(f"LLM post-processing failed: {e}, using original")
        return md_chunk


def llm_final_quality_check(converter, md: str) -> str:
    """Final comprehensive quality check for full_llm mode."""
    if not converter.cfg.llm:
        return md
    if not _ensure_llm_client(converter, "final quality check"):
        return md

    max_chunk_size = 12000
    if len(md) <= max_chunk_size:
        return llm_final_quality_check_chunk(converter, md)

    lines = md.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1
        if line.strip().startswith("## ") and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        elif current_size + line_size > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    checked_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Final quality check chunk {i+1}/{len(chunks)}...", end="\r")
        checked_chunks.append(llm_final_quality_check_chunk(converter, chunk))
    print()

    return "\n\n".join(checked_chunks)


def llm_final_quality_check_chunk(converter, md_chunk: str) -> str:
    """Final comprehensive quality check for a chunk."""
    prompt = f"""You are performing a FINAL COMPREHENSIVE QUALITY CHECK on a Markdown document converted from PDF. 
This is the LAST pass to ensure EVERYTHING is PERFECT. Fix ALL remaining issues.

CRITICAL QUALITY REQUIREMENTS (must be PERFECT):

1. TITLE/HEADING STRUCTURE (MUST BE PERFECT):
   - Ensure proper hierarchy: H1 for title only, H2 for main sections (I, II, III, IV, V, APPENDIX, REFERENCES), H3 for subsections (A, B, C, etc.)
   - Remove any duplicate headings (keep only first occurrence)
   - Remove author names from headings (they should be plain text)
   - Ensure headings are on their own lines (not in middle of paragraphs)
   - Fix any misclassified headings (e.g., headings wrapped in $...$ should be proper markdown headings)
   - Ensure logical order: I, II, III, IV, V, then APPENDIX, then REFERENCES

2. FORMULAS - BOTH INLINE AND DISPLAY (MUST BE PERFECT):
   - Inline formulas ($...$): Must be proper LaTeX, no garbled Unicode, correct subscripts/superscripts
   - Display formulas ($$...$$): Must be proper LaTeX, no nested $ symbols, correct formatting
   - Fix ALL Unicode math symbols to LaTeX (for example: alpha -> \\alpha, beta -> \\beta, <= -> \\leq, >= -> \\geq)
   - Remove equation numbers from inside formulas (they should be outside or use \\tag)
   - Merge split formulas that belong together
   - Fix spacing: "log k" -> "\\log k", "exp" -> "\\exp", etc.
   - Ensure proper grouping with braces

3. IMAGES (MUST BE PERFECT):
   - All images must have proper markdown syntax: ![Figure](./assets/filename.png)
   - Ensure figure captions are properly formatted (bold "Fig. X." prefix)
   - Remove any duplicate image references
   - Ensure images are on their own lines with blank lines around them

4. TABLES (MUST BE PERFECT):
   - All tables must have proper Markdown format with | separators
   - All rows must have the same number of columns
   - Must have proper header row and separator row (---)
   - Fix any garbled text in tables
   - Ensure proper alignment

5. REFERENCES (MUST BE PERFECT):
   - Each reference must start with [number] followed by a space
   - Format: [number] Author names. Title. Conference/Journal, pages, year.
   - Fix all garbled text and mojibake
   - Ensure proper spacing and punctuation
   - Each reference on a separate line
   - Remove duplicate references

6. BODY TEXT (MUST BE PERFECT):
   - Fix ALL mojibake/garbled text (for example: broken ligatures should become fi/fl/ffi/ffl, broken dash encodings should become proper dashes)
   - Fix all ligatures and special characters
   - Ensure proper paragraph structure
   - Preserve mathematical notation in text
   - Fix em dashes and quotes so misencoded punctuation becomes proper Unicode punctuation

7. OVERALL STRUCTURE:
   - Ensure proper spacing between sections
   - Remove any empty or duplicate content
   - Ensure logical flow

OUTPUT REQUIREMENTS:
- Return ONLY the perfected Markdown
- NO explanations, NO comments, NO code fences
- Maintain all content, just fix errors and improve quality
- Ensure EVERYTHING is perfect - this is the final pass

INPUT MARKDOWN (make it PERFECT):
{md_chunk}
"""

    try:
        resp = converter.llm_worker._llm_create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at perfecting PDF-to-Markdown conversions. This is the final quality check - ensure EVERYTHING is perfect.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=converter.cfg.llm.max_tokens,
        )
        result = resp.choices[0].message.content or md_chunk
        return _strip_code_fences(result).strip()
    except Exception as e:
        print(f"Final quality check failed: {e}, using original")
        return md_chunk
