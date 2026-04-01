from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


_REF_LINE_RE = re.compile(r"^\[(\d+)\]\s+")


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
        text = re.sub(r"^```(?:\w+)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def llm_fix_references(converter, md: str) -> str:
    """Fix references section formatting with LLM."""
    if not converter.cfg.llm:
        return md
    if not _ensure_llm_client(converter, "references"):
        return md

    lines = md.splitlines()
    ref_start = None
    for i, line in enumerate(lines):
        if re.match(r"^#+\s+References", line, re.IGNORECASE):
            ref_start = i
            break

    if ref_start is None:
        return md

    first_ref_line = None
    for idx in range(ref_start + 1, min(ref_start + 10, len(lines))):
        line = lines[idx].strip()
        if line:
            first_ref_line = line
            break
    if first_ref_line and re.match(r"^\[\d+\]", first_ref_line):
        formatted_count = 0
        total_count = 0
        for idx in range(ref_start + 1, min(ref_start + 30, len(lines))):
            line = lines[idx].strip()
            if line:
                total_count += 1
                if re.match(r"^\[\d+\]", line):
                    formatted_count += 1
        if total_count > 0 and formatted_count / total_count >= 0.8:
            return md

    print("Fixing references formatting...")
    ref_lines = lines[ref_start + 1 :]
    ref_text = "\n".join(ref_lines)

    max_ref_chunk = 20000
    if len(ref_text) <= max_ref_chunk:
        formatted_refs = llm_format_references_chunk(converter, ref_text)
        if formatted_refs:
            return "\n".join(lines[: ref_start + 1]) + "\n\n" + formatted_refs
    else:
        ref_chunks = []
        mid_point = len(ref_lines) // 2
        ref_chunks.append("\n".join(ref_lines[:mid_point]))
        ref_chunks.append("\n".join(ref_lines[mid_point:]))

        formatted_chunks = [None] * len(ref_chunks)
        with ThreadPoolExecutor(max_workers=min(2, len(ref_chunks))) as ex:
            fut_to_idx = {
                ex.submit(llm_format_references_chunk, converter, chunk): i
                for i, chunk in enumerate(ref_chunks)
            }
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                print(f"Processing references chunk {i+1}/{len(ref_chunks)}...", end="\r")
                try:
                    formatted = fut.result()
                except Exception:
                    formatted = None
                formatted_chunks[i] = formatted if formatted else ref_chunks[i]
        print()

        if formatted_chunks:
            return "\n".join(lines[: ref_start + 1]) + "\n\n" + "\n\n".join(formatted_chunks)

    return md


def llm_format_references_chunk(converter, ref_text: str) -> Optional[str]:
    """Format a chunk of references using LLM."""
    try:
        prompt = f"""Format this references section properly. Each reference should be on its own line, properly formatted.

Requirements:
1. Each reference should start with [number] followed by a space
2. Format should be: [number] Author names. Title. Conference/Journal, pages, year.
3. Fix any garbled text and mojibake (for example: broken author names, ligatures, or punctuation)
4. Ensure proper spacing and punctuation
5. Keep all citation numbers exactly as they appear
6. Each reference on a separate line
7. Remove any duplicate references
8. Fix special characters properly (for example: accents, apostrophes, and Greek letters should be valid Unicode)

References section:
{ref_text}

Return ONLY the formatted references, one per line, no explanations:"""

        resp = converter.llm_worker._llm_create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at formatting academic references. Return properly formatted references with correct Unicode characters.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=6000,
        )
        formatted_refs = (resp.choices[0].message.content or "").strip()
        formatted_refs = _strip_code_fences(formatted_refs)
        return formatted_refs if formatted_refs else None
    except Exception as e:
        print(f"Failed to format references chunk: {e}")
        return None


def _split_numbered_reference_lines(text: str) -> list[str]:
    return [str(line or "").strip() for line in str(text or "").splitlines() if _REF_LINE_RE.match(str(line or "").strip())]


def _reference_number(line: str) -> Optional[str]:
    match = _REF_LINE_RE.match(str(line or "").strip())
    if not match:
        return None
    return str(match.group(1))


def llm_polish_references(converter, md: str) -> str:
    """Lightly repair OCR noise in already-formatted references while preserving numbering."""
    if not converter.cfg.llm:
        return md
    if not _ensure_llm_client(converter, "reference polish"):
        return md

    lines = md.splitlines()
    ref_start = None
    for i, line in enumerate(lines):
        if re.match(r"^#+\s+References?\s*$", str(line or "").strip(), re.IGNORECASE):
            ref_start = i
            break
    if ref_start is None:
        return md

    ref_lines = lines[ref_start + 1 :]
    indexed_positions: list[int] = []
    indexed_lines: list[str] = []
    for idx, line in enumerate(ref_lines):
        stripped = str(line or "").strip()
        if _REF_LINE_RE.match(stripped):
            indexed_positions.append(idx)
            indexed_lines.append(stripped)
    if len(indexed_lines) < 3:
        return md

    chunk_size = 18
    polished_chunks: dict[int, str] = {}
    any_changed = False

    for start in range(0, len(indexed_lines), chunk_size):
        chunk_lines = indexed_lines[start : start + chunk_size]
        prompt = (
            "Lightly repair OCR noise in these already-numbered academic references.\n"
            "Requirements:\n"
            "1. Return exactly the same number of lines as the input.\n"
            "2. Keep each leading citation number [n] exactly unchanged and in the same order.\n"
            "3. Do not add, remove, merge, split, or reorder references.\n"
            "4. Only fix obvious OCR or formatting issues: missing hyphens or spaces, broken Unicode, ligatures, punctuation, and easy spelling mistakes.\n"
            "5. Do not invent metadata such as DOI, venue, volume, or pages.\n"
            "6. Keep one reference per line and return only the corrected lines.\n\n"
            "INPUT:\n"
            + "\n".join(chunk_lines)
        )
        try:
            resp = converter.llm_worker._llm_create(
                messages=[
                    {
                        "role": "system",
                        "content": "You repair OCR noise in academic references. Preserve numbering and order exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=min(6000, int(getattr(converter.cfg.llm, "max_tokens", 6000) or 6000)),
            )
            content = _strip_code_fences((resp.choices[0].message.content or "").strip())
        except Exception as e:
            print(f"Failed to polish references chunk: {e}")
            continue

        out_lines = _split_numbered_reference_lines(content)
        if len(out_lines) != len(chunk_lines):
            continue
        if any(_reference_number(src) != _reference_number(out) for src, out in zip(chunk_lines, out_lines)):
            continue

        for rel_idx, out_line in enumerate(out_lines):
            abs_idx = start + rel_idx
            polished_chunks[abs_idx] = out_line
            if out_line != chunk_lines[rel_idx]:
                any_changed = True

    if not any_changed:
        return md

    rebuilt_ref_lines = list(ref_lines)
    for abs_idx, out_line in polished_chunks.items():
        pos = indexed_positions[abs_idx]
        rebuilt_ref_lines[pos] = out_line
    return "\n".join(lines[: ref_start + 1] + rebuilt_ref_lines)


def llm_fix_tables(converter, md: str) -> str:
    """Fix tables formatting with LLM."""
    if not converter.cfg.llm:
        return md
    if not _ensure_llm_client(converter, "tables"):
        return md

    lines = md.splitlines()
    fixed_lines = []
    in_table = False
    table_lines = []
    fixed_table_count = 0

    print("Checking for tables to fix...")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                table_text = "\n".join(table_lines)
                if "---" in table_text and len(table_lines) >= 3:
                    fixed_lines.extend(table_lines)
                elif len(table_lines) >= 3:
                    try:
                        prompt = f"""Fix this Markdown table. Ensure proper formatting, alignment, and fix any garbled text.

Requirements:
1. Ensure all rows have the same number of columns
2. Fix any garbled text or mojibake
3. Ensure proper alignment with | separators
4. Keep header row and separator row (---)
5. Fix any spacing issues

Table:
{table_text}

Return ONLY the fixed table in Markdown format, no explanations:"""

                        resp = converter.llm_worker._llm_create(
                            messages=[
                                {"role": "system", "content": "You are an expert at formatting Markdown tables. Return properly formatted tables."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=2000,
                        )
                        fixed_table = (resp.choices[0].message.content or "").strip()
                        fixed_table = _strip_code_fences(fixed_table)

                        if fixed_table and fixed_table != table_text:
                            fixed_lines.extend(fixed_table.splitlines())
                            fixed_table_count += 1
                        else:
                            fixed_lines.extend(table_lines)
                    except Exception:
                        fixed_lines.extend(table_lines)
                else:
                    fixed_lines.extend(table_lines)

                in_table = False
                table_lines = []

            fixed_lines.append(line)

    if in_table and table_lines:
        fixed_lines.extend(table_lines)

    if fixed_table_count > 0:
        print(f"Fixed {fixed_table_count} tables")

    return "\n".join(fixed_lines)


def llm_fix_tables_with_screenshot(converter, md: str, pdf_path: Path, save_dir: Path) -> str:
    """Fix tables formatting with LLM, or screenshot if too difficult."""
    if not converter.cfg.llm:
        return converter._llm_fix_tables(md)
    if not _ensure_llm_client(converter, "tables"):
        return md

    lines = md.splitlines()
    fixed_lines = []
    in_table = False
    table_lines = []
    table_caption = None
    fixed_table_count = 0
    screenshot_count = 0
    table_num = 1

    print("Checking for tables to fix (with screenshot fallback)...")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i > 0 and (stripped.lower().startswith("table ") or "*Table" in stripped or "Table" in stripped):
            table_caption = stripped

        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                table_text = "\n".join(table_lines)
                has_separator = any("---" in tl for tl in table_lines)
                has_multiple_rows = len([l for l in table_lines if l.strip().startswith("|")]) >= 3
                if has_separator and has_multiple_rows:
                    fixed_lines.extend(table_lines)
                elif len(table_lines) >= 3:
                    try:
                        prompt = f"""Fix this Markdown table. Ensure proper formatting, alignment, and fix any garbled text.

Requirements:
1. Ensure all rows have the same number of columns
2. Fix any garbled text or mojibake
3. Ensure proper alignment with | separators
4. Keep header row and separator row (---)
5. Fix any spacing issues

Table:
{table_text}

Return ONLY the fixed table in Markdown format, no explanations. If the table is too complex or corrupted, return "SCREENSHOT" as the only word:"""

                        resp = converter.llm_worker._llm_create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert at formatting Markdown tables. Return properly formatted tables or 'SCREENSHOT' if too difficult.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=2000,
                        )
                        fixed_table = (resp.choices[0].message.content or "").strip()
                        fixed_table = _strip_code_fences(fixed_table)

                        if fixed_table.upper() == "SCREENSHOT" or "SCREENSHOT" in fixed_table.upper():
                            screenshot_path = converter._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                            if screenshot_path:
                                fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                if table_caption:
                                    fixed_lines.append(f"*{table_caption}*")
                                screenshot_count += 1
                                table_num += 1
                            else:
                                fixed_lines.extend(table_lines)
                        elif fixed_table and fixed_table != table_text:
                            fixed_lines.extend(fixed_table.splitlines())
                            if table_caption:
                                fixed_lines.append(f"*{table_caption}*")
                            fixed_table_count += 1
                        else:
                            screenshot_path = converter._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                            if screenshot_path:
                                fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                if table_caption:
                                    fixed_lines.append(f"*{table_caption}*")
                                screenshot_count += 1
                                table_num += 1
                            else:
                                fixed_lines.extend(table_lines)
                    except Exception as e:
                        print(f"LLM table fix failed, trying screenshot: {e}")
                        screenshot_path = converter._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                        if screenshot_path:
                            fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                            if table_caption:
                                fixed_lines.append(f"*{table_caption}*")
                            screenshot_count += 1
                            table_num += 1
                        else:
                            fixed_lines.extend(table_lines)
                else:
                    fixed_lines.extend(table_lines)

                in_table = False
                table_lines = []
                table_caption = None

            fixed_lines.append(line)

    if in_table and table_lines:
        fixed_lines.extend(table_lines)

    if fixed_table_count > 0:
        print(f"Fixed {fixed_table_count} tables")
    if screenshot_count > 0:
        print(f"Screenshot {screenshot_count} difficult tables")

    return "\n".join(fixed_lines)
