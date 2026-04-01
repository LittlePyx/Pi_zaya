from __future__ import annotations

import json
import os
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


def llm_fix_inline_formulas(converter, md: str) -> str:
    """Fix inline formulas with batch LLM processing for speed."""
    if not converter.cfg.llm:
        return md
    if not _ensure_llm_client(converter, "inline formulas"):
        return md

    pattern = r"\$([^$]+)\$"
    all_formulas = []
    formula_positions = []

    lines = md.splitlines()
    for line_idx, line in enumerate(lines):
        if "$" in line and not line.strip().startswith("$$"):
            matches = list(re.finditer(pattern, line))
            for match in matches:
                formula_text = match.group(1)
                if formula_text.startswith("$") or formula_text.endswith("$"):
                    continue

                has_garbled = any(ch in formula_text for ch in ["ˆ", "�"]) or (" " in formula_text and "\\" not in formula_text)
                if "\\" in formula_text and "\\hat" in formula_text and not has_garbled:
                    continue

                all_formulas.append(formula_text)
                formula_positions.append((line_idx, match.start(), match.end(), formula_text))

    if not all_formulas:
        return md

    print(f"Fixing {len(all_formulas)} inline formulas in batch...")
    fixed_formulas = {}
    result_text = ""
    try:
        prompt = f"""Fix these {len(all_formulas)} inline math expressions to proper LaTeX. Return a JSON array with the fixed formulas.

CRITICAL FIXES for each formula:
1. Fix garbled Unicode or mojibake: broken hat symbols -> "\\hat", "C ( r )" -> "C(r)", "C ( r ) 2" -> "C(r)^2"
2. Use proper LaTeX: subscripts x_i, superscripts x^2, functions \\hat{{C}}
3. Remove ALL extra spaces between symbols
4. Group properly with braces

Input formulas:
{json.dumps(all_formulas, ensure_ascii=False)}

Return JSON array with fixed formulas in the same order, e.g.:
["\\hat{{C}}(r) - C(r)^2", "x_1 + x_2", ...]

Return ONLY the JSON array, no other text:"""

        resp = converter.llm_worker._llm_create(
            messages=[
                {"role": "system", "content": "You are a LaTeX math expert. Return only valid JSON array with fixed formulas."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=3000,
        )
        result_text = (resp.choices[0].message.content or "").strip()
        if result_text.startswith("```"):
            result_text = re.sub(r"^```(?:\w+)?\n?", "", result_text)
            result_text = re.sub(r"\n?```$", "", result_text)

        fixed_batch = json.loads(result_text)
        for i, fixed_formula in enumerate(fixed_batch):
            if i >= len(all_formulas):
                break
            original = all_formulas[i]
            fixed = str(fixed_formula).strip()
            if fixed.startswith("$") and fixed.endswith("$"):
                fixed = fixed[1:-1].strip()
            elif fixed.startswith("$$") and fixed.endswith("$$"):
                fixed = fixed[2:-2].strip()
            fixed = fixed.replace("\\hat C", "\\hat{C}").replace("\\hat{ C", "\\hat{C")
            fixed = re.sub(r"C\s*\(\s*r\s*\)\s*2", r"C(r)^2", fixed)
            fixed = re.sub(r"C\s*\(\s*r\s*\)", r"C(r)", fixed)
            fixed = re.sub(r"\s+", " ", fixed)
            fixed_formulas[original] = fixed
    except json.JSONDecodeError as e:
        print(f"JSON parse failed, trying manual extraction: {e}")
        if "[" in result_text and "]" in result_text:
            try:
                start = result_text.find("[")
                end = result_text.rfind("]") + 1
                if start >= 0 and end > start:
                    json_text = result_text[start:end]
                    fixed_batch = json.loads(json_text)
                    for i, fixed_formula in enumerate(fixed_batch):
                        if i >= len(all_formulas):
                            break
                        original = all_formulas[i]
                        fixed = str(fixed_formula).strip()
                        if fixed.startswith("$"):
                            fixed = fixed.strip("$").strip()
                        fixed = fixed.replace("\\hat C", "\\hat{C}")
                        fixed = re.sub(r"C\s*\(\s*r\s*\)\s*2", r"C(r)^2", fixed)
                        fixed = re.sub(r"C\s*\(\s*r\s*\)", r"C(r)", fixed)
                        fixed_formulas[original] = fixed
            except Exception:
                pass
    except Exception as e:
        print(f"Failed to fix inline formulas: {e}")

    fixed_lines = lines.copy()
    fixed_count = 0
    for line_idx, match_start, match_end, formula_text in reversed(formula_positions):
        if formula_text in fixed_formulas:
            fixed = fixed_formulas[formula_text]
            if fixed and fixed != formula_text:
                line = fixed_lines[line_idx]
                fixed_lines[line_idx] = line[:match_start] + f"${fixed}$" + line[match_end:]
                fixed_count += 1

    if fixed_count > 0:
        print(f"Fixed {fixed_count} inline formulas")

    return "\n".join(fixed_lines)


def llm_fix_display_math(converter, md: str) -> str:
    """Fix display math blocks - remove nested $ symbols and clean formatting."""
    if not converter.cfg.llm:
        return md

    def _env_bool(name: str, default: bool = False) -> bool:
        try:
            raw = str(os.environ.get(name, "") or "").strip().lower()
            if not raw:
                return bool(default)
            return raw in {"1", "true", "yes", "y", "on"}
        except Exception:
            return bool(default)

    enable_llm_fix = _env_bool("KB_PDF_ENABLE_LLM_DISPLAY_MATH_FIX", False)
    if not _ensure_llm_client(converter, "display math"):
        return md

    lines = md.splitlines()
    fixed_lines = []
    in_display_math = False
    math_lines = []
    fixed_count = 0

    print("Fixing display math blocks...")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "$$":
            if in_display_math:
                math_text = "\n".join(math_lines)
                if enable_llm_fix and ("$" in math_text or "\\[" in math_text or "\\]" in math_text):
                    try:
                        prompt = f"""Fix this display math block. Remove any nested $ symbols and fix formatting.

Input:
{math_text}

Requirements:
1. Remove ALL $ symbols inside the math block (display math uses $$...$$, no $ inside)
2. Remove any \\[ or \\] symbols (use $$ only for display math)
3. Fix garbled characters and spacing (for example: recover commands like "\\hat{{x}}" and collapse spaced tokens like "C ( r )" into "C(r)")
4. Use proper LaTeX syntax (e.g., "Z_{{t}} f" -> "\\int_{{t_0}}^{{t_1}}" where t_0 and t_1 are time bounds)
5. Return ONLY the cleaned math content without $$ or \\[ \\] delimiters

LaTeX:"""

                        resp = converter.llm_worker._llm_create(
                            messages=[
                                {"role": "system", "content": "You are a LaTeX math expert. Fix display math blocks by removing nested $ symbols and fixing formatting."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=1000,
                        )
                        fixed_math = (resp.choices[0].message.content or "").strip()
                        if fixed_math.startswith("$$") and fixed_math.endswith("$$"):
                            fixed_math = fixed_math[2:-2].strip()
                        elif fixed_math.startswith("$") and fixed_math.endswith("$"):
                            fixed_math = fixed_math[1:-1].strip()
                        if fixed_math.startswith("```"):
                            fixed_math = re.sub(r"^```(?:\w+)?\n?", "", fixed_math)
                            fixed_math = re.sub(r"\n?```$", "", fixed_math)

                        if fixed_math and fixed_math != math_text:
                            fixed_math = fixed_math.replace("\\[", "").replace("\\]", "")
                            fixed_lines.extend(["$$", fixed_math, "$$"])
                            fixed_count += 1
                        else:
                            cleaned_math = math_text.replace("$", "").replace("\\[", "").replace("\\]", "")
                            fixed_lines.extend(["$$", cleaned_math, "$$"])
                    except Exception:
                        cleaned_math = math_text.replace("$", "").replace("\\[", "").replace("\\]", "")
                        fixed_lines.extend(["$$", cleaned_math, "$$"])
                else:
                    cleaned_lines = [l for l in math_lines if l.strip()]
                    if cleaned_lines:
                        cleaned_lines = [ln.replace("$", "").replace("\\[", "").replace("\\]", "") for ln in cleaned_lines]
                        fixed_lines.append("$$")
                        fixed_lines.extend(cleaned_lines)
                        fixed_lines.append("$$")
                in_display_math = False
                math_lines = []
            else:
                if i + 1 < len(lines) and lines[i + 1].strip() == "$$":
                    continue
                in_display_math = True
        elif in_display_math:
            math_lines.append(line)
        else:
            fixed_lines.append(line)

    if fixed_count > 0:
        print(f"Fixed {fixed_count} display math blocks")

    return "\n".join(fixed_lines)
