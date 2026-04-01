from __future__ import annotations

import re


_TEXT_WORD_RE = re.compile(
    r"\b(the|and|or|is|are|was|were|in|on|at|to|for|of|with|by)\b",
    re.IGNORECASE,
)
_MATH_CHAR_RE = re.compile(r"[+\-*/^_{}\[\]()=<>\\]")
_UNICODE_MATH_RE = re.compile(r"[\u0370-\u03FF\u2200-\u22FF]")
_CJK_RE = re.compile(r"[\u3400-\u9FFF]+")
_BOX_RE = re.compile(r"[\uFFFD\u25A1\u25A0]+")


def merge_split_formulas(md: str) -> str:
    """Merge consecutive formula blocks that were split across lines."""
    lines = md.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "$$":
            formula_parts = []
            i += 1

            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()

                if next_stripped == "$$":
                    i += 1
                    break
                if next_stripped:
                    formula_parts.append(next_stripped)
                i += 1

            if formula_parts:
                merged = re.sub(r"\s+", " ", " ".join(formula_parts))
                result.extend(["$$", merged, "$$", ""])
        else:
            result.append(line)
            i += 1

    final_result = []
    i = 0
    while i < len(result):
        line = result[i]
        stripped = line.strip()

        if stripped == "$$":
            formula_blocks = []
            i += 1

            current_formula = []
            while i < len(result) and result[i].strip() != "$$":
                if result[i].strip():
                    current_formula.append(result[i].strip())
                i += 1

            if i < len(result) and result[i].strip() == "$$":
                i += 1
                if current_formula:
                    formula_blocks.append(" ".join(current_formula))

            blank_count = 0
            inline_math_blocks = []
            while i < len(result) and blank_count < 3:
                next_line = result[i]
                next_stripped = next_line.strip()

                if next_stripped == "":
                    blank_count += 1
                    i += 1
                elif next_stripped == "$$":
                    i += 1
                    next_formula = []
                    while i < len(result) and result[i].strip() != "$$":
                        if result[i].strip():
                            next_formula.append(result[i].strip())
                        i += 1
                    if i < len(result) and result[i].strip() == "$$":
                        i += 1
                        if next_formula:
                            formula_blocks.append(" ".join(next_formula))
                    blank_count = 0
                elif re.match(r"^\$[^$]+\$$", next_stripped):
                    inline_math_blocks.append(next_stripped)
                    i += 1
                    blank_count = 0
                else:
                    if len(next_stripped) < 50 and _MATH_CHAR_RE.search(next_stripped):
                        inline_math_blocks.append(next_stripped)
                        i += 1
                        blank_count = 0
                    else:
                        break

            all_parts = formula_blocks + inline_math_blocks
            if all_parts:
                merged = re.sub(r"\s+", " ", " ".join(all_parts))
                final_result.extend(["$$", merged, "$$", ""])
        else:
            final_result.append(line)
            i += 1

    return "\n".join(final_result)


def merge_fragmented_formulas(md: str) -> str:
    """
    Aggressively merge fragmented formula pieces that are on separate lines.
    Detects formula fragments and merges them into complete formulas.
    """
    lines = md.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if is_formula_fragment(stripped):
            formula_parts = [stripped]
            j = i + 1

            while j < len(lines) and j < i + 10:
                next_line = lines[j]
                next_stripped = next_line.strip()

                if not next_stripped:
                    j += 1
                    continue

                if is_formula_fragment(next_stripped):
                    formula_parts.append(next_stripped)
                    j += 1
                elif looks_like_formula_continuation(next_stripped, formula_parts):
                    formula_parts.append(next_stripped)
                    j += 1
                else:
                    break

            if len(formula_parts) > 1:
                merged_formula = " ".join(formula_parts)
                merged_formula = re.sub(r"\s+", " ", merged_formula)
                merged_formula = re.sub(r"\s*([+\-*/=,()\[\]{}])\s*", r"\1", merged_formula)
                merged_formula = re.sub(r"([a-zA-Z])\s+([a-zA-Z])", r"\1\2", merged_formula)
                if not merged_formula.startswith("$$"):
                    merged_formula = f"$${merged_formula}$$"
                result.append(merged_formula)
                i = j
                continue

        result.append(line)
        i += 1

    return "\n".join(result)


def is_formula_fragment(text: str) -> bool:
    """Check if a line is a formula fragment (part of a larger formula)."""
    if not text or len(text) < 1:
        return False

    if text.startswith("$$") or text.endswith("$$"):
        return True

    fragment_patterns = [
        r"^(?:\\sum|\\int|\\prod)\b",
        r"^[a-zA-Z]_\{[^}]+\}",
        r"^[a-zA-Z]\^\{[^}]+\}",
        r"^[a-zA-Z]\^[0-9]",
        r"^[a-zA-Z]_[0-9a-z]",
        r"^\\[a-zA-Z]+\{",
        r"^[+\-*/=]",
        r"^[()\[\]{}]",
        r"^[0-9]+\s*[+\-*/=]",
        r"^[a-zA-Z]\([^)]*\)",
        r"^<[^>]+>",
        r"^[a-zA-Z]\s*=\s*[0-9]",
    ]

    for pattern in fragment_patterns:
        if re.match(pattern, text):
            return True

    if len(text) <= 15 and (_MATH_CHAR_RE.search(text) or _UNICODE_MATH_RE.search(text)):
        if not _TEXT_WORD_RE.search(text):
            return True

    return False


def looks_like_formula_continuation(text: str, existing_parts: list[str]) -> bool:
    """Check if text looks like a continuation of the formula fragments."""
    if not text:
        return False

    has_math = bool(_MATH_CHAR_RE.search(text) or _UNICODE_MATH_RE.search(text))
    has_text_words = bool(_TEXT_WORD_RE.search(text))

    if has_math and not has_text_words:
        return True

    if re.match(r"^[()\[\]{}+\-*/=,<>\\]", text.strip()):
        return True

    return False


def basic_formula_cleanup(md: str) -> str:
    """
    Basic formula cleanup - only remove obviously broken fragments.
    CRITICAL: Preserve all $$...$$ and $...$ blocks.
    """
    lines = md.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "$$ $$" or (stripped == "$$" and i + 1 < len(lines) and lines[i + 1].strip() == "$$"):
            i += 1
            continue

        if stripped in [")(", "()", ")((", "()("]:
            i += 1
            continue

        if len(stripped) <= 5 and re.match(r"^[Nn]$|^i\s*=\s*1$|^\\sum\s*$", stripped):
            in_math_block = False
            for j in range(i - 1, max(-1, i - 20), -1):
                if j < 0:
                    break
                if lines[j].strip() == "$$":
                    in_math_block = True
                    break

            if not in_math_block:
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    if next_stripped and len(next_stripped) <= 10 and (_MATH_CHAR_RE.search(next_stripped) or _UNICODE_MATH_RE.search(next_stripped)):
                        i += 1
                        continue
                i += 1
                continue

        if stripped.count("$$") >= 4:
            parts = re.split(r"\$\$", stripped)
            formula_parts = []
            for part in parts:
                part = part.strip()
                if part and not re.match(r"^[)(]+$", part):
                    part = re.sub(r"^[)(]+", "", part)
                    part = re.sub(r"[)(]+$", "", part)
                    if part and len(part) > 2:
                        formula_parts.append(part)

            if formula_parts:
                merged = re.sub(r"\s+", " ", " ".join(formula_parts))
                if len(merged) > 10:
                    line = f"$${merged}$$"

        result.append(line)
        i += 1

    return "\n".join(result)


def is_likely_formula(text: str) -> bool:
    """Check if text is likely a formula (not regular text)."""
    if not text or len(text.strip()) < 2:
        return False

    text = text.strip()

    if (text.startswith("$$") and text.endswith("$$")) or (
        text.startswith("$") and text.endswith("$") and text.count("$") == 2
    ):
        return True

    formula_indicators = [
        r"\\[a-zA-Z]+\{",
        r"[a-zA-Z]_\{[^}]+\}",
        r"[a-zA-Z]\^\{[^}]+\}",
        r"\\[a-zA-Z]+",
        r"[\u0370-\u03FF]",
        r"[\u2200-\u22FF]",
    ]

    has_formula_pattern = any(re.search(pattern, text) for pattern in formula_indicators)

    text_indicators = [
        r"\b(the|and|or|is|are|was|were|in|on|at|to|for|of|with|by)\b",
        r"[.!?,;:]",
        r"[A-Z][a-z]+\s+[A-Z][a-z]+",
    ]

    has_text_pattern = any(re.search(pattern, text, re.IGNORECASE) for pattern in text_indicators)

    if has_formula_pattern and not has_text_pattern:
        return True

    if len(text) < 20 and re.search(r"[+\-*/^_{}\[\]()=]", text) and not has_text_pattern:
        return True

    return False


def fix_vision_formula_errors(md: str) -> str:
    """
    Fix common formula errors from vision model output.
    """
    lines = md.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            formula_content = stripped[2:-2].strip()
            if not is_likely_formula(formula_content):
                result.append(formula_content)
                i += 1
                continue

        if stripped.startswith("$$") and not stripped.endswith("$$"):
            formula_parts = [line]
            i += 1
            found_closing = False
            while i < len(lines):
                next_line = lines[i]
                formula_parts.append(next_line)
                if next_line.strip().endswith("$$"):
                    found_closing = True
                    break
                i += 1

            if found_closing:
                merged = " ".join(part.strip() for part in formula_parts)
                formula_content = merged.replace("$$", "").strip()
                if is_likely_formula(formula_content):
                    result.append(merged)
                else:
                    result.append(formula_content)
            else:
                formula_content = " ".join(part.strip() for part in formula_parts).replace("$$", "").strip()
                if is_likely_formula(formula_content):
                    result.append(" ".join(part.strip() for part in formula_parts))
                else:
                    result.append(formula_content)
            i += 1
            continue

        if "$" in line:
            inline_formula_pattern = r"\$([^$]+)\$"
            matches = list(re.finditer(inline_formula_pattern, line))
            for match in reversed(matches):
                formula_content = match.group(1)
                if not is_likely_formula(formula_content):
                    line = line[:match.start()] + formula_content + line[match.end():]

            if "$" in line:
                line = re.sub(r"([A-Za-z])'\s+([a-z])(?![_^{])", r"\1'_{\2}", line)
                line = re.sub(r"([A-Za-z])'\s+_\{([^}]+)\}", r"\1'_{\2}", line)
                line = re.sub(
                    r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega)([a-z])(?![_^{])",
                    r"\\\1_{\2}",
                    line,
                )
                line = re.sub(r"\\(partial)\s+([a-z])\s+([a-z])(?![_^{])", r"\\\1 \2_{\3}", line)
                line = re.sub(
                    r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega)([a-z])(?![_^{\\])",
                    r"\\\1_{\2}",
                    line,
                )

                line = re.sub(r"(\$[^$]*)" + _CJK_RE.pattern + r"([^$]*\$)", r"\1\2", line)
                line = re.sub(r"(\$\$[^$]*)" + _CJK_RE.pattern + r"([^$]*\$\$)", r"\1\2", line)
                line = re.sub(r"(\$[^$]*)" + _BOX_RE.pattern + r"([^$]*\$)", r"\1\2", line)
                line = re.sub(r"(\$\$[^$]*)" + _BOX_RE.pattern + r"([^$]*\$\$)", r"\1\2", line)
                line = re.sub(r"([a-zA-Z])\s+_(\w)", r"\1_\2", line)
                line = re.sub(r"([a-zA-Z])\s+\^(\w)", r"\1^\2", line)

                if line.count("$") >= 2 and line.count("$") % 2 == 0:
                    line = re.sub(r"\$\s+([^$]+)\s+\$", r"$ \1 $", line)

                line = re.sub(
                    r"\$([^$]+)\$\s+\$([^$]+)\$",
                    lambda m: f"${m.group(1)} {m.group(2)}$"
                    if is_likely_formula(m.group(1) + " " + m.group(2))
                    else f"${m.group(1)}$ ${m.group(2)}$",
                    line,
                )

        if "|" in line and line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if len(cells) >= 2 and "---" not in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if "|" in next_line and "---" not in next_line:
                    sep = "| " + " | ".join(["---"] * len(cells)) + " |"
                    result.append(line)
                    result.append(sep)
                    i += 1
                    continue

        result.append(line)
        i += 1

    return "\n".join(result)


def formula_to_plain_text(formula: str) -> str:
    """Convert LaTeX formula to plain text for references."""
    if not formula:
        return ""

    text = formula
    text = re.sub(r"_\{([^}]+)\}", r" \1", text)
    text = re.sub(r"_([a-z0-9])", r" \1", text)
    text = re.sub(r"\^\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\^([a-z0-9])", r"\1", text)
    text = re.sub(r"\\alpha", "alpha", text)
    text = re.sub(r"\\beta", "beta", text)
    text = re.sub(r"\\gamma", "gamma", text)
    text = re.sub(r"\\delta", "delta", text)
    text = re.sub(r"\\[a-z]+\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
