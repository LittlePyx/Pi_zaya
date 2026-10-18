from __future__ import annotations

from pathlib import Path
import re
import unicodedata


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _fraction_tail(spans: list[dict], number: str) -> bool:
    visible = [span for span in spans if str(span.get("text") or "").strip()]
    if len(visible) < 2 or str(visible[0].get("text") or "").strip() != number:
        return False
    first, punctuation = visible[:2]
    if not str(punctuation.get("text") or "").lstrip().startswith("."):
        return False
    try:
        return (
            0 < float(first["size"]) <= float(punctuation["size"]) - 1.25
            and float(first["origin"][1]) > float(punctuation["origin"][1]) + 0.5
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def demote_fraction_heading_fragments(text: str, pdf_path: Path) -> tuple[str, bool]:
    """Remove heading markup only for exact same-page lowered denominators."""
    lines = text.splitlines()
    candidates = []
    page_number = 0
    fence = ""
    in_math = False
    for index, line in enumerate(lines):
        if line.startswith(("    ", "\t")):
            continue
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if fence_match:
            marker, suffix = fence_match.groups()
            if not fence:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
                fence = ""
            continue
        if fence:
            continue
        if stripped == "$$":
            in_math = not in_math
        if in_math:
            continue
        marker = re.fullmatch(r"<!--\s*kb_page:\s*(\d+)\s*-->", stripped, re.I)
        if marker:
            page_number = int(marker.group(1))
        if re.match(r"^#{1,6}\s+(?:References|Bibliography)\s*$", stripped, re.I):
            break
        match = re.fullmatch(r"#{2,6}\s+((\d{1,2})\s+\.(?:\s+.*)?)", stripped)
        if match and page_number > 0:
            candidates.append((index, page_number, match.group(1), match.group(2)))
    if not candidates:
        return text, False
    try:
        import fitz
        with fitz.open(str(pdf_path)) as doc:
            for index, page_number, body, number in candidates:
                if not 1 <= page_number <= doc.page_count:
                    continue
                matches = []
                for block in doc[page_number - 1].get_text("dict").get("blocks", []):
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if _key("".join(str(span.get("text") or "") for span in spans)) == _key(body):
                            matches.append(_fraction_tail(spans, number))
                if matches and all(matches):
                    lines[index] = re.sub(r"^\s*#{2,6}\s+", "", lines[index], count=1)
    except (ImportError, OSError, ValueError, RuntimeError):
        return text, False
    fixed = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    return fixed, fixed != text
