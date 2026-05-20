from __future__ import annotations

import re


_FOCUSED_READING_PATH_PROMPT_RE = re.compile(
    r"(?i)(?:"
    r"\bwhere\s+should\s+i\s+start\b|\bwhat\s+should\s+i\s+read\s+first\b|"
    r"\bwhich\s+(?:paragraphs?|sections?|parts?)\s+should\s+i\s+(?:read|start)\b|"
    r"应该先读|先读哪|先看哪|从哪里开始读|从哪(?:几|些)?(?:段|节|部分)开始|"
    r"读哪(?:几|些)?(?:段|节|部分)|哪(?:几|些)?(?:段|节|小节|部分).{0,12}读|阅读路线|阅读顺序"
    r")",
)

_FULL_DOC_MAP_PROMPT_RE = re.compile(
    r"(?i)(?:\bdoc\s*map\b|\bsection(?:\s*-\s*|\s+)by(?:\s*-\s*|\s+)section\b|"
    r"\btable\s+of\s+contents\b|\btoc\b|\boutline\b|目录|大纲|每一块|每段总结|按段总结|章节概览)"
)


def _paper_guide_prompt_requests_focused_reading_path(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(_FOCUSED_READING_PATH_PROMPT_RE.search(q)) and not bool(_FULL_DOC_MAP_PROMPT_RE.search(q))


def _doc_map_record_heading(rec: dict) -> str:
    return str((rec or {}).get("heading_path") or "").strip()


def _doc_map_record_anchor(rec: dict) -> str:
    return str((rec or {}).get("locate_anchor") or "").strip()


def _looks_like_title_or_author_doc_map_record(rec: dict) -> bool:
    heading = _doc_map_record_heading(rec)
    anchor = _doc_map_record_anchor(rec)
    if not anchor:
        return True
    heading_low = heading.lower()
    anchor_low = anchor.lower()
    if "/" not in heading and re.search(r"@(westlake|zhejiang|edu|gmail|outlook)|university|{.*@|author", anchor_low):
        return True
    if "/" not in heading and heading_low and anchor_low.startswith(heading_low[: min(28, len(heading_low))]):
        return True
    return False


def _focused_doc_map_score(rec: dict, *, prompt: str, order_index: int) -> float:
    heading = _doc_map_record_heading(rec)
    anchor = _doc_map_record_anchor(rec)
    leaf = heading.rsplit("/", 1)[-1].strip() if heading else ""
    heading_low = heading.lower()
    leaf_low = leaf.lower()
    anchor_low = anchor.lower()
    low = f"{leaf_low}\n{anchor_low}"
    prompt_low = str(prompt or "").lower()
    score = 0.0
    if _looks_like_title_or_author_doc_map_record(rec):
        return -1000.0
    if "abstract" in leaf_low:
        score += 100.0
    if re.search(r"\bfigure\s*1\b|图\s*1", leaf_low) or "overview" in anchor_low:
        score += 96.0
    if re.search(r"\bfigure\s*2\b|图\s*2", leaf_low) and re.search(r"image formation|video sci|overview", heading_low):
        score += 88.0
    if re.search(r"^3\.\s*method\b|^method$", leaf_low):
        score += 92.0
    if re.search(r"image formation|video sci|成像模型", leaf_low) or re.search(r"formation process|sci measurement|binary masks|压缩", anchor_low):
        score += 90.0
    if re.search(r"proposed framework|framework|框架", leaf_low) or re.search(r"photometric consistency|optimiz(?:e|ing)|优化", anchor_low):
        score += 88.0
    if re.search(r"background on nerf", leaf_low) or re.search(r"nerf transfers|volumetric rendering|view-dependent|视角", anchor_low):
        score += 84.0
    if re.search(r"introduction", leaf_low) or re.search(r"motivation|challenge", anchor_low):
        score += 72.0
    if "related work" in leaf_low or re.search(r"prior works?", anchor_low):
        score += 45.0
    if re.search(r"experiments?|experimental setup", leaf_low) or re.search(r"evaluate|dataset", anchor_low):
        score += 8.0
    if re.search(r"压缩|compressed|sci|mask", prompt_low) and re.search(r"compressed|snapshot compressive|sci|mask|measurement|压缩", low):
        score += 12.0
    if re.search(r"3d|三维|视角|view|scene", prompt_low) and re.search(r"3d|scene|view|nerf|camera|pose|render", low):
        score += 12.0
    if not re.search(r"实验|结果|数据集|experiment|result|dataset|benchmark", prompt_low):
        if re.search(r"experiments?|experimental setup|additional study", heading_low):
            score -= 80.0
    return score - min(float(order_index) * 0.02, 2.0)


def _focused_doc_map_order_key(rec: dict) -> tuple[int, str]:
    heading = _doc_map_record_heading(rec).lower()
    anchor = _doc_map_record_anchor(rec).lower()
    surface = f"{heading}\n{anchor}"
    if "abstract" in surface:
        return (10, heading)
    if re.search(r"\bfigure\s*1\b|图\s*1", surface):
        return (20, heading)
    if "introduction" in surface:
        return (30, heading)
    if re.search(r"\b3\.\s*method\b", surface) and "3.1" not in surface and "3.2" not in surface and "3.3" not in surface:
        return (40, heading)
    if "background on nerf" in surface:
        return (50, heading)
    if "image formation" in surface or "video sci" in surface:
        return (60, heading)
    if "proposed framework" in surface:
        return (70, heading)
    if re.search(r"\bfigure\s*2\b|图\s*2", surface):
        return (80, heading)
    if "related work" in surface:
        return (90, heading)
    if "experiment" in surface:
        return (100, heading)
    return (999, heading)


def _select_focused_doc_map_records(
    records: list[dict],
    *,
    prompt: str,
    max_items: int = 6,
) -> list[dict]:
    ranked: list[tuple[float, int, dict]] = []
    seen: set[str] = set()
    for idx, rec in enumerate(list(records or [])):
        if not isinstance(rec, dict):
            continue
        anchor = _doc_map_record_anchor(rec)
        if not anchor:
            continue
        key = f"{_doc_map_record_heading(rec).lower()}|{anchor[:160].lower()}"
        if key in seen:
            continue
        seen.add(key)
        score = _focused_doc_map_score(rec, prompt=prompt, order_index=idx)
        if score < 20.0:
            continue
        ranked.append((score, idx, rec))
    if not ranked:
        return [
            rec
            for rec in list(records or [])
            if isinstance(rec, dict) and not _looks_like_title_or_author_doc_map_record(rec)
        ][:max_items]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    picked = [dict(item[2]) for item in ranked[: max(1, int(max_items or 6))]]
    picked.sort(key=_focused_doc_map_order_key)
    return picked
