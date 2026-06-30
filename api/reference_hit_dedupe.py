from __future__ import annotations

from collections.abc import Callable
import hashlib
import re

from api.reference_card_quality import ref_card_polish_status
from api.reference_value_utils import _positive_int


def _refs_hit_ui_meta(hit: dict | None) -> dict:
    return (hit or {}).get("ui_meta") if isinstance((hit or {}).get("ui_meta"), dict) else {}


def _refs_hit_meta(hit: dict | None) -> dict:
    return (hit or {}).get("meta") if isinstance((hit or {}).get("meta"), dict) else {}


def _refs_hit_reader_open(hit: dict | None) -> dict:
    ui_meta = _refs_hit_ui_meta(hit)
    reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
    if reader_open:
        return reader_open
    raw = (hit or {}).get("reader_open") if isinstance((hit or {}).get("reader_open"), dict) else {}
    return raw if isinstance(raw, dict) else {}


def _refs_norm_key_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower()).strip()


def _refs_hit_source_key(hit: dict | None) -> str:
    ui_meta = _refs_hit_ui_meta(hit)
    meta = _refs_hit_meta(hit)
    source = (
        str(ui_meta.get("source_path") or "").strip()
        or str(meta.get("source_path") or "").strip()
        or str(_refs_hit_reader_open(hit).get("sourcePath") or "").strip()
        or str(ui_meta.get("display_name") or "").strip()
    )
    return _refs_norm_key_text(source.replace("\\", "/"))


def _refs_hit_heading_key(hit: dict | None) -> str:
    ui_meta = _refs_hit_ui_meta(hit)
    meta = _refs_hit_meta(hit)
    reader_open = _refs_hit_reader_open(hit)
    heading = (
        str(ui_meta.get("heading_path") or "").strip()
        or str(ui_meta.get("section_label") or "").strip()
        or str(meta.get("ref_best_heading_path") or "").strip()
        or str(meta.get("heading_path") or "").strip()
        or str(reader_open.get("headingPath") or "").strip()
    )
    return _refs_norm_key_text(heading)


def _refs_hit_locate_key(hit: dict | None) -> str:
    reader_open = _refs_hit_reader_open(hit)
    ui_meta = _refs_hit_ui_meta(hit)
    primary = reader_open.get("primaryEvidence") if isinstance(reader_open.get("primaryEvidence"), dict) else {}
    primary_ui = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
    block_id = str(reader_open.get("blockId") or primary.get("block_id") or primary_ui.get("block_id") or "").strip()
    anchor_id = str(reader_open.get("anchorId") or primary.get("anchor_id") or primary_ui.get("anchor_id") or "").strip()
    anchor_kind = str(reader_open.get("anchorKind") or "").strip().lower()
    anchor_num = str(reader_open.get("anchorNumber") or "").strip()
    if block_id or anchor_id:
        return "loc:" + "|".join([block_id, anchor_id, anchor_kind, anchor_num])
    return ""


def _refs_hit_exact_locate_score(hit: dict | None) -> float:
    ui_meta = _refs_hit_ui_meta(hit)
    reader_open = _refs_hit_reader_open(hit)
    primary = reader_open.get("primaryEvidence") if isinstance(reader_open.get("primaryEvidence"), dict) else {}
    primary_ui = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
    score = 0.0
    if bool(reader_open.get("strictLocate")):
        score += 0.55
    if str(reader_open.get("blockId") or primary.get("block_id") or primary_ui.get("block_id") or "").strip():
        score += 0.30
    if str(reader_open.get("anchorId") or primary.get("anchor_id") or primary_ui.get("anchor_id") or "").strip():
        score += 0.20
    if str(reader_open.get("anchorKind") or "").strip() or _positive_int(reader_open.get("anchorNumber")) > 0:
        score += 0.10
    if bool(primary or primary_ui):
        score += 0.15
    return min(1.30, score)


def _refs_hit_polish_score(hit: dict | None) -> float:
    ui_meta = _refs_hit_ui_meta(hit)
    if not ui_meta:
        return 0.0
    status = str(ref_card_polish_status(ui_meta).get("polish_status") or "").strip().lower()
    if status == "full":
        return 0.60
    if status == "heuristic":
        return 0.20
    if status == "pending":
        return 0.05
    if status == "failed":
        return -0.20
    return 0.0


def _refs_hit_evidence_text(hit: dict | None) -> str:
    ui_meta = _refs_hit_ui_meta(hit)
    meta = _refs_hit_meta(hit)
    reader_open = _refs_hit_reader_open(hit)
    primary = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
    reader_primary = reader_open.get("primaryEvidence") if isinstance(reader_open.get("primaryEvidence"), dict) else {}
    parts: list[str] = []
    for value in (
        primary.get("highlight_snippet"),
        primary.get("snippet"),
        reader_primary.get("highlight_snippet"),
        reader_primary.get("snippet"),
        reader_open.get("highlightSnippet"),
        reader_open.get("snippet"),
        ui_meta.get("summary_line"),
        ui_meta.get("why_line"),
        (hit or {}).get("text"),
    ):
        text = str(value or "").strip()
        if text:
            parts.append(text)
    for key in ("ref_show_snippets", "ref_snippets", "ref_overview_snippets"):
        arr = meta.get(key)
        if isinstance(arr, list):
            for item in arr[:3]:
                text = str(item or "").strip()
                if text:
                    parts.append(text)
    return " ".join(parts)


def _refs_dedupe_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", str(text or "").lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "paper",
        "section",
        "method",
        "these",
        "those",
        "from",
        "into",
        "where",
        "which",
        "what",
        "how",
        "\u8fd9\u6761",
        "\u547d\u4e2d",
        "\u8bc1\u636e",
        "\u8bba\u6587",
        "\u7ae0\u8282",
        "\u65b9\u6cd5",
        "\u76f8\u5173",
        "\u53ef\u4ee5",
        "\u7528\u4e8e",
    }
    return {token for token in tokens if token not in stop}


def _refs_evidence_similarity(left: str, right: str) -> float:
    a = _refs_dedupe_tokens(left)
    b = _refs_dedupe_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _refs_evidence_fingerprint(text: str) -> str:
    tokens = sorted(_refs_dedupe_tokens(text))
    if not tokens:
        return ""
    return hashlib.sha1(" ".join(tokens[:32]).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _refs_hits_are_near_duplicates(left: dict, right: dict) -> bool:
    left_source = _refs_hit_source_key(left)
    right_source = _refs_hit_source_key(right)
    if not left_source or left_source != right_source:
        return False
    left_loc = _refs_hit_locate_key(left)
    right_loc = _refs_hit_locate_key(right)
    if left_loc and right_loc and left_loc == right_loc:
        return True
    left_heading = _refs_hit_heading_key(left)
    right_heading = _refs_hit_heading_key(right)
    if left_heading and right_heading and left_heading != right_heading:
        return False
    left_text = _refs_hit_evidence_text(left)
    right_text = _refs_hit_evidence_text(right)
    if left_heading and right_heading and (not left_text or not right_text):
        return True
    left_fp = _refs_evidence_fingerprint(left_text)
    right_fp = _refs_evidence_fingerprint(right_text)
    if left_fp and left_fp == right_fp:
        return True
    return _refs_evidence_similarity(left_text, right_text) >= 0.72


def _refs_hit_duplicate_rank(
    *,
    prompt: str,
    hit: dict,
    idx: int,
    focus_match_count: Callable[[str, dict], int],
    section_intent_score: Callable[[str, dict], float],
    display_score: Callable[[dict], float],
) -> tuple[float, float, float, float, float, float, int]:
    meta = _refs_hit_meta(hit)
    answer_source_boost = 1.0 if str((meta or {}).get("ref_display_reason") or "").strip().lower() == "answer_hit_top" else 0.0
    return (
        answer_source_boost,
        float(focus_match_count(prompt, hit)),
        float(section_intent_score(prompt, hit)),
        _refs_hit_exact_locate_score(hit),
        _refs_hit_polish_score(hit),
        display_score(hit),
        -int(idx),
    )


def _merge_refs_duplicate_into(keeper: dict, duplicate: dict) -> dict:
    hit = dict(keeper or {})
    ui = dict(_refs_hit_ui_meta(hit))
    ui["merged_duplicate_count"] = _positive_int(ui.get("merged_duplicate_count")) + 1
    headings = [
        str(item or "").strip()
        for item in list(ui.get("merged_duplicate_headings") or [])
        if str(item or "").strip()
    ]
    duplicate_heading = str(_refs_hit_ui_meta(duplicate).get("heading_path") or "").strip()
    if duplicate_heading and duplicate_heading not in headings:
        headings.append(duplicate_heading)
    if headings:
        ui["merged_duplicate_headings"] = headings[:6]
    hit["ui_meta"] = ui
    meta = dict(_refs_hit_meta(hit))
    meta["merged_duplicate_count"] = _positive_int(meta.get("merged_duplicate_count")) + 1
    hit["meta"] = meta
    return hit


def _dedupe_refs_hits_for_display(
    *,
    prompt: str,
    hits: list[dict],
    focus_match_count: Callable[[str, dict], int],
    section_intent_score: Callable[[str, dict], float],
    display_score: Callable[[dict], float],
) -> tuple[list[dict], int]:
    rows = [dict(hit) for hit in list(hits or []) if isinstance(hit, dict)]
    if len(rows) <= 1:
        return rows, 0
    kept: list[tuple[int, dict]] = []
    removed = 0
    for idx, hit in enumerate(rows):
        match_pos = -1
        for pos, (_keeper_idx, keeper) in enumerate(kept):
            if _refs_hits_are_near_duplicates(keeper, hit):
                match_pos = pos
                break
        if match_pos < 0:
            kept.append((idx, hit))
            continue
        keeper_idx, keeper = kept[match_pos]
        hit_rank = _refs_hit_duplicate_rank(
            prompt=prompt,
            hit=hit,
            idx=idx,
            focus_match_count=focus_match_count,
            section_intent_score=section_intent_score,
            display_score=display_score,
        )
        keeper_rank = _refs_hit_duplicate_rank(
            prompt=prompt,
            hit=keeper,
            idx=keeper_idx,
            focus_match_count=focus_match_count,
            section_intent_score=section_intent_score,
            display_score=display_score,
        )
        if hit_rank > keeper_rank:
            kept[match_pos] = (idx, _merge_refs_duplicate_into(hit, keeper))
        else:
            kept[match_pos] = (keeper_idx, _merge_refs_duplicate_into(keeper, hit))
        removed += 1
    return [hit for _idx, hit in kept], removed
