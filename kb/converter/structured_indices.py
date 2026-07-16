from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kb.citation_meta import extract_first_author_family_hint, extract_first_doi, extract_year_hint
from kb.evidence_text import looks_author_list_context, looks_bibliography_entry_context
from kb.inpaper_citation_grounding import parse_ref_num_set
from kb.reference_index import (
    _fallback_title_from_raw_reference,
    build_reference_catalog_from_md,
    extract_references_map_from_md,
    load_reference_catalog_for_md,
    reference_catalog_to_map,
)
from kb.source_blocks import build_source_blocks, doc_id_for_path, normalize_inline_markdown
from kb.table_index import build_table_index_payload

STRUCTURED_INDEX_VERSION = 5
_INDEX_VERSION = STRUCTURED_INDEX_VERSION
_EQUATION_CONTEXT_KINDS = {"paragraph", "list_item", "blockquote", "table"}
_INLINE_REF_RE = re.compile(
    r"(?<!\[)\[(\d{1,4}(?:\s*(?:[-\u2013\u2014\u2212,;\uff0c\u3001\uff1b])\s*\d{1,4})*)\](?!\])"
)
_REFERENCE_HEADING_RE = re.compile(
    r"(?:^|/)\s*(?:\d{1,3}\s*[.)]?\s*)?(?:references?|bibliography|literature\s+cited|"
    r"works\s+cited|cited\s+references|reference\s+list|references\s+and\s+notes|"
    r"\u53c2\u8003\u6587\u732e|\u5f15\u7528\u6587\u732e)\s*[:\uff1a]?\s*$",
    re.IGNORECASE,
)
_REFERENCE_ENTRY_RE = re.compile(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[\.)])\s+\S+")
_MENTION_BOUNDARY_CHARS = ".!?;\n\u3002\uff01\uff1f\uff1b"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_text(text: Any, *, limit: int = 0) -> str:
    value = normalize_inline_markdown(str(text or ""))
    if limit > 0:
        return value[:limit]
    return value


def _collapse_equation_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_meta_text(value: Any, *, limit: int = 0) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n.;:,")
    if limit > 0:
        return text[:limit]
    return text


def _extract_page_from_asset_name(*names: Any) -> int:
    for raw in names:
        name = Path(str(raw or "").strip()).name
        if not name:
            continue
        match = re.search(r"(?:^|[_-])page[_-]?(\d+)(?:[_-]|$)", name, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            page = int(match.group(1) or 0)
        except Exception:
            page = 0
        if page > 0:
            return page
    return 0


def _block_page(block: dict[str, Any] | None) -> int:
    if not isinstance(block, dict):
        return 0
    for key in ("page_start", "page", "page_end"):
        try:
            value = int(block.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _extract_caption_panel_clauses(text: Any) -> list[dict[str, str]]:
    src = re.sub(r"\s+", " ", str(text or "").strip())
    if not src:
        return []
    clauses: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in re.finditer(
        r"(?:^|(?<=[.;:]))\s*([A-Ga-g])\s+(?=[A-Z])(.+?)(?=(?:[.;:]\s*[A-Ga-g]\s+(?=[A-Z]))|$)",
        src,
        flags=re.DOTALL,
    ):
        letter = str(match.group(1) or "").strip().lower()
        body = _clean_meta_text(match.group(2) or "", limit=400)
        if (not letter) or (not body):
            continue
        clause = f"{letter} {body}".strip()
        key = (letter, clause.lower())
        if key in seen:
            continue
        seen.add(key)
        clauses.append({"panel_letter": letter, "clause": clause})
    return clauses


def _collect_caption_continuation(
    *,
    blocks: list[dict[str, Any]],
    figure_block: dict[str, Any] | None = None,
    figure_number: int = 0,
) -> str:
    figure_block_id = str((figure_block or {}).get("block_id") or "").strip()
    figure_key = str((figure_block or {}).get("figure_key") or "").strip().lower()
    ordered: list[tuple[int, str]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("figure_role") or "").strip().lower() != "caption_continuation":
            continue
        try:
            block_fig_num = int(block.get("paper_figure_number") or 0)
        except Exception:
            block_fig_num = 0
        linked_figure_block_id = str(block.get("linked_figure_block_id") or "").strip()
        block_key = str(block.get("figure_key") or "").strip().lower()
        if figure_block_id and linked_figure_block_id == figure_block_id:
            pass
        elif figure_key and block_key:
            if block_key != figure_key:
                continue
        elif figure_number > 0 and block_fig_num == figure_number:
            pass
        else:
            continue
        text = _clean_text(block.get("raw_text") or block.get("text") or "", limit=1200)
        if not text:
            continue
        ordered.append((int(block.get("order_index") or 0), text))
    ordered.sort(key=lambda item: int(item[0]))
    return " ".join(text for _idx, text in ordered).strip()


def _fallback_reference_authors(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\[(\d{1,4})\]\s*", "", text)
    text = re.sub(r"^(\d{1,4})\.\s*", "", text)
    title_hint = _clean_meta_text(_fallback_title_from_raw_reference(text), limit=320)
    if title_hint:
        low = text.lower()
        idx = low.find(title_hint.lower())
        if idx > 0:
            prefix = _clean_meta_text(text[:idx], limit=240)
            if prefix and len(prefix.split()) <= 12:
                return prefix
    segments = [seg.strip(" .;:,") for seg in re.split(r"\.\s+", text) if seg.strip(" .;:,")]
    author_parts: list[str] = []
    for seg in segments:
        low = seg.lower()
        if re.search(r"\b(19\d{2}|20\d{2})\b", seg):
            break
        if extract_first_doi(seg):
            break
        if len(seg.split()) >= 10:
            break
        if (
            "," in seg
            or " et al" in low
            or " and " in low
            or "&" in seg
            or re.search(r"\b[A-Z]\.", seg)
        ):
            author_parts.append(seg)
            if len(author_parts) >= 2:
                break
            continue
        if author_parts:
            break
    if author_parts:
        return _clean_meta_text(". ".join(author_parts), limit=240)
    family = extract_first_author_family_hint(text)
    return _clean_meta_text(family, limit=120)


def _heading_is_references(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and _REFERENCE_HEADING_RE.search(text))


def _is_reference_entry_text(value: Any) -> bool:
    text = _clean_text(value, limit=1200).strip()
    return bool(text and _REFERENCE_ENTRY_RE.match(text))


def _normalize_ref_marker_spec(value: Any) -> str:
    return re.sub(r"[\uff0c\u3001;\uff1b]", ",", str(value or "").strip())


def _looks_like_inline_reference_marker(text: str, match: re.Match[str]) -> bool:
    source = str(text or "")
    try:
        start = int(match.start())
        end = int(match.end())
    except Exception:
        return True
    before = source[start - 1] if start > 0 else ""
    after = source[end] if end < len(source) else ""
    # Converted OCR/Markdown can split notation like s2ISM into "s [2]ISM".
    # That is not a bibliography citation and should not seed SystemB cards.
    if after and re.match(r"[A-Za-z0-9_]", after):
        return False
    if before and re.match(r"[A-Za-z0-9_]", before) and not before.isspace():
        return False
    return True


def _mention_context(text: Any, start: int, end: int, *, max_chars: int = 520) -> str:
    source = str(text or "")
    if not source:
        return ""
    limit = max(180, int(max_chars or 520))
    start = max(0, min(len(source), int(start)))
    end = max(start, min(len(source), int(end)))

    left = -1
    for ch in _MENTION_BOUNDARY_CHARS:
        left = max(left, source.rfind(ch, 0, start))
    left = 0 if left < 0 else left + 1

    right_candidates = [source.find(ch, end) for ch in _MENTION_BOUNDARY_CHARS]
    right_candidates = [idx for idx in right_candidates if idx >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(source)

    if right - left > limit:
        center = (start + end) // 2
        half = limit // 2
        left = max(0, center - half)
        right = min(len(source), left + limit)
        left = max(0, min(left, max(0, right - limit)))

    context = re.sub(r"\s+", " ", source[left:right]).strip()
    if not context:
        return ""
    if left > 0:
        context = "..." + context.lstrip()
    if right < len(source):
        context = context.rstrip() + "..."
    return context[:limit].strip()


def _collect_reference_mentions(blocks: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str, int, str]] = set()
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").strip().lower()
        if kind not in {"paragraph", "list_item", "blockquote", "table"}:
            continue
        heading_path = str(block.get("heading_path") or "").strip()
        if _heading_is_references(heading_path):
            continue
        text = _clean_text(block.get("raw_text") or block.get("text") or "", limit=5000)
        if len(text) < 12:
            continue
        if _is_reference_entry_text(text):
            continue
        if looks_author_list_context(text) or looks_bibliography_entry_context(text):
            continue
        for match in _INLINE_REF_RE.finditer(text):
            if not _looks_like_inline_reference_marker(text, match):
                continue
            nums = parse_ref_num_set(_normalize_ref_marker_spec(match.group(1)), max_items=64)
            if not nums:
                continue
            context = _mention_context(text, int(match.start()), int(match.end()), max_chars=520)
            if not context:
                continue
            if looks_author_list_context(context) or looks_bibliography_entry_context(context):
                continue
            line_start = int(block.get("line_start") or 0)
            line_end = int(block.get("line_end") or 0)
            page = _block_page(block)
            location_parts = [part for part in (heading_path, f"p. {page}" if page > 0 else "") if part]
            for num in nums:
                key = (int(num), context.lower(), line_start, heading_path)
                if key in seen:
                    continue
                seen.add(key)
                rec: dict[str, Any] = {
                    "ref_num": int(num),
                    "citation_context": context,
                    "heading_path": heading_path,
                    "location_label": " / ".join(location_parts),
                    "block_id": str(block.get("block_id") or "").strip(),
                    "anchor_id": str(block.get("anchor_id") or "").strip(),
                    "anchor_kind": kind,
                    "line_start": line_start,
                    "line_end": line_end,
                }
                if page > 0:
                    rec["page_start"] = int(page)
                    rec["page_end"] = int(page)
                out.setdefault(int(num), []).append(rec)
    for num, rows in list(out.items()):
        rows.sort(key=lambda item: (int(item.get("line_start") or 0), int(item.get("line_end") or 0)))
        out[num] = rows[:8]
    return out


def _reference_meta_text(meta: dict[str, Any], *keys: str, limit: int = 0) -> str:
    for key in keys:
        value = _clean_meta_text(meta.get(key), limit=limit)
        if value:
            return value
    return ""


def _load_existing_figure_rows(assets_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(assets_dir / "figure_index.json")
    figures = payload.get("figures")
    if not isinstance(figures, list):
        return []
    return [dict(item) for item in figures if isinstance(item, dict)]


def _build_figure_meta_by_asset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        rec = dict(row)
        raw_name = Path(str(rec.get("asset_name_raw") or rec.get("asset_name") or "").strip()).name
        alias_name = Path(str(rec.get("asset_name_alias") or "").strip()).name
        if raw_name:
            out[raw_name] = rec
        if alias_name:
            out[alias_name] = rec
    return out


def _build_anchor_index_payload(md_path: Path, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        rec: dict[str, Any] = {
            "anchor_id": str(block.get("anchor_id") or "").strip(),
            "block_id": str(block.get("block_id") or "").strip(),
            "kind": str(block.get("kind") or "").strip(),
            "heading_path": str(block.get("heading_path") or "").strip(),
            "order_index": int(block.get("order_index") or 0),
            "line_start": int(block.get("line_start") or 0),
            "line_end": int(block.get("line_end") or 0),
            "text": _clean_text(block.get("raw_text") or block.get("text") or "", limit=1200),
        }
        page = _block_page(block)
        if page > 0:
            rec["page_start"] = page
            rec["page_end"] = page
        for key in (
            "number",
            "figure_id",
            "figure_ident",
            "figure_scope",
            "figure_key",
            "figure_role",
            "asset_name",
            "asset_name_alias",
        ):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                rec[key] = value.strip()
            elif isinstance(value, int) and value > 0:
                rec[key] = value
        paper_figure_number = int(block.get("paper_figure_number") or 0)
        if paper_figure_number > 0:
            rec["paper_figure_number"] = paper_figure_number
        if rec["anchor_id"] and rec["block_id"]:
            anchors.append(rec)
    return {
        "version": _INDEX_VERSION,
        "doc_id": doc_id_for_path(md_path),
        "doc_path": str(md_path.resolve(strict=False)),
        "anchor_count": int(len(anchors)),
        "anchors": anchors,
    }


def _nearest_equation_context(blocks: list[dict[str, Any]], index: int, *, step: int) -> str:
    cursor = int(index) + int(step)
    while 0 <= cursor < len(blocks):
        block = blocks[cursor]
        if not isinstance(block, dict):
            cursor += step
            continue
        kind = str(block.get("kind") or "").strip().lower()
        text = _clean_text(block.get("raw_text") or block.get("text") or "", limit=1000)
        if kind in _EQUATION_CONTEXT_KINDS and text:
            return text
        cursor += step
    return ""


_CONVERSION_RETRY_MARKER_RE = re.compile(r"<!--\s*kb:conversion_retry\s+(.+?)\s*-->", flags=re.I)
_CONVERSION_MARKER_ATTR_RE = re.compile(r"([A-Za-z_][\w-]*)=(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))")


def _conversion_equation_fallbacks(md_text: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallbacks: list[dict[str, Any]] = []
    for match in _CONVERSION_RETRY_MARKER_RE.finditer(str(md_text or "")):
        attrs = {
            item.group(1).lower(): next((value for value in item.groups()[1:] if value is not None), "")
            for item in _CONVERSION_MARKER_ATTR_RE.finditer(match.group(1))
        }
        if str(attrs.get("kind") or "").lower() != "equation":
            continue
        line_number = str(md_text or "").count("\n", 0, match.start()) + 1
        try:
            page = int(attrs.get("page") or 0)
        except Exception:
            page = 0
        try:
            equation_number = int(attrs.get("number") or 0)
        except Exception:
            equation_number = 0

        before: tuple[int, dict[str, Any]] | None = None
        after: tuple[int, dict[str, Any]] | None = None
        nearest: tuple[int, dict[str, Any]] | None = None
        heading_path = ""
        for block in blocks:
            if not isinstance(block, dict):
                continue
            line_start = int(block.get("line_start") or 0)
            line_end = int(block.get("line_end") or line_start)
            distance_to_marker = min(abs(line_number - line_start), abs(line_number - line_end))
            if nearest is None or distance_to_marker < nearest[0]:
                nearest = (distance_to_marker, block)
            kind = str(block.get("kind") or "").lower()
            if kind not in _EQUATION_CONTEXT_KINDS:
                continue
            if line_end <= line_number:
                distance = line_number - line_end
                if before is None or distance < before[0]:
                    before = (distance, block)
            elif line_start >= line_number:
                distance = line_start - line_number
                if after is None or distance < after[0]:
                    after = (distance, block)
        if nearest:
            heading_path = str(nearest[1].get("heading_path") or "").strip()
        if not heading_path and before:
            heading_path = str(before[1].get("heading_path") or "").strip()
        if not heading_path and after:
            heading_path = str(after[1].get("heading_path") or "").strip()

        def _context(row: tuple[int, dict[str, Any]] | None) -> str:
            if not row:
                return ""
            block = row[1]
            if str(block.get("kind") or "").lower() not in _EQUATION_CONTEXT_KINDS:
                return ""
            return _clean_text(block.get("raw_text") or block.get("text") or "", limit=1000)

        label = f"Equation ({equation_number})" if equation_number > 0 else "Equation"
        locate_anchor = f"{label} is preserved as a source image"
        if page > 0:
            locate_anchor += f" on page {page}"
        locate_anchor += "; use the source image for exact notation."
        fallbacks.append({
            "equation_number": equation_number,
            "equation_markdown": "",
            "normalized_tex": "",
            "evidence_status": "image_only",
            "asset_name": str(attrs.get("asset") or "").strip(),
            "locate_anchor": locate_anchor,
            "context_before": _context(before),
            "context_after": _context(after),
            "block_id": "",
            "anchor_id": "",
            "heading_path": heading_path,
            "line_start": line_number,
            "line_end": line_number,
            **({"page_start": page, "page_end": page} if page > 0 else {}),
        })
    return fallbacks


def _build_equation_index_payload(md_path: Path, blocks: list[dict[str, Any]], md_text: str = "") -> dict[str, Any]:
    equations: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if str(block.get("kind") or "").strip().lower() != "equation":
            continue
        raw_equation = str(block.get("raw_text") or block.get("text") or "").strip()
        if not raw_equation:
            continue
        page = _block_page(block)
        equations.append(
            {
                "equation_number": int(block.get("number") or 0),
                "equation_markdown": raw_equation,
                "normalized_tex": _collapse_equation_text(raw_equation),
                "context_before": _nearest_equation_context(blocks, idx, step=-1),
                "context_after": _nearest_equation_context(blocks, idx, step=1),
                "block_id": str(block.get("block_id") or "").strip(),
                "anchor_id": str(block.get("anchor_id") or "").strip(),
                "heading_path": str(block.get("heading_path") or "").strip(),
                "line_start": int(block.get("line_start") or 0),
                "line_end": int(block.get("line_end") or 0),
                **({"page_start": page, "page_end": page} if page > 0 else {}),
            }
        )
    equations.extend(_conversion_equation_fallbacks(md_text, blocks))
    return {
        "version": _INDEX_VERSION,
        "doc_id": doc_id_for_path(md_path),
        "doc_path": str(md_path.resolve(strict=False)),
        "equation_count": int(len(equations)),
        "equations": equations,
    }


def _row_figure_number(row: dict[str, Any]) -> int:
    try:
        return int(row.get("paper_figure_number") or row.get("fig_no") or 0)
    except Exception:
        return 0


def _block_figure_number(block: dict[str, Any]) -> int:
    try:
        return int(block.get("paper_figure_number") or block.get("number") or 0)
    except Exception:
        return 0


def _figure_key(value: dict[str, Any]) -> str:
    key = str(value.get("figure_key") or "").strip().lower()
    if key:
        return key
    number = _block_figure_number(value)
    if number <= 0:
        number = _row_figure_number(value)
    scope = str(value.get("figure_scope") or "main").strip().lower() or "main"
    return f"{scope}:{number}" if number > 0 else ""


def _score_figure_row_match(row: dict[str, Any], block: dict[str, Any]) -> int:
    score = 0
    explicit_row_key = str(row.get("figure_key") or "").strip().lower()
    explicit_row_scope = str(row.get("figure_scope") or "").strip().lower()
    row_key = explicit_row_key
    if not row_key and explicit_row_scope:
        row_number = _row_figure_number(row)
        if row_number > 0:
            row_key = f"{explicit_row_scope}:{row_number}"
    block_key = _figure_key(block)
    if row_key and block_key:
        if row_key != block_key:
            return -100
        score += 14
    row_figure_id = str(row.get("figure_id") or "").strip()
    block_figure_id = str(block.get("figure_id") or "").strip()
    if row_figure_id and block_figure_id and row_figure_id == block_figure_id:
        score += 10

    row_names = {
        Path(str(row.get("asset_name_raw") or row.get("asset_name") or "").strip()).name,
        Path(str(row.get("asset_name_alias") or "").strip()).name,
    }
    block_names = {
        Path(str(block.get("asset_name") or "").strip()).name,
        Path(str(block.get("asset_name_alias") or "").strip()).name,
    }
    row_names.discard("")
    block_names.discard("")
    if row_names & block_names:
        score += 8

    row_number = _row_figure_number(row)
    block_number = _block_figure_number(block)
    if row_number > 0 and block_number > 0 and row_number == block_number:
        score += 6
    return score


def _find_best_figure_block(row: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("kind") or "").strip().lower() != "figure":
            continue
        score = _score_figure_row_match(row, block)
        if score <= 0:
            continue
        ranked.append((score, -int(block.get("order_index") or 0), block))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def _find_best_caption_block(
    row: dict[str, Any],
    blocks: list[dict[str, Any]],
    *,
    figure_block: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    figure_block_id = str((figure_block or {}).get("block_id") or "").strip()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        figure_role = str(block.get("figure_role") or "").strip().lower()
        if figure_role not in {"caption", "caption_continuation"}:
            continue
        score = _score_figure_row_match(row, block)
        if figure_block_id and str(block.get("linked_figure_block_id") or "").strip() == figure_block_id:
            score += 12
        if score <= 0:
            continue
        ranked.append((score, -int(block.get("order_index") or 0), block))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def _build_figure_index_payload(
    md_path: Path,
    *,
    blocks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    # If we have explicit figure metadata rows (typically produced by the converter),
    # use them for high-quality matching. Otherwise, fall back to deriving a minimal
    # index from parsed source blocks so older guides still get a usable figure index.
    if not rows:
        figures: list[dict[str, Any]] = []
        figure_blocks = [
            block
            for block in blocks
            if isinstance(block, dict) and str(block.get("kind") or "").strip().lower() == "figure"
        ]
        for fig in figure_blocks:
            try:
                figure_number = int(fig.get("paper_figure_number") or fig.get("number") or 0)
            except Exception:
                figure_number = 0
            if figure_number <= 0:
                continue
            figure_scope = str(fig.get("figure_scope") or "main").strip().lower() or "main"
            figure_key = str(fig.get("figure_key") or f"{figure_scope}:{figure_number}").strip()
            figure_block_id = str(fig.get("block_id") or "").strip()
            figure_anchor_id = str(fig.get("anchor_id") or "").strip()
            heading_path = str(fig.get("heading_path") or "").strip()
            # Find a caption block that matches the same figure number if present.
            caption_block = None
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                role = str(block.get("figure_role") or "").strip().lower()
                if role not in {"caption", "caption_continuation"}:
                    continue
                try:
                    cap_num = int(block.get("paper_figure_number") or 0)
                except Exception:
                    cap_num = 0
                cap_key = _figure_key(block)
                if (figure_key and cap_key and cap_key != figure_key) or cap_num != figure_number:
                    continue
                caption_block = block
                break
            caption_block_id = str((caption_block or {}).get("block_id") or "").strip()
            caption_anchor_id = str((caption_block or {}).get("anchor_id") or "").strip()
            caption_text = _clean_text((caption_block or {}).get("raw_text") or (caption_block or {}).get("text") or "", limit=1200)
            asset_name = str(fig.get("asset_name") or "").strip()
            asset_name_alias = str(fig.get("asset_name_alias") or "").strip()
            caption_continuation = _collect_caption_continuation(
                blocks=blocks,
                figure_block=fig,
                figure_number=figure_number,
            )
            combined_caption = " ".join(part for part in [caption_text, caption_continuation] if part).strip()
            locate_anchor = caption_text or _clean_text(fig.get("raw_text") or fig.get("text") or "", limit=600)
            rec = {
                "paper_figure_number": int(figure_number),
                "figure_scope": figure_scope,
                "figure_key": figure_key,
                "heading_path": heading_path,
                "anchor_id": caption_anchor_id or figure_anchor_id,
                "figure_block_id": figure_block_id,
                "caption_block_id": caption_block_id,
                "caption_anchor_id": caption_anchor_id,
                "block_ids": [value for value in [figure_block_id, caption_block_id] if value],
                "caption": caption_text,
                "locate_anchor": locate_anchor,
            }
            if asset_name:
                rec["asset_name"] = asset_name
            if asset_name_alias:
                rec["asset_name_alias"] = asset_name_alias
            page = (
                _extract_page_from_asset_name(asset_name, asset_name_alias)
                or _block_page(caption_block)
                or _block_page(fig)
            )
            if page > 0:
                rec["page"] = page
            if caption_continuation:
                rec["caption_continuation"] = caption_continuation
            panel_clauses = _extract_caption_panel_clauses(combined_caption)
            if panel_clauses:
                rec["panel_clauses"] = panel_clauses
            figures.append(rec)
        # Always return a payload (even when empty) so existing DB guides have a
        # stable contract: figure_index.json exists and can be cached/checked.
        return {
            "version": _INDEX_VERSION,
            "doc_id": doc_id_for_path(md_path),
            "doc_path": str(md_path.resolve(strict=False)),
            "figures": figures,
        }
    figures: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        figure_block = _find_best_figure_block(rec, blocks)
        caption_block = _find_best_caption_block(rec, blocks, figure_block=figure_block)
        matched_identity = figure_block or caption_block or {}
        matched_number = _block_figure_number(matched_identity)
        if _row_figure_number(rec) <= 0 and matched_number > 0:
            rec["paper_figure_number"] = matched_number
        matched_scope = str(matched_identity.get("figure_scope") or "").strip().lower()
        matched_key = _figure_key(matched_identity)
        if matched_scope:
            rec["figure_scope"] = matched_scope
        if matched_key:
            rec["figure_key"] = matched_key
        if not _clean_text(rec.get("caption") or "", limit=1200) and caption_block:
            rec["caption"] = _clean_text(
                caption_block.get("raw_text") or caption_block.get("text") or "",
                limit=1200,
            )
        heading_path = (
            str((figure_block or {}).get("heading_path") or "").strip()
            or str((caption_block or {}).get("heading_path") or "").strip()
            or str(rec.get("heading_path") or "").strip()
        )
        anchor_id = (
            str((figure_block or {}).get("anchor_id") or "").strip()
            or str((caption_block or {}).get("anchor_id") or "").strip()
            or str(rec.get("anchor_id") or "").strip()
        )
        figure_block_id = str((figure_block or {}).get("block_id") or "").strip() or str(rec.get("figure_block_id") or "").strip()
        caption_block_id = str((caption_block or {}).get("block_id") or "").strip() or str(rec.get("caption_block_id") or "").strip()
        block_ids = [value for value in [figure_block_id, caption_block_id] if value]
        if heading_path:
            rec["heading_path"] = heading_path
        if anchor_id:
            rec["anchor_id"] = anchor_id
        if figure_block_id:
            rec["figure_block_id"] = figure_block_id
        if caption_block_id:
            rec["caption_block_id"] = caption_block_id
        caption_anchor_id = str((caption_block or {}).get("anchor_id") or "").strip()
        if caption_anchor_id:
            rec["caption_anchor_id"] = caption_anchor_id
        if block_ids:
            rec["block_ids"] = block_ids
        page = 0
        try:
            page = int(rec.get("page") or 0)
        except Exception:
            page = 0
        if page <= 0:
            page = _extract_page_from_asset_name(
                rec.get("asset_name"),
                rec.get("asset_name_raw"),
                rec.get("asset_name_alias"),
                (figure_block or {}).get("asset_name"),
                (figure_block or {}).get("asset_name_alias"),
            )
        if page <= 0:
            page = _block_page(caption_block) or _block_page(figure_block)
        if page > 0:
            rec["page"] = page
        caption_continuation = _clean_text(rec.get("caption_continuation") or "", limit=1200) or _collect_caption_continuation(
            blocks=blocks,
            figure_block=figure_block,
            figure_number=_row_figure_number(rec),
        )
        if caption_continuation:
            rec["caption_continuation"] = caption_continuation
        locate_anchor = (
            _clean_text(rec.get("caption") or "", limit=600)
            or _clean_text((caption_block or {}).get("raw_text") or (caption_block or {}).get("text") or "", limit=600)
            or _clean_text((figure_block or {}).get("raw_text") or (figure_block or {}).get("text") or "", limit=600)
        )
        if locate_anchor:
            rec["locate_anchor"] = locate_anchor
        panel_clauses = _extract_caption_panel_clauses(
            " ".join(
                part
                for part in [
                    rec.get("caption"),
                    rec.get("caption_continuation"),
                ]
                if str(part or "").strip()
            )
        )
        if panel_clauses:
            rec["panel_clauses"] = panel_clauses
        figures.append(rec)
    return {
        "version": _INDEX_VERSION,
        "doc_id": doc_id_for_path(md_path),
        "doc_path": str(md_path.resolve(strict=False)),
        "figures": figures,
    }


def _reference_map_missing_numbers(ref_map: dict[int, str]) -> list[int]:
    refs: set[int] = set()
    for raw_key, raw_value in (ref_map if isinstance(ref_map, dict) else {}).items():
        try:
            n = int(raw_key)
        except Exception:
            continue
        if n > 0 and str(raw_value or "").strip():
            refs.add(n)
    if len(refs) < 8:
        return []
    first = min(refs)
    last = max(refs)
    if first != 1 or last < 8:
        return []
    return sorted(set(range(first, last + 1)) - refs)


def _reference_catalog_stale_for_markdown(catalog_map: dict[int, str], md_ref_map: dict[int, str]) -> bool:
    if not md_ref_map:
        return False
    if not catalog_map:
        return True
    catalog_count = len(catalog_map)
    md_count = len(md_ref_map)
    if md_count >= catalog_count + 3:
        return True
    catalog_max = max(catalog_map.keys(), default=0)
    md_max = max(md_ref_map.keys(), default=0)
    if md_max > catalog_max and md_count >= catalog_count:
        return True
    catalog_missing = _reference_map_missing_numbers(catalog_map)
    md_missing = _reference_map_missing_numbers(md_ref_map)
    return bool(catalog_missing and len(md_missing) < len(catalog_missing))


def _build_reference_index_payload(
    md_path: Path,
    md_text: str,
    *,
    blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    catalog = load_reference_catalog_for_md(md_path)
    ref_map = reference_catalog_to_map(catalog)
    md_ref_map = extract_references_map_from_md(md_text)
    if _reference_catalog_stale_for_markdown(ref_map, md_ref_map):
        catalog = build_reference_catalog_from_md(
            md_text,
            source_path=str(md_path.resolve(strict=False)),
            source_name=md_path.name,
        )
        ref_map = reference_catalog_to_map(catalog)
    refs_by_num: dict[int, dict[str, Any]] = {}
    for item in list(catalog.get("refs") or []):
        if not isinstance(item, dict):
            continue
        try:
            ref_num = int(item.get("reference_number") or 0)
        except Exception:
            ref_num = 0
        if ref_num > 0:
            refs_by_num[ref_num] = dict(item)

    mention_map = _collect_reference_mentions(list(blocks or []))
    mention_total = sum(len(rows) for rows in mention_map.values())
    references: list[dict[str, Any]] = []
    for ref_num in sorted(ref_map.keys()):
        raw = str(ref_map.get(ref_num) or "").strip()
        if not raw:
            continue
        meta = refs_by_num.get(int(ref_num)) or {}
        title = _reference_meta_text(meta, "title", limit=320) or _clean_meta_text(_fallback_title_from_raw_reference(raw), limit=320)
        authors = _reference_meta_text(meta, "authors", "author", limit=240) or _fallback_reference_authors(raw)
        venue = _reference_meta_text(meta, "venue", "journal", "journal_title", "container-title", limit=240)
        year = _reference_meta_text(meta, "year", limit=12) or extract_year_hint(raw)
        doi = _reference_meta_text(meta, "doi", "DOI", limit=160) or extract_first_doi(raw)
        rec: dict[str, Any] = {
            "ref_num": int(ref_num),
            "reference_entry_id": str(meta.get("reference_entry_id") or f"ref_{int(ref_num):04d}"),
            "text": raw,
            "doi": doi,
            "year": year,
            "parse_confidence": float(meta.get("parse_confidence") or 0.0),
        }
        if title:
            rec["title"] = title
        if authors:
            rec["authors"] = authors
        if venue:
            rec["venue"] = venue
        for field in ("volume", "issue", "pages", "match_method"):
            value = _reference_meta_text(meta, field, limit=120)
            if value:
                rec[field] = value
        if "crossref_ok" in meta:
            rec["crossref_ok"] = bool(meta.get("crossref_ok"))
        mentions = [dict(item) for item in mention_map.get(int(ref_num), []) if isinstance(item, dict)]
        if mentions:
            rec["mention_count"] = int(len(mentions))
            rec["citation_mentions"] = mentions
            first = mentions[0]
            first_context = _clean_text(first.get("citation_context"), limit=520)
            first_location = _clean_meta_text(first.get("location_label"), limit=240)
            if first_context:
                rec["first_citation_context"] = first_context
            if first_location:
                rec["first_citation_location"] = first_location
        references.append(rec)
    return {
        "version": _INDEX_VERSION,
        "doc_id": doc_id_for_path(md_path),
        "doc_path": str(md_path.resolve(strict=False)),
        "ref_count": int(len(references)),
        "citation_mention_count": int(mention_total),
        "tail_continuity_status": str(catalog.get("tail_continuity_status") or "").strip(),
        "missing_numbers": list(catalog.get("missing_numbers") or []),
        "references": references,
    }


def rebuild_structured_indices_for_markdown(
    md_path: Path | str,
    *,
    md_text: str | None = None,
    assets_dir: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(str(md_path or "")).expanduser()
    if md_text is None:
        try:
            md_text = path.read_text(encoding="utf-8")
        except Exception:
            md_text = ""
    md_text = str(md_text or "")

    resolved_assets_dir = Path(str(assets_dir or (path.parent / "assets"))).expanduser()
    resolved_assets_dir.mkdir(parents=True, exist_ok=True)

    figure_rows = _load_existing_figure_rows(resolved_assets_dir)
    blocks = build_source_blocks(
        md_text,
        doc_id=doc_id_for_path(path),
        figure_meta_by_asset=_build_figure_meta_by_asset(figure_rows),
    )

    anchor_payload = _build_anchor_index_payload(path, blocks)
    equation_payload = _build_equation_index_payload(path, blocks, md_text)
    reference_payload = _build_reference_index_payload(path, md_text, blocks=blocks)
    figure_payload = _build_figure_index_payload(path, blocks=blocks, rows=figure_rows)
    table_payload = build_table_index_payload(path, blocks, version=_INDEX_VERSION)

    _write_json(resolved_assets_dir / "anchor_index.json", anchor_payload)
    _write_json(resolved_assets_dir / "equation_index.json", equation_payload)
    _write_json(resolved_assets_dir / "reference_index.json", reference_payload)
    _write_json(resolved_assets_dir / "table_index.json", table_payload)
    if figure_payload is None:
        figure_payload = {
            "version": _INDEX_VERSION,
            "doc_id": doc_id_for_path(path),
            "doc_path": str(path.resolve(strict=False)),
            "figures": [],
        }
    _write_json(resolved_assets_dir / "figure_index.json", figure_payload)

    return {
        "anchor_index": anchor_payload,
        "equation_index": equation_payload,
        "reference_index": reference_payload,
        "figure_index": figure_payload,
        "table_index": table_payload,
    }
