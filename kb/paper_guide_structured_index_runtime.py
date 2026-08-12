from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kb.citation_meta import extract_year_hint


_FIGURE_SCOPE_ALIASES = {
    "main": "main",
    "figure": "main",
    "extended": "extended_data",
    "extended_data": "extended_data",
    "extended-data": "extended_data",
    "supplement": "supplementary",
    "supplemental": "supplementary",
    "supplementary": "supplementary",
}


def normalize_figure_scope(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    return _FIGURE_SCOPE_ALIASES.get(raw, "")


def extract_figure_scope_from_text(text: str, *, default_main: bool = False) -> str:
    src = str(text or "")
    if not src:
        return "main" if default_main else ""
    if re.search(r"\bextended\s+data\s+fig(?:ure)?\.?\s*\d+\b|(?:扩展数据|扩展)\s*图\s*\d+", src, re.IGNORECASE):
        return "extended_data"
    if re.search(
        r"\b(?:supplementary|supplemental)\s+fig(?:ure)?\.?\s*S?\s*\d+\b|"
        r"\bfig(?:ure)?\.?\s*S\s*\d+\b|补充\s*图\s*\d+",
        src,
        re.IGNORECASE,
    ):
        return "supplementary"
    if re.search(r"\bfig(?:ure)?\.?\s*\d+\b|(?:第\s*\d+\s*张?图|图\s*\d+)", src, re.IGNORECASE):
        return "main"
    return "main" if default_main else ""


def figure_key_for_scope(scope: Any, figure_number: int) -> str:
    normalized = normalize_figure_scope(scope)
    try:
        number = int(figure_number or 0)
    except Exception:
        number = 0
    if not normalized or number <= 0:
        return ""
    return f"{normalized}:{number}"


def _figure_row_scope(row: dict) -> tuple[str, bool]:
    explicit_scope = normalize_figure_scope(row.get("figure_scope"))
    raw_key = str(row.get("figure_key") or "").strip().lower()
    key_scope = normalize_figure_scope(raw_key.split(":", 1)[0]) if ":" in raw_key else ""
    return explicit_scope or key_scope, bool(explicit_scope or raw_key)


def filter_figure_index_rows(
    entries: list[dict] | None,
    *,
    figure_number: int,
    figure_scope: str = "",
) -> list[dict]:
    """Filter same-number figures without mixing explicit semantic scopes.

    When a scope is requested, exact scoped rows win. Rows from legacy indices
    that have neither ``figure_scope`` nor ``figure_key`` are considered only
    when no exact scoped row exists. Explicit rows from another scope are never
    used as a fallback.
    """
    try:
        target_number = int(figure_number or 0)
    except Exception:
        target_number = 0
    if target_number <= 0:
        return []
    requested_scope = normalize_figure_scope(figure_scope)
    numbered: list[dict] = []
    exact: list[dict] = []
    legacy: list[dict] = []
    for raw in list(entries or []):
        if not isinstance(raw, dict):
            continue
        try:
            row_number = int(raw.get("paper_figure_number") or raw.get("figure_number") or raw.get("fig_no") or raw.get("number") or 0)
        except Exception:
            row_number = 0
        if row_number != target_number:
            continue
        row = dict(raw)
        numbered.append(row)
        row_scope, has_explicit_identity = _figure_row_scope(row)
        if requested_scope and row_scope == requested_scope:
            exact.append(row)
        elif requested_scope and not has_explicit_identity:
            legacy.append(row)
    if not requested_scope:
        return numbered
    return exact or legacy


def _load_paper_guide_index_payload(md_path: Path | str, file_name: str) -> dict[str, Any]:
    path = Path(str(md_path or "")).expanduser()
    index_path = path.parent / "assets" / str(file_name or "").strip()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_paper_guide_index_rows(md_path: Path | str, *, file_name: str, key: str) -> list[dict]:
    payload = _load_paper_guide_index_payload(md_path, file_name)
    rows = payload.get(key)
    if not isinstance(rows, list):
        return []
    return [dict(item) for item in rows if isinstance(item, dict)]


def load_paper_guide_anchor_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="anchor_index.json", key="anchors")


def load_paper_guide_equation_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="equation_index.json", key="equations")


def load_paper_guide_figure_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="figure_index.json", key="figures")


def load_paper_guide_table_index(md_path: Path | str) -> list[dict]:
    return _load_paper_guide_index_rows(md_path, file_name="table_index.json", key="tables")


def load_paper_guide_reference_index(md_path: Path | str) -> list[dict]:
    rows = _load_paper_guide_index_rows(
        md_path,
        file_name="reference_index.json",
        key="references",
    )
    for row in rows:
        raw = str(row.get("text") or row.get("raw") or "").strip()
        source_year = extract_year_hint(raw)
        indexed_year = str(row.get("year") or "").strip()
        if source_year and source_year != indexed_year:
            # arXiv identifiers such as ``arXiv:2004.04906`` contain a
            # year-looking prefix before the actual publication year at the
            # end of the bibliography entry. The shared citation parser uses
            # the final year; normalize older persisted structured indices at
            # read time so validation and cards expose the source-accurate
            # value without requiring a corpus rebuild.
            row["year"] = source_year
            row["year_repaired_from_text"] = True
    return rows
