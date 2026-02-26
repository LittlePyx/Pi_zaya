from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from kb.pdf_tools import abbreviate_venue, _safe_base_len_for_paths

KB_DISPLAY_FULL_NAME_KEY = "_kb_display_full_name"
KB_STORAGE_FILENAME_KEY = "_kb_storage_filename"
KB_DOC_ID_KEY = "_kb_doc_id"
KB_NAME_VENUE_KEY = "_kb_name_venue"
KB_NAME_YEAR_KEY = "_kb_name_year"
KB_NAME_TITLE_KEY = "_kb_name_title"


def sanitize_filename_component(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r'[<>:"/\\\\|?*]+', "-", s)
    s = s.replace("\u0000", "").strip()
    return s.strip(" .-_")


def _trim_left_on_boundary(text: str, budget: int) -> str:
    part = text[: max(0, int(budget))]
    if len(part) <= 10:
        return part.rstrip(" .-_")
    m = re.search(r"[- _][^- _]{1,16}$", part)
    if m and m.start() >= max(8, len(part) // 2):
        part = part[: m.start()]
    return part.rstrip(" .-_")


def _trim_right_on_boundary(text: str, budget: int) -> str:
    part = text[-max(0, int(budget)) :]
    if len(part) <= 10:
        return part.lstrip(" .-_")
    m = re.match(r"^[^- _]{1,16}[- _]", part)
    if m:
        cand = part[m.end() :]
        if len(cand) >= max(8, len(part) // 2):
            part = cand
    return part.lstrip(" .-_")


def trim_middle_readable(text: str, *, max_len: int, ellipsis: str = "...") -> str:
    s = (text or "").strip()
    lim = int(max_len or 0)
    if lim <= 0:
        return ""
    if len(s) <= lim:
        return s
    if lim <= len(ellipsis) + 6:
        return s[:lim].rstrip(" .-_")
    keep = lim - len(ellipsis)
    left_budget = keep // 2
    right_budget = keep - left_budget
    left = _trim_left_on_boundary(s, left_budget)
    right = _trim_right_on_boundary(s, right_budget)
    if not left:
        left = s[:left_budget].rstrip(" .-_")
    if not right:
        right = s[-right_budget:].lstrip(" .-_")
    out = f"{left}{ellipsis}{right}".strip(" .-_")
    if len(out) <= lim:
        return out
    # Final clamp if boundary trimming still overflowed.
    return (out[:lim]).rstrip(" .-_")


def safe_storage_base_len(
    *,
    pdf_dir: Path | str | None = None,
    md_out_root: Path | str | None = None,
    default_base_max: int = 88,
) -> int:
    try:
        lim = _safe_base_len_for_paths(pdf_dir=pdf_dir, md_out_root=md_out_root, default_base_max=default_base_max)
        return max(40, min(160, int(lim)))
    except Exception:
        return max(40, min(160, int(default_base_max)))


def fit_storage_base_name(base: str, *, max_len: int, suffix: str = "") -> str:
    lim = max(8, int(max_len or 0))
    tail = str(suffix or "")
    core_budget = max(8, lim - len(tail))
    core = sanitize_filename_component(base) or "paper"
    if len(core) > core_budget:
        core = trim_middle_readable(core, max_len=core_budget, ellipsis="...")
        core = sanitize_filename_component(core) or "paper"
    out = f"{core}{tail}".strip(" .-_")
    if not out:
        out = "paper"
    if len(out) > lim:
        out = trim_middle_readable(out, max_len=lim, ellipsis="...")
        out = sanitize_filename_component(out) or "paper"
    return out


def _compose_parts(*parts: str) -> str:
    items = [sanitize_filename_component(p) for p in parts if sanitize_filename_component(p)]
    return "-".join(items) if items else "paper"


def build_display_pdf_filename(
    *,
    venue: str,
    year: str,
    title: str,
    fallback_name: str = "",
    shorten_venue: bool = True,
) -> str:
    venue_part = abbreviate_venue(venue) if shorten_venue else venue
    base = _compose_parts(venue_part, year, title)
    if (not base) or (base == "paper"):
        fb = sanitize_filename_component(Path(fallback_name or "").stem or fallback_name or "")
        if fb:
            base = fb
    return f"{base}.pdf"


def build_storage_base_name(
    *,
    venue: str,
    year: str,
    title: str,
    pdf_dir: Path | str | None = None,
    md_out_root: Path | str | None = None,
    max_len: int | None = None,
    shorten_venue: bool = True,
) -> str:
    venue_part = abbreviate_venue(venue) if shorten_venue else venue
    venue_s = sanitize_filename_component(venue_part)
    year_s = sanitize_filename_component(year)
    title_s = sanitize_filename_component(title)
    base_full = _compose_parts(venue_s, year_s, title_s)
    target_len = int(max_len) if isinstance(max_len, int) and max_len > 0 else safe_storage_base_len(
        pdf_dir=pdf_dir,
        md_out_root=md_out_root,
        default_base_max=88,
    )
    target_len = max(40, min(160, target_len))
    if len(base_full) <= target_len:
        return base_full

    prefix = _compose_parts(venue_s, year_s)
    if prefix and title_s:
        title_budget = target_len - len(prefix) - 1
        if title_budget >= 10:
            title_cut = trim_middle_readable(title_s, max_len=title_budget, ellipsis="...")
            out = _compose_parts(prefix, title_cut)
            if len(out) <= target_len:
                return out

    return fit_storage_base_name(base_full, max_len=target_len)


def build_doc_id(*parts: str) -> str:
    raw = "|".join(str(x or "").strip() for x in parts if str(x or "").strip())
    if not raw:
        raw = "paper"
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:12]


def merge_citation_meta_file_labels(
    citation_meta: dict | None,
    *,
    display_full_name: str,
    storage_filename: str,
    doc_id: str | None = None,
) -> dict:
    out: dict[str, Any] = dict(citation_meta) if isinstance(citation_meta, dict) else {}
    disp = str(display_full_name or "").strip()
    stor = str(storage_filename or "").strip()
    if disp:
        out[KB_DISPLAY_FULL_NAME_KEY] = disp
    if stor:
        out[KB_STORAGE_FILENAME_KEY] = stor
    did = str(doc_id or "").strip()
    if (not did) and (disp or stor):
        did = build_doc_id(disp, stor)
    if did:
        out[KB_DOC_ID_KEY] = did
    return out


def merge_citation_meta_name_fields(
    citation_meta: dict | None,
    *,
    venue: str = "",
    year: str = "",
    title: str = "",
) -> dict:
    out: dict[str, Any] = dict(citation_meta) if isinstance(citation_meta, dict) else {}
    venue_s = sanitize_filename_component(venue)
    year_s = str(year or "").strip()
    title_s = sanitize_filename_component(title)
    if venue_s:
        out[KB_NAME_VENUE_KEY] = venue_s
    if title_s:
        out[KB_NAME_TITLE_KEY] = title_s
    if year_s and re.fullmatch(r"(19\d{2}|20\d{2})", year_s):
        out[KB_NAME_YEAR_KEY] = year_s
    return out


def citation_meta_name_fields(citation_meta: dict | None) -> dict[str, str]:
    if not isinstance(citation_meta, dict):
        return {"venue": "", "year": "", "title": ""}
    venue = str(citation_meta.get(KB_NAME_VENUE_KEY) or "").strip()
    year = str(citation_meta.get(KB_NAME_YEAR_KEY) or "").strip()
    title = str(citation_meta.get(KB_NAME_TITLE_KEY) or "").strip()
    if year and (not re.fullmatch(r"(19\d{2}|20\d{2})", year)):
        year = ""
    return {"venue": venue, "year": year, "title": title}


def citation_meta_display_pdf_name(citation_meta: dict | None) -> str:
    if not isinstance(citation_meta, dict):
        return ""
    return str(citation_meta.get(KB_DISPLAY_FULL_NAME_KEY) or "").strip()


def citation_meta_storage_filename(citation_meta: dict | None) -> str:
    if not isinstance(citation_meta, dict):
        return ""
    return str(citation_meta.get(KB_STORAGE_FILENAME_KEY) or "").strip()
