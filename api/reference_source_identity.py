from __future__ import annotations

import re

from api.reference_rendering import _parse_filename_meta


def _source_filename(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return ""
    parts = re.split(r"[\\/]+", s)
    return str(parts[-1] or "").strip() if parts else s


def _source_identity_keys(source_path: str) -> set[str]:
    raw = str(source_path or "").strip()
    if not raw:
        return set()
    out: set[str] = set()
    norm = raw.replace("\\", "/").strip().lower()
    if norm:
        out.add(norm)

    name = _source_filename(raw).strip().lower()
    if name:
        out.add(name)
        if name.endswith(".en.md"):
            pdf_name = name[:-6] + ".pdf"
            stem_name = name[:-6]
            out.add(pdf_name)
            out.add(stem_name)
        elif name.endswith(".md"):
            pdf_name = name[:-3] + ".pdf"
            stem_name = name[:-3]
            out.add(pdf_name)
            out.add(stem_name)
    return {item for item in out if item}


def _same_source_identity(source_path: str, bound_source_path: str) -> bool:
    left = _source_identity_keys(source_path)
    right = _source_identity_keys(bound_source_path)
    if not left or not right:
        return False
    return bool(left.intersection(right))


def _normalize_title_identity(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if low.endswith(".en.md"):
        raw = raw[:-6]
    elif low.endswith(".md") or low.endswith(".pdf"):
        raw = raw[:-3] if low.endswith(".md") else raw[:-4]
    raw = re.sub(r"(19\d{2}|20\d{2})\s*-\s*", r"\1 - ", raw)
    raw = re.sub(r"[_/\\]+", " ", raw)
    raw = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip().lower()
    return raw


def _title_identity_keys(source_like: str) -> set[str]:
    raw = str(source_like or "").strip()
    if not raw:
        return set()
    out: set[str] = set()

    def _push(value: str):
        norm = _normalize_title_identity(value)
        if norm:
            out.add(norm)

    _push(raw)
    name = _source_filename(raw)
    if name:
        _push(name)
    _venue, _year, parsed_title = _parse_filename_meta(raw)
    if parsed_title:
        _push(parsed_title)
    base = name or raw
    m = re.search(r"(?:19\d{2}|20\d{2})\s*-\s*(.+)$", base)
    if m:
        _push(str(m.group(1) or "").strip())
    return {item for item in out if item}


def _same_source_title_identity(left_source: str, right_source: str) -> bool:
    left = _title_identity_keys(left_source)
    right = _title_identity_keys(right_source)
    if not left or not right:
        return False

    def _first_identity_token(value: str) -> str:
        stop = {
            "the", "and", "for", "with", "from", "into", "using", "based", "towards",
            "conference", "symposium", "workshop", "journal", "transactions", "letters",
            "ieee", "cvpr", "iccv", "eccv", "neurips", "iclr", "icml",
        }
        tokens = [tok for tok in str(value or "").split() if tok]
        for tok in tokens:
            if re.fullmatch(r"(19\d{2}|20\d{2})", tok):
                continue
            if tok in stop:
                continue
            if len(tok) < 3:
                continue
            return tok
        return tokens[0] if tokens else ""

    if left.intersection(right):
        return True
    for a in left:
        for b in right:
            if min(len(a), len(b)) < 20:
                continue
            if (a in b) or (b in a):
                return True
            a_tokens = set(a.split())
            b_tokens = set(b.split())
            if len(a_tokens) < 4 or len(b_tokens) < 4:
                continue
            overlap = len(a_tokens.intersection(b_tokens))
            smaller = min(len(a_tokens), len(b_tokens))
            if smaller <= 0:
                continue
            if (overlap / float(smaller)) >= 0.75 and _first_identity_token(a) == _first_identity_token(b):
                return True
    return False
