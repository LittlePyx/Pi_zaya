from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from kb.citation_meta import (
    extract_first_doi,
    extract_first_author_family_hint,
    extract_year_hint,
    fetch_best_crossref_for_reference,
    fetch_best_crossref_meta,
    fetch_crossref_references_by_doi,
    is_promising_reference_text,
    normalize_title_for_match,
)
from kb.store import compute_file_sha1


INDEX_FILE_NAME = "references_index.json"
CROSSREF_CACHE_FILE_NAME = "crossref_cache.json"

_REF_HEAD_RE = re.compile(
    r"^#{1,6}\s+(references(?:\s+and\s+(?:notes|links))?|bibliography)\b",
    re.IGNORECASE,
)
_REF_START_BRACKET_RE = re.compile(r"^\[(\d{1,4})\]\s*(.*\S)?\s*$")
_REF_START_DOT_RE = re.compile(r"^(\d{1,4})\.\s+(.*\S)?\s*$")
_REF_ANY_START_RE = re.compile(r"(?:\[\d{1,4}\]|\d{1,4}\.)\s+")
_DOI_TRIM_RE = re.compile(r"^[ \t\r\n.,;:(){}\[\]<>]+|[ \t\r\n.,;:(){}\[\]<>]+$")
_SOURCE_DOI_HTTP_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
_SOURCE_DOI_RE = re.compile(r"\bdoi\s*:?\s*(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", re.IGNORECASE)


def _to_os_path(path_like: str | Path) -> str:
    p = Path(path_like).expanduser()
    try:
        s = str(p.resolve(strict=False))
    except Exception:
        s = str(p)
    if os.name != "nt":
        return s
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        tail = s.lstrip("\\")
        return "\\\\?\\UNC\\" + tail
    return "\\\\?\\" + s


def _path_is_file(path_like: str | Path) -> bool:
    try:
        return bool(os.path.isfile(_to_os_path(path_like)))
    except Exception:
        try:
            return Path(path_like).is_file()
        except Exception:
            return False


def _norm_source_key(path_like: str | Path) -> str:
    s = str(path_like or "").strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    try:
        return str(p.resolve()).strip().lower()
    except Exception:
        return str(p).strip().lower()


def _iter_md_files(
    src: Path,
    *,
    glob: str = "*.md",
    exclude_dirs: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> list[Path]:
    ex_dirs = {str(x).strip().lower() for x in (exclude_dirs or set()) if str(x).strip()}
    ex_names = {str(x).strip().lower() for x in (exclude_names or set()) if str(x).strip()}
    if _path_is_file(src):
        return [src]

    pool: list[Path] = []
    for p in src.rglob(glob):
        if not _path_is_file(p):
            continue
        parts = [str(x).strip().lower() for x in p.parts]
        if any(part in ex_dirs for part in parts):
            continue
        pool.append(p)

    # Detect directories that already have a main .en.md file.
    has_en_in_dir: set[str] = set()
    for p in pool:
        n = p.name.lower()
        if n.endswith(".en.md"):
            has_en_in_dir.add(str(p.parent).strip().lower())

    out: list[Path] = []
    for p in pool:
        n = p.name.lower()
        if n in ex_names:
            # Keep output.md as a fallback only when this directory has no .en.md.
            if n == "output.md":
                dir_k = str(p.parent).strip().lower()
                if dir_k not in has_en_in_dir:
                    out.append(p)
            continue
        out.append(p)
    return sorted(out)


def _read_text_tail(path: Path, *, max_bytes: int = 1_500_000) -> str:
    os_p = _to_os_path(path)
    try:
        size = int(os.path.getsize(os_p))
    except Exception:
        size = 0
    try:
        if size <= 0:
            return ""
        if size <= int(max_bytes):
            with open(os_p, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="ignore")
        with open(os_p, "rb") as f:
            f.seek(max(0, size - int(max_bytes)))
            raw = f.read(int(max_bytes))
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_references_map_from_md(md_text: str) -> dict[int, str]:
    md = (md_text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not md.strip():
        return {}
    lines = md.split("\n")

    ref_i: int | None = None
    for i, ln in enumerate(lines):
        s = (ln or "").strip()
        if _REF_HEAD_RE.match(s):
            ref_i = i
            break

    def _fallback_ref_start_index(_lines: list[str]) -> int | None:
        total = len(_lines)
        if total < 80:
            return None
        scan_from = max(0, int(total * 0.45))
        cand_idx: list[int] = []
        cand_num: list[int] = []
        for j in range(scan_from, total):
            s0 = str(_lines[j] or "").strip()
            m0 = _REF_START_BRACKET_RE.match(s0) or _REF_START_DOT_RE.match(s0)
            if not m0:
                continue
            try:
                n0 = int(m0.group(1))
            except Exception:
                continue
            cand_idx.append(j)
            cand_num.append(n0)
        if len(cand_idx) < 8:
            return None

        best_idx = -1
        best_cnt = -1
        for i0 in cand_idx:
            c0 = 0
            hi = i0 + 260
            for j0 in cand_idx:
                if i0 <= j0 <= hi:
                    c0 += 1
            if c0 > best_cnt:
                best_cnt = c0
                best_idx = i0
        if best_idx < 0 or best_cnt < 8:
            return None

        cluster_nums: list[int] = []
        for j, n0 in zip(cand_idx, cand_num):
            if best_idx <= j <= (best_idx + 260):
                cluster_nums.append(int(n0))
        if len(cluster_nums) < 8:
            return None
        if max(cluster_nums) < 8:
            return None
        inc = 0
        for a, b in zip(cluster_nums, cluster_nums[1:]):
            if b >= a:
                inc += 1
        if inc < int(0.60 * max(1, len(cluster_nums) - 1)):
            return None

        for j, n0 in zip(cand_idx, cand_num):
            if (best_idx - 40) <= j <= (best_idx + 60) and int(n0) == 1:
                return max(0, j - 1)
        return max(0, best_idx - 1)

    if ref_i is None:
        ref_i = _fallback_ref_start_index(lines)
    if ref_i is None:
        return {}

    out: dict[int, str] = {}
    cur_n: int | None = None
    cur_buf: list[str] = []

    def _flush() -> None:
        nonlocal cur_n, cur_buf
        if cur_n is None:
            cur_buf = []
            return
        merged = " ".join(x.strip() for x in cur_buf if str(x or "").strip()).strip()
        if merged:
            out[int(cur_n)] = merged
        cur_n = None
        cur_buf = []

    def _split_embedded_ref_segments(line: str) -> list[str]:
        s0 = (line or "").strip()
        if not s0:
            return []
        starts = [0]
        for m0 in _REF_ANY_START_RE.finditer(s0):
            i0 = int(m0.start())
            if i0 <= 0:
                continue
            prev = s0[i0 - 1]
            # Split when marker is not glued to an alnum token.
            if str(prev).isalnum():
                continue
            starts.append(i0)
        starts = sorted(set(starts))
        if len(starts) <= 1:
            return [s0]
        out_segs: list[str] = []
        for i0, st0 in enumerate(starts):
            ed0 = starts[i0 + 1] if (i0 + 1) < len(starts) else len(s0)
            seg = s0[st0:ed0].strip()
            if seg:
                out_segs.append(seg)
        return out_segs

    for raw in lines[ref_i + 1 :]:
        for s in _split_embedded_ref_segments(raw):
            if not s:
                continue

            if re.match(r"^#{1,6}\s+\S+", s) and (not _REF_HEAD_RE.match(s)):
                if out:
                    _flush()
                    return out

            m = _REF_START_BRACKET_RE.match(s) or _REF_START_DOT_RE.match(s)
            if m:
                _flush()
                try:
                    cur_n = int(m.group(1))
                except Exception:
                    cur_n = None
                    continue
                rest = (m.group(2) or "").strip()
                if rest:
                    cur_buf = [f"[{int(cur_n)}] {rest}"]
                else:
                    cur_buf = [f"[{int(cur_n)}]"]
                continue

            if cur_n is not None:
                cur_buf.append(s)

    _flush()
    return _cleanup_reference_number_noise(out)


def _cleanup_reference_number_noise(ref_map: dict[int, str]) -> dict[int, str]:
    """
    Remove obvious OCR-induced fake reference numbers:
    - years misread as [2019], [2024], ...
    - extreme outliers when a dense low-index reference list already exists.
    """
    if not isinstance(ref_map, dict) or (not ref_map):
        return {}

    out: dict[int, str] = {}
    removed_year_like = False
    for k, v in ref_map.items():
        try:
            n = int(k)
        except Exception:
            continue
        if n <= 0:
            continue
        # Years are not reference indices.
        if 1900 <= n <= 2099:
            removed_year_like = True
            continue
        txt = str(v or "").strip()
        if not txt:
            continue
        out[n] = txt

    if not out:
        return {}

    nums = sorted(out.keys())
    total = len(nums)
    has_extreme = any(n >= 1000 for n in nums)
    core_n = sum(1 for n in nums if 1 <= n <= 400)

    # If the list is mostly low-index refs but contains huge outliers,
    # cap indices by a data-driven upper bound.
    if (
        total >= 15
        and (core_n / max(1, total)) >= 0.70
        and (has_extreme or removed_year_like)
    ):
        cap = max(80, min(500, int(total * 1.6) + 20))
        out = {n: t for n, t in out.items() if n <= cap}

    return out


def _extract_query_title(entry: str) -> str:
    s = (entry or "").strip()
    if not s:
        return ""
    s = re.sub(r"^\[(\d{1,4})\]\s*", "", s)
    s = re.sub(r"^(\d{1,4})\.\s*", "", s)
    s = re.sub(r"https?://\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdoi\s*:?\s*10\.\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\barxiv\s*:?\s*\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return ""

    def _looks_author_segment(seg: str) -> bool:
        t = str(seg or "").strip()
        if not t:
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", t)
        if len(words) < 4:
            return False
        comma_n = t.count(",")
        and_n = len(re.findall(r"\b(?:and|et al)\b|&", t, flags=re.IGNORECASE))
        init_n = len(re.findall(r"\b[A-Z]\.?\b", t))
        cap_n = sum(1 for w in words if w[:1].isupper())
        cap_ratio = cap_n / max(1, len(words))
        return bool((comma_n >= 2) and (cap_ratio >= 0.55) and ((and_n >= 1) or (init_n >= 2)))

    def _looks_venue_segment(seg: str) -> bool:
        t = str(seg or "").strip().lower()
        if not t:
            return False
        starts = (
            "in proceedings of",
            "proceedings of",
            "in proc",
            "ieee transactions",
            "acm transactions",
            "journal of",
            "optics ",
            "applied optics",
            "nat. commun",
            "nature ",
            "science ",
            "arxiv preprint",
        )
        if any(t.startswith(x) for x in starts):
            return True
        if len(t) <= 110 and re.search(r"\b(cvpr|iccv|eccv|neurips|icml|iclr|aaai|ijcai|kdd|sigir)\b", t):
            return True
        return False

    # Heuristic 1: title often lies between first and second period.
    m_title = re.search(r"^[^.]{4,260}\.\s*([^.]{8,260})\.", s)
    if m_title:
        cand = (m_title.group(1) or "").strip(" .;:")
        if cand and (not _looks_author_segment(cand)) and (not _looks_venue_segment(cand)):
            return cand[:260]

    # Heuristic 2: choose best scored segment among sentence-like chunks.
    segs = [x.strip(" .;:") for x in re.split(r"\.\s+", s) if x.strip(" .;:")]
    if not segs:
        return s[:260]

    best = ""
    best_score = -10_000.0
    for idx, seg in enumerate(segs):
        w_n = len(seg.split())
        score = float(min(280, len(seg)))
        if 4 <= w_n <= 24:
            score += 20.0
        if ":" in seg:
            score += 16.0
        if "-" in seg:
            score += 5.0
        if _looks_author_segment(seg):
            score -= 140.0
        if _looks_venue_segment(seg):
            score -= 120.0
        if re.search(r"\b(19\d{2}|20\d{2})\b", seg):
            score -= 15.0
        # The second segment is very often the paper title.
        if idx == 1:
            score += 18.0
        if score > best_score:
            best_score = score
            best = seg

    if best:
        return best[:260]
    return segs[0][:260]


def _clean_doi_for_url(doi: str) -> str:
    d = (doi or "").strip()
    if not d:
        return ""
    return _DOI_TRIM_RE.sub("", d)


def _doi_url(doi: str) -> str:
    d = _clean_doi_for_url(doi)
    if not d:
        return ""
    return "https://doi.org/" + quote(d, safe="/-._;()")


def _read_text_head(path: Path, *, max_bytes: int = 220_000) -> str:
    os_p = _to_os_path(path)
    try:
        with open(os_p, "rb") as f:
            raw = f.read(int(max_bytes))
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_source_doi_from_md_head(md_head: str) -> str:
    s = (md_head or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return ""
    # Ignore references section.
    m_ref = _REF_HEAD_RE.search(s)
    if m_ref:
        s = s[: m_ref.start()]
    m = _SOURCE_DOI_HTTP_RE.search(s)
    if m:
        return _clean_doi_for_url(m.group(1))
    m2 = _SOURCE_DOI_RE.search(s)
    if m2:
        return _clean_doi_for_url(m2.group(1))
    return _clean_doi_for_url(extract_first_doi(s))


def _norm_path_key(path_like: str | Path) -> str:
    s = str(path_like or "").strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    try:
        return str(p.resolve()).strip().lower()
    except Exception:
        return str(p).strip().lower()


def _load_library_citation_meta_map(library_db_path: Path | None) -> dict[str, dict]:
    p = Path(library_db_path) if library_db_path else None
    if (p is None) or (not p.exists()):
        return {}
    out: dict[str, dict] = {}
    try:
        conn = sqlite3.connect(str(p), timeout=20)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT path, citation_meta FROM pdf_files").fetchall()
        conn.close()
        for r in rows:
            try:
                path_s = str(r["path"] or "").strip()
            except Exception:
                path_s = ""
            if not path_s:
                continue
            key = _norm_path_key(path_s)
            if not key:
                continue
            raw_meta = r["citation_meta"]
            if not raw_meta:
                continue
            try:
                obj = json.loads(raw_meta)
            except Exception:
                continue
            if isinstance(obj, dict):
                out[key] = obj
    except Exception:
        return {}
    return out


def _best_source_doi_from_citation_meta(cm: dict[str, Any], *, crossref_enabled: bool) -> str:
    if not isinstance(cm, dict):
        return ""
    for k in ("doi", "DOI"):
        doi = _clean_doi_for_url(str(cm.get(k) or ""))
        if doi:
            return doi

    if not crossref_enabled:
        return ""

    title = str(cm.get("title") or "").strip()
    if not title:
        return ""
    year = str(cm.get("year") or cm.get("published_year") or "").strip()
    if year and (not re.fullmatch(r"(19\d{2}|20\d{2})", year)):
        year = ""
    venue = str(cm.get("venue") or cm.get("journal") or cm.get("container-title") or "").strip()

    try:
        meta = fetch_best_crossref_meta(
            query_title=title,
            expected_year=year,
            expected_venue=venue,
            doi_hint="",
            min_score=0.90,
            allow_title_only=True,
        )
    except Exception:
        meta = None
    if not isinstance(meta, dict):
        return ""
    return _clean_doi_for_url(str(meta.get("doi") or ""))


def _pdf_stem_from_md_path(md_path: Path) -> str:
    stem = str(Path(md_path).stem or "").strip()
    if stem.lower().endswith(".en"):
        stem = stem[:-3].strip()
    return stem


_DOC_BAD_TITLE_HINTS = {
    "markdown quality analysis report",
    "quality report",
    "output",
    "table of contents",
    "introduction",
    "results",
    "discussion",
    "methods",
    "conclusion",
    "acknowledgements",
    "acknowledgments",
    "supplementary materials",
    "references",
    "references and notes",
}


def _extract_doc_title_from_md_head(md_head: str) -> str:
    s = (md_head or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return ""
    for ln in s.split("\n"):
        t = str(ln or "").strip()
        if not t.startswith("#"):
            continue
        t = re.sub(r"^#{1,6}\s*", "", t).strip()
        if not t:
            continue
        low = t.lower()
        if _REF_HEAD_RE.match("# " + t):
            continue
        if low in _DOC_BAD_TITLE_HINTS:
            continue
        if len(t) < 8:
            continue
        return t[:320]
    return ""


def _parse_doc_name_hints(md_path: Path) -> tuple[str, str, str]:
    stem = _pdf_stem_from_md_path(md_path)
    if not stem:
        return "", "", ""
    m = re.match(r"^(?P<venue>.+?)-(?P<year>19\d{2}|20\d{2})-(?P<title>.+)$", stem)
    if not m:
        return "", "", stem
    venue = str(m.group("venue") or "").strip()
    year = str(m.group("year") or "").strip()
    title = str(m.group("title") or "").strip()
    return venue, year, title


def _infer_source_doi_from_doc_hints(
    md_path: Path,
    md_head: str,
    cache: dict,
    *,
    crossref_enabled: bool,
) -> str:
    if not crossref_enabled:
        return ""
    venue0, year0, title0 = _parse_doc_name_hints(md_path)
    title_head = _extract_doc_title_from_md_head(md_head)
    title_h = title_head or title0
    if title0 and title_head:
        n_head = normalize_title_for_match(title_head)
        n_name = normalize_title_for_match(title0)
        sim = difflib.SequenceMatcher(None, n_head, n_name).ratio() if (n_head and n_name) else 0.0
        # Many papers start with section headers in OCR output; prefer filename title when it is
        # substantially longer or semantically far from the extracted heading.
        if (len(title_head) < max(16, int(len(title0) * 0.72))) or (sim < 0.72):
            title_h = title0
    if not title_h:
        return ""
    year_h = year0 if re.fullmatch(r"(19\d{2}|20\d{2})", str(year0 or "").strip()) else ""
    venue_h = venue0

    key = (
        f"{normalize_title_for_match(title_h)[:220]}|"
        f"{str(year_h or '').strip()}|"
        f"{normalize_title_for_match(venue_h)[:120]}"
    )
    source_work_cache = cache.get("source_work")
    if not isinstance(source_work_cache, dict):
        source_work_cache = {}
        cache["source_work"] = source_work_cache
    if key in source_work_cache:
        return _clean_doi_for_url(str(source_work_cache.get(key) or ""))

    doi = ""
    try:
        meta = fetch_best_crossref_meta(
            query_title=title_h,
            expected_year=year_h,
            expected_venue=venue_h,
            doi_hint="",
            min_score=0.90,
            allow_title_only=True,
        )
    except Exception:
        meta = None
    if isinstance(meta, dict):
        doi = _clean_doi_for_url(str(meta.get("doi") or ""))

    if (not doi) and year_h:
        try:
            meta2 = fetch_best_crossref_meta(
                query_title=title_h,
                expected_year=year_h,
                expected_venue="",
                doi_hint="",
                min_score=0.92,
                allow_title_only=True,
            )
        except Exception:
            meta2 = None
        if isinstance(meta2, dict):
            doi = _clean_doi_for_url(str(meta2.get("doi") or ""))

    if (not doi) and (not year_h):
        try:
            meta3 = fetch_best_crossref_meta(
                query_title=title_h,
                expected_year="",
                expected_venue="",
                doi_hint="",
                min_score=0.96,
                allow_title_only=True,
            )
        except Exception:
            meta3 = None
        if isinstance(meta3, dict):
            doi = _clean_doi_for_url(str(meta3.get("doi") or ""))

    source_work_cache[key] = doi
    return doi


def _norm_name_key(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _lookup_pdf_for_md_doc(md_path: Path, pdf_root: Path | None) -> Path | None:
    if pdf_root is None:
        return None
    root = Path(pdf_root)
    if not root.exists():
        return None

    stem = _pdf_stem_from_md_path(md_path)
    if not stem:
        return None
    direct = [root / f"{stem}.pdf", root / f"{stem}.PDF"]
    for p in direct:
        if p.exists():
            return p

    target = _norm_name_key(stem)
    if not target:
        return None
    best: Path | None = None
    best_score = 0.0
    try:
        for p in root.glob("*.pdf"):
            cand = _norm_name_key(p.stem)
            if not cand:
                continue
            if cand == target:
                return p
            sim = difflib.SequenceMatcher(None, target, cand).ratio()
            if sim > best_score:
                best_score = sim
                best = p
    except Exception:
        return None
    if best is not None and best_score >= 0.90:
        return best
    return None


def _source_ref_row_to_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("author") or ""),
        str(row.get("article-title") or ""),
        str(row.get("journal-title") or ""),
        str(row.get("year") or ""),
        str(row.get("volume") or ""),
        str(row.get("first-page") or ""),
        str(row.get("DOI") or ""),
        str(row.get("unstructured") or ""),
    ]
    return " ".join(x for x in parts if x).strip()


def _normalize_source_ref_row(row: dict[str, Any]) -> dict[str, str]:
    doi = _clean_doi_for_url(str(row.get("DOI") or ""))
    out = {
        "doi": doi,
        "title": str(row.get("article-title") or "").strip(),
        "venue": str(row.get("journal-title") or "").strip(),
        "year": str(row.get("year") or "").strip(),
        "volume": str(row.get("volume") or "").strip(),
        "pages": str(row.get("first-page") or "").strip(),
        "author": str(row.get("author") or "").strip(),
        "unstructured": str(row.get("unstructured") or "").strip(),
    }
    out["text"] = _source_ref_row_to_text(row)
    return out


def _load_source_reference_rows(
    source_doi: str,
    cache: dict,
    *,
    crossref_enabled: bool,
) -> list[dict[str, str]]:
    d = _clean_doi_for_url(source_doi).lower()
    if not d:
        return []
    key = f"doi:{d}"

    src_cache = cache.get("source_refs")
    if not isinstance(src_cache, dict):
        src_cache = {}
        cache["source_refs"] = src_cache

    if key in src_cache:
        v = src_cache.get(key)
        if isinstance(v, list):
            out: list[dict[str, str]] = []
            for r in v:
                if isinstance(r, dict):
                    out.append({k: str(v2 or "") for k, v2 in r.items()})
            return out
        return []

    if not crossref_enabled:
        return []

    rows_raw = fetch_crossref_references_by_doi(source_doi)
    out_rows: list[dict[str, str]] = []
    for r in rows_raw:
        if not isinstance(r, dict):
            continue
        out_rows.append(_normalize_source_ref_row(r))
    src_cache[key] = out_rows
    return out_rows


def _similarity(a: str, b: str) -> float:
    na = normalize_title_for_match(a)
    nb = normalize_title_for_match(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    sa = set(na.split())
    sb = set(nb.split())
    jac = (len(sa & sb) / len(sa | sb)) if (sa and sb) else 0.0
    return min(1.0, (0.68 * seq) + (0.32 * jac))


def _match_source_reference(local_entry: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    raw = str(local_entry or "").strip()
    if (not raw) or (not rows):
        return None
    local_doi = _clean_doi_for_url(extract_first_doi(raw)).lower()
    local_year = extract_year_hint(raw)
    local_author = extract_first_author_family_hint(raw)

    if local_doi:
        for r in rows:
            if _clean_doi_for_url(str(r.get("doi") or "")).lower() == local_doi:
                return r

    best: dict[str, str] | None = None
    best_score = -1.0
    for r in rows:
        cand_text = str(r.get("text") or "")
        if not cand_text:
            continue
        sc = _similarity(raw, cand_text)
        cand_year = str(r.get("year") or "").strip()
        if local_year:
            if cand_year == local_year:
                sc += 0.14
            else:
                try:
                    if cand_year and (abs(int(cand_year) - int(local_year)) <= 1):
                        sc += 0.05
                    else:
                        sc -= 0.08
                except Exception:
                    sc -= 0.05
        cand_author = str(r.get("author") or "").strip().lower()
        if local_author:
            if local_author and cand_author and (local_author in cand_author):
                sc += 0.10
            else:
                sc -= 0.04
        if str(r.get("doi") or "").strip():
            sc += 0.03
        sc = max(0.0, min(1.0, sc))
        if sc > best_score:
            best_score = sc
            best = r

    if best is None:
        return None
    if best_score < 0.63:
        return None
    return best


def _match_source_reference_by_number(
    ref_num: int,
    local_entry: str,
    rows: list[dict[str, str]],
) -> dict[str, str] | None:
    try:
        idx = int(ref_num) - 1
    except Exception:
        return None
    if (idx < 0) or (idx >= len(rows)):
        return None
    row = rows[idx]
    if not isinstance(row, dict):
        return None

    raw = str(local_entry or "").strip()
    if not raw:
        return None

    local_doi = _clean_doi_for_url(extract_first_doi(raw)).lower()
    row_doi = _clean_doi_for_url(str(row.get("doi") or "")).lower()
    if local_doi and row_doi and (local_doi == row_doi):
        return row

    cand_text = str(row.get("text") or "").strip()
    if not cand_text:
        return None

    sc = _similarity(raw, cand_text)
    local_year = extract_year_hint(raw)
    cand_year = str(row.get("year") or "").strip()
    if local_year and cand_year:
        if local_year == cand_year:
            sc += 0.10
        else:
            try:
                if abs(int(local_year) - int(cand_year)) <= 1:
                    sc += 0.04
            except Exception:
                pass

    local_author = extract_first_author_family_hint(raw)
    cand_author = str(row.get("author") or "").strip().lower()
    if local_author and cand_author and (local_author in cand_author):
        sc += 0.08
    if row_doi:
        sc += 0.02

    sc = max(0.0, min(1.0, sc))
    if sc >= 0.56:
        return row
    return None


def _load_json(path: Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_reference_index(db_dir: Path) -> dict:
    p = Path(db_dir) / INDEX_FILE_NAME
    data = _load_json(p)
    if not isinstance(data, dict):
        return {}
    docs = data.get("docs")
    if not isinstance(docs, dict):
        data["docs"] = {}
    return data


def _crossref_cache_load(db_dir: Path) -> dict:
    p = Path(db_dir) / CROSSREF_CACHE_FILE_NAME
    data = _load_json(p)
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("doi"), dict):
        data["doi"] = {}
    if not isinstance(data.get("bib"), dict):
        data["bib"] = {}
    if not isinstance(data.get("source_refs"), dict):
        data["source_refs"] = {}
    if not isinstance(data.get("source_work"), dict):
        data["source_work"] = {}
    if not isinstance(data.get("title"), dict):
        data["title"] = {}
    return data


def _crossref_cache_save(db_dir: Path, cache: dict) -> None:
    out = {
        "version": 1,
        "updated_at": time.time(),
        "doi": cache.get("doi") if isinstance(cache.get("doi"), dict) else {},
        "bib": cache.get("bib") if isinstance(cache.get("bib"), dict) else {},
        "source_refs": cache.get("source_refs") if isinstance(cache.get("source_refs"), dict) else {},
        "source_work": cache.get("source_work") if isinstance(cache.get("source_work"), dict) else {},
        "title": cache.get("title") if isinstance(cache.get("title"), dict) else {},
    }
    _save_json(Path(db_dir) / CROSSREF_CACHE_FILE_NAME, out)


def _crossref_preflight_ok(*, timeout_s: float = 2.5) -> bool:
    url = "https://api.crossref.org/works"
    for _ in range(2):
        try:
            resp = requests.get(
                url,
                params={"rows": 0},
                headers={"User-Agent": "Pi-zaya-KB/1.0 (Research Assistant)"},
                timeout=float(max(2.5, timeout_s)),
            )
            code = int(resp.status_code or 0)
            # 429 means reachable but rate-limited.
            if code in {200, 429}:
                return True
        except Exception:
            continue
    return False


def _lookup_crossref_meta_for_entry(
    entry: str,
    cache: dict,
    *,
    crossref_enabled: bool,
    enable_title_lookup: bool,
) -> tuple[dict | None, str]:
    raw = (entry or "").strip()
    doi_hint = extract_first_doi(raw)
    doi_key = _clean_doi_for_url(doi_hint).lower()
    meta: dict | None = None

    doi_cache = cache.get("doi")
    if not isinstance(doi_cache, dict):
        doi_cache = {}
        cache["doi"] = doi_cache
    title_cache = cache.get("title")
    if not isinstance(title_cache, dict):
        title_cache = {}
        cache["title"] = title_cache
    bib_cache = cache.get("bib")
    if not isinstance(bib_cache, dict):
        bib_cache = {}
        cache["bib"] = bib_cache

    if crossref_enabled and doi_key:
        if doi_key in doi_cache:
            cached = doi_cache.get(doi_key)
            meta = cached if isinstance(cached, dict) else None
        else:
            found = fetch_best_crossref_meta(
                query_title="",
                doi_hint=doi_hint,
                allow_title_only=False,
                min_score=0.90,
            )
            meta = found if isinstance(found, dict) else None
            doi_cache[doi_key] = meta
        if isinstance(meta, dict):
            return meta, doi_hint

    # Bibliographic query fallback (author/journal/year/volume/pages etc.)
    ref_key = normalize_title_for_match(raw)[:260]
    if crossref_enabled and ref_key:
        if ref_key in bib_cache:
            cached = bib_cache.get(ref_key)
            meta = cached if isinstance(cached, dict) else None
        else:
            found = fetch_best_crossref_for_reference(reference_text=raw, min_score=0.62)
            meta = found if isinstance(found, dict) else None
            bib_cache[ref_key] = meta
        if isinstance(meta, dict):
            return meta, doi_hint

    if (not crossref_enabled) or (not enable_title_lookup) or (not is_promising_reference_text(raw)):
        return None, doi_hint

    title_hint = _extract_query_title(raw)
    title_key = normalize_title_for_match(title_hint)[:260]
    if not title_key or len(title_key) < 8:
        return None, doi_hint

    if title_key in title_cache:
        cached = title_cache.get(title_key)
        meta = cached if isinstance(cached, dict) else None
    else:
        found = fetch_best_crossref_meta(
            query_title=title_hint,
            doi_hint="",
            allow_title_only=True,
            min_score=0.95,
        )
        meta = found if isinstance(found, dict) else None
        title_cache[title_key] = meta

    return meta, doi_hint


def _path_suffix_match_score(want_path: str, cand_path: str, *, max_parts: int = 8) -> int:
    want = str(want_path or "").strip()
    cand = str(cand_path or "").strip()
    if not want or not cand:
        return 0
    wp = [str(x).strip().lower() for x in Path(want).parts if str(x).strip()]
    cp = [str(x).strip().lower() for x in Path(cand).parts if str(x).strip()]
    if not wp or not cp:
        return 0
    lim = min(len(wp), len(cp), int(max_parts))
    score = 0
    for k in range(1, lim + 1):
        if wp[-k:] == cp[-k:]:
            score = k
        else:
            break
    return int(score)


def resolve_reference_entry(
    index_data: dict,
    source_path: str,
    ref_num: int,
    *,
    source_sha1: str = "",
) -> dict | None:
    if not isinstance(index_data, dict):
        return None
    docs = index_data.get("docs")
    if not isinstance(docs, dict) or not docs:
        return None
    try:
        n = int(ref_num)
    except Exception:
        return None
    if n <= 0:
        return None

    ref_key = str(n)
    src_key = _norm_source_key(source_path)
    if src_key and src_key in docs:
        d = docs.get(src_key)
        if isinstance(d, dict):
            refs = d.get("refs")
            if isinstance(refs, dict):
                item = refs.get(ref_key)
                if isinstance(item, dict):
                    return {
                        "source_path": str(d.get("path") or source_path),
                        "source_name": str(d.get("name") or Path(source_path).name),
                        "ref_num": n,
                        "ref": item,
                    }

    source_sha1 = str(source_sha1 or "").strip().lower()
    if source_sha1:
        for d in docs.values():
            if not isinstance(d, dict):
                continue
            d_sha1 = str(d.get("sha1") or "").strip().lower()
            if not d_sha1 or d_sha1 != source_sha1:
                continue
            refs = d.get("refs")
            if not isinstance(refs, dict):
                continue
            item = refs.get(ref_key)
            if isinstance(item, dict):
                return {
                    "source_path": str(d.get("path") or source_path),
                    "source_name": str(d.get("name") or Path(source_path).name),
                    "ref_num": n,
                    "ref": item,
                }

    # Fallback: match by shared path suffix when source root changed across machines.
    if str(source_path or "").strip():
        best_score = 0
        best_matches: list[dict] = []
        for d in docs.values():
            if not isinstance(d, dict):
                continue
            refs = d.get("refs")
            if not isinstance(refs, dict):
                continue
            item = refs.get(ref_key)
            if not isinstance(item, dict):
                continue
            score = _path_suffix_match_score(str(source_path), str(d.get("path") or ""))
            if score <= 1:
                continue
            rec = {
                "source_path": str(d.get("path") or source_path),
                "source_name": str(d.get("name") or Path(source_path).name),
                "ref_num": n,
                "ref": item,
                "_score": int(score),
            }
            if score > best_score:
                best_score = int(score)
                best_matches = [rec]
            elif score == best_score:
                best_matches.append(rec)
        if best_matches and best_score >= 2:
            if len(best_matches) == 1:
                out = dict(best_matches[0])
                out.pop("_score", None)
                return out
            # If multiple docs tie, require a stronger shared suffix.
            if best_score >= 4:
                out = dict(best_matches[0])
                out.pop("_score", None)
                return out

    # Fallback: match by basename/stem when source path changed.
    want_name = Path(str(source_path or "")).name.lower()
    want_stem = Path(str(source_path or "")).stem.lower()
    for d in docs.values():
        if not isinstance(d, dict):
            continue
        d_name = str(d.get("name") or "").lower()
        d_stem = str(d.get("stem") or "").lower()
        if want_name and d_name and (want_name != d_name):
            if (not want_stem) or (want_stem != d_stem):
                continue
        refs = d.get("refs")
        if not isinstance(refs, dict):
            continue
        item = refs.get(ref_key)
        if isinstance(item, dict):
            return {
                "source_path": str(d.get("path") or source_path),
                "source_name": str(d.get("name") or Path(source_path).name),
                "ref_num": n,
                "ref": item,
            }

    return None


def build_reference_index(
    *,
    src_root: Path,
    db_dir: Path,
    incremental: bool = True,
    enable_title_lookup: bool = True,
    crossref_time_budget_s: float = 45.0,
    pdf_root: Path | None = None,
    library_db_path: Path | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    src = Path(src_root).expanduser().resolve()
    out_dir = Path(db_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files_raw = _iter_md_files(
        src,
        glob="*.md",
        exclude_dirs={"temp", "__pycache__", ".git"},
        exclude_names={"assets_manifest.md", "quality_report.md", "output.md"},
    )
    md_files = list(md_files_raw)

    prev = load_reference_index(out_dir)
    prev_docs = prev.get("docs") if isinstance(prev.get("docs"), dict) else {}
    if not isinstance(prev_docs, dict):
        prev_docs = {}

    cache = _crossref_cache_load(out_dir)
    crossref_enabled = _crossref_preflight_ok(timeout_s=3.5)
    crossref_start_ts = time.monotonic()
    crossref_budget_exhausted = False
    need_crossref_enrich = bool(crossref_enabled and enable_title_lookup)
    lib_citation_meta_map = _load_library_citation_meta_map(library_db_path)
    pdf_root_obj = Path(pdf_root).expanduser().resolve() if pdf_root else None

    start_cursor = 0
    if md_files:
        try:
            start_cursor = int(prev.get("next_cursor", 0) or 0) % len(md_files)
        except Exception:
            start_cursor = 0
        if start_cursor > 0:
            md_files = md_files[start_cursor:] + md_files[:start_cursor]

    docs_out: dict[str, dict] = {}
    docs_updated = 0
    docs_reused = 0
    refs_total = 0
    refs_with_doi = 0
    refs_crossref_ok = 0
    refs_source_map_ok = 0
    total_docs = len(md_files)

    def _emit_progress(stage: str, *, docs_done: int, current: str = "") -> None:
        if not callable(progress_cb):
            return
        try:
            progress_cb(
                {
                    "stage": str(stage or "").strip(),
                    "current": str(current or "").strip(),
                    "docs_done": int(max(0, docs_done)),
                    "docs_total": int(max(0, total_docs)),
                    "refs_total": int(max(0, refs_total)),
                    "refs_with_doi": int(max(0, refs_with_doi)),
                    "refs_crossref_ok": int(max(0, refs_crossref_ok)),
                    "refs_source_map_ok": int(max(0, refs_source_map_ok)),
                }
            )
        except Exception:
            return

    _emit_progress("prepare", docs_done=0, current="")

    for doc_i, p in enumerate(md_files, start=1):
        src_path = str(p.resolve())
        src_key = _norm_source_key(src_path)
        if not src_key:
            continue
        _emit_progress("doc_start", docs_done=max(0, doc_i - 1), current=p.name)

        try:
            sha1 = compute_file_sha1(p)
        except Exception:
            sha1 = ""

        prev_doc = prev_docs.get(src_key) if isinstance(prev_docs, dict) else None
        if (
            incremental
            and isinstance(prev_doc, dict)
            and str(prev_doc.get("sha1") or "") == str(sha1 or "")
            and isinstance(prev_doc.get("refs"), dict)
            and (
                (not need_crossref_enrich)
                or bool(prev_doc.get("crossref_enriched"))
            )
        ):
            docs_out[src_key] = prev_doc
            docs_reused += 1
            refs = prev_doc.get("refs")
            if isinstance(refs, dict):
                refs_total += len(refs)
                for rv in refs.values():
                    if not isinstance(rv, dict):
                        continue
                    if str(rv.get("doi") or "").strip():
                        refs_with_doi += 1
                    if bool(rv.get("crossref_ok")):
                        refs_crossref_ok += 1
                    mm = str(rv.get("match_method") or "").strip()
                    if mm.startswith("source_work_reference"):
                        refs_source_map_ok += 1
            _emit_progress("doc_done", docs_done=doc_i, current=p.name)
            continue

        md_head = _read_text_head(p, max_bytes=220_000)
        md_tail = _read_text_tail(p, max_bytes=1_500_000)
        ref_map = extract_references_map_from_md(md_tail)
        refs_obj: dict[str, dict] = {}
        unresolved_promising = 0
        crossref_active_source = bool(crossref_enabled)

        pdf_candidate = _lookup_pdf_for_md_doc(p, pdf_root_obj)
        source_doi = ""
        if pdf_candidate is not None:
            cm = lib_citation_meta_map.get(_norm_path_key(pdf_candidate))
            if isinstance(cm, dict):
                source_doi = _best_source_doi_from_citation_meta(cm, crossref_enabled=crossref_active_source)
        if not source_doi:
            source_doi = _extract_source_doi_from_md_head(md_head)
        if not source_doi:
            source_doi = _infer_source_doi_from_doc_hints(
                p,
                md_head,
                cache,
                crossref_enabled=crossref_active_source,
            )
        source_ref_rows = _load_source_reference_rows(
            source_doi,
            cache,
            crossref_enabled=crossref_active_source,
        )
        ref_nums_sorted = sorted(ref_map.keys())
        can_map_by_exact_order = bool(
            source_ref_rows
            and ref_nums_sorted
            and (len(ref_nums_sorted) == len(source_ref_rows))
            and (ref_nums_sorted == list(range(1, len(ref_nums_sorted) + 1)))
        )
        can_map_by_prefix_order = False
        if source_ref_rows and ref_nums_sorted and ref_nums_sorted and (ref_nums_sorted[0] == 1):
            prefix_n = min(len(source_ref_rows), len(ref_nums_sorted))
            if prefix_n > 0:
                want = list(range(1, prefix_n + 1))
                got = ref_nums_sorted[:prefix_n]
                can_map_by_prefix_order = (got == want)
        can_map_by_number_loose = False
        if source_ref_rows and ref_nums_sorted:
            try:
                in_range_n = sum(1 for x in ref_nums_sorted if 1 <= int(x) <= len(source_ref_rows))
            except Exception:
                in_range_n = 0
            if len(ref_nums_sorted) >= 8:
                in_range_ratio = in_range_n / max(1, len(ref_nums_sorted))
                can_map_by_number_loose = bool(in_range_ratio >= 0.75)

        for n in sorted(ref_map.keys()):
            raw = str(ref_map.get(n) or "").strip()
            if not raw:
                continue

            meta = None
            doi_hint = extract_first_doi(raw)
            promising_ref = bool(is_promising_reference_text(raw))

            # Skip obvious non-bibliographic list items (often numbered body text),
            # but only when we do not have source-reference rows to anchor by number.
            if (not promising_ref) and (not source_ref_rows):
                if (not str(doi_hint or "").strip()) and (len(raw.split()) >= 16):
                    continue

            if source_ref_rows:
                mapped_by_number = False
                mapped_exact_order = False
                mapped_loose_number = False
                mapped = None
                if can_map_by_exact_order or (can_map_by_prefix_order and int(n) <= len(source_ref_rows)):
                    idx = int(n) - 1
                    if 0 <= idx < len(source_ref_rows):
                        row = source_ref_rows[idx]
                        if isinstance(row, dict):
                            mapped = row
                            mapped_by_number = True
                            mapped_exact_order = True
                if not isinstance(mapped, dict):
                    mapped = _match_source_reference_by_number(int(n), raw, source_ref_rows)
                    if isinstance(mapped, dict):
                        mapped_by_number = True
                if (not isinstance(mapped, dict)) and can_map_by_number_loose:
                    idx = int(n) - 1
                    if 0 <= idx < len(source_ref_rows):
                        row = source_ref_rows[idx]
                        if isinstance(row, dict):
                            mapped = row
                            mapped_by_number = True
                            mapped_loose_number = True
                if not isinstance(mapped, dict):
                    mapped = _match_source_reference(raw, source_ref_rows)
                if isinstance(mapped, dict):
                    meta = {
                        "title": str(mapped.get("title") or "").strip(),
                        "authors": "",
                        "venue": str(mapped.get("venue") or "").strip(),
                        "year": str(mapped.get("year") or "").strip(),
                        "volume": str(mapped.get("volume") or "").strip(),
                        "issue": "",
                        "pages": str(mapped.get("pages") or "").strip(),
                        "doi": str(mapped.get("doi") or "").strip(),
                        "match_method": (
                            "source_work_reference_order_exact"
                            if mapped_exact_order
                            else (
                                "source_work_reference_numbered_loose"
                                if mapped_loose_number
                                else ("source_work_reference_numbered" if mapped_by_number else "source_work_reference")
                            )
                        ),
                        "match_score": 0.98,
                    }
                    refs_source_map_ok += 1

            if not isinstance(meta, dict):
                crossref_active_now = bool(
                    crossref_enabled and ((time.monotonic() - crossref_start_ts) <= float(max(5.0, crossref_time_budget_s)))
                )
                meta, doi_hint = _lookup_crossref_meta_for_entry(
                    raw,
                    cache,
                    crossref_enabled=crossref_active_now,
                    enable_title_lookup=bool(enable_title_lookup),
                )
                if crossref_enabled and (not crossref_budget_exhausted):
                    if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                        crossref_budget_exhausted = True
            doi = ""
            doi_url = ""
            if isinstance(meta, dict):
                doi = str(meta.get("doi") or "").strip()
            if not doi:
                doi = _clean_doi_for_url(doi_hint)
            if doi:
                doi_url = _doi_url(doi)

            # DOI backfill: when mapped/bibliographic metadata is sparse or polluted,
            # fetch canonical Crossref work metadata by DOI to stabilize title/authors/venue.
            if doi:
                try:
                    title0 = str(meta.get("title") or "").strip() if isinstance(meta, dict) else ""
                    authors0 = str(meta.get("authors") or "").strip() if isinstance(meta, dict) else ""
                    venue0 = str(meta.get("venue") or "").strip() if isinstance(meta, dict) else ""
                    year0 = str(meta.get("year") or "").strip() if isinstance(meta, dict) else ""
                    vol0 = str(meta.get("volume") or "").strip() if isinstance(meta, dict) else ""
                    pages0 = str(meta.get("pages") or "").strip() if isinstance(meta, dict) else ""
                    title_words = [w for w in re.split(r"\s+", title0) if w]
                    noisy_title = (not title0) or (len(title0) >= 200) or (len(title_words) >= 28)
                    needs_backfill = (not authors0) or (not venue0) or (not year0) or (noisy_title) or ((not vol0) and (not pages0))
                    if needs_backfill and crossref_enabled:
                        by_doi = fetch_best_crossref_meta(
                            query_title=("" if noisy_title else title0),
                            doi_hint=doi,
                            allow_title_only=True,
                            min_score=0.88,
                        )
                        if isinstance(by_doi, dict) and by_doi:
                            merged = dict(meta) if isinstance(meta, dict) else {}
                            if noisy_title or (not str(merged.get("title") or "").strip()):
                                merged["title"] = str(by_doi.get("title") or merged.get("title") or "").strip()
                            if not str(merged.get("authors") or "").strip():
                                merged["authors"] = str(by_doi.get("authors") or "").strip()
                            if not str(merged.get("venue") or "").strip():
                                merged["venue"] = str(by_doi.get("venue") or "").strip()
                            if not str(merged.get("year") or "").strip():
                                merged["year"] = str(by_doi.get("year") or "").strip()
                            if not str(merged.get("volume") or "").strip():
                                merged["volume"] = str(by_doi.get("volume") or "").strip()
                            if not str(merged.get("issue") or "").strip():
                                merged["issue"] = str(by_doi.get("issue") or "").strip()
                            if not str(merged.get("pages") or "").strip():
                                merged["pages"] = str(by_doi.get("pages") or "").strip()
                            if not str(merged.get("doi") or "").strip():
                                merged["doi"] = str(by_doi.get("doi") or "").strip()
                            merged["match_method"] = str(merged.get("match_method") or "doi_backfill")
                            mm0 = str(merged.get("match_method") or "")
                            if ("doi_backfill" not in mm0) and mm0:
                                merged["match_method"] = mm0 + "+doi_backfill"
                            meta = merged
                except Exception:
                    pass

            rec = {
                "num": int(n),
                "raw": raw,
                "doi": doi,
                "doi_url": doi_url,
                "title": str(meta.get("title") or "").strip() if isinstance(meta, dict) else "",
                "authors": str(meta.get("authors") or "").strip() if isinstance(meta, dict) else "",
                "venue": str(meta.get("venue") or "").strip() if isinstance(meta, dict) else "",
                "year": str(meta.get("year") or "").strip() if isinstance(meta, dict) else "",
                "volume": str(meta.get("volume") or "").strip() if isinstance(meta, dict) else "",
                "issue": str(meta.get("issue") or "").strip() if isinstance(meta, dict) else "",
                "pages": str(meta.get("pages") or "").strip() if isinstance(meta, dict) else "",
                "crossref_ok": bool(isinstance(meta, dict)),
                "match_method": str(meta.get("match_method") or "").strip() if isinstance(meta, dict) else "",
            }
            refs_obj[str(int(n))] = rec
            if promising_ref:
                has_resolved = bool(str(rec.get("doi") or "").strip() or bool(rec.get("crossref_ok")))
                if not has_resolved:
                    unresolved_promising += 1

        crossref_enriched_doc = (not need_crossref_enrich) or (int(unresolved_promising) <= 0)

        docs_out[src_key] = {
            "path": src_path,
            "name": p.name,
            "stem": p.stem.lower(),
            "sha1": sha1,
            "source_doi": source_doi,
            "crossref_enriched": bool(crossref_enriched_doc),
            "refs": refs_obj,
        }
        docs_updated += 1
        refs_total += len(refs_obj)
        for rv in refs_obj.values():
            if str(rv.get("doi") or "").strip():
                refs_with_doi += 1
            if bool(rv.get("crossref_ok")):
                refs_crossref_ok += 1
        _emit_progress("doc_done", docs_done=doc_i, current=p.name)

    _emit_progress("saving", docs_done=total_docs, current="")
    out_data = {
        "version": 1,
        "updated_at": time.time(),
        "doc_count": len(docs_out),
        "next_cursor": (int(start_cursor + 1) % max(1, len(md_files_raw))) if md_files_raw else 0,
        "docs": docs_out,
    }
    _save_json(out_dir / INDEX_FILE_NAME, out_data)
    _crossref_cache_save(out_dir, cache)
    _emit_progress("done", docs_done=total_docs, current="")

    return {
        "docs_total": len(md_files),
        "docs_indexed": len(docs_out),
        "docs_updated": int(docs_updated),
        "docs_reused": int(docs_reused),
        "refs_total": int(refs_total),
        "refs_with_doi": int(refs_with_doi),
        "refs_crossref_ok": int(refs_crossref_ok),
        "refs_source_map_ok": int(refs_source_map_ok),
        "crossref_enabled": 1 if bool(crossref_enabled) else 0,
        "crossref_budget_exhausted": 1 if bool(crossref_budget_exhausted) else 0,
    }
