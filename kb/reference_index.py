from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

import requests

from kb.citation_meta import (
    extract_first_doi,
    extract_first_author_family_hint,
    extract_year_hint,
    fetch_best_crossref_for_reference,
    fetch_best_crossref_meta,
    fetch_best_openalex_meta,
    fetch_crossref_references_by_doi,
    is_promising_reference_text,
    normalize_title_for_match,
)
from kb.store import atomic_write_json, compute_file_sha1
from kb.source_filters import is_excluded_source_path


INDEX_FILE_NAME = "references_index.json"
CROSSREF_CACHE_FILE_NAME = "crossref_cache.json"
REFERENCE_CATALOG_FILE_NAME = "reference_catalog.json"
REFERENCE_LOOKUP_VERSION = 12
# Increment this independently of metadata lookup behavior whenever Markdown
# reference parsing changes. It prevents unchanged documents from reusing refs
# produced by an older parser.
REFERENCE_PARSER_VERSION = 3

_REF_HEAD_RE = re.compile(
    r"^#{1,6}\s+(references(?:\s+and\s+(?:notes|links))?|bibliography)\b",
    re.IGNORECASE,
)
_REF_START_BRACKET_RE = re.compile(r"^\[(\d{1,4})\]\s*(.*\S)?\s*$")
_REF_START_DOT_RE = re.compile(r"^(\d{1,4})\.\s+(.*\S)?\s*$")
_REF_ANY_START_RE = re.compile(r"(?:\[\d{1,4}\]|\d{1,4}\.)\s+")
_REF_EMBED_BRACKET_SPLIT_RE = re.compile(r"\[\d{1,4}\]\s+")
_WRAPPED_PAGE_RANGE_TAIL_RE = re.compile(
    r"\b(?:pp?\.?|pages?)\s*(\d{1,5})\s*([-\u2013\u2014])\s*$",
    re.IGNORECASE,
)
_POST_REFERENCES_PLAIN_STOP_RE = re.compile(
    r"^(?:#{1,6}\s+)?(?:appendix|appendices|annex|"
    r"supplementary(?:\s+materials?)?|supplemental(?:\s+(?:information|materials?))?)\b",
    re.IGNORECASE,
)
_DOI_TRIM_RE = re.compile(r"^[ \t\r\n.,;:(){}\[\]<>]+|[ \t\r\n.,;:(){}\[\]<>]+$")
_SOURCE_DOI_HTTP_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
_SOURCE_DOI_RE = re.compile(r"\bdoi\s*:?\s*(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", re.IGNORECASE)
_QUOTED_TITLE_RE = re.compile(r"[\"“”]([^\"“”]{8,260})[\"“”]")
_YEAR_ANY_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_REFERENCE_TRUNCATION_HINT_RE = re.compile(
    r"(?:\.\.\.|…|\b(?:incomplete\s+(?:reference|entry|visible|text|line)|partially visible|not fully visible|unreadable|illegible)\b)",
    re.IGNORECASE,
)
_REFERENCE_TITLE_SIGNAL_RE = re.compile(
    r"\b("
    r"adaptive|algorithm|algorithms|application|applications|architecture|benchmark|camera|"
    r"compressed|compressive|communication|communications|computational|dataset|denoising|"
    r"detection|detector|detectors|fields|histogram|holography|image|imaging|information|"
    r"learning|mathematical|measurement|measurements|method|methods|microscopy|network|networks|neural|"
    r"optical|photon|photonic|photons|principle|principles|quantum|radiance|reconstruction|"
    r"recovery|sciences|sensing|signal|snapshot|spectral|splatting|stereo|texture|theory|"
    r"transform|transforms|uncertainty|video|wavelet|wavelets"
    r")\b",
    re.IGNORECASE,
)
_REFERENCE_PERIOD_PLACEHOLDER = "<KB_REF_PERIOD>"
_REFERENCE_PERIOD_ABBREVIATIONS = (
    "appl",
    "assoc",
    "adv",
    "biomed",
    "conf",
    "condens",
    "commun",
    "comput",
    "ed",
    "eds",
    "electron",
    "eng",
    "express",
    "fig",
    "imag",
    "inf",
    "int",
    "lett",
    "mag",
    "math",
    "matter",
    "nat",
    "no",
    "nucl",
    "opt",
    "phys",
    "process",
    "proc",
    "rev",
    "sci",
    "sel",
    "selected",
    "sig",
    "soc",
    "suppl",
    "technol",
    "theory",
    "top",
    "topics",
    "trans",
    "vol",
)
_COMPACT_REFERENCE_VENUE_RE = re.compile(
    r"\b("
    r"acm|appl|applied|ieee|j|journal|laser|light|nat|nature|opt|optics|"
    r"photonics|phys|physical|proc|proceedings|rev|review|sci|science|trans|transactions"
    r")\.?\b",
    re.IGNORECASE,
)


def _is_reference_map_post_references_stop_line(text: str) -> bool:
    st = (text or "").strip()
    if not st or _REF_HEAD_RE.match(st):
        return False
    if _POST_REFERENCES_PLAIN_STOP_RE.match(st):
        return True
    ref_start_match = bool(_REF_START_BRACKET_RE.match(st) or _REF_START_DOT_RE.match(st))
    if ref_start_match:
        return False
    if re.search(r"\bsupplement(?:ary|al)\s+(?:information|materials?)\b", st, re.IGNORECASE):
        return not re.search(r"\b(?:doi|arxiv|proc(?:eedings)?\.?)\b", st, re.IGNORECASE)
    if re.match(
        r"^[A-Z]\.\s+(?:appendix|additional|supplementary|supplemental|proof(?:s)?|derivation(?:s)?|"
        r"implementation(?:\s+details?)?|experiment(?:s)?|results?|ablation(?:s)?|details?|extra)\b",
        st,
        re.IGNORECASE,
    ):
        return True
    return False


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
        if is_excluded_source_path(str(p)):
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
        # Some publishers, notably Nature, place an unheaded reference list
        # before Methods/Extended Data. Start early enough to catch that shape,
        # but require a dense bibliographic sequence below to avoid body lists.
        scan_from = max(0, int(total * 0.20))
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
        cluster_lines: list[str] = []
        for j, n0 in zip(cand_idx, cand_num):
            if best_idx <= j <= (best_idx + 260):
                cluster_nums.append(int(n0))
                cluster_lines.append(str(_lines[j] or "").strip())
        if len(cluster_nums) < 8:
            return None
        if max(cluster_nums) < 8:
            return None
        promising_count = sum(1 for line in cluster_lines if is_promising_reference_text(line))
        if promising_count < max(5, int(len(cluster_lines) * 0.35)):
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
        # Only split embedded [n] markers.
        # Embedded "1." patterns frequently appear in body text (e.g., "Fig. 1.")
        # and should not be treated as reference starts.
        for m0 in _REF_EMBED_BRACKET_SPLIT_RE.finditer(s0):
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

    def _merge_wrapped_page_range_candidate(candidate_n: int, rest: str) -> bool:
        """Fold a page-range tail misread as a bracketed reference number."""
        if cur_n is None or (not cur_buf) or int(candidate_n) == int(cur_n) + 1:
            return False
        current_text = " ".join(str(x or "").strip() for x in cur_buf if str(x or "").strip())
        match = _WRAPPED_PAGE_RANGE_TAIL_RE.search(current_text)
        if not match:
            return False
        try:
            page_start = int(match.group(1))
            page_end = int(candidate_n)
        except Exception:
            return False
        if page_end <= page_start or page_end > page_start + 300:
            return False

        for idx in range(len(cur_buf) - 1, -1, -1):
            fragment = str(cur_buf[idx] or "").rstrip()
            if not fragment:
                continue
            cur_buf[idx] = re.sub(
                r"([-\u2013\u2014])\s*$",
                lambda m: f"{m.group(1)}{page_end}",
                fragment,
            )
            break
        if rest:
            cur_buf.append(rest)
        return True

    reference_tail = lines[ref_i + 1 :]

    def _page_after_marker_is_author_biography(raw_index: int) -> bool:
        page_lines: list[str] = []
        for candidate_raw in reference_tail[raw_index + 1 :]:
            candidate = str(candidate_raw or "").strip()
            if re.match(r"^<!--\s*kb_page:\s*\d+\s*-->$", candidate, re.IGNORECASE):
                break
            if candidate:
                page_lines.append(candidate)
        if any(
            _REF_START_BRACKET_RE.match(candidate) or _REF_START_DOT_RE.match(candidate)
            for candidate in page_lines
        ):
            return False
        return bool(
            re.search(
                r"(?:"
                r"^(?:\*\*|__)[^*_\n]{2,100}(?:\*\*|__)\s+(?:received|earned|obtained)\b"
                r"|\breceived\s+(?:his|her|their|a|the)\b[^\n]{0,120}\bdegree\b"
                r"|\b(?:his|her|their)\s+research\s+interests?\s+include\b"
                r")",
                "\n".join(page_lines),
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )

    def _running_header_before_next_reference(raw_index: int) -> bool:
        current_number = int(cur_n or max(out, default=0))
        if current_number <= 0:
            return False
        for candidate_raw in reference_tail[raw_index + 1 : raw_index + 8]:
            candidate = str(candidate_raw or "").strip()
            if not candidate or re.match(r"^<!--\s*kb_page:\s*\d+\s*-->$", candidate, re.IGNORECASE):
                continue
            match = _REF_START_BRACKET_RE.match(candidate) or _REF_START_DOT_RE.match(candidate)
            if not match:
                return False
            try:
                return int(match.group(1)) == current_number + 1
            except Exception:
                return False
        return False

    in_references = True
    for raw_index, raw in enumerate(reference_tail):
        for s in _split_embedded_ref_segments(raw):
            if not s:
                continue
            if _REF_HEAD_RE.match(s):
                _flush()
                in_references = True
                continue
            if not in_references:
                continue
            if re.match(r"^<!--\s*kb_page:\s*\d+\s*-->$", s, re.IGNORECASE):
                if (out or cur_n is not None) and _page_after_marker_is_author_biography(raw_index):
                    _flush()
                    in_references = False
                continue

            if (out or cur_n is not None) and _is_reference_map_post_references_stop_line(s):
                if _running_header_before_next_reference(raw_index):
                    continue
                _flush()
                in_references = False
                continue

            if re.match(r"^#{1,6}\s+\S+", s) and (not _REF_HEAD_RE.match(s)):
                if out:
                    _flush()
                    in_references = False
                    continue

            m = _REF_START_BRACKET_RE.match(s) or _REF_START_DOT_RE.match(s)
            if m:
                try:
                    candidate_n = int(m.group(1))
                except Exception:
                    continue
                rest = (m.group(2) or "").strip()
                if _merge_wrapped_page_range_candidate(candidate_n, rest):
                    continue
                _flush()
                cur_n = candidate_n
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
    max_n = nums[-1] if nums else 0
    # Typical references are dense and low-index; a huge max index strongly suggests OCR noise.
    has_large_gap_outlier = bool(
        total >= 15
        and max_n >= 120
        and max_n >= max(120, int(total * 2.2) + 20)
    )

    # If the list is mostly low-index refs but contains huge outliers,
    # cap indices by a data-driven upper bound.
    if (
        total >= 15
        and (core_n / max(1, total)) >= 0.70
        and (has_extreme or removed_year_like or has_large_gap_outlier)
    ):
        cap = max(80, min(500, int(total * 1.6) + 20))
        out = {n: t for n, t in out.items() if n <= cap}

    return out


def _reference_parse_confidence(entry_text: str) -> float:
    raw = str(entry_text or "").strip()
    if not raw:
        return 0.0
    score = 0.50
    word_count = len(re.findall(r"\S+", raw))
    if len(raw) >= 40:
        score += 0.16
    if word_count >= 8:
        score += 0.12
    if _YEAR_ANY_RE.search(raw):
        score += 0.08
    if extract_first_doi(raw):
        score += 0.09
    if raw.count(".") >= 2 or raw.count(",") >= 3:
        score += 0.06
    if _REFERENCE_TRUNCATION_HINT_RE.search(raw):
        score -= 0.28
    if len(raw) < 20 or word_count < 4:
        score -= 0.18
    return max(0.05, min(0.99, float(score)))


def build_reference_catalog_from_ref_map(
    ref_map: dict[int, str],
    *,
    source_path: str = "",
    source_name: str = "",
    source_sha1: str = "",
) -> dict:
    refs_in = ref_map if isinstance(ref_map, dict) else {}
    ref_nums = sorted(int(k) for k in refs_in.keys() if int(k) > 0)
    missing_numbers: list[int] = []
    continuity_status = "empty"
    if ref_nums:
        if ref_nums[0] <= ref_nums[-1]:
            want = set(range(ref_nums[0], ref_nums[-1] + 1))
            missing_numbers = sorted(int(n) for n in (want - set(ref_nums)))
        if missing_numbers:
            continuity_status = "gapped"
        elif ref_nums[0] != 1:
            continuity_status = "offset"
        else:
            continuity_status = "continuous"

    rows: list[dict] = []
    for n in ref_nums:
        text = str(refs_in.get(int(n)) or "").strip()
        if not text:
            continue
        rows.append(
            {
                "reference_number": int(n),
                "reference_text": text,
                "reference_entry_id": f"ref_{int(n):04d}",
                "parse_confidence": _reference_parse_confidence(text),
            }
        )

    return {
        "version": int(REFERENCE_PARSER_VERSION),
        "source_path": str(source_path or "").strip(),
        "source_name": str(source_name or "").strip(),
        "source_sha1": str(source_sha1 or "").strip(),
        "ref_count": int(len(rows)),
        "first_ref_num": int(ref_nums[0]) if ref_nums else 0,
        "max_ref_num": int(ref_nums[-1]) if ref_nums else 0,
        "tail_continuity_status": continuity_status,
        "missing_numbers": list(missing_numbers[:128]),
        "refs": rows,
    }


def build_reference_catalog_from_md(
    md_text: str,
    *,
    source_path: str = "",
    source_name: str = "",
    source_sha1: str = "",
) -> dict:
    ref_map = extract_references_map_from_md(md_text)
    return build_reference_catalog_from_ref_map(
        ref_map,
        source_path=source_path,
        source_name=source_name,
        source_sha1=source_sha1,
    )


def _reference_catalog_path_for_md(md_path: Path | str) -> Path:
    path = Path(str(md_path or "")).expanduser()
    return path.parent / REFERENCE_CATALOG_FILE_NAME


def load_reference_catalog_for_md(md_path: Path | str) -> dict:
    p = _reference_catalog_path_for_md(md_path)
    data = _load_json(p)
    if not isinstance(data, dict):
        return {}
    try:
        catalog_version = int(data.get("version") or 0)
    except Exception:
        catalog_version = 0
    if catalog_version < int(REFERENCE_PARSER_VERSION):
        return {}
    refs = data.get("refs")
    if not isinstance(refs, list):
        data["refs"] = []
    return data


def reference_catalog_to_map(catalog: dict | None) -> dict[int, str]:
    if not isinstance(catalog, dict):
        return {}
    refs = catalog.get("refs")
    if not isinstance(refs, list):
        return {}
    out: dict[int, str] = {}
    for item in refs:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("reference_number") or 0)
        except Exception:
            n = 0
        text = str(item.get("reference_text") or "").strip()
        if n <= 0 or not text:
            continue
        out[int(n)] = text
    return out


def persist_reference_catalog_for_md(
    md_path: Path | str,
    *,
    md_text: str | None = None,
    source_sha1: str = "",
) -> dict:
    path = Path(str(md_path or "")).expanduser()
    source_name = path.name
    source_path = str(path.resolve(strict=False))
    if md_text is None:
        try:
            md_text = path.read_text(encoding="utf-8")
        except Exception:
            md_text = ""
    catalog = build_reference_catalog_from_md(
        str(md_text or ""),
        source_path=source_path,
        source_name=source_name,
        source_sha1=source_sha1,
    )
    _save_json(_reference_catalog_path_for_md(path), catalog)
    return catalog


def _strip_reference_number(raw: str) -> str:
    s = str(raw or "").strip()
    s = re.sub(r"^\[(\d{1,4})\]\s*", "", s).strip()
    s = re.sub(r"^(\d{1,4})\.\s*", "", s).strip()
    return s


def _protect_reference_periods(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = re.sub(r"\b([A-Z])\.", rf"\1{_REFERENCE_PERIOD_PLACEHOLDER}", s)
    for token in _REFERENCE_PERIOD_ABBREVIATIONS:
        s = re.sub(
            rf"\b({re.escape(token)})\.",
            lambda m: f"{m.group(1)}{_REFERENCE_PERIOD_PLACEHOLDER}",
            s,
            flags=re.IGNORECASE,
        )
    return s


def _restore_reference_periods(text: str) -> str:
    return str(text or "").replace(_REFERENCE_PERIOD_PLACEHOLDER, ".")


def _clean_reference_markup_text(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = re.sub(r"<!--\s*kb_page:\s*\d+\s*-->", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _clean_reference_title_text(text: str) -> str:
    s = _clean_reference_markup_text(text).strip(" .;:,")
    if not s:
        return ""
    s = re.sub(r"\s*\([^()]{0,90},\s*(?:18|19|20)\d{2}\)\s*$", "", s).strip(" .;:,")
    s = re.sub(r"\s*,?\s*\((?:18|19|20)\d{2}\)\s*$", "", s).strip(" .;:,")
    s = re.sub(r"\s*,\s*(?:18|19|20)\d{2}\s*$", "", s).strip(" .;:,")
    return s[:260]


def _extract_reference_year_hint(text: str) -> str:
    year = extract_year_hint(str(text or ""))
    if year:
        return year
    raw = str(text or "")

    def _fix_ocr_year(match: re.Match[str]) -> str:
        value = str(match.group(0) or "")
        return value.translate(str.maketrans({"S": "5", "s": "5", "O": "0", "o": "0", "I": "1", "l": "1"}))

    for m in re.finditer(r"\b(?:18|19|20)[0-9SsoOIl]{2}\b", raw):
        fixed = _fix_ocr_year(m)
        if re.fullmatch(r"(?:18|19|20)\d{2}", fixed):
            return fixed
    return ""


def _looks_like_compact_titleless_reference(text: str) -> bool:
    t = _strip_reference_number(_clean_reference_markup_text(text))
    t = re.sub(r"https?://\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdoi\s*:?\s*10\.\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip(" .;:,")
    if not t or len(t) < 20:
        return False
    if not _YEAR_ANY_RE.search(t):
        return False
    venue_match = _COMPACT_REFERENCE_VENUE_RE.search(t)
    if not venue_match:
        return False
    signal_scope = t[: venue_match.start()].strip()
    if _REFERENCE_TITLE_SIGNAL_RE.search(signal_scope):
        return False
    yearish = re.search(
        r"\b(?:18|19|20)\d{2}\b[^\n]{0,60}\b[A-Za-z]?\d{1,8}[A-Za-z]?\b"
        r"(?:[^\n]{0,40}\b[A-Za-z]?\d{2,8}[A-Za-z]?\b)?",
        t,
    )
    if not yearish:
        return False
    comma_n = t.count(",")
    initial_n = len(re.findall(r"\b[A-Z]\.", t))
    return bool(comma_n >= 2 or initial_n >= 2)


def _looks_like_venue_volume_fragment(text: str) -> bool:
    t = _strip_reference_number(_clean_reference_markup_text(text))
    t = re.sub(r"https?://\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdoi\s*:?\s*10\.\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip(" .;:,")
    if not t or len(t) < 8 or len(t) > 180:
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) > 18:
        return False
    venue_match = _COMPACT_REFERENCE_VENUE_RE.search(t)
    if not venue_match:
        return False
    signal_scope = t[: venue_match.start()].strip(" .;:,")
    if signal_scope and _REFERENCE_TITLE_SIGNAL_RE.search(signal_scope):
        return False
    has_year = bool(_YEAR_ANY_RE.search(t))
    has_volume_page = bool(
        re.search(r"\b\d{1,4}\s*\(\s*\d{1,4}\s*\)\s*[:,]\s*[A-Za-z]?\d{1,8}[A-Za-z0-9\-]*", t)
        or re.search(r"\b\d{1,4}\s*,\s*[A-Za-z]?\d{1,8}[A-Za-z0-9\-]*", t)
        or re.search(r"\b\d{1,4}\s+[A-Za-z]?\d{2,8}[A-Za-z0-9\-]*\s*\((?:18|19|20)\d{2}\)", t)
        or re.search(r"\b(?:18|19|20)\d{2}\s+\d{1,4}\s+[A-Za-z]?\d{2,8}[A-Za-z0-9\-]*", t)
        or re.search(r"\b(?:pp|pages?)\.?\s*\d", t, flags=re.IGNORECASE)
    )
    return bool(has_year and has_volume_page)


def _reference_sentence_segments(text: str) -> list[str]:
    s = re.sub(r"\s{2,}", " ", str(text or "")).strip()
    if not s:
        return []
    protected = _protect_reference_periods(s)
    parts = [
        _restore_reference_periods(x).strip(" .;:")
        for x in re.split(r"\.\s+", protected)
        if _restore_reference_periods(x).strip(" .;:")
    ]
    return parts


def _looks_reference_author_segment(seg: str) -> bool:
    t = str(seg or "").strip(" .;:,")
    if not t:
        return False
    low = t.lower()
    if low.startswith(("in ", "in:", "proceedings", "journal of", "abstracts of")):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", t)
    if len(words) < 2 or len(words) > 80:
        return False
    comma_n = t.count(",")
    and_n = len(re.findall(r"\b(?:and|et al)\b|&", t, flags=re.IGNORECASE))
    init_n = len(re.findall(r"\b[A-Z]{1,3}\.?\b", t))
    cap_n = sum(1 for w in words if w[:1].isupper())
    cap_ratio = cap_n / max(1, len(words))
    family_initial_pairs = len(re.findall(r"\b[A-Z][A-Za-z'\-]{1,24}\s+[A-Z]{1,3}\.?\b", t))
    given_family_pairs = len(
        re.findall(r"\b[A-Z][a-z][A-Za-z'\-]+(?:\s+[A-Z]\.?)?\s+[A-Z][A-Za-z'\-]+\b", t)
    )
    has_connector = bool((comma_n >= 1) or (and_n >= 1) or (family_initial_pairs >= 2))
    return bool(
        cap_ratio >= 0.45
        and (
            (has_connector and (init_n >= 1 or family_initial_pairs >= 1 or given_family_pairs >= 1))
            or (family_initial_pairs >= 2)
            or (given_family_pairs >= 2 and and_n >= 1)
        )
    )


def _reference_title_looks_author_noise(title: str) -> bool:
    t = str(title or "").strip()
    if not t:
        return False
    if ":" in t:
        return False
    title_signal = bool(_REFERENCE_TITLE_SIGNAL_RE.search(t))
    if _looks_reference_author_segment(t):
        return not title_signal
    words = [w for w in re.split(r"\s+", t) if w]
    if 2 <= len(words) <= 8 and re.search(r"\band\b", t, flags=re.IGNORECASE):
        cap_n = sum(1 for w in words if w[:1].isupper())
        init_n = len(re.findall(r"\b[A-Z]{1,3}\.?\b", t))
        return bool((not title_signal) and (cap_n / max(1, len(words))) >= 0.55 and init_n >= 1)
    return False


def _clean_raw_reference_authors(text: str) -> str:
    s = str(text or "").strip(" .;:,")
    if not s:
        return ""
    s = re.sub(r"\b(?:ed|eds|editor|editors)\.?\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\((?:ed|eds|editor|editors)\.?\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:19|20)\d{2}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .;:,")
    if not _looks_reference_author_segment(s):
        if _looks_single_reference_author_segment(s):
            return s[:320]
        if re.search(r"\bet\s+al\.?$", s, flags=re.IGNORECASE) and re.search(r"[A-Za-z\u00C0-\u024F]", s):
            return s[:320]
        return ""
    return s[:320]


def _reference_authors_look_noisy(authors: str, *, raw_title: str = "") -> bool:
    a = _clean_reference_markup_text(str(authors or "")).strip(" .;:,")
    if not a:
        return False
    if _looks_like_venue_volume_fragment(a):
        return True
    title = _clean_reference_markup_text(str(raw_title or "")).strip(" .;:,")
    if title and len(title) >= 8 and title.lower() in a.lower() and len(a) >= len(title) + 16:
        return True
    words = [w for w in re.split(r"\s+", a) if w]
    if len(a) > 520 or len(words) > 80:
        return True
    if _REFERENCE_TITLE_SIGNAL_RE.search(a):
        lower_words = sum(1 for w in words if w[:1].islower())
        if len(words) >= 12 and lower_words >= 4:
            return True
    return False


def _split_inline_reference_author_title(seg: str) -> tuple[str, str]:
    t = _strip_reference_number(str(seg or "").strip(" .;:,"))
    if not t:
        return "", ""
    best: tuple[str, str] = ("", "")
    for m in re.finditer(r"(?:\b[A-Z]\.\s*){1,4}", t):
        authors = t[: m.end()].strip(" .;:,")
        title = t[m.end() :].strip(" .;:,")
        if not title:
            continue
        clean_authors = _clean_raw_reference_authors(authors)
        if clean_authors and _is_plausible_reference_title(title):
            best = (clean_authors, title[:260])
    return best


def _split_comma_reference_author_title_venue(text: str) -> tuple[str, str, str]:
    t = _strip_reference_number(_clean_reference_markup_text(text)).strip(" .;:,")
    if not t or "," not in t:
        return "", "", ""
    best: tuple[str, str, str] = ("", "", "")
    for m in re.finditer(r",", t):
        authors_part = t[: m.start()].strip(" .;:,")
        rest = t[m.end() :].strip(" .;:,")
        if not authors_part or not rest:
            continue
        if _REFERENCE_TITLE_SIGNAL_RE.search(authors_part):
            continue
        if re.match(r"^(?:and|&)\s+[A-Z]", rest):
            continue
        authors = _clean_raw_reference_authors(authors_part)
        if not authors and _looks_single_reference_author_segment(authors_part):
            authors = authors_part[:320]
        if not authors:
            continue
        title_part = rest
        tail = ""
        split = re.search(r",\s*(?=(?:in:?\s+)?(?:\*|[A-Z]))", rest)
        if not split:
            split = re.search(
                r",\s*(?=(?:physica|nat\.?|nature|sci\.?|science|opt\.?|j\.?|journal|"
                r"ieee|acm|proc\.?|proceedings)\b)",
                rest,
                flags=re.IGNORECASE,
            )
        if split:
            title_part = rest[: split.start()]
            tail = rest[split.end() :].strip(" .;:,")
        title = _clean_reference_title_text(title_part)
        if not _is_plausible_reference_title(title):
            continue
        venue = _extract_venue_from_reference_tail(tail)
        return authors, title, venue
    return best


def _looks_single_reference_author_segment(seg: str) -> bool:
    t = _strip_reference_number(_clean_reference_markup_text(seg)).strip(" .;:,")
    if not t or _REFERENCE_TITLE_SIGNAL_RE.search(t) or _YEAR_ANY_RE.search(t):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", t)
    if not (2 <= len(words) <= 5):
        return False
    cap_n = sum(1 for w in words if w[:1].isupper())
    return bool(cap_n == len(words))


def _extract_venue_from_reference_tail(tail: str) -> str:
    s = _clean_reference_markup_text(str(tail or "")).strip(" .;:,")
    if not s:
        return ""
    s = re.sub(r"https?://\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdoi\s*:?\s*10\.\S+.*$", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\barxiv\s*:?\s*\S+.*$", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip(" .;:,")
    if not s:
        return ""
    first_sentence = (_reference_sentence_segments(s) or [s])[0].strip(" .;:,")
    first_sentence = re.sub(r"^in:?\s+", "", first_sentence, flags=re.IGNORECASE).strip(" .;:,")
    first_sentence = re.sub(
        r"^(?:18|19|20)\d{2}\s+(?=(?:IEEE|ACM|Proceedings|Proc|International|Conference|Journal|Trans|IEICE)\b)",
        "",
        first_sentence,
        flags=re.IGNORECASE,
    ).strip(" .;:,")
    comma_parts = [p.strip(" .;:,") for p in first_sentence.split(",") if p.strip(" .;:,")]
    cand = comma_parts[0] if comma_parts else first_sentence
    cand = _clean_reference_markup_text(cand).strip(" .;:,")
    cand = re.sub(r"\b(?:pp|pages?)\.?\s*\d.*$", "", cand, flags=re.IGNORECASE).strip(" .;:,")
    cand = re.sub(r"\b(?:vol|volume|no|number)\.?\s*\d.*$", "", cand, flags=re.IGNORECASE).strip(" .;:,")
    cand = re.sub(r"\s+(?:\d{1,4}|[A-Za-z]?\d{3,8}[A-Za-z]?)\b.*$", "", cand).strip(" .;:,")
    if not cand or len(cand) > 180:
        return ""
    if re.fullmatch(r"(?:19|20)\d{2}", cand):
        return ""
    if len([w for w in re.split(r"\s+", cand) if w]) > 14 and (not _looks_like_venue_or_publisher(cand)):
        return ""
    return cand[:180]


def _fallback_meta_from_raw_reference(raw: str) -> dict[str, str]:
    s = _strip_reference_number(raw)
    if not s:
        return {}

    out: dict[str, str] = {}
    year = _extract_reference_year_hint(s)
    if year:
        out["year"] = year

    mq = _QUOTED_TITLE_RE.search(s)
    if mq:
        title = str(mq.group(1) or "").strip(" .;:,")
        if _is_plausible_reference_title(title):
            out["title"] = title[:260]
            out["authors"] = _clean_raw_reference_authors(s[: mq.start()])
            out["venue"] = _extract_venue_from_reference_tail(s[mq.end() :])
            out["match_method"] = "raw_meta"
            return {k: v for k, v in out.items() if str(v or "").strip()}

    my = re.match(r"(?P<authors>.+?)\s*\((?P<year>(?:19|20)\d{2})\)\s*(?P<tail>.+)$", s)
    if my:
        authors = _clean_raw_reference_authors(str(my.group("authors") or ""))
        tail = str(my.group("tail") or "").strip()
        segs = _reference_sentence_segments(tail)
        title = segs[0].strip(" .;:,") if segs else ""
        if authors:
            out["authors"] = authors
        out["year"] = str(my.group("year") or "").strip()
        if _is_plausible_reference_title(title):
            out["title"] = title[:260]
            out["venue"] = _extract_venue_from_reference_tail(". ".join(segs[1:]))
            out["match_method"] = "raw_meta"
            return {k: v for k, v in out.items() if str(v or "").strip()}

    mpub = re.match(r"(?P<head>.+?)\s*\((?P<publisher>[^()]{3,160})\)\.?\s*$", s)
    if mpub:
        head = str(mpub.group("head") or "").strip(" .;:,")
        publisher = str(mpub.group("publisher") or "").strip(" .;:,")
        publisherish = bool(
            re.search(
                r"\b(?:press|publisher|elsevier|springer|wiley|cambridge|academic|"
                r"prentice(?:-hall)?|american\s+mathematical\s+society|voss|kluwer|plenum)\b",
                publisher,
                flags=re.IGNORECASE,
            )
        )
        if publisherish:
            candidates: list[tuple[str, str]] = []
            head_parts = [p.strip(" .;:,") for p in head.rsplit(",", 1)]
            if len(head_parts) == 2:
                candidates.append((head_parts[0], head_parts[1]))
            surname_initial = re.match(
                r"^(?P<authors>[A-Z][A-Za-z'\-]+,\s*(?:[A-Z]\.\s*){1,4}"
                r"(?:\s*(?:&|and)\s*[A-Z][A-Za-z'\-]+,\s*(?:[A-Z]\.\s*){1,4})*)"
                r"\s+(?P<title>.+)$",
                head,
            )
            if surname_initial:
                candidates.append(
                    (
                        str(surname_initial.group("authors") or ""),
                        str(surname_initial.group("title") or ""),
                    )
                )
            for authors_raw, title_raw in candidates:
                authors = _clean_raw_reference_authors(authors_raw)
                title = _clean_reference_title_text(title_raw)
                if authors and _is_plausible_reference_title(title):
                    out["authors"] = authors
                    out["title"] = title[:260]
                    out["venue"] = publisher[:180]
                    out["match_method"] = "raw_meta"
                    return {k: v for k, v in out.items() if str(v or "").strip()}

    segs = _reference_sentence_segments(s)
    if len(segs) <= 1:
        comma_authors, comma_title, comma_venue = _split_comma_reference_author_title_venue(s)
        if comma_title:
            if comma_authors:
                out["authors"] = comma_authors
            out["title"] = comma_title
            if comma_venue:
                out["venue"] = comma_venue
            out["match_method"] = "raw_meta"
            return {k: v for k, v in out.items() if str(v or "").strip()}

    if segs:
        authors, title = _split_inline_reference_author_title(segs[0])
        if authors and title:
            out["authors"] = authors
            out["title"] = title
            out["venue"] = _extract_venue_from_reference_tail(". ".join(segs[1:]))
            out["match_method"] = "raw_meta"
            return {k: v for k, v in out.items() if str(v or "").strip()}
    if len(segs) >= 2 and re.search(r"\bet\s+al\.?$", segs[0], flags=re.IGNORECASE):
        authors = _clean_raw_reference_authors(segs[0])
        title = _clean_reference_title_text(segs[1])
        if authors and _is_plausible_reference_title(title):
            out["authors"] = authors
            out["title"] = title[:260]
            out["venue"] = _extract_venue_from_reference_tail(". ".join(segs[2:]))
            out["match_method"] = "raw_meta"
            return {k: v for k, v in out.items() if str(v or "").strip()}
    if len(segs) >= 2 and (_looks_reference_author_segment(segs[0]) or _looks_single_reference_author_segment(segs[0])):
        title = _clean_reference_title_text(segs[1])
        if _is_plausible_reference_title(title):
            out["authors"] = _clean_raw_reference_authors(segs[0])
            if not out["authors"] and _looks_single_reference_author_segment(segs[0]):
                out["authors"] = _strip_reference_number(_clean_reference_markup_text(segs[0])).strip(" .;:,")
            out["title"] = title[:260]
            out["venue"] = _extract_venue_from_reference_tail(". ".join(segs[2:]))
            out["match_method"] = "raw_meta"
            return {k: v for k, v in out.items() if str(v or "").strip()}

    if len(segs) > 1:
        comma_authors, comma_title, comma_venue = _split_comma_reference_author_title_venue(s)
        if comma_title:
            if comma_authors:
                out["authors"] = comma_authors
            out["title"] = comma_title
            if comma_venue:
                out["venue"] = comma_venue
            out["match_method"] = "raw_meta"
            return {k: v for k, v in out.items() if str(v or "").strip()}

    if _looks_like_compact_titleless_reference(s):
        out["match_method"] = "raw_meta"
        return {k: v for k, v in out.items() if str(v or "").strip()}

    title = _extract_query_title(s).strip(" .;:,")
    if _is_plausible_reference_title(title):
        out["title"] = title[:260]
        out["match_method"] = "raw_meta"
    return {k: v for k, v in out.items() if str(v or "").strip()}


def _merge_raw_reference_meta(meta: dict | None, raw_meta: dict[str, str]) -> dict | None:
    if not isinstance(raw_meta, dict) or not raw_meta:
        return dict(meta) if isinstance(meta, dict) else None
    merged = dict(meta) if isinstance(meta, dict) else {}
    changed = False

    for key in ("year", "volume", "issue", "pages", "doi"):
        value = str(raw_meta.get(key) or "").strip()
        if value and not str(merged.get(key) or "").strip():
            merged[key] = value
            changed = True

    raw_title = str(raw_meta.get("title") or "").strip()
    cur_title = str(merged.get("title") or "").strip()
    raw_authors = str(raw_meta.get("authors") or "").strip()
    cur_authors = str(merged.get("authors") or "").strip()
    if raw_authors and (
        not cur_authors
        or _reference_authors_look_noisy(cur_authors, raw_title=cur_title or raw_title)
    ):
        merged["authors"] = raw_authors
        changed = True

    raw_venue = str(raw_meta.get("venue") or "").strip()
    cur_venue = str(merged.get("venue") or "").strip()
    if raw_venue and (not cur_venue or _should_replace_venue_with_external(cur_venue, raw_venue)):
        merged["venue"] = raw_venue
        changed = True

    changed_title = False
    if raw_title:
        should_replace = (
            not cur_title
            or _reference_title_looks_author_noise(cur_title)
            or _looks_like_venue_volume_fragment(cur_title)
            or (not _is_plausible_reference_title(cur_title))
        )
        if should_replace:
            merged["title"] = raw_title
            changed = True
            changed_title = True

    if changed:
        base_mm = str(merged.get("match_method") or "").strip()
        raw_mm = str(raw_meta.get("match_method") or "raw_meta").strip()
        if changed_title and raw_mm and "raw_title" not in raw_mm:
            raw_mm = f"raw_title+{raw_mm}"
        if not base_mm:
            merged["match_method"] = raw_mm
        elif raw_mm and raw_mm not in base_mm:
            merged["match_method"] = f"{base_mm}+{raw_mm}"
    return merged if merged else None


def _reference_match_method_has_external_resolution(match_method: str, doi: str = "") -> bool:
    if _clean_doi_for_url(str(doi or "")):
        return True
    tokens = {p.strip().lower() for p in str(match_method or "").split("+") if p.strip()}
    if any(token.startswith("source_work_reference") for token in tokens):
        return True
    if any(token.startswith("openalex") or token.startswith("semantic_scholar") for token in tokens):
        return True
    return bool(tokens.intersection({"bibliographic", "title", "doi", "doi_backfill"}))


def _extract_query_title(entry: str) -> str:
    s = (entry or "").strip()
    if not s:
        return ""
    s = _strip_reference_number(s)
    s = re.sub(r"https?://\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdoi\s*:?\s*10\.\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\barxiv\s*:?\s*\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return ""

    # Prefer explicit quoted title segments when present.
    mq = _QUOTED_TITLE_RE.search(s)
    if mq:
        cand_q = str(mq.group(1) or "").strip(" .;:,")
        if cand_q:
            return cand_q[:260]

    def _looks_author_segment(seg: str) -> bool:
        return _looks_reference_author_segment(seg) or _looks_single_reference_author_segment(seg)

    def _looks_venue_segment(seg: str) -> bool:
        t = str(seg or "").strip().lower()
        if not t:
            return False
        if _looks_like_venue_volume_fragment(seg):
            return True
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

    protected_segs = _reference_sentence_segments(s)
    if protected_segs:
        inline_authors, inline_title = _split_inline_reference_author_title(protected_segs[0])
        if inline_authors and inline_title:
            return inline_title[:260]
    if len(protected_segs) >= 2 and _looks_author_segment(protected_segs[0]):
        cand0 = protected_segs[1].strip(" .;:,")
        if cand0 and (not _looks_author_segment(cand0)) and (not _looks_venue_segment(cand0)):
            return cand0[:260]

    _, comma_title, _ = _split_comma_reference_author_title_venue(s)
    if comma_title:
        return comma_title[:260]

    # Heuristic 1: title often lies between first and second period.
    m_title = re.search(r"^[^.]{4,260}\.\s*([^.]{8,260})\.", s)
    if m_title:
        cand = (m_title.group(1) or "").strip(" .;:")
        if cand and (not _looks_author_segment(cand)) and (not _looks_venue_segment(cand)):
            return cand[:260]

    # Heuristic 2: choose best scored segment among sentence-like chunks.
    segs = protected_segs or [x.strip(" .;:") for x in re.split(r"\.\s+", s) if x.strip(" .;:")]
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


_TITLE_LOOKUP_VENUE_TOKEN_RE = re.compile(
    r"\b("
    r"journal|transactions?|conference|proceedings|workshop|symposium|letters|express|photonics|"
    r"communications?|science|nature|opt(?:ics)?|phys(?:ics)?|pattern analysis|machine intelligence|"
    r"trans|proc|ieee|acm|appl|intell|theory|"
    r"cvpr|iccv|eccv|neurips|icml|iclr|aaai|ijcai|kdd|sigir|arxiv"
    r")\b",
    flags=re.I,
)


def _looks_like_venue_or_publisher(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return True
    if t.startswith(("in:", "in ", "proceedings", "abstracts of", "workshop", "conference")):
        return True
    if re.search(r"\b(pp?|vol|no)\.?\s*\d", t):
        return True
    if re.search(r"\b\d+\(\d+\)\s*:\s*\d", t):
        return True
    venueish = (
        "ieee",
        "acm",
        "springer",
        "wiley",
        "press",
        "university",
        "journal",
        "transactions",
        "proceedings",
        "conference",
        "workshop",
    )
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) <= 8 and any(re.search(rf"\b{re.escape(v)}\b", t) for v in venueish):
        return True
    return False


def _is_plausible_reference_title(text: str) -> bool:
    t = str(text or "").strip(" .;:,")
    if not t:
        return False
    if _looks_like_compact_titleless_reference(t):
        return False
    if _looks_like_venue_volume_fragment(t):
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(t) < 8 or len(t) > 260:
        return False
    if len(words) < 2 or len(words) > 32:
        return False
    if _reference_title_looks_author_noise(t):
        return False
    explicit_venue_title = bool(re.search(r"\b(?:journal|transactions|proceedings|conference)\b", t, flags=re.IGNORECASE))
    if _looks_like_venue_or_publisher(t) and (explicit_venue_title or not _REFERENCE_TITLE_SIGNAL_RE.search(t)):
        return False
    if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", t)) < 2:
        return False
    return True


def _fallback_title_from_raw_reference(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s = re.sub(r"^\[(\d{1,4})\]\s*", "", s).strip()
    s = re.sub(r"^(\d{1,4})\.\s*", "", s).strip()

    mq = _QUOTED_TITLE_RE.search(s)
    if mq:
        q = str(mq.group(1) or "").strip(" .;:,")
        if _is_plausible_reference_title(q):
            return q[:260]

    # Common citation form: Authors (YEAR) Title. Venue...
    my = re.search(r"\((?:19|20)\d{2}\)\s*([^.]{6,260})\.", s)
    if my:
        y = str(my.group(1) or "").strip(" .;:,")
        if _is_plausible_reference_title(y):
            return y[:260]

    t0 = _extract_query_title(s).strip(" .;:,")
    if _is_plausible_reference_title(t0):
        return t0[:260]
    return ""


def _should_try_title_lookup(entry: str, title_hint: str) -> bool:
    raw = str(entry or "").strip()
    title = str(title_hint or "").strip()
    if not raw or not title:
        return False
    if is_promising_reference_text(raw):
        return True

    title_words = [w for w in re.split(r"\s+", title) if w]
    if len(title) < 12 or len(title_words) < 2 or len(title_words) > 24:
        return False

    low = raw.lower()
    has_author_shape = bool((" and " in low) or (" et al" in low) or ("&" in raw) or (raw.count(",") >= 2))
    has_venue_shape = bool(_TITLE_LOOKUP_VENUE_TOKEN_RE.search(raw))
    has_title_sentence = bool(re.search(r"\.\s+[^.]{8,260}(?:\.\s+|,\s+[A-Z])", raw))
    has_quoted_title = bool(_QUOTED_TITLE_RE.search(raw))
    title_only_ok = bool(_TITLE_ONLY_NON_ARTICLE_RE.search(" ".join([raw, title])))
    has_year = bool(_extract_reference_year_hint(raw))
    if has_quoted_title and has_author_shape and len(title_words) >= 2:
        return True
    if (has_title_sentence or title_only_ok) and has_author_shape and has_year:
        return True
    return bool((has_title_sentence or has_quoted_title) and has_author_shape and has_venue_shape)


def _reference_meta_is_sparse(meta: dict | None) -> bool:
    if not isinstance(meta, dict):
        return True
    title = str(meta.get("title") or "").strip()
    authors = str(meta.get("authors") or "").strip()
    venue = str(meta.get("venue") or "").strip()
    year = str(meta.get("year") or "").strip()
    doi = _clean_doi_for_url(str(meta.get("doi") or ""))
    title_words = [w for w in re.split(r"\s+", title) if w]
    noisy_title = (not title) or (len(title) >= 200) or (len(title_words) >= 28)
    weak_authors = (not authors) or (len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", authors)) < 2)
    return bool((not doi) or noisy_title or weak_authors or (not venue) or (not year))


def _should_replace_venue_with_external(current: str, new: str) -> bool:
    cur = str(current or "").strip()
    nxt = str(new or "").strip()
    if not nxt:
        return False
    if not cur:
        return True
    cur_norm = cur.lower().strip(" .;:,")
    cur_words = [w for w in re.split(r"\s+", cur) if w]
    new_words = [w for w in re.split(r"\s+", nxt) if w]
    generic_short = cur_norm in {"press", "publisher", "journal", "conference", "proceedings"}
    if generic_short and len(nxt) > len(cur):
        return True
    return bool(len(cur_words) <= 2 and len(new_words) > len(cur_words) and len(nxt) >= len(cur) + 6)


def _merge_reference_meta(base: dict | None, supplement: dict | None) -> dict | None:
    if not isinstance(supplement, dict):
        return dict(base) if isinstance(base, dict) else None
    if not isinstance(base, dict):
        return dict(supplement)

    merged = dict(base)
    base_doi = _clean_doi_for_url(str(merged.get("doi") or ""))
    supp_doi = _clean_doi_for_url(str(supplement.get("doi") or ""))
    if base_doi and supp_doi and (base_doi.lower() != supp_doi.lower()):
        return merged

    def _fill(
        key: str,
        *,
        prefer_longer: bool = False,
        allow_noisy_replace: bool = False,
        prefer_better_venue: bool = False,
    ) -> None:
        cur = str(merged.get(key) or "").strip()
        new = str(supplement.get(key) or "").strip()
        if not new:
            return
        if not cur:
            merged[key] = new
            return
        if allow_noisy_replace:
            cur_words = [w for w in re.split(r"\s+", cur) if w]
            cur_noisy = (
                (len(cur) >= 200)
                or (len(cur_words) >= 28)
                or _reference_title_looks_author_noise(cur)
                or _looks_like_venue_volume_fragment(cur)
                or (not _is_plausible_reference_title(cur))
            )
            if cur_noisy and _is_plausible_reference_title(new):
                merged[key] = new
                return
        if prefer_better_venue and _should_replace_venue_with_external(cur, new):
            merged[key] = new
            return
        if prefer_longer and len(new) > len(cur):
            merged[key] = new

    _fill("doi")
    _fill("title", allow_noisy_replace=True)
    _fill("authors")
    _fill("venue", prefer_better_venue=True)
    _fill("year")
    _fill("volume")
    _fill("issue")
    _fill("pages", prefer_longer=True)

    base_mm = str(merged.get("match_method") or "").strip()
    supp_mm = str(supplement.get("match_method") or "").strip()
    if supp_mm:
        if not base_mm:
            merged["match_method"] = supp_mm
        elif supp_mm not in base_mm:
            merged["match_method"] = f"{base_mm}+{supp_mm}"
    return merged


def _assess_doc_crossref_enrichment(refs: dict[str, dict] | None) -> tuple[int, int]:
    unresolved_promising = 0
    sparse_promising = 0
    if not isinstance(refs, dict):
        return unresolved_promising, sparse_promising

    for rec in refs.values():
        if not isinstance(rec, dict):
            continue
        raw = str(rec.get("raw") or "").strip()
        if (not raw) or (not is_promising_reference_text(raw)):
            continue
        doi = _clean_doi_for_url(str(rec.get("doi") or ""))
        crossref_ok = bool(rec.get("crossref_ok"))
        title = str(rec.get("title") or "").strip()
        authors = str(rec.get("authors") or "").strip()
        if (not doi) and (not crossref_ok):
            unresolved_promising += 1
            continue
        if (not title) or (not authors):
            sparse_promising += 1

    return int(unresolved_promising), int(sparse_promising)


def _doc_crossref_enriched(refs: dict[str, dict] | None) -> bool:
    unresolved_promising, sparse_promising = _assess_doc_crossref_enrichment(refs)
    return bool((int(unresolved_promising) <= 0) and (int(sparse_promising) <= 0))


def _reference_sync_retry_ttl_s() -> float:
    raw = str(os.environ.get("KB_REF_SYNC_RETRY_TTL_S") or "").strip()
    try:
        value = float(raw) if raw else 24.0 * 60.0 * 60.0
    except Exception:
        value = 24.0 * 60.0 * 60.0
    return float(max(300.0, value))


def _doc_crossref_retry_due(prev_doc: dict[str, Any] | None, *, now_ts: float | None = None) -> bool:
    if not isinstance(prev_doc, dict):
        return True
    try:
        last_attempt = float(prev_doc.get("crossref_last_attempt_at") or 0.0)
    except Exception:
        last_attempt = 0.0
    if last_attempt <= 0.0:
        return True
    now = float(time.time() if now_ts is None else now_ts)
    return bool((now - last_attempt) >= _reference_sync_retry_ttl_s())


def _doc_ref_quality_tuple(refs: dict[str, dict] | None) -> tuple[int, int, int, int, int, int, int]:
    if not isinstance(refs, dict):
        return (-10**9, -10**9, -10**9, -10**9, -10**9, -10**9, -10**9)
    unresolved = 0
    doi_n = 0
    cross_n = 0
    authors_n = 0
    title_n = 0
    venue_n = 0
    pages_n = 0
    for rec in refs.values():
        if not isinstance(rec, dict):
            continue
        doi = _clean_doi_for_url(str(rec.get("doi") or ""))
        cross_ok = bool(rec.get("crossref_ok"))
        if doi:
            doi_n += 1
        if cross_ok:
            cross_n += 1
        if _reference_has_usable_authors(str(rec.get("authors") or "")):
            authors_n += 1
        if _reference_has_usable_title(str(rec.get("title") or "")):
            title_n += 1
        if _reference_has_usable_venue(str(rec.get("venue") or "")):
            venue_n += 1
        if str(rec.get("pages") or "").strip():
            pages_n += 1
        if (not doi) and (not cross_ok):
            unresolved += 1
    # Larger tuple is better.
    return (-int(unresolved), int(doi_n), int(cross_n), int(authors_n), int(title_n), int(venue_n), int(pages_n))


def _prefer_previous_doc_refs(prev_refs: dict[str, dict] | None, new_refs: dict[str, dict] | None) -> bool:
    return bool(_doc_ref_quality_tuple(prev_refs) > _doc_ref_quality_tuple(new_refs))


def _doc_refs_have_parser_generated_metadata(refs: dict[str, dict] | None) -> bool:
    if not isinstance(refs, dict):
        return False
    for rec in refs.values():
        if not isinstance(rec, dict):
            continue
        tokens = {p.strip().lower() for p in str(rec.get("match_method") or "").split("+") if p.strip()}
        if tokens.intersection({"raw_meta", "raw_title"}):
            return True
    return False


def _reference_refs_missing_numbers(refs: dict[str, dict] | None) -> list[int]:
    if not isinstance(refs, dict):
        return []
    nums: set[int] = set()
    for key in refs.keys():
        try:
            n = int(str(key).strip())
        except Exception:
            continue
        if n > 0:
            nums.add(n)
    if len(nums) < 2:
        return []
    first = min(nums)
    last = max(nums)
    if first != 1 or last <= first:
        return []
    return sorted(set(range(first, last + 1)) - nums)


def _previous_doc_refs_stale_against_catalog(
    prev_doc: dict[str, Any] | None,
    prev_refs: dict[str, dict] | None,
) -> bool:
    if not isinstance(prev_doc, dict) or not isinstance(prev_refs, dict):
        return False
    try:
        catalog_count = int(prev_doc.get("reference_catalog_ref_count") or 0)
    except Exception:
        catalog_count = 0
    catalog_missing_raw = prev_doc.get("reference_catalog_missing_numbers")
    catalog_missing: list[int] = []
    if isinstance(catalog_missing_raw, list):
        for item in catalog_missing_raw:
            try:
                n = int(item)
            except Exception:
                continue
            if n > 0:
                catalog_missing.append(n)
    refs_missing = _reference_refs_missing_numbers(prev_refs)
    if catalog_count > len(prev_refs):
        return True
    if (not catalog_missing) and refs_missing:
        return True
    if catalog_missing and refs_missing and len(refs_missing) > len(catalog_missing):
        return True
    return False


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


REFERENCE_METADATA_STATUS_CODES = (
    "complete",
    "crossref_enriched",
    "bibliographic_ready",
    "doi_sparse_refreshable",
    "title_lookup_retryable",
    "non_article_source_ok",
    "no_doi_expected",
    "truncated_reference",
    "low_confidence_match",
)
REFERENCE_MISSING_REASON_CODES = (
    "doi_sparse_refreshable",
    "title_lookup_retryable",
    "non_article_source_ok",
    "no_doi_expected",
    "truncated_reference",
    "low_confidence_match",
)

_NON_ARTICLE_SOURCE_RE = re.compile(
    r"\b("
    r"book|chapter|edited\s+volume|monograph|dataset|data\s+set|software|code|github|"
    r"protocol|protocols\.io|standard|iso|technical\s+report|report|white\s+paper|"
    r"manual|thesis|dissertation|patent|preprint|arxiv|bioRxiv|medRxiv|zenodo|figshare|"
    r"institution|university|publisher|press|repository|springer|wiley|academic|plenum|"
    r"mcgraw(?:-hill)?|artech|kluwer|elsevier|american\s+mathematical\s+society|"
    r"prentice(?:-hall)?|voss"
    r")\b",
    flags=re.IGNORECASE,
)
_WEB_SOURCE_RE = re.compile(
    r"\b(?:https?://(?!(?:dx\.)?doi\.org)|www\.|available\s+at|accessed\s+(?:on|at)|web\s*site|website)\b",
    flags=re.IGNORECASE,
)
_NO_DOI_EXPECTED_RE = re.compile(
    r"\b("
    r"arxiv|preprint|book|chapter|monograph|dataset|data\s+set|software|code|github|"
    r"protocol|standard|technical\s+report|report|white\s+paper|manual|thesis|"
    r"dissertation|patent|repository|publisher|press|handbook|textbook|url|"
    r"available\s+at|accessed\s+(?:on|at)|web\s*site|website|springer|wiley|academic|"
    r"plenum|mcgraw(?:-hill)?|artech|kluwer|elsevier|american\s+mathematical\s+society|"
    r"prentice(?:-hall)?|voss"
    r")\b|https?://(?!doi\.org)",
    flags=re.IGNORECASE,
)
_PUBLISHER_SOURCE_RE = re.compile(
    r"\b(?:press|publisher|elsevier|springer(?:-verlag)?|wiley|academic|cambridge|"
    r"american\s+mathematical\s+society|prentice(?:-hall)?|voss|kluwer|plenum|"
    r"mcgraw(?:-hill)?|artech)\b",
    flags=re.IGNORECASE,
)
_TITLE_ONLY_NON_ARTICLE_RE = re.compile(
    r"\b(?:technical\s+report|report|preprint|arxiv|dataset|data\s+set|software|"
    r"toolbox|benchmark|challenge|plenoptic\s+camera)\b",
    flags=re.IGNORECASE,
)


def _float_meta_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _reference_has_usable_title(title: str) -> bool:
    t = str(title or "").strip()
    if not t:
        return False
    if _looks_like_compact_titleless_reference(t):
        return False
    if _looks_like_venue_volume_fragment(t):
        return False
    if re.fullmatch(r"references?\s*\d*", t, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"reference\s+\d{1,4}", t, flags=re.IGNORECASE):
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(t) < 8 or len(t) > 260 or len(words) < 2 or len(words) > 36:
        return False
    if _reference_title_looks_author_noise(t):
        return False
    explicit_venue_title = bool(re.search(r"\b(?:journal|transactions|proceedings|conference)\b", t, flags=re.IGNORECASE))
    if _looks_like_venue_or_publisher(t) and (len(words) <= 3 or explicit_venue_title):
        return False
    return bool(len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", t)) >= 2)


def _reference_has_usable_authors(authors: str) -> bool:
    a = str(authors or "").strip()
    if not a:
        return False
    if _reference_authors_look_noisy(a):
        return False
    low = a.lower().strip(" .;:,[]()")
    if not low or "unknown author" in low or low in {"unknown", "et al", "et al."}:
        return False
    return bool(len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", a)) >= 1)


def _reference_has_usable_venue(venue: str) -> bool:
    v = str(venue or "").strip()
    if not v:
        return False
    if v.lower().strip(" .;:,") in {"unknown", "journal", "conference", "references", "reference"}:
        return False
    return True


def _reference_has_usable_year(year: str) -> bool:
    return bool(re.fullmatch(r"(?:18|19|20)\d{2}", str(year or "").strip()))


def _reference_looks_source_truncated(raw: str) -> bool:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        return False
    body = _strip_reference_number(text).strip()
    if not body:
        return False
    if re.search(r"\bvol\.?\s*$", body, flags=re.IGNORECASE):
        return True
    if len(body) <= 90 and not _extract_reference_year_hint(body):
        if re.search(r"\b(?:app|appl|proc|conf|journal|trans)\.?\s*$", body, flags=re.IGNORECASE):
            return True
        if body.count('"') % 2 == 1 or body.count("“") != body.count("”"):
            return True
    if len(body) <= 160 and not _extract_reference_year_hint(body):
        if re.search(r"[“\"].{8,140}$", body) and not re.search(r"[”\"]\s*(?:,|\bin\b|\w)", body):
            return True
    return False


def _reference_has_external_metadata(rec: Mapping[str, Any], match_method: str, doi: str) -> bool:
    if bool(rec.get("crossref_ok")):
        return True
    tokens = {p.strip().lower() for p in str(match_method or "").split("+") if p.strip()}
    if any(token.startswith("source_work_reference") for token in tokens):
        return True
    if any(token.startswith("openalex") or token.startswith("semantic_scholar") for token in tokens):
        return True
    return bool(tokens.intersection({"bibliographic", "title", "doi", "doi_backfill", "shelf_metadata_repair"}))


def classify_reference_metadata(
    rec: Mapping[str, Any] | None,
    *,
    enable_title_lookup: bool = True,
) -> dict[str, Any]:
    """Classify one reference into a sync/backfill action bucket."""
    data: Mapping[str, Any] = rec if isinstance(rec, Mapping) else {}
    raw = str(data.get("raw") or data.get("cite_fmt") or "").strip()
    doi = _clean_doi_for_url(str(data.get("doi") or data.get("doi_url") or ""))
    if not doi and raw:
        doi = _clean_doi_for_url(extract_first_doi(raw))
    title = str(data.get("title") or "").strip()
    authors = str(data.get("authors") or "").strip()
    venue = str(data.get("venue") or "").strip()
    year = str(data.get("year") or "").strip()
    match_method = str(data.get("match_method") or "").strip()
    parse_confidence = _float_meta_value(data.get("parse_confidence"), _reference_parse_confidence(raw))
    match_score = _float_meta_value(data.get("match_score"), 1.0 if match_method else 0.0)

    has_title = _reference_has_usable_title(title)
    has_authors = _reference_has_usable_authors(authors)
    has_venue = _reference_has_usable_venue(venue)
    has_year = _reference_has_usable_year(year)
    text_for_kind = " ".join([raw, title, venue, match_method])
    publisher_signal = bool(_PUBLISHER_SOURCE_RE.search(text_for_kind))
    title_only_ok_signal = bool(_TITLE_ONLY_NON_ARTICLE_RE.search(text_for_kind))
    non_article_signal = bool(_NON_ARTICLE_SOURCE_RE.search(text_for_kind) or publisher_signal or title_only_ok_signal)
    no_doi_expected_signal = bool(_NO_DOI_EXPECTED_RE.search(text_for_kind) or publisher_signal or title_only_ok_signal)
    web_source_signal = bool(_WEB_SOURCE_RE.search(text_for_kind))
    external_metadata = _reference_has_external_metadata(data, match_method, doi)

    missing_fields: list[str] = []
    if not doi:
        missing_fields.append("doi")
    if not has_title:
        missing_fields.append("title")
    if not has_authors:
        missing_fields.append("authors")
    if not has_venue:
        missing_fields.append("venue")
    if not has_year:
        missing_fields.append("year")

    status = "low_confidence_match"
    reason = "low_confidence_match"
    action = "source_repair"
    ready = False

    core_complete = bool(doi and has_title and has_authors and has_venue and has_year)
    external_ready = bool(
        external_metadata
        and has_year
        and (
            (has_title and (has_authors or has_venue))
            or (has_authors and has_venue)
        )
    )
    if core_complete:
        status = "crossref_enriched" if external_metadata else "complete"
        reason = ""
        action = "none"
        ready = True
    elif external_ready:
        status = "crossref_enriched"
        reason = ""
        action = "none"
        ready = True
    else:
        truncated = bool(_REFERENCE_TRUNCATION_HINT_RE.search(raw) or parse_confidence < 0.32 or _reference_looks_source_truncated(raw))
        if (not doi) and web_source_signal:
            status = "no_doi_expected"
            reason = "no_doi_expected"
            action = "non_article_ok"
            ready = True
        elif truncated:
            status = "truncated_reference"
            reason = "truncated_reference"
            action = "source_repair"
        elif doi and _reference_meta_is_sparse(dict(data)):
            status = "doi_sparse_refreshable"
            reason = "doi_sparse_refreshable"
            action = "auto_backfill"
        elif (
            non_article_signal
            and has_title
            and (has_year or publisher_signal or title_only_ok_signal)
            and (has_venue or has_authors or no_doi_expected_signal)
        ):
            status = "non_article_source_ok"
            reason = "no_doi_expected" if (not doi and no_doi_expected_signal) else "non_article_source_ok"
            action = "non_article_ok"
            ready = True
        elif (
            (not doi)
            and no_doi_expected_signal
            and has_title
            and (has_year or publisher_signal or title_only_ok_signal)
            and (has_authors or has_venue or web_source_signal or title_only_ok_signal)
        ):
            status = "no_doi_expected"
            reason = "no_doi_expected"
            action = "non_article_ok"
            ready = True
        elif has_title and has_venue and has_year:
            status = "bibliographic_ready"
            reason = ""
            action = "none"
            ready = True
        else:
            title_hint = _extract_query_title(raw)
            can_retry_title = bool(
                enable_title_lookup
                and (not doi)
                and raw
                and title_hint
                and _should_try_title_lookup(raw, title_hint)
            )
            if can_retry_title:
                status = "title_lookup_retryable"
                reason = "title_lookup_retryable"
                action = "retry"
            elif parse_confidence < 0.52 or (match_score and match_score < 0.74):
                status = "low_confidence_match"
                reason = "low_confidence_match"
                action = "source_repair"
            elif (not has_title) or (not has_authors):
                status = "low_confidence_match"
                reason = "low_confidence_match"
                action = "source_repair"
            else:
                status = "title_lookup_retryable" if not doi else "low_confidence_match"
                reason = status
                action = "retry" if status == "title_lookup_retryable" else "source_repair"

    return {
        "metadata_status": status,
        "missing_reason": reason,
        "metadata_action": action,
        "metadata_ready": bool(ready),
        "metadata_missing_fields": missing_fields,
    }


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


def _best_source_doi_from_citation_meta(
    cm: dict[str, Any],
    *,
    crossref_enabled: bool,
    stats: dict[str, int] | None = None,
) -> str:
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
        _stat_inc(stats, "crossref_network_attempts")
        _stat_inc(stats, "source_work_network_attempts")
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
    stats: dict[str, int] | None = None,
) -> str:
    venue0, year0, title0 = _parse_doc_name_hints(md_path)
    title_head = _extract_doc_title_from_md_head(md_head)
    title_h = title_head or title0
    if title0 and title_head:
        filename_title_truncated = ("..." in title0) or ("\u2026" in title0)
        if filename_title_truncated:
            title_h = title_head
        else:
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
        cached_value = source_work_cache.get(key)
        if _is_fresh_crossref_cache_miss(cached_value):
            _stat_inc(stats, "crossref_negative_hits")
            _stat_inc(stats, "source_work_negative_hits")
            return ""
        if not _is_crossref_cache_miss(cached_value):
            cached_doi = _clean_doi_for_url(str(cached_value or ""))
            if cached_doi:
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "source_work_cache_hits")
                return cached_doi

    if not crossref_enabled:
        return ""

    doi = ""
    try:
        _stat_inc(stats, "crossref_network_attempts")
        _stat_inc(stats, "source_work_network_attempts")
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
            _stat_inc(stats, "crossref_network_attempts")
            _stat_inc(stats, "source_work_network_attempts")
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
            _stat_inc(stats, "crossref_network_attempts")
            _stat_inc(stats, "source_work_network_attempts")
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

    if doi:
        source_work_cache[key] = doi
    else:
        source_work_cache[key] = _crossref_cache_miss("source_work_not_found")
        _stat_inc(stats, "crossref_negative_writes")
        _stat_inc(stats, "source_work_negative_writes")
    return doi


def _norm_name_key(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-").replace("\xad", "-")
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
    stats: dict[str, int] | None = None,
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
        if _is_fresh_crossref_cache_miss(v):
            _stat_inc(stats, "crossref_negative_hits")
            _stat_inc(stats, "source_refs_negative_hits")
            _stat_inc(stats, "source_refs_cached_miss")
            return []
        if isinstance(v, list):
            out: list[dict[str, str]] = []
            for r in v:
                if isinstance(r, dict):
                    out.append({k: str(v2 or "") for k, v2 in r.items()})
            if out:
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "source_refs_cache_hits")
                return out
            # Stale empty cache should retry when Crossref is available.
            if not crossref_enabled:
                return []
        elif not (_is_crossref_cache_miss(v) and crossref_enabled):
            return []

    if not crossref_enabled:
        return []

    _stat_inc(stats, "crossref_network_attempts")
    _stat_inc(stats, "source_refs_network_attempts")
    rows_raw = fetch_crossref_references_by_doi(source_doi)
    out_rows: list[dict[str, str]] = []
    for r in rows_raw:
        if not isinstance(r, dict):
            continue
        out_rows.append(_normalize_source_ref_row(r))
    if out_rows:
        src_cache[key] = out_rows
    else:
        src_cache[key] = _crossref_cache_miss("source_refs_empty")
        _stat_inc(stats, "crossref_negative_writes")
        _stat_inc(stats, "source_refs_negative_writes")
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


def _normalized_phrase_in_text(phrase: str, text_norm: str) -> bool:
    phrase_norm = normalize_title_for_match(phrase)
    if not phrase_norm or not text_norm:
        return False
    if phrase_norm in text_norm:
        return True
    toks = [t for t in phrase_norm.split() if len(t) >= 2]
    if not toks:
        return False
    text_tokens = set(text_norm.split())
    return bool(len(toks) >= 2 and all(t in text_tokens for t in toks))


def _reference_contains_number(text: str, value: str) -> bool:
    v = str(value or "").strip()
    if not v or not re.fullmatch(r"\d{1,8}", v):
        return False
    return bool(re.search(rf"(?<!\d){re.escape(v)}(?!\d)", str(text or "")))


def _reference_contains_nearby_number(text: str, value: str, *, max_delta: int = 3) -> bool:
    v = str(value or "").strip()
    if not v or not re.fullmatch(r"\d{3,8}", v):
        return False
    try:
        want = int(v)
    except Exception:
        return False
    for m in re.finditer(r"(?<!\d)(\d{3,8})(?!\d)", str(text or "")):
        try:
            got = int(m.group(1))
        except Exception:
            continue
        if abs(got - want) <= int(max_delta):
            return True
    return False


def _source_reference_row_field_score(local_entry: str, row: dict[str, str]) -> float:
    raw = _strip_reference_number(str(local_entry or "").strip())
    if (not raw) or (not isinstance(row, dict)):
        return 0.0
    raw_norm = normalize_title_for_match(raw)
    if not raw_norm:
        return 0.0

    score = 0.0
    title = str(row.get("title") or "").strip()
    if title:
        title_sim = _similarity(raw, title)
        if _normalized_phrase_in_text(title, raw_norm):
            score += 0.34
        elif title_sim >= 0.62:
            score += min(0.30, title_sim * 0.38)

    cand_author = str(row.get("author") or "").strip()
    cand_author_norm = normalize_title_for_match(cand_author)
    cand_author_tokens = [t for t in cand_author_norm.split() if len(t) >= 3]
    if cand_author_tokens and any(re.search(rf"\b{re.escape(t)}\b", raw_norm) for t in cand_author_tokens[:3]):
        score += 0.24

    venue = str(row.get("venue") or "").strip()
    if venue and _normalized_phrase_in_text(venue, raw_norm):
        score += 0.20

    cand_year = str(row.get("year") or "").strip()
    local_year = _extract_reference_year_hint(raw)
    if cand_year and local_year and cand_year == local_year:
        score += 0.18

    pages = str(row.get("pages") or "").strip()
    if pages and _reference_contains_number(raw, pages):
        score += 0.20
    elif pages and _reference_contains_nearby_number(raw, pages, max_delta=3):
        score += 0.14

    volume = str(row.get("volume") or "").strip()
    if volume and _reference_contains_number(raw, volume):
        score += 0.08

    if str(row.get("doi") or "").strip() and score >= 0.20:
        score += 0.04

    return max(0.0, min(1.0, score))


def _score_source_reference_row(
    local_entry: str,
    row: dict[str, str],
    *,
    penalize_mismatch: bool,
    year_match_bonus: float,
    year_near_bonus: float,
    year_mismatch_penalty: float,
    author_match_bonus: float,
    author_mismatch_penalty: float,
    doi_bonus: float,
) -> float:
    raw = str(local_entry or "").strip()
    if (not raw) or (not isinstance(row, dict)):
        return -1.0

    local_doi = _clean_doi_for_url(extract_first_doi(raw)).lower()
    row_doi = _clean_doi_for_url(str(row.get("doi") or "")).lower()
    if local_doi and row_doi and (local_doi == row_doi):
        return 1.0

    cand_text = str(row.get("text") or row.get("unstructured") or "").strip()
    if not cand_text:
        return -1.0

    sc = max(_similarity(raw, cand_text), _source_reference_row_field_score(raw, row))
    local_year = _extract_reference_year_hint(raw)
    cand_year = str(row.get("year") or "").strip()
    if local_year:
        if cand_year == local_year:
            sc += float(year_match_bonus)
        else:
            try:
                if cand_year and (abs(int(cand_year) - int(local_year)) <= 1):
                    sc += float(year_near_bonus)
                elif penalize_mismatch:
                    sc -= float(year_mismatch_penalty)
            except Exception:
                if penalize_mismatch:
                    sc -= float(max(0.0, year_mismatch_penalty * 0.6))

    local_author = extract_first_author_family_hint(raw)
    cand_author = str(row.get("author") or "").strip().lower()
    if local_author:
        if cand_author and (local_author in cand_author):
            sc += float(author_match_bonus)
        elif penalize_mismatch:
            sc -= float(author_mismatch_penalty)

    if row_doi:
        sc += float(doi_bonus)

    return max(0.0, min(1.0, sc))


def _source_reference_row_has_alignment_signal(local_entry: str, row: dict[str, str]) -> bool:
    raw = str(local_entry or "").strip()
    if not raw or (not isinstance(row, dict)):
        return False
    local_doi = _clean_doi_for_url(extract_first_doi(raw)).lower()
    row_doi = _clean_doi_for_url(str(row.get("doi") or "")).lower()
    if local_doi and row_doi:
        return True
    structured_fields = [
        str(row.get("title") or "").strip(),
        str(row.get("author") or "").strip(),
        str(row.get("venue") or "").strip(),
        str(row.get("year") or "").strip(),
        str(row.get("volume") or "").strip(),
        str(row.get("pages") or "").strip(),
    ]
    structured_count = sum(1 for x in structured_fields if x)
    if row_doi and structured_count >= 2:
        return True
    if structured_count >= 3 and (str(row.get("author") or "").strip() or str(row.get("title") or "").strip()):
        return True

    fields = [
        str(row.get("title") or "").strip(),
        str(row.get("author") or "").strip(),
        str(row.get("venue") or "").strip(),
        str(row.get("year") or "").strip(),
        str(row.get("unstructured") or "").strip(),
    ]
    rich_text = " ".join(x for x in fields if x).strip()
    if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", rich_text)) >= 4:
        return True

    cand_text = str(row.get("text") or "").strip()
    if cand_text:
        no_doi = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", " ", cand_text, flags=re.I)
        if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", no_doi)) >= 4:
            return True
    return False


def _assess_source_reference_alignment(ref_map: dict[int, str], rows: list[dict[str, str]]) -> bool:
    if (not isinstance(ref_map, dict)) or (not isinstance(rows, list)) or (not rows):
        return False

    nums = [
        int(n)
        for n in sorted(ref_map.keys())
        if int(n) > 0 and int(n) <= len(rows) and str(ref_map.get(int(n)) or "").strip()
    ]
    if len(nums) < 4:
        # Not enough evidence to reject order mapping.
        return True

    idx_candidates = {
        0,
        1,
        2,
        max(0, len(nums) // 2 - 1),
        len(nums) // 2,
        max(0, len(nums) - 3),
        max(0, len(nums) - 2),
        len(nums) - 1,
    }
    sampled_nums = [nums[i] for i in sorted(idx_candidates) if 0 <= i < len(nums)]

    scores: list[float] = []
    for n in sampled_nums:
        row = rows[int(n) - 1]
        if not _source_reference_row_has_alignment_signal(str(ref_map.get(int(n)) or ""), row):
            continue
        sc = _score_source_reference_row(
            str(ref_map.get(int(n)) or ""),
            row,
            penalize_mismatch=True,
            year_match_bonus=0.14,
            year_near_bonus=0.05,
            year_mismatch_penalty=0.08,
            author_match_bonus=0.10,
            author_mismatch_penalty=0.04,
            doi_bonus=0.03,
        )
        if sc >= 0.0:
            scores.append(sc)

    if len(scores) < 3:
        return True

    tail_scores: list[float] = []
    for n in nums[-min(6, len(nums)) :]:
        row = rows[int(n) - 1]
        if not _source_reference_row_has_alignment_signal(str(ref_map.get(int(n)) or ""), row):
            continue
        sc = _score_source_reference_row(
            str(ref_map.get(int(n)) or ""),
            row,
            penalize_mismatch=True,
            year_match_bonus=0.14,
            year_near_bonus=0.05,
            year_mismatch_penalty=0.08,
            author_match_bonus=0.10,
            author_mismatch_penalty=0.04,
            doi_bonus=0.03,
        )
        if sc >= 0.0:
            tail_scores.append(sc)
    if len(tail_scores) >= 3:
        tail_low_n = sum(1 for sc in tail_scores if sc < 0.45)
        tail_avg = sum(tail_scores) / max(1, len(tail_scores))
        early_strong = any(sc >= 0.66 for sc in scores)
        if early_strong and tail_low_n >= 3 and tail_avg < 0.38:
            return False

    strong_n = sum(1 for sc in scores if sc >= 0.66)
    ok_n = sum(1 for sc in scores if sc >= 0.56)
    avg_sc = sum(scores) / max(1, len(scores))
    need_ok = max(3, int((len(scores) * 0.60) + 0.999))
    need_strong = max(2, int((len(scores) * 0.30) + 0.999))
    return bool(ok_n >= need_ok and strong_n >= need_strong and avg_sc >= 0.58)


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
        sc = _score_source_reference_row(
            raw,
            r,
            penalize_mismatch=True,
            year_match_bonus=0.14,
            year_near_bonus=0.05,
            year_mismatch_penalty=0.08,
            author_match_bonus=0.10,
            author_mismatch_penalty=0.04,
            doi_bonus=0.03,
        )
        if sc < 0.0:
            continue
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

    sc = _score_source_reference_row(
        raw,
        row,
        penalize_mismatch=False,
        year_match_bonus=0.10,
        year_near_bonus=0.04,
        year_mismatch_penalty=0.0,
        author_match_bonus=0.08,
        author_mismatch_penalty=0.0,
        doi_bonus=0.02,
    )
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
    atomic_write_json(Path(path), obj)


def _crossref_negative_cache_ttl_s() -> float:
    raw = str(os.environ.get("KB_CROSSREF_NEGATIVE_TTL_S") or "").strip()
    try:
        value = float(raw) if raw else 7.0 * 24.0 * 60.0 * 60.0
    except Exception:
        value = 7.0 * 24.0 * 60.0 * 60.0
    return float(max(300.0, value))


def _crossref_cache_miss(reason: str = "not_found") -> dict[str, Any]:
    return {
        "_kb_cache": "miss",
        "reason": str(reason or "not_found"),
        "ts": float(time.time()),
        "lookup_version": int(REFERENCE_LOOKUP_VERSION),
    }


def _is_crossref_cache_miss(value: Any) -> bool:
    return bool(isinstance(value, dict) and str(value.get("_kb_cache") or "") == "miss")


def _is_fresh_crossref_cache_miss(value: Any, *, now_ts: float | None = None) -> bool:
    if not _is_crossref_cache_miss(value):
        return False
    try:
        ts = float(value.get("ts") or 0.0)
    except Exception:
        ts = 0.0
    if ts <= 0.0:
        return False
    try:
        lookup_version = int(value.get("lookup_version") or 0)
    except Exception:
        lookup_version = 0
    if lookup_version < int(REFERENCE_LOOKUP_VERSION):
        return False
    now = float(time.time() if now_ts is None else now_ts)
    return bool((now - ts) < _crossref_negative_cache_ttl_s())


def _is_crossref_meta_cache_hit(value: Any) -> bool:
    return bool(isinstance(value, dict) and value and (not _is_crossref_cache_miss(value)))


def _crossref_cache_value_is_fresh(value: Any) -> bool:
    return bool(_is_crossref_meta_cache_hit(value) or _is_fresh_crossref_cache_miss(value))


def _stat_inc(stats: dict[str, int] | None, key: str, amount: int = 1) -> None:
    if not isinstance(stats, dict):
        return
    try:
        stats[key] = int(stats.get(key, 0) or 0) + int(amount)
    except Exception:
        stats[key] = int(amount)


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
    stats: dict[str, int] | None = None,
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
    openalex_title_cache = cache.get("openalex_title")
    if not isinstance(openalex_title_cache, dict):
        openalex_title_cache = {}
        cache["openalex_title"] = openalex_title_cache

    if crossref_enabled and doi_key:
        if doi_key in doi_cache:
            cached = doi_cache.get(doi_key)
            if _is_crossref_meta_cache_hit(cached):
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "doi_cache_hits")
                meta = cached
            elif _is_fresh_crossref_cache_miss(cached):
                _stat_inc(stats, "crossref_negative_hits")
                _stat_inc(stats, "doi_negative_hits")
                meta = None
            elif cached is None:
                _stat_inc(stats, "crossref_network_attempts")
                _stat_inc(stats, "doi_network_attempts")
                found = fetch_best_crossref_meta(
                    query_title="",
                    doi_hint=doi_hint,
                    allow_title_only=False,
                    min_score=0.90,
                )
                meta = found if isinstance(found, dict) else None
                if isinstance(meta, dict):
                    doi_cache[doi_key] = meta
                else:
                    doi_cache[doi_key] = _crossref_cache_miss("doi_not_found")
                    _stat_inc(stats, "crossref_negative_writes")
                    _stat_inc(stats, "doi_negative_writes")
            elif _is_crossref_cache_miss(cached):
                _stat_inc(stats, "crossref_network_attempts")
                _stat_inc(stats, "doi_network_attempts")
                found = fetch_best_crossref_meta(
                    query_title="",
                    doi_hint=doi_hint,
                    allow_title_only=False,
                    min_score=0.90,
                )
                meta = found if isinstance(found, dict) else None
                doi_cache[doi_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("doi_not_found")
                if not isinstance(meta, dict):
                    _stat_inc(stats, "crossref_negative_writes")
                    _stat_inc(stats, "doi_negative_writes")
            else:
                meta = None
        else:
            _stat_inc(stats, "crossref_network_attempts")
            _stat_inc(stats, "doi_network_attempts")
            found = fetch_best_crossref_meta(
                query_title="",
                doi_hint=doi_hint,
                allow_title_only=False,
                min_score=0.90,
            )
            meta = found if isinstance(found, dict) else None
            doi_cache[doi_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("doi_not_found")
            if not isinstance(meta, dict):
                _stat_inc(stats, "crossref_negative_writes")
                _stat_inc(stats, "doi_negative_writes")
        if isinstance(meta, dict):
            return meta, doi_hint

    # Bibliographic query fallback (author/journal/year/volume/pages etc.)
    ref_key = normalize_title_for_match(raw)[:260]
    if crossref_enabled and ref_key:
        if ref_key in bib_cache:
            cached = bib_cache.get(ref_key)
            if _is_crossref_meta_cache_hit(cached):
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "bib_cache_hits")
                meta = cached
            elif _is_fresh_crossref_cache_miss(cached):
                _stat_inc(stats, "crossref_negative_hits")
                _stat_inc(stats, "bib_negative_hits")
                meta = None
            elif cached is None:
                _stat_inc(stats, "crossref_network_attempts")
                _stat_inc(stats, "bib_network_attempts")
                found = fetch_best_crossref_for_reference(reference_text=raw, min_score=0.62)
                meta = found if isinstance(found, dict) else None
                if isinstance(meta, dict):
                    bib_cache[ref_key] = meta
                else:
                    bib_cache[ref_key] = _crossref_cache_miss("bib_not_found")
                    _stat_inc(stats, "crossref_negative_writes")
                    _stat_inc(stats, "bib_negative_writes")
            elif _is_crossref_cache_miss(cached):
                _stat_inc(stats, "crossref_network_attempts")
                _stat_inc(stats, "bib_network_attempts")
                found = fetch_best_crossref_for_reference(reference_text=raw, min_score=0.62)
                meta = found if isinstance(found, dict) else None
                bib_cache[ref_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("bib_not_found")
                if not isinstance(meta, dict):
                    _stat_inc(stats, "crossref_negative_writes")
                    _stat_inc(stats, "bib_negative_writes")
            else:
                meta = None
        else:
            _stat_inc(stats, "crossref_network_attempts")
            _stat_inc(stats, "bib_network_attempts")
            found = fetch_best_crossref_for_reference(reference_text=raw, min_score=0.62)
            meta = found if isinstance(found, dict) else None
            bib_cache[ref_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("bib_not_found")
            if not isinstance(meta, dict):
                _stat_inc(stats, "crossref_negative_writes")
                _stat_inc(stats, "bib_negative_writes")
        if isinstance(meta, dict):
            return meta, doi_hint

    if (not crossref_enabled) or (not enable_title_lookup):
        return None, doi_hint

    title_hint = _extract_query_title(raw)
    title_key = normalize_title_for_match(title_hint)[:260]
    if not title_key or len(title_key) < 8:
        return None, doi_hint
    if not _should_try_title_lookup(raw, title_hint):
        return None, doi_hint

    if title_key in title_cache:
        cached = title_cache.get(title_key)
        if _is_crossref_meta_cache_hit(cached):
            _stat_inc(stats, "crossref_cache_hits")
            _stat_inc(stats, "title_cache_hits")
            meta = cached
        elif _is_fresh_crossref_cache_miss(cached):
            _stat_inc(stats, "crossref_negative_hits")
            _stat_inc(stats, "title_negative_hits")
            meta = None
        elif cached is None:
            year_hint = extract_year_hint(raw)
            has_quoted_title = bool(_QUOTED_TITLE_RE.search(raw))
            min_score = 0.95
            if has_quoted_title:
                min_score = 0.90
            _stat_inc(stats, "crossref_network_attempts")
            _stat_inc(stats, "title_network_attempts")
            found = fetch_best_crossref_meta(
                query_title=title_hint,
                expected_year=year_hint,
                doi_hint="",
                allow_title_only=True,
                min_score=float(min_score),
            )
            meta = found if isinstance(found, dict) else None
            title_cache[title_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("title_not_found")
            if not isinstance(meta, dict):
                _stat_inc(stats, "crossref_negative_writes")
                _stat_inc(stats, "title_negative_writes")
        elif _is_crossref_cache_miss(cached):
            year_hint = extract_year_hint(raw)
            has_quoted_title = bool(_QUOTED_TITLE_RE.search(raw))
            min_score = 0.95
            if has_quoted_title:
                min_score = 0.90
            _stat_inc(stats, "crossref_network_attempts")
            _stat_inc(stats, "title_network_attempts")
            found = fetch_best_crossref_meta(
                query_title=title_hint,
                expected_year=year_hint,
                doi_hint="",
                allow_title_only=True,
                min_score=float(min_score),
            )
            meta = found if isinstance(found, dict) else None
            title_cache[title_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("title_not_found")
            if not isinstance(meta, dict):
                _stat_inc(stats, "crossref_negative_writes")
                _stat_inc(stats, "title_negative_writes")
        else:
            meta = None
    else:
        year_hint = extract_year_hint(raw)
        has_quoted_title = bool(_QUOTED_TITLE_RE.search(raw))
        min_score = 0.95
        if has_quoted_title:
            min_score = 0.90
        _stat_inc(stats, "crossref_network_attempts")
        _stat_inc(stats, "title_network_attempts")
        found = fetch_best_crossref_meta(
            query_title=title_hint,
            expected_year=year_hint,
            doi_hint="",
            allow_title_only=True,
            min_score=float(min_score),
        )
        meta = found if isinstance(found, dict) else None
        title_cache[title_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("title_not_found")
        if not isinstance(meta, dict):
            _stat_inc(stats, "crossref_negative_writes")
            _stat_inc(stats, "title_negative_writes")

    if isinstance(meta, dict):
        return meta, doi_hint

    cached_openalex = openalex_title_cache.get(title_key)
    if _is_crossref_meta_cache_hit(cached_openalex):
        _stat_inc(stats, "crossref_cache_hits")
        _stat_inc(stats, "openalex_cache_hits")
        meta = dict(cached_openalex)
    elif _is_fresh_crossref_cache_miss(cached_openalex):
        _stat_inc(stats, "crossref_negative_hits")
        _stat_inc(stats, "openalex_negative_hits")
        meta = None
    else:
        _stat_inc(stats, "crossref_network_attempts")
        _stat_inc(stats, "openalex_network_attempts")
        try:
            found = fetch_best_openalex_meta(
                query_title=title_hint,
                reference_text=raw,
                expected_year=extract_year_hint(raw),
                min_score=0.90,
            )
        except Exception:
            found = None
        meta = found if isinstance(found, dict) else None
        openalex_title_cache[title_key] = meta if isinstance(meta, dict) else _crossref_cache_miss("openalex_title_not_found")
        if not isinstance(meta, dict):
            _stat_inc(stats, "crossref_negative_writes")
            _stat_inc(stats, "openalex_negative_writes")

    return meta, doi_hint


def _cached_crossref_meta_for_record(
    rec: Mapping[str, Any],
    cache: dict,
    *,
    enable_title_lookup: bool,
    stats: dict[str, int] | None = None,
) -> tuple[dict | None, str, str]:
    if not isinstance(rec, Mapping) or not isinstance(cache, dict):
        return None, "", ""

    raw = str(rec.get("raw") or rec.get("cite_fmt") or "").strip()
    doi_hint = _clean_doi_for_url(str(rec.get("doi") or rec.get("doi_url") or ""))
    if not doi_hint and raw:
        doi_hint = _clean_doi_for_url(extract_first_doi(raw))

    doi_cache = cache.get("doi") if isinstance(cache.get("doi"), dict) else {}
    bib_cache = cache.get("bib") if isinstance(cache.get("bib"), dict) else {}
    title_cache = cache.get("title") if isinstance(cache.get("title"), dict) else {}
    openalex_title_cache = cache.get("openalex_title") if isinstance(cache.get("openalex_title"), dict) else {}

    if doi_hint:
        cached = doi_cache.get(doi_hint.lower())
        if _is_crossref_meta_cache_hit(cached):
            _stat_inc(stats, "crossref_cache_hits")
            _stat_inc(stats, "doi_cache_hits")
            return dict(cached), doi_hint, "doi"

    if raw:
        ref_key = normalize_title_for_match(raw)[:260]
        cached = bib_cache.get(ref_key) if ref_key else None
        if _is_crossref_meta_cache_hit(cached):
            _stat_inc(stats, "crossref_cache_hits")
            _stat_inc(stats, "bib_cache_hits")
            return dict(cached), doi_hint, "bibliographic"

    if raw and enable_title_lookup:
        title_hint = _extract_query_title(raw)
        title_key = normalize_title_for_match(title_hint)[:260]
        if title_key and len(title_key) >= 8 and _should_try_title_lookup(raw, title_hint):
            cached = title_cache.get(title_key)
            if _is_crossref_meta_cache_hit(cached):
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "title_cache_hits")
                return dict(cached), doi_hint, "title"
            cached = openalex_title_cache.get(title_key)
            if _is_crossref_meta_cache_hit(cached):
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "openalex_cache_hits")
                return dict(cached), doi_hint, "openalex_title"

    return None, doi_hint, ""


def _hydrate_reference_record_from_crossref_cache(
    rec: Mapping[str, Any],
    cache: dict,
    *,
    enable_title_lookup: bool,
    stats: dict[str, int] | None = None,
) -> tuple[dict, bool]:
    out = dict(rec) if isinstance(rec, Mapping) else {}
    if not out:
        return out, False

    meta, doi_hint, method = _cached_crossref_meta_for_record(
        out,
        cache,
        enable_title_lookup=bool(enable_title_lookup),
        stats=stats,
    )
    if not isinstance(meta, dict) or not meta:
        return out, False

    supplement = dict(meta)
    if doi_hint and not str(supplement.get("doi") or "").strip():
        supplement["doi"] = doi_hint
    supp_method = str(supplement.get("match_method") or "").strip()
    if not supp_method:
        supplement["match_method"] = method
    elif method and method not in supp_method:
        supplement["match_method"] = f"{supp_method}+{method}"

    merged = _merge_reference_meta(out, supplement)
    if not isinstance(merged, dict):
        merged = dict(out)
    doi = _clean_doi_for_url(str(merged.get("doi") or doi_hint or ""))
    if doi:
        merged["doi"] = doi
        merged["doi_url"] = str(merged.get("doi_url") or _doi_url(doi)).strip()
    match_method = str(merged.get("match_method") or "").strip()
    merged["crossref_ok"] = bool(_reference_match_method_has_external_resolution(match_method, doi))
    merged.update(classify_reference_metadata(merged, enable_title_lookup=bool(enable_title_lookup)))
    changed = merged != out
    if changed:
        _stat_inc(stats, "refs_cache_hydrated")
    return merged, bool(changed)


def _hydrate_doc_refs_from_crossref_cache(
    refs: dict[str, dict] | None,
    cache: dict,
    *,
    enable_title_lookup: bool,
    stats: dict[str, int] | None = None,
) -> tuple[dict[str, dict], int]:
    if not isinstance(refs, dict):
        return {}, 0
    out: dict[str, dict] = {}
    changed = 0
    for key, value in refs.items():
        if isinstance(value, Mapping):
            hydrated, did_change = _hydrate_reference_record_from_crossref_cache(
                value,
                cache,
                enable_title_lookup=bool(enable_title_lookup),
                stats=stats,
            )
            out[str(key)] = hydrated
            if did_change:
                changed += 1
        else:
            out[str(key)] = value
    return out, int(changed)


def _prefetch_doi_meta_parallel(
    ref_map: dict[int, str],
    cache: dict,
    *,
    crossref_enabled: bool,
    max_workers: int,
    max_prefetch: int,
    stats: dict[str, int] | None = None,
) -> int:
    if (not crossref_enabled) or (int(max_workers) <= 1) or (not isinstance(ref_map, dict)):
        return 0

    doi_cache = cache.get("doi")
    if not isinstance(doi_cache, dict):
        doi_cache = {}
        cache["doi"] = doi_cache

    want: list[str] = []
    seen: set[str] = set()
    lim = int(max(0, max_prefetch))
    if lim <= 0:
        return 0
    for n in sorted(ref_map.keys()):
        raw = str(ref_map.get(int(n)) or "").strip()
        if not raw:
            continue
        d = _clean_doi_for_url(extract_first_doi(raw))
        if not d:
            continue
        k = d.lower()
        if k in seen:
            continue
        if k in doi_cache and _crossref_cache_value_is_fresh(doi_cache.get(k)):
            if _is_crossref_meta_cache_hit(doi_cache.get(k)):
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "doi_cache_hits")
            elif _is_fresh_crossref_cache_miss(doi_cache.get(k)):
                _stat_inc(stats, "crossref_negative_hits")
                _stat_inc(stats, "doi_negative_hits")
            continue
        seen.add(k)
        want.append(d)
        if len(want) >= lim:
            break

    if len(want) < 2:
        return 0
    _stat_inc(stats, "crossref_network_attempts", len(want))
    _stat_inc(stats, "doi_network_attempts", len(want))

    def _job(doi: str) -> tuple[str, dict | None]:
        key = _clean_doi_for_url(doi).lower()
        try:
            meta = fetch_best_crossref_meta(
                query_title="",
                doi_hint=doi,
                allow_title_only=False,
                min_score=0.90,
            )
            return key, (meta if isinstance(meta, dict) else None)
        except Exception:
            return key, None

    done = 0
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futs = [ex.submit(_job, d) for d in want]
        for fut in as_completed(futs):
            try:
                key, meta = fut.result()
            except Exception:
                continue
            if isinstance(meta, dict):
                doi_cache[key] = meta
                done += 1
            elif key:
                doi_cache[key] = _crossref_cache_miss("doi_not_found")
                _stat_inc(stats, "crossref_negative_writes")
                _stat_inc(stats, "doi_negative_writes")
    return int(done)


def _prefetch_reference_meta_parallel(
    ref_map: dict[int, str],
    cache: dict,
    *,
    crossref_enabled: bool,
    enable_title_lookup: bool,
    max_workers: int,
    max_prefetch: int,
    stats: dict[str, int] | None = None,
) -> int:
    if (not crossref_enabled) or (int(max_workers) <= 1) or (not isinstance(ref_map, dict)):
        return 0

    bib_cache = cache.get("bib")
    if not isinstance(bib_cache, dict):
        bib_cache = {}
        cache["bib"] = bib_cache
    title_cache = cache.get("title")
    if not isinstance(title_cache, dict):
        title_cache = {}
        cache["title"] = title_cache

    lim = int(max(0, max_prefetch))
    if lim <= 0:
        return 0

    bib_jobs: list[tuple[str, str]] = []
    seen_bib: set[str] = set()
    for n in sorted(ref_map.keys()):
        raw = str(ref_map.get(int(n)) or "").strip()
        if not raw:
            continue
        if str(extract_first_doi(raw) or "").strip():
            # DOI paths are prefetched separately.
            continue
        if not is_promising_reference_text(raw):
            continue
        ref_key = normalize_title_for_match(raw)[:260]
        if (not ref_key) or (ref_key in seen_bib):
            continue
        if ref_key in bib_cache and _crossref_cache_value_is_fresh(bib_cache.get(ref_key)):
            if _is_crossref_meta_cache_hit(bib_cache.get(ref_key)):
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "bib_cache_hits")
            elif _is_fresh_crossref_cache_miss(bib_cache.get(ref_key)):
                _stat_inc(stats, "crossref_negative_hits")
                _stat_inc(stats, "bib_negative_hits")
            continue
        seen_bib.add(ref_key)
        bib_jobs.append((ref_key, raw))
        if len(bib_jobs) >= lim:
            break

    if len(bib_jobs) < 2:
        return 0
    _stat_inc(stats, "crossref_network_attempts", len(bib_jobs))
    _stat_inc(stats, "bib_network_attempts", len(bib_jobs))

    def _bib_job(ref_key: str, raw: str) -> tuple[str, str, dict | None]:
        try:
            meta = fetch_best_crossref_for_reference(reference_text=raw, min_score=0.62)
            return ref_key, raw, (meta if isinstance(meta, dict) else None)
        except Exception:
            return ref_key, raw, None

    title_need: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(bib_jobs))) as ex:
        futs = [ex.submit(_bib_job, k, r) for k, r in bib_jobs]
        for fut in as_completed(futs):
            try:
                ref_key, raw, meta = fut.result()
            except Exception:
                continue
            if isinstance(meta, dict):
                bib_cache[ref_key] = meta
                done += 1
                continue
            if ref_key:
                bib_cache[ref_key] = _crossref_cache_miss("bib_not_found")
                _stat_inc(stats, "crossref_negative_writes")
                _stat_inc(stats, "bib_negative_writes")
            if not enable_title_lookup:
                continue
            title_need.append(raw)

    if (not enable_title_lookup) or (not title_need):
        return int(done)

    title_jobs: list[tuple[str, str, str, float]] = []
    seen_title: set[str] = set()
    for raw in title_need:
        title_hint = _extract_query_title(raw)
        title_key = normalize_title_for_match(title_hint)[:260]
        if not title_key or len(title_key) < 8:
            continue
        if not _should_try_title_lookup(raw, title_hint):
            continue
        if title_key in seen_title:
            continue
        cached = title_cache.get(title_key)
        if _crossref_cache_value_is_fresh(cached):
            if _is_crossref_meta_cache_hit(cached):
                _stat_inc(stats, "crossref_cache_hits")
                _stat_inc(stats, "title_cache_hits")
            elif _is_fresh_crossref_cache_miss(cached):
                _stat_inc(stats, "crossref_negative_hits")
                _stat_inc(stats, "title_negative_hits")
            continue
        seen_title.add(title_key)
        min_score = 0.90 if bool(_QUOTED_TITLE_RE.search(raw)) else 0.95
        title_jobs.append((title_key, title_hint, extract_year_hint(raw), float(min_score)))
        if len(title_jobs) >= lim:
            break

    if len(title_jobs) < 2:
        return int(done)
    _stat_inc(stats, "crossref_network_attempts", len(title_jobs))
    _stat_inc(stats, "title_network_attempts", len(title_jobs))

    def _title_job(title_key: str, title_hint: str, year_hint: str, min_score: float) -> tuple[str, dict | None]:
        try:
            meta = fetch_best_crossref_meta(
                query_title=title_hint,
                expected_year=year_hint,
                doi_hint="",
                allow_title_only=True,
                min_score=float(min_score),
            )
            return title_key, (meta if isinstance(meta, dict) else None)
        except Exception:
            return title_key, None

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(title_jobs))) as ex:
        futs = [ex.submit(_title_job, k, t, y, s) for k, t, y, s in title_jobs]
        for fut in as_completed(futs):
            try:
                title_key, meta = fut.result()
            except Exception:
                continue
            if isinstance(meta, dict):
                title_cache[title_key] = meta
                done += 1
            elif title_key:
                title_cache[title_key] = _crossref_cache_miss("title_not_found")
                _stat_inc(stats, "crossref_negative_writes")
                _stat_inc(stats, "title_negative_writes")
    return int(done)


def _prepare_doc_context_prefetch(
    md_path: Path,
    *,
    pdf_root_obj: Path | None,
    lib_citation_meta_map: dict[str, dict],
    crossref_enabled: bool,
) -> dict[str, Any]:
    md_head = _read_text_head(md_path, max_bytes=220_000)
    md_tail = _read_text_tail(md_path, max_bytes=1_500_000)
    catalog = load_reference_catalog_for_md(md_path)
    ref_map = reference_catalog_to_map(catalog)
    if not ref_map:
        ref_map = extract_references_map_from_md(md_tail)

    source_doi = ""
    try:
        pdf_candidate = _lookup_pdf_for_md_doc(md_path, pdf_root_obj)
    except Exception:
        pdf_candidate = None
    if pdf_candidate is not None:
        cm = lib_citation_meta_map.get(_norm_path_key(pdf_candidate))
        if isinstance(cm, dict):
            # Prefetch stage avoids extra title-based network DOI lookup here.
            source_doi = _best_source_doi_from_citation_meta(cm, crossref_enabled=False)
    if not source_doi:
        source_doi = _extract_source_doi_from_md_head(md_head)

    return {
        "md_head": md_head,
        "md_tail": md_tail,
        "ref_map": ref_map if isinstance(ref_map, dict) else {},
        "reference_catalog": catalog if isinstance(catalog, dict) else {},
        "source_doi": _clean_doi_for_url(source_doi),
        # Keep preparation I/O-only. Source references are loaded in the main pass so
        # the shared cache and Crossref time budget are respected.
        "source_ref_rows": [],
    }


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
    doi_prefetch_workers: int = 1,
    doc_prepare_workers: int = 1,
    quality_gate: bool = False,
    quality_gate_autofix: bool = True,
    allow_blocked_quality: bool = False,
    pdf_root: Path | None = None,
    library_db_path: Path | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    run_started_ts = time.time()
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
    docs_quality_blocked = 0
    refs_total = 0
    refs_with_doi = 0
    refs_with_title = 0
    refs_with_authors = 0
    refs_with_venue = 0
    refs_metadata_ready = 0
    refs_metadata_user_ready = 0
    refs_unresolved = 0
    refs_crossref_ok = 0
    refs_source_map_ok = 0
    crossref_stats: dict[str, int] = {
        "crossref_cache_hits": 0,
        "crossref_negative_hits": 0,
        "crossref_negative_writes": 0,
        "crossref_network_attempts": 0,
        "docs_retry_suppressed": 0,
        "source_refs_cached_miss": 0,
    }
    total_docs = len(md_files)
    prep_futures: dict[str, Any] = {}
    prep_results: dict[str, dict[str, Any]] = {}
    prep_executor: ThreadPoolExecutor | None = None

    def _emit_progress(stage: str, *, docs_done: int, current: str = "") -> None:
        if not callable(progress_cb):
            return
        try:
            metadata_missing_doi = max(0, refs_total - refs_with_doi)
            metadata_missing_title = max(0, refs_total - refs_with_title)
            metadata_missing_authors = max(0, refs_total - refs_with_authors)
            metadata_missing_venue = max(0, refs_total - refs_with_venue)
            progress_stats: dict[str, Any] = {
                **dict(crossref_stats),
                "docs_total": int(max(0, total_docs)),
                "docs_indexed": int(max(0, len(docs_out))),
                "refs_total": int(max(0, refs_total)),
                "refs_with_doi": int(max(0, refs_with_doi)),
                "refs_with_title": int(max(0, refs_with_title)),
                "refs_with_authors": int(max(0, refs_with_authors)),
                "refs_with_venue": int(max(0, refs_with_venue)),
                "refs_metadata_ready": int(max(0, refs_metadata_ready)),
                "refs_metadata_user_ready": int(max(0, refs_metadata_user_ready)),
                "refs_missing_doi": int(metadata_missing_doi),
                "refs_missing_title": int(metadata_missing_title),
                "refs_missing_authors": int(metadata_missing_authors),
                "refs_missing_venue": int(metadata_missing_venue),
                "refs_unresolved": int(max(0, refs_unresolved)),
                "refs_crossref_ok": int(max(0, refs_crossref_ok)),
                "refs_source_map_ok": int(max(0, refs_source_map_ok)),
            }
            progress_cb(
                {
                    "stage": str(stage or "").strip(),
                    "current": str(current or "").strip(),
                    "docs_done": int(max(0, docs_done)),
                    "docs_total": int(max(0, total_docs)),
                    "refs_total": int(max(0, refs_total)),
                    "refs_metadata_ready": int(max(0, refs_metadata_ready)),
                    "refs_metadata_user_ready": int(max(0, refs_metadata_user_ready)),
                    "refs_with_doi": int(max(0, refs_with_doi)),
                    "refs_with_title": int(max(0, refs_with_title)),
                    "refs_with_authors": int(max(0, refs_with_authors)),
                    "refs_with_venue": int(max(0, refs_with_venue)),
                    "refs_crossref_ok": int(max(0, refs_crossref_ok)),
                    "refs_source_map_ok": int(max(0, refs_source_map_ok)),
                    "stats": progress_stats,
                }
            )
        except Exception:
            return

    def _record_ref_stats(refs: Any) -> None:
        nonlocal refs_total, refs_with_doi, refs_with_title, refs_with_authors, refs_with_venue
        nonlocal refs_metadata_ready, refs_metadata_user_ready, refs_unresolved, refs_crossref_ok, refs_source_map_ok
        if not isinstance(refs, dict):
            return
        refs_total += len(refs)
        for rv in refs.values():
            if not isinstance(rv, dict):
                continue
            classification = classify_reference_metadata(rv, enable_title_lookup=bool(enable_title_lookup))
            doi = str(rv.get("doi") or "").strip()
            title = str(rv.get("title") or "").strip()
            authors = str(rv.get("authors") or "").strip()
            venue = str(rv.get("venue") or "").strip()
            raw = str(rv.get("raw") or rv.get("cite_fmt") or "").strip()
            mm = str(rv.get("match_method") or "").strip()
            crossref_ok = bool(rv.get("crossref_ok") or _reference_match_method_has_external_resolution(mm, doi))
            if doi:
                refs_with_doi += 1
            if _reference_has_usable_title(title):
                refs_with_title += 1
            if _reference_has_usable_authors(authors):
                refs_with_authors += 1
            if _reference_has_usable_venue(venue):
                refs_with_venue += 1
            metadata_ready = bool(classification.get("metadata_ready"))
            if metadata_ready:
                refs_metadata_ready += 1
            if (not metadata_ready) and (not doi) and (not crossref_ok):
                refs_unresolved += 1
            if crossref_ok:
                refs_crossref_ok += 1
            if mm.startswith("source_work_reference"):
                refs_source_map_ok += 1
            status = str(classification.get("metadata_status") or rv.get("metadata_status") or "").strip()
            reason = str(classification.get("missing_reason") or rv.get("missing_reason") or "").strip()
            action = str(classification.get("metadata_action") or rv.get("metadata_action") or "").strip()
            user_ready = bool(metadata_ready or crossref_ok or action == "auto_backfill")
            if status in REFERENCE_METADATA_STATUS_CODES:
                _stat_inc(crossref_stats, f"refs_metadata_status_{status}")
            if reason in REFERENCE_MISSING_REASON_CODES:
                _stat_inc(crossref_stats, f"refs_missing_reason_{reason}")
            if action:
                _stat_inc(crossref_stats, f"refs_action_{action}")
            if action in {"retry", "source_repair"}:
                _stat_inc(crossref_stats, "refs_action_retry_or_source_repair")
            if action == "non_article_ok" and _WEB_SOURCE_RE.search(" ".join([raw, title, venue, mm])):
                _stat_inc(crossref_stats, "refs_web_source_ok")
            if user_ready:
                refs_metadata_user_ready += 1

    _emit_progress("prepare", docs_done=0, current="")
    if (not bool(quality_gate)) and int(doc_prepare_workers) > 1 and total_docs > 1:
        prep_max_workers = int(max(1, min(8, int(doc_prepare_workers), int(total_docs))))
        prep_executor = ThreadPoolExecutor(max_workers=prep_max_workers)
        for p0 in md_files:
            k0 = _norm_source_key(str(p0.resolve()))
            if not k0:
                continue
            prep_futures[k0] = prep_executor.submit(
                _prepare_doc_context_prefetch,
                p0,
                pdf_root_obj=pdf_root_obj,
                lib_citation_meta_map=lib_citation_meta_map,
                crossref_enabled=bool(crossref_enabled),
            )

    try:
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
            prev_refs_obj = prev_doc.get("refs") if isinstance(prev_doc, dict) else None
            prev_needs_rebuild_for_enrich = False
            if need_crossref_enrich and isinstance(prev_refs_obj, dict):
                prev_unresolved, _prev_sparse = _assess_doc_crossref_enrichment(prev_refs_obj)
                # Rebuild only when promising references remain unresolved.
                # Sparse-but-resolved docs are often already good enough and re-running can regress quality.
                prev_needs_rebuild_for_enrich = bool(int(prev_unresolved) > 0)
            try:
                prev_lookup_version = int(prev_doc.get("reference_lookup_version") or 1) if isinstance(prev_doc, dict) else 0
            except Exception:
                prev_lookup_version = 0
            try:
                prev_parser_version = int(prev_doc.get("reference_parser_version") or 0) if isinstance(prev_doc, dict) else 0
            except Exception:
                prev_parser_version = 0
            prev_parser_stale = bool(
                isinstance(prev_doc, dict)
                and int(prev_parser_version) < int(REFERENCE_PARSER_VERSION)
            )
            prev_lookup_stale = bool(
                need_crossref_enrich
                and prev_needs_rebuild_for_enrich
                and int(prev_lookup_version) < int(REFERENCE_LOOKUP_VERSION)
            )
            prev_crossref_retry_due = bool(prev_lookup_stale or _doc_crossref_retry_due(prev_doc, now_ts=time.time()))
            prev_needs_rebuild_for_catalog = _previous_doc_refs_stale_against_catalog(prev_doc, prev_refs_obj)
            prev_has_quality_gate = bool(
                isinstance(prev_doc, dict)
                and isinstance(prev_doc.get("quality_gate"), dict)
                and str(prev_doc.get("index_status") or "").strip()
            )
            if (
                incremental
                and isinstance(prev_doc, dict)
                and str(prev_doc.get("sha1") or "") == str(sha1 or "")
                and isinstance(prev_refs_obj, dict)
                and ((not bool(quality_gate)) or prev_has_quality_gate)
                and (not prev_parser_stale)
                and (not prev_needs_rebuild_for_catalog)
                and (
                    (not need_crossref_enrich)
                    or (not prev_needs_rebuild_for_enrich)
                    or (not prev_crossref_retry_due)
                )
            ):
                kept_doc = dict(prev_doc)
                kept_refs, hydrated_count = _hydrate_doc_refs_from_crossref_cache(
                    prev_refs_obj,
                    cache,
                    enable_title_lookup=bool(enable_title_lookup),
                    stats=crossref_stats,
                )
                if int(hydrated_count) > 0:
                    kept_doc["refs"] = kept_refs
                    unresolved_now, sparse_now = _assess_doc_crossref_enrichment(kept_refs)
                    kept_doc["crossref_enriched"] = bool(_doc_crossref_enriched(kept_refs))
                    kept_doc["crossref_unresolved_promising"] = int(unresolved_now)
                    kept_doc["crossref_sparse_promising"] = int(sparse_now)
                    kept_doc["reference_lookup_version"] = int(REFERENCE_LOOKUP_VERSION)
                    docs_updated += 1
                    _stat_inc(crossref_stats, "docs_cache_hydrated")
                else:
                    docs_reused += 1
                docs_out[src_key] = kept_doc
                if prev_needs_rebuild_for_enrich and (not prev_crossref_retry_due):
                    _stat_inc(crossref_stats, "docs_retry_suppressed")
                _record_ref_stats(kept_doc.get("refs"))
                _emit_progress("doc_done", docs_done=doc_i, current=p.name)
                continue

            quality_fields: dict[str, Any] = {}
            if bool(quality_gate):
                try:
                    from kb.converter.quality_gate import index_quality_document_fields, prepare_markdown_for_index

                    quality_assessment = prepare_markdown_for_index(
                        p,
                        auto_repair=bool(quality_gate_autofix),
                        allow_blocked=bool(allow_blocked_quality),
                    )
                    quality_fields = index_quality_document_fields(quality_assessment)
                    if not bool(quality_assessment.get("indexable")):
                        docs_quality_blocked += 1
                        _emit_progress("doc_done", docs_done=doc_i, current=p.name)
                        continue
                except Exception:
                    docs_quality_blocked += 1
                    _emit_progress("doc_done", docs_done=doc_i, current=p.name)
                    continue

                try:
                    sha1 = compute_file_sha1(p)
                except Exception:
                    sha1 = ""

            prepped: dict[str, Any] = {}
            if src_key in prep_results:
                prepped = prep_results.get(src_key) or {}
            elif src_key in prep_futures:
                fut = prep_futures.get(src_key)
                if fut is not None:
                    try:
                        prepped = fut.result() or {}
                    except Exception:
                        prepped = {}
                prep_results[src_key] = prepped

            md_head = str(prepped.get("md_head") or "")
            md_tail = str(prepped.get("md_tail") or "")
            ref_map_prepped = prepped.get("ref_map")
            reference_catalog_prepped = prepped.get("reference_catalog")
            source_doi_prepped = _clean_doi_for_url(str(prepped.get("source_doi") or ""))
            source_ref_rows_prepped = prepped.get("source_ref_rows")
            if not isinstance(source_ref_rows_prepped, list):
                source_ref_rows_prepped = []

            if not md_head:
                md_head = _read_text_head(p, max_bytes=220_000)
            if not md_tail:
                md_tail = _read_text_tail(p, max_bytes=1_500_000)
            ref_map = (
                ref_map_prepped
                if isinstance(ref_map_prepped, dict)
                else extract_references_map_from_md(md_tail)
            )
            reference_catalog = (
                dict(reference_catalog_prepped)
                if isinstance(reference_catalog_prepped, dict)
                else {}
            )
            catalog_sha1 = str(reference_catalog.get("source_sha1") or "").strip()
            if (
                (not reference_catalog)
                or (not reference_catalog_to_map(reference_catalog))
                or (str(sha1 or "").strip() and catalog_sha1 != str(sha1 or "").strip())
            ):
                reference_catalog = build_reference_catalog_from_ref_map(
                    ref_map,
                    source_path=src_path,
                    source_name=p.name,
                    source_sha1=sha1,
                )
                try:
                    _save_json(_reference_catalog_path_for_md(p), reference_catalog)
                except Exception:
                    pass
            refs_obj: dict[str, dict] = {}
            unresolved_promising = 0
            sparse_promising = 0
            crossref_active_source = bool(
                crossref_enabled and ((time.monotonic() - crossref_start_ts) <= float(max(5.0, crossref_time_budget_s)))
            )
            doc_crossref_attempted = False
            doc_crossref_attempt_ts = 0.0

            source_doi = _clean_doi_for_url(source_doi_prepped)
            source_ref_rows = list(source_ref_rows_prepped or [])
            if not source_doi:
                pdf_candidate = _lookup_pdf_for_md_doc(p, pdf_root_obj)
                if pdf_candidate is not None:
                    cm = lib_citation_meta_map.get(_norm_path_key(pdf_candidate))
                    if isinstance(cm, dict):
                        source_doi = _best_source_doi_from_citation_meta(
                            cm,
                            crossref_enabled=crossref_active_source,
                            stats=crossref_stats,
                        )
            if not source_doi:
                source_doi = _extract_source_doi_from_md_head(md_head)
            if not source_doi:
                source_doi = _infer_source_doi_from_doc_hints(
                    p,
                    md_head,
                    cache,
                    crossref_enabled=crossref_active_source,
                    stats=crossref_stats,
                )
                if crossref_enabled and (not crossref_budget_exhausted):
                    if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                        crossref_budget_exhausted = True
                if crossref_budget_exhausted:
                    crossref_active_source = False
            if source_doi and source_ref_rows:
                src_cache = cache.get("source_refs")
                if not isinstance(src_cache, dict):
                    src_cache = {}
                    cache["source_refs"] = src_cache
                src_key_rows = f"doi:{_clean_doi_for_url(source_doi).lower()}"
                if src_key_rows and (src_key_rows not in src_cache):
                    src_cache[src_key_rows] = source_ref_rows
            if (not source_ref_rows) and source_doi:
                if crossref_active_source:
                    doc_crossref_attempted = True
                    doc_crossref_attempt_ts = doc_crossref_attempt_ts or time.time()
                source_ref_rows = _load_source_reference_rows(
                    source_doi,
                    cache,
                    crossref_enabled=crossref_active_source,
                    stats=crossref_stats,
                )
                if crossref_enabled and (not crossref_budget_exhausted):
                    if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                        crossref_budget_exhausted = True
            if int(doi_prefetch_workers) > 1 and source_ref_rows:
                crossref_active_prefetch = bool(
                    crossref_enabled and ((time.monotonic() - crossref_start_ts) <= float(max(5.0, crossref_time_budget_s)))
                )
                if crossref_active_prefetch:
                    doc_crossref_attempted = True
                    doc_crossref_attempt_ts = doc_crossref_attempt_ts or time.time()
                source_doi_ref_map: dict[int, str] = {}
                for i_row, row in enumerate(source_ref_rows, start=1):
                    if not isinstance(row, dict):
                        continue
                    doi_row = _clean_doi_for_url(str(row.get("doi") or ""))
                    if doi_row:
                        source_doi_ref_map[int(i_row)] = doi_row
                if source_doi_ref_map:
                    _prefetch_doi_meta_parallel(
                        source_doi_ref_map,
                        cache,
                        crossref_enabled=crossref_active_prefetch,
                        max_workers=int(max(1, doi_prefetch_workers)),
                        max_prefetch=int(max(10, int(doi_prefetch_workers) * 80)),
                        stats=crossref_stats,
                    )
                if crossref_enabled and (not crossref_budget_exhausted):
                    if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                        crossref_budget_exhausted = True
            if int(doi_prefetch_workers) > 1:
                crossref_active_prefetch = bool(
                    crossref_enabled and ((time.monotonic() - crossref_start_ts) <= float(max(5.0, crossref_time_budget_s)))
                )
                if crossref_active_prefetch:
                    doc_crossref_attempted = True
                    doc_crossref_attempt_ts = doc_crossref_attempt_ts or time.time()
                _prefetch_doi_meta_parallel(
                    ref_map,
                    cache,
                    crossref_enabled=crossref_active_prefetch,
                    max_workers=int(max(1, doi_prefetch_workers)),
                    max_prefetch=int(max(10, int(doi_prefetch_workers) * 40)),
                    stats=crossref_stats,
                )
                if crossref_enabled and (not crossref_budget_exhausted):
                    if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                        crossref_budget_exhausted = True
            order_mapping_quality_ok = _assess_source_reference_alignment(ref_map, source_ref_rows) if source_ref_rows else False
            should_prefetch_crossref_meta = bool((not source_ref_rows) or (not order_mapping_quality_ok))
            if int(doi_prefetch_workers) > 1 and should_prefetch_crossref_meta:
                crossref_active_prefetch = bool(
                    crossref_enabled and ((time.monotonic() - crossref_start_ts) <= float(max(5.0, crossref_time_budget_s)))
                )
                if crossref_active_prefetch:
                    doc_crossref_attempted = True
                    doc_crossref_attempt_ts = doc_crossref_attempt_ts or time.time()
                _prefetch_reference_meta_parallel(
                    ref_map,
                    cache,
                    crossref_enabled=crossref_active_prefetch,
                    enable_title_lookup=bool(enable_title_lookup),
                    max_workers=int(max(1, doi_prefetch_workers)),
                    max_prefetch=int(max(10, int(doi_prefetch_workers) * 50)),
                    stats=crossref_stats,
                )
                if crossref_enabled and (not crossref_budget_exhausted):
                    if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                        crossref_budget_exhausted = True
            ref_nums_sorted = sorted(ref_map.keys())
            can_map_by_exact_order = bool(
                source_ref_rows
                and order_mapping_quality_ok
                and ref_nums_sorted
                and (len(ref_nums_sorted) == len(source_ref_rows))
                and (ref_nums_sorted == list(range(1, len(ref_nums_sorted) + 1)))
            )
            can_map_by_prefix_order = False
            if source_ref_rows and order_mapping_quality_ok and ref_nums_sorted and ref_nums_sorted and (ref_nums_sorted[0] == 1):
                prefix_n = min(len(source_ref_rows), len(ref_nums_sorted))
                if prefix_n > 0:
                    want = list(range(1, prefix_n + 1))
                    got = ref_nums_sorted[:prefix_n]
                    can_map_by_prefix_order = (got == want)
            can_map_by_number_loose = False
            if source_ref_rows and order_mapping_quality_ok and ref_nums_sorted:
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
                            "authors": str(mapped.get("author") or "").strip(),
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
    
                if not isinstance(meta, dict):
                    crossref_active_now = bool(
                        crossref_enabled and ((time.monotonic() - crossref_start_ts) <= float(max(5.0, crossref_time_budget_s)))
                    )
                    if crossref_active_now:
                        doc_crossref_attempted = True
                        doc_crossref_attempt_ts = doc_crossref_attempt_ts or time.time()
                    meta, doi_hint = _lookup_crossref_meta_for_entry(
                        raw,
                        cache,
                        crossref_enabled=crossref_active_now,
                        enable_title_lookup=bool(enable_title_lookup),
                        stats=crossref_stats,
                    )
                    if crossref_enabled and (not crossref_budget_exhausted):
                        if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                            crossref_budget_exhausted = True
    
                crossref_active_now = bool(
                    crossref_enabled and ((time.monotonic() - crossref_start_ts) <= float(max(5.0, crossref_time_budget_s)))
                )
                if isinstance(meta, dict) and str(meta.get("match_method") or "").startswith("source_work_reference"):
                    if crossref_active_now and _reference_meta_is_sparse(meta):
                        doc_crossref_attempted = True
                        doc_crossref_attempt_ts = doc_crossref_attempt_ts or time.time()
                        supplemental_meta, doi_hint2 = _lookup_crossref_meta_for_entry(
                            raw,
                            cache,
                            crossref_enabled=True,
                            enable_title_lookup=bool(enable_title_lookup),
                            stats=crossref_stats,
                        )
                        if str(doi_hint2 or "").strip() and (not str(doi_hint or "").strip()):
                            doi_hint = doi_hint2
                        meta = _merge_reference_meta(meta, supplemental_meta)
                        if crossref_enabled and (not crossref_budget_exhausted):
                            if (time.monotonic() - crossref_start_ts) > float(max(5.0, crossref_time_budget_s)):
                                crossref_budget_exhausted = True

                raw_meta = _fallback_meta_from_raw_reference(raw)
                meta = _merge_raw_reference_meta(meta, raw_meta)
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
                        match0 = str(meta.get("match_method") or "").strip() if isinstance(meta, dict) else ""
                        title_words = [w for w in re.split(r"\s+", title0) if w]
                        noisy_title = (not title0) or (len(title0) >= 200) or (len(title_words) >= 28)
                        needs_backfill = (not authors0) or (not venue0) or (not year0) or (noisy_title) or ((not vol0) and (not pages0))
                        doi_key = _clean_doi_for_url(doi).lower()
                        doi_cache = cache.get("doi")
                        if not isinstance(doi_cache, dict):
                            doi_cache = {}
                            cache["doi"] = doi_cache
                        by_doi = None
                        doi_backfill_skip_miss = False
                        cached_doi_meta: dict | None = None
                        if needs_backfill and doi_key:
                            cached_doi = doi_cache.get(doi_key)
                            if _is_crossref_meta_cache_hit(cached_doi):
                                _stat_inc(crossref_stats, "crossref_cache_hits")
                                _stat_inc(crossref_stats, "doi_cache_hits")
                                cached_doi_meta = dict(cached_doi) if isinstance(cached_doi, dict) else None
                                if (
                                    isinstance(cached_doi_meta, dict)
                                    and _reference_meta_is_sparse(cached_doi_meta)
                                    and crossref_enabled
                                ):
                                    _stat_inc(crossref_stats, "doi_sparse_cache_refreshes")
                                else:
                                    by_doi = cached_doi_meta
                            elif _is_fresh_crossref_cache_miss(cached_doi):
                                _stat_inc(crossref_stats, "crossref_negative_hits")
                                _stat_inc(crossref_stats, "doi_negative_hits")
                                doi_backfill_skip_miss = True
                        if (not isinstance(by_doi, dict)) and needs_backfill and crossref_enabled and (not doi_backfill_skip_miss):
                            _stat_inc(crossref_stats, "crossref_network_attempts")
                            _stat_inc(crossref_stats, "doi_network_attempts")
                            doc_crossref_attempted = True
                            doc_crossref_attempt_ts = doc_crossref_attempt_ts or time.time()
                            by_doi = fetch_best_crossref_meta(
                                query_title=("" if noisy_title else title0),
                                doi_hint=doi,
                                allow_title_only=True,
                                min_score=0.88,
                            )
                            if isinstance(by_doi, dict) and by_doi and doi_key:
                                doi_cache[doi_key] = by_doi
                            elif isinstance(cached_doi_meta, dict) and cached_doi_meta:
                                by_doi = cached_doi_meta
                            elif doi_key:
                                doi_cache[doi_key] = _crossref_cache_miss("doi_backfill_not_found")
                                _stat_inc(crossref_stats, "crossref_negative_writes")
                                _stat_inc(crossref_stats, "doi_negative_writes")
                            if isinstance(by_doi, dict) and by_doi:
                                merged = dict(meta) if isinstance(meta, dict) else {}
                                if noisy_title or ("raw_meta" in match0) or (not str(merged.get("title") or "").strip()):
                                    merged["title"] = str(by_doi.get("title") or merged.get("title") or "").strip()
                                if not str(merged.get("authors") or "").strip():
                                    merged["authors"] = str(by_doi.get("authors") or "").strip()
                                venue_new = str(by_doi.get("venue") or "").strip()
                                if _should_replace_venue_with_external(str(merged.get("venue") or ""), venue_new):
                                    merged["venue"] = venue_new
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
                        elif isinstance(by_doi, dict) and by_doi:
                            merged = dict(meta) if isinstance(meta, dict) else {}
                            if noisy_title or ("raw_meta" in match0) or (not str(merged.get("title") or "").strip()):
                                merged["title"] = str(by_doi.get("title") or merged.get("title") or "").strip()
                            if not str(merged.get("authors") or "").strip():
                                merged["authors"] = str(by_doi.get("authors") or "").strip()
                            venue_new = str(by_doi.get("venue") or "").strip()
                            if _should_replace_venue_with_external(str(merged.get("venue") or ""), venue_new):
                                merged["venue"] = venue_new
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
    
                title_fallback = _fallback_title_from_raw_reference(raw)
                if isinstance(meta, dict):
                    if title_fallback and (not str(meta.get("title") or "").strip()):
                        meta["title"] = title_fallback
                        mm = str(meta.get("match_method") or "").strip()
                        if mm:
                            if "raw_title" not in mm:
                                meta["match_method"] = f"{mm}+raw_title"
                        else:
                            meta["match_method"] = "raw_title"
    
                match_method = (
                    str(meta.get("match_method") or "").strip()
                    if isinstance(meta, dict)
                    else ("raw_title" if str(title_fallback or "").strip() else "")
                )
                rec = {
                    "num": int(n),
                    "raw": raw,
                    "doi": doi,
                    "doi_url": doi_url,
                    "title": (
                        str(meta.get("title") or "").strip()
                        if isinstance(meta, dict)
                        else str(title_fallback or "").strip()
                    ),
                    "authors": str(meta.get("authors") or "").strip() if isinstance(meta, dict) else "",
                    "venue": str(meta.get("venue") or "").strip() if isinstance(meta, dict) else "",
                    "year": str(meta.get("year") or "").strip() if isinstance(meta, dict) else "",
                    "volume": str(meta.get("volume") or "").strip() if isinstance(meta, dict) else "",
                    "issue": str(meta.get("issue") or "").strip() if isinstance(meta, dict) else "",
                    "pages": str(meta.get("pages") or "").strip() if isinstance(meta, dict) else "",
                    "crossref_ok": bool(
                        isinstance(meta, dict)
                        and _reference_match_method_has_external_resolution(match_method, doi)
                    ),
                    "match_method": match_method,
                    "match_score": _float_meta_value(meta.get("match_score"), 0.0) if isinstance(meta, dict) else 0.0,
                    "parse_confidence": _reference_parse_confidence(raw),
                    "tail_continuity_status": str(reference_catalog.get("tail_continuity_status") or "").strip(),
                }
                rec.update(classify_reference_metadata(rec, enable_title_lookup=bool(enable_title_lookup)))
                refs_obj[str(int(n))] = rec
                if promising_ref:
                    has_resolved = bool(str(rec.get("doi") or "").strip() or bool(rec.get("crossref_ok")))
                    if not has_resolved:
                        unresolved_promising += 1
                    elif (not str(rec.get("title") or "").strip()) or (not str(rec.get("authors") or "").strip()):
                        sparse_promising += 1
    
            if need_crossref_enrich:
                unresolved_promising, sparse_promising = _assess_doc_crossref_enrichment(refs_obj)
            crossref_enriched_doc = (not need_crossref_enrich) or _doc_crossref_enriched(refs_obj)
            crossref_last_attempt_at = 0.0
            if need_crossref_enrich and doc_crossref_attempted:
                crossref_last_attempt_at = float(doc_crossref_attempt_ts or time.time())
            elif isinstance(prev_doc, dict):
                try:
                    crossref_last_attempt_at = float(prev_doc.get("crossref_last_attempt_at") or 0.0)
                except Exception:
                    crossref_last_attempt_at = 0.0
    
            built_doc = {
                "path": src_path,
                "name": p.name,
                "stem": p.stem.lower(),
                "sha1": sha1,
                "source_doi": source_doi,
                "crossref_enriched": bool(crossref_enriched_doc),
                "crossref_last_attempt_at": float(crossref_last_attempt_at),
                "crossref_unresolved_promising": int(unresolved_promising),
                "crossref_sparse_promising": int(sparse_promising),
                "crossref_retry_ttl_s": int(_reference_sync_retry_ttl_s()) if need_crossref_enrich else 0,
                "reference_lookup_version": int(REFERENCE_LOOKUP_VERSION),
                "reference_parser_version": int(REFERENCE_PARSER_VERSION),
                "reference_catalog_status": str(reference_catalog.get("tail_continuity_status") or "").strip(),
                "reference_catalog_ref_count": int(reference_catalog.get("ref_count") or 0),
                "reference_catalog_missing_numbers": list(reference_catalog.get("missing_numbers") or []),
                "refs": refs_obj,
                **quality_fields,
            }
    
            if (
                incremental
                and isinstance(prev_doc, dict)
                and str(prev_doc.get("sha1") or "") == str(sha1 or "")
                and isinstance(prev_doc.get("refs"), dict)
                and (not prev_parser_stale)
                and (not prev_needs_rebuild_for_catalog)
                and (not (prev_lookup_stale and _doc_refs_have_parser_generated_metadata(prev_doc.get("refs"))))
                and _prefer_previous_doc_refs(prev_doc.get("refs"), refs_obj)
            ):
                kept_doc = dict(prev_doc)
                if quality_fields:
                    kept_doc.update(quality_fields)
                if need_crossref_enrich and doc_crossref_attempted:
                    prev_unresolved, prev_sparse = _assess_doc_crossref_enrichment(
                        prev_doc.get("refs") if isinstance(prev_doc.get("refs"), dict) else None
                    )
                    kept_doc["crossref_last_attempt_at"] = float(doc_crossref_attempt_ts or time.time())
                    kept_doc["crossref_unresolved_promising"] = int(prev_unresolved)
                    kept_doc["crossref_sparse_promising"] = int(prev_sparse)
                    kept_doc["crossref_retry_ttl_s"] = int(_reference_sync_retry_ttl_s())
                    kept_doc["reference_lookup_version"] = int(REFERENCE_LOOKUP_VERSION)
                kept_refs, hydrated_count = _hydrate_doc_refs_from_crossref_cache(
                    prev_doc.get("refs") if isinstance(prev_doc.get("refs"), dict) else None,
                    cache,
                    enable_title_lookup=bool(enable_title_lookup),
                    stats=crossref_stats,
                )
                if int(hydrated_count) > 0:
                    kept_doc["refs"] = kept_refs
                    unresolved_now, sparse_now = _assess_doc_crossref_enrichment(kept_refs)
                    kept_doc["crossref_enriched"] = bool(_doc_crossref_enriched(kept_refs))
                    kept_doc["crossref_unresolved_promising"] = int(unresolved_now)
                    kept_doc["crossref_sparse_promising"] = int(sparse_now)
                    kept_doc["reference_lookup_version"] = int(REFERENCE_LOOKUP_VERSION)
                    docs_updated += 1
                    _stat_inc(crossref_stats, "docs_cache_hydrated")
                else:
                    docs_reused += 1
                docs_out[src_key] = kept_doc
                _record_ref_stats(kept_doc.get("refs"))
                _emit_progress("doc_done", docs_done=doc_i, current=p.name)
                continue
    
            docs_out[src_key] = built_doc
            docs_updated += 1
            _record_ref_stats(refs_obj)
            _emit_progress("doc_done", docs_done=doc_i, current=p.name)
    finally:
        if prep_executor is not None:
            try:
                prep_executor.shutdown(wait=False, cancel_futures=False)
            except Exception:
                pass

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

    elapsed_s = max(0.0, time.time() - run_started_ts)
    metadata_missing_doi = max(0, refs_total - refs_with_doi)
    metadata_missing_title = max(0, refs_total - refs_with_title)
    metadata_missing_authors = max(0, refs_total - refs_with_authors)
    metadata_missing_venue = max(0, refs_total - refs_with_venue)
    out_stats: dict[str, Any] = {
        "docs_total": len(md_files),
        "docs_indexed": len(docs_out),
        "docs_updated": int(docs_updated),
        "docs_reused": int(docs_reused),
        "docs_quality_blocked": int(docs_quality_blocked),
        "refs_total": int(refs_total),
        "refs_with_doi": int(refs_with_doi),
        "refs_with_title": int(refs_with_title),
        "refs_with_authors": int(refs_with_authors),
        "refs_with_venue": int(refs_with_venue),
        "refs_metadata_ready": int(refs_metadata_ready),
        "refs_metadata_user_ready": int(refs_metadata_user_ready),
        "refs_missing_doi": int(metadata_missing_doi),
        "refs_missing_title": int(metadata_missing_title),
        "refs_missing_authors": int(metadata_missing_authors),
        "refs_missing_venue": int(metadata_missing_venue),
        "refs_unresolved": int(refs_unresolved),
        "refs_crossref_ok": int(refs_crossref_ok),
        "refs_source_map_ok": int(refs_source_map_ok),
        "crossref_enabled": 1 if bool(crossref_enabled) else 0,
        "crossref_budget_exhausted": 1 if bool(crossref_budget_exhausted) else 0,
        "elapsed_s": round(float(elapsed_s), 3),
    }
    for code in REFERENCE_METADATA_STATUS_CODES:
        out_stats[f"refs_metadata_status_{code}"] = 0
    for code in REFERENCE_MISSING_REASON_CODES:
        out_stats[f"refs_missing_reason_{code}"] = 0
    for action in ("none", "auto_backfill", "retry", "source_repair", "non_article_ok", "retry_or_source_repair"):
        out_stats[f"refs_action_{action}"] = 0
    out_stats["refs_web_source_ok"] = 0
    out_stats.update({k: int(v) for k, v in crossref_stats.items()})
    return out_stats
