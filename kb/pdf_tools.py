from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
import json
import html as html_lib
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

import fitz  # PyMuPDF
from .citation_meta import extract_first_doi, fetch_best_crossref_meta, title_similarity


@dataclass
class PdfMetaSuggestion:
    venue: str = ""
    year: str = ""
    title: str = ""
    crossref_meta: dict | None = None  # Stored Crossref metadata if trusted


# Keep abbreviations explicit and deterministic (avoid LLM guessing for filenames).
# This built-in map acts as a safe fallback when the external json is missing/invalid.
_DEFAULT_VENUE_ABBR_MAP: dict[str, str] = {
    # Optics / photonics
    "laser photonics reviews": "LPR",
    "light science applications": "LSA",
    "advanced photonics": "AdvPhoton",
    "photonics research": "PhotonicsRes",
    "optica": "Optica",
    "optics express": "OE",
    "optics letters": "OL",
    "applied optics": "AO",
    "journal of lightwave technology": "JLT",
    "ieee photonics technology letters": "PTL",
    "ieee journal of selected topics in quantum electronics": "JSTQE",
    "laser physics letters": "LaserPhysLett",
    # Nature / Science / Cell families
    "nature": "Nature",
    "science": "Science",
    "ieee transactions on image processing": "IEEE-TIP",
    "ieee trans image process": "IEEE-TIP",
    "ieee transactions on pattern analysis and machine intelligence": "IEEE-TPAMI",
    "ieee trans pattern anal mach intell": "IEEE-TPAMI",
    "ieee transactions on visualization and computer graphics": "IEEE-TVCG",
    "ieee transactions on circuits and systems for video technology": "IEEE-TCSVT",
    "ieee transactions on medical imaging": "IEEE-TMI",
    "ieee transactions on multimedia": "IEEE-TMM",
    "ieee transactions on neural networks and learning systems": "IEEE-TNNLS",
    "ieee journal of biomedical and health informatics": "IEEE-JBHI",
    "pattern recognition": "PR",
    "international journal of computer vision": "IJCV",
    "computer vision and image understanding": "CVIU",
    "medical image analysis": "MedIA",
    "knowledge based systems": "KBS",
    "information fusion": "InfoFusion",
    "signal processing": "SignalProcess",
    "expert systems with applications": "ESWA",
    "journal of machine learning research": "JMLR",
    "acm transactions on graphics": "ACM-TOG",
    "acm tog": "ACM-TOG",
    "proceedings of the acm on computer graphics and interactive techniques": "PACM-CGIT",
    "acm computing surveys": "ACM-CSUR",
    "acm transactions on intelligent systems and technology": "ACM-TIST",
    "acm transactions on information systems": "ACM-TOIS",
    "nature machine intelligence": "NatMachIntell",
    "nature methods": "NatMethods",
    "nature materials": "NatMater",
    "nature physics": "NatPhys",
    "nature electronics": "NatElectron",
    "nature biotechnology": "NatBiotechnol",
    "nature biomedical engineering": "NatBiomedEng",
    "nature computational science": "NatComputSci",
    "nature communications": "NatCommun",
    "nature photonics": "NatPhoton",
    "science advances": "SciAdv",
    "cell": "Cell",
    "cell reports": "CellRep",
    "the lancet": "Lancet",
    "new england journal of medicine": "NEJM",
    "jama": "JAMA",
    # APS / physics
    "physical review letters": "PRL",
    "physical review x": "PRX",
    "physical review applied": "PRApplied",
    "physical review research": "PRResearch",
    "physical review a": "PRA",
    "physical review b": "PRB",
    "physical review c": "PRC",
    "physical review d": "PRD",
    "physical review e": "PRE",
    # CV/ML/NLP conferences
    "cvpr": "CVPR",
    "conference on computer vision and pattern recognition": "CVPR",
    "proceedings of the ieee cvf conference on computer vision and pattern recognition": "CVPR",
    "ieee cvf conference on computer vision and pattern recognition": "CVPR",
    "ieee conference on computer vision and pattern recognition": "CVPR",
    "proceedings of the ieee cvf international conference on computer vision": "ICCV",
    "ieee cvf international conference on computer vision": "ICCV",
    "european conference on computer vision": "ECCV",
    "iccv": "ICCV",
    "eccv": "ECCV",
    "wacv": "WACV",
    "accv": "ACCV",
    "icip": "ICIP",
    "ieee international conference on image processing": "ICIP",
    "proceedings of the ieee international conference on image processing": "ICIP",
    "icassp": "ICASSP",
    "icme": "ICME",
    "neurips": "NeurIPS",
    "icml": "ICML",
    "iclr": "ICLR",
    "aaai": "AAAI",
    "ijcai": "IJCAI",
    "kdd": "KDD",
    "www": "WWW",
    "the web conference": "WWW",
    "sigir": "SIGIR",
    "acl": "ACL",
    "emnlp": "EMNLP",
    "naacl": "NAACL",
    "coling": "COLING",
    # HCI / robotics
    "chi": "CHI",
    "uist": "UIST",
    "icra": "ICRA",
    "iros": "IROS",
    "robotics science and systems": "RSS",
    # Archives / preprints
    "siggraph asia": "SIGGRAPH-Asia",
    "siggraph": "SIGGRAPH",
    "arxiv": "arXiv",
}
_VENUE_ABBR_JSON_PATH = Path(__file__).resolve().with_name("venue_abbr_map.json")


# Bump this whenever the PDF meta extraction heuristics change in a way that should
# invalidate any UI/session caches that store extracted metadata.
PDF_META_EXTRACT_VERSION = "2026-02-19.1"


def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _sanitize_component(s: str) -> str:
    s = html_lib.unescape((s or "").strip())
    s = re.sub(r"\s+", " ", s)
    # Windows file name forbidden chars
    s = re.sub(r'[<>:"/\\\\|?*]+', "-", s)
    s = s.replace("\u0000", "").strip()
    # avoid trailing dots/spaces
    s = s.strip(" .-_")
    return s


def _venue_key(venue: str) -> str:
    v = html_lib.unescape((venue or "").strip()).lower()
    v = v.replace("&", " ")
    v = re.sub(r"[^a-z0-9]+", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _normalize_venue_abbr_map(raw_map: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for k, v in (raw_map or {}).items():
        key = _venue_key(str(k))
        val = _sanitize_component(str(v))
        if key and val:
            normalized[key] = val
    return normalized


def _load_venue_abbr_map() -> dict[str, str]:
    merged = _normalize_venue_abbr_map(_DEFAULT_VENUE_ABBR_MAP)
    try:
        if _VENUE_ABBR_JSON_PATH.exists():
            data = json.loads(_VENUE_ABBR_JSON_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(_normalize_venue_abbr_map(data))
    except Exception:
        # Keep runtime stable if the json config is malformed.
        pass
    return merged


_VENUE_ABBR_MAP = _load_venue_abbr_map()


def abbreviate_venue(venue: str) -> str:
    raw = _sanitize_component(venue)
    if not raw:
        return ""

    def _keys(raw_text: str) -> list[str]:
        txt = html_lib.unescape((raw_text or "").strip())
        if not txt:
            return []
        cands: list[str] = []
        seen: set[str] = set()

        def _add(s: str) -> None:
            k = _venue_key(s)
            if k and (k not in seen):
                seen.add(k)
                cands.append(k)

        txt0 = txt
        txt1 = re.sub(r"^\s*(?:19\d{2}|20\d{2})\s+", "", txt0, flags=re.IGNORECASE)
        txt2 = re.sub(r"^\s*(?:proceedings\s+of|in\s+proceedings\s+of)\s+", "", txt1, flags=re.IGNORECASE)
        txt3 = re.sub(r"\([^)]*\)", " ", txt2)
        txt4 = re.sub(r"^\s*(?:ieee(?:\s+cvf)?|acm)\s+", "", txt3, flags=re.IGNORECASE)

        for s in (txt0, txt1, txt2, txt3, txt4):
            _add(s)
        return cands

    # 1) Direct/normalized phrase matching.
    for key in _keys(raw):
        if key in _VENUE_ABBR_MAP:
            return _VENUE_ABBR_MAP[key]

    # 2) Acronym fallback: prefer explicit acronyms in parentheses, then standalone tokens.
    ignored = {"IEEE", "ACM", "CVF", "USA", "UK", "EU"}
    ac_tokens: list[str] = []
    for seg in re.findall(r"\(([^)]{1,28})\)", raw):
        for tk in re.split(r"[^A-Za-z0-9]+", seg):
            t = (tk or "").strip().upper()
            if re.fullmatch(r"[A-Z][A-Z0-9]{1,11}", t):
                ac_tokens.append(t)
    for tk in re.findall(r"\b[A-Z][A-Z0-9]{1,11}\b", raw):
        ac_tokens.append(str(tk).upper())

    seen_ac: set[str] = set()
    for ac in ac_tokens:
        if (not ac) or (ac in seen_ac) or (ac in ignored):
            continue
        seen_ac.add(ac)
        k = _venue_key(ac)
        if k in _VENUE_ABBR_MAP:
            return _VENUE_ABBR_MAP[k]
        # Keep well-formed conference/journal acronyms as-is.
        if 3 <= len(ac) <= 10:
            return ac

    # 3) Soft fallback: strip verbose wrappers before truncating title in filenames.
    compact = re.sub(r"^\s*(?:19\d{2}|20\d{2})\s+", "", raw, flags=re.IGNORECASE)
    compact = re.sub(r"^\s*(?:proceedings\s+of|in\s+proceedings\s+of)\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"^\s*(?:ieee(?:\s+cvf)?|acm)\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", " ", compact).strip(" .-_")
    compact = _sanitize_component(compact)
    if compact and (len(compact) + 6 <= len(raw)):
        return compact

    return raw


def venue_abbreviation_pairs() -> list[tuple[str, str]]:
    pairs = [
        ("Laser & Photonics Reviews", "LPR"),
        ("Light: Science & Applications", "LSA"),
        ("Optica", "Optica"),
        ("Optics Express", "OE"),
        ("Optics Letters", "OL"),
        ("Applied Optics", "AO"),
        ("Photonics Research", "PhotonicsRes"),
        ("IEEE Transactions on Image Processing", "IEEE-TIP"),
        ("IEEE Transactions on Pattern Analysis and Machine Intelligence", "IEEE-TPAMI"),
        ("ACM Transactions on Graphics", "ACM-TOG"),
        ("Physical Review Letters", "PRL"),
        ("Physical Review X", "PRX"),
        ("Nature Communications", "NatCommun"),
        ("Nature Photonics", "NatPhoton"),
        ("Science Advances", "SciAdv"),
        ("CVPR", "CVPR"),
        ("ICCV", "ICCV"),
        ("ECCV", "ECCV"),
        ("WACV", "WACV"),
        ("NeurIPS", "NeurIPS"),
        ("ICML", "ICML"),
        ("AAAI", "AAAI"),
        ("ACL", "ACL"),
        ("EMNLP", "EMNLP"),
    ]
    return pairs


def _safe_base_len_for_paths(
    *,
    pdf_dir: Path | str | None = None,
    md_out_root: Path | str | None = None,
    safe_path_limit: int = 230,
    default_base_max: int = 88,
) -> int:
    """
    Choose a conservative base-name cap so that both:
    - <pdf_dir>/<base>.pdf
    - <md_out_root>/<base>/<base>.en.md
    stay comfortably below problematic path lengths on Windows/tooling.
    """
    cap = int(default_base_max)
    cap = max(40, min(160, cap))
    limit = int(max(120, safe_path_limit))

    try:
        if pdf_dir:
            pdf_root_len = len(str(Path(pdf_dir).expanduser().resolve()))
            # "<root>\<base>.pdf"
            pdf_budget = limit - pdf_root_len - len("\\") - len(".pdf")
            cap = min(cap, int(pdf_budget))
    except Exception:
        pass

    try:
        if md_out_root:
            md_root_len = len(str(Path(md_out_root).expanduser().resolve()))
            # "<root>\<base>\<base>.en.md"
            md_budget = (limit - md_root_len - len("\\") * 2 - len(".en.md")) // 2
            cap = min(cap, int(md_budget))
    except Exception:
        pass

    return max(40, min(160, int(cap)))


def _truncate_with_hash(text: str, max_len: int) -> str:
    s = (text or "").strip()
    lim = max(16, int(max_len))
    if len(s) <= lim:
        return s
    digest = hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]
    tail = "-" + digest
    head_len = max(8, lim - len(tail))
    head = s[:head_len].rstrip(" .-_")
    return (head + tail)[:lim]


def build_base_name(
    venue: str,
    year: str,
    title: str,
    *,
    shorten_venue: bool = True,
    pdf_dir: Path | str | None = None,
    md_out_root: Path | str | None = None,
    max_len: int | None = None,
) -> str:
    venue = abbreviate_venue(venue) if shorten_venue else _sanitize_component(venue)
    year = _sanitize_component(year)
    title = _sanitize_component(title)

    parts = []
    if venue:
        parts.append(venue)
    if year:
        parts.append(year)
    if title:
        parts.append(title)
    base = "-".join(parts) if parts else "paper"

    target_len = int(max_len) if isinstance(max_len, int) and max_len > 0 else _safe_base_len_for_paths(
        pdf_dir=pdf_dir,
        md_out_root=md_out_root,
    )
    target_len = max(40, min(160, target_len))
    if len(base) <= target_len:
        return base

    # Keep venue/year stable, shorten title first, then add hash for uniqueness.
    prefix_parts: list[str] = []
    if venue:
        prefix_parts.append(venue)
    if year:
        prefix_parts.append(year)
    prefix = "-".join(prefix_parts).strip("-")

    if prefix and title:
        # Reserve room for "<prefix>-<title_part>" plus hash suffix.
        digest = hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest()[:8]
        suffix = "-" + digest
        body_budget = max(16, target_len - len(suffix))
        title_budget = body_budget - len(prefix) - 1
        if title_budget >= 10:
            title_cut = title[:title_budget].rstrip(" .-_")
            body = f"{prefix}-{title_cut}".strip("-")
            return (body + suffix)[:target_len]

    return _truncate_with_hash(base, target_len)


def copy_upload_to_dir(uploaded_file: Any, dst_dir: Path) -> Path:
    """
    Streamlit uploaded_file has .name and .getbuffer().
    Save to a tmp name to allow user to edit metadata before final rename.
    """
    ensure_dir(dst_dir)
    name = getattr(uploaded_file, "name", "upload.pdf") or "upload.pdf"
    name = _sanitize_component(Path(name).stem) or "upload"
    tmp_path = Path(dst_dir) / f"__upload__{name}.pdf"
    data = uploaded_file.getbuffer()
    tmp_path.write_bytes(data)
    return tmp_path


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_RE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _guess_year(text: str) -> str:
    """
    Try to guess *publication* year (not a random cited year).

    Heuristics (in priority order):
    - "Publication date: <Month> <YYYY>" (ACM DL style)
    - "Published/Accepted/Received" nearby year
    - arXiv id "arXiv:YYMM.xxxxx" -> 20YY
    - Otherwise: prefer the most frequent / most recent year, but only from short top text
    """
    t = " ".join((text or "").split())
    if not t:
        return ""

    # ACM-style
    m = re.search(r"Publication\s*date\s*:\s*[A-Za-z]+\s+(19\d{2}|20\d{2})", t, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Generic "Published/Accepted/Received" nearby year
    m = re.search(r"(Published|Accepted|Received)\b[^.\n]{0,80}?\b(19\d{2}|20\d{2})\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(2)

    # arXiv ID: arXiv:2403.20018 -> 2024
    m = re.search(r"arXiv\s*:\s*(\d{2})(\d{2})\.\d{4,5}", t, flags=re.IGNORECASE)
    if m:
        yy = int(m.group(1))
        year = 2000 + yy if 0 <= yy <= 99 else 0
        if 1900 <= year <= 2099:
            return str(year)

    years = _RE_YEAR.findall(t)
    if not years:
        return ""

    ys = [int(y) for y in years if 1900 <= int(y) <= 2099]
    if not ys:
        return ""

    # Prefer the most frequent year; tie-break by most recent.
    freq: dict[int, int] = {}
    for y in ys:
        freq[y] = freq.get(y, 0) + 1
    best = sorted(freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]
    return str(best)


def _guess_venue(text: str) -> str:
    t = " ".join((text or "").split())
    # IMPORTANT: order matters (match more specific venues before generic ones).
    # Example: "Science Advances" must not be collapsed to "Science".
    patterns: list[tuple[str, str | None]] = [
        # Science family (specific) — must come before generic Science markers.
        (r"\bScience\s+Advances\b", "Science Advances"),
        (r"\bSci\.\s*Adv\.\b", "Science Advances"),
        (r"\bSci\s*\.?\s*Adv\.?\b", "Science Advances"),
        (r"\bSciAdv\b", "Science Advances"),
        (r"\badvances\.science(?:mag)?\.org\b", "Science Advances"),
        (r"\bscienceadvances\.org\b", "Science Advances"),

        # Nature family (specific) — must come before generic Nature markers.
        (r"\bNature\s+Communications\b", "Nature Communications"),
        (r"\bNat\.\s*Commun\.\b", "Nature Communications"),
        (r"\bNature\s+Photonics\b", "Nature Photonics"),
        (r"\bNat\.\s*Photon\.\b", "Nature Photonics"),
        (r"\bnaturecommunications\.com\b", "Nature Communications"),
        (r"\bnature(?:-|\.)?photonics\b", "Nature Photonics"),
        (r"\bNature\s+Machine\s+Intelligence\b", "Nature Machine Intelligence"),
        (r"\bNature\s+Methods\b", "Nature Methods"),

        # Strong publisher/header markers (avoid "Computer Science"/"Optical Science" false matches)
        (r"\bnature\.com\b", "Nature"),
        (r"\bNature\s*\|\s*Vol\b", "Nature"),
        (r"\bscience\.org\b", "Science"),
        (r"\bScience\s*\|\s*Vol\b", "Science"),
        (r"\bAAAS\b", "Science"),

        # Optics / imaging journals
        (r"\bLight\s*:\s*Science\s*(?:&|and)\s*Applications\b", "Light: Science & Applications"),
        (r"\bLaser\s*(?:&|and)\s*Photonics\s+Reviews\b", "Laser & Photonics Reviews"),
        (r"\bPhotonics\s+Research\b", "Photonics Research"),
        (r"\bAdvanced\s+Photonics\b", "Advanced Photonics"),
        (r"\bOptica\b", "Optica"),
        (r"\bOptics\s+Express\b", "Optics Express"),
        (r"\bOptics\s+Letters\b", "Optics Letters"),
        (r"\bApplied\s+Optics\b", "Applied Optics"),
        (r"\bAppl\.\s*Opt\.\b", "Applied Optics"),
        (r"\bPhysical\s+Review\s+Letters\b", "Physical Review Letters"),
        (r"\bPhys\.\s*Rev\.\s*Lett\.\b", "Physical Review Letters"),
        (r"\bPhysical\s+Review\s+X\b", "Physical Review X"),
        # ACM / IEEE (keep existing patterns)
        (r"ACM\s+TOG\b", None),
        (r"ACM\s+Trans(?:actions)?\.?\s+Graph(?:ics)?\.?", None),
        (r"ACM\s+Transactions\s+on\s+Graphics", None),
        (r"Proc\.\s+ACM\s+Comput\.\s+Graph\.\s+Interact\.\s+Tech\.", None),
        (r"IEEE\s+Trans\.\s+Pattern\s+Anal\.\s+Mach\.\s+Intell\.", None),
        (r"IEEE\s+Trans\.\s+Image\s+Process\.", None),
        (r"IEEE\s+Transactions\s+on\s+[A-Za-z ]+", None),
        # Conferences (with year if present)
        (r"\bCVPR\s+\d{4}\b", None),
        (r"\bICCV\s+\d{4}\b", None),
        (r"\bECCV\s+\d{4}\b", None),
        (r"\bWACV\s+\d{4}\b", None),
        (r"\bNeurIPS\s+\d{4}\b", None),
        (r"\bICML\s+\d{4}\b", None),
        (r"\bICLR\s+\d{4}\b", None),
        (r"\bAAAI\s+\d{4}\b", None),
        (r"\bIJCAI\s+\d{4}\b", None),
        (r"\bSIGGRAPH(?:\s+Asia)?\s+\d{4}\b", None),
        # Conferences (without year)
        (r"\bCVPR\b", "CVPR"),
        (r"\bICCV\b", "ICCV"),
        (r"\bECCV\b", "ECCV"),
        (r"\bWACV\b", "WACV"),
        (r"\bNeurIPS\b", "NeurIPS"),
        (r"\bICML\b", "ICML"),
        (r"\bICLR\b", "ICLR"),
        (r"\bAAAI\b", "AAAI"),
        (r"\bIJCAI\b", "IJCAI"),
        (r"\bACL\b", "ACL"),
        (r"\bEMNLP\b", "EMNLP"),
        (r"\bNAACL\b", "NAACL"),
        # Preprints
        (r"\barXiv\b", "arXiv"),
    ]
    for pat, canon in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return canon or m.group(0)

    # Proceedings long-form (map to short venue when possible)
    m = re.search(r"Proceedings\s+of\s+the\s+IEEE/CVF\s+Conference\s+on\s+Computer\s+Vision\s+and\s+Pattern\s+Recognition", t, flags=re.IGNORECASE)
    if m:
        return "CVPR"
    m = re.search(r"Proceedings\s+of\s+the\s+IEEE/CVF\s+International\s+Conference\s+on\s+Computer\s+Vision", t, flags=re.IGNORECASE)
    if m:
        return "ICCV"
    m = re.search(r"European\s+Conference\s+on\s+Computer\s+Vision", t, flags=re.IGNORECASE)
    if m:
        return "ECCV"
    m = re.search(r"Conference\s+on\s+Computer\s+Vision\s+and\s+Pattern\s+Recognition", t, flags=re.IGNORECASE)
    if m:
        return "CVPR"
    m = re.search(r"International\s+Conference\s+on\s+Machine\s+Learning", t, flags=re.IGNORECASE)
    if m:
        return "ICML"
    m = re.search(r"International\s+Conference\s+on\s+Learning\s+Representations", t, flags=re.IGNORECASE)
    if m:
        return "ICLR"
    return ""


def _is_generic_venue(venue: str) -> bool:
    v = _venue_key(venue)
    return v in {"science", "nature"}


def _parse_filename_meta(stem_or_name: str) -> tuple[str, str, str]:
    stem = Path(stem_or_name or "").stem or (stem_or_name or "")
    if stem.lower().endswith(".en"):
        stem = stem[:-3]
    parts = [p.strip() for p in stem.split("-") if p.strip()]
    if len(parts) < 3:
        return "", "", ""
    year_idx = -1
    for i, p in enumerate(parts):
        if re.fullmatch(r"(19\d{2}|20\d{2})", p):
            year_idx = i
            break
    if year_idx <= 0 or year_idx >= (len(parts) - 1):
        return "", "", ""
    venue = _sanitize_component("-".join(parts[:year_idx]))
    year = parts[year_idx]
    title = _sanitize_component("-".join(parts[year_idx + 1 :]))
    return venue, year, title


def _extract_top_text(page: fitz.Page, y_frac: float = 0.40) -> str:
    """
    Extract text from the top area of the first page to avoid picking years from references.
    """
    try:
        h = float(page.rect.height)
        y_max = h * float(y_frac)
        blocks = page.get_text("blocks")
    except Exception:
        return ""

    parts: list[str] = []
    for b in blocks:
        if len(b) < 5:
            continue
        y0 = float(b[1])
        if y0 > y_max:
            continue
        txt = (b[4] or "").strip()
        if not txt:
            continue
        parts.append(txt)

    return "\n".join(parts)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _title_from_font_spans(page: fitz.Page) -> str:
    """
    Try to extract the paper title from the first page by font-size cues.
    This reads the PDF *content*, not the filename.
    """
    try:
        d = page.get_text("dict")
    except Exception:
        return ""

    W = float(page.rect.width)
    H = float(page.rect.height)
    y_max = H * 0.45

    lines: list[tuple[float, float, str]] = []  # (y0, size, text)
    max_size = 0.0

    for b in d.get("blocks", []) or []:
        if b.get("type") != 0:
            continue
        bbox = b.get("bbox") or None
        if not bbox or len(bbox) < 4:
            continue
        y0 = float(bbox[1])
        if y0 > y_max:
            continue
        for ln in b.get("lines", []) or []:
            spans = ln.get("spans", []) or []
            if not spans:
                continue
            text = "".join([s.get("text", "") for s in spans]).strip()
            text = _normalize_ws(text)
            if not text:
                continue
            size = max(float(s.get("size", 0.0) or 0.0) for s in spans)
            # Filter obvious non-title junk
            low = text.lower()
            if "abstract" == low or low.startswith("abstract "):
                continue
            if "arxiv" in low and len(text) < 30:
                continue
            if re.search(r"university|inria|max planck|institute|department", low):
                continue
            # avoid very narrow single-word lines
            if len(text) < 8 and (len(text.split()) <= 1):
                continue
            # Often author list is wide but has many commas
            if text.count(",") >= 3 and len(text) < 120:
                continue
            lines.append((y0, float(size), text))
            max_size = max(max_size, float(size))

    if not lines or max_size <= 0.0:
        return ""

    # Keep only the largest-font lines (title tends to be the largest on page 1).
    # Allow multi-line title: include nearby lines with close font size.
    keep: list[tuple[float, float, str]] = []
    for y0, size, text in lines:
        if size >= (max_size - 0.6):
            keep.append((y0, size, text))
    keep.sort(key=lambda x: x[0])

    # Merge consecutive lines that look like a single title block.
    merged: list[str] = []
    last_y = None
    for y0, _size, text in keep:
        if last_y is None:
            merged.append(text)
        else:
            if (y0 - last_y) <= (H * 0.03):
                # same title block
                merged.append(text)
            else:
                break
        last_y = y0

    title = _normalize_ws(" ".join(merged))
    # Basic sanity: titles are usually centered-ish / not too long
    if len(title) < 10 or len(title) > 180:
        return ""
    # Avoid lines that are likely headers
    if W > 10 and title.lower().startswith(("proceedings of", "arxiv:", "https://", "doi:")):
        return ""
    return title


def _guess_title_from_first_page(doc: fitz.Document) -> str:
    try:
        page = doc.load_page(0)
    except Exception:
        return ""

    # Prefer font-size based title extraction if available.
    title2 = _title_from_font_spans(page)
    if title2:
        return title2

    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, block_type)
    text_blocks = []
    for b in blocks:
        if len(b) < 5:
            continue
        txt = (b[4] or "").strip()
        if not txt:
            continue
        # skip obvious headers/footers
        if len(txt) < 6:
            continue
        text_blocks.append((float(b[1]), float(b[0]), txt))

    if not text_blocks:
        return ""

    # sort by y then x
    text_blocks.sort(key=lambda x: (x[0], x[1]))

    # title often is near top and relatively short (not authors list)
    candidates = []
    for y, x, txt in text_blocks[:20]:
        one = " ".join(txt.split())
        if len(one) < 10 or len(one) > 140:
            continue
        # heuristic: avoid "Abstract" and affiliations
        if re.search(r"\babstract\b", one, flags=re.IGNORECASE):
            continue
        if re.search(r"university|inria|max planck|institute|department", one, flags=re.IGNORECASE):
            continue
        candidates.append(one)

    return candidates[0] if candidates else ""


def _llm_extract_meta_from_text(settings: Any, text: str) -> PdfMetaSuggestion | None:
    """
    Optional, higher-precision extraction using the configured LLM.
    This uses *PDF text* only (no filename).
    """
    if not settings or not getattr(settings, "api_key", None):
        return None
    t = _normalize_ws(text)
    if not t or len(t) < 80:
        return None

    try:
        from .llm import DeepSeekChat  # local import to avoid hard dependency for scripts
    except Exception:
        return None

    ds = DeepSeekChat(settings)
    sys_msg = (
        "你是文献元数据抽取器。\n"
        "你必须只输出 JSON：{\"title\":string,\"year\":string,\"venue\":string}。\n"
        "规则：\n"
        "1) title：论文标题，尽量与文中标题一致，不要加作者。\n"
        "2) year：4位年份（如 2024），必须是发表/会议年份；不确定就留空。\n"
        "3) venue：期刊/会议/平台的简称或全称（如 CVPR 2023, Optics Express, arXiv），不确定就留空。\n"
        "4) 不要根据文件名猜。\n"
    )
    user_msg = f"PDF第一页/前两页顶部文字（可能含作者/摘要/版权信息）：\n{text}\n"
    try:
        out = (ds.chat(messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}], temperature=0.0, max_tokens=220) or "").strip()
    except Exception:
        return None

    # Allow fenced json
    if out.startswith("```"):
        out = out.strip()
        out = out.strip("`")
        out = out.replace("json", "", 1).strip()

    try:
        import json

        data = json.loads(out)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    title = _sanitize_component(str(data.get("title") or "").strip())
    year = _sanitize_component(str(data.get("year") or "").strip())
    venue = _sanitize_component(str(data.get("venue") or "").strip())

    if year and not re.fullmatch(r"(19\d{2}|20\d{2})", year):
        year = ""

    # light sanity: title too short is likely wrong
    if title and len(title) < 8:
        title = ""

    if not (title or year or venue):
        return None
    return PdfMetaSuggestion(venue=venue, year=year, title=title)


def extract_pdf_meta_suggestion(pdf_path: Path, *, settings: Any | None = None) -> PdfMetaSuggestion:
    pdf_path = Path(pdf_path)
    file_venue, file_year, file_title = _parse_filename_meta(pdf_path.name)
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return PdfMetaSuggestion()

    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()
    if not title or title.lower() in ("untitled", "unknown"):
        title = _guess_title_from_first_page(doc)

    # First page text for venue/year guesses (prefer top region to avoid references noise)
    first_text = ""
    top_text = ""
    try:
        page0 = doc.load_page(0)
        first_text = page0.get_text("text") or ""
        top_text = _extract_top_text(page0, y_frac=0.42)
        # Some PDFs place venue/year on page 2 header; include a small slice.
        if doc.page_count >= 2:
            try:
                page1 = doc.load_page(1)
                top_text2 = _extract_top_text(page1, y_frac=0.28)
                if top_text2:
                    top_text = (top_text + "\n" + top_text2).strip()
                # Also include a limited amount of page-2 full text for venue hints.
                p1_text = page1.get_text("text") or ""
                if p1_text:
                    first_text = (first_text + "\n" + p1_text[:4000]).strip()
            except Exception:
                pass
    except Exception:
        first_text = ""
        top_text = ""

    year = _guess_year(top_text) or _guess_year(first_text)
    venue = _guess_venue(top_text) or _guess_venue(first_text)
    if (not venue or _is_generic_venue(venue)) and file_venue and (not _is_generic_venue(file_venue)):
        venue = file_venue
    if (not title) and file_title:
        title = file_title

    # Optional LLM refinement when heuristics are missing/suspicious.
    need_llm = (not title) or (not year) or (not venue) or _is_generic_venue(venue)
    if need_llm and settings:
        sugg = _llm_extract_meta_from_text(settings, top_text or first_text)
        if sugg:
            if not title and sugg.title:
                title = sugg.title
            if not year and sugg.year:
                year = sugg.year
            if sugg.venue and (not venue or _is_generic_venue(venue)):
                venue = sugg.venue

    # Crossref refinement (DOI first, then strict title search).
    # For rename suggestions we still keep a conservative gate to avoid bad overwrites.
    try:
        doi_hint = extract_first_doi(str(meta.get("doi") or ""))
    except Exception:
        doi_hint = ""
    if not doi_hint:
        doi_hint = extract_first_doi(top_text) or extract_first_doi(first_text)

    query_title = title or file_title
    expected_year = year or file_year
    expected_venue = venue
    if (not expected_venue or _is_generic_venue(expected_venue)) and file_venue and (not _is_generic_venue(file_venue)):
        expected_venue = file_venue

    cross = None
    cross_trusted = False
    try:
        cross = fetch_best_crossref_meta(
            query_title=query_title,
            expected_year=expected_year,
            expected_venue=expected_venue,
            doi_hint=doi_hint,
            min_score=0.84,
        )
    except Exception:
        cross = None

    # Retry once with relaxed constraints when year guess is noisy.
    if (not isinstance(cross, dict)) and query_title:
        retry_venue = "" if _is_generic_venue(expected_venue) else expected_venue
        try:
            cross = fetch_best_crossref_meta(
                query_title=query_title,
                expected_year="",
                expected_venue=retry_venue,
                doi_hint=doi_hint,
                min_score=0.90,
            )
        except Exception:
            cross = None

    # Very strict title-only fallback (for cases where venue/year extraction is empty but title is clean).
    if (not isinstance(cross, dict)) and query_title and len(_normalize_ws(query_title)) >= 24:
        try:
            cross = fetch_best_crossref_meta(
                query_title=query_title,
                expected_year="",
                expected_venue="",
                doi_hint=doi_hint,
                min_score=0.97,
                allow_title_only=True,
            )
        except Exception:
            cross = None

    crossref_meta_stored = None
    if isinstance(cross, dict):
        c_title = _sanitize_component(str(cross.get("title") or "").strip())
        c_year = _sanitize_component(str(cross.get("year") or "").strip())
        c_venue = _sanitize_component(str(cross.get("venue") or "").strip())
        c_method = str(cross.get("match_method") or "").strip()
        c_score = float(cross.get("match_score") or 0.0)
        c_trusted = (c_method == "doi") or (c_score >= 0.90)
        cross_trusted = bool(c_trusted)

        if c_title:
            t_sim = title_similarity(title, c_title) if title else 1.0
            if (not title) or c_trusted or (t_sim >= 0.88):
                title = c_title

        if c_trusted:
            if c_year and re.fullmatch(r"(19\d{2}|20\d{2})", c_year):
                year = c_year
            else:
                year = ""
            if c_venue:
                venue = c_venue
            # Store trusted Crossref metadata for later use
            crossref_meta_stored = dict(cross)
        else:
            # For filename suggestions, prefer empty year over a potentially wrong year.
            year = ""
    else:
        # User preference: if Crossref cannot be confirmed, keep year empty.
        year = ""

    # Safety fallback: avoid generic top-level journal names when untrusted.
    if _is_generic_venue(venue) and (not cross_trusted):
        venue = ""
    if (not venue) and file_venue and (not _is_generic_venue(file_venue)):
        venue = file_venue
    if (not title) and file_title:
        title = file_title

    # Cleanups
    title = _sanitize_component(title)
    venue = _sanitize_component(venue)
    year = _sanitize_component(year)

    return PdfMetaSuggestion(venue=venue, year=year, title=title, crossref_meta=crossref_meta_stored)


def open_in_explorer(path: Path) -> None:
    """
    Open a directory, or reveal a file in Explorer.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        if p.is_dir():
            subprocess.Popen(["explorer", str(p)])
        else:
            subprocess.Popen(["explorer", "/select,", str(p)])
    except Exception:
        pass


def run_pdf_to_md(
    pdf_path: Path,
    out_root: Path,
    no_llm: bool,
    keep_debug: bool,
    eq_image_fallback: bool,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    heartbeat_s: float = 1.0,
    stall_timeout_s: float | None = None,
    speed_mode: str | None = None,
    _safe_retry_attempt: int = 0,
) -> tuple[bool, str]:
    """
    Convert a PDF into a markdown folder under out_root/pdf_stem.

    Preferred path:
    - Use an external converter script (more capable) if provided via KB_PDF_CONVERTER
      or if a repo-local pdf_to_md.py exists.
    Fallback:
    - Use a built-in fast text extractor (PyMuPDF) so conversion works out-of-the-box
      for collaborators.
    """
    pdf_path = Path(pdf_path)
    out_root = Path(out_root)
    ensure_dir(out_root)
    # In no-LLM mode, extracted equations are often garbled. Force image fallback to keep rendering faithful.
    if bool(no_llm) and (not bool(eq_image_fallback)):
        eq_image_fallback = True

    if stall_timeout_s is None:
        try:
            # Output-stall watchdog remains opt-in because some stages can stay quiet for long periods.
            # Enable via env: KB_PDF_PROGRESS_STALL_TIMEOUT_S=<seconds>.
            raw_stall = (os.environ.get("KB_PDF_PROGRESS_STALL_TIMEOUT_S") or "").strip()
            stall_timeout_s = float(raw_stall) if raw_stall else 0.0
        except Exception:
            stall_timeout_s = 0.0
    try:
        heartbeat_s = max(0.25, float(heartbeat_s))
    except Exception:
        heartbeat_s = 1.0

    def _fallback_convert() -> tuple[bool, str]:
        out_dir = out_root / pdf_path.stem
        ensure_dir(out_dir)
        md_path = out_dir / f"{pdf_path.stem}.en.md"

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            return False, f"open pdf failed: {e}"

        parts: list[str] = []
        try:
            total_pages = int(getattr(doc, "page_count", 0) or 0)
            for i in range(total_pages):
                if cancel_cb is not None:
                    try:
                        if bool(cancel_cb()):
                            return False, "cancelled"
                    except Exception:
                        pass
                try:
                    page = doc.load_page(i)
                    txt = (page.get_text("text") or "").strip()
                except Exception:
                    txt = ""
                if txt:
                    parts.append(txt)
                if progress_cb is not None:
                    try:
                        progress_cb(i + 1, total_pages, f"fallback page {i+1}/{total_pages}")
                    except Exception:
                        pass
        finally:
            try:
                doc.close()
            except Exception:
                pass

        body = "\n\n---\n\n".join(parts).strip()
        if not body:
            body = "（未能从 PDF 提取到可检索的文本：可能是扫描版，或文本被嵌入为图片。）"

        try:
            md_path.write_text(body, encoding="utf-8")
            if keep_debug:
                (out_dir / "_converter.txt").write_text("fallback=pymupdf_text\n", encoding="utf-8")
        except Exception as e:
            return False, f"write md failed: {e}"

        return True, str(out_dir)

    # Prefer an explicit path override (portable across machines / folders):
    # - KB_PDF_CONVERTER: absolute path to a converter script (e.g. pdf_to_md.py)
    # - fallback: resolve to repo-local ../pdf_to_md.py from this file.
    #   Keep legacy fallbacks to test2.py for older layouts.
    override = (os.environ.get("KB_PDF_CONVERTER") or "").strip().strip('"').strip("'")
    if override:
        script = Path(override).expanduser()
    else:
        local_script = Path(__file__).resolve().parents[1] / "pdf_to_md.py"
        legacy_script = Path(__file__).resolve().parents[2] / "pdf_to_md.py"
        local_compat = Path(__file__).resolve().parents[1] / "test2.py"
        legacy_compat = Path(__file__).resolve().parents[2] / "test2.py"
        if local_script.exists():
            script = local_script
        elif legacy_script.exists():
            script = legacy_script
        elif local_compat.exists():
            script = local_compat
        else:
            script = legacy_compat
    if not script.exists():
        # Collaborator-friendly fallback: still produce an .en.md so the app can run.
        return _fallback_convert()

    # -u: unbuffered, so we can parse per-page progress from stdout in real time.
    args = [sys.executable, "-u", str(script), "--pdf", str(pdf_path), "--out", str(out_root)]

    def _env_int(name: str, default: int = 0, *, lo: int = 0, hi: int = 256) -> int:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return int(default)
        try:
            v = int(raw)
        except Exception:
            return int(default)
        return max(lo, min(hi, v))

    def _env_bool(name: str, default: bool = False) -> bool:
        raw = (os.environ.get(name) or "").strip().lower()
        if not raw:
            return bool(default)
        return raw in {"1", "true", "yes", "y", "on"}

    def _probe_pdf_pages(path: Path) -> int:
        try:
            with fitz.open(str(path)) as d:
                return int(getattr(d, "page_count", 0) or 0)
        except Exception:
            return 0

    def _auto_llm_workers_defaults(pages: int, cpu: int) -> tuple[int, int]:
        # Keep quality settings intact; only tune concurrency.
        if pages <= 3:
            workers = 1
        elif pages <= 8:
            workers = 2
        elif pages <= 20:
            workers = 4
        else:
            workers = 6

        llm_workers = 1 if pages <= 4 else 3

        if cpu <= 2:
            workers = 1
            llm_workers = 1
        elif cpu <= 4:
            workers = min(workers, 2)
        elif cpu <= 6:
            workers = min(workers, 3)
        else:
            workers = min(workers, 6)

        max_inflight = 12
        while (workers * llm_workers) > max_inflight:
            if llm_workers > 1:
                llm_workers -= 1
            elif workers > 1:
                workers -= 1
            else:
                break
        return max(1, workers), max(1, llm_workers)

    def _auto_no_llm_workers_default(pages: int, cpu: int) -> int:
        if pages <= 2:
            return 1
        if cpu <= 2:
            return 1
        return max(2, min(12, cpu - 1))

    page_count = _probe_pdf_pages(pdf_path)
    cpu_count = max(1, int(os.cpu_count() or 1))

    # Web/UI conversion defaults to an adaptive profile:
    # - keep quality features on
    # - tune concurrency by document size and host CPU
    classify_batch_size = _env_int("KB_PDF_CLASSIFY_BATCH_SIZE", default=80, lo=8, hi=256)
    if classify_batch_size > 0:
        args.extend(["--classify-batch-size", str(classify_batch_size)])

    if bool(no_llm):
        ui_workers_default = _auto_no_llm_workers_default(page_count, cpu_count)
        ui_llm_workers_default = 0
    else:
        ui_workers_default, ui_llm_workers_default = _auto_llm_workers_defaults(page_count, cpu_count)

    ui_workers = _env_int("KB_PDF_WORKERS", default=ui_workers_default, lo=0, hi=64)
    if ui_workers > 0:
        args.extend(["--workers", str(ui_workers)])
    ui_llm_workers = _env_int("KB_PDF_LLM_WORKERS", default=ui_llm_workers_default, lo=0, hi=32)
    if ui_llm_workers > 0:
        args.extend(["--llm-workers", str(ui_llm_workers)])

    ui_llm_timeout = _env_int("KB_PDF_LLM_TIMEOUT_S", default=25, lo=1, hi=600)
    if ui_llm_timeout > 0:
        args.extend(["--llm-timeout", str(ui_llm_timeout)])
    ui_llm_retries = _env_int("KB_PDF_LLM_RETRIES", default=0, lo=0, hi=20)
    args.extend(["--llm-retries", str(ui_llm_retries)])

    # Page-progress watchdog (separate from output-stall watchdog):
    # terminate if no page has finished for too long, to avoid endless "14/15 alive ...".
    # Default is conservative to avoid killing genuinely heavy pages.
    if bool(no_llm):
        page_stall_default = 240.0
    else:
        # Example (default): timeout=25,retries=0 -> 300s floor; timeout=120,retries=0 -> 600s.
        page_stall_default = max(
            300.0,
            min(1800.0, float(ui_llm_timeout) * float(max(1, ui_llm_retries + 2)) * 2.5),
        )
    try:
        raw_page_stall = (os.environ.get("KB_PDF_PAGE_STALL_TIMEOUT_S") or "").strip()
        page_stall_timeout_s = float(raw_page_stall) if raw_page_stall else float(page_stall_default)
    except Exception:
        page_stall_timeout_s = float(page_stall_default)
    if page_stall_timeout_s <= 0:
        page_stall_timeout_s = 0.0

    auto_page_llm_threshold_default = 8
    if page_count >= 15:
        auto_page_llm_threshold_default = 10
    if page_count >= 30:
        auto_page_llm_threshold_default = 12
    auto_page_llm_threshold = _env_int(
        "KB_PDF_AUTO_PAGE_LLM_THRESHOLD",
        default=auto_page_llm_threshold_default,
        lo=0,
        hi=200,
    )
    args.extend(["--auto-page-llm-threshold", str(auto_page_llm_threshold)])
    if _env_bool("KB_PDF_FAST", default=False):
        args.append("--fast")

    if progress_cb is not None:
        try:
            progress_cb(
                0,
                0,
                (
                    "converter profile: "
                    f"script={str(script)}, "
                    f"workers={ui_workers if ui_workers > 0 else 'auto'}, "
                    f"llm_workers={ui_llm_workers if ui_llm_workers > 0 else 'auto'}, "
                    f"llm_timeout={ui_llm_timeout}s, llm_retries={ui_llm_retries}, "
                    f"page_stall_timeout={int(page_stall_timeout_s)}s, "
                    f"auto_page_llm_threshold={auto_page_llm_threshold}, "
                    f"classify_batch={classify_batch_size}, "
                    f"pages={page_count}, cpu={cpu_count}"
                ),
            )
        except Exception:
            pass
    if progress_cb is not None:
        try:
            # Provide immediate total-page visibility in UI even before child stdout arrives.
            progress_cb(0, max(0, int(page_count or 0)), "converter starting...")
        except Exception:
            pass

    if keep_debug:
        args.append("--keep-debug")
    if no_llm:
        args.append("--no-llm")
    if eq_image_fallback:
        args.append("--eq-image-fallback")
    
    # Add speed mode from parameter or environment variable
    if speed_mode is None:
        speed_mode = os.environ.get("KB_PDF_SPEED_MODE", "balanced")
    args.extend(["--speed-mode", str(speed_mode)])

    def _terminate_proc(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is not None:
                return
        except Exception:
            return
        try:
            proc.terminate()
            proc.wait(timeout=4)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)
        except Exception:
            pass

    p_total = 0
    p_done = 0
    try:
        # Stream stdout so we can parse per-page progress emitted by the converter.
        # Example lines:
        # - "Detected body font size: ... | pages: 12 | range: 1-12"
        # - "Processing page 3/12 ..."
        p_total = max(0, int(page_count or 0))
        p_done = 0
        cp_out = []
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            progress_cb and progress_cb(0, p_total, f"converter pid={int(proc.pid)}")
        except Exception:
            pass
        assert proc.stdout is not None
        line_q: queue.Queue[Optional[str]] = queue.Queue()

        def _reader() -> None:
            try:
                for line in proc.stdout:
                    line_q.put(line)
            except Exception:
                pass
            finally:
                try:
                    line_q.put(None)
                except Exception:
                    pass

        rd = threading.Thread(target=_reader, daemon=True)
        rd.start()

        re_pages = re.compile(r"\bpages\s*:\s*(\d+)\b", flags=re.IGNORECASE)
        re_prog = re.compile(r"Processing\s+page\s+(\d+)\s*/\s*(\d+)", flags=re.IGNORECASE)
        re_done = re.compile(r"Finished\s+page\s+(\d+)\s*/\s*(\d+)", flags=re.IGNORECASE)
        re_page_hint = re.compile(r"\bPage\s+(\d+)\s*:", flags=re.IGNORECASE)
        re_page_bracket = re.compile(r"\[\s*Page\s+(\d+)\s*\]", flags=re.IGNORECASE)
        re_fail_page_a = re.compile(r"\berror\s+processing\s+page\s+(\d+)\b", flags=re.IGNORECASE)
        re_fail_page_b = re.compile(r"\berror\s+page\s+(\d+)\b", flags=re.IGNORECASE)
        re_fail_page_c = re.compile(r"\bpage\s+(\d+)\s+failed\b", flags=re.IGNORECASE)
        re_running_pages = re.compile(r"still\s+running\s+pages\s*:\s*\[([0-9,\s]+)\]", flags=re.IGNORECASE)
        done_pages: set[int] = set()
        current_page_inflight = 0
        last_line_ts = time.time()
        last_done_ts = time.time()
        last_heartbeat_ts = 0.0
        rc_override: Optional[int] = None
        while True:
            got_line = False
            line: Optional[str] = None
            try:
                line = line_q.get(timeout=0.35)
                got_line = True
            except queue.Empty:
                got_line = False

            if got_line and (line is None):
                break

            if got_line:
                s = (line or "").rstrip()
                if s:
                    cp_out.append(s)
                    last_line_ts = time.time()
                m1 = re_pages.search(s)
                if m1:
                    try:
                        p_total = max(p_total, int(m1.group(1)))
                    except Exception:
                        pass
                m2 = re_prog.search(s)
                if m2:
                    try:
                        current_page_inflight = max(current_page_inflight, int(m2.group(1)))
                        p_total = max(p_total, int(m2.group(2)))
                    except Exception:
                        pass
                m3 = re_done.search(s)
                if m3:
                    try:
                        old_done = int(p_done)
                        pg = int(m3.group(1))
                        p_total = max(p_total, int(m3.group(2)))
                        done_pages.add(pg)
                        p_done = max(p_done, len(done_pages))
                        if p_done > old_done:
                            last_done_ts = time.time()
                    except Exception:
                        pass
                m4 = re_page_hint.search(s)
                if m4:
                    try:
                        current_page_inflight = max(current_page_inflight, int(m4.group(1)))
                    except Exception:
                        pass
                m5 = re_page_bracket.search(s)
                if m5:
                    try:
                        current_page_inflight = max(current_page_inflight, int(m5.group(1)))
                    except Exception:
                        pass
                m6 = re_running_pages.search(s)
                if m6:
                    try:
                        vals = [int(x.strip()) for x in str(m6.group(1) or "").split(",") if x.strip().isdigit()]
                        if vals:
                            current_page_inflight = max(current_page_inflight, max(vals))
                    except Exception:
                        pass
                # Treat failed pages as "processed" for progress accounting.
                # Otherwise UI can appear stuck at N-1/N during long tail post-processing.
                fm = re_fail_page_a.search(s) or re_fail_page_b.search(s) or re_fail_page_c.search(s)
                if fm:
                    try:
                        old_done = int(p_done)
                        pgf = int(fm.group(1))
                        if pgf > 0:
                            done_pages.add(pgf)
                            p_done = max(p_done, len(done_pages))
                            current_page_inflight = max(current_page_inflight, pgf)
                            if p_done > old_done:
                                last_done_ts = time.time()
                    except Exception:
                        pass
                try:
                    progress_cb and progress_cb(p_done, p_total, s)
                except Exception:
                    pass

            now = time.time()
            if cancel_cb is not None:
                try:
                    if bool(cancel_cb()):
                        rc_override = -2
                        _terminate_proc(proc)
                        break
                except Exception:
                    pass

            if (stall_timeout_s is not None) and (stall_timeout_s > 0):
                try:
                    if (now - last_line_ts) >= float(stall_timeout_s):
                        rc_override = -3
                        _terminate_proc(proc)
                        break
                except Exception:
                    pass

            # Page-progress stall watchdog (independent from stdout heartbeat activity).
            # If no page finishes for too long while pages remain, abort to avoid endless hangs.
            if (page_stall_timeout_s is not None) and (page_stall_timeout_s > 0):
                try:
                    if (p_total > 0) and (p_done < p_total) and ((now - last_done_ts) >= float(page_stall_timeout_s)):
                        rc_override = -4
                        stalled_page = max(1, int(p_done) + 1, int(current_page_inflight or 0))
                        try:
                            progress_cb and progress_cb(
                                p_done,
                                p_total,
                                f"converter page-progress stalled at {stalled_page}/{p_total} for {int(now-last_done_ts)}s; terminating",
                            )
                        except Exception:
                            pass
                        _terminate_proc(proc)
                        break
                except Exception:
                    pass

            if (progress_cb is not None) and ((now - last_heartbeat_ts) >= heartbeat_s):
                last_idle_s = max(0, int(now - last_line_ts))
                if p_total > 0:
                    if p_done >= p_total:
                        base_msg = f"Post-processing after pages {p_total}/{p_total} ..."
                    else:
                        live_page = max(1, p_done + 1, current_page_inflight)
                        live_page = min(p_total, live_page)
                        base_msg = f"Processing page {live_page}/{p_total} ..."
                else:
                    base_msg = "converter running..." if not cp_out else cp_out[-1]
                heartbeat_msg = f"{base_msg} (alive {last_idle_s}s)"
                try:
                    progress_cb(p_done, p_total, heartbeat_msg)
                except Exception:
                    pass
                last_heartbeat_ts = now

            if (proc.poll() is not None) and line_q.empty():
                break

        if rc_override is not None:
            rc = int(rc_override)
        else:
            try:
                rc = int(proc.wait(timeout=8) or 0)
            except Exception:
                _terminate_proc(proc)
                rc = int(proc.wait() or 0)
    except Exception as e:
        # If the external converter is misconfigured on collaborator machines, don't hard-fail.
        ok2, out2 = _fallback_convert()
        if ok2:
            return True, out2
        return False, str(e)

    if rc == -2:
        return False, "cancelled"
    if rc == -3:
        # Source-level resilience: one automatic safe-profile retry for LLM mode.
        # This addresses common root causes of stalls (provider throttling / concurrency contention)
        # without requiring users to manually tune env vars.
        if (not bool(no_llm)) and int(_safe_retry_attempt) < 1:
            try:
                progress_cb and progress_cb(
                    p_done,
                    p_total,
                    "converter stalled on output heartbeat; retrying once with conservative profile (workers=1, llm_workers=1)",
                )
            except Exception:
                pass
            env_overrides = {
                "KB_PDF_WORKERS": "1",
                "KB_PDF_LLM_WORKERS": "1",
                "KB_LLM_MAX_INFLIGHT": "1",
                "KB_PDF_LLM_TIMEOUT_S": str(max(120, int(ui_llm_timeout))),
                "KB_PDF_LLM_RETRIES": str(max(1, int(ui_llm_retries))),
                "KB_PDF_PAGE_STALL_TIMEOUT_S": str(max(1200, int(page_stall_timeout_s or 0))),
            }
            old_env: dict[str, str | None] = {}
            try:
                for k, v in env_overrides.items():
                    old_env[k] = os.environ.get(k)
                    os.environ[k] = str(v)
                return run_pdf_to_md(
                    pdf_path=pdf_path,
                    out_root=out_root,
                    no_llm=no_llm,
                    keep_debug=keep_debug,
                    eq_image_fallback=eq_image_fallback,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                    heartbeat_s=heartbeat_s,
                    stall_timeout_s=stall_timeout_s,
                    speed_mode=speed_mode,
                    _safe_retry_attempt=int(_safe_retry_attempt) + 1,
                )
            finally:
                for k, ov in old_env.items():
                    if ov is None:
                        try:
                            del os.environ[k]
                        except Exception:
                            pass
                    else:
                        os.environ[k] = ov
        timeout_hint = int(float(stall_timeout_s or 0))
        return False, f"converter stalled (no output for {timeout_hint}s)"
    if rc == -4:
        # Source-level resilience: one automatic safe-profile retry for LLM mode.
        # This addresses common root causes of stalls (provider throttling / concurrency contention)
        # without requiring users to manually tune env vars.
        if (not bool(no_llm)) and int(_safe_retry_attempt) < 1:
            try:
                progress_cb and progress_cb(
                    p_done,
                    p_total,
                    "converter stalled on page completion; retrying once with conservative profile (workers=1, llm_workers=1)",
                )
            except Exception:
                pass
            env_overrides = {
                "KB_PDF_WORKERS": "1",
                "KB_PDF_LLM_WORKERS": "1",
                "KB_LLM_MAX_INFLIGHT": "1",
                "KB_PDF_LLM_TIMEOUT_S": str(max(120, int(ui_llm_timeout))),
                "KB_PDF_LLM_RETRIES": str(max(1, int(ui_llm_retries))),
                "KB_PDF_PAGE_STALL_TIMEOUT_S": str(max(1200, int(page_stall_timeout_s or 0))),
            }
            old_env: dict[str, str | None] = {}
            try:
                for k, v in env_overrides.items():
                    old_env[k] = os.environ.get(k)
                    os.environ[k] = str(v)
                return run_pdf_to_md(
                    pdf_path=pdf_path,
                    out_root=out_root,
                    no_llm=no_llm,
                    keep_debug=keep_debug,
                    eq_image_fallback=eq_image_fallback,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                    heartbeat_s=heartbeat_s,
                    stall_timeout_s=stall_timeout_s,
                    speed_mode=speed_mode,
                    _safe_retry_attempt=int(_safe_retry_attempt) + 1,
                )
            finally:
                for k, ov in old_env.items():
                    if ov is None:
                        try:
                            del os.environ[k]
                        except Exception:
                            pass
                    else:
                        os.environ[k] = ov
        timeout_hint = int(float(page_stall_timeout_s or 0))
        return False, f"converter stalled (no page finished for {timeout_hint}s)"

    if rc != 0:
        tail = "\n".join(cp_out).strip()[-800:]
        # Fall back to built-in extraction so users still get something usable.
        ok2, out2 = _fallback_convert()
        if ok2:
            return True, out2
        return False, f"exit={rc} {tail}"

    return True, str(out_root / pdf_path.stem)
