from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF


@dataclass
class PdfMetaSuggestion:
    venue: str = ""
    year: str = ""
    title: str = ""


# Bump this whenever the PDF meta extraction heuristics change in a way that should
# invalidate any UI/session caches that store extracted metadata.
PDF_META_EXTRACT_VERSION = "2026-02-09.1"


def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _sanitize_component(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    # Windows file name forbidden chars
    s = re.sub(r'[<>:"/\\\\|?*]+', "-", s)
    s = s.replace("\u0000", "").strip()
    # avoid trailing dots/spaces
    s = s.strip(" .-_")
    return s


def build_base_name(venue: str, year: str, title: str) -> str:
    venue = _sanitize_component(venue)
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
    # clamp
    return base[:160]


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
        (r"\badvances\.science(?:mag)?\.org\b", "Science Advances"),
        (r"\bscienceadvances\.org\b", "Science Advances"),

        # Nature family (specific) — must come before generic Nature markers.
        (r"\bNature\s+Communications\b", "Nature Communications"),
        (r"\bNat\.\s*Commun\.\b", "Nature Communications"),

        # Strong publisher/header markers (avoid "Computer Science"/"Optical Science" false matches)
        (r"\bnature\.com\b", "Nature"),
        (r"\bNature\s*\|\s*Vol\b", "Nature"),
        (r"\bscience\.org\b", "Science"),
        (r"\bScience\s*\|\s*Vol\b", "Science"),
        (r"\bAAAS\b", "Science"),

        # Optics / imaging journals
        (r"\bOptics\s+Express\b", "Optics Express"),
        (r"\bApplied\s+Optics\b", "Applied Optics"),
        (r"\bAppl\.\s*Opt\.\b", "Applied Optics"),
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
        (r"\bNeurIPS\s+\d{4}\b", None),
        (r"\bICLR\s+\d{4}\b", None),
        (r"\bSIGGRAPH(?:\s+Asia)?\s+\d{4}\b", None),
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
    return ""


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

    # Optional LLM refinement when heuristics are missing/suspicious.
    need_llm = (not title) or (not year) or (not venue)
    if need_llm and settings:
        sugg = _llm_extract_meta_from_text(settings, top_text or first_text)
        if sugg:
            if not title and sugg.title:
                title = sugg.title
            if not year and sugg.year:
                year = sugg.year
            if not venue and sugg.venue:
                venue = sugg.venue

    # Cleanups
    title = _sanitize_component(title)
    venue = _sanitize_component(venue)
    year = _sanitize_component(year)

    return PdfMetaSuggestion(venue=venue, year=year, title=title)


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
) -> tuple[bool, str]:
    """
    Convert a PDF into a markdown folder under out_root/pdf_stem.

    Preferred path:
    - Use an external converter script (more capable) if provided via KB_PDF_CONVERTER
      or if a repo-local test2.py exists.
    Fallback:
    - Use a built-in fast text extractor (PyMuPDF) so conversion works out-of-the-box
      for collaborators.
    """
    pdf_path = Path(pdf_path)
    out_root = Path(out_root)
    ensure_dir(out_root)

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
            for i in range(int(getattr(doc, "page_count", 0) or 0)):
                try:
                    page = doc.load_page(i)
                    txt = (page.get_text("text") or "").strip()
                except Exception:
                    txt = ""
                if txt:
                    parts.append(txt)
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
    # - KB_PDF_CONVERTER: absolute path to a converter script (e.g. test2.py)
    # - fallback: resolve to repo-relative ../..../test2.py from this file
    override = (os.environ.get("KB_PDF_CONVERTER") or "").strip().strip('"').strip("'")
    if override:
        script = Path(override).expanduser()
    else:
        script = Path(__file__).resolve().parents[2] / "test2.py"
    if not script.exists():
        # Collaborator-friendly fallback: still produce an .en.md so the app can run.
        return _fallback_convert()

    args = [sys.executable, str(script), "--pdf", str(pdf_path), "--out", str(out_root)]
    if keep_debug:
        args.append("--keep-debug")
    if no_llm:
        args.append("--no-llm")
    if eq_image_fallback:
        args.append("--eq-image-fallback")

    try:
        cp = subprocess.run(args, capture_output=True, text=True, check=False)
    except Exception as e:
        # If the external converter is misconfigured on collaborator machines, don't hard-fail.
        ok2, out2 = _fallback_convert()
        if ok2:
            return True, out2
        return False, str(e)

    if cp.returncode != 0:
        tail = (cp.stderr or cp.stdout or "").strip()[-800:]
        # Fall back to built-in extraction so users still get something usable.
        ok2, out2 = _fallback_convert()
        if ok2:
            return True, out2
        return False, f"exit={cp.returncode} {tail}"

    return True, str(out_root / pdf_path.stem)
