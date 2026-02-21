#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_to_md_v2.py

Research-grade PDF -> Markdown converter for RAG/LLM pipelines.
Architecture rewrite (not patching test2.py).

Primary output layout (required):
  <output>/<paper_name>/
    paper.md
    images/
    tables/
    meta.json

Compatibility extras (for legacy test2.py downstream):
  <output>/<paper_name>/
    assets/                  (mirror of images/)
    <paper_name>.en.md       (copy of paper.md)
    assets_manifest.md       (optional)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


_LLM_RUNTIME: Optional["LLMRuntime"] = None


LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

GREEK_TO_LATEX = {
    "\u03b1": r"\alpha",
    "\u03b2": r"\beta",
    "\u03b3": r"\gamma",
    "\u03b4": r"\delta",
    "\u03b5": r"\epsilon",
    "\u03b8": r"\theta",
    "\u03bb": r"\lambda",
    "\u03bc": r"\mu",
    "\u03c0": r"\pi",
    "\u03c3": r"\sigma",
    "\u03c4": r"\tau",
    "\u03c6": r"\phi",
    "\u03c9": r"\omega",
    "\u0393": r"\Gamma",
    "\u0394": r"\Delta",
    "\u039b": r"\Lambda",
    "\u03a0": r"\Pi",
    "\u03a3": r"\Sigma",
    "\u03a6": r"\Phi",
    "\u03a9": r"\Omega",
}

MATH_SYMBOL_TO_LATEX = {
    "\u2264": r"\leq",
    "\u2265": r"\geq",
    "\u2260": r"\neq",
    "\u2248": r"\approx",
    "\u2208": r"\in",
    "\u2209": r"\notin",
    "\u2211": r"\sum",
    "\u220f": r"\prod",
    "\u222b": r"\int",
    "\u221e": r"\infty",
    "\u2192": r"\to",
    "\u00d7": r"\times",
    "\u00b7": r"\cdot",
}

FIGURE_CAP_RE = re.compile(r"^\s*(?:Fig\.|Figure)\s*\d+\b", re.IGNORECASE)
TABLE_CAP_RE = re.compile(r"^\s*Table\s*(?:\d+|[IVXLC]+)\b", re.IGNORECASE)
REFERENCE_HEAD_RE = re.compile(
    r"^\s*(?:references(?:\s+and\s+links)?|bibliography|reference(?:s)?)\s*$",
    re.IGNORECASE,
)
PAGE_NUMBER_RE = re.compile(r"^\s*[\(\[]?\s*(?:\d+|[ivxlcdm]+)\s*[\)\]]?\s*$", re.IGNORECASE)
# Strict style used for mode decisions; avoids matching "1 Introduction".
NUMBERED_REF_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[\.\)])\s+(.+?)\s*$")
# Relaxed style used only when splitting already-detected reference content.
NUMBERED_REF_RELAXED_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[\.\)]?)\s+(.+?)\s*$")
AUTHOR_YEAR_START_RE = re.compile(
    r"^\s*[A-Z][A-Za-z'`\- ]{1,40},\s*(?:[A-Z]\.\s*){1,4}.*?\(\d{4}[a-z]?\)",
    re.IGNORECASE,
)
SECTION_HEAD_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Za-z][A-Za-z0-9,\- ]{1,120}$")

HF_STRONG_PATTERNS = [
    re.compile(r"\bOPTICS\s+EXPRESS\b", re.IGNORECASE),
    re.compile(r"\bVol\.\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bNo\.\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bReceived\b.*\bpublished\b", re.IGNORECASE),
    re.compile(r"\(\s*[cC]\s*\)\s*\d{4}\s*OSA\b", re.IGNORECASE),
    re.compile(r"#\d{4,}\s*-\s*\$\d", re.IGNORECASE),
]

FRONTMATTER_SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "results",
    "conclusion",
}

LLM_TASK_PROMPTS: dict[str, str] = {
    "structure_headings": (
        "Given heading candidates, fix heading levels only.\n"
        "Return JSON: {\"items\":[{\"index\":int,\"level\":int,\"label\":string,\"confidence\":number}]}\n"
        "Rules: keep original wording; do not invent sections."
    ),
    "split_references": (
        "Split reference lines into entries.\n"
        "Return JSON: {\"entries\":[string],\"confidence\":number}\n"
        "Rules: no hallucination; keep source text content."
    ),
    "table_repair_or_fallback": (
        "Decide whether a table candidate can be repaired.\n"
        "Return JSON: {\"action\":\"repair\"|\"fallback\",\"markdown_table\":string,\"confidence\":number}\n"
        "If unreliable, choose fallback."
    ),
    "validate_formula_latex": (
        "Validate a LaTeX formula candidate.\n"
        "Return JSON: {\"is_valid\":boolean,\"latex\":string,\"confidence\":number,\"reason\":string}\n"
        "If invalid, keep latex empty or minimally corrected."
    ),
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def safe_name(name: str) -> str:
    out = re.sub(r"[^\w\-\.]+", "_", name.strip())
    out = re.sub(r"_+", "_", out).strip("._")
    return out or "paper"


def fs_path(path: Path) -> str:
    """
    Normalize Windows long paths for robust file I/O.
    """
    p = Path(path)
    if os.name != "nt":
        return str(p)
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    try:
        s = str(p.resolve())
    except Exception:
        s = str(p)
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def sha8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r", " ").replace("\n", " ")
    for k, v in LIGATURES.items():
        t = t.replace(k, v)
    t = (
        t.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2212", "-")
    )
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def clean_template_text(text: str) -> str:
    t = normalize_text(text).lower()
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_balanced_brackets(text: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for ch in text:
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack:
                return False
            top = stack.pop()
            if pairs[top] != ch:
                return False
    return not stack


def rect_area(rect: "fitz.Rect") -> float:
    return max(0.0, float(rect.width) * float(rect.height))


def intersection_area(a: "fitz.Rect", b: "fitz.Rect") -> float:
    inter = fitz.Rect(a) & fitz.Rect(b)
    return rect_area(inter)


def overlap_ratio(a: "fitz.Rect", b: "fitz.Rect") -> float:
    ia = intersection_area(a, b)
    da = max(1.0, rect_area(a))
    return ia / da


def rows_to_markdown(rows: list[list[Any]]) -> Optional[str]:
    if not rows or len(rows) < 1:
        return None
    norm_rows: list[list[str]] = []
    max_cols = max((len(r) for r in rows), default=0)
    if max_cols <= 1:
        return None
    for row in rows:
        vals = [normalize_text("" if c is None else str(c)) for c in row]
        vals = vals + [""] * (max_cols - len(vals))
        norm_rows.append(vals)
    if not norm_rows:
        return None
    header = norm_rows[0]
    body = norm_rows[1:] if len(norm_rows) > 1 else []
    if not any(cell for cell in header):
        header = [f"col_{i+1}" for i in range(max_cols)]
    lines = [
        "| " + " | ".join(c.replace("|", r"\|") for c in header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(c.replace("|", r"\|") for c in row) + " |")
    out = "\n".join(lines)
    if out.count("|") < (max_cols * 2):
        return None
    return out


def markdown_table_quality(md: str) -> float:
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0
    pipe_density = sum(ln.count("|") for ln in lines) / max(1.0, len(lines))
    return min(1.0, pipe_density / 6.0)


def fix_hyphen_breaks(text: str) -> str:
    return re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)


def likely_math_text(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if TABLE_CAP_RE.match(t) or FIGURE_CAP_RE.match(t):
        return False
    if len(t) > 320:
        return False
    symbol_n = len(re.findall(r"[=<>^_\\+\-/*∑∫∞≤≥≠≈]", t))
    greek_n = len(re.findall(r"[α-ωΑ-Ω]", t))
    math_tokens = re.findall(r"\b(?:arg\s*min|arg\s*max|log|exp|sin|cos|tan|softmax|L\d+)\b", t, flags=re.IGNORECASE)
    if symbol_n >= 2 and (symbol_n + greek_n) / max(1, len(t)) >= 0.03:
        return True
    if greek_n >= 2 and symbol_n >= 1:
        return True
    if math_tokens and symbol_n >= 1:
        return True
    if re.search(r"^\s*\(?\s*\d+\s*\)?\s*$", t):
        return False
    words = re.findall(r"[A-Za-z]{2,}", t)
    if len(words) >= 8 and symbol_n <= 1 and greek_n == 0:
        return False
    non_word = re.findall(r"[^\w\s]", t)
    if len(non_word) >= 5 and (len(non_word) / max(1, len(t))) > 0.15:
        return True
    return False


def is_garbled_math_fragment(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if len(t) > 40:
        return False
    if re.search(r"[A-Za-z]{5,}", t):
        return False
    token_n = len(re.findall(r"[α-ωΑ-ΩΔΛΣΠΦΩ∂∇+\-=/(),.]", t))
    alpha_n = len(re.findall(r"[A-Za-z]", t))
    non_ascii_n = sum(1 for ch in t if ord(ch) > 127)
    punct_n = len(re.findall(r"[+\-=/(),.;:]", t))
    if token_n >= 4 and alpha_n <= 4:
        return True
    if len(t) <= 32 and non_ascii_n >= 1 and punct_n >= 1 and alpha_n <= 3:
        return True
    return False


def has_control_chars(text: str) -> bool:
    return bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text))


def looks_like_garbled_formula_line(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if len(t) > 180:
        return False
    words = re.findall(r"[A-Za-z]{2,}", t)
    if has_control_chars(t):
        if len(words) <= 14:
            return True
        if likely_math_text(t):
            return True
        if re.search(r"\brect\b|dxdyd|dx|dy|[=∑Δλ]", t, flags=re.IGNORECASE):
            return True
    if not likely_math_text(t):
        return False
    if has_control_chars(t):
        return True
    mojibake_n = len(re.findall(r"[鈭鈮锟�螖位]", t))
    weird_n = len(re.findall(r"[^\x20-\x7Eα-ωΑ-Ω≤≥≠≈∞∑∫×·\\{}^_+=\-*/().,:;<>|[\]\s]", t))
    if mojibake_n >= 1 and weird_n >= 1:
        return True
    if weird_n / max(1, len(t)) > 0.18:
        return True
    return False


def heading_text_quality(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if likely_math_text(t):
        return False
    if FIGURE_CAP_RE.match(t) or TABLE_CAP_RE.match(t):
        return False
    if has_control_chars(t):
        return False
    if re.search(r"\b(?:fig(?:ure)?|table)\.?\s*\d+\b", t, flags=re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", t)
    if len(words) == 0:
        return False
    if len(words) > 15 or len(t) > 120:
        return False
    if t.count(",") > 1:
        return False
    if re.search(r"[!?]", t):
        return False
    if t.endswith(".") and len(words) >= 4:
        return False
    digit_ratio = len(re.findall(r"\d", t)) / max(1, len(t))
    if digit_ratio > 0.30:
        return False
    if re.search(r"[01]{8,}", t):
        return False
    if re.search(r"\b\d+\s*(?:x|×)\s*10\s*[-−]?\d+\b", t, flags=re.IGNORECASE):
        return False
    if re.match(r"^\s*\d+\s*(?:nm|mm|cm|ms|s)\s*$", t, flags=re.IGNORECASE):
        return False
    return True


def looks_like_reference_entry(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if NUMBERED_REF_RE.match(t):
        return True
    if re.match(r"^\s*\d+\)\s+", t):
        return True
    if AUTHOR_YEAR_START_RE.match(t):
        return True
    if re.search(r"\bdoi\b|\barxiv\b|http[s]?://", t, flags=re.IGNORECASE):
        return True
    return False


def looks_like_content_section_heading(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if SECTION_HEAD_RE.match(t):
        return True
    if t.lower() in FRONTMATTER_SECTION_KEYWORDS:
        return True
    if re.match(r"^\s*(?:appendix|supplementary)\b", t, flags=re.IGNORECASE):
        return True
    return False


def normalize_heading_text(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return t
    t = re.sub(r"^\s*(\d+(?:\.\d+)*)\s*\.\s*", r"\1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences_keep(text: str) -> list[str]:
    chunks = [normalize_text(x) for x in re.split(r"\n+", text) if normalize_text(x)]
    return chunks


def estimate_body_font_size(pages: list[list["RawBlock"]]) -> float:
    sizes: list[float] = []
    for blocks in pages:
        for b in blocks:
            if len(b.text) < 3:
                continue
            if likely_math_text(b.text):
                continue
            if TABLE_CAP_RE.match(b.text) or FIGURE_CAP_RE.match(b.text):
                continue
            sizes.append(float(b.font_size))
    if not sizes:
        return 10.0
    try:
        return float(statistics.median(sizes))
    except Exception:
        return float(sizes[len(sizes) // 2])


@dataclass
class LLMConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_s: float = 40.0


@dataclass
class Config:
    input_path: Path
    output_dir: Path
    workers: int = 4
    use_llm: bool = False
    use_ocr: bool = False
    recursive: bool = False
    keep_compat_test2: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class RawBlock:
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    bold: bool
    page_index: int

    @property
    def rect(self) -> "fitz.Rect":
        return fitz.Rect(self.bbox)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])


@dataclass
class Element:
    kind: str
    text: str
    page_index: int
    bbox: tuple[float, float, float, float]
    markdown: str
    priority: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rect(self) -> "fitz.Rect":
        return fitz.Rect(self.bbox)


@dataclass
class FormulaStats:
    total_formula: int = 0
    latex_success: int = 0
    fallback_images: int = 0
    llm_attempts: int = 0


@dataclass
class TableStats:
    total_tables: int = 0
    structured_tables: int = 0
    fallback_images: int = 0
    ocr_hint_count: int = 0


@dataclass
class ReferenceStats:
    detected: bool = False
    start_page: Optional[int] = None
    entry_count: int = 0
    style: str = "unknown"
    confidence: float = 0.0
    llm_attempts: int = 0


class LLMRuntime:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled and cfg.api_key and OpenAI is not None)
        self.client = None
        if self.enabled:
            self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_s)

    def _validate_task_output(self, task_name: str, obj: dict[str, Any]) -> bool:
        try:
            if task_name == "validate_formula_latex":
                return (
                    isinstance(obj.get("is_valid"), bool)
                    and isinstance(obj.get("latex", ""), str)
                    and isinstance(float(obj.get("confidence", 0.0)), float)
                    and isinstance(obj.get("reason", ""), str)
                )
            if task_name == "split_references":
                entries = obj.get("entries")
                return isinstance(entries, list) and all(isinstance(x, str) for x in entries)
            if task_name == "table_repair_or_fallback":
                return (
                    obj.get("action") in {"repair", "fallback"}
                    and isinstance(obj.get("markdown_table", ""), str)
                    and isinstance(float(obj.get("confidence", 0.0)), float)
                )
            if task_name == "structure_headings":
                items = obj.get("items")
                if not isinstance(items, list):
                    return False
                for it in items:
                    if not isinstance(it, dict):
                        return False
                    if not isinstance(it.get("index"), int):
                        return False
                    if not isinstance(it.get("level"), int):
                        return False
                    if not isinstance(it.get("label", ""), str):
                        return False
                return True
        except Exception:
            return False
        return isinstance(obj, dict)

    def call_llm(self, task_name: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled or self.client is None:
            return None
        template = LLM_TASK_PROMPTS.get(task_name)
        if not template:
            return None
        prompt = (
            "You are a strict JSON API. Return one JSON object only.\n\n"
            f"Task: {task_name}\n"
            f"Instruction:\n{template}\n\n"
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Output strict JSON only. No markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            txt = (resp.choices[0].message.content or "").strip()
            s = txt.find("{")
            e = txt.rfind("}")
            if s < 0 or e < s:
                return None
            obj = json.loads(txt[s : e + 1])
            if not isinstance(obj, dict):
                return None
            if not self._validate_task_output(task_name, obj):
                return None
            return obj
        except Exception:
            return None


def call_llm(task_name: str, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    if _LLM_RUNTIME is None:
        return None
    return _LLM_RUNTIME.call_llm(task_name, payload)


class PDFToMarkdownV2:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.formula_stats = FormulaStats()
        self.table_stats = TableStats()
        self.reference_stats = ReferenceStats()
        self.warnings: list[str] = []
        self.header_templates_top: dict[str, int] = {}
        self.header_templates_bottom: dict[str, int] = {}
        self.removed_header_footer_examples: list[str] = []
        self.figure_count: int = 0
        self.layout_per_page: list[dict[str, Any]] = []
        self._paper_elapsed_s: float = 0.0
        self._img_seq = 0
        self._table_seq = 0
        self._formula_seq = 0

    def convert(self, pdf_path: Path) -> Path:
        if fitz is None:
            raise RuntimeError("Missing dependency: PyMuPDF (fitz). Install with `pip install pymupdf`.")
        t0 = time.perf_counter()
        paper_name = safe_name(pdf_path.stem)
        out_dir = self.cfg.output_dir / paper_name
        images_dir = out_dir / "images"
        tables_dir = out_dir / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(str(pdf_path)) as doc:
            page_count = len(doc)
            raw_pages = [self._extract_raw_blocks(doc[i], i) for i in range(page_count)]
            body_font = estimate_body_font_size(raw_pages)
            self._build_header_footer_templates(doc, raw_pages)
            per_page_cols: list[dict[str, Any]] = []
            for pi in range(page_count):
                page_w = float(doc[pi].rect.width)
                info = self._detect_page_columns(raw_pages[pi], page_w)
                per_page_cols.append(info)
                self.layout_per_page.append(
                    {
                        "page": pi + 1,
                        "layout": "double" if info.get("is_double_column") else "single",
                        "split_x": info.get("split_x"),
                        "gutter": info.get("gutter"),
                    }
                )

            double_pages = [x for x in per_page_cols if x.get("is_double_column")]
            split_candidates = [float(x["split_x"]) for x in double_pages if x.get("split_x") is not None]
            col_info = {
                "is_double_column": len(double_pages) >= max(2, int(math.ceil(page_count * 0.35))),
                "split_x": statistics.median(split_candidates) if split_candidates else None,
                "gutter": statistics.median([float(x["gutter"]) for x in double_pages if x.get("gutter")]) if double_pages else None,
            }

            page_markdown: list[str] = []
            reference_lines: list[str] = []
            references_started = False

            for pi in range(page_count):
                page = doc[pi]
                page_col = per_page_cols[pi]
                elements = self._parse_page(
                    page=page,
                    page_index=pi,
                    raw_blocks=raw_pages[pi],
                    body_font_size=body_font,
                    is_double_column=bool(page_col.get("is_double_column")),
                    global_split_x=page_col.get("split_x"),
                    images_dir=images_dir,
                    tables_dir=tables_dir,
                    pdf_path=pdf_path,
                )

                sorted_elements = self._sort_elements_page(
                    elements=elements,
                    page_width=float(page.rect.width),
                    is_double_column=bool(page_col.get("is_double_column")),
                    split_x=page_col.get("split_x"),
                )
                sorted_elements = self._merge_split_heading_fragments(sorted_elements)
                sorted_elements = self._llm_refine_heading_levels(sorted_elements)
                if pi == 0:
                    sorted_elements = self._prevent_media_before_frontmatter(
                        sorted_elements,
                        page_height=float(page.rect.height),
                    )

                page_out: list[str] = [f"<!-- kb_page: {pi+1} -->"]
                for el in sorted_elements:
                    txt = normalize_text(el.text)
                    ref_head_m = re.search(r"\breferences(?:\s+and\s+links)?\b[:\s]*", txt, flags=re.IGNORECASE)
                    txt_l = txt.lower()
                    ref_head_ok = bool(
                        ref_head_m
                        and (
                            txt_l.startswith("references")
                            or txt_l.startswith("# references")
                            or ("references and links" in txt_l)
                            or (len(txt) <= 80 and ref_head_m.start() <= 12)
                        )
                    )
                    if not references_started and (REFERENCE_HEAD_RE.match(txt) or ref_head_ok):
                        references_started = True
                        self.reference_stats.detected = True
                        self.reference_stats.start_page = pi + 1
                        if ref_head_m is not None:
                            tail = txt[ref_head_m.end() :].strip()
                            if tail:
                                reference_lines.extend(split_sentences_keep(tail))
                        continue

                    if references_started:
                        if (
                            looks_like_content_section_heading(txt)
                            and (not REFERENCE_HEAD_RE.match(txt))
                            and (not looks_like_reference_entry(txt))
                        ):
                            references_started = False
                            if el.kind == "heading":
                                page_out.append(el.markdown)
                            else:
                                page_out.append(f"## {txt}")
                            continue
                        if el.kind in {"body", "heading", "reference", "caption"}:
                            for ln in split_sentences_keep(el.text):
                                if looks_like_reference_entry(ln) or len(reference_lines) > 0:
                                    reference_lines.append(ln)
                        continue
                    page_out.append(el.markdown)

                page_markdown.append(self._merge_paragraphs("\n\n".join(x for x in page_out if x.strip())))

            references_md = self._render_references(reference_lines)
            paper_md_parts = [x for x in page_markdown if x.strip()]
            if references_md.strip():
                paper_md_parts.append("# References\n\n" + references_md)
            paper_md = "\n\n".join(paper_md_parts).strip() + "\n"
            paper_md = fix_hyphen_breaks(paper_md)

            (out_dir / "paper.md").write_text(paper_md, encoding="utf-8")
            self._paper_elapsed_s = float(time.perf_counter() - t0)
            meta = self._build_meta(pdf_path, page_count, col_info)
            (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_compat_outputs(out_dir=out_dir, paper_name=paper_name)
            return out_dir

    def _extract_raw_blocks(self, page: "fitz.Page", page_index: int) -> list[RawBlock]:
        data = page.get_text("dict")
        out: list[RawBlock] = []
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            lines = block.get("lines", []) or []
            text_lines: list[str] = []
            max_size = 0.0
            bold = False
            for line in lines:
                spans = line.get("spans", []) or []
                parts: list[str] = []
                for sp in spans:
                    t = normalize_text(str(sp.get("text", "")))
                    if t:
                        parts.append(t)
                    try:
                        max_size = max(max_size, float(sp.get("size", 0.0)))
                    except Exception:
                        pass
                    try:
                        if int(sp.get("flags", 0)) & (2**4):
                            bold = True
                    except Exception:
                        pass
                if parts:
                    text_lines.append(" ".join(parts))
            text = "\n".join(text_lines).strip()
            if not text:
                continue
            bbox = tuple(float(x) for x in block.get("bbox", (0, 0, 0, 0)))
            out.append(
                RawBlock(
                    bbox=bbox,
                    text=text,
                    font_size=max_size if max_size > 0 else 10.0,
                    bold=bold,
                    page_index=page_index,
                )
            )
        return out

    def _build_header_footer_templates(self, doc: "fitz.Document", raw_pages: list[list[RawBlock]]) -> None:
        page_count = len(doc)
        top_counter: dict[str, int] = {}
        bot_counter: dict[str, int] = {}
        for pi in range(page_count):
            ph = float(doc[pi].rect.height)
            top_cut = ph * 0.12
            bot_cut = ph * 0.88
            for b in raw_pages[pi]:
                rect = b.rect
                t = normalize_text(b.text)
                if not t:
                    continue
                key = clean_template_text(t)
                if not key:
                    continue
                if rect.y1 <= top_cut:
                    top_counter[key] = top_counter.get(key, 0) + 1
                elif rect.y0 >= bot_cut:
                    bot_counter[key] = bot_counter.get(key, 0) + 1
        min_rep = max(2, int(math.ceil(page_count * 0.3)))
        self.header_templates_top = {k: v for k, v in top_counter.items() if v >= min_rep}
        self.header_templates_bottom = {k: v for k, v in bot_counter.items() if v >= min_rep}

    def _detect_document_columns(self, raw_pages: list[list[RawBlock]], doc: "fitz.Document") -> dict[str, Any]:
        centers: list[float] = []
        page_widths: list[float] = []
        for pi, blocks in enumerate(raw_pages):
            pw = float(doc[pi].rect.width)
            page_widths.append(pw)
            for b in blocks:
                if b.width >= pw * 0.70:
                    continue
                cx = (b.bbox[0] + b.bbox[2]) / 2.0
                centers.append(cx)
        if not centers:
            return {"is_double_column": False, "split_x": None, "gutter": None}
        pw_mid = statistics.median(page_widths) if page_widths else 1000.0
        arr = sorted(centers)
        best_gap = 0.0
        split_x: Optional[float] = None
        lo = pw_mid * 0.25
        hi = pw_mid * 0.75
        for i in range(len(arr) - 1):
            a = arr[i]
            b = arr[i + 1]
            mid = (a + b) / 2.0
            if mid < lo or mid > hi:
                continue
            gap = b - a
            if gap > best_gap:
                best_gap = gap
                split_x = mid
        if split_x is None:
            return {"is_double_column": False, "split_x": None, "gutter": None}
        left_n = sum(1 for c in centers if c < split_x)
        right_n = len(centers) - left_n
        is_double = bool(
            best_gap >= pw_mid * 0.08
            and left_n >= max(8, int(len(centers) * 0.2))
            and right_n >= max(8, int(len(centers) * 0.2))
        )
        gutter = best_gap if is_double else None
        return {"is_double_column": is_double, "split_x": split_x if is_double else None, "gutter": gutter}

    def _detect_page_columns(self, blocks: list[RawBlock], page_width: float) -> dict[str, Any]:
        centers = []
        for b in blocks:
            if b.width >= page_width * 0.70:
                continue
            centers.append((b.bbox[0] + b.bbox[2]) / 2.0)
        if len(centers) < 8:
            return {"is_double_column": False, "split_x": None, "gutter": None}
        arr = sorted(centers)
        best_gap = 0.0
        split_x: Optional[float] = None
        lo = page_width * 0.25
        hi = page_width * 0.75
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i + 1]
            mid = (a + b) / 2.0
            if mid < lo or mid > hi:
                continue
            gap = b - a
            if gap > best_gap:
                best_gap = gap
                split_x = mid
        if split_x is None:
            return {"is_double_column": False, "split_x": None, "gutter": None}
        left_n = sum(1 for c in centers if c < split_x)
        right_n = len(centers) - left_n
        is_double = bool(best_gap >= page_width * 0.08 and left_n >= 4 and right_n >= 4)
        return {"is_double_column": is_double, "split_x": split_x if is_double else None, "gutter": best_gap if is_double else None}

    def _collect_visual_rects(self, page: "fitz.Page") -> list["fitz.Rect"]:
        rects: list[fitz.Rect] = []
        page_rect = page.rect
        page_area = max(1.0, rect_area(page_rect))
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") == 1 and block.get("bbox"):
                rr = fitz.Rect(block["bbox"])
                if not self._is_visual_strip_noise(rr, page_rect):
                    rects.append(rr)
        try:
            infos = page.get_image_info(xrefs=True) or []
            for info in infos:
                bbox = info.get("bbox")
                if bbox:
                    rr = fitz.Rect(bbox)
                    if not self._is_visual_strip_noise(rr, page_rect):
                        rects.append(rr)
        except Exception:
            pass
        try:
            drawings = page.get_drawings()
            for d in drawings:
                r = d.get("rect")
                if not r:
                    continue
                rr = fitz.Rect(r)
                if self._is_visual_strip_noise(rr, page_rect):
                    continue
                if rect_area(rr) >= page_area * 0.0015:
                    rects.append(rr)
        except Exception:
            pass
        return self._coalesce_visual_rects(self._dedupe_rects(rects), page_rect)

    def _dedupe_rects(self, rects: list["fitz.Rect"]) -> list["fitz.Rect"]:
        out: list[fitz.Rect] = []
        for r in sorted(rects, key=lambda x: (x.y0, x.x0, x.y1, x.x1)):
            keep = True
            for e in out:
                if overlap_ratio(r, e) >= 0.85 and overlap_ratio(e, r) >= 0.85:
                    keep = False
                    break
            if keep:
                out.append(r)
        return out

    def _is_visual_strip_noise(self, rect: "fitz.Rect", page_rect: "fitz.Rect") -> bool:
        pw = float(page_rect.width)
        ph = float(page_rect.height)
        very_wide = rect.width >= pw * 0.88
        short_band = rect.height <= ph * 0.08
        near_top_bottom = rect.y0 <= ph * 0.09 or rect.y1 >= ph * 0.91
        return bool(very_wide and short_band and near_top_bottom)

    def _coalesce_visual_rects(self, rects: list["fitz.Rect"], page_rect: "fitz.Rect") -> list["fitz.Rect"]:
        if not rects:
            return []
        merged = [fitz.Rect(r) for r in sorted(rects, key=lambda x: (x.y0, x.x0))]
        changed = True
        while changed:
            changed = False
            out: list[fitz.Rect] = []
            while merged:
                cur = merged.pop(0)
                i = 0
                while i < len(merged):
                    other = merged[i]
                    inter = fitz.Rect(cur) & fitz.Rect(other)
                    x_overlap = max(0.0, min(cur.x1, other.x1) - max(cur.x0, other.x0))
                    y_overlap = max(0.0, min(cur.y1, other.y1) - max(cur.y0, other.y0))
                    x_gap = max(0.0, max(cur.x0, other.x0) - min(cur.x1, other.x1))
                    y_gap = max(0.0, max(cur.y0, other.y0) - min(cur.y1, other.y1))
                    should_merge = False
                    if rect_area(inter) > 0:
                        should_merge = True
                    elif x_overlap >= 12 and y_gap <= 14:
                        should_merge = True
                    elif y_overlap >= 12 and x_gap <= 14:
                        should_merge = True
                    if should_merge:
                        cur = fitz.Rect(min(cur.x0, other.x0), min(cur.y0, other.y0), max(cur.x1, other.x1), max(cur.y1, other.y1))
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1
                if not self._is_visual_strip_noise(cur, page_rect):
                    out.append(cur)
            merged = sorted(out, key=lambda x: (x.y0, x.x0))
        return merged

    def _is_header_footer_noise(self, block: RawBlock, page_height: float) -> bool:
        text = normalize_text(block.text)
        if not text:
            return True
        top_band = page_height * 0.15
        bottom_band = page_height * 0.85
        key = clean_template_text(text)
        in_top = block.rect.y1 <= top_band
        in_bottom = block.rect.y0 >= bottom_band
        for pat in HF_STRONG_PATTERNS:
            if pat.search(text) and (in_top or in_bottom or len(text) <= 240):
                if len(self.removed_header_footer_examples) < 12:
                    self.removed_header_footer_examples.append(text[:220])
                return True
        if in_top and key in self.header_templates_top:
            if len(self.removed_header_footer_examples) < 12:
                self.removed_header_footer_examples.append(text[:220])
            return True
        if in_bottom and key in self.header_templates_bottom:
            if len(self.removed_header_footer_examples) < 12:
                self.removed_header_footer_examples.append(text[:220])
            return True
        if (in_top or in_bottom) and PAGE_NUMBER_RE.match(text):
            return True
        return False

    def _is_text_inside_visual(self, block: RawBlock, visuals: list["fitz.Rect"]) -> bool:
        if not visuals:
            return False
        br = block.rect
        barea = max(1.0, rect_area(br))
        ov = max((intersection_area(br, vr) for vr in visuals), default=0.0)
        ratio = ov / barea
        if ratio >= 0.60:
            return True
        cx = (br.x0 + br.x1) / 2.0
        cy = (br.y0 + br.y1) / 2.0
        center_inside = any((vr.x0 <= cx <= vr.x1 and vr.y0 <= cy <= vr.y1) for vr in visuals)
        if center_inside and len(normalize_text(block.text)) < 40:
            return True
        return False

    def _detect_heading_level(self, block: RawBlock, body_font_size: float) -> Optional[int]:
        text = normalize_text(block.text)
        if not text:
            return None
        if REFERENCE_HEAD_RE.match(text):
            return 1
        if re.match(r"^\s*abstract\b", text, flags=re.IGNORECASE):
            return 2
        if TABLE_CAP_RE.match(text) or FIGURE_CAP_RE.match(text):
            return None
        if likely_math_text(text):
            return None
        delta = float(block.font_size) - float(body_font_size)
        if re.match(r"^\d+\.$", text):
            return 2
        m_num = re.match(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+)$", text)
        if m_num:
            if not heading_text_quality(text):
                return None
            suffix = normalize_text(m_num.group(2))
            if heading_text_quality(suffix):
                depth = len(m_num.group(1).split("."))
                return max(2, min(4, depth))
            return None
        if text.lower().startswith("references and links"):
            return 1
        if not heading_text_quality(text):
            return None
        if text.isupper() and len(text.split()) <= 8 and (delta >= 0.4 or block.bold):
            return 2
        if delta >= 1.4 and (block.bold or len(text) <= 90):
            return 2
        if delta >= 0.8 and block.bold and len(text) <= 110:
            return 3
        return None

    def _convert_formula_candidate(self, text: str) -> str:
        t = normalize_text(text)
        t = re.sub(r"\(\s*\d+\s*\)$", "", t).strip()
        for k, v in GREEK_TO_LATEX.items():
            t = t.replace(k, v)
        for k, v in MATH_SYMBOL_TO_LATEX.items():
            t = t.replace(k, " " + v + " ")
        t = re.sub(r"\s+", " ", t).strip()
        t = t.replace("\u2212", "-")
        return t

    def _formula_confidence(self, raw: str, latex: str) -> tuple[float, str]:
        if not latex:
            return 0.0, "empty"
        score = 1.0
        reason = "ok"
        if not is_balanced_brackets(latex):
            score -= 0.45
            reason = "unbalanced_brackets"
        bad = re.findall(r"[\uFFFD\u25A1]", latex)
        if bad:
            score -= 0.35
            reason = "garbled"
        ctrl_n = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", latex))
        if ctrl_n > 0:
            score -= 0.60
            reason = "control_chars"
        if re.search(r"[鈭鈮锟�]", latex):
            score -= 0.35
            reason = "mojibake"
        illegal = re.findall(r"[^A-Za-z0-9\s\\{}^_+=\-*/().,:;<>|\[\]]", latex)
        illegal_ratio = len(illegal) / max(1, len(latex))
        if illegal_ratio > 0.05:
            score -= 0.45
            reason = "illegal_char_ratio"
        if raw.count(" ") <= 1 and len(raw) <= 3:
            score -= 0.25
            reason = "too_short"
        return max(0.0, min(1.0, score)), reason

    def _is_display_math_candidate(self, block: RawBlock, page: "fitz.Page") -> bool:
        text = normalize_text(block.text)
        if not text:
            return False
        if re.match(r"^\s*abstract\b", text, flags=re.IGNORECASE):
            return False
        words = re.findall(r"[A-Za-z]{2,}", text)
        symbol_n = len(re.findall(r"[=<>^_\\+\-/*∑∫∞≤≥≠≈]", text))
        greek_n = len(re.findall(r"[α-ωΑ-Ω]", text))
        has_eq = ("=" in text) or bool(re.search(r"\b(?:arg\s*min|arg\s*max|min|max)\b", text, flags=re.IGNORECASE))
        if len(words) >= 16 and symbol_n <= 2 and greek_n == 0:
            return False
        if symbol_n < 2 and greek_n < 2 and (not has_eq):
            return False
        pw = float(page.rect.width)
        centered = block.rect.x0 >= pw * 0.12 and block.rect.x1 <= pw * 0.88
        if len(words) <= 10 and (symbol_n + greek_n) >= 2:
            return True
        if centered and has_eq and (symbol_n + greek_n) >= 2:
            return True
        return False

    def _render_formula(
        self,
        page: "fitz.Page",
        block: RawBlock,
        images_dir: Path,
        use_llm: bool,
    ) -> Element:
        self.formula_stats.total_formula += 1
        raw = normalize_text(block.text)
        latex = self._convert_formula_candidate(raw)
        conf, reason = self._formula_confidence(raw, latex)

        if conf < 0.78 and use_llm:
            self.formula_stats.llm_attempts += 1
            res = call_llm(
                "validate_formula_latex",
                {
                    "raw_text": raw,
                    "latex_candidate": latex,
                    "page": block.page_index + 1,
                },
            )
            if isinstance(res, dict):
                try:
                    llm_latex = normalize_text(str(res.get("latex", "")))
                    llm_conf = float(res.get("confidence", 0))
                    llm_valid = bool(res.get("is_valid", False))
                    if llm_valid and llm_latex:
                        latex = llm_latex
                        conf = max(conf, llm_conf)
                        reason = "llm_repaired"
                except Exception:
                    pass

        if conf >= 0.78:
            self.formula_stats.latex_success += 1
            inline = len(latex) <= 50 and ("\n" not in raw) and ("=" not in raw)
            if inline:
                md = f"${latex}$"
            else:
                md = "$$\n" + latex + "\n$$"
            return Element(
                kind="formula",
                text=raw,
                page_index=block.page_index,
                bbox=block.bbox,
                markdown=md,
                priority=40,
            )

        self._formula_seq += 1
        img_name = self._save_crop(
            page=page,
            rect=block.rect,
            out_dir=images_dir,
            prefix="formula",
            page_index=block.page_index,
            seq=self._formula_seq,
            extra=raw[:80],
        )

        # Optional external formula OCR (Mathpix). Disabled by default.
        mathpix_latex = maybe_mathpix_recognize(images_dir / img_name)
        if mathpix_latex:
            m_conf, _ = self._formula_confidence(raw, mathpix_latex)
            if m_conf >= 0.72:
                self.formula_stats.latex_success += 1
                return Element(
                    kind="formula",
                    text=raw,
                    page_index=block.page_index,
                    bbox=block.bbox,
                    markdown="$$\n" + mathpix_latex + "\n$$",
                    priority=40,
                )

        self.formula_stats.fallback_images += 1
        bbox_txt = [round(x, 2) for x in block.bbox]
        md = (
            f"<!-- FORMULA_FALLBACK: page={block.page_index+1} bbox={bbox_txt} reason={reason} -->\n"
            f"![](images/{img_name})"
        )
        return Element(
            kind="formula_fallback",
            text=raw,
            page_index=block.page_index,
            bbox=block.bbox,
            markdown=md,
            priority=40,
        )

    def _save_crop(
        self,
        page: "fitz.Page",
        rect: "fitz.Rect",
        out_dir: Path,
        prefix: str,
        page_index: int,
        seq: int,
        extra: str = "",
    ) -> str:
        pad = 4.0
        clip = fitz.Rect(
            max(0.0, rect.x0 - pad),
            max(0.0, rect.y0 - pad),
            min(float(page.rect.width), rect.x1 + pad),
            min(float(page.rect.height), rect.y1 + pad),
        )
        min_w = min(float(page.rect.width) * 0.20, 120.0)
        min_h = min(float(page.rect.height) * 0.10, 90.0)
        if clip.width < min_w:
            cx = (clip.x0 + clip.x1) / 2.0
            clip.x0 = max(0.0, cx - min_w / 2.0)
            clip.x1 = min(float(page.rect.width), cx + min_w / 2.0)
        if clip.height < min_h:
            cy = (clip.y0 + clip.y1) / 2.0
            clip.y0 = max(0.0, cy - min_h / 2.0)
            clip.y1 = min(float(page.rect.height), cy + min_h / 2.0)
        digest = sha8(f"{page_index+1}|{seq}|{clip.x0:.2f}|{clip.y0:.2f}|{clip.x1:.2f}|{clip.y1:.2f}|{extra}")
        name = f"{prefix}_p{page_index+1:03d}_{seq:03d}_{digest}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False)
        pix.save(str(out_dir / name))
        return name

    def _extract_tables(
        self,
        page: "fitz.Page",
        page_index: int,
        tables_dir: Path,
        images_dir: Path,
        pdf_path: Path,
        raw_blocks: list[RawBlock],
    ) -> list[Element]:
        _ = images_dir
        elements: list[Element] = []
        page_tables: list[tuple[fitz.Rect, str, float]] = []
        page_area = max(1.0, rect_area(page.rect))
        caption_rects = [b.rect for b in raw_blocks if TABLE_CAP_RE.match(normalize_text(b.text))]

        def near_table_caption(rect: "fitz.Rect") -> bool:
            for cr in caption_rects:
                if cr.y0 < rect.y0 - 120 or cr.y0 > rect.y1 + 120:
                    continue
                x_overlap = max(0.0, min(rect.x1, cr.x1) - max(rect.x0, cr.x0))
                if x_overlap >= 15:
                    return True
            return False

        def table_rows_stats(rows: list[list[Any]]) -> tuple[int, int, float, float]:
            row_n = len(rows)
            col_n = max((len(r) for r in rows), default=0)
            if row_n == 0 or col_n == 0:
                return row_n, col_n, 1.0, 0.0
            blank_rows = 0
            non_empty = 0
            for r in rows:
                vals = [normalize_text("" if c is None else str(c)) for c in r]
                nz = sum(1 for v in vals if v)
                if nz == 0:
                    blank_rows += 1
                non_empty += nz
            blank_ratio = blank_rows / max(1, row_n)
            avg_non_empty = non_empty / max(1, row_n)
            return row_n, col_n, blank_ratio, avg_non_empty

        def accept_table_candidate(rect: "fitz.Rect", rows: list[list[Any]], md: str) -> bool:
            row_n, col_n, blank_ratio, avg_non_empty = table_rows_stats(rows)
            if row_n < 2 or col_n < 2:
                return False
            if blank_ratio > 0.42:
                return False
            if avg_non_empty < 1.6:
                return False
            flat_cells = [normalize_text("" if c is None else str(c)) for r in rows for c in r]
            nz_cells = [c for c in flat_cells if c]
            if not nz_cells:
                return False
            single_char_ratio = sum(1 for c in nz_cells if len(c) == 1) / max(1, len(nz_cells))
            long_sentence_like = sum(1 for c in nz_cells if len(c.split()) >= 8)
            numeric_ratio = sum(1 for c in nz_cells if re.search(r"\d", c)) / max(1, len(nz_cells))
            if single_char_ratio > 0.34 and (not near_table_caption(rect)):
                return False
            if long_sentence_like >= 3 and (not near_table_caption(rect)):
                return False
            if numeric_ratio < 0.08 and (not near_table_caption(rect)):
                return False
            if (not near_table_caption(rect)) and row_n > 5:
                return False
            if (not near_table_caption(rect)) and numeric_ratio < 0.25:
                return False
            area_ratio = rect_area(rect) / page_area
            if area_ratio > 0.25 and (not near_table_caption(rect)):
                return False
            if row_n > 60 and (not near_table_caption(rect)):
                return False
            q = markdown_table_quality(md)
            if q < 0.35:
                return False
            return True

        if hasattr(page, "find_tables"):
            strategies = [
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                {"vertical_strategy": "text", "horizontal_strategy": "text"},
            ]
            for kwargs in strategies:
                try:
                    finder = page.find_tables(**kwargs)
                    tables = getattr(finder, "tables", finder)
                except Exception:
                    continue
                for tb in tables or []:
                    try:
                        bbox = fitz.Rect(tb.bbox)
                        rows = tb.extract() or []
                        md = rows_to_markdown(rows)
                    except Exception:
                        continue
                    if not md:
                        continue
                    q = markdown_table_quality(md)
                    if q >= 0.35 and accept_table_candidate(bbox, rows, md):
                        page_tables.append((bbox, md, q))

        if (not page_tables) and (pdfplumber is not None):
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    pg = pdf.pages[page_index]
                    found = pg.find_tables()
                    for tb in found or []:
                        rows = tb.extract()
                        md = rows_to_markdown(rows or [])
                        if not md:
                            continue
                        q = markdown_table_quality(md)
                        x0, top, x1, bottom = tb.bbox
                        bbox = fitz.Rect(float(x0), float(top), float(x1), float(bottom))
                        if q >= 0.35 and accept_table_candidate(bbox, rows or [], md):
                            page_tables.append((bbox, md, q))
            except Exception:
                pass

        page_tables = sorted(page_tables, key=lambda x: x[2], reverse=True)
        keep: list[tuple[fitz.Rect, str, float]] = []
        for rect, md, q in page_tables:
            if any(overlap_ratio(rect, kr) >= 0.70 for kr, _, _ in keep):
                continue
            keep.append((rect, md, q))

        for rect, md, q in keep:
            _ = q
            self.table_stats.total_tables += 1
            self.table_stats.structured_tables += 1
            self._table_seq += 1
            digest = sha8(md)
            table_file = f"table_p{page_index+1:03d}_{self._table_seq:03d}_{digest}.md"
            (tables_dir / table_file).write_text(md + "\n", encoding="utf-8")
            elements.append(
                Element(
                    kind="table",
                    text=md,
                    page_index=page_index,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    markdown=md,
                    priority=30,
                )
            )
        return elements

    def _table_fallback_from_caption(
        self,
        page: "fitz.Page",
        caption_block: RawBlock,
        visuals: list["fitz.Rect"],
        images_dir: Path,
    ) -> Optional[Element]:
        if not TABLE_CAP_RE.match(normalize_text(caption_block.text)):
            return None
        candidates = [vr for vr in visuals if vr.y1 <= caption_block.rect.y0 + 15.0]
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda r: caption_block.rect.y0 - r.y1)
        target = candidates[0]
        self.table_stats.total_tables += 1
        self.table_stats.fallback_images += 1
        self._table_seq += 1
        img_name = self._save_crop(
            page=page,
            rect=target,
            out_dir=images_dir,
            prefix="table_fallback",
            page_index=caption_block.page_index,
            seq=self._table_seq,
            extra=caption_block.text,
        )
        ocr_hint = ""
        if self.cfg.use_ocr and pytesseract is not None and Image is not None:
            txt = self._ocr_rect(page, target)
            if txt:
                self.table_stats.ocr_hint_count += 1
                ocr_hint = "\n<!-- TABLE_OCR_HINT: " + normalize_text(txt)[:800] + " -->"
        md = (
            f"<!-- TABLE_FALLBACK: page={caption_block.page_index+1} reason=structure_low_confidence -->\n"
            f"![Table fallback](images/{img_name})\n"
            f"{caption_block.text}{ocr_hint}"
        )
        return Element(
            kind="table_fallback",
            text=caption_block.text,
            page_index=caption_block.page_index,
            bbox=(target.x0, target.y0, target.x1, target.y1),
            markdown=md,
            priority=32,
        )

    def _ocr_rect(self, page: "fitz.Page", rect: "fitz.Rect") -> str:
        if pytesseract is None or Image is None:
            return ""
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), clip=rect, alpha=False)
            bio = io.BytesIO(pix.tobytes("png"))
            with Image.open(bio) as im:
                return pytesseract.image_to_string(im, lang="eng")
        except Exception:
            return ""

    def _reconstruct_table_from_text(
        self,
        raw_blocks: list[RawBlock],
        start_idx: int,
    ) -> tuple[Optional[str], set[int]]:
        rows: list[list[str]] = []
        consumed: set[int] = set()

        def is_numeric(tok: str) -> bool:
            return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:%|e[+-]?\d+)?", tok, flags=re.IGNORECASE))

        for j in range(start_idx + 1, min(len(raw_blocks), start_idx + 14)):
            t = normalize_text(raw_blocks[j].text)
            if not t:
                continue
            if TABLE_CAP_RE.match(t) or FIGURE_CAP_RE.match(t) or REFERENCE_HEAD_RE.match(t):
                break
            if t.startswith("#"):
                break
            words = t.split()
            if len(words) > 18 and (not any(is_numeric(w) for w in words)):
                break
            if len(words) < 2:
                continue

            nums = [k for k, w in enumerate(words) if is_numeric(w)]
            if nums:
                k = nums[0]
                if k <= 0:
                    continue
                label = " ".join(words[:k]).strip()
                vals = words[k:]
                cells = [label] + vals
            else:
                # possible header row
                if len(rows) == 0 and 2 <= len(words) <= 8:
                    cells = words
                else:
                    break
            rows.append(cells)
            consumed.add(j)

        if len(rows) < 2:
            return None, set()

        max_cols = max(len(r) for r in rows)
        min_cols = min(len(r) for r in rows)
        if max_cols < 2 or (max_cols - min_cols) > 2:
            return None, set()

        if not any(any(re.search(r"\d", c) for c in r[1:]) for r in rows[1:]):
            return None, set()

        aligned: list[list[str]] = []
        for r in rows:
            rr = r + [""] * (max_cols - len(r))
            aligned.append(rr[:max_cols])
        md = rows_to_markdown(aligned)
        if not md:
            return None, set()
        if self.cfg.use_llm:
            res = call_llm(
                "table_repair_or_fallback",
                {
                    "caption_index": start_idx,
                    "rows": aligned,
                    "markdown_candidate": md,
                },
            )
            if isinstance(res, dict):
                action = str(res.get("action", "")).strip().lower()
                repaired = normalize_text(str(res.get("markdown_table", "")))
                if action == "fallback":
                    return None, set()
                if action == "repair" and repaired and repaired.count("|") >= md.count("|") * 0.5:
                    md = repaired.replace("\\n", "\n")
        return md, consumed

    def _extract_figures(
        self,
        page: "fitz.Page",
        page_index: int,
        visuals: list["fitz.Rect"],
        text_blocks: list[RawBlock],
        images_dir: Path,
        table_rects: list["fitz.Rect"],
    ) -> tuple[list[Element], set[int]]:
        elements: list[Element] = []
        consumed_caption_idx: set[int] = set()
        page_rect = page.rect
        page_area = max(1.0, rect_area(page_rect))
        page_w = float(page_rect.width)
        page_h = float(page_rect.height)
        table_caption_rects = [b.rect for b in text_blocks if TABLE_CAP_RE.match(normalize_text(b.text))]
        figure_captions = [(i, b, normalize_text(b.text)) for i, b in enumerate(text_blocks) if FIGURE_CAP_RE.match(normalize_text(b.text))]
        figure_caption_count = len(figure_captions)
        valid_visuals: list[fitz.Rect] = []
        for vr in visuals:
            if self._is_visual_strip_noise(vr, page_rect):
                continue
            if rect_area(vr) < page_area * 0.003:
                continue
            if vr.y1 < 30 or vr.y0 > page_h - 30:
                continue
            if any(overlap_ratio(vr, tr) >= 0.5 for tr in table_rects):
                continue
            valid_visuals.append(vr)

        used_visual_ids: set[int] = set()

        # Caption-first matching avoids binding captions to stray top/bottom strips.
        for cap_idx, cap_block, cap_text in sorted(figure_captions, key=lambda x: (x[1].rect.y0, x[1].rect.x0)):
            best_id: Optional[int] = None
            best_score: float = 1e18
            cap_rect = cap_block.rect
            for vid, vr in enumerate(valid_visuals):
                if vid in used_visual_ids:
                    continue
                if vr.y1 > cap_rect.y0 + 18:
                    continue
                dy = cap_rect.y0 - vr.y1
                if dy < -12 or dy > page_h * 0.50:
                    continue
                x_overlap = max(0.0, min(vr.x1, cap_rect.x1) - max(vr.x0, cap_rect.x0))
                if x_overlap < 10:
                    continue
                area_ratio = rect_area(vr) / page_area
                if area_ratio < 0.004:
                    continue
                center_dx = abs(((vr.x0 + vr.x1) / 2.0) - ((cap_rect.x0 + cap_rect.x1) / 2.0))
                thin_penalty = 0.0
                if vr.width >= page_w * 0.85 and vr.height <= page_h * 0.10:
                    thin_penalty += 600.0
                if vr.height <= page_h * 0.055:
                    thin_penalty += 260.0
                score = (dy * 2.0) + (center_dx * 0.12) + thin_penalty - (area_ratio * 220.0)
                if score < best_score:
                    best_score = score
                    best_id = vid
            if best_id is None:
                continue

            vr = valid_visuals[best_id]
            used_visual_ids.add(best_id)
            consumed_caption_idx.add(cap_idx)
            self._img_seq += 1
            img_name = self._save_crop(
                page=page,
                rect=vr,
                out_dir=images_dir,
                prefix="figure",
                page_index=page_index,
                seq=self._img_seq,
                extra=cap_text,
            )
            md = f"![Figure](images/{img_name})\n{cap_text}"
            elements.append(
                Element(
                    kind="figure",
                    text=cap_text,
                    page_index=page_index,
                    bbox=(vr.x0, vr.y0, vr.x1, vr.y1),
                    markdown=md,
                    priority=25,
                )
            )
            self.figure_count += 1

        # Uncaptioned figure fallback only when no figure captions are present on the page.
        if figure_caption_count == 0:
            for vid, vr in enumerate(sorted(valid_visuals, key=lambda r: (r.y0, r.x0))):
                if vid in used_visual_ids:
                    continue
                area_ratio = rect_area(vr) / page_area
                if area_ratio < 0.08:
                    continue
                if page_index == 0 and vr.y0 < page_h * 0.62:
                    continue
                if vr.y0 < page_h * 0.18 or vr.y1 > page_h * 0.97:
                    continue
                near_table_caption = False
                for tr in table_caption_rects:
                    if tr.y0 < vr.y1 - 20 or tr.y0 > vr.y1 + 120:
                        continue
                    x_overlap = max(0.0, min(vr.x1, tr.x1) - max(vr.x0, tr.x0))
                    if x_overlap >= 12:
                        near_table_caption = True
                        break
                if near_table_caption:
                    continue
                self._img_seq += 1
                img_name = self._save_crop(
                    page=page,
                    rect=vr,
                    out_dir=images_dir,
                    prefix="figure",
                    page_index=page_index,
                    seq=self._img_seq,
                    extra="",
                )
                elements.append(
                    Element(
                        kind="figure",
                        text="Figure",
                        page_index=page_index,
                        bbox=(vr.x0, vr.y0, vr.x1, vr.y1),
                        markdown=f"![Figure](images/{img_name})",
                        priority=25,
                    )
                )
                self.figure_count += 1
        return elements, consumed_caption_idx

    def _parse_page(
        self,
        page: "fitz.Page",
        page_index: int,
        raw_blocks: list[RawBlock],
        body_font_size: float,
        is_double_column: bool,
        global_split_x: Optional[float],
        images_dir: Path,
        tables_dir: Path,
        pdf_path: Path,
    ) -> list[Element]:
        _ = is_double_column
        _ = global_split_x
        visuals = self._collect_visual_rects(page)
        table_elements = self._extract_tables(page, page_index, tables_dir, images_dir, pdf_path, raw_blocks)
        table_rects = [te.rect for te in table_elements]
        figure_elements, consumed_caption_idx = self._extract_figures(
            page=page,
            page_index=page_index,
            visuals=visuals,
            text_blocks=raw_blocks,
            images_dir=images_dir,
            table_rects=table_rects,
        )

        out: list[Element] = []
        out.extend(table_elements)
        out.extend(figure_elements)
        figure_rects = [fe.rect for fe in figure_elements]
        consumed_table_text_idx: set[int] = set()
        front_title_done = False
        front_authors_done = False
        front_affil_done = False

        page_height = float(page.rect.height)
        for i, b in enumerate(raw_blocks):
            text = normalize_text(b.text)
            if not text:
                continue
            if self._is_header_footer_noise(b, page_height):
                continue
            if i in consumed_caption_idx:
                continue
            if i in consumed_table_text_idx:
                continue
            if any(overlap_ratio(b.rect, tr) >= 0.55 for tr in table_rects):
                continue
            is_ref_like_block = bool(re.search(r"\breferences(?:\s+and\s+links)?\b", text, flags=re.IGNORECASE))

            if figure_rects and (not FIGURE_CAP_RE.match(text)):
                short_label_like = len(text) <= 35 and len(text.split()) <= 6 and (not re.search(r"[.?!:;]", text))
                if short_label_like:
                    near_fig = False
                    for fr in figure_rects:
                        if intersection_area(b.rect, fr) > 0:
                            near_fig = True
                            break
                        y_gap = min(abs(b.rect.y0 - fr.y1), abs(fr.y0 - b.rect.y1))
                        x_overlap = max(0.0, min(b.rect.x1, fr.x1) - max(b.rect.x0, fr.x0))
                        if y_gap <= 12 and x_overlap >= 8:
                            near_fig = True
                            break
                    if near_fig:
                        continue

            if self._is_text_inside_visual(b, visuals):
                if not (FIGURE_CAP_RE.match(text) or TABLE_CAP_RE.match(text) or is_ref_like_block):
                    continue

            if is_garbled_math_fragment(text):
                self.warnings.append(f"Page {page_index+1}: dropped low-confidence math fragment '{text[:30]}'")
                continue

            if looks_like_garbled_formula_line(text):
                out.append(self._render_formula(page=page, block=b, images_dir=images_dir, use_llm=self.cfg.use_llm))
                continue

            if page_index == 0 and b.rect.y0 < page_height * 0.45:
                if re.match(r"^\s*abstract\s*[:：]", text, flags=re.IGNORECASE):
                    abs_body = re.sub(r"^\s*abstract\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
                    md = "## Abstract"
                    if abs_body:
                        md += "\n" + abs_body
                    out.append(
                        Element(
                            kind="heading",
                            text="Abstract",
                            page_index=page_index,
                            bbox=b.bbox,
                            markdown=md,
                            priority=9,
                            meta={"frontmatter": "abstract"},
                        )
                    )
                    continue

                words = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", text)
                if (not front_title_done) and b.font_size >= (body_font_size + 2.0) and len(words) >= 4 and len(text) <= 220:
                    front_title_done = True
                    out.append(
                        Element(
                            kind="heading",
                            text=text,
                            page_index=page_index,
                            bbox=b.bbox,
                            markdown=f"# {text}",
                            priority=5,
                            meta={"frontmatter": "title"},
                        )
                    )
                    continue
                if (not front_authors_done) and len(words) >= 2 and len(words) <= 40:
                    if re.search(r"\b[A-Z]\.\s*[A-Z][a-z]+", text) or ("," in text and ";" not in text):
                        front_authors_done = True
                        out.append(
                            Element(
                                kind="body",
                                text=text,
                                page_index=page_index,
                                bbox=b.bbox,
                                markdown=f"**Authors:** {text}",
                                priority=8,
                                meta={"frontmatter": "authors"},
                            )
                        )
                        continue
                if (not front_affil_done) and re.search(
                    r"\b(university|department|institute|laboratory|school|college|box)\b|@",
                    text,
                    flags=re.IGNORECASE,
                ):
                    front_affil_done = True
                    out.append(
                        Element(
                            kind="body",
                            text=text,
                            page_index=page_index,
                            bbox=b.bbox,
                            markdown=f"**Affiliations:** {text}",
                            priority=9,
                            meta={"frontmatter": "affiliation"},
                        )
                    )
                    continue

            if TABLE_CAP_RE.match(text):
                has_table_near = any(abs(tr.y1 - b.rect.y0) <= 35 and overlap_ratio(tr, b.rect) < 0.85 for tr in table_rects)
                if has_table_near:
                    continue
                text_table_md, consumed_ids = self._reconstruct_table_from_text(raw_blocks, i)
                if text_table_md:
                    self.table_stats.total_tables += 1
                    self.table_stats.structured_tables += 1
                    self._table_seq += 1
                    digest = sha8(text_table_md)
                    table_file = f"table_p{page_index+1:03d}_{self._table_seq:03d}_{digest}.md"
                    (tables_dir / table_file).write_text(text_table_md + "\n", encoding="utf-8")
                    consumed_table_text_idx.update(consumed_ids)
                    out.append(
                        Element(
                            kind="table",
                            text=text,
                            page_index=page_index,
                            bbox=b.bbox,
                            markdown=f"{text}\n\n{text_table_md}",
                            priority=28,
                        )
                    )
                    continue
                fallback = self._table_fallback_from_caption(page, b, visuals, images_dir)
                if fallback:
                    out.append(fallback)
                    continue
                out.append(
                    Element(
                        kind="caption",
                        text=text,
                        page_index=page_index,
                        bbox=b.bbox,
                        markdown=text,
                        priority=18,
                    )
                )
                continue

            if FIGURE_CAP_RE.match(text):
                out.append(
                    Element(
                        kind="caption",
                        text=text,
                        page_index=page_index,
                        bbox=b.bbox,
                        markdown=text,
                        priority=18,
                    )
                )
                continue

            heading_level = self._detect_heading_level(b, body_font_size)
            if heading_level is not None:
                heading_txt = normalize_heading_text(text)
                md = f"{'#' * max(1, min(6, heading_level))} {heading_txt}"
                out.append(
                    Element(
                        kind="heading",
                        text=heading_txt,
                        page_index=page_index,
                        bbox=b.bbox,
                        markdown=md,
                        priority=10,
                    )
                )
                continue

            if likely_math_text(text) and self._is_display_math_candidate(b, page):
                out.append(self._render_formula(page=page, block=b, images_dir=images_dir, use_llm=self.cfg.use_llm))
                continue

            out.append(
                Element(
                    kind="body",
                    text=text,
                    page_index=page_index,
                    bbox=b.bbox,
                    markdown=text,
                    priority=20,
                    meta={"font_size": b.font_size, "bold": b.bold},
                )
            )

        if self.cfg.use_ocr and pytesseract is not None and Image is not None:
            text_elems = [e for e in out if e.kind in {"body", "heading", "formula"}]
            if len(text_elems) <= 2:
                ocr_txt = self._ocr_rect(page, page.rect)
                ocr_txt = normalize_text(ocr_txt)
                if len(ocr_txt) >= 40:
                    out.append(
                        Element(
                            kind="body",
                            text=ocr_txt,
                            page_index=page_index,
                            bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                            markdown=ocr_txt,
                            priority=50,
                        )
                    )
                else:
                    self.warnings.append(
                        f"Page {page_index+1}: OCR enabled but extracted little text; scanned page may remain image-only."
                    )
        return out

    def _sort_elements_page(
        self,
        elements: list[Element],
        page_width: float,
        is_double_column: bool,
        split_x: Optional[float],
    ) -> list[Element]:
        if not elements:
            return []
        if not is_double_column or split_x is None:
            return sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0], e.priority))

        spanning_threshold = page_width * 0.72
        cross_margin = max(8.0, page_width * 0.02)
        full: list[Element] = []
        narrow: list[Element] = []
        for e in elements:
            w = e.bbox[2] - e.bbox[0]
            cross = e.bbox[0] <= split_x - cross_margin and e.bbox[2] >= split_x + cross_margin
            if w >= spanning_threshold or cross:
                full.append(e)
            else:
                narrow.append(e)
        full = sorted(full, key=lambda e: (e.bbox[1], e.bbox[0], e.priority))
        narrow = sorted(narrow, key=lambda e: (e.bbox[1], e.bbox[0], e.priority))

        def sort_cols(items: list[Element]) -> list[Element]:
            left = [x for x in items if (x.bbox[0] + x.bbox[2]) / 2.0 <= split_x]
            right = [x for x in items if x not in left]
            left.sort(key=lambda e: (e.bbox[1], e.bbox[0], e.priority))
            right.sort(key=lambda e: (e.bbox[1], e.bbox[0], e.priority))
            return left + right

        out: list[Element] = []
        cursor_y = -1e9
        used_ids: set[int] = set()
        for f in full:
            seg_ids = [idx for idx, n in enumerate(narrow) if idx not in used_ids and cursor_y <= n.bbox[1] < f.bbox[1]]
            seg = [narrow[idx] for idx in seg_ids]
            used_ids.update(seg_ids)
            out.extend(sort_cols(seg))
            out.append(f)
            cursor_y = max(cursor_y, f.bbox[3])
        tail = [n for idx, n in enumerate(narrow) if idx not in used_ids]
        out.extend(sort_cols(tail))

        final: list[Element] = []
        seen: set[tuple[int, int, int, int, str]] = set()
        for e in out:
            key = (
                int(e.bbox[0] * 10),
                int(e.bbox[1] * 10),
                int(e.bbox[2] * 10),
                int(e.bbox[3] * 10),
                e.markdown[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            final.append(e)
        return final

    def _merge_split_heading_fragments(self, elements: list[Element]) -> list[Element]:
        out: list[Element] = []
        i = 0
        while i < len(elements):
            cur = elements[i]
            if i + 1 < len(elements):
                nxt = elements[i + 1]
                cur_txt = normalize_text(cur.text)
                nxt_txt = normalize_text(nxt.text)
                if (
                    re.match(r"^\d+(?:\.\d+)*\.?$", cur_txt)
                    and nxt.kind in {"heading", "body"}
                    and re.match(r"^[A-Za-z][A-Za-z0-9 ,:\-]{2,120}$", nxt_txt)
                    and (nxt.bbox[1] - cur.bbox[1]) < 50
                ):
                    merged = normalize_heading_text(f"{cur_txt.rstrip('.')} {nxt_txt}".strip())
                    lvl = max(2, min(4, len(cur_txt.rstrip('.').split('.'))))
                    out.append(
                        Element(
                            kind="heading",
                            text=merged,
                            page_index=cur.page_index,
                            bbox=(
                                min(cur.bbox[0], nxt.bbox[0]),
                                min(cur.bbox[1], nxt.bbox[1]),
                                max(cur.bbox[2], nxt.bbox[2]),
                                max(cur.bbox[3], nxt.bbox[3]),
                            ),
                            markdown=f"{'#' * lvl} {merged}",
                            priority=min(cur.priority, nxt.priority),
                            meta={"merged_split_heading": True},
                        )
                    )
                    i += 2
                    continue
            out.append(cur)
            i += 1
        return out

    def _prevent_media_before_frontmatter(self, elements: list[Element], page_height: float) -> list[Element]:
        if not elements:
            return elements
        section_idx: Optional[int] = None
        abstract_idx: Optional[int] = None
        for i, el in enumerate(elements):
            txt = normalize_text(el.text)
            low = txt.lower()
            if abstract_idx is None and low.startswith("abstract"):
                abstract_idx = i
            if looks_like_content_section_heading(txt):
                section_idx = i
                break
        if section_idx is None:
            section_idx = len(elements)
        front_limit = section_idx
        if abstract_idx is not None:
            front_limit = max(front_limit, abstract_idx + 2)
        media_idx = [
            i
            for i, el in enumerate(elements)
            if i < front_limit
            and el.kind in {"figure", "table", "table_fallback"}
            and el.bbox[1] < page_height * 0.62
        ]
        if not media_idx:
            return elements
        front = [el for i, el in enumerate(elements[:front_limit]) if i not in media_idx]
        media = [elements[i] for i in media_idx]
        tail = elements[front_limit:]
        return front + media + tail

    def _llm_refine_heading_levels(self, elements: list[Element]) -> list[Element]:
        if (not self.cfg.use_llm) or (not elements):
            return elements
        candidates = []
        for i, e in enumerate(elements):
            if e.kind == "heading":
                candidates.append({"index": i, "text": normalize_text(e.text), "markdown": e.markdown})
        if not candidates:
            return elements
        res = call_llm(
            "structure_headings",
            {
                "candidates": candidates[:200],
                "rule": "Only correct heading levels/labels. Keep order and wording.",
            },
        )
        if not isinstance(res, dict) or (not isinstance(res.get("items"), list)):
            return elements
        out = list(elements)
        for item in res["items"]:
            try:
                idx = int(item["index"])
                lvl = int(item["level"])
                lbl = normalize_text(str(item.get("label", "")))
            except Exception:
                continue
            if idx < 0 or idx >= len(out):
                continue
            if out[idx].kind != "heading":
                continue
            lvl = max(1, min(4, lvl))
            if lbl:
                out[idx].text = lbl
            out[idx].markdown = f"{'#' * lvl} {out[idx].text}"
        return out

    def _merge_paragraphs(self, md: str) -> str:
        lines = md.splitlines()
        out: list[str] = []
        buf: list[str] = []

        def flush() -> None:
            nonlocal buf
            if not buf:
                return
            text = " ".join(x.strip() for x in buf if x.strip())
            text = fix_hyphen_breaks(text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                out.append(text)
            buf = []

        block_start_re = re.compile(r"^\s*(?:#|!|\||\$\$|<!--)")
        for ln in lines:
            s = ln.rstrip()
            if not s.strip():
                flush()
                if out and out[-1] != "":
                    out.append("")
                continue
            if block_start_re.match(s):
                flush()
                out.append(s)
                continue
            if TABLE_CAP_RE.match(s) or FIGURE_CAP_RE.match(s):
                flush()
                out.append(s)
                continue
            buf.append(s)
        flush()
        text = "\n".join(out)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _render_references(self, lines: list[str]) -> str:
        lines = [normalize_text(x) for x in lines if normalize_text(x)]
        lines = [x for x in lines if not REFERENCE_HEAD_RE.match(x)]
        def _is_ref_noise(x: str) -> bool:
            return bool(
                re.search(r"#\d{4,}\s*-\s*\$\d", x)
                or re.search(r"\bOPTICS\s+EXPRESS\b", x, flags=re.IGNORECASE)
                or re.search(r"\bReceived\b.*\bpublished\b", x, flags=re.IGNORECASE)
            )
        lines = [x for x in lines if not _is_ref_noise(x)]
        expanded_lines: list[str] = []
        for ln in lines:
            if len(re.findall(r"\b(?:\d+\.\s+|\d+\s+)[A-Z]", ln)) >= 2:
                parts = re.split(r"\s+(?=(?:\d+\.\s+|\d+\s+)[A-Z])", ln)
                expanded_lines.extend([p.strip() for p in parts if p.strip()])
            else:
                expanded_lines.append(ln)
        lines = expanded_lines
        if not lines:
            return ""

        def split_merged_entry(entry: str) -> list[str]:
            year_matches = list(re.finditer(r"(?:19|20)\d{2}[a-z]?", entry))
            if len(year_matches) < 2:
                return [entry]
            y2 = year_matches[1].start()
            head = entry[:y2]
            split_pos: Optional[int] = None
            candidates = list(re.finditer(r"\.\s+(?=[A-Z][A-Za-z'`\- ]{1,30},\s*(?:[A-Z]\.\s*){1,4})", head))
            if candidates:
                split_pos = candidates[-1].end()
            if split_pos is None:
                c2 = list(re.finditer(r"\s(?=[A-Z]\.\s+[A-Z][A-Za-z'`\- ]{1,25},\s*[A-Z]\.)", head))
                if c2:
                    split_pos = c2[-1].start() + 1
            if split_pos is None:
                return [entry]
            left = entry[:split_pos].strip()
            right = entry[split_pos:].strip()
            if not left or not right:
                return [entry]
            return [left, right]

        numbered_starts = sum(1 for ln in lines if NUMBERED_REF_RELAXED_RE.match(ln))
        author_year_starts = sum(1 for ln in lines if AUTHOR_YEAR_START_RE.match(ln))
        style = "numbered" if numbered_starts >= author_year_starts else "author_year"

        entries: list[str] = []
        if style == "numbered":
            cur = ""
            for ln in lines:
                m = NUMBERED_REF_RELAXED_RE.match(ln)
                if m:
                    if cur.strip():
                        entries.append(cur.strip())
                    cur = (m.group(3) or "").strip()
                else:
                    if cur:
                        cur = (cur + " " + ln).strip()
                    else:
                        cur = ln
            if cur.strip():
                entries.append(cur.strip())
        else:
            cur = ""
            for ln in lines:
                if AUTHOR_YEAR_START_RE.match(ln):
                    if cur.strip():
                        entries.append(cur.strip())
                    cur = ln
                else:
                    if cur:
                        cur = (cur + " " + ln).strip()
                    else:
                        cur = ln
            if cur.strip():
                entries.append(cur.strip())

        confidence = 0.0
        if entries:
            confidence = min(1.0, len(entries) / max(1.0, len(lines) * 0.45))
        if confidence < 0.55 and self.cfg.use_llm:
            self.reference_stats.llm_attempts += 1
            res = call_llm(
                "split_references",
                {
                    "raw_lines": lines[:500],
                    "paper_context": "academic references",
                },
            )
            if isinstance(res, dict) and isinstance(res.get("entries"), list):
                llm_entries = [normalize_text(str(x)) for x in res["entries"] if normalize_text(str(x))]
                if llm_entries:
                    entries = llm_entries
                    confidence = max(confidence, 0.65)

        if not entries:
            entries = [" ".join(lines)]
            confidence = 0.15
            style = "unknown"
            self.warnings.append("Reference splitting confidence is low; fallback to single merged entry.")
        elif len(entries) == 1:
            entries = split_merged_entry(entries[0])
            if len(entries) > 1:
                confidence = max(confidence, 0.7)

        self.reference_stats.entry_count = len(entries)
        self.reference_stats.confidence = round(confidence, 3)
        self.reference_stats.style = style

        out = []
        for i, ref in enumerate(entries, start=1):
            out.append(f"{i}. {fix_hyphen_breaks(ref)}")
        return "\n".join(out)

    def _build_meta(self, pdf_path: Path, page_count: int, col_info: dict[str, Any]) -> dict[str, Any]:
        return {
            "pdf_path": str(pdf_path),
            "page_count": int(page_count),
            "is_double_column": bool(col_info.get("is_double_column", False)),
            "layout_per_page": self.layout_per_page,
            "removed_header_footer_examples": self.removed_header_footer_examples[:20],
            "figure_count": int(self.figure_count),
            "table_count": int(self.table_stats.total_tables),
            "formula_count": int(self.formula_stats.total_formula),
            "formula_fallback_count": int(self.formula_stats.fallback_images),
            "table_fallback_count": int(self.table_stats.fallback_images),
            "reference_detected": bool(self.reference_stats.detected),
            "reference_confidence": float(self.reference_stats.confidence),
            "elapsed_seconds": round(float(self._paper_elapsed_s), 3),
            "formula_stats": {
                "total_formula": self.formula_stats.total_formula,
                "latex_success": self.formula_stats.latex_success,
                "fallback_images": self.formula_stats.fallback_images,
                "llm_attempts": self.formula_stats.llm_attempts,
            },
            "table_stats": {
                "total_tables": self.table_stats.total_tables,
                "structured_tables": self.table_stats.structured_tables,
                "fallback_images": self.table_stats.fallback_images,
                "ocr_hint_count": self.table_stats.ocr_hint_count,
            },
            "reference_stats": {
                "detected": self.reference_stats.detected,
                "start_page": self.reference_stats.start_page,
                "entry_count": self.reference_stats.entry_count,
                "style": self.reference_stats.style,
                "confidence": self.reference_stats.confidence,
                "llm_attempts": self.reference_stats.llm_attempts,
            },
            "header_footer_template": {
                "top": self.header_templates_top,
                "bottom": self.header_templates_bottom,
            },
            "warnings": self.warnings,
        }

    def _write_compat_outputs(self, out_dir: Path, paper_name: str) -> None:
        if not self.cfg.keep_compat_test2:
            return
        paper_md = out_dir / "paper.md"
        if paper_md.exists():
            dst_en = out_dir / f"{paper_name}.en.md"
            shutil.copyfile(fs_path(paper_md), fs_path(dst_en))
        images_dir = out_dir / "images"
        assets_dir = out_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        for img in images_dir.glob("*"):
            if img.is_file():
                dst = assets_dir / img.name
                if not dst.exists():
                    try:
                        os.link(fs_path(img), fs_path(dst))
                    except Exception:
                        shutil.copyfile(fs_path(img), fs_path(dst))
        pngs = sorted(p.name for p in assets_dir.glob("*.png"))
        if pngs:
            manifest = "\n".join(f"- ![asset](./assets/{name})" for name in pngs) + "\n"
            (out_dir / "assets_manifest.md").write_text(manifest, encoding="utf-8")


def maybe_mathpix_recognize(image_path: Path) -> Optional[str]:
    """
    Optional formula OCR via Mathpix.
    Enabled only when env PDF2MD_V2_ENABLE_MATHPIX=true.
    """
    if str(os.environ.get("PDF2MD_V2_ENABLE_MATHPIX", "false")).lower() not in {"1", "true", "yes", "on"}:
        return None
    app_id = os.environ.get("MATHPIX_APP_ID", "").strip()
    app_key = os.environ.get("MATHPIX_APP_KEY", "").strip()
    if not app_id or not app_key:
        return None
    if requests is None:
        return None
    try:
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "src": f"data:image/png;base64,{data}",
            "formats": ["latex_styled"],
            "ocr": ["math"],
        }
        resp = requests.post(
            "https://api.mathpix.com/v3/text",
            headers={"app_id": app_id, "app_key": app_key, "Content-type": "application/json"},
            json=payload,
            timeout=40,
        )
        if resp.status_code != 200:
            return None
        obj = resp.json()
        latex = normalize_text(str(obj.get("latex_styled", "")))
        return latex or None
    except Exception:
        return None


def parse_args(argv: Optional[list[str]] = None) -> Config:
    ap = argparse.ArgumentParser(
        description="Research-grade PDF parser: PDF -> Markdown (RAG-friendly) with formulas/tables/layout handling."
    )
    ap.add_argument("--input", required=True, help="Input PDF file or directory.")
    ap.add_argument("--output", required=True, help="Output root directory.")
    ap.add_argument("--workers", type=int, default=4, help="Worker count for batch conversion.")
    ap.add_argument("--use-llm", type=parse_bool, default=False, help="Enable LLM-assisted tasks (default: false).")
    ap.add_argument("--use-ocr", type=parse_bool, default=False, help="Enable OCR fallback for scanned PDFs.")
    ap.add_argument("--recursive", type=parse_bool, default=False, help="When input is a directory, recurse.")
    ap.add_argument(
        "--compat-test2",
        type=parse_bool,
        default=True,
        help="Write extra compatibility files (assets/, paper_name.en.md).",
    )

    ap.add_argument("--llm-api-key", default=os.environ.get("PDF2MD_V2_LLM_API_KEY", ""))
    ap.add_argument("--llm-base-url", default=os.environ.get("PDF2MD_V2_LLM_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--llm-model", default=os.environ.get("PDF2MD_V2_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--llm-timeout", type=float, default=float(os.environ.get("PDF2MD_V2_LLM_TIMEOUT", "40")))
    args = ap.parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    llm_cfg = LLMConfig(
        enabled=bool(args.use_llm),
        api_key=str(args.llm_api_key).strip(),
        base_url=str(args.llm_base_url).strip(),
        model=str(args.llm_model).strip(),
        timeout_s=float(args.llm_timeout),
    )
    return Config(
        input_path=in_path,
        output_dir=out_dir,
        workers=max(1, int(args.workers)),
        use_llm=bool(args.use_llm),
        use_ocr=bool(args.use_ocr),
        recursive=bool(args.recursive),
        keep_compat_test2=bool(args.compat_test2),
        llm=llm_cfg,
    )


def collect_pdf_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is neither PDF nor directory: {input_path}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted([p for p in input_path.glob(pattern) if p.is_file()])


def convert_one(pdf_path: Path, cfg: Config) -> tuple[Path, Optional[str]]:
    worker = PDFToMarkdownV2(cfg)
    try:
        out_dir = worker.convert(pdf_path)
        return out_dir, None
    except Exception as e:
        return cfg.output_dir / safe_name(pdf_path.stem), str(e)


def main(argv: Optional[list[str]] = None) -> None:
    cfg = parse_args(argv)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    global _LLM_RUNTIME
    _LLM_RUNTIME = LLMRuntime(cfg.llm)
    if cfg.use_llm and not _LLM_RUNTIME.enabled:
        print("WARNING: --use-llm=true but LLM runtime is unavailable (missing key/openai package). Falling back to non-LLM.")
    if cfg.use_ocr and (pytesseract is None or Image is None):
        print("WARNING: --use-ocr=true but pytesseract/Pillow unavailable; OCR fallback will be skipped.")

    pdf_files = collect_pdf_files(cfg.input_path, cfg.recursive)
    if not pdf_files:
        raise SystemExit("No PDF files found.")

    started = time.time()
    print(f"Found {len(pdf_files)} PDF(s).")

    failures: list[tuple[Path, str]] = []
    if len(pdf_files) == 1:
        out_dir, err = convert_one(pdf_files[0], cfg)
        if err:
            failures.append((pdf_files[0], err))
            print(f"[FAILED] {pdf_files[0].name}: {err}")
        else:
            print(f"[OK] {pdf_files[0].name} -> {out_dir}")
    else:
        workers = min(cfg.workers, len(pdf_files))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(convert_one, p, cfg): p for p in pdf_files}
            for fut in as_completed(fut_map):
                p = fut_map[fut]
                out_dir, err = fut.result()
                if err:
                    failures.append((p, err))
                    print(f"[FAILED] {p.name}: {err}")
                else:
                    print(f"[OK] {p.name} -> {out_dir}")

    elapsed = time.time() - started
    print(f"Done in {elapsed:.2f}s")
    if failures:
        print("Failures:")
        for p, err in failures:
            print(f"- {p}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()




