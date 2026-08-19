from __future__ import annotations

import difflib
import heapq
import hashlib
import json
import os
import re
import shutil
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .post_processing import postprocess_markdown
from .post_math_rules import contains_bare_tagged_display_math, repair_inline_math_prose_boundaries_document
from .post_references import _is_post_references_resume_heading_line
from .quality_acceptance import (
    is_prose_dominant_display_math_block,
    summarize_conversion_quality,
)
from .quality_compare import compare_markdown_quality
from .reference_markdown import (
    fix_references_format,
    normalize_references_page_text,
    _is_plausible_reference_number,
    _looks_like_author_year_reference_text,
)
from .pdf_reference_text import (
    consecutive_reference_chain_positions,
    is_ambiguous_reference_running_line,
    is_reference_running_line,
    merge_standalone_reference_continuations,
    reference_ordered_page_text,
    trim_reference_publisher_tail,
)
from .reference_page_vl import reference_markdown_entry_count
from .tables import (
    markdown_table_issue_counts,
    markdown_table_issue_spans,
    normalize_markdown_tables_document,
    repair_detached_markdown_table_rows_document,
)
from .text_utils import _normalize_text, contains_only_detached_accent_mojibake, normalize_detached_accents
from kb.inpaper_citation_grounding import iter_inpaper_numeric_citations, parse_ref_num_set
from kb.reference_index import extract_references_map_from_md

try:
    import fitz
except ImportError:
    fitz = None


PAGE_MARKER_RE = re.compile(r"<!--\s*kb_page:\s*(\d+)\s*-->", re.IGNORECASE)
DISPLAY_MATH_DELIMITER_RE = re.compile(r"^\s*\$\$\s*$")
IMAGE_LINE_RE = re.compile(r"^(\s*)!\[([^\]]*)]\(([^)]+)\)\s*$")
CONVERSION_RETRY_MARKER_RE = re.compile(r"<!--\s*kb:conversion_retry\s+(.+?)\s*-->", re.IGNORECASE)
CONVERSION_RETRY_ATTR_RE = re.compile(r"([A-Za-z_][\w-]*)=(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))")
CAPTION_LINE_RE = re.compile(
    r"^\s*(?:\*{1,2}\s*)?(?:fig(?:ure)?\.?|table|algorithm)\s*(?:S?\d+[A-Za-z]?|[A-Za-z](?:\.\d+)?|[IVXLC]+)\b",
    re.IGNORECASE,
)
REFERENCES_HEADING_RE = re.compile(r"^#{1,6}\s+(?:References|Bibliography)\s*$", re.IGNORECASE)
BODY_SECTION_HEADING_RE = re.compile(
    r"^#{1,6}\s+(?:\d+(?:\.\d+)*\.?\s+|[IVXLC]+\.\s+)?"
    r"(?:introduction|background|related\s+work|theory|principle|comparison|method(?:s|ology)?|"
    r"experiment(?:s|al)?|results?|discussion|conclusions?|challenges?|outlooks?|future\s+work|"
    r"implementation|analysis|system|structure)\b",
    re.IGNORECASE,
)

_TRANSACTIONAL_STRUCTURE_ISSUES = {
    "missing_page_markers",
    "page_marker_gaps",
    "missing_source_pages",
    "source_page_marker_alignment",
    "source_page_count_mismatch",
    "source_table_page_alignment",
    "missing_references",
    "reference_index_truncated",
    "references_before_body",
    "out_of_order_sections",
}


_PAGE_MARKER_OFFSETS_CACHE_MAX_ITEMS = 32
_PAGE_MARKER_OFFSETS_CACHE_LOCK = threading.Lock()
_PAGE_MARKER_OFFSETS_CACHE: dict[tuple[str, int, int, int, str, bool], dict[int, int]] = {}


def _page_marker_offsets_cache_key(
    md_text: str,
    pdf_path: Path,
    *,
    snap_to_line_start: bool,
) -> tuple[str, int, int, int, str, bool] | None:
    try:
        path = Path(pdf_path).expanduser().resolve()
        stat = path.stat()
        text = str(md_text or "")
        digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        return (
            os.path.normcase(str(path)),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            len(text),
            digest,
            bool(snap_to_line_start),
        )
    except Exception:
        return None


def _get_cached_page_marker_offsets(
    key: tuple[str, int, int, int, str, bool] | None,
) -> dict[int, int] | None:
    if key is None:
        return None
    with _PAGE_MARKER_OFFSETS_CACHE_LOCK:
        cached = _PAGE_MARKER_OFFSETS_CACHE.get(key)
        return dict(cached) if cached is not None else None


def _cache_page_marker_offsets(
    key: tuple[str, int, int, int, str, bool] | None,
    offsets: dict[int, int],
) -> None:
    if key is None:
        return
    with _PAGE_MARKER_OFFSETS_CACHE_LOCK:
        _PAGE_MARKER_OFFSETS_CACHE[key] = dict(offsets)
        while len(_PAGE_MARKER_OFFSETS_CACHE) > _PAGE_MARKER_OFFSETS_CACHE_MAX_ITEMS:
            oldest = next(iter(_PAGE_MARKER_OFFSETS_CACHE), None)
            if oldest is None:
                break
            _PAGE_MARKER_OFFSETS_CACHE.pop(oldest, None)


def _reference_map_missing_numbers(ref_map: dict[int, str]) -> list[int]:
    refs: set[int] = set()
    for raw_key, raw_value in (ref_map if isinstance(ref_map, dict) else {}).items():
        try:
            n = int(raw_key)
        except Exception:
            continue
        if n <= 0 or not str(raw_value or "").strip():
            continue
        refs.add(n)
    if len(refs) < 8:
        return []
    first = min(refs)
    last = max(refs)
    if first != 1 or last < 8:
        return []
    return sorted(int(n) for n in (set(range(first, last + 1)) - refs))


def _reference_gap_is_material(ref_map: dict[int, str]) -> bool:
    missing = _reference_map_missing_numbers(ref_map)
    if not missing:
        return False
    last = 0
    for raw_key in (ref_map if isinstance(ref_map, dict) else {}).keys():
        try:
            last = max(last, int(raw_key))
        except Exception:
            continue
    if last < 8:
        return False
    # A small tail gap can come from intentionally omitted supplement-only refs.
    # Interior gaps are almost always conversion damage.
    return any(int(n) < last for n in missing)


def _reference_map_has_short_truncated_entries(ref_map: dict[int, str]) -> bool:
    rows: list[tuple[int, str]] = []
    for raw_key, raw_value in (ref_map if isinstance(ref_map, dict) else {}).items():
        try:
            n = int(raw_key)
        except Exception:
            continue
        text = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        if n > 0 and text:
            rows.append((n, text))
    if len(rows) < 8:
        return False
    bad_short = 0
    for _, text in sorted(rows):
        body = re.sub(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}[.)])\s*", "", text).strip(" .;:,")
        if not body:
            continue
        if re.search(r"\b(?:18|19|20)\d{2}\b|https?://|www\.|\bdoi\s*:|10\.\d{4,9}/", body, flags=re.IGNORECASE):
            continue
        if re.search(
            r"\b(?:journal|proceedings?|proc\.?|ieee|acm|spie|springer|wiley|opt\.?|optics?|"
            r"photonics?|phys\.?|nature|science|letters?|review|commun\.?|express|trans\.?)\b"
            r".*(?:\b\d{1,4}\s*,\s*)?[A-Za-z]?\d{3,}\s*$",
            body,
            flags=re.IGNORECASE,
        ):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", body)
        # A bibliography entry that reaches the next PDF page can be much
        # longer than the compact-entry heuristic below while still ending in
        # a converter-owned line-wrap hyphen (for example ``view syn-``).
        # Treat that terminal hyphen as sufficient evidence of truncation so
        # the source-backed reference rebuild gets a chance to join the tail.
        if body.endswith("-") and len(words) >= 4:
            return True
        if len(body) <= 32 and len(words) <= 5 and re.search(r"\bet\s+al\.?$", body, flags=re.IGNORECASE):
            return True
        if (
            len(body) <= 140
            and len(words) >= 4
            and (
                body.endswith(("-", ","))
                or re.search(r"\b(?:a|an|and|based|for|from|in|of|on|over|the|to|using|with)\s*$", body, re.IGNORECASE)
            )
        ):
            bad_short += 1
            continue
        if len(body) <= 90 and len(words) >= 4:
            bad_short += 1
    return bad_short >= 2


def _reference_map_has_inflated_tail(before_map: dict[int, str], recovered_map: dict[int, str]) -> bool:
    before_nums = sorted(
        int(k)
        for k, v in (before_map if isinstance(before_map, dict) else {}).items()
        if str(v or "").strip() and str(k).isdigit() and int(k) > 0
    )
    recovered_nums = sorted(
        int(k)
        for k, v in (recovered_map if isinstance(recovered_map, dict) else {}).items()
        if str(v or "").strip() and str(k).isdigit() and int(k) > 0
    )
    if len(before_nums) < 12 or len(recovered_nums) < 8:
        return False
    if len(recovered_nums) >= len(before_nums):
        return False
    if (len(recovered_nums) / max(1, len(before_nums))) < 0.45:
        return False
    if len(before_nums) < int(len(recovered_nums) * 1.35):
        return False
    if _reference_map_missing_numbers(recovered_map):
        return False
    if min(recovered_nums) != 1 or max(recovered_nums) < len(recovered_nums) * 0.85:
        return False
    recovered_max = max(recovered_nums)
    tail_nums = [n for n in before_nums if n > recovered_max]
    return len(tail_nums) >= max(5, int(len(before_nums) * 0.20))


CONVERSION_QUALITY_RESULT_FILENAME = "conversion_quality_result.json"
CONVERSION_QUALITY_RULES_VERSION = 15
MAX_CONVERSION_REPAIR_ATTEMPTS = 30
PAGE_ALIGNMENT_NGRAMS = (8, 6)
PAGE_ALIGNMENT_DEFAULT_NGRAM = PAGE_ALIGNMENT_NGRAMS[0]
SOURCE_PAGE_COVERAGE_THRESHOLD = 0.66
SOURCE_PAGE_MIN_RARE_TOKENS = 60
SOURCE_PAGE_EMPTY_MARKER_MIN_ALNUM_CHARS = 500
SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD = 0.32
SOURCE_PAGE_MIN_WRAPPED_WORDS = 3
SOURCE_PAGE_MAX_WRAP_PREFIX_CHARS = 12
SOURCE_PAGE_MAX_WRAP_SUFFIX_CHARS = 24
SOURCE_PAGE_MISSING_WRAP_RATIO = 0.20
SOURCE_PAGE_MIN_PROSE_BLOCK_TOKENS = 80
SOURCE_PAGE_MIN_ANCHORED_OMITTED_WORDS = 8
SOURCE_PAGE_MIN_PROSE_SENTENCES = 2
PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS = 1200
PAGE_ALIGNMENT_BEAM_SIZE = 300
PAGE_ALIGNMENT_BEAM_PER_MATCH = 80
LEADING_PAGE_ALIGNMENT_MAX_OFFSET = 900
LEADING_PAGE_ALIGNMENT_WINDOW_CHARS = 3200
LEADING_PAGE_DROP_MAX_PREVIOUS_COVERAGE = 0.50
LEADING_PAGE_DROP_MIN_COVERAGE_MARGIN = 0.12
PAGE_ALIGNMENT_STOP_WORDS = {
    "the", "of", "and", "in", "to", "for", "with", "on", "by", "from", "that", "this",
    "these", "those", "using", "used", "into", "such", "which", "were", "their", "have",
    "has", "been", "between", "through", "under", "over", "figure", "table", "results",
    "method", "methods", "abstract", "introduction",
}


CONVERSION_REPAIR_STRATEGIES: dict[str, dict[str, Any]] = {
    "missing_markdown": {
        "label": "Convert the PDF because no Markdown output exists",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "No converted Markdown exists, so the source PDF must be converted.",
        "strategies": [],
    },
    "missing_images": {
        "label": "Reconvert the paper to rebuild missing image assets",
        "safe": False,
        "action": "reconvert",
        "scope": "assets",
        "speed_mode": "normal",
        "reason": "Markdown references image assets that are missing on disk.",
        "strategies": ["repair_missing_image_links"],
    },
    "mojibake": {
        "label": "Reconvert with vision mode to avoid encoding artifacts",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "Encoding artifacts usually originate in extraction and should not be indexed.",
        "strategies": [],
    },
    "detached_accents": {
        "label": "Reattach detached accents in extracted names",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "The detected encoding artifacts are limited to safely reattachable name accents.",
        "strategies": ["normalize_detached_accents"],
    },
    "weak_structure": {
        "label": "Reconvert with higher-quality structure extraction",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "The output has too little heading structure for reliable retrieval and location.",
        "strategies": [],
    },
    "missing_references": {
        "label": "Reconvert to recover the reference section",
        "safe": False,
        "action": "reconvert",
        "scope": "references",
        "speed_mode": "normal",
        "reason": "The bibliography is missing, which breaks upstream citations and metadata repair.",
        "strategies": [],
    },
    "source_text_loss": {
        "label": "Reconvert because body text appears to be missing from the Markdown",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "The converted Markdown is much shorter than the source PDF or reaches references before recovering the body.",
        "strategies": [],
    },
    "missing_source_pages": {
        "label": "Recover missing PDF page text into Markdown",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "One or more source PDF pages have too little text represented in Markdown.",
        "strategies": ["recover_missing_source_pages"],
    },
    "source_page_marker_alignment": {
        "label": "Realign Markdown page anchors from the source PDF",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "The source PDF page starts do not line up with the Markdown page anchors.",
        "strategies": ["realign_page_markers_from_pdf", "recover_missing_source_pages"],
    },
    "source_page_count_mismatch": {
        "label": "Reconvert pages missing from the Markdown page sequence",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "Multiple source PDF pages are absent from the Markdown page-marker sequence.",
        "strategies": [],
    },
    "source_table_page_alignment": {
        "label": "Move page anchors before tables found on the next source page",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "A table has a high-confidence source-page match, but the next page anchor is currently placed after it.",
        "strategies": ["realign_table_page_markers_from_pdf"],
    },
    "reference_index_truncated": {
        "label": "Rebuild the reference section from the source PDF text layer",
        "safe": True,
        "action": "autofix",
        "scope": "references",
        "reason": "The visible reference lines are interrupted or not extractable as a complete reference index.",
        "strategies": ["pdf_reference_backfill"],
    },
    "references_before_body": {
        "label": "Move an early References block behind recovered body sections",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["move_early_references_to_end"],
    },
    "analyzer_errors": {
        "label": "Run deterministic Markdown post-processing",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": [
            "repair_detached_table_rows",
            "repair_inline_math_boundaries",
            "normalize_markdown_tables",
            "demote_malformed_numbered_headings",
            "postprocess_markdown",
            "balance_display_math",
            "figure_metadata_captions",
        ],
    },
    "analyzer_warnings": {
        "label": "Normalize headings, captions, tables, and layout noise",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": [
            "normalize_markdown_tables",
            "postprocess_markdown",
            "figure_metadata_captions",
            "pdf_text_captions",
            "recover_ambiguous_table_pages",
        ],
    },
    "collapsed_table_rows": {
        "label": "Recover collapsed table rows or preserve the source PDF page",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Multiple logical data rows were packed into cells with literal HTML break markers.",
        "strategies": ["recover_ambiguous_table_pages", "normalize_markdown_tables"],
    },
    "ambiguous_table_break_rows": {
        "label": "Preserve ambiguous tables as source PDF page evidence",
        "safe": True,
        "action": "autofix",
        "scope": "pages",
        "reason": "When row coordinates cannot be reconstructed safely, retain the original PDF page image and text layer instead of guessing cells.",
        "strategies": ["recover_ambiguous_table_pages"],
    },
    "duplicate_table_representations": {
        "label": "Remove a nearby lower-quality duplicate table",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "The same table data appears in nearby compact and structured representations.",
        "strategies": ["normalize_markdown_tables"],
    },
    "fragmented_table_columns": {
        "label": "Reconvert a table with fragmented columns",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "A wide table contains broken header and decimal fragments that cannot be safely realigned from Markdown alone.",
        "strategies": [],
    },
    "missing_abstract": {
        "label": "Infer and insert Abstract heading from front matter",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["abstract_heading_only", "postprocess_markdown"],
    },
    "missing_page_markers": {
        "label": "Insert a fallback page anchor at the Markdown start",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["ensure_page_anchor"],
    },
    "page_marker_gaps": {
        "label": "Normalize duplicate or out-of-order page anchors",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["realign_page_markers_from_pdf", "normalize_page_markers", "postprocess_markdown"],
    },
    "missing_captions": {
        "label": "Recover visible captions from alt text and figure metadata sidecars",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["postprocess_markdown", "figure_metadata_captions", "pdf_text_captions"],
    },
    "unclosed_display_math": {
        "label": "Close a trailing display-math block when the delimiter is unbalanced",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["balance_display_math"],
    },
    "conversion_retry_math_text": {
        "label": "Recover unresolved math text from its source PDF page",
        "safe": True,
        "action": "autofix",
        "scope": "pages",
        "reason": "Retain visual evidence and recover authoritative prose from the same PDF page.",
        "strategies": ["recover_conversion_retry_pages"],
    },
    "conversion_retry_equation": {
        "label": "Ground equation fallbacks in their source PDF pages",
        "safe": True,
        "action": "autofix",
        "scope": "pages",
        "reason": "Keep the equation image and add the authoritative text layer from the same PDF page.",
        "strategies": ["recover_conversion_retry_pages"],
    },
    "conversion_retry_other": {
        "label": "Review unresolved conversion retries",
        "safe": False,
        "action": "review",
        "scope": "document",
        "reason": "The Markdown contains an unresolved converter retry marker with an unrecognized kind.",
        "strategies": [],
    },
    "prose_dominant_display_math": {
        "label": "Unwrap prose captured as display math",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Natural-language prose was captured inside a display-math block, which breaks reading and formula rendering.",
        "strategies": ["unwrap_prose_display_math", "postprocess_markdown"],
    },
    "display_math_markdown_link": {
        "label": "Reconvert Markdown links captured inside display math",
        "safe": False,
        "action": "reconvert",
        "scope": "document",
        "speed_mode": "normal",
        "reason": "A Markdown link appears inside display math and cannot be rendered as valid TeX.",
        "strategies": [],
    },
    "source_page_text_corruption": {
        "label": "Reconvert source pages with damaged column or line-wrap text",
        "safe": False,
        "action": "reconvert",
        "scope": "pages",
        "speed_mode": "normal",
        "reason": (
            "One or more converted pages lost the leading half of source words at PDF line wraps, "
            "so those pages are not reliable enough for retrieval or answer evidence."
        ),
        "strategies": ["recover_corrupted_source_pages", "postprocess_markdown"],
    },
    "source_page_prose_omission": {
        "label": "Restore source-proven words missing from interior prose",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": (
            "One or more converted pages dropped source-proven words from the middle of long prose blocks, "
            "so bounded differences are restored from the PDF text layer before indexing."
        ),
        "strategies": ["recover_source_prose_omissions", "postprocess_markdown"],
    },
    "numeric_only_headings": {
        "label": "Demote plot-axis numbers captured as headings",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Numeric plot labels were captured as document headings and would distort reader navigation.",
        "strategies": ["postprocess_markdown"],
    },
    "plain_method_subheadings": {
        "label": "Promote recognized Methods subsection titles",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Recognized Methods subsection titles were left as body text, weakening navigation and retrieval paths.",
        "strategies": ["postprocess_markdown"],
    },
    "heading_level_jumps": {
        "label": "Rebalance heading levels using the established heading policy",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "strategies": ["normalize_heading_levels"],
    },
    "collapsed_heading_hierarchy": {
        "label": "Promote collapsed review-paper headings for better navigation",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "A review-style paper has many section headings collapsed under H3, which weakens reader navigation and retrieval heading paths.",
        "strategies": ["promote_collapsed_review_headings"],
    },
    "stray_inline_math": {
        "label": "Clean stray inline math markup",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Prose contains bare LaTeX or OCR math leftovers that should be normalized as inline math.",
        "strategies": ["postprocess_markdown", "promote_collapsed_review_headings"],
    },
    "adjacent_inline_math_superscript": {
        "label": "Repair a detached numeric superscript beside an inline formula",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Adjacent inline-math delimiters produce invalid KaTeX; the surrounding formula and punctuation distinguish a unit power from a citation marker.",
        "strategies": ["postprocess_markdown"],
    },
    "legacy_numeric_superscript_citation": {
        "label": "Normalize a legacy numeric superscript citation",
        "safe": True,
        "action": "autofix",
        "scope": "markdown",
        "reason": "Raw HTML or LaTeX superscript citation wrappers render as source markup unless they are normalized to canonical citation brackets.",
        "strategies": ["postprocess_markdown"],
    },
    "out_of_order_sections": {
        "label": "Review numbered sections that appear out of source order",
        "safe": False,
        "action": "review",
        "scope": "document",
        "reason": "A later numbered section appears before an earlier one, which can indicate cross-page text displacement.",
        "strategies": ["restore_numbered_headings_from_pdf", "demote_malformed_numbered_headings"],
    },
}


def conversion_repair_strategy_for_issue(code: str) -> dict[str, Any]:
    clean = str(code or "").strip().lower()
    strategy = CONVERSION_REPAIR_STRATEGIES.get(clean)
    if not strategy:
        return {
            "label": "",
            "safe": False,
            "strategies": [],
        }
    return {
        "label": str(strategy.get("label") or ""),
        "safe": bool(strategy.get("safe")),
        "action": str(strategy.get("action") or ("autofix" if bool(strategy.get("safe")) else "")),
        "scope": str(strategy.get("scope") or ""),
        "speed_mode": str(strategy.get("speed_mode") or ""),
        "reason": str(strategy.get("reason") or ""),
        "strategies": [str(item) for item in list(strategy.get("strategies") or []) if str(item or "").strip()],
    }


def conversion_quality_result_path(md_path: Path | str) -> Path:
    return Path(md_path).expanduser().parent / CONVERSION_QUALITY_RESULT_FILENAME


def plan_conversion_quality_repair(issue_codes: list[str] | None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    codes = []
    seen: set[str] = set()
    for raw in list(issue_codes or []):
        code = str(raw or "").strip().lower()
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    issue_actions: list[dict[str, Any]] = []
    reconvert_actions: list[dict[str, Any]] = []
    autofix_actions: list[dict[str, Any]] = []
    review_actions: list[dict[str, Any]] = []
    for code in codes:
        strategy = conversion_repair_strategy_for_issue(code)
        action = str(strategy.get("action") or "").strip() or ("autofix" if bool(strategy.get("safe")) else "review")
        row = {
            "code": code,
            "action": action,
            "safe": bool(strategy.get("safe")),
            "scope": str(strategy.get("scope") or ""),
            "label": str(strategy.get("label") or code),
            "reason": str(strategy.get("reason") or ""),
            "speed_mode": str(strategy.get("speed_mode") or ""),
            "strategies": list(strategy.get("strategies") or []),
        }
        issue_actions.append(row)
        if action == "reconvert":
            reconvert_actions.append(row)
        elif action == "autofix":
            autofix_actions.append(row)
        else:
            review_actions.append(row)

    if reconvert_actions:
        scope_priority = {"document": 5, "pages": 4, "assets": 3, "references": 2, "markdown": 1}
        primary = sorted(
            reconvert_actions,
            key=lambda item: scope_priority.get(str(item.get("scope") or ""), 0),
            reverse=True,
        )[0]
        return {
            "action": "reconvert",
            "scope": str(primary.get("scope") or "document"),
            "speed_mode": str(primary.get("speed_mode") or "normal") or "normal",
            "no_llm": False,
            "replace": True,
            "md_autofix_first": bool(autofix_actions),
            "reason": str(primary.get("reason") or primary.get("label") or "Source conversion needs repair."),
            "issue_codes": codes,
            "reconvert_issue_codes": [str(item.get("code") or "") for item in reconvert_actions],
            "autofix_issue_codes": [str(item.get("code") or "") for item in autofix_actions],
            "review_issue_codes": [str(item.get("code") or "") for item in review_actions],
            "issue_actions": issue_actions,
            "metrics": dict(metrics or {}),
        }
    if autofix_actions:
        return {
            "action": "autofix",
            "scope": "markdown",
            "speed_mode": "",
            "no_llm": False,
            "replace": False,
            "md_autofix_first": True,
            "reason": "Remaining issues are covered by deterministic Markdown repair.",
            "issue_codes": codes,
            "reconvert_issue_codes": [],
            "autofix_issue_codes": [str(item.get("code") or "") for item in autofix_actions],
            "review_issue_codes": [str(item.get("code") or "") for item in review_actions],
            "issue_actions": issue_actions,
            "metrics": dict(metrics or {}),
        }
    if review_actions:
        return {
            "action": "review",
            "scope": "manual",
            "speed_mode": "",
            "no_llm": False,
            "replace": False,
            "md_autofix_first": False,
            "reason": "The remaining issues do not have a safe automated repair yet.",
            "issue_codes": codes,
            "reconvert_issue_codes": [],
            "autofix_issue_codes": [],
            "review_issue_codes": [str(item.get("code") or "") for item in review_actions],
            "issue_actions": issue_actions,
            "metrics": dict(metrics or {}),
        }
    return {
        "action": "none",
        "scope": "",
        "speed_mode": "",
        "no_llm": False,
        "replace": False,
        "md_autofix_first": False,
        "reason": "No conversion quality issues detected.",
        "issue_codes": [],
        "reconvert_issue_codes": [],
        "autofix_issue_codes": [],
        "review_issue_codes": [],
        "issue_actions": [],
        "metrics": dict(metrics or {}),
    }


def _current_markdown_stat(path: Path) -> dict[str, int]:
    try:
        stat = path.stat()
        return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
    except Exception:
        return {"mtime_ns": 0, "size": 0}


def conversion_quality_report_is_stale(md_path: Path | str, report: Mapping[str, Any] | None) -> bool:
    path = Path(md_path).expanduser()
    payload = report if isinstance(report, Mapping) else {}
    if int(payload.get("quality_rules_version") or 0) != CONVERSION_QUALITY_RULES_VERSION:
        return True
    current = _current_markdown_stat(path)
    return (
        int(payload.get("md_mtime_ns") or 0) != current["mtime_ns"]
        or int(payload.get("md_size") or 0) != current["size"]
    )


def load_conversion_quality_result(md_path: Path | str) -> dict[str, Any]:
    report_path = conversion_quality_result_path(md_path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def append_conversion_repair_attempt(
    md_path: Path | str,
    *,
    event: str,
    status: str,
    action: str = "",
    scope: str = "",
    speed_mode: str = "",
    issue_codes: list[str] | None = None,
    task_id: str = "",
    repair_run_id: str = "",
    source: str = "",
    reason: str = "",
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report_path = conversion_quality_result_path(path)
    report = load_conversion_quality_result(path)
    if not report and path.exists():
        try:
            report = write_conversion_quality_result(path)
        except Exception:
            report = {}
    if not isinstance(report, dict):
        report = {}

    row = {
        "event": str(event or "").strip() or "repair",
        "status": str(status or "").strip() or "info",
        "action": str(action or "").strip(),
        "scope": str(scope or "").strip(),
        "speed_mode": str(speed_mode or "").strip(),
        "issue_codes": [str(item or "").strip() for item in list(issue_codes or []) if str(item or "").strip()][:30],
        "task_id": str(task_id or "").strip(),
        "repair_run_id": str(repair_run_id or "").strip(),
        "source": str(source or "").strip(),
        "reason": str(reason or "").strip()[:500],
        "detail": str(detail or "").strip()[:800],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(extra, dict) and extra:
        row["extra"] = {
            str(k): v
            for k, v in extra.items()
            if isinstance(k, str) and k and isinstance(v, (str, int, float, bool, type(None), list, dict))
        }

    attempts = report.get("repair_attempts") if isinstance(report.get("repair_attempts"), list) else []
    attempts = [item for item in attempts if isinstance(item, dict)]
    attempts.append(row)
    report["repair_attempts"] = attempts[-MAX_CONVERSION_REPAIR_ATTEMPTS:]
    report["latest_repair_attempt"] = row
    report.setdefault("schema_version", 1)
    report.setdefault("kind", "conversion_quality_result")
    report["md_path"] = str(path)
    try:
        stat = path.stat()
        report["md_mtime_ns"] = int(stat.st_mtime_ns)
        report["md_size"] = int(stat.st_size)
    except Exception:
        report.setdefault("md_mtime_ns", 0)
        report.setdefault("md_size", 0)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def write_conversion_quality_result(
    md_path: Path | str,
    *,
    auto_repair_result: dict[str, Any] | None = None,
    auto_repair_enabled: bool = True,
    source_pdf_path: Path | str | None = None,
    allow_source_pdf_inference: bool = True,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report_path = conversion_quality_result_path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics = _metric_view(path, text)
    repair = dict(auto_repair_result or {})
    cached_metrics = repair.get("after") if isinstance(repair.get("after"), dict) else {}
    cached_source_quality = (
        repair.get("source_quality_after")
        if isinstance(repair.get("source_quality_after"), dict)
        else {}
    )
    if cached_source_quality and cached_metrics == metrics:
        source_quality = dict(cached_source_quality)
    else:
        source_quality = _source_quality_view(
            path,
            text,
            metrics,
            source_pdf_path=source_pdf_path,
            allow_source_pdf_inference=allow_source_pdf_inference,
        )
    repair.pop("repaired_text", None)
    remaining = [
        str(code or "").strip().lower()
        for code in list(repair.get("remaining_issue_codes") or _issue_codes_from_context(path, text, metrics, source_quality=source_quality))
        if str(code or "").strip()
    ]
    md_stat = _current_markdown_stat(path)
    prev = load_conversion_quality_result(path)
    exhausted_issue_codes = _persistent_source_autofix_codes(repair) if auto_repair_result is not None else set()
    if auto_repair_result is None:
        prev_stat_matches = not conversion_quality_report_is_stale(path, prev)
        prev_auto_repair = prev.get("auto_repair") if isinstance(prev.get("auto_repair"), dict) else {}
        if prev_stat_matches:
            exhausted_issue_codes = {
                str(code or "").strip().lower()
                for code in list((prev_auto_repair or {}).get("exhausted_issue_codes") or [])
                if str(code or "").strip().lower() in remaining
            }
    repair_plan = plan_conversion_quality_repair(remaining, metrics=metrics)
    retry_pages = [
        int(page)
        for page in list(source_quality.get("evidence_unreliable_pages") or [])
        if str(page or "").isdigit() and int(page) > 0
    ]
    if exhausted_issue_codes:
        repair_plan = _escalate_persistent_source_autofix(
            repair_plan,
            {
                "issue_codes_before": sorted(exhausted_issue_codes),
                "remaining_issue_codes": remaining,
            },
            source_available=bool(source_quality.get("source_pdf_available")),
        )
    if retry_pages and str(repair_plan.get("action") or "").strip().lower() == "reconvert":
        repair_plan["retry_pages"] = sorted(set(retry_pages))[:500]
    recommended_action = str(repair_plan.get("action") or "review")
    prev_attempts = prev.get("repair_attempts") if isinstance(prev.get("repair_attempts"), list) else []
    prev_attempts = [item for item in prev_attempts if isinstance(item, dict)][-MAX_CONVERSION_REPAIR_ATTEMPTS:]
    latest_attempt = prev.get("latest_repair_attempt") if isinstance(prev.get("latest_repair_attempt"), dict) else (prev_attempts[-1] if prev_attempts else {})
    payload = {
        "schema_version": 1,
        "quality_rules_version": CONVERSION_QUALITY_RULES_VERSION,
        "kind": "conversion_quality_result",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "md_path": str(path),
        "md_mtime_ns": md_stat["mtime_ns"],
        "md_size": md_stat["size"],
        "auto_repair_enabled": bool(auto_repair_enabled),
        "auto_repair": {
            "changed": bool(repair.get("changed")),
            "unsafe": bool(repair.get("unsafe")),
            "applied": [str(item) for item in list(repair.get("applied") or []) if str(item or "").strip()][:20],
            "attempted_issue_codes": [
                str(item)
                for item in list(repair.get("attempted_issue_codes") or [])
                if str(item or "").strip()
            ][:30],
            "issue_codes_before": [str(item) for item in list(repair.get("issue_codes_before") or []) if str(item or "").strip()][:30],
            "issue_codes_after": [str(item) for item in list(repair.get("issue_codes_after") or []) if str(item or "").strip()][:30],
            "remaining_issue_codes": remaining[:30],
            "exhausted_issue_codes": sorted(exhausted_issue_codes)[:30],
            "regression_reasons": [str(item) for item in list(repair.get("regression_reasons") or []) if str(item or "").strip()][:20],
        },
        "repair_plan": repair_plan,
        "repair_attempts": prev_attempts,
        "latest_repair_attempt": latest_attempt,
        "recommended_action": recommended_action,
        "needs_reconvert": recommended_action == "reconvert",
        "metrics": metrics,
        "source_quality": source_quality,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


_PERSISTENT_SOURCE_AUTOFIX_ISSUES = {
    "page_marker_gaps",
    "source_page_marker_alignment",
    "source_table_page_alignment",
    "source_page_prose_omission",
    "conversion_retry_math_text",
    "conversion_retry_equation",
}


def _persistent_source_autofix_codes(repair_result: Mapping[str, Any]) -> set[str]:
    attempted = {
        str(code or "").strip().lower()
        for code in list(repair_result.get("attempted_issue_codes") or repair_result.get("issue_codes_before") or [])
        if str(code or "").strip()
    }
    remaining = {
        str(code or "").strip().lower()
        for code in list(repair_result.get("remaining_issue_codes") or [])
        if str(code or "").strip()
    }
    return attempted & remaining & _PERSISTENT_SOURCE_AUTOFIX_ISSUES


def _escalate_persistent_source_autofix(
    repair_plan: dict[str, Any],
    repair_result: Mapping[str, Any],
    *,
    source_available: bool,
) -> dict[str, Any]:
    """Stop offering a deterministic repair when the same source issue survives it."""
    persistent = _persistent_source_autofix_codes(repair_result)
    if not persistent:
        return repair_plan

    plan = dict(repair_plan)
    page_scoped = persistent == {"source_page_prose_omission"}
    persistent_reason = (
        "Source-proven prose omissions remain after bounded text-layer repair."
        if page_scoped
        else "Page anchors remain inconsistent after deterministic Markdown repair."
    )
    issue_actions: list[dict[str, Any]] = []
    for raw in list(plan.get("issue_actions") or []):
        row = dict(raw) if isinstance(raw, Mapping) else {}
        code = str(row.get("code") or "").strip().lower()
        if code in persistent:
            row.update(
                {
                    "action": "reconvert",
                    "safe": False,
                    "scope": "pages" if page_scoped else "document",
                    "reason": persistent_reason,
                    "speed_mode": "normal",
                    "strategies": ["source_reconversion"],
                }
            )
        issue_actions.append(row)

    autofix_codes = [
        str(code)
        for code in list(plan.get("autofix_issue_codes") or [])
        if str(code).strip().lower() not in persistent
    ]
    if source_available:
        plan.update(
            {
                "action": "reconvert",
                "scope": "pages" if page_scoped else "document",
                "speed_mode": "normal",
                "no_llm": False,
                "replace": True,
                "md_autofix_first": bool(autofix_codes),
                "reason": (
                    "Bounded source-text repair could not resolve every omission; retry only the affected pages."
                    if page_scoped
                    else "Page anchors remain inconsistent after deterministic repair; rerun source conversion."
                ),
                "reconvert_issue_codes": sorted(persistent),
                "autofix_issue_codes": autofix_codes,
                "issue_actions": issue_actions,
            }
        )
    else:
        retry_marker_codes = persistent & {
            "conversion_retry_math_text",
            "conversion_retry_equation",
        }
        for row in issue_actions:
            code = str(row.get("code") or "").strip().lower()
            if code in retry_marker_codes:
                row.update(
                    {
                        "action": "reconvert",
                        "safe": False,
                        "scope": "document",
                        "speed_mode": "normal",
                        "reason": "An unresolved conversion retry requires the source PDF to be restored and reconverted.",
                        "strategies": ["source_reconversion"],
                    }
                )
            elif code in persistent:
                row.update(
                    {
                        "action": "review",
                        "scope": "manual",
                        "speed_mode": "",
                        "reason": "Page anchors remain inconsistent, but the source PDF is unavailable.",
                        "strategies": [],
                    }
                )
        manual_codes = persistent - retry_marker_codes
        review_codes = list(plan.get("review_issue_codes") or []) + sorted(manual_codes)
        if retry_marker_codes:
            plan.update(
                {
                    "action": "reconvert",
                    "scope": "document",
                    "speed_mode": "normal",
                    "no_llm": False,
                    "replace": True,
                    "md_autofix_first": bool(autofix_codes),
                    "reason": "Unresolved conversion retries remain; restore the source PDF and rerun source conversion.",
                    "reconvert_issue_codes": sorted(retry_marker_codes),
                    "autofix_issue_codes": autofix_codes,
                    "review_issue_codes": list(dict.fromkeys(str(code) for code in review_codes)),
                    "issue_actions": issue_actions,
                }
            )
        else:
            plan.update(
                {
                    "action": "review",
                    "scope": "manual",
                    "speed_mode": "",
                    "no_llm": False,
                    "replace": False,
                    "md_autofix_first": bool(autofix_codes),
                    "reason": "Page anchors need review because the source PDF is unavailable.",
                    "reconvert_issue_codes": [],
                    "autofix_issue_codes": autofix_codes,
                    "review_issue_codes": list(dict.fromkeys(str(code) for code in review_codes)),
                    "issue_actions": issue_actions,
                }
            )
    return plan


def _metric_view(md_path: Path, text: str) -> dict[str, Any]:
    metrics = summarize_conversion_quality(md_path, text)
    return asdict(metrics)


def _issue_codes_from_metrics(metrics: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if int(metrics.get("missing_image_count") or 0) > 0:
        out.append("missing_images")
    if int(metrics.get("unclosed_display_math_block_count") or 0) > 0:
        out.append("unclosed_display_math")
    if int(metrics.get("conversion_retry_math_text_count") or 0) > 0:
        out.append("conversion_retry_math_text")
    if int(metrics.get("conversion_retry_equation_count") or 0) > 0:
        out.append("conversion_retry_equation")
    if int(metrics.get("conversion_retry_other_count") or 0) > 0:
        out.append("conversion_retry_other")
    if int(metrics.get("prose_dominant_display_math_block_count") or 0) > 0:
        out.append("prose_dominant_display_math")
    if int(metrics.get("display_math_markdown_link_count") or 0) > 0:
        out.append("display_math_markdown_link")
    if int(metrics.get("adjacent_inline_math_superscript_hazard_count") or 0) > 0:
        out.append("adjacent_inline_math_superscript")
    if int(metrics.get("legacy_numeric_superscript_citation_count") or 0) > 0:
        out.append("legacy_numeric_superscript_citation")
    if int(metrics.get("mojibake_count") or 0) > 0:
        out.append("mojibake")
    if int(metrics.get("analyzer_error_count") or 0) > 0:
        out.append("analyzer_errors")
    if int(metrics.get("heading_count") or 0) <= 1:
        out.append("weak_structure")
    if not bool(metrics.get("has_abstract_heading")):
        out.append("missing_abstract")
    if int(metrics.get("page_marker_count") or 0) <= 0:
        out.append("missing_page_markers")
    if int(metrics.get("page_marker_gap_count") or 0) > 0:
        out.append("page_marker_gaps")
    if int(metrics.get("extracted_reference_count") or 0) <= 0 and int(metrics.get("reference_line_count") or 0) <= 0:
        out.append("missing_references")
    if int(metrics.get("image_count") or 0) > 0 and int(metrics.get("caption_count") or 0) <= 0:
        out.append("missing_captions")
    if int(metrics.get("analyzer_warning_count") or 0) > 3:
        out.append("analyzer_warnings")
    if int(metrics.get("heading_level_jump_count") or 0) > 0:
        out.append("heading_level_jumps")
    ambiguous_break_rows = int(metrics.get("ambiguous_table_break_row_count") or 0)
    if int(metrics.get("collapsed_table_row_count") or 0) > 0:
        out.append("collapsed_table_rows")
    if ambiguous_break_rows > 0:
        out.append("ambiguous_table_break_rows")
    if (
        int(metrics.get("duplicate_table_count") or 0) > 0
        or int(metrics.get("fragmented_table_duplicate_count") or 0) > 0
    ):
        out.append("duplicate_table_representations")
    if (
        int(metrics.get("fragmented_table_column_count") or 0)
        > int(metrics.get("fragmented_table_duplicate_count") or 0)
    ):
        out.append("fragmented_table_columns")
    return out


def _dedupe_codes(codes: list[str]) -> list[str]:
    out: list[str] = []
    for raw in codes:
        code = str(raw or "").strip().lower()
        if code and code not in out:
            out.append(code)
    return out


def _collapsed_heading_hierarchy_likely(text: str, metrics: dict[str, Any], quality: dict[str, Any]) -> bool:
    if str((quality or {}).get("document_type") or "").strip().lower() != "review":
        return False
    heading_count = int(metrics.get("heading_count") or 0)
    h2_count = int(metrics.get("h2_count") or 0)
    h3_plus_count = int(metrics.get("h3_plus_count") or 0)
    if heading_count < 10 or h3_plus_count < 8:
        return False
    if h2_count >= 8:
        return False
    if h3_plus_count < max(8, h2_count * 2):
        return False
    # Require at least a few topical H3 headings. This avoids flagging papers
    # that only use H3 for appendices, captions, or small front-matter blocks.
    topical = 0
    for line in str(text or "").splitlines():
        match = re.match(r"^###\s+(.+?)\s*$", line)
        if not match:
            continue
        title = str(match.group(1) or "").strip()
        if _review_heading_promotable(title):
            topical += 1
        if topical >= 4:
            return True
    return False


_STRAY_INLINE_LATEX_RE = re.compile(
    r"\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega|"
    r"Theta|Sigma|hat|mathbf|boldsymbol)"
    r"(?:\s*(?:[_^]\{[^{}\n]{1,60}\}|[_^][A-Za-z0-9]))?"
    r"(?:\s*(?:[<>=]|\\leq|\\geq|\\neq|\\approx|\\in|\\notin)\s*"
    r"(?:[-+]?\d+(?:\.\d+)?|[A-Za-z](?:[_^]\{[^{}\n]{1,60}\}|[_^][A-Za-z0-9])?|"
    r"\\[A-Za-z]+(?:\{[^{}\n]{1,60}\})?|\[[^\]\n]{1,80}\]))+",
)
_STRAY_INLINE_LATEX_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:~|\\,)?\s*"
    r"\\mu\\mathrm\{[A-Za-z]{1,8}\}",
)
_STRAY_INLINE_CITATION_DOLLAR_RE = re.compile(
    r"\[\s*\d{1,4}(?:\s*[,;\u2013-]\s*\d{1,4})*\s*\]\$\s*(?=(?:and|or|the|this|that|these|those|[A-Z]))"
)
_STRAY_INLINE_CDOT_RE = re.compile(r"\bloss\s*\(\s*c(?:dot|[.\u00b7\u22c5])\s*\)", re.IGNORECASE)
_STRAY_INLINE_UNCLOSED_SENTENCE_RE = re.compile(
    r"\$(\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega|"
    r"Theta|Sigma|hat|mathbf|boldsymbol)[^$\n]{1,180}?)([.?!])(?=\s+[A-Z])"
)


def _stray_inline_math_likely(text: str) -> bool:
    if contains_bare_tagged_display_math(text):
        return True
    if not text or ("\\" not in text and "$" not in text and "cdot" not in text):
        return False
    in_fence = False
    in_math = False
    in_refs = False
    for line in str(text or "").splitlines():
        st = line.strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            continue
        if st == "$$":
            in_math = not in_math
            continue
        if REFERENCES_HEADING_RE.match(st):
            in_refs = True
            continue
        if in_fence or in_math or in_refs:
            continue
        if re.match(r"^#{1,6}\s+", st) or re.match(r"^\s*!\[[^\]]*\]\([^)]+\)", st):
            continue
        stray_citation_dollar = any(
            line[: match.start()].count("$") % 2 == 0
            for match in _STRAY_INLINE_CITATION_DOLLAR_RE.finditer(line)
        )
        if (
            stray_citation_dollar
            or _STRAY_INLINE_CDOT_RE.search(line)
            or _STRAY_INLINE_UNCLOSED_SENTENCE_RE.search(line)
        ):
            return True
        probe = re.sub(r"\$[^$\n]*\$", " ", line)
        if _STRAY_INLINE_LATEX_RE.search(probe) or _STRAY_INLINE_LATEX_UNIT_RE.search(probe):
            return True
    return False


def _fragmented_math_definition_likely(text: str) -> bool:
    """Detect variable-definition prose split into OCR shards outside math."""
    if not text:
        return False
    visible: list[str] = []
    in_fence = False
    in_math = False
    in_refs = False
    for raw in str(text or "").splitlines():
        st = str(raw or "").strip()
        if re.match(r"^\s*```", raw):
            in_fence = not in_fence
            continue
        if st == "$$":
            in_math = not in_math
            continue
        if REFERENCES_HEADING_RE.match(st):
            in_refs = True
            continue
        if in_fence or in_math or in_refs or not st:
            continue
        visible.append(st)

    definition_re = re.compile(r"\b(?:refers\s+to|denotes|indicates|represents)\b", re.IGNORECASE)
    for idx, line in enumerate(visible):
        probe = re.sub(r"[`*_{}$^\\]", " ", line)
        probe = re.sub(r"\s+", " ", probe).strip()
        if not re.fullmatch(r"where\s+[A-Za-z](?:\s+[A-Za-z0-9]){0,3}", probe, flags=re.IGNORECASE):
            continue
        following = " ".join(visible[idx + 1 : idx + 7])
        if definition_re.search(following):
            return True
    return False


def _out_of_order_numbered_sections_likely(text: str) -> bool:
    sections: list[int] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if REFERENCES_HEADING_RE.match(stripped):
            break
        match = re.match(r"^##\s+(\d{1,2})(?:\.\d+)*[.)]?\s+\S", stripped)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= 12:
            sections.append(number)
    if len(sections) < 2:
        return False
    highest = sections[0]
    for number in sections[1:]:
        if number < highest:
            return True
        highest = max(highest, number)
    return False


def _demote_malformed_numbered_formula_headings(md: str) -> tuple[str, bool]:
    """Demote source artifacts that unmistakably contain equation syntax."""

    text = str(md or "")
    lines = text.splitlines()
    out: list[str] = []
    highest = 0
    changed = False
    for index, line in enumerate(lines):
        match = re.match(r"^##\s+(\d{1,2})\s+#+\s*$", line.strip())
        if not match:
            formula_heading = re.match(r"^##\s+(\d{1,2})\s+(.+?)\s*$", line.strip())
            if formula_heading:
                formula_title = str(formula_heading.group(2) or "").strip()
                next_nonempty = next(
                    (
                        str(candidate or "").strip()
                        for candidate in lines[index + 1 : index + 5]
                        if str(candidate or "").strip()
                    ),
                    "",
                )
                if (
                    re.search(r"\b(?:[A-Za-z]*loss|mse|mae)\s*=", formula_title, re.IGNORECASE)
                    and next_nonempty == "$$"
                ):
                    out.append(formula_title)
                    changed = True
                    continue
            structural = re.match(r"^##\s+(\d{1,2})(?:\.\d+)*[.)]?\s+\S", line.strip())
            if structural:
                highest = max(highest, int(structural.group(1)))
            out.append(line)
            continue
        number = int(match.group(1))
        # A literal trailing ``#`` is not a valid numbered section title. If
        # the same/later section has already appeared, this is an equation
        # shard regardless of whether another OCR glyph precedes the ``$$``.
        if highest >= number:
            out.append(str(match.group(1)))
            changed = True
            continue
        out.append(line)
    fixed = "\n".join(out)
    if text.endswith("\n"):
        fixed += "\n"
    return fixed, changed and fixed != text


def _numbered_heading_title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = re.sub(r"[`*_{}\[\]()]", " ", normalized)
    return re.sub(r"[^0-9a-z]+", "", normalized)


def _restore_numbered_headings_from_pdf_text(
    md_text: str,
    md_path: Path,
    source_pdf_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Restore subsection numbers only when the same PDF page proves them.

    Vision conversion can drop the leading parent section in a two-column
    heading (``2.3`` becomes ``3``), which then resembles displaced content.
    Title equality, page-marker equality, and a unique source match make this
    repair deterministic; ambiguous or top-level-only headings are untouched.
    """

    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if fitz is None or pdf_path is None or not pdf_path.is_file():
        return text, False

    source_by_page: dict[int, dict[str, list[str]]] = {}
    source_heading_re = re.compile(
        r"^\s*(\d{1,2}(?:\.\d+)+)\.?\s+(.{3,180}?)\s*$"
    )
    try:
        doc = fitz.open(str(pdf_path))
        try:
            for page_index in range(int(doc.page_count)):
                candidates: dict[str, list[str]] = {}
                for raw_line in str(doc.load_page(page_index).get_text("text") or "").splitlines():
                    match = source_heading_re.match(raw_line)
                    if not match:
                        continue
                    number = str(match.group(1) or "").strip(".")
                    title_key = _numbered_heading_title_key(match.group(2))
                    if len(title_key) < 8:
                        continue
                    candidates.setdefault(title_key, []).append(number)
                if candidates:
                    source_by_page[page_index + 1] = candidates
        finally:
            doc.close()
    except Exception:
        return text, False

    heading_re = re.compile(
        r"^(#{2,6})\s+(\d{1,2}(?:\.\d+)*)\.?\s+(.+?)\s*$"
    )
    current_page = 0
    changed = False
    out: list[str] = []
    for line in text.splitlines():
        marker = PAGE_MARKER_RE.fullmatch(str(line or "").strip())
        if marker:
            current_page = int(marker.group(1))
            out.append(line)
            continue
        match = heading_re.match(str(line or ""))
        if not match or current_page <= 0:
            out.append(line)
            continue
        title = str(match.group(3) or "").strip()
        title_key = _numbered_heading_title_key(title)
        source_numbers = source_by_page.get(current_page, {}).get(title_key, [])
        if len(source_numbers) != 1:
            out.append(line)
            continue
        source_number = source_numbers[0]
        source_depth = len(source_number.split("."))
        desired_marks = "#" * min(6, source_depth + 1)
        repaired = f"{desired_marks} {source_number}. {title}"
        out.append(repaired)
        changed = changed or repaired != line
    fixed = "\n".join(out)
    if text.endswith("\n"):
        fixed += "\n"
    return fixed, changed and fixed != text


def _demote_source_proven_nonheading_numbered_headings(
    md_text: str,
    md_path: Path,
    source_pdf_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Demote out-of-sequence H2 artifacts only when PDF typography proves it.

    Vision conversion occasionally promotes a footnote, equation label, or
    table cell such as ``4 As detailed ...`` to a top-level section.  Sequence
    order alone is not sufficient evidence because a converted page can itself
    be displaced.  This repair therefore also requires an exact same-page text
    match whose font is materially smaller than the document's real H2 text.
    """

    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if fitz is None or pdf_path is None or not pdf_path.is_file():
        return text, False

    lines = text.splitlines()
    current_page = 0
    candidates: list[dict[str, Any]] = []
    heading_re = re.compile(r"^##\s+(\d{1,4})[.)]?\s+(.+?)\s*$")
    for line_index, line in enumerate(lines):
        stripped = str(line or "").strip()
        marker = PAGE_MARKER_RE.fullmatch(stripped)
        if marker:
            current_page = int(marker.group(1))
            continue
        if REFERENCES_HEADING_RE.match(stripped):
            break
        match = heading_re.match(stripped)
        if not match or current_page <= 0:
            continue
        title = str(match.group(2) or "").strip()
        title_key = _numbered_heading_title_key(title)
        if len(title_key) < 3:
            continue
        candidates.append(
            {
                "line_index": line_index,
                "page": current_page,
                "number": int(match.group(1)),
                "title": title,
                "title_key": title_key,
            }
        )
    if len(candidates) < 3:
        return text, False

    numbers = [int(item["number"]) for item in candidates]
    violating_indices: set[int] = set()
    violating_indices.update(
        index for index, number in enumerate(numbers) if number > 12
    )
    highest = numbers[0]
    for index, number in enumerate(numbers[1:], start=1):
        if number < highest:
            violating_indices.add(index)
        highest = max(highest, number)
    lowest = numbers[-1]
    for index in range(len(numbers) - 2, -1, -1):
        number = numbers[index]
        if number > lowest:
            violating_indices.add(index)
        lowest = min(lowest, number)
    if not violating_indices:
        return text, False

    def _matching_line_size(page: Any, title_key: str) -> float:
        best = 0.0
        try:
            blocks = list((page.get_text("dict") or {}).get("blocks") or [])
        except Exception:
            return 0.0
        for block in blocks:
            for source_line in list(block.get("lines") or []):
                spans = list(source_line.get("spans") or [])
                source_text = "".join(str(span.get("text") or "") for span in spans).strip()
                source_match = re.match(r"^\s*\d{1,4}[.)]?\s*(.+?)\s*$", source_text)
                source_title = source_match.group(1) if source_match else source_text
                source_key = _numbered_heading_title_key(source_title)
                shared = min(len(source_key), len(title_key))
                if shared < 5:
                    continue
                if not (source_key.startswith(title_key[:shared]) or title_key.startswith(source_key[:shared])):
                    continue
                best = max(best, max((float(span.get("size") or 0.0) for span in spans), default=0.0))
        return best

    matched_sizes: list[float] = []
    try:
        doc = fitz.open(str(pdf_path))
        try:
            for item in candidates:
                page_no = int(item["page"])
                if not 1 <= page_no <= int(doc.page_count):
                    item["source_font_size"] = 0.0
                    continue
                size = _matching_line_size(doc.load_page(page_no - 1), str(item["title_key"]))
                item["source_font_size"] = size
                if size > 0:
                    matched_sizes.append(size)
        finally:
            doc.close()
    except Exception:
        return text, False
    if not matched_sizes:
        return text, False

    reference_size = max(matched_sizes)
    if reference_size < 7.0:
        return text, False
    changed = False
    for index in sorted(violating_indices):
        item = candidates[index]
        source_size = float(item.get("source_font_size") or 0.0)
        if source_size <= 0 or source_size > reference_size - 1.25:
            continue
        line_index = int(item["line_index"])
        lines[line_index] = re.sub(r"^##\s+", "", lines[line_index], count=1)
        changed = True
    fixed = "\n".join(lines)
    if text.endswith("\n"):
        fixed += "\n"
    return fixed, changed and fixed != text


_NUMERIC_ONLY_HEADING_RE = re.compile(
    r"^#{1,6}\s+(?:\d+(?:\.\d+)?\s*){2,}$",
    re.MULTILINE,
)
_PLAIN_METHOD_SUBHEADING_TITLES = {
    "deep feature fusion",
    "image reconstruction",
    "loss function",
    "network structure",
    "network training",
    "noise modeling of spad arrays",
    "noise parameter calibration",
    "shallow feature extraction",
}


def _plain_method_subheading_count(text: str) -> int:
    lines = str(text or "").splitlines()
    in_methods = False
    count = 0
    for line in lines:
        stripped = str(line or "").strip()
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
        if heading:
            title = str(heading.group(1) or "").strip().lower()
            if re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?methods?\b", title):
                in_methods = True
                continue
            if in_methods and re.match(
                r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:references|discussion|conclusions?)\b",
                title,
            ):
                in_methods = False
            continue
        if in_methods and stripped.lower() in _PLAIN_METHOD_SUBHEADING_TITLES:
            count += 1
    return count


def _issue_codes_from_context(
    md_path: Path,
    text: str,
    metrics: dict[str, Any],
    *,
    source_quality: dict[str, Any] | None = None,
) -> list[str]:
    quality = source_quality if isinstance(source_quality, dict) else _source_quality_view(md_path, text, metrics)
    codes = _issue_codes_from_metrics(metrics)
    if "mojibake" in codes and contains_only_detached_accent_mojibake(text):
        codes = ["detached_accents" if code == "mojibake" else code for code in codes]
    if bool(quality.get("abstract_not_applicable")):
        codes = [code for code in codes if code not in {"missing_abstract", "missing_references", "weak_structure"}]
    elif "missing_abstract" in codes and not bool(quality.get("abstract_autofix_likely")):
        codes = [code for code in codes if code != "missing_abstract"]
    if bool(quality.get("source_text_loss")):
        codes.insert(0, "source_text_loss")
        codes = [code for code in codes if code not in {"missing_abstract", "weak_structure"}]
    if int(quality.get("missing_source_page_count") or 0) > 0:
        codes.append("missing_source_pages")
    if int(quality.get("source_page_text_corruption_count") or 0) > 0:
        codes.append("source_page_text_corruption")
    if int(quality.get("source_page_prose_omission_count") or 0) > 0:
        codes.append("source_page_prose_omission")
    if int(quality.get("source_page_anchor_issue_count") or 0) > 0:
        codes.append("source_page_marker_alignment")
    if (
        int(quality.get("missing_pdf_page_marker_count") or 0) > 0
        or int(quality.get("duplicate_pdf_page_marker_count") or 0) > 0
    ):
        codes.append("source_page_marker_alignment")
    if (
        int(quality.get("page_marker_shortfall") or 0) >= 1
        or int(quality.get("out_of_range_page_marker_count") or 0) > 0
    ):
        codes.append("source_page_count_mismatch")
    if int(quality.get("source_table_page_anchor_issue_count") or 0) > 0:
        codes.append("source_table_page_alignment")
    if bool(quality.get("reference_index_truncated")):
        codes.append("reference_index_truncated")
    if bool(quality.get("references_before_body")):
        codes.append("references_before_body")
    if _collapsed_heading_hierarchy_likely(text, metrics, quality):
        codes.append("collapsed_heading_hierarchy")
    if _stray_inline_math_likely(text):
        codes.append("stray_inline_math")
    if _fragmented_math_definition_likely(text):
        codes.append("conversion_retry_math_text")
    if _out_of_order_numbered_sections_likely(text):
        codes.append("out_of_order_sections")
    if _NUMERIC_ONLY_HEADING_RE.search(text):
        codes.append("numeric_only_headings")
    if _plain_method_subheading_count(text) >= 2:
        codes.append("plain_method_subheadings")
    return _dedupe_codes(codes)


def _ensure_page_anchor(md: str) -> tuple[str, bool]:
    text = str(md or "")
    if PAGE_MARKER_RE.search(text):
        return text, False
    stripped = text.lstrip()
    prefix_len = len(text) - len(stripped)
    fixed = f"{text[:prefix_len]}<!-- kb_page: 1 -->\n\n{stripped}" if stripped else "<!-- kb_page: 1 -->"
    return fixed, fixed != text


def _normalize_page_marker_sequence(md: str) -> tuple[str, bool]:
    text = str(md or "")
    matches = list(PAGE_MARKER_RE.finditer(text))
    if len(matches) <= 1:
        return text, False
    removable_spans: list[tuple[int, int]] = []
    for previous, current in zip(matches, matches[1:]):
        if int(previous.group(1)) != int(current.group(1)):
            continue
        if text[int(previous.end()) : int(current.start())].strip():
            continue
        removable_spans.append((int(current.start()), int(current.end())))
    if not removable_spans:
        return text, False
    fixed = text
    for start, end in reversed(removable_spans):
        fixed = fixed[:start] + fixed[end:]
    return fixed, fixed != text


def _balance_display_math(md: str) -> tuple[str, bool]:
    lines = str(md or "").splitlines()
    out: list[str] = []
    in_math = False
    changed = False
    structural_re = re.compile(r"^\s*(?:#{1,6}\s+\S.*|!\[[^\]]*]\([^)]+\)|\|.+\|)\s*$")
    for line in lines:
        if DISPLAY_MATH_DELIMITER_RE.match(line or ""):
            in_math = not in_math
            out.append(line)
            continue
        if in_math and structural_re.match(line or ""):
            out.append("$$")
            changed = True
            in_math = False
        out.append(line)
    if in_math:
        out.append("$$")
        changed = True
    fixed = "\n".join(out)
    return fixed, changed and fixed != str(md or "")


def _unwrap_prose_dominant_display_math(md: str) -> tuple[str, bool]:
    """Remove display delimiters only from blocks classified as prose-heavy.

    The block body is preserved exactly so mixed prose/equation evidence is
    never discarded. Genuine equation-only blocks retain their delimiters.
    """

    original = str(md or "")
    lines = original.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        if not DISPLAY_MATH_DELIMITER_RE.match(lines[index]):
            out.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not DISPLAY_MATH_DELIMITER_RE.match(lines[end]):
            end += 1
        if end >= len(lines):
            out.extend(lines[index:])
            break
        block_lines = lines[index + 1 : end]
        if is_prose_dominant_display_math_block("\n".join(block_lines)):
            out.extend(block_lines)
        else:
            out.extend(lines[index : end + 1])
        index = end + 1
    repaired = "\n".join(out)
    if original.endswith("\n"):
        repaired += "\n"
    return repaired, repaired != original


def _asset_name_from_image_target(target: str) -> str:
    raw = str(target or "").strip().strip("<>")
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    if not raw:
        return ""
    return Path(unquote(raw.replace("\\", "/"))).name


def _normalize_caption_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    if not CAPTION_LINE_RE.match(text):
        return ""
    if re.fullmatch(r"(?i)(?:fig(?:ure)?\.?|table|algorithm)\s*\d*[A-Za-z]?", text):
        return ""
    if len(text.split()) < 4 and len(text) < 32:
        return ""
    return text


def _format_caption_line(caption: str) -> str:
    text = _normalize_caption_text(caption)
    if not text:
        return ""
    m = re.match(r"(?i)^(fig(?:ure)?\.?|table|algorithm)\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?)\.?\s*(.*)$", text)
    if not m:
        return f"*{text}*"
    kind_raw = m.group(1).lower()
    kind = "Figure" if kind_raw.startswith("fig") else ("Table" if kind_raw.startswith("table") else "Algorithm")
    ident = m.group(2)
    tail = (m.group(3) or "").strip()
    prefix = f"**{kind} {ident}.**"
    return f"{prefix} {tail}".rstrip()


def _load_figure_metadata_captions(md_path: Path) -> dict[str, str]:
    assets_dir = md_path.parent / "assets"
    if not assets_dir.exists():
        return {}
    out: dict[str, str] = {}

    def add(asset_name: str, caption: Any) -> None:
        name = str(asset_name or "").strip()
        line = _format_caption_line(str(caption or ""))
        if name and line and name not in out:
            out[name] = line

    for path in [assets_dir / "figure_index.json", *sorted(assets_dir.glob("page_*_fig_index.json"))]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        figures = payload.get("figures") if isinstance(payload, dict) else None
        if not isinstance(figures, list):
            continue
        for item in figures:
            if not isinstance(item, dict):
                continue
            add(str(item.get("asset_name") or item.get("asset_name_raw") or item.get("asset_name_alias") or ""), item.get("caption"))

    for path in sorted(assets_dir.glob("*.meta.json"))[:500]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        add(str(payload.get("asset_name") or payload.get("asset_name_raw") or f"{path.stem}.png"), payload.get("caption"))

    return out


def _has_caption_nearby(lines: list[str], image_idx: int) -> bool:
    for idx in range(image_idx + 1, min(len(lines), image_idx + 5)):
        st = (lines[idx] or "").strip()
        if not st:
            continue
        return bool(CAPTION_LINE_RE.match(st))
    return False


def _inject_figure_metadata_captions(md_path: Path, md: str) -> tuple[str, bool]:
    captions = _load_figure_metadata_captions(md_path)
    if not captions:
        return md, False
    lines = str(md or "").splitlines()
    out: list[str] = []
    changed = False
    for idx, line in enumerate(lines):
        out.append(line)
        m = IMAGE_LINE_RE.match(line or "")
        if not m or _has_caption_nearby(lines, idx):
            continue
        asset_name = _asset_name_from_image_target(m.group(3) or "")
        caption_line = captions.get(asset_name) or _format_caption_line(m.group(2) or "")
        if not caption_line:
            continue
        out.append("")
        out.append(caption_line)
        changed = True
    return "\n".join(out), changed


def _asset_page_and_fig(asset_name: str) -> tuple[int, int]:
    m = re.search(r"page_(\d+)_fig_(\d+)", str(asset_name or ""), flags=re.IGNORECASE)
    if not m:
        return (0, 0)
    try:
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return (0, 0)


def _caption_fig_no_from_text(text: str) -> int:
    m = re.match(r"^\s*(?:\*{1,2}\s*)?fig(?:ure)?\.?\s*S?(\d{1,4})\b", str(text or ""), flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _pdf_page_caption_candidates(page: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        text = _normalize_caption_text(raw)
        if not text:
            return
        key = re.sub(r"\s+", " ", text).strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append({"fig_no": _caption_fig_no_from_text(text), "caption": text})

    try:
        data = page.get_text("dict") or {}
        for block in data.get("blocks", []) or []:
            parts: list[str] = []
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                raw = "".join(str(span.get("text", "")) for span in spans)
                if raw.strip():
                    parts.append(raw.strip())
            if parts:
                add(" ".join(parts))
    except Exception:
        pass

    try:
        for raw in str(page.get_text("text") or "").splitlines():
            add(raw)
    except Exception:
        pass
    return out


def _load_pdf_text_captions_by_page(source_pdf_path: Path | str | None) -> dict[int, list[dict[str, Any]]]:
    if fitz is None or not source_pdf_path:
        return {}
    path = Path(source_pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return {}
    except Exception:
        return {}
    out: dict[int, list[dict[str, Any]]] = {}
    try:
        doc = fitz.open(str(path))
    except Exception:
        return {}
    try:
        for page_index in range(len(doc)):
            try:
                candidates = _pdf_page_caption_candidates(doc.load_page(page_index))
            except Exception:
                candidates = []
            if candidates:
                out[page_index + 1] = candidates
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


def _choose_pdf_caption_for_image(
    *,
    asset_name: str,
    alt_text: str,
    captions_by_page: dict[int, list[dict[str, Any]]],
) -> str:
    page_no, page_fig_index = _asset_page_and_fig(asset_name)
    if page_no <= 0:
        return ""
    candidates = list(captions_by_page.get(page_no) or [])
    if not candidates:
        return ""
    alt_fig_no = _caption_fig_no_from_text(alt_text)
    if alt_fig_no > 0:
        for item in candidates:
            try:
                if int(item.get("fig_no") or 0) == alt_fig_no:
                    return _format_caption_line(str(item.get("caption") or ""))
            except Exception:
                continue
    numbered = [item for item in candidates if int(item.get("fig_no") or 0) > 0]
    if len(numbered) == 1:
        return _format_caption_line(str(numbered[0].get("caption") or ""))
    if page_fig_index > 0 and page_fig_index <= len(numbered):
        return _format_caption_line(str(numbered[page_fig_index - 1].get("caption") or ""))
    if len(candidates) == 1:
        return _format_caption_line(str(candidates[0].get("caption") or ""))
    return ""


def _inject_pdf_text_captions(md_path: Path, md: str, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    captions_by_page = _load_pdf_text_captions_by_page(pdf_path)
    if not captions_by_page:
        return md, False
    lines = str(md or "").splitlines()
    out: list[str] = []
    changed = False
    for idx, line in enumerate(lines):
        out.append(line)
        m = IMAGE_LINE_RE.match(line or "")
        if not m or _has_caption_nearby(lines, idx):
            continue
        asset_name = _asset_name_from_image_target(m.group(3) or "")
        caption_line = _choose_pdf_caption_for_image(
            asset_name=asset_name,
            alt_text=m.group(2) or "",
            captions_by_page=captions_by_page,
        )
        if not caption_line:
            continue
        out.append("")
        out.append(caption_line)
        changed = True
    return "\n".join(out), changed


def _markdown_image_target_path(md_path: Path, target: str) -> Path | None:
    raw = str(target or "").strip().strip("<>")
    if not raw or raw.startswith(("http://", "https://", "data:", "#")):
        return None
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    if not raw:
        return None
    decoded = unquote(raw.replace("\\", "/"))
    p = Path(decoded)
    return p if p.is_absolute() else (md_path.parent / p)


def _target_page_hint(target: str) -> int:
    raw = str(target or "")
    patterns = [
        r"(?:^|[_\-/])p0*(\d{1,5})(?:[_\-.]|$)",
        r"(?:^|[_\-/])page[_-]?0*(\d{1,5})(?:[_\-.]|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _target_region_hint(target: str) -> int:
    raw = str(target or "")
    for pattern in [r"(?:^|[_\-/])r0*(\d{1,4})(?:[_\-.]|$)", r"(?:^|[_\-/])fig0*(\d{1,4})(?:[_\-.]|$)"]:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _image_assets_by_page(assets_dir: Path) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = {}
    try:
        children = sorted(assets_dir.iterdir())
    except Exception:
        return out
    for child in children:
        if not child.is_file() or child.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        page_no, _fig_no = _asset_page_and_fig(child.name)
        if page_no <= 0:
            continue
        out.setdefault(page_no, []).append(child)
    return out


def _next_caption_fig_no(lines: list[str], image_idx: int) -> int:
    for idx in range(image_idx + 1, min(len(lines), image_idx + 5)):
        st = (lines[idx] or "").strip()
        if not st:
            continue
        if CAPTION_LINE_RE.match(st):
            return _caption_fig_no_from_text(st)
        return 0
    return 0


def _format_asset_target_like(original_target: str, asset: Path, assets_dir: Path) -> str:
    raw = str(original_target or "").strip()
    normalized = raw.replace("\\", "/")
    name = asset.name
    if normalized.startswith("./assets/"):
        return f"./assets/{name}"
    if normalized.startswith("assets/"):
        return f"assets/{name}"
    try:
        rel = asset.relative_to(assets_dir.parent).as_posix()
        return f"./{rel}" if normalized.startswith("./") else rel
    except Exception:
        return f"./assets/{name}"


def _choose_replacement_image_asset(
    *,
    candidates: list[Path],
    target: str,
    alt_text: str,
    nearby_caption_fig_no: int,
) -> Path | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    wanted_fig = _caption_fig_no_from_text(alt_text) or int(nearby_caption_fig_no or 0)
    if wanted_fig > 0:
        matches = [p for p in candidates if _asset_page_and_fig(p.name)[1] == wanted_fig]
        if len(matches) == 1:
            return matches[0]

    region = _target_region_hint(target)
    if region > 0:
        matches = [p for p in candidates if _asset_page_and_fig(p.name)[1] == region]
        if len(matches) == 1:
            return matches[0]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _repair_missing_image_links(md_path: Path, md: str) -> tuple[str, bool]:
    assets_dir = md_path.parent / "assets"
    if not assets_dir.exists():
        return md, False
    by_page = _image_assets_by_page(assets_dir)
    if not by_page:
        return md, False

    lines = str(md or "").splitlines()
    out: list[str] = []
    current_page = 0
    changed = False
    for idx, line in enumerate(lines):
        page_match = PAGE_MARKER_RE.search(line or "")
        if page_match:
            try:
                current_page = int(page_match.group(1))
            except Exception:
                current_page = 0

        m = IMAGE_LINE_RE.match(line or "")
        if not m:
            out.append(line)
            continue

        target = m.group(3) or ""
        target_path = _markdown_image_target_path(md_path, target)
        if target_path is None or target_path.exists():
            out.append(line)
            continue

        page_no = _target_page_hint(target) or current_page
        replacement = _choose_replacement_image_asset(
            candidates=list(by_page.get(page_no) or []),
            target=target,
            alt_text=m.group(2) or "",
            nearby_caption_fig_no=_next_caption_fig_no(lines, idx),
        )
        if replacement is None:
            out.append(line)
            continue

        new_target = _format_asset_target_like(target, replacement, assets_dir)
        out.append(f"{m.group(1) or ''}![{m.group(2) or ''}]({new_target})")
        changed = True

    fixed = "\n".join(out)
    return fixed, changed and fixed != str(md or "")


def _looks_like_abstract_candidate_line(text: str) -> bool:
    st = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(st) < 140:
        return False
    if st.startswith(("![", "|", "<!--")) or re.match(r"^#{1,6}\s+", st):
        return False
    low = st.lower()
    if any(
        needle in low
        for needle in [
            "contents lists available",
            "article info",
            "keywords:",
            "ocis codes",
            "doi:",
            "copyright",
            "creative commons",
            "published by",
            "received:",
            "accepted:",
        ]
    ):
        return False
    words = re.findall(r"[A-Za-z]{3,}", st)
    if len(words) < 22:
        return False
    signal_hits = len(
        re.findall(
            r"\b(?:we|this|our|paper|study|work|approach|method|results?|demonstrate|"
            r"propose|show|capable|promise|however)\b",
            low,
        )
    )
    return signal_hits >= 1 or len(words) >= 35


def _insert_abstract_heading_only(md: str) -> tuple[str, bool]:
    text = str(md or "")
    if re.search(r"(?mi)^#{1,6}\s+Abstract\s*$", text):
        return text, False
    lines = text.splitlines()
    out: list[str] = []
    changed = False
    spaced_re = re.compile(r"^(?:#*\s*)?A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*$", re.IGNORECASE)
    inline_re = re.compile(r"^\s*(?:\*{1,2})?\s*Abstract\s*(?:\*{1,2})?\s*:?\s*(.*)$", re.IGNORECASE)

    for line in lines:
        st = (line or "").strip()
        if spaced_re.match(st):
            out.append("## Abstract")
            changed = True
            continue
        m_inline = inline_re.match(st)
        if m_inline and not re.match(r"^#{1,6}\s+", st):
            tail = (m_inline.group(1) or "").strip()
            out.append("## Abstract")
            if tail:
                out.extend(["", tail])
            changed = True
            continue
        out.append(line)
    if changed:
        return "\n".join(out), True

    first_section_idx = len(lines)
    section_re = re.compile(
        r"^#{1,6}\s+(?:"
        r"\d+(?:\.\d+)*\.?\s+\S+|[IVXLC]+\.\s+\S+|"
        r"(?:introduction|background|related\s+work|results?|method(?:s|ology)?|discussion|"
        r"conclusions?|references|bibliography|structure\b|article\b))",
        re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        if section_re.match((line or "").strip()):
            first_section_idx = idx
            break

    candidate_idx = -1
    for idx in range(0, first_section_idx):
        if _looks_like_abstract_candidate_line((lines[idx] or "").strip()):
            candidate_idx = idx
            break
    if candidate_idx < 0:
        return text, False

    insert = ["## Abstract", ""]
    if candidate_idx > 0 and (lines[candidate_idx - 1] or "").strip():
        insert.insert(0, "")
    fixed_lines = list(lines)
    fixed_lines[candidate_idx:candidate_idx] = insert
    return "\n".join(fixed_lines), True


def _pdf_source_stats(source_pdf_path: Path | str | None) -> dict[str, Any]:
    if fitz is None or not source_pdf_path:
        return {"available": False, "page_count": 0, "text_chars": 0}
    path = Path(source_pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return {"available": False, "page_count": 0, "text_chars": 0, "path": str(path)}
    except Exception:
        return {"available": False, "page_count": 0, "text_chars": 0, "path": str(path)}
    text_chars = 0
    page_count = 0
    try:
        doc = fitz.open(str(path))
    except Exception:
        return {"available": False, "page_count": 0, "text_chars": 0, "path": str(path)}
    try:
        page_count = len(doc)
        for page_index in range(page_count):
            try:
                text_chars += len(str(doc.load_page(page_index).get_text("text") or "").strip())
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return {"available": True, "page_count": int(page_count), "text_chars": int(text_chars), "path": str(path)}


def _document_profile(md_path: Path, text: str) -> dict[str, Any]:
    name = f"{md_path.parent.name} {md_path.name}".lower()
    file_name = md_path.name.lower()
    front = re.sub(r"\s+", " ", str(text or "")[:5000]).lower()
    file_title_hint = re.sub(r"^.*?\b(?:19|20)\d{2}\b[\s._-]*", "", file_name).strip()
    is_supplement = bool(
        re.search(r"\b(?:supplement|supplementary|supplemental)\b", name)
        or re.search(r"\b(?:supplement|supplementary|supplemental)\s+(?:document|material|information|doi)\b", front)
        or "parent article doi" in front
    )
    is_review = bool(
        re.search(r"\b(?:review|survey)\b", file_title_hint)
        or re.search(r"\b(?:review article|survey article|systematic review|brief review|comprehensive review)\b", front)
    )
    doc_type = "supplementary" if is_supplement else ("review" if is_review else "research_article")
    return {
        "document_type": doc_type,
        "abstract_required": not is_supplement,
        "abstract_not_applicable": bool(is_supplement),
    }


def _reference_layout(text: str) -> dict[str, Any]:
    lines = str(text or "").splitlines()
    ref_idx = -1
    for idx, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match((line or "").strip()):
            ref_idx = idx
            break
    if ref_idx < 0:
        return {
            "references_line": -1,
            "references_char": -1,
            "references_char_ratio": 1.0,
            "body_heading_after_references_line": -1,
            "reference_line_count_before_body": 0,
            "references_before_body": False,
        }
    char_pos = len("\n".join(lines[:ref_idx]))
    char_ratio = float(char_pos / max(1, len(str(text or ""))))
    ref_count = 0
    body_after = -1
    for idx in range(ref_idx + 1, len(lines)):
        st = (lines[idx] or "").strip()
        if re.match(r"^\[\d{1,4}\]\s+", st):
            ref_count += 1
            continue
        if BODY_SECTION_HEADING_RE.match(st):
            body_after = idx
            break
    # A normal main-body heading after three or more bibliography entries is a
    # structural ordering error regardless of where the first References
    # heading happens to fall.  The old 20% position threshold missed long
    # reviews whose bibliography was accidentally inserted around mid-file.
    before_body = bool(ref_idx >= 0 and body_after > ref_idx and ref_count >= 3)
    return {
        "references_line": int(ref_idx + 1),
        "references_char": int(char_pos),
        "references_char_ratio": round(char_ratio, 4),
        "body_heading_after_references_line": int(body_after + 1) if body_after >= 0 else -1,
        "reference_line_count_before_body": int(ref_count),
        "references_before_body": before_body,
    }


def _source_text_loss_likely(text: str, metrics: dict[str, Any], pdf_stats: dict[str, Any], ref_layout: dict[str, Any]) -> bool:
    if not bool(pdf_stats.get("available")):
        return False
    pdf_pages = int(pdf_stats.get("page_count") or 0)
    pdf_text_chars = int(pdf_stats.get("text_chars") or 0)
    md_chars = int(metrics.get("chars") or len(str(text or "")))
    ref_lines = int(metrics.get("reference_line_count") or 0)
    non_ref_body_chars = md_chars
    ref_pos = int(ref_layout.get("references_char") or -1)
    if ref_pos >= 0:
        non_ref_body_chars = max(0, ref_pos)
    has_body_after_refs = int(ref_layout.get("body_heading_after_references_line") or -1) > 0
    if pdf_pages >= 4 and pdf_text_chars >= 5000 and md_chars < int(pdf_text_chars * 0.22):
        return True
    if pdf_pages >= 4 and ref_lines >= 5 and non_ref_body_chars < max(900, pdf_pages * 180) and not has_body_after_refs:
        return True
    if pdf_pages >= 6 and int(metrics.get("heading_count") or 0) <= 1 and ref_lines >= 8:
        return True
    return False


def _page_alignment_quality(metrics: dict[str, Any], pdf_stats: dict[str, Any], text: str = "") -> dict[str, Any]:
    pdf_pages = int(pdf_stats.get("page_count") or 0)
    markers = int(metrics.get("page_marker_count") or 0)
    marker_numbers = [
        int(match.group(1))
        for match in PAGE_MARKER_RE.finditer(str(text or ""))
        if str(match.group(1) or "").isdigit()
    ]
    valid_marker_numbers = [number for number in marker_numbers if pdf_pages <= 0 or 1 <= number <= pdf_pages]
    out_of_range_markers = sorted({number for number in marker_numbers if pdf_pages > 0 and not 1 <= number <= pdf_pages})
    valid_marker_set = set(valid_marker_numbers)
    valid_markers = len(valid_marker_set)
    missing_markers = [number for number in range(1, pdf_pages + 1) if number not in valid_marker_set]
    duplicate_markers = sorted(
        number for number in valid_marker_set if valid_marker_numbers.count(number) > 1
    )
    max_marker = max(valid_marker_set, default=0)
    ratio = float(valid_markers / max(1, pdf_pages)) if pdf_pages > 0 else 0.0
    if pdf_pages <= 0:
        confidence = "unknown"
    elif valid_markers <= 0:
        confidence = "missing"
    elif ratio >= 0.8:
        confidence = "high"
    elif ratio >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "pdf_pages": int(pdf_pages),
        "page_marker_count": int(markers),
        "valid_page_marker_count": int(valid_markers),
        "valid_page_marker_occurrence_count": int(len(valid_marker_numbers)),
        "out_of_range_page_markers": out_of_range_markers,
        "out_of_range_page_marker_count": int(len(out_of_range_markers)),
        "missing_pdf_page_markers": missing_markers,
        "missing_pdf_page_marker_count": int(len(missing_markers)),
        "duplicate_pdf_page_markers": duplicate_markers,
        "duplicate_pdf_page_marker_count": int(len(duplicate_markers)),
        # A sparse-but-correct anchor set (for example pages 1 and 4 of a
        # four-page PDF) can be safely realigned. Only a missing terminal page
        # range proves that conversion stopped before the source PDF ended.
        "page_marker_shortfall": max(0, int(pdf_pages - max_marker)),
        "page_marker_count_shortfall": int(len(missing_markers)),
        "max_page_marker": int(max_marker),
        "matched_page_ratio": round(ratio, 4) if pdf_pages > 0 else 0.0,
        "page_alignment_confidence": confidence,
    }


def _rare_source_tokens(text: str) -> set[str]:
    return {
        tok
        for tok, _ in _word_tokens_with_offsets(text)
        if len(tok) >= 5 and tok not in PAGE_ALIGNMENT_STOP_WORDS and not tok.isdigit()
    }


def _page_marker_occurrences(text: str) -> list[dict[str, int]]:
    matches = list(PAGE_MARKER_RE.finditer(str(text or "")))
    out: list[dict[str, int]] = []
    for idx, match in enumerate(matches):
        try:
            page_no = int(match.group(1))
        except Exception:
            continue
        segment_end = int(matches[idx + 1].start()) if idx + 1 < len(matches) else len(str(text or ""))
        out.append(
            {
                "page": int(page_no),
                "start": int(match.start()),
                "end": int(match.end()),
                "segment_start": int(match.end()),
                "segment_end": int(segment_end),
            }
        )
    return out


_SOURCE_WRAPPED_WORD_RE = re.compile(
    rf"([A-Za-z]{{2,{SOURCE_PAGE_MAX_WRAP_PREFIX_CHARS}}})-\s*\n\s*"
    rf"([a-z]{{2,{SOURCE_PAGE_MAX_WRAP_SUFFIX_CHARS}}})"
)


def _source_page_wrapped_word_damage(page_text: str, local_segment: str) -> dict[str, Any]:
    """Find PDF line-wrap words whose leading half disappeared in Markdown.

    Two-column PDFs commonly expose words as ``algo-\nrithms`` in their text
    layer. A healthy conversion emits ``algorithms`` (or at least keeps the two
    adjacent halves). A damaged column merge can instead leave only ``rithms``.
    Comparing against the source page makes this signal precise without using a
    language dictionary or penalizing legitimate technical vocabulary.
    """
    source = str(page_text or "")
    local = str(local_segment or "").lower()
    pairs: list[tuple[str, str]] = []
    for match in _SOURCE_WRAPPED_WORD_RE.finditer(source):
        prefix = str(match.group(1) or "").lower()
        suffix = str(match.group(2) or "").lower()
        if len(prefix + suffix) < 5:
            continue
        pairs.append((prefix, suffix))

    missing: list[dict[str, str]] = []
    for prefix, suffix in pairs:
        joined = re.escape(prefix + suffix)
        first = re.escape(prefix)
        second = re.escape(suffix)
        preserved = re.search(
            rf"\b{joined}\b|\b{first}\s*-\s*{second}\b|\b{first}\s+{second}\b",
            local,
            flags=re.IGNORECASE,
        )
        if preserved:
            continue
        missing.append(
            {
                "source": f"{prefix}-{suffix}",
                "expected": f"{prefix}{suffix}",
            }
        )

    pair_count = len(pairs)
    missing_count = len(missing)
    missing_ratio = float(missing_count / max(1, pair_count))
    corrupted = bool(
        pair_count >= SOURCE_PAGE_MIN_WRAPPED_WORDS
        and missing_count >= SOURCE_PAGE_MIN_WRAPPED_WORDS
        and missing_ratio >= SOURCE_PAGE_MISSING_WRAP_RATIO
    )
    return {
        "source_wrapped_word_count": int(pair_count),
        "missing_wrapped_word_count": int(missing_count),
        "missing_wrapped_word_ratio": round(missing_ratio, 4),
        "missing_wrapped_word_examples": missing[:12],
        "text_corruption": corrupted,
    }


_SOURCE_PROSE_BLOCK_SKIP_RE = re.compile(
    r"^\s*(?:"
    r"\[\s*\d{1,4}\s*\]\s+|"
    r"fig(?:ure)?\.?\s*\d|table\s*\d|algorithm\s*\d|"
    r"acknowledg(?:e)?ments?|author\s+details?|funding|"
    r"supplementary\s+information|publisher(?:'s|’s)?\s+note|open\s+access"
    r")\b",
    re.IGNORECASE,
)


def _source_prose_tokens(text: str) -> list[str]:
    return [token for token, _start, _end in _source_prose_token_spans(text)]


def _source_prose_token_spans(text: str) -> list[tuple[str, int, int]]:
    """Return comparison tokens while retaining Markdown character offsets."""

    value = str(text or "")
    out: list[tuple[str, int, int]] = []
    for match in re.finditer(
        r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})|[A-Za-z0-9]+",
        value,
    ):
        if match.group(1) and match.group(2):
            token = f"{match.group(1)}{match.group(2)}"
        else:
            token = str(match.group(0) or "")
        if token:
            out.append((token.lower(), int(match.start()), int(match.end())))
    return out


def _merge_source_prose_ligature_spans(
    spans: list[tuple[str, int, int]],
    local_vocabulary: set[str],
) -> list[tuple[str, int, int]]:
    """Join PDF ligature fragments only when the converted page proves the word."""

    merged: list[tuple[str, int, int]] = []
    token_index = 0
    while token_index < len(spans):
        token, start, end = spans[token_index]
        if token_index + 1 < len(spans):
            next_token, _next_start, next_end = spans[token_index + 1]
            joined = token + next_token
            if joined in local_vocabulary:
                merged.append((joined, start, next_end))
                token_index += 2
                continue
        merged.append((token, start, end))
        token_index += 1
    return merged


def _eligible_source_prose_block(block: str) -> bool:
    value = str(block or "").strip()
    if not value or _SOURCE_PROSE_BLOCK_SKIP_RE.match(value):
        return False
    if len(_source_prose_tokens(value)) < SOURCE_PAGE_MIN_PROSE_BLOCK_TOKENS:
        return False
    alpha_chars = sum(char.isalpha() for char in value)
    alnum_chars = sum(char.isalnum() for char in value)
    if alpha_chars / max(1, alnum_chars) < 0.90:
        return False
    return len(re.findall(r"[.!?](?:\s|$)", value)) >= SOURCE_PAGE_MIN_PROSE_SENTENCES


def _anchored_local_prose_window(
    source_tokens: list[str],
    local_tokens: list[str],
    local_positions: dict[str, list[int]],
) -> list[str]:
    """Bound paragraph comparison to its anchored region on the converted page.

    Comparing every source block with every token on a long two-column page is
    quadratic and made final quality verification dominate conversion time.
    Stable token offsets identify the same paragraph even when a short interior
    phrase is missing.  When there are too few anchors, retain the original
    whole-page comparison so the quality gate never loses recall.
    """

    if len(local_tokens) <= max(320, len(source_tokens) * 3):
        return local_tokens

    deltas: list[int] = []
    for source_index, token in enumerate(source_tokens):
        positions = local_positions.get(token) or []
        # Very common words are weak anchors and create large cross products.
        if not positions or len(positions) > 8:
            continue
        deltas.extend(local_index - source_index for local_index in positions)
    if len(deltas) < 6:
        return local_tokens

    # A small deletion shifts the suffix offsets by only a few positions. Find
    # the densest offset cluster instead of requiring one exact delta.
    sorted_deltas = sorted(deltas)
    cluster_width = max(12, min(64, len(source_tokens) // 3))
    best_start = 0
    best_end = 0
    left = 0
    for right, delta in enumerate(sorted_deltas):
        while delta - sorted_deltas[left] > cluster_width:
            left += 1
        if right - left > best_end - best_start:
            best_start, best_end = left, right
    cluster = sorted_deltas[best_start : best_end + 1]
    if len(cluster) < 6:
        return local_tokens

    margin = max(40, min(160, len(source_tokens) // 2))
    window_start = max(0, min(cluster) - margin)
    window_end = min(len(local_tokens), max(cluster) + len(source_tokens) + margin)
    if window_end - window_start < len(source_tokens):
        return local_tokens
    return local_tokens[window_start:window_end]


def _source_page_prose_omission_damage(block_texts: list[str], local_segment: str) -> dict[str, Any]:
    """Detect source words deleted from inside otherwise matching prose.

    Set-based page coverage can look healthy when a converter drops short
    phrases from a long two-column paragraph.  Compare long prose blocks in
    sequence and count only deletions bounded by stable three-token anchors on
    both sides.  This deliberately ignores captions, front matter, formulas,
    references, and wholesale rewrites so a retry is requested only for a
    high-confidence interior omission.
    """

    local_tokens = _source_prose_tokens(local_segment)
    local_vocabulary = set(local_tokens)
    if not local_tokens:
        return {
            "assessed_prose_block_count": 0,
            "anchored_omitted_word_count": 0,
            "anchored_omission_group_count": 0,
            "anchored_omission_examples": [],
            "text_omission": False,
        }
    local_positions: dict[str, list[int]] = {}
    for local_index, token in enumerate(local_tokens):
        local_positions.setdefault(token, []).append(local_index)

    assessed = 0
    omitted_words = 0
    omission_groups = 0
    examples: list[str] = []
    for raw_block in block_texts:
        block = str(raw_block or "").strip()
        if not _eligible_source_prose_block(block):
            continue
        source_tokens_raw = _source_prose_tokens(block)

        # PDF text layers often split a ligature into tokens such as
        # ``di`` + ``erent``.  Join it only when the resulting token is
        # actually present in the converted page, preventing false omissions.
        source_tokens = [
            token
            for token, _start, _end in _merge_source_prose_ligature_spans(
                [(token, index, index + 1) for index, token in enumerate(source_tokens_raw)],
                local_vocabulary,
            )
        ]

        assessed += 1
        comparison_tokens = _anchored_local_prose_window(
            source_tokens,
            local_tokens,
            local_positions,
        )
        opcodes = difflib.SequenceMatcher(
            a=source_tokens,
            b=comparison_tokens,
            autojunk=False,
        ).get_opcodes()
        for opcode_index, (tag, source_start, source_end, _local_start, _local_end) in enumerate(opcodes):
            if tag != "delete" or opcode_index <= 0 or opcode_index + 1 >= len(opcodes):
                continue
            previous = opcodes[opcode_index - 1]
            following = opcodes[opcode_index + 1]
            if previous[0] != "equal" or following[0] != "equal":
                continue
            if previous[2] - previous[1] < 3 or following[2] - following[1] < 3:
                continue
            missing = [
                token
                for token in source_tokens[source_start:source_end]
                if token.isalpha() and len(token) >= 2
            ]
            if not missing:
                continue
            omitted_words += len(missing)
            omission_groups += 1
            if len(examples) < 12:
                examples.append(" ".join(missing[:18]))

    return {
        "assessed_prose_block_count": int(assessed),
        "anchored_omitted_word_count": int(omitted_words),
        "anchored_omission_group_count": int(omission_groups),
        "anchored_omission_examples": examples[:12],
        "text_omission": bool(omitted_words >= SOURCE_PAGE_MIN_ANCHORED_OMITTED_WORDS),
    }


def _source_page_coverage_quality(text: str, pdf_path: Path | None) -> dict[str, Any]:
    empty = {
        "min_source_page_coverage": 0.0,
        "missing_source_page_count": 0,
        "missing_source_pages": [],
        "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
        "source_page_text_corruption_count": 0,
        "source_page_text_corruption_pages": [],
        "source_page_prose_omission_count": 0,
        "source_page_prose_omission_pages": [],
        "evidence_unreliable_pages": [],
        "page_evidence_profiles": [],
    }
    if fitz is None or pdf_path is None:
        return dict(empty)
    try:
        path = Path(pdf_path).expanduser()
        if not path.exists() or not path.is_file():
            return dict(empty)
    except Exception:
        return dict(empty)

    md_tokens = _rare_source_tokens(text)
    if len(md_tokens) < SOURCE_PAGE_MIN_RARE_TOKENS:
        return dict(empty)
    marker_pages = {
        int(match.group(1))
        for match in PAGE_MARKER_RE.finditer(str(text or ""))
        if str(match.group(1) or "").isdigit()
    }
    marker_segments: dict[int, tuple[int, int]] = {}
    for item in _page_marker_occurrences(text):
        page_no = int(item.get("page") or 0)
        if page_no > 0 and page_no not in marker_segments:
            marker_segments[page_no] = (
                int(item.get("segment_start") or 0),
                int(item.get("segment_end") or 0),
            )
    references_heading = re.search(
        r"(?mi)^#{1,6}\s+(?:References|Bibliography)\s*$",
        str(text or ""),
    )
    references_offset = int(references_heading.start()) if references_heading else -1
    inferred_offsets: dict[int, int] = {}
    try:
        inferred_offsets = _page_marker_offsets_from_pdf_text(str(text or ""), path, snap_to_line_start=False)
    except Exception:
        inferred_offsets = {}

    def local_segment_for_page(page_no: int) -> str:
        if page_no in marker_segments:
            start, end = marker_segments[page_no]
            return str(text or "")[max(0, start) : max(0, end)]
        known = sorted((int(page), int(offset)) for page, offset in inferred_offsets.items() if int(page) > 0)
        previous = [(page, offset) for page, offset in known if page < page_no]
        following = [(page, offset) for page, offset in known if page > page_no]
        if not previous or not following:
            return ""
        start = previous[-1][1]
        end = following[0][1]
        if end <= start:
            return ""
        return str(text or "")[max(0, start) : max(0, end)]

    low_pages: list[dict[str, Any]] = []
    corrupted_pages: list[dict[str, Any]] = []
    prose_omission_pages: list[dict[str, Any]] = []
    page_profiles: list[dict[str, Any]] = []
    min_coverage = 1.0
    assessed = 0
    try:
        doc = fitz.open(str(path))
    except Exception:
        return dict(empty)
    try:
        for page_index in range(len(doc)):
            page_no = page_index + 1
            try:
                page = doc.load_page(page_index)
                page_text = str(page.get_text("text") or "")
                page_block_texts = [
                    str(block[4] or "")
                    for block in list(page.get_text("blocks", sort=True) or [])
                    if len(block) >= 5
                ]
            except Exception:
                page_text = ""
                page_block_texts = []
            if marker_pages and page_no < min(marker_pages) and _pdf_page_looks_like_download_landing_page(page_text):
                continue
            page_tokens = _rare_source_tokens(page_text)
            local_segment = local_segment_for_page(page_no)
            if len(page_tokens) < SOURCE_PAGE_MIN_RARE_TOKENS:
                source_alnum_chars = len(re.findall(r"[A-Za-z0-9]", page_text))
                empty_marked_page = bool(
                    page_no in marker_segments
                    and not _rare_source_tokens(local_segment)
                    and source_alnum_chars >= SOURCE_PAGE_EMPTY_MARKER_MIN_ALNUM_CHARS
                )
                if (
                    empty_marked_page
                ):
                    assessed += 1
                    min_coverage = 0.0
                    low_pages.append(
                        {
                            "page": int(page_no),
                            "coverage": 0.0,
                            "local_coverage": 0.0,
                            "source_token_count": int(len(page_tokens)),
                            "has_page_marker": True,
                            "reason": "empty_page_marker_segment",
                        }
                    )
                page_profiles.append(
                    {
                        "page": int(page_no),
                        "status": "unreliable" if empty_marked_page else "unassessed",
                        "source_token_count": int(len(page_tokens)),
                        "has_page_marker": bool(page_no in marker_pages),
                        "reason_codes": ["empty_page_marker_segment"] if empty_marked_page else [],
                    }
                )
                continue
            assessed += 1
            coverage = len(page_tokens.intersection(md_tokens)) / max(1, len(page_tokens))
            local_coverage: float | None = None
            local_token_count: int | None = None
            if page_no in marker_segments:
                local_token_count = len(_rare_source_tokens(local_segment))
            if local_segment:
                local_tokens = _rare_source_tokens(local_segment)
                local_coverage = len(page_tokens.intersection(local_tokens)) / max(1, len(page_tokens))
            marker_start = int(marker_segments.get(page_no, (-1, -1))[0])
            within_references = bool(
                references_offset >= 0
                and marker_start >= references_offset
            )
            # Bibliography pages intentionally normalize line wraps, author
            # accents, venue punctuation, and page ranges. Generic prose-gap
            # diagnostics misclassify those changes as missing prose. Reference
            # numbering, continuity, and terminal-hyphen checks provide the
            # source-specific quality contract for these pages instead.
            wrap_damage = (
                {}
                if within_references
                else _source_page_wrapped_word_damage(page_text, local_segment)
            )
            prose_omission = (
                {}
                if within_references
                else _source_page_prose_omission_damage(page_block_texts, local_segment)
            )
            if bool(wrap_damage.get("text_corruption")):
                corrupted_pages.append(
                    {
                        "page": int(page_no),
                        "source_wrapped_word_count": int(wrap_damage.get("source_wrapped_word_count") or 0),
                        "missing_wrapped_word_count": int(wrap_damage.get("missing_wrapped_word_count") or 0),
                        "missing_wrapped_word_ratio": float(wrap_damage.get("missing_wrapped_word_ratio") or 0.0),
                        "examples": list(wrap_damage.get("missing_wrapped_word_examples") or [])[:12],
                        "reason": "missing_wrapped_word_prefixes",
                    }
                )
            if bool(prose_omission.get("text_omission")):
                prose_omission_pages.append(
                    {
                        "page": int(page_no),
                        "assessed_prose_block_count": int(
                            prose_omission.get("assessed_prose_block_count") or 0
                        ),
                        "anchored_omitted_word_count": int(
                            prose_omission.get("anchored_omitted_word_count") or 0
                        ),
                        "anchored_omission_group_count": int(
                            prose_omission.get("anchored_omission_group_count") or 0
                        ),
                        "examples": list(prose_omission.get("anchored_omission_examples") or [])[:12],
                        "reason": "source_prose_omission",
                    }
                )
            empty_marked_page_segment = bool(
                page_no in marker_segments and int(local_token_count or 0) == 0
            )
            local_low = bool(
                local_coverage is not None
                and local_coverage < SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD
                and page_no not in set(inferred_offsets)
            )
            low_global = bool(page_no not in marker_pages and coverage < SOURCE_PAGE_COVERAGE_THRESHOLD)
            reason_codes: list[str] = []
            if empty_marked_page_segment:
                reason_codes.append("empty_page_marker_segment")
                low_pages.append(
                    {
                        "page": int(page_no),
                        "coverage": round(float(coverage), 4),
                        "local_coverage": 0.0,
                        "source_token_count": int(len(page_tokens)),
                        "has_page_marker": True,
                        "reason": "empty_page_marker_segment",
                    }
                )
                min_coverage = min(min_coverage, 0.0)
            else:
                min_coverage = min(min_coverage, coverage)
                if local_low or low_global:
                    reason = "low_local_page_overlap" if local_low else "low_text_overlap"
                    reason_codes.append(reason)
                    low_pages.append(
                        {
                            "page": int(page_no),
                            "coverage": round(float(coverage), 4),
                            "local_coverage": round(float(local_coverage), 4) if local_coverage is not None else None,
                            "source_token_count": int(len(page_tokens)),
                            "has_page_marker": bool(page_no in marker_pages),
                            "reason": reason,
                        }
                    )
            if bool(wrap_damage.get("text_corruption")):
                reason_codes.append("missing_wrapped_word_prefixes")
            if bool(prose_omission.get("text_omission")):
                reason_codes.append("source_prose_omission")
            page_profiles.append(
                {
                    "page": int(page_no),
                    "status": "unreliable" if reason_codes else "ready",
                    "coverage": round(float(coverage), 4),
                    "local_coverage": round(float(local_coverage), 4) if local_coverage is not None else None,
                    "source_token_count": int(len(page_tokens)),
                    "has_page_marker": bool(page_no in marker_pages),
                    "missing_wrapped_word_count": int(wrap_damage.get("missing_wrapped_word_count") or 0),
                    "missing_wrapped_word_ratio": float(wrap_damage.get("missing_wrapped_word_ratio") or 0.0),
                    "anchored_omitted_word_count": int(
                        prose_omission.get("anchored_omitted_word_count") or 0
                    ),
                    "reason_codes": reason_codes,
                }
            )
    finally:
        try:
            doc.close()
        except Exception:
            pass
    unreliable_pages = sorted(
        {
            int(item.get("page") or 0)
            for item in low_pages + corrupted_pages + prose_omission_pages
            if int(item.get("page") or 0) > 0
        }
    )
    return {
        "min_source_page_coverage": round(float(min_coverage if assessed > 0 else 0.0), 4),
        "missing_source_page_count": int(len(low_pages)),
        "missing_source_pages": low_pages[:20],
        "source_page_coverage_threshold": SOURCE_PAGE_COVERAGE_THRESHOLD,
        "source_page_text_corruption_count": int(len(corrupted_pages)),
        "source_page_text_corruption_pages": corrupted_pages[:50],
        "source_page_prose_omission_count": int(len(prose_omission_pages)),
        "source_page_prose_omission_pages": prose_omission_pages[:50],
        "evidence_unreliable_pages": unreliable_pages[:500],
        "page_evidence_profiles": page_profiles[:500],
    }


_TABLE_PAGE_ALIGNMENT_STOP_WORDS = {
    "and", "comparison", "comparisons", "different", "figure", "for", "from", "index",
    "method", "methods", "network", "ours", "psnr", "result", "results", "ssim", "table",
    "the", "under", "with",
}


def _table_page_alignment_tokens(value: str) -> set[str]:
    cleaned = PAGE_MARKER_RE.sub(" ", str(value or "")).lower()
    tokens: set[str] = set()
    for token in re.findall(r"[a-z][a-z0-9+\-]{2,}|(?<![a-z0-9])\d+(?:\.\d+)?", cleaned):
        normalized = token.lstrip("0") or "0" if token[0].isdigit() else token
        if normalized not in _TABLE_PAGE_ALIGNMENT_STOP_WORDS:
            tokens.add(normalized)
    return tokens


def _source_table_page_anchor_alignment_quality(text: str, pdf_path: Path | None) -> dict[str, Any]:
    empty = {
        "source_table_page_anchor_issue_count": 0,
        "source_table_page_anchor_issues": [],
    }
    if fitz is None or pdf_path is None:
        return empty
    path = Path(pdf_path).expanduser()
    if not path.exists() or not path.is_file():
        return empty

    lines = str(text or "").splitlines()
    marker_lines: dict[int, int] = {}
    current_page = 0
    tables: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        marker = PAGE_MARKER_RE.search(lines[index])
        if marker:
            current_page = int(marker.group(1))
            marker_lines.setdefault(current_page, index)
        if lines[index].lstrip().startswith("|") and lines[index].count("|") >= 2:
            start = index
            block: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith("|") and lines[index].count("|") >= 2:
                block.append(lines[index])
                index += 1
            if len(block) >= 3 and current_page > 0:
                context = " ".join(line for line in lines[max(0, start - 5) : start] if line.strip())
                probe_tokens = _table_page_alignment_tokens(context + " " + " ".join(block[:4]))
                if len(probe_tokens) >= 8:
                    tables.append(
                        {
                            "line": start,
                            "current_page": current_page,
                            "probe_tokens": probe_tokens,
                            "header": block[0][:240],
                        }
                    )
            continue
        index += 1

    if not tables:
        return empty
    try:
        doc = fitz.open(str(path))
    except Exception:
        return empty
    try:
        page_tokens = [
            _table_page_alignment_tokens(str(doc.load_page(page_index).get_text("text") or ""))
            for page_index in range(len(doc))
        ]
    finally:
        try:
            doc.close()
        except Exception:
            pass

    issues: list[dict[str, Any]] = []
    for table in tables:
        probe_tokens = set(table["probe_tokens"])
        scores = [len(probe_tokens.intersection(tokens)) / max(1, len(probe_tokens)) for tokens in page_tokens]
        if not scores:
            continue
        ranked = sorted(range(len(scores)), key=lambda page_index: scores[page_index], reverse=True)
        best_page = ranked[0] + 1
        best_score = scores[ranked[0]]
        second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
        current_page = int(table["current_page"])
        table_line = int(table["line"])
        if best_page != current_page + 1:
            continue
        if best_score < 0.60 or best_score - second_score < 0.12:
            continue
        if int(marker_lines.get(best_page, -1)) <= table_line:
            continue
        context_range = range(max(0, table_line - 5), table_line)
        asset_anchor_lines = [
            line_index
            for line_index in context_range
            if re.search(rf"page[_-]0*{best_page}[_-]", lines[line_index], flags=re.IGNORECASE)
        ]
        caption_anchor_lines = [
            line_index
            for line_index in context_range
            if CAPTION_LINE_RE.match(lines[line_index].strip())
        ]
        anchor_line = (
            min(asset_anchor_lines)
            if asset_anchor_lines
            else (min(caption_anchor_lines) if caption_anchor_lines else table_line)
        )
        issues.append(
            {
                "line": anchor_line + 1,
                "table_line": table_line + 1,
                "current_page": current_page,
                "source_page": best_page,
                "source_score": round(float(best_score), 4),
                "source_margin": round(float(best_score - second_score), 4),
                "header": str(table.get("header") or ""),
            }
        )
    return {
        "source_table_page_anchor_issue_count": len(issues),
        "source_table_page_anchor_issues": issues[:20],
    }


def _source_page_anchor_alignment_quality(text: str, pdf_path: Path | None) -> dict[str, Any]:
    empty = {
        "source_page_anchor_issue_count": 0,
        "missing_source_page_marker_count": 0,
        "misaligned_source_page_marker_count": 0,
        "source_page_anchor_issues": [],
        "source_page_anchor_alignment_threshold": PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS,
    }
    if fitz is None or pdf_path is None:
        return empty
    path = Path(pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return empty
    except Exception:
        return empty
    occurrences = _page_marker_occurrences(text)
    if not occurrences:
        return empty

    marker_offsets: dict[int, int] = {}
    marker_segments: dict[int, tuple[int, int]] = {}
    for item in occurrences:
        page_no = int(item.get("page") or 0)
        if page_no <= 0 or page_no in marker_offsets:
            continue
        marker_offsets[page_no] = int(item.get("start") or 0)
        marker_segments[page_no] = (
            int(item.get("segment_start") or 0),
            int(item.get("segment_end") or 0),
        )

    inferred = _page_marker_offsets_from_pdf_text(text, path, snap_to_line_start=False)
    inferred_pages = {int(page) for page in inferred if int(page) > 0}
    marker_pages = set(marker_offsets)

    issues: list[dict[str, Any]] = []
    for page_no in sorted(page for page in inferred_pages - marker_pages if page > 1):
        issues.append(
            {
                "page": int(page_no),
                "reason": "source_page_text_present_without_anchor",
                "inferred_offset": int(inferred.get(page_no) or 0),
            }
        )

    try:
        doc = fitz.open(str(path))
    except Exception:
        doc = None
    if doc is not None:
        try:
            for page_no, (segment_start, segment_end) in sorted(marker_segments.items()):
                if page_no <= 0 or page_no > len(doc):
                    continue
                try:
                    page_text = str(doc.load_page(page_no - 1).get_text("text") or "")
                except Exception:
                    page_text = ""
                page_tokens = _rare_source_tokens(page_text)
                if len(page_tokens) < SOURCE_PAGE_MIN_RARE_TOKENS:
                    continue
                segment = str(text or "")[max(0, segment_start) : max(0, segment_end)]
                segment_tokens = _rare_source_tokens(segment)
                coverage = len(page_tokens.intersection(segment_tokens)) / max(1, len(page_tokens))
                inferred_offset = inferred.get(page_no)
                marker_offset = marker_offsets.get(page_no)
                inferred_far = (
                    inferred_offset is None
                    or marker_offset is None
                    or abs(int(marker_offset) - int(inferred_offset)) > PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS
                )
                if coverage < SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD and inferred_far:
                    issues.append(
                        {
                            "page": int(page_no),
                            "reason": "page_anchor_segment_low_source_overlap",
                            "segment_coverage": round(float(coverage), 4),
                        }
                    )
        finally:
            try:
                doc.close()
            except Exception:
                pass

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in issues:
        page_no = int(item.get("page") or 0)
        reason = str(item.get("reason") or "")
        key = (page_no, reason)
        if page_no <= 0 or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    missing_count = sum(1 for item in deduped if str(item.get("reason") or "") == "source_page_text_present_without_anchor")
    return {
        "source_page_anchor_issue_count": int(len(deduped)),
        "missing_source_page_marker_count": int(missing_count),
        "misaligned_source_page_marker_count": int(len(deduped) - missing_count),
        "source_page_anchor_issues": deduped[:20],
        "source_page_anchor_alignment_threshold": PAGE_ALIGNMENT_ANCHOR_DRIFT_CHARS,
    }


_NON_CITATION_NUMERIC_ARRAY_CONTEXT_RE = re.compile(
    r"(?:channels?|heads?|blocks?|dimensions?|sizes?|resolutions?|widths?|depths?|"
    r"batch\s+sizes?|sequence\s+lengths?|kernel\s+sizes?|strides?|values?)"
    r"\s+(?:are|is|=)\s*$",
    re.IGNORECASE,
)


def _plausible_reference_citation_indices(body: str) -> set[int]:
    """Return numeric citations while excluding explicit parameter arrays.

    This filter is deliberately local to reference-completeness diagnostics;
    it does not change retrieval or citation grounding.  Architecture prose
    such as ``number of channels are [48,96,192,384]`` otherwise makes a
    complete 109-entry bibliography look truncated.
    """

    cited: set[int] = set()
    for spec, start, _end, _style in iter_inpaper_numeric_citations(body):
        prefix = str(body or "")[max(0, int(start) - 96):int(start)]
        if _NON_CITATION_NUMERIC_ARRAY_CONTEXT_RE.search(prefix):
            continue
        cited.update(parse_ref_num_set(spec, max_items=256))
    return cited


def _reference_index_truncated(text: str, metrics: dict[str, Any]) -> bool:
    reference_lines = int(metrics.get("reference_line_count") or 0)
    max_index = int(metrics.get("max_reference_index") or 0)
    extracted = int(metrics.get("extracted_reference_count") or 0)
    if reference_lines < 8 or max_index < 8:
        return False
    if extracted <= 0:
        return True
    ref_map = extract_references_map_from_md(text)
    first_index = min(ref_map) if ref_map else 0
    ref_heading = re.search(r"(?mi)^#{1,6}\s+(?:References|Bibliography)\s*$", str(text or ""))
    body = str(text or "")[: int(ref_heading.start())] if ref_heading else str(text or "")
    cited_indices = _plausible_reference_citation_indices(body)
    if first_index == 1 and max(cited_indices, default=0) > max_index:
        return True
    if first_index > 1:
        if re.search(r"\[\s*1(?:\s*[,;\-–]\s*\d{1,4})*\s*\]", body):
            return True
    if _reference_gap_is_material(ref_map):
        return True
    if _reference_map_has_short_truncated_entries(ref_map):
        return True
    expected = max(reference_lines, max_index)
    return extracted < max(5, int(expected * 0.55))


def _source_quality_view(
    md_path: Path,
    text: str,
    metrics: dict[str, Any],
    *,
    source_pdf_path: Path | str | None = None,
    allow_source_pdf_inference: bool = True,
) -> dict[str, Any]:
    pdf_path = (
        Path(source_pdf_path).expanduser()
        if source_pdf_path
        else (_guess_source_pdf_for_md(md_path) if allow_source_pdf_inference else None)
    )
    pdf_stats = _pdf_source_stats(pdf_path)
    profile = _document_profile(md_path, text)
    ref_layout = _reference_layout(text)
    abstract_candidate_text, abstract_changed = _insert_abstract_heading_only(text)
    _ = abstract_candidate_text
    abstract_autofix_likely = bool(abstract_changed)
    source_text_loss = False if bool(profile.get("abstract_not_applicable")) else _source_text_loss_likely(text, metrics, pdf_stats, ref_layout)
    page_quality = _page_alignment_quality(metrics, pdf_stats, text)
    page_coverage = _source_page_coverage_quality(text, pdf_path if pdf_path and bool(pdf_stats.get("available")) else None)
    page_anchor_quality = _source_page_anchor_alignment_quality(text, pdf_path if pdf_path and bool(pdf_stats.get("available")) else None)
    table_page_anchor_quality = _source_table_page_anchor_alignment_quality(
        text,
        pdf_path if pdf_path and bool(pdf_stats.get("available")) else None,
    )
    return {
        **profile,
        "source_pdf_path": str((pdf_stats or {}).get("path") or (pdf_path or "")),
        "source_pdf_available": bool(pdf_stats.get("available")),
        "pdf_page_count": int(pdf_stats.get("page_count") or 0),
        "pdf_text_chars": int(pdf_stats.get("text_chars") or 0),
        **page_quality,
        **page_coverage,
        **page_anchor_quality,
        **table_page_anchor_quality,
        **ref_layout,
        "abstract_autofix_likely": abstract_autofix_likely,
        "source_text_loss": bool(source_text_loss),
        "reference_index_truncated": _reference_index_truncated(text, metrics),
    }


def _move_early_references_to_end(md: str) -> tuple[str, bool]:
    lines = str(md or "").splitlines()
    ref_idx = -1
    for idx, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match((line or "").strip()):
            ref_idx = idx
            break
    if ref_idx < 0:
        return str(md or ""), False
    ref_count = 0
    body_idx = -1
    for idx in range(ref_idx + 1, len(lines)):
        st = (lines[idx] or "").strip()
        if re.match(r"^\[\d{1,4}\]\s+", st):
            ref_count += 1
            continue
        if BODY_SECTION_HEADING_RE.match(st):
            body_idx = idx
            break
    if body_idx <= ref_idx or ref_count < 3:
        return str(md or ""), False
    char_pos = len("\n".join(lines[:ref_idx]))
    if char_pos > int(max(1, len(str(md or ""))) * 0.2):
        return str(md or ""), False
    head = lines[:ref_idx]
    ref_block = lines[ref_idx:body_idx]
    body = lines[body_idx:]

    def _trim_blank_edges(items: list[str]) -> list[str]:
        out = list(items)
        while out and not (out[0] or "").strip():
            out.pop(0)
        while out and not (out[-1] or "").strip():
            out.pop()
        return out

    fixed_parts = _trim_blank_edges(head) + [""] + _trim_blank_edges(body) + [""] + _trim_blank_edges(ref_block)
    fixed = "\n".join(fixed_parts).strip() + "\n"
    return fixed, fixed != str(md or "")


def _normalize_markdown_tables_only(md: str) -> tuple[str, bool]:
    fixed = normalize_markdown_tables_document(md)
    return fixed, fixed != str(md or "")


def _repair_detached_table_rows_only(md: str) -> tuple[str, bool]:
    fixed = repair_detached_markdown_table_rows_document(md)
    return fixed, fixed != str(md or "")


def _repair_inline_math_boundaries_only(md: str) -> tuple[str, bool]:
    fixed = repair_inline_math_prose_boundaries_document(md)
    return fixed, fixed != str(md or "")


def _table_issue_page_spans(md: str) -> list[dict[str, int]]:
    text = str(md or "")
    lines = text.splitlines()
    out: list[dict[str, int]] = []
    for item in markdown_table_issue_spans(text):
        start = int(item.get("start") or 0)
        prefix = "\n".join(lines[:start])
        markers = list(PAGE_MARKER_RE.finditer(prefix))
        page_no = int(markers[-1].group(1)) if markers else 0
        if page_no <= 0:
            continue
        out.append({**item, "page": page_no})
    return out


def _bounded_source_recovery_paragraphs(text: str, *, max_chars: int = 700) -> str:
    """Split a long PDF text-layer block at sentence boundaries for reading."""

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text or "")) if part.strip()]
    paragraphs: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + 1 + len(part) > max_chars:
            paragraphs.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def _prepare_ambiguous_table_page_assets(md_path: Path, md_text: str, pdf_path: Path) -> list[int]:
    """Render authoritative PDF pages before transactional table recovery."""

    if fitz is None or not pdf_path.is_file():
        return []
    page_spans = _table_issue_page_spans(md_text)
    pages = sorted({int(item.get("page") or 0) for item in page_spans if int(item.get("page") or 0) > 0})
    if not pages:
        return []
    assets_dir = md_path.parent / "assets"
    recovery_dir = md_path.parent / ".conversion_cache" / "table_recovery"
    assets_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    lines = str(md_text or "").splitlines()
    prepared: list[int] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []
    try:
        for page_no in pages:
            if page_no > len(doc):
                continue
            asset_path = assets_dir / f"page_{page_no}_table_recovery.png"
            if not asset_path.is_file():
                page = doc.load_page(page_no - 1)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.25, 2.25), alpha=False)
                pixmap.save(str(asset_path))
            raw_blocks = [
                "\n".join(lines[int(item["start"]):int(item["end"])])
                for item in page_spans
                if int(item.get("page") or 0) == page_no
            ]
            (recovery_dir / f"page_{page_no}_original_tables.md").write_text(
                "\n\n---\n\n".join(raw_blocks).rstrip() + "\n",
                encoding="utf-8",
            )
            prepared.append(page_no)
    finally:
        doc.close()
    return prepared


def _recover_ambiguous_table_pages_from_pdf_text(
    md: str,
    md_path: Path,
    source_pdf_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Replace lossy tables with the rendered source page plus its text layer."""

    text = str(md or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    page_spans = _table_issue_page_spans(text)
    if not page_spans:
        return text, False
    lines = text.splitlines()
    replacements: dict[int, list[str]] = {}
    handled_pages: set[int] = set()
    for item in page_spans:
        page_no = int(item.get("page") or 0)
        asset_name = f"page_{page_no}_table_recovery.png"
        asset_path = md_path.parent / "assets" / asset_name
        fallback = _pdf_page_fallback_markdown(pdf_path, page_no)
        fallback_body = PAGE_MARKER_RE.sub("", fallback, count=1).strip()
        if (
            page_no <= 0
            or not asset_path.is_file()
            or len(_source_prose_tokens(fallback_body)) < SOURCE_PAGE_MIN_PROSE_BLOCK_TOKENS
        ):
            continue
        start = int(item.get("start") or 0)
        end = int(item.get("end") or start)
        if page_no not in handled_pages:
            replacement = [
                f"<!-- kb_table_source_recovery: {page_no} -->",
                "",
                f"**Table evidence.** Original table preserved from source PDF page {page_no}.",
                "",
                f"![Source PDF page {page_no} containing the recovered table](./assets/{asset_name})",
                "",
                f"<!-- kb_source_recovery: {page_no} -->",
                "",
                _bounded_source_recovery_paragraphs(fallback_body),
            ]
            handled_pages.add(page_no)
        else:
            replacement = [f"<!-- additional damaged table preserved in source PDF page {page_no} recovery above -->"]
        replacements[start] = replacement
        for index in range(start + 1, end):
            replacements[index] = []
    if not replacements:
        return text, False
    out: list[str] = []
    for index, line in enumerate(lines):
        if index in replacements:
            out.extend(replacements[index])
        else:
            out.append(line)
    fixed = "\n".join(out)
    if text.endswith("\n"):
        fixed += "\n"
    return fixed, fixed != text


def _escape_source_recovery_literal_headings(md: str) -> tuple[str, bool]:
    """Escape short lowercase PDF column labels that begin with ``#``.

    PDF text is not Markdown.  In a source-recovery segment a label such as
    ``# masks sampled`` is a table column, not an H1 heading.
    """

    out: list[str] = []
    in_source_recovery = False
    changed = False
    for line in str(md or "").splitlines():
        if re.match(r"^<!--\s*kb_source_recovery:\s*\d+\s*-->$", line.strip(), re.IGNORECASE):
            in_source_recovery = True
            out.append(line)
            continue
        if in_source_recovery and PAGE_MARKER_RE.fullmatch(line.strip()):
            in_source_recovery = False
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        title = str(match.group(2) or "").strip() if match else ""
        if (
            in_source_recovery
            and match
            and len(match.group(1)) == 1
            and title == title.lower()
            and len(title.split()) <= 6
        ):
            out.append("\\" + line)
            changed = True
        else:
            out.append(line)
    fixed = "\n".join(out)
    if str(md or "").endswith("\n"):
        fixed += "\n"
    return fixed, changed


def _normalize_heading_level_jumps(md: str) -> tuple[str, bool]:
    original = str(md or "")
    escaped, escaped_changed = _escape_source_recovery_literal_headings(original)
    lines = escaped.splitlines()
    out: list[str] = []
    previous_level = 0
    in_fence = False
    in_math = False
    changed = escaped_changed

    for line in lines:
        stripped = str(line or "").strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            continue
        if stripped == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_fence or in_math:
            out.append(line)
            continue

        match = re.match(r"^(#{1,6})(\s+.+)$", line)
        if not match:
            out.append(line)
            continue

        level = len(match.group(1))
        target_level = level
        if previous_level > 0 and level > previous_level + 1:
            target_level = min(6, previous_level + 1)
        if target_level != level:
            out.append("#" * target_level + match.group(2))
            changed = True
        else:
            out.append(line)
        previous_level = target_level

    fixed = "\n".join(out)
    if original.endswith("\n"):
        fixed += "\n"
    return fixed, changed and fixed != original


_REVIEW_PROMOTABLE_HEADING_RE = re.compile(
    r"\b(?:"
    r"problem|statement|classical|spatial|domain|filter(?:ing)?|variational|regularization|"
    r"non[-\s]?local|sparse|representation|low[-\s]?rank|minimization|transform|technique|"
    r"adaptive|bm3d|cnn|mlp|deep\s+learning|experiments?|metrics?|comparison|methods?|models?|"
    r"performance|conclusions?"
    r")\b",
    re.IGNORECASE,
)


def _review_heading_promotable(title: str) -> bool:
    text = re.sub(r"\s+", " ", str(title or "")).strip()
    if len(text) < 3 or len(text) > 110:
        return False
    if REFERENCES_HEADING_RE.match(f"## {text}") or re.match(r"^(?:abstract|keywords?)$", text, re.IGNORECASE):
        return False
    if CAPTION_LINE_RE.match(text):
        return False
    if re.search(r"[=\\^_{}]|^\(?\d{1,3}\)?\s*$", text):
        return False
    if re.search(r"[.!?]\s+\S", text):
        return False
    return bool(_REVIEW_PROMOTABLE_HEADING_RE.search(text))


def _promote_collapsed_review_headings(md: str) -> tuple[str, bool]:
    text = str(md or "")
    lines = text.splitlines()
    h2_count = sum(1 for line in lines if re.match(r"^##\s+\S", line))
    h3_count = sum(1 for line in lines if re.match(r"^###\s+\S", line))
    if h3_count < 8 or h2_count >= 8 or h3_count < max(8, h2_count * 2):
        return text, False

    out: list[str] = []
    in_fence = False
    in_math = False
    changed = False
    for line in lines:
        stripped = str(line or "").strip()
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            out.append(line)
            continue
        if stripped == "$$":
            in_math = not in_math
            out.append(line)
            continue
        if in_fence or in_math:
            out.append(line)
            continue

        match = re.match(r"^(###)(\s+)(.+?)\s*$", line)
        if match and _review_heading_promotable(match.group(3)):
            out.append(f"##{match.group(2)}{match.group(3).strip()}")
            changed = True
            continue
        out.append(line)

    fixed = "\n".join(out)
    if text.endswith("\n"):
        fixed += "\n"
    return fixed, changed and fixed != text


def _regression_reasons(base_text: str, candidate_text: str) -> list[str]:
    comparison = compare_markdown_quality(base_text, candidate_text)
    base = comparison.get("base") if isinstance(comparison.get("base"), dict) else {}
    cand = comparison.get("candidate") if isinstance(comparison.get("candidate"), dict) else {}
    flags = comparison.get("regression_flags") if isinstance(comparison.get("regression_flags"), dict) else {}
    reasons = [str(key) for key, value in flags.items() if bool(value)]
    if "tables_dropped" in reasons:
        base_table_issues = markdown_table_issue_counts(base_text)
        candidate_table_issues = markdown_table_issue_counts(candidate_text)
        base_duplicate_count = int(base_table_issues.get("duplicate_table_count") or 0) + int(
            base_table_issues.get("fragmented_duplicate_count") or 0
        )
        candidate_duplicate_count = int(candidate_table_issues.get("duplicate_table_count") or 0) + int(
            candidate_table_issues.get("fragmented_duplicate_count") or 0
        )
        tables_dropped = max(
            0,
            int(base.get("table_block_count") or 0) - int(cand.get("table_block_count") or 0),
        )
        duplicate_delta = max(0, base_duplicate_count - candidate_duplicate_count)
        if (
            0 < tables_dropped <= duplicate_delta
            and int(candidate_table_issues.get("literal_break_count") or 0) == 0
            and int(candidate_table_issues.get("fragmented_column_count") or 0) == 0
        ):
            reasons = [reason for reason in reasons if reason != "tables_dropped"]
    base_chars = int(base.get("chars") or 0)
    cand_chars = int(cand.get("chars") or 0)
    if reasons == ["analyzer_warnings_increased"]:
        base_no_markers = re.sub(r"\s*<!--\s*kb_page:\s*\d+\s*-->\s*", " ", str(base_text or ""))
        cand_no_markers = re.sub(r"\s*<!--\s*kb_page:\s*\d+\s*-->\s*", " ", str(candidate_text or ""))
        marker_only_change = (
            re.sub(r"\s+", " ", base_no_markers).strip()
            == re.sub(r"\s+", " ", cand_no_markers).strip()
        )
        structural_gain = bool(
            (not bool(base.get("has_abstract_heading"))) and bool(cand.get("has_abstract_heading"))
        ) or int(cand.get("caption_count") or 0) > int(base.get("caption_count") or 0)
        no_content_loss = (
            int(cand.get("chars") or 0) >= int(int(base.get("chars") or 0) * 0.9)
            and int(cand.get("image_count") or 0) >= int(base.get("image_count") or 0)
            and int(cand.get("table_block_count") or 0) >= int(base.get("table_block_count") or 0)
            and int(cand.get("reference_line_count") or 0) >= int(base.get("reference_line_count") or 0)
        )
        base_warning_count = int(base.get("analyzer_warning_count") or 0)
        cand_warning_count = int(cand.get("analyzer_warning_count") or 0)
        low_warning_delta = cand_warning_count <= max(3, base_warning_count + 2)
        content_backfill_gain = base_chars > 0 and cand_chars >= base_chars + max(1000, int(base_chars * 0.08))
        if marker_only_change or (structural_gain and no_content_loss) or (content_backfill_gain and no_content_loss and low_warning_delta):
            reasons = []
    if base_chars > 1000 and cand_chars < int(base_chars * 0.82):
        reasons.append("content_shrank_too_much")
    return reasons


def _transactional_structure_reasons(
    before_issue_codes: list[str],
    after_issue_codes: list[str],
    active_issue_codes: list[str],
) -> list[str]:
    """Reject a partial repair while page/reference structure is still unsafe."""
    before = {str(code or "").strip().lower() for code in before_issue_codes if str(code or "").strip()}
    after = {str(code or "").strip().lower() for code in after_issue_codes if str(code or "").strip()}
    active = {str(code or "").strip().lower() for code in active_issue_codes if str(code or "").strip()}
    reasons: list[str] = []
    for code in sorted(after & _TRANSACTIONAL_STRUCTURE_ISSUES):
        if code not in before:
            reasons.append(f"blocking_structure_introduced:{code}")
        elif code in active and code not in {"source_page_marker_alignment"}:
            reasons.append(f"blocking_structure_remains:{code}")
    return reasons


def _source_pdf_name_candidates(md_path: Path) -> list[str]:
    names: list[str] = []
    parent_name = str(md_path.parent.name or "").strip()
    if parent_name:
        names.append(parent_name)
    stem = str(md_path.stem or "").strip()
    if stem.endswith(".en"):
        stem = stem[:-3]
    if stem and stem not in names:
        names.append(stem)
    out: list[str] = []
    for name in names:
        clean = name[:-4] if name.lower().endswith(".pdf") else name
        if clean and f"{clean}.pdf" not in out:
            out.append(f"{clean}.pdf")
    return out


def _candidate_pdf_roots(md_path: Path) -> list[Path]:
    roots: list[Path] = []

    def add(raw: str | Path | None) -> None:
        if raw is None:
            return
        try:
            p = Path(raw).expanduser()
        except Exception:
            return
        if p and p not in roots:
            roots.append(p)

    add(md_path.parent)
    add(md_path.parent.parent)
    add(os.environ.get("KB_PDF_DIR"))

    pref_paths = [
        Path.cwd() / "user_prefs.json",
        Path(__file__).resolve().parents[2] / "user_prefs.json",
        md_path.parent.parent / "user_prefs.json",
    ]
    for pref_path in pref_paths:
        try:
            payload = json.loads(pref_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            add(payload.get("pdf_dir"))
    return roots


def _guess_source_pdf_for_md(md_path: Path) -> Path | None:
    names = _source_pdf_name_candidates(md_path)
    if not names:
        return None
    for root in _candidate_pdf_roots(md_path):
        for name in names:
            candidate = root / name
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except Exception:
                continue
    lowered = {name.lower() for name in names}
    for root in _candidate_pdf_roots(md_path):
        try:
            for candidate in root.glob("*.pdf"):
                if candidate.name.lower() in lowered:
                    return candidate
        except Exception:
            continue
    return None


def _word_tokens_with_offsets(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for m in re.finditer(r"[^\W_]+", str(text or ""), flags=re.UNICODE):
        raw = unicodedata.normalize("NFKC", m.group(0) or "").lower()
        raw = raw.replace("\u00ad", "")
        for sub in re.finditer(r"[a-z][a-z0-9]{2,}|\d{2,}", raw):
            out.append((sub.group(0), int(m.start())))
    return out


def _page_alignment_candidates(tokens: list[tuple[str, int]], width: int = PAGE_ALIGNMENT_DEFAULT_NGRAM) -> list[tuple[tuple[str, ...], int, int]]:
    if len(tokens) < width:
        return []
    total = len(tokens)
    starts: list[int] = []
    starts.extend(range(0, min(total, 220), 8 if width <= 6 else 12))
    starts.extend(range(max(0, total // 3 - 80), min(total, total // 3 + 100), 16 if width <= 6 else 20))
    starts.extend(range(max(0, (2 * total) // 3 - 80), min(total, (2 * total) // 3 + 100), 16 if width <= 6 else 20))
    out: list[tuple[tuple[str, ...], int, int]] = []
    seen: set[tuple[str, ...]] = set()
    rare_min = 2 if width <= 6 else 3
    for start in starts:
        if start + width > total:
            continue
        gram = tuple(tok for tok, _ in tokens[start : start + width])
        if gram in seen:
            continue
        seen.add(gram)
        rare = sum(
            1
            for tok in gram
            if len(tok) >= 5 and tok not in PAGE_ALIGNMENT_STOP_WORDS and not tok.isdigit()
        )
        if rare < rare_min:
            continue
        out.append((gram, rare, start))
    return out


def _line_start_for_offset(text: str, offset: int) -> int:
    pos = max(0, min(len(text), int(offset)))
    return str(text or "").rfind("\n", 0, pos) + 1


def _page_alignment_md_grams(md_tokens: list[tuple[str, int]], width: int) -> dict[tuple[str, ...], list[int]]:
    md_grams: dict[tuple[str, ...], list[int]] = {}
    for idx in range(0, len(md_tokens) - width + 1):
        gram = tuple(tok for tok, _ in md_tokens[idx : idx + width])
        bucket = md_grams.setdefault(gram, [])
        if len(bucket) < 30:
            bucket.append(idx)
    return md_grams


def _page_alignment_page_candidates(
    page_tokens: list[tuple[str, int]],
    md_grams: dict[tuple[str, ...], list[int]],
    width: int,
) -> list[tuple[int, int, int]]:
    by_md_index: dict[int, tuple[int, int]] = {}
    for gram, rare, page_token_start in _page_alignment_candidates(page_tokens, width):
        for md_token_index in md_grams.get(gram, []):
            if md_token_index < 10:
                continue
            old = by_md_index.get(md_token_index)
            if old is None or rare > old[0] or (rare == old[0] and page_token_start < old[1]):
                by_md_index[md_token_index] = (rare, page_token_start)
    choices = [(idx, rare, start) for idx, (rare, start) in by_md_index.items()]
    choices.sort(key=lambda item: (int(item[0]), int(item[2]), -int(item[1])))
    return choices[:160]


def _select_page_alignment_offsets(
    md_tokens: list[tuple[str, int]],
    page_tokens_by_index: list[list[tuple[str, int]]],
    width: int,
) -> dict[int, int]:
    if len(md_tokens) < width:
        return {1: 0}
    md_grams = _page_alignment_md_grams(md_tokens, width)
    page_candidates = [
        []
        if page_index == 0
        else _page_alignment_page_candidates(page_tokens, md_grams, width)
        for page_index, page_tokens in enumerate(page_tokens_by_index)
    ]

    # State: matched pages, quality score, last Markdown token index, selected (page_no, token_index) pairs.
    states: list[tuple[int, float, int, tuple[tuple[int, int], ...]]] = [(0, 0.0, -1, tuple())]
    for page_no in range(2, len(page_tokens_by_index) + 1):
        new_states = list(states)
        for matched, quality, last_token_index, offsets in states:
            for md_token_index, rare, page_token_start in page_candidates[page_no - 1]:
                if md_token_index <= last_token_index:
                    continue
                score = quality + float(rare * 4.0) - float(page_token_start) * 0.4
                new_states.append(
                    (
                        matched + 1,
                        score,
                        int(md_token_index),
                        offsets + ((int(page_no), int(md_token_index)),),
                    )
                )
        if len(new_states) > PAGE_ALIGNMENT_BEAM_SIZE:
            def _state_key(
                item: tuple[int, float, int, tuple[tuple[int, int], ...]],
            ) -> tuple[int, float, int]:
                return int(item[0]), float(item[1]), -int(item[2])

            by_matched: dict[int, list[tuple[int, float, int, tuple[tuple[int, int], ...]]]] = {}
            for state in new_states:
                by_matched.setdefault(int(state[0]), []).append(state)
            # This is equivalent to sorting the full candidate set and taking
            # at most PAGE_ALIGNMENT_BEAM_PER_MATCH from each matched-page
            # bucket, then the global beam.  Heap selection avoids repeatedly
            # sorting tens of thousands of states on long papers while keeping
            # the same evidence-alignment scores and acceptance behavior.
            eligible: list[tuple[int, float, int, tuple[tuple[int, int], ...]]] = []
            for bucket in by_matched.values():
                eligible.extend(
                    heapq.nlargest(PAGE_ALIGNMENT_BEAM_PER_MATCH, bucket, key=_state_key)
                )
            states = heapq.nlargest(PAGE_ALIGNMENT_BEAM_SIZE, eligible, key=_state_key)
        else:
            states = new_states

    best = max(states, key=lambda item: (int(item[0]), float(item[1]), -int(item[2])), default=states[0])
    offsets: dict[int, int] = {1: 0}
    for page_no, md_token_index in best[3]:
        try:
            offsets[int(page_no)] = int(md_tokens[int(md_token_index)][1])
        except Exception:
            continue
    return offsets


def _source_page_token_coverage(page_text: str, segment_text: str) -> float:
    source_tokens = _rare_source_tokens(page_text)
    if not source_tokens:
        return 0.0
    segment_tokens = _rare_source_tokens(segment_text)
    if not segment_tokens:
        return 0.0
    return len(source_tokens.intersection(segment_tokens)) / max(1, len(source_tokens))


def _pdf_page_looks_like_download_landing_page(page_text: str) -> bool:
    text = _clean_pdf_page_block_text(page_text)
    if not text:
        return False
    low = text.lower()
    signals = 0
    for pattern in (
        "latest updates",
        "pdf download",
        "total citations",
        "total downloads",
        "citation in bibtex",
        "open access support",
        "dl.acm.org",
    ):
        if pattern in low:
            signals += 1
    if signals >= 2:
        return True
    return bool(signals >= 1 and "published" in low and ("doi" in low or "acm" in low))


def _adjust_offsets_for_skipped_leading_pdf_pages(
    md_text: str,
    pdf_path: Path,
    offsets: dict[int, int],
) -> dict[int, int]:
    """
    Some publisher PDFs include a download/landing page before the actual paper.
    If conversion intentionally starts at the article page, the alignment code may
    still synthesize page 1 at offset 0 and then split the true first article page.
    Prefer the first strongly matched later source page when it starts near the
    beginning and earlier PDF pages only weakly match the Markdown head.
    """
    if fitz is None or not offsets:
        return offsets
    later = sorted((int(page), int(offset)) for page, offset in offsets.items() if int(page) > 1)
    if not later:
        return offsets
    first_page, first_offset = later[0]
    if first_page <= 1 or first_offset < 0 or first_offset > LEADING_PAGE_ALIGNMENT_MAX_OFFSET:
        return offsets

    head_len = max(LEADING_PAGE_ALIGNMENT_WINDOW_CHARS, first_offset + LEADING_PAGE_ALIGNMENT_WINDOW_CHARS)
    head = str(md_text or "")[: min(len(str(md_text or "")), int(head_len))]
    if len(_rare_source_tokens(head)) < 20:
        return offsets

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return offsets
    try:
        if first_page > len(doc):
            return offsets
        first_text = str(doc.load_page(first_page - 1).get_text("text") or "")
        first_coverage = _source_page_token_coverage(first_text, head)
        previous_coverages: list[float] = []
        for page_index in range(0, max(0, first_page - 1)):
            previous_text = str(doc.load_page(page_index).get_text("text") or "")
            previous_coverages.append(_source_page_token_coverage(previous_text, head))
    except Exception:
        return offsets
    finally:
        try:
            doc.close()
        except Exception:
            pass

    best_previous = max(previous_coverages, default=0.0)
    skipped_pages_are_landing_pages = bool(previous_coverages)
    if skipped_pages_are_landing_pages:
        try:
            doc = fitz.open(str(pdf_path))
            try:
                skipped_pages_are_landing_pages = all(
                    _pdf_page_looks_like_download_landing_page(
                        str(doc.load_page(page_index).get_text("text") or "")
                    )
                    for page_index in range(0, max(0, first_page - 1))
                )
            finally:
                doc.close()
        except Exception:
            skipped_pages_are_landing_pages = False

    if (
        skipped_pages_are_landing_pages
        and first_coverage >= SOURCE_PAGE_SEGMENT_COVERAGE_THRESHOLD
        and best_previous < LEADING_PAGE_DROP_MAX_PREVIOUS_COVERAGE
        and (first_coverage - best_previous) >= LEADING_PAGE_DROP_MIN_COVERAGE_MARGIN
    ):
        adjusted = {int(page): int(offset) for page, offset in offsets.items() if int(page) >= first_page}
        adjusted[first_page] = 0
        return dict(sorted(adjusted.items(), key=lambda item: int(item[0])))
    return offsets


def _page_marker_offsets_from_pdf_text(
    md_text: str,
    pdf_path: Path,
    *,
    snap_to_line_start: bool = True,
) -> dict[int, int]:
    if fitz is None:
        return {}
    path = Path(pdf_path).expanduser()
    try:
        if not path.exists() or not path.is_file():
            return {}
    except Exception:
        return {}

    cache_key = _page_marker_offsets_cache_key(
        md_text,
        path,
        snap_to_line_start=snap_to_line_start,
    )
    cached = _get_cached_page_marker_offsets(cache_key)
    if cached is not None:
        return cached

    md_tokens = _word_tokens_with_offsets(md_text)
    if len(md_tokens) < min(PAGE_ALIGNMENT_NGRAMS):
        return {}

    offsets: dict[int, int] = {1: 0}
    pdf_page_count = 0
    try:
        doc = fitz.open(str(path))
    except Exception:
        return offsets
    try:
        pdf_page_count = len(doc)
        page_tokens_by_index: list[list[tuple[str, int]]] = []
        for page_index in range(pdf_page_count):
            try:
                page_text = doc.load_page(page_index).get_text("text")
            except Exception:
                page_text = ""
            page_tokens_by_index.append(_word_tokens_with_offsets(page_text))

        best_offsets: dict[int, int] = {1: 0}
        for width in PAGE_ALIGNMENT_NGRAMS:
            current_offsets = _select_page_alignment_offsets(md_tokens, page_tokens_by_index, width)

            if len(current_offsets) > len(best_offsets):
                best_offsets = current_offsets
        offsets = _adjust_offsets_for_skipped_leading_pdf_pages(str(md_text or ""), path, best_offsets)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    matched_later_pages = len([p for p in offsets if p > 1])
    if matched_later_pages <= 0:
        offsets = {1: 0}
    elif pdf_page_count >= 6 and matched_later_pages < 2:
        offsets = {1: 0}
    elif snap_to_line_start:
        offsets = {
            int(page_no): _line_start_for_offset(md_text, int(offset))
            for page_no, offset in offsets.items()
        }
    _cache_page_marker_offsets(cache_key, offsets)
    return offsets


def _recover_page_markers_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    if PAGE_MARKER_RE.search(text):
        return text, False
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    offsets = _page_marker_offsets_from_pdf_text(text, pdf_path)
    if not offsets:
        return text, False
    fixed = text
    for page_no, offset in sorted(offsets.items(), key=lambda item: int(item[1]), reverse=True):
        marker = f"<!-- kb_page: {int(page_no)} -->"
        pos = max(0, min(len(fixed), int(offset)))
        needs_after = pos < len(fixed) and fixed[pos : pos + 2] != "\n\n"
        insert = marker + ("\n\n" if needs_after else "\n")
        if pos > 0 and not fixed[:pos].endswith("\n\n"):
            insert = "\n" + insert
        fixed = fixed[:pos] + insert + fixed[pos:]
    return fixed, fixed != text


def _insert_page_marker_at_offset(md_text: str, offset: int, page_no: int) -> str:
    text = str(md_text or "")
    pos = max(0, min(len(text), int(offset)))
    marker = f"<!-- kb_page: {int(page_no)} -->"
    left = text[:pos]
    right = text[pos:]
    insert = marker
    if left and not left.endswith("\n\n"):
        insert = ("\n" if left.endswith("\n") else "\n\n") + insert
    if right and not right.startswith("\n\n"):
        insert = insert + ("\n" if right.startswith("\n") else "\n\n")
    if not left and right and not right.startswith("\n"):
        insert += "\n\n"
    return left + insert + right


def _page_marker_insert_offset(md_text: str, offset: int) -> int:
    text = str(md_text or "")
    pos = max(0, min(len(text), int(offset)))
    line_start = _line_start_for_offset(text, pos)
    line_end = text.find("\n", pos)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    if re.match(r"\s*\[\s*\d{1,4}\s*]\s+", line):
        return int(line_start)
    return int(pos)


def _is_existing_converter_page_asset(md_path: Path, raw_target: str) -> bool:
    target = unquote(str(raw_target or "").strip())
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")].strip()
    else:
        target = re.split(r"\s+[\"']", target, maxsplit=1)[0].strip()
    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not target or re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", target, flags=re.IGNORECASE):
        return False
    relative = Path(target)
    if relative.is_absolute():
        return False
    try:
        candidate = (md_path.parent / relative).resolve()
        assets_root = (md_path.parent / "assets").resolve()
        candidate.relative_to(assets_root)
        return candidate.is_file()
    except Exception:
        return False


def _realign_page_markers_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    if not PAGE_MARKER_RE.search(text):
        return text, False
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    markerless = PAGE_MARKER_RE.sub("", text)
    offsets = _page_marker_offsets_from_pdf_text(markerless, pdf_path, snap_to_line_start=False)
    matched_later = len([page for page in offsets if int(page) > 1])
    if matched_later <= 0:
        return text, False
    pdf_page_count = 0
    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            try:
                pdf_page_count = len(doc)
            finally:
                doc.close()
        except Exception:
            pdf_page_count = 0
    if pdf_page_count >= 6:
        required = max(2, int((pdf_page_count - 1) * 0.45))
        if matched_later < required:
            return text, False
    page_alignment = _page_alignment_quality(
        _metric_view(md_path, text),
        _pdf_source_stats(pdf_path),
        text,
    )
    anchor_alignment = _source_page_anchor_alignment_quality(text, pdf_path)
    if not any(
        int(page_alignment.get(key) or 0) > 0
        for key in (
            "missing_pdf_page_marker_count",
            "duplicate_pdf_page_marker_count",
            "out_of_range_page_marker_count",
        )
    ) and int(anchor_alignment.get("source_page_anchor_issue_count") or 0) <= 0:
        return text, False

    # Text alignment cannot place an image-only page. Converter-owned asset names
    # retain the physical PDF page number, so use the first page asset as a safe
    # anchor for otherwise unmatched pages.
    if pdf_page_count > 0:
        asset_pattern = re.compile(
            r"(?im)^.*?!\[[^\]]*]\(([^\n)]*?page[_-](\d{1,5})[_-][^\n)]*)\).*$"
        )
        for match in asset_pattern.finditer(markerless):
            page_no = int(match.group(2))
            if (
                1 <= page_no <= pdf_page_count
                and page_no not in offsets
                and _is_existing_converter_page_asset(md_path, match.group(1))
            ):
                offsets[page_no] = _line_start_for_offset(markerless, int(match.start()))

    existing_pages = {
        int(match.group(1))
        for match in PAGE_MARKER_RE.finditer(text)
        if 1 <= int(match.group(1)) <= pdf_page_count
    }

    cleaned_offsets: dict[int, int] = {}
    last_offset = -1
    for page_no, offset in sorted(offsets.items(), key=lambda item: (int(item[0]), int(item[1]))):
        page = int(page_no)
        pos = _page_marker_insert_offset(markerless, int(offset))
        if page <= 0 or pos < last_offset:
            continue
        if page > 1 and pos == last_offset:
            continue
        cleaned_offsets[page] = pos
        last_offset = pos
    if len(cleaned_offsets) <= 1:
        return text, False
    if pdf_page_count > 0 and len(existing_pages) == pdf_page_count and existing_pages.difference(cleaned_offsets):
        # Re-alignment may move a proven page anchor, but it must never erase
        # one merely because that page lacks a distinctive text n-gram.
        return text, False

    fixed = markerless
    for page_no, offset in sorted(cleaned_offsets.items(), key=lambda item: int(item[1]), reverse=True):
        fixed = _insert_page_marker_at_offset(fixed, int(offset), int(page_no))
    return fixed, fixed != text


def _realign_table_page_markers_from_pdf_text(
    md_text: str,
    md_path: Path,
    source_pdf_path: Path | str | None = None,
) -> tuple[str, bool]:
    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if pdf_path is None:
        return text, False
    quality = _source_table_page_anchor_alignment_quality(text, pdf_path)
    issues = [item for item in list(quality.get("source_table_page_anchor_issues") or []) if isinstance(item, dict)]
    moves: dict[int, int] = {}
    for issue in issues:
        page_no = int(issue.get("source_page") or 0)
        line_index = int(issue.get("line") or 0) - 1
        if page_no > 1 and line_index >= 0:
            moves[page_no] = min(line_index, moves.get(page_no, line_index))
    if not moves:
        return text, False

    trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    move_lines = {line_index: page_no for page_no, line_index in moves.items()}
    output: list[str] = []
    for line_index, line in enumerate(lines):
        marker = PAGE_MARKER_RE.search(line)
        if marker and int(marker.group(1)) in moves:
            continue
        page_no = move_lines.get(line_index)
        if page_no is not None:
            if output and output[-1].strip():
                output.append("")
            output.append(f"<!-- kb_page: {page_no} -->")
            output.append("")
        output.append(line)
    fixed = "\n".join(output)
    if trailing_newline:
        fixed += "\n"
    return fixed, fixed != text


def _clean_pdf_page_block_text(raw: str) -> str:
    lines = [line.strip() for line in str(raw or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    text = "\n".join(lines)
    def join_hyphen(match: re.Match) -> str:
        left = str(match.group(1) or "")
        right = str(match.group(2) or "")
        if left.lower() in {"what", "goes"}:
            return f"{left}-{right}"
        return f"{left}{right}"

    text = re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z][A-Za-z]*)", join_hyphen, text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _normalize_text(text)


def _pdf_page_has_references_heading_text(text: str) -> bool:
    return bool(
        re.search(
            r"(?mi)^\s*(?:references?|bibliography|references?\s+and\s+links|literature\s+cited)\s*$",
            str(text or ""),
        )
    )


PDF_REFERENCE_START_LINE_RE = re.compile(r"^\s*(?:\[\s*(\d{1,4})\s*\]|(\d{1,4})[.)])\s+\S")
PDF_REFERENCE_STANDALONE_NUMBER_RE = re.compile(
    r"^\s*(?:\[\s*(\d{1,4})\s*\]|(\d{1,4})[.)])\s*$"
)


def _pdf_page_reference_start_numbers(text: str) -> list[int]:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    numbers: list[int] = []
    for idx, raw in enumerate(lines):
        line = str(raw or "").strip()
        if not line:
            continue
        match = PDF_REFERENCE_START_LINE_RE.match(line)
        if match and _is_plausible_reference_number(match.group(1) or match.group(2)):
            if _pdf_reference_window_has_signal(lines, idx):
                numbers.append(int(match.group(1) or match.group(2)))
            continue
        standalone = PDF_REFERENCE_STANDALONE_NUMBER_RE.match(line)
        standalone_number = (standalone.group(1) or standalone.group(2)) if standalone else None
        if standalone_number and _is_plausible_reference_number(standalone_number):
            if _pdf_reference_window_has_signal(lines, idx):
                numbers.append(int(standalone_number))
    return numbers


def _pdf_reference_window_has_signal(lines: list[str], idx: int) -> bool:
    window = " ".join(str(line or "").strip() for line in lines[idx : idx + 6])
    window = re.sub(r"\s+", " ", window).strip()
    if not window:
        return False
    if re.search(r"\b(?:18|19|20)\d{2}\b|https?://|www\.|\bdoi\s*:|10\.\d{4,9}/|arxiv", window, re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:journal|proceedings?|proc\.?|conference|ieee|acm|spie|springer|wiley|"
        r"elsevier|opt\.|optics?|photonics?|phys\.|nature|science|letters?|review|"
        r"commun\.|express|trans\.|vol\.|pp\.)\b",
        window,
        re.IGNORECASE,
    ):
        return True
    first = re.sub(r"\s+", " ", str(lines[idx] or "")).strip()
    if PDF_REFERENCE_STANDALONE_NUMBER_RE.match(first) and idx + 1 < len(lines):
        first = re.sub(r"\s+", " ", str(lines[idx + 1] or "")).strip()
    return bool(
        first.count(",") >= 2
        and re.match(r"^[A-Z][A-Za-z'\-]{1,40}(?:\s+[A-Z]\.?)?(?:,|\s)", first)
    )


def _pdf_page_has_reference_block_text(text: str) -> bool:
    numbers = _pdf_page_reference_start_numbers(text)
    if len(numbers) < 3:
        return False
    return len(consecutive_reference_chain_positions(numbers)) >= 3


def _pdf_reference_continuation_page_has_signal(text: str) -> bool:
    """Accept dense numbered pages after a References section has started.

    Two-column PDF text can omit a few entries from the stricter bibliographic
    signal detector even though the raw page still contains many bracketed
    reference starts. Once a prior page has established the References section,
    three plausible numbered starts are enough to continue collecting pages.
    """

    numbers = [
        int(match.group(1))
        for match in re.finditer(r"(?m)^\s*\[\s*(\d{1,4})\s*]\s+\S", str(text or ""))
        if _is_plausible_reference_number(match.group(1))
    ]
    return len(set(numbers)) >= 3


def _trim_pdf_page_text_to_first_reference(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    candidates: list[tuple[int, int]] = []
    for idx, raw_line in enumerate(lines):
        line = str(raw_line or "").strip()
        match = PDF_REFERENCE_START_LINE_RE.match(line)
        standalone = PDF_REFERENCE_STANDALONE_NUMBER_RE.match(line)
        raw_num = (
            (match.group(1) or match.group(2))
            if match
            else ((standalone.group(1) or standalone.group(2)) if standalone else None)
        )
        if raw_num and _is_plausible_reference_number(raw_num) and _pdf_reference_window_has_signal(lines, idx):
            candidates.append((idx, int(raw_num)))
    chain = consecutive_reference_chain_positions([number for _, number in candidates])
    if chain:
        line_idx = candidates[chain[0]][0]
        return "\n".join(lines[line_idx:]).strip()
    return raw.strip()


def _drop_pdf_reference_running_lines(text: str) -> str:
    lines: list[str] = []
    raw_lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    seen_reference_start = False
    for idx, raw in enumerate(raw_lines):
        line = str(raw or "").strip()
        prev_line = str(raw_lines[idx - 1] or "").strip() if idx > 0 else ""
        next_line = str(raw_lines[idx + 1] or "").strip() if idx + 1 < len(raw_lines) else ""
        if not line:
            lines.append(raw)
            continue
        if is_reference_running_line(line) and (
            not is_ambiguous_reference_running_line(line) or not seen_reference_start
        ):
            continue
        if re.fullmatch(r"Research\s+Article", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"(?:TOPICAL\s+REVIEW|FRONTIERS\s+OF\s+PHYSICS)", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{4,6}-\d{1,3}", line):
            continue
        if re.fullmatch(r".+\bet\s+al\.,\s+Front\.\s+Phys\.\s+\d.+", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^Vol\.\s+\d+\b.*\b(?:Optica|Optics?|Photonics?)\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^#\d+\b", line):
            continue
        if re.fullmatch(r"https?://doi\.org/\S+", line, flags=re.IGNORECASE) and (
            re.match(r"^#\d+\b", prev_line) or re.match(r"^Journal\s+©\s+\d{4}", next_line, flags=re.IGNORECASE)
        ):
            continue
        if re.fullmatch(r"Journal\s+©\s+\d{4}", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^Received\s+\d{1,2}\s+\w+\s+\d{4}\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\(?C\)?\s+\d{4}\s+OSA\b", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\d{1,2}\s+\w+\s+\d{4}\s*/\s*Vol\.", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        lines.append(raw)
        if PDF_REFERENCE_START_LINE_RE.match(line) or PDF_REFERENCE_STANDALONE_NUMBER_RE.match(line):
            seen_reference_start = True
    return merge_standalone_reference_continuations("\n".join(lines).strip())


def _merge_pdf_page_paragraphs(parts: list[str]) -> list[str]:
    out: list[str] = []
    for raw in parts:
        part = str(raw or "").strip()
        if not part:
            continue
        if not out:
            out.append(part)
            continue
        prev = out[-1].rstrip()
        if prev.endswith("-"):
            out[-1] = prev[:-1] + part.lstrip()
            continue
        if not re.search(r"[.!?]\s*$", prev) and re.match(r"^[a-z]", part):
            out[-1] = prev + " " + part
            continue
        out.append(part)
    return out


def _pdf_page_fallback_markdown(pdf_path: Path, page_no: int) -> str:
    if fitz is None:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return ""
    try:
        if page_no <= 0 or page_no > len(doc):
            return ""
        page = doc.load_page(page_no - 1)
        W = float(page.rect.width)
        H = float(page.rect.height)
        main_blocks: list[tuple[int, float, str]] = []
        footnote_blocks: list[tuple[int, float, str]] = []
        for block in list(page.get_text("blocks") or []):
            if len(block) < 5:
                continue
            try:
                x0, y0, x1, y1 = float(block[0]), float(block[1]), float(block[2]), float(block[3])
            except Exception:
                continue
            text = _clean_pdf_page_block_text(str(block[4] or ""))
            text = re.sub(r"(?m)^(\s*)(#{1,6})(?=\s)", r"\1\\\2", text)
            if not text:
                continue
            width = max(0.0, x1 - x0)
            if re.search(r"\bwww\.(?:advancedsciencenews|lpr-journal)\.com\b", text, re.IGNORECASE):
                continue
            if x0 > W * 0.90 or width < W * 0.035:
                continue
            if y0 > H * 0.86 and re.search(r"\b(?:copyright|wiley|photonics\s+rev|\d+\s+of\s+\d+)\b", text, re.IGNORECASE):
                continue
            if re.fullmatch(r"(?:references|bibliography)", text, re.IGNORECASE):
                continue
            if re.match(r"^\d{1,4}\.\s+[A-Z]", text) and y0 > H * 0.65:
                continue
            if width >= W * 0.62:
                col = 0
            else:
                col = 0 if x0 < W * 0.50 else 1
            main_blocks.append((col, y0, text))
        main = _merge_pdf_page_paragraphs([text for _, _, text in sorted(main_blocks, key=lambda item: (item[0], item[1]))])
        footnotes = [text for _, _, text in sorted(footnote_blocks, key=lambda item: (item[0], item[1]))]
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if not main and not footnotes:
        return ""
    parts = [f"<!-- kb_page: {int(page_no)} -->"]
    parts.extend(main)
    parts.extend(f"> {note}" for note in footnotes)
    return "\n\n".join(part for part in parts if part).strip()


def _line_start_before(text: str, offset: int) -> int:
    return str(text or "").rfind("\n", 0, max(0, int(offset))) + 1


def _insertion_offset_for_missing_page(text: str, page_no: int) -> int:
    markers: list[tuple[int, int, int]] = []
    for match in PAGE_MARKER_RE.finditer(str(text or "")):
        try:
            start = int(match.start())
            markers.append((int(match.group(1)), start, _line_start_before(text, start)))
        except Exception:
            continue
    ref_match = re.search(r"(?mi)^#{1,6}\s+References\s*$", str(text or ""))
    ref_offset = int(ref_match.start()) if ref_match else -1
    body_next_offsets = [
        line_start
        for page, _offset, line_start in markers
        if page > page_no and line_start >= 0 and (ref_offset < 0 or line_start < ref_offset)
    ]
    if body_next_offsets:
        return min(body_next_offsets)
    if ref_offset >= 0:
        reference_pages = [page for page, offset, _line_start in markers if offset >= ref_offset]
        first_reference_page = min(reference_pages) if reference_pages else 0
        if first_reference_page > 0 and page_no < first_reference_page:
            return ref_offset
    next_offsets = [line_start for page, _offset, line_start in markers if page > page_no and line_start >= 0]
    if next_offsets:
        return min(next_offsets)
    if ref_offset >= 0:
        reference_pages = [page for page, offset, _line_start in markers if offset >= ref_offset]
        first_reference_page = min(reference_pages) if reference_pages else 0
        if first_reference_page <= 0 or page_no < first_reference_page:
            return ref_offset
    return len(str(text or ""))


def _recover_missing_source_pages_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    coverage = _source_page_coverage_quality(text, pdf_path)
    page_alignment = _page_alignment_quality(
        _metric_view(md_path, text),
        _pdf_source_stats(pdf_path),
        text,
    )
    anchor_alignment = _source_page_anchor_alignment_quality(text, pdf_path)
    pages = [
        int(item.get("page") or 0)
        for item in list(coverage.get("missing_source_pages") or [])
        if int(item.get("page") or 0) > 0
    ]
    pages.extend(
        int(page_no)
        for page_no in list(page_alignment.get("missing_pdf_page_markers") or [])
        if int(page_no or 0) > 0
    )
    pages.extend(
        int(item.get("page") or 0)
        for item in list(anchor_alignment.get("source_page_anchor_issues") or [])
        if isinstance(item, dict)
        and str(item.get("reason") or "") == "page_anchor_segment_low_source_overlap"
        and float(item.get("segment_coverage") or 0.0) < 0.20
        and int(item.get("page") or 0) > 0
    )
    existing_marker_pages = {
        int(match.group(1))
        for match in PAGE_MARKER_RE.finditer(text)
        if int(match.group(1)) > 0
    }
    if pages and existing_marker_pages and fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            try:
                first_marker = min(existing_marker_pages)
                pages = [
                    page_no
                    for page_no in pages
                    if not (
                        page_no < first_marker
                        and 1 <= page_no <= len(doc)
                        and _pdf_page_looks_like_download_landing_page(
                            str(doc.load_page(page_no - 1).get_text("text") or "")
                        )
                    )
                ]
            finally:
                doc.close()
        except Exception:
            pass
    if not pages:
        return text, False
    fixed = text
    for page_no in sorted(set(pages), reverse=True):
        fallback = _pdf_page_fallback_markdown(pdf_path, page_no)
        if not fallback:
            continue
        existing_marker = next(
            (
                item
                for item in _page_marker_occurrences(fixed)
                if int(item.get("page") or 0) == int(page_no)
            ),
            None,
        )
        if existing_marker is not None:
            marker_start = int(existing_marker.get("start") or 0)
            segment_start = int(existing_marker.get("segment_start") or 0)
            segment_end = int(existing_marker.get("segment_end") or len(fixed))
            current_body = fixed[segment_start:segment_end].strip()
            if not _rare_source_tokens(current_body):
                fixed = fixed[:marker_start] + fallback.strip() + "\n\n" + fixed[segment_end:]
                continue
            fallback_body = PAGE_MARKER_RE.sub("", fallback, count=1).strip()
            # A low-coverage page can still contain valuable figures, captions,
            # or tables. Keep that converted material and add the authoritative
            # PDF text inside the same page segment. Inserting another full page
            # block used to duplicate the page marker and could move the source
            # recovery past the next page, creating an alignment regression.
            references_tail = ""
            references_match = re.search(r"(?mi)^#{1,6}\s+References\s*$", current_body)
            if references_match:
                references_tail = current_body[references_match.start() :].strip()
                current_body = current_body[: references_match.start()].strip()
            replacement = "\n\n".join(
                part
                for part in (
                    f"<!-- kb_page: {page_no} -->",
                    current_body,
                    f"<!-- kb_source_recovery: {page_no} -->",
                    fallback_body,
                    references_tail,
                )
                if part
            ).strip() + "\n\n"
            fixed = fixed[:marker_start] + replacement + fixed[segment_end:]
            continue
        pos = _insertion_offset_for_missing_page(fixed, page_no)
        fallback_body = PAGE_MARKER_RE.sub("", fallback, count=1).strip()
        insert = "\n\n".join(
            (
                f"<!-- kb_page: {page_no} -->",
                f"<!-- kb_source_recovery: {page_no} -->",
                fallback_body,
            )
        ).strip() + "\n\n"
        if pos > 0 and not fixed[:pos].endswith("\n\n"):
            insert = "\n" + insert
        fixed = fixed[:pos] + insert + fixed[pos:]
    return fixed, fixed != text


def _bounded_source_prose_edits(
    block_texts: list[str],
    local_segment: str,
) -> list[tuple[int, int, str]]:
    """Build source-backed edits bounded by stable token sequences.

    Only pages already classified with a high-confidence omission are eligible.
    Individual differences must also have three unchanged tokens on both sides.
    Replacing the small gap between those anchors preserves Markdown headings,
    figures, paragraph breaks, and citations outside the damaged phrase.
    """

    damage = _source_page_prose_omission_damage(block_texts, local_segment)
    if not bool(damage.get("text_omission")):
        return []

    local_spans = _source_prose_token_spans(local_segment)
    if not local_spans:
        return []
    local_tokens = [token for token, _start, _end in local_spans]
    local_vocabulary = set(local_tokens)
    proposed: list[tuple[int, int, str]] = []

    for raw_block in block_texts:
        block = str(raw_block or "").strip()
        if not _eligible_source_prose_block(block):
            continue
        source_text = _clean_pdf_page_block_text(block)
        raw_source_spans = _source_prose_token_spans(source_text)
        source_spans = _merge_source_prose_ligature_spans(raw_source_spans, local_vocabulary)
        source_tokens = [token for token, _start, _end in source_spans]
        if len(source_tokens) < SOURCE_PAGE_MIN_PROSE_BLOCK_TOKENS:
            continue

        opcodes = difflib.SequenceMatcher(
            a=source_tokens,
            b=local_tokens,
            autojunk=False,
        ).get_opcodes()
        for opcode_index, (tag, _source_start, _source_end, _local_start, _local_end) in enumerate(opcodes):
            if tag not in {"delete", "replace"} or opcode_index <= 0 or opcode_index + 1 >= len(opcodes):
                continue
            previous = opcodes[opcode_index - 1]
            following = opcodes[opcode_index + 1]
            if previous[0] != "equal" or following[0] != "equal":
                continue
            if previous[2] - previous[1] < 3 or following[2] - following[1] < 3:
                continue

            source_gap_start = source_spans[previous[2] - 1][2]
            source_gap_end = source_spans[following[1]][1]
            local_gap_start = local_spans[previous[4] - 1][2]
            local_gap_end = local_spans[following[3]][1]
            if source_gap_end <= source_gap_start or local_gap_end < local_gap_start:
                continue
            if source_gap_end - source_gap_start > 320 or local_gap_end - local_gap_start > 320:
                continue
            replacement = source_text[source_gap_start:source_gap_end]
            current = local_segment[local_gap_start:local_gap_end]
            if not replacement.strip() or replacement == current:
                continue
            proposed.append((int(local_gap_start), int(local_gap_end), replacement))

    if not proposed:
        return []

    # Source blocks can overlap at column boundaries. Keep a deterministic,
    # non-overlapping set and apply it from the end of the page.
    accepted: list[tuple[int, int, str]] = []
    for edit in sorted(set(proposed), key=lambda item: (item[0], item[1], item[2])):
        start, end, _replacement = edit
        if any(start < accepted_end and end > accepted_start for accepted_start, accepted_end, _ in accepted):
            continue
        accepted.append(edit)
    return accepted


def _recover_source_prose_omissions_from_pdf_text(
    md_text: str,
    md_path: Path,
    source_pdf_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Repair page-local prose gaps from the source PDF text layer."""

    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if fitz is None or not pdf_path:
        return text, False
    coverage = _source_page_coverage_quality(text, pdf_path)
    pages = [
        int(item.get("page") or 0)
        for item in list(coverage.get("source_page_prose_omission_pages") or [])
        if int(item.get("page") or 0) > 0
    ]
    if not pages:
        return text, False

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return text, False
    fixed = text
    try:
        for page_no in sorted(set(pages), reverse=True):
            occurrence = next(
                (
                    item
                    for item in _page_marker_occurrences(fixed)
                    if int(item.get("page") or 0) == int(page_no)
                ),
                None,
            )
            if occurrence is None or not 1 <= page_no <= len(doc):
                continue
            segment_start = int(occurrence.get("segment_start") or 0)
            segment_end = int(occurrence.get("segment_end") or len(fixed))
            local_segment = fixed[segment_start:segment_end]
            try:
                page = doc.load_page(page_no - 1)
                block_texts = [
                    str(block[4] or "")
                    for block in list(page.get_text("blocks", sort=True) or [])
                    if len(block) >= 5
                ]
            except Exception:
                continue
            edits = _bounded_source_prose_edits(block_texts, local_segment)
            if not edits:
                continue
            repaired_segment = local_segment
            for start, end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
                repaired_segment = repaired_segment[:start] + replacement + repaired_segment[end:]
            fixed = fixed[:segment_start] + repaired_segment + fixed[segment_end:]
    finally:
        doc.close()
    return fixed, fixed != text


def _novel_source_recovery_prose(fallback_body: str, current_body: str) -> str:
    """Keep only source prose that is not already represented on the page.

    A corrupted page may still have trustworthy figures, captions, tables, and
    prose. Appending the complete PDF text fallback duplicates all of those
    structures in the reader and retrieval index. PDF text blocks are already
    separated by blank lines, so retain acknowledgements/funding plus eligible
    prose blocks whose ordered token sequence is not covered by the converted
    page. Short labels, flattened tables, and repeated captions remain backed
    by the existing structured Markdown and are intentionally omitted here.
    """

    current_tokens = _source_prose_tokens(current_body)
    kept: list[str] = []
    for raw_block in re.split(r"\n\s*\n", str(fallback_body or "")):
        block = str(raw_block or "").strip()
        if not block:
            continue
        block_tokens = _source_prose_tokens(block)
        if not block_tokens:
            continue
        special_prose = bool(
            re.match(
                r"^\s*(?:acknowledg(?:e)?ments?|funding|author\s+details?)\b",
                block,
                flags=re.IGNORECASE,
            )
        )
        if not special_prose:
            if _SOURCE_PROSE_BLOCK_SKIP_RE.match(block):
                continue
            alpha_chars = sum(char.isalpha() for char in block)
            alnum_chars = sum(char.isalnum() for char in block)
            substantial_prose = bool(
                len(block_tokens) >= SOURCE_PAGE_MIN_RARE_TOKENS
                and alpha_chars / max(1, alnum_chars) >= 0.70
                and re.search(r"[.!?](?:\s|$)", block)
            )
            if not _eligible_source_prose_block(block) and not substantial_prose:
                continue
        longest = difflib.SequenceMatcher(
            a=block_tokens,
            b=current_tokens,
            autojunk=False,
        ).find_longest_match(0, len(block_tokens), 0, len(current_tokens)).size
        covered_ratio = float(longest) / max(1, len(block_tokens))
        # Only discard near-exact duplicates. A long paragraph with one
        # converter omission can have high overall overlap while the longest
        # stable span is materially shorter; that source block is the evidence
        # needed to repair the page and must remain available.
        if covered_ratio >= 0.95:
            continue
        kept.append(block)
    return "\n\n".join(kept).strip()


def _recover_corrupted_source_pages_from_pdf_text(
    md_text: str,
    md_path: Path,
    source_pdf_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Replace only source-proven corrupted page segments with PDF block text.

    Existing page-local image links are retained so visual evidence remains
    available, while prose comes from source text blocks sorted by column and
    vertical position. The caller re-runs the source comparison before accepting
    the repair.
    """
    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path:
        return text, False
    coverage = _source_page_coverage_quality(text, pdf_path)
    pages = [
        int(item.get("page") or 0)
        for item in list(coverage.get("source_page_text_corruption_pages") or [])
        if int(item.get("page") or 0) > 0
    ]
    if not pages:
        return text, False

    fixed = text
    for page_no in sorted(set(pages), reverse=True):
        occurrence = next(
            (
                item
                for item in _page_marker_occurrences(fixed)
                if int(item.get("page") or 0) == int(page_no)
            ),
            None,
        )
        if occurrence is None:
            continue
        fallback = _pdf_page_fallback_markdown(pdf_path, page_no)
        if not fallback:
            continue
        marker_start = int(occurrence.get("start") or 0)
        segment_end = int(occurrence.get("segment_end") or len(fixed))
        current_segment = fixed[marker_start:segment_end]
        current_body = PAGE_MARKER_RE.sub("", current_segment, count=1).strip()
        fallback_body = PAGE_MARKER_RE.sub("", fallback, count=1).strip()
        prefix_lines = fixed[:marker_start].splitlines()
        within_references = False
        for prefix_line in reversed(prefix_lines):
            stripped = str(prefix_line or "").strip()
            if not stripped:
                continue
            if REFERENCES_HEADING_RE.match(stripped):
                within_references = True
                break
            if re.match(r"^#{1,6}\s+\S", stripped):
                break
        has_structured_evidence = bool(
            re.search(r"(?m)^\s*!\[[^\]]*]\([^)]+\)\s*$", current_body)
            or re.search(r"(?m)^\s*\|.+\|\s*$", current_body)
            or re.search(r"(?m)^\s*\$\$\s*$", current_body)
        )
        # Retain page-local visual and structured evidence, then append a clean
        # source-text recovery. Replacing the whole segment fixed OCR prose but
        # could drop valid Markdown tables and cause the transactional quality
        # gate to reject an otherwise source-accurate repair.
        if within_references and not has_structured_evidence:
            # Dense reference pages are prose-only and especially vulnerable
            # to column interleaving. Appending the fallback duplicates and
            # reorders bibliography entries during reference post-processing;
            # a source-text replacement is both safer and idempotent here.
            replacement_parts = [
                f"<!-- kb_page: {page_no} -->",
                f"<!-- kb_source_recovery: {page_no} -->",
                fallback_body,
            ]
        else:
            prior_recovery = re.search(
                rf"<!--\s*kb_source_recovery:\s*{int(page_no)}\s*-->",
                current_body,
                flags=re.IGNORECASE,
            )
            if prior_recovery:
                current_body = current_body[: prior_recovery.start()].rstrip()
            recovery_body = (
                _novel_source_recovery_prose(fallback_body, current_body)
                if has_structured_evidence
                else fallback_body
            )
            replacement_parts = [
                f"<!-- kb_page: {page_no} -->",
                current_body,
                f"<!-- kb_source_recovery: {page_no} -->" if recovery_body else "",
                recovery_body,
            ]
        replacement = "\n\n".join(part for part in replacement_parts if part).strip() + "\n\n"
        fixed = fixed[:marker_start] + replacement + fixed[segment_end:]
    return fixed, fixed != text


def _conversion_retry_attrs(marker: re.Match[str]) -> dict[str, str]:
    return {
        item.group(1).lower(): next(
            (value for value in item.groups()[1:] if value is not None),
            "",
        )
        for item in CONVERSION_RETRY_ATTR_RE.finditer(marker.group(1))
    }


def _recover_conversion_retry_pages_from_pdf_text(
    md_text: str,
    md_path: Path,
    source_pdf_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Replace unresolved retry fragments with source-page text.

    Math-text fragments are removed only after the same PDF page yields an
    authoritative text fallback. Equation comments are cleared only when the
    referenced image asset exists and remains linked in that page segment.
    """

    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    if not pdf_path or not CONVERSION_RETRY_MARKER_RE.search(text):
        return text, False

    pages: set[int] = set()
    for marker in CONVERSION_RETRY_MARKER_RE.finditer(text):
        attrs = _conversion_retry_attrs(marker)
        try:
            page_no = int(attrs.get("page") or 0)
        except Exception:
            page_no = 0
        if page_no > 0 and str(attrs.get("kind") or "").lower() in {"math_text", "equation"}:
            pages.add(page_no)

    fixed = text
    for page_no in sorted(pages, reverse=True):
        occurrence = next(
            (
                item
                for item in _page_marker_occurrences(fixed)
                if int(item.get("page") or 0) == page_no
            ),
            None,
        )
        fallback = _pdf_page_fallback_markdown(pdf_path, page_no)
        if occurrence is None or not fallback:
            continue
        marker_start = int(occurrence.get("start") or 0)
        segment_end = int(occurrence.get("segment_end") or len(fixed))
        current_segment = fixed[marker_start:segment_end]
        page_markers: list[tuple[re.Match[str], dict[str, str]]] = []
        page_is_recoverable = True
        for marker in CONVERSION_RETRY_MARKER_RE.finditer(current_segment):
            attrs = _conversion_retry_attrs(marker)
            try:
                marker_page = int(attrs.get("page") or 0)
            except Exception:
                marker_page = 0
            if marker_page != page_no:
                continue
            kind = str(attrs.get("kind") or "").strip().lower()
            if kind not in {"math_text", "equation"}:
                page_is_recoverable = False
                break
            if kind == "equation":
                asset = str(attrs.get("asset") or "").strip()
                asset_path = md_path.parent / "assets" / asset
                if not asset or Path(asset).name != asset or not asset_path.is_file() or asset not in current_segment:
                    page_is_recoverable = False
                    break
            page_markers.append((marker, attrs))
        fallback_body = PAGE_MARKER_RE.sub("", fallback, count=1).strip()
        if (
            not page_is_recoverable
            or not page_markers
            or len(_source_prose_tokens(fallback_body)) < SOURCE_PAGE_MIN_PROSE_BLOCK_TOKENS
        ):
            continue

        cleaned_lines: list[str] = []
        for line in current_segment.splitlines():
            relevant: list[dict[str, str]] = []
            for marker in CONVERSION_RETRY_MARKER_RE.finditer(line):
                attrs = _conversion_retry_attrs(marker)
                try:
                    marker_page = int(attrs.get("page") or 0)
                except Exception:
                    marker_page = 0
                if marker_page == page_no:
                    relevant.append(attrs)
            if not relevant:
                cleaned_lines.append(line)
                continue
            cleaned = CONVERSION_RETRY_MARKER_RE.sub("", line).rstrip()
            if any(str(attrs.get("kind") or "").lower() == "math_text" for attrs in relevant):
                if re.match(r"^\s*(?:#{1,6}\s+|!\[)", cleaned):
                    cleaned_lines.append(cleaned)
                continue
            if cleaned.strip():
                cleaned_lines.append(cleaned)

        current_body = PAGE_MARKER_RE.sub("", "\n".join(cleaned_lines), count=1).strip()
        replacement = "\n\n".join(
            part
            for part in (
                f"<!-- kb_page: {page_no} -->",
                current_body,
                f"<!-- kb_source_recovery: {page_no} -->",
                fallback_body,
            )
            if part
        ).strip() + "\n\n"
        fixed = fixed[:marker_start] + replacement + fixed[segment_end:]
    return fixed, fixed != text


_INLINE_NUMERIC_REFERENCE_MARKER_RE = re.compile(r"(?<!\S)\[\s*(\d{1,4})\s*\]\s+")


def _split_collapsed_numeric_reference_lines(text: str) -> str:
    """Split a line containing a verified ``[n] ... [n+1]`` reference chain."""
    output: list[str] = []
    for raw in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        matches = [
            match
            for match in _INLINE_NUMERIC_REFERENCE_MARKER_RE.finditer(raw)
            if _is_plausible_reference_number(match.group(1))
        ]
        numbers = [int(match.group(1)) for match in matches]
        chain = consecutive_reference_chain_positions(numbers)
        if len(chain) < 3:
            output.append(raw)
            continue
        chain_matches = [matches[index] for index in chain]
        prefix = raw[: chain_matches[0].start()].strip()
        if prefix:
            output.append(prefix)
        for idx, match in enumerate(chain_matches):
            end = chain_matches[idx + 1].start() if idx + 1 < len(chain_matches) else len(raw)
            fragment = raw[match.start() : end].strip()
            if fragment:
                output.append(fragment)
    return "\n".join(output)


def _reference_chain_matches(text: str) -> list[re.Match[str]]:
    matches = [
        match
        for match in _INLINE_NUMERIC_REFERENCE_MARKER_RE.finditer(str(text or ""))
        if _is_plausible_reference_number(match.group(1))
    ]
    # Converter page text may contain body citations before a collapsed
    # bibliography on the same page.  The generic PDF helper deliberately
    # rejects competing disjoint chains, but here the cache-page continuity
    # checks below provide the stronger guard we need.  Prefer the longest
    # ordered chain and, on a tie, one beginning at 1 and occurring later.
    candidates: list[list[int]] = []
    numbers = [int(match.group(1)) for match in matches]
    for start, first in enumerate(numbers):
        expected = int(first)
        positions = [start]
        for idx in range(start + 1, len(numbers)):
            if numbers[idx] == expected + 1:
                positions.append(idx)
                expected = numbers[idx]
        candidates.append(positions)
    if not candidates:
        return []
    candidates.sort(
        key=lambda positions: (
            len(positions),
            int(numbers[positions[0]] == 1),
            positions[0],
        ),
        reverse=True,
    )
    return [matches[index] for index in candidates[0]]


def _extract_cached_reference_markdown(md_path: Path) -> tuple[str, int]:
    """Recover a bibliography from validated converter page-cache text.

    Page-cache output preserves the original page order even when a later
    Markdown repair misplaced the body and references.  Only a chain beginning
    at reference 1 and containing at least five consecutive entries is trusted.
    """
    pages_dir = Path(md_path).expanduser().parent / ".conversion_cache" / "pages"
    try:
        page_dirs = sorted(
            (path for path in pages_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
    except Exception:
        return "", 0

    recovered_pages: list[tuple[int, str]] = []
    last_reference_number = 0
    for page_dir in page_dirs:
        page_file = page_dir / "page.txt"
        try:
            page_text = page_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        chain = _reference_chain_matches(page_text)
        if not chain:
            if recovered_pages:
                break
            continue
        first_number = int(chain[0].group(1))
        last_number = int(chain[-1].group(1))
        if not recovered_pages:
            if first_number != 1 or len(chain) < 5:
                continue
        elif first_number > last_reference_number + 2 or last_number <= last_reference_number:
            break

        trimmed = page_text[chain[0].start() :]
        trimmed = _drop_pdf_reference_running_lines(trimmed)
        trimmed = trim_reference_publisher_tail(trimmed)
        trimmed = _split_collapsed_numeric_reference_lines(trimmed).strip()
        if not trimmed:
            break
        try:
            page_no = int(page_dir.name)
        except Exception:
            break
        recovered_pages.append((page_no, trimmed))
        last_reference_number = last_number

    if not recovered_pages:
        return "", 0
    raw_lines: list[str] = ["## References"]
    for page_no, page_text in recovered_pages:
        raw_lines.extend(["", f"<!-- kb_page: {int(page_no)} -->", page_text])
    formatted = fix_references_format("\n".join(raw_lines)).strip()
    count = reference_markdown_entry_count(formatted)
    reference_map = extract_references_map_from_md(formatted)
    if count < 5 or not reference_map:
        return "", 0
    numbers = sorted(int(number) for number in reference_map if int(number) > 0)
    if numbers[0] != 1 or numbers != list(range(1, numbers[-1] + 1)):
        return "", 0
    return formatted, len(numbers)


def _extract_pdf_reference_markdown(pdf_path: Path) -> tuple[str, int]:
    if fitz is None:
        return "", 0
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return "", 0
    page_texts: list[tuple[int, str]] = []
    try:
        in_references = False
        for page_index in range(len(doc)):
            try:
                page = doc.load_page(page_index)
                plain_text = str(page.get_text("text") or "")
                ordered_text = reference_ordered_page_text(page, fallback_text=plain_text).strip()
                plain_numbers = _pdf_page_reference_start_numbers(plain_text)
                ordered_numbers = _pdf_page_reference_start_numbers(ordered_text)
                plain_chain = consecutive_reference_chain_positions(plain_numbers)
                ordered_chain = consecutive_reference_chain_positions(ordered_numbers)
                page_text = (
                    plain_text.strip()
                    if len(plain_chain) > len(ordered_chain)
                    else ordered_text
                )
            except Exception:
                page_text = ""
            if not page_text:
                continue
            has_heading = _pdf_page_has_references_heading_text(page_text)
            has_reference_block = _pdf_page_has_reference_block_text(page_text)
            if not in_references:
                if not has_heading and not has_reference_block:
                    continue
                in_references = True
                if not has_heading:
                    page_text = _trim_pdf_page_text_to_first_reference(page_text)
                else:
                    heading_match = re.search(
                        r"(?mi)^\s*(?:references?|bibliography|references?\s+and\s+links|literature\s+cited)\s*$",
                        page_text,
                    )
                    if heading_match:
                        page_text = page_text[heading_match.start() :]
            elif (
                not has_heading
                and not has_reference_block
                and not _pdf_reference_continuation_page_has_signal(page_text)
            ):
                break
            page_text = _drop_pdf_reference_running_lines(page_text)
            page_text = trim_reference_publisher_tail(page_text)
            if page_text:
                page_texts.append((page_index + 1, page_text))
    finally:
        try:
            doc.close()
        except Exception:
            pass
    if not page_texts:
        return "", 0
    raw_lines: list[str] = ["## References"]
    for page_no, page_text in page_texts:
        normalized = normalize_references_page_text(page_text)
        if not re.match(r"(?i)^#{1,6}\s+References\b", normalized.strip()):
            normalized = "## References\n\n" + normalized.lstrip()
        normalized = re.sub(r"(?im)^#{1,6}\s+References\b.*$", "## References", normalized.strip(), count=1)
        body_lines = normalized.splitlines()
        if body_lines and re.match(r"(?i)^#{1,6}\s+References\b", body_lines[0].strip()):
            body_lines = body_lines[1:]
        body_lines = [line for line in body_lines if str(line or "").strip()]
        if not body_lines:
            continue
        raw_lines.extend(["", f"<!-- kb_page: {int(page_no)} -->", *body_lines])
    formatted = fix_references_format("\n".join(raw_lines)).strip()
    return formatted.strip(), reference_markdown_entry_count(formatted)


def _reference_run_start_in_tail(lines: list[str]) -> int:
    numbered: list[tuple[int, int]] = []
    for idx, raw in enumerate(lines):
        st = str(raw or "").strip()
        inline_chain = _reference_chain_matches(st)
        if len(inline_chain) >= 3 and int(inline_chain[0].group(1)) == 1:
            return idx
        match = re.match(r"^\s*\[\s*(\d{1,4})\s*\]\s+", st)
        if match and _is_plausible_reference_number(match.group(1)):
            numbered.append((idx, int(match.group(1))))
    positions = consecutive_reference_chain_positions([number for _, number in numbered])
    if len(positions) >= 5 and int(numbered[positions[0]][1]) == 1:
        return int(numbered[positions[0]][0])
    return -1


def _post_reference_preserved_tail_start(lines: list[str], start: int) -> int:
    heading_re = re.compile(
        r"^(?:#{1,6}\s+)?(?:author\s+biograph(?:y|ies)|author\s+information|"
        r"about\s+the\s+authors?|biograph(?:y|ies))\s*$",
        re.IGNORECASE,
    )
    for idx in range(max(0, int(start)), len(lines)):
        st = str(lines[idx] or "").strip()
        is_biography_heading = bool(heading_re.match(st))
        is_biography_prose = bool(
            re.match(
                r"^[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,5}\s+"
                r"received\s+(?:his|her|their)\b.{0,120}\bdegree\b",
                st,
                re.IGNORECASE,
            )
        )
        if not (is_biography_heading or is_biography_prose):
            continue
        begin = idx
        while begin > start and not str(lines[begin - 1] or "").strip():
            begin -= 1
        if begin > start and PAGE_MARKER_RE.fullmatch(str(lines[begin - 1] or "").strip()):
            begin -= 1
        return begin
    return len(lines)


def _partition_misplaced_reference_tail(lines: list[str]) -> tuple[list[str], list[str]]:
    """Keep misplaced body before rebuilt refs and true post-ref material after."""
    raw_reference_start = _reference_run_start_in_tail(lines)
    if raw_reference_start < 0:
        return _clean_post_reference_body_tail(lines), []
    post_start = _post_reference_preserved_tail_start(lines, raw_reference_start + 1)
    body = _clean_post_reference_body_tail(lines[:raw_reference_start])
    post = [str(line or "") for line in lines[post_start:]] if post_start < len(lines) else []
    return body, post


def _replace_references_section(md: str, references_md: str) -> str:
    text = str(md or "")
    lines = text.splitlines()
    ref_idx = -1
    for idx, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match((line or "").strip()):
            ref_idx = idx
            break
    refs = str(references_md or "").strip()
    if not refs:
        return text
    if ref_idx < 0:
        # Fresh converter output may contain the complete body followed by a
        # collapsed numeric bibliography without ever emitting a References
        # heading.  Replace that raw run in place instead of appending a second
        # bibliography after author biographies.
        raw_reference_start = _reference_run_start_in_tail(lines)
        if raw_reference_start < 0:
            return (text.rstrip() + "\n\n" + refs).strip()
        post_start = _post_reference_preserved_tail_start(lines, raw_reference_start + 1)
        body_lines = lines[:raw_reference_start]
        post_reference_tail = lines[post_start:] if post_start < len(lines) else []
        refs_for_tail = _drop_leading_duplicate_reference_page_marker(
            refs,
            _last_page_marker_in_lines(body_lines),
        )
        parts = [*body_lines, "", *refs_for_tail.splitlines()]
        if post_reference_tail:
            parts.extend(["", *post_reference_tail])
        return "\n".join(parts).strip()

    start_idx = ref_idx
    stray_start = ref_idx
    stray_count = 0
    idx = ref_idx - 1
    while idx >= 0 and ref_idx - idx <= 12:
        st = (lines[idx] or "").strip()
        if not st:
            idx -= 1
            continue
        if _looks_markdown_reference_payload_line(st):
            stray_start = idx
            stray_count += 1
            idx -= 1
            continue
        break
    if stray_count > 0:
        start_idx = stray_start

    tail_idx = len(lines)
    ref_signal = 0
    non_ref_run = 0
    non_ref_start = -1

    def include_preceding_page_marker(index: int) -> int:
        begin = max(ref_idx + 1, int(index))
        probe = begin - 1
        while probe > ref_idx and not str(lines[probe] or "").strip():
            probe -= 1
        if probe > ref_idx and PAGE_MARKER_RE.fullmatch(str(lines[probe] or "").strip()):
            return probe
        return begin

    for idx in range(ref_idx + 1, len(lines)):
        st = (lines[idx] or "").strip()
        if not st:
            continue
        if REFERENCES_HEADING_RE.match(st):
            continue
        if _looks_markdown_reference_payload_line(st):
            ref_signal += 1
            non_ref_run = 0
            non_ref_start = -1
            continue
        if ref_signal >= 1 and _is_post_references_resume_heading_line(st):
            tail_idx = include_preceding_page_marker(idx)
            break
        if ref_signal >= 3 and _post_reference_body_heading_line(st):
            tail_idx = include_preceding_page_marker(non_ref_start if non_ref_start >= 0 else idx)
            break
        if ref_signal >= 8:
            if non_ref_run == 0:
                non_ref_start = idx
            non_ref_run += 1
            if non_ref_run >= 8 and non_ref_start >= 0:
                explicit_resume = next(
                    (
                        probe
                        for probe in range(non_ref_start, len(lines))
                        if _is_post_references_resume_heading_line(str(lines[probe] or "").strip())
                    ),
                    -1,
                )
                tail_idx = include_preceding_page_marker(
                    explicit_resume if explicit_resume >= 0 else non_ref_start
                )
                break
    tail_lines = lines[tail_idx:] if tail_idx < len(lines) else []
    if tail_lines and _post_reference_tail_should_precede_references(tail_lines):
        clean_tail, post_reference_tail = _partition_misplaced_reference_tail(tail_lines)
        refs_for_tail = _drop_leading_duplicate_reference_page_marker(
            refs,
            _last_page_marker_in_lines([*lines[:start_idx], *clean_tail]),
        )
        parts = [*lines[:start_idx], *clean_tail, "", *refs_for_tail.splitlines()]
        if post_reference_tail:
            parts.extend(["", *post_reference_tail])
        return "\n".join(parts).strip()

    parts = [*lines[:start_idx], *refs.splitlines()]
    if tail_idx < len(lines):
        parts.extend(["", *tail_lines])
    return "\n".join(parts).strip()


def _last_page_marker_before_references(md: str) -> int:
    text = str(md or "")
    ref_match = re.search(r"(?mi)^#{1,6}\s+References\s*$", text)
    ref_offset = int(ref_match.start()) if ref_match else len(text)
    last_page = 0
    for match in PAGE_MARKER_RE.finditer(text):
        if int(match.start()) >= ref_offset:
            continue
        try:
            last_page = int(match.group(1))
        except Exception:
            continue
    return last_page


def _last_page_marker_in_lines(lines: list[str]) -> int:
    last_page = 0
    for line in list(lines or []):
        for match in PAGE_MARKER_RE.finditer(str(line or "")):
            try:
                last_page = int(match.group(1))
            except Exception:
                continue
    return last_page


def _clean_post_reference_body_tail(lines: list[str]) -> list[str]:
    raw = [str(line or "") for line in list(lines or [])]

    def next_nonempty(start: int) -> str:
        for candidate in raw[start + 1 :]:
            if str(candidate or "").strip():
                return str(candidate or "").strip()
        return ""

    out: list[str] = []
    for idx, line in enumerate(raw):
        st = line.strip()
        if _looks_markdown_reference_payload_line(st):
            continue
        if PAGE_MARKER_RE.fullmatch(st) and _looks_markdown_reference_payload_line(next_nonempty(idx)):
            continue
        out.append(line)

    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    compact: list[str] = []
    blank_run = 0
    for line in out:
        if not line.strip():
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        compact.append(line)
    return compact


def _drop_leading_duplicate_reference_page_marker(references_md: str, page_no: int) -> str:
    if page_no <= 0:
        return str(references_md or "")
    pattern = rf"(?im)^(#{{1,6}}\s+References\s*)\n\s*<!--\s*kb_page:\s*{int(page_no)}\s*-->\s*\n*"
    return re.sub(pattern, r"\1\n\n", str(references_md or "").strip(), count=1)


def _looks_markdown_reference_payload_line(line: str) -> bool:
    st = str(line or "").strip()
    if not st or PAGE_MARKER_RE.fullmatch(st):
        return False
    if _looks_like_author_year_reference_text(st):
        return True
    match = re.match(r"^\s*(?:\[\s*(\d{1,4})\s*\]|(\d{1,4})[.)])\s+(.+)$", st)
    if not match:
        return False
    try:
        n = int(match.group(1) or match.group(2) or 0)
    except Exception:
        return False
    if n <= 0 or 1800 <= n <= 2099:
        return False
    body = str(match.group(3) or "").strip()
    if len(body) < 20:
        return False
    if re.match(
        r"^(?:abstract|introduction|background|related\s+work|method(?:s|ology)?|"
        r"experiment(?:s|al)?|results?|discussion|conclusions?|funding|"
        r"acknowledg(?:e)?ments?|appendix|supplementary|supplemental)\b",
        body,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(r"\b(?:18|19|20)\d{2}\b", body)
        or re.search(r"\b(?:doi\s*:|10\.\d{4,9}/|arxiv\s*:)", body, re.IGNORECASE)
        or re.search(
            r"\b(?:journal|proceedings?|proc\.?|opt\.|phys\.|nat\.|nature|science|"
            r"ieee|acm|appl\.|express|letters?|review|commun\.|photonics)\b",
            body,
            re.IGNORECASE,
        )
    )


def _post_reference_body_heading_line(line: str) -> bool:
    st = str(line or "").strip()
    if not st:
        return False
    st = re.sub(r"^#{1,6}\s+", "", st).strip()
    return bool(
        re.fullmatch(
            r"(?:\d+(?:\.\d+)*\.?\s+)?(?:conclusions?|method(?:s|ology)?|discussion|"
            r"challenges?|outlooks?|future\s+work|funding|acknowledg(?:e)?ments?)\s*[:.]?",
            st,
            re.IGNORECASE,
        )
    )


def _post_reference_tail_should_precede_references(lines: list[str]) -> bool:
    sample = "\n".join(str(line or "") for line in list(lines or [])[:30])
    if re.search(r"\b(?:supplementary|supplemental|appendix|appendices)\b", sample, re.IGNORECASE) or re.search(
        r"(?mi)^\s*#{1,6}\s+[A-Z]\.?\s*$",
        sample,
    ):
        return False
    return bool(
        re.search(
            r"(?mi)^\s*(?:#{1,6}\s+)?(?:\d+(?:\.\d+)*\.?\s+)?(?:conclusions?|method(?:s|ology)?|"
            r"discussion|challenges?|outlooks?|future\s+work|funding|acknowledg(?:e)?ments?)\b",
            sample,
        )
    )


def _backfill_references_from_pdf_text(md_text: str, md_path: Path, source_pdf_path: Path | str | None = None) -> tuple[str, bool]:
    text = str(md_text or "")
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(md_path)
    before_map = extract_references_map_from_md(text)
    before_extracted = len(before_map)
    before_missing = set(_reference_map_missing_numbers(before_map))
    references_md, recovered_count = _extract_pdf_reference_markdown(pdf_path) if pdf_path else ("", 0)
    cached_references_md, cached_count = _extract_cached_reference_markdown(md_path)
    recovered_truncated = _reference_map_has_short_truncated_entries(
        extract_references_map_from_md(references_md)
    )
    cached_truncated = _reference_map_has_short_truncated_entries(
        extract_references_map_from_md(cached_references_md)
    )
    if (
        cached_count >= 5
        and cached_count > recovered_count
        and not cached_truncated
    ) or (
        cached_count == recovered_count
        and recovered_truncated
        and not cached_truncated
    ):
        references_md, recovered_count = cached_references_md, cached_count
    recovered_map = extract_references_map_from_md(references_md)
    recovered_missing = set(_reference_map_missing_numbers(recovered_map))
    fills_existing_gap = bool(before_missing and len(recovered_missing) < len(before_missing))
    inflated_tail = _reference_map_has_inflated_tail(before_map, recovered_map)
    if not references_md or recovered_count < 5:
        return text, False
    if recovered_count < before_extracted and not (fills_existing_gap or inflated_tail):
        return text, False
    if before_missing and not fills_existing_gap:
        return text, False
    references_md = _drop_leading_duplicate_reference_page_marker(
        references_md,
        _last_page_marker_before_references(text),
    )
    fixed = _replace_references_section(text, references_md)
    return fixed, fixed != text


def repair_markdown_text(
    md_path: Path | str,
    md_text: str,
    *,
    issue_codes: list[str] | None = None,
    default_to_postprocess: bool = False,
    source_pdf_path: Path | str | None = None,
    allow_source_pdf_inference: bool = True,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    before_text = str(md_text or "")
    had_source_recovery = "<!-- kb_source_recovery:" in before_text
    before_metrics = _metric_view(path, before_text)
    source_repairs_enabled = bool(source_pdf_path) or bool(allow_source_pdf_inference)
    before_source_quality = _source_quality_view(
        path,
        before_text,
        before_metrics,
        source_pdf_path=source_pdf_path,
        allow_source_pdf_inference=allow_source_pdf_inference,
    )
    requested_codes = [str(code or "").strip().lower() for code in list(issue_codes or []) if str(code or "").strip()]
    before_issue_codes = _issue_codes_from_context(path, before_text, before_metrics, source_quality=before_source_quality)
    active_codes = requested_codes or before_issue_codes
    active_strategy_names: set[str] = set()
    for code in active_codes:
        for name in conversion_repair_strategy_for_issue(code).get("strategies") or []:
            active_strategy_names.add(str(name))
    if not active_strategy_names and default_to_postprocess:
        active_strategy_names.update({"postprocess_markdown"})

    text = before_text
    applied: list[str] = []

    if active_strategy_names:
        if "repair_detached_table_rows" in active_strategy_names:
            text, changed = _repair_detached_table_rows_only(text)
            if changed:
                applied.append("repair_detached_table_rows")

        if "repair_inline_math_boundaries" in active_strategy_names:
            text, changed = _repair_inline_math_boundaries_only(text)
            if changed:
                applied.append("repair_inline_math_boundaries")

        if source_repairs_enabled and "pdf_reference_backfill" in active_strategy_names:
            # Rebuild the bibliography before page-anchor repair. A complete
            # source-backed reference block often restores its own missing page
            # markers; doing this later can duplicate a provisional page
            # recovery and create an artificial marker gap.
            text, changed = _backfill_references_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("pdf_reference_backfill")

        if "normalize_detached_accents" in active_strategy_names:
            normalized = normalize_detached_accents(text)
            if normalized != text:
                text = normalized
                applied.append("normalize_detached_accents")

        if "ensure_page_anchor" in active_strategy_names:
            changed = False
            if source_repairs_enabled:
                text, changed = _recover_page_markers_from_pdf_text(text, path, source_pdf_path)
                if changed:
                    applied.append("recover_page_markers_from_pdf")
            if not changed:
                text, changed = _ensure_page_anchor(text)
                if changed:
                    applied.append("ensure_page_anchor")

        if source_repairs_enabled and "realign_page_markers_from_pdf" in active_strategy_names:
            text, changed = _realign_page_markers_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("realign_page_markers_from_pdf")

        if source_repairs_enabled and "realign_table_page_markers_from_pdf" in active_strategy_names:
            text, changed = _realign_table_page_markers_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("realign_table_page_markers_from_pdf")

        if "normalize_page_markers" in active_strategy_names:
            text, changed = _normalize_page_marker_sequence(text)
            if changed:
                applied.append("normalize_page_markers")

        if "balance_display_math" in active_strategy_names:
            text, changed = _balance_display_math(text)
            if changed:
                applied.append("balance_display_math")

        if "unwrap_prose_display_math" in active_strategy_names:
            text, changed = _unwrap_prose_dominant_display_math(text)
            if changed:
                applied.append("unwrap_prose_display_math")

        if "figure_metadata_captions" in active_strategy_names:
            text, changed = _inject_figure_metadata_captions(path, text)
            if changed:
                applied.append("figure_metadata_captions")

        if source_repairs_enabled and "pdf_text_captions" in active_strategy_names:
            text, changed = _inject_pdf_text_captions(path, text, source_pdf_path)
            if changed:
                applied.append("pdf_text_captions")

        if "repair_missing_image_links" in active_strategy_names:
            text, changed = _repair_missing_image_links(path, text)
            if changed:
                applied.append("repair_missing_image_links")

        if "abstract_heading_only" in active_strategy_names:
            text, changed = _insert_abstract_heading_only(text)
            if changed:
                applied.append("abstract_heading_only")

        if "move_early_references_to_end" in active_strategy_names:
            text, changed = _move_early_references_to_end(text)
            if changed:
                applied.append("move_early_references_to_end")

        if "normalize_markdown_tables" in active_strategy_names:
            text, changed = _normalize_markdown_tables_only(text)
            if changed:
                applied.append("normalize_markdown_tables")

        if source_repairs_enabled and "recover_ambiguous_table_pages" in active_strategy_names:
            text, changed = _recover_ambiguous_table_pages_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_ambiguous_table_pages")

        if "normalize_heading_levels" in active_strategy_names:
            text, changed = _normalize_heading_level_jumps(text)
            if changed:
                applied.append("normalize_heading_levels")

        if source_repairs_enabled and "restore_numbered_headings_from_pdf" in active_strategy_names:
            text, changed = _restore_numbered_headings_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("restore_numbered_headings_from_pdf")

        if "demote_malformed_numbered_headings" in active_strategy_names:
            source_changed = False
            if source_repairs_enabled:
                text, source_changed = _demote_source_proven_nonheading_numbered_headings(
                    text,
                    path,
                    source_pdf_path,
                )
            text, changed = _demote_malformed_numbered_formula_headings(text)
            if source_changed or changed:
                applied.append("demote_malformed_numbered_headings")

        if source_repairs_enabled and "recover_missing_source_pages" in active_strategy_names:
            text, changed = _recover_missing_source_pages_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_missing_source_pages")

        if source_repairs_enabled and "recover_source_prose_omissions" in active_strategy_names:
            text, changed = _recover_source_prose_omissions_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_source_prose_omissions")

        if source_repairs_enabled and (
            "recover_corrupted_source_pages" in active_strategy_names or had_source_recovery
        ):
            text, changed = _recover_corrupted_source_pages_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_corrupted_source_pages")

        if source_repairs_enabled and "recover_conversion_retry_pages" in active_strategy_names:
            text, changed = _recover_conversion_retry_pages_from_pdf_text(text, path, source_pdf_path)
            if changed:
                applied.append("recover_conversion_retry_pages")

        if "postprocess_markdown" in active_strategy_names:
            postprocessed = postprocess_markdown(text)
            if postprocessed != text:
                text = postprocessed
                applied.append("postprocess_markdown")

        if source_repairs_enabled and (
            "recover_corrupted_source_pages" in active_strategy_names
            or "postprocess_markdown" in active_strategy_names
            or had_source_recovery
        ):
            # Reference formatting may deliberately rebuild an unnumbered
            # bibliography and thereby discard a source-text recovery made
            # earlier in this transaction. Post-processing can also expose a
            # previously latent page-level corruption. Re-run the source-
            # proven repair after it; healthy pages are a no-op.
            text, changed = _recover_corrupted_source_pages_from_pdf_text(text, path, source_pdf_path)
            if changed and "recover_corrupted_source_pages" not in applied:
                applied.append("recover_corrupted_source_pages")

        if "promote_collapsed_review_headings" in active_strategy_names:
            text, changed = _promote_collapsed_review_headings(text)
            if changed:
                applied.append("promote_collapsed_review_headings")

        # Post-processing can remove leading comments when legacy files are oddly shaped;
        # preserve at least one stable reader anchor for quality-center repair.
        text, changed = _ensure_page_anchor(text)
        if changed:
            applied.append("ensure_page_anchor")

    changed_text = text != before_text
    if changed_text:
        after_metrics = _metric_view(path, text)
        after_source_quality = _source_quality_view(
            path,
            text,
            after_metrics,
            source_pdf_path=source_pdf_path,
            allow_source_pdf_inference=allow_source_pdf_inference,
        )
        after_issue_codes = _issue_codes_from_context(path, text, after_metrics, source_quality=after_source_quality)
    else:
        after_metrics = before_metrics
        after_source_quality = before_source_quality
        after_issue_codes = before_issue_codes
    regression_reasons = _regression_reasons(before_text, text) if changed_text else []
    if (
        "prose_dominant_display_math" in active_codes
        and "prose_dominant_display_math" not in after_issue_codes
        and "display_math_dropped" in regression_reasons
    ):
        # Removing a display-math block is the intended repair when that block
        # was deterministically classified and unwrapped as natural-language
        # prose. Other regression checks still protect real equations, tables,
        # figures, references, and overall content volume.
        regression_reasons = [reason for reason in regression_reasons if reason != "display_math_dropped"]
    if (
        "recover_ambiguous_table_pages" in applied
        and not markdown_table_issue_spans(text)
        and "kb_table_source_recovery" in text
    ):
        regression_reasons = [reason for reason in regression_reasons if reason != "tables_dropped"]
    transactional_target_codes = [
        code
        for code in active_codes
        if any(
            str(name or "") in active_strategy_names
            for name in conversion_repair_strategy_for_issue(code).get("strategies") or []
        )
    ]
    transactional_reasons = (
        _transactional_structure_reasons(before_issue_codes, after_issue_codes, transactional_target_codes)
        if changed_text
        else []
    )
    regression_reasons = _dedupe_codes([*regression_reasons, *transactional_reasons])
    if (
        changed_text
        and regression_reasons
        and "pdf_reference_backfill" in applied
        and set(regression_reasons).issubset({"reference_lines_dropped", "reference_index_regressed"})
        and _reference_map_has_inflated_tail(
            extract_references_map_from_md(before_text),
            extract_references_map_from_md(text),
        )
    ):
        regression_reasons = []
    safe_to_use = changed_text and not regression_reasons
    attempted_applied = list(applied)
    final_applied = list(applied)

    if changed_text and regression_reasons:
        fallback_text = before_text
        fallback_applied: list[str] = []
        fallback_issue_codes_current = list(before_issue_codes)
        fallback_detached_table_merge_accepted = False
        for label in applied:
            candidate = fallback_text
            changed = False
            if label == "repair_detached_table_rows":
                candidate, changed = _repair_detached_table_rows_only(fallback_text)
            elif label == "repair_inline_math_boundaries":
                candidate, changed = _repair_inline_math_boundaries_only(fallback_text)
            elif source_repairs_enabled and label == "recover_page_markers_from_pdf":
                candidate, changed = _recover_page_markers_from_pdf_text(fallback_text, path, source_pdf_path)
            elif label == "ensure_page_anchor":
                candidate, changed = _ensure_page_anchor(fallback_text)
            elif source_repairs_enabled and label == "realign_page_markers_from_pdf":
                candidate, changed = _realign_page_markers_from_pdf_text(fallback_text, path, source_pdf_path)
            elif source_repairs_enabled and label == "realign_table_page_markers_from_pdf":
                candidate, changed = _realign_table_page_markers_from_pdf_text(fallback_text, path, source_pdf_path)
            elif label == "normalize_page_markers":
                candidate, changed = _normalize_page_marker_sequence(fallback_text)
            elif label == "unwrap_prose_display_math":
                candidate, changed = _unwrap_prose_dominant_display_math(fallback_text)
            elif label == "figure_metadata_captions":
                candidate, changed = _inject_figure_metadata_captions(path, fallback_text)
            elif source_repairs_enabled and label == "pdf_text_captions":
                candidate, changed = _inject_pdf_text_captions(path, fallback_text, source_pdf_path)
            elif label == "repair_missing_image_links":
                candidate, changed = _repair_missing_image_links(path, fallback_text)
            elif label == "abstract_heading_only":
                candidate, changed = _insert_abstract_heading_only(fallback_text)
            elif label == "move_early_references_to_end":
                candidate, changed = _move_early_references_to_end(fallback_text)
            elif label == "normalize_markdown_tables":
                candidate, changed = _normalize_markdown_tables_only(fallback_text)
            elif source_repairs_enabled and label == "recover_ambiguous_table_pages":
                candidate, changed = _recover_ambiguous_table_pages_from_pdf_text(
                    fallback_text,
                    path,
                    source_pdf_path,
                )
            elif label == "normalize_heading_levels":
                candidate, changed = _normalize_heading_level_jumps(fallback_text)
            elif source_repairs_enabled and label == "restore_numbered_headings_from_pdf":
                candidate, changed = _restore_numbered_headings_from_pdf_text(
                    fallback_text,
                    path,
                    source_pdf_path,
                )
            elif label == "demote_malformed_numbered_headings":
                candidate = fallback_text
                source_changed = False
                if source_repairs_enabled:
                    candidate, source_changed = _demote_source_proven_nonheading_numbered_headings(
                        candidate,
                        path,
                        source_pdf_path,
                    )
                candidate, formula_changed = _demote_malformed_numbered_formula_headings(candidate)
                changed = source_changed or formula_changed
            elif label == "promote_collapsed_review_headings":
                candidate, changed = _promote_collapsed_review_headings(fallback_text)
            elif source_repairs_enabled and label == "recover_missing_source_pages":
                candidate, changed = _recover_missing_source_pages_from_pdf_text(fallback_text, path, source_pdf_path)
            elif source_repairs_enabled and label == "recover_source_prose_omissions":
                candidate, changed = _recover_source_prose_omissions_from_pdf_text(
                    fallback_text,
                    path,
                    source_pdf_path,
                )
            elif source_repairs_enabled and label == "recover_corrupted_source_pages":
                candidate, changed = _recover_corrupted_source_pages_from_pdf_text(
                    fallback_text,
                    path,
                    source_pdf_path,
                )
            elif source_repairs_enabled and label == "recover_conversion_retry_pages":
                candidate, changed = _recover_conversion_retry_pages_from_pdf_text(
                    fallback_text,
                    path,
                    source_pdf_path,
                )
            elif source_repairs_enabled and label == "pdf_reference_backfill":
                candidate, changed = _backfill_references_from_pdf_text(fallback_text, path, source_pdf_path)
            if not changed or candidate == fallback_text:
                continue
            step_reasons = _regression_reasons(fallback_text, candidate)
            if label == "repair_detached_table_rows":
                # This narrow transform only removes an empty placeholder and
                # joins its following content row to the established table.
                # A table-block count decrease therefore reflects a structural
                # merge, not loss of any non-empty cell.
                had_only_structural_table_drop = bool(step_reasons) and set(step_reasons) == {"tables_dropped"}
                step_reasons = [reason for reason in step_reasons if reason != "tables_dropped"]
            if label == "recover_missing_source_pages" and "kb_source_recovery" in candidate:
                # An authoritative page-text recovery can legitimately expose
                # long raw source lines that the Markdown analyzer warns about.
                # The page-level source gate below still has to confirm that
                # the target alignment/missing-page issue was resolved.
                step_reasons = [reason for reason in step_reasons if reason != "analyzer_warnings_increased"]
            if (
                label == "unwrap_prose_display_math"
                and "prose_dominant_display_math" not in _issue_codes_from_metrics(_metric_view(path, candidate))
            ):
                step_reasons = [reason for reason in step_reasons if reason != "display_math_dropped"]
            if (
                label == "recover_ambiguous_table_pages"
                and not markdown_table_issue_spans(candidate)
                and "kb_table_source_recovery" in candidate
            ):
                step_reasons = [reason for reason in step_reasons if reason != "tables_dropped"]
            step_target_codes = [
                code
                for code in active_codes
                if label in conversion_repair_strategy_for_issue(code).get("strategies", [])
            ]
            candidate_issue_codes: list[str] | None = None
            if any(code in _TRANSACTIONAL_STRUCTURE_ISSUES for code in step_target_codes):
                candidate_metrics = _metric_view(path, candidate)
                candidate_source_quality = _source_quality_view(
                    path,
                    candidate,
                    candidate_metrics,
                    source_pdf_path=source_pdf_path,
                    allow_source_pdf_inference=allow_source_pdf_inference,
                )
                candidate_issue_codes = _issue_codes_from_context(
                    path,
                    candidate,
                    candidate_metrics,
                    source_quality=candidate_source_quality,
                )
                step_reasons = _dedupe_codes(
                    [
                        *step_reasons,
                        *_transactional_structure_reasons(
                            fallback_issue_codes_current,
                            candidate_issue_codes,
                            step_target_codes,
                        ),
                    ]
                )
            if step_reasons:
                continue
            fallback_text = candidate
            fallback_applied.append(label)
            if label == "repair_detached_table_rows" and had_only_structural_table_drop:
                fallback_detached_table_merge_accepted = True
            if candidate_issue_codes is not None:
                fallback_issue_codes_current = candidate_issue_codes
        if fallback_text != before_text:
            fallback_reasons = _regression_reasons(before_text, fallback_text)
            if fallback_detached_table_merge_accepted:
                fallback_reasons = [reason for reason in fallback_reasons if reason != "tables_dropped"]
            if (
                "recover_ambiguous_table_pages" in fallback_applied
                and not markdown_table_issue_spans(fallback_text)
                and "kb_table_source_recovery" in fallback_text
            ):
                fallback_reasons = [reason for reason in fallback_reasons if reason != "tables_dropped"]
            fallback_metrics = _metric_view(path, fallback_text)
            fallback_source_quality = _source_quality_view(
                path,
                fallback_text,
                fallback_metrics,
                source_pdf_path=source_pdf_path,
                allow_source_pdf_inference=allow_source_pdf_inference,
            )
            fallback_issue_codes = _issue_codes_from_context(
                path,
                fallback_text,
                fallback_metrics,
                source_quality=fallback_source_quality,
            )
            if (
                "unwrap_prose_display_math" in fallback_applied
                and "prose_dominant_display_math" not in fallback_issue_codes
            ):
                fallback_reasons = [reason for reason in fallback_reasons if reason != "display_math_dropped"]
            fallback_reasons = _dedupe_codes(
                [
                    *fallback_reasons,
                    *_transactional_structure_reasons(
                        before_issue_codes,
                        fallback_issue_codes,
                        [
                            code
                            for code in active_codes
                            if any(
                                str(name or "") in set(fallback_applied)
                                for name in conversion_repair_strategy_for_issue(code).get("strategies") or []
                            )
                        ],
                    ),
                ]
            )
            if not fallback_reasons:
                text = fallback_text
                final_applied = fallback_applied
                after_metrics = fallback_metrics
                after_source_quality = fallback_source_quality
                after_issue_codes = fallback_issue_codes
                changed_text = True
                regression_reasons = []
                safe_to_use = True

    final_text = text if safe_to_use or not changed_text else before_text
    rolled_back = bool(final_text == before_text and attempted_applied and changed_text and regression_reasons)
    if rolled_back:
        final_applied = []

    return {
        "ok": bool(safe_to_use or not changed_text),
        "changed": bool(safe_to_use),
        "unsafe": bool(changed_text and regression_reasons),
        "path": str(path),
        "backup_path": "",
        "applied": final_applied,
        "attempted_applied": attempted_applied,
        "rolled_back": rolled_back,
        "attempted_issue_codes": active_codes,
        "issue_codes_before": before_issue_codes,
        "issue_codes_after": after_issue_codes if safe_to_use or not changed_text else before_issue_codes,
        "remaining_issue_codes": after_issue_codes if safe_to_use or not changed_text else before_issue_codes,
        "regression_reasons": regression_reasons,
        "before": before_metrics,
        "after": after_metrics if safe_to_use or not changed_text else before_metrics,
        "source_quality_before": before_source_quality,
        "source_quality_after": after_source_quality if safe_to_use or not changed_text else before_source_quality,
        "repaired_text": final_text,
    }


def repair_markdown_quality(
    md_path: Path | str,
    *,
    issue_codes: list[str] | None = None,
    create_backup: bool = True,
    source_pdf_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    before_text = path.read_text(encoding="utf-8", errors="replace")
    requested_codes = {str(code or "").strip().lower() for code in list(issue_codes or [])}
    pdf_path = Path(source_pdf_path).expanduser() if source_pdf_path else _guess_source_pdf_for_md(path)
    if pdf_path and requested_codes & {
        "collapsed_table_rows",
        "ambiguous_table_break_rows",
        "analyzer_warnings",
    }:
        _prepare_ambiguous_table_page_assets(path, before_text, pdf_path)
    result = repair_markdown_text(
        path,
        before_text,
        issue_codes=issue_codes,
        default_to_postprocess=True,
        source_pdf_path=source_pdf_path,
    )
    text = str(result.get("repaired_text") or before_text)
    safe_to_write = bool(result.get("changed"))

    backup_path = ""
    if safe_to_write:
        if create_backup:
            backup = path.with_suffix(path.suffix + ".bak")
            try:
                shutil.copyfile(path, backup)
                backup_path = str(backup)
            except Exception:
                backup_path = ""
        path.write_text(text, encoding="utf-8")

    out = dict(result)
    out["backup_path"] = backup_path
    out.pop("repaired_text", None)
    return out
