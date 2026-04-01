from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import TypedDict
import threading

from kb import runtime_state as RUNTIME


class SourceBlock(TypedDict, total=False):
    doc_id: str
    block_id: str
    anchor_id: str
    kind: str
    heading_path: str
    order_index: int
    line_start: int
    line_end: int
    text: str
    raw_text: str
    number: int
    figure_id: str
    figure_ident: str
    paper_figure_number: int
    figure_role: str
    linked_figure_block_id: str
    asset_name: str
    asset_name_alias: str
    caption_text: str


_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_MD_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_EQ_NUMBER_RE = re.compile(
    r"(?:\b(?:eq|equation|formula)\s*[#(]?\s*|[\(])(\d{1,4})(?:\s*[)])",
    re.IGNORECASE,
)
_EQ_TAG_RE = re.compile(r"\\tag\{\s*(\d{1,4})\s*\}", re.IGNORECASE)
_FIG_NUMBER_RE = re.compile(r"(?:\bfig(?:ure)?\.?\s*#?\s*|图\s*|第\s*)(\d{1,4})(?:\s*张图)?", re.IGNORECASE)
_INLINE_EQ_RE = re.compile(r"\$[^$]{1,280}\$")
_DISPLAY_EQ_RE = re.compile(r"\$\$[\s\S]{1,6000}\$\$")
_EQ_ENV_RE = re.compile(r"\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}", re.IGNORECASE)
_TEX_CMD_RE = re.compile(r"\\[a-zA-Z]{2,}")
_STRUCT_CITE_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*[A-Za-z0-9_-]{4,24}\s*:\s*\d{1,4}\s*\]\]",
    re.IGNORECASE,
)
_STRUCT_CITE_SINGLE_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*[A-Za-z0-9_-]{4,24}(?:\s*:\s*\d{1,4})?\s*\](?!\])",
    re.IGNORECASE,
)
_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_CJK_SEQ_RE = re.compile(r"[\u4e00-\u9fff]{1,}")


def _anchor_id(kind: str, index: int) -> str:
    prefix_map = {
        "heading": "hd",
        "paragraph": "p",
        "equation": "eq",
        "figure": "fg",
        "list_item": "li",
        "blockquote": "bq",
        "code": "cd",
        "table": "tb",
    }
    prefix = prefix_map.get(str(kind or "").strip().lower(), "b")
    return f"{prefix}_{int(max(1, index)):05d}"


def normalize_inline_markdown(input_text: str) -> str:
    text = str(input_text or "")
    if not text:
        return ""
    text = _STRUCT_CITE_RE.sub(" ", text)
    text = _STRUCT_CITE_SINGLE_RE.sub(" ", text)
    text = _SID_INLINE_RE.sub(" ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_match_text(input_text: str) -> str:
    return normalize_inline_markdown(input_text).lower()


def tokenize_match_text(input_text: str) -> list[str]:
    src = normalize_match_text(input_text)
    if not src:
        return []
    out: list[str] = []
    out.extend(_LATIN_TOKEN_RE.findall(src))
    for seq in _CJK_SEQ_RE.findall(src):
        if len(seq) <= 2:
            out.append(seq)
            continue
        for idx in range(0, len(seq) - 1):
            out.append(seq[idx : idx + 2])
    return out


def has_equation_signal(text: str) -> bool:
    src = str(text or "")
    if not src:
        return False
    low = src.lower()
    if "$$" in src:
        return True
    if "\\begin{equation" in low or "\\[" in src:
        return True
    if _INLINE_EQ_RE.search(src):
        return True
    if _TEX_CMD_RE.search(src) and re.search(r"[=^_]", src):
        return True
    return False


def is_display_equation_block(text: str) -> bool:
    src = str(text or "").strip()
    if not src:
        return False
    low = src.lower()
    if _DISPLAY_EQ_RE.search(src):
        return True
    if _EQ_ENV_RE.search(src):
        return True
    if "\\[" in src and "\\]" in src:
        return True

    # Fallback: allow concise single-line formula-like blocks.
    if not has_equation_signal(src):
        return False
    non_empty_lines = [line for line in src.splitlines() if str(line).strip()]
    if len(non_empty_lines) > 2:
        return False

    formula_marker = bool(
        ("=" in src)
        or ("\\tag{" in low)
        or ("\\sum" in src)
        or ("\\int" in src)
        or ("\\frac" in src)
        or ("\\mathcal" in src)
        or ("\\mathbf" in src)
    )
    if not formula_marker:
        return False

    clean = normalize_inline_markdown(src)
    if not clean:
        return False
    latin_words = re.findall(r"[A-Za-z]{3,}", clean)
    cjk_words = re.findall(r"[\u4e00-\u9fff]{2,}", clean)
    # Too many natural-language words usually means paragraph with inline math.
    return (len(latin_words) + len(cjk_words)) <= 12


def extract_equation_number(text: str) -> int:
    src = str(text or "")
    if not src:
        return 0
    m_tag = _EQ_TAG_RE.search(src)
    if m_tag:
        try:
            number = int(str(m_tag.group(1) or "0"))
        except Exception:
            number = 0
        if number > 0:
            return number
    m = _EQ_NUMBER_RE.search(src)
    if not m:
        return 0
    try:
        number = int(str(m.group(1) or "0"))
    except Exception:
        return 0
    return number if number > 0 else 0


def extract_figure_number(text: str) -> int:
    src = str(text or "")
    if not src:
        return 0
    m = _FIG_NUMBER_RE.search(src)
    if not m:
        return 0
    try:
        number = int(str(m.group(1) or "0"))
    except Exception:
        return 0
    return number if number > 0 else 0


def _formula_tokens(text: str) -> list[str]:
    src = str(text or "")
    if not src:
        return []
    out: list[str] = []
    out.extend(x.lower() for x in (_TEX_CMD_RE.findall(src) or []))
    out.extend(x.lower() for x in (re.findall(r"[A-Za-z](?:_[A-Za-z0-9]+)?(?:\^[A-Za-z0-9]+)?", src) or []))
    out.extend(re.findall(r"\b\d{1,4}\b", src) or [])
    return out


def _token_overlap_score(a: str, b: str) -> float:
    ta = set(tokenize_match_text(a))
    tb = set(tokenize_match_text(b))
    if not ta or not tb:
        return 0.0
    overlap = sum(1 for token in ta if token in tb)
    return overlap / math.sqrt(max(1, len(ta)) * max(1, len(tb)))


def _formula_overlap_score(a: str, b: str) -> float:
    ta = set(_formula_tokens(a))
    tb = set(_formula_tokens(b))
    if not ta or not tb:
        return 0.0
    overlap = sum(1 for token in ta if token in tb)
    return overlap / math.sqrt(max(1, len(ta)) * max(1, len(tb)))


def _heading_overlap_score(needle: str, heading: str) -> float:
    n = normalize_match_text(needle)
    h = normalize_match_text(heading)
    if not n or not h:
        return 0.0
    score = 0.0
    if h in n or n in h:
        score += 0.65
    score += _token_overlap_score(n, h)
    return score


def _equation_number_score(block: SourceBlock, target_number: int) -> float:
    if int(target_number or 0) <= 0:
        return 0.0
    try:
        block_number = int(block.get("number") or 0)
    except Exception:
        block_number = 0
    if block_number > 0 and block_number == int(target_number):
        return 1.15
    text = normalize_match_text(str(block.get("text") or ""))
    if not text:
        return 0.0
    if re.search(rf"\(\s*{int(target_number)}\s*\)", text):
        return 1.0
    if re.search(rf"\beq(?:uation)?\s*\.?\s*{int(target_number)}\b", text, flags=re.IGNORECASE):
        return 0.95
    return 0.0


def _figure_number_score(block: SourceBlock, target_number: int) -> float:
    if int(target_number or 0) <= 0:
        return 0.0
    text = normalize_match_text(str(block.get("text") or block.get("raw_text") or ""))
    if not text:
        return 0.0
    try:
        block_number = int(block.get("paper_figure_number") or block.get("number") or 0)
    except Exception:
        block_number = 0
    if block_number > 0 and block_number == int(target_number):
        score = 1.08
        if str(block.get("figure_role") or "").strip().lower() == "caption":
            score += 0.12
        return score
    if re.search(rf"\bfig(?:ure)?\.?\s*#?\s*{int(target_number)}\b", text, flags=re.IGNORECASE):
        return 0.96
    return 0.0


def doc_id_for_path(md_path: Path | str) -> str:
    raw = str(md_path or "").strip().replace("\\", "/").lower()
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:12]


def _load_figure_identity_by_asset(md_path: Path | str) -> dict[str, dict]:
    path = Path(str(md_path or "")).expanduser()
    assets_dir = path.parent / "assets"
    if not assets_dir.exists():
        return {}

    figure_index_path = assets_dir / "figure_index.json"
    rows: list[dict] = []
    if figure_index_path.exists():
        try:
            payload = json.loads(figure_index_path.read_text(encoding="utf-8"))
            figures = payload.get("figures") if isinstance(payload, dict) else None
            if isinstance(figures, list):
                rows = [dict(item) for item in figures if isinstance(item, dict)]
        except Exception:
            rows = []

    if not rows:
        for meta_path in sorted(assets_dir.glob("page_*_fig_index.json")):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            figures = payload.get("figures") if isinstance(payload, dict) else None
            if not isinstance(figures, list):
                continue
            rows.extend(dict(item) for item in figures if isinstance(item, dict))

    out: dict[str, dict] = {}
    for row in rows:
        raw_name = str(row.get("asset_name_raw") or row.get("asset_name") or "").strip()
        alias_name = str(row.get("asset_name_alias") or "").strip()
        if raw_name:
            out[raw_name] = dict(row)
        if alias_name:
            out[alias_name] = dict(row)
    return out


def build_source_blocks(
    md_text: str,
    *,
    doc_id: str,
    figure_meta_by_asset: dict[str, dict] | None = None,
) -> list[SourceBlock]:
    lines = str(md_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return []

    blocks: list[SourceBlock] = []
    counters: dict[str, int] = {}
    heading_stack: list[tuple[int, str]] = []
    para_buf: list[str] = []
    para_start = 0
    table_buf: list[str] = []
    table_start = 0
    code_buf: list[str] = []
    code_start = 0
    in_fence = False
    fence_mark = ""
    order_index = 0
    pending_figure_context: dict[str, object] | None = None

    def heading_path() -> str:
        return " / ".join(item[1] for item in heading_stack if item[1]).strip()

    def push(
        kind: str,
        raw_text: str,
        *,
        line_start: int,
        line_end: int,
        number: int = 0,
        extras: dict[str, object] | None = None,
    ) -> SourceBlock | None:
        nonlocal order_index
        clean = normalize_inline_markdown(raw_text)
        if kind not in {"heading", "figure"} and len(clean) < 4:
            return None
        counters[kind] = int(counters.get(kind, 0) or 0) + 1
        order_index += 1
        block: SourceBlock = {
            "doc_id": doc_id,
            "block_id": f"blk_{doc_id}_{order_index:05d}",
            "anchor_id": _anchor_id(kind, counters[kind]),
            "kind": kind,
            "heading_path": heading_path(),
            "order_index": order_index,
            "line_start": int(max(1, line_start)),
            "line_end": int(max(line_start, line_end)),
            "text": clean[:4000],
            "raw_text": str(raw_text or "")[:12000],
        }
        if int(number or 0) > 0:
            block["number"] = int(number)
            if kind == "figure":
                block["paper_figure_number"] = int(number)
        if extras:
            for key, value in extras.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                block[key] = value
        blocks.append(block)
        return block

    def flush_paragraph(end_line: int) -> None:
        nonlocal para_buf, para_start, pending_figure_context
        if not para_buf:
            return
        raw = "\n".join(para_buf).strip()
        para_buf = []
        if not raw:
            return
        is_equation = is_display_equation_block(raw)
        extras: dict[str, object] | None = None
        if pending_figure_context:
            pending_number = int(pending_figure_context.get("paper_figure_number") or pending_figure_context.get("number") or 0)
            caption_number = extract_figure_number(raw)
            if pending_number > 0 and caption_number == pending_number:
                extras = {
                    "figure_id": str(pending_figure_context.get("figure_id") or "").strip(),
                    "figure_ident": str(pending_figure_context.get("figure_ident") or "").strip(),
                    "paper_figure_number": pending_number,
                    "figure_role": "caption",
                    "linked_figure_block_id": str(pending_figure_context.get("figure_block_id") or "").strip(),
                    "asset_name": str(pending_figure_context.get("asset_name") or "").strip(),
                    "asset_name_alias": str(pending_figure_context.get("asset_name_alias") or "").strip(),
                    "caption_text": normalize_inline_markdown(raw)[:1200],
                }
            pending_figure_context = None
        push(
            "equation" if is_equation else "paragraph",
            raw,
            line_start=para_start,
            line_end=end_line,
            number=(extract_equation_number(raw) if is_equation else 0),
            extras=extras,
        )

    def flush_table(end_line: int) -> None:
        nonlocal table_buf, table_start
        if not table_buf:
            return
        raw = "\n".join(table_buf).strip()
        table_buf = []
        if raw:
            push("table", raw, line_start=table_start, line_end=end_line)

    def flush_code(end_line: int) -> None:
        nonlocal code_buf, code_start
        if not code_buf:
            return
        raw = "\n".join(code_buf).strip()
        code_buf = []
        if raw:
            push("code", raw, line_start=code_start, line_end=end_line)

    for line_no, raw in enumerate(lines, start=1):
        line = str(raw or "")
        fence = _MD_FENCE_RE.match(line)
        if in_fence:
            if fence and str(fence.group(1) or "").startswith(fence_mark):
                flush_code(line_no)
                in_fence = False
                fence_mark = ""
                continue
            code_buf.append(line)
            continue
        if fence:
            flush_paragraph(max(1, line_no - 1))
            flush_table(max(1, line_no - 1))
            in_fence = True
            fence_mark = str(fence.group(1) or "")[:3]
            code_buf = []
            code_start = line_no + 1
            continue

        heading = _MD_HEADING_RE.match(line)
        if heading:
            flush_paragraph(max(1, line_no - 1))
            flush_table(max(1, line_no - 1))
            pending_figure_context = None
            level = len(str(heading.group(1) or ""))
            text = normalize_inline_markdown(str(heading.group(2) or ""))
            if text:
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                push("heading", text, line_start=line_no, line_end=line_no)
            continue

        if _MD_TABLE_RE.match(line):
            flush_paragraph(max(1, line_no - 1))
            pending_figure_context = None
            if not table_buf:
                table_start = line_no
            table_buf.append(line)
            continue
        flush_table(max(1, line_no - 1))

        image_match = _MD_IMAGE_RE.search(line)
        if image_match:
            flush_paragraph(max(1, line_no - 1))
            alt_text = str(image_match.group(1) or "").strip()
            raw_path = str(image_match.group(2) or "").strip().strip('"').strip("'")
            asset_name = Path(raw_path).name
            meta = dict((figure_meta_by_asset or {}).get(asset_name) or {})
            figure_number = 0
            try:
                figure_number = int(meta.get("paper_figure_number") or meta.get("fig_no") or 0)
            except Exception:
                figure_number = 0
            if figure_number <= 0:
                figure_number = extract_figure_number(alt_text or line)
            generic_alt = bool(re.fullmatch(r"fig(?:ure)?\.?", alt_text, flags=re.IGNORECASE))
            figure_text = alt_text or str(line or "").strip()
            if (not alt_text or generic_alt) and figure_number > 0:
                figure_text = f"Figure {figure_number}"
            if figure_text:
                figure_block = push(
                    "figure",
                    figure_text,
                    line_start=line_no,
                    line_end=line_no,
                    number=figure_number,
                    extras={
                        "figure_id": str(meta.get("figure_id") or "").strip(),
                        "figure_ident": str(meta.get("figure_ident") or meta.get("fig_ident") or "").strip(),
                        "paper_figure_number": figure_number if figure_number > 0 else None,
                        "asset_name": asset_name,
                        "asset_name_alias": str(meta.get("asset_name_alias") or "").strip(),
                        "caption_text": str(meta.get("caption") or "").strip(),
                    },
                )
                pending_figure_context = {
                    "figure_block_id": str((figure_block or {}).get("block_id") or "").strip(),
                    "figure_id": str(meta.get("figure_id") or "").strip(),
                    "figure_ident": str(meta.get("figure_ident") or meta.get("fig_ident") or "").strip(),
                    "paper_figure_number": figure_number,
                    "asset_name": asset_name,
                    "asset_name_alias": str(meta.get("asset_name_alias") or "").strip(),
                }
            continue

        if not line.strip():
            flush_paragraph(max(1, line_no - 1))
            continue

        list_match = _MD_LIST_RE.match(line)
        if list_match:
            flush_paragraph(max(1, line_no - 1))
            pending_figure_context = None
            push("list_item", str(list_match.group(1) or ""), line_start=line_no, line_end=line_no)
            continue

        quote_match = _MD_BLOCKQUOTE_RE.match(line)
        if quote_match:
            flush_paragraph(max(1, line_no - 1))
            pending_figure_context = None
            push("blockquote", str(quote_match.group(1) or ""), line_start=line_no, line_end=line_no)
            continue

        if not para_buf:
            para_start = line_no
        para_buf.append(line)

    flush_table(len(lines))
    flush_paragraph(len(lines))
    if in_fence:
        flush_code(len(lines))
    return blocks


def source_blocks_to_reader_anchors(blocks: list[SourceBlock]) -> list[dict]:
    out: list[dict] = []
    for block in blocks or []:
        rec = {
            "anchor_id": str(block.get("anchor_id") or "").strip(),
            "block_id": str(block.get("block_id") or "").strip(),
            "kind": str(block.get("kind") or "").strip(),
            "heading_path": str(block.get("heading_path") or "").strip(),
            "text": str(block.get("text") or ""),
            "line_start": int(block.get("line_start") or 0),
            "line_end": int(block.get("line_end") or 0),
        }
        if int(block.get("number") or 0) > 0:
            rec["number"] = int(block.get("number") or 0)
        if str(block.get("figure_id") or "").strip():
            rec["figure_id"] = str(block.get("figure_id") or "").strip()
        if int(block.get("paper_figure_number") or 0) > 0:
            rec["paper_figure_number"] = int(block.get("paper_figure_number") or 0)
        if str(block.get("figure_role") or "").strip():
            rec["figure_role"] = str(block.get("figure_role") or "").strip()
        out.append(rec)
    return out


def load_source_blocks(md_path: Path | str, *, md_text: str | None = None) -> list[SourceBlock]:
    path = Path(str(md_path or "")).expanduser()
    if md_text is None:
        return load_source_blocks_cached(path)
    return build_source_blocks(
        md_text,
        doc_id=doc_id_for_path(path),
        figure_meta_by_asset=_load_figure_identity_by_asset(path),
    )


def load_source_blocks_cached(md_path: Path | str) -> list[SourceBlock]:
    """
    Cache SourceBlocks in-process keyed by path+mtime+size.
    This is a hot path for paper-guide targeted scans, and re-parsing full markdown
    on every request can add noticeable latency on large docs.
    """
    path = Path(str(md_path or "")).expanduser()
    figure_index_path = path.parent / "assets" / "figure_index.json"
    try:
        st = path.stat()
        key = f"{str(path)}|{int(st.st_mtime)}|{int(st.st_size)}"
        try:
            fig_st = figure_index_path.stat()
            key += f"|{int(fig_st.st_mtime)}|{int(fig_st.st_size)}"
        except Exception:
            pass
    except Exception:
        key = str(path)

    lock: threading.Lock = getattr(RUNTIME, "CACHE_LOCK", threading.Lock())
    cache: dict = getattr(RUNTIME, "CACHE", {})
    with lock:
        bucket = cache.setdefault("source_blocks_v1", {})
        v = bucket.get(key)
    if isinstance(v, list) and v:
        return v

    try:
        md_text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        md_text = ""
    figure_meta_by_asset = _load_figure_identity_by_asset(path)
    blocks = build_source_blocks(
        md_text,
        doc_id=doc_id_for_path(path),
        figure_meta_by_asset=figure_meta_by_asset,
    ) if md_text else []

    with lock:
        bucket = cache.setdefault("source_blocks_v1", {})
        bucket[key] = blocks
        if len(bucket) > 120:
            # Drop oldest insertion order (Py>=3.7).
            for k in list(bucket.keys())[: max(1, len(bucket) // 2)]:
                bucket.pop(k, None)
    return blocks


def match_source_blocks(
    blocks: list[SourceBlock],
    *,
    snippet: str = "",
    heading_path: str = "",
    prefer_kind: str = "",
    target_number: int = 0,
    limit: int = 3,
    score_floor: float | None = None,
) -> list[dict]:
    query = normalize_inline_markdown(snippet)
    heading = normalize_inline_markdown(heading_path)
    prefer_kind_norm = str(prefer_kind or "").strip().lower()
    if (not query) and (not heading) and int(target_number or 0) <= 0:
        return []

    ranked: list[dict] = []
    query_norm = normalize_match_text(query)
    for block in blocks or []:
        text = str(block.get("text") or "")
        if not text:
            continue
        block_kind = str(block.get("kind") or "").strip().lower()
        score = 0.0
        if query:
            score += _token_overlap_score(query, text)
            text_norm = normalize_match_text(text)
            if query_norm and text_norm:
                if text_norm.find(query_norm) >= 0:
                    score += 0.78
                head = query_norm[: min(72, len(query_norm))]
                if len(head) >= 20 and text_norm.find(head) >= 0:
                    score += 0.24
            if has_equation_signal(query) or has_equation_signal(text):
                score += 0.82 * _formula_overlap_score(query, text)
        if heading:
            score += 0.34 * _heading_overlap_score(heading, str(block.get("heading_path") or ""))
        if int(target_number or 0) > 0:
            if prefer_kind_norm == "figure" or block_kind == "figure" or str(block.get("figure_role") or "").strip().lower() == "caption":
                score += _figure_number_score(block, int(target_number))
            else:
                score += _equation_number_score(block, int(target_number))
        if prefer_kind_norm and block_kind == prefer_kind_norm:
            score += 0.18
        if has_equation_signal(query) and block_kind == "equation":
            score += 0.24
        ranked.append({"block": block, "score": float(score)})

    ranked.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    if not ranked:
        return []

    if int(target_number or 0) > 0:
        floor = 0.28
    elif has_equation_signal(query):
        floor = 0.18
    elif query:
        floor = 0.22
    else:
        floor = 0.42
    if score_floor is not None:
        try:
            floor = max(0.0, float(score_floor))
        except Exception:
            pass

    out: list[dict] = []
    seen: set[str] = set()
    for row in ranked:
        score = float(row.get("score") or 0.0)
        block = row.get("block")
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("block_id") or "").strip()
        if (not block_id) or (block_id in seen) or score < floor:
            continue
        seen.add(block_id)
        out.append({"score": score, "block": block})
        if len(out) >= max(1, int(limit)):
            break
    return out


def split_answer_segments(answer_markdown: str) -> list[dict]:
    lines = str(answer_markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: list[dict] = []
    buf: list[str] = []
    segment_kind = "paragraph"
    in_fence = False

    def push(kind: str, raw_text: str) -> None:
        raw = str(raw_text or "").strip()
        if not raw:
            return
        text = normalize_inline_markdown(raw)
        if len(text) < 10:
            return
        segments.append(
            {
                "kind": kind,
                "raw_text": raw[:4000],
                "raw_markdown": raw[:4000],
                "text": text[:1600],
                "snippet_key": normalize_match_text(text[:360]),
            }
        )

    def flush() -> None:
        nonlocal buf, segment_kind
        if not buf:
            return
        push(segment_kind, "\n".join(buf))
        buf = []
        segment_kind = "paragraph"

    for raw in lines:
        line = str(raw or "")
        if _MD_FENCE_RE.match(line):
            if in_fence:
                in_fence = False
                flush()
            else:
                flush()
                in_fence = True
            continue
        if in_fence:
            continue
        if not line.strip():
            flush()
            continue
        if _MD_HEADING_RE.match(line):
            flush()
            continue
        list_match = _MD_LIST_RE.match(line)
        if list_match:
            flush()
            push("list_item", str(list_match.group(1) or ""))
            continue
        if _MD_TABLE_RE.match(line):
            flush()
            push("table", line)
            continue
        quote_match = _MD_BLOCKQUOTE_RE.match(line)
        if quote_match:
            flush()
            push("blockquote", str(quote_match.group(1) or ""))
            continue
        segment_kind = "paragraph"
        buf.append(line)
    flush()
    return segments
