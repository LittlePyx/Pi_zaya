from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

from kb import runtime_state as RUNTIME
from kb.bg_queue_state import (
    begin_next_task_or_idle as bg_begin_next_task_or_idle,
    cancel_all as bg_cancel_all,
    enqueue as bg_enqueue,
    finish_task as bg_finish_task,
    remove_queued_tasks_for_pdf as bg_remove_queued_tasks_for_pdf,
    should_cancel as bg_should_cancel,
    snapshot as bg_snapshot,
    update_page_progress as bg_update_page_progress,
)
from kb.chat_store import ChatStore
from kb.file_ops import _resolve_md_output_paths
from kb.llm import DeepSeekChat
from kb.pdf_tools import run_pdf_to_md
from kb.reference_index import load_reference_index, resolve_reference_entry
from kb.retrieval_engine import (
    _collect_doc_overview_snippets,
    _deep_read_md_for_context,
    _enrich_grouped_refs_with_llm_pack,
    _extract_md_headings,
    _group_hits_by_doc_for_refs,
    _search_hits_with_fallback,
    _top_heading,
)
from kb.retrieval_heuristics import (
    _is_probably_bad_heading,
    _quick_answer_for_prompt,
    _should_bypass_kb_retrieval,
    _should_prioritize_attached_image,
)
from kb.store import load_all_chunks
from kb.retriever import BM25Retriever
from kb.source_blocks import (
    extract_equation_number,
    has_equation_signal,
    load_source_blocks,
    match_source_blocks,
    normalize_inline_markdown,
    normalize_match_text,
    split_answer_segments,
)
from ui.chat_widgets import _normalize_math_markdown
from ui.strings import S

_LIVE_ASSISTANT_PREFIX = "__KB_LIVE_TASK__:"
_CITE_SINGLE_BRACKET_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\](?!\])",
    re.IGNORECASE,
)
_CITE_SID_ONLY_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*\]\]",
    re.IGNORECASE,
)
_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_SID_INLINE_RE = re.compile(r"\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]", re.IGNORECASE)
_SID_HEADER_LINE_RE = re.compile(
    r"(?im)^\s*\[\d{1,3}\]\s*\[\s*SID\s*:\s*[A-Za-z0-9_-]{4,24}\s*\][^\n]*\n?",
    re.IGNORECASE,
)
_VISION_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
_DOC_FIGURE_CACHE_LOCK = threading.Lock()
_DOC_FIGURE_CACHE: dict[str, tuple[float, list[dict]]] = {}
_DOC_FIGURE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MD_IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_FIG_NUMBER_PATTERNS = (
    re.compile(r"\bfig(?:ure)?\.?\s*(\d{1,3})\b", flags=re.IGNORECASE),
    re.compile(r"\bfigure\s*#?\s*(\d{1,3})\b", flags=re.IGNORECASE),
    re.compile(r"图\s*([0-9]{1,3})\b"),
    re.compile(r"第\s*([0-9]{1,3})\s*张图"),
    re.compile(r"(?:^|[_\-/])fig(?:ure)?[_\-]?(\d{1,3})(?:\D|$)", flags=re.IGNORECASE),
)
_DISPLAY_EQ_SEG_RE = re.compile(r"\$\$[\s\S]{1,6000}\$\$")
_EQ_ENV_SEG_RE = re.compile(r"\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}", re.IGNORECASE)
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_CJK_WORD_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_FORMULA_TOKEN_RE = re.compile(r"(\\[a-zA-Z]{2,}|[=^_]|\\sum|\\int|\\frac|\\mathcal|\\mathbf)")
_FORMULA_CMD_RE = re.compile(r"\\[a-zA-Z]{2,}")
_SEG_SENT_SPLIT_RE = re.compile(r"(?<=[\u3002\uff01\uff1f.!;:\uff1b\uff1a])\s+")
_CLAIM_EXPERIMENT_HINT_RE = re.compile(
    r"(\bexperiment(?:al)?\b|\bsetup\b|\bresult(?:s)?\b|\bablation\b|\bbaseline\b|\bcomparison\b|"
    r"\bground[ -]?truth\b|\bpose\b|\bcamera\b|\btrain(?:ing)?\b|\bevaluation\b|\bdataset\b|"
    r"\bmetric\b|\bsota\b|实验|对比|基线|位姿|真值|训练|数据集|指标)"
    ,
    re.IGNORECASE,
)
_CLAIM_METHOD_HINT_RE = re.compile(
    r"(\bmethod\b|\bapproach\b|\bpipeline\b|\bframework\b|\barchitecture\b|\bmodule\b|\bnetwork\b|"
    r"\binput\b|\boutput\b|\brender(?:ing)?\b|\breconstruct(?:ion)?\b|\bprior\b|方法|流程|框架|输入|输出|重建|渲染)"
    ,
    re.IGNORECASE,
)
_GENERIC_HEADING_HINTS = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminar",
    "conclusion",
    "discussion",
    "reference",
)
_EXPERIMENT_HEADING_HINTS = (
    "experiment",
    "experimental",
    "setup",
    "results",
    "ablation",
    "evaluation",
    "dataset",
    "implementation",
    "baseline",
    "comparison",
)
_METHOD_HEADING_HINTS = (
    "method",
    "approach",
    "pipeline",
    "framework",
    "architecture",
    "model",
    "overview",
    "algorithm",
)
_QUOTE_PATTERNS = (
    re.compile(r'["\u201c\u201d]\s*([^"\u201c\u201d]{6,320}?)\s*["\u201c\u201d]'),
    re.compile(r"[\u2018\u2019']\s*([^\u2018\u2019']{6,260}?)\s*[\u2018\u2019']"),
    re.compile(r"[\u300c\u300d\u300e\u300f\u300a\u300b]\s*([^\u300c\u300d\u300e\u300f\u300a\u300b]{6,320}?)\s*[\u300d\u300f\u300b]"),
)
_SHELL_ONLY_RE = re.compile(
    r"^(?:"
    r"\u8bf4\u660e|\u8868\u660e|\u53ef\u89c1|\u56e0\u6b64|\u6240\u4ee5|\u603b\u4e4b|\u7efc\u4e0a|"
    r"\u7531\u6b64\u53ef\u89c1|\u8fdb\u4e00\u6b65\u8bf4\u660e|\u8fdb\u4e00\u6b65\u8868\u660e|\u8fdb\u4e00\u6b65\u8bc1\u5b9e|"
    r"\u63d0\u793a|\u6ce8\u610f|\u4e0b\u4e00\u6b65|\u5efa\u8bae"
    r")\s*[:\uFF1A]?$",
    re.IGNORECASE,
)
_SHELL_PREFIX_RE = re.compile(
    r"^(?:"
    r"\u6587\u4e2d\u63d0\u5230|\u6587\u4e2d\u6307\u51fa|\u4f5c\u8005\u6307\u51fa|"
    r"\u8868\u683c\u6807\u9898\u4e0e\u65b9\u6cd5\u547d\u540d\u660e\u786e\u4e3a|"
    r"\u76f4\u63a5\u8bc1\u636e(?:\uff08[^)\uff09]{0,48}[\)\uff09])?|"
    r"\u95f4\u63a5\u8bc1\u636e(?:\uff08[^)\uff09]{0,48}[\)\uff09])?|"
    r"\u5ef6\u4f38\u601d\u8003\u9898|\u9ad8\u4ef7\u503c\u95ee\u9898"
    r").{0,220}(?:"
    r"\u8bf4\u660e|\u8868\u660e|\u610f\u5473\u7740|\u63d0\u793a|\u53ef\u89c1|\u8bc1\u5b9e"
    r")?\s*[:\uFF1A]$",
    re.IGNORECASE,
)
_CRITICAL_FACT_HINT_RE = re.compile(
    r"("
    r"\b(?:ground[ -]?truth|pose|camera|pipeline|baseline|training|input|output|dataset|metric|ablation|"
    r"equation|formula|table|figure|fig|hardware|dmd|compression|snapshot|rendering)\b|"
    r"\b(?:nerf|scinerf|pnp|ffdnet|gap-tv)\b|"
    r"(?:\u4f4d\u59ff|\u76f8\u673a|\u8bad\u7ec3|\u8f93\u5165|\u8f93\u51fa|\u6d41\u7a0b|\u516c\u5f0f|\u53d8\u91cf|"
    r"\u8868|\u56fe|\u786c\u4ef6|\u538b\u7f29\u6bd4|\u771f\u503c|\u57fa\u7ebf|\u5bf9\u6bd4|\u5b9e\u9a8c)"
    r")",
    re.IGNORECASE,
)
_SUMMARY_NOVELTY_HINT_RE = re.compile(
    r"("
    r"\b(?:first(?:\s+to)?|novel|novelty|innovation|innovative|contribution(?:s)?|"
    r"we\s+propose|introduc(?:e|es|ed|ing)|camera\s+pose\s+stamps?|"
    r"transformation\s+network)\b|"
    r"(?:首次|首个|首创|创新点|创新|贡献|提出)"
    r")",
    re.IGNORECASE,
)
_SUMMARY_RESULT_HINT_RE = re.compile(
    r"("
    r"\b(?:extensive\s+experiments?|experiments?\s+demonstrate|high[- ]quality|novel[- ]view|"
    r"outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|superior|static(?:\s+and)?\s+dynamic|"
    r"dynamic\s+scenes?|static\s+scenes?|multi-view\s+consisten(?:t|cy))\b|"
    r"(?:实验表明|高质量|新视角|优于|静态|动态|多视角一致)"
    r")",
    re.IGNORECASE,
)
_CONTRIBUTION_BLOCK_HINT_RE = re.compile(
    r"("
    r"\b(?:first(?:\s+to)?|novel|we\s+propose|introduc(?:e|es|ed|ing)|"
    r"main\s+contribution(?:s)?|camera\s+pose\s+stamps?|transformation\s+network)\b|"
    r"(?:首次|首个|创新|贡献|提出)"
    r")",
    re.IGNORECASE,
)
_RESULT_BLOCK_HINT_RE = re.compile(
    r"("
    r"\b(?:extensive\s+experiments?|experiments?\s+demonstrate|high[- ]quality|novel[- ]view|"
    r"outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|superior|static(?:\s+and)?\s+dynamic|"
    r"dynamic\s+scenes?|static\s+scenes?)\b|"
    r"(?:实验表明|高质量|新视角|优于|静态|动态)"
    r")",
    re.IGNORECASE,
)
_CONTRIBUTION_LEADIN_HINT_RE = re.compile(
    r"\b(?:our\s+main\s+contributions?\s+can\s+be\s+summarized\s+as\s+follows|"
    r"main\s+contributions?\s+are\s+as\s+follows|contributions?\s+are\s+summarized\s+as\s+follows)\b",
    re.IGNORECASE,
)
_DEFINITION_LIKE_BLOCK_HINT_RE = re.compile(
    r"("
    r"\bis\s+defined\s+as\b|\bdefined\s+as:?\b|\bparameterized\s+as\b|\bcan\s+be\s+calculated\s+by\b|"
    r"\balpha\s+blending\b|\bjacobian\b|"
    r"\bwhere\b.{0,48}\b(?:denotes?|represents?|stands?\s+for)\b|"
    r"(?:定义为|表示为|记为|其中.{0,16}(?:表示|代表|对应))"
    r")",
    re.IGNORECASE,
)
_QUOTE_HEADING_LIKE_RE = re.compile(
    r"^(?:"
    r"abstract|introduction|background|related work|preliminar(?:y|ies)?|method(?:ology)?|"
    r"experiment(?:al)?(?: setup)?|result(?:s)?|discussion|conclusion|"
    r"baseline(?: methods?)?|evaluation metrics?|implementation details?|"
    r"references?|appendix|supplement(?:ary)?"
    r")$",
    re.IGNORECASE,
)
_FIGURE_CLAIM_RE = re.compile(
    r"(\bfig(?:ure)?\.?\s*\d{1,3}\b|(?:^|[^\w])figure\s*#?\s*\d{1,3}\b|图\s*\d{1,3}\b|第\s*\d{1,3}\s*张图)",
    re.IGNORECASE,
)


def _perf_log(stage: str, **metrics) -> None:
    parts: list[str] = []
    for key, val in metrics.items():
        if isinstance(val, float):
            parts.append(f"{key}={val:.3f}s")
        else:
            parts.append(f"{key}={val}")
    try:
        print("[kb-perf]", stage, " ".join(parts), flush=True)
    except Exception:
        pass


def _warm_refs_citation_meta_background(source_paths: list[str], *, library_db_path: Path | str | None) -> None:
    uniq_paths: list[str] = []
    seen: set[str] = set()
    for src in source_paths or []:
        s = str(src or "").strip()
        if (not s) or (s in seen):
            continue
        seen.add(s)
        uniq_paths.append(s)
        if len(uniq_paths) >= 8:
            break
    if not uniq_paths:
        return

    def _run() -> None:
        try:
            from api.reference_ui import ensure_source_citation_meta
            from api.routers.library import _md_dir, _pdf_dir
            from kb.library_store import LibraryStore
        except Exception:
            return

        try:
            pdf_root = _pdf_dir()
        except Exception:
            pdf_root = None
        try:
            md_root = _md_dir()
        except Exception:
            md_root = None
        try:
            lib_store = LibraryStore(library_db_path) if library_db_path else None
        except Exception:
            lib_store = None

        def _one(src: str) -> None:
            try:
                ensure_source_citation_meta(
                    source_path=src,
                    pdf_root=pdf_root,
                    md_root=md_root,
                    lib_store=lib_store,
                )
            except Exception:
                return

        max_workers = max(1, min(4, len(uniq_paths)))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_one, src) for src in uniq_paths]
                for fu in as_completed(futs):
                    try:
                        fu.result()
                    except Exception:
                        continue
        except Exception:
            for src in uniq_paths:
                _one(src)

    try:
        threading.Thread(target=_run, daemon=True, name="kb_refs_meta_warm").start()
    except Exception:
        pass


_PAPER_GUIDE_PREFETCH_LOCK = threading.Lock()
_PAPER_GUIDE_PREFETCH_RECENT: dict[str, float] = {}
_PAPER_GUIDE_PREFETCH_TTL_S = 20.0 * 60.0


def _normalize_fs_path_for_match(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw).expanduser().resolve(strict=False)
    except Exception:
        p = Path(raw).expanduser()
    return str(p).replace("\\", "/").strip().lower()


def _source_basename_identity(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    name = Path(raw).name or raw
    low = str(name).strip().lower()
    low = re.sub(r"\.en\.md$", "", low, flags=re.IGNORECASE)
    low = re.sub(r"\.md$", "", low, flags=re.IGNORECASE)
    low = re.sub(r"\.pdf$", "", low, flags=re.IGNORECASE)
    low = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def _resolve_paper_guide_md_path(
    source_path: str,
    *,
    md_root: Path | str | None = None,
    db_dir: Path | str | None = None,
) -> Path | None:
    raw = str(source_path or "").strip()
    if not raw:
        return None
    src = Path(raw).expanduser()
    try:
        if src.is_file() and src.suffix.lower().endswith(".md"):
            return src.resolve(strict=False)
    except Exception:
        pass

    if src.suffix.lower() != ".pdf":
        return None

    roots: list[Path] = []
    if md_root:
        try:
            roots.append(Path(md_root).expanduser())
        except Exception:
            pass
    if db_dir:
        try:
            roots.append(Path(db_dir).expanduser())
        except Exception:
            pass
        try:
            roots.append(Path(db_dir).expanduser().parent / "md_output")
        except Exception:
            pass
    try:
        roots.append(src.parent)
    except Exception:
        pass

    uniq_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in roots:
        key = _normalize_fs_path_for_match(str(root))
        if (not key) or (key in seen_roots):
            continue
        seen_roots.add(key)
        uniq_roots.append(root)

    for root in uniq_roots:
        try:
            _md_dir, md_main, md_exists = _resolve_md_output_paths(root, src)
        except Exception:
            continue
        if not md_exists:
            continue
        try:
            if md_main.is_file():
                return md_main.resolve(strict=False)
        except Exception:
            continue

    return None


def kickoff_paper_guide_prefetch(
    *,
    source_path: str,
    source_name: str = "",
    db_dir: Path | str | None = None,
    md_root: Path | str | None = None,
    library_db_path: Path | str | None = None,
) -> bool:
    raw_source = str(source_path or "").strip()
    if not raw_source:
        return False
    md_path = _resolve_paper_guide_md_path(raw_source, md_root=md_root, db_dir=db_dir)
    key = _normalize_fs_path_for_match(str(md_path) if md_path is not None else raw_source)
    if not key:
        return False

    now = time.time()
    with _PAPER_GUIDE_PREFETCH_LOCK:
        prev = float(_PAPER_GUIDE_PREFETCH_RECENT.get(key) or 0.0)
        if (now - prev) < _PAPER_GUIDE_PREFETCH_TTL_S:
            return False
        _PAPER_GUIDE_PREFETCH_RECENT[key] = now
        if len(_PAPER_GUIDE_PREFETCH_RECENT) > 240:
            old_items = sorted(_PAPER_GUIDE_PREFETCH_RECENT.items(), key=lambda item: float(item[1] or 0.0))
            drop_n = len(_PAPER_GUIDE_PREFETCH_RECENT) - 200
            for k_old, _ in old_items[: max(0, drop_n)]:
                _PAPER_GUIDE_PREFETCH_RECENT.pop(k_old, None)

    seed_name = str(source_name or "").strip() or Path(raw_source).name or Path(raw_source).stem

    def _run() -> None:
        t0 = time.perf_counter()
        deep_jobs = 0
        try:
            if md_path is not None:
                _extract_md_headings(md_path, max_n=96)
                _collect_doc_overview_snippets(md_path, max_n=4, snippet_chars=420)

                deep_queries: list[str] = []
                for q in (
                    f"{seed_name} contribution method experiment limitation",
                    f"{seed_name} abstract introduction method",
                    f"{seed_name} experiment setup results",
                    f"{seed_name} limitation failure future work",
                ):
                    qt = str(q or "").strip()
                    if qt and (qt not in deep_queries):
                        deep_queries.append(qt)
                    if len(deep_queries) >= 4:
                        break
                if deep_queries:
                    max_workers = max(1, min(3, len(deep_queries)))
                    try:
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            futs = [
                                ex.submit(
                                    _deep_read_md_for_context,
                                    md_path,
                                    q,
                                    max_snippets=3,
                                    snippet_chars=1200,
                                )
                                for q in deep_queries
                            ]
                            for fu in as_completed(futs):
                                try:
                                    fu.result()
                                except Exception:
                                    continue
                                deep_jobs += 1
                    except Exception:
                        for q in deep_queries:
                            try:
                                _deep_read_md_for_context(md_path, q, max_snippets=3, snippet_chars=1200)
                                deep_jobs += 1
                            except Exception:
                                continue

            warm_paths = [raw_source]
            if md_path is not None:
                warm_paths.insert(0, str(md_path))
            _warm_refs_citation_meta_background(warm_paths, library_db_path=library_db_path)
            _perf_log(
                "paper_guide.prefetch",
                elapsed=time.perf_counter() - t0,
                source=raw_source,
                md=str(md_path or ""),
                deep_jobs=deep_jobs,
            )
        except Exception as exc:
            _perf_log(
                "paper_guide.prefetch",
                elapsed=time.perf_counter() - t0,
                source=raw_source,
                md=str(md_path or ""),
                error=str(exc)[:120],
            )

    try:
        threading.Thread(target=_run, daemon=True, name="kb_paper_guide_prefetch").start()
    except Exception:
        return False
    return True


def _split_kb_miss_notice(text: str) -> tuple[str, str]:
    s = str(text or "").lstrip()
    prefix = "未命中知识库片段"
    if not s.startswith(prefix):
        return "", str(text or "")

    nl = s.find("\n")
    if nl != -1:
        return s[:nl].strip(), s[nl + 1 :].lstrip("\n")

    for sep in ("。", ".", "！", "!", "？", "?", ";", "；"):
        idx = s.find(sep)
        if 0 <= idx <= 80:
            return s[: idx + 1].strip(), s[idx + 1 :].lstrip()

    return prefix, s[len(prefix) :].lstrip("：: \t")


def _reconcile_kb_notice(answer: str, *, has_hits: bool) -> str:
    notice, body = _split_kb_miss_notice(answer)
    body = str(body or "").strip()
    if has_hits:
        return body or str(answer or "").strip()

    if notice:
        return str(answer or "").strip()
    if not body:
        return "未命中知识库片段"
    return f"未命中知识库片段。\n\n{body}"

_DEICTIC_DOC_RE = re.compile(
    r"(\bthis paper\b|\bthat paper\b|\bthis article\b|\bthat article\b|\bin this paper\b|\bin that paper\b|"
    r"\bthe paper\b|\bthe article\b|"
    r"这篇文章|那篇文章|这篇论文|那篇论文|本文|这篇文献|那篇文献|文中|文里|文章里|论文里)",
    flags=re.I,
)
_EXPLICIT_DOC_RE = re.compile(
    r"(\.pdf\b|[A-Za-z]+-\d{4}[-_ ][A-Za-z0-9][A-Za-z0-9 _\-]{8,}|"
    r"[A-Z][A-Za-z0-9&'._\-]+(?: [A-Za-z0-9&'._\-]+){3,})",
    flags=re.I,
)


def _needs_conversational_source_hint(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    if _EXPLICIT_DOC_RE.search(q):
        return False
    return bool(_DEICTIC_DOC_RE.search(q))


def _pick_recent_source_hint(*, conv_id: str, user_msg_id: int, chat_store: ChatStore) -> str:
    cid = str(conv_id or "").strip()
    if not cid:
        return ""
    try:
        refs_by_user = chat_store.list_message_refs(cid) or {}
    except Exception:
        refs_by_user = {}
    items = sorted(
        (
            (int(mid), rec)
            for mid, rec in refs_by_user.items()
            if isinstance(rec, dict) and int(mid or 0) > 0 and int(mid or 0) < int(user_msg_id or 0)
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    for _mid, rec in items:
        hits = rec.get("hits") or []
        if not isinstance(hits, list):
            continue
        for h in hits:
            if not isinstance(h, dict):
                continue
            meta = h.get("meta", {}) or {}
            src = str(meta.get("source_path") or "").strip()
            if not src:
                continue
            p = Path(src)
            cand0 = re.sub(r"\.en\.md$", ".pdf", p.name, flags=re.I)
            cand1 = re.sub(r"\.en$", "", p.stem, flags=re.I)
            for cand in (cand0, cand1, p.name, p.stem):
                s = str(cand or "").strip()
                if s:
                    return s
    return ""


def _augment_prompt_with_source_hint(prompt: str, source_hint: str) -> str:
    q = str(prompt or "").strip()
    hint = str(source_hint or "").strip()
    if (not q) or (not hint):
        return q
    return f"{hint} {q}".strip()


_INPAPER_QUERY_RE = re.compile(
    r"(\bfig(?:ure)?\b|\beq(?:uation)?\b|\bformula\b|\btheorem\b|\blemma\b|\bdefinition\b|\bproposition\b|\bcorollary\b|"
    r"图|公式|定理|引理|定义|命题|推论|这篇|本文|文中|这篇文章|这篇论文)",
    flags=re.I,
)


def _needs_bound_source_hint(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    if re.search(r"(\.pdf\b|[A-Za-z]+-\d{4}[-_ ][A-Za-z0-9])", q, flags=re.I):
        return False
    if _DEICTIC_DOC_RE.search(q):
        return True
    return bool(_INPAPER_QUERY_RE.search(q))


def _pick_recent_bound_source_hints(*, conv_id: str, chat_store: ChatStore, limit: int = 2) -> list[str]:
    cid = str(conv_id or "").strip()
    if not cid:
        return []
    try:
        rows = chat_store.list_conversation_sources(cid, limit=max(1, int(limit)))
    except Exception:
        rows = []
    out: list[str] = []
    seen: set[str] = set()
    for rec in rows or []:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("source_name") or "").strip()
        src = str(rec.get("source_path") or "").strip()
        cand = name or Path(src).name or Path(src).stem
        if (not cand) or (cand in seen):
            continue
        seen.add(cand)
        out.append(cand)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _source_stem_identity(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw)
        name = str(p.name or raw).strip()
    except Exception:
        name = raw
    return _source_basename_identity(name)


def _is_hit_from_bound_source(
    hit: dict,
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> bool:
    meta = (hit or {}).get("meta", {}) or {}
    hit_source = str(meta.get("source_path") or "").strip()
    if not hit_source:
        return False
    hit_path_norm = _normalize_fs_path_for_match(hit_source)
    if not hit_path_norm:
        return False

    target_path_norm = _normalize_fs_path_for_match(bound_source_path)
    if target_path_norm:
        if hit_path_norm == target_path_norm:
            return True
        # Allow strict stem-equality fallback only for pdf<->md path variants.
        hit_stem = _source_stem_identity(hit_source)
        target_stem = _source_stem_identity(bound_source_path)
        if target_stem and hit_stem and (hit_stem == target_stem) and (len(target_stem) >= 8):
            return True
        return False

    target_name = _source_basename_identity(bound_source_name)
    if not target_name:
        return False
    hit_stem = _source_stem_identity(hit_source)
    return bool(hit_stem and (hit_stem == target_name))


def _filter_hits_for_paper_guide(
    hits_raw: list[dict],
    *,
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    out: list[dict] = []
    for hit in hits_raw or []:
        if not isinstance(hit, dict):
            continue
        if _is_hit_from_bound_source(
            hit,
            bound_source_path=bound_source_path,
            bound_source_name=bound_source_name,
        ):
            out.append(hit)
    return out


def _paper_guide_fallback_deepread_hits(
    *,
    bound_source_path: str,
    bound_source_name: str,
    query: str,
    top_k: int,
    db_dir: Path | str | None = None,
) -> list[dict]:
    md_path = _resolve_paper_guide_md_path(
        bound_source_path,
        db_dir=db_dir,
    )
    if md_path is None:
        return []

    q = str(query or "").strip()
    if not q:
        q = f"{bound_source_name or md_path.stem} contribution method experiment limitation"
    deep_hits = _deep_read_md_for_context(
        md_path,
        q,
        max_snippets=max(2, min(int(top_k or 4), 4)),
        snippet_chars=1200,
    )
    out: list[dict] = []
    for idx, h in enumerate(deep_hits, start=1):
        if not isinstance(h, dict):
            continue
        rec = dict(h)
        meta = rec.get("meta", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        meta["source_path"] = str(md_path)
        meta["paper_guide_fallback"] = True
        rec["meta"] = meta
        try:
            score0 = float(rec.get("score") or 0.0)
        except Exception:
            score0 = 0.0
        if score0 <= 0.0:
            rec["score"] = max(0.01, 1.0 - (idx - 1) * 0.1)
        out.append(rec)
    return out


_ANSWER_INTENT_COMPARE_RE = re.compile(
    r"(\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|\bdifference\b|\btrade[\s-]?off\b|"
    r"对比|区别|优劣|哪个好|怎么选|选型)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_IDEA_RE = re.compile(
    r"(\bidea\b|\bnovelty\b|\binnovation\b|\bfeasib(?:le|ility)\b|\bhypothesis\b|\bbrainstorm\b|"
    r"想法|创新|可行性|可行|值得做|是否可做|研究点子)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_EXPERIMENT_RE = re.compile(
    r"(\bexperiment\b|\bablation\b|\bbaseline\b|\bmetric\b|\bevaluation\b|\bprotocol\b|\breproduc(?:e|ibility)\b|"
    r"实验|对照|指标|评估|消融|复现|验证方案)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_TROUBLESHOOT_RE = re.compile(
    r"(\bdebug\b|\btroubleshoot\b|\berror\b|\bissue\b|\bfail(?:ed|ure)?\b|\bwhy not\b|"
    r"报错|排查|卡住|失败|不收敛|跑不通)",
    flags=re.IGNORECASE,
)
_ANSWER_INTENT_WRITING_RE = re.compile(
    r"(\bwrite\b|\bwriting\b|\brewrite\b|\bedit\b|\bwording\b|\brelated work\b|\babstract\b|"
    r"写作|润色|改写|表达|摘要|相关工作)",
    flags=re.IGNORECASE,
)
_ANSWER_CITE_HINT_RE = re.compile(r"(\[\d+\]|\[\[CITE:)", flags=re.IGNORECASE)
_ANSWER_LIMITS_HINT_RE = re.compile(
    r"(\bmay\b|\bmight\b|\buncertain\b|\bunknown\b|\bnot shown\b|\binsufficient\b|\bassum(?:e|ption)\b|"
    r"可能|不确定|未知|未给出|证据不足|假设)",
    flags=re.IGNORECASE,
)
_ANSWER_SECTION_PREFIX_RE = re.compile(
    r"(?im)^\s*(Conclusion|Evidence|Limits|Next Steps|结论|依据|证据|边界|限制|局限|下一步建议|下一步)\s*[:：]"
)
_ANSWER_NEXT_STEPS_HEADER_RE = re.compile(r"(?im)^\s*(Next Steps|Next Step|下一步建议|下一步|建议)\s*[:：]")
_ANSWER_ORDERED_LIST_RE = re.compile(r"(?m)^\s*1\.\s+\S+")
_ANSWER_INTENTS = {"reading", "compare", "idea", "experiment", "troubleshoot", "writing"}
_ANSWER_DEPTHS = {"L1", "L2", "L3"}
_ANSWER_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def _normalize_answer_mode_hint(answer_mode_hint: str) -> str:
    hint = str(answer_mode_hint or "").strip().lower()
    alias = {
        "read": "reading",
        "reading": "reading",
        "literature": "reading",
        "compare": "compare",
        "comparison": "compare",
        "idea": "idea",
        "experiment": "experiment",
        "exp": "experiment",
        "debug": "troubleshoot",
        "troubleshoot": "troubleshoot",
        "writing": "writing",
    }
    return alias.get(hint, "")


def _detect_answer_intent(prompt: str, *, answer_mode_hint: str = "") -> str:
    mode_hint = _normalize_answer_mode_hint(answer_mode_hint)
    if mode_hint:
        return mode_hint
    q = str(prompt or "").strip()
    if not q:
        return "reading"
    if _ANSWER_INTENT_COMPARE_RE.search(q):
        return "compare"
    if _ANSWER_INTENT_IDEA_RE.search(q):
        return "idea"
    if _ANSWER_INTENT_EXPERIMENT_RE.search(q):
        return "experiment"
    if _ANSWER_INTENT_TROUBLESHOOT_RE.search(q):
        return "troubleshoot"
    if _ANSWER_INTENT_WRITING_RE.search(q):
        return "writing"
    return "reading"


def _detect_answer_depth(prompt: str, *, intent: str, auto_depth: bool) -> str:
    if not auto_depth:
        return "L2"
    q = str(prompt or "")
    q_len = len(q)
    technical_markers = len(
        re.findall(
            r"(\bfig(?:ure)?\b|\beq(?:uation)?\b|\bmethod\b|\balgorithm\b|\bproof\b|"
            r"实验|公式|定理|算法|模型|指标|对照|复杂度|损失函数)",
            q,
            flags=re.IGNORECASE,
        )
    )
    if intent in {"idea", "experiment", "compare"} and (q_len >= 48 or technical_markers >= 2):
        return "L3"
    if intent in {"reading", "troubleshoot", "writing"} and q_len <= 24 and technical_markers == 0:
        return "L1"
    return "L2"


def _prefer_zh_locale(*texts: str) -> bool:
    joined = " ".join(str(t or "") for t in texts if str(t or ""))
    if not joined:
        return False
    cjk = len(_ANSWER_CJK_CHAR_RE.findall(joined))
    if cjk <= 0:
        return False
    latin = len(re.findall(r"[A-Za-z]", joined))
    return cjk >= 4 or cjk >= max(2, latin // 3)


def _normalize_answer_section_name(raw: str) -> str:
    s = str(raw or "").strip().lower().replace(" ", "")
    if s in {"conclusion", "结论"}:
        return "conclusion"
    if s in {"evidence", "依据", "证据"}:
        return "evidence"
    if s in {"limits", "边界", "限制", "局限"}:
        return "limits"
    if s in {"nextsteps", "下一步", "下一步建议"}:
        return "next_steps"
    return ""


def _extract_answer_section_keys(text: str) -> list[str]:
    keys: list[str] = []
    for m in _ANSWER_SECTION_PREFIX_RE.finditer(str(text or "")):
        key = _normalize_answer_section_name(str(m.group(1) or ""))
        if key:
            keys.append(key)
    return keys


def _has_sufficient_answer_sections(text: str, *, has_hits: bool) -> bool:
    keys = set(_extract_answer_section_keys(text))
    if "conclusion" not in keys:
        return False
    if "next_steps" not in keys:
        return False
    if bool(has_hits) and ("evidence" not in keys):
        return False
    return len(keys) >= 2


def _build_answer_quality_probe(
    answer: str,
    *,
    has_hits: bool,
    contract_enabled: bool,
    intent: str = "",
    depth: str = "",
) -> dict:
    text = str(answer or "").strip()
    keys = set(_extract_answer_section_keys(text))
    has_conclusion = "conclusion" in keys
    has_evidence = "evidence" in keys
    has_limits = "limits" in keys
    has_next_steps = "next_steps" in keys
    core_sections = int(has_conclusion) + int(has_evidence) + int(has_next_steps)
    core_ratio = round(float(core_sections) / 3.0, 3)
    has_citations = bool(_ANSWER_CITE_HINT_RE.search(text) or re.search(r"\[\d{1,3}\]", text))
    evidence_required = bool(has_hits)
    evidence_ok = (not evidence_required) or bool(has_evidence)
    minimum_ok = bool(has_conclusion and has_next_steps and evidence_ok)
    return {
        "contract_enabled": bool(contract_enabled),
        "intent": str(intent or ""),
        "depth": str(depth or ""),
        "char_count": len(text),
        "has_hits": bool(has_hits),
        "has_conclusion": bool(has_conclusion),
        "has_evidence": bool(has_evidence),
        "has_limits": bool(has_limits),
        "has_next_steps": bool(has_next_steps),
        "has_citations": bool(has_citations),
        "evidence_required": bool(evidence_required),
        "evidence_ok": bool(evidence_ok),
        "core_section_coverage": core_ratio,
        "minimum_ok": bool(minimum_ok),
    }


def _build_answer_contract_system_rules(*, intent: str, depth: str, has_hits: bool) -> str:
    lines = [
        "Answer Contract v1 (enabled):",
        "- Keep the response compact but structured.",
        "- Use this section order when possible: Conclusion, Evidence, Limits, Next Steps.",
        "- Keep the answer in the same language as the user's query.",
        "- Conclusion should answer the user's core question directly in 1-3 sentences.",
        "- Avoid redundancy; keep total bullet/action lines concise (usually <= 8).",
    ]
    if has_hits:
        lines.append("- Evidence should be grounded in retrieved snippets and include citations when available.")
    else:
        lines.append("- If retrieval has no hits, avoid fabrication and clearly mark the answer as general guidance.")
    if depth == "L1":
        lines.append("- Depth=L1: concise response, at most one next-step action.")
    elif depth == "L3":
        lines.append("- Depth=L3: include assumptions/boundaries and 2-3 concrete follow-up actions.")
    else:
        lines.append("- Depth=L2: include at least one evidence item and 1-2 concrete next-step actions.")

    intent_rules = {
        "reading": "- Intent=reading: focus on contribution, key evidence, and where to read next.",
        "compare": "- Intent=compare: emphasize differences, applicability boundary, and trade-offs.",
        "idea": "- Intent=idea: include feasibility, risk, and a minimum validation path.",
        "experiment": "- Intent=experiment: include variables/controls, metrics, and expected outcomes.",
        "troubleshoot": "- Intent=troubleshoot: prioritize likely causes, diagnosis steps, and fix order.",
        "writing": "- Intent=writing: provide structure edits and directly reusable wording suggestions.",
    }
    lines.append(intent_rules.get(intent, intent_rules["reading"]))
    return "\n" + "\n".join(lines) + "\n"


def _answer_contract_enabled(task: dict | None) -> bool:
    return bool((task or {}).get("answer_contract_v1", False))


def _build_paper_guide_grounding_rules(*, answer_contract_v1: bool) -> str:
    lines = [
        "Paper-guide formula grounding:",
        "- When the question asks about a formula/model, quote the equation from retrieved context as-is, including equation tag/number when available.",
        "- If exact equation text is not present in retrieved context, say it is not explicitly available; do not invent a generic replacement formula.",
        "- When quoting a retrieved long equation or numbered equation, render the equation itself as display math and keep the explanation sentence separate.",
        "- Do not compress a retrieved long equation plus its explanation into one mixed prose sentence if that would blur the locate target.",
        "- Do not introduce display-math formulas for figure explanations, mechanism summaries, or optimization sketches unless that exact display formula was retrieved from the paper.",
    ]
    if answer_contract_v1:
        lines.extend(
            [
                "- Keep the answer compact: 3-4 sections, avoid repetitive paragraphs.",
                "- Only keep claims with direct support under Evidence; move unsupported general knowledge to Limits.",
            ]
        )
    else:
        lines.extend(
            [
                "- Keep the answer compact and direct, but do not force a fixed section template.",
                "- Distinguish retrieved evidence from supplemental general knowledge in prose; do not present unsupported context as direct paper evidence.",
            ]
        )
    return "\n" + "\n".join(lines) + "\n"


def _extract_cited_sentences(text: str, *, limit: int = 2) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seg in re.split(r"(?<=[。！？.!?])\s*", str(text or "")):
        s = str(seg or "").strip()
        if (not s) or (not _ANSWER_CITE_HINT_RE.search(s)):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _build_default_next_steps(*, intent: str, has_hits: bool, locale: str = "en") -> list[str]:
    by_intent_en = {
        "reading": [
            "Check the cited section/figure to verify the conclusion details.",
            "Compare this result against one baseline paper from the same period.",
            "Write a 3-line takeaway linked to the cited evidence for your notes.",
        ],
        "compare": [
            "Build a side-by-side table with assumptions, compute cost, and expected gains.",
            "Choose one metric where methods diverge most and run a small pilot test.",
            "Document the boundary conditions where each method is likely to fail.",
        ],
        "idea": [
            "Define the minimum viable experiment that can falsify this idea within one week.",
            "List the top 2 technical risks and one mitigation for each risk.",
            "Search for one recent paper that tests a similar hypothesis and compare setup.",
        ],
        "experiment": [
            "Fix the control group and only vary one key factor in the first run.",
            "Predefine evaluation metrics and stopping criteria before training starts.",
            "Record reproducibility settings (seed, environment, data split) for each run.",
        ],
        "troubleshoot": [
            "Reproduce the issue on a minimal case and log exact error/output deltas.",
            "Check environment, dependency, and data preprocessing mismatches first.",
            "Apply one change at a time and validate with a short regression test.",
        ],
        "writing": [
            "Rewrite one paragraph using claim -> evidence -> implication order.",
            "Replace broad statements with measurable or cited wording.",
            "Run a final pass for logical transitions between adjacent paragraphs.",
        ],
    }
    by_intent_zh = {
        "reading": [
            "优先核对被引用的具体段落/图表，确认结论对应的原文证据。",
            "找一篇同阶段基线文献做并行对比，判断结论是否稳健。",
            "把本次结论整理成 3 行读书卡片，并附上对应引文编号。",
        ],
        "compare": [
            "做一张并排对比表：假设条件、计算代价、预期收益各列一行。",
            "挑一个差异最大的指标先做小样本验证，快速判断优劣。",
            "补充两种方法各自的失效边界，避免误用到不适配场景。",
        ],
        "idea": [
            "定义一个一周内可执行的最小可证伪实验来验证这个想法。",
            "列出前 2 个技术风险，并给出每个风险的缓解方案。",
            "补查 1 篇近期相似假设论文，重点对比实验设置与结果。",
        ],
        "experiment": [
            "先固定对照组，只改变一个关键变量完成首轮实验。",
            "训练前预先写清评价指标与停止条件，减少后验偏差。",
            "逐次记录复现配置（随机种子、环境、数据切分）便于回溯。",
        ],
        "troubleshoot": [
            "先构造最小复现样例，并记录精确报错与输出差异。",
            "优先检查环境、依赖版本和数据预处理是否一致。",
            "每次只改一个变量，并配合短回归测试验证修复效果。",
        ],
        "writing": [
            "按“结论-证据-意义”重写一段，先提升主线清晰度。",
            "把泛化表述替换为可量化或可引用的具体表述。",
            "最后做一轮段间衔接检查，确保上下文过渡自然。",
        ],
    }
    use_zh = str(locale or "").strip().lower().startswith("zh")
    by_intent = by_intent_zh if use_zh else by_intent_en
    steps = list(by_intent.get(intent, by_intent["reading"]))
    if not has_hits:
        if use_zh:
            steps.insert(0, "补充 1-2 篇目标论文或具体章节，下次回答才能更好地基于证据。")
        else:
            steps.insert(0, "Add one or two target papers or sections so the next answer can be evidence-grounded.")
    return steps


def _apply_answer_contract_v1(
    answer: str,
    *,
    prompt: str,
    has_hits: bool,
    intent: str,
    depth: str,
) -> str:
    _ = str(prompt or "")
    src = str(answer or "").strip()
    if not src:
        return src

    notice, body0 = _split_kb_miss_notice(src)
    body = str(body0 if notice else src).strip()
    if not body:
        return src
    if _has_sufficient_answer_sections(body, has_hits=has_hits):
        return src

    paras = [p.strip() for p in re.split(r"\n{2,}", body) if str(p or "").strip()]
    if not paras:
        return src

    conclusion = re.sub(_ANSWER_SECTION_PREFIX_RE, "", paras[0], count=1).strip() or paras[0]
    tail = paras[1:]
    evidence: list[str] = []
    extra: list[str] = []
    for p in tail:
        p1 = str(p or "").strip()
        if _ANSWER_NEXT_STEPS_HEADER_RE.search(p1):
            # Avoid carrying old next-steps fragments into extra paragraphs.
            continue
        if _ANSWER_CITE_HINT_RE.search(p):
            evidence.append(p)
        else:
            extra.append(p)
    if has_hits and not evidence:
        evidence = _extract_cited_sentences(body, limit=2)

    prefer_zh = _prefer_zh_locale(prompt, src)
    labels = {
        "conclusion": "结论" if prefer_zh else "Conclusion",
        "evidence": "依据" if prefer_zh else "Evidence",
        "limits": "边界" if prefer_zh else "Limits",
        "next_steps": "下一步" if prefer_zh else "Next Steps",
    }
    if has_hits and (not evidence):
        evidence.append(
            "命中库内片段支持该结论，建议优先核对对应原文段落/图表。"
            if prefer_zh
            else "Retrieved library snippets support this conclusion; verify against the cited source passage/figure."
        )

    limits: list[str] = []
    if _ANSWER_LIMITS_HINT_RE.search(body):
        limits.append(
            "部分细节依赖假设或在当前上下文中未明确给出。"
            if prefer_zh
            else "Some details may depend on assumptions or are not explicit in the available context."
        )
    if not has_hits:
        limits.append(
            "当前未检索到可直接引用的库内片段，本回答属于通用指导。"
            if prefer_zh
            else "This answer is general guidance because no direct library snippets were retrieved."
        )
    if (not limits) and depth == "L3":
        limits.append(
            "在将结论作为最终证据前，请回到原文核验关键假设。"
            if prefer_zh
            else "Validate key assumptions against the original paper before using this as final evidence."
        )

    depth_level = depth if depth in _ANSWER_DEPTHS else "L2"
    step_limit = 1 if depth_level == "L1" else (3 if depth_level == "L3" else 2)
    body_limit = 2 if depth_level == "L1" else (6 if depth_level == "L3" else 4)
    steps = _build_default_next_steps(
        intent=intent if intent in _ANSWER_INTENTS else "reading",
        has_hits=has_hits,
        locale="zh" if prefer_zh else "en",
    )[:step_limit]

    parts: list[str] = []
    if notice:
        parts.append(notice)
    parts.append(f"{labels['conclusion']}: {conclusion}")
    if has_hits and evidence:
        parts.append(f"{labels['evidence']}:\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(evidence[:3], start=1)))
    if limits:
        parts.append(f"{labels['limits']}:\n" + "\n".join(f"- {item}" for item in limits[:2]))
    if steps:
        parts.append(f"{labels['next_steps']}:\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(steps, start=1)))
    if extra:
        parts.append("\n\n".join(extra[:body_limit]))

    contracted = "\n\n".join(parts).strip()
    # Guardrail: avoid over-shrinking long answers when we auto-repair structure.
    # Keep the structured scaffold, but preserve more original detail if contraction is too aggressive.
    if len(src) >= 700 and len(contracted) < int(len(src) * 0.65):
        details = "\n\n".join(extra).strip()
        if details and (details not in contracted):
            details_title = "补充细节" if prefer_zh else "Additional Details"
            contracted = f"{contracted}\n\n{details_title}:\n{details}".strip()
    return contracted


def _enhance_kb_miss_fallback(
    answer: str,
    *,
    has_hits: bool,
    intent: str,
    depth: str,
    contract_enabled: bool,
) -> str:
    src = str(answer or "").strip()
    if (not src) or has_hits:
        return src
    notice, body0 = _split_kb_miss_notice(src)
    if not notice:
        return src
    body = str(body0 or "").strip()
    if not body:
        body = "当前没有检索到可直接引用的库内片段，先给你一个可执行的通用路径。"
    if not contract_enabled:
        return f"{notice}\n\n{body}".strip()
    if _ANSWER_NEXT_STEPS_HEADER_RE.search(body):
        return f"{notice}\n\n{body}".strip()
    if _ANSWER_ORDERED_LIST_RE.search(body):
        return f"{notice}\n\n{body}".strip()

    depth_level = depth if depth in _ANSWER_DEPTHS else "L2"
    step_limit = 1 if depth_level == "L1" else (3 if depth_level == "L3" else 2)
    intent_norm = intent if intent in _ANSWER_INTENTS else "reading"
    prefer_zh = _prefer_zh_locale(body, notice)
    steps = _build_default_next_steps(
        intent=intent_norm,
        has_hits=False,
        locale="zh" if prefer_zh else "en",
    )[:step_limit]
    if not steps:
        return f"{notice}\n\n{body}".strip()
    step_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))
    next_steps_title = "下一步建议" if prefer_zh else "Next Steps"
    return f"{notice}\n\n{body}\n\n{next_steps_title}:\n{step_lines}".strip()


# Backward-compat for long-lived Streamlit processes that loaded older runtime_state.
if not hasattr(RUNTIME, "BG_LOCK"):
    RUNTIME.BG_LOCK = threading.Lock()
if not hasattr(RUNTIME, "BG_STATE"):
    RUNTIME.BG_STATE = {
        "queue": [],
        "running": False,
        "done": 0,
        "total": 0,
        "current": "",
        "cur_page_done": 0,
        "cur_page_total": 0,
        "cur_page_msg": "",
        "cancel": False,
        "last": "",
    }
if not hasattr(RUNTIME, "GEN_QUALITY_EVENTS"):
    RUNTIME.GEN_QUALITY_EVENTS = []

_BG_STATE = RUNTIME.BG_STATE
_BG_LOCK = RUNTIME.BG_LOCK


def _cite_source_id(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return "s0000000"
    return "s" + hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]

def _live_assistant_text(task_id: str) -> str:
    return f"{_LIVE_ASSISTANT_PREFIX}{str(task_id or '').strip()}"

def _is_live_assistant_text(text: str) -> bool:
    return str(text or "").strip().startswith(_LIVE_ASSISTANT_PREFIX)

def _live_assistant_task_id(text: str) -> str:
    s = str(text or "").strip()
    if not s.startswith(_LIVE_ASSISTANT_PREFIX):
        return ""
    return s[len(_LIVE_ASSISTANT_PREFIX) :].strip()

def _gen_get_task(session_id: str) -> dict | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    with RUNTIME.GEN_LOCK:
        t = RUNTIME.GEN_TASKS.get(sid)
        return dict(t) if isinstance(t, dict) else None

def _gen_update_task(session_id: str, task_id: str, **patch) -> None:
    sid = (session_id or "").strip()
    tid = (task_id or "").strip()
    if (not sid) or (not tid):
        return
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if not isinstance(cur, dict):
            return
        if str(cur.get("id") or "") != tid:
            return
        nxt = dict(cur)
        nxt.update(patch)
        nxt["updated_at"] = time.time()
        RUNTIME.GEN_TASKS[sid] = nxt

def _gen_should_cancel(session_id: str, task_id: str) -> bool:
    sid = (session_id or "").strip()
    tid = (task_id or "").strip()
    if (not sid) or (not tid):
        return True
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if not isinstance(cur, dict):
            return True
        if str(cur.get("id") or "") != tid:
            return True
        return bool(cur.get("cancel") or False)

def _gen_mark_cancel(session_id: str, task_id: str) -> bool:
    sid = (session_id or "").strip()
    tid = (task_id or "").strip()
    if (not sid) or (not tid):
        return False
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if not isinstance(cur, dict):
            return False
        if str(cur.get("id") or "") != tid:
            return False
        if str(cur.get("status") or "") != "running":
            return False
        if bool(cur.get("answer_ready") or False):
            # During post-answer refs refinement, keep answer stable and do not flip to canceled.
            return False
        cur2 = dict(cur)
        cur2["cancel"] = True
        cur2["stage"] = "canceled"
        cur2["updated_at"] = time.time()
        RUNTIME.GEN_TASKS[sid] = cur2
        return True


def _gen_record_answer_quality(
    *,
    session_id: str,
    task_id: str,
    conv_id: str,
    answer_quality: dict | None,
) -> None:
    q = dict(answer_quality or {})
    if not q:
        return
    try:
        max_keep = int(str(os.environ.get("KB_ANSWER_QUALITY_KEEP", "800") or "800"))
    except Exception:
        max_keep = 800
    max_keep = max(100, min(5000, max_keep))
    sample = {
        "ts": float(time.time()),
        "session_id": str(session_id or ""),
        "task_id": str(task_id or ""),
        "conv_id": str(conv_id or ""),
        "intent": str(q.get("intent") or ""),
        "depth": str(q.get("depth") or ""),
        "contract_enabled": bool(q.get("contract_enabled", False)),
        "has_hits": bool(q.get("has_hits", False)),
        "has_conclusion": bool(q.get("has_conclusion", False)),
        "has_evidence": bool(q.get("has_evidence", False)),
        "has_next_steps": bool(q.get("has_next_steps", False)),
        "evidence_required": bool(q.get("evidence_required", False)),
        "evidence_ok": bool(q.get("evidence_ok", False)),
        "minimum_ok": bool(q.get("minimum_ok", False)),
        "core_section_coverage": float(q.get("core_section_coverage") or 0.0),
        "char_count": int(q.get("char_count") or 0),
    }
    with RUNTIME.GEN_LOCK:
        events = getattr(RUNTIME, "GEN_QUALITY_EVENTS", None)
        if not isinstance(events, list):
            events = []
            RUNTIME.GEN_QUALITY_EVENTS = events
        events.append(sample)
        overflow = len(events) - max_keep
        if overflow > 0:
            del events[:overflow]


def _gen_answer_quality_summary(
    *,
    limit: int = 200,
    intent: str = "",
    depth: str = "",
    only_failed: bool = False,
) -> dict:
    try:
        n_limit = int(limit)
    except Exception:
        n_limit = 200
    n_limit = max(20, min(2000, n_limit))
    intent_filter = str(intent or "").strip().lower()
    depth_filter = str(depth or "").strip().upper()
    failed_only = bool(only_failed)
    with RUNTIME.GEN_LOCK:
        events0 = getattr(RUNTIME, "GEN_QUALITY_EVENTS", None)
        events = list(events0) if isinstance(events0, list) else []
    if intent_filter:
        events = [x for x in events if str(x.get("intent") or "").strip().lower() == intent_filter]
    if depth_filter:
        events = [x for x in events if str(x.get("depth") or "").strip().upper() == depth_filter]
    if failed_only:
        events = [x for x in events if not bool(x.get("minimum_ok", False))]
    if n_limit and len(events) > n_limit:
        events = events[-n_limit:]
    total = len(events)
    if total <= 0:
        return {
            "limit": n_limit,
            "filters": {
                "intent": intent_filter,
                "depth": depth_filter,
                "only_failed": failed_only,
            },
            "total": 0,
            "failed_count": 0,
            "failed_rate": 0.0,
            "structure_complete_rate": 0.0,
            "evidence_coverage_rate": 0.0,
            "next_steps_coverage_rate": 0.0,
            "minimum_ok_rate": 0.0,
            "avg_core_section_coverage": 0.0,
            "by_intent": {},
            "by_depth": {},
            "fail_reasons": {},
        }

    structure_ok = 0
    evidence_ok = 0
    next_steps_ok = 0
    minimum_ok = 0
    failed_count = 0
    core_cov_sum = 0.0
    by_intent: dict[str, dict] = {}
    by_depth: dict[str, dict] = {}
    fail_reasons: dict[str, int] = {}

    for rec in events:
        has_conclusion = bool(rec.get("has_conclusion", False))
        has_next_steps = bool(rec.get("has_next_steps", False))
        has_evidence = bool(rec.get("has_evidence", False))
        evidence_required = bool(rec.get("evidence_required", False))
        rec_structure_ok = bool(has_conclusion and has_next_steps)
        rec_evidence_ok = bool((not evidence_required) or has_evidence)
        rec_next_ok = bool(has_next_steps)
        rec_minimum_ok = bool(rec.get("minimum_ok", False))
        rec_core_cov = float(rec.get("core_section_coverage") or 0.0)

        structure_ok += int(rec_structure_ok)
        evidence_ok += int(rec_evidence_ok)
        next_steps_ok += int(rec_next_ok)
        minimum_ok += int(rec_minimum_ok)
        failed_count += int(not rec_minimum_ok)
        core_cov_sum += rec_core_cov

        intent = str(rec.get("intent") or "unknown").strip().lower() or "unknown"
        bucket = by_intent.setdefault(
            intent,
            {
                "count": 0,
                "structure_complete_rate": 0.0,
                "evidence_coverage_rate": 0.0,
                "next_steps_coverage_rate": 0.0,
                "minimum_ok_rate": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["_structure_ok"] = int(bucket.get("_structure_ok", 0)) + int(rec_structure_ok)
        bucket["_evidence_ok"] = int(bucket.get("_evidence_ok", 0)) + int(rec_evidence_ok)
        bucket["_next_ok"] = int(bucket.get("_next_ok", 0)) + int(rec_next_ok)
        bucket["_minimum_ok"] = int(bucket.get("_minimum_ok", 0)) + int(rec_minimum_ok)

        depth = str(rec.get("depth") or "unknown").strip().upper() or "UNKNOWN"
        d_bucket = by_depth.setdefault(
            depth,
            {
                "count": 0,
                "minimum_ok_rate": 0.0,
                "avg_char_count": 0.0,
            },
        )
        d_bucket["count"] += 1
        d_bucket["_minimum_ok"] = int(d_bucket.get("_minimum_ok", 0)) + int(rec_minimum_ok)
        d_bucket["_char_sum"] = int(d_bucket.get("_char_sum", 0)) + int(rec.get("char_count") or 0)

        if not rec_minimum_ok:
            reasons: list[str] = []
            if not has_conclusion:
                reasons.append("missing_conclusion")
            if not has_next_steps:
                reasons.append("missing_next_steps")
            if evidence_required and (not has_evidence):
                reasons.append("missing_evidence")
            if not reasons:
                reasons.append("other")
            for r in reasons:
                fail_reasons[r] = int(fail_reasons.get(r, 0)) + 1

    def _ratio(ok: int, n: int) -> float:
        if n <= 0:
            return 0.0
        return round(float(ok) / float(n), 3)

    for bucket in by_intent.values():
        c = int(bucket.get("count") or 0)
        bucket["structure_complete_rate"] = _ratio(int(bucket.get("_structure_ok") or 0), c)
        bucket["evidence_coverage_rate"] = _ratio(int(bucket.get("_evidence_ok") or 0), c)
        bucket["next_steps_coverage_rate"] = _ratio(int(bucket.get("_next_ok") or 0), c)
        bucket["minimum_ok_rate"] = _ratio(int(bucket.get("_minimum_ok") or 0), c)
        bucket.pop("_structure_ok", None)
        bucket.pop("_evidence_ok", None)
        bucket.pop("_next_ok", None)
        bucket.pop("_minimum_ok", None)

    for d_bucket in by_depth.values():
        c = int(d_bucket.get("count") or 0)
        d_bucket["minimum_ok_rate"] = _ratio(int(d_bucket.get("_minimum_ok") or 0), c)
        if c > 0:
            d_bucket["avg_char_count"] = round(float(int(d_bucket.get("_char_sum") or 0)) / float(c), 1)
        else:
            d_bucket["avg_char_count"] = 0.0
        d_bucket.pop("_minimum_ok", None)
        d_bucket.pop("_char_sum", None)

    return {
        "limit": n_limit,
        "filters": {
            "intent": intent_filter,
            "depth": depth_filter,
            "only_failed": failed_only,
        },
        "total": total,
        "failed_count": failed_count,
        "failed_rate": _ratio(failed_count, total),
        "structure_complete_rate": _ratio(structure_ok, total),
        "evidence_coverage_rate": _ratio(evidence_ok, total),
        "next_steps_coverage_rate": _ratio(next_steps_ok, total),
        "minimum_ok_rate": _ratio(minimum_ok, total),
        "avg_core_section_coverage": round(core_cov_sum / float(total), 3),
        "by_intent": by_intent,
        "by_depth": by_depth,
        "fail_reasons": dict(sorted(fail_reasons.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))),
    }


def _gen_store_answer(task: dict, answer: str) -> None:
    conv_id = str(task.get("conv_id") or "")
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = ChatStore(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid > 0:
        ok = chat_store.update_message_content(amid, answer)
        if not ok:
            chat_store.append_message(conv_id, "assistant", answer)
    else:
        chat_store.append_message(conv_id, "assistant", answer)

def _gen_store_partial(task: dict, partial: str) -> None:
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = ChatStore(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid <= 0:
        return
    txt = str(partial or "").strip()
    if not txt:
        return
    try:
        chat_store.update_message_content(amid, txt)
    except Exception:
        pass


def _is_display_formula_segment(text: str, *, segment_kind: str = "") -> bool:
    src = str(text or "").strip()
    if not src:
        return False
    low = src.lower()
    if _DISPLAY_EQ_SEG_RE.search(src):
        return True
    if _EQ_ENV_SEG_RE.search(src):
        return True
    if "\\[" in src and "\\]" in src:
        return True

    # Allow numbered equation mentions to trigger equation-mode mapping.
    n0 = extract_equation_number(src)
    if n0 > 0 and has_equation_signal(src):
        return True

    if not has_equation_signal(src):
        return False

    # Inline-math explanatory bullets should stay as paragraph mapping.
    clean = normalize_inline_markdown(src)
    if not clean:
        return False
    latin_words = len(_LATIN_WORD_RE.findall(clean))
    cjk_words = len(_CJK_WORD_RE.findall(clean))
    formula_tokens = len(_FORMULA_TOKEN_RE.findall(src))
    non_empty_lines = [line for line in src.splitlines() if str(line).strip()]
    kind = str(segment_kind or "").strip().lower()

    if formula_tokens >= 8 and (latin_words + cjk_words) <= 12 and len(non_empty_lines) <= 3:
        return True
    if kind == "paragraph" and formula_tokens >= 10 and (latin_words + cjk_words) <= 14:
        return True
    return False


def _formula_token_overlap_score(a: str, b: str) -> float:
    ta = set(_FORMULA_CMD_RE.findall(str(a or "")))
    tb = set(_FORMULA_CMD_RE.findall(str(b or "")))
    if not ta or not tb:
        return 0.0
    overlap = sum(1 for token in ta if token in tb)
    return float(overlap) / (max(1.0, (len(ta) * len(tb)) ** 0.5))


def _text_token_overlap_score(a: str, b: str) -> float:
    x = normalize_match_text(a)
    y = normalize_match_text(b)
    if (not x) or (not y):
        return 0.0
    ta = set(re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{1,2}", x))
    tb = set(re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{1,2}", y))
    if not ta or not tb:
        return 0.0
    overlap = sum(1 for token in ta if token in tb)
    return float(overlap) / (max(1.0, (len(ta) * len(tb)) ** 0.5))


def _normalize_formula_compare_text(text: str) -> str:
    src = str(text or "")
    if not src:
        return ""
    s = src.lower()
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\tag\{\s*\d{1,4}\s*\}", "", s)
    s = re.sub(r"\s+", "", s)
    return s[:2000]


def _formula_char_similarity(a: str, b: str) -> float:
    x = _normalize_formula_compare_text(a)
    y = _normalize_formula_compare_text(b)
    if (not x) or (not y):
        return 0.0
    try:
        return float(SequenceMatcher(None, x, y).ratio())
    except Exception:
        return 0.0


def _segment_snippet_aliases(text: str) -> list[str]:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return []
    aliases: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        key = normalize_match_text(str(raw or ""))
        if (not key) or (key in seen):
            return
        seen.add(key)
        aliases.append(key[:360])

    _push(src)
    if len(src) > 200:
        _push(src[:200])

    sent_list = [s.strip() for s in _SEG_SENT_SPLIT_RE.split(src) if str(s).strip()]
    if sent_list:
        first = sent_list[0]
        if len(first) >= 14:
            _push(first)
        if len(sent_list) >= 2:
            pair = f"{sent_list[0]} {sent_list[1]}".strip()
            if len(pair) >= 18:
                _push(pair)

    return aliases[:6]


def _strip_provenance_noise_text(text: str) -> str:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return ""
    src = re.sub(r"\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]", " ", src)
    src = re.sub(r"\(\s*\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\s*\)", " ", src)
    src = re.sub(r"(?:see|\u53c2\u89c1)\s*\[\d{1,3}(?:\s*[-,\u2013\u2014]\s*\d{1,3})*\]", " ", src, flags=re.IGNORECASE)
    src = re.sub(r"\s+", " ", src)
    return src.strip()


def _extract_quoted_spans(text: str, *, min_len: int = 10) -> list[str]:
    src = _strip_provenance_noise_text(text)
    if not src:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pattern in _QUOTE_PATTERNS:
        for m in pattern.finditer(src):
            item = str(m.group(1) or "").strip()
            if len(item) < max(1, int(min_len)):
                continue
            key = normalize_match_text(item)
            if (not key) or (key in seen):
                continue
            seen.add(key)
            out.append(item[:360])
            if len(out) >= 6:
                return out
    return out


def _longest_quoted_span(text: str, *, min_len: int = 10) -> str:
    spans = _extract_quoted_spans(text, min_len=min_len)
    if not spans:
        return ""
    spans.sort(key=lambda item: len(str(item or "")), reverse=True)
    return str(spans[0] or "").strip()


def _is_heading_like_quote_span(text: str) -> bool:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return True
    compact = re.sub(r"\s+", " ", raw).strip(" :;,.!?()[]{}\"'“”‘’")
    if not compact:
        return True
    if _QUOTE_HEADING_LIKE_RE.fullmatch(compact):
        return True
    if re.search(r"[。！？!?;；:：]", compact):
        return False
    latin_words = _LATIN_WORD_RE.findall(compact)
    if 0 < len(latin_words) <= 8 and len(compact) <= 80:
        verb_like = re.search(
            r"\b(?:is|are|was|were|be|being|been|can|cannot|could|should|would|will|use|used|"
            r"using|estimate|estimated|show|shown|train|training|feed|feeding|make|making|"
            r"exploit|exploits|provide|providing|compare|comparison)\b",
            compact,
            re.IGNORECASE,
        )
        if not verb_like:
            return True
    if len(compact) <= 28 and not re.search(r"\d", compact):
        return True
    return False


def _is_rhetorical_shell_sentence(text: str) -> bool:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return True
    if _SHELL_ONLY_RE.match(raw):
        return True
    if _SHELL_PREFIX_RE.match(raw):
        return True
    if raw.endswith(":") or raw.endswith("："):
        informative_tail = re.sub(r"[:：]\s*$", "", raw).strip()
        if len(informative_tail) <= 32:
            return True
        if re.search(r"(?:\u8bf4\u660e|\u8868\u660e|\u610f\u5473\u7740|\u63d0\u793a|\u53ef\u89c1|\u8bc1\u5b9e)\s*$", informative_tail):
            return True
    return False


_NON_SOURCE_SEGMENT_HINTS = (
    "通用知识",
    "非检索片段内容",
    "非检索内容",
    "外部常识补充",
    "未出现在本文检索片段中",
    "未出现在当前检索片段中",
    "未出现在检索片段中",
    "generic knowledge",
    "non-retrieved",
    "not present in retrieved context",
    "not in retrieved context",
    "not explicitly available in retrieved context",
)


def _is_explicit_non_source_segment(text: str) -> bool:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return False
    low = raw.lower()
    if "补充说明" in raw and ("检索片段" in raw or "retrieved context" in low):
        return True
    for hint in _NON_SOURCE_SEGMENT_HINTS:
        if hint and ((hint in raw) or (hint in low)):
            return True
    return False


def _opens_non_source_scope(text: str) -> bool:
    raw = str(text or "").strip()
    clean = _strip_provenance_noise_text(raw)
    if (not clean) or (not _is_explicit_non_source_segment(raw)):
        return False
    head = clean[:48].lower()
    if clean.endswith(":") or clean.endswith("："):
        return True
    return ("supplement" in head) or ("补充说明" in clean[:24]) or ("说明" in clean[:16])


def _is_non_source_scope_boundary(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    clean = _strip_provenance_noise_text(raw)
    if not clean:
        return False
    if _is_explicit_non_source_segment(raw):
        return False
    if raw.startswith("---"):
        return True
    if re.match(r"^\s{0,3}#{1,6}\s+", raw):
        return True
    return bool(
        re.match(
            r"^\s*(?:结论|依据|边界|限制|下一步|note|notes|boundary|limitations?|conclusion|next steps?)\s*[:：]",
            clean,
            re.IGNORECASE,
        )
    )


def _critical_fact_score(text: str) -> float:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return 0.0
    score = 0.0
    if len(raw) >= 18:
        score += 0.12
    if len(raw) >= 32:
        score += 0.12
    if len(raw) >= 56:
        score += 0.08
    if re.search(r"\d", raw):
        score += 0.08
    if re.search(r"[A-Z][A-Za-z0-9+-]{1,}", raw):
        score += 0.08
    if has_equation_signal(raw):
        score += 0.14
    if extract_equation_number(raw) > 0:
        score += 0.12
    if _FIGURE_CLAIM_RE.search(raw):
        score += 0.12
    if _CLAIM_EXPERIMENT_HINT_RE.search(raw):
        score += 0.10
    if _CLAIM_METHOD_HINT_RE.search(raw):
        score += 0.08
    if _CRITICAL_FACT_HINT_RE.search(raw):
        score += 0.14
    return float(score)


def _formula_anchor_text(raw_markdown: str, segment_text: str, primary_block: dict | None) -> str:
    raw = str(raw_markdown or "").strip()
    seg = str(segment_text or "").strip()
    if raw:
        m = _DISPLAY_EQ_SEG_RE.search(raw)
        if m:
            return str(m.group(0) or "").strip()[:900]
        if _EQ_ENV_SEG_RE.search(raw):
            return raw[:900]
    if isinstance(primary_block, dict):
        block_text = str(primary_block.get("raw_text") or primary_block.get("text") or "").strip()
        if block_text:
            return block_text[:900]
    return seg[:900]


def _dedupe_str_items(items: list[object] | tuple[object, ...] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        text = str(item or "").strip()
        if (not text) or (text in seen):
            continue
        seen.add(text)
        out.append(text)
    return out


def _segment_claim_meta(
    *,
    segment_text: str,
    raw_markdown: str,
    segment_kind: str,
    evidence_mode: str,
    primary_block: dict | None,
    evidence_quote: str,
    mapping_quality: float,
) -> dict[str, object]:
    seg_text = _strip_provenance_noise_text(segment_text)
    raw_md = str(raw_markdown or "").strip()
    kind = str(segment_kind or "").strip().lower()
    mode = str(evidence_mode or "").strip().lower()
    quote_spans = _extract_quoted_spans(raw_md or seg_text, min_len=10)
    quote_anchor = _longest_quoted_span(raw_md or seg_text, min_len=10)
    heading_like_quote = bool(quote_anchor) and _is_heading_like_quote_span(quote_anchor)
    anchor_text = ""
    anchor_kind = ""
    claim_type = "critical_fact_claim"
    explicit_non_source = _is_explicit_non_source_segment(raw_md or seg_text)
    eq_number = extract_equation_number(raw_md or seg_text) if has_equation_signal(raw_md or seg_text) else 0
    figure_number = _extract_figure_number(raw_md or seg_text) if _FIGURE_CLAIM_RE.search(raw_md or seg_text) else 0
    primary_kind = str((primary_block or {}).get("kind") or "").strip().lower()

    if explicit_non_source:
        claim_type = "shell_sentence"
        anchor_kind = ""
        anchor_text = ""
    elif _is_display_formula_segment(raw_md or seg_text, segment_kind=kind):
        claim_type = "formula_claim"
        anchor_kind = "equation"
        anchor_text = _formula_anchor_text(raw_md, seg_text, primary_block)
    elif primary_kind == "figure" or figure_number > 0:
        claim_type = "figure_claim"
        anchor_kind = "figure"
        figure_anchor = str(evidence_quote or seg_text or raw_md).strip()
        if figure_number > 0 and not re.search(
            rf"(?:\bfig(?:ure)?\.?\s*{int(figure_number)}\b|图\s*{int(figure_number)}\b|第\s*{int(figure_number)}\s*张图)",
            figure_anchor,
            re.IGNORECASE,
        ):
            figure_anchor = f"Figure {int(figure_number)}. {figure_anchor}".strip()
        anchor_text = figure_anchor[:480]
    elif kind == "blockquote":
        claim_type = "blockquote_claim"
        anchor_kind = "blockquote"
        anchor_text = str(evidence_quote or seg_text).strip()[:600]
    elif quote_spans and (not heading_like_quote):
        claim_type = "quote_claim"
        anchor_kind = "quote"
        anchor_text = quote_anchor[:600]
    elif heading_like_quote:
        claim_type = "shell_sentence"
        anchor_kind = ""
        anchor_text = ""
    elif _is_rhetorical_shell_sentence(seg_text):
        claim_type = "shell_sentence"
        anchor_kind = ""
        anchor_text = ""
    else:
        claim_type = "critical_fact_claim"
        anchor_kind = "sentence"
        anchor_text = str(evidence_quote or seg_text).strip()[:320]

    has_identity = bool(str((primary_block or {}).get("block_id") or "").strip()) and bool(
        str(anchor_kind or "").strip()
    ) and bool(str(anchor_text or evidence_quote or "").strip())
    must_locate = bool(
        mode == "direct"
        and has_identity
        and (
            claim_type in {"quote_claim", "blockquote_claim", "formula_claim", "figure_claim"}
            or (
                claim_type == "critical_fact_claim"
                and _critical_fact_score(anchor_text or seg_text) >= 0.34
                and (len(str(anchor_text or "").strip()) >= 14)
            )
        )
    )
    if claim_type == "shell_sentence":
        must_locate = False
    if claim_type == "formula_claim" and eq_number <= 0:
        try:
            eq_number = int((primary_block or {}).get("number") or 0)
        except Exception:
            eq_number = 0
    return {
        "claim_type": claim_type,
        "must_locate": bool(must_locate),
        "anchor_kind": anchor_kind,
        "anchor_text": str(anchor_text or "").strip(),
        "equation_number": int(eq_number or 0),
        "quote_spans": quote_spans,
    }


_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION = 4


_EQUATION_EXPLANATION_PREFIX_RE = re.compile(
    r"^\s*(?:where|wherein|here|with|in which|式中|其中|其中，|其中,|其中：|其中:|其中各项|其中变量)\b",
    re.IGNORECASE,
)
_EQUATION_EXPLANATION_HINT_RE = re.compile(
    r"(?:\bwhere\b|\bdenotes?\b|\brepresents?\b|\bstands?\s+for\b|\bis\s+the\b|\bare\s+the\b|"
    r"\bcorresponds?\s+to\b|表示|记为|定义为|代表|对应|指代|表示为)",
    re.IGNORECASE,
)


def _equation_explanation_score(
    *,
    segment_text: str,
    raw_markdown: str,
    formula_text: str,
    equation_number: int = 0,
    source_distance: int = 0,
) -> float:
    raw = _strip_provenance_noise_text(raw_markdown or segment_text)
    if not raw:
        return 0.0
    if _is_rhetorical_shell_sentence(raw):
        return 0.0
    score = 0.0
    if _EQUATION_EXPLANATION_PREFIX_RE.search(raw):
        score += 0.6
    if _EQUATION_EXPLANATION_HINT_RE.search(raw):
        score += 0.42
    if has_equation_signal(raw) and not _is_display_formula_segment(raw):
        score += 0.22
    if int(equation_number or 0) > 0:
        raw_eq = extract_equation_number(raw)
        if raw_eq > 0 and raw_eq == int(equation_number):
            score += 0.5
    score += 0.58 * _formula_token_overlap_score(raw, formula_text)
    score += 0.24 * _formula_char_similarity(raw, formula_text)
    score += 0.16 * _text_token_overlap_score(raw, formula_text)
    if int(source_distance or 0) == 1:
        score += 0.14
    elif int(source_distance or 0) == 2:
        score += 0.06
    return float(score)


def _figure_block_number(block: dict | None) -> int:
    if not isinstance(block, dict):
        return 0
    try:
        n = int(block.get("number") or 0)
    except Exception:
        n = 0
    if n > 0:
        return int(n)
    return _extract_figure_number(str(block.get("raw_text") or block.get("text") or ""))


def _select_figure_claim_binding(
    segment: dict | None,
    block_lookup: dict[str, dict] | None = None,
) -> tuple[dict | None, dict | None]:
    if not isinstance(segment, dict):
        return None, None
    lookup = block_lookup or {}
    blocks = [block for block in lookup.values() if isinstance(block, dict)]
    if not blocks:
        return None, None

    raw = " ".join(
        [
            str(segment.get("anchor_text") or ""),
            str(segment.get("evidence_quote") or ""),
            str(segment.get("raw_markdown") or segment.get("text") or ""),
        ]
    ).strip()
    figure_number = _extract_figure_number(raw)
    if figure_number <= 0:
        current_primary = lookup.get(str(segment.get("primary_block_id") or "").strip()) or {}
        figure_number = _figure_block_number(current_primary)
    if figure_number <= 0:
        return None, None

    def _order_of(block: dict | None) -> int:
        if not isinstance(block, dict):
            return 0
        try:
            order = int(block.get("order_index") or 0)
        except Exception:
            order = 0
        if order > 0:
            return order
        try:
            line_start = int(block.get("line_start") or 0)
        except Exception:
            line_start = 0
        return max(0, line_start)

    def _caption_score(block: dict | None, figure_order: int) -> float:
        if not isinstance(block, dict):
            return -1.0
        text = str(block.get("raw_text") or block.get("text") or "").strip()
        if _extract_figure_number(text) != figure_number:
            return -1.0
        score = 0.0
        if re.match(rf"^\s*(?:fig(?:ure)?\.?\s*{figure_number}\b|图\s*{figure_number}\b)", text, re.IGNORECASE):
            score += 1.3
        if "single snapshot compressed image" in text.lower():
            score += 0.22
        order = _order_of(block)
        if figure_order > 0 and order > 0:
            dist = abs(order - figure_order)
            if dist == 1:
                score += 0.42
            elif dist == 0:
                score += 0.28
            else:
                score += max(0.0, 0.18 - (0.04 * float(dist)))
        return score

    figure_blocks = [
        block for block in blocks
        if str(block.get("kind") or "").strip().lower() == "figure"
        and _figure_block_number(block) == figure_number
    ]
    current_primary_id = str(segment.get("primary_block_id") or "").strip()
    current_primary = lookup.get(current_primary_id) or {}

    figure_block: dict | None = None
    if figure_blocks:
        figure_blocks.sort(
            key=lambda block: (
                0 if str(block.get("block_id") or "").strip() == current_primary_id else 1,
                _order_of(block) if _order_of(block) > 0 else 10**9,
                str(block.get("block_id") or ""),
            )
        )
        figure_block = figure_blocks[0]

    figure_order = _order_of(figure_block or current_primary)
    caption_candidates = [
        block for block in blocks
        if str(block.get("kind") or "").strip().lower() in {"paragraph", "list_item", "blockquote"}
        and _extract_figure_number(str(block.get("raw_text") or block.get("text") or "")) == figure_number
    ]
    caption_block: dict | None = None
    if caption_candidates:
        caption_candidates.sort(
            key=lambda block: (
                -_caption_score(block, figure_order),
                _order_of(block) if _order_of(block) > 0 else 10**9,
                str(block.get("block_id") or ""),
            )
        )
        caption_block = caption_candidates[0]
        if _caption_score(caption_block, figure_order) < 0.18:
            caption_block = None

    return figure_block, caption_block


_QUOTE_ELLIPSIS_RE = re.compile(r"(?:\[\s*(?:\.{3,}|…)\s*\]|\.{3,}|…)")


def _quote_excerpt_fragments(text: str) -> list[str]:
    raw = _strip_provenance_noise_text(text)
    if not raw:
        return []
    normalized = _QUOTE_ELLIPSIS_RE.sub(" … ", raw)
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\s+…\s+", normalized):
        clean = normalize_inline_markdown(part).strip(" \"'`*[](){}")
        if len(clean) < 14:
            continue
        key = normalize_match_text(clean)
        if (not key) or (key in seen):
            continue
        seen.add(key)
        out.append(clean[:360])
        if len(out) >= 4:
            break
    return out


def _ordered_fragment_match_score(needle: str, haystack: str) -> tuple[float, int]:
    fragments = _quote_excerpt_fragments(needle)
    if len(fragments) < 2:
        return 0.0, 0
    haystack_norm = normalize_match_text(haystack)
    if not haystack_norm:
        return 0.0, 0
    matched = 0
    cursor = 0
    for fragment in fragments:
        fragment_norm = normalize_match_text(fragment)
        if not fragment_norm:
            continue
        idx = haystack_norm.find(fragment_norm, cursor)
        if idx < 0:
            break
        matched += 1
        cursor = idx + len(fragment_norm)
    if matched <= 0:
        return 0.0, 0
    score = 0.26 * float(matched)
    if matched >= 2:
        score += 0.84
    if matched >= len(fragments):
        score += 0.24
    return float(score), int(matched)


def _quote_binding_score(needle: str, block: dict | None) -> tuple[float, str, int]:
    if not isinstance(block, dict):
        return -1.0, "", 0
    block_text_raw = str(block.get("raw_text") or block.get("text") or "").strip()
    block_text = normalize_inline_markdown(block_text_raw)
    probe = normalize_inline_markdown(needle)
    if not block_text or not probe:
        return -1.0, "", 0
    block_norm = normalize_match_text(block_text)
    probe_norm = normalize_match_text(probe)

    score = 0.0
    if probe_norm and probe_norm in block_norm:
        score += 1.54

    ordered_score, matched_fragments = _ordered_fragment_match_score(probe, block_text)
    score += ordered_score

    support_quote, quote_score = _best_evidence_quote_match(probe, block)
    score += 0.92 * float(quote_score)
    score += 0.52 * _text_token_overlap_score(probe, block_text)
    try:
        score += 0.34 * float(SequenceMatcher(None, probe_norm[:520], block_norm[:520]).ratio())
    except Exception:
        pass

    heading_path = str(block.get("heading_path") or "").strip()
    score += 0.08 * _text_token_overlap_score(probe, heading_path)
    if str(block.get("kind") or "").strip().lower() in {"paragraph", "blockquote"}:
        score += 0.05

    return float(score), str(support_quote or "").strip(), int(matched_fragments)


def _select_quote_claim_binding(
    segment: dict | None,
    block_lookup: dict[str, dict] | None = None,
) -> tuple[dict | None, str]:
    if not isinstance(segment, dict):
        return None, ""
    lookup = block_lookup or {}
    blocks = [
        block
        for block in lookup.values()
        if isinstance(block, dict)
        and str(block.get("kind") or "").strip().lower() in {"paragraph", "list_item", "blockquote"}
    ]
    if not blocks:
        return None, ""

    raw_markdown = str(segment.get("raw_markdown") or segment.get("text") or "").strip()
    quoted_spans = _extract_quoted_spans(raw_markdown, min_len=12)
    seeds = _dedupe_str_items(
        quoted_spans
        + [
            str(segment.get("anchor_text") or "").strip(),
            str(segment.get("evidence_quote") or "").strip(),
        ]
    )
    if not seeds and str(segment.get("kind") or "").strip().lower() == "blockquote":
        seeds = [normalize_inline_markdown(raw_markdown or str(segment.get("text") or ""))[:600]]
    if not seeds:
        return None, ""

    current_primary_id = str(segment.get("primary_block_id") or "").strip()
    current_primary = lookup.get(current_primary_id) if current_primary_id else None
    current_best = max((_quote_binding_score(seed, current_primary)[0] for seed in seeds), default=-1.0)

    best_block: dict | None = None
    best_quote = ""
    best_score = -1.0
    best_fragment_hits = 0
    for block in blocks:
        block_best_score = -1.0
        block_best_quote = ""
        block_best_hits = 0
        for seed in seeds:
            score, support_quote, matched_fragments = _quote_binding_score(seed, block)
            if score > block_best_score:
                block_best_score = score
                block_best_quote = support_quote
                block_best_hits = matched_fragments
        if block_best_score > best_score:
            best_block = block
            best_quote = block_best_quote
            best_score = block_best_score
            best_fragment_hits = block_best_hits

    if not isinstance(best_block, dict):
        return None, ""
    if best_score < 0.96:
        return None, ""
    if current_primary_id and str(best_block.get("block_id") or "").strip() == current_primary_id:
        return best_block, best_quote
    if current_best > 0 and best_score < (current_best + 0.12):
        return None, ""

    if best_fragment_hits >= 2:
        block_text = str(best_block.get("raw_text") or best_block.get("text") or "").strip()
        if block_text:
            best_quote = normalize_inline_markdown(block_text)[:900]
    elif not best_quote:
        best_quote = _best_evidence_quote(seeds[0], best_block)

    return best_block, str(best_quote or "").strip()


def _extract_display_formula_snippet(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    m = _DISPLAY_EQ_SEG_RE.search(raw)
    if m:
        return str(m.group(0) or "").strip()[:1200]
    if _EQ_ENV_SEG_RE.search(raw):
        return raw[:1200]
    if "\\[" in raw and "\\]" in raw:
        return raw[:1200]
    return ""


def _is_formula_claim_source_grounded(segment: dict | None, primary_block: dict | None) -> bool:
    if not isinstance(segment, dict):
        return False
    formula_src = _extract_display_formula_snippet(str(segment.get("raw_markdown") or segment.get("text") or ""))
    if not formula_src:
        return True
    support_formula = (
        _extract_display_formula_snippet(str((primary_block or {}).get("raw_text") or (primary_block or {}).get("text") or ""))
        or _extract_display_formula_snippet(str(segment.get("evidence_quote") or ""))
        or str(segment.get("evidence_quote") or "").strip()
    )
    if not support_formula or not has_equation_signal(support_formula):
        return False
    if normalize_match_text(formula_src) == normalize_match_text(support_formula):
        return True
    formula_tok = _formula_token_overlap_score(formula_src, support_formula)
    formula_char = _formula_char_similarity(formula_src, support_formula)
    formula_text = _text_token_overlap_score(formula_src, support_formula)
    if formula_tok >= 0.84 and formula_char >= 0.78:
        return True
    if formula_char >= 0.88 and formula_text >= 0.76:
        return True
    return False


def _formula_claim_alignment_score(segment: dict | None, primary_block: dict | None) -> float:
    if not isinstance(segment, dict) or not isinstance(primary_block, dict):
        return 0.0
    formula_src = _extract_display_formula_snippet(str(segment.get("raw_markdown") or segment.get("text") or ""))
    support_formula = _extract_display_formula_snippet(
        str(primary_block.get("raw_text") or primary_block.get("text") or "")
    )
    if (not formula_src) or (not support_formula):
        return 0.0
    score = 0.0
    if normalize_match_text(formula_src) == normalize_match_text(support_formula):
        score += 0.9
    score += 0.94 * _formula_token_overlap_score(formula_src, support_formula)
    score += 0.78 * _formula_char_similarity(formula_src, support_formula)
    score += 0.18 * _text_token_overlap_score(formula_src, support_formula)
    try:
        seg_eq = int(segment.get("equation_number") or 0)
    except Exception:
        seg_eq = 0
    try:
        block_eq = int(primary_block.get("number") or 0)
    except Exception:
        block_eq = 0
    if seg_eq > 0 and block_eq > 0 and seg_eq == block_eq:
        score += 0.72
    return float(score)


def _apply_provenance_required_coverage_contract(
    segments: list[dict] | None,
    *,
    block_lookup: dict[str, dict] | None = None,
) -> list[dict]:
    out: list[dict] = []
    lookup = block_lookup or {}
    block_order = {
        str(block_id): int((block or {}).get("order_index") or 0)
        for block_id, block in lookup.items()
        if str(block_id or "").strip()
    }
    formula_bundles: list[dict[str, object]] = []
    non_source_scope_active = False

    for seg0 in list(segments or []):
        if not isinstance(seg0, dict):
            continue
        seg = dict(seg0)
        claim_type = str(seg.get("claim_type") or "").strip().lower()
        evidence_mode = str(seg.get("evidence_mode") or "").strip().lower()
        segment_id = str(seg.get("segment_id") or "").strip()
        primary_block_id = str(seg.get("primary_block_id") or "").strip()
        raw_markdown = str(seg.get("raw_markdown") or seg.get("text") or "").strip()
        explicit_non_source = _is_explicit_non_source_segment(raw_markdown)
        seg["related_block_ids"] = _dedupe_str_items(seg.get("related_block_ids") or [])
        if claim_type in {"quote_claim", "blockquote_claim"} and evidence_mode == "direct":
            quote_block, quote_support = _select_quote_claim_binding(seg, lookup)
            if isinstance(quote_block, dict):
                quote_anchor = _longest_quoted_span(raw_markdown or str(seg.get("text") or ""), min_len=12)
                primary_block_id = str(quote_block.get("block_id") or "").strip()
                primary_anchor_id = str(quote_block.get("anchor_id") or "").strip()
                primary_heading_path = str(quote_block.get("heading_path") or "").strip()
                old_primary_block_id = str(seg.get("primary_block_id") or "").strip()
                evidence_block_ids = _dedupe_str_items(
                    [primary_block_id]
                    + list(seg.get("evidence_block_ids") or [])
                )
                support_block_ids = _dedupe_str_items(
                    ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != primary_block_id else [])
                    + list(seg.get("support_block_ids") or [])
                )
                seg["primary_block_id"] = primary_block_id
                seg["primary_anchor_id"] = primary_anchor_id
                seg["primary_heading_path"] = primary_heading_path
                seg["evidence_block_ids"] = evidence_block_ids
                seg["support_block_ids"] = support_block_ids
                if quote_support:
                    seg["evidence_quote"] = quote_support[:900]
                seg["anchor_text"] = str(quote_anchor or quote_support or seg.get("text") or "").strip()[:600]
        primary_block = lookup.get(primary_block_id) if primary_block_id else None
        if claim_type == "formula_claim" and evidence_mode == "direct":
            if not _is_formula_claim_source_grounded(seg, primary_block):
                seg["must_locate"] = False
                seg["locate_policy"] = "hidden"
                seg["locate_surface_policy"] = "hidden"
                seg["formula_origin"] = "derived"
                seg["claim_group_id"] = ""
                seg["claim_group_kind"] = ""
                out.append(seg)
                if explicit_non_source and _opens_non_source_scope(raw_markdown):
                    non_source_scope_active = True
                continue
        if claim_type == "figure_claim" and evidence_mode == "direct":
            figure_block, caption_block = _select_figure_claim_binding(seg, lookup)
            if figure_block or caption_block:
                primary_block = figure_block or caption_block or {}
                primary_block_id = str(primary_block.get("block_id") or "").strip()
                primary_anchor_id = str(primary_block.get("anchor_id") or "").strip()
                primary_heading_path = str(primary_block.get("heading_path") or "").strip()
                old_primary_block_id = str(seg.get("primary_block_id") or "").strip()
                evidence_block_ids = _dedupe_str_items(
                    [primary_block_id]
                    + ([str(caption_block.get("block_id") or "").strip()] if isinstance(caption_block, dict) else [])
                    + list(seg.get("evidence_block_ids") or [])
                )
                support_block_ids = _dedupe_str_items(
                    ([old_primary_block_id] if old_primary_block_id and old_primary_block_id != primary_block_id else [])
                    + list(seg.get("support_block_ids") or [])
                )
                seg["primary_block_id"] = primary_block_id
                seg["primary_anchor_id"] = primary_anchor_id
                seg["primary_heading_path"] = primary_heading_path
                seg["evidence_block_ids"] = evidence_block_ids
                seg["support_block_ids"] = support_block_ids
                if isinstance(caption_block, dict):
                    caption_text = str(caption_block.get("raw_text") or caption_block.get("text") or "").strip()
                    if caption_text:
                        seg["anchor_text"] = caption_text[:480]
                        seg["evidence_quote"] = caption_text[:900]
        if non_source_scope_active and _is_non_source_scope_boundary(raw_markdown):
            non_source_scope_active = False
        non_source_scoped = bool(explicit_non_source or non_source_scope_active)
        locate_policy = "hidden"
        locate_surface_policy = "hidden"
        claim_group_id = ""
        claim_group_kind = ""
        formula_origin = str(seg.get("formula_origin") or "").strip().lower()

        if non_source_scoped:
            locate_policy = "hidden"
            locate_surface_policy = "hidden"
            seg["must_locate"] = False
        elif claim_type in {"quote_claim", "blockquote_claim"} and evidence_mode == "direct":
            locate_policy = "required"
            locate_surface_policy = "primary"
            claim_group_kind = "quote_bundle"
            claim_group_id = f"quote_bundle:{primary_block_id or segment_id or uuid.uuid4().hex[:8]}"
        elif claim_type == "formula_claim" and evidence_mode == "direct":
            locate_policy = "required"
            locate_surface_policy = "primary"
            formula_origin = "source"
        elif claim_type == "figure_claim" and evidence_mode == "direct":
            locate_policy = "required"
            locate_surface_policy = "primary"
        elif claim_type == "shell_sentence":
            locate_policy = "hidden"
        elif bool(seg.get("must_locate")):
            locate_policy = "required"
            locate_surface_policy = "primary"
        elif evidence_mode == "direct":
            locate_policy = "optional"
            locate_surface_policy = "hidden"

        if claim_type == "formula_claim" and evidence_mode == "direct" and not non_source_scoped:
            bundle_key = primary_block_id or segment_id or uuid.uuid4().hex[:8]
            claim_group_id = f"formula_bundle:{bundle_key}"
            claim_group_kind = "formula_bundle"
            formula_bundles.append(
                {
                    "segment_id": segment_id,
                    "segment_index": int(seg.get("segment_index") or 0),
                    "primary_block_id": primary_block_id,
                    "primary_anchor_id": str(seg.get("primary_anchor_id") or "").strip(),
                    "primary_heading_path": str(seg.get("primary_heading_path") or "").strip(),
                    "equation_number": int(seg.get("equation_number") or 0),
                    "anchor_text": str(seg.get("anchor_text") or "").strip(),
                    "evidence_quote": str(seg.get("evidence_quote") or "").strip(),
                    "claim_group_id": claim_group_id,
                    "claim_group_kind": claim_group_kind,
                    "source_block_order": int(block_order.get(primary_block_id) or 0),
                    "alignment_score": _formula_claim_alignment_score(seg, primary_block),
                }
            )

        seg["locate_policy"] = locate_policy
        seg["locate_surface_policy"] = locate_surface_policy
        seg["claim_group_id"] = claim_group_id
        seg["claim_group_kind"] = claim_group_kind
        seg["formula_origin"] = formula_origin
        out.append(seg)
        if explicit_non_source and _opens_non_source_scope(raw_markdown):
            non_source_scope_active = True

    if not out or not formula_bundles:
        quote_groups: dict[str, list[dict]] = {}
        for seg in out:
            if not isinstance(seg, dict):
                continue
            if str(seg.get("claim_group_kind") or "").strip().lower() != "quote_bundle":
                continue
            quote_groups.setdefault(str(seg.get("claim_group_id") or "").strip(), []).append(seg)
        for items in quote_groups.values():
            dedupe_by_key: dict[str, list[dict]] = {}
            for seg in items:
                key = " :: ".join(
                    [
                        str(seg.get("primary_block_id") or "").strip(),
                        normalize_match_text(
                            str(seg.get("anchor_text") or seg.get("evidence_quote") or seg.get("text") or "")
                        )[:420],
                    ]
                ).strip()
                if not key:
                    continue
                dedupe_by_key.setdefault(key, []).append(seg)
            for dup_items in dedupe_by_key.values():
                if len(dup_items) <= 1:
                    continue
                dup_items.sort(
                    key=lambda seg: (
                        -len(str(seg.get("anchor_text") or seg.get("evidence_quote") or "")),
                        int(seg.get("segment_index") or 0),
                    )
                )
                keep = dup_items[0]
                keep["locate_surface_policy"] = "primary"
                for seg in dup_items[1:]:
                    seg["must_locate"] = False
                    seg["locate_policy"] = "hidden"
                    seg["locate_surface_policy"] = "hidden"
        return out

    bundle_items_by_id: dict[str, list[dict[str, object]]] = {}
    for bundle in formula_bundles:
        bundle_id = str(bundle.get("claim_group_id") or "").strip()
        if not bundle_id:
            continue
        bundle_items_by_id.setdefault(bundle_id, []).append(bundle)

    bundle_formula_segment_ids: set[str] = set()
    representative_bundles: list[dict[str, object]] = []
    for bundle_id, items in bundle_items_by_id.items():
        ranked_items = sorted(
            items,
            key=lambda item: (
                -float(item.get("alignment_score") or 0.0),
                -(1 if int(item.get("equation_number") or 0) > 0 else 0),
                -len(str(item.get("anchor_text") or item.get("evidence_quote") or "")),
                int(item.get("segment_index") or 0),
            ),
        )
        if not ranked_items:
            continue
        primary_item = ranked_items[0]
        representative_bundles.append(primary_item)
        primary_segment_id = str(primary_item.get("segment_id") or "").strip()
        for seg in out:
            if str(seg.get("claim_group_id") or "").strip() != bundle_id:
                continue
            if str(seg.get("claim_type") or "").strip().lower() != "formula_claim":
                continue
            segment_id = str(seg.get("segment_id") or "").strip()
            if not segment_id:
                continue
            bundle_formula_segment_ids.add(segment_id)
            if segment_id == primary_segment_id:
                seg["must_locate"] = True
                seg["locate_policy"] = "required"
                seg["locate_surface_policy"] = "primary"
                seg["formula_origin"] = "source"
                continue
            seg["must_locate"] = False
            seg["locate_policy"] = "hidden"
            seg["locate_surface_policy"] = "hidden"
            seg["formula_origin"] = "derived"

    used_segment_ids: set[str] = set(bundle_formula_segment_ids)
    for bundle in representative_bundles:
        formula_segment_id = str(bundle.get("segment_id") or "").strip()
        if formula_segment_id:
            used_segment_ids.add(formula_segment_id)
        formula_block_id = str(bundle.get("primary_block_id") or "").strip()
        formula_anchor_id = str(bundle.get("primary_anchor_id") or "").strip()
        formula_heading_path = str(bundle.get("primary_heading_path") or "").strip()
        formula_number = int(bundle.get("equation_number") or 0)
        formula_text = str(bundle.get("anchor_text") or bundle.get("evidence_quote") or "").strip()
        formula_order = int(bundle.get("source_block_order") or 0)
        claim_group_id = str(bundle.get("claim_group_id") or "").strip()
        claim_group_kind = str(bundle.get("claim_group_kind") or "").strip()
        formula_seg_index = int(bundle.get("segment_index") or 0)
        if not formula_block_id:
            continue

        for seg in out:
            segment_id = str(seg.get("segment_id") or "").strip()
            if (not segment_id) or (segment_id in used_segment_ids):
                continue
            evidence_mode = str(seg.get("evidence_mode") or "").strip().lower()
            if evidence_mode != "direct":
                continue
            claim_type = str(seg.get("claim_type") or "").strip().lower()
            if claim_type in {"formula_claim", "quote_claim", "blockquote_claim", "figure_claim"}:
                continue
            raw_markdown = str(seg.get("raw_markdown") or seg.get("text") or "").strip()
            segment_text = str(seg.get("text") or "").strip()
            if not segment_text:
                continue
            source_primary_block_id = str(seg.get("primary_block_id") or "").strip()
            source_order = int(block_order.get(source_primary_block_id) or 0)
            source_distance = 0
            if formula_order > 0 and source_order > 0:
                source_distance = abs(source_order - formula_order)
            else:
                source_distance = abs(int(seg.get("segment_index") or 0) - formula_seg_index)
            if source_distance <= 0 or source_distance > 2:
                continue
            score = _equation_explanation_score(
                segment_text=segment_text,
                raw_markdown=raw_markdown,
                formula_text=formula_text,
                equation_number=formula_number,
                source_distance=source_distance,
            )
            if score < 0.58:
                continue
            explanation_block = lookup.get(source_primary_block_id) if source_primary_block_id else None
            explanation_block_id = str((explanation_block or {}).get("block_id") or source_primary_block_id).strip()
            explanation_anchor_id = str((explanation_block or {}).get("anchor_id") or seg.get("primary_anchor_id") or "").strip()
            explanation_heading_path = str((explanation_block or {}).get("heading_path") or seg.get("primary_heading_path") or "").strip()
            explanation_anchor_kind = str(seg.get("anchor_kind") or "").strip().lower() or "sentence"
            if explanation_anchor_kind in {"equation", "figure", "quote"}:
                explanation_anchor_kind = "sentence"
            related_block_ids = _dedupe_str_items([formula_block_id] + list(seg.get("related_block_ids") or []))
            if (not explanation_block_id) or explanation_block_id == formula_block_id:
                seg["claim_type"] = "equation_explanation_claim"
                seg["must_locate"] = False
                seg["locate_policy"] = "hidden"
                seg["locate_surface_policy"] = "hidden"
                seg["formula_origin"] = "explanation"
                seg["claim_group_id"] = claim_group_id
                seg["claim_group_kind"] = claim_group_kind
                seg["related_block_ids"] = related_block_ids
                used_segment_ids.add(segment_id)
                continue
            evidence_block_ids = _dedupe_str_items(
                [explanation_block_id, formula_block_id]
                + list(seg.get("evidence_block_ids") or [])
            )
            support_block_ids = _dedupe_str_items(
                ([formula_block_id] if formula_block_id and formula_block_id != explanation_block_id else [])
                + ([source_primary_block_id] if source_primary_block_id and source_primary_block_id not in {formula_block_id, explanation_block_id} else [])
                + list(seg.get("support_block_ids") or [])
            )
            seg["claim_type"] = "equation_explanation_claim"
            seg["must_locate"] = True
            seg["anchor_kind"] = explanation_anchor_kind
            seg["equation_number"] = int(formula_number or 0)
            seg["primary_block_id"] = explanation_block_id
            seg["primary_anchor_id"] = explanation_anchor_id
            seg["primary_heading_path"] = explanation_heading_path or formula_heading_path
            seg["evidence_block_ids"] = evidence_block_ids
            seg["support_block_ids"] = support_block_ids
            seg["locate_policy"] = "required"
            seg["locate_surface_policy"] = "secondary"
            seg["claim_group_id"] = claim_group_id
            seg["claim_group_kind"] = claim_group_kind
            seg["formula_origin"] = "explanation"
            seg["related_block_ids"] = related_block_ids
            anchor_text = str(seg.get("anchor_text") or "").strip()
            if not anchor_text:
                seg["anchor_text"] = str(segment_text or raw_markdown)[:900]
            used_segment_ids.add(segment_id)

    quote_groups: dict[str, list[dict]] = {}
    for seg in out:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("claim_group_kind") or "").strip().lower() != "quote_bundle":
            continue
        quote_groups.setdefault(str(seg.get("claim_group_id") or "").strip(), []).append(seg)
    for items in quote_groups.values():
        dedupe_by_key: dict[str, list[dict]] = {}
        for seg in items:
            key = " :: ".join(
                [
                    str(seg.get("primary_block_id") or "").strip(),
                    normalize_match_text(
                        str(seg.get("anchor_text") or seg.get("evidence_quote") or seg.get("text") or "")
                    )[:420],
                ]
            ).strip()
            if not key:
                continue
            dedupe_by_key.setdefault(key, []).append(seg)
        for dup_items in dedupe_by_key.values():
            if len(dup_items) <= 1:
                continue
            dup_items.sort(
                key=lambda seg: (
                    -len(str(seg.get("anchor_text") or seg.get("evidence_quote") or "")),
                    int(seg.get("segment_index") or 0),
                )
            )
            keep = dup_items[0]
            keep["locate_surface_policy"] = "primary"
            for seg in dup_items[1:]:
                seg["must_locate"] = False
                seg["locate_policy"] = "hidden"
                seg["locate_surface_policy"] = "hidden"

    return out


def _strict_identity_missing_reasons(segment: dict | None) -> list[str]:
    if not isinstance(segment, dict):
        return ["invalid_segment"]
    reasons: list[str] = []
    primary_block_id = str(segment.get("primary_block_id") or "").strip()
    evidence_block_ids = [
        str(item or "").strip()
        for item in list(segment.get("evidence_block_ids") or [])
        if str(item or "").strip()
    ]
    anchor_kind = str(segment.get("anchor_kind") or "").strip().lower()
    anchor_text = str(segment.get("anchor_text") or "").strip()
    evidence_quote = str(segment.get("evidence_quote") or "").strip()
    if not primary_block_id:
        reasons.append("missing_primary_block_id")
    if not evidence_block_ids:
        reasons.append("missing_evidence_block_ids")
    if not anchor_kind:
        reasons.append("missing_anchor_kind")
    if not (anchor_text or evidence_quote):
        reasons.append("missing_anchor_text_or_evidence_quote")
    return reasons


def _apply_provenance_strict_identity_contract(
    segments: list[dict] | None,
) -> tuple[list[dict], dict[str, object]]:
    hardened: list[dict] = []
    must_locate_candidate_count = 0
    strict_identity_count = 0
    identity_missing_reasons: dict[str, int] = {}
    identity_missing_segments: list[dict[str, object]] = []

    for seg0 in list(segments or []):
        if not isinstance(seg0, dict):
            continue
        seg = dict(seg0)
        missing_reasons: list[str] = []
        if bool(seg.get("must_locate")):
            must_locate_candidate_count += 1
            missing_reasons = _strict_identity_missing_reasons(seg)
            if missing_reasons:
                seg["must_locate"] = False
                for reason in missing_reasons:
                    identity_missing_reasons[reason] = int(identity_missing_reasons.get(reason) or 0) + 1
                identity_missing_segments.append(
                    {
                        "segment_id": str(seg.get("segment_id") or "").strip(),
                        "claim_type": str(seg.get("claim_type") or "").strip(),
                        "reasons": missing_reasons,
                    }
                )
            else:
                strict_identity_count += 1
        seg["strict_identity_missing_reasons"] = missing_reasons
        hardened.append(seg)

    must_locate_count = sum(1 for seg in hardened if bool(seg.get("must_locate")))
    strict_identity_ready = bool(must_locate_candidate_count == strict_identity_count)
    if must_locate_candidate_count <= 0:
        strict_identity_ready = True
    return hardened, {
        "provenance_schema_version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
        "must_locate_candidate_count": int(must_locate_candidate_count),
        "must_locate_count": int(must_locate_count),
        "strict_identity_count": int(strict_identity_count),
        "strict_identity_ready": bool(strict_identity_ready),
        "identity_missing_reasons": identity_missing_reasons,
        "identity_missing_segments": identity_missing_segments,
    }


def _segment_type_from_text(text: str, *, segment_kind: str = "") -> str:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return "other"
    head = src[:48]
    kind = str(segment_kind or "").strip().lower()
    if _is_display_formula_segment(src, segment_kind=kind):
        return "equation_explanation"
    if head.startswith("结论") or head.startswith("核心结论"):
        return "claim"
    if head.startswith("依据") or head.startswith("证据") or head.startswith("原文"):
        return "evidence"
    if head.startswith("下一步") or head.startswith("建议") or head.startswith("行动"):
        return "next_step"
    if kind == "list_item":
        return "bullet"
    return "prose"


def _segment_focus_tags(text: str) -> set[str]:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return set()
    tags: set[str] = set()
    if _CLAIM_EXPERIMENT_HINT_RE.search(src):
        tags.add("experiment")
    if _CLAIM_METHOD_HINT_RE.search(src):
        tags.add("method")
    if has_equation_signal(src):
        tags.add("formula")
    return tags


def _summary_segment_tags(text: str) -> set[str]:
    src = normalize_inline_markdown(str(text or ""))
    if not src:
        return set()
    tags: set[str] = set()
    if _SUMMARY_NOVELTY_HINT_RE.search(src):
        tags.add("novelty")
    if _SUMMARY_RESULT_HINT_RE.search(src):
        tags.add("result")
    return tags


def _summary_block_adjustment(segment_text: str, block: dict | None) -> float:
    if not isinstance(block, dict):
        return 0.0
    tags = _summary_segment_tags(segment_text)
    if not tags:
        return 0.0
    block_text = normalize_inline_markdown(str(block.get("raw_text") or block.get("text") or ""))
    if not block_text:
        return 0.0
    heading = normalize_match_text(str(block.get("heading_path") or ""))
    kind = str(block.get("kind") or "").strip().lower()
    score = 0.0
    text_has_contribution = bool(_CONTRIBUTION_BLOCK_HINT_RE.search(block_text))
    text_has_result = bool(_RESULT_BLOCK_HINT_RE.search(block_text))
    definition_like = bool(_DEFINITION_LIKE_BLOCK_HINT_RE.search(block_text))
    leadin_only = bool(_CONTRIBUTION_LEADIN_HINT_RE.search(block_text))

    if kind == "heading":
        score -= 0.22
    if kind == "equation":
        score -= 0.36

    if "novelty" in tags:
        if text_has_contribution:
            score += 0.72
            if kind == "list_item":
                score += 0.18
            if "abstract" in heading:
                score += 0.18
            elif "introduction" in heading:
                score += 0.10
        if definition_like:
            score -= 0.88
        if leadin_only:
            score -= 0.14

    if "result" in tags:
        if text_has_result:
            score += 0.58
            if kind == "list_item":
                score += 0.12
            if any(token in heading for token in _EXPERIMENT_HEADING_HINTS):
                score += 0.16
            elif "abstract" in heading:
                score += 0.08
        if definition_like:
            score -= 0.46
        if leadin_only:
            score -= 0.10

    return float(score)


def _heading_focus_adjustment(segment_text: str, heading_path: str) -> float:
    heading = normalize_match_text(heading_path)
    if not heading:
        return 0.0
    tags = _segment_focus_tags(segment_text)
    generic = any(token in heading for token in _GENERIC_HEADING_HINTS)
    experiment = any(token in heading for token in _EXPERIMENT_HEADING_HINTS)
    method = any(token in heading for token in _METHOD_HEADING_HINTS)
    score = 0.0
    if generic:
        score -= 0.18
        if "abstract" in heading:
            score -= 0.08
        if "related work" in heading or "reference" in heading:
            score -= 0.12
    if "experiment" in tags:
        if experiment:
            score += 0.26
        elif generic:
            score -= 0.14
    if "method" in tags:
        if method:
            score += 0.18
        elif generic and ("experiment" not in tags):
            score -= 0.06
    return score


def _is_generic_heading_path(heading_path: str) -> bool:
    heading = normalize_match_text(heading_path)
    if not heading:
        return False
    return any(token in heading for token in _GENERIC_HEADING_HINTS)


def _best_evidence_quote_match(segment_text: str, block: dict | None) -> tuple[str, float]:
    if not isinstance(block, dict):
        return "", 0.0
    block_text_raw = str(block.get("raw_text") or block.get("text") or "").strip()
    if not block_text_raw:
        return "", 0.0
    block_text = normalize_inline_markdown(block_text_raw)
    if not block_text:
        return "", 0.0
    if has_equation_signal(segment_text) or str(block.get("kind") or "").strip().lower() == "equation":
        snippet = block_text[:320]
        score = _text_token_overlap_score(segment_text, snippet)
        if snippet:
            try:
                score += 0.22 * float(SequenceMatcher(None, normalize_match_text(segment_text)[:420], normalize_match_text(snippet)[:420]).ratio())
            except Exception:
                pass
        return snippet, score

    segment_key = normalize_match_text(segment_text)
    best = ""
    best_score = -1.0
    sentences = [s.strip() for s in _SEG_SENT_SPLIT_RE.split(block_text) if str(s).strip()]
    if not sentences:
        sentences = [block_text]
    for sent in sentences[:16]:
        sent_norm = normalize_match_text(sent)
        if not sent_norm:
            continue
        score = 0.0
        if sent_norm == segment_key:
            score += 1.2
        elif segment_key and (segment_key in sent_norm or sent_norm in segment_key):
            score += 0.82
        score += _text_token_overlap_score(segment_text, sent)
        try:
            score += 0.28 * float(SequenceMatcher(None, segment_key[:420], sent_norm[:420]).ratio())
        except Exception:
            pass
        if has_equation_signal(sent):
            score += 0.22 * _formula_token_overlap_score(segment_text, sent)
            score += 0.12 * _formula_char_similarity(segment_text, sent)
        if score > best_score:
            best = sent
            best_score = score
    if best and best_score >= 0.16:
        return best[:320], max(0.0, best_score)
    fallback = block_text[:220]
    fallback_score = _text_token_overlap_score(segment_text, fallback)
    try:
        fallback_score += 0.18 * float(
            SequenceMatcher(None, normalize_match_text(segment_text)[:420], normalize_match_text(fallback)[:420]).ratio()
        )
    except Exception:
        pass
    return fallback, max(0.0, fallback_score)


def _best_evidence_quote(segment_text: str, block: dict | None) -> str:
    return _best_evidence_quote_match(segment_text, block)[0]


def _expand_match_snippet_hints(text: str, *, max_items: int = 6) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _push(item: str) -> None:
        clean = normalize_inline_markdown(item)
        if len(clean) < 14:
            return
        key = normalize_match_text(clean)
        if (not key) or (key in seen):
            return
        seen.add(key)
        out.append(clean[:360])

    for line in raw.splitlines():
        stripped = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", str(line or "")).strip()
        if not stripped:
            continue
        clean = normalize_inline_markdown(stripped)
        if len(clean) >= 24:
            _push(stripped)
            if len(out) >= max_items:
                return out

    clean_raw = normalize_inline_markdown(raw)
    if clean_raw:
        for sent in [s.strip() for s in _SEG_SENT_SPLIT_RE.split(clean_raw) if str(s).strip()]:
            if len(sent) < 22:
                continue
            _push(sent)
            if len(out) >= max_items:
                return out
        _push(clean_raw)
    return out[:max_items]


def _block_support_metrics(segment_text: str, block: dict | None) -> dict[str, float | str | bool]:
    if not isinstance(block, dict):
        return {
            "quote": "",
            "quote_score": 0.0,
            "support_score": 0.0,
            "heading_adjust": 0.0,
            "generic_heading": False,
        }
    block_text = normalize_inline_markdown(str(block.get("raw_text") or block.get("text") or ""))
    if not block_text:
        return {
            "quote": "",
            "quote_score": 0.0,
            "support_score": 0.0,
            "heading_adjust": 0.0,
            "generic_heading": False,
        }
    heading_path = str(block.get("heading_path") or "").strip()
    quote, quote_score = _best_evidence_quote_match(segment_text, block)
    block_overlap = _text_token_overlap_score(segment_text, block_text)
    block_ratio = 0.0
    try:
        block_ratio = float(
            SequenceMatcher(
                None,
                normalize_match_text(segment_text)[:420],
                normalize_match_text(block_text)[:420],
            ).ratio()
        )
    except Exception:
        block_ratio = 0.0
    support_score = max(quote_score, block_overlap + (0.22 * block_ratio))
    heading_adjust = _heading_focus_adjustment(segment_text, heading_path)
    return {
        "quote": quote,
        "quote_score": float(max(0.0, quote_score)),
        "support_score": float(max(0.0, support_score)),
        "heading_adjust": float(heading_adjust),
        "generic_heading": bool(_is_generic_heading_path(heading_path)),
    }


def _extract_json_object_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{[\s\S]*\}", text)
    return str(m.group(0) or "").strip() if m else ""


def _pick_blocks_with_llm(
    *,
    ds: DeepSeekChat | None,
    segment_text: str,
    segment_kind: str,
    is_formula_segment: bool,
    candidate_rows: list[dict],
    max_pick: int = 2,
) -> list[str]:
    if ds is None:
        return []
    rows = [row for row in list(candidate_rows or []) if isinstance(row, dict)]
    if len(rows) < 2:
        return []

    cand_rows = rows[: min(6, len(rows))]
    allowed_ids: set[str] = set()
    cand_lines: list[str] = []
    for idx, row in enumerate(cand_rows, start=1):
        block = row.get("block")
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        allowed_ids.add(block_id)
        score = float(row.get("score") or 0.0)
        kind = str(block.get("kind") or "").strip()
        heading = str(block.get("heading_path") or "").strip()
        number = int(block.get("number") or 0)
        block_text_raw = str(block.get("raw_text") or block.get("text") or "")
        text = normalize_inline_markdown(block_text_raw)[:420]
        cand_lines.append(
            f"{idx}. block_id={block_id} kind={kind} number={number} score={score:.3f}\n"
            f"   heading={heading}\n"
            f"   text={text}"
        )
    if len(cand_lines) < 2:
        return []

    sys_msg = (
        "You are a strict evidence mapper. Choose source blocks that directly support the answer segment. "
        "If no candidate clearly supports, return empty ids."
    )
    formula_rule = (
        "- For formula segments, prioritize same equation number; if absent, allow mathematically equivalent form with different symbols.\n"
        if is_formula_segment else
        "- For non-formula segments, choose directly supporting original statements, not broad topic neighbors.\n"
    )
    user_msg = (
        "Task: choose at most "
        f"{max(1, int(max_pick))} block_id values.\n"
        "Rules:\n"
        "- Prefer exact semantic grounding over loose topic similarity.\n"
        + formula_rule +
        "- Do NOT guess.\n"
        "Return JSON only: {\"ids\": [\"block_id\", ...]}.\n\n"
        f"segment_kind={segment_kind}\n"
        f"is_formula_segment={int(bool(is_formula_segment))}\n"
        f"segment_text={segment_text[:520]}\n\n"
        "candidates:\n"
        + "\n".join(cand_lines)
    )
    try:
        out = ds.chat(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=220,
        )
    except Exception:
        return []

    obj_text = _extract_json_object_text(out)
    if not obj_text:
        return []
    try:
        obj = json.loads(obj_text)
    except Exception:
        return []
    ids_raw = obj.get("ids") if isinstance(obj, dict) else []
    if not isinstance(ids_raw, list):
        return []
    picked: list[str] = []
    seen: set[str] = set()
    for it in ids_raw:
        bid = str(it or "").strip()
        if (not bid) or (bid in seen) or (bid not in allowed_ids):
            continue
        seen.add(bid)
        picked.append(bid)
        if len(picked) >= max(1, int(max_pick)):
            break
    return picked


def _ensure_provenance_block_entry(block_map: dict[str, dict], block: dict) -> None:
    block_id = str(block.get("block_id") or "").strip()
    if (not block_id) or (block_id in block_map):
        return
    block_map[block_id] = {
        "block_id": block_id,
        "anchor_id": str(block.get("anchor_id") or "").strip(),
        "kind": str(block.get("kind") or "").strip(),
        "heading_path": str(block.get("heading_path") or "").strip(),
        "text": str(block.get("text") or ""),
        "line_start": int(block.get("line_start") or 0),
        "line_end": int(block.get("line_end") or 0),
        "number": int(block.get("number") or 0),
    }


def _collect_paper_guide_block_pool(
    *,
    blocks: list[dict],
    answer_hits: list[dict],
    bound_source_path: str,
    bound_source_name: str,
) -> list[dict]:
    best_by_block: dict[str, dict] = {}

    def _block_formula_seed_score(block: dict) -> float:
        kind = str(block.get("kind") or "").strip().lower()
        text = str(block.get("text") or "")
        score = 0.0
        if kind == "equation":
            score += 0.34
        if "$$" in text or "\\begin{equation" in text.lower():
            score += 0.38
        if has_equation_signal(text):
            score += 0.22
        try:
            if int(block.get("number") or 0) > 0:
                score += 0.18
        except Exception:
            pass
        return score

    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        if not _is_hit_from_bound_source(
            hit,
            bound_source_path=bound_source_path,
            bound_source_name=bound_source_name,
        ):
            continue
        meta = hit.get("meta", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        heading_hints: list[str] = []
        for value in (
            meta.get("ref_best_heading_path"),
            meta.get("heading_path"),
            meta.get("top_heading"),
        ):
            text = normalize_inline_markdown(str(value or ""))
            if text and text not in heading_hints:
                heading_hints.append(text)
        raw_locs = meta.get("ref_locs")
        if isinstance(raw_locs, list):
            for loc in raw_locs[:4]:
                if not isinstance(loc, dict):
                    continue
                hp = normalize_inline_markdown(str(loc.get("heading_path") or loc.get("heading") or ""))
                if hp and hp not in heading_hints:
                    heading_hints.append(hp)
        snippet_hints: list[str] = []
        raw_snips = meta.get("ref_show_snippets")
        if isinstance(raw_snips, list):
            for item in raw_snips[:4]:
                for text in _expand_match_snippet_hints(str(item or ""), max_items=4):
                    if text and text not in snippet_hints:
                        snippet_hints.append(text)
        hit_text = normalize_inline_markdown(str(hit.get("text") or ""))
        for text in _expand_match_snippet_hints(str(hit.get("text") or ""), max_items=6):
            if text and text not in snippet_hints:
                snippet_hints.append(text)
        prefer_kind = str(meta.get("anchor_target_kind") or "").strip().lower()
        try:
            target_number = int(meta.get("anchor_target_number") or 0)
        except Exception:
            target_number = 0

        match_jobs: list[tuple[str, str]] = []
        if snippet_hints:
            for snippet in snippet_hints[:4]:
                for heading in heading_hints[:2] or [""]:
                    match_jobs.append((snippet, heading))
        elif heading_hints:
            for heading in heading_hints[:2]:
                match_jobs.append(("", heading))
        elif target_number > 0:
            match_jobs.append(("", ""))

        for snippet, heading in match_jobs[:8]:
            rows = match_source_blocks(
                blocks,
                snippet=snippet,
                heading_path=heading,
                prefer_kind=prefer_kind,
                target_number=target_number,
                limit=3,
            )
            for row in rows:
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                block_id = str(block.get("block_id") or "").strip()
                if not block_id:
                    continue
                score = float(row.get("score") or 0.0)
                prev = best_by_block.get(block_id)
                if prev is None or score > float(prev.get("score") or 0.0):
                    best_by_block[block_id] = {
                        "score": score,
                        "block": block,
                    }

    # Always keep a small set of formula-capable blocks in pool.
    # This avoids missing equation grounding when refs snippets are paragraph-heavy.
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        seed_score = _block_formula_seed_score(block)
        if seed_score <= 0.0:
            continue
        prev = best_by_block.get(block_id)
        if prev is None or seed_score > float(prev.get("score") or 0.0):
            best_by_block[block_id] = {
                "score": seed_score,
                "block": block,
            }

    ranked = sorted(best_by_block.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return ranked[:24]


def _build_paper_guide_answer_provenance(
    *,
    answer: str,
    answer_hits: list[dict],
    bound_source_path: str,
    bound_source_name: str,
    db_dir: Path | str | None,
    settings_obj: object | None = None,
    llm_rerank: bool = False,
) -> dict | None:
    source_path = str(bound_source_path or "").strip()
    if not source_path:
        return None
    md_path = _resolve_paper_guide_md_path(source_path, db_dir=db_dir)
    if md_path is None:
        return None
    try:
        blocks = load_source_blocks(md_path)
    except Exception:
        return None
    if not blocks:
        return None
    block_lookup = {
        str(block.get("block_id") or "").strip(): dict(block)
        for block in blocks
        if isinstance(block, dict) and str(block.get("block_id") or "").strip()
    }

    evidence_pool = _collect_paper_guide_block_pool(
        blocks=blocks,
        answer_hits=answer_hits,
        bound_source_path=bound_source_path,
        bound_source_name=bound_source_name,
    )
    mapping_mode = "fast"
    empty_contract_meta = {
        "provenance_schema_version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
        "must_locate_candidate_count": 0,
        "must_locate_count": 0,
        "strict_identity_count": 0,
        "strict_identity_ready": True,
        "identity_missing_reasons": {},
        "identity_missing_segments": [],
    }

    if not evidence_pool:
        return {
            "version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
            "source_path": source_path,
            "source_name": str(bound_source_name or Path(source_path).name or "").strip(),
            "md_path": str(md_path),
            "doc_id": str(blocks[0].get("doc_id") or ""),
            "segments": [],
            "block_map": {},
            "status": "no_evidence_pool",
            "mapping_mode": mapping_mode,
            "llm_rerank_enabled": bool(llm_rerank),
            "llm_rerank_calls": 0,
            **empty_contract_meta,
        }

    candidate_blocks = [dict(item.get("block") or {}) for item in evidence_pool if isinstance(item.get("block"), dict)]
    block_map: dict[str, dict] = {}
    segments_out: list[dict] = []
    segments = split_answer_segments(answer)
    llm_picker: DeepSeekChat | None = None
    llm_calls_used = 0
    try:
        llm_max_calls = int(os.environ.get("KB_PROVENANCE_LLM_MAX_CALLS", "3") or 3)
    except Exception:
        llm_max_calls = 3
    llm_max_calls = max(0, min(6, llm_max_calls))
    if bool(llm_rerank) and (llm_max_calls > 0):
        try:
            llm_picker = DeepSeekChat(settings_obj) if settings_obj is not None else None
        except Exception:
            llm_picker = None
    def _is_formula_block(block: dict) -> bool:
        kind = str(block.get("kind") or "").strip().lower()
        text = str(block.get("text") or "")
        if kind == "equation":
            return True
        if "$$" in text or "\\begin{equation" in text.lower():
            return True
        if has_equation_signal(text) and ("=" in text or "\\tag{" in text.lower()):
            return True
        return False

    global_formula_blocks = [dict(block) for block in (blocks or []) if isinstance(block, dict) and _is_formula_block(block)]
    if not global_formula_blocks:
        global_formula_blocks = [dict(block) for block in candidate_blocks if isinstance(block, dict) and _is_formula_block(block)]

    for idx, segment in enumerate(segments, start=1):
        seg_text = normalize_inline_markdown(str(segment.get("text") or ""))
        if len(seg_text) < 10:
            continue
        seg_kind = str(segment.get("kind") or "paragraph")
        raw_markdown = str(segment.get("raw_markdown") or segment.get("raw_text") or seg_text).strip()
        word_count = len(_LATIN_WORD_RE.findall(seg_text)) + len(_CJK_WORD_RE.findall(seg_text))
        is_formula = _is_display_formula_segment(seg_text, segment_kind=seg_kind)
        eq_number = extract_equation_number(seg_text) if has_equation_signal(seg_text) else 0
        quoted_spans = _extract_quoted_spans(raw_markdown or seg_text, min_len=12)
        quote_anchor = _longest_quoted_span(raw_markdown or seg_text, min_len=12)
        probe_text = str(quote_anchor or seg_text).strip()
        summary_tags = _summary_segment_tags(seg_text) if not is_formula else set()
        prefer_kind = "equation" if is_formula else ""
        base_blocks = candidate_blocks
        if (not is_formula) and (str(seg_kind).strip().lower() == "blockquote" or bool(quote_anchor)):
            base_blocks = blocks
        if is_formula and global_formula_blocks:
            base_blocks = global_formula_blocks
        if is_formula:
            rank_limit = 10
        elif summary_tags:
            rank_limit = max(12, min(24, len(base_blocks or [])))
        else:
            rank_limit = 8 if quote_anchor else 5
        ranked = match_source_blocks(
            base_blocks,
            snippet=probe_text,
            prefer_kind=prefer_kind,
            target_number=eq_number,
            limit=rank_limit,
            score_floor=(0.12 if summary_tags else None),
        )
        if (not ranked) and (not is_formula) and quote_anchor and base_blocks is not candidate_blocks:
            ranked = match_source_blocks(
                candidate_blocks,
                snippet=probe_text,
                prefer_kind="",
                target_number=0,
                limit=6,
            )
        if is_formula and (not ranked):
            ranked = match_source_blocks(
                candidate_blocks,
                snippet=seg_text,
                prefer_kind="equation",
                target_number=eq_number,
                limit=10,
            )
        if is_formula:
            ranked_formula = []
            ranked_other = []
            for row in ranked:
                block0 = row.get("block")
                if isinstance(block0, dict) and _is_formula_block(block0):
                    score = float(row.get("score") or 0.0)
                    formula_text = str(block0.get("text") or "")
                    formula_tok = _formula_token_overlap_score(seg_text, formula_text)
                    formula_char = _formula_char_similarity(seg_text, formula_text)
                    score += (0.92 * formula_tok) + (0.65 * formula_char)
                    if eq_number > 0:
                        try:
                            if int(block0.get("number") or 0) == int(eq_number):
                                score += 1.15
                        except Exception:
                            pass
                    ranked_formula.append(
                        {
                            "score": score,
                            "block": block0,
                            "formula_tok": formula_tok,
                            "formula_char": formula_char,
                        }
                    )
                else:
                    ranked_other.append(row)
            ranked_formula.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            if eq_number <= 0 and ranked_formula:
                top_tok = float(ranked_formula[0].get("formula_tok") or 0.0)
                top_char = float(ranked_formula[0].get("formula_char") or 0.0)
                # Keep mathematically-related equations; only drop clearly unrelated noise.
                if top_tok < 0.24 and top_char < 0.42:
                    ranked_formula = []
            # Formula segment: keep equation blocks first. Non-equation rows are only a late fallback.
            ranked = ranked_formula + ([] if ranked_formula else ranked_other)
        else:
            ranked_text = []
            ranked_formula = []
            for row in ranked:
                block0 = row.get("block")
                if isinstance(block0, dict) and _is_formula_block(block0):
                    ranked_formula.append(row)
                else:
                    ranked_text.append(row)
            # Non-formula segments should prefer prose blocks first.
            ranked = ranked_text + ranked_formula

        segment_mapping_source = "fast"
        chosen_ids: list[str] = []
        best_score = float(ranked[0].get("score") or 0.0) if ranked else 0.0
        second_score = float(ranked[1].get("score") or 0.0) if len(ranked) > 1 else 0.0
        score_gap = best_score - second_score

        formula_needs_llm = bool(
            is_formula and (
                (eq_number > 0 and score_gap < 0.18)
                or (eq_number <= 0 and ((best_score < 1.55) or (score_gap < 0.26)))
            )
        )
        if (
            llm_picker is not None
            and llm_calls_used < llm_max_calls
            and len(ranked) >= 2
            and (
                formula_needs_llm
                or ((best_score < 0.98) and (score_gap < 0.14))
            )
        ):
            llm_ids = _pick_blocks_with_llm(
                ds=llm_picker,
                segment_text=probe_text,
                segment_kind=seg_kind,
                is_formula_segment=is_formula,
                candidate_rows=ranked[:6],
                max_pick=(2 if is_formula else 1),
            )
            llm_calls_used += 1
            if llm_ids:
                by_id = {
                    str((row.get("block") or {}).get("block_id") or "").strip(): row
                    for row in ranked
                    if isinstance(row.get("block"), dict)
                }
                boosted: list[dict] = []
                consumed: set[str] = set()
                for bid in llm_ids:
                    row = by_id.get(bid)
                    if not isinstance(row, dict):
                        continue
                    block0 = row.get("block")
                    if not isinstance(block0, dict):
                        continue
                    consumed.add(bid)
                    boosted.append({
                        "score": float(row.get("score") or 0.0) + 1.25,
                        "block": block0,
                    })
                for row in ranked:
                    block0 = row.get("block")
                    if not isinstance(block0, dict):
                        continue
                    bid = str(block0.get("block_id") or "").strip()
                    if bid in consumed:
                        continue
                    boosted.append(row)
                ranked = boosted
                best_score = float(ranked[0].get("score") or 0.0) if ranked else best_score
                segment_mapping_source = "llm_refined"

        primary_support_metrics: dict[str, float | str | bool] = {
            "quote": "",
            "quote_score": 0.0,
            "support_score": 0.0,
            "heading_adjust": 0.0,
            "generic_heading": False,
            "summary_adjust": 0.0,
        }
        if (not is_formula) and ranked:
            rescored: list[dict] = []
            for row in ranked:
                block0 = row.get("block")
                if not isinstance(block0, dict):
                    continue
                metrics = _block_support_metrics(probe_text, block0)
                summary_adjust = _summary_block_adjustment(seg_text, block0)
                capped_base = min(float(row.get("score") or 0.0), 1.32)
                final_score = capped_base
                final_score += 0.96 * float(metrics.get("support_score") or 0.0)
                final_score += float(metrics.get("heading_adjust") or 0.0)
                final_score += summary_adjust
                rescored.append(
                    {
                        "score": final_score,
                        "block": block0,
                        "support_score": float(metrics.get("support_score") or 0.0),
                        "quote_score": float(metrics.get("quote_score") or 0.0),
                        "support_quote": str(metrics.get("quote") or ""),
                        "heading_adjust": float(metrics.get("heading_adjust") or 0.0),
                        "generic_heading": bool(metrics.get("generic_heading")),
                        "summary_adjust": float(summary_adjust),
                    }
                )
            rescored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            ranked = rescored
            best_score = float(ranked[0].get("score") or 0.0) if ranked else 0.0
            second_score = float(ranked[1].get("score") or 0.0) if len(ranked) > 1 else 0.0
            score_gap = best_score - second_score

        min_score = 0.44 if is_formula else 0.63
        dynamic_floor = max(min_score, best_score - (0.14 if is_formula else 0.10))
        keep_limit = 2 if is_formula else 2
        for row in ranked:
            score = float(row.get("score") or 0.0)
            block = row.get("block")
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("block_id") or "").strip()
            if not block_id:
                continue
            block_is_formula = _is_formula_block(block)
            if not is_formula and block_is_formula and score < (dynamic_floor + 0.12):
                continue
            if not is_formula:
                support_score = float(row.get("support_score") or 0.0)
                quote_score = float(row.get("quote_score") or 0.0)
                heading_adjust = float(row.get("heading_adjust") or 0.0)
                generic_heading = bool(row.get("generic_heading"))
                summary_adjust = float(row.get("summary_adjust") or 0.0)
                if summary_tags:
                    if support_score < 0.18 and summary_adjust < 0.60:
                        continue
                    if generic_heading and support_score < 0.42 and summary_adjust < 0.60:
                        continue
                else:
                    if support_score < 0.24:
                        continue
                    if generic_heading and support_score < 0.42:
                        continue
                if quote_score < 0.18 and heading_adjust < -0.10:
                    continue
                if summary_tags and summary_adjust <= -0.48 and support_score < 0.58:
                    continue
            if score < dynamic_floor:
                continue
            if block_id in chosen_ids:
                continue
            chosen_ids.append(block_id)
            _ensure_provenance_block_entry(block_map, block)
            if len(chosen_ids) == 1:
                primary_support_metrics = {
                    "quote": str(row.get("support_quote") or ""),
                    "quote_score": float(row.get("quote_score") or 0.0),
                    "support_score": float(row.get("support_score") or 0.0),
                    "heading_adjust": float(row.get("heading_adjust") or 0.0),
                    "generic_heading": bool(row.get("generic_heading")),
                    "summary_adjust": float(row.get("summary_adjust") or 0.0),
                }
            if len(chosen_ids) >= keep_limit:
                break

        if is_formula and not chosen_ids:
            eq_ranked = match_source_blocks(
                global_formula_blocks or blocks,
                snippet=seg_text,
                prefer_kind="equation",
                target_number=eq_number,
                limit=8,
            )
            eq_ranked2: list[dict] = []
            for row in eq_ranked:
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                if not _is_formula_block(block):
                    continue
                formula_text = str(block.get("text") or "")
                formula_tok = _formula_token_overlap_score(seg_text, formula_text)
                formula_char = _formula_char_similarity(seg_text, formula_text)
                score = float(row.get("score") or 0.0)
                score += (0.88 * formula_tok) + (0.62 * formula_char)
                if eq_number > 0:
                    try:
                        if int(block.get("number") or 0) == int(eq_number):
                            score += 1.1
                    except Exception:
                        pass
                eq_ranked2.append(
                    {
                        "score": score,
                        "block": block,
                        "formula_tok": formula_tok,
                        "formula_char": formula_char,
                    }
                )
            eq_ranked2.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            if eq_number <= 0 and eq_ranked2:
                top_tok = float(eq_ranked2[0].get("formula_tok") or 0.0)
                top_char = float(eq_ranked2[0].get("formula_char") or 0.0)
                if top_tok < 0.24 and top_char < 0.42:
                    eq_ranked2 = []
            eq_best = float(eq_ranked2[0].get("score") or 0.0) if eq_ranked2 else 0.0
            eq_floor = max(0.24, eq_best - 0.26)
            for row in eq_ranked2:
                score = float(row.get("score") or 0.0)
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                block_id = str(block.get("block_id") or "").strip()
                if (not block_id) or (score < eq_floor) or (block_id in chosen_ids):
                    continue
                chosen_ids.append(block_id)
                _ensure_provenance_block_entry(block_map, block)
                if len(chosen_ids) >= 2:
                    break
            if chosen_ids:
                best_score = max(best_score, eq_best)
        if is_formula and (not chosen_ids) and word_count >= 10:
            # Mixed "formula + explanation" segments can still be mapped to prose evidence.
            prose_ranked = match_source_blocks(
                blocks,
                snippet=seg_text,
                prefer_kind="",
                target_number=0,
                limit=6,
            )
            prose_floor = 0.5
            for row in prose_ranked:
                score = float(row.get("score") or 0.0)
                block = row.get("block")
                if not isinstance(block, dict):
                    continue
                if _is_formula_block(block):
                    continue
                block_id = str(block.get("block_id") or "").strip()
                if (not block_id) or (score < prose_floor):
                    continue
                chosen_ids.append(block_id)
                _ensure_provenance_block_entry(block_map, block)
                if len(chosen_ids) >= 2:
                    break
            if chosen_ids:
                best_score = max(best_score, float(prose_ranked[0].get("score") or 0.0))

        evidence_mode = "direct" if chosen_ids else "synthesis"
        snippet_aliases = _segment_snippet_aliases(seg_text)
        for quote_span in quoted_spans:
            key = normalize_match_text(quote_span)
            if (not key) or (key in snippet_aliases):
                continue
            snippet_aliases.append(key[:360])
            if len(snippet_aliases) >= 8:
                break
        primary_block_id = str(chosen_ids[0] or "").strip() if chosen_ids else ""
        support_block_ids = [str(item or "").strip() for item in chosen_ids[1:] if str(item or "").strip()]
        primary_block = block_lookup.get(primary_block_id) if primary_block_id else None
        if primary_block and (not is_formula):
            support_score = float(primary_support_metrics.get("support_score") or 0.0)
            generic_heading = bool(primary_support_metrics.get("generic_heading"))
            summary_adjust = float(primary_support_metrics.get("summary_adjust") or 0.0)
            reject_primary = False
            if summary_tags:
                if support_score < 0.18 and summary_adjust < 0.60:
                    reject_primary = True
                if generic_heading and support_score < 0.42 and summary_adjust < 0.60:
                    reject_primary = True
            elif support_score < 0.24 or (generic_heading and support_score < 0.42):
                reject_primary = True
            if reject_primary:
                chosen_ids = []
                primary_block_id = ""
                support_block_ids = []
                primary_block = None
                evidence_mode = "synthesis"
                primary_support_metrics = {
                    "quote": "",
                    "quote_score": 0.0,
                    "support_score": 0.0,
                    "heading_adjust": 0.0,
                    "generic_heading": False,
                    "summary_adjust": 0.0,
                }
        primary_anchor_id = str((primary_block or {}).get("anchor_id") or "").strip()
        primary_heading_path = str((primary_block or {}).get("heading_path") or "").strip()
        evidence_quote = str(primary_support_metrics.get("quote") or "")
        if (not evidence_quote) and primary_block:
            evidence_quote = _best_evidence_quote(probe_text, primary_block)
        evidence_confidence = round(best_score, 4) if chosen_ids else 0.0
        mapping_quality = round(float(primary_support_metrics.get("support_score") or 0.0), 4) if chosen_ids else 0.0
        claim_meta = _segment_claim_meta(
            segment_text=seg_text,
            raw_markdown=raw_markdown,
            segment_kind=seg_kind,
            evidence_mode=evidence_mode,
            primary_block=primary_block,
            evidence_quote=evidence_quote,
            mapping_quality=mapping_quality,
        )
        segments_out.append(
            {
                "segment_id": f"seg_{idx:03d}",
                "segment_index": idx,
                "kind": seg_kind,
                "segment_type": _segment_type_from_text(seg_text, segment_kind=seg_kind),
                "text": seg_text,
                "raw_markdown": raw_markdown[:4000],
                "snippet_key": str(segment.get("snippet_key") or normalize_match_text(seg_text[:360])),
                "snippet_aliases": snippet_aliases,
                "evidence_mode": evidence_mode,
                "evidence_block_ids": chosen_ids,
                "primary_block_id": primary_block_id,
                "primary_anchor_id": primary_anchor_id,
                "primary_heading_path": primary_heading_path,
                "support_block_ids": support_block_ids,
                "evidence_quote": evidence_quote,
                "evidence_confidence": evidence_confidence,
                "mapping_quality": mapping_quality,
                "mapping_source": segment_mapping_source,
                "claim_type": str(claim_meta.get("claim_type") or "").strip(),
                "must_locate": bool(claim_meta.get("must_locate")),
                "anchor_kind": str(claim_meta.get("anchor_kind") or "").strip(),
                "anchor_text": str(claim_meta.get("anchor_text") or "").strip()[:900],
                "equation_number": int(claim_meta.get("equation_number") or 0),
            }
        )

    if llm_calls_used > 0:
        mapping_mode = "llm_refined"
    segments_with_policy = _apply_provenance_required_coverage_contract(
        segments_out,
        block_lookup=block_lookup,
    )
    segments_hardened, contract_meta = _apply_provenance_strict_identity_contract(segments_with_policy)

    return {
        "version": int(_PAPER_GUIDE_PROVENANCE_SCHEMA_VERSION),
        "source_path": source_path,
        "source_name": str(bound_source_name or Path(source_path).name or "").strip(),
        "md_path": str(md_path),
        "doc_id": str(blocks[0].get("doc_id") or ""),
        "segments": segments_hardened,
        "block_map": block_map,
        "status": "ready",
        "candidate_block_count": len(candidate_blocks),
        "mapping_mode": mapping_mode,
        "llm_rerank_enabled": bool(llm_rerank),
        "llm_rerank_calls": int(llm_calls_used),
        **contract_meta,
    }


def _gen_store_answer_provenance(task: dict, *, answer: str, answer_hits: list[dict]) -> None:
    if not bool(task.get("paper_guide_mode")):
        return
    source_path = str(task.get("paper_guide_bound_source_path") or "").strip()
    if not source_path:
        return
    chat_db = Path(str(task.get("chat_db") or "")).expanduser()
    chat_store = ChatStore(chat_db)
    try:
        amid = int(task.get("assistant_msg_id") or 0)
    except Exception:
        amid = 0
    if amid <= 0:
        return
    provenance = _build_paper_guide_answer_provenance(
        answer=answer,
        answer_hits=list(answer_hits or []),
        bound_source_path=source_path,
        bound_source_name=str(task.get("paper_guide_bound_source_name") or "").strip(),
        db_dir=task.get("db_dir"),
        settings_obj=task.get("settings_obj"),
        llm_rerank=bool(task.get("llm_rerank", True)),
    )
    if not isinstance(provenance, dict):
        return
    try:
        chat_store.merge_message_meta(amid, {"provenance": provenance})
    except Exception:
        pass


def _gen_store_answer_provenance_fast(task: dict, *, answer: str, answer_hits: list[dict]) -> None:
    """Build/store paper-guide provenance with deterministic fast heuristics only."""
    task_copy = dict(task or {})
    task_copy["llm_rerank"] = False
    _gen_store_answer_provenance(task_copy, answer=answer, answer_hits=answer_hits)


def _should_run_provenance_async_refine(task: dict) -> bool:
    if not bool((task or {}).get("paper_guide_mode")):
        return False
    source_path = str((task or {}).get("paper_guide_bound_source_path") or "").strip()
    if not source_path:
        return False
    try:
        enabled = bool(int(str(os.environ.get("KB_PROVENANCE_ASYNC_LLM", "1") or "1")))
    except Exception:
        enabled = True
    if not enabled:
        return False
    if not bool((task or {}).get("llm_rerank", True)):
        return False
    settings_obj = (task or {}).get("settings_obj")
    if settings_obj is None:
        return False
    return bool(getattr(settings_obj, "api_key", None))


def _gen_store_answer_provenance_async(task: dict, *, answer: str, answer_hits: list[dict]) -> None:
    task_copy = dict(task or {})
    # Keep post-answer latency low: run optional LLM rerank only in background.
    task_copy["llm_rerank"] = True

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            _gen_store_answer_provenance(task_copy, answer=answer, answer_hits=answer_hits)
            _perf_log("gen.provenance_async", elapsed=time.perf_counter() - t0, ok=1)
        except Exception as exc:
            _perf_log("gen.provenance_async", elapsed=time.perf_counter() - t0, ok=0, err=str(exc)[:120])

    try:
        threading.Thread(target=_run, daemon=True, name="kb_gen_provenance_async").start()
    except Exception:
        pass

def _gen_worker(session_id: str, task_id: str) -> None:
    task = _gen_get_task(session_id) or {}
    if str(task.get("id") or "") != str(task_id or ""):
        return

    worker_t0 = time.perf_counter()
    _gen_update_task(session_id, task_id, status="running", stage="starting", started_at=time.time())

    try:
        conv_id = str(task.get("conv_id") or "")
        prompt = str(task.get("prompt") or "").strip()
        raw_image_atts = task.get("image_attachments") or []
        chat_db = Path(str(task.get("chat_db") or "")).expanduser()
        db_dir = Path(str(task.get("db_dir") or "")).expanduser()
        top_k = int(task.get("top_k") or 6)
        temperature = float(task.get("temperature") or 0.15)
        max_tokens = int(task.get("max_tokens") or 1200)
        deep_read = bool(task.get("deep_read"))
        answer_contract_v1 = _answer_contract_enabled(task)
        answer_depth_auto = bool(task.get("answer_depth_auto", True))
        answer_mode_hint = str(task.get("answer_mode_hint") or "").strip()
        answer_intent = _detect_answer_intent(prompt, answer_mode_hint=answer_mode_hint)
        answer_depth = _detect_answer_depth(prompt, intent=answer_intent, auto_depth=answer_depth_auto)
        llm_rerank = bool(task.get("llm_rerank", True))
        settings_obj = task.get("settings_obj")
        chat_store = ChatStore(chat_db)
        preferred_sources_raw = task.get("preferred_sources") or []
        paper_guide_mode = bool(task.get("paper_guide_mode"))
        paper_guide_bound_source_path = str(task.get("paper_guide_bound_source_path") or "").strip()
        paper_guide_bound_source_name = str(task.get("paper_guide_bound_source_name") or "").strip()
        paper_guide_bound_source_ready = bool(task.get("paper_guide_bound_source_ready"))

        image_attachments: list[dict] = []
        if isinstance(raw_image_atts, list):
            for it in raw_image_atts:
                if not isinstance(it, dict):
                    continue
                p0 = Path(str(it.get("path") or "")).expanduser()
                if (not str(p0)) or (not p0.exists()) or (not p0.is_file()):
                    continue
                mime0 = str(it.get("mime") or "").strip().lower()
                if not mime0.startswith("image/"):
                    mime0 = _VISION_IMAGE_MIME_BY_SUFFIX.get(p0.suffix.lower(), "")
                if not mime0.startswith("image/"):
                    continue
                image_attachments.append(
                    {
                        "path": str(p0),
                        "name": str(it.get("name") or p0.name),
                        "mime": mime0,
                        "sha1": str(it.get("sha1") or "").strip().lower(),
                    }
                )
        if len(image_attachments) > 4:
            image_attachments = image_attachments[:4]

        if (not conv_id) or ((not prompt) and (not image_attachments)):
            raise RuntimeError("invalid task")
        if _gen_should_cancel(session_id, task_id):
            raise RuntimeError("canceled")
        if paper_guide_mode and paper_guide_bound_source_ready and paper_guide_bound_source_path:
            try:
                kickoff_paper_guide_prefetch(
                    source_path=paper_guide_bound_source_path,
                    source_name=paper_guide_bound_source_name,
                    db_dir=db_dir,
                    library_db_path=getattr(settings_obj, "library_db_path", None),
                )
            except Exception:
                pass

        quick_answer = _quick_answer_for_prompt(prompt) if prompt else None
        image_first_prompt = bool(image_attachments) and _should_prioritize_attached_image(prompt)
        bypass_kb = bool(prompt) and (_should_bypass_kb_retrieval(prompt) or image_first_prompt)
        if quick_answer is not None:
            try:
                umid0 = int(task.get("user_msg_id") or 0)
            except Exception:
                umid0 = 0
            if umid0 > 0:
                try:
                    chat_store.upsert_message_refs(
                        user_msg_id=umid0,
                        conv_id=conv_id,
                        prompt=prompt,
                        prompt_sig=str(task.get("prompt_sig") or ""),
                        hits=[],
                        scores=[],
                        used_query="",
                        used_translation=False,
                    )
                except Exception:
                    pass
            _gen_store_answer(task, quick_answer)
            try:
                _gen_store_answer_provenance_fast(task, answer=quick_answer, answer_hits=[])
            except Exception as exc:
                _perf_log("gen.provenance_inline_fast", ok=0, err=str(exc)[:120])
            _perf_log("gen.quick_answer", total=time.perf_counter() - worker_t0, conv_id=conv_id)
            _gen_update_task(session_id, task_id, status="done", stage="done", answer=quick_answer, partial=quick_answer, char_count=len(quick_answer), finished_at=time.time())
            return

        try:
            cur_user_msg_id = int(task.get("user_msg_id") or 0)
        except Exception:
            cur_user_msg_id = 0
        retrieval_prompt = str(prompt or "").strip()
        preferred_source_hints: list[str] = []
        if isinstance(preferred_sources_raw, list):
            seen_pref: set[str] = set()
            for it in preferred_sources_raw:
                cand = str(it or "").strip()
                if (not cand) or (cand in seen_pref):
                    continue
                seen_pref.add(cand)
                preferred_source_hints.append(cand)
                if len(preferred_source_hints) >= 3:
                    break
        if paper_guide_mode:
            for cand in (paper_guide_bound_source_path, paper_guide_bound_source_name):
                cand_norm = str(cand or "").strip()
                if (not cand_norm) or (cand_norm in preferred_source_hints):
                    continue
                preferred_source_hints.insert(0, cand_norm)
            if len(preferred_source_hints) > 3:
                preferred_source_hints = preferred_source_hints[:3]
        inferred_source_hint = ""
        if retrieval_prompt and _needs_conversational_source_hint(retrieval_prompt):
            inferred_source_hint = _pick_recent_source_hint(
                conv_id=conv_id,
                user_msg_id=cur_user_msg_id,
                chat_store=chat_store,
            )
            if inferred_source_hint:
                retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, inferred_source_hint)
        if retrieval_prompt and _needs_bound_source_hint(retrieval_prompt):
            if preferred_source_hints:
                for h in preferred_source_hints[:2]:
                    retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, h)
            else:
                bound_hints = _pick_recent_bound_source_hints(conv_id=conv_id, chat_store=chat_store, limit=2)
                for h in bound_hints:
                    retrieval_prompt = _augment_prompt_with_source_hint(retrieval_prompt, h)

        t_load0 = time.perf_counter()
        chunks = load_all_chunks(db_dir)
        retriever = BM25Retriever(chunks)
        _perf_log("gen.load_retriever", elapsed=time.perf_counter() - t_load0, chunks=len(chunks))

        hits_raw: list[dict] = []
        scores_raw: list[float] = []
        used_query = ""
        used_translation = False
        hits: list[dict] = []
        grouped_docs: list[dict] = []
        answer_grouped_docs: list[dict] = []
        refs_async_will_run = False
        refs_async_seed_docs: list[dict] = []
        if prompt and (not bypass_kb):
            _gen_update_task(session_id, task_id, stage="retrieve")
            t_ret0 = time.perf_counter()
            hits_raw, scores_raw, used_query, used_translation = _search_hits_with_fallback(
                retrieval_prompt,
                retriever,
                top_k=top_k,
                settings=settings_obj,
            )
            if paper_guide_mode and paper_guide_bound_source_ready and (paper_guide_bound_source_path or paper_guide_bound_source_name):
                scoped_hits = _filter_hits_for_paper_guide(
                    hits_raw,
                    bound_source_path=paper_guide_bound_source_path,
                    bound_source_name=paper_guide_bound_source_name,
                )
                if (not scoped_hits) and paper_guide_bound_source_path:
                    scoped_hits = _paper_guide_fallback_deepread_hits(
                        bound_source_path=paper_guide_bound_source_path,
                        bound_source_name=paper_guide_bound_source_name,
                        query=(retrieval_prompt or prompt or ""),
                        top_k=max(2, min(int(top_k or 4), 4)),
                        db_dir=db_dir,
                    )
                    if scoped_hits:
                        _perf_log(
                            "gen.paper_guide_scope_fallback",
                            docs=len(scoped_hits),
                            source=paper_guide_bound_source_name or paper_guide_bound_source_path,
                        )
                if len(scoped_hits) != len(hits_raw):
                    _perf_log(
                        "gen.paper_guide_scope",
                        before=len(hits_raw),
                        after=len(scoped_hits),
                        source=paper_guide_bound_source_name or paper_guide_bound_source_path,
                    )
                hits_raw = scoped_hits
                scores_raw = [float(h.get("score", 0.0) or 0.0) for h in hits_raw]
            _perf_log(
                "gen.retrieve",
                elapsed=time.perf_counter() - t_ret0,
                hits_raw=len(hits_raw),
                translated=bool(used_translation),
            )
            hits = _group_hits_by_top_heading(hits_raw, top_k=top_k)

            _gen_update_task(session_id, task_id, stage="refs")
            if not getattr(retriever, "is_empty", False):
                try:
                    t_seed0 = time.perf_counter()
                    grouped_docs = _group_hits_by_doc_for_refs(
                        hits_raw,
                        prompt_text=retrieval_prompt,
                        top_k_docs=top_k,
                        deep_query=(used_query or retrieval_prompt or prompt or ""),
                        deep_read=False,  # fast seed first; deep-read is moved to async refs enrichment
                        llm_rerank=False,
                        settings=settings_obj,
                    )
                    _perf_log("gen.refs_seed", elapsed=time.perf_counter() - t_seed0, docs=len(grouped_docs))
                except Exception:
                    grouped_docs = []
            answer_grouped_docs = list(grouped_docs or [])
            answer_hit_limit = max(1, min(int(top_k), 4))
            guide_strict_mode = bool(paper_guide_mode and paper_guide_bound_source_ready)
            answer_doc_cap = max(answer_hit_limit, min(int(top_k), 4 if guide_strict_mode else 3))
            should_sync_deep_seed = bool(hits_raw) and (
                guide_strict_mode
                or _needs_bound_source_hint(prompt or retrieval_prompt or "")
            )
            if should_sync_deep_seed:
                try:
                    t_answer_seed0 = time.perf_counter()
                    rebuilt_for_answer = _group_hits_by_doc_for_refs(
                        hits_raw,
                        prompt_text=retrieval_prompt,
                        top_k_docs=answer_doc_cap,
                        deep_query=(used_query or retrieval_prompt or prompt or ""),
                        deep_read=True,
                        llm_rerank=False,
                        settings=settings_obj,
                    )
                    if rebuilt_for_answer:
                        answer_grouped_docs = rebuilt_for_answer
                    _perf_log(
                        "gen.answer_refs_seed",
                        elapsed=time.perf_counter() - t_answer_seed0,
                        docs=len(answer_grouped_docs),
                    )
                except Exception:
                    pass
            else:
                _perf_log("gen.answer_refs_seed", elapsed=0.0, docs=len(answer_grouped_docs), mode="fast_only")
            # Keep answer path focused on evidence readiness. LLM ref-pack enrichment is
            # deferred to async so it does not block first answer latency.
            _perf_log("gen.answer_refs_enrich", elapsed=0.0, docs=len(answer_grouped_docs), mode="async_only")

            try:
                refs_async_enabled = bool(int(str(os.environ.get("KB_REFS_ASYNC_ENRICH", "1") or "1")))
            except Exception:
                refs_async_enabled = True
            try:
                refs_async_in_paper_guide = bool(int(str(os.environ.get("KB_REFS_ASYNC_ENRICH_IN_PAPER_GUIDE", "0") or "0")))
            except Exception:
                refs_async_in_paper_guide = False
            allow_refs_async = bool(refs_async_enabled and ((not paper_guide_mode) or refs_async_in_paper_guide))

            refs_async_will_run = bool(
                allow_refs_async
                and llm_rerank
                and prompt
                and grouped_docs
                and settings_obj
                and getattr(settings_obj, "api_key", None)
            )
            if refs_async_will_run and grouped_docs:
                try:
                    for d in grouped_docs:
                        if not isinstance(d, dict):
                            continue
                        meta_d = d.get("meta", {}) or {}
                        if not isinstance(meta_d, dict):
                            meta_d = {}
                        meta_d["ref_pack_state"] = "pending"
                        d["meta"] = meta_d
                except Exception:
                    pass
                try:
                    refs_async_seed_docs = copy.deepcopy(grouped_docs)
                except Exception:
                    refs_async_seed_docs = list(grouped_docs)
        else:
            _gen_update_task(
                session_id,
                task_id,
                stage=(
                    "retrieve skipped (image-first prompt)"
                    if image_first_prompt
                    else ("retrieve skipped (general coding request)" if bypass_kb else "retrieve (image-only)")
                ),
            )

        try:
            umid = int(task.get("user_msg_id") or 0)
        except Exception:
            umid = 0
        if umid > 0:
            try:
                chat_store.upsert_message_refs(
                    user_msg_id=umid,
                    conv_id=conv_id,
                    prompt=prompt,
                    prompt_sig=str(task.get("prompt_sig") or ""),
                    hits=list(grouped_docs or []),
                    scores=list(scores_raw or []),
                    used_query=str(used_query or ""),
                    used_translation=bool(used_translation),
                )
            except Exception:
                pass

        def _finalize_task_after_refs_async() -> None:
            snap = _gen_get_task(session_id) or {}
            if str(snap.get("id") or "") != str(task_id or ""):
                return
            if (str(snap.get("status") or "") == "running") and bool(snap.get("answer_ready") or False):
                ans = str(snap.get("answer") or snap.get("partial") or "").strip()
                _gen_update_task(
                    session_id,
                    task_id,
                    status="done",
                    stage="done",
                    answer=ans,
                    partial=ans,
                    char_count=len(ans),
                    finished_at=time.time(),
                )

        if refs_async_will_run and (umid > 0) and refs_async_seed_docs:
            _gen_update_task(session_id, task_id, refs_async_pending=True, refs_async_state="running")

            def _bg_enrich_refs() -> None:
                try:
                    refs_async_top_k_docs = int(str(os.environ.get("KB_REFS_ASYNC_TOP_K", "3") or "3"))
                except Exception:
                    refs_async_top_k_docs = 3
                refs_async_top_k_docs = max(1, min(max(1, int(top_k or 1)), refs_async_top_k_docs))

                try:
                    refs_async_timeout_s = float(str(os.environ.get("KB_REFS_ASYNC_TIMEOUT_S", "12") or "12"))
                except Exception:
                    refs_async_timeout_s = 12.0
                refs_async_timeout_s = max(4.0, min(30.0, refs_async_timeout_s))

                try:
                    refs_async_max_retries = int(str(os.environ.get("KB_REFS_ASYNC_MAX_RETRIES", "0") or "0"))
                except Exception:
                    refs_async_max_retries = 0
                refs_async_max_retries = max(0, min(1, refs_async_max_retries))

                settings_for_refs = settings_obj
                try:
                    settings_for_refs = replace(
                        settings_obj,
                        timeout_s=min(float(getattr(settings_obj, "timeout_s", refs_async_timeout_s) or refs_async_timeout_s), refs_async_timeout_s),
                        max_retries=refs_async_max_retries,
                    )
                except Exception:
                    settings_for_refs = settings_obj

                def _push_partial(partial_docs: list[dict]) -> None:
                    try:
                        cs = ChatStore(chat_db)
                        cs.upsert_message_refs(
                            user_msg_id=umid,
                            conv_id=conv_id,
                            prompt=prompt,
                            prompt_sig=str(task.get("prompt_sig") or ""),
                            hits=list(partial_docs or []),
                            scores=list(scores_raw or []),
                            used_query=str(used_query or ""),
                            used_translation=bool(used_translation),
                        )
                    except Exception:
                        pass

                seed_docs = list(refs_async_seed_docs)[:refs_async_top_k_docs]
                try:
                    if hits_raw:
                        t_rebuild0 = time.perf_counter()
                        rebuilt_docs = _group_hits_by_doc_for_refs(
                            hits_raw,
                            prompt_text=retrieval_prompt,
                            top_k_docs=refs_async_top_k_docs,
                            deep_query=(used_query or retrieval_prompt or prompt or ""),
                            deep_read=True,
                            llm_rerank=False,
                            settings=settings_obj,
                        )
                        if rebuilt_docs:
                            seed_docs = rebuilt_docs
                            for d in seed_docs:
                                if not isinstance(d, dict):
                                    continue
                                meta_d = d.get("meta", {}) or {}
                                if not isinstance(meta_d, dict):
                                    meta_d = {}
                                meta_d["ref_pack_state"] = "pending"
                                d["meta"] = meta_d
                            _push_partial(seed_docs)
                        _perf_log("gen.refs_rebuild", elapsed=time.perf_counter() - t_rebuild0, docs=len(seed_docs))
                except Exception:
                    seed_docs = list(refs_async_seed_docs)

                try:
                    t_pack0 = time.perf_counter()
                    enriched = _enrich_grouped_refs_with_llm_pack(
                        list(seed_docs),
                        question=(prompt or used_query or ""),
                        settings=settings_for_refs,
                        top_k_docs=refs_async_top_k_docs,
                        progress_cb=_push_partial,
                    )
                    _perf_log(
                        "gen.refs_enrich",
                        elapsed=time.perf_counter() - t_pack0,
                        docs=len(enriched),
                        top_k=refs_async_top_k_docs,
                        timeout=refs_async_timeout_s,
                        retries=refs_async_max_retries,
                    )
                except Exception:
                    enriched = []
                snap0 = _gen_get_task(session_id) or {}
                same_task = str(snap0.get("id") or "") == str(task_id or "")
                answer_ready0 = bool(snap0.get("answer_ready") or False)
                # If another task has already replaced this session slot, still allow refs
                # enrichment to be persisted for the original answered message.
                if same_task and _gen_should_cancel(session_id, task_id) and (not answer_ready0):
                    _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="canceled")
                    return
                if enriched:
                    try:
                        cs = ChatStore(chat_db)
                        cs.upsert_message_refs(
                            user_msg_id=umid,
                            conv_id=conv_id,
                            prompt=prompt,
                            prompt_sig=str(task.get("prompt_sig") or ""),
                            hits=list(enriched),
                            scores=list(scores_raw or []),
                            used_query=str(used_query or ""),
                            used_translation=bool(used_translation),
                        )
                    except Exception:
                        pass
                    _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="done", refs_async_docs=int(len(enriched)))
                    try:
                        warm_paths: list[str] = []
                        for d in list(enriched or []):
                            if not isinstance(d, dict):
                                continue
                            meta_d = d.get("meta", {}) or {}
                            src = str(meta_d.get("source_path") or "").strip()
                            if src:
                                warm_paths.append(src)
                        _warm_refs_citation_meta_background(
                            warm_paths,
                            library_db_path=getattr(settings_obj, "library_db_path", None),
                        )
                    except Exception:
                        pass
                else:
                    _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="empty")
                _finalize_task_after_refs_async()

            try:
                threading.Thread(target=_bg_enrich_refs, daemon=True).start()
            except Exception:
                _gen_update_task(session_id, task_id, refs_async_pending=False, refs_async_state="error")
                _finalize_task_after_refs_async()

        _gen_update_task(session_id, task_id, stage="context", used_query=str(used_query or ""), used_translation=bool(used_translation), refs_done=True)

        ctx_parts: list[str] = []
        doc_first_idx: dict[str, int] = {}
        # Keep prompt compact for fast first-token latency.
        answer_hit_limit = max(1, min(int(top_k), 4))
        answer_seed = answer_grouped_docs or grouped_docs or hits
        answer_hits = _build_answer_hits_for_generation(
            grouped_docs=list(answer_seed or []),
            heading_hits=list(hits or []),
            top_n=answer_hit_limit,
        )
        anchor_grounded_answer = _has_anchor_grounded_answer_hits(answer_hits)
        locked_citation_source = _pick_locked_citation_source(list(answer_seed or answer_hits))
        answer_hits = _ensure_locked_source_in_answer_hits(
            answer_hits,
            source_rec=locked_citation_source,
            seed_docs=list(answer_seed or []),
            top_n=answer_hit_limit,
        )
        for i, h in enumerate(answer_hits, start=1):
            meta = h.get("meta", {}) or {}
            src = (meta.get("source_path", "") or "").strip()
            if src and src not in doc_first_idx:
                doc_first_idx[src] = i
            src_name = Path(src).name if src else ""
            focus_heading = (
                str(meta.get("ref_best_heading_path") or "").strip()
                or str(meta.get("top_heading") or "").strip()
                or str(_top_heading(meta.get("heading_path", "")) or "").strip()
            )
            top = "" if _is_probably_bad_heading(focus_heading) else focus_heading
            sid = _cite_source_id(src)
            header = f"[{i}] [SID:{sid}] {src_name or 'unknown'}" + (f" | {top}" if top else "")
            body = ""
            rs = meta.get("ref_show_snippets")
            if isinstance(rs, list):
                parts: list[str] = []
                seen_parts: set[str] = set()
                for s0 in rs[:2]:
                    s = str(s0 or "").strip()
                    if not s:
                        continue
                    k = hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:12]
                    if k in seen_parts:
                        continue
                    seen_parts.add(k)
                    parts.append(s)
                if parts:
                    body = "\n\n".join(parts)
            if not body:
                body = h.get("text", "") or ""
            ctx_parts.append(header + "\n" + body)

        deep_added = 0
        deep_docs = 0
        if deep_read and answer_hits:
            deep_budget_s = 9.0
            deep_begin = time.monotonic()
            q_fine = (used_query or retrieval_prompt or prompt or "").strip()
            items = list(doc_first_idx.items())[: min(3, max(1, len(doc_first_idx)))]
            total = len(items)
            for n, (src, idx0) in enumerate(items, start=1):
                if _gen_should_cancel(session_id, task_id):
                    raise RuntimeError("canceled")
                if (time.monotonic() - deep_begin) >= deep_budget_s:
                    _gen_update_task(session_id, task_id, stage="deep-read skipped (timeout)")
                    break
                _gen_update_task(session_id, task_id, stage=f"deep-read {n}/{total}")
                p = Path(src)
                extras: list[dict] = []
                if q_fine:
                    extras.extend(_deep_read_md_for_context(p, q_fine, max_snippets=2, snippet_chars=1000))
                if not extras:
                    continue
                deep_docs += 1
                seen_snip = set()
                extras2: list[str] = []
                for ex in sorted(extras, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True):
                    t = (ex.get("text") or "").strip()
                    if not t:
                        continue
                    k = hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()[:12]
                    if k in seen_snip:
                        continue
                    seen_snip.add(k)
                    extras2.append(t)
                    if len(extras2) >= 2:
                        break
                if not extras2:
                    continue
                try:
                    base = ctx_parts[idx0 - 1]
                except Exception:
                    continue
                for t in extras2:
                    if t in base:
                        continue
                    base += "\n\n（深读补充定位：来自原文）\n" + t
                    deep_added += 1
                ctx_parts[idx0 - 1] = base
            _perf_log("gen.deep_read", elapsed=time.monotonic() - deep_begin, docs=deep_docs, added=deep_added)

        _gen_update_task(
            session_id,
            task_id,
            deep_read_docs=int(deep_docs),
            deep_read_added=int(deep_added),
            answer_intent=answer_intent,
            answer_depth=answer_depth,
            answer_contract_v1=bool(answer_contract_v1),
            citation_locked_sid=str((locked_citation_source or {}).get("sid") or ""),
            stage="answer",
        )
        ctx = "\n\n---\n\n".join(ctx_parts)

        system = (
            "你的名字是 π-zaya。\n"
            "如果用户问‘你是谁/你叫什么/你是谁开发的’之类的问题，统一回答：我是 P&I Lab 开发的 π-zaya。\n"
            "你是我的个人知识库助手。优先基于我提供的检索片段回答问题。\n"
            "规则：\n"
            "1) 如果检索片段存在：优先基于片段回答；需要引用时，必须用 [[CITE:<sid>:<ref_num>]] 结构化标注。\n"
            "2) 如果检索片段为空：也要给出可用的通用回答，但开头必须写明‘未命中知识库片段’。\n"
            "3) 不要编造不存在的论文、公式、数据或结论。\n"
            "4) 不要输出‘参考定位/Top-K/引用列表’之类的额外段落（我会在页面里单独展示）。\n"
            "5) 数学公式输出格式：短的变量/符号用 $...$（行内）；较长的等式/推导用 $$...$$（行间）。不要用反引号包裹公式。\n"
            "   - 只要在解释“公式(n)/Eq. (n)”或公式中出现 \\frac / \\sum / \\int / \\mathcal / \\mathbf 等复杂结构，就优先改成行间公式。\n"
            "   - 长公式尽量单独成行；不要把整条长公式塞进一句话中间。\n"
            "   - 公式后的 where 句、变量定义句、解释句尽量另起一句或另起一条，不要和长公式挤在同一行。\n"
            "6) 先直接回答用户最核心的问题，再补充必要依据、步骤或限制条件。\n"
            "7) 如果上下文不足以支撑某个细节，要明确说明该部分是基于通用知识的补充，而不是检索片段直接给出的结论。\n"
            "8) 如果用户请求代码、伪代码、步骤或公式推导，要给出可直接使用的结果，不要只讲概念。\n"
            "9) 代码必须放在 fenced code block 中；优先给出最小正确、可运行、命名清晰的实现，并简要说明关键参数、边界条件或复杂度。\n"
            "10) 回答要信息密度高，避免空泛套话、重复表述和模板化总结。\n"
        )
        system += (
            "\nStructured citation protocol:\n"
            "- Context headers contain [SID:<sid>] identifiers.\n"
            "- When citing paper references, MUST use [[CITE:<sid>:<ref_num>]].\n"
            "- Example: [[CITE:s1a2b3c4:24]] or [[CITE:s1a2b3c4:24]][[CITE:s1a2b3c4:25]].\n"
            "- Do NOT output free-form numeric citations like [24] / [2][4].\n"
            "- NEVER output malformed markers like [[CITE:<sid>]] or [CITE:<sid>] (missing ref_num).\n"
        )
        if locked_citation_source:
            locked_sid = str(locked_citation_source.get("sid") or "").strip()
            locked_name = str(locked_citation_source.get("source_name") or "").strip()
            system += (
                "\nCitation source lock:\n"
                f"- This answer is primarily grounded in [SID:{locked_sid}] {locked_name}.\n"
                f"- Include at least one valid [[CITE:{locked_sid}:<ref_num>]] when the answer uses retrieved evidence.\n"
                "- Only switch to another SID when the same reference number cannot be verified in the locked source.\n"
            )
        if image_first_prompt:
            system += (
                "\nImage-first rule:\n"
                "- The user is asking about the attached image itself.\n"
                "- Analyze the attached image first.\n"
                "- Use retrieved paper context only as secondary background, not as a substitute for visual inspection.\n"
            )
        if anchor_grounded_answer:
            system += (
                "\nAnchor-grounded answer rule:\n"
                "- The requested numbered figure/equation/theorem is already matched in the retrieved library context.\n"
                "- Answer from the matched snippets and the same document's retrieved context.\n"
                "- Do NOT say the item is missing, unavailable, inferred only from a public version, or that later sections may possibly add details unless the retrieved context explicitly shows that.\n"
                "- If a detail is not shown in the retrieved context, say it is not shown in the retrieved context; do not speculate that it might appear later.\n"
            )
        if answer_contract_v1:
            system += _build_answer_contract_system_rules(
                intent=answer_intent,
                depth=answer_depth,
                has_hits=bool(answer_hits),
            )
        if paper_guide_mode and paper_guide_bound_source_ready:
            system += _build_paper_guide_grounding_rules(answer_contract_v1=bool(answer_contract_v1))
        prompt_for_user = prompt or "[Image attachment only request]"
        user = (
            f"Question:\\n{prompt_for_user}\\n\\n"
            f"Retrieved context (with deep-read supplements):\\n{ctx if ctx else '(none)'}\\n"
        )
        if anchor_grounded_answer:
            user += (
                "\\nAnchor-grounded retrieval: the requested numbered item is already matched in the library snippets above. "
                "Resolve the answer from those snippets and any explicit follow-up context already retrieved from the same document.\\n"
            )
        if image_attachments:
            user += (
                f"\\nAttached images: {len(image_attachments)}. "
                "These images are part of the current request. Inspect them directly before answering. "
                "Do not claim that no image was uploaded.\\n"
            )
        history = chat_store.get_messages(conv_id)
        try:
            cur_user_msg_id = int(task.get("user_msg_id") or 0)
        except Exception:
            cur_user_msg_id = 0
        try:
            cur_assistant_msg_id = int(task.get("assistant_msg_id") or 0)
        except Exception:
            cur_assistant_msg_id = 0

        hist = _filter_history_for_multimodal_turn(
            history,
            cur_user_msg_id=cur_user_msg_id,
            cur_assistant_msg_id=cur_assistant_msg_id,
            has_current_images=bool(image_attachments),
        )
        hist = hist[-10:]
        user_content: str | list[dict] = user
        if image_attachments:
            mm_parts: list[dict] = [{"type": "text", "text": user}]
            for it in image_attachments:
                try:
                    p_img = Path(str(it.get("path") or "")).expanduser()
                    if (not p_img.exists()) or (not p_img.is_file()):
                        continue
                    data_img = p_img.read_bytes()
                    if (not data_img) or (len(data_img) > 8 * 1024 * 1024):
                        continue
                    mime_img = str(it.get("mime") or "").strip().lower() or _VISION_IMAGE_MIME_BY_SUFFIX.get(p_img.suffix.lower(), "image/png")
                    b64 = base64.b64encode(data_img).decode("ascii")
                    mm_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_img};base64,{b64}"}})
                except Exception:
                    continue
            if len(mm_parts) > 1:
                user_content = mm_parts
        messages = [{"role": "system", "content": system}, *hist, {"role": "user", "content": user_content}]
        ds = DeepSeekChat(settings_obj)
        partial = ""
        streamed = False
        last_store_ts = 0.0
        last_store_len = 0
        t_answer0 = time.perf_counter()
        try:
            for piece in ds.chat_stream(messages=messages, temperature=temperature, max_tokens=max_tokens):
                if _gen_should_cancel(session_id, task_id):
                    raise RuntimeError("canceled")
                partial += piece
                streamed = True
                _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))
                now = time.monotonic()
                # Reduce sqlite write frequency while still keeping crash-recovery checkpoints.
                if (
                    ((now - last_store_ts) >= 0.9 and (len(partial) - last_store_len) >= 48)
                    or (("\n\n" in piece) and (len(partial) - last_store_len) >= 120)
                ):
                    _gen_store_partial(task, partial)
                    last_store_ts = now
                    last_store_len = len(partial)
        except Exception:
            if streamed:
                if _gen_should_cancel(session_id, task_id):
                    raise RuntimeError("canceled")
            else:
                resp = ds.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
                partial = str(resp or "")
                _gen_update_task(session_id, task_id, stage="answer", partial=partial, char_count=len(partial))

        if _gen_should_cancel(session_id, task_id):
            answer = (str(partial or "").strip() + "\n\n（已停止生成）").strip() or "（已停止生成）"
            _gen_store_answer(task, answer)
            _gen_update_task(session_id, task_id, status="canceled", stage="canceled", answer=answer, partial=answer, char_count=len(answer), finished_at=time.time())
            return

        answer = _normalize_math_markdown(_strip_model_ref_section(_sanitize_structured_cite_tokens(partial or ""))).strip() or "（未返回文本）"
        answer = _reconcile_kb_notice(answer, has_hits=bool(answer_hits))
        if answer_contract_v1:
            answer = _apply_answer_contract_v1(
                answer,
                prompt=prompt,
                has_hits=bool(answer_hits),
                intent=answer_intent,
                depth=answer_depth,
            )
        answer = _enhance_kb_miss_fallback(
            answer,
            has_hits=bool(answer_hits),
            intent=answer_intent,
            depth=answer_depth,
            contract_enabled=bool(answer_contract_v1),
        )
        answer = _maybe_append_library_figure_markdown(answer, prompt=prompt, answer_hits=answer_hits)
        answer, citation_validation = _validate_structured_citations(
            answer,
            answer_hits=answer_hits,
            db_dir=db_dir,
            locked_source=locked_citation_source,
        )
        answer_quality = _build_answer_quality_probe(
            answer,
            has_hits=bool(answer_hits),
            contract_enabled=bool(answer_contract_v1),
            intent=answer_intent,
            depth=answer_depth,
        )
        _gen_store_answer(task, answer)
        _gen_record_answer_quality(
            session_id=session_id,
            task_id=task_id,
            conv_id=conv_id,
            answer_quality=answer_quality,
        )
        t_prov0 = time.perf_counter()
        try:
            _gen_store_answer_provenance_fast(task, answer=answer, answer_hits=answer_hits)
            _perf_log("gen.provenance_inline_fast", elapsed=time.perf_counter() - t_prov0, ok=1)
        except Exception as exc:
            _perf_log("gen.provenance_inline_fast", elapsed=time.perf_counter() - t_prov0, ok=0, err=str(exc)[:120])
        if _should_run_provenance_async_refine(task):
            try:
                _gen_store_answer_provenance_async(task, answer=answer, answer_hits=answer_hits)
                _perf_log("gen.provenance_async_schedule", ok=1)
            except Exception as exc:
                _perf_log("gen.provenance_async_schedule", ok=0, err=str(exc)[:120])
        _gen_update_task(
            session_id,
            task_id,
            status="done",
            stage="done",
            answer=answer,
            partial=answer,
            char_count=len(answer),
            answer_ready=True,
            answer_quality=answer_quality,
            citation_validation=citation_validation,
            finished_at=time.time(),
        )
        _perf_log("gen.answer", elapsed=time.perf_counter() - t_answer0, chars=len(answer))
        _perf_log("gen.total", elapsed=time.perf_counter() - worker_t0, conv_id=conv_id)

    except Exception as e:
        if str(e) == "canceled":
            snap = _gen_get_task(session_id) or {}
            partial = str(snap.get("partial") or "").strip()
            answer = (partial + "\n\n（已停止生成）").strip() or "（已停止生成）"
            try:
                _gen_store_answer(task, answer)
            except Exception:
                pass
            _gen_update_task(session_id, task_id, status="canceled", stage="canceled", answer=answer, partial=answer, char_count=len(answer), finished_at=time.time())
            return

        err = S["llm_fail"].format(err=str(e))
        try:
            _gen_store_answer(task, err)
        except Exception:
            pass
        _gen_update_task(session_id, task_id, status="error", stage="error", error=str(e), answer=err, partial=err, char_count=len(err), finished_at=time.time())

def _gen_start_task(task: dict) -> bool:
    sid = str(task.get("session_id") or "").strip()
    tid = str(task.get("id") or "").strip()
    if (not sid) or (not tid):
        return False
    with RUNTIME.GEN_LOCK:
        cur = RUNTIME.GEN_TASKS.get(sid)
        if (
            isinstance(cur, dict)
            and str(cur.get("status") or "") == "running"
            and (not bool(cur.get("answer_ready") or False))
        ):
            return False
        item = dict(task)
        item.setdefault("status", "running")
        item.setdefault("stage", "starting")
        item.setdefault("partial", "")
        item.setdefault("char_count", 0)
        item.setdefault("cancel", False)
        item.setdefault("created_at", time.time())
        item.setdefault("updated_at", time.time())
        RUNTIME.GEN_TASKS[sid] = item
    try:
        threading.Thread(target=_gen_worker, args=(sid, tid), daemon=True).start()
    except Exception:
        with RUNTIME.GEN_LOCK:
            cur = RUNTIME.GEN_TASKS.get(sid)
            if isinstance(cur, dict) and str(cur.get("id") or "") == tid:
                cur2 = dict(cur)
                cur2["status"] = "error"
                cur2["stage"] = "error"
                cur2["answer"] = "线程启动失败"
                cur2["finished_at"] = time.time()
                RUNTIME.GEN_TASKS[sid] = cur2
        return False
    return True

def _bg_enqueue(task: dict) -> None:
    if "_tid" not in task:
        task = dict(task)
        task["_tid"] = uuid.uuid4().hex
    bg_enqueue(_BG_STATE, _BG_LOCK, task)
    _bg_ensure_started()

def _bg_remove_queued_tasks_for_pdf(pdf_path: Path) -> int:
    """
    Remove queued (not running) conversion tasks for a given PDF.
    Returns removed count.
    """
    return bg_remove_queued_tasks_for_pdf(_BG_STATE, _BG_LOCK, pdf_path)

def _bg_cancel_all() -> None:
    bg_cancel_all(_BG_STATE, _BG_LOCK, "正在停止当前转换…")

def _bg_snapshot() -> dict:
    return bg_snapshot(_BG_STATE, _BG_LOCK)

def _bg_worker_loop() -> None:
    while True:
        task = bg_begin_next_task_or_idle(_BG_STATE, _BG_LOCK)

        if task is None:
            time.sleep(0.35)
            continue

        pdf = Path(task["pdf"])
        out_root = Path(task["out_root"])
        db_dir = Path(task.get("db_dir") or "").expanduser() if task.get("db_dir") else None
        no_llm = bool(task.get("no_llm", False))
        # Equation image fallback should be a last resort.
        # - For full_llm (quality-first), prefer editable/searchable LaTeX over screenshots.
        # - In no-LLM degraded runs, `kb/pdf_tools.run_pdf_to_md` will force-enable it to preserve fidelity.
        eq_image_fallback = bool(task.get("eq_image_fallback", False))
        replace = bool(task.get("replace", False))
        speed_mode = str(task.get("speed_mode", "balanced"))
        if speed_mode == "ultra_fast":
            # Keep VL/LLM path in ultra_fast; converter itself handles speed/quality tradeoff.
            # Forcing no_llm here causes a dramatic quality drop that does not match UI semantics.
            eq_image_fallback = False
        task_id = str(task.get("_tid") or "")

        try:
            md_folder = out_root / pdf.stem
            if replace and md_folder.exists():
                # Safety: only delete inside out_root
                try:
                    md_root = out_root.resolve()
                    target = md_folder.resolve()
                    if str(target).lower().startswith(str(md_root).lower()):
                        import shutil

                        shutil.rmtree(md_folder, ignore_errors=True)
                except Exception:
                    pass

            def _on_progress(page_done: int, page_total: int, msg: str = "") -> None:
                try:
                    bg_update_page_progress(_BG_STATE, _BG_LOCK, page_done, page_total, msg, task_id=task_id)
                except Exception:
                    pass

            def _should_cancel() -> bool:
                return bg_should_cancel(_BG_STATE, _BG_LOCK)

            ok, out_folder = run_pdf_to_md(
                pdf_path=pdf,
                out_root=out_root,
                no_llm=no_llm,
                keep_debug=False,
                eq_image_fallback=eq_image_fallback,
                progress_cb=_on_progress,
                cancel_cb=_should_cancel,
                speed_mode=speed_mode,
            )
            if ok:
                msg = f"OK: {out_folder}"
            else:
                txt = str(out_folder or "").strip().lower()
                msg = "CANCELLED" if txt == "cancelled" else f"FAIL: {out_folder}"

            # Auto-ingest can add noticeable latency in the conversion UI.
            # Skip it in ultra_fast mode to keep end-to-end time near the 5s target.
            do_auto_ingest = ok and bool(db_dir) and (speed_mode != "ultra_fast")
            if do_auto_ingest and db_dir:
                try:
                    ingest_py = Path(__file__).resolve().parent / "ingest.py"
                    _, md_main, md_exists = _resolve_md_output_paths(out_root, pdf)
                    if ingest_py.exists() and md_exists:
                        subprocess.run(
                            [os.sys.executable, str(ingest_py), "--src", str(md_main), "--db", str(db_dir), "--incremental"],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        msg = f"OK+INGEST: {out_folder}"
                except Exception:
                    # Not fatal; conversion still succeeded.
                    pass
        except Exception as e:
            msg = f"FAIL: {e}"

        bg_finish_task(_BG_STATE, _BG_LOCK, msg, task_id=task_id)

def _bg_ensure_started() -> None:
    worker_ver = "2026-02-12.bg.v4"
    t = getattr(RUNTIME, "BG_THREAD", None)
    running_ver = str(getattr(RUNTIME, "BG_WORKER_VERSION", "") or "")
    if t is not None and t.is_alive():
        # Never interrupt an active conversion thread on app rerun/hot-reload.
        # Otherwise users observe "sudden stop" in the middle of conversion.
        if running_ver != worker_ver:
            try:
                RUNTIME.BG_WORKER_VERSION = worker_ver
            except Exception:
                pass
        return
    t = threading.Thread(target=_bg_worker_loop, daemon=True)
    RUNTIME.BG_THREAD = t
    RUNTIME.BG_WORKER_VERSION = worker_ver
    t.start()

def _build_bg_task(
    *,
    pdf_path: Path,
    out_root: Path,
    db_dir: Path,
    no_llm: bool,
    replace: bool = False,
    speed_mode: str = "balanced",
) -> dict:
    pdf = Path(pdf_path)
    mode = str(speed_mode)
    return {
        "_tid": uuid.uuid4().hex,
        "pdf": str(pdf),
        "out_root": str(out_root),
        "db_dir": str(db_dir),
        # no_llm is controlled only by user-selected mode "no_llm".
        # ultra_fast should remain VL-based (lower quality, but still LLM).
        "no_llm": bool(no_llm),
        # Default OFF across all normal modes; enable only explicitly.
        # In no-LLM runs we still force-enable it inside `run_pdf_to_md` for fidelity.
        "eq_image_fallback": False,
        "replace": bool(replace),
        "speed_mode": mode,
        "name": pdf.name,
    }

def _group_hits_by_top_heading(hits: list[dict], top_k: int) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    grouped: list[dict] = []
    for h in hits:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        top = _top_heading(meta.get("heading_path", ""))
        key = (src, top)
        if key in seen:
            continue
        seen.add(key)
        gh = dict(h)
        gh_meta = dict(meta)
        gh_meta["top_heading"] = top
        gh["meta"] = gh_meta
        grouped.append(gh)
        if len(grouped) >= max(1, int(top_k)):
            break
    return grouped


def _hit_source_path(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("meta", {}) or {}
    return str(meta.get("source_path") or "").strip()


def _build_answer_hits_for_generation(
    *,
    grouped_docs: list[dict],
    heading_hits: list[dict],
    top_n: int,
) -> list[dict]:
    try:
        limit = max(1, int(top_n))
    except Exception:
        limit = 1
    out: list[dict] = []
    seen_src: set[str] = set()

    def _push(pool: list[dict]) -> None:
        nonlocal out
        for h in pool or []:
            if not isinstance(h, dict):
                continue
            src = _hit_source_path(h)
            if src and (src in seen_src):
                continue
            out.append(h)
            if src:
                seen_src.add(src)
            if len(out) >= limit:
                return

    _push(grouped_docs)
    if len(out) < limit:
        _push(heading_hits)
    if out:
        return out[:limit]
    return list((grouped_docs or heading_hits or [])[:limit])


def _ensure_locked_source_in_answer_hits(
    answer_hits: list[dict],
    *,
    source_rec: dict | None,
    seed_docs: list[dict],
    top_n: int,
) -> list[dict]:
    try:
        limit = max(1, int(top_n))
    except Exception:
        limit = 1
    out = list(answer_hits or [])[:limit]
    if not source_rec:
        return out
    locked_src = str((source_rec or {}).get("source_path") or "").strip()
    if not locked_src:
        return out
    if any(_hit_source_path(h) == locked_src for h in out):
        return out
    locked_hit = None
    for cand in seed_docs or []:
        if _hit_source_path(cand) == locked_src:
            locked_hit = cand
            break
    if not isinstance(locked_hit, dict):
        return out
    out2 = [locked_hit]
    for h in out:
        if _hit_source_path(h) == locked_src:
            continue
        out2.append(h)
        if len(out2) >= limit:
            break
    return out2[:limit]


def _should_prefer_grouped_docs_for_answer(grouped_docs: list[dict]) -> bool:
    for doc in grouped_docs or []:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta", {}) or {}
        try:
            doc_score = float(meta.get("explicit_doc_match_score") or 0.0)
        except Exception:
            doc_score = 0.0
        if doc_score >= 6.0:
            return True
        if str(meta.get("anchor_target_kind") or "").strip():
            try:
                anchor_score = float(meta.get("anchor_match_score") or 0.0)
            except Exception:
                anchor_score = 0.0
            if anchor_score > 0.0:
                return True
    return False


def _has_anchor_grounded_answer_hits(answer_hits: list[dict]) -> bool:
    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        if not str(meta.get("anchor_target_kind") or "").strip():
            continue
        try:
            anchor_score = float(meta.get("anchor_match_score") or 0.0)
        except Exception:
            anchor_score = 0.0
        if anchor_score > 0.0:
            return True
    return False


def _aggregate_answer_sources(answer_hits: list[dict]) -> list[dict]:
    agg_by_src: dict[str, dict] = {}
    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if not src:
            continue
        rec = agg_by_src.get(src)
        if not isinstance(rec, dict):
            rec = {
                "source_path": src,
                "sid": _cite_source_id(src),
                "source_name": _source_name_from_md_path(src),
                "hits": 0,
                "explicit_doc_score": 0.0,
                "anchor_score": 0.0,
                "source_sha1": "",
            }
            agg_by_src[src] = rec
        rec["hits"] = int(rec.get("hits") or 0) + 1
        sha1 = str(meta.get("source_sha1") or "").strip().lower()
        if sha1 and (not str(rec.get("source_sha1") or "").strip()):
            rec["source_sha1"] = sha1
        try:
            rec["explicit_doc_score"] = max(
                float(rec.get("explicit_doc_score") or 0.0),
                float(meta.get("explicit_doc_match_score") or 0.0),
            )
        except Exception:
            pass
        try:
            rec["anchor_score"] = max(
                float(rec.get("anchor_score") or 0.0),
                float(meta.get("anchor_match_score") or 0.0),
            )
        except Exception:
            pass
    out = list(agg_by_src.values())
    out.sort(
        key=lambda item: (
            float(item.get("anchor_score") or 0.0),
            float(item.get("explicit_doc_score") or 0.0),
            int(item.get("hits") or 0),
            str(item.get("source_name") or ""),
        ),
        reverse=True,
    )
    return out


def _pick_locked_citation_source(answer_hits: list[dict]) -> dict | None:
    ranked = _aggregate_answer_sources(answer_hits)
    if not ranked:
        return None
    if len(ranked) == 1:
        rec = dict(ranked[0])
        rec["lock_reason"] = "single_source"
        return rec

    top = ranked[0]
    second = ranked[1]
    top_anchor = float(top.get("anchor_score") or 0.0)
    sec_anchor = float(second.get("anchor_score") or 0.0)
    if top_anchor > 0.0 and top_anchor >= max(1.0, sec_anchor + 0.25):
        rec = dict(top)
        rec["lock_reason"] = "anchor_dominant"
        return rec

    top_doc = float(top.get("explicit_doc_score") or 0.0)
    sec_doc = float(second.get("explicit_doc_score") or 0.0)
    if top_doc >= 6.0 and top_doc >= max(6.0, sec_doc + 1.5):
        rec = dict(top)
        rec["lock_reason"] = "explicit_doc_dominant"
        return rec

    top_hits = int(top.get("hits") or 0)
    sec_hits = int(second.get("hits") or 0)
    if top_hits >= max(2, sec_hits * 2) and top_doc >= max(4.0, sec_doc):
        rec = dict(top)
        rec["lock_reason"] = "hit_dominant"
        return rec
    return None


def _validate_structured_citations(
    answer: str,
    *,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_source: dict | None = None,
) -> tuple[str, dict]:
    text = str(answer or "")
    if ("[[CITE:" not in text) and ("[CITE:" not in text):
        return text, {
            "raw_count": 0,
            "kept": 0,
            "rewritten": 0,
            "dropped": 0,
            "locked_sid": str((locked_source or {}).get("sid") or ""),
        }

    cleaned = _sanitize_structured_cite_tokens(text)
    raw_tokens = list(_CITE_CANON_RE.finditer(cleaned))
    if not raw_tokens:
        return cleaned, {
            "raw_count": 0,
            "kept": 0,
            "rewritten": 0,
            "dropped": 0,
            "locked_sid": str((locked_source or {}).get("sid") or ""),
        }

    sid_to_source: dict[str, str] = {}
    sha_by_source: dict[str, str] = {}
    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if not src:
            continue
        sid = _cite_source_id(src).lower()
        sid_to_source[sid] = src
        sha1 = str(meta.get("source_sha1") or "").strip().lower()
        if sha1 and (src not in sha_by_source):
            sha_by_source[src] = sha1

    try:
        index_data = load_reference_index(Path(db_dir).expanduser()) if db_dir else {}
    except Exception:
        index_data = {}

    locked_sid = str((locked_source or {}).get("sid") or "").strip().lower()
    locked_source_path = str((locked_source or {}).get("source_path") or "").strip()
    if locked_sid and locked_source_path and (locked_sid not in sid_to_source):
        sid_to_source[locked_sid] = locked_source_path
        sha_locked = str((locked_source or {}).get("source_sha1") or "").strip().lower()
        if sha_locked and (locked_source_path not in sha_by_source):
            sha_by_source[locked_source_path] = sha_locked

    def _resolves(sid: str, ref_num: int) -> bool:
        sp = sid_to_source.get(str(sid or "").strip().lower())
        if (not sp) or (int(ref_num) <= 0):
            return False
        try:
            got = resolve_reference_entry(
                index_data,
                sp,
                int(ref_num),
                source_sha1=sha_by_source.get(sp, ""),
            )
        except Exception:
            got = None
        return bool(isinstance(got, dict) and isinstance(got.get("ref"), dict))

    stats = {
        "raw_count": int(len(raw_tokens)),
        "kept": 0,
        "rewritten": 0,
        "dropped": 0,
        "locked_sid": locked_sid,
    }

    def _repl(m: re.Match[str]) -> str:
        sid = str(m.group(1) or "").strip().lower()
        try:
            n = int(m.group(2) or 0)
        except Exception:
            n = 0
        if n <= 0:
            stats["dropped"] = int(stats["dropped"]) + 1
            return ""

        if locked_sid:
            if _resolves(locked_sid, n):
                if sid == locked_sid:
                    stats["kept"] = int(stats["kept"]) + 1
                    return f"[[CITE:{locked_sid}:{n}]]"
                stats["rewritten"] = int(stats["rewritten"]) + 1
                return f"[[CITE:{locked_sid}:{n}]]"
            if sid and _resolves(sid, n):
                # Locked source could not validate this ref number; keep the
                # original only when it resolves cleanly in the cited source.
                stats["kept"] = int(stats["kept"]) + 1
                return f"[[CITE:{sid}:{n}]]"
            stats["dropped"] = int(stats["dropped"]) + 1
            return ""

        if sid and _resolves(sid, n):
            stats["kept"] = int(stats["kept"]) + 1
            return f"[[CITE:{sid}:{n}]]"
        stats["dropped"] = int(stats["dropped"]) + 1
        return ""

    out = _CITE_CANON_RE.sub(_repl, cleaned)
    return out, stats


def _filter_history_for_multimodal_turn(
    history: list[dict],
    *,
    cur_user_msg_id: int,
    cur_assistant_msg_id: int,
    has_current_images: bool,
) -> list[dict]:
    hist: list[dict] = []
    suppress_followup_assistant = False

    for m in history or []:
        if m.get("role") not in ("user", "assistant"):
            continue
        try:
            mid = int(m.get("id") or 0)
        except Exception:
            mid = 0
        if mid and mid in {cur_user_msg_id, cur_assistant_msg_id}:
            continue
        if _is_live_assistant_text(str(m.get("content") or "")):
            continue

        attachments = m.get("attachments") if isinstance(m.get("attachments"), list) else []
        has_message_images = any(
            isinstance(it, dict) and str(it.get("mime") or "").strip().lower().startswith("image/")
            for it in attachments
        )

        if has_current_images and str(m.get("role") or "") == "user":
            suppress_followup_assistant = bool(has_message_images)
            if has_message_images:
                continue
        elif has_current_images and str(m.get("role") or "") == "assistant" and suppress_followup_assistant:
            continue
        else:
            suppress_followup_assistant = False

        hist.append(m)

    return hist

def _strip_model_ref_section(answer: str) -> str:
    if not answer:
        return answer
    # Common markers we used in prompts previously.
    for marker in ("可参考定位", "参考定位"):
        idx = answer.find(marker)
        if idx > 0:
            return answer[:idx].rstrip()
    return answer


def _sanitize_structured_cite_tokens(answer: str) -> str:
    s = str(answer or "")
    if not s:
        return s
    # Normalize accidental single-bracket form to canonical form expected by renderer.
    s = _CITE_SINGLE_BRACKET_RE.sub(lambda m: f"[[CITE:{m.group(1)}:{m.group(2)}]]", s)
    # Drop malformed sid-only tokens; they have no ref number and cannot be resolved.
    s = _CITE_SID_ONLY_RE.sub("", s)
    # Strip internal source id markers that may leak from retrieval context headers.
    s = _SID_HEADER_LINE_RE.sub("", s)
    s = _SID_INLINE_RE.sub("", s)
    return s


def _extract_figure_number(text: str) -> int:
    raw = str(text or "").strip()
    if not raw:
        return 0
    for pat in _FIG_NUMBER_PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            n = 0
        if n > 0:
            return n
    return 0


def _requested_figure_number(prompt: str, answer_hits: list[dict]) -> int:
    n = _extract_figure_number(prompt)
    if n > 0:
        return n
    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        kind = str(meta.get("anchor_target_kind") or "").strip().lower()
        if kind != "figure":
            continue
        try:
            n2 = int(meta.get("anchor_target_number") or 0)
        except Exception:
            n2 = 0
        if n2 > 0:
            return n2
    return 0


def _source_name_from_md_path(source_path: str) -> str:
    src = Path(str(source_path or "").strip())
    name = src.name or src.stem or "unknown-source"
    if name.lower().endswith(".en.md"):
        return re.sub(r"\.en\.md$", ".pdf", name, flags=re.IGNORECASE)
    if name.lower().endswith(".md"):
        return re.sub(r"\.md$", ".pdf", name, flags=re.IGNORECASE)
    return name


def _resolve_doc_image_path(md_path: Path, raw_ref: str) -> Path | None:
    ref = str(raw_ref or "").strip().strip("'").strip('"')
    if not ref:
        return None
    low = ref.lower()
    if low.startswith(("http://", "https://", "data:")):
        return None
    if "?" in ref:
        ref = ref.split("?", 1)[0]
    if "#" in ref:
        ref = ref.split("#", 1)[0]
    ref = ref.replace("\\", "/")
    cand = Path(ref)
    if not cand.is_absolute():
        cand = (md_path.parent / cand).resolve()
    else:
        cand = cand.resolve()
    if (not cand.exists()) or (not cand.is_file()):
        return None
    if cand.suffix.lower() not in _DOC_FIGURE_IMAGE_EXTS:
        return None
    return cand


def _collect_doc_figure_assets(md_path: Path) -> list[dict]:
    p = Path(md_path).expanduser()
    if (not p.exists()) or (not p.is_file()):
        return []
    try:
        mtime = float(p.stat().st_mtime)
    except Exception:
        mtime = 0.0
    key = str(p.resolve())
    with _DOC_FIGURE_CACHE_LOCK:
        cached = _DOC_FIGURE_CACHE.get(key)
        if isinstance(cached, tuple) and len(cached) == 2:
            old_mtime, old_items = cached
            if float(old_mtime) == mtime:
                return [dict(x) for x in (old_items or [])]

    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = text.splitlines()
    out: list[dict] = []
    seen_paths: set[str] = set()

    for i, ln in enumerate(lines):
        for m in _MD_IMAGE_LINK_RE.finditer(ln):
            alt = str(m.group(1) or "").strip()
            raw_img = str(m.group(2) or "").strip()
            img_path = _resolve_doc_image_path(p, raw_img)
            if img_path is None:
                continue
            sp = str(img_path)
            if sp in seen_paths:
                continue
            seen_paths.add(sp)
            next_line = str(lines[i + 1] or "").strip() if (i + 1) < len(lines) else ""
            prev_line = str(lines[i - 1] or "").strip() if i > 0 else ""
            caption = next_line if _extract_figure_number(next_line) > 0 else ""
            if (not caption) and (_extract_figure_number(prev_line) > 0):
                caption = prev_line
            number = _extract_figure_number(caption) or _extract_figure_number(alt) or _extract_figure_number(raw_img)
            label = caption or alt or img_path.name
            out.append(
                {
                    "path": sp,
                    "number": int(number or 0),
                    "label": str(label or "").strip(),
                }
            )

    with _DOC_FIGURE_CACHE_LOCK:
        _DOC_FIGURE_CACHE[key] = (mtime, [dict(x) for x in out])
        if len(_DOC_FIGURE_CACHE) > 512:
            try:
                for k in list(_DOC_FIGURE_CACHE.keys())[:128]:
                    _DOC_FIGURE_CACHE.pop(k, None)
            except Exception:
                pass
    return out


def _build_doc_figure_card(*, source_path: str, figure_num: int) -> dict | None:
    src = Path(str(source_path or "").strip())
    if (not src.exists()) or (not src.is_file()):
        return None
    items = _collect_doc_figure_assets(src)
    if not items:
        return None
    selected = next((it for it in items if int(it.get("number") or 0) == int(figure_num)), None)
    if selected is None:
        return None
    img_path = str(selected.get("path") or "").strip()
    if not img_path:
        return None
    src_name = _source_name_from_md_path(str(source_path or ""))
    label = str(selected.get("label") or "").strip()
    if len(label) > 140:
        label = label[:140].rstrip() + "..."
    return {
        "source_name": src_name,
        "figure_num": int(figure_num),
        "label": label,
        "url": f"/api/references/asset?path={quote(img_path, safe='')}",
    }


def _score_figure_card_source_binding(*, prompt: str, meta: dict, figure_num: int, source_path: str) -> float:
    q = str(prompt or "").strip().lower()
    m = meta if isinstance(meta, dict) else {}
    src = str(source_path or "").strip()
    src_name = _source_name_from_md_path(src).lower()
    src_stem = Path(src_name).stem.lower()

    score = 0.0
    try:
        score += 2.0 * float(m.get("explicit_doc_match_score") or 0.0)
    except Exception:
        pass

    kind = str(m.get("anchor_target_kind") or "").strip().lower()
    try:
        n0 = int(m.get("anchor_target_number") or 0)
    except Exception:
        n0 = 0
    try:
        a0 = float(m.get("anchor_match_score") or 0.0)
    except Exception:
        a0 = 0.0
    if kind == "figure" and n0 > 0:
        if int(figure_num) == int(n0):
            score += 40.0 + max(0.0, a0)
        else:
            score -= 16.0
    elif kind and kind != "figure":
        score -= 10.0

    if q:
        if src_name and src_name in q:
            score += 36.0
        if src_stem and src_stem in q:
            score += 26.0
        if src_stem:
            tokens = [t for t in re.split(r"[^a-z0-9]+", src_stem) if len(t) >= 4]
            if tokens:
                overlap = sum(1 for t in set(tokens) if t in q)
                score += min(18.0, 4.0 * float(overlap))

    return float(score)


def _maybe_append_library_figure_markdown(answer: str, *, prompt: str, answer_hits: list[dict]) -> str:
    base = str(answer or "").rstrip()
    if (not base) or (not answer_hits):
        return base
    # Avoid duplicate injection on retries/rerenders.
    if "/api/references/asset?path=" in base:
        return base
    target_num = _requested_figure_number(prompt, answer_hits)
    if target_num <= 0:
        return base

    cards_scored: list[tuple[float, dict]] = []
    seen_src: set[str] = set()
    for hit in answer_hits:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if (not src) or (src in seen_src):
            continue
        seen_src.add(src)
        card = _build_doc_figure_card(source_path=src, figure_num=target_num)
        if card is None:
            continue
        score = _score_figure_card_source_binding(
            prompt=prompt,
            meta=meta,
            figure_num=target_num,
            source_path=src,
        )
        cards_scored.append((score, card))

    if not cards_scored:
        return base

    cards_scored.sort(key=lambda x: float(x[0]), reverse=True)
    cards = [cards_scored[0][1]]

    lines: list[str] = ["### 文献图示（库内截图）"]
    for card in cards:
        src_name = str(card.get("source_name") or "unknown-source")
        fig_num = int(card.get("figure_num") or target_num)
        url = str(card.get("url") or "").strip()
        label = str(card.get("label") or "").strip()
        alt = f"{src_name} Fig. {fig_num}"
        lines.append(f"![{alt}]({url})")
        if label:
            lines.append(f"*来源：{src_name}，Fig. {fig_num}。{label}*")
        else:
            lines.append(f"*来源：{src_name}，Fig. {fig_num}（库内截图）*")

    return f"{base}\n\n" + "\n\n".join(lines)
