from __future__ import annotations

import hashlib
import re
from pathlib import Path

from kb.file_ops import _resolve_md_output_paths
from kb.llm import DeepSeekChat
from kb.source_blocks import (
    extract_equation_number,
    has_equation_signal,
    load_source_blocks,
    match_source_blocks,
    normalize_inline_markdown,
    normalize_match_text,
    split_answer_segments,
)

_FIG_NUMBER_PATTERNS = (
    re.compile(r"\bfig(?:ure)?\.?\s*(\d{1,3})(?:\s*[a-z])?\b", flags=re.IGNORECASE),
    re.compile(r"\bfigure\s*#?\s*(\d{1,3})(?:\s*[a-z])?\b", flags=re.IGNORECASE),
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
    r"\bmetric\b|\bsota\b|实验|对比|基线|位姿|真值|训练|数据集|指标)",
    re.IGNORECASE,
)
_CLAIM_METHOD_HINT_RE = re.compile(
    r"(\bmethod\b|\bapproach\b|\bpipeline\b|\bframework\b|\barchitecture\b|\bmodule\b|\bnetwork\b|"
    r"\binput\b|\boutput\b|\brender(?:ing)?\b|\breconstruct(?:ion)?\b|\bprior\b|方法|流程|框架|输入|输出|重建|渲染)",
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
    r"(?:位姿|相机|训练|输入|输出|流程|公式|变量|表|图|硬件|压缩比|真值|基线|对比|实验)"
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


def _cite_source_id(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return "s0000000"
    return "s" + hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]


def _source_name_from_md_path(source_path: str) -> str:
    src = Path(str(source_path or "").strip())
    name = src.name or src.stem or "unknown-source"
    if name.lower().endswith(".en.md"):
        return re.sub(r"\.en\.md$", ".pdf", name, flags=re.IGNORECASE)
    if name.lower().endswith(".md"):
        return re.sub(r"\.md$", ".pdf", name, flags=re.IGNORECASE)
    return name
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
    r"(?:定义为|表示|记为|其中.{0,16}(?:表示|代表|对应))"
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
_EQUATION_EXPLANATION_PREFIX_RE = re.compile(
    r"^\s*(?:where|wherein|here|with|in which|式中|其中|其中,|其中:|其中各项|其中变量)\b",
    re.IGNORECASE,
)
_EQUATION_EXPLANATION_HINT_RE = re.compile(
    r"(?:\bwhere\b|\bdenotes?\b|\brepresents?\b|\bstands?\s+for\b|\bis\s+the\b|\bare\s+the\b|"
    r"\bcorresponds?\s+to\b|表示|记为|定义为|代表|对应|指代)",
    re.IGNORECASE,
)
_QUOTE_ELLIPSIS_RE = re.compile(r"(?:\[\s*(?:\.{3,}|…)\s*\]|\.{3,}|…)")


def _trim_paper_guide_prompt_field(text: str, *, max_chars: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    s = s.replace("|", "/")
    try:
        limit = max(24, int(max_chars))
    except Exception:
        limit = 160
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)].rstrip() + "..."


def _trim_paper_guide_prompt_snippet(text: str, *, max_chars: int = 420) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [re.sub(r"[ \t]+", " ", str(line or "").strip()) for line in raw.splitlines()]
    kept: list[str] = []
    last_blank = False
    for line in lines:
        if not line:
            if kept and (not last_blank):
                kept.append("")
            last_blank = True
            continue
        kept.append(line)
        last_blank = False
    s = "\n".join(kept).strip()
    if not s:
        return ""
    s = re.sub(r"\n{3,}", "\n\n", s)
    try:
        limit = max(80, int(max_chars))
    except Exception:
        limit = 420
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)].rstrip() + "..."


def _extract_paper_guide_abstract_excerpt(text: str, *, max_chars: int = 560) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [str(line or "").rstrip() for line in raw.splitlines()]
    start_idx = -1
    for idx, line in enumerate(lines):
        s = str(line or "").strip()
        if not s:
            continue
        if re.match(r"^\s*#\s*abstract\b", s, flags=re.IGNORECASE):
            start_idx = idx + 1
            break
        if normalize_match_text(s) in {"abstract", "摘要"}:
            start_idx = idx + 1
            break
    body_lines: list[str] = []
    if start_idx >= 0:
        for line in lines[start_idx:]:
            s = str(line or "").rstrip()
            if re.match(r"^\s*#{1,6}\s+", s) and body_lines:
                break
            body_lines.append(s)
    if not body_lines:
        body_lines = lines[:]
        if body_lines and body_lines[0].lstrip().startswith("#"):
            body_lines = body_lines[1:]
        while len(body_lines) > 1:
            first = str(body_lines[0] or "").strip()
            if not first:
                body_lines = body_lines[1:]
                continue
            if ("@" in first) or ("$^{" in first):
                body_lines = body_lines[1:]
                continue
            if (len(first) <= 96) and (not re.search(r"[.!?。！？]$", first)):
                body_lines = body_lines[1:]
                continue
            break
    body = "\n".join(body_lines).strip()
    return _trim_paper_guide_prompt_snippet(body, max_chars=max_chars)
