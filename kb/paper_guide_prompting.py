from __future__ import annotations

import re

from kb.paper_guide_shared import (
    _extract_paper_guide_abstract_excerpt,
    _trim_paper_guide_prompt_field,
    _trim_paper_guide_prompt_snippet,
)
from kb.paper_guide_provenance import _extract_figure_number
from kb.source_blocks import normalize_match_text

_PAPER_GUIDE_OVERVIEW_PROMPT_RE = re.compile(
    r"(\bwhat problem\b|\bsolve(?:s|d)?\b|\bmain contribution\b|\bcore contribution\b|\bkey contribution\b|"
    r"\bmain idea\b|\bsummary\b|\bwhat does this paper do\b|解决.*问题|核心贡献|主要贡献|这篇.*讲了什么)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_OVERVIEW_PROMPT_RE_CLEAN = re.compile(
    r"(\u8fd9\u7bc7(?:\u6587\u7ae0|\u8bba\u6587|\u6587\u732e).{0,8}\u8bb2.{0,4}\u4ec0\u4e48|"
    r"\u89e3\u51b3.*\u95ee\u9898|\u6838\u5fc3\u8d21\u732e|\u4e3b\u8981\u8d21\u732e|\u4e3b\u8981\u60f3\u6cd5|\u603b\u7ed3)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_ABSTRACT_PROMPT_RE = re.compile(
    r"(\babstract\b|\bshow\b.*\babstract\b|\bquote\b.*\babstract\b|\boriginal\b.*\babstract\b|"
    r"\btranslate\b.*\babstract\b|\babstract\b.*\btranslate\b|"
    r"\u6458\u8981|\u6458\u8981\u539f\u6587|\u539f\u6587.*\u6458\u8981|\u6458\u8981.*\u539f\u6587|"
    r"\u7ffb\u8bd1.*\u6458\u8981|\u6458\u8981.*\u7ffb\u8bd1|\u628a\u6458\u8981\u539f\u6587\u7ed9\u51fa|\u7ed9\u51fa\u6458\u8981)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_COMPARE_PROMPT_RE = re.compile(
    r"(\badvantage\b|\btrade[\s-]?off\b|\bcompared with\b|\bversus\b|\bvs\.?\b|"
    r"\bopen[-\s]?pinhole\b|\bclosed[-\s]?pinhole\b|\bcnr\b|\bnec\b|\bresolution\b|"
    r"优势|代价|对比|区别|取舍|分辨率|噪声|信噪比)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_COMPARE_PROMPT_RE_CLEAN = re.compile(
    r"(\u5bf9\u6bd4|\u533a\u522b|\u53d6\u820d|\u4f18\u52bf|\u4ee3\u4ef7|\u5206\u8fa8\u7387|\u566a\u58f0|\u4fe1\u566a\u6bd4)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_REPRO_PROMPT_RE = re.compile(
    r"(\breproduc(?:e|ibility)\b|\breplicate\b|\bimplement\b|\bsetup\b|\bhardware\b|\bacquisition\b|"
    r"\bparameter\b|\bprotocol\b|\bmaterials and methods\b|复现|搭建|硬件|采集|参数|流程|方法细节)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_REPRO_PROMPT_RE_CLEAN = re.compile(
    r"(\u590d\u73b0|\u642d\u5efa|\u786c\u4ef6|\u91c7\u96c6|\u53c2\u6570|\u6d41\u7a0b|\u65b9\u6cd5\u7ec6\u8282)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_STRENGTH_PROMPT_RE = re.compile(
    r"(\bstrongest evidence\b|\blimitation\b|\bweakness\b|\bindirect(?:ly)? supported\b|\bwhat is missing\b|"
    r"证据|局限|不足|缺点|薄弱|支撑)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_STRENGTH_PROMPT_RE_CLEAN = re.compile(
    r"(\u8bc1\u636e|\u5c40\u9650|\u4e0d\u8db3|\u7f3a\u70b9|\u8584\u5f31|\u652f\u6491)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_METHOD_PROMPT_RE = re.compile(
    r"(\bhow does it work\b|\bmethod\b|\bmechanism\b|\bprinciple\b|\balgorithm\b|\bnetwork training\b|\btraining\b|\boptimizer\b|\blearning rate\b|\bbatch size\b|\bepoch(?:s)?\b|\bpytorch\b|\bhyperparameter(?:s)?\b|\bloss\b|\bimplementation details?\b|\btraining details?\b|\bapr\b|"
    r"\bwhy\b.*\bapr\b|原理|方法|机制|算法|为什么.*APR|怎么做到的)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_EQUATION_PROMPT_RE = re.compile(
    r"(\bwhat\s+does\s+(?:equation|eq\.?|formula)\b|\b(?:equation|eq\.?|formula)\s*\(?\d+\)?\b|"
    r"\bvariable(?:s)?\b.*\b(?:define|definition|mean|means|denote|denotes|represent|represents)\b|"
    r"\bwhere\s+do(?:es)?\s+the\s+author(?:s)?\s+define\b|"
    r"\bwhat\s+do(?:es)?\s+.*\b(?:denote|represent|mean)\b|"
    r"\u516c\u5f0f|\u65b9\u7a0b|\u53d8\u91cf|\u7b26\u53f7)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_EQUATION_PROMPT_RE_CLEAN = re.compile(
    r"(?i)("
    r"(?:\u516c\u5f0f|\u65b9\u7a0b)\s*\d+|"
    r"(?:\u516c\u5f0f|\u65b9\u7a0b).{0,12}(?:\u662f\u4ec0\u4e48|\u8bb2\u4ec0\u4e48|\u63cf\u8ff0|\u5b9a\u4e49)|"
    r"(?:\u53d8\u91cf|\u7b26\u53f7).{0,12}(?:\u542b\u4e49|\u5b9a\u4e49|\u4ee3\u8868)|"
    r"(?:\u54ea\u91cc|\u54ea\u513f).{0,10}(?:\u5b9a\u4e49|\u89e3\u91ca).{0,10}(?:\u53d8\u91cf|\u7b26\u53f7)"
    r")",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE = re.compile(
    r"(\bwhich prior work\b|\bwhat (?:in-paper )?citation\b|\bwhat reference\b|\bwhich references?\b|"
    r"\battributed to\b|\bintroduced with\b|\bwhen introducing it\b|\bwho introduced\b|"
    # Chinese prompts: include generic “引用了/标出引用号/引用编号” patterns, not only “引用的是哪篇”.
    r"引用的是哪篇|文内参考|参考文献.*哪(?:篇|一|些)|归因于哪篇|"
    r"(?:标出|给出|指出).{0,16}(?:引用|引文)(?:号|编号)?|"
    r"(?:引用|引文)(?:号|编号)|"
    r"引用.*?(?:先前工作|前人工作|相关工作|参考文献|文献))",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE_CLEAN = re.compile(
    r"(?i)("
    r"\u5f15\u7528.*?(?:\u54ea\u7bc7|\u54ea\u4e00\u7bc7|\u54ea\u4e2a|\u54ea\u6761|\u54ea\u4e2a\u5de5\u4f5c|\u54ea\u4e2a\u6587\u732e)|"
    r"(?:\u54ea\u7bc7|\u54ea\u4e00\u7bc7|\u54ea\u4e2a)\u5148\u524d\u5de5\u4f5c|"
    r"\u6587\u5185.*?(?:\u5f15\u7528|\u5f15\u8ff0|\u53c2\u8003\u6587\u732e).*?(?:\u662f|\u5bf9\u5e94)|"
    r"(?:\u53c2\u8003\u6587\u732e|\u5f15\u7528)\u7f16\u53f7|"
    r"\u5f52\u56e0\u4e8e.*?(?:\u54ea\u7bc7|\u54ea\u4e2a\u5de5\u4f5c|\u54ea\u4e2a\u6587\u732e)"
    r")",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_CITATION_LOOKUP_ATTRIBUTION_RE = re.compile(
    r"(?i)\b(?:we use|we used|we invert|introduced(?: as)?|introducing|"
    r"attributed to|akin to|as detailed in|as described in|cited as|"
    r"performed using|using the .*? method|based on)\b"
)
_PAPER_GUIDE_CITATION_LOOKUP_QUERY_STOPWORDS = {
    "reference",
    "references",
    "citation",
    "cite",
    "cited",
    "paper",
    "prior",
    "work",
    "works",
    "where",
    "what",
    "which",
    "when",
    "they",
    "does",
    "with",
    "into",
    "from",
    "that",
    "stated",
    "exactly",
    "connection",
    "introduced",
    "introducing",
    "basis",
    "example",
    "examples",
    "section",
    "acquisition",
    "strategy",
    "list",
    "bibliography",
}
_PAPER_GUIDE_METHOD_PROMPT_RE_CLEAN = re.compile(
    r"(\u8fd9\u4e2a?\u65b9\u6cd5.{0,8}(?:\u5177\u4f53)?\u4ecb\u7ecd|\u65b9\u6cd5.{0,4}\u4ecb\u7ecd|"
    r"\u600e\u4e48\u5de5\u4f5c|\u600e\u4e48\u5b9e\u73b0|\u539f\u7406|\u673a\u5236|\u7b97\u6cd5)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_METHOD_DETAIL_FAMILY_RE = re.compile(
    r"(\bnetwork training\b|\btrain(?:ing)?\s+the\s+network\b|\bhow\s+they\s+train\b|"
    r"\boptimizer\b|\blearning rate\b|\bbatch size\b|\bepoch(?:s)?\b|\bpytorch\b|"
    r"\bhyperparameter(?:s)?\b|\bimplementation details?\b|\btraining details?\b)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_BROAD_OVERVIEW_HINT_RE = re.compile(
    r"(\bproblem is this paper solving\b|\bwhat problem\b|\bwhat is the basic idea\b|\bbasic idea\b|"
    r"\bplain language\b|\bwhy is\b.*\binteresting\b|\bwhat kinds? of applications\b|\bapplications?\b|\breview emphasize\b|"
    r"\bwhat\s+(?:is|are)\b.{0,80}\bdoing here\b|\bin simple terms\b)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_FIGURE_PROMPT_RE = re.compile(
    r"(\bfig(?:ure)?\b|\bpanel\b|\bcaption\b|\blegend\b|\bwalk me through\b.*\bfig(?:ure)?\b|"
    r"图\s*\d+|图注|配图|面板|讲解.*图|解释.*图)",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_REF_SECTION_RE = re.compile(
    r"\b(references?|bibliography|works?\s+cited|citation|appendi(?:x|ces)|supplementary|acknowledg(e)?ments?)\b",
    flags=re.IGNORECASE,
)
_PAPER_GUIDE_BOX_TARGET_RE = re.compile(r"\bbox\s*(\d{1,2})\b", flags=re.IGNORECASE)
_PAPER_GUIDE_BOX_HEADER_RE = re.compile(
    r"(?i)^\s*(?:\*{0,2}\[\s*box\s*(\d{1,2})\b|\*{0,2}box\s*(\d{1,2})\b)"
)
_PAPER_GUIDE_BOX_ONLY_PROMPT_RE = re.compile(
    r"(?i)(?:\bfrom\s+box\s*\d+\s+only\b|\bbox\s*\d+\s+only\b|仅限\s*box\s*\d+\b|只看\s*box\s*\d+\b)"
)
_PAPER_GUIDE_DISCUSSION_ONLY_PROMPT_RE = re.compile(
    r"(?i)(?:\bfrom\s+the\s+discussion(?:\s+section)?\s+only\b|\bdiscussion(?:\s+section)?\s+only\b|仅限\s*discussion\b|只看\s*discussion\b)"
)


def _paper_guide_requested_box_numbers(prompt: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for m in _PAPER_GUIDE_BOX_TARGET_RE.finditer(str(prompt or "").strip()):
        try:
            n = int(m.group(1) or 0)
        except Exception:
            n = 0
        if n <= 0 or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _paper_guide_requested_literal_section_titles(prompt: str) -> list[str]:
    q = str(prompt or "").strip()
    if not q:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"(?i)\b(?:from|in)\s+(?:the\s+)?[\"'“”‘’]([^\"'“”‘’\n]{3,120}?)['\"“”‘’]\s+section(?:\s+only)?\b",
        q,
    ):
        title = re.sub(r"\s+", " ", str(match.group(1) or "")).strip(" \t\r\n\"'“”‘’.:;,-")
        key = title.lower()
        if (not title) or (key in seen):
            continue
        seen.add(key)
        out.append(title)
    return out


def _paper_guide_requested_section_targets(prompt: str) -> list[str]:
    q = str(prompt or "").strip()
    if not q:
        return []
    q_low = q.lower()
    out: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        key = str(name or "").strip().lower()
        if (not key) or (key in seen):
            return
        seen.add(key)
        out.append(key)

    if re.search(r"(?i)\bdiscussion(?:\s+section)?\b", q_low):
        _add("discussion")
    if re.search(r"(?i)\b(?:limitations?|strengths?\s+and\s+limitations?)\b", q_low):
        _add("limitations")
    if re.search(r"(?i)\b(?:future\s+work|future\s+direction(?:s)?|future\s+extension(?:s)?)\b", q_low):
        _add("future_work")
    if re.search(r"(?i)\bconclusions?(?:\s+section)?\b", q_low):
        _add("conclusion")
    if re.search(r"(?i)\b(?:materials?\s+and\s+methods?|methods?|methodology)\b", q_low):
        _add("methods")
    if re.search(r"(?i)\bresults?(?:\s+section)?\b", q_low):
        _add("results")
    if re.search(r"(?i)\bintroduction\b", q_low):
        _add("introduction")
    if re.search(r"(?i)\bacquisition(?:\s+and\s+image\s+reconstruction)?\s+strateg", q_low):
        _add("acquisition_strategy")
    if re.search(r"(?i)\babstract\b", q_low):
        _add("abstract")
    if re.search(r"(?i)\b(?:references?|reference\s+list|works?\s+cited|bibliography)\b", q_low):
        _add("references")
    for title in _paper_guide_requested_literal_section_titles(q):
        _add(title)
    return out


def _paper_guide_requested_heading_hints(prompt: str) -> list[str]:
    q = str(prompt or "").strip()
    if not q:
        return []
    hints: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        item = str(value or "").strip()
        key = item.lower()
        if (not item) or (key in seen):
            return
        seen.add(key)
        hints.append(item)

    section_aliases = {
        "discussion": ["discussion", "discussion section"],
        "limitations": ["limitations", "limitation", "strengths and limitations"],
        "future_work": ["future work", "future directions", "future direction"],
        "conclusion": ["conclusion", "conclusions"],
        "methods": ["methods", "materials and methods", "methodology"],
        "results": ["results", "results section"],
        "introduction": ["introduction"],
        "acquisition_strategy": ["acquisition and image reconstruction strategies", "acquisition strategy"],
        "abstract": ["abstract"],
        "references": ["references", "reference list", "works cited"],
    }
    for sec in _paper_guide_requested_section_targets(q):
        for alias in section_aliases.get(sec, [sec]):
            _add(alias)
    for box_num in _paper_guide_requested_box_numbers(q):
        _add(f"box {int(box_num)}")
    figure_num = _extract_figure_number(q)
    if figure_num > 0:
        _add(f"figure {int(figure_num)}")
        _add(f"fig. {int(figure_num)}")
        _add("caption")
        _add("panel")
    return hints


def _paper_guide_text_matches_requested_section(text: str, section_name: str) -> bool:
    low = normalize_match_text(text)
    sec = str(section_name or "").strip().lower()
    if (not low) or (not sec):
        return False
    aliases = {
        "discussion": ("discussion",),
        "limitations": ("limitations", "limitation", "strengths and limitations"),
        "future_work": ("future work", "future direction", "future directions", "future extension", "future extensions"),
        "conclusion": ("conclusion", "conclusions"),
        "methods": ("materials and methods", "methods", "methodology"),
        "results": ("results", "result"),
        "introduction": ("introduction",),
        "acquisition_strategy": ("acquisition and image reconstruction strategies", "acquisition strategy"),
        "abstract": ("abstract",),
        "references": ("references", "reference list", "works cited", "bibliography"),
    }
    return any(normalize_match_text(alias) in low for alias in aliases.get(sec, (sec,)))


def _paper_guide_text_matches_requested_box(text: str, box_num: int) -> bool:
    try:
        n = int(box_num)
    except Exception:
        n = 0
    if n <= 0:
        return False
    return bool(re.search(rf"(?i)\bbox\s*{int(n)}\b", str(text or "").strip()))


def _paper_guide_text_matches_requested_targets(text: str, *, prompt: str = "") -> bool:
    src = str(text or "").strip()
    q = str(prompt or "").strip()
    if (not src) or (not q):
        return False
    if any(_paper_guide_text_matches_requested_box(src, n) for n in _paper_guide_requested_box_numbers(q)):
        return True
    if any(_paper_guide_text_matches_requested_section(src, sec) for sec in _paper_guide_requested_section_targets(q)):
        return True
    fig_num = _extract_figure_number(q)
    if fig_num > 0 and _extract_figure_number(src) == fig_num:
        return True
    return False


def _paper_guide_box_header_number(text: str) -> int:
    src = str(text or "").strip()
    if not src:
        return 0
    m = _PAPER_GUIDE_BOX_HEADER_RE.search(src)
    if not m:
        return 0
    for group_idx in (1, 2):
        try:
            n = int(m.group(group_idx) or 0)
        except Exception:
            n = 0
        if n > 0:
            return n
    return 0


def _paper_guide_prompt_family(prompt: str, *, intent: str = "") -> str:
    q = str(prompt or "").strip()
    if not q:
        return ""
    if _PAPER_GUIDE_BOX_ONLY_PROMPT_RE.search(q):
        return "box_only"
    if _PAPER_GUIDE_DISCUSSION_ONLY_PROMPT_RE.search(q) and (
        _PAPER_GUIDE_STRENGTH_PROMPT_RE.search(q) or _PAPER_GUIDE_STRENGTH_PROMPT_RE_CLEAN.search(q)
    ):
        return "strength_limits"
    if _PAPER_GUIDE_DISCUSSION_ONLY_PROMPT_RE.search(q):
        return "discussion_only"
    if _PAPER_GUIDE_FIGURE_PROMPT_RE.search(q):
        return "figure_walkthrough"
    if _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE.search(q) or _PAPER_GUIDE_CITATION_LOOKUP_PROMPT_RE_CLEAN.search(q):
        return "citation_lookup"
    # Allow citation lookup to explicitly scope to "in the Abstract" without being misclassified as abstract.
    if _PAPER_GUIDE_ABSTRACT_PROMPT_RE.search(q):
        return "abstract"
    if _PAPER_GUIDE_EQUATION_PROMPT_RE.search(q) or _PAPER_GUIDE_EQUATION_PROMPT_RE_CLEAN.search(q):
        return "equation"
    if _PAPER_GUIDE_REPRO_PROMPT_RE.search(q) or _PAPER_GUIDE_REPRO_PROMPT_RE_CLEAN.search(q) or intent == "experiment":
        return "reproduce"
    tradeoff_without_explicit_comparison = bool(
        re.search(r"(?i)\btrade[\s-]?off\b", q)
        and not re.search(
            r"(?i)\b(?:compared with|versus|vs\.?|open[-\s]?pinhole|closed[-\s]?pinhole|baseline|same incident illumination power)\b",
            q,
        )
    )
    if _PAPER_GUIDE_STRENGTH_PROMPT_RE.search(q) or _PAPER_GUIDE_STRENGTH_PROMPT_RE_CLEAN.search(q):
        return "strength_limits"
    if tradeoff_without_explicit_comparison and (
        _paper_guide_requested_section_targets(q)
        or re.search(r"(?i)\b(?:dynamic range|quantization electronics|mean square error|bottleneck|limitation|limitations|weakness)\b", q)
    ):
        return "strength_limits"
    if _PAPER_GUIDE_COMPARE_PROMPT_RE.search(q) or _PAPER_GUIDE_COMPARE_PROMPT_RE_CLEAN.search(q) or intent == "compare":
        return "compare"
    if _PAPER_GUIDE_METHOD_DETAIL_FAMILY_RE.search(q):
        return "method"
    if _PAPER_GUIDE_BROAD_OVERVIEW_HINT_RE.search(q):
        return "overview"
    if _PAPER_GUIDE_METHOD_PROMPT_RE.search(q) or _PAPER_GUIDE_METHOD_PROMPT_RE_CLEAN.search(q):
        return "method"
    if _PAPER_GUIDE_OVERVIEW_PROMPT_RE.search(q) or _PAPER_GUIDE_OVERVIEW_PROMPT_RE_CLEAN.search(q):
        return "overview"
    return ""


def _paper_guide_prompt_requests_exact_method_support(prompt: str) -> bool:
    src = str(prompt or "").strip()
    if not src:
        return False
    return bool(
        re.search(
            r"(?i)\b(?:"
            r"where\b|where in the pipeline|point me to|exact supporting part|"
            r"exact supporting sentence(?:s)?|exact sentence|exact method paragraph|which part of the paper|"
            r"applied back|re-applied|shift vectors|original iism dataset|"
            r"implementation details?|training details?|network training|optimizer|learning rate|batch size|"
            r"iterations?|epochs?|hyperparameter(?:s)?|pytorch|adam"
            r")\b",
            src,
        )
    )


def _augment_paper_guide_retrieval_prompt(
    prompt: str,
    *,
    family: str = "",
    intent: str = "",
    output_mode: str = "",
) -> str:
    q = str(prompt or "").strip()
    if not q:
        return q
    family_norm = str(family or "").strip().lower() or _paper_guide_prompt_family(q, intent=intent)
    explicit_hints = _paper_guide_requested_heading_hints(q)
    if (not family_norm) and (not explicit_hints):
        return q
    explicit_ref_list_request = bool(
        re.search(r"(?i)\b(?:reference\s+list|works?\s+cited|bibliography)\b", q)
    )

    extras_by_family = {
        "abstract": [
            "abstract",
            "summary",
        ],
        "figure_walkthrough": [
            "figure",
            "caption",
            "legend",
            "panel",
        ],
        "overview": [
            "abstract",
            "introduction",
            "results",
            "discussion",
            "contribution",
            "we introduce",
            "we propose",
        ],
        "compare": [
            "results",
            "resolution",
            "contrast",
            "contrast-to-noise ratio",
            "CNR",
            "noise equivalent contrast",
            "NEC",
            "open pinhole",
            "closed pinhole",
            "iISM-APR",
        ],
        "reproduce": [
            "materials and methods",
            "microscope setup",
            "hardware control",
            "data acquisition",
            "data analysis",
            "adaptive pixel-reassignment",
            "APR",
            "software packages",
        ],
        "equation": [
            "equation",
            "formula",
            "where",
            "defines",
            "denotes",
            "represents",
            "variable",
            "derivation",
        ],
        "strength_limits": [
            "results",
            "discussion",
            "resolution",
            "CNR",
            "limitations",
            "bottleneck",
            "trade-off",
            "acquisition time",
            "modulation rate",
            "DMD",
            "SLM",
            "evidence",
        ],
        "method": [
            "principle of interferometric ISM",
            "adaptive pixel-reassignment",
            "APR",
            "equation",
            "analysis",
            "algorithm",
            "phase correlation",
            "image registration",
            "RVT",
        ],
    }
    extras = list(extras_by_family.get(family_norm) or [])
    if family_norm == "citation_lookup":
        extras = [
            "reference list",
            "works cited",
            "bibliography",
            "reference",
            "references",
            "citation",
            "cited as",
            "introduced with",
            "introduced as",
            "attributed to",
            "prior work",
            "in-paper citation",
        ]
        if not explicit_ref_list_request:
            extras = [item for item in extras if item not in {"reference list", "works cited", "bibliography"}]
    extras.extend(explicit_hints)
    if family_norm == "figure_walkthrough":
        figure_num = _extract_figure_number(q)
        if figure_num > 0:
            extras = [f"figure {figure_num}", f"fig. {figure_num}", *extras]
    if family_norm == "method" and _paper_guide_prompt_requests_exact_method_support(q):
        extras.extend(["methods", "materials and methods", "analysis"])
    if (family_norm == "overview") and (str(output_mode or "").strip().lower() == "critical_review"):
        extras.extend(["limitations", "discussion"])
    # Topic-specific recall helpers (works even when prompt family is ambiguous).
    if re.search(r"(深度学习|神经网络|卷积网络|CNN|neural\s+network|deep\s+learning)", q, flags=re.IGNORECASE):
        extras.extend(["deep learning", "neural network", "CNN"])
    if re.search(r"(重建|reconstruction|压缩感知|compressed\s+sensing)", q, flags=re.IGNORECASE):
        extras.extend([
            "acquisition and image reconstruction strategies",
            "understanding compressed sensing",
            "optimization",
            "l1 norm",
            "total variation",
            "hadamard",
            "fourier",
            "wavelet",
            "matching pursuit",
        ])

    norm = normalize_match_text(q)
    missing: list[str] = []
    for extra in extras:
        if normalize_match_text(extra) in norm:
            continue
        missing.append(extra)
    if not missing:
        return q
    return f"{q} {' '.join(missing[:8])}".strip()


def _paper_guide_allows_citeless_answer(prompt_family: str) -> bool:
    return str(prompt_family or "").strip().lower() == "abstract"


def _looks_like_reference_list_snippet_local(text: str) -> bool:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return False
    if _PAPER_GUIDE_REF_SECTION_RE.search(s[:160]):
        return True
    if len(re.findall(r"\[\d{1,3}\]", s)) >= 2:
        return True
    if re.match(r"^\[\d{1,3}\]\s+[A-Z][A-Za-z][^.!?]{8,}", s):
        low = s.lower()
        if (
            re.search(r"\b(?:19|20)\d{2}\b", s)
            or "proceedings" in low
            or "conference" in low
            or "arxiv" in low
            or "ieee" in low
            or "springer" in low
        ):
            return True
    return False


def _merge_paper_guide_deepread_context(base: str, extra: str, *, prompt_family: str = "", prompt: str = "") -> str:
    base_text = str(base or "").strip()
    extra_text = str(extra or "").strip()
    if not extra_text:
        return base_text
    if not base_text:
        return extra_text
    if extra_text in base_text:
        return base_text

    family = str(prompt_family or "").strip().lower()
    header, _sep, body = base_text.partition("\n")
    body_text = str(body or "").strip()
    if family == "abstract":
        abstract_excerpt = _extract_paper_guide_abstract_excerpt(extra_text, max_chars=1800) or extra_text
        if header.strip():
            return f"{header}\n{abstract_excerpt}".strip()
        return abstract_excerpt
    if family == "figure_walkthrough" and re.search(r"(?:^|\b)(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|panel\b|caption\b)", extra_text, flags=re.IGNORECASE):
        if header.strip() and body_text:
            return f"{header}\n{extra_text}\n\n(Deep-read source excerpt)\n{body_text}".strip()
        if header.strip():
            return f"{header}\n{extra_text}".strip()
        if body_text:
            return f"{extra_text}\n\n(Deep-read source excerpt)\n{body_text}".strip()
        return extra_text
    if _paper_guide_text_matches_requested_targets(extra_text, prompt=prompt):
        if header.strip() and body_text:
            return f"{header}\n{extra_text}\n\n(Deep-read source excerpt)\n{body_text}".strip()
        if header.strip():
            return f"{header}\n{extra_text}".strip()
        if body_text:
            return f"{extra_text}\n\n(Deep-read source excerpt)\n{body_text}".strip()
        return extra_text
    return f"{base_text}\n\n(Deep-read source excerpt)\n{extra_text}".strip()


def _paper_guide_evidence_card_use_hint(prompt_family: str) -> str:
    family = str(prompt_family or "").strip().lower()
    if family == "abstract":
        return "Use the abstract span directly; preserve sentence order; no title/authors."
    if family == "figure_walkthrough":
        return "Describe only the figure/panels stated here; preserve panel letters and figure numbers exactly."
    if family == "compare":
        return "Use only the reported comparison and numbers from this card; mark missing values as not stated."
    if family == "citation_lookup":
        return "Extract the exact in-paper ref numbers or reference-list entries from this card; do not answer 'not stated' when the refs are explicit."
    if family == "method":
        return "Explain the mechanism only if this card states it explicitly; otherwise say not stated."
    if family == "reproduce":
        return "List only hardware, acquisition, or analysis details explicitly shown here."
    if family == "equation":
        return "Keep the equation and its variable-definition sentence together; do not replace symbols with a generic paraphrase."
    if family == "strength_limits":
        return "Keep strengths and limitations tied to the evidence scope shown here."
    return "Summarize only what this evidence card explicitly supports."


def _build_paper_guide_evidence_cards_block(
    cards: list[dict],
    *,
    prompt: str = "",
    prompt_family: str = "",
    max_cards: int = 4,
) -> str:
    try:
        limit = max(1, int(max_cards))
    except Exception:
        limit = 4

    family = str(prompt_family or "").strip().lower() or _paper_guide_prompt_family(prompt)
    lines: list[str] = []
    seen: set[tuple[int, str, str]] = set()
    for raw_card in cards or []:
        if len(lines) >= limit:
            break
        if not isinstance(raw_card, dict):
            continue
        try:
            doc_idx = max(1, int(raw_card.get("doc_idx") or 0))
        except Exception:
            doc_idx = len(lines) + 1
        sid = _trim_paper_guide_prompt_field(str(raw_card.get("sid") or "").strip(), max_chars=24)
        heading = _trim_paper_guide_prompt_field(str(raw_card.get("heading") or "").strip(), max_chars=96)
        cue = _trim_paper_guide_prompt_field(str(raw_card.get("cue") or "").strip(), max_chars=160)
        source_path = str(raw_card.get("source_path") or "").strip()
        primary = str(raw_card.get("snippet") or "").strip()
        deepread_texts = [str(item or "").strip() for item in list(raw_card.get("deepread_texts") or []) if str(item or "").strip()]

        snippet = ""
        if family == "abstract":
            for cand in deepread_texts + [primary]:
                snippet = _extract_paper_guide_abstract_excerpt(cand, max_chars=560)
                if snippet:
                    break
        elif family == "figure_walkthrough":
            for cand in deepread_texts + [primary]:
                if re.search(r"(?:^|\b)(?:fig(?:ure)?\.?\s*\d+|图\s*\d+|panel\b|caption\b)", cand, flags=re.IGNORECASE):
                    snippet = _trim_paper_guide_prompt_snippet(cand, max_chars=520)
                    if snippet:
                        break
        if not snippet:
            for cand in [primary, *deepread_texts]:
                snippet = _trim_paper_guide_prompt_snippet(cand, max_chars=420)
                if snippet:
                    break
        if not snippet:
            continue

        refs: list[int] = []
        for item in list(raw_card.get("candidate_refs") or []):
            try:
                refs.append(int(item))
            except Exception:
                continue
        key = (int(doc_idx), normalize_match_text(source_path), normalize_match_text(snippet[:240]))
        if key in seen:
            continue
        seen.add(key)

        parts = [f"DOC-{doc_idx}"]
        if sid:
            parts.append(f"sid={sid}")
        if heading:
            parts.append(f"heading={heading}")
        if refs:
            parts.append("refs=" + ", ".join(str(n) for n in refs[:6]))
            if sid:
                parts.append(f"cite_example=[[CITE:{sid}:{refs[0]}]]")
        parts.append("use=" + _paper_guide_evidence_card_use_hint(family))
        lines.append("- " + " | ".join(parts))
        lines.append("  snippet: " + snippet.replace("\n", "\n  "))
        if cue and cue not in snippet:
            lines.append("  cue: " + cue)

    if not lines:
        return ""

    return (
        "Paper-guide evidence cards:\n"
        "- Treat each DOC-k card as a claim boundary: keep paper-grounded claims inside that card's snippet and use-rule.\n"
        "- If a DOC-k card includes cite_example=[[CITE:<sid>:<ref_num>]], reuse that exact marker on the claim or bullet derived from that card.\n"
        "- If a needed detail is absent from all cards, say it is not stated in the retrieved paper evidence.\n"
        + "\n".join(lines)
    )


def _build_paper_guide_citation_grounding_block(
    answer_hits: list[dict],
    *,
    max_blocks: int = 4,
    hit_source_path=None,
    paper_guide_focus_heading=None,
    cite_source_id=None,
    extract_candidate_ref_nums=None,
    extract_candidate_ref_cue_texts=None,
) -> str:
    try:
        limit = max(1, int(max_blocks))
    except Exception:
        limit = 4

    source_path_getter = hit_source_path or (lambda hit: str((((hit or {}).get("meta") or {}).get("source_path") or "")).strip())
    heading_getter = paper_guide_focus_heading or (lambda hit: str((((hit or {}).get("meta") or {}).get("heading_path") or "")).strip())
    sid_getter = cite_source_id or (lambda src: "")
    candidate_ref_getter = extract_candidate_ref_nums or (lambda _hits, **_kwargs: [])
    cue_text_getter = extract_candidate_ref_cue_texts or (lambda _hit, **_kwargs: [])

    lines: list[str] = []
    seen: set[tuple[str, str, tuple[int, ...], str]] = set()
    for i, hit in enumerate(answer_hits or [], start=1):
        if len(lines) >= limit:
            break
        src = str(source_path_getter(hit) or "").strip()
        if not src:
            continue
        candidate_refs = candidate_ref_getter([hit], source_path=src, max_candidates=6)
        if not candidate_refs:
            continue
        heading = _trim_paper_guide_prompt_field(str(heading_getter(hit) or "").strip(), max_chars=96)
        cue_texts = cue_text_getter(hit, max_cues=1, max_chars=160)
        cue = _trim_paper_guide_prompt_field(str(cue_texts[0] or "").strip(), max_chars=160) if cue_texts else ""
        key = (
            src.lower(),
            heading.lower(),
            tuple(int(n) for n in candidate_refs[:6]),
            cue.lower(),
        )
        if key in seen:
            continue
        seen.add(key)

        sid = str(sid_getter(src) or "").strip()
        refs_txt = ", ".join(str(int(n)) for n in candidate_refs[:6])
        parts = [f"DOC-{i}", f"sid={sid}", f"refs={refs_txt}"]
        if heading:
            parts.append(f"heading={heading}")
        if cue:
            parts.append(f"cue={cue}")
        lines.append("- " + " | ".join(parts))

    if not lines:
        return ""

    return (
        "Paper-guide citation grounding hints:\n"
        "- Match each claim to the same DOC-k evidence block before choosing [[CITE:<sid>:<ref_num>]].\n"
        "- Prefer ref numbers listed on that DOC-k line; only go outside that list when DOI or author-year text in the claim gives a stronger identity signal.\n"
        + "\n".join(lines)
    )


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


_PAPER_GUIDE_DOC_MAP_PROMPT_RE = re.compile(
    r"(?i)(?:\bdoc\s*map\b|\breading\s+map\b|\boutline\b|\btable\s+of\s+contents\b|\btoc\b|"
    r"\bsection(?:\s*-\s*|\s+)by(?:\s*-\s*|\s+)section\b|\bverbatim\s+anchor(?:\s+sentence)?s?\b|"
    r"\banchor\s+sentence(?:s)?\b|\bmajor\s+section(?:s)?\b|"
    r"\u76ee\u5f55|\u5927\u7eb2|\u6587\u6863\u5730\u56fe|\u5730\u56fe|\u6982\u89c8|\u7ae0\u8282\u6982\u89c8|\u6bcf\u4e00\u5757|\u6bcf\u6bb5\u603b\u7ed3|\u6309\u6bb5\u603b\u7ed3|\u5148\u603b\u7ed3\u4e00\u4e0b)",
)


def _paper_guide_prompt_requests_doc_map(prompt: str) -> bool:
    q = str(prompt or "").strip()
    if not q:
        return False
    return bool(_PAPER_GUIDE_DOC_MAP_PROMPT_RE.search(q))
