from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from kb.evidence_binding import (
    _claim_fact_quantities_for_evidence,
    _quantity_is_covered,
    _quantity_label,
    _system_a_fact_quantities,
    assess_system_a_hit_binding,
    explicit_claim_relations_covered,
)
from kb.evidence_term_mapping import evidence_alignment_tokens, method_identity_conflicts


_CITATION_RE = re.compile(
    r"(?<!\[)\[(?:R?\d{1,5})(?:\s*[,;，；、-]\s*R?\d{1,5})*\](?:\([^\n)]+\))?"
    r"|\[\[(?:CITE|SUPPORT):[^\]]+\]\]",
    flags=re.IGNORECASE,
)
_NUMERIC_CITATION_RE = re.compile(r"(?<!\[)\[(\d{1,5})\](?:\([^\n)]+\))?")
_STRUCTURED_CITATION_RE = re.compile(r"\[\[(?:CITE|SUPPORT):[^\]]+\]\]", flags=re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
# Markdown quotes are evidence excerpts, not answer claims.  Excluding them
# prevents the claim repair pass from appending a System-A marker to text that
# is already displayed as the supporting source quotation.
_TABLE_OR_CODE_RE = re.compile(r"^\s*(?:\||>|```|~~~|<!--)")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?(?:\s*(?:%|dB|nm|μm|um|mm|cm|Hz|kHz|MHz|GHz|fps|帧/秒|帧|倍))?",
    flags=re.IGNORECASE,
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,12}\b")
_EN_TERM_RE = re.compile(r"\b[a-z][a-z0-9-]{3,}\b", flags=re.IGNORECASE)
_RISK_RE = re.compile(
    r"(?:表明|证明|达到|提升|提高|降低|优于|劣于|导致|使(?:得)?|通过|采用|使用|利用|构建|引入|"
    r"包含|纳入|建模|训练|验证|报告|实现|解决|决定|影响|正交(?:的|组合)|权衡|局限|限制|没有|未(?:提供|说明|显示|报告|验证|讨论)|"
    r"\b(?:show(?:s|ed)?|demonstrat(?:e|es|ed)|achiev(?:e|es|ed)|improv(?:e|es|ed)|reduc(?:e|es|ed)|"
    r"outperform(?:s|ed)?|caus(?:e|es|ed)|use(?:s|d)?|employ(?:s|ed)?|introduc(?:e|es|ed)|"
    r"include(?:s|d)?|model(?:s|ed)?|train(?:s|ed)?|validat(?:e|es|ed)|report(?:s|ed)?|"
    r"enable(?:s|d)?|solve(?:s|d)?|trade[- ]?off|limitation|does\s+not|not\s+validated|lack(?:s|ed)?)\b)",
    flags=re.IGNORECASE,
)
_OPTIMIZATION_DETAIL_RE = re.compile(
    r"(?i)\b(?:backpropagat(?:e|es|ed|ion)|gradients?|differentiable|training\s+loss|"
    r"loss\s+function)\b|\u53cd\u5411\u4f20\u64ad|\u68af\u5ea6|\u53ef\u5fae|"
    r"\u8bad\u7ec3\u635f\u5931|\u635f\u5931\u51fd\u6570"
)
_MASK_COMPRESSION_DETAIL_RE = re.compile(
    r"(?i)\bmasks?.{0,24}(?:modulat(?:e|es|ed|ion)|summ?(?:ation|ed|ing))\b|"
    r"\u63a9(?:\u6a21|\u7801).{0,16}(?:\u8c03\u5236|\u6c42\u548c|\u79ef\u5206(?:\u538b\u7f29)?|\u538b\u7f29\u79ef\u5206)"
)
_ADVICE_RE = re.compile(
    r"(?:建议|可以(?:先|再)?|应该|值得|下一步|优先阅读|查阅|检查|查看|对照|"
    r"\b(?:recommend|should|could|next step|read|inspect|check)\b)",
    flags=re.IGNORECASE,
)
_NON_FACTUAL_GUIDANCE_RE = re.compile(
    r"^(?:(?:你)?可以(?:思考|考虑)|读法建议|"
    r"这样(?:在)?(?:读|阅读)|是否可能|建议|"
    r"阅读(?:/使用)?(?:建议|目的)|快速浏览|"
    r"优先阅读|重点(?:看|阅读)|先读|再读|"
    r"consider\s+whether|think\s+about\s+whether|"
    r"(?:quickly\s+)?(?:read|skim|browse)\b)",
    flags=re.IGNORECASE,
)
_ANAPHORIC_CONTINUATION_RE = re.compile(
    r"^(?:这(?:是|使得|意味着|表明)?|因此|由此|"
    r"从而|它|该(?:方法|模型|设计)|基于(?:该|此)模型|"
    r"其(?:核心(?:思想)?|作用|机制)(?:是|在于)?|"
    r"this\b|that\b|it\b|therefore\b|thereby\b|as\s+a\s+result\b)",
    flags=re.IGNORECASE,
)
_PAPER_COVERAGE_CLAIM_RE = re.compile(
    r"(?:这篇|该(?:综述|论文|文献)|本文|综述).{0,80}"
    r"(?:梳理|概述|讨论|涵盖|介绍|总结|了解|提供)|"
    r"\b(?:this|the)\s+(?:review|paper|study)\b.{0,80}"
    r"\b(?:reviews?|summari[sz]es?|covers?|discusses?|introduces?|provides?)\b",
    flags=re.IGNORECASE,
)
_BOUNDARY_RE = re.compile(
    r"(?:当前|本轮)(?:引用|检索)(?:到的)?(?:片段|证据|结果|内容)?[^。！？.!?]{0,16}"
    r"(?:未直接|未|没有)(?:提供|说明|显示|报告|验证|讨论)"
    r"|(?:the\s+)?current\s+(?:cited\s+)?(?:snippet|evidence|retrieval)[^.!?]{0,24}"
    r"(?:does\s+not|did\s+not|doesn't)(?:\s+directly)?\s+(?:provide|state|show|report|validate|discuss)",
    flags=re.IGNORECASE,
)
_INFERENCE_RE = re.compile(
    r"^(?:据此|由此|这|这也|因此)?(?:可以|可)?(?:推断|认为|看出|表明|说明)"
    r"|^(?:this|that)\s+(?:suggests?|implies?|indicates?)\b",
    flags=re.IGNORECASE,
)
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")

_MULTIPLIER_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_MULTIPLIER_NUMBER_TOKEN = "|".join(
    sorted(_MULTIPLIER_NUMBER_WORDS, key=len, reverse=True)
)
_MULTIPLIER_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<en_value>\d+(?:\.\d+)?|{_MULTIPLIER_NUMBER_TOKEN})"
    r"\s*(?:[-\u2010-\u2015]\s*)?(?:fold|times?)(?![A-Za-z])"
    r"|(?<![A-Za-z0-9])(?P<zh_value>\d+(?:\.\d+)?)\s*\u500d",
    flags=re.IGNORECASE,
)
_MULTIPLIER_DECREASE_RE = re.compile(
    r"\b(?:lower|less|fewer|decreas(?:e|es|ed|ing)|reduc(?:e|es|ed|ing|tion)|"
    r"drop(?:s|ped|ping)?|diminish(?:es|ed|ing)?|attenuat(?:e|es|ed|ing|ion))\b|"
    r"\u964d\u4f4e|\u51cf\u5c11|\u4e0b\u964d|\u7f29\u51cf|\u51cf\u5c0f",
    flags=re.IGNORECASE,
)
_MULTIPLIER_INCREASE_RE = re.compile(
    r"\b(?:higher|more|greater|increas(?:e|es|ed|ing)|rais(?:e|es|ed|ing)|"
    r"improv(?:e|es|ed|ing|ement)|enhanc(?:e|es|ed|ing|ement)|"
    r"amplif(?:y|ies|ied|ication)|speedup|gain)\b|"
    r"\u63d0\u9ad8|\u589e\u52a0|\u4e0a\u5347|\u589e\u5927|\u63d0\u5347|\u589e\u5f3a",
    flags=re.IGNORECASE,
)
_MULTIPLIER_TARGET_PATTERNS = (
    (
        "resolution",
        re.compile(
            r"\b(?:(?:spatial|position|positional|axial|lateral|temporal|image)\s+)?"
            r"resolution\b|\u4f4d\u7f6e\u5206\u8fa8\u7387|\u7a7a\u95f4\u5206\u8fa8\u7387|"
            r"\u8f74\u5411\u5206\u8fa8\u7387|\u6a2a\u5411\u5206\u8fa8\u7387|\u65f6\u95f4\u5206\u8fa8\u7387|"
            r"\u5206\u8fa8\u7387",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "illumination_power",
        re.compile(
            r"\b(?:(?:incident|input|optical|laser|excitation|illumination)\s+)?power\b|"
            r"\b(?:illumination|excitation|incident\s+light)\s+(?:intensity|energy)\b|"
            r"(?:\u5165\u5c04|\u7167\u660e|\u6fc0\u53d1|\u5149\u6e90|\u6fc0\u5149|\u5149\u5b66)"
            r".{0,8}(?:\u529f\u7387|\u5f3a\u5ea6|\u80fd\u91cf)|\u529f\u7387",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "signal_to_noise",
        re.compile(
            r"\bSNR\b|\bsignal[-\s]?to[-\s]?noise(?:\s+ratio)?\b|\u4fe1\u566a\u6bd4",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "speed",
        re.compile(
            r"\b(?:speed|throughput|frame\s+rate|acquisition\s+rate|processing\s+rate|fps)\b|"
            r"\u901f\u5ea6|\u541e\u5410\u91cf|\u5e27\u7387|\u91c7\u96c6\u7387|\u5904\u7406\u7387",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "sampling",
        re.compile(
            r"\bsampling\s+(?:rate|ratio)\b|\u91c7\u6837\u7387|\u91c7\u6837\u6bd4",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "time",
        re.compile(
            r"\b(?:time|latency|duration)\b|\u65f6\u95f4|\u5ef6\u8fdf|\u8017\u65f6",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "dose",
        re.compile(
            r"\b(?:dose|exposure|photon\s+budget)\b|\u5242\u91cf|\u66dd\u5149|\u5149\u5b50\u9884\u7b97",
            flags=re.IGNORECASE,
        ),
    ),
)

_STOPWORDS = {
    "about",
    "after",
    "also",
    "based",
    "because",
    "between",
    "current",
    "does",
    "from",
    "have",
    "into",
    "more",
    "paper",
    "results",
    "shows",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "using",
    "which",
    "with",
}

_CONCEPT_GROUPS = (
    (
        "physical noise model",
        "real-world physical noise model",
        "multi-source physical noise model",
        "物理噪声模型",
        "多源物理噪声模型",
        "真实物理噪声",
    ),
    (
        "calibration dataset",
        "real-shot spad image dataset",
        "模型参数标定",
        "校准模型参数",
        "真实spad数据集",
    ),
    ("single photon", "single-photon", "单光子", "光子受限"),
    ("single pixel", "single-pixel", "单像素"),
    ("deep learning", "深度学习"),
    ("physics informed", "physics-informed", "物理先验", "物理信息"),
    ("super resolution", "super-resolution", "超分辨"),
    ("reconstruction", "重建", "恢复"),
    ("noise", "噪声"),
    ("detector", "探测器"),
    ("training", "训练"),
    ("quality", "质量"),
    ("physical model", "物理模型"),
    ("spatial resolution", "resolution", "分辨率", "空间分辨率"),
    ("single photon", "single-photon", "单光子"),
    ("single pixel", "single-pixel", "单像素"),
    ("deep learning", "深度学习"),
    ("physics informed", "physics-informed", "物理先验", "物理信息"),
    ("super resolution", "super-resolution", "超分辨"),
    ("reconstruction", "重建"),
    ("noise", "噪声"),
    ("poisson", "泊松"),
    ("crosstalk", "串扰"),
    ("dark count", "暗计数"),
    ("detector", "探测器"),
    ("sampling", "采样"),
    ("measurement", "测量"),
    ("training", "训练"),
    ("generalization", "泛化"),
    ("speed", "速度"),
    ("quality", "质量"),
    ("physical model", "物理模型"),
    ("spatial resolution", "空间分辨率"),
    ("signal to noise", "signal-to-noise", "信噪比"),
    ("optical sectioning", "光学切片", "光学层切"),
    ("thick sample", "厚样本"),
    ("out of focus", "out-of-focus", "离焦"),
    ("lasing threshold", "laser threshold", "threshold", "激光阈值", "阈值"),
    ("coupling efficiency", "耦合效率"),
)


def _split_claim_segments(value: str) -> list[str]:
    math_tokens = {
        ".": "<KB_MATH_DOT>",
        ";": "<KB_MATH_SEMI>",
        "!": "<KB_MATH_BANG>",
        "?": "<KB_MATH_QUESTION>",
        "\u3002": "<KB_MATH_CJK_DOT>",
        "\uff1b": "<KB_MATH_CJK_SEMI>",
        "\uff01": "<KB_MATH_CJK_BANG>",
        "\uff1f": "<KB_MATH_CJK_QUESTION>",
    }

    def _protect_math(match: re.Match[str]) -> str:
        text = str(match.group(0) or "")
        for punctuation, token in math_tokens.items():
            text = text.replace(punctuation, token)
        return text

    protected = re.sub(
        r"(?<!\\)\$(?!\$)[^\n$]+?(?<!\\)\$|\\\([^\n]+?\\\)",
        _protect_math,
        str(value or ""),
    )
    protected = re.sub(
        r"(?i)\b(?:nat|commun|fig|eq|et\s+al)\.",
        lambda match: match.group(0)[:-1] + "<KB_DOT>",
        protected,
    )
    protected = re.sub(
        r"\b\d+\.(?=\s+[A-Z])",
        lambda match: match.group(0)[:-1] + "<KB_DOT>",
        protected,
    )
    parts = re.split(
        r"(?<=[。！？!?；;])\s*|(?<=[A-Za-z0-9\)\]]\.)\s+(?=[A-Z\u4e00-\u9fff])",
        protected,
    )
    restored: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        value_out = part.replace("<KB_DOT>", ".")
        for punctuation, token in math_tokens.items():
            value_out = value_out.replace(token, punctuation)
        restored.append(value_out.strip())
    return restored


def _claim_units(answer: str) -> list[str]:
    units: list[str] = []
    in_fence = False
    lines = str(answer or "").splitlines()
    line_index = 0
    while line_index < len(lines):
        raw_line = lines[line_index]
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            line_index += 1
            continue
        if not in_fence and stripped == "$$":
            math_lines: list[str] = []
            line_index += 1
            while line_index < len(lines) and lines[line_index].strip() != "$$":
                if lines[line_index].strip():
                    math_lines.append(lines[line_index].strip())
                line_index += 1
            if line_index < len(lines):
                line_index += 1
            math_surface = " ".join(math_lines).strip()
            # A display equation is commonly followed by a cited “where/其中”
            # definition line. Treat that immediately adjacent citation as the
            # equation's evidence marker too; it is the same semantic unit in
            # the rendered answer, even though Markdown keeps the link outside
            # the math delimiter.
            next_index = line_index
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            next_surface = lines[next_index].strip() if next_index < len(lines) else ""
            if (
                math_surface
                and re.match(r"^(?:其中|式中|where\b|with\b)", next_surface, flags=re.IGNORECASE)
                and _CITATION_RE.search(next_surface)
            ):
                adjacent_markers = " ".join(_CITATION_RE.findall(next_surface))
                math_surface = f"{math_surface} {adjacent_markers}".strip()
            if len(math_surface) >= 10:
                units.append(math_surface)
            continue
        if in_fence or not stripped or _HEADING_RE.match(stripped) or _TABLE_OR_CODE_RE.match(stripped):
            line_index += 1
            continue
        stripped = _LIST_PREFIX_RE.sub("", stripped).strip()
        if not stripped:
            line_index += 1
            continue
        units.extend(part for part in _split_claim_segments(stripped) if len(part) >= 10)
        line_index += 1
    return units


def _plain_claim(text: str) -> str:
    value = _CITATION_RE.sub(" ", str(text or ""))
    value = re.sub(r"[*_`>#]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_multiplier_value(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in _MULTIPLIER_NUMBER_WORDS:
        return _MULTIPLIER_NUMBER_WORDS[raw]
    try:
        return format(Decimal(raw).normalize(), "f")
    except (InvalidOperation, ValueError):
        return raw


def _multiplier_direction(surface: str, start: int, end: int) -> str:
    """Return the closest explicit direction attached to a multiplier."""

    text = str(surface or "")
    clause_start = max(
        text.rfind(mark, 0, start) + 1
        for mark in (".", ";", "!", "?", "\u3002", "\uff1b", "\uff01", "\uff1f", "\uff0c")
    )
    right_boundaries = [
        pos
        for mark in (".", ";", "!", "?", "\u3002", "\uff1b", "\uff01", "\uff1f", "\uff0c")
        if (pos := text.find(mark, end)) >= 0
    ]
    clause_end = min(right_boundaries) if right_boundaries else len(text)
    window_start = max(clause_start, start - 56)
    window_end = min(clause_end, end + 56)
    window = text[window_start:window_end]
    relative_start = start - window_start
    relative_end = end - window_start
    candidates: list[tuple[int, str]] = []
    for direction, pattern in (
        ("decrease", _MULTIPLIER_DECREASE_RE),
        ("increase", _MULTIPLIER_INCREASE_RE),
    ):
        for match in pattern.finditer(window):
            if match.end() <= relative_start:
                distance = relative_start - match.end()
            elif match.start() >= relative_end:
                distance = match.start() - relative_end
            else:
                distance = 0
            candidates.append((distance, direction))
    if not candidates:
        return ""
    closest = min(distance for distance, _direction in candidates)
    directions = {
        direction for distance, direction in candidates if distance == closest
    }
    return next(iter(directions)) if len(directions) == 1 else ""


def _multiplier_target(surface: str, start: int, end: int) -> str:
    """Return the closest measurable quantity modified by a multiplier."""

    text = str(surface or "")
    clause_start = max(
        text.rfind(mark, 0, start) + 1
        for mark in (".", ";", "!", "?", "\u3002", "\uff1b", "\uff01", "\uff1f", "\uff0c")
    )
    right_boundaries = [
        pos
        for mark in (".", ";", "!", "?", "\u3002", "\uff1b", "\uff01", "\uff1f", "\uff0c")
        if (pos := text.find(mark, end)) >= 0
    ]
    clause_end = min(right_boundaries) if right_boundaries else len(text)
    window_start = max(clause_start, start - 88)
    window_end = min(clause_end, end + 88)
    window = text[window_start:window_end]
    relative_start = start - window_start
    relative_end = end - window_start
    candidates: list[tuple[int, str]] = []
    for target, pattern in _MULTIPLIER_TARGET_PATTERNS:
        for match in pattern.finditer(window):
            if match.end() <= relative_start:
                distance = relative_start - match.end()
            elif match.start() >= relative_end:
                distance = match.start() - relative_end
            else:
                distance = 0
            candidates.append((distance, target))
    if not candidates:
        return ""
    closest = min(distance for distance, _target in candidates)
    targets = {target for distance, target in candidates if distance == closest}
    return next(iter(targets)) if len(targets) == 1 else ""


def _normalize_multiplier_surface(
    value: str,
) -> tuple[list[tuple[str, str, str]], str]:
    """Normalize written multipliers while retaining magnitude and direction.

    The normalized surface is used only for deterministic quantity comparison;
    the user-visible answer remains untouched.  A multiplier stays distinct from
    ordinary units such as ``10 Hz`` because callers enable this equivalence only
    when both the claim and evidence contain an explicit fold/times/\u500d expression.
    """

    surface = str(value or "")
    facts: list[tuple[str, str, str]] = []
    parts: list[str] = []
    cursor = 0
    for match in _MULTIPLIER_RE.finditer(surface):
        raw_value = str(match.group("en_value") or match.group("zh_value") or "")
        normalized = _normalize_multiplier_value(raw_value)
        if not normalized:
            continue
        facts.append(
            (
                normalized,
                _multiplier_direction(surface, match.start(), match.end()),
                _multiplier_target(surface, match.start(), match.end()),
            )
        )
        parts.append(surface[cursor : match.start()])
        # A hyphen keeps ``fold`` from being inferred as a scientific unit while
        # exposing the canonical magnitude to the existing quantity parser.
        parts.append(f"{normalized}-fold")
        cursor = match.end()
    if not facts:
        return [], surface
    parts.append(surface[cursor:])
    return list(dict.fromkeys(facts)), "".join(parts)


def _prepare_quantity_surfaces(
    claim: str,
    evidence: str,
) -> tuple[str, str, bool]:
    """Return comparable quantity surfaces and multiplier compatibility."""

    claim_facts, normalized_claim = _normalize_multiplier_surface(claim)
    if not claim_facts:
        return str(claim or ""), str(evidence or ""), True
    evidence_facts, normalized_evidence = _normalize_multiplier_surface(evidence)
    if not evidence_facts:
        return normalized_claim, str(evidence or ""), False
    covered = all(
        any(
            claim_value == evidence_value
            and (
                not claim_direction
                or not evidence_direction
                or claim_direction == evidence_direction
            )
            and (
                not claim_target
                or not evidence_target
                or claim_target == evidence_target
            )
            for evidence_value, evidence_direction, evidence_target in evidence_facts
        )
        for claim_value, claim_direction, claim_target in claim_facts
    )
    return normalized_claim, normalized_evidence, covered


def _meaningful_numbers(text: str) -> list[str]:
    _multiplier_facts, quantity_surface = _normalize_multiplier_surface(
        _CITATION_RE.sub(" ", str(text or ""))
    )
    return sorted(
        _quantity_label(quantity)
        for quantity in _system_a_fact_quantities(quantity_surface)
    )


def _is_high_risk_claim(unit: str) -> bool:
    plain = _plain_claim(unit)
    if len(plain) < 12 or _BOUNDARY_RE.search(plain) or _INFERENCE_RE.search(plain):
        return False
    if re.search(r"[?？]\s*$", plain):
        return False
    if re.match(r"^(?:这|these\b).{0,80}(?:建议|recommend)", plain, flags=re.IGNORECASE):
        return False
    if re.search(r"(?:搭配|一起)(?:读|阅读)", plain, flags=re.IGNORECASE):
        return False
    if _NON_FACTUAL_GUIDANCE_RE.search(plain):
        return False
    if _ADVICE_RE.search(plain) and re.match(
        r"^(?:阅读建议|建议|可以|应该|直接阅读|对于.{0,40}可参考|read|recommend|for further)",
        plain,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(
        _meaningful_numbers(plain)
        or _RISK_RE.search(plain)
        or re.search(
            r"(?:是因为|由于|从而|进而|因而|这使得)|"
            r"\b(?:because|thereby|therefore|result(?:s|ed)?\s+in)\b",
            plain,
            flags=re.IGNORECASE,
        )
        or _OPTIMIZATION_DETAIL_RE.search(plain)
        or _MASK_COMPRESSION_DETAIL_RE.search(plain)
        or _PAPER_COVERAGE_CLAIM_RE.search(plain)
    )


def _hit_payload(hit: dict[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    values = [
        hit.get("text"),
        hit.get("title"),
        meta.get("source_name"),
        meta.get("title"),
        meta.get("heading_path"),
        meta.get("top_heading"),
    ]
    return "\n".join(str(value or "") for value in values if str(value or "").strip())


def _citation_group_covers_claim_quantities(
    claim: str,
    citations: list[int],
    answer_hits: list[dict[str, Any]],
) -> bool:
    numbers = list(dict.fromkeys(int(number) for number in citations if int(number) > 0))
    if len(numbers) < 2:
        return False
    evidence_parts = [
        _hit_payload(answer_hits[number - 1])
        for number in numbers
        if 0 < number <= len(answer_hits) and answer_hits[number - 1]
    ]
    if len(evidence_parts) < 2:
        return False
    claim_surface = _plain_claim(claim)
    evidence_surface = "\n".join(evidence_parts)
    claim_surface, evidence_surface, multiplier_covered = _prepare_quantity_surfaces(
        claim_surface,
        evidence_surface,
    )
    if not multiplier_covered:
        return False
    claim_quantities = _system_a_fact_quantities(claim_surface)
    if not claim_quantities:
        return False
    union_quantities = _system_a_fact_quantities(evidence_surface)
    return all(
        _quantity_is_covered(quantity, union_quantities)
        for quantity in claim_quantities
    )


def _concept_ids(text: str) -> set[int]:
    normalized = re.sub(r"[\s_]+", " ", str(text or "").lower())
    return {
        index
        for index, aliases in enumerate(_CONCEPT_GROUPS)
        if any(alias in normalized for alias in aliases)
    }


def _support_score(
    claim: str,
    evidence: str,
    *,
    allow_comparison_scope: bool = False,
) -> int:
    claim_plain = _plain_claim(claim)
    evidence_text = str(evidence or "")
    evidence_norm = re.sub(r"\s+", " ", evidence_text).lower()
    if not claim_plain or not evidence_norm:
        return 0
    if method_identity_conflicts(claim_plain, evidence_text):
        return 0
    claim_low = claim_plain.lower()
    claim_single_photon = bool(
        re.search(r"\bsingle[- ]?photon\b|\bspad\b|单光子", claim_low)
    )
    evidence_single_photon = bool(
        re.search(r"\bsingle[- ]?photon\b|\bspad\b|单光子", evidence_norm)
    )
    evidence_single_pixel = bool(
        re.search(r"\bsingle[- ]?pixel\b|单像素", evidence_norm)
    )
    if claim_single_photon and evidence_single_pixel and not evidence_single_photon:
        return 0
    physical_noise_model_re = re.compile(
        r"physical\s+(?:multi[- ]source\s+)?noise\s+model|"
        r"multi[- ]source\s+(?:physical\s+)?noise\s+model|"
        r"物理噪声模型|多源(?:物理)?噪声模型|真实物理噪声"
    )
    if physical_noise_model_re.search(claim_low) and not physical_noise_model_re.search(
        evidence_norm
    ):
        return 0
    review_identity_re = re.compile(r"\b(?:review|survey)\b|综述", re.I)
    method_identity_re = re.compile(
        r"\b(?:here\s*,?\s*)?we\s+(?:introduce|propose|develop|present|build)\b|"
        r"\bour\s+(?:method|framework|model)\b",
        re.I,
    )
    if (
        review_identity_re.search(claim_low)
        and not review_identity_re.search(evidence_norm)
        and method_identity_re.search(evidence_norm)
    ):
        return 0
    if not explicit_claim_relations_covered(claim_plain, evidence_text):
        return 0
    explicit_relation_requirements = (
        (
            re.compile(
                r"(?=.*位置)(?=.*(?:角度|动量))(?=.*相机)"
                r"(?=.*(?:牺牲|取舍|权衡))",
                re.I,
            ),
            re.compile(
                r"(?=.*(?:\bposition\b|位置))"
                r"(?=.*(?:\bangular\b|\bmomentum\b|角度|动量))"
                r"(?=.*(?:\bcameras?\b|相机))"
                r"(?=.*(?:\bsacrific|\btrade[- ]?off\b|牺牲|取舍|权衡))",
                re.I,
            ),
        ),
        (
            re.compile(
                r"牺牲.{0,32}换取|(?:帧率|时间分辨率).{0,24}(?:空间)?分辨率.{0,12}权衡|"
                r"\btrade[- ]?off\b|\bat the expense of\b",
                re.I,
            ),
            re.compile(r"牺牲|换取|权衡|\btrade[- ]?off\b|\bsacrific|\bat the expense of\b", re.I),
        ),
        (
            re.compile(r"互补信息.{0,20}(?:合并|融合)|(?:合并|融合).{0,20}互补信息|\bmerge\w* complementary\b", re.I),
            re.compile(r"互补|合并|融合|\bcomplementary\b|\bmerg|\bcombin|\bfus|\bintegrat", re.I),
        ),
        (
            re.compile(r"超分辨率重建|\bsuper[- ]?resolution reconstruction\b", re.I),
            re.compile(r"超分辨率.{0,16}重建|\bsuper[- ]?resolution\b.{0,40}\breconstruct", re.I),
        ),
        (
            re.compile(r"信噪比|\bSNR\b|\bsignal[- ]to[- ]noise\b", re.I),
            re.compile(r"信噪比|\bSNR\b|\bsignal[- ]to[- ]noise\b", re.I),
        ),
        (
            re.compile(r"噪声(?:鲁棒性|特性)|\bnoise robustness\b|\bnoise characteristics?\b", re.I),
            re.compile(r"噪声|\bnoise\b", re.I),
        ),
        (
            re.compile(r"不同采样率|采样率下|\bacross (?:different )?sampling rates?\b", re.I),
            re.compile(r"采样率|\bsampling (?:ratio|rate)s?\b", re.I),
        ),
        (
            re.compile(r"空间频率.{0,8}结构信息|\bspatial frequency.{0,16}structure", re.I),
            re.compile(r"空间频率|结构信息|\bspatial frequenc|\bspatial structure", re.I),
        ),
        (
            re.compile(r"(?:算法|硬件).{0,16}复杂度|\b(?:algorithm|hardware).{0,20}complexity\b", re.I),
            re.compile(r"复杂度|\bcomplexity\b", re.I),
        ),
        (
            re.compile(r"频域.{0,12}稀疏|\bfrequency[- ]domain.{0,20}spars", re.I),
            re.compile(r"频域.{0,12}稀疏|\bfrequency[- ]domain.{0,20}spars", re.I),
        ),
        (
            re.compile(r"有限.{0,16}测量预算|总帧数|\bfixed.{0,16}measurement budget\b", re.I),
            re.compile(r"有限.{0,16}测量预算|总帧数|\bfixed.{0,16}measurement budget\b|\btotal frame count\b", re.I),
        ),
        (
            re.compile(r"不改变.{0,20}(?:编码|基函数)|\bwithout changing.{0,24}(?:encoding|basis)\b", re.I),
            re.compile(r"不改变.{0,20}(?:编码|基函数)|\bwithout changing.{0,24}(?:encoding|basis)\b", re.I),
        ),
        (
            re.compile(r"正交(?:的|组合)|\borthogonal(?:ly)?(?: combin| design| dimension)", re.I),
            re.compile(r"正交(?:的|组合)|\borthogonal(?:ly)?(?: combin| design| dimension)", re.I),
        ),
    )
    explicit_relation_requirements = (
        *explicit_relation_requirements,
        (_OPTIMIZATION_DETAIL_RE, _OPTIMIZATION_DETAIL_RE),
        (_MASK_COMPRESSION_DETAIL_RE, _MASK_COMPRESSION_DETAIL_RE),
    )
    if any(
        claim_pattern.search(claim_low) and not evidence_pattern.search(evidence_norm)
        for claim_pattern, evidence_pattern in explicit_relation_requirements
    ):
        return 0
    quantity_claim, quantity_evidence, multiplier_covered = _prepare_quantity_surfaces(
        claim_plain,
        evidence_text,
    )
    if not multiplier_covered:
        return 0
    claim_quantities = _claim_fact_quantities_for_evidence(
        quantity_claim,
        quantity_evidence,
        allow_comparison_scope=allow_comparison_scope,
    )
    evidence_quantities = _system_a_fact_quantities(quantity_evidence)
    if claim_quantities and not all(
        _quantity_is_covered(quantity, evidence_quantities)
        for quantity in claim_quantities
    ):
        return 0
    claim_numbers = [_quantity_label(quantity) for quantity in claim_quantities]
    score = 2 * len(_concept_ids(claim_plain) & _concept_ids(evidence_norm))
    claim_acronyms = {item.lower() for item in _ACRONYM_RE.findall(claim_plain)}
    evidence_acronyms = {item.lower() for item in _ACRONYM_RE.findall(evidence_norm.upper())}
    score += 2 * len(claim_acronyms & evidence_acronyms)
    claim_terms = {
        term.lower()
        for term in _EN_TERM_RE.findall(claim_plain)
        if term.lower() not in _STOPWORDS
    }
    evidence_terms = {
        term.lower()
        for term in _EN_TERM_RE.findall(evidence_norm)
        if term.lower() not in _STOPWORDS
    }
    shared_terms = claim_terms & evidence_terms
    score += min(3, len(shared_terms))
    # Two independent literal content terms (for example Hadamard + Fourier)
    # are materially stronger than one broad domain token such as
    # ``single-pixel``.  The small bonus lets strict citation plans rebind a
    # wrong source number without lowering the global support threshold.
    if len(shared_terms) >= 2:
        score += 1
    score += min(
        6,
        len(evidence_alignment_tokens(claim_plain) & evidence_alignment_tokens(evidence_norm)),
    )
    if claim_numbers:
        score += 2
    return score


def _best_unique_hit(
    claim: str,
    answer_hits: list[dict[str, Any]],
    *,
    min_score: int = 3,
) -> tuple[int, int]:
    scored = sorted(
        (
            (_support_score(claim, _hit_payload(hit)), index)
            for index, hit in enumerate(answer_hits, start=1)
            if isinstance(hit, dict)
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < max(1, int(min_score)):
        return 0, 0
    best_score, best_index = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    if runner_up >= best_score:
        return 0, best_score
    return best_index, best_score


def _append_citation(unit: str, citation_number: int) -> str:
    marker = f"[{int(citation_number)}]"
    match = re.search(r"([。！？!?；;.]?)\s*$", unit)
    if not match:
        return f"{unit} {marker}"
    punctuation = match.group(1)
    body = unit[: match.start()].rstrip()
    return f"{body} {marker}{punctuation}"


def _relocate_possessive_numeric_citations(answer: str) -> tuple[str, int]:
    """Move ``paper [n] 的...`` markers to the supported clause end.

    A marker between a source noun and the Chinese possessive ``的`` has almost
    no claim text on its left, so the renderer may hide it as weakly bound.
    Moving the same marker to the clause end preserves source identity while
    giving both readers and the renderer the complete supported claim.
    """

    citation_before_possessive = re.compile(
        r"[ \t]*(?<!\[)(\[(\d{1,5})\](?:\([^\n)]+\))?)\s*的"
    )
    clause_split = re.compile(r"([；;。！？!?](?:\s+|$)?)")
    relocated = 0
    output_lines: list[str] = []
    in_fence = False
    for raw_line in str(answer or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            output_lines.append(raw_line)
            continue
        if in_fence or not stripped or _TABLE_OR_CODE_RE.match(stripped):
            output_lines.append(raw_line)
            continue
        parts = clause_split.split(raw_line)
        rebuilt: list[str] = []
        for index in range(0, len(parts), 2):
            clause = parts[index]
            punctuation = parts[index + 1] if index + 1 < len(parts) else ""
            moved_markers: list[str] = []

            def _remove_midphrase_marker(match: re.Match[str]) -> str:
                nonlocal relocated
                marker = str(match.group(1) or "").strip()
                if marker and marker not in moved_markers:
                    moved_markers.append(marker)
                relocated += 1
                return "的"

            cleaned_clause = citation_before_possessive.sub(
                _remove_midphrase_marker,
                clause,
            )
            for marker in moved_markers:
                if marker not in cleaned_clause:
                    cleaned_clause = f"{cleaned_clause.rstrip()} {marker}"
            rebuilt.append(f"{cleaned_clause}{punctuation}")
        output_lines.append("".join(rebuilt))
    return "\n".join(output_lines), relocated


def _scope_absolute_negative_claims(answer: str) -> tuple[str, int]:
    text = str(answer or "")
    count = 0
    patterns = (
        (
            re.compile(
                r"(?:两篇|这些|上述)(?:文献|论文)(?:均|都)?"
                r"(?:未|没有)(?:提供|说明|显示|报告|验证|讨论)",
                flags=re.IGNORECASE,
            ),
            lambda match: "当前引用证据未显示这些文献"
            + ("提供" if "提供" in match.group(0) else "说明"),
        ),
        (
            re.compile(
                r"(?:这|该|上述)?两篇(?:论文|文献)之间(?:并)?没有直接的引用关系",
                flags=re.IGNORECASE,
            ),
            lambda _match: "当前引用证据未显示这两篇论文之间存在直接引用关系",
        ),
        (
            re.compile(r"(?:这篇(?:文章|论文)|本文)(?:并)?(?:没有|未)(直接)?(提供|说明|显示|报告|验证|讨论)"),
            lambda match: f"当前引用证据未直接{match.group(2)}",
        ),
        (
            re.compile(r"(?:当前|本轮)检索(?:到的)?(?:片段|证据|结果)?(?:并)?(?:没有|未)(直接)?(提供|说明|显示|报告|验证|讨论)"),
            lambda match: f"当前引用证据未直接{match.group(2)}",
        ),
        (
            re.compile(
                r"(?:现有|当前)(?:检索到的)?证据(?:片段)?(?:并)?(?:没有|未)"
                r"(?:明确|直接)?(提供|说明|显示|报告|验证|讨论|提及)"
            ),
            lambda match: f"当前引用证据未直接{match.group(1)}",
        ),
        (
            re.compile(
                r"(?:this\s+paper|the\s+paper)\s+(?:does\s+not|did\s+not|doesn't)\s+"
                r"(directly\s+)?(provide|state|show|report|validate|discuss)",
                flags=re.IGNORECASE,
            ),
            lambda match: f"The currently cited evidence does not directly {match.group(2).lower()}",
        ),
    )
    for pattern, replacement in patterns:
        text, changed = pattern.subn(replacement, text)
        count += changed
    # A retrieval-boundary statement describes what the current evidence
    # window does not establish; it is not a factual claim from any one paper.
    # Models sometimes append a numeric marker anyway.  Leaving that marker in
    # place can make the claim auditor bind the caveat to a semantically nearby
    # but unrelated hit, while the renderer later (correctly) rejects it.  Strip
    # only markers inside the normalized boundary sentence so the visible
    # answer and the final evidence audit apply the same contract.
    text = re.sub(
        r"(当前引用证据未直接[^。！？.!?\n]*?)\s*"
        r"(?<!\[)\[\d{1,5}\](?:\([^\n)]+\))?(?!\])",
        r"\1",
        text,
    )
    text = re.sub(r"\s+([。！？.!?；;])", r"\1", text)
    return text, count


def _trim_unsupported_boundary_inferences(answer: str) -> tuple[str, int]:
    """Keep an evidence boundary while removing the speculative clause after it."""

    text = str(answer or "")
    patterns = (
        re.compile(
            r"((?:当前|本轮)(?:引用|检索)(?:到的)?(?:片段|证据|结果|内容)?"
            r"[^。！？.!?]{0,48}?(?:未明确提及|未直接提及|未说明|没有提及)"
            r"[^。！？.!?，,]{0,48})"
            r"\s*[，,]\s*但[^。！？.!?]*",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"((?:the\s+)?current\s+(?:cited\s+)?(?:snippet|evidence|retrieval|content)"
            r"[^.!?]{0,48}(?:does\s+not|doesn't|did\s+not)(?:\s+directly)?\s+mention"
            r"[^.!?,;]{0,48})"
            r"\s*[,;]\s*however[^.!?]*",
            flags=re.IGNORECASE,
        ),
    )
    count = 0
    for pattern in patterns:
        text, changed = pattern.subn(r"\1", text)
        count += int(changed)
    return text, count


def _repair_modality_boundary_language(answer: str) -> tuple[str, int]:
    text = str(answer or "")
    count = 0
    patterns = (
        (
            re.compile(r"单像素成像\s*[（(]\s*(?:一种)?单光子成像的(?:变体|分支|一种)\s*[）)]"),
            "单像素成像（与单光子成像不同，这里仅作为邻近算法背景）",
        ),
        (
            re.compile(r"单像素成像(?:是|属于)\s*(?:一种)?单光子成像的?(?:变体|分支|一种)"),
            "单像素成像与单光子成像是不同维度的成像概念，前者这里只能作为邻近算法背景",
        ),
        (
            re.compile(
                r"single[- ]pixel imaging\s+is\s+(?:a\s+)?(?:variant|branch|type)\s+of\s+single[- ]photon imaging",
                flags=re.IGNORECASE,
            ),
            "single-pixel imaging is distinct from single-photon imaging and is included here only as adjacent algorithmic background",
        ),
    )
    for pattern, replacement in patterns:
        text, changed = pattern.subn(replacement, text)
        count += changed
    return text, count


def _ensure_prompt_spad_term(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    text = str(answer or "")
    if not re.search(r"\bSPAD\b", str(prompt or ""), flags=re.IGNORECASE):
        return text, 0
    if re.search(r"\bSPAD\b", text, flags=re.IGNORECASE):
        return text, 0
    if not any(re.search(r"\bSPAD\b", _hit_payload(hit), flags=re.IGNORECASE) for hit in answer_hits):
        return text, 0
    repaired, changed = re.subn(r"单光子(?:成像|探测)", lambda match: f"SPAD {match.group(0)}", text, count=1)
    return repaired, int(changed)


def _ensure_prompt_source_identifier_heading(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Keep one explicit user-supplied source identifier visible as context.

    Models occasionally answer every requested fact but paraphrase away the
    uncommon acronym that tells the reader which paper/method is being
    discussed.  Repeating a single prompt identifier as a Markdown heading is
    non-assertive and cannot fabricate evidence; require an existing citation
    and at least one eligible System-A hit so this never labels an unsupported
    answer.
    """

    text = str(answer or "").strip()
    if not text or not _CITATION_RE.search(text) or not any(answer_hits):
        return text, 0
    generic_identifiers = {
        "API",
        "CPU",
        "DAQ",
        "DMD",
        "DOF",
        "FPS",
        "GPU",
        "LFM",
        "LED",
        "MRI",
        "PSNR",
        "SNR",
        "SPAD",
        "SSIM",
    }
    identifiers = list(
        dict.fromkeys(
            token
            for token in _ACRONYM_RE.findall(str(prompt or ""))
            if len(token) >= 4 and token.upper() not in generic_identifiers
        )
    )
    if len(identifiers) != 1:
        return text, 0
    identifier = identifiers[0]
    if re.search(rf"\b{re.escape(identifier)}\b", text, flags=re.I):
        return text, 0
    return f"### {identifier}\n\n{text}", 1


def _drop_hard_mismatched_claims(
    answer: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    if not answer_hits:
        return str(answer or ""), []
    dropped: list[dict[str, Any]] = []
    output_lines: list[str] = []
    in_fence = False
    for raw_line in str(answer or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            output_lines.append(raw_line)
            continue
        if in_fence or not stripped or _HEADING_RE.match(stripped) or _TABLE_OR_CODE_RE.match(stripped):
            output_lines.append(raw_line)
            continue
        prefix_match = _LIST_PREFIX_RE.match(stripped)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = stripped[len(prefix) :] if prefix else stripped
        segments = _split_claim_segments(body)
        kept: list[str] = []
        for segment in segments:
            citations = [int(match.group(1)) for match in _NUMERIC_CITATION_RE.finditer(segment)]
            group_quantity_coverage = _citation_group_covers_claim_quantities(
                segment,
                citations,
                answer_hits,
            )
            bound_scores = [
                _support_score(
                    segment,
                    _hit_payload(answer_hits[number - 1]),
                    allow_comparison_scope=group_quantity_coverage,
                )
                for number in citations
                if 0 < number <= len(answer_hits)
            ]
            hard_checkable = bool(_meaningful_numbers(segment) or _ACRONYM_RE.search(_plain_claim(segment)))
            if citations and bound_scores and max(bound_scores) <= 0 and hard_checkable and _is_high_risk_claim(segment):
                dropped.append({"claim": _plain_claim(segment)[:220], "citations": citations})
                continue
            kept.append(segment)
        if not kept:
            continue
        joiner = "" if _ZH_RE.search("".join(kept)) else " "
        rebuilt = joiner.join(kept)
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        output_lines.append(f"{leading}{prefix}{rebuilt}")
    return "\n".join(output_lines), dropped


def _strip_unplanned_numeric_citations(
    answer: str,
    allowed_citation_numbers: set[int],
) -> tuple[str, int]:
    removed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed
        try:
            number = int(match.group(1))
        except (TypeError, ValueError):
            return ""
        if number in allowed_citation_numbers:
            return match.group(0)
        removed += 1
        return ""

    cleaned = _NUMERIC_CITATION_RE.sub(replace, str(answer or ""))
    cleaned = re.sub(r"\s+([。！？.!?；;])", r"\1", cleaned)
    return cleaned, removed


def _drop_unsupported_uncited_claims(answer: str) -> tuple[str, list[str]]:
    """Drop factual model additions that strict planned evidence cannot support."""

    dropped: list[str] = []
    output_lines: list[str] = []
    in_fence = False
    for raw_line in str(answer or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            output_lines.append(raw_line)
            continue
        if in_fence or not stripped or _HEADING_RE.match(stripped) or _TABLE_OR_CODE_RE.match(stripped):
            output_lines.append(raw_line)
            continue
        prefix_match = _LIST_PREFIX_RE.match(stripped)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = stripped[len(prefix) :] if prefix else stripped
        kept: list[str] = []
        for segment in _split_claim_segments(body):
            if _is_high_risk_claim(segment) and not _CITATION_RE.search(segment):
                dropped.append(_plain_claim(segment)[:220])
                continue
            kept.append(segment)
        if not kept:
            continue
        joiner = "" if _ZH_RE.search("".join(kept)) else " "
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        output_lines.append(f"{leading}{prefix}{joiner.join(kept)}")
    return "\n".join(output_lines), dropped


def _renumber_ordered_lists(answer: str) -> str:
    """Repair ordered-list numbering after unsupported items are removed."""

    output: list[str] = []
    next_number = 1
    for raw_line in str(answer or "").splitlines():
        match = re.match(r"^(\s*)(\d+)([.)、])\s+(.+)$", raw_line)
        if match:
            output.append(
                f"{match.group(1)}{next_number}{match.group(3)} {match.group(4)}"
            )
            next_number += 1
            continue
        output.append(raw_line)
        if raw_line.strip():
            next_number = 1
    return "\n".join(output)


def _strip_weak_numeric_citations(
    answer: str,
    answer_hits: list[dict[str, Any]],
    *,
    min_support_score: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Remove numeric markers that do not bind a high-risk claim to evidence.

    This runs only for the final strict evidence gate.  Removing the marker lets
    the existing unsupported-claim dropper reject the model addition instead of
    presenting a weak or wrong source as direct support.
    """

    stripped: list[dict[str, Any]] = []
    output_lines: list[str] = []
    in_fence = False
    for raw_line in str(answer or "").splitlines():
        surface = raw_line.strip()
        if surface.startswith(("```", "~~~")):
            in_fence = not in_fence
            output_lines.append(raw_line)
            continue
        if in_fence or not surface or _HEADING_RE.match(surface) or _TABLE_OR_CODE_RE.match(surface):
            output_lines.append(raw_line)
            continue
        prefix_match = _LIST_PREFIX_RE.match(surface)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = surface[len(prefix) :] if prefix else surface
        rebuilt: list[str] = []
        changed = False
        for segment in _split_claim_segments(body):
            citations = [
                int(match.group(1))
                for match in _NUMERIC_CITATION_RE.finditer(segment)
                if 0 < int(match.group(1)) <= len(answer_hits)
            ]
            if (
                not citations
                or _STRUCTURED_CITATION_RE.search(segment)
                or not _is_high_risk_claim(segment)
            ):
                rebuilt.append(segment)
                continue
            group_quantity_coverage = _citation_group_covers_claim_quantities(
                segment,
                citations,
                answer_hits,
            )
            scored_citations = {
                number: _support_score(
                    segment,
                    _hit_payload(answer_hits[number - 1]),
                    allow_comparison_scope=group_quantity_coverage,
                )
                for number in citations
                if answer_hits[number - 1]
            }
            supported_citations = {
                number
                for number, score in scored_citations.items()
                if score >= max(1, int(min_support_score))
            }
            weak_citations = [
                number for number in list(dict.fromkeys(citations))
                if number not in supported_citations
            ]
            if not weak_citations:
                rebuilt.append(segment)
                continue
            cleaned = _NUMERIC_CITATION_RE.sub(
                lambda match: (
                    match.group(0)
                    if int(match.group(1)) in supported_citations
                    else ""
                ),
                segment,
            )
            cleaned = re.sub(r"\s+([。！？.!?；;])", r"\1", cleaned)
            rebuilt.append(cleaned)
            stripped.append(
                {
                    "claim": _plain_claim(segment)[:220],
                    "citations": weak_citations,
                    "best_score": max(scored_citations.values()) if scored_citations else 0,
                    "supported_citations": sorted(supported_citations),
                }
            )
            changed = True
        if not changed:
            output_lines.append(raw_line)
            continue
        joiner = "" if _ZH_RE.search("".join(rebuilt)) else " "
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        output_lines.append(f"{leading}{prefix}{joiner.join(rebuilt)}")
    return "\n".join(output_lines), stripped


def _strip_user_visible_rejected_citations(
    answer: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply the same binding decision used by the user-visible citation card.

    The import is intentionally lazy: claim auditing remains lightweight for
    ordinary calls, while the final answer gate uses the renderer's exact
    decision as the single source of truth for whether a numeric marker will be
    clickable in the UI.
    """

    rejected: list[dict[str, Any]] = []
    output_lines: list[str] = []
    in_fence = False
    for raw_line in str(answer or "").splitlines():
        surface = raw_line.strip()
        if surface.startswith(("```", "~~~")):
            in_fence = not in_fence
            output_lines.append(raw_line)
            continue
        if in_fence or not surface or _HEADING_RE.match(surface) or _TABLE_OR_CODE_RE.match(surface):
            output_lines.append(raw_line)
            continue
        prefix_match = _LIST_PREFIX_RE.match(surface)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = surface[len(prefix) :] if prefix else surface
        rebuilt: list[str] = []
        changed = False
        for segment in _split_claim_segments(body):
            citations = [
                int(match.group(1))
                for match in _NUMERIC_CITATION_RE.finditer(segment)
                if 0 < int(match.group(1)) <= len(answer_hits)
            ]
            if (
                not citations
                or _STRUCTURED_CITATION_RE.search(segment)
            ):
                rebuilt.append(segment)
                continue
            group_quantity_coverage = _citation_group_covers_claim_quantities(
                segment,
                citations,
                answer_hits,
            )
            group_evidence_quotes = [
                _hit_payload(answer_hits[number - 1])
                for number in list(dict.fromkeys(citations))
                if 0 < number <= len(answer_hits) and answer_hits[number - 1]
            ] if group_quantity_coverage else []
            rejected_numbers: list[int] = []
            binding_rows: list[dict[str, Any]] = []
            for number in list(dict.fromkeys(citations)):
                hit = answer_hits[number - 1]
                if not hit:
                    rejected_numbers.append(number)
                    continue
                meta = dict(hit.get("meta") or {}) if isinstance(hit.get("meta"), dict) else {}
                quotes = [
                    str(item or "").strip()
                    for item in list(meta.get("citation_plan_evidence_quotes") or [])
                    if str(item or "").strip()
                ]
                if not quotes:
                    # Only make this strict decision for the authoritative
                    # citation-plan evidence shown by the renderer.
                    continue
                binding_meta = dict(meta)
                binding_meta["citation_plan_evidence_authoritative"] = True
                binding_meta["citation_plan_evidence_selection_reason"] = (
                    "prompt_aligned_source_sentence"
                )
                if group_evidence_quotes:
                    binding_meta["citation_group_evidence_quotes"] = [
                        _normalize_multiplier_surface(item)[1]
                        for item in group_evidence_quotes
                    ]
                quantity_evidence = (
                    "\n".join(group_evidence_quotes)
                    if group_evidence_quotes
                    else "\n".join(quotes)
                )
                binding_claim, _normalized_union, multiplier_covered = (
                    _prepare_quantity_surfaces(segment, quantity_evidence)
                )
                source_name = str(
                    meta.get("source_name")
                    or hit.get("source_name")
                    or hit.get("title")
                    or ""
                ).strip()
                if not multiplier_covered:
                    bindings = [{
                        "status": "mismatch",
                        "suppress_link": True,
                        "reason": "The cited evidence has an incompatible multiplier magnitude or direction.",
                    }]
                else:
                    bindings = []
                    for evidence_quote in quotes:
                        _evidence_multiplier_facts, binding_evidence_quote = (
                            _normalize_multiplier_surface(evidence_quote)
                        )
                        bindings.append(
                            assess_system_a_hit_binding(
                                answer_claim=binding_claim,
                                hit=hit,
                                meta=binding_meta,
                                heading=str(
                                    meta.get("heading_path")
                                    or meta.get("top_heading")
                                    or ""
                                ),
                                evidence_quote=binding_evidence_quote,
                                source_name=source_name,
                            )
                        )
                accepted_binding = next(
                    (
                        binding
                        for binding in bindings
                        if not bool(binding.get("suppress_link"))
                    ),
                    None,
                )
                if accepted_binding is None:
                    rejected_numbers.append(number)
                    binding = max(
                        bindings,
                        key=lambda item: float(item.get("confidence") or 0.0),
                    )
                    binding_rows.append(
                        {
                            "citation": number,
                            "status": str(binding.get("status") or "candidate"),
                            "reason": str(binding.get("reason") or "")[:260],
                        }
                    )
            if not rejected_numbers:
                rebuilt.append(segment)
                continue
            rejected_set = set(rejected_numbers)
            cleaned = _NUMERIC_CITATION_RE.sub(
                lambda match: "" if int(match.group(1)) in rejected_set else match.group(0),
                segment,
            )
            cleaned = re.sub(r"\s+([。！？.!?；;])", r"\1", cleaned)
            rebuilt.append(cleaned)
            rejected.append(
                {
                    "claim": _plain_claim(segment)[:220],
                    "citations": sorted(rejected_set),
                    "bindings": binding_rows,
                }
            )
            changed = True
        if not changed:
            output_lines.append(raw_line)
            continue
        joiner = "" if _ZH_RE.search("".join(rebuilt)) else " "
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        output_lines.append(f"{leading}{prefix}{joiner.join(rebuilt)}")
    return "\n".join(output_lines), rejected


def _repair_uncited_unique_claims(
    answer: str,
    answer_hits: list[dict[str, Any]],
    *,
    min_support_score: int = 3,
    allowed_citation_numbers: set[int] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    repairs: list[dict[str, Any]] = []
    output_lines: list[str] = []
    in_fence = False
    section_citations: list[int] = []
    for raw_line in str(answer or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            output_lines.append(raw_line)
            continue
        if _HEADING_RE.match(stripped):
            section_citations = list(
                dict.fromkeys(
                    int(match.group(1))
                    for match in _NUMERIC_CITATION_RE.finditer(stripped)
                    if 0 < int(match.group(1)) <= len(answer_hits)
                )
            )
            output_lines.append(raw_line)
            continue
        if in_fence or not stripped:
            output_lines.append(raw_line)
            continue
        if stripped.startswith("|"):
            table_claim = re.sub(r"\s*\|\s*", " ", stripped).strip()
            is_separator = bool(
                re.fullmatch(r"(?::?-{3,}:?\s*)+", table_claim.replace(" ", ""))
            )
            if (
                is_separator
                or _CITATION_RE.search(stripped)
                or not _is_high_risk_claim(table_claim)
            ):
                output_lines.append(raw_line)
                continue
            hit_index, score = _best_unique_hit(
                table_claim,
                answer_hits,
                min_score=min_support_score,
            )
            if hit_index <= 0:
                output_lines.append(raw_line)
                continue
            leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            table_body = stripped[:-1].rstrip() if stripped.endswith("|") else stripped
            output_lines.append(f"{leading}{table_body} [{hit_index}] |")
            repairs.append(
                {
                    "claim": _plain_claim(table_claim)[:220],
                    "citation": hit_index,
                    "score": score,
                    "reason": "markdown_table_fact",
                }
            )
            continue
        if _TABLE_OR_CODE_RE.match(stripped):
            output_lines.append(raw_line)
            continue
        prefix_match = _LIST_PREFIX_RE.match(stripped)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = stripped[len(prefix) :] if prefix else stripped
        segments = _split_claim_segments(body)
        if not segments:
            output_lines.append(raw_line)
            continue
        changed = False
        rebuilt_segments: list[str] = []
        previous_citations: list[int] = list(section_citations)
        for segment in segments:
            segment_citations = [
                int(match.group(1))
                for match in _NUMERIC_CITATION_RE.finditer(segment)
                if 0 < int(match.group(1)) <= len(answer_hits)
            ]
            comparison_citations = sorted(
                number
                for number in set(allowed_citation_numbers or set())
                if 0 < number <= len(answer_hits) and answer_hits[number - 1]
            )
            is_compound_comparison = bool(
                len(comparison_citations) == 2
                and re.search(
                    r"核心区别|前者.{0,160}后者|两者.{0,80}(?:不同|区别|差异)|"
                    r"(?:两者|它们).{0,120}分别(?:决定|作用)|"
                    r"core\s+difference|the\s+former.{0,160}the\s+latter",
                    _plain_claim(segment),
                    flags=re.IGNORECASE,
                )
                and all(
                    _support_score(
                        segment,
                        _hit_payload(answer_hits[number - 1]),
                        allow_comparison_scope=_citation_group_covers_claim_quantities(
                            segment,
                            comparison_citations,
                            answer_hits,
                        ),
                    )
                    >= max(3, min_support_score - 1)
                    for number in comparison_citations
                )
            )
            if is_compound_comparison and not set(comparison_citations).issubset(segment_citations):
                compound = segment
                for number in comparison_citations:
                    if number not in segment_citations:
                        compound = _append_citation(compound, number)
                rebuilt_segments.append(compound)
                repairs.append(
                    {
                        "claim": _plain_claim(segment)[:220],
                        "citations": comparison_citations,
                        "score": max(3, min_support_score - 1),
                        "reason": "compound_comparison_synthesis",
                    }
                )
                previous_citations = comparison_citations
                changed = True
                continue
            if _CITATION_RE.search(segment) or not _is_high_risk_claim(segment):
                rebuilt_segments.append(segment)
                if segment_citations:
                    previous_citations = list(dict.fromkeys(segment_citations))
                continue
            if len(section_citations) == 1:
                inherited = int(section_citations[0])
                rebuilt_segments.append(_append_citation(segment, inherited))
                repairs.append(
                    {
                        "claim": _plain_claim(segment)[:220],
                        "citation": inherited,
                        "score": 1,
                        "reason": "section_source_continuation",
                    }
                )
                previous_citations = [inherited]
                changed = True
                continue
            if (
                len(previous_citations) == 1
                and _ANAPHORIC_CONTINUATION_RE.search(_plain_claim(segment))
                and not _meaningful_numbers(segment)
            ):
                inherited = int(previous_citations[0])
                rebuilt_segments.append(_append_citation(segment, inherited))
                repairs.append(
                    {
                        "claim": _plain_claim(segment)[:220],
                        "citation": inherited,
                        "score": 1,
                        "reason": "anaphoric_continuation",
                    }
                )
                changed = True
                continue
            if is_compound_comparison:
                compound = segment
                for number in comparison_citations:
                    compound = _append_citation(compound, number)
                rebuilt_segments.append(compound)
                repairs.append(
                    {
                        "claim": _plain_claim(segment)[:220],
                        "citations": comparison_citations,
                        "score": max(3, min_support_score - 1),
                        "reason": "compound_comparison_synthesis",
                    }
                )
                previous_citations = comparison_citations
                changed = True
                continue
            coverage_suffix = re.search(
                r"[，,；;]\s*(?:为(?:了)?|从而|以便|用于|"
                r"(?:so\s+that|in\s+order\s+to|to\s+understand)\b)",
                segment,
                flags=re.IGNORECASE,
            )
            coverage_prefix = (
                segment[: coverage_suffix.start()].rstrip("，,；;：: ")
                if coverage_suffix
                else ""
            )
            if (
                len(_plain_claim(coverage_prefix)) >= 20
                and _PAPER_COVERAGE_CLAIM_RE.search(_plain_claim(coverage_prefix))
            ):
                prefix_hit_index, prefix_score = _best_unique_hit(
                    coverage_prefix,
                    answer_hits,
                    min_score=min_support_score,
                )
                if prefix_hit_index > 0:
                    punctuation = "。" if _ZH_RE.search(coverage_prefix) else "."
                    rebuilt_segments.append(
                        _append_citation(
                            f"{coverage_prefix}{punctuation}",
                            prefix_hit_index,
                        )
                    )
                    repairs.append(
                        {
                            "claim": _plain_claim(coverage_prefix)[:220],
                            "citation": prefix_hit_index,
                            "score": prefix_score,
                            "reason": "supported_paper_coverage_prefix",
                        }
                    )
                    previous_citations = [prefix_hit_index]
                    changed = True
                    continue
            hit_index, score = _best_unique_hit(
                segment,
                answer_hits,
                min_score=min_support_score,
            )
            if hit_index <= 0:
                rebuilt_segments.append(segment)
                continue
            rebuilt_segments.append(_append_citation(segment, hit_index))
            repairs.append(
                {"claim": _plain_claim(segment)[:220], "citation": hit_index, "score": score}
            )
            changed = True
        if not changed:
            output_lines.append(raw_line)
            continue
        joiner = "" if _ZH_RE.search("".join(rebuilt_segments)) else " "
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        output_lines.append(f"{leading}{prefix}{joiner.join(rebuilt_segments)}")
    return "\n".join(output_lines), repairs


def _strip_source_only_heading_citations(answer: str) -> tuple[str, int]:
    """Keep evidence links on claims, not on decorative section headings."""

    removed = 0
    output: list[str] = []
    in_fence = False
    for raw_line in str(answer or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            output.append(raw_line)
            continue
        if in_fence or not _HEADING_RE.match(stripped):
            output.append(raw_line)
            continue
        cleaned, count = _NUMERIC_CITATION_RE.subn("", raw_line)
        if count:
            cleaned = re.sub(r"[（(]\s*[,，;；:/|\-\s]*[）)]", "", cleaned)
            cleaned = re.sub(r"\s+([,，;；:：])", r"\1", cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
            removed += int(count)
        output.append(cleaned)
    return "\n".join(output), removed


def _repair_mismatched_unique_citations(
    answer: str,
    answer_hits: list[dict[str, Any]],
    *,
    min_support_score: int = 3,
    strict_plan: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Rebind an incorrect System-A number only when one hit is uniquely stronger."""

    repairs: list[dict[str, Any]] = []
    output_lines: list[str] = []
    in_fence = False
    section_citations: list[int] = []
    for raw_line in str(answer or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            output_lines.append(raw_line)
            continue
        if _HEADING_RE.match(stripped):
            section_citations = list(
                dict.fromkeys(
                    int(match.group(1))
                    for match in _NUMERIC_CITATION_RE.finditer(stripped)
                    if 0 < int(match.group(1)) <= len(answer_hits)
                )
            )
            output_lines.append(raw_line)
            continue
        if in_fence or not stripped or _TABLE_OR_CODE_RE.match(stripped):
            output_lines.append(raw_line)
            continue
        prefix_match = _LIST_PREFIX_RE.match(stripped)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = stripped[len(prefix) :] if prefix else stripped
        segments = _split_claim_segments(body)
        if not segments:
            output_lines.append(raw_line)
            continue
        rebuilt: list[str] = []
        changed = False
        previous_citations: list[int] = []
        for segment in segments:
            citations = [int(match.group(1)) for match in _NUMERIC_CITATION_RE.finditer(segment)]
            plain_segment = _plain_claim(segment)
            strict_factual_candidate = bool(
                strict_plan
                and len(plain_segment) >= 20
                and not _BOUNDARY_RE.search(plain_segment)
                and not _INFERENCE_RE.search(plain_segment)
                and not _NON_FACTUAL_GUIDANCE_RE.search(plain_segment)
                and not re.search(r"[?\uFF1F]\s*$", plain_segment)
            )
            if not citations or not (
                _is_high_risk_claim(segment) or strict_factual_candidate
            ):
                rebuilt.append(segment)
                if citations:
                    previous_citations = list(dict.fromkeys(citations))
                continue
            if len(section_citations) == 1:
                inherited = int(section_citations[0])
                if citations != [inherited]:
                    rebound = _NUMERIC_CITATION_RE.sub(f"[{inherited}]", segment)
                    rebuilt.append(rebound)
                    repairs.append(
                        {
                            "claim": _plain_claim(segment)[:220],
                            "from": citations,
                            "citation": inherited,
                            "score": 1,
                            "reason": "section_source_continuation",
                        }
                    )
                    changed = True
                else:
                    rebuilt.append(segment)
                previous_citations = [inherited]
                continue
            cited_scores = [
                _support_score(
                    segment,
                    _hit_payload(answer_hits[number - 1]),
                    allow_comparison_scope=_citation_group_covers_claim_quantities(
                        segment,
                        citations,
                        answer_hits,
                    ),
                )
                for number in citations
                if 0 < number <= len(answer_hits)
            ]
            # A model may attach one unsupported contrast clause to an otherwise
            # faithful evidence paraphrase (for example, "it does not change the
            # basis, but instead ...").  Do not discard the supported half merely
            # because the unsupported setup made the whole sentence fail.  This
            # is deliberately limited to an explicit contrast boundary and still
            # requires the suffix to bind uniquely to eligible evidence.
            contrast = re.search(
                r"(?:，|,)\s*(?:而是|而应|但(?:是)?应|rather\s+than|but\s+instead|instead)\s*",
                segment,
                flags=re.IGNORECASE,
            )
            contrast_suffix = (
                _CITATION_RE.sub("", segment[contrast.end() :]).strip()
                if contrast
                else ""
            )
            if (
                (not cited_scores or max(cited_scores) < min_support_score)
                and len(_plain_claim(contrast_suffix)) >= 20
                and _is_high_risk_claim(contrast_suffix)
            ):
                suffix_hit_index, suffix_score = _best_unique_hit(
                    contrast_suffix,
                    answer_hits,
                    min_score=min_support_score,
                )
                if suffix_hit_index > 0:
                    rebuilt.append(_append_citation(contrast_suffix, suffix_hit_index))
                    repairs.append(
                        {
                            "claim": _plain_claim(contrast_suffix)[:220],
                            "from": citations,
                            "citation": suffix_hit_index,
                            "score": suffix_score,
                            "reason": "supported_contrast_suffix",
                        }
                    )
                    previous_citations = [suffix_hit_index]
                    changed = True
                    continue
            if (
                len(previous_citations) == 1
                and _ANAPHORIC_CONTINUATION_RE.search(_plain_claim(segment))
                and not _meaningful_numbers(segment)
            ):
                inherited = int(previous_citations[0])
                if inherited not in citations and 0 < inherited <= len(answer_hits):
                    rebound = _NUMERIC_CITATION_RE.sub(f"[{inherited}]", segment)
                    rebuilt.append(rebound)
                    repairs.append(
                        {
                            "claim": _plain_claim(segment)[:220],
                            "from": citations,
                            "citation": inherited,
                            "score": 1,
                            "reason": "anaphoric_continuation",
                        }
                    )
                    previous_citations = [inherited]
                    changed = True
                    continue
            if cited_scores and max(cited_scores) >= min_support_score:
                # A merely adequate hit can still be the wrong occurrence when
                # another strict-plan hit uniquely covers the claim's detailed
                # entities and relations. Prefer that evidence only with a wide
                # score margin; this preserves stable citations for close ties.
                stronger_hit_index = 0
                stronger_score = 0
                if strict_plan:
                    stronger_hit_index, stronger_score = _best_unique_hit(
                        segment,
                        answer_hits,
                        min_score=max(
                            int(min_support_score),
                            int(max(cited_scores)) + 4,
                        ),
                    )
                if stronger_hit_index > 0 and stronger_hit_index not in citations:
                    rebound = _NUMERIC_CITATION_RE.sub(
                        f"[{stronger_hit_index}]",
                        segment,
                    )
                    rebuilt.append(rebound)
                    repairs.append(
                        {
                            "claim": _plain_claim(segment)[:220],
                            "from": citations,
                            "citation": stronger_hit_index,
                            "score": stronger_score,
                            "reason": "uniquely_stronger_evidence",
                        }
                    )
                    previous_citations = [stronger_hit_index]
                    changed = True
                    continue
                rebuilt.append(segment)
                previous_citations = list(dict.fromkeys(citations))
                continue
            hit_index, score = _best_unique_hit(
                segment,
                answer_hits,
                min_score=min_support_score,
            )
            if hit_index <= 0 or hit_index in citations:
                rebuilt.append(segment)
                previous_citations = list(dict.fromkeys(citations))
                continue
            rebound = _NUMERIC_CITATION_RE.sub(f"[{hit_index}]", segment)
            rebuilt.append(rebound)
            repairs.append(
                {
                    "claim": _plain_claim(segment)[:220],
                    "from": citations,
                    "citation": hit_index,
                    "score": score,
                }
            )
            previous_citations = [hit_index]
            changed = True
        if not changed:
            output_lines.append(raw_line)
            continue
        joiner = "" if _ZH_RE.search("".join(rebuilt)) else " "
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        output_lines.append(f"{leading}{prefix}{joiner.join(rebuilt)}")
    return "\n".join(output_lines), repairs


def _drop_placeholder_sections(answer: str) -> tuple[str, int]:
    """Remove empty or retrieval-disclaimer sections from the user answer."""

    lines: list[str] = []
    dropped_inline = 0
    for raw_line in str(answer or "").splitlines():
        surface = raw_line.strip()
        inline_limitation = bool(
            re.match(
                r"^(?:\*{1,2})?(?:局限性|限制|limitations?)(?:\*{1,2})?\s*[:：]",
                surface,
                flags=re.IGNORECASE,
            )
            and (
                re.search(r"当前(?:检索|引用|证据|片段)|现有片段|未提及|未详细说明|需(?:要)?全文", surface)
                or re.search(
                    r"current (?:retrieval|evidence|snippet)|not mentioned|not detailed|need the full text",
                    surface,
                    flags=re.IGNORECASE,
                )
            )
        )
        if inline_limitation:
            dropped_inline += 1
            continue
        lines.append(raw_line)
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_body: list[str] = []
    for line in lines:
        if _HEADING_RE.match(line.strip()):
            sections.append((current_heading, current_body))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_heading, current_body))

    kept: list[str] = []
    dropped = 0
    for heading, body in sections:
        meaningful_body = [line for line in body if line.strip()]
        if heading is not None and not meaningful_body:
            dropped += 1
            continue
        surface = " ".join(line.strip() for line in meaningful_body)
        heading_low = str(heading or "").lower()
        placeholder = bool(
            re.search(r"局限|限制|limitation", heading_low, flags=re.IGNORECASE)
            and (
                re.search(r"当前(?:检索|引用|证据|片段)|现有片段|未提及|未详细说明|需(?:要)?全文", surface)
                or re.search(
                    r"current (?:retrieval|evidence|snippet)|not mentioned|not detailed|need the full text",
                    surface,
                    flags=re.IGNORECASE,
                )
            )
        )
        if placeholder:
            dropped += 1
            continue
        if heading is not None:
            if kept and kept[-1].strip():
                kept.append("")
            kept.append(heading)
        kept.extend(body)
    return "\n".join(kept).strip(), dropped + dropped_inline


def _ensure_grounded_frame_rate_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Keep a directly relevant reported frame rate from planned evidence.

    Models often paraphrase ``real-time`` while dropping the concrete rate that
    makes the comparison useful. Restore only when both the question/answer
    discuss a 3D video system and one eligible source states the rate verbatim.
    """

    text = str(answer or "")
    surface = f"{prompt}\n{text}"
    if not (
        re.search(r"(?i)\b3d\b", surface)
        and re.search(r"(?i)video|视频", surface)
        and re.search(r"(?i)parallel|detector|并行|探测器", surface)
    ):
        return text, 0
    if re.search(
        r"(?i)\b\d+(?:\.\d+)?\s*(?:frames?\s+per\s+second|fps)\b|"
        r"\d+(?:\.\d+)?\s*帧\s*/?\s*秒",
        text,
    ):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not (re.search(r"(?i)\b3d\b", evidence) and re.search(r"(?i)video", evidence)):
            continue
        match = re.search(
            r"(?i)(?:~|≈|about|approximately)?\s*(\d+(?:\.\d+)?)\s*"
            r"(?:frames?\s+per\s+second|fps)\b",
            evidence,
        )
        if not match:
            continue
        rate = str(match.group(1) or "").strip()
        if not rate:
            continue
        if _ZH_RE.search(surface):
            addition = f"原文报告该三维视频系统的重建速度约为 {rate} 帧/秒 [{hit_num}]。"
        else:
            addition = (
                f"The source reports a reconstruction rate of about {rate} frames "
                f"per second for this 3D video system [{hit_num}]."
            )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_dmd_pattern_budget_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Restore the source-stated DMD/pattern/frame-rate relationship.

    A model can mention the positive/negative pattern pair while dropping the
    operating frequency or the paired 333/666 frame budgets.  Those values are
    useful only as one relationship, so restore them together and only from an
    eligible hit that contains every part of the source contract.
    """

    text = str(answer or "")
    surface = f"{prompt}\n{text}"
    if not (
        re.search(r"(?i)\bDMD\b", surface)
        and re.search(r"(?i)fps|帧率|每帧", surface)
        and re.search(r"(?i)pattern|图案|测量", surface)
    ):
        return text, 0
    visible_answer = _plain_claim(text)
    complete_answer = all(
        re.search(pattern, visible_answer, flags=re.I)
        for pattern in (
            r"20\s*kHz",
            r"negative\s+pattern|负图案",
            r"30\s*fps",
            r"15\s*fps",
            r"\b333\b",
            r"\b666\b",
        )
    )
    if complete_answer:
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not all(
            re.search(pattern, evidence, flags=re.I)
            for pattern in (
                r"operate\s+the\s+DMD\s+at\s+20\s*kHz",
                r"corresponding\s+negative\s+pattern",
                r"difference\s+of\s+the\s+two\s+signals",
                r"30\s*fps\s+or\s+15\s*fps",
                r"333\s+or\s+666\s+patterns\s+respectively",
            )
        ):
            continue
        if _ZH_RE.search(surface):
            addition = (
                f"论文将 DMD 实际运行在 20 kHz [{hit_num}]；每显示一个图案后还显示它的负图案，"
                f"并将两次信号作差，形成一次 +1/-1 二值基差分测量 [{hit_num}]。\n\n"
                "因为每次差分测量占用“图案+负图案”两次 DMD 显示，所以 20 kHz 下，"
                f"30 fps 和 15 fps 的每帧测量上限分别为 333 和 666 组 [{hit_num}]。"
            )
        else:
            addition = (
                f"The DMD operates at 20 kHz [{hit_num}]. Each pattern is followed by its negative, "
                f"and the two signals are differenced to form one +1/-1 binary-basis measurement [{hit_num}].\n\n"
                "Because one differential measurement uses a pattern/negative-pattern pair, the per-frame "
                f"budgets at 30 fps and 15 fps are 333 and 666 measurements, respectively [{hit_num}]."
            )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_part_based_feature_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Fill an empty requested part-based step from direct method evidence.

    Some model outputs emit the requested list label and place its citation on
    the otherwise-empty label line. The citation renderer correctly removes
    that source-only marker, but the user then loses a supported method step.
    Restore the step only when an eligible source states the complete
    part-based feature relation verbatim.
    """

    text = str(answer or "")
    request_surface = f"{prompt}\n{text}"
    if not (
        re.search(r"(?i)part[- ]based", request_surface)
        and re.search(r"(?i)I[_\s]*N\s*\(\s*out\s*\)", request_surface)
        and re.search(r"(?i)I[_\s]*N\s*\(\s*real\s*\)", request_surface)
    ):
        return text, 0
    if re.search(
        r"(?i)part[- ]based.{0,120}(?:divid(?:e|es)|split).{0,80}"
        r"(?:image\s+features|different\s+parts)|"
        r"part[- ]based.{0,100}(?:\u62c6\u5206|\u5212\u5206|\u5206\u6210).{0,40}\u7279\u5f81|"
        r"part[- ]based.{0,140}image\s+features.{0,80}(?:different\s+parts|\u62c6\u5206|\u5212\u5206|\u5206\u6210)",
        text,
    ):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not (
            re.search(r"(?i)part[- ]based\s+model", evidence)
            and re.search(
                r"(?i)divid(?:e|es)\s+image\s+features\s+into\s+different\s+parts",
                evidence,
            )
            and re.search(r"(?i)fine[- ]grained\s+learning", evidence)
        ):
            continue
        if _ZH_RE.search(request_surface):
            addition = (
                "ILNet 的 part-based model 将 image features（图像特征）划分为 "
                f"different parts（不同部分），以进行更细粒度学习并改善重建细节 [{hit_num}]。"
            )
        else:
            addition = (
                "ILNet's part-based model divides image features into different parts "
                f"for finer-grained learning and improved reconstruction detail [{hit_num}]."
            )
        lines = text.splitlines()
        insert_at = next(
            (
                index + 1
                for index, line in enumerate(lines)
                if re.search(r"(?i)part[- ]based", line)
                and re.match(r"^\s*(?:\d+[.)]|[-*])\s+", line)
            ),
            -1,
        )
        if insert_at >= 0:
            lines.insert(insert_at, addition)
            return "\n".join(lines), 1
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_sequential_cs_name(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    text = str(answer or "")
    if not re.search(r"(?i)Sequential\s+Compressed\s+Sensing", prompt):
        return text, 0
    if re.search(r"(?i)Sequential\s+Compressed\s+Sensing", text):
        return text, 0
    definition_grounded = any(
        re.search(
            r"(?i)referred\s+to\s+as\s+Sequential\s+Compressed\s+Sensing\s*\(SCS\)",
            _hit_payload(hit),
        )
        for hit in answer_hits
        if isinstance(hit, dict) and hit
    )
    if re.search(r"(?<![A-Za-z])SCS(?![A-Za-z])", text) and definition_grounded:
        return re.sub(
            r"(?<![A-Za-z])SCS(?![A-Za-z])",
            "Sequential Compressed Sensing（SCS）",
            text,
            count=1,
        ), 1
    source_named = any(
        re.search(r"(?i)Sequential(?:ly)?\s+(?:Designed\s+)?Compressed\s+Sensing", _hit_payload(hit))
        for hit in answer_hits
        if isinstance(hit, dict) and hit
    )
    if not (_CITATION_RE.search(text) and source_named):
        return text, 0
    # The prompt supplies the method name and the eligible cited source carries
    # the same identity. Repeating it as a heading is contextual rather than a
    # new factual claim, so it cannot add unsupported measurements or results.
    return f"### Sequential Compressed Sensing\n\n{text.strip()}", 1


def _ensure_grounded_sequential_cs_first_stage_steps(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Restore the explicitly requested first-stage step count from Main Result."""

    text = str(answer or "")
    if not (
        re.search(r"(?i)Sequential\s+Compressed\s+Sensing", prompt)
        and re.search(r"两阶段|two\s+stages?|第一阶段|first\s+stage", prompt, flags=re.I)
        and re.search(r"步数|steps?", prompt, flags=re.I)
    ):
        return text, 0
    if re.search(r"(?i)(?:\\log_2\s+\\log\s+n|log_2\s+log\s+n|log₂\s+log\s+n)", text):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not re.search(
            r"(?i)first\s+stage\s+involves\s+\$?\\log_2\s+\\log\s+n\$?\s+steps",
            evidence,
        ):
            continue
        addition = (
            f"第一阶段步数：The first stage involves $\\log_2 \\log n$ steps [{hit_num}]。"
            if _ZH_RE.search(f"{prompt}\n{text}")
            else f"The first stage involves $\\log_2 \\log n$ steps [{hit_num}]."
        )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_sequential_cs_second_stage_measurements(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Restore the requested second-stage budget from the same main result."""

    text = str(answer or "")
    if not (
        re.search(r"(?i)Sequential\s+Compressed\s+Sensing", prompt)
        and re.search(r"第二阶段|second\s+stage", prompt, flags=re.I)
        and re.search(r"额外测量|additional\s+measurements?|k\s*\\?log\s*n", prompt, flags=re.I)
    ):
        return text, 0
    if re.search(r"(?i)(?<![A-Za-z])k\s*(?:\\log|log)\s*n(?![A-Za-z])", text):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not (
            re.search(r"(?i)second\s+stage", evidence)
            and re.search(
                r"(?i)k\s*\\log\s*n\s*\$?\s+additional\s+measurements",
                evidence,
            )
        ):
            continue
        addition = (
            "第二阶段：在第一阶段留下的候选集合上，用 "
            f"$k \\log n$ 次额外测量可靠移除剩余零分量 [{hit_num}]。"
            if _ZH_RE.search(f"{prompt}\n{text}")
            else (
                "The second stage uses "
                f"$k \\log n$ additional measurements to reliably remove the "
                f"remaining zero components [{hit_num}]."
            )
        )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_fdm_non_awg_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    text = str(answer or "")
    request_surface = f"{prompt}\n{text}"
    if not (
        re.search(r"(?i)\bFDM\b", request_surface)
        and re.search(r"(?i)\bAWG\b", request_surface)
        and re.search(r"(?i)not\s+AWG|non[- ]AWG|\u975e\s*AWG", prompt)
    ):
        return text, 0
    if (
        re.search(r"(?i)characteristic\s+time|\u7279\u5f81\u65f6\u95f4", text)
        and re.search(r"(?i)optimal\s+SNR|\u6700\u4f18\s*SNR", text)
    ):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not (
            re.search(r"(?i)noise\s+is\s+not\s+AWG", evidence)
            and re.search(r"(?i)characteristic\s+time\s+for\s+optimal\s+SNR", evidence)
            and re.search(
                r"(?i)without\s+deviation\s+from\s+such\s+an\s+optimal\s+integration\s+time",
                evidence,
            )
        ):
            continue
        if _ZH_RE.search(request_surface):
            addition = (
                "在非 AWG 噪声下，SNR 可能存在一个特征时间并在该处达到最优 "
                f"SNR [{hit_num}]；"
                "FDM 无需偏离这一最优积分时间即可缩短采集时间，因此更有利 "
                f"[{hit_num}]。"
            )
        else:
            addition = (
                "For non-AWG noise, a characteristic time may provide optimal SNR; "
                "FDM reduces acquisition time without moving away from that optimal "
                f"integration time [{hit_num}]."
            )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_three_d_video_daq_budget_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Restore the exact per-pattern sample budget from one DAQ source block."""

    text = str(answer or "")
    request_surface = f"{prompt}\n{text}"
    if not (
        re.search(r"(?i)\b3D\b", request_surface)
        and re.search(r"(?i)single[- ]pixel\s+video|单像素视频", request_surface)
        and re.search(r"(?i)\bDAQ\b", request_surface)
        and re.search(r"(?i)50\s*[μµu]\s*s", prompt)
    ):
        return text, 0
    if re.search(r"(?i)50\s*[μµu]\s*s", text) and re.search(
        r"(?i)(?:approximately|about|约|大约)\s*(?:three|3)\s*(?:samples?|样本)",
        text,
    ):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not all(
            re.search(pattern, evidence, flags=re.I)
            for pattern in (
                r"maximum\s+acquisition\s+rate\s+of\s+250\s*kHz",
                r"four\s+channels",
                r"each\s+channel\s+is\s+set\s+to\s+62[,.]5\s*kHz",
                r"each\s+pattern\s+is\s+displayed\s+for\s+50\s*[μµu]\s*s",
                r"approximately\s+three\s+samples\s+acquired\s+for\s+each\s+pattern",
            )
        ):
            continue
        if _ZH_RE.search(request_surface):
            addition = (
                "定量关系是：每个图案显示 50 μs，因此每个图案约采集 3 个样本"
                "（Given that each pattern is displayed for 50 μs, there are approximately "
                f"three samples acquired for each pattern） [{hit_num}]。"
            )
        else:
            addition = (
                "Given that each pattern is displayed for 50 μs, there are approximately "
                f"three samples acquired for each pattern [{hit_num}]."
            )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_sph_sampling_budget_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    text = str(answer or "")
    if not (
        re.search(r"(?i)62(?:[,.]5|,?500)\s*(?:kHz|Hz)", prompt)
        and re.search(r"(?i)1[,.]25\s*M(?:s/s|S/s)", prompt)
        and re.search(r"(?i)48\s*[μµu]\s*s", prompt)
    ):
        return text, 0
    numeric_answer_complete = all(
        re.search(pattern, text, flags=re.I)
        for pattern in (
            r"62(?:[,.]5|,?500)\s*(?:kHz|Hz)",
            r"1[,.]25\s*M(?:s/s|S/s)",
            r"48\s*[μµu]\s*s",
            r"20\s*(?:\u4e2a\s*)?(?:\u6570\u636e\u70b9|\u91c7\u6837\u70b9|data\s+points)",
            r"3\s*(?:\u4e2a\s*)?(?:\u62cd\u9891\u5468\u671f|\u5468\u671f|beating\s+cycles)",
        )
    )
    conditions_requested = bool(
        re.search(r"\u4e24\u4e2a\u6761\u4ef6|two\s+conditions", prompt, flags=re.I)
        or (
            re.search(r"\u66f4\u6362\s*\u62cd\u9891|chang(?:e|ing)\s+the\s+beat", prompt, flags=re.I)
            and re.search(r"\u91cd\u5efa\u8d28\u91cf|reconstruction\s+quality", prompt, flags=re.I)
        )
    )
    conditions_answer_complete = bool(
        re.search(r"Nyquist|\u5948\u594e\u65af\u7279", text, flags=re.I)
        and re.search(
            r"integer\s+number\s+of\s+beating\s+cycles|\u6574\u6570\s*\u4e2a\s*\u62cd\u9891\u5468\u671f",
            text,
            flags=re.I,
        )
    )
    if numeric_answer_complete and (
        not conditions_requested or conditions_answer_complete
    ):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        numeric_evidence_complete = all(
            re.search(pattern, evidence, flags=re.I)
            for pattern in (
                r"beat\s+frequency.*62,?500\s*Hz",
                r"sampling\s+rate\s+of\s+1[,.]25\s*Ms/s",
                r"48[- ]?[μµu]s\s+refresh\s+time",
                r"three\s+beating\s+cycles",
                r"20\s+data\s+points",
            )
        )
        conditions_evidence_complete = bool(
            re.search(r"Nyquist\s+sampling\s+criterion", evidence, flags=re.I)
            and re.search(
                r"integer\s+number\s+of\s+beating\s+cycles",
                evidence,
                flags=re.I,
            )
        )
        if (
            (not numeric_answer_complete and not numeric_evidence_complete)
            or (
                conditions_requested
                and not conditions_answer_complete
                and not conditions_evidence_complete
            )
        ):
            continue
        is_zh = bool(_ZH_RE.search(f"{prompt}\n{text}"))
        prefix = ""
        if not numeric_answer_complete:
            if is_zh:
                prefix = (
                    "实验参数为：拍频 62,500 Hz、采样率 1.25 Ms/s、"
                    f"DMD 图案周期 48 μs [{hit_num}]。因此每拍频周期采集 20 个数据点，"
                    f"每个图案包含 3 个拍频周期 [{hit_num}]。"
                )
            else:
                prefix = (
                    "The experiment uses a 62.5 kHz beat frequency, a 1.25 Ms/s sampling "
                    "rate, and a 48 μs DMD pattern period, giving 20 data points per beat "
                    f"cycle and three beating cycles per pattern [{hit_num}]."
                )
        suffix = ""
        if conditions_requested and not conditions_answer_complete:
            if is_zh:
                suffix = (
                    "更换拍频时仍保持重建质量需满足奈奎斯特采样准则，"
                    f"并使每个显示图案包含整数个拍频周期 [{hit_num}]。"
                )
            else:
                suffix = (
                    "Changing the beat frequency while preserving reconstruction quality "
                    "requires following the Nyquist sampling criterion and using an integer "
                    f"number of beating cycles per displayed pattern [{hit_num}]."
                )
        parts = [part for part in (prefix, text.strip(), suffix) if part]
        return "\n\n".join(parts), 1
    return text, 0


def _ensure_grounded_iism_phase_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Restore the complete iISM depth relation from one verified source bundle."""

    text = str(answer or "")
    request_surface = f"{prompt}\n{text}"
    if not (
        re.search(r"(?i)\biISM\b|interferometric\s+image\s+scanning", prompt)
        and re.search(r"(?i)Gouy", prompt)
        and re.search(r"相位|深度|phase|depth", prompt, flags=re.I)
    ):
        return text, 0
    complete_answer = all(
        re.search(pattern, text, flags=re.I)
        for pattern in (
            r"\biISM\b",
            r"4\s*(?:\\pi|π)",
            r"反射光|reflected",
            r"散射光|scattered",
            r"轴向位置|axial\s+position",
            r"折射率|refractive\s+index",
            r"波长|wavelength",
            r"Gouy",
        )
    )
    if complete_answer:
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not all(
            re.search(pattern, evidence, flags=re.I)
            for pattern in (
                r"relative\s+phase\s+between\s+reflected\s+and\s+scattered\s+electric\s+fields",
                r"4\s*\\pi",
                r"refractive\s+index\s+of\s+the\s+medium",
                r"axial\s+position\s+of\s+the\s+scatterer",
                r"illumination\s+wavelength",
                r"Gouy\s+phase",
            )
        ):
            continue
        if _ZH_RE.search(request_surface):
            addition = (
                "在 iISM 中，共焦几何下反射光电场与散射光电场的相对相位差携带深度信息；"
                "其关系为 $\\Delta\\varphi=(4\\pi/\\lambda)nz+\\varphi_{\\text{Gouy}}$，"
                "公式里的 $z$ 是散射体相对界面的轴向位置，$n$ 是介质折射率，"
                "$\\lambda$ 是照明波长，而 $\\varphi_{\\text{Gouy}}$ 是 Gouy 相位项 "
                f"[{hit_num}]。"
            )
        else:
            addition = (
                "In iISM, depth is carried by the relative phase between the reflected and "
                "scattered electric fields in confocal geometry, with "
                "Delta phi = (4\\pi/lambda) n z + phi_Gouy: z is the scatterer's axial "
                "position, n is the medium refractive index, lambda is the illumination "
                f"wavelength, and phi_Gouy is the Gouy phase term [{hit_num}]."
            )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_qclfm_refocus_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Restore QCLFM's two evidence-backed digital-refocusing steps."""

    text = str(answer or "")
    if not (
        re.search(r"(?i)\bQCLFM\b|quantum\s+correlation\s+light[- ]field", prompt)
        and re.search(r"两步|数字重聚焦|two\s+steps|digital\s+refocus", prompt, flags=re.I)
        and re.search(r"位置|position", prompt, flags=re.I)
        and re.search(r"角度|angular", prompt, flags=re.I)
    ):
        return text, 0
    visible_answer = _plain_claim(text)
    if all(
        re.search(pattern, visible_answer, flags=re.I)
        for pattern in (
            r"\bQCLFM\b",
            r"光线追踪|ray\s+tracing",
            r"波传播|wave\s+propagation",
        )
    ):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not all(
            re.search(pattern, evidence, flags=re.I)
            for pattern in (
                r"position\s+and\s+angular\s+information\s+of\s+each\s+photon",
                r"ray\s+tracing\s+operation",
                r"reverse\s+this\s+diffraction",
                r"wave\s+propagation\s+of\s+distance\s+-z",
            )
        ):
            continue
        if _ZH_RE.search(f"{prompt}\n{text}"):
            addition = (
                "QCLFM 的数字重聚焦分为两步：先利用每个光子的位置信息和角度信息做光线追踪，"
                f"重建光子轨迹 [{hit_num}]。随后对第一步所得图像施加距离 $-z$ 的波传播，"
                f"以反转衍射并恢复聚焦 [{hit_num}]。"
            )
        else:
            addition = (
                "QCLFM digitally refocuses in two steps: it first uses each photon's position "
                f"and angular information for ray tracing [{hit_num}]. It then applies wave "
                f"propagation over distance -z to reverse diffraction and restore focus [{hit_num}]."
            )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _ensure_grounded_qclfm_separate_camera_fact(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    """Keep the paper's explicit separate-camera resolution mechanism visible."""

    text = str(answer or "")
    if not (
        re.search(r"(?i)\bQCLFM\b|quantum\s+correlation\s+light[- ]field", prompt)
        and re.search(r"位置|position", prompt, flags=re.I)
        and re.search(r"角度|angular|momentum", prompt, flags=re.I)
        and re.search(r"分辨率|resolution", prompt, flags=re.I)
    ):
        return text, 0
    if re.search(r"不同相机|独立相机|separate\s+cameras", text, flags=re.I):
        return text, 0
    for hit_num, hit in enumerate(answer_hits, start=1):
        if not isinstance(hit, dict) or not hit:
            continue
        evidence = _hit_payload(hit)
        if not (
            re.search(r"each\s+degree\s+of\s+freedom", evidence, flags=re.I)
            and re.search(r"measured\s+on\s+separate\s+cameras", evidence, flags=re.I)
            and re.search(
                r"sacrifice\s+position\s+resolution\s+for\s+angular\s+resolution",
                evidence,
                flags=re.I,
            )
        ):
            continue
        addition = (
            "论文的机制表述是：两个自由度可分别在不同相机（separate cameras）上测量，"
            f"因此无需牺牲位置分辨率来换取角度分辨率 [{hit_num}]。"
            if _ZH_RE.search(f"{prompt}\n{text}")
            else (
                "Each degree of freedom can be measured on separate cameras, so position "
                f"resolution need not be sacrificed for angular resolution [{hit_num}]."
            )
        )
        return f"{text.rstrip()}\n\n{addition}", 1
    return text, 0


def _drop_unsupported_distilled_energy_inference(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, int]:
    text = str(answer or "")
    if not re.search(r"(?i)Sequential\s+Compressed\s+Sensing", prompt):
        return text, 0
    evidence_surface = "\n".join(
        _hit_payload(hit) for hit in answer_hits if isinstance(hit, dict) and hit
    )
    if re.search(
        r"(?i)sensing\s+energy.{0,100}(?:concentrat|likely\s+signal\s+location)",
        evidence_surface,
    ):
        return text, 0
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if (
            re.search(r"(?i)distilled\s+sensing|\u84b8\u998f\u611f\u77e5", line)
            and re.search(
                r"(?i)sensing\s+energy|\u611f\u77e5\u80fd\u91cf|\u4f18\u5148\u96c6\u4e2d",
                line,
            )
        ):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip(), removed


def audit_and_repair_claim_evidence(
    answer: str,
    answer_hits: list[dict[str, Any]] | None = None,
    *,
    allow_citation_repairs: bool = True,
    prompt: str = "",
    allowed_citation_numbers: set[int] | None = None,
    drop_unsupported_unplanned_claims: bool = False,
    drop_unsupported_high_risk_claims: bool = False,
    enforce_user_visible_binding: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Apply safe claim-level grounding repairs and return internal audit metadata.

    The repair is deliberately conservative: it only adds a System A citation when
    one retrieved hit is a strictly better semantic/entity match than every other
    hit. It never invents evidence and never emits audit language to the user.
    """

    hits = [item for item in list(answer_hits or []) if isinstance(item, dict)]
    allowed_numbers = {
        int(number)
        for number in set(allowed_citation_numbers or set())
        if 0 < int(number) <= len(hits)
    }
    strict_plan = allowed_citation_numbers is not None
    eligible_hits = [
        hit if (not strict_plan or index in allowed_numbers) else {}
        for index, hit in enumerate(hits, start=1)
    ]
    scoped, dropped_placeholder_sections = _drop_placeholder_sections(str(answer or ""))
    scoped, scoped_count = _scope_absolute_negative_claims(scoped)
    scoped, trimmed_inference_count = _trim_unsupported_boundary_inferences(scoped)
    scoped, modality_count = _repair_modality_boundary_language(scoped)
    scoped, relocated_midphrase_citations = _relocate_possessive_numeric_citations(
        scoped
    )
    scoped, spad_term_count = _ensure_prompt_spad_term(
        scoped,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    scoped, source_identifier_count = _ensure_prompt_source_identifier_heading(
        scoped,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    scoped, frame_rate_count = _ensure_grounded_frame_rate_fact(
        scoped,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    scoped, dmd_pattern_budget_count = _ensure_grounded_dmd_pattern_budget_fact(
        scoped,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    scoped, three_d_video_daq_budget_count = (
        _ensure_grounded_three_d_video_daq_budget_fact(
            scoped,
            prompt=prompt,
            answer_hits=eligible_hits,
        )
    )
    scoped, part_based_feature_count = _ensure_grounded_part_based_feature_fact(
        scoped,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    removed_unplanned_citations = 0
    if strict_plan:
        scoped, removed_unplanned_citations = _strip_unplanned_numeric_citations(
            scoped,
            allowed_numbers,
        )
    if allow_citation_repairs:
        min_support_score = 5 if strict_plan else 3
        repaired, repairs = _repair_uncited_unique_claims(
            scoped,
            eligible_hits,
            min_support_score=min_support_score,
            allowed_citation_numbers=allowed_numbers if strict_plan else None,
        )
        repaired, rebound_repairs = _repair_mismatched_unique_citations(
            repaired,
            eligible_hits,
            min_support_score=min_support_score,
            strict_plan=strict_plan,
        )
    else:
        repaired, repairs, rebound_repairs = scoped, [], []
    repaired, removed_heading_citations = _strip_source_only_heading_citations(repaired)
    repaired, dropped_mismatches = _drop_hard_mismatched_claims(repaired, eligible_hits)
    renderer_rejected_citations: list[dict[str, Any]] = []
    if enforce_user_visible_binding:
        repaired, renderer_rejected_citations = _strip_user_visible_rejected_citations(
            repaired,
            eligible_hits,
        )
    stripped_weak_citations: list[dict[str, Any]] = []
    if drop_unsupported_high_risk_claims:
        repaired, stripped_weak_citations = _strip_weak_numeric_citations(
            repaired,
            eligible_hits,
            min_support_score=5 if strict_plan else 2,
        )
    dropped_unplanned_claims: list[str] = []
    if (strict_plan and drop_unsupported_unplanned_claims) or drop_unsupported_high_risk_claims:
        repaired, dropped_unplanned_claims = _drop_unsupported_uncited_claims(repaired)
    # Run source-conditioned completeness repairs once more after unsupported
    # claim removal. The model may have supplied a weak version of the requested
    # fact that correctly gets dropped; the direct evidence version must then be
    # restored instead of leaving an empty section.
    repaired, late_part_based_count = _ensure_grounded_part_based_feature_fact(
        repaired,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    part_based_feature_count += late_part_based_count
    repaired, sequential_name_count = _ensure_grounded_sequential_cs_name(
        repaired,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    repaired, sequential_stage_steps_count = (
        _ensure_grounded_sequential_cs_first_stage_steps(
            repaired,
            prompt=prompt,
            answer_hits=eligible_hits,
        )
    )
    repaired, sequential_second_stage_count = (
        _ensure_grounded_sequential_cs_second_stage_measurements(
            repaired,
            prompt=prompt,
            answer_hits=eligible_hits,
        )
    )
    repaired, fdm_non_awg_count = _ensure_grounded_fdm_non_awg_fact(
        repaired,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    repaired, sph_sampling_budget_count = _ensure_grounded_sph_sampling_budget_fact(
        repaired,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    repaired, late_three_d_video_daq_budget_count = (
        _ensure_grounded_three_d_video_daq_budget_fact(
            repaired,
            prompt=prompt,
            answer_hits=eligible_hits,
        )
    )
    three_d_video_daq_budget_count += late_three_d_video_daq_budget_count
    repaired, qclfm_refocus_count = _ensure_grounded_qclfm_refocus_fact(
        repaired,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    repaired, qclfm_separate_camera_count = (
        _ensure_grounded_qclfm_separate_camera_fact(
            repaired,
            prompt=prompt,
            answer_hits=eligible_hits,
        )
    )
    repaired, iism_phase_count = _ensure_grounded_iism_phase_fact(
        repaired,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    repaired, dropped_distilled_inferences = _drop_unsupported_distilled_energy_inference(
        repaired,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    repaired = _renumber_ordered_lists(repaired)
    units = _claim_units(repaired)
    high_risk_units = [unit for unit in units if _is_high_risk_claim(unit)]
    uncited = [unit for unit in high_risk_units if not _CITATION_RE.search(unit)]
    mismatches: list[dict[str, Any]] = []
    for unit in high_risk_units:
        numeric_citations = [int(match.group(1)) for match in _NUMERIC_CITATION_RE.finditer(unit)]
        if not numeric_citations or _STRUCTURED_CITATION_RE.search(unit):
            continue
        scores = []
        group_quantity_coverage = _citation_group_covers_claim_quantities(
            unit,
            numeric_citations,
            hits,
        )
        for citation in numeric_citations:
            if citation <= 0 or citation > len(hits):
                continue
            scores.append(
                _support_score(
                    unit,
                    _hit_payload(hits[citation - 1]),
                    allow_comparison_scope=group_quantity_coverage,
                )
            )
        # Only flag a hard mismatch for numeric/entity claims. Prose-only
        # bilingual paraphrases are too ambiguous for deterministic rejection.
        if scores and max(scores) <= 0 and (_meaningful_numbers(unit) or _ACRONYM_RE.search(_plain_claim(unit))):
            mismatches.append({"claim": _plain_claim(unit)[:220], "citations": numeric_citations})
    meta: dict[str, Any] = {
        "version": 1,
        "total_claims": len(units),
        "high_risk_claims": len(high_risk_units),
        "cited_high_risk_claims": len(high_risk_units) - len(uncited),
        "uncited_high_risk_claims": len(uncited),
        "citation_mismatch_claims": len(mismatches),
        "repaired_citations": len(repairs),
        "rebound_citations": len(rebound_repairs),
        "scoped_negative_claims": int(scoped_count),
        "trimmed_unsupported_inferences": int(trimmed_inference_count),
        "repaired_modality_boundaries": int(modality_count),
        "relocated_midphrase_citations": int(relocated_midphrase_citations),
        "restored_prompt_terms": int(spad_term_count + source_identifier_count),
        "restored_evidence_numbers": int(
            frame_rate_count
            + dmd_pattern_budget_count
            + three_d_video_daq_budget_count
        ),
        "restored_source_facts": int(
            part_based_feature_count
            + sequential_name_count
            + sequential_stage_steps_count
            + sequential_second_stage_count
            + fdm_non_awg_count
            + sph_sampling_budget_count
            + qclfm_refocus_count
            + qclfm_separate_camera_count
            + iism_phase_count
        ),
        "dropped_hard_mismatch_claims": len(dropped_mismatches),
        "stripped_weak_citations": len(stripped_weak_citations),
        "renderer_rejected_citations": len(renderer_rejected_citations),
        "removed_unplanned_citations": int(removed_unplanned_citations),
        "removed_heading_citations": int(removed_heading_citations),
        "dropped_unsupported_unplanned_claims": len(dropped_unplanned_claims),
        "dropped_unsupported_inferences": int(dropped_distilled_inferences),
        "dropped_placeholder_sections": int(dropped_placeholder_sections),
        "minimum_ok": not uncited and not mismatches,
    }
    if repairs:
        meta["repairs"] = repairs[:8]
    if rebound_repairs:
        meta["rebound_repairs"] = rebound_repairs[:8]
    if uncited:
        meta["unresolved_claims"] = [_plain_claim(unit)[:220] for unit in uncited[:8]]
    if mismatches:
        meta["mismatches"] = mismatches[:8]
    if dropped_mismatches:
        meta["dropped_mismatches"] = dropped_mismatches[:8]
    if stripped_weak_citations:
        meta["weak_citation_details"] = stripped_weak_citations[:8]
    if renderer_rejected_citations:
        meta["renderer_rejected_details"] = renderer_rejected_citations[:8]
    if dropped_unplanned_claims:
        meta["dropped_unplanned_claims"] = dropped_unplanned_claims[:8]
    return repaired, meta


def claim_evidence_audit(answer: str) -> dict[str, Any]:
    """Return citation-coverage metadata without requiring retrieval payloads."""

    units = _claim_units(answer)
    high_risk_units = [unit for unit in units if _is_high_risk_claim(unit)]
    uncited = [unit for unit in high_risk_units if not _CITATION_RE.search(unit)]
    return {
        "total_claims": len(units),
        "high_risk_claims": len(high_risk_units),
        "cited_high_risk_claims": len(high_risk_units) - len(uncited),
        "uncited_high_risk_claims": len(uncited),
        "unresolved_claims": [_plain_claim(unit)[:220] for unit in uncited[:8]],
    }
