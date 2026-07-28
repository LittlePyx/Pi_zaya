from __future__ import annotations

import re
from typing import Any

from kb.evidence_binding import assess_system_a_hit_binding
from kb.evidence_term_mapping import evidence_alignment_tokens


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
)


def _split_claim_segments(value: str) -> list[str]:
    protected = re.sub(
        r"(?i)\b(?:nat|commun|fig|eq|et\s+al)\.",
        lambda match: match.group(0)[:-1] + "<KB_DOT>",
        str(value or ""),
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
    return [part.replace("<KB_DOT>", ".").strip() for part in parts if part.strip()]


def _claim_units(answer: str) -> list[str]:
    units: list[str] = []
    in_fence = False
    for raw_line in str(answer or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or _HEADING_RE.match(stripped) or _TABLE_OR_CODE_RE.match(stripped):
            continue
        stripped = _LIST_PREFIX_RE.sub("", stripped).strip()
        if not stripped:
            continue
        units.extend(part for part in _split_claim_segments(stripped) if len(part) >= 10)
    return units


def _plain_claim(text: str) -> str:
    value = _CITATION_RE.sub(" ", str(text or ""))
    value = re.sub(r"[*_`>#]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _meaningful_numbers(text: str) -> list[str]:
    plain = _CITATION_RE.sub(" ", str(text or ""))
    values: list[str] = []
    for match in _NUMBER_RE.finditer(plain):
        raw_value = re.sub(r"\s+", "", match.group(0)).lower()
        value = re.search(r"\d+(?:\.\d+)?", match.group(0))
        value = value.group(0) if value else ""
        # Bare single-digit list/order markers are not scientific claims.
        if re.fullmatch(r"\d", raw_value):
            continue
        if re.fullmatch(r"\d{4}", raw_value) and 1900 <= int(raw_value) <= 2100:
            # Publication years identify papers, but are not scientific result
            # values that must occur in the supporting evidence sentence.
            continue
        if "." in value:
            try:
                value = str(float(value))
            except ValueError:
                pass
        values.append(value)
    return values


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


def _concept_ids(text: str) -> set[int]:
    normalized = re.sub(r"[\s_]+", " ", str(text or "").lower())
    return {
        index
        for index, aliases in enumerate(_CONCEPT_GROUPS)
        if any(alias in normalized for alias in aliases)
    }


def _support_score(claim: str, evidence: str) -> int:
    claim_plain = _plain_claim(claim)
    evidence_norm = re.sub(r"\s+", " ", str(evidence or "")).lower()
    if not claim_plain or not evidence_norm:
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
    explicit_relation_requirements = (
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
    if any(
        claim_pattern.search(claim_low) and not evidence_pattern.search(evidence_norm)
        for claim_pattern, evidence_pattern in explicit_relation_requirements
    ):
        return 0
    claim_numbers = _meaningful_numbers(claim_plain)
    if claim_numbers and not all(number in re.sub(r"\s+", "", evidence_norm) for number in claim_numbers):
        return 0
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
            bound_scores = [
                _support_score(segment, _hit_payload(answer_hits[number - 1]))
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
            scored_citations = {
                number: _support_score(
                    segment,
                    _hit_payload(answer_hits[number - 1]),
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
                evidence_quote = quotes[0]
                source_name = str(
                    meta.get("source_name")
                    or hit.get("source_name")
                    or hit.get("title")
                    or ""
                ).strip()
                binding = assess_system_a_hit_binding(
                    answer_claim=segment,
                    hit=hit,
                    meta=binding_meta,
                    heading=str(meta.get("heading_path") or meta.get("top_heading") or ""),
                    evidence_quote=evidence_quote,
                    source_name=source_name,
                )
                if bool(binding.get("suppress_link")):
                    rejected_numbers.append(number)
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
                    _support_score(segment, _hit_payload(answer_hits[number - 1]))
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
            if not citations or not _is_high_risk_claim(segment):
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
                _support_score(segment, _hit_payload(answer_hits[number - 1]))
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
    scoped, spad_term_count = _ensure_prompt_spad_term(
        scoped,
        prompt=prompt,
        answer_hits=eligible_hits,
    )
    scoped, frame_rate_count = _ensure_grounded_frame_rate_fact(
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
        for citation in numeric_citations:
            if citation <= 0 or citation > len(hits):
                continue
            scores.append(_support_score(unit, _hit_payload(hits[citation - 1])))
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
        "restored_prompt_terms": int(spad_term_count),
        "restored_evidence_numbers": int(frame_rate_count),
        "dropped_hard_mismatch_claims": len(dropped_mismatches),
        "stripped_weak_citations": len(stripped_weak_citations),
        "renderer_rejected_citations": len(renderer_rejected_citations),
        "removed_unplanned_citations": int(removed_unplanned_citations),
        "removed_heading_citations": int(removed_heading_citations),
        "dropped_unsupported_unplanned_claims": len(dropped_unplanned_claims),
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
