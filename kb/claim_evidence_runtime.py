from __future__ import annotations

import re
from typing import Any


_CITATION_RE = re.compile(
    r"(?<!\[)\[(?:R?\d{1,5})(?:\s*[,;，；、-]\s*R?\d{1,5})*\](?:\([^\n)]+\))?"
    r"|\[\[(?:CITE|SUPPORT):[^\]]+\]\]",
    flags=re.IGNORECASE,
)
_NUMERIC_CITATION_RE = re.compile(r"(?<!\[)\[(\d{1,5})\](?:\([^\n)]+\))?")
_STRUCTURED_CITATION_RE = re.compile(r"\[\[(?:CITE|SUPPORT):[^\]]+\]\]", flags=re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_TABLE_OR_CODE_RE = re.compile(r"^\s*(?:\||```|~~~|<!--)")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?(?:\s*(?:%|dB|nm|μm|um|mm|cm|Hz|kHz|MHz|GHz|fps|帧/秒|帧|倍))?",
    flags=re.IGNORECASE,
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,12}\b")
_EN_TERM_RE = re.compile(r"\b[a-z][a-z0-9-]{3,}\b", flags=re.IGNORECASE)
_RISK_RE = re.compile(
    r"(?:表明|证明|达到|提升|提高|降低|优于|劣于|导致|使(?:得)?|通过|采用|使用|利用|构建|引入|"
    r"包含|纳入|建模|训练|验证|报告|实现|解决|权衡|局限|限制|没有|未(?:提供|说明|显示|报告|验证|讨论)|"
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
_BOUNDARY_RE = re.compile(
    r"(?:当前|本轮)(?:引用|检索)(?:到的)?(?:片段|证据|结果)?[^。！？.!?]{0,16}"
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
        r"(?<=[。！？!?；;])\s*|(?<=[A-Za-z0-9\)])\.\s+(?=[A-Z\u4e00-\u9fff])",
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
    if _ADVICE_RE.search(plain) and re.match(
        r"^(?:阅读建议|建议|可以|应该|直接阅读|对于.{0,40}可参考|read|recommend|for further)",
        plain,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(_meaningful_numbers(plain) or _RISK_RE.search(plain))


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
    score += min(3, len(claim_terms & evidence_terms))
    if claim_numbers:
        score += 2
    return score


def _best_unique_hit(claim: str, answer_hits: list[dict[str, Any]]) -> tuple[int, int]:
    scored = sorted(
        (
            (_support_score(claim, _hit_payload(hit)), index)
            for index, hit in enumerate(answer_hits, start=1)
            if isinstance(hit, dict)
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < 3:
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


def _repair_uncited_unique_claims(answer: str, answer_hits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    repairs: list[dict[str, Any]] = []
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
        if not segments:
            output_lines.append(raw_line)
            continue
        changed = False
        rebuilt_segments: list[str] = []
        for segment in segments:
            if _CITATION_RE.search(segment) or not _is_high_risk_claim(segment):
                rebuilt_segments.append(segment)
                continue
            hit_index, score = _best_unique_hit(segment, answer_hits)
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


def _repair_mismatched_unique_citations(
    answer: str,
    answer_hits: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Rebind an incorrect System-A number only when one hit is uniquely stronger."""

    repairs: list[dict[str, Any]] = []
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
        if not segments:
            output_lines.append(raw_line)
            continue
        rebuilt: list[str] = []
        changed = False
        for segment in segments:
            citations = [int(match.group(1)) for match in _NUMERIC_CITATION_RE.finditer(segment)]
            if not citations or not _is_high_risk_claim(segment):
                rebuilt.append(segment)
                continue
            cited_scores = [
                _support_score(segment, _hit_payload(answer_hits[number - 1]))
                for number in citations
                if 0 < number <= len(answer_hits)
            ]
            if cited_scores and max(cited_scores) >= 3:
                rebuilt.append(segment)
                continue
            hit_index, score = _best_unique_hit(segment, answer_hits)
            if hit_index <= 0 or hit_index in citations:
                rebuilt.append(segment)
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


def audit_and_repair_claim_evidence(
    answer: str,
    answer_hits: list[dict[str, Any]] | None = None,
    *,
    allow_citation_repairs: bool = True,
    prompt: str = "",
) -> tuple[str, dict[str, Any]]:
    """Apply safe claim-level grounding repairs and return internal audit metadata.

    The repair is deliberately conservative: it only adds a System A citation when
    one retrieved hit is a strictly better semantic/entity match than every other
    hit. It never invents evidence and never emits audit language to the user.
    """

    hits = [item for item in list(answer_hits or []) if isinstance(item, dict)]
    scoped, dropped_placeholder_sections = _drop_placeholder_sections(str(answer or ""))
    scoped, scoped_count = _scope_absolute_negative_claims(scoped)
    scoped, modality_count = _repair_modality_boundary_language(scoped)
    scoped, spad_term_count = _ensure_prompt_spad_term(
        scoped,
        prompt=prompt,
        answer_hits=hits,
    )
    if allow_citation_repairs:
        repaired, repairs = _repair_uncited_unique_claims(scoped, hits)
        repaired, rebound_repairs = _repair_mismatched_unique_citations(repaired, hits)
    else:
        repaired, repairs, rebound_repairs = scoped, [], []
    repaired, dropped_mismatches = _drop_hard_mismatched_claims(repaired, hits)
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
        "repaired_modality_boundaries": int(modality_count),
        "restored_prompt_terms": int(spad_term_count),
        "dropped_hard_mismatch_claims": len(dropped_mismatches),
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
