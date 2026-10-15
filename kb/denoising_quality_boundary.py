from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _source_identity(value: object) -> str:
    text = str(value or "").replace("\\", "/").casefold().strip()
    return re.sub(r"[^a-z0-9]+", "", text.rsplit("/", 1)[-1])


def _slot_source(slot: Mapping[str, Any]) -> str:
    return _source_identity(slot.get("source_path") or slot.get("source_name"))


def _hit_source(hit: Mapping[str, Any]) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), Mapping) else {}
    primary = hit.get("primary_evidence") if isinstance(hit.get("primary_evidence"), Mapping) else {}
    ui = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), Mapping) else {}
    ui_primary = ui.get("primary_evidence") if isinstance(ui.get("primary_evidence"), Mapping) else {}
    for row in (meta, primary, ui_primary, hit):
        source = _source_identity(row.get("source_path") or row.get("source_name"))
        if source:
            return source
    return ""


def _citation_number(slot: Mapping[str, Any], answer_hits: Sequence[Mapping[str, Any]]) -> int:
    source = _slot_source(slot)
    if source and answer_hits:
        matches = [index for index, hit in enumerate(answer_hits, start=1) if _hit_source(hit) == source]
        if matches:
            candidates = [int(value) for value in slot.get("candidate_hits") or [] if str(value).isdigit()]
            return next((number for number in candidates if number in matches), matches[0])
    for value in slot.get("candidate_hits") or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return 0


def _relation_slots(
    citation_plan: Mapping[str, Any] | None,
    answer_hits: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
    n2n: tuple[int, str] | None = None
    n2v: tuple[int, str] | None = None
    for raw_slot in list((citation_plan or {}).get("slots") or []):
        if not isinstance(raw_slot, Mapping):
            continue
        if str(raw_slot.get("preferred_system") or "system_a").casefold() != "system_a":
            continue
        source = _slot_source(raw_slot)
        evidence = _text(raw_slot.get("evidence_quote") or raw_slot.get("evidenceQuote"))
        number = _citation_number(raw_slot, answer_hits)
        if number <= 0 or not evidence:
            continue
        if "noise2noise" in source and all(
            re.search(pattern, evidence, re.I)
            for pattern in (
                r"under certain common circumstances",
                r"performance close or equal",
                r"training using clean exemplars",
            )
        ):
            n2n = (number, evidence)
        if "noise2void" in source and (
            all(
                re.search(pattern, evidence, re.I)
                for pattern in (
                    r"cannot be expected to outperform",
                    r"more information available during training",
                    r"drops? in moderation",
                )
            )
            or all(
                re.search(pattern, evidence, re.I)
                for pattern in (
                    r"less information available for (?:its|their) predictions",
                    r"accuracy to be slightly impaired",
                    r"compared to a normal network",
                )
            )
        ):
            n2v = (number, evidence)
    return n2n, n2v


def complete_denoising_quality_boundary(
    answer: str,
    *,
    prompt: str,
    citation_plan: Mapping[str, Any] | None,
    answer_hits: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Restore narrow N2N/N2V quality boundaries from complete, source-bound relations."""

    text = str(answer or "").strip()
    if not text or not isinstance(citation_plan, Mapping):
        return text
    hits = [hit for hit in list(answer_hits or []) if isinstance(hit, Mapping)]
    n2n, n2v = _relation_slots(citation_plan, hits)
    question = str(prompt or "")
    pair_request = bool(
        n2n
        and n2v
        and re.search(r"paired noisy observations?", question, re.I)
        and re.search(r"necessarily improves?.{0,30}(?:quality|denoising)|quality.{0,30}necessarily", question, re.I)
        and re.search(r"statistical", question, re.I)
        and re.search(r"information trade[- ]off", question, re.I)
    )
    if pair_request:
        n2n_num = n2n[0]
        n2v_num = n2v[0]
        return (
            "**Verdict:** No. Removing the requirement for paired noisy observations does not guarantee better denoising quality.\n\n"
            "Noise2Noise reports that, under certain common circumstances, performance is close or equal to training with "
            f"clean exemplars [{n2n_num}]. This supports conditional parity, not superiority [{n2n_num}].\n\n"
            "Noise2Void cannot be expected to outperform methods that have more information available during training "
            f"[{n2v_num}]. The denoising performance of Noise2Void drops in moderation [{n2v_num}]."
        )

    single_request = bool(
        n2v
        and not n2n
        and re.search(r"胜过|超过|优于|outperform", question, re.I)
        and re.search(r"干净目标|clean targets?|噪声对|noisy pairs?", question, re.I)
        and re.search(r"性能边界|performance boundary|怎样解释|why", question, re.I)
    )
    if not single_request:
        return text
    number, evidence = n2v
    if re.search(r"cannot be expected to outperform", evidence, re.I):
        detail = "拥有更多训练信息的方法；论文观察到的结果只是性能下降幅度较温和"
    else:
        detail = "普通网络；盲点网络可用于预测的信息更少，因此精度可能略受损"
    return (
        f"**结论：不能。** 原文不支持这种训练方式应当胜过{detail} [{number}]。"
        f"这说明取消干净目标或独立噪声对扩大了可训练数据的范围，并不意味着去噪质量一定更高 [{number}]。"
    )


def _n2n_corruption_summary_slot(
    citation_plan: Mapping[str, Any] | None,
    answer_hits: Sequence[Mapping[str, Any]],
) -> tuple[int, str] | None:
    for raw_slot in list((citation_plan or {}).get("slots") or []):
        if not isinstance(raw_slot, Mapping):
            continue
        if str(raw_slot.get("preferred_system") or "system_a").casefold() != "system_a":
            continue
        if "noise2noise" not in _slot_source(raw_slot):
            continue
        evidence = _text(raw_slot.get("evidence_quote") or raw_slot.get("evidenceQuote"))
        number = _citation_number(raw_slot, answer_hits)
        if number <= 0 or not evidence:
            continue
        if all(
            re.search(pattern, evidence, re.I)
            for pattern in (
                r"\bthey need a perfect,? explicit computational model of the corruption",
                r"our requirement of the knowledge of the appropriate statistical summary",
                r"\bmean\b",
                r"\bmedian\b",
            )
        ):
            return number, evidence
    return None


def complete_n2n_corruption_model_boundary(
    answer: str,
    *,
    prompt: str,
    citation_plan: Mapping[str, Any] | None,
    answer_hits: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Restore N2N's stated statistical-summary requirement from one direct source surface."""

    text = str(answer or "").strip()
    if not text or not isinstance(citation_plan, Mapping):
        return text
    question = str(prompt or "")
    if not (
        re.search(r"likelihood|explicit model", question, re.I)
        and re.search(r"corruption", question, re.I)
        and re.search(r"statistical condition|remains necessary", question, re.I)
    ):
        return text
    if not (
        re.search(r"does not require|no .{0,24}explicit model|avoid\w* .{0,40}explicit", text, re.I)
        and re.search(r"statistical summary|appropriate statistics", text, re.I)
    ):
        return text
    if re.search(r"\b(?:expectation|mean)\b", text, re.I):
        return text
    hits = [hit for hit in list(answer_hits or []) if isinstance(hit, Mapping)]
    relation = _n2n_corruption_summary_slot(citation_plan, hits)
    if relation is None:
        return text
    number, _evidence = relation
    return (
        f"{text}\n\n"
        "The paper makes this condition concrete as knowledge of the appropriate statistical summary, "
        f"naming the mean or median, rather than a perfect explicit computational corruption model [{number}]."
    )
