from __future__ import annotations

import hashlib
import re
import time
import uuid
from copy import deepcopy
from typing import Any, Callable

from kb.evidence_matrix import evidence_matrix_hits
from kb.research_brief import research_brief_evidence


_CITATION_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
_LIST_RE = re.compile(r"^(\s*(?:[-*+]\s+|\d+[.)]\s+))")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


def research_brief_content_hash(content: object) -> str:
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def _citation_numbers(value: object) -> list[int]:
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(str(value or "")):
        for part in re.split(r"\s*,\s*", match.group(1)):
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                try:
                    start = int(start_raw.strip())
                    end = int(end_raw.strip())
                except ValueError:
                    continue
                candidates = range(start, end + 1) if 0 < start <= end and end - start < 100 else ()
            else:
                try:
                    candidates = (int(part.strip()),)
                except ValueError:
                    candidates = ()
            for number in candidates:
                if number > 0 and number not in numbers:
                    numbers.append(number)
    return numbers


def _claim_spans(content: str) -> list[dict[str, Any]]:
    """Return citation-bearing Markdown blocks without normalizing their bytes."""

    lines = content.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    spans: list[dict[str, Any]] = []
    index = 0
    heading = ""
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if _HEADING_RE.match(line):
            heading = stripped.lstrip("#").strip()
            index += 1
            continue
        if not stripped:
            index += 1
            continue
        start_index = index
        is_list = bool(_LIST_RE.match(line))
        index += 1
        while index < len(lines):
            candidate = lines[index]
            if not candidate.strip() or _HEADING_RE.match(candidate):
                break
            if _LIST_RE.match(candidate):
                break
            if not is_list and _LIST_RE.match(lines[start_index]):
                break
            index += 1
        start = offsets[start_index]
        end = offsets[index] if index < len(offsets) else len(content)
        while end > start and content[end - 1] in "\r\n":
            end -= 1
        text = content[start:end]
        citations = _citation_numbers(text)
        if citations:
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "heading": heading,
                    "citation_numbers": citations,
                    "list_prefix": (_LIST_RE.match(text).group(1) if _LIST_RE.match(text) else ""),
                }
            )
    return spans


def _source_identity(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def _normal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _evidence_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _source_identity(item.get("source_path") or item.get("source_name")),
        _normal(item.get("evidence_quote")),
        _normal(item.get("source_evidence_quote")),
        _normal(item.get("matrix_field")),
        _normal(item.get("comparison_audit_id")),
    )


def _hit_evidence_key(hit: dict[str, Any]) -> tuple[str, str, str, str, str]:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return (
        _source_identity(meta.get("source_path") or meta.get("source_name")),
        _normal(hit.get("text")),
        _normal(meta.get("comparison_source_quote")),
        _normal(meta.get("matrix_field")),
        _normal(meta.get("comparison_audit_id")),
    )


def _compatible(old: dict[str, Any], hit: dict[str, Any]) -> bool:
    old_key = _evidence_key(old)
    hit_key = _hit_evidence_key(hit)
    return bool(
        old_key[0]
        and old_key[0] == hit_key[0]
        and old_key[3] == hit_key[3]
        and old_key[4] == hit_key[4]
    )


def stable_matrix_hits(
    old_evidence: list[dict[str, Any]],
    matrix_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep surviving citation slots stable while replacing invalid matrix evidence.

    Citation verification is positional. Empty historical slots are filled only with
    current matrix evidence and marked as fillers, so unchanged higher-numbered
    citations do not need to be rewritten merely because an earlier claim vanished.
    """

    latest = [deepcopy(item) for item in evidence_matrix_hits(matrix_record, limit=100)]
    if not latest:
        return []
    old_by_number = {
        int(item.get("citation_number") or 0): item
        for item in old_evidence
        if isinstance(item, dict) and int(item.get("citation_number") or 0) > 0
    }
    old_max = max(old_by_number, default=0)
    slot_count = max(old_max, len(latest))
    slots: list[dict[str, Any] | None] = [None] * slot_count
    assigned: set[int] = set()

    for number, old in sorted(old_by_number.items()):
        for latest_index, hit in enumerate(latest):
            if latest_index in assigned or _evidence_key(old) != _hit_evidence_key(hit):
                continue
            slots[number - 1] = deepcopy(hit)
            assigned.add(latest_index)
            break

    for number, old in sorted(old_by_number.items()):
        if slots[number - 1] is not None:
            continue
        for latest_index, hit in enumerate(latest):
            if latest_index in assigned or not _compatible(old, hit):
                continue
            slots[number - 1] = deepcopy(hit)
            assigned.add(latest_index)
            break

    for latest_index, hit in enumerate(latest):
        if latest_index in assigned:
            continue
        try:
            empty_index = slots.index(None)
        except ValueError:
            slots.append(deepcopy(hit))
        else:
            slots[empty_index] = deepcopy(hit)
        assigned.add(latest_index)

    filler = next((deepcopy(item) for item in slots if isinstance(item, dict)), deepcopy(latest[0]))
    for index, item in enumerate(slots):
        if item is not None:
            continue
        replacement = deepcopy(filler)
        meta = replacement.get("meta") if isinstance(replacement.get("meta"), dict) else {}
        replacement["meta"] = {**meta, "citation_slot_filler": True}
        slots[index] = replacement
    return [item for item in slots if isinstance(item, dict)]


def _remap_citations(text: str, mapping: dict[int, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        numbers = _citation_numbers(match.group(0))
        remapped = [mapping[number] for number in numbers if number in mapping]
        return f"[{', '.join(str(number) for number in remapped)}]" if remapped else ""

    return _CITATION_RE.sub(replace, text)


def _model_candidates(
    focus_hits: list[dict[str, Any]],
    actual_numbers: list[int],
    *,
    locale: str,
    settings: Any,
    max_tokens: int,
    model_generator: Callable[..., dict[str, Any]] | None,
) -> tuple[dict[int, str], dict[str, Any]]:
    if not focus_hits or model_generator is None:
        return {}, {"mode": "extractive", "elapsed_ms": 0.0, "reason": "model_not_requested"}
    english = str(locale or "").lower().startswith("en")
    prompt = (
        "Rewrite only the affected research-brief claims from the numbered evidence snippets. "
        "Return exactly one Markdown bullet sentence per snippet, in snippet order. Each sentence must name "
        "its paper or method, use no facts beyond that snippet, and end with exactly its local citation [n]. "
        "Do not add headings, comparisons, recommendations, or statements about missing evidence."
        if english
        else
        "仅根据带编号的证据片段改写受影响的研究简报主张。按片段顺序，每个片段只返回一个 Markdown 项目符号完整句；"
        "每句必须点明对应论文或方法，只使用该片段中的事实，并以唯一的本地引用 [n] 结尾。"
        "不要增加标题、跨来源比较、建议或证据缺失陈述。"
    )
    started = time.perf_counter()
    try:
        generated = model_generator(
            prompt,
            focus_hits,
            settings=settings,
            agent_notes={
                "answer_contract": "research_brief_incremental_update",
                "evidence_gate": {
                    "answer_mode": "evidence_grounded",
                    "source_blend": "local_grounded",
                    "source_policy": "local_only",
                },
            },
            temperature=0.0,
            max_tokens=max(200, min(int(max_tokens or 800), 1_600)),
            defer_quality_gate_repair=True,
        )
        answer = str(generated.get("answer") or "").strip()
        from kb.agent.tools import verify_answer_citations

        verification_payload = verify_answer_citations(answer, focus_hits, answer_mode="evidence_grounded")
        verification = verification_payload.get("verification") if isinstance(verification_payload.get("verification"), dict) else {}
        if (
            int(verification.get("total_claims") or 0) <= 0
            or int(verification.get("unsupported_claims") or 0) > 0
            or float(verification.get("support_ratio") or 0.0) < 0.999
        ):
            raise ValueError("focused model candidate did not pass citation verification")
        local_to_actual = {index: number for index, number in enumerate(actual_numbers, start=1)}
        candidates: dict[int, str] = {}
        for span in _claim_spans(answer):
            local_numbers = list(span.get("citation_numbers") or [])
            if len(local_numbers) != 1 or local_numbers[0] not in local_to_actual:
                continue
            actual = local_to_actual[local_numbers[0]]
            candidates[actual] = _remap_citations(str(span.get("text") or ""), local_to_actual)
        return candidates, {
            "mode": "model_synthesis" if len(candidates) == len(actual_numbers) else "mixed_fallback",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "candidate_count": len(candidates),
            "requested_count": len(actual_numbers),
        }
    except Exception as exc:
        return {}, {
            "mode": "extractive",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "reason": str(exc)[:500],
        }


def _extractive_claim(hit: dict[str, Any], number: int) -> str:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source = re.sub(r"\s+", " ", str(meta.get("source_name") or meta.get("title") or "Source")).strip()
    quote = _CITATION_RE.sub("", str(hit.get("text") or ""))
    quote = re.sub(r"\s+", " ", quote).strip()
    if not quote:
        return ""
    quote = quote.rstrip(".!?。！？")
    return f"- {source}: {quote} [{number}]."


def _style_replacement(old_text: str, claims: list[str]) -> str:
    if not claims:
        return ""
    old_prefix_match = _LIST_RE.match(old_text)
    if old_prefix_match:
        prefix = old_prefix_match.group(1)
        styled = []
        for claim in claims:
            body = _LIST_RE.sub("", claim, count=1).strip()
            styled.append(f"{prefix}{body}")
        return "\n".join(styled)
    return "\n".join(_LIST_RE.sub("", claim, count=1).strip() for claim in claims)


def build_research_brief_update_plan(
    brief: dict[str, Any],
    *,
    historical_matrix: dict[str, Any],
    current_matrix: dict[str, Any],
    impact: dict[str, Any],
    locale: str = "zh",
    settings: Any = None,
    max_tokens: int = 800,
    model_generator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    content = str(brief.get("content_markdown") or "")
    old_evidence = [item for item in list(brief.get("evidence") or []) if isinstance(item, dict)]
    old_by_number = {
        int(item.get("citation_number") or 0): item
        for item in old_evidence
        if int(item.get("citation_number") or 0) > 0
    }
    affected = {
        int(number)
        for number in list(impact.get("affected_citation_numbers") or [])
        if int(number or 0) > 0
    }
    current_hits = stable_matrix_hits(old_evidence, current_matrix)
    current_evidence = research_brief_evidence(current_hits)
    current_by_number = {
        int(item.get("citation_number") or 0): item
        for item in current_evidence
        if int(item.get("citation_number") or 0) > 0
    }
    affected_spans = [
        span
        for span in _claim_spans(content)
        if affected.intersection(set(span.get("citation_numbers") or []))
    ]

    target_numbers: list[int] = []
    span_targets: list[list[int]] = []
    for span in affected_spans:
        targets: list[int] = []
        for number in list(span.get("citation_numbers") or []):
            old = old_by_number.get(number)
            hit = current_hits[number - 1] if 0 < number <= len(current_hits) else None
            current = current_by_number.get(number)
            if not old or not hit or not current:
                continue
            if number in affected and not _compatible(old, hit):
                continue
            targets.append(number)
            if number not in target_numbers:
                target_numbers.append(number)
        span_targets.append(targets)

    focus_hits = [current_hits[number - 1] for number in target_numbers if 0 < number <= len(current_hits)]
    model_candidates, generation = _model_candidates(
        focus_hits,
        target_numbers,
        locale=locale,
        settings=settings,
        max_tokens=max_tokens,
        model_generator=model_generator,
    )
    items: list[dict[str, Any]] = []
    for index, (span, targets) in enumerate(zip(affected_spans, span_targets), start=1):
        claims: list[str] = []
        modes: list[str] = []
        for number in targets:
            candidate = model_candidates.get(number)
            if candidate:
                claims.append(candidate)
                modes.append("model_synthesis")
            else:
                fallback = _extractive_claim(current_hits[number - 1], number)
                if fallback:
                    claims.append(fallback)
                    modes.append("extractive_fallback")
        proposed = _style_replacement(str(span.get("text") or ""), claims)
        before_numbers = list(span.get("citation_numbers") or [])
        item_id = f"change-{index}-{uuid.uuid4().hex[:10]}"
        items.append(
            {
                "id": item_id,
                "start": int(span.get("start") or 0),
                "end": int(span.get("end") or 0),
                "heading": str(span.get("heading") or ""),
                "old_markdown": str(span.get("text") or ""),
                "proposed_markdown": proposed,
                "action": "replace" if proposed else "delete",
                "recommended": "accept",
                "citation_numbers_before": before_numbers,
                "citation_numbers_after": _citation_numbers(proposed),
                "affected_citation_numbers": sorted(affected.intersection(before_numbers)),
                "generation_modes": sorted(set(modes)),
            }
        )

    preview = apply_research_brief_update_decisions(
        content,
        items,
        {str(item["id"]): "accept" for item in items},
    )
    affected_chars = sum(int(item["end"]) - int(item["start"]) for item in items)
    return {
        "contract_version": 1,
        "brief_id": str(brief.get("id") or ""),
        "base_brief_revision": int(brief.get("revision") or 1),
        "base_content_hash": research_brief_content_hash(content),
        "matrix_id": str(current_matrix.get("id") or ""),
        "source_matrix_revision": int(historical_matrix.get("revision") or 1),
        "target_matrix_revision": int(current_matrix.get("revision") or 1),
        "items": items,
        "preview_content_markdown": preview["content_markdown"],
        "impact": deepcopy(impact),
        "generation": generation,
        "preservation": {
            "base_character_count": len(content),
            "affected_character_count": affected_chars,
            "unaffected_character_count": max(0, len(content) - affected_chars),
            "unaffected_preservation_ratio": round(
                max(0, len(content) - affected_chars) / max(1, len(content)),
                4,
            ),
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def apply_research_brief_update_decisions(
    base_content: str,
    items: list[dict[str, Any]],
    decisions: dict[str, str],
) -> dict[str, Any]:
    content = str(base_content or "")
    accepted: list[str] = []
    rejected: list[str] = []
    replacements: list[tuple[int, int, str]] = []
    for item in items:
        item_id = str(item.get("id") or "")
        decision = str(decisions.get(item_id) or "reject").strip().lower()
        if decision == "accept":
            accepted.append(item_id)
            replacements.append(
                (
                    int(item.get("start") or 0),
                    int(item.get("end") or 0),
                    str(item.get("proposed_markdown") or ""),
                )
            )
        else:
            rejected.append(item_id)
    for start, end, replacement in sorted(replacements, reverse=True):
        if start < 0 or end < start or end > len(content):
            raise ValueError("invalid research brief update-plan span")
        content = f"{content[:start]}{replacement}{content[end:]}"
    return {
        "content_markdown": content,
        "accepted_item_ids": accepted,
        "rejected_item_ids": rejected,
        "all_accepted": not rejected,
    }
