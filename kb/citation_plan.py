from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from kb.reference_query_family import (
    extract_requested_paper_count,
    prompt_requests_answer_audit,
    strip_negated_reference_trail_requests,
)


_ORIGIN_INTENT_RE = re.compile(
    r"(?i)(怎么来|从哪(?:里)?来|来源|出处|源头|借鉴|上游|前人|已有|先前|之前|早期|"
    r"谁提出|谁发明|谁最早|原创|不是.*原创|origin|source|upstream|prior|previous|"
    r"borrowed|based on|inspired|who proposed|who introduced|come from|came from)"
)
_STRONG_ORIGIN_INTENT_RE = re.compile(
    r"(?i)(怎么来|从哪(?:里)?来|出处|源头|借鉴|上游|前人|谁提出|谁发明|谁最早|原创|不是.*原创|"
    r"origin|upstream|borrowed|based on|inspired|who proposed|who introduced|come from|came from)"
)
_SOURCE_MARKER_REQUEST_RE = re.compile(
    r"(?i)(?:来源|引用|证据)(?:编号|序号|标号)|(?:编号|序号|标号).{0,8}(?:来源|引用|证据)|"
    r"(?:标出|标注|给出|注明|附上).{0,24}(?:来源|引用|证据|依据)|"
    r"(?:每个|各个|逐(?:条|项|句)|结论).{0,24}(?:来源|引用|证据|依据)|"
    r"(?:可点击|点回|回原文).{0,20}(?:来源|引用|证据|依据)|"
    r"source\s+(?:number|marker|citation)|citation\s+(?:number|marker)|"
    r"(?:each|every).{0,16}(?:claim|conclusion|sentence).{0,24}(?:source|citation|evidence)"
)
_COMPARE_INTENT_RE = re.compile(
    r"(?i)(对比|比较|区别|差异|哪个更|优缺点|trade-?off|versus|vs\.?|compare|comparison|difference)"
)
_METHOD_INTENT_RE = re.compile(
    r"(?i)(怎么做|如何做|如何实现|流程|步骤|训练|公式|算法|方法|method|implementation|pipeline|train|derive)"
)
_BEGINNER_INTENT_RE = re.compile(
    r"(?i)(看不懂|入门|初学|小白|通俗|简单讲|overview|explain|intuitive|beginner|plain language)"
)


def _compact_text(value: Any, *, max_len: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "..."


def _first_text(raw: Mapping[str, Any], *keys: str, max_len: int = 240) -> str:
    for key in keys:
        text = _compact_text(raw.get(key), max_len=max_len)
        if text:
            return text
    return ""


def _source_name(source_path: str) -> str:
    text = str(source_path or "").strip()
    if not text:
        return ""
    name = Path(text).name or text
    for suffix in (".en.md", ".md"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _positive_ints(values: Any, *, limit: int = 6) -> list[int]:
    out: list[int] = []
    for raw in list(values or []):
        try:
            n = int(raw)
        except Exception:
            continue
        if n <= 0 or n in out:
            continue
        out.append(n)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _citation_intent(prompt: str, *, prompt_family: str = "") -> str:
    routing_prompt = strip_negated_reference_trail_requests(prompt)
    raw = " ".join([routing_prompt, str(prompt_family or "")]).strip()
    family = str(prompt_family or "").strip().lower()
    origin_match = bool(_ORIGIN_INTENT_RE.search(raw))
    marker_request = bool(_SOURCE_MARKER_REQUEST_RE.search(raw))
    if family == "citation_lookup" or bool(_STRONG_ORIGIN_INTENT_RE.search(raw)) or (origin_match and not marker_request):
        return "origin_lookup"
    if _COMPARE_INTENT_RE.search(raw) or family == "compare":
        return "comparison"
    if _METHOD_INTENT_RE.search(raw) or family in {"method", "reproduce", "figure_walkthrough"}:
        return "method_explain"
    if _BEGINNER_INTENT_RE.search(raw) or family in {"overview", "strength_limits"}:
        return "beginner_overview"
    return "answer_grounding"


def _budget_for_intent(intent: str) -> dict[str, int]:
    if intent == "origin_lookup":
        return {"system_a": 1, "system_b": 1}
    if intent == "comparison":
        return {"system_a": 2, "system_b": 0}
    if intent == "method_explain":
        return {"system_a": 2, "system_b": 1}
    if intent == "beginner_overview":
        return {"system_a": 2, "system_b": 1}
    return {"system_a": 2, "system_b": 1}


def _system_b_slots(
    opportunities: Sequence[Mapping[str, Any]] | None,
    *,
    intent: str,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw0 in list(opportunities or []):
        if not isinstance(raw0, Mapping):
            continue
        raw = dict(raw0)
        try:
            ref_num = int(raw.get("ref_num") or 0)
        except Exception:
            ref_num = 0
        sid = str(raw.get("sid") or "").strip()
        if ref_num <= 0 or not sid:
            continue
        key = (sid.lower(), ref_num)
        if key in seen:
            continue
        seen.add(key)
        label = _first_text(raw, "label", "topic", "title", "ref_title", "ref_raw", max_len=160)
        source_path = str(raw.get("source_path") or "").strip()
        slots.append(
            {
                "claim_type": "origin" if intent == "origin_lookup" else "upstream_reference",
                "preferred_system": "system_b",
                "topic": label or f"reference {ref_num}",
                "candidate_refs": [ref_num],
                "candidate_cite_examples": [f"[[CITE:{sid}:{ref_num}]]"],
                "sid": sid,
                "source_path": source_path,
                "source_name": _source_name(source_path),
                "heading_path": _first_text(raw, "heading_path", "heading", max_len=180),
                "evidence_quote": _first_text(raw, "evidence_quote", "quote", "snippet", max_len=220),
                "instruction": (
                    "Use this only on a sentence that explains where a method, concept, or prior-work thread comes from."
                ),
            }
        )
        if len(slots) >= max(1, int(max_items)):
            break
    return slots


def _system_a_slots(
    *,
    support_slots: Sequence[Mapping[str, Any]] | None,
    answer_hits: Sequence[Mapping[str, Any]] | None,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_slot(raw: Mapping[str, Any], *, hit_num: int = 0) -> None:
        source_path = str(raw.get("source_path") or "").strip()
        heading = _first_text(raw, "heading_path", "heading", "ref_best_heading_path", max_len=180)
        snippet = _first_text(
            raw,
            "evidence_atom_text",
            "evidence_quote",
            "locate_anchor",
            "snippet",
            "text",
            max_len=220,
        )
        identity = "|".join([source_path.lower(), heading.lower(), snippet[:120].lower(), str(hit_num)])
        if not (source_path or heading or snippet) or identity in seen:
            return
        seen.add(identity)
        candidate_hits = [int(hit_num)] if int(hit_num or 0) > 0 else []
        slots.append(
            {
                "claim_type": _first_text(raw, "claim_type", max_len=80) or "paper_evidence",
                "preferred_system": "system_a",
                "topic": heading or _source_name(source_path) or "retrieved evidence",
                "candidate_hits": candidate_hits,
                "support_example": _first_text(raw, "support_example", max_len=80),
                "source_path": source_path,
                "source_name": _source_name(source_path),
                "heading_path": heading,
                "evidence_quote": snippet,
                "candidate_refs": _positive_ints(raw.get("candidate_refs"), limit=4),
                "instruction": "Use this for factual claims supported by the retrieved paper text itself.",
            }
        )

    for raw in list(support_slots or []):
        if isinstance(raw, Mapping):
            add_slot(dict(raw))
        if len(slots) >= max(1, int(max_items)):
            return slots

    for idx, hit0 in enumerate(list(answer_hits or []), start=1):
        if not isinstance(hit0, Mapping):
            continue
        hit = dict(hit0)
        meta = dict(hit.get("meta") or {}) if isinstance(hit.get("meta"), Mapping) else {}
        raw = {
            "source_path": meta.get("source_path"),
            "heading_path": meta.get("heading_path") or meta.get("ref_best_heading_path"),
            "evidence_quote": meta.get("evidence_quote") or hit.get("text"),
            "text": hit.get("text"),
            "claim_type": meta.get("claim_type"),
        }
        add_slot(raw, hit_num=idx)
        if len(slots) >= max(1, int(max_items)):
            break
    return slots


def build_citation_plan(
    *,
    prompt: str,
    prompt_family: str = "",
    answer_hits: Sequence[Mapping[str, Any]] | None = None,
    support_slots: Sequence[Mapping[str, Any]] | None = None,
    reference_opportunities: Sequence[Mapping[str, Any]] | None = None,
    max_slots: int = 5,
) -> dict[str, Any]:
    intent = _citation_intent(prompt, prompt_family=prompt_family)
    budget = _budget_for_intent(intent)
    requested_paper_count = extract_requested_paper_count(prompt)
    requested_system_a = min(8, int(requested_paper_count or 0))
    answer_audit = prompt_requests_answer_audit(prompt)
    if answer_audit:
        intent = "answer_audit"
        requested_system_a = min(8, len(list(answer_hits or [])))
        budget = {"system_a": requested_system_a, "system_b": 0}
    if requested_system_a > 0:
        budget["system_a"] = max(int(budget.get("system_a") or 0), requested_system_a)
    sys_b = (
        _system_b_slots(reference_opportunities, intent=intent, max_items=3)
        if int(budget.get("system_b") or 0) > 0
        else []
    )
    system_a_limit = max(3, requested_system_a)
    sys_a = _system_a_slots(
        support_slots=support_slots,
        answer_hits=answer_hits,
        max_items=system_a_limit,
    )
    slots = (sys_b if intent == "origin_lookup" else []) + sys_a
    if intent != "origin_lookup":
        slots.extend(sys_b)
    slots = slots[: max(1, int(max(max_slots, system_a_limit)))]
    return {
        "version": 1,
        "source": "citation_plan_builder",
        "intent": intent,
        "budget": dict(budget),
        "system_a_enabled": bool(int(budget.get("system_a") or 0) > 0 and sys_a),
        "system_b_enabled": bool(int(budget.get("system_b") or 0) > 0 and sys_b),
        "slots": [dict(slot) for slot in slots if isinstance(slot, dict)],
    }


def build_citation_plan_prompt_block(plan: Mapping[str, Any] | None) -> str:
    if not isinstance(plan, Mapping) or not plan:
        return ""
    slots = [dict(item) for item in list(plan.get("slots") or []) if isinstance(item, Mapping)]
    if not slots:
        return ""
    budget = dict(plan.get("budget") or {}) if isinstance(plan.get("budget"), Mapping) else {}
    lines = [
        "Citation plan (follow before adding citations):",
        f"- intent={str(plan.get('intent') or '').strip() or 'answer_grounding'}",
        f"- per paragraph budget: SystemA={int(budget.get('system_a') or 0)}, SystemB={int(budget.get('system_b') or 0)}",
        "- SystemA = retrieved paper text evidence; SystemB = a retrieved paper's bibliography/reference item.",
        "- Put a citation immediately after the sentence it supports; do not cite decorative or summary-only sentences.",
        "- Use SystemB only for origin, prior-work, method-source, or 'where did this idea come from' claims.",
        "- Use SystemA for claims about what the retrieved paper itself says, shows, defines, or reports.",
    ]
    for idx, slot in enumerate(slots[:6], start=1):
        preferred = str(slot.get("preferred_system") or "").strip() or "system_a"
        topic = _compact_text(slot.get("topic"), max_len=120) or "evidence"
        parts = [f"{idx}. preferred_system={preferred}", f"topic={topic}"]
        cite_examples = [str(x or "").strip() for x in list(slot.get("candidate_cite_examples") or []) if str(x or "").strip()]
        if cite_examples:
            parts.append("cite_example=" + " ".join(cite_examples[:2]))
        support_example = str(slot.get("support_example") or "").strip()
        if support_example:
            parts.append(f"support_example={support_example}")
        candidate_hits = _positive_ints(slot.get("candidate_hits"), limit=3)
        if candidate_hits:
            parts.append("hit=" + ",".join(str(n) for n in candidate_hits))
        heading = _compact_text(slot.get("heading_path"), max_len=100)
        if heading:
            parts.append(f"heading={heading}")
        quote = _compact_text(slot.get("evidence_quote"), max_len=160)
        if quote:
            parts.append(f"evidence={quote}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def citation_plan_prefers_system_b(
    plan: Mapping[str, Any] | None,
    *,
    context: str = "",
    ref_num: int = 0,
) -> bool:
    if not isinstance(plan, Mapping) or not plan:
        return False
    budget = dict(plan.get("budget") or {}) if isinstance(plan.get("budget"), Mapping) else {}
    if int(budget.get("system_b") or 0) <= 0:
        return False
    slots = [dict(item) for item in list(plan.get("slots") or []) if isinstance(item, Mapping)]
    try:
        n = int(ref_num or 0)
    except Exception:
        n = 0
    for slot in slots:
        if str(slot.get("preferred_system") or "").strip().lower() != "system_b":
            continue
        refs = _positive_ints(slot.get("candidate_refs"), limit=12)
        if n > 0 and refs and n in refs:
            return True
    intent = str(plan.get("intent") or "").strip().lower()
    if n <= 0 and bool(plan.get("system_b_enabled")):
        return True
    if intent == "origin_lookup" and bool(plan.get("system_b_enabled")):
        return bool(_ORIGIN_INTENT_RE.search(str(context or "")))
    return False
