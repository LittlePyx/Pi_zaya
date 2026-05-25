from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REF_CARD_POLISH_CONTRACT_VERSION = 1
CITATION_CARD_QUALITY_CONTRACT_VERSION = 1

POLISH_STATUSES = {"full", "heuristic", "pending", "failed"}
LLM_SUMMARY_GENERATIONS = {"llm_grounded", "llm_pack", "llm_abstract"}
LLM_WHY_GENERATIONS = {"llm_grounded", "llm_pack"}
PENDING_GENERATIONS = {"pending", "pending_section_seed", "pending_focus_seed"}
FAILED_GENERATIONS = {"failed", "error", "render_failed", "polish_failed"}

_TEMPLATE_PHRASES = (
    "the paper cites",
    "this is stated in",
    "this hit is directly relevant",
    "why this is relevant",
    "no summary available",
)
_GENERIC_LOCATOR_PHRASES = (
    "not located",
    "unknown location",
    "尚未",
    "未定位",
    "无法定位",
)
_BROKEN_EVIDENCE_PHRASES = (
    "has attrac",
    "rson can be",
    "$^{",
    "\\begin{",
)
_MARKDOWN_HEADING_RE = re.compile(r"(^|\n)\s{0,3}#{1,6}\s+\S")
_MARKDOWN_TABLE_RULE_RE = re.compile(r"(^|\n)\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*(\n|$)")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first_text(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _text(mapping.get(key))
        if text:
            return text
    return ""


def _intish(value: Any) -> int:
    try:
        return int(str(value or "").strip())
    except Exception:
        return 0


def _citation_route(detail: Mapping[str, Any]) -> str:
    explicit = _norm(detail.get("citation_route") or detail.get("route"))
    if explicit in {"system_a", "system-b", "system_b", "a", "b"}:
        return "system_b" if explicit in {"system-b", "system_b", "b"} else "system_a"
    return "system_b" if bool(detail.get("is_inpaper")) else "system_a"


def _has_raw_markdown(text: str) -> bool:
    if "[[CITE:" in text or "```" in text:
        return True
    if _MARKDOWN_HEADING_RE.search(text):
        return True
    return bool(_MARKDOWN_TABLE_RULE_RE.search(text))


def _has_template_phrase(text: str) -> bool:
    lowered = _norm(text)
    return any(phrase in lowered for phrase in _TEMPLATE_PHRASES)


def _has_generic_locator(text: str) -> bool:
    lowered = _norm(text)
    return not lowered or any(phrase in lowered for phrase in _GENERIC_LOCATOR_PHRASES)


def _looks_broken_evidence(text: str) -> bool:
    stripped = _text(text)
    lowered = _norm(stripped)
    if not stripped:
        return False
    if any(phrase in lowered for phrase in _BROKEN_EVIDENCE_PHRASES):
        return True
    if stripped.startswith("...") or stripped.startswith("…"):
        return True
    tail = re.search(r"\s([A-Za-z]{2,})\.{3}$", stripped)
    if tail and len(stripped) < 220:
        word = tail.group(1).lower()
        return len(word) <= 5 or word in {"attrac", "appro", "recons"}
    return False


def _quality_issue(name: str, *, field: str = "", detail: Any = "", severity: str = "error") -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "severity": severity}
    if field:
        out["field"] = field
    if detail != "":
        out["detail"] = detail
    return out


def _visible_citation_texts(data: Mapping[str, Any], *, route: str) -> dict[str, str]:
    if route == "system_b":
        return {
            "card_takeaway": _first_text(data, ("card_takeaway", "upstream_work_role", "user_question_relation", "support_relation")),
            "card_evidence": _first_text(data, ("card_evidence", "citation_context", "evidence_quote", "context", "answer_claim")),
            "card_locator": _first_text(data, ("card_locator", "location_label", "heading_path")),
            "card_reference_entry": _first_text(data, ("card_reference_entry", "raw", "cite_fmt")),
            "card_support_explanation": _first_text(data, ("card_support_explanation", "why_line")),
        }
    return {
        "card_takeaway": _first_text(data, ("card_takeaway",)),
        "card_claim": _first_text(data, ("card_claim", "answer_claim")),
        "card_evidence": _first_text(data, ("card_evidence", "evidence_quote", "primary_evidence_quote", "quote", "summary_line", "raw", "cite_fmt")),
        "card_locator": _first_text(data, ("card_locator", "location_label", "heading_path")),
        "card_support_explanation": _first_text(data, ("card_support_explanation", "support_relation", "binding_reason", "why_line")),
    }


def citation_detail_quality(detail: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a compact quality contract for one rendered citation popover."""

    data = _as_dict(detail)
    route = _citation_route(data)
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def fail(name: str, *, field: str = "", detail: Any = "") -> None:
        failures.append(_quality_issue(name, field=field, detail=detail, severity="error"))

    def warn(name: str, *, field: str = "", detail: Any = "") -> None:
        warnings.append(_quality_issue(name, field=field, detail=detail, severity="warning"))

    if _norm(data.get("source")) == "inline_marker":
        fail("inline_marker_not_rendered")

    num = _intish(data.get("num") or data.get("ref_num"))
    if num <= 0:
        fail("missing_citation_number")
    if not _text(data.get("anchor")):
        fail("missing_click_anchor")

    source_identity = _first_text(data, ("source_name", "source_path", "title", "raw", "cite_fmt"))
    if not source_identity:
        fail("missing_source_identity")

    visible_texts = _visible_citation_texts(data, route=route)
    for field, text in visible_texts.items():
        if not text:
            continue
        if _has_raw_markdown(text):
            fail("raw_markdown_visible", field=field, detail=text[:120])
        if _has_template_phrase(text):
            fail("template_phrase_visible", field=field, detail=text[:120])

    locator = visible_texts.get("card_locator", "")
    if route == "system_a":
        evidence = visible_texts.get("card_evidence", "")
        if len(evidence) < 24:
            fail("system_a_missing_evidence", field="evidence_quote")
        elif _looks_broken_evidence(evidence):
            fail("system_a_broken_evidence", field="evidence_quote", detail=evidence[:160])
        if _has_generic_locator(locator):
            fail("system_a_missing_locator", field="location_label")
    else:
        reference_identity = _first_text(data, ("title", "raw", "cite_fmt", "card_reference_entry"))
        takeaway = visible_texts.get("card_takeaway", "")
        context = visible_texts.get("card_evidence", "")
        if not reference_identity:
            fail("system_b_missing_reference_identity")
        if len(takeaway) < 24:
            fail("system_b_missing_takeaway", field="card_takeaway")
        if len(context) < 24:
            fail("system_b_missing_citing_context", field="citation_context")
        if _has_generic_locator(locator):
            fail("system_b_missing_locator", field="location_label")
        if _norm(context) == _norm(reference_identity):
            fail("system_b_context_is_reference_entry", field="citation_context")
        if _norm(data.get("card_locator_label")) in {"current paper citation", "current paper location"}:
            warn("system_b_locator_label_should_be_user_facing", field="card_locator_label")

    score = max(0.0, 1.0 - len(failures) * 0.22 - len(warnings) * 0.05)
    return {
        "quality_contract_version": CITATION_CARD_QUALITY_CONTRACT_VERSION,
        "ok": not failures,
        "score": round(score, 3),
        "route": route,
        "num": num,
        "anchor": _text(data.get("anchor")),
        "failures": failures,
        "warnings": warnings,
    }


def summarize_citation_detail_quality(details: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    items = [citation_detail_quality(item) for item in details if isinstance(item, Mapping)]
    route_counts = {"system_a": 0, "system_b": 0}
    ok_route_counts = {"system_a": 0, "system_b": 0}
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        route = str(item.get("route") or "")
        if route in route_counts:
            route_counts[route] += 1
            if bool(item.get("ok")):
                ok_route_counts[route] += 1
        for failure in list(item.get("failures") or []):
            if isinstance(failure, Mapping):
                failures.append({"index": idx, **dict(failure)})
        for warning in list(item.get("warnings") or []):
            if isinstance(warning, Mapping):
                warnings.append({"index": idx, **dict(warning)})
    return {
        "quality_contract_version": CITATION_CARD_QUALITY_CONTRACT_VERSION,
        "ok": not failures,
        "count": len(items),
        "route_counts": route_counts,
        "ok_route_counts": ok_route_counts,
        "failures": failures,
        "warnings": warnings,
        "min_score": min((float(item.get("score") or 0.0) for item in items), default=1.0),
    }


def _field_polish_status(*, generation: str, kind: str = "", field: str = "") -> str:
    gen = _norm(generation)
    kind_norm = _norm(kind)
    field_norm = _norm(field)
    if gen in FAILED_GENERATIONS:
        return "failed"
    if gen in PENDING_GENERATIONS or gen.startswith("pending_"):
        return "pending"
    if field_norm == "summary" and gen in LLM_SUMMARY_GENERATIONS:
        return "full"
    if field_norm == "why" and gen in LLM_WHY_GENERATIONS:
        return "full"
    if field_norm == "summary" and kind_norm in {"abstract", "metadata"}:
        return "heuristic"
    return "heuristic"


def ref_card_polish_status(
    ui_meta: Mapping[str, Any] | None,
    *,
    hit_meta: Mapping[str, Any] | None = None,
    render_status: str = "",
    display_state: str = "",
) -> dict[str, Any]:
    ui = _as_dict(ui_meta)
    meta = _as_dict(hit_meta)
    render_status_norm = _norm(render_status)
    display_state_norm = _norm(display_state)
    ref_state = _norm(meta.get("ref_pack_state"))
    summary_generation = _norm(ui.get("summary_generation"))
    why_generation = _norm(ui.get("why_generation"))
    summary_kind = _norm(ui.get("summary_kind")) or "guide"

    summary_status = _field_polish_status(
        generation=summary_generation,
        kind=summary_kind,
        field="summary",
    )
    why_status = _field_polish_status(
        generation=why_generation,
        kind=summary_kind,
        field="why",
    )

    if (
        render_status_norm == "failed"
        or display_state_norm == "failed"
        or ref_state == "failed"
        or summary_status == "failed"
        or why_status == "failed"
    ):
        status = "failed"
    elif (
        display_state_norm == "pending"
        or ref_state == "pending"
        or bool(ui.get("score_pending"))
        or summary_status == "pending"
        or why_status == "pending"
    ):
        status = "pending"
    elif summary_status == "full" and why_status == "full":
        status = "full"
    else:
        status = "heuristic"

    return {
        "polish_contract_version": REF_CARD_POLISH_CONTRACT_VERSION,
        "polish_status": status,
        "summary_polish_status": summary_status,
        "why_polish_status": why_status,
        "polish_source": "llm" if status == "full" else ("pending" if status == "pending" else ("failed" if status == "failed" else "rules")),
        "polish_detail": ";".join(
            part
            for part in (
                f"summary:{summary_generation or 'unset'}->{summary_status}",
                f"why:{why_generation or 'unset'}->{why_status}",
            )
            if part
        ),
    }


def attach_ref_card_polish_contract(
    ui_meta: Mapping[str, Any] | None,
    *,
    hit_meta: Mapping[str, Any] | None = None,
    render_status: str = "",
    display_state: str = "",
) -> dict[str, Any]:
    ui = _as_dict(ui_meta)
    if not ui:
        return {}
    ui.update(
        ref_card_polish_status(
            ui,
            hit_meta=hit_meta,
            render_status=render_status,
            display_state=display_state,
        )
    )
    return ui


def attach_refs_pack_polish_contract(pack: Mapping[str, Any] | None) -> dict[str, Any]:
    out = _as_dict(pack)
    hits = [dict(hit) for hit in list(out.get("hits") or []) if isinstance(hit, Mapping)]
    render_status = _norm(out.get("render_status"))
    display_state = _norm(out.get("display_state"))
    counts = {status: 0 for status in sorted(POLISH_STATUSES)}
    next_hits: list[dict[str, Any]] = []
    for hit in hits:
        ui_meta = _as_dict(hit.get("ui_meta"))
        hit_meta = _as_dict(hit.get("meta"))
        if ui_meta:
            ui_meta = attach_ref_card_polish_contract(
                ui_meta,
                hit_meta=hit_meta,
                render_status=render_status,
                display_state=display_state,
            )
            hit["ui_meta"] = ui_meta
            status = _norm(ui_meta.get("polish_status")) or "heuristic"
            counts[status if status in counts else "heuristic"] += 1
        next_hits.append(hit)

    if next_hits or "hits" in out:
        out["hits"] = next_hits

    if display_state == "pending" or bool(out.get("pending")):
        pack_status = "pending"
    elif render_status == "failed" or display_state == "failed":
        pack_status = "failed"
    elif next_hits and counts.get("full", 0) == len(next_hits):
        pack_status = "full"
    elif next_hits:
        pack_status = "heuristic"
    else:
        pack_status = "pending" if bool(out.get("enrichment_pending")) else (render_status or "heuristic")
        if pack_status not in POLISH_STATUSES:
            pack_status = "heuristic"

    out["polish_contract_version"] = REF_CARD_POLISH_CONTRACT_VERSION
    out["polish_status"] = pack_status
    out["polish_counts"] = {key: int(value) for key, value in counts.items()}
    return out


def refs_pack_has_full_llm_copy(pack: Mapping[str, Any] | None) -> bool:
    normalized = attach_refs_pack_polish_contract(pack)
    hits = [hit for hit in list(normalized.get("hits") or []) if isinstance(hit, Mapping)]
    if not hits:
        return True
    return all(_norm(_as_dict(hit.get("ui_meta")).get("polish_status")) == "full" for hit in hits)
