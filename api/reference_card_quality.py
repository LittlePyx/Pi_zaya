from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REF_CARD_POLISH_CONTRACT_VERSION = 1

POLISH_STATUSES = {"full", "heuristic", "pending", "failed"}
LLM_SUMMARY_GENERATIONS = {"llm_grounded", "llm_pack", "llm_abstract"}
LLM_WHY_GENERATIONS = {"llm_grounded", "llm_pack"}
PENDING_GENERATIONS = {"pending", "pending_section_seed", "pending_focus_seed"}
FAILED_GENERATIONS = {"failed", "error", "render_failed", "polish_failed"}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


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
