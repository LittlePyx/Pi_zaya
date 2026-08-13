from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from api.deps import get_chat_store, get_settings
from kb.agent.runner import run_research_agent
from kb.agent.tools import generate_grounded_answer, verify_answer_citations
from kb.maintenance import create_auto_snapshot
from kb.research_brief import (
    research_brief_bibliography,
    research_brief_bibtex,
    research_brief_context,
    research_brief_docx,
    research_brief_evidence,
    generate_research_brief_from_matrix,
    research_brief_markdown,
    research_brief_prompt,
    research_brief_quality,
    research_brief_ris,
    select_research_brief_sources,
)
from kb.research_brief_lineage import (
    matrix_contract_fingerprint,
    research_brief_lineage,
)
from kb.research_brief_update import (
    apply_research_brief_update_decisions,
    build_research_brief_update_plan,
    research_brief_content_hash,
    stable_matrix_hits,
)


router = APIRouter(prefix="/api", tags=["research-briefs"])

_MAX_TITLE_CHARS = 240
_MAX_OBJECTIVE_CHARS = 4_000
_MAX_CONTENT_CHARS = 160_000
_MAX_ITEM_KEYS = 8


class ResearchBriefCreateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field("Untitled research brief", max_length=_MAX_TITLE_CHARS)
    objective: str = Field("", max_length=_MAX_OBJECTIVE_CHARS)
    content_markdown: str = Field("", max_length=_MAX_CONTENT_CHARS)
    source_conv_id: str | None = Field(None, max_length=120)


class ResearchBriefUpdateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)
    title: str | None = Field(None, max_length=_MAX_TITLE_CHARS)
    objective: str | None = Field(None, max_length=_MAX_OBJECTIVE_CHARS)
    content_markdown: str | None = Field(None, max_length=_MAX_CONTENT_CHARS)


class ResearchBriefGenerateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field("Research brief", max_length=_MAX_TITLE_CHARS)
    objective: str = Field("", max_length=_MAX_OBJECTIVE_CHARS)
    item_keys: list[str] = Field(default_factory=list, max_length=_MAX_ITEM_KEYS)
    source_conv_id: str | None = Field(None, max_length=120)
    brief_id: str | None = Field(None, max_length=120)
    matrix_id: str | None = Field(None, max_length=120)
    expected_revision: int | None = Field(None, ge=1)
    locale: str = Field("zh", max_length=16)
    top_k: int = Field(8, ge=2, le=20)
    max_tokens: int = Field(1_800, ge=400, le=4_096)


class ResearchBriefRestoreBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    revision: int = Field(..., ge=1)
    expected_revision: int = Field(..., ge=1)


class ResearchBriefUpdatePlanCreateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)
    locale: str = Field("zh", max_length=16)
    max_tokens: int = Field(800, ge=200, le=1_600)


class ResearchBriefUpdateDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_id: str = Field(..., min_length=1, max_length=120)
    decision: Literal["accept", "reject"]


class ResearchBriefUpdatePlanApplyBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expected_revision: int = Field(..., ge=1)
    decisions: list[ResearchBriefUpdateDecision] = Field(default_factory=list, max_length=100)


def _project_or_404(project_id: str) -> dict:
    project = get_chat_store().get_project(str(project_id or "").strip())
    if project is None:
        raise HTTPException(404, "project not found")
    return project


def _brief_or_404(brief_id: str) -> dict:
    record = get_chat_store().get_research_brief(str(brief_id or "").strip())
    if record is None:
        raise HTTPException(404, "research brief not found")
    return record


def _brief_with_lineage(
    record: dict,
    *,
    include_impact: bool = True,
    summary_only: bool = False,
    matrix_cache: dict[str, dict | None] | None = None,
) -> dict:
    enriched = dict(record or {})
    quality = enriched.get("quality") if isinstance(enriched.get("quality"), dict) else {}
    matrix_id = str(quality.get("source_matrix_id") or "").strip()
    store = get_chat_store()
    cache = matrix_cache if matrix_cache is not None else {}
    if matrix_id not in cache:
        cache[matrix_id] = store.get_evidence_matrix(matrix_id) if matrix_id else None
    current_matrix = cache.get(matrix_id)
    if isinstance(current_matrix, dict) and str(current_matrix.get("project_id") or "") != str(enriched.get("project_id") or current_matrix.get("project_id") or ""):
        current_matrix = None
    historical_matrix: dict | None = None
    saved_revision = int(quality.get("source_matrix_revision") or 0)
    current_revision = int((current_matrix or {}).get("revision") or 0)
    if matrix_id and saved_revision > 0 and current_revision > saved_revision and not summary_only:
        historical_matrix = store.get_evidence_matrix_revision(matrix_id, saved_revision)
    enriched["lineage"] = research_brief_lineage(
        enriched,
        current_matrix=current_matrix,
        historical_matrix=historical_matrix,
        include_impact=include_impact,
        summary_only=summary_only,
    )
    return enriched


def _briefs_with_lineage(records: list[dict]) -> list[dict]:
    cache: dict[str, dict | None] = {}
    return [
        _brief_with_lineage(
            record,
            include_impact=False,
            summary_only=True,
            matrix_cache=cache,
        )
        for record in records
    ]


def _conflict_response(record: dict | None) -> None:
    current_revision = int((record or {}).get("revision") or 0)
    raise HTTPException(409, f"research brief revision conflict; current revision is {current_revision}")


def _brief_update_matrices(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    matrix_id = str(quality.get("source_matrix_id") or "").strip()
    source_revision = int(quality.get("source_matrix_revision") or 0)
    if not matrix_id or source_revision <= 0:
        raise HTTPException(400, "incremental updates require a matrix-backed research brief")
    store = get_chat_store()
    current = store.get_evidence_matrix(matrix_id)
    if current is None or str(current.get("project_id") or "") != str(record.get("project_id") or ""):
        raise HTTPException(409, "source evidence matrix is unavailable")
    if str(current.get("quality_status") or "") != "verified":
        raise HTTPException(409, "the latest evidence matrix must be verified before planning an update")
    current_revision = int(current.get("revision") or 1)
    if current_revision <= source_revision:
        historical = current
    else:
        historical = store.get_evidence_matrix_revision(matrix_id, source_revision)
    if historical is None:
        raise HTTPException(409, "the source evidence-matrix revision is unavailable")
    lineage = research_brief_lineage(
        record,
        current_matrix=current,
        historical_matrix=historical,
        include_impact=True,
    )
    if str(lineage.get("status") or "") != "matrix_updated":
        raise HTTPException(
            409,
            "an incremental update plan requires a verified matrix change with auditable impact; "
            f"current lineage status is {lineage.get('status') or 'unknown'}",
        )
    return historical, current, lineage


def _download_name(record: dict, suffix: str) -> str:
    title = str(record.get("title") or "research-brief").strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "-", title).strip("-")[:64] or "research-brief"
    return f"{slug}-r{int(record.get('revision') or 1)}.{suffix}"


@router.get("/projects/{project_id}/research-briefs")
def list_research_briefs(project_id: str, limit: int = Query(80, ge=1, le=300)):
    _project_or_404(project_id)
    return _briefs_with_lineage(get_chat_store().list_research_briefs(project_id, limit=limit))


@router.post("/projects/{project_id}/research-briefs")
def create_research_brief(project_id: str, body: ResearchBriefCreateBody):
    _project_or_404(project_id)
    record = get_chat_store().create_research_brief(
        project_id=project_id,
        title=body.title,
        objective=body.objective,
        content_markdown=body.content_markdown,
        source_conv_id=body.source_conv_id,
        quality_status="draft",
        quality={
            "contract_version": 1,
            "reasons": ["manual_draft"],
            "edited_after_verification": False,
        },
    )
    if record is None:
        raise HTTPException(404, "project not found")
    return _brief_with_lineage(record)


@router.post("/projects/{project_id}/research-briefs/generate")
def generate_research_brief(project_id: str, body: ResearchBriefGenerateBody):
    _project_or_404(project_id)
    store = get_chat_store()
    current: dict[str, Any] | None = None
    if body.brief_id:
        current = _brief_or_404(body.brief_id)
        if str(current.get("project_id") or "") != str(project_id or ""):
            raise HTTPException(404, "research brief not found in project")
        if body.expected_revision is None:
            raise HTTPException(400, "expected_revision is required when regenerating a research brief")
        if int(current.get("revision") or 1) != int(body.expected_revision):
            _conflict_response(current)
        current_quality = current.get("quality") if isinstance(current.get("quality"), dict) else {}
        current_matrix_id = str(current_quality.get("source_matrix_id") or "").strip()
        requested_matrix_id = str(body.matrix_id or "").strip()
        if current_matrix_id and not requested_matrix_id:
            raise HTTPException(400, "matrix_id is required when regenerating a matrix-backed research brief")
        if current_matrix_id and requested_matrix_id != current_matrix_id:
            raise HTTPException(
                400,
                "an existing research brief cannot switch evidence matrices; create a new brief instead",
            )
    settings = get_settings()
    prompt = research_brief_prompt(body.objective, locale=body.locale)
    source_matrix: dict[str, Any] | None = None
    if body.matrix_id:
        source_matrix = store.get_evidence_matrix(body.matrix_id)
        if source_matrix is None or str(source_matrix.get("project_id") or "") != str(project_id or ""):
            raise HTTPException(404, "evidence matrix not found in project")
        if str(source_matrix.get("quality_status") or "") != "verified":
            raise HTTPException(400, "research briefs require a verified evidence matrix")
        if current is not None:
            current_quality = current.get("quality") if isinstance(current.get("quality"), dict) else {}
            saved_matrix_revision = int(current_quality.get("source_matrix_revision") or 0)
            latest_matrix_revision = int(source_matrix.get("revision") or 1)
            if saved_matrix_revision > 0 and latest_matrix_revision > saved_matrix_revision:
                historical_matrix = store.get_evidence_matrix_revision(
                    str(source_matrix.get("id") or ""),
                    saved_matrix_revision,
                )
                lineage = research_brief_lineage(
                    current,
                    current_matrix=source_matrix,
                    historical_matrix=historical_matrix,
                    include_impact=True,
                )
                if str(lineage.get("status") or "") == "matrix_updated":
                    raise HTTPException(
                        409,
                        "the evidence matrix changed; create and review an incremental update plan instead of replacing the full brief",
                    )
        selected_items = [
            item
            for item in list(source_matrix.get("source_items") or [])
            if isinstance(item, dict)
        ]
        if not selected_items:
            raise HTTPException(400, "verified evidence matrix has no source items")
        payload = generate_research_brief_from_matrix(
            prompt,
            matrix_record=source_matrix,
            settings=settings,
            max_tokens=body.max_tokens,
        )
    else:
        shelf = store.get_citation_shelf(
            project_id=project_id,
            scope="project",
        )
        shelf_items = [item for item in list((shelf or {}).get("items") or []) if isinstance(item, dict)]
        requested_keys = {str(key or "").strip() for key in body.item_keys if str(key or "").strip()}
        if requested_keys:
            unavailable_keys = sorted(
                key
                for key in requested_keys
                if not select_research_brief_sources(shelf_items, item_keys=[key])
            )
            if unavailable_keys:
                raise HTTPException(
                    400,
                    "selected literature-basket items lack local full-text evidence: "
                    + ", ".join(unavailable_keys[:8]),
                )
        selected_items = select_research_brief_sources(
            shelf_items,
            item_keys=body.item_keys,
        )
        if not selected_items:
            raise HTTPException(
                400,
                "no selected literature-basket item has local full-text evidence",
            )
        context = research_brief_context(
            selected_items,
            conversation_id=str(body.source_conv_id or "").strip(),
        )
        payload = run_research_agent(
            prompt,
            db_dir=settings.db_dir,
            settings=settings,
            top_k=max(body.top_k, len(selected_items)),
            temperature=0.1,
            max_tokens=body.max_tokens,
            query_scope="basket",
            selected_research_context=context,
            answer_contract="research_brief",
        )
    answer = str(payload.get("answer") or "").strip()
    agent_trace = payload.get("agent_trace") if isinstance(payload.get("agent_trace"), dict) else {}
    evidence = research_brief_evidence(
        [item for item in list(payload.get("hits") or []) if isinstance(item, dict)]
    )
    bibliography = research_brief_bibliography(selected_items, evidence)
    quality_status, quality = research_brief_quality(
        answer=answer,
        agent_trace=agent_trace,
        selected_items=selected_items,
        evidence=evidence,
    )
    if source_matrix is not None:
        quality["source_matrix_id"] = str(source_matrix.get("id") or "")
        quality["source_matrix_revision"] = int(source_matrix.get("revision") or 1)
        quality["source_matrix_quality_status"] = str(source_matrix.get("quality_status") or "")
        quality["source_matrix_title"] = str(source_matrix.get("title") or "")
        quality["source_matrix_fingerprint"] = matrix_contract_fingerprint(source_matrix)
    title = str(body.title or "").strip() or str(body.objective or "").strip()[:_MAX_TITLE_CHARS]
    if body.brief_id:
        record, conflict = store.update_research_brief(
            body.brief_id,
            expected_revision=body.expected_revision,
            title=title,
            objective=body.objective,
            content_markdown=answer,
            evidence=evidence,
            bibliography=bibliography,
            agent_trace=agent_trace,
            quality_status=quality_status,
            quality=quality,
        )
        if conflict:
            _conflict_response(record)
        if record is None:
            raise HTTPException(404, "research brief not found")
        return _brief_with_lineage(record)
    record = store.create_research_brief(
        project_id=project_id,
        source_conv_id=body.source_conv_id,
        title=title,
        objective=body.objective,
        content_markdown=answer,
        evidence=evidence,
        bibliography=bibliography,
        agent_trace=agent_trace,
        quality_status=quality_status,
        quality=quality,
    )
    if record is None:
        raise HTTPException(404, "project not found")
    return _brief_with_lineage(record)


@router.get("/research-briefs/{brief_id}")
def get_research_brief(brief_id: str):
    return _brief_with_lineage(_brief_or_404(brief_id))


@router.patch("/research-briefs/{brief_id}")
def update_research_brief(brief_id: str, body: ResearchBriefUpdateBody):
    current = _brief_or_404(brief_id)
    content_changed = body.content_markdown is not None or body.objective is not None
    quality = dict(current.get("quality") or {})
    quality_status: str | None = None
    if content_changed:
        quality.update(
            {
                "contract_version": 1,
                "edited_after_verification": True,
                "reasons": sorted({
                    *[str(item) for item in list(quality.get("reasons") or []) if str(item)],
                    "edited_after_verification",
                }),
            }
        )
        quality_status = "draft"
    record, conflict = get_chat_store().update_research_brief(
        brief_id,
        expected_revision=body.expected_revision,
        title=body.title,
        objective=body.objective,
        content_markdown=body.content_markdown,
        quality_status=quality_status,
        quality=quality if content_changed else None,
    )
    if conflict:
        _conflict_response(record)
    if record is None:
        raise HTTPException(404, "research brief not found")
    return _brief_with_lineage(record)


@router.post("/research-briefs/{brief_id}/update-plans")
def create_research_brief_update_plan(brief_id: str, body: ResearchBriefUpdatePlanCreateBody):
    record = _brief_or_404(brief_id)
    if int(record.get("revision") or 1) != int(body.expected_revision):
        _conflict_response(record)
    historical_matrix, current_matrix, lineage = _brief_update_matrices(record)
    impact = lineage.get("impact") if isinstance(lineage.get("impact"), dict) else {}
    plan = build_research_brief_update_plan(
        record,
        historical_matrix=historical_matrix,
        current_matrix=current_matrix,
        impact=impact,
        locale=body.locale,
        settings=get_settings(),
        max_tokens=body.max_tokens,
        model_generator=generate_grounded_answer,
    )
    fingerprint = matrix_contract_fingerprint(current_matrix)
    saved, conflict = get_chat_store().create_research_brief_update_plan(
        brief_id,
        expected_revision=body.expected_revision,
        matrix_id=str(current_matrix.get("id") or ""),
        matrix_revision=int(current_matrix.get("revision") or 1),
        matrix_fingerprint=fingerprint,
        payload=plan,
    )
    if conflict:
        _conflict_response(_brief_or_404(brief_id))
    if saved is None:
        raise HTTPException(404, "research brief not found")
    return saved


@router.get("/research-briefs/{brief_id}/update-plans/current")
def get_current_research_brief_update_plan(brief_id: str):
    _brief_or_404(brief_id)
    plan = get_chat_store().get_open_research_brief_update_plan(brief_id)
    if plan is None:
        raise HTTPException(404, "open research brief update plan not found")
    return plan


@router.delete("/research-briefs/{brief_id}/update-plans/{plan_id}")
def discard_research_brief_update_plan(brief_id: str, plan_id: str):
    _brief_or_404(brief_id)
    if not get_chat_store().set_research_brief_update_plan_status(
        brief_id,
        plan_id,
        status="discarded",
    ):
        raise HTTPException(404, "open research brief update plan not found")
    return {"ok": True}


@router.post("/research-briefs/{brief_id}/update-plans/{plan_id}/apply")
def apply_research_brief_update_plan(
    brief_id: str,
    plan_id: str,
    body: ResearchBriefUpdatePlanApplyBody,
):
    store = get_chat_store()
    current = _brief_or_404(brief_id)
    if int(current.get("revision") or 1) != int(body.expected_revision):
        _conflict_response(current)
    plan = store.get_research_brief_update_plan(brief_id, plan_id)
    if plan is None or str(plan.get("status") or "") != "open":
        raise HTTPException(404, "open research brief update plan not found")
    if int(plan.get("base_brief_revision") or 0) != int(body.expected_revision):
        raise HTTPException(409, "research brief update plan is stale")
    base_content = str(current.get("content_markdown") or "")
    if str(plan.get("base_content_hash") or "") != research_brief_content_hash(base_content):
        raise HTTPException(409, "research brief content changed after the update plan was created")
    matrix = store.get_evidence_matrix(str(plan.get("matrix_id") or ""))
    if matrix is None or str(matrix.get("project_id") or "") != str(current.get("project_id") or ""):
        raise HTTPException(409, "source evidence matrix is unavailable")
    if str(matrix.get("quality_status") or "") != "verified":
        raise HTTPException(409, "the source evidence matrix is no longer verified")
    if (
        int(matrix.get("revision") or 1) != int(plan.get("target_matrix_revision") or 0)
        or matrix_contract_fingerprint(matrix) != str(plan.get("matrix_fingerprint") or "")
    ):
        raise HTTPException(409, "source evidence matrix changed after the update plan was created")

    items = [item for item in list(plan.get("items") or []) if isinstance(item, dict)]
    item_ids = {str(item.get("id") or "") for item in items}
    decisions = {str(item.item_id): str(item.decision) for item in body.decisions}
    unknown = sorted(set(decisions) - item_ids)
    if unknown:
        raise HTTPException(400, "update decisions contain unknown change items: " + ", ".join(unknown[:8]))
    merged = apply_research_brief_update_decisions(base_content, items, decisions)
    answer = str(merged.get("content_markdown") or "")
    hits = stable_matrix_hits(
        [item for item in list(current.get("evidence") or []) if isinstance(item, dict)],
        matrix,
    )
    evidence = research_brief_evidence(hits)
    selected_items = [item for item in list(matrix.get("source_items") or []) if isinstance(item, dict)]
    bibliography = research_brief_bibliography(selected_items, evidence)
    verification_payload = verify_answer_citations(answer, hits, answer_mode="evidence_grounded")
    verification = verification_payload.get("verification") if isinstance(verification_payload.get("verification"), dict) else {}
    verification_passed = bool(
        str(verification.get("evidence_status") or "").strip().lower() == "grounded"
        and int(verification.get("total_claims") or 0) > 0
        and int(verification.get("unsupported_claims") or 0) == 0
        and float(verification.get("support_ratio") or 0.0) >= 0.999
    )
    rejected = list(merged.get("rejected_item_ids") or [])
    trace = {
        "mode": "research_agent",
        "question_type": "multi_paper_comparison",
        "context": {
            "query_scope": "basket",
            "answer_contract": "research_brief_incremental_update",
            "source_matrix_id": str(matrix.get("id") or ""),
            "source_matrix_revision": int(matrix.get("revision") or 1),
            "base_brief_revision": int(current.get("revision") or 1),
            "update_plan_id": str(plan.get("id") or ""),
        },
        "plan": items,
        "steps": [
            {
                "tool": "review_incremental_update",
                "status": "done",
                "observation": "Applied only accepted change items and preserved every unaffected Markdown span.",
                "output": {
                    "accepted_item_ids": list(merged.get("accepted_item_ids") or []),
                    "rejected_item_ids": rejected,
                },
                "error": "",
                "elapsed_ms": 0,
            },
            {
                "tool": "verify_answer_citations",
                "status": "done",
                "observation": str(verification_payload.get("observation") or ""),
                "output": verification,
                "error": "",
                "elapsed_ms": 0,
            },
        ],
        "verification": verification,
        "status": "done",
        "errors": [],
        "summary": {
            "query_scope": "basket",
            "quality_gate_status": "passed" if verification_passed else "failed",
            "quality_gate_warnings": [
                "incremental_update_rejected_changes" if rejected else "",
                "incremental_update_full_audit_failed" if not verification_passed else "",
                "incremental_extractive_fallback"
                if any(
                    "extractive_fallback" in list(item.get("generation_modes") or [])
                    for item in items
                )
                else "",
            ],
            **{
                key: verification.get(key)
                for key in (
                    "total_claims",
                    "supported_claims",
                    "unsupported_claims",
                    "support_ratio",
                    "evidence_status",
                )
            },
        },
    }
    quality_status, quality = research_brief_quality(
        answer=answer,
        agent_trace=trace,
        selected_items=selected_items,
        evidence=evidence,
    )
    if rejected:
        quality_status = "needs_review"
        quality["reasons"] = sorted(
            {
                *[str(item) for item in list(quality.get("reasons") or []) if str(item)],
                "incremental_update_rejected_changes",
            }
        )
    quality.update(
        {
            "source_matrix_id": str(matrix.get("id") or ""),
            "source_matrix_revision": int(matrix.get("revision") or 1),
            "source_matrix_quality_status": str(matrix.get("quality_status") or ""),
            "source_matrix_title": str(matrix.get("title") or ""),
            "source_matrix_fingerprint": matrix_contract_fingerprint(matrix),
            "incremental_update": {
                "contract_version": 1,
                "plan_id": str(plan.get("id") or ""),
                "base_brief_revision": int(current.get("revision") or 1),
                "source_matrix_revision": int(plan.get("source_matrix_revision") or 0),
                "target_matrix_revision": int(matrix.get("revision") or 1),
                "accepted_item_ids": list(merged.get("accepted_item_ids") or []),
                "rejected_item_ids": rejected,
                "base_content_hash": research_brief_content_hash(base_content),
                "generation": dict(plan.get("generation") or {}),
                "preservation": dict(plan.get("preservation") or {}),
            },
        }
    )
    record, conflict = store.update_research_brief(
        brief_id,
        expected_revision=body.expected_revision,
        content_markdown=answer,
        evidence=evidence,
        bibliography=bibliography,
        agent_trace=trace,
        quality_status=quality_status,
        quality=quality,
    )
    if conflict:
        _conflict_response(record)
    if record is None:
        raise HTTPException(404, "research brief not found")
    store.set_research_brief_update_plan_status(brief_id, plan_id, status="applied")
    return _brief_with_lineage(record)


@router.get("/research-briefs/{brief_id}/revisions")
def list_research_brief_revisions(brief_id: str, limit: int = Query(40, ge=1, le=200)):
    _brief_or_404(brief_id)
    return _briefs_with_lineage(get_chat_store().list_research_brief_revisions(brief_id, limit=limit))


@router.get("/research-briefs/{brief_id}/revisions/{revision}")
def get_research_brief_revision(brief_id: str, revision: int):
    _brief_or_404(brief_id)
    record = get_chat_store().get_research_brief_revision(brief_id, revision)
    if record is None:
        raise HTTPException(404, "research brief revision not found")
    return _brief_with_lineage(record)


@router.post("/research-briefs/{brief_id}/restore")
def restore_research_brief(brief_id: str, body: ResearchBriefRestoreBody):
    _brief_or_404(brief_id)
    record, conflict = get_chat_store().restore_research_brief_revision(
        brief_id,
        body.revision,
        expected_revision=body.expected_revision,
    )
    if conflict:
        _conflict_response(record)
    if record is None:
        raise HTTPException(404, "research brief revision not found")
    return _brief_with_lineage(record)


@router.delete("/research-briefs/{brief_id}")
def delete_research_brief(brief_id: str):
    record = _brief_or_404(brief_id)
    snapshot = create_auto_snapshot(
        get_settings(),
        action="research_brief_delete",
        label=brief_id,
        metadata={
            "brief_id": brief_id,
            "project_id": str(record.get("project_id") or ""),
            "revision": int(record.get("revision") or 1),
        },
    )
    if bool(snapshot.get("block_operation")):
        detail = str(snapshot.get("error") or snapshot.get("reason") or "automatic backup failed")
        raise HTTPException(503, f"automatic backup failed before research_brief_delete: {detail}")
    if not get_chat_store().delete_research_brief(brief_id):
        raise HTTPException(404, "research brief not found")
    return {"ok": True, "auto_backup": snapshot}


@router.get("/research-briefs/{brief_id}/export")
def export_research_brief(
    brief_id: str,
    format: Literal["markdown", "docx", "bibtex", "ris"] = Query("markdown"),
):
    record = _brief_with_lineage(_brief_or_404(brief_id))
    lineage = record.get("lineage") if isinstance(record.get("lineage"), dict) else {}
    if lineage and not bool(lineage.get("export_allowed", True)):
        reasons = ", ".join(str(item) for item in list(lineage.get("reasons") or []) if str(item))
        raise HTTPException(
            409,
            "research brief export blocked because its matrix lineage cannot be verified"
            + (f": {reasons}" if reasons else ""),
        )
    if format == "docx":
        return Response(
            content=research_brief_docx(record),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{_download_name(record, "docx")}"'},
        )
    if format == "bibtex":
        content = research_brief_bibtex(record)
        media_type = "application/x-bibtex; charset=utf-8"
        suffix = "bib"
    elif format == "ris":
        content = research_brief_ris(record)
        media_type = "application/x-research-info-systems; charset=utf-8"
        suffix = "ris"
    else:
        content = research_brief_markdown(record)
        media_type = "text/markdown; charset=utf-8"
        suffix = "md"
    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{_download_name(record, suffix)}"'},
    )
