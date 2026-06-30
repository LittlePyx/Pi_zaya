from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .types import (
    EvidenceMatrixRow,
    EvidenceStatus,
    QuestionType,
    ResearchRun,
    ResearchSubtask,
    SourcePolicy,
)


_LIMITATION_RE = re.compile(
    r"\b(limitations?|challenges?|weakness(?:es)?|failure|fails?|future\s+work|open\s+problem)\b",
    flags=re.IGNORECASE,
)
_EXPERIMENT_RE = re.compile(
    r"\b(experiments?|results?|datasets?|ablation|metrics?|benchmark|evaluation|figure|table)\b",
    flags=re.IGNORECASE,
)


def _clip(value: Any, limit: int = 220) -> str:
    if isinstance(value, (list, tuple, set)):
        value = " / ".join(str(item or "").strip() for item in value if str(item or "").strip())
    text = " ".join(str(value or "").split()).strip()
    return text[: max(20, int(limit or 220))]


def _meta(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    return meta if isinstance(meta, dict) else {}


def _source_name(hit: dict[str, Any]) -> str:
    meta = _meta(hit)
    source_path = str(meta.get("source_path") or "").strip()
    return (
        str(meta.get("source_name") or meta.get("title") or "").strip()
        or Path(source_path).name
        or str(hit.get("id") or "").strip()
    )


def _source_path(hit: dict[str, Any]) -> str:
    return str(_meta(hit).get("source_path") or "").strip()


def _heading(hit: dict[str, Any]) -> str:
    meta = _meta(hit)
    return str(meta.get("heading_path") or meta.get("top_heading") or "").strip()


def _first_sentence(value: Any, limit: int = 220) -> str:
    text = _clip(value, limit=600)
    if not text:
        return ""
    match = re.search(r"(.+?[.!?\u3002\uff01\uff1f])(?:\s|$)", text)
    return _clip(match.group(1) if match else text, limit=limit)


def _answer_mode(agent_notes: dict[str, Any] | None) -> str:
    if not isinstance(agent_notes, dict):
        return ""
    gate = agent_notes.get("evidence_gate")
    if not isinstance(gate, dict):
        return ""
    return str(gate.get("answer_mode") or "").strip()


def infer_source_policy(
    *,
    hits: list[dict[str, Any]],
    agent_notes: dict[str, Any] | None,
) -> SourcePolicy:
    mode = _answer_mode(agent_notes)
    if mode == "hybrid_local_external":
        return "local_plus_external_background"
    if mode in {"external_academic_llm", "general_llm"} or not hits:
        return "external_allowed_with_notice"
    return "local_only"


def _support_status(has_quote: bool, verification_status: EvidenceStatus) -> EvidenceStatus:
    if verification_status == "grounded" and has_quote:
        return "grounded"
    if has_quote:
        return "needs_review"
    return "insufficient"


def _comparison_rows(
    comparisons: list[dict[str, Any]],
    *,
    verification_status: EvidenceStatus,
) -> list[EvidenceMatrixRow]:
    rows: list[EvidenceMatrixRow] = []
    for idx, item in enumerate(comparisons[:8], start=1):
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        first_evidence = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
        quote = _clip(first_evidence.get("text_preview") or first_evidence.get("evidence_preview") or "", 240)
        source_name = str(item.get("source_name") or item.get("paper") or "").strip()
        rows.append(
            EvidenceMatrixRow(
                paper=str(item.get("paper") or source_name or "Source").strip(),
                source_name=source_name,
                source_path=str(item.get("source_path") or "").strip(),
                method=_clip(item.get("method"), 180),
                dataset_or_experiment=_clip(first_evidence.get("heading_path") or item.get("supporting_headings"), 160),
                key_result=_clip(item.get("relation_to_question"), 180),
                limitation=_clip(item.get("limitation"), 180),
                evidence_quote=quote,
                citation=f"[{idx}]",
                heading_path=_clip(first_evidence.get("heading_path") or "", 180),
                support_status=_support_status(bool(quote), verification_status),
            )
        )
    return rows


def _hit_rows(
    hits: list[dict[str, Any]],
    *,
    verification_status: EvidenceStatus,
) -> list[EvidenceMatrixRow]:
    rows: list[EvidenceMatrixRow] = []
    seen: set[str] = set()
    for hit in hits[:12]:
        if not isinstance(hit, dict):
            continue
        source = _source_name(hit)
        path = _source_path(hit)
        heading = _heading(hit)
        key = f"{path}|{source}"
        if key in seen:
            continue
        seen.add(key)
        text = str(hit.get("text") or "")
        limitation = _first_sentence(text, 160) if _LIMITATION_RE.search(text) else ""
        experiment = heading if _EXPERIMENT_RE.search(" ".join([heading, text])) else heading
        quote = _first_sentence(text, 260)
        rows.append(
            EvidenceMatrixRow(
                paper=source or "Source",
                source_name=source,
                source_path=path,
                method=quote,
                dataset_or_experiment=_clip(experiment, 160),
                key_result=quote,
                limitation=limitation or "Not identified in retrieved evidence.",
                evidence_quote=quote,
                citation=f"[{len(rows) + 1}]",
                heading_path=heading,
                support_status=_support_status(bool(quote), verification_status),
            )
        )
        if len(rows) >= 8:
            break
    return rows


def _subtasks(
    *,
    question_type: QuestionType,
    hits: list[dict[str, Any]],
    agent_notes: dict[str, Any] | None,
    verification_status: EvidenceStatus,
) -> list[ResearchSubtask]:
    tasks = [
        ResearchSubtask(
            goal="Classify task and apply the selected research scope.",
            status="done",
            tool="planner",
            observation=f"Planned {question_type}.",
        ),
        ResearchSubtask(
            goal="Retrieve local evidence from the indexed library.",
            status="done" if hits else "skipped",
            tool="retrieve_evidence",
            observation=f"{len(hits)} usable local evidence hit(s).",
        ),
    ]
    if question_type == "multi_paper_comparison":
        tasks.append(
            ResearchSubtask(
                goal="Extract per-paper method, result, limitation, and evidence cells.",
                status="done" if isinstance((agent_notes or {}).get("comparisons"), list) or hits else "skipped",
                tool="compare_papers",
                observation="Prepared comparison-oriented evidence matrix rows.",
            )
        )
    elif question_type == "reference_followup":
        tasks.append(
            ResearchSubtask(
                goal="Resolve upstream references from retrieved citing-paper evidence.",
                status="done" if isinstance((agent_notes or {}).get("references"), list) else "skipped",
                tool="retrieve_references",
                observation="Reference evidence is available when local bibliography entries resolve.",
            )
        )
    elif question_type == "reading_guide":
        tasks.append(
            ResearchSubtask(
                goal="Convert retrieved sections into reading waypoints.",
                status="done" if isinstance((agent_notes or {}).get("guide"), list) else "skipped",
                tool="build_reading_guide",
                observation="Reading waypoints are available when section evidence is retrieved.",
            )
        )
    tasks.append(
        ResearchSubtask(
            goal="Verify local citation support and disclose non-local context.",
            status="done" if verification_status != "not_applicable" else "skipped",
            tool="verify_answer_citations",
            observation=f"Verification status: {verification_status}.",
        )
    )
    return tasks


def build_research_run(
    query: str,
    *,
    question_type: QuestionType,
    hits: list[dict[str, Any]],
    agent_notes: dict[str, Any] | None = None,
    scope_context: dict[str, Any] | None = None,
    verification_status: EvidenceStatus = "insufficient",
    failed: bool = False,
) -> ResearchRun:
    local_hits = [hit for hit in list(hits or []) if isinstance(hit, dict)]
    comparisons = []
    if isinstance(agent_notes, dict) and isinstance(agent_notes.get("comparisons"), list):
        comparisons = [item for item in list(agent_notes.get("comparisons") or []) if isinstance(item, dict)]
    matrix = _comparison_rows(comparisons, verification_status=verification_status) if comparisons else _hit_rows(
        local_hits,
        verification_status=verification_status,
    )
    source_policy = infer_source_policy(hits=local_hits, agent_notes=agent_notes)
    subtasks = _subtasks(
        question_type=question_type,
        hits=local_hits,
        agent_notes=agent_notes,
        verification_status=verification_status,
    )
    run_seed = "|".join(
        [
            str(query or "").strip(),
            str(question_type),
            str((scope_context or {}).get("query_scope") or ""),
            str(len(local_hits)),
            str(len(matrix)),
        ]
    )
    run_id = f"rr_{hashlib.sha1(run_seed.encode('utf-8', errors='ignore')).hexdigest()[:12]}"
    status = "failed" if failed else "verified"
    metrics = {
        "subtask_count": len(subtasks),
        "evidence_matrix_rows": len(matrix),
        "source_count": len({row.source_path or row.source_name for row in matrix if row.source_path or row.source_name}),
        "local_evidence_hit_count": len(local_hits),
    }
    return ResearchRun(
        run_id=run_id,
        status=status,
        source_policy=source_policy,
        query_scope=str((scope_context or {}).get("query_scope") or ""),
        question=_clip(query, 500),
        subtasks=subtasks,
        evidence_matrix=matrix,
        metrics=metrics,
    )
