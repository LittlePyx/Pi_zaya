from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from kb.evidence_watch import source_identity
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks
from kb.tokenize import tokenize


RESEARCH_GAP_CONTRACT_VERSION = 1
ACTIVE_RESEARCH_GAP_STATUSES = {"open", "in_progress"}
SEARCHABLE_RESEARCH_GAP_KINDS = {
    "missing_cell",
    "unsupported_cell",
    "comparison_not_comparable",
}

_FIELD_LABELS = {
    "method": "method architecture algorithm mechanism",
    "dataset_or_experiment": "dataset experiment evaluation protocol setup samples hardware",
    "metric": "metric quantitative evaluation PSNR SSIM LPIPS RMSE accuracy latency runtime",
    "key_result": "result performance comparison improvement ablation finding",
    "limitation": "limitation challenge failure drawback trade-off future work",
}
_BASE_PRIORITY = {
    "source_change": 88,
    "unsupported_cell": 84,
    "brief_stale": 76,
    "matrix_needs_review": 72,
    "comparison_not_comparable": 64,
    "missing_cell": 48,
}
_REFERENCE_HEADING_RE = re.compile(r"(?:^|[/ >])(?:references?|bibliography|works cited)(?:$|[/ >])", re.I)
_GENERIC_CANDIDATE_TOKENS = {
    "compare",
    "comparison",
    "imaging",
    "method",
    "paper",
    "result",
    "study",
}


def _text(value: object, *, limit: int = 1_200) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()[: max(0, int(limit))]


def _hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _priority(score: int) -> str:
    if score >= 85:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _row_label(row: dict[str, Any] | None) -> str:
    row = row or {}
    return _text(row.get("paper") or row.get("source_name") or row.get("source_path"), limit=500)


def _matrix_brief_impact(
    matrix: dict[str, Any],
    briefs: list[dict[str, Any]],
    *,
    source_path: str = "",
) -> dict[str, Any]:
    matrix_id = str(matrix.get("id") or "")
    identity = source_identity(source_path)
    affected: list[dict[str, Any]] = []
    citations: set[tuple[str, int]] = set()
    for brief in briefs:
        quality = brief.get("quality") if isinstance(brief.get("quality"), dict) else {}
        if str(quality.get("source_matrix_id") or "") != matrix_id:
            continue
        citation_numbers = sorted(
            {
                int(item.get("citation_number") or 0)
                for item in list(brief.get("evidence") or [])
                if isinstance(item, dict)
                and int(item.get("citation_number") or 0) > 0
                and (not identity or source_identity(item.get("source_path")) == identity)
            }
        )
        affected.append(
            {
                "brief_id": str(brief.get("id") or ""),
                "title": str(brief.get("title") or ""),
                "revision": int(brief.get("revision") or 1),
                "citation_numbers": citation_numbers,
            }
        )
        citations.update((str(brief.get("id") or ""), number) for number in citation_numbers)
    return {
        "affected_briefs": affected,
        "affected_brief_count": len(affected),
        "affected_citation_count": len(citations),
    }


def _candidate_query(
    matrix: dict[str, Any],
    *,
    field: str = "",
    row: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
) -> str:
    terms = [str(matrix.get("objective") or ""), _FIELD_LABELS.get(field, "")]
    if row:
        terms.append(str(row.get("paper") or row.get("source_name") or ""))
    spec = (audit or {}).get("input") if isinstance((audit or {}).get("input"), dict) else {}
    for dimension in list(spec.get("dimensions") or []):
        if not isinstance(dimension, dict):
            continue
        terms.extend([str(dimension.get("left_value") or ""), str(dimension.get("right_value") or "")])
    terms.extend(
        [
            str(spec.get("left_target") or ""),
            str(spec.get("right_target") or ""),
            str(spec.get("left_result") or ""),
            str(spec.get("right_result") or ""),
        ]
    )
    return _text(" ".join(term for term in terms if term), limit=1_000)


def _gap(
    *,
    project_id: str,
    kind: str,
    identity_parts: list[object],
    title: str,
    detail: str,
    matrix: dict[str, Any] | None = None,
    brief: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
    field: str = "",
    comparison_id: str = "",
    reasons: list[str] | None = None,
    impact: dict[str, Any] | None = None,
    candidate_query: str = "",
    severity: str = "warning",
    dismissible: bool = True,
) -> dict[str, Any]:
    matrix = matrix or {}
    brief = brief or {}
    row = row or {}
    impact_payload = dict(impact or {})
    score = int(_BASE_PRIORITY.get(kind, 40))
    score += min(12, int(impact_payload.get("affected_brief_count") or 0) * 4)
    score += min(10, int(impact_payload.get("affected_citation_count") or 0) * 2)
    score += min(8, int(impact_payload.get("affected_comparison_count") or 0) * 3)
    score = max(0, min(100, score))
    matrix_id = str(matrix.get("id") or "")
    gap_key = _hash([RESEARCH_GAP_CONTRACT_VERSION, project_id, kind, matrix_id, *identity_parts])
    return {
        "gap_key": gap_key,
        "contract_version": RESEARCH_GAP_CONTRACT_VERSION,
        "project_id": project_id,
        "kind": kind,
        "severity": severity,
        "priority": _priority(score),
        "priority_score": score,
        "title": _text(title, limit=500),
        "detail": _text(detail, limit=1_200),
        "matrix_id": matrix_id,
        "matrix_title": str(matrix.get("title") or ""),
        "matrix_revision": int(matrix.get("revision") or 0),
        "brief_id": str(brief.get("id") or ""),
        "brief_title": str(brief.get("title") or ""),
        "brief_revision": int(brief.get("revision") or 0),
        "row_id": str(row.get("id") or ""),
        "row_label": _row_label(row),
        "field": field,
        "comparison_id": comparison_id,
        "source_path": str(row.get("source_path") or ""),
        "source_name": str(row.get("source_name") or row.get("paper") or ""),
        "reasons": sorted({_text(reason, limit=240) for reason in list(reasons or []) if _text(reason, limit=240)}),
        "impact": impact_payload,
        "candidate_query": _text(candidate_query, limit=1_000),
        "candidate_searchable": kind in SEARCHABLE_RESEARCH_GAP_KINDS and bool(candidate_query),
        "dismissible": bool(dismissible),
    }


def build_project_research_gaps(
    *,
    project_id: str,
    matrices: list[dict[str, Any]],
    briefs: list[dict[str, Any]] | None = None,
    evidence_changes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pid = str(project_id or "").strip()
    active_briefs = [item for item in list(briefs or []) if isinstance(item, dict)]
    gaps: list[dict[str, Any]] = []
    for matrix in [item for item in list(matrices or []) if isinstance(item, dict)]:
        rows = [item for item in list(matrix.get("rows") or []) if isinstance(item, dict)]
        rows_by_id = {str(row.get("id") or ""): row for row in rows if str(row.get("id") or "")}
        quality = matrix.get("quality") if isinstance(matrix.get("quality"), dict) else {}
        audits = [item for item in list(matrix.get("comparison_audits") or []) if isinstance(item, dict)]
        comparison_by_row: dict[str, int] = {}
        for audit in audits:
            for key in ("left_row_id", "right_row_id"):
                row_id = str(audit.get(key) or "")
                if row_id:
                    comparison_by_row[row_id] = comparison_by_row.get(row_id, 0) + 1

        for kind, quality_key in (("missing_cell", "missing_cells"), ("unsupported_cell", "unsupported_cells")):
            for item in list(quality.get(quality_key) or []):
                if not isinstance(item, dict):
                    continue
                row_id = str(item.get("row_id") or "")
                field = str(item.get("field") or "")
                row = rows_by_id.get(row_id, {})
                impact = _matrix_brief_impact(matrix, active_briefs, source_path=str(row.get("source_path") or ""))
                impact["affected_comparison_count"] = int(comparison_by_row.get(row_id, 0))
                label = _row_label(row) or row_id
                gaps.append(
                    _gap(
                        project_id=pid,
                        kind=kind,
                        identity_parts=[row_id, field],
                        title=f"{label}: {field.replace('_', ' ')}",
                        detail=(
                            "The matrix has no source-grounded value for this field."
                            if kind == "missing_cell"
                            else "The current value does not pass the same-source evidence contract."
                        ),
                        matrix=matrix,
                        row=row,
                        field=field,
                        reasons=[quality_key],
                        impact=impact,
                        candidate_query=_candidate_query(matrix, field=field, row=row),
                        severity="error" if kind == "unsupported_cell" else "warning",
                    )
                )

        for audit in audits:
            if str(audit.get("status") or "") == "verified":
                continue
            audit_id = str(audit.get("id") or "")
            left_id = str(audit.get("left_row_id") or "")
            right_id = str(audit.get("right_row_id") or "")
            reasons = [str(item or "") for item in list(audit.get("reasons") or []) if str(item or "")]
            impact = _matrix_brief_impact(matrix, active_briefs)
            impact["affected_comparison_count"] = 1
            gaps.append(
                _gap(
                    project_id=pid,
                    kind="comparison_not_comparable",
                    identity_parts=[audit_id or _hash(audit.get("input") or {})],
                    title=f"Comparison needs evidence: {_row_label(rows_by_id.get(left_id))} / {_row_label(rows_by_id.get(right_id))}",
                    detail="The saved comparison did not pass the complete task, dataset, protocol, metric, target, value, and evidence contract.",
                    matrix=matrix,
                    comparison_id=audit_id,
                    reasons=reasons,
                    impact=impact,
                    candidate_query=_candidate_query(matrix, audit=audit),
                )
            )

        represented_reasons = {"unsupported_cells", "no_rows", "selected_sources_without_rows", "selected_sources_without_evidence", "unexpected_sources", "no_supported_cells"}
        matrix_reasons = [str(item or "") for item in list(quality.get("reasons") or []) if str(item or "") in represented_reasons]
        if list(quality.get("unsupported_cells") or []):
            matrix_reasons = [reason for reason in matrix_reasons if reason != "unsupported_cells"]
        if str(matrix.get("quality_status") or "") != "verified" and matrix_reasons:
            impact = _matrix_brief_impact(matrix, active_briefs)
            impact["affected_comparison_count"] = len(audits)
            gaps.append(
                _gap(
                    project_id=pid,
                    kind="matrix_needs_review",
                    identity_parts=["matrix_quality"],
                    title=f"Matrix needs review: {matrix.get('title') or matrix.get('id')}",
                    detail="The matrix cannot currently be used as verified synthesis evidence.",
                    matrix=matrix,
                    reasons=matrix_reasons,
                    impact=impact,
                    candidate_query=_candidate_query(matrix),
                    severity="error",
                )
            )

    matrix_by_id = {str(item.get("id") or ""): item for item in matrices if isinstance(item, dict)}
    for brief in active_briefs:
        lineage = brief.get("lineage") if isinstance(brief.get("lineage"), dict) else {}
        status = str(lineage.get("status") or "untracked")
        matrix_id = str(lineage.get("source_matrix_id") or "")
        if not matrix_id or status in {"current", "current_equivalent"}:
            continue
        matrix = matrix_by_id.get(matrix_id, {"id": matrix_id, "title": lineage.get("source_matrix_title") or ""})
        impact = dict(lineage.get("impact") or {}) if isinstance(lineage.get("impact"), dict) else {}
        impact.setdefault("affected_brief_count", 1)
        impact.setdefault("affected_citation_count", len(list(impact.get("affected_citation_numbers") or [])))
        gaps.append(
            _gap(
                project_id=pid,
                kind="brief_stale",
                identity_parts=[str(brief.get("id") or ""), status],
                title=f"Brief needs review: {brief.get('title') or brief.get('id')}",
                detail=f"The brief's matrix lineage is {status}; its historical evidence must stay visible until reviewed.",
                matrix=matrix,
                brief=brief,
                reasons=[str(item or "") for item in list(lineage.get("reasons") or [])],
                impact=impact,
                severity="error" if status in {"matrix_missing", "integrity_mismatch", "revision_mismatch"} else "warning",
            )
        )

    for event in [item for item in list(evidence_changes or []) if isinstance(item, dict)]:
        matrix = matrix_by_id.get(
            str(event.get("matrix_id") or ""),
            {"id": event.get("matrix_id"), "title": event.get("matrix_title"), "revision": event.get("matrix_revision")},
        )
        impact = dict(event.get("impact") or {}) if isinstance(event.get("impact"), dict) else {}
        impact["affected_comparison_count"] = len(list(impact.get("affected_comparison_ids") or []))
        gaps.append(
            _gap(
                project_id=pid,
                kind="source_change",
                identity_parts=[str(event.get("event_key") or event.get("id") or "")],
                title=f"Source change: {event.get('source_name') or event.get('source_path')}",
                detail=f"The evidence source reports {event.get('kind')}; review it in the matrix change inbox.",
                matrix=matrix,
                reasons=[str(event.get("kind") or "")],
                impact=impact,
                severity=str(event.get("severity") or "warning"),
                dismissible=False,
            )
        )

    gaps.sort(
        key=lambda item: (
            -int(item.get("priority_score") or 0),
            str(item.get("matrix_title") or "").casefold(),
            str(item.get("row_label") or item.get("brief_title") or item.get("title") or "").casefold(),
            str(item.get("field") or ""),
        )
    )
    return gaps


def research_gap_summary(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    active = [item for item in list(gaps or []) if isinstance(item, dict)]
    return {
        "total": len(active),
        "open": sum(1 for item in active if str(item.get("status") or "open") == "open"),
        "in_progress": sum(1 for item in active if str(item.get("status") or "") == "in_progress"),
        "high": sum(1 for item in active if str(item.get("priority") or "") == "high"),
        "medium": sum(1 for item in active if str(item.get("priority") or "") == "medium"),
        "low": sum(1 for item in active if str(item.get("priority") or "") == "low"),
        "searchable": sum(1 for item in active if bool(item.get("candidate_searchable"))),
        "affected_matrix_count": len({str(item.get("matrix_id") or "") for item in active if str(item.get("matrix_id") or "")}),
        "affected_brief_count": len({str(item.get("brief_id") or "") for item in active if str(item.get("brief_id") or "")}),
    }


def find_research_gap_candidates(
    gap: dict[str, Any],
    *,
    db_dir: str | Path,
    excluded_source_paths: list[str] | None = None,
    limit: int = 5,
    chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    query = _text(gap.get("candidate_query"), limit=1_000)
    if not query or not bool(gap.get("candidate_searchable")):
        return []
    corpus = [item for item in (chunks if chunks is not None else load_all_chunks(Path(db_dir))) if isinstance(item, dict)]
    top_k = max(20, min(80, int(limit or 5) * 10))
    hits = BM25Retriever(corpus).search(query, top_k=top_k)
    excluded = {source_identity(item) for item in list(excluded_source_paths or []) if source_identity(item)}
    query_tokens = {str(item).casefold() for item in tokenize(query) if str(item).strip()}
    if not hits:
        # BM25's IDF can be non-positive in very small, repetitive corpora.  A
        # strict lexical fallback recovers only chunks with at least two
        # meaningful query terms; it never emits arbitrary zero-match chunks.
        meaningful_query_tokens = query_tokens.difference(_GENERIC_CANDIDATE_TOKENS)
        fallback_hits: list[dict[str, Any]] = []
        for index, chunk in enumerate(corpus):
            meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
            if meta.get("evidence_ready") is False:
                continue
            text_value = str(chunk.get("text") or "")
            chunk_tokens = {str(item).casefold() for item in tokenize(text_value) if str(item).strip()}
            overlap_count = len(meaningful_query_tokens.intersection(chunk_tokens))
            if overlap_count < 2:
                continue
            fallback_hits.append(
                {
                    "score": float(overlap_count),
                    "id": chunk.get("id", str(index)),
                    "text": text_value,
                    "meta": dict(meta),
                }
            )
        hits = sorted(fallback_hits, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("id") or "")))[:top_k]
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for hit in hits:
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        source_path = str(meta.get("source_path") or "").strip()
        identity = source_identity(source_path)
        heading = _text(meta.get("heading_path") or meta.get("top_heading"), limit=800)
        quote = _text(hit.get("text"), limit=900)
        if not identity or identity in excluded or identity in seen or not quote:
            continue
        if _REFERENCE_HEADING_RE.search(heading):
            continue
        seen.add(identity)
        hit_tokens = {str(item).casefold() for item in tokenize(quote) if str(item).strip()}
        overlap = sorted(query_tokens.intersection(hit_tokens))[:16]
        chunk_id = str(hit.get("id") or meta.get("chunk_id") or "")
        candidate_id = _hash([str(gap.get("gap_key") or ""), identity, chunk_id])
        candidates.append(
            {
                "id": candidate_id,
                "gap_id": str(gap.get("id") or ""),
                "gap_key": str(gap.get("gap_key") or ""),
                "source_path": source_path,
                "source_name": Path(source_path).name,
                "title": Path(source_path).stem,
                "chunk_id": chunk_id,
                "score": round(float(hit.get("score") or 0.0), 6),
                "evidence_quote": quote,
                "heading_path": heading,
                "location_label": _text(meta.get("location_label") or heading, limit=500),
                "page_start": meta.get("page_start") or meta.get("page") or None,
                "page_end": meta.get("page_end") or meta.get("page") or None,
                "block_id": str(meta.get("block_id") or ""),
                "anchor_id": str(meta.get("anchor_id") or meta.get("anchor") or ""),
                "matched_terms": overlap,
                "match_reason": "Local indexed passage shares the deterministic gap query terms.",
            }
        )
        if len(candidates) >= max(1, min(12, int(limit or 5))):
            break
    return candidates
