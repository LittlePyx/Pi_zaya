from __future__ import annotations

import re
from typing import Any

from .types import AgentVerification, EvidenceStatus


_CITATION_RE = re.compile(r"(?:\[[0-9][0-9,\-\s]*\]|\[\[CITE:[^\]]+\]\])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,}")


def split_answer_claims(answer: str) -> list[str]:
    text = str(answer or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        clean = _BULLET_PREFIX_RE.sub("", str(part or "").strip())
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) < 12:
            continue
        if clean.endswith(":") and len(clean) < 80:
            continue
        chunks.append(clean)
    return chunks


def _claim_terms(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(str(text or ""))}


def _source_summary(hit: dict[str, Any], *, overlap_terms: int) -> dict[str, Any]:
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    try:
        score = float(hit.get("score") or 0.0)
    except Exception:
        score = 0.0
    return {
        "source_name": str(meta.get("source_name") or meta.get("title") or "").strip(),
        "source_path": str(meta.get("source_path") or "").strip(),
        "heading_path": str(meta.get("heading_path") or meta.get("top_heading") or "").strip(),
        "score": score,
        "overlap_terms": int(overlap_terms),
        "evidence_preview": str(hit.get("text") or "").strip()[:220],
    }


def _matched_evidence_sources(claim: str, evidence_hits: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    claim_terms = _claim_terms(claim)
    if not claim_terms:
        return []
    matches: list[dict[str, Any]] = []
    for hit in evidence_hits[:8]:
        if not isinstance(hit, dict):
            continue
        text = str(hit.get("text") or "")
        hit_terms = _claim_terms(text)
        overlap = len(claim_terms & hit_terms)
        if overlap >= 2:
            matches.append(_source_summary(hit, overlap_terms=overlap))
    matches.sort(key=lambda item: (int(item.get("overlap_terms") or 0), float(item.get("score") or 0.0)), reverse=True)
    return matches[: max(1, int(limit or 3))]


def _unsupported_reason(*, citation_present: bool, matched_evidence_count: int, hit_count: int) -> str:
    if not citation_present:
        return "missing_citation"
    if hit_count <= 0:
        return "no_evidence_hits"
    if matched_evidence_count <= 0:
        return "missing_evidence_overlap"
    return ""


def assess_evidence_status(
    *,
    evidence_hit_count: int,
    total_claims: int,
    supported_claims: int,
    unsupported_claims: int,
    support_ratio: float,
) -> tuple[EvidenceStatus, list[str]]:
    """Classify answer grounding with simple, explainable thresholds."""
    hit_count = max(0, int(evidence_hit_count or 0))
    total = max(0, int(total_claims or 0))
    supported = max(0, int(supported_claims or 0))
    unsupported = max(0, int(unsupported_claims or 0))
    ratio = max(0.0, min(1.0, float(support_ratio or 0.0)))
    reasons: list[str] = []

    if hit_count <= 0:
        reasons.append("no_evidence_hits")
    if total <= 0:
        reasons.append("no_checkable_claims")
    if total > 0 and supported <= 0:
        reasons.append("no_supported_claims")
    if total > 0 and ratio < 0.5:
        reasons.append("low_support_ratio")

    if "no_evidence_hits" in reasons or "no_supported_claims" in reasons or "low_support_ratio" in reasons:
        return "insufficient", reasons

    if unsupported > 0:
        reasons.append("unsupported_claims")
    if hit_count < 2:
        reasons.append("low_evidence_count")
    if total <= 0:
        return "needs_review", reasons
    if reasons:
        return "needs_review", reasons
    return "grounded", []


def verify_answer_citations(answer: str, evidence_hits: list[dict[str, Any]] | None = None) -> AgentVerification:
    hits = [h for h in list(evidence_hits or []) if isinstance(h, dict)]
    claims = split_answer_claims(answer)
    claim_rows: list[dict[str, Any]] = []
    supported = 0
    for idx, claim in enumerate(claims, start=1):
        citation_present = bool(_CITATION_RE.search(claim))
        matched_sources = _matched_evidence_sources(claim, hits) if hits else []
        matched_evidence_count = len(matched_sources)
        overlap = matched_evidence_count > 0
        is_supported = bool(citation_present and matched_evidence_count > 0)
        unsupported_reason = "" if is_supported else _unsupported_reason(
            citation_present=citation_present,
            matched_evidence_count=matched_evidence_count,
            hit_count=len(hits),
        )
        if is_supported:
            supported += 1
        claim_rows.append(
            {
                "index": idx,
                "text": claim[:280],
                "claim_text": claim[:280],
                "has_citation": citation_present,
                "citation_present": citation_present,
                "has_evidence_overlap": overlap,
                "matched_evidence_count": matched_evidence_count,
                "matched_sources": matched_sources,
                "supported": is_supported,
                "unsupported_reason": unsupported_reason,
            }
        )
    total = len(claim_rows)
    unsupported = max(0, total - supported)
    ratio = round((supported / total), 4) if total else 0.0
    evidence_status, evidence_status_reasons = assess_evidence_status(
        evidence_hit_count=len(hits),
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        support_ratio=ratio,
    )
    return AgentVerification(
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        support_ratio=ratio,
        evidence_status=evidence_status,
        evidence_hit_count=len(hits),
        evidence_status_reasons=evidence_status_reasons,
        claims=claim_rows,
    )
