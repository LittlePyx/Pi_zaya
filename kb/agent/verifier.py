from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .types import AgentVerification, EvidenceStatus


_CITATION_RE = re.compile(r"(?:\[[0-9][0-9,\-\s]*\]|\[\[CITE:[^\]]+\]\])")
_NUMERIC_CITATION_RE = re.compile(r"(?<!\[)\[(\d+(?:\s*(?:,|-)\s*\d+)*)\](?!\])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,}")
_SOURCE_NOTICE_RE = re.compile(
    r"^(?:Note:\s*(?:local citations|no matching local knowledge-base evidence)|"
    r"(?:\u6ce8\u610f|\u6ce8)[:\uff1a]\s*(?:\u5e26\s*\[n\]|\u672c\u5730\u77e5\u8bc6\u5e93|local))",
    flags=re.IGNORECASE,
)
_HYBRID_NOTICE_RE = re.compile(
    r"^(?:Note:\s*local citations|(?:\u6ce8\u610f|\u6ce8)[:\uff1a]\s*\u5e26\s*\[n\])",
    flags=re.IGNORECASE,
)
_EXTERNAL_BACKGROUND_PREFIX_RE = re.compile(
    r"^(?:#{1,6}\s*)?"
    r"(?:external\s+(?:context|background)|general\s+background|background|broader\s+context|context|"
    r"\u5916\u90e8\u8865\u5145|\u5916\u90e8\u80cc\u666f|\u901a\u7528\u80cc\u666f|\u80cc\u666f|\u8865\u5145\u8bf4\u660e)"
    r"\s*[:\uff1a]",
    flags=re.IGNORECASE,
)
_GENERAL_BACKGROUND_RE = re.compile(
    r"\b(?:generally|in general|typically|often|usually|background|broader literature|"
    r"outside the local|external context|common pattern|commonly)\b"
    r"|(?:\u4e00\u822c\u6765\u8bf4|\u901a\u5e38|\u5e38\u89c1|\u5b66\u672f\u4e0a|\u5916\u90e8|\u80cc\u666f|\u8865\u5145)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ClassifiedAnswerClaim:
    text: str
    kind: str
    reason: str = ""


def _clean_answer_part(part: str) -> str:
    clean = _BULLET_PREFIX_RE.sub("", str(part or "").strip())
    clean = re.sub(r"^\s*#{1,6}\s*", "", clean).strip()
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean.strip("`*_ ")


def _candidate_answer_parts(answer: str) -> list[str]:
    text = str(answer or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        clean = _clean_answer_part(part)
        if not clean:
            continue
        if _SOURCE_NOTICE_RE.search(clean):
            chunks.append(clean)
            continue
        if len(clean) < 12:
            continue
        if clean.endswith((":", "\uff1a")) and len(clean) < 80:
            continue
        chunks.append(clean)
    return chunks


def _hybrid_external_allowed(answer: str, *, answer_mode: str = "") -> bool:
    if str(answer_mode or "").strip() == "hybrid_local_external":
        return True
    return any(_HYBRID_NOTICE_RE.search(part) for part in _candidate_answer_parts(answer))


def _looks_like_external_background(claim: str, *, hybrid_external_allowed: bool) -> bool:
    if not hybrid_external_allowed or _CITATION_RE.search(claim):
        return False
    text = str(claim or "").strip()
    return bool(_EXTERNAL_BACKGROUND_PREFIX_RE.search(text) or _GENERAL_BACKGROUND_RE.search(text))


def classify_answer_claims(answer: str, *, answer_mode: str = "") -> list[ClassifiedAnswerClaim]:
    """Classify answer sentences by verification scope.

    Local paper claims remain citation-checked. External background is tracked,
    but it does not lower local evidence support ratios in hybrid answers.
    """
    hybrid_allowed = _hybrid_external_allowed(answer, answer_mode=answer_mode)
    classified: list[ClassifiedAnswerClaim] = []
    for part in _candidate_answer_parts(answer):
        if _SOURCE_NOTICE_RE.search(part):
            classified.append(ClassifiedAnswerClaim(text=part, kind="source_notice", reason="source_disclosure"))
        elif _looks_like_external_background(part, hybrid_external_allowed=hybrid_allowed):
            classified.append(ClassifiedAnswerClaim(text=part, kind="external_background", reason="hybrid_external_context"))
        else:
            classified.append(ClassifiedAnswerClaim(text=part, kind="local_claim", reason="local_evidence_required"))
    return classified


def split_answer_claims(answer: str) -> list[str]:
    return [item.text for item in classify_answer_claims(answer) if item.kind != "source_notice"]


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


def _citation_numbers(claim: str, *, limit: int = 24) -> list[int]:
    numbers: list[int] = []
    seen: set[int] = set()
    for match in _NUMERIC_CITATION_RE.finditer(str(claim or "")):
        for part in re.split(r"\s*,\s*", match.group(1)):
            raw = part.strip()
            if "-" in raw:
                start_raw, end_raw = [item.strip() for item in raw.split("-", 1)]
                try:
                    start = int(start_raw)
                    end = int(end_raw)
                except ValueError:
                    continue
                if start <= 0 or end < start or end - start >= limit:
                    continue
                candidates = range(start, end + 1)
            else:
                try:
                    candidates = (int(raw),)
                except ValueError:
                    continue
            for number in candidates:
                if number <= 0 or number in seen:
                    continue
                seen.add(number)
                numbers.append(number)
                if len(numbers) >= limit:
                    return numbers
    return numbers


def _matched_cited_evidence_sources(
    claim: str,
    evidence_hits: list[dict[str, Any]],
    citation_numbers: list[int],
) -> list[dict[str, Any]]:
    claim_terms = _claim_terms(claim)
    if not claim_terms:
        return []
    matches: list[dict[str, Any]] = []
    for citation_number in citation_numbers:
        hit_index = citation_number - 1
        if hit_index < 0 or hit_index >= len(evidence_hits):
            continue
        hit = evidence_hits[hit_index]
        if not isinstance(hit, dict):
            continue
        text = str(hit.get("text") or "")
        hit_terms = _claim_terms(text)
        overlap = len(claim_terms & hit_terms)
        if overlap >= 2:
            summary = _source_summary(hit, overlap_terms=overlap)
            summary["citation_index"] = citation_number
            matches.append(summary)
    matches.sort(key=lambda item: (int(item.get("overlap_terms") or 0), float(item.get("score") or 0.0)), reverse=True)
    return matches[: max(1, min(3, len(matches)))]


def _unsupported_reason(
    *,
    citation_present: bool,
    citation_numbers: list[int],
    bound_citation_count: int,
    matched_evidence_count: int,
    hit_count: int,
) -> str:
    if not citation_present:
        return "missing_citation"
    if hit_count <= 0:
        return "no_evidence_hits"
    if not citation_numbers:
        return "unbound_citation"
    if bound_citation_count < len(citation_numbers):
        return "citation_index_out_of_range"
    if matched_evidence_count <= 0:
        return "citation_evidence_mismatch"
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


def verify_answer_citations(
    answer: str,
    evidence_hits: list[dict[str, Any]] | None = None,
    *,
    answer_mode: str = "",
) -> AgentVerification:
    hits = [h for h in list(evidence_hits or []) if isinstance(h, dict)]
    classified_claims = classify_answer_claims(answer, answer_mode=answer_mode)
    claim_rows: list[dict[str, Any]] = []
    supported = 0
    source_notice_count = len([item for item in classified_claims if item.kind == "source_notice"])
    external_background_claims = len([item for item in classified_claims if item.kind == "external_background"])
    local_claim_index = 0
    for item in classified_claims:
        if item.kind == "source_notice":
            continue
        claim = item.text
        row_index = len(claim_rows) + 1
        if item.kind == "external_background":
            claim_rows.append(
                {
                    "index": row_index,
                    "text": claim[:280],
                    "claim_text": claim[:280],
                    "claim_kind": "external_background",
                    "verification_scope": "external_background",
                    "classification_reason": item.reason,
                    "has_citation": False,
                    "citation_present": False,
                    "has_evidence_overlap": False,
                    "matched_evidence_count": 0,
                    "matched_sources": [],
                    "supported": None,
                    "unsupported_reason": "",
                }
            )
            continue
        local_claim_index += 1
        citation_present = bool(_CITATION_RE.search(claim))
        citation_numbers = _citation_numbers(claim)
        bound_citation_numbers = [number for number in citation_numbers if number <= len(hits)]
        unresolved_citation_numbers = [number for number in citation_numbers if number > len(hits)]
        matched_sources = _matched_cited_evidence_sources(claim, hits, bound_citation_numbers) if hits else []
        matched_evidence_count = len(matched_sources)
        overlap = matched_evidence_count > 0
        is_supported = bool(
            citation_present
            and citation_numbers
            and not unresolved_citation_numbers
            and matched_evidence_count > 0
        )
        if not citation_present:
            citation_binding = "none"
        elif not citation_numbers:
            citation_binding = "unbound"
        elif unresolved_citation_numbers and matched_evidence_count > 0:
            citation_binding = "partial"
        elif unresolved_citation_numbers:
            citation_binding = "unresolved"
        elif matched_evidence_count > 0:
            citation_binding = "bound"
        else:
            citation_binding = "mismatch"
        unsupported_reason = "" if is_supported else _unsupported_reason(
            citation_present=citation_present,
            citation_numbers=citation_numbers,
            bound_citation_count=len(bound_citation_numbers),
            matched_evidence_count=matched_evidence_count,
            hit_count=len(hits),
        )
        if is_supported:
            supported += 1
        claim_rows.append(
            {
                "index": row_index,
                "local_claim_index": local_claim_index,
                "text": claim[:280],
                "claim_text": claim[:280],
                "claim_kind": "local_claim",
                "verification_scope": "local_evidence",
                "classification_reason": item.reason,
                "has_citation": citation_present,
                "citation_present": citation_present,
                "citation_numbers": citation_numbers,
                "bound_citation_numbers": bound_citation_numbers,
                "unresolved_citation_numbers": unresolved_citation_numbers,
                "citation_binding": citation_binding,
                "has_evidence_overlap": overlap,
                "matched_evidence_count": matched_evidence_count,
                "matched_sources": matched_sources,
                "supported": is_supported,
                "unsupported_reason": unsupported_reason,
            }
        )
    local_claims = local_claim_index
    total = local_claims
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
        local_claims=local_claims,
        external_background_claims=external_background_claims,
        source_notice_count=source_notice_count,
        support_ratio=ratio,
        evidence_status=evidence_status,
        evidence_hit_count=len(hits),
        evidence_status_reasons=evidence_status_reasons,
        claims=claim_rows,
    )
