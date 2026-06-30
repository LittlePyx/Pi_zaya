from __future__ import annotations

import re
from typing import Any

from .types import AgentVerification


_CITATION_RE = re.compile(r"(?:\[[0-9][0-9,\-\s]*\]|\[\[CITE:[^\]]+\]\])")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


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


def _has_evidence_overlap(claim: str, evidence_hits: list[dict[str, Any]]) -> bool:
    claim_terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,}", claim)
    }
    if not claim_terms:
        return False
    for hit in evidence_hits[:8]:
        text = str((hit or {}).get("text") or "")
        hit_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,}", text)
        }
        if len(claim_terms & hit_terms) >= 2:
            return True
    return False


def verify_answer_citations(answer: str, evidence_hits: list[dict[str, Any]] | None = None) -> AgentVerification:
    hits = [h for h in list(evidence_hits or []) if isinstance(h, dict)]
    claims = split_answer_claims(answer)
    claim_rows: list[dict[str, Any]] = []
    supported = 0
    for idx, claim in enumerate(claims, start=1):
        has_citation = bool(_CITATION_RE.search(claim))
        overlap = _has_evidence_overlap(claim, hits) if hits else False
        is_supported = bool(has_citation and (overlap or hits))
        if is_supported:
            supported += 1
        claim_rows.append(
            {
                "index": idx,
                "text": claim[:280],
                "has_citation": has_citation,
                "has_evidence_overlap": overlap,
                "supported": is_supported,
            }
        )
    total = len(claim_rows)
    unsupported = max(0, total - supported)
    ratio = round((supported / total), 4) if total else 0.0
    return AgentVerification(
        total_claims=total,
        supported_claims=supported,
        unsupported_claims=unsupported,
        support_ratio=ratio,
        claims=claim_rows,
    )
