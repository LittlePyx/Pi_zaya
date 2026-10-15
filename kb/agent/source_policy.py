from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .types import EvidenceStatus, SourcePolicy


SourceBlend = Literal["local_grounded", "hybrid_local_external", "external_academic", "general_llm"]
AnswerMode = Literal["evidence_grounded", "hybrid_local_external", "external_academic_llm", "general_llm"]
SourceNotice = Literal["none", "hybrid", "external"]
ExternalSupplementMode = Literal["disabled", "model", "web"]


def normalize_external_supplement_mode(value: object) -> ExternalSupplementMode:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"model", "llm", "model_knowledge", "background"}:
        return "model"
    if raw in {"web", "web_search", "search", "internet"}:
        return "web"
    return "disabled"


@dataclass(frozen=True)
class AnswerSourceDecision:
    source_blend: SourceBlend
    answer_mode: AnswerMode
    source_policy: SourcePolicy
    source_notice: SourceNotice
    evidence_status: EvidenceStatus
    evidence_hit_count: int = 0
    candidate_hit_count: int = 0
    retrieval_confidence: str = ""
    reasons: list[str] | None = None
    instruction: str = ""
    external_supplement_mode: ExternalSupplementMode = "disabled"

    def to_evidence_gate(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons or [])[:8]
        return data


def _dedupe_reasons(*groups: list[str]) -> list[str]:
    reasons: list[str] = []
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if text and text not in reasons:
                reasons.append(text)
    return reasons[:8]


def _confidence_level(value: str, *, candidate_hit_count: int, fallback: str = "medium") -> str:
    level = str(value or "").strip()
    if level:
        return level
    if candidate_hit_count <= 0:
        return "none"
    return fallback


def decide_answer_source(
    *,
    hit_count: int,
    candidate_hit_count: int,
    retrieval_confidence: str = "",
    retrieval_reasons: list[str] | None = None,
    academic_question: bool = False,
    local_grounding_requested: bool = False,
    external_supplement_mode: str = "disabled",
) -> AnswerSourceDecision:
    """Choose how to blend local evidence and external model context.

    The returned `answer_mode` preserves the older public/runtime strings. The
    `source_blend` value is the clearer strategy label used for traces and evals.
    """
    hits = max(0, int(hit_count or 0))
    candidates = max(0, int(candidate_hit_count or hits or 0))
    supplement_mode = normalize_external_supplement_mode(external_supplement_mode)
    confidence_reasons = [str(item or "").strip() for item in list(retrieval_reasons or []) if str(item or "").strip()]

    if hits <= 0:
        base_reasons = (
            ["no_evidence_hits"]
            if candidates <= 0
            else ["low_retrieval_confidence", "no_usable_local_evidence"]
        )
        confidence = _confidence_level(retrieval_confidence, candidate_hit_count=candidates, fallback="low")
        if supplement_mode != "disabled":
            reasons = _dedupe_reasons(base_reasons, ["not_based_on_local_knowledge_base"], confidence_reasons)
            if local_grounding_requested and "local_grounding_requested" not in reasons:
                reasons.append("local_grounding_requested")
            reasons.append("web_search_requested" if supplement_mode == "web" else "external_model_requested")
            return AnswerSourceDecision(
                source_blend="external_academic",
                answer_mode="external_academic_llm",
                source_policy="external_allowed_with_notice",
                source_notice="external",
                evidence_status="not_applicable",
                evidence_hit_count=0,
                candidate_hit_count=candidates,
                retrieval_confidence=confidence,
                reasons=reasons[:8],
                instruction=(
                    "Use external web search only. Clearly state that reliable local indexed evidence was not found; "
                    "do not use model-only knowledge as a silent fallback."
                    if supplement_mode == "web"
                    else "Use external model knowledge only. Clearly state that reliable local indexed evidence was not found; "
                    "do not claim the answer is grounded in the knowledge base."
                ),
                external_supplement_mode=supplement_mode,
            )
        return AnswerSourceDecision(
            source_blend="local_grounded",
            answer_mode="evidence_grounded",
            source_policy="local_only",
            source_notice="none",
            evidence_status="insufficient",
            evidence_hit_count=0,
            candidate_hit_count=candidates,
            retrieval_confidence=confidence,
            reasons=_dedupe_reasons(base_reasons, confidence_reasons),
            instruction="Do not infer an answer. Say that indexed evidence is insufficient.",
            external_supplement_mode="disabled",
        )

    thin_or_low_confidence = hits < 2 or str(retrieval_confidence or "").strip().lower() in {"low", "none"}
    if supplement_mode != "disabled" and thin_or_low_confidence:
        return AnswerSourceDecision(
            source_blend="hybrid_local_external",
            answer_mode="hybrid_local_external",
            source_policy="local_plus_external_background",
            source_notice="hybrid",
            evidence_status="needs_review",
            evidence_hit_count=hits,
            candidate_hit_count=candidates,
            retrieval_confidence=_confidence_level(retrieval_confidence, candidate_hit_count=candidates),
            reasons=_dedupe_reasons(
                [
                    "low_evidence_count" if hits < 2 else "low_retrieval_confidence",
                    "web_search_requested" if supplement_mode == "web" else "external_model_requested",
                ],
                confidence_reasons,
            ),
            instruction=(
                "Use the retrieved snippet as local authority. Web search may fill background gaps, but model-only "
                "knowledge must not replace a failed web search and paper-specific claims must stay tied to local evidence."
                if supplement_mode == "web"
                else "Use the retrieved snippet as local authority. External model knowledge may clarify context, "
                "but paper-specific claims must stay tied to local evidence."
            ),
            external_supplement_mode=supplement_mode,
        )

    return AnswerSourceDecision(
        source_blend="local_grounded",
        answer_mode="evidence_grounded",
        source_policy="local_only",
        source_notice="none",
        evidence_status="grounded",
        evidence_hit_count=hits,
        candidate_hit_count=candidates,
        retrieval_confidence=_confidence_level(retrieval_confidence, candidate_hit_count=candidates, fallback="high"),
        reasons=_dedupe_reasons(confidence_reasons),
        instruction="Answer only from retrieved evidence and cite supported claims.",
        external_supplement_mode=supplement_mode,
    )


def source_policy_from_gate(agent_notes: dict[str, Any] | None, *, has_hits: bool) -> SourcePolicy:
    gate = agent_notes.get("evidence_gate") if isinstance(agent_notes, dict) else {}
    if isinstance(gate, dict):
        policy = str(gate.get("source_policy") or "").strip()
        if policy in {
            "local_only",
            "local_plus_external_background",
            "external_allowed_with_notice",
            "trusted_sites_only",
        }:
            return policy  # type: ignore[return-value]
        mode = str(gate.get("answer_mode") or "").strip()
        if mode == "hybrid_local_external":
            return "local_plus_external_background"
        if mode in {"external_academic_llm", "general_llm"}:
            return "external_allowed_with_notice"
    return "local_only" if has_hits else "external_allowed_with_notice"
