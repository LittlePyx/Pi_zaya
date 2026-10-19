from __future__ import annotations

import re

from kb.explicit_noise_relations import iid_relation_covered, known_noise_model_covered, named_synthetic_sample_covered, raw_output_noise_covered, sampling_ability_covered
from kb.claim_source_eligibility import retrieval_limit_claim, omitted_derivation_covered
from kb.noise_model_prerequisites import noise_model_prerequisites_covered, objective_term_covered
from kb.explicit_claim_metrics import named_metrics_covered, qualitative_outperformance_covered
from kb.explicit_training_assignments import explicit_training_assignments_covered, named_equation_covered, noise_augmentation_training_covered, prediction_average_covered


def citation_plan_locator_quote(locator_slot: dict, slots: list[dict]) -> str:
    located = {
        (s.get("block_id") or s.get("blockId"), s.get("anchor_id") or s.get("anchorId"))
        for s in slots if s.get("block_id") or s.get("blockId") or s.get("anchor_id") or s.get("anchorId")
    }
    if len(located) > 1:
        return str(locator_slot.get("evidence_quote") or locator_slot.get("evidenceQuote") or "").strip()
    return "\n".join(dict.fromkeys(str(s.get("evidence_quote") or s.get("evidenceQuote") or "").strip() for s in slots))


def claim_passage_concepts_covered(claim: str, evidence: str) -> bool:
    """A general statistical summary does not establish conditional/zero means."""
    if retrieval_limit_claim(claim) or not omitted_derivation_covered(claim, evidence):
        return False
    if not noise_model_prerequisites_covered(claim, evidence) or not objective_term_covered(claim, evidence):
        return False
    if not named_metrics_covered(claim, evidence) or not qualitative_outperformance_covered(claim, evidence):
        return False
    concepts = (r"\bBernoulli\b|伯努利", r"zero[- ]mean|零均值", r"\bconditional\s+(?:expect\w*|mean)\b|条件(?:期望|均值)",
                r"standard\s+deviation|标准差",
                r"\bunconditional\s+(?:expect\w*|mean)\b|无条件(?:期望|均值)",
                r"spatially\s+correlated\s+noise|空间相关噪声",
                r"sampl\w*\s+from\s+(?:the\s+)?noise\s+distribution|从噪声分布中?采样")
    return (all(not re.search(c, claim, re.I) or re.search(c, evidence, re.I) for c in concepts)
            and explicit_training_assignments_covered(claim, evidence) and named_equation_covered(claim, evidence) and prediction_average_covered(claim, evidence)
            and noise_augmentation_training_covered(claim, evidence)
            and known_noise_model_covered(claim, evidence)
            and named_synthetic_sample_covered(claim, evidence)
            and iid_relation_covered(claim, evidence) and raw_output_noise_covered(claim, evidence)
            and sampling_ability_covered(claim, evidence))


def claim_occurrence_matches(claim: str, bound_claims: list[str]) -> bool:
    def key(value: str) -> str:
        value = re.sub(r"\[\d{1,4}\](?:\([^\n)]*\))?", "", str(value or ""))
        return re.sub(r"[\W_]+", "", value.casefold())

    actual = key(claim)
    return any(len(bound := key(value)) >= 20 and bound in actual for value in bound_claims)


def claim_bound_candidates(rows: list[dict], claim: str) -> list[dict]:
    matched = [row for row in rows if claim_occurrence_matches(claim, row.get("passage_bound_claims") or [])]
    return matched or [row for row in rows if not row.get("passage_bound_claims")]


def bind_source_passage_primary(card: dict, primary: dict, hits: list[dict]) -> dict:
    """Bind a compound context excerpt to its first verified source passage."""
    quote = str(
        primary.get("snippet")
        or primary.get("evidence_quote")
        or card.get("snippet")
        or card.get("evidence_quote")
        or ""
    )
    match = re.match(
        r"^\[Source passage \d+: .*?\bp\.\s*(\d+)\]\s*(.*?)(?=\[Source passage \d+:|\Z)",
        quote,
        re.S,
    )
    if not match:
        return primary
    source = re.sub(
        r"/+",
        "/",
        str(primary.get("source_path") or card.get("source_path") or "").replace(
            "\\", "/"
        ),
    ).casefold()
    body = match.group(2).strip()
    for hit in hits:
        meta = hit.get("meta") or {}
        if (
            re.sub(
                r"/+", "/", str(meta.get("source_path") or "").replace("\\", "/")
            ).casefold()
            != source
        ):
            continue
        for passage in meta.get("source_passages") or []:
            actual = str(passage.get("text") or "").strip()
            if (
                len(actual) < 40
                or re.sub(r"\s+", " ", body)[:100] != re.sub(r"\s+", " ", actual)[:100]
            ):
                continue
            if int(passage.get("page_start") or 0) != int(match.group(1)):
                continue
            return {
                **primary,
                **passage,
                "source_path": primary.get("source_path") or card.get("source_path"),
                "snippet": actual,
                "evidence_quote": actual,
            }
    return primary
