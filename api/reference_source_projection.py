from __future__ import annotations

import re

from api.claim_passage_plan import _verified_passage
from api.citation_occurrence_support import verified_occurrence_candidates
from kb.citation_card import CITATION_CARD_EVIDENCE_MAX_LEN
from kb.evidence_term_mapping import evidence_alignment_tokens
from kb.evidence_text import compound_claim_evidence_excerpt, pick_readable_evidence_text
from kb.source_excerpt import scoped_source_blocks


def verified_reference_detail(detail: dict) -> dict | None:
    """Project one source-verified occurrence without mixing quote and locator."""
    if detail.get("evidence_source") != "verified_source_excerpt":
        return None
    quote = str(detail.get("evidence_quote") or "").strip()
    passage = _verified_passage({**detail, "text": quote}, str(detail.get("source_path") or ""))
    blocks = scoped_source_blocks(detail)
    if len(quote) > 1800 or not passage or len(blocks) != 1 or quote not in str(blocks[0].get("raw_text") or ""):
        return None
    result = {**detail, "anchor_kind": passage["anchor_kind"]}
    claims = [str(detail.get("answer_claim") or ""), *list(detail.get("answer_claims") or [])]
    claims = list(dict.fromkeys(c for c in claims if c and verified_occurrence_candidates(detail, [detail], c)))
    result.update(answer_claim=claims[0] if claims else "", answer_claims=claims)
    takeaway = str(detail.get("card_takeaway") or "")
    result["card_takeaway"] = takeaway if takeaway and verified_occurrence_candidates(detail, [detail], takeaway) else ""
    # Downstream copy selection must see this passage only, including fallback
    # fields that formerly carried another occurrence or an entire answer.
    for key in ("evidence_quote", "reader_evidence_quote", "citation_plan_evidence_quote",
                "citation_plan_reader_evidence_quote", "citation_plan_full_evidence_quote", "raw", "summary_line"):
        result[key] = quote
    return result


def answer_citation_evidence_quote(detail: dict, *, prompt: str = "") -> str:
    """Prefer the sentence in the cited block that best matches the answer claim."""

    current = str(detail.get("evidence_quote") or detail.get("summary_line") or "").strip()
    if verified_reference_detail(detail):
        return current
    claim_core = " ".join(
        part
        for part in (
            str(detail.get("answer_claim") or "").strip(),
            " ".join(str(value or "").strip() for value in list(detail.get("answer_claims") or [])),
            str(detail.get("card_takeaway") or "").strip(),
        )
        if part
    )
    claim = " ".join(
        part for part in (claim_core, str(prompt or "").strip()) if part
    )
    claim_terms = evidence_alignment_tokens(claim)
    if not claim_terms:
        return current

    candidates: list[str] = []
    seen_candidates: set[str] = set()
    for raw_value in (
        detail.get("citation_plan_evidence_quote"),
        detail.get("citation_plan_reader_evidence_quote"),
        detail.get("reader_evidence_quote"),
        detail.get("raw"),
        current,
    ):
        raw = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        if not raw:
            continue
        # Citation-plan slots are whitespace-normalized before persistence, so
        # a Markdown heading and its first sentence can share one line.  Strip
        # only a known section label instead of treating the whole line as a
        # heading and discarding the evidence body.
        readable_source = re.sub(
            r"^\s*#{1,6}\s+(?:abstract|introduction|conclusion|discussion|results?)\s+",
            "",
            raw,
            count=1,
            flags=re.IGNORECASE,
        )
        compound = compound_claim_evidence_excerpt(
            readable_source,
            # The user prompt improves candidate ranking, but it is not part
            # of the cited claim. Mixing a CJK question into an English
            # compound claim can add an uncovered token and make the compact
            # multi-step extractor reject an otherwise complete excerpt.
            claim=claim_core or claim,
            max_len=CITATION_CARD_EVIDENCE_MAX_LEN,
        )
        readable = compound or pick_readable_evidence_text(
            readable_source,
            source=str(detail.get("source_name") or ""),
            title=str(detail.get("title") or detail.get("card_title") or ""),
            claim=claim_core or claim,
            heading=str(detail.get("heading_path") or detail.get("location_label") or ""),
            # Selection and display must share one budget.  Otherwise the
            # overlay can lose the last step of a compound mechanism before
            # the 520-character card contract ever sees it.
            max_len=CITATION_CARD_EVIDENCE_MAX_LEN,
        )
        for candidate in (
            readable,
            *re.split(r"(?<=[.!?])\s+", readable_source),
        ):
            value = str(candidate or "").strip()
            key = value.casefold()
            if 24 <= len(value) <= 700 and key not in seen_candidates:
                seen_candidates.add(key)
                candidates.append(value)
    if not candidates:
        return current

    def score(value: str) -> tuple[int, int, int]:
        overlap = len(claim_terms & evidence_alignment_tokens(value))
        number_overlap = len(
            set(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?", claim))
            & set(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?", value))
        )
        return overlap, number_overlap, -abs(len(value) - 180)

    best = max(candidates, key=score)
    if score(best)[:2] > score(current)[:2]:
        current_overlap = claim_terms & evidence_alignment_tokens(current)
        best_overlap = claim_terms & evidence_alignment_tokens(best)
        current_sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", current)
            if sentence.strip()
        ]
        best_sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", best)
            if sentence.strip()
        ]
        if (
            current_overlap
            and len(current_sentences) >= 2
            and len(best_sentences) == 1
            and len(current) <= CITATION_CARD_EVIDENCE_MAX_LEN
            and not current_overlap.issubset(best_overlap)
        ):
            # The compact render quote can already contain a complete mechanism
            # chain.  A later sentence may match more answer tokens while
            # covering a different step; replacing the whole chain with that
            # sentence loses evidence instead of improving alignment.
            return current
        return best
    if (
        score(best)[:2] == score(current)[:2]
        and len(current) > CITATION_CARD_EVIDENCE_MAX_LEN
        and len(best) <= CITATION_CARD_EVIDENCE_MAX_LEN
    ):
        return best
    return current


def reference_question_facet_priority(prompt: str, quote: str) -> int:
    from kb.direct_question_evidence import _QUALITY_BOUNDARY_QUESTION, quality_boundary_quote

    return int(bool(_QUALITY_BOUNDARY_QUESTION.search(prompt) and quality_boundary_quote(quote)))
