from __future__ import annotations

from api.claim_passage_plan import _supports, _verified_passage
from kb.claim_evidence_runtime import _support_score
from kb.claim_source_scope import owned_source_body_claim


def verified_occurrence_candidates(current: dict, candidates: list[dict], claim: str) -> list[dict] | None:
    """Recheck each occurrence against its displayed, physically verified quote.

    Legacy metadata-only cards retain their existing path. A source-verified
    card cannot authorize later claims merely because its first claim passed.
    """
    if current.get("evidence_source") != "verified_source_excerpt":
        return None
    result = []
    for row in candidates:
        source = str(row.get("source_path") or "")
        passage = _verified_passage({**row, "text": row.get("evidence_quote")}, source)
        if not passage or _support_score(owned_source_body_claim(claim, str(row.get('source_name') or source)), passage['text']) < 5:
            continue
        proof = {}
        reason = row.get('citation_plan_evidence_selection_reason') or row.get('selection_reason')
        if row.get('citation_plan_slot') and reason in {'prompt_contract_block', 'prompt_aligned_source_sentence'}:
            proof = {'citation_plan_evidence_selection_reason': reason,
                     'primary_block_id': passage['block_id'], 'primary_anchor_id': passage['anchor_id'],
                     'page_start': passage['page_start']}
        if _supports(claim, passage["text"], str(passage.get("heading_path") or ""), str(row.get("source_name") or ""), binding_meta=proof):
            result.append(row)
    return result


def refresh_verified_occurrence_claims(markdown: str, rows: list[dict]) -> list[dict]:
    from api.citation_display_registry import _ANSWER_CITATION_LINK_RE, _citation_claim_context

    claims: dict[str, list[str]] = {}
    for match in _ANSWER_CITATION_LINK_RE.finditer(markdown):
        claims.setdefault(match[2], []).append(_citation_claim_context(markdown, match.start(), match.end()))
    result = []
    for row in rows:
        if row.get("evidence_source") != "verified_source_excerpt":
            result.append(row)
            continue
        actual = claims.get(str(row.get("anchor") or ""), [])
        if actual:
            result.append({**row, "answer_claim": actual[0]})
    return result


def expand_verified_occurrence_quotes(markdown: str, rows: list[dict], link_pattern, claim_context) -> list[dict]:
    """Use the actual located block when a preview omits a supported clause.

    Every old supported occurrence must remain supported. Unstated facts never
    qualify, and the same 1800-character literal/physical bounds still apply.
    """
    from kb.citation_marker_context import numeric_marker_is_content
    from kb.source_excerpt import scoped_source_blocks
    from api.source_card_contract import refresh_source_citation_cards

    claims={}
    for m in link_pattern.finditer(markdown):
        if not numeric_marker_is_content(markdown,m.start()):
            claims.setdefault(m[2],[]).append(claim_context(markdown,m.start(),m.end()))
    result=[]
    for row in rows:
        units=claims.get(str(row.get('anchor') or ''),[])
        current=verified_occurrence_candidates(row,[row],units[0]) if units else None
        blocks=scoped_source_blocks(row) if current is not None else []
        if len(blocks)!=1:
            result.append(row);continue
        raw=str(blocks[0].get('raw_text') or '').strip()
        if not 40<=len(raw)<=1800 or raw==row.get('evidence_quote'):
            result.append(row);continue
        def supported(detail):
            return {i for i,c in enumerate(units) if verified_occurrence_candidates(detail,[detail],c)}
        before=supported(row)
        candidate=refresh_source_citation_cards([{**row,'evidence_quote':raw,'card_evidence':raw,'raw':raw,'summary_line':raw}],render_locale=str(row.get('render_locale') or ''))[0]
        after=supported(candidate)
        result.append(candidate if before<after else row)
    return result
