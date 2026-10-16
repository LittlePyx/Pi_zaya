from __future__ import annotations

import re
from pathlib import Path

from api.citation_display_registry import _canonical_source_path_identity, _citation_claim_context
from kb.citation_marker_context import numeric_marker_is_content
from kb.citation_sentences import citation_sentence_spans
from kb.claim_evidence_runtime import _claim_units, _support_score
from kb.evidence_binding import assess_system_a_hit_binding
from kb.evidence_term_mapping import evidence_alignment_tokens, specific_method_identities
from kb.source_blocks import load_source_blocks_cached, normalize_match_text
from kb.source_passage_evidence import claim_passage_concepts_covered


def message_citation_plan(rec: dict | None) -> dict:
    if not isinstance(rec, dict):
        return {}
    meta = dict(rec.get("meta") or {}) if isinstance(rec.get("meta"), dict) else {}
    answer_quality = dict(meta.get("answer_quality") or {}) if isinstance(meta.get("answer_quality"), dict) else {}
    plan = answer_quality.get("citation_plan")
    if isinstance(plan, dict) and plan:
        return dict(plan)
    contracts = dict(meta.get("paper_guide_contracts") or {}) if isinstance(meta.get("paper_guide_contracts"), dict) else {}
    plan = contracts.get("citation_plan")
    return dict(plan) if isinstance(plan, dict) and plan else {}


def _supports(claim: str, quote: str, heading: str, source: str) -> bool:
    # Related summary-statistic prose cannot substitute for a target theorem.
    if not claim_passage_concepts_covered(claim, quote):
        return False
    if _support_score(claim, f"{quote}\n{heading}\n{source}") < 5:
        return False
    if specific_method_identities(claim) & specific_method_identities(quote):
        def requirements(text):
            facts = []
            for match in re.finditer(r"\b(?:requires?|needs?)\s+([^.;\n]+)", text, re.I):
                negative = bool(re.search(r"\b(?:not|never)\s*$", text[:match.start()], re.I))
                facts.append((not negative, evidence_alignment_tokens(match[1])))
            for match in re.finditer(r"\bneither\s+(.+?)\s+nor\s+(.+?)\s+(?:is|are)\s+(?:needed|required|necessary)", text, re.I):
                facts.append((False, evidence_alignment_tokens(match[1] + " " + match[2])))
            return facts
        if any(a != b and len(x & y) >= 2 for a,x in requirements(claim) for b,y in requirements(quote)):
            return False
    binding = assess_system_a_hit_binding(
        answer_claim=claim, hit={"text": quote},
        meta={"citation_plan_evidence_authoritative": True, "citation_plan_full_evidence_quote": quote},
        heading=heading, evidence_quote=quote, source_name=source,
    )
    return not binding.get("suppress_link", True)


def _verified_passage(passage: dict, source: str) -> dict:
    """Check the retrieved passage's text and complete locator against its own source."""
    quote = str(passage.get("text") or "").strip()
    if passage.get("source_path") and _canonical_source_path_identity(str(passage["source_path"])) != _canonical_source_path_identity(source):
        return {}
    if len(quote) < 40 or not passage.get("block_id") or not passage.get("page_start"):
        return {}
    path = Path(source)
    if not path.is_file():
        from api.reference_ui import _resolve_source_md_path
        path = _resolve_source_md_path(source)
    if path is None:
        return {}
    for block in load_source_blocks_cached(path):
        if str(block.get("block_id") or "") != str(passage["block_id"]):
            continue
        if any(str(block.get(key) or "") != str(passage.get(key) or "")
               for key in ("anchor_id", "page_start", "page_end", "heading_path")):
            return {}
        if normalize_match_text(quote) not in normalize_match_text(str(block.get("text") or "")):
            return {}
        return {**passage, "text": quote, "anchor_kind": str(block.get("kind") or "paragraph"),
                "source_block_text": str(block.get("text") or "")}
    return {}


def _slot_supports(claim: str, slot: dict) -> bool:
    passage = _verified_passage({**slot, "text": slot.get("evidence_quote")}, str(slot.get("source_path") or ""))
    if slot.get("block_id") and not passage:
        return False
    quote = passage["source_block_text"] if passage else str(slot.get("evidence_quote") or "")
    return _supports(claim, quote, str(slot.get("heading_path") or ""), str(slot.get("source_name") or ""))


def prepare_claim_passage_plan(rec: dict, ref_pack: dict | None, answer: str) -> dict:
    """Add only retrieved, verified same-source passages needed by cited claims.

    Keep document numbers stable. The renderer can then give each passage its
    own anchor/card instead of using one source-level quote for every sentence.
    """
    plan = message_citation_plan(rec)
    slots = [dict(slot) for slot in plan.get("slots", []) if isinstance(slot, dict)]
    if not slots or not isinstance(ref_pack, dict):
        return rec
    canonical = list((rec.get("meta") or {}).get("canonical_hit_paths") or [])
    hits = [hit for hit in ref_pack.get("hits", []) if isinstance(hit, dict)]
    valid_slots = [slot for slot in slots if not slot.get("passage_bound_claims") or _verified_passage(
        {**slot, "text": slot.get("evidence_quote")}, str(slot.get("source_path") or ""))]
    added = len(valid_slots) != len(slots)
    slots = valid_slots
    # Older render plans may have created a full-block fallback beside the
    # original excerpt of that same verified block. Reuse the original slot,
    # rechecking every inherited claim; do not let two views compete again.
    for slot in list(slots):
        if slot.get("evidence_selection_reason") != "claim_source_passage":
            continue
        primary = next((s for s in slots if s is not slot
                        and s.get("evidence_selection_reason") != "claim_source_passage"
                        and s.get("block_id") == slot.get("block_id")
                        and _canonical_source_path_identity(str(s.get("source_path") or "")) == _canonical_source_path_identity(str(slot.get("source_path") or ""))
                        and _verified_passage({**s, "text": s.get("evidence_quote")}, str(s.get("source_path") or ""))), None)
        if primary is not None:
            claims = [c for c in slot.get("passage_bound_claims", []) if _slot_supports(c, primary)]
            primary["passage_bound_claims"] = list(dict.fromkeys([*primary.get("passage_bound_claims", []), *claims]))
            slots.remove(slot)
            added = True
    for slot in slots:
        passage = _verified_passage({**slot, "text": slot.get("evidence_quote")}, str(slot.get("source_path") or ""))
        full = passage.get("source_block_text") if passage else None
        if full and slot.get("citation_plan_full_evidence_quote") != full:
            slot["citation_plan_full_evidence_quote"] = full
            added = True
        elif not full and slot.pop("citation_plan_full_evidence_quote", None) is not None:
            added = True
    for claim in _claim_units(answer):
        for marker in re.finditer(r"(?<!\[)\[(\d{1,4})\](?!\])", claim):
            if numeric_marker_is_content(claim, marker.start()):
                continue
            local_claim = _citation_claim_context(claim, marker.start(), marker.end())
            number = int(marker[1])
            if not 0 < number <= len(canonical):
                continue
            identity = _canonical_source_path_identity(str(canonical[number - 1]))
            source_slots = [slot for slot in slots
                            if str(slot.get("preferred_system") or "system_a").lower() == "system_a"
                            and _canonical_source_path_identity(str(slot.get("source_path") or "")) == identity]
            if not source_slots:
                continue
            supported_slots = [slot for slot in source_slots if _slot_supports(local_claim, slot)]
            excerpt_slots = [slot for slot in supported_slots if _support_score(local_claim, str(slot.get("evidence_quote") or "")) >= 5 and _supports(
                local_claim, str(slot.get("evidence_quote") or ""),
                str(slot.get("heading_path") or ""), str(slot.get("source_name") or ""))]
            if len(excerpt_slots) == 1:
                supported_slots = excerpt_slots
            if supported_slots:
                unique_secondary = (
                    len(supported_slots) == 1 and supported_slots[0] is not source_slots[0]
                    and bool(_verified_passage(
                        {**supported_slots[0], "text": supported_slots[0].get("evidence_quote")},
                        str(supported_slots[0].get("source_path") or ""),
                    ))
                )
                if len(supported_slots) == 1 and (unique_secondary or supported_slots[0].get("passage_bound_claims")):
                    slot = supported_slots[0]
                    claims = list(slot.get("passage_bound_claims") or [])
                    if local_claim not in claims:
                        slot["passage_bound_claims"] = [*claims, local_claim]
                        added = True
                continue
            candidates = {}
            for hit in hits:
                meta = hit.get("meta") or {}
                if _canonical_source_path_identity(str(meta.get("source_path") or "")) != identity:
                    continue
                for raw in meta.get("source_passages") or []:
                    if not isinstance(raw, dict):
                        continue
                    passage = _verified_passage(raw, str(meta.get("source_path") or ""))
                    if not passage or not _supports(local_claim, passage["text"],
                                                    str(passage.get("heading_path") or ""),
                                                    str(meta.get("source_name") or "")):
                        continue
                    candidates[str(passage["block_id"])] = passage
            # Multiple physically different matches are ambiguous; never pick
            # a page by retrieval order or turn a namesake paper into evidence.
            if len(candidates) != 1:
                continue
            passage = next(iter(candidates.values()))
            slots.append({
                "claim_type": "paper_evidence", "preferred_system": "system_a",
                "source_path": source_slots[0]["source_path"],
                "source_name": source_slots[0].get("source_name", ""),
                "candidate_hits": [number], "evidence_quote": passage["text"],
                "citation_plan_full_evidence_quote": passage["source_block_text"],
                **{key: passage.get(key) for key in ("heading_path", "page_start", "page_end", "block_id", "anchor_id")},
                "anchor_kind": passage["anchor_kind"], "strict_locate": True,
                "source_passage_bundle": True,
                "evidence_selection_reason": "claim_source_passage",
                "passage_bound_claims": [local_claim],
            })
            added = True
    if not added:
        return rec
    result = dict(rec)
    meta = dict(rec.get("meta") or {})
    expanded = {**plan, "slots": slots}
    for key in ("answer_quality", "paper_guide_contracts"):
        meta[key] = {**dict(meta.get(key) or {}), "citation_plan": expanded}
    result["meta"] = meta
    return result


def prepare_claim_passage_render(rec: dict, ref_pack: dict | None, answer: str) -> tuple[dict, str]:
    """Give an uncited sentence its own verified passage within a cited paragraph.

    Only distribute one existing paragraph-end document coordinate. Do not
    infer a source for mixed-source, uncited, mathematical or code paragraphs.
    Raw generation and retrieved evidence remain unchanged.
    """
    if not isinstance(ref_pack, dict):
        return rec, answer
    canonical = list((rec.get("meta") or {}).get("canonical_hit_paths") or [])
    slots = [dict(s) for s in message_citation_plan(rec).get("slots", []) if isinstance(s, dict)]
    insertions = []
    in_fence = False
    for paragraph in re.finditer(r'[^\n]+', answer):
        text = paragraph[0]
        if text.strip().startswith(('```', '~~~', '$$')):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        markers = [m for m in re.finditer(r'(?<!\[)\[(\d{1,4})\](?!\])', text)
                   if not numeric_marker_is_content(text, m.start())]
        if len(markers) != 1 or not re.fullmatch(r'[\s.!?。！？]*', text[markers[0].end():]):
            continue
        number = int(markers[0][1])
        if not 0 < number <= len(canonical):
            continue
        source = str(canonical[number-1])
        verified = []
        for slot in slots:
            if str(slot.get('preferred_system') or 'system_a').lower() != 'system_a':
                continue
            if _canonical_source_path_identity(str(slot.get('source_path') or '')) != _canonical_source_path_identity(source):
                continue
            passage = _verified_passage({**slot, 'text': slot.get('evidence_quote')}, source)
            if passage:
                verified.append((slot, passage))
        if len({p['block_id'] for _, p in verified}) < 2:
            continue
        for start, end in citation_sentence_spans(text):
            claim = text[start:end].strip()
            if start <= markers[0].start() < end or len(claim) < 20:
                continue
            losses = set(re.findall(r'\bL\s*(\d+)\b', claim, re.I))
            matches = [(slot, p) for slot, p in verified
                       if losses <= set(re.findall(r'\bL\s*(\d+)\b', p['source_block_text'], re.I)) and _supports(
                claim, p['source_block_text'], str(slot.get('heading_path') or ''),
                str(slot.get('source_name') or ''),
            )]
            if len({p['block_id'] for _, p in matches}) != 1:
                continue
            slot, _ = matches[0]
            # Binding stores the precise sentence, not the whole paragraph.
            slot['passage_bound_claims'] = list(dict.fromkeys([*slot.get('passage_bound_claims', []), claim]))
            insertions.append((paragraph.start()+end, f' [{number}]'))
    if insertions:
        for position, marker in reversed(insertions):
            answer = answer[:position]+marker+answer[position:]
        rec = {**rec, 'meta': dict(rec.get('meta') or {})}
        plan = {**message_citation_plan(rec), 'slots': slots}
        for key in ('answer_quality', 'paper_guide_contracts'):
            rec['meta'][key] = {**dict(rec['meta'].get(key) or {}), 'citation_plan': plan}
    return prepare_claim_passage_plan(rec, ref_pack, answer), answer
