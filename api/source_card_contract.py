from __future__ import annotations

import re

from api.claim_passage_plan import _supports, _verified_passage
from api.citation_display_registry import _canonical_source_path_identity
from kb.citation_card import (
    CITATION_CARD_EVIDENCE_MAX_LEN,
    refresh_citation_card_contract,
)
from kb.citation_sentences import citation_sentence_spans
from kb.source_excerpt import literal_excerpt, scoped_source_blocks
from kb.source_blocks import normalize_match_text
from kb.claim_evidence_runtime import _claim_units
from kb.citation_marker_context import numeric_marker_is_content
from kb.claim_source_eligibility import complete_source_prefix, incomplete_source_prose
from kb.refusal_context_excerpt import bind_refusal_context_excerpt
from kb.source_locator_receipts import relocate_repaired_source_plan

# The reader already carries this bounded source context. A verified continuous
# excerpt may need more room than the 520-character synthesized card preview.
_SOURCE_EXCERPT_MAX_LEN = 1800


def _restore_leading_qualifier(block: dict, quote: str) -> str:
    """Keep a located sentence's logical prefix when its preview starts late."""
    raw = str(block.get('raw_text') or '')
    if not quote or raw.count(quote) != 1:
        return quote
    offset = raw.index(quote)
    for start, end in citation_sentence_spans(raw):
        if start < offset < end:
            prefix = raw[start:offset]
            if re.search(r'\b(?:without|not|never|unless|only\s+if|provided\s+that|assuming|if)\b|除非|只有|若|如果|不(?:需要|使用|满足)', prefix, re.I):
                restored = raw[start:offset + len(quote)].strip()
                if len(restored) <= _SOURCE_EXCERPT_MAX_LEN:
                    return restored
            break
    return quote


def _verified_bound_reader_excerpt(detail: dict) -> tuple[dict, str] | None:
    """Retain a verified claim-specific passage after display shortening."""
    if detail.get('evidence_source') not in {'citation_plan_claim_window', 'verified_source_excerpt'} or detail.get('reader_evidence_source') not in {'citation_plan_located_block', 'verified_source_block'}:
        return None
    actual = str(detail.get('answer_claim') or '').strip()
    def key(value: str) -> str:
        return re.sub(r'[\W_]+', '', re.sub(r'\[\d{1,5}\](?:\([^\n)]*\))?', '', value).casefold())
    actual_key = key(actual)
    bound = {key(str(c)): str(c) for c in detail.get('passage_bound_claims') or []}
    matches = [c for k, c in bound.items() if k == actual_key or (actual.endswith(('...', '…')) and len(actual_key) >= 40 and k.startswith(actual_key))]
    if not actual_key or len(matches) != 1 or not all(detail.get(k) for k in ('source_path', 'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')):
        return None
    blocks = scoped_source_blocks(detail)
    if len(blocks) != 1 or any(detail.get(k) != blocks[0].get(k) for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')):
        return None
    block = blocks[0]; claim = matches[0]
    current = literal_excerpt(block, str(detail.get('evidence_quote') or ''))
    if current and _supports(claim, current, '', ''):
        return None
    quote = literal_excerpt(block, str(detail.get('reader_evidence_quote') or ''))
    return (block, quote) if quote and len(quote) <= _SOURCE_EXCERPT_MAX_LEN and _supports(claim, quote, '', '') else None


def _verified_reader_excerpt(detail: dict) -> tuple[dict, str] | None:
    """Recover only the already located plan passage after preview resolution fails."""
    if detail.get("evidence_source") != "citation_plan_claim_window" or detail.get("reader_evidence_source") != "citation_plan_located_block":
        return None
    if not all(detail.get(k) for k in (
        "source_path", "block_id", "anchor_id", "heading_path", "page_start", "page_end",
    )):
        return None
    blocks = scoped_source_blocks(detail)
    if len(blocks) != 1 or any(str(detail.get(k)) != str(blocks[0].get(k)) for k in (
        "block_id", "anchor_id", "heading_path", "page_start", "page_end",
    )):
        return None
    quote = literal_excerpt(blocks[0], str(detail.get("reader_evidence_quote") or ""))
    if quote and len(quote) <= _SOURCE_EXCERPT_MAX_LEN and _supports(str(detail.get("answer_claim") or ""), quote, "", ""):
        return blocks[0], quote
    return None


def _split_compound_sources(
    slots: list[dict], answer: str, canonical: list
) -> tuple[list[dict], list[dict]]:
    result = []
    retired = []
    for slot in slots:
        if (
            not isinstance(slot, dict)
            or not slot.get("compound_same_page_evidence")
            or slot.get("block_id")
            or slot.get("anchor_id")
        ):
            result.append(slot)
            continue
        quote = str(slot.get("evidence_quote") or "")
        groups = {}
        valid = True
        for start, end in citation_sentence_spans(quote):
            sentence = quote[start:end].strip()
            matches = [
                (b, literal_excerpt(b, sentence)) for b in scoped_source_blocks(slot)
            ]
            matches = [(b, q) for b, q in matches if q]
            if len(matches) != 1:
                valid = False
                break
            block, _literal = matches[0]
            groups[block["block_id"]] = block
        candidates = []
        source_numbers = {
            i + 1
            for i, path in enumerate(canonical)
            if _canonical_source_path_identity(str(path))
            == _canonical_source_path_identity(str(slot.get("source_path") or ""))
        }
        for block in groups.values():
            # A matching original slot may carry the rest of this same block.
            # Reading that block supplies no new document/page or fake locator.
            text = str(block["raw_text"])
            claims = [
                claim
                for claim in _claim_units(answer)
                if any(
                    int(marker[1]) in source_numbers
                    and not numeric_marker_is_content(claim, marker.start())
                    for marker in re.finditer(r"(?<!\[)\[(\d{1,4})\](?!\])", claim)
                )
                and _supports(claim, text, "", "")
            ]
            candidate = {
                **slot,
                **{
                    k: block[k]
                    for k in (
                        "block_id",
                        "anchor_id",
                        "heading_path",
                        "page_start",
                        "page_end",
                    )
                },
                "evidence_quote": text,
                "anchor_kind": block["kind"],
                "strict_locate": True,
                "compound_same_page_evidence": False,
                "passage_bound_claims": claims,
                "evidence_selection_reason": "verified_compound_source_block",
            }
            if not claims or not _verified_passage(
                {**candidate, "text": text}, str(slot.get("source_path") or "")
            ):
                valid = False
            candidates.append(candidate)
        if (
            valid
            and len(candidates) >= 2
            and len({s["page_start"] for s in candidates}) == 1
        ):
            retired.append(slot)
            result.extend(candidates)
        else:
            result.append(slot)
    if retired:
        # The verified occurrence replaces a stale/truncated view of that same
        # physical block, while other occurrences and source budgets stay put.
        verified = {
            (
                _canonical_source_path_identity(str(s.get("source_path") or "")),
                s.get("block_id"),
            ): s
            for s in result
            if isinstance(s, dict)
            and s.get("evidence_selection_reason") == "verified_compound_source_block"
        }
        kept = []
        for slot in result:
            key = (
                (
                    _canonical_source_path_identity(str(slot.get("source_path") or "")),
                    slot.get("block_id"),
                )
                if isinstance(slot, dict)
                else None
            )
            replacement = verified.get(key)
            if replacement is None or slot is replacement:
                kept.append(slot)
            else:
                retired.append(slot)
        result = kept
    return result, retired


def _repair_located_message_plan(message: dict) -> dict:
    """Follow verified structural receipts even on a retained failed answer.

    This derives only source coordinates and an explicit reading quotation;
    every historical quality flag and the stored answer stay unchanged.
    """
    meta = message.get('meta') or {}
    quality = meta.get('answer_quality') or {}
    plan = quality.get('citation_plan')
    if message.get('role') != 'assistant' or not isinstance(plan, dict):
        return message
    paths = list(meta.get('canonical_hit_paths') or [])
    repaired = relocate_repaired_source_plan(plan, paths)
    repaired = bind_refusal_context_excerpt(str(message.get('content') or ''), repaired, paths)
    if repaired == plan:
        return message
    meta = dict(meta)
    for key in ('answer_quality', 'paper_guide_contracts'):
        meta[key] = {**dict(meta.get(key) or {}), 'citation_plan': repaired}
    return {**message, 'meta': meta}


def prepare_source_card_message(message: dict) -> dict:
    """Resolve page-only plans before binding both complete and partial answers.

    Keep the answer's failure and all semantic/support gates unchanged. A
    page-only plan can qualify only after its entire quote is re-read in one
    unique block of its own canonical source. Located plans may follow only
    an exact PDF-proven repair receipt before all ordinary support checks.
    """
    rec = _repair_located_message_plan(dict(message or {}))
    meta = dict(rec["meta"]) if isinstance(rec.get("meta"), dict) else {}
    quality = (
        dict(meta["answer_quality"])
        if isinstance(meta.get("answer_quality"), dict)
        else {}
    )
    if (
        rec.get("role") != "assistant"
        or quality.get("has_hits") is not True
        or not isinstance(quality.get("claim_evidence"), dict)
        or quality["claim_evidence"].get("minimum_ok") is not True
        or quality.get("resolution_status")
        in {"insufficient_evidence", "not_found", "scoped_not_found"}
        or meta.get("answer_contract")
        or meta.get("answerContract")
    ):
        return rec
    plan = quality.get("citation_plan")
    if not isinstance(plan, dict):
        return rec
    partial = (
        quality.get("minimum_ok") is False
        and quality.get("incomplete_fragment") is True
        and quality.get("contract_enabled") is False
    )
    if quality.get("minimum_ok") is not True and not partial:
        return rec
    paths = meta.get("canonical_hit_paths") or []
    canonical = {_canonical_source_path_identity(str(p)) for p in paths}
    originals, retired = _split_compound_sources(
        plan.get("slots") or [], str(rec.get("content") or ""), paths
    )
    slots = []
    changed = bool(retired)
    for original in originals:
        slot = dict(original) if isinstance(original, dict) else original
        if (
            not isinstance(slot, dict)
            or slot.get("block_id")
            or slot.get("anchor_id")
            or slot.get("preferred_system", "system_a") != "system_a"
            or _canonical_source_path_identity(str(slot.get("source_path") or ""))
            not in canonical
        ):
            slots.append(slot)
            continue
        quote = str(slot.get("evidence_quote") or "")
        matches = [(b, literal_excerpt(b, quote)) for b in scoped_source_blocks(slot)]
        matches = [
            (b, q)
            for b, q in matches
            if q and normalize_match_text(quote) in normalize_match_text(q)
        ]
        if len(matches) == 1:
            block, literal = matches[0]
            candidate = {
                **slot,
                **{
                    k: block[k]
                    for k in (
                        "block_id",
                        "anchor_id",
                        "heading_path",
                        "page_start",
                        "page_end",
                    )
                },
                "anchor_kind": block["kind"],
                "strict_locate": True,
                "evidence_quote": literal,
            }
            if _verified_passage(
                {**candidate, "text": literal}, str(slot["source_path"])
            ):
                slot = candidate
                changed = True
        slots.append(slot)
    if changed:
        expanded = {**plan, "slots": slots}
        if retired:
            expanded["source_compound_audit"] = [
                *plan.get("source_compound_audit", []),
                *retired,
            ]
        for key in ("answer_quality", "paper_guide_contracts"):
            meta[key] = {
                **(meta[key] if isinstance(meta.get(key), dict) else {}),
                "citation_plan": expanded,
            }
        rec["meta"] = meta
    return rec


def _source_excerpt(detail: dict) -> tuple[dict, str] | None:
    bound = _verified_bound_reader_excerpt(detail)
    if bound:
        return bound
    blocks = scoped_source_blocks(detail)
    if not blocks:
        return None
    quote = str(detail.get("evidence_quote") or detail.get("raw") or "")
    if len(blocks)==1 and all(detail.get(k) and detail.get(k)==blocks[0].get(k) for k in ('block_id','anchor_id','heading_path','page_start','page_end')):
        raw=str(blocks[0].get('raw_text') or '').strip()
        prefix=complete_source_prefix(raw)
        # The full parent was already located exactly. This is a literal,
        # complete assertion, not a shorter fuzzy match or a new locator.
        if quote in {raw,prefix} and prefix!=raw and len(prefix)>=24:
            return blocks[0],prefix
    matches = [(b, literal_excerpt(b, quote)) for b in blocks]
    matches = [(b, q) for b, q in matches if q]
    if len(matches) == 1:
        block, excerpt = matches[0]
        if len(excerpt) <= _SOURCE_EXCERPT_MAX_LEN:
            raw = str(block.get('raw_text') or '').strip()
            # A short variable declaration needs its existing prose context.
            # Never stitch blocks or restore a lost logical continuation.
            if (block.get('kind') == 'paragraph' and len(excerpt) < 120
                    and re.search(r'(?<!\\)\$[^$\n]+\$|:=|≜|\\triangleq\b', excerpt)
                    and excerpt in raw and len(excerpt) < len(raw) <= _SOURCE_EXCERPT_MAX_LEN
                    and not incomplete_source_prose(raw)
                    and not re.search(r'\b(?:because|and|or|but|therefore|thus|hence|where|with|of|from|to|given|denotes|requires|assume|is|are|such that)\s*[,.;:]?\s*$', raw, re.I)):
                from api.citation_occurrence_support import verified_occurrence_candidates
                candidate = {**detail, 'evidence_source': 'verified_source_excerpt', 'evidence_quote': raw}
                if verified_occurrence_candidates(candidate, [candidate], str(detail.get('answer_claim') or '')):
                    return block, raw
            return block, excerpt
    # A compound context is not a single locator. Select a complete, already
    # quoted source sentence that independently passes the existing support
    # contract. Keep the original compound text for auditing.
    claim = str(detail.get("answer_claim") or "")
    if len(blocks) == 1 and all(detail.get(k) and detail[k] == blocks[0].get(k) for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')):
        raw = str(blocks[0].get('raw_text') or '').strip()
        if len(raw) <= _SOURCE_EXCERPT_MAX_LEN and _supports(claim, raw, '', str(detail.get('source_name') or '')):
            return blocks[0], raw
    for start, end in citation_sentence_spans(quote):
        sentence = quote[start:end].strip()
        candidates = [(b, literal_excerpt(b, sentence)) for b in blocks]
        candidates = [
            (b, q)
            for b, q in candidates
            if q and len(q) <= CITATION_CARD_EVIDENCE_MAX_LEN
        ]
        if len(candidates) == 1:
            block, excerpt = candidates[0]
            if _supports(claim, excerpt, "", ""):
                return block, excerpt
    return _verified_reader_excerpt(detail)


def refresh_source_citation_cards(
    cite_details: list[dict], *, render_locale: str = ""
) -> list[dict]:
    """Use a literal source excerpt and its own locator in every projection."""
    result = []
    for original in cite_details or []:
        if not isinstance(original, dict):
            continue
        detail = refresh_citation_card_contract(original, locale=render_locale)
        if detail.get("is_inpaper") or detail.get("citation_route") != "system_a":
            result.append(detail)
            continue
        resolved = _source_excerpt(detail)
        if resolved is None:
            # A located plan preview is not a legacy metadata-only card. If
            # no continuous excerpt or supported reader context survives,
            # it cannot remain as an actionable extra card in the packet.
            if detail.get('evidence_source') in {'citation_plan_claim_window', 'verified_source_excerpt'}:
                blocks = scoped_source_blocks(detail)
                if len(blocks) == 1 and all(detail.get(k) and detail[k] == blocks[0].get(k) for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')):
                    continue
            result.append(detail)
            continue
        block, quote = resolved
        quote = _restore_leading_qualifier(block, quote)
        if str(original.get("evidence_quote") or "") != quote:
            detail["source_excerpt_previous_quote"] = (
                original.get("evidence_quote") or original.get("raw") or ""
            )
        old_heading = str(detail.get("heading_path") or "")
        detail.update(
            {
                k: block[k]
                for k in (
                    "block_id",
                    "anchor_id",
                    "heading_path",
                    "page_start",
                    "page_end",
                )
            }
        )
        detail.update(
            anchor_kind=block["kind"],
            strict_locate=True,
            reader_evidence_quote=str(block["raw_text"]),
            reader_evidence_source="verified_source_block",
        )
        if detail.get("evidence_source") != "verified_source_excerpt":
            detail["source_excerpt_previous_source"] = (
                detail.get("evidence_source") or ""
            )
        detail["evidence_source"] = "verified_source_excerpt"
        for key in ("evidence_quote", "card_evidence", "summary_line", "raw"):
            detail[key] = quote
        if old_heading and old_heading != block["heading_path"]:
            for key in ("location_label", "card_locator", "card_subtitle"):
                detail[key] = str(detail.get(key) or "").replace(
                    old_heading, block["heading_path"]
                )
        detail["card_quality_flags"] = [
            f
            for f in detail.get("card_quality_flags", [])
            if f not in {"missing_evidence_quote", "evidence_quote_filtered"}
        ]
        # Refresh labels/quality without passing source text through the prose
        # finisher again: it can remove a reference or append synthetic dots.
        view = dict(detail.get("card_view") or {})
        view['summary'] = detail['summary_line']
        view["header"] = {
            **view.get("header", {}),
            "subtitle": detail.get("card_subtitle") or "",
        }
        view["quality"] = {
            **view.get("quality", {}),
            "flags": detail["card_quality_flags"],
        }
        view["sections"] = [
            {**section, "text": quote}
            if section.get("id") == "evidence"
            else dict(section)
            for section in view.get("sections", [])
        ]
        for section in view["sections"]:
            if section.get("id") == "locator":
                section["text"] = (
                    detail.get("card_locator") or detail.get("location_label") or ""
                )
        if not any(s.get("id") == "evidence" for s in view["sections"]):
            view["sections"].insert(
                0,
                {
                    "id": "evidence",
                    "label": detail.get("card_evidence_label") or "Source evidence",
                    "text": quote,
                    "kind": "quote",
                    "hint": "",
                    "tone": "",
                },
            )
        detail["card_view"] = view
        detail["card_visible_sections"] = [s["id"] for s in view["sections"]]
        result.append(detail)
    return result
