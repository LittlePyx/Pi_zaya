from __future__ import annotations

import re
from pathlib import Path

from kb.citation_marker_context import numeric_marker_is_content
from kb.citation_sentences import citation_sentence_spans
from kb.claim_source_eligibility import retrieval_limit_claim
from kb.source_excerpt import literal_excerpt, scoped_source_blocks

_LABELS = ('已核对的原文片段（阅读上下文）', 'Checked source excerpt (reading context)')
_CITE = re.compile(r'(?<!\[)\[(\d{1,5})\](?!\])')


def _bounded(answer: str) -> bool:
    first = answer.splitlines()[0].lstrip('*#> -') if answer else ''
    first = re.sub(r'^(?:结论|Conclusion|Verdict)\s*[:：]\s*', '', first, flags=re.I)
    return retrieval_limit_claim(first)


def _readable_context(sentence: str) -> bool:
    return bool(80 <= len(sentence) <= 240 and re.search(r'[.!?。！？]["”]?$', sentence)
                and '...' not in sentence and '…' not in sentence
                and not re.search(r'[$`]|(?<![A-Za-z0-9])\d|\b(?:PSNR|SSIM|LPIPS|AUC|sensitivity|results?|values?|performance|accuracy)\b', sentence, re.I)
                and re.search(r'\b(?:assumptions?|limitations?|not\s+(?:evaluated|validated)|comparison\s+may\s+be\s+misleading)\b|适用条件|局限|未验证', sentence, re.I))


def _paths_equal(left: str, right: str) -> bool:
    return bool(left and right and Path(left).resolve() == Path(right).resolve())


def _contexts(plan: dict, paths: list[str]):
    from kb.converter.quality_gate import _unreliable_pages_from_report
    from kb.converter.quality_repair import load_conversion_quality_result

    for slot in plan.get('slots') or []:
        if not isinstance(slot, dict) or slot.get('preferred_system', 'system_a') != 'system_a':
            continue
        path = str(slot.get('source_path') or '')
        numbers = [int(n) for n in slot.get('candidate_hits') or [] if str(n).isdigit() and 0 < int(n) <= len(paths) and _paths_equal(path, paths[int(n) - 1])]
        if not numbers or not Path(path).is_file():
            continue
        blocks = scoped_source_blocks(slot)
        excluded = set(_unreliable_pages_from_report(load_conversion_quality_result(Path(path))))
        raw = str(slot.get('evidence_quote') or '')
        for start, end in citation_sentence_spans(raw):
            sentence = raw[start:end].strip()
            if not _readable_context(sentence):
                continue
            matches = [(b, literal_excerpt(b, sentence)) for b in blocks if int(b.get('page_start') or 0) not in excluded]
            matches = [(b, q) for b, q in matches if q and _readable_context(q)]
            if len(matches) != 1:
                continue
            block, quote = matches[0]
            yield numbers[0], quote, {
                **slot, **{k: block[k] for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')},
                'evidence_quote': quote, 'anchor_kind': block['kind'], 'strict_locate': True,
                'evidence_selection_reason': 'claim_source_passage', 'source_passage_bundle': True,
                'passage_bound_claims': [quote], 'candidate_hits': [numbers[0]],
            }


def _prune_unverified_context_facts(answer: str, plan: dict, paths: list[str]) -> str:
    # A title or a neighbouring result cannot turn a bounded refusal's generic
    # paper overview into directly supported, locatable source evidence.
    from api.claim_passage_plan import _supports
    from kb.converter.quality_gate import _unreliable_pages_from_report
    from kb.converter.quality_repair import load_conversion_quality_result

    quotes: dict[int, list[tuple[str, str]]] = {}
    for slot in plan.get('slots') or []:
        if not isinstance(slot, dict) or slot.get('preferred_system', 'system_a') != 'system_a':
            continue
        path = str(slot.get('source_path') or '')
        numbers = [int(n) for n in slot.get('candidate_hits') or [] if str(n).isdigit() and 0 < int(n) <= len(paths) and _paths_equal(path, paths[int(n) - 1])]
        if not numbers or not Path(path).is_file():
            continue
        excluded = set(_unreliable_pages_from_report(load_conversion_quality_result(Path(path))))
        blocks = [b for b in scoped_source_blocks(slot) if int(b.get('page_start') or 0) not in excluded]
        raw = str(slot.get('evidence_quote') or '')
        pieces = [raw, *(raw[a:b].strip() for a, b in citation_sentence_spans(raw))]
        verified = [(q, Path(path).name) for block in blocks for piece in pieces if (q := literal_excerpt(block, piece))]
        for number in numbers:
            quotes.setdefault(number, []).extend(verified)
    spans = []
    for start, end in citation_sentence_spans(answer):
        unit = answer[start:end]
        numbers = [int(m[1]) for m in _CITE.finditer(unit) if not numeric_marker_is_content(answer, start + m.start())]
        if not numbers or all(any(_supports(unit, q, '', source) for q, source in quotes.get(number, [])) for number in numbers):
            spans.append(unit)
    return ''.join(spans).strip()


def append_refusal_context_excerpt(answer: str, plan: dict, paths: list[str]) -> str:
    """Show one checked, nonnumeric quotation without citing the refusal itself."""
    if not _bounded(answer):
        return answer
    answer = _prune_unverified_context_facts(answer, plan, paths)
    if any(not numeric_marker_is_content(answer, m.start()) for m in _CITE.finditer(answer)):
        return answer
    found = next(_contexts(plan, paths), None)
    if not found:
        return answer
    number, quote, _slot = found
    label = _LABELS[0] if re.search(r'[\u4e00-\u9fff]', answer) else _LABELS[1]
    return answer.rstrip() + f'\n\n{label}\n\n> {quote} [{number}]'


def bind_refusal_context_excerpt(answer: str, plan: dict, paths: list[str]) -> dict:
    """Bind only that explicit literal reading quotation in a copied plan."""
    if not _bounded(answer) or not any(label in answer for label in _LABELS):
        return plan
    cites = [m for m in _CITE.finditer(answer) if not numeric_marker_is_content(answer, m.start())]
    if len(cites) != 1:
        return plan
    quote_line = re.search(r'^> (.+) \[(\d{1,5})\]\s*$', answer, re.M)
    if not quote_line:
        return plan
    for number, quote, context in _contexts(plan, paths):
        if quote != quote_line[1] or number != int(quote_line[2]):
            continue
        slots = list(plan.get('slots') or [])
        index = next((i for i, slot in enumerate(slots) if slot.get('source_path') == context['source_path'] and str(number) in [str(n) for n in slot.get('candidate_hits', [])]), None)
        if index is None:
            continue
        if slots[index] == context:
            return plan
        slots[index] = context
        return {**plan, 'slots': slots}
    return plan
