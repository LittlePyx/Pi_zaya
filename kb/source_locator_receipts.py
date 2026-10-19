from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from kb.source_blocks import load_source_blocks


_KIND = 'pdf_inline_prose_structure_v1'
_LOCATOR = ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def _receipt_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.locator-repair.json')


def _transform(kind: str, before: str, pdf: Path) -> tuple[str, bool]:
    if kind == _KIND:
        from kb.converter.pdf_inline_prose import recover_inline_prose_fragments
        return recover_inline_prose_fragments(before, pdf)
    if kind == 'pdf_paragraph_structure_v2':
        from kb.converter.pdf_paragraph_join import recover_pdf_paragraph_structure
        return recover_pdf_paragraph_structure(before, pdf)
    return before, False


def _validated_receipt(path: Path, current: str) -> tuple[dict, list[dict]] | None:
    try:
        receipt_path = _receipt_path(path)
        if not receipt_path.is_file() or receipt_path.stat().st_size > 8_000_000:
            return None
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        if not isinstance(receipt, dict) or Path(receipt['source_path']).resolve() != path.resolve():
            return None
        pdf = Path(receipt['pdf_path']).resolve()
        pdf_root = os.environ.get('KB_PDF_DIR')
        if pdf_root and not pdf.is_relative_to(Path(pdf_root).resolve()):
            return None
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != receipt['pdf_sha256']:
            return None
        history = receipt.get('history', [])
        if not isinstance(history, list):
            return None
        latest = {k:receipt[k] for k in ('kind','before_text','before_sha256','after_sha256')}
        rows = [*history, latest]
        previous = None
        for row in rows:
            before = row['before_text']
            if not isinstance(before, str) or _sha(before) != row['before_sha256'] or (previous is not None and previous != before):
                return None
            fixed, changed = _transform(row['kind'], before, pdf)
            if not changed or _sha(fixed) != row['after_sha256']:
                return None
            previous = fixed
        return (receipt,rows) if previous == current else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_pdf_prose_repair(path: Path, before: str, after: str, pdf_path: Path | None) -> None:
    """Retain each exact PDF-proven structural transition for old citations."""
    receipt = None
    if pdf_path and Path(pdf_path).is_file():
        fixed, changed = _transform('pdf_paragraph_structure_v2',before,Path(pdf_path))
        if changed and fixed == after:
            prior = _validated_receipt(path,before)
            history = prior[1] if prior and Path(prior[0]['pdf_path']).resolve() == Path(pdf_path).resolve() else []
            receipt = {'kind':'pdf_paragraph_structure_v2', 'source_path':str(path.resolve()),
                       'pdf_path':str(Path(pdf_path).resolve()),
                       'pdf_sha256':hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest(),
                       'before_text':before,'before_sha256':_sha(before),'after_sha256':_sha(after),
                       'history':history}
    path.write_text(after,encoding='utf8')
    if receipt is not None:
        # Readers validate the complete chain and the current content hash.
        # Atomic replacement never exposes a partially serialized receipt.
        import tempfile
        fd,name = tempfile.mkstemp(prefix=path.name+'.locator-',suffix='.tmp',dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd,'w',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False)
            temporary.replace(_receipt_path(path))
        finally:
            temporary.unlink(missing_ok=True)


def _verified_transition(path: Path) -> list[tuple] | None:
    try:
        if not _receipt_path(path).is_file():
            return None
        current = path.read_text(encoding='utf8')
        validated = _validated_receipt(path,current)
        if not validated:
            return None
        _receipt,rows = validated
        blocks = load_source_blocks(path,md_text=current)
        receipt_id = _sha(_receipt_path(path).read_text(encoding='utf8'))
        return [(load_source_blocks(path,md_text=row['before_text']),blocks,receipt_id) for row in rows]
    except (OSError,ValueError,TypeError):
        return None


def _relocate_over_history(record: dict, source: str, transitions: list[tuple]) -> dict:
    candidates = [_relocate(record,source,transition) for transition in transitions]
    candidates = [c for c in candidates if c is not record]
    return candidates[0] if candidates and all(c == candidates[0] for c in candidates) else record


def _flat(text: str) -> str:
    return re.sub(r'\s+', '', text)


def _relocate(record: dict, source: str, transition: tuple) -> dict:
    """Require the entire old tuple and literal before following one receipt."""
    old_blocks, new_blocks, receipt_id = transition
    if not all(record.get(k) for k in _LOCATOR):
        return record
    if record.get('source_path') and Path(record['source_path']).resolve() != Path(source).resolve():
        return record
    matches = [b for b in old_blocks if all(str(record.get(k) or '') == str(b.get(k) or '') for k in _LOCATOR)]
    if len(matches) != 1:
        return record
    old = matches[0]
    key = 'evidence_quote' if 'evidence_quote' in record else 'text'
    quote = str(record.get(key) or '').strip()
    if len(quote) < 40 or _flat(quote) not in _flat(str(old.get('raw_text') or '')):
        return record
    # Only a verified removed display wrapper may be discarded. Every inner
    # character, variable, operator and capitalization remains significant.
    literal = quote
    if old.get('kind') == 'equation' and quote.startswith('$$') and quote.endswith('$$'):
        literal = quote[2:-2].strip()
    targets = [b for b in new_blocks if all(b.get(k) == old.get(k) for k in ('page_start', 'page_end', 'heading_path'))
               and _flat(literal) in _flat(str(b.get('raw_text') or ''))]
    if len(targets) != 1:
        return record
    target = targets[0]
    raw = str(target.get('raw_text') or '').strip()
    if len(raw) > 1800 and _flat(raw) == _flat(str(old.get('raw_text') or '')):
        # An unchanged later paragraph may only have shifted block numbers.
        # Keep its existing bounded quotation; never enlarge the quote budget.
        from kb.source_excerpt import literal_excerpt
        raw = literal_excerpt(target, literal)
    if not 40 <= len(raw) <= 1800:
        return record
    result = {**record, **{k: target[k] for k in _LOCATOR}, 'anchor_kind': target['kind'], key: raw,
              'source_locator_repair': {'receipt_sha256': receipt_id,
                                        'prior_locator': {k: record[k] for k in _LOCATOR},
                                        'prior_quote': quote}}
    if 'citation_plan_full_evidence_quote' in result:
        result['citation_plan_full_evidence_quote'] = raw
    return result


def relocate_repaired_source_plan(plan: dict, canonical_paths: list[str]) -> dict:
    """Derive a current plan; all normal support and occurrence gates run next.

    This never changes the stored answer or grants a canonical/grounded flag.
    Missing, stale or contradictory provenance leaves strict rejection intact.
    """
    canonical = {str(Path(p).resolve()) for p in canonical_paths if p}
    transitions = {}
    slots = []
    for slot in plan.get('slots') or []:
        if not isinstance(slot, dict):
            slots.append(slot)
            continue
        source = str(slot.get('source_path') or '')
        identity = str(Path(source).resolve()) if source else ''
        if identity not in canonical:
            slots.append(slot)
            continue
        if identity not in transitions:
            transitions[identity] = _verified_transition(Path(source))
        transition = transitions[identity]
        if not transition:
            slots.append(slot)
            continue
        current = _relocate_over_history(slot, source, transition)
        # Children cannot revive an unverified parent or unrelated source.
        if current is not slot and slot.get('source_passage_bundle') is True:
            children = [_relocate_over_history(c, source, transition) if isinstance(c, dict) else c
                        for c in slot.get('direct_source_evidence') or []]
            current = {**current, 'direct_source_evidence': children,
                       'source_evidence_quotes': [c.get('text', '') for c in children if isinstance(c, dict)]}
        slots.append(current)
    return {**plan, 'slots': slots} if slots != plan.get('slots') else plan
