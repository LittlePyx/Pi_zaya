from __future__ import annotations

from pathlib import Path
import re

from kb.citation_marker_context import numeric_marker_is_content
from kb.citation_sentences import citation_sentence_spans
from kb.claim_source_eligibility import retrieval_limit_claim
from kb.explicit_source_attribution import source_attribution_covered
from kb.n2n_target_condition import _verified_block
from kb.source_blocks import load_source_blocks_cached


_CITE = re.compile(r'(?<!\[)\[(\d{1,4})\](?!\])')
_LOCATOR = ('block_id','anchor_id','heading_path','page_start','page_end')


def _path(value: str) -> str:
    return str(Path(value).resolve()).replace('\\','/').casefold() if value else ''


def _literal_quote(source: str, row: dict) -> str:
    if row.get('source_path') and _path(str(row['source_path'])) != _path(source):
        return ''
    quote = str(row.get('evidence_quote') or row.get('text') or '').strip()
    flat = lambda text: re.sub(r'\s+','',text)
    if len(quote)<40:return ''
    if all(row.get(k) for k in _LOCATOR):
        block = _verified_block(source,row)
        return quote if block and flat(quote) in flat(str(block.get('raw_text') or '')) else ''
    # A preflight slot can omit its locator. Resolve only its unique literal
    # passage for this negative check; never export a locator or support flag.
    try:
        matches=[b for b in load_source_blocks_cached(Path(source))
                 if all(not row.get(k) or row[k]==b.get(k) for k in _LOCATOR)
                 and flat(quote) in flat(str(b.get('raw_text') or ''))]
    except (OSError,ValueError):return ''
    return quote if len(matches)==1 else ''


def scope_explicit_source_facts(answer: str, *, citation_plan: dict | None, answer_hits: list[dict] | None, slot_hit_numbers) -> str:
    """Omit explicitly unsupported additions before the answer quality gate.

    Only a cited source, one paragraph-end source, or a single verified source
    shared by every supplied hit is eligible for this negative check.
    Source text must be supplied by the existing plan/retrieval pack and pass
    complete physical/literal verification. No new evidence or fact is added.
    """
    hits = list(answer_hits or [])
    sources = {}
    quotes = {}
    names = {}
    for slot in (citation_plan or {}).get('slots') or []:
        if not isinstance(slot,dict) or slot.get('preferred_system')!='system_a':continue
        source = str(slot.get('source_path') or '')
        quote = _literal_quote(source,slot) if source else ''
        if not quote:continue
        identity = _path(source)
        for n in slot_hit_numbers(slot,hits):
            if 0<int(n)<=len(hits) and _path(str((hits[int(n)-1].get('meta') or {}).get('source_path') or ''))==identity:
                sources.setdefault(int(n),set()).add(identity)
        quotes.setdefault(identity,[]).append(quote);names[identity]=str(slot.get('source_name') or source)
        for child in slot.get('direct_source_evidence') or []:
            if isinstance(child,dict) and (q:=_literal_quote(source,child)):
                quotes[identity].append(q)
    for hit in hits:
        meta = hit.get('meta') or {};source=str(meta.get('source_path') or '');identity=_path(source)
        if identity not in quotes:continue
        for row in meta.get('source_passages') or []:
            if isinstance(row,dict) and (q:=_literal_quote(source,row)):
                quotes[identity].append(q)
    single_source = set(quotes) if len(quotes)==1 and {_path(str((h.get('meta') or {}).get('source_path') or '')) for h in hits}==set(quotes) else set()
    removals=[]
    for paragraph in re.finditer(r'[^\n]+(?:\n(?![ \t]*\n)[^\n]+)*',answer):
        body=paragraph[0]
        if re.search(r'(?m)^\s*(?:>|```|~~~|\$\$)',body):continue
        numbers=[int(m[1]) for m in _CITE.finditer(body) if not numeric_marker_is_content(answer,paragraph.start()+m.start())]
        inherited=set().union(*(sources.get(n,set()) for n in numbers)) if numbers else set()
        if not numbers and not body.lstrip().startswith(('"', "'", '“', '‘')):
            inherited = single_source
        for start,end in citation_sentence_spans(body):
            unit=body[start:end]
            own=[int(m[1]) for m in _CITE.finditer(unit) if not numeric_marker_is_content(answer,paragraph.start()+start+m.start())]
            paths=set().union(*(sources.get(n,set()) for n in own)) if own else inherited if len(inherited)==1 else set()
            if not paths or retrieval_limit_claim(unit) or numeric_marker_is_content(answer,paragraph.start()+start):continue
            evidence=[(q,names[p]) for p in paths for q in quotes.get(p,[])]
            # Routing markers precede citation formatting. They are not method
            # names (e.g. SUPPORT/DOC) in a paper's baseline enumeration.
            claim = re.sub(r'\[\[SUPPORT:DOC-\d+\]\]', lambda m: m[0] if numeric_marker_is_content(unit,m.start()) else '', unit)
            if evidence and not any(source_attribution_covered(claim,q,name) for q,name in evidence):
                removals.append((paragraph.start()+start,paragraph.start()+end))
    for start,end in reversed(removals):answer=answer[:start]+answer[end:]
    return answer.strip()
