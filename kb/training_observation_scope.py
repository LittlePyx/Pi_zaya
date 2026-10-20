from __future__ import annotations

from pathlib import Path
import re

from kb.citation_marker_context import numeric_marker_is_content
from kb.n2n_target_condition import _verified_block
from kb.source_question_facets import _additive_training_pair_question


def scope_training_observation_answer(answer: str, *, prompt: str, citation_plan: dict | None, answer_hits: list[dict] | None = None) -> str:
    """Do not import a toy example's observation notation into a training task.

    This only removes prose using an observation expression contradicted by
    one uniquely retrieved and physically verified training procedure. It
    cannot invent an equation, substitute variables, or grant a citation.
    """
    if not _additive_training_pair_question(str(prompt or '')):
        return answer
    rows = []
    for slot in (citation_plan or {}).get('slots') or []:
        if not isinstance(slot,dict):continue
        source = str(slot.get('source_path') or '')
        rows.append((source,{**slot,'text':slot.get('evidence_quote')}))
        rows.extend((source,child) for child in slot.get('direct_source_evidence') or [] if isinstance(child,dict))
    for hit in answer_hits or []:
        meta = hit.get('meta') or {}
        source = str(meta.get('source_path') or '')
        rows.extend((source,row) for row in meta.get('source_passages') or [{**meta,'text':hit.get('text')}] if isinstance(row,dict))
    paths = {str(Path(source).resolve()) for source,_ in rows if source}
    if len(paths) != 1:
        return answer
    procedures = {}
    for source,row in rows:
        if not source:continue
        block = _verified_block(source,row)
        if not block:continue
        raw = str(block.get('raw_text') or '')
        pair = re.search(r'train\s+the\s+network\s+to\s+predict\s+([A-Z])\s+given\s+([A-Z])\s+as\s+input',raw)
        if not pair:continue
        target,network_input = pair.groups()
        compact = re.sub(r'\s+|\$', '',raw)
        chain = re.search(rf'(?<![A-Za-z0-9_]){re.escape(network_input)}(?:≜|:=|=)([A-Z](?:[+-][A-Z])*(?:=[A-Z](?:[+-][A-Z])*)*)',compact)
        if not chain:continue
        procedures.setdefault((target,network_input),set()).update([network_input,*chain[1].split('=')])
    if len(procedures) != 1:
        return answer
    allowed = next(iter(procedures.values()))
    expression = re.compile(r'\b(?:estimate|recover|denois)\w*[^.;\n]{0,60}?\bfrom\s+\$?([A-Z](?:\s*[+-]\s*[A-Z])+)(?![A-Za-z0-9_])',re.I)
    kept = []
    fence = ''
    for paragraph in re.split(r'(\n[ \t]*\n)',answer):
        protected = bool(fence)
        for line in paragraph.splitlines():
            marker = re.match(r'^(`{3,}|~{3,})',line.strip())
            if marker:
                if not fence:fence=marker[1]
                elif marker[1][0]==fence[0] and len(marker[1])>=len(fence):fence=''
                protected=True
        if protected or paragraph.lstrip().startswith(('>','$$')):
            kept.append(paragraph);continue
        invalid = any(not numeric_marker_is_content(paragraph,m.start()) and re.sub(r'\s+','',m[1]) not in allowed
                      for m in expression.finditer(paragraph))
        if not invalid:kept.append(paragraph)
    return ''.join(kept).strip()
