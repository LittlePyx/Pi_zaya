from __future__ import annotations

import copy
import json

import fitz
import pytest

from api.claim_passage_plan import _supports, _verified_passage
from kb.converter.pdf_inline_prose import recover_inline_prose_fragments
from kb.source_blocks import load_source_blocks
from kb.source_locator_receipts import relocate_repaired_source_plan, write_pdf_prose_repair


BEFORE = 'Our training procedure is as follows. We create'
BODY = 'a noisier version, Z := Y + M. We then'
AFTER = 'train the network to predict the observed image using L2 loss.'


def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('KB_PDF_DIR', str(tmp_path))
    pdf = tmp_path / 'study.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        for xy, text in [((62, 100), BEFORE), ((50, 112), BODY), ((50, 124), AFTER)]:
            page.insert_text(xy, text, fontsize=10)
        doc.save(pdf)
    md = tmp_path / 'study.en.md'
    before = '<!-- kb_page: 1 -->\n# Study\n## Method\n\n' + BEFORE + '\n\n$$\n' + BODY + '\n$$\n\n' + AFTER
    md.write_text(before, encoding='utf8')
    blocks = load_source_blocks(md)
    equation = next(b for b in blocks if b['kind'] == 'equation')
    training = next(b for b in blocks if b.get('raw_text') == AFTER)
    def record(block, key):
        return {**{k: block[k] for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end')},
                'source_path': str(md), key: block['raw_text']}
    slot = {**record(training, 'evidence_quote'), 'source_passage_bundle': True,
            'direct_source_evidence': [record(equation, 'text')]}
    fixed, changed = recover_inline_prose_fragments(before, pdf)
    assert changed
    write_pdf_prose_repair(md, before, fixed, pdf)
    return pdf, md, before, fixed, {'slots': [slot]}


def test_pdf_proven_join_derives_current_full_locators_without_mutating_the_plan(tmp_path, monkeypatch):
    _, md, _, fixed, plan = fixture(tmp_path, monkeypatch)
    original = copy.deepcopy(plan)
    result = relocate_repaired_source_plan(plan, [str(md)])
    assert plan == original
    slot = result['slots'][0]
    child = slot['direct_source_evidence'][0]
    block = next(b for b in load_source_blocks(md) if b['kind'] == 'paragraph')
    for record, key in ((slot, 'evidence_quote'), (child, 'text')):
        assert record[key] == block['raw_text'] and record[key] in fixed
        assert all(record[k] == block[k] for k in ('block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end'))
        assert _verified_passage({**record, 'text': record[key]}, str(md))
        assert 'canonical_evidence_contract_grounded' not in record
    assert not _supports('The study uses 300 clinical subjects.', slot['evidence_quote'], '', '')
    assert relocate_repaired_source_plan(result, [str(md)]) == result


def test_shifted_unchanged_long_paragraph_keeps_its_bounded_literal_and_failed_flags(tmp_path, monkeypatch):
    from api.source_card_contract import prepare_source_card_message

    pdf, md, before, _, _ = fixture(tmp_path, monkeypatch)
    quote = 'It is important to note that these methods make different assumptions about the task, and thus direct comparison may be misleading.'
    long = ('The experiment describes the available training observations. ' * 40) + quote
    before += '\n\n## Comparison\n\n' + long
    md.write_text(before, encoding='utf8')
    old = next(b for b in load_source_blocks(md) if b['raw_text'] == long)
    slot = {**{k: old[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')},
            'source_path': str(md), 'evidence_quote': quote, 'candidate_hits': [1], 'preferred_system': 'system_a'}
    fixed, changed = recover_inline_prose_fragments(before, pdf)
    assert changed
    write_pdf_prose_repair(md, before, fixed, pdf)
    plan = {'slots': [slot]}
    moved = relocate_repaired_source_plan(plan, [str(md)])
    assert moved != plan and moved['slots'][0]['evidence_quote'] == quote
    assert _verified_passage({**moved['slots'][0], 'text': quote}, str(md))
    quality = {'has_hits': True, 'minimum_ok': False, 'paper_count_ok': False,
               'claim_evidence': {'minimum_ok': False}, 'citation_plan': plan}
    message = {'role': 'assistant', 'content': 'The retrieved evidence cannot establish a universal ranking.\n\nChecked source excerpt (reading context)\n\n> ' + quote + ' [1]',
               'meta': {'answer_quality': quality, 'canonical_hit_paths': [str(md)]}}
    original = copy.deepcopy(message)
    derived = prepare_source_card_message(message)
    assert message == original and derived['content'] == message['content']
    assert derived['meta']['answer_quality']['minimum_ok'] is False
    assert derived['meta']['answer_quality']['paper_count_ok'] is False
    assert derived['meta']['answer_quality']['claim_evidence']['minimum_ok'] is False
    current = derived['meta']['answer_quality']['citation_plan']['slots'][0]
    assert current['evidence_quote'] == quote and current['passage_bound_claims'] == [quote]
    assert _verified_passage({**current, 'text': quote}, str(md))
    assert 'canonical_evidence_contract_grounded' not in current
    bad = copy.deepcopy(plan); bad['slots'][0]['anchor_id'] = 'wrong'
    assert relocate_repaired_source_plan(bad, [str(md)]) == bad


@pytest.mark.parametrize('failure', ['missing', 'current', 'pdf', 'before', 'after_hash', 'kind', 'source', 'pdf_root',
                                     'block_id', 'anchor_id', 'heading_path', 'page_start', 'page_end',
                                     'variable', 'operator', 'case', 'index', 'canonical'])
def test_stale_or_contradictory_provenance_cannot_revive_a_plan(tmp_path, monkeypatch, failure):
    pdf, md, _, fixed, plan = fixture(tmp_path, monkeypatch)
    receipt = md.with_suffix(md.suffix + '.locator-repair.json')
    data = json.loads(receipt.read_text(encoding='utf8'))
    paths = [str(md)]
    slot = plan['slots'][0]
    if failure == 'missing': receipt.unlink()
    if failure == 'current': md.write_text(fixed + '\nChanged conclusion.', encoding='utf8')
    if failure == 'pdf': pdf.write_bytes(pdf.read_bytes() + b'\nChanged PDF')
    if failure == 'before': data['before_text'] += 'Inserted unsupported source.'
    if failure == 'after_hash': data['after_sha256'] = '0' * 64
    if failure == 'kind': data['kind'] = 'manual_edit'
    if failure == 'source': data['source_path'] = str(tmp_path / 'other.md')
    if failure == 'pdf_root': monkeypatch.setenv('KB_PDF_DIR', str(tmp_path / 'other'))
    if failure in {'block_id', 'anchor_id', 'heading_path'}: slot[failure] = 'wrong'
    if failure in {'page_start', 'page_end'}: slot[failure] = 2
    # Contradictory variable/operator/case/index in a declared child remains
    # stale even when its valid parent can follow the source repair.
    if failure in {'variable', 'operator', 'case', 'index'}:
        old = slot['direct_source_evidence'][0]['text']
        replacements = {'variable': ('Y + M', 'X + M'), 'operator': ('Y + M', 'Y - M'),
                        'case': ('Z :=', 'z :='), 'index': ('Z :=', 'Z_2 :=')}
        a, b = replacements[failure]
        slot['direct_source_evidence'][0]['text'] = old.replace(a, b)
    if failure == 'canonical': paths = []
    if failure in {'before', 'after_hash', 'kind', 'source'}:
        receipt.write_text(json.dumps(data), encoding='utf8')
    result = relocate_repaired_source_plan(plan, paths)
    if failure in {'variable', 'operator', 'case', 'index'}:
        child = result['slots'][0]['direct_source_evidence'][0]
        assert child == slot['direct_source_evidence'][0]
        assert not _verified_passage(child, str(md))
    else:
        assert result == plan


def test_arbitrary_edits_do_not_issue_a_source_repair_receipt(tmp_path, monkeypatch):
    pdf, md, before, fixed, _ = fixture(tmp_path, monkeypatch)
    receipt = md.with_suffix(md.suffix + '.locator-repair.json')
    receipt.unlink()
    changed = fixed.replace('Y + M', 'Y - M')
    write_pdf_prose_repair(md, before, changed, pdf)
    assert md.read_text(encoding='utf8') == changed
    assert not receipt.exists()


def test_duplicate_current_paragraph_is_not_a_unique_repair_target(tmp_path, monkeypatch):
    pdf, md, before, fixed, plan = fixture(tmp_path, monkeypatch)
    joined = BEFORE + ' ' + BODY + ' ' + AFTER
    before += '\n\n' + joined
    fixed += '\n\n' + joined
    write_pdf_prose_repair(md, before, fixed, pdf)
    assert relocate_repaired_source_plan(plan, [str(md)]) == plan


def test_recomputed_hashes_do_not_substitute_for_pdf_geometry(tmp_path, monkeypatch):
    import hashlib

    _, md, before, fixed, plan = fixture(tmp_path, monkeypatch)
    receipt = md.with_suffix(md.suffix + '.locator-repair.json')
    data = json.loads(receipt.read_text(encoding='utf8'))
    data['before_text'] = before.replace('Y + M', 'Y - M')
    data['before_sha256'] = hashlib.sha256(data['before_text'].encode()).hexdigest()
    current = fixed.replace('Y + M', 'Y - M')
    data['after_sha256'] = hashlib.sha256(current.encode()).hexdigest()
    md.write_text(current, encoding='utf8')
    receipt.write_text(json.dumps(data), encoding='utf8')
    assert relocate_repaired_source_plan(plan, [str(md)]) == plan
