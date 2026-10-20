from __future__ import annotations

import copy
import hashlib
import json
import re

import fitz
import pytest

from api.claim_passage_plan import _verified_passage
from kb.converter.pdf_inline_prose import recover_inline_prose_fragments
from kb.converter.pdf_paragraph_join import join_verified_pdf_paragraphs, _mathcal_fonts_agree
from kb.converter.quality_repair import repair_markdown_text
from kb.source_blocks import load_source_blocks
from kb.source_locator_receipts import relocate_repaired_source_plan, write_pdf_prose_repair


LINES=['We consider a scenario with a known noise distribution.',
       'We observe Y := X + N and draw an additional sample M.',
       'The sample is drawn from the same known distribution.',
       'We then use the resulting observations to train our network.']


def fixture(tmp_path,layout='normal',training=False):
    pdf=tmp_path/'source.pdf'
    with fitz.open() as doc:
        page=doc.new_page()
        for repeat in range(2 if layout=='duplicate' else 1):
            for i,line in enumerate(LINES):
                x=62 if i==0 else 90 if i==1 and layout=='centered' else 62 if i==2 and layout=='new_paragraph' else 50
                if layout=='no_indent':x=50
                if layout=='column' and i==2:x=330
                y=100+repeat*240+i*12+(60 if layout=='gap' and i>=2 else 0)
                page.insert_text((x,y),line,fontsize=10)
        if training:
            for i,line in enumerate(['Our training procedure is as follows. We create','a noisier version, Z := Y + M. We then','train the network to predict the observed image using L2 loss.']):
                page.insert_text((62 if i==0 else 50,200+i*12),line,fontsize=10)
        doc.new_page();doc.save(pdf)
    text='<!-- kb_page: 1 -->\n\n# Study\n\n## Method\n\n'+'\n\n'.join(LINES)+'\n'
    return pdf,text


def test_complete_pdf_paragraph_is_joined_without_changing_any_token(tmp_path):
    pdf,text=fixture(tmp_path)
    result,changed=join_verified_pdf_paragraphs(text,pdf)
    assert changed and ' '.join(LINES) in result
    assert re.sub(r'\s+','',result)==re.sub(r'\s+','',text)
    assert join_verified_pdf_paragraphs(result,pdf)==(result,False)
    repair=repair_markdown_text(tmp_path/'source.en.md',text,issue_codes=['fragmented_pdf_prose'],source_pdf_path=pdf)
    assert repair['changed'] and repair['repaired_text']==result
    assert 'postprocess_markdown' not in repair['applied']


@pytest.mark.parametrize('layout',['centered','new_paragraph','column','no_indent','gap','duplicate'])
def test_geometry_and_unique_page_identity_are_required(tmp_path,layout):
    pdf,text=fixture(tmp_path,layout)
    assert join_verified_pdf_paragraphs(text,pdf)==(text,False)


@pytest.mark.parametrize('failure',['variable','operator','case','index','font','page','page_break','code','indented','math','references'])
def test_changed_math_or_protected_scopes_are_not_repaired(tmp_path,failure):
    pdf,text=fixture(tmp_path)
    if failure=='variable':text=text.replace('X + N','X + M')
    if failure=='operator':text=text.replace('X + N','X - N')
    if failure=='case':text=text.replace('Y :=','y :=')
    if failure=='index':text=text.replace('Y :=','Y_2 :=')
    if failure=='font':text=text.replace('X + N',r'$\mathcal{X}$ + N')
    if failure=='page':text=text.replace('kb_page: 1','kb_page: 2')
    if failure=='page_break':text=text.replace(LINES[1],'<!-- kb_page: 2 -->\n\n'+LINES[1])
    if failure=='code':text='```markdown\n'+text+'\n```'
    if failure=='indented':text=text.replace(LINES[0],'    '+LINES[0])
    if failure=='math':text='<!-- kb_page: 1 -->\n\n$$\n\n'+text+'\n\n$$'
    if failure=='references':text=text.replace('## Method','## 6 References')
    assert join_verified_pdf_paragraphs(text,pdf)==(text,False)


def test_mathematical_font_roles_are_checked_in_both_directions():
    assert _mathcal_fonts_agree(r'$\mathcal{A}$',[('A','CMSY10')])
    assert not _mathcal_fonts_agree('A',[('A','CMSY10')])
    assert not _mathcal_fonts_agree(r'$\mathcal{A}$',[('A','CMMI10')])


def test_two_real_structural_transitions_keep_both_old_locator_versions(tmp_path,monkeypatch):
    monkeypatch.setenv('KB_PDF_DIR',str(tmp_path))
    pdf,before=fixture(tmp_path,training=True)
    before+='\nOur training procedure is as follows. We create\n\n$$\na noisier version, Z := Y + M. We then\n$$\n\ntrain the network to predict the observed image using L2 loss.\n'
    md=tmp_path/'source.en.md';md.write_text(before,encoding='utf8')
    old_blocks=load_source_blocks(md,md_text=before)
    first,changed=recover_inline_prose_fragments(before,pdf);assert changed
    md.write_text(first,encoding='utf8')
    sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
    legacy={'kind':'pdf_inline_prose_structure_v1','source_path':str(md.resolve()),'pdf_path':str(pdf.resolve()),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'before_text':before,'before_sha256':sha(before),'after_sha256':sha(first)}
    receipt=md.with_suffix(md.suffix+'.locator-repair.json');receipt.write_text(json.dumps(legacy),encoding='utf8')
    def plan(block):
        return {'slots':[{**{k:block[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')},'source_path':str(md),'evidence_quote':block['raw_text']}]}
    older=plan(next(b for b in old_blocks if b['kind']=='equation'))
    newer=plan(next(b for b in load_source_blocks(md,md_text=first) if b.get('raw_text')==LINES[1]))
    final,changed=join_verified_pdf_paragraphs(first,pdf);assert changed
    write_pdf_prose_repair(md,first,final,pdf)
    data=json.loads(receipt.read_text(encoding='utf8'));assert len(data['history'])==1
    for original in (older,newer):
        preserved=copy.deepcopy(original);moved=relocate_repaired_source_plan(original,[str(md)])
        assert original==preserved and moved!=original
        slot=moved['slots'][0]
        assert _verified_passage({**slot,'text':slot['evidence_quote']},str(md))
    data['history'][0]['before_text']=before.replace('Y + M','Y - M')
    data['history'][0]['before_sha256']=sha(data['history'][0]['before_text'])
    receipt.write_text(json.dumps(data),encoding='utf8')
    assert relocate_repaired_source_plan(older,[str(md)])==older
    assert relocate_repaired_source_plan(newer,[str(md)])==newer
