from __future__ import annotations

from api.claim_passage_plan import _verified_passage
from api.reference_ui import _source_block_to_answer_primary_evidence
from kb.source_blocks import load_source_blocks_cached


def test_current_source_primary_keeps_the_full_physical_heading(tmp_path):
    path = tmp_path / 'newmethod.en.md'
    raw = 'We then train the network to predict Y given Z as input, using a pixel-wise loss function.'
    path.write_text('<!-- kb_page: 3 -->\n# NewMethod: A Study of Image Restoration\n## 3. Method\n### 3.2. Training\n\n' + raw, encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if b['kind'] == 'paragraph')
    primary = _source_block_to_answer_primary_evidence(block=block,prompt='What is the input and target?',source_path=str(path),display_name='NewMethod',terms=['train','predict'])
    assert all(primary[k] == block[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end'))
    assert primary['heading_path'].startswith('NewMethod:')
    assert _verified_passage({**primary,'text':primary['snippet']},str(path))
    # Canonicalizing at construction does not authorize contradictory tuples.
    for key,value in (('heading_path','3. Method / 3.2. Training'),('anchor_id','wrong'),('page_start',9)):
        assert not _verified_passage({**primary,key:value,'text':primary['snippet']},str(path))
