from __future__ import annotations

import os

import pytest

from api.claim_passage_plan import _verified_passage
from kb.explicit_training_assignments import explicit_training_assignments_covered
from kb.noise_model_prerequisites import noise_model_prerequisites_covered
from kb.source_blocks import load_source_blocks
from kb.training_observation_scope import scope_training_observation_answer


RAW='Our training procedure is as follows. Given Y, we create a noisier version, Z := Y + M = X + N + M. We then train the network to predict Y given Z as input, using a pixel-wise loss.'
PROMPT='In the standard additive procedure, what is the input and training target?'


def fixture(tmp_path):
    path=tmp_path/'paper.en.md';path.write_text('<!-- kb_page: 3 -->\n# Paper\n## Method\n\n'+RAW,encoding='utf8')
    b=next(b for b in load_source_blocks(path) if b['kind']=='paragraph')
    slot={**{k:b[k] for k in ('block_id','anchor_id','heading_path','page_start','page_end')},'source_path':str(path),'evidence_quote':RAW}
    return path,slot


def test_observation_is_not_a_training_target_declaration():
    observation='We have an unobserved clean image X and observe a single noisy realization Y of it.'
    correct='The training target is the observed noisy image Y, not the unseen clean image X.'
    assert not explicit_training_assignments_covered(correct,observation)
    assert explicit_training_assignments_covered(correct,RAW)
    assert not explicit_training_assignments_covered('The training target is the unseen clean image X.',RAW)
    assert not explicit_training_assignments_covered('The network input is the observed noisy image Y.',RAW)
    assert not explicit_training_assignments_covered('The network input is the doubly-noisy image.', 'We can draw additional synthetic noise samples.')


@pytest.mark.parametrize('source,expected',[
    ('We learn from single noisy measurements of each object.',True),
    ('We learn from multiple noisy measurements of each object.',False),
    ('We learn from single clean measurements of each object.',False),
])
def test_single_noisy_measurement_preserves_quantity_and_noise_state(source,expected):
    assert bool(noise_model_prerequisites_covered('仅凭单次含噪观测学习去噪函数。',source)) is expected


def test_toy_observation_notation_is_not_imported_into_a_verified_training_procedure(tmp_path):
    path,slot=fixture(tmp_path);plan={'slots':[slot]}
    answer='The network predicts Y from Z.\n\nIts output can be corrected to estimate X from X + Z. In other words, X enters only through correction.\n\nRead the method for its assumptions.'
    result=scope_training_observation_answer(answer,prompt=PROMPT,citation_plan=plan)
    assert 'X + Z' not in result and 'X enters only' not in result
    assert 'predicts Y from Z' in result and 'Read the method' in result
    assert scope_training_observation_answer('We recover X from X + N + M.',prompt=PROMPT,citation_plan=plan)=='We recover X from X + N + M.'
    assert scope_training_observation_answer(answer,prompt='Explain the motivation example.',citation_plan=plan)==answer
    assert scope_training_observation_answer(answer,prompt='In the non-additive procedure, what is the input and target?',citation_plan=plan)==answer
    assert scope_training_observation_answer(answer,prompt=PROMPT,citation_plan={'slots':[{**slot,'anchor_id':'wrong'}]})==answer
    quoted='> We recover X from X + Z.\n\n```python\nrecover X from X + Z\n```'
    assert scope_training_observation_answer(quoted,prompt=PROMPT,citation_plan=plan)==quoted


def test_same_second_same_size_source_change_invalidates_literal_proof(tmp_path):
    path,slot=fixture(tmp_path)
    tick=(path.stat().st_mtime_ns//1_000_000_000)*1_000_000_000+100_000_000
    os.utime(path,ns=(tick,tick))
    assert _verified_passage({**slot,'text':RAW},str(path))
    size=path.stat().st_size
    changed=path.read_text(encoding='utf8').replace('Y + M','Y - M')
    path.write_text(changed,encoding='utf8');os.utime(path,ns=(tick+100_000,tick+100_000))
    assert path.stat().st_size==size and path.stat().st_mtime_ns//1_000_000_000==tick//1_000_000_000
    assert not _verified_passage({**slot,'text':RAW},str(path))
