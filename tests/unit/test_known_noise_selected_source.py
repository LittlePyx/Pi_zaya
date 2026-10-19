from __future__ import annotations

import copy

from kb.direct_question_evidence import prioritize_selected_source_relation_hits
from kb.source_question_facets import noise_model_premise_priority


def test_known_noise_premise_replaces_only_its_own_selected_source(tmp_path, monkeypatch):
    first = tmp_path / 'single.md'; second = tmp_path / 'known.md'
    first.write_text('<!-- kb_page: 1 -->\nWe train on Bernoulli-masked instances of the single noisy image.', encoding='utf8')
    premise = 'N is drawn from a known noise distribution. Because it is known, we can draw an additional synthetic noise sample.'
    second.write_text('<!-- kb_page: 3 -->\n## Method\n\n' + premise, encoding='utf8')
    selected = [{'text': first.read_text(encoding='utf8'), 'meta': {'source_path': str(first)}}, {'text': 'A statistical noise model is required.', 'meta': {'source_path': str(second)}}]
    before = copy.deepcopy(selected)
    prompt = 'Compare the training observations and noise model required by these papers.'
    result = prioritize_selected_source_relation_hits(selected, [], limit=2, prompt=prompt)
    assert selected == before and result[0] == selected[0]
    assert len(result) == 2 and result[1]['text'] == premise
    assert result[1]['meta']['page_start'] == 3
    monkeypatch.setattr('kb.converter.quality_repair.load_conversion_quality_result', lambda _path: {'source_quality': {'evidence_unreliable_pages': [3]}})
    assert prioritize_selected_source_relation_hits(selected, [], limit=2, prompt=prompt) == selected
    assert not noise_model_premise_priority(prompt, 'A prior method uses a known noise distribution and can draw an additional synthetic noise sample.')
