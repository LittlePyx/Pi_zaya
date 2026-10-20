from __future__ import annotations

import copy
from pathlib import Path

from kb.citation_sentences import citation_sentence_spans, split_definition_bridges
from kb.direct_question_evidence import direct_question_evidence_hits
from kb.local_retrieval_selector import select_product_retrieval
from kb.local_semantic_runtime import LocalSemanticRuntime


def test_separate_input_and_synthetic_draw_definitions_without_changing_the_answer():
    text = ('Given the observed noisy image $Y \\triangleq X+N$, the method creates a noisier version '
            '$Z \\triangleq Y+M=X+N+M$, where $M$ is an additional synthetic noise sample drawn from the known distribution.')
    spans = citation_sentence_spans(text)
    assert ''.join(text[a:b] for a, b in spans) == text
    assert len(spans) == 3
    parts = split_definition_bridges(text)
    assert len(parts) == 3
    assert 'Y \\triangleq X+N' in parts[0]
    assert 'Z \\triangleq Y+M' in parts[1]
    assert parts[2].startswith('where $M$ is an additional synthetic noise')


def test_non_definition_commas_and_literal_content_are_not_split():
    for text in (
        'The method estimates a denoised image, the method creates a prediction.',
        'A variable is named, where $M$ is an unknown target.',
        'The caption says "Y=X+N, the method creates Z=Y+M".',
        '$h(Y, the method creates Z=Y+M)$',
        '`Y=X+N, the method creates Z=Y+M`',
    ):
        assert len(split_definition_bridges(text)) == 1


def test_reconstruction_question_retrieves_the_explicit_operation(tmp_path: Path):
    paper = tmp_path / 'denoiser.md'
    paper.write_text(
        '# Residual denoiser\n\n<!-- kb_page: 1 -->\n\n'
        'Our network is designed to predict the residual image, the difference between the noisy observation and the latent clean image.\n\n'
        '<!-- kb_page: 2 -->\n\n'
        'The input is a noisy observation y = x + v. We train a residual mapping R(y) ≈ v and then have x = y − R(y).\n', encoding='utf8')
    hits = direct_question_evidence_hits(paper, 'Does the DnCNN network output the denoised image directly? How is the denoised image obtained?')
    assert hits and 'x = y − R(y)' in hits[0]['text']
    assert hits[0]['meta']['page_start'] == 2
    paper.write_text(paper.read_text().replace('x = y − R(y)', 'x = y + R(y)'), encoding='utf8')
    assert not any('x = y + R(y)' in h['text'] for h in direct_question_evidence_hits(
        paper, 'Does the DnCNN network output the denoised image directly?'))


def test_distinct_physical_blocks_with_equal_prose_keep_the_hybrid_prefix(tmp_path: Path):
    paper = tmp_path / 'denoiser.md'
    paragraph = 'Our model combines residual learning and batch normalization for fast and stable training.'
    paper.write_text('# Denoiser\n\n' + '\n\n'.join(
        f'<!-- kb_page: {page} -->\n\n{paragraph}' for page in range(1, 7)), encoding='utf8')
    sparse = direct_question_evidence_hits(paper, 'How do residual learning and batch normalization accelerate and stabilize training?')
    assert len(sparse) >= 5
    original = copy.deepcopy(sparse)
    class Dense:
        is_empty = False
        def search_scoped(self, query, *, source_paths, top_k=6):
            return [{'id': 'novel', 'text': 'other evidence', 'score': 1., 'meta': {'source_path': str(paper)}}]
    class Service:
        def load_ready(self, db_dir):
            return LocalSemanticRuntime(Dense(), index_cache_hit=True, cold_start=False)
    result = select_product_retrieval(query='training', sparse_hits=sparse, db_dir=tmp_path,
        retrieval_mode='hybrid_local', source_scope='current_paper', source_paths=[str(paper)],
        runtime_service=Service(), product_enabled=True)
    assert result.used_mode == 'hybrid_local'
    assert [h['id'] for h in result.hits[:5]] == [h['id'] for h in original[:5]]
    assert all(all(h[k] == v for k, v in before.items()) for h, before in zip(result.hits[:5], original[:5]))
    assert len({h['id'] for h in sparse}) == len(sparse)
    assert sparse == original
