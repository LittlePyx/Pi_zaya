from __future__ import annotations

from kb.claim_evidence_runtime import _scope_absolute_negative_claims, audit_and_repair_claim_evidence


def test_retrieval_limit_survives_a_colon_joined_paper_claim_without_stealing_its_citation():
    boundary = '当前检索到的证据无法确定这两篇论文在同一项随机临床试验中的患者人数或诊断敏感度差值'
    fact = 'Noisier2Noise reports PSNR comparisons on the KODAK image dataset.'
    answer, _ = audit_and_repair_claim_evidence(boundary + '：' + fact + ' [1]', [{'text': fact, 'meta': {'source_name': 'noisier2noise.pdf'}}], allowed_citation_numbers={1}, drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert boundary in answer
    assert fact in answer and '[1]' in answer
    assert '[1]' not in answer.split(boundary)[0]


def test_retrieval_prefix_cannot_protect_an_unsupported_clinical_fact_after_colon():
    boundary = '当前检索到的证据无法确定患者人数'
    unsupported = 'Both papers report 900 patients with sensitivity of 98 percent.'
    answer, _ = audit_and_repair_claim_evidence(boundary + '：' + unsupported + ' [1]', [{'text': 'Experiments report image denoising PSNR on KODAK.', 'meta': {'source_name': 'paper.pdf'}}], allowed_citation_numbers={1}, drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert boundary in answer
    assert '900' not in answer and '98' not in answer and '[1]' not in answer


def test_math_code_and_link_colons_are_not_boundary_separators():
    for value in ('$A:B$', '`A:B`', '[source](http://example.test/page)'):
        text = '当前检索到的证据无法确定变量 ' + value + ' 的值。'
        result, _ = _scope_absolute_negative_claims(text)
        assert result == text
