from kb.agent.verifier import split_answer_claims, verify_answer_citations


def test_verifier_counts_supported_and_unsupported_claims():
    answer = "Retrieval improves grounding [1]. Unsupported general statement."
    result = verify_answer_citations(
        answer,
        [{"text": "The retrieval module improves grounding quality.", "meta": {"source_name": "demo"}}],
    )

    assert result.total_claims == 2
    assert result.supported_claims == 1
    assert result.unsupported_claims == 1
    assert result.claims[0]["has_citation"] is True
    assert result.claims[1]["has_citation"] is False


def test_split_answer_claims_ignores_tiny_fragments_and_headings():
    assert split_answer_claims("Evidence:\n- Short.\nThe method uses BM25 retrieval [1].") == [
        "The method uses BM25 retrieval [1]."
    ]
