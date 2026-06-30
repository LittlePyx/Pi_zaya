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
    assert result.evidence_status == "needs_review"
    assert "unsupported_claims" in result.evidence_status_reasons
    assert result.claims[0]["has_citation"] is True
    assert result.claims[0]["citation_present"] is True
    assert result.claims[0]["matched_evidence_count"] == 1
    assert result.claims[0]["matched_sources"][0]["source_name"] == "demo"
    assert result.claims[0]["unsupported_reason"] == ""
    assert result.claims[1]["has_citation"] is False
    assert result.claims[1]["unsupported_reason"] == "missing_citation"


def test_verifier_flags_cited_claim_without_matching_evidence():
    result = verify_answer_citations(
        "The model uses contrastive decoding [1].",
        [{"text": "The retrieval module improves grounding quality.", "meta": {"source_name": "demo"}}],
    )

    assert result.total_claims == 1
    assert result.supported_claims == 0
    assert result.unsupported_claims == 1
    assert result.evidence_status == "insufficient"
    assert "no_supported_claims" in result.evidence_status_reasons
    assert result.claims[0]["citation_present"] is True
    assert result.claims[0]["matched_evidence_count"] == 0
    assert result.claims[0]["matched_sources"] == []
    assert result.claims[0]["unsupported_reason"] == "missing_evidence_overlap"


def test_split_answer_claims_ignores_tiny_fragments_and_headings():
    assert split_answer_claims("Evidence:\n- Short.\nThe method uses BM25 retrieval [1].") == [
        "The method uses BM25 retrieval [1]."
    ]


def test_verifier_marks_fully_supported_answer_as_grounded():
    result = verify_answer_citations(
        "The retrieval module improves grounding quality [1].",
        [
            {"text": "The retrieval module improves grounding quality.", "meta": {"source_name": "demo-a"}},
            {"text": "Additional retrieval evidence is available.", "meta": {"source_name": "demo-b"}},
        ],
    )

    assert result.evidence_status == "grounded"
    assert result.evidence_hit_count == 2
    assert result.evidence_status_reasons == []


def test_verifier_marks_no_evidence_as_insufficient():
    result = verify_answer_citations("The model uses retrieval [1].", [])

    assert result.evidence_status == "insufficient"
    assert result.evidence_hit_count == 0
    assert "no_evidence_hits" in result.evidence_status_reasons


def test_verifier_ignores_source_notice_lines():
    result = verify_answer_citations(
        "Note: local citations [n] come from the knowledge base; uncited background may use external model context.\n\n"
        "The retrieval module improves grounding quality [1].",
        [{"text": "The retrieval module improves grounding quality.", "meta": {"source_name": "demo"}}],
    )

    assert result.total_claims == 1
    assert result.supported_claims == 1


def test_verifier_ignores_chinese_source_notice_lines():
    result = verify_answer_citations(
        "\u6ce8\u610f\uff1a\u5e26 [n] \u7684\u5185\u5bb9\u6765\u81ea\u672c\u5730\u77e5\u8bc6\u5e93\uff1b"
        "\u672a\u5e26\u672c\u5730\u5f15\u7528\u7684\u80cc\u666f\u89e3\u91ca\u53ef\u80fd\u6765\u81ea\u5916\u90e8\u6a21\u578b\u8865\u5145\u3002\n\n"
        "The retrieval module improves grounding quality [1].",
        [{"text": "The retrieval module improves grounding quality.", "meta": {"source_name": "demo"}}],
    )

    assert result.total_claims == 1
    assert result.supported_claims == 1
    assert result.source_notice_count == 1


def test_verifier_tracks_hybrid_external_background_separately():
    result = verify_answer_citations(
        "Note: local citations [n] come from the knowledge base; uncited background may use external model context.\n\n"
        "The paper uses retrieval before generation [1].\n"
        "External context: retrieval commonly helps reduce unsupported generation.",
        [
            {"text": "The paper uses retrieval before generation.", "meta": {"source_name": "demo-a"}},
            {"text": "Additional retrieval evidence is available.", "meta": {"source_name": "demo-b"}},
        ],
        answer_mode="hybrid_local_external",
    )

    assert result.total_claims == 1
    assert result.local_claims == 1
    assert result.supported_claims == 1
    assert result.unsupported_claims == 0
    assert result.external_background_claims == 1
    assert result.source_notice_count == 1
    assert result.evidence_status == "grounded"
    assert result.claims[0]["claim_kind"] == "local_claim"
    assert result.claims[1]["claim_kind"] == "external_background"
    assert result.claims[1]["supported"] is None


def test_verifier_still_flags_uncited_local_claim_in_hybrid_answer():
    result = verify_answer_citations(
        "Note: local citations [n] come from the knowledge base; uncited background may use external model context.\n\n"
        "The paper uses retrieval before generation.\n"
        "Background: retrieval often improves answer grounding.",
        [{"text": "The paper uses retrieval before generation.", "meta": {"source_name": "demo"}}],
        answer_mode="hybrid_local_external",
    )

    assert result.total_claims == 1
    assert result.supported_claims == 0
    assert result.unsupported_claims == 1
    assert result.external_background_claims == 1
    assert result.claims[0]["unsupported_reason"] == "missing_citation"
    assert result.evidence_status == "insufficient"
