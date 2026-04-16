from kb.task_runtime import _sanitize_paper_guide_answer_for_user


def test_sanitize_paper_guide_answer_strips_internal_doc_context_labels():
    raw = (
        "In summary, Figure 1 walks the reader from hardware (a) to APR gains (f,g), "
        "all grounded in the experimental setup and quantitative metrics reported in DOC-1 and DOC-2."
    )
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "DOC-1" not in out
    assert "DOC-2" not in out
    assert "the supporting excerpts" in out


def test_sanitize_paper_guide_answer_keeps_structured_cites_for_non_method_family():
    raw = "The paper overcomes the trade-off between sectioning and SNR [[CITE:s1234abcd:1]]."
    out = _sanitize_paper_guide_answer_for_user(
        raw,
        has_hits=True,
        prompt_family="overview",
    )
    assert "[[CITE:s1234abcd:1]]" not in out


def test_sanitize_paper_guide_answer_canonicalizes_negative_shell_to_does_not_specify():
    raw = "The retrieved paper does not state the GPU model used [[CITE:s1234abcd:1]]."
    out = _sanitize_paper_guide_answer_for_user(
        raw,
        has_hits=True,
        prompt_family="reproduce",
    )
    assert "[[CITE:" not in out
    assert "does not specify the GPU model used" in out
