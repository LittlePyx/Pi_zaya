from kb.paper_guide_postprocess import _strip_model_ref_section
from kb.task_runtime import _sanitize_paper_guide_answer_for_user


def test_sanitize_paper_guide_answer_strips_internal_doc_context_labels():
    raw = (
        "In summary, Figure 1 walks the reader from hardware (a) to APR gains (f,g), "
        "all grounded in the experimental setup and quantitative metrics reported in DOC-1 and DOC-2."
    )
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "DOC-1" not in out
    assert "DOC-2" not in out
    assert "the supporting excerpts" not in out
    assert "source evidence" in out


def test_strip_model_ref_section_removes_leading_internal_reference_block():
    assert _strip_model_ref_section("Reference locate\n- [1] internal trace") == ""
    assert _strip_model_ref_section("Answer first.\n\nReference locate\n- [1] internal trace") == "Answer first."


def test_sanitize_paper_guide_answer_strips_empty_reference_number_shells():
    raw = "这句话列举了 5 类应用方向及其对应文献编号（共 6 篇参考文献：,,,, ），说明压缩感知的适用范围。"
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "共 6 篇参考文献" not in out
    assert ",,,," not in out
    assert "说明压缩感知的适用范围" in out


def test_sanitize_paper_guide_answer_strips_empty_reference_bullets_after_cite_removal():
    raw = (
        "这句话在同一句中列举了 5 类应用方向，并分别标注了对应参考文献编号：\n\n"
        "- 显微成像（microscopy）：\n"
        "- 遥感（remote sensing）：\n\n"
        "作者想表达的对比点是：压缩感知具有跨领域普适性。"
    )
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "显微成像（microscopy）：" not in out
    assert "遥感（remote sensing）：" not in out
    assert "并在原句中保留了对应参考文献编号" in out
    assert "作者想表达的对比点" in out


def test_sanitize_paper_guide_answer_cleans_empty_intro_tail_and_refs_label():
    raw = (
        "位于 原文证据的首句：这句话引用了多篇文献（ refs 9, 16, 24, 46, 73），即： "
        "作者想表达的对比点是：压缩感知具有跨领域普适性。"
    )
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "位于 原文证据" not in out
    assert "refs" not in out
    assert "即：" not in out
    assert "参考文献 9, 16, 24, 46, 73" in out


def test_sanitize_paper_guide_answer_keeps_structured_cites_for_non_method_family():
    raw = "The paper overcomes the trade-off between sectioning and SNR [[CITE:s1234abcd:1]]."
    out = _sanitize_paper_guide_answer_for_user(
        raw,
        has_hits=True,
        prompt_family="overview",
    )
    assert "[[CITE:s1234abcd:1]]" not in out


def test_sanitize_paper_guide_answer_preserves_structured_cites_for_citation_lookup():
    raw = "建议优先点开的文内参考：ADMM [[CITE:s7f6b9404:4]]、ADMM-Net [[CITE:s7f6b9404:21]]。"
    out = _sanitize_paper_guide_answer_for_user(
        raw,
        has_hits=True,
        prompt_family="citation_lookup",
    )
    assert "[[CITE:s7f6b9404:4]]" in out
    assert "[[CITE:s7f6b9404:21]]" in out


def test_sanitize_paper_guide_answer_can_preserve_validated_ordinary_system_b_cites():
    raw = "ADMM is prior optimization machinery. To follow the citation trail, open ADMM [[CITE:s1234abcd:4]]."
    out = _sanitize_paper_guide_answer_for_user(
        raw,
        has_hits=True,
        prompt_family="overview",
        preserve_structured_cites=True,
    )
    assert "[[CITE:s1234abcd:4]]" in out


def test_sanitize_paper_guide_answer_canonicalizes_negative_shell_to_does_not_specify():
    raw = "The retrieved paper does not state the GPU model used [[CITE:s1234abcd:1]]."
    out = _sanitize_paper_guide_answer_for_user(
        raw,
        has_hits=True,
        prompt_family="reproduce",
    )
    assert "[[CITE:" not in out
    assert "does not specify the GPU model used" in out


def test_sanitize_strips_orphaned_markdown_bold_markers():
    raw = "The sampling rate is **** higher than the baseline. The method **** improves SNR."
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "****" not in out
    assert "higher than the baseline" in out


def test_sanitize_removes_empty_chinese_parenthetical_shells():
    raw = "该方法的性能（依据）优于基线，且（基于，）鲁棒性更好。"
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "（依据）" not in out
    assert "（基于，）" not in out
    assert "该方法的性能优于基线" in out


def test_sanitize_fixes_chinese_conjunction_fragments():
    raw = "该方法提高了分辨率。和讨论了不同的采样策略。"
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    # Conjunction fragment should be merged with previous sentence via a comma
    assert "。和讨论了" not in out
    assert "，和讨论了" in out


def test_sanitize_removes_orphaned_chinese_right_bracket_before_punctuation():
    raw = "该模型基于），压缩感知理论提出了一种新方法。"
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "基于），" not in out
    assert "基于压缩感知" in out or "基于，" in out


def test_sanitize_removes_orphaned_chinese_left_bracket_after_period():
    raw = "该方法提高了分辨率。（讨论了不同的采样策略。"
    out = _sanitize_paper_guide_answer_for_user(raw, has_hits=True)
    assert "。（讨论了" not in out
    assert "讨论了不同的采样策略" in out
