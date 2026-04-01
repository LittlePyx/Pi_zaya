from __future__ import annotations


def test_detect_answer_intent_prefers_explicit_hint():
    from kb import task_runtime

    intent = task_runtime._detect_answer_intent(
        "Can you compare method A and method B?",
        answer_mode_hint="writing",
    )
    assert intent == "writing"


def test_detect_answer_intent_by_prompt():
    from kb import task_runtime

    assert task_runtime._detect_answer_intent("compare transformer and cnn for this task") == "compare"
    assert task_runtime._detect_answer_intent("这个 idea 可行吗，风险是什么") == "idea"
    assert task_runtime._detect_answer_intent("如何设计实验和对照组") == "experiment"
    assert task_runtime._detect_answer_intent("训练报错了，帮我排查") == "troubleshoot"
    assert task_runtime._detect_answer_intent("帮我润色 related work 这一段") == "writing"


def test_detect_answer_depth_auto_and_fixed():
    from kb import task_runtime

    assert task_runtime._detect_answer_depth("ok?", intent="reading", auto_depth=True) == "L1"
    assert task_runtime._detect_answer_depth(
        "idea feasibility with experiment design and evaluation metrics for a new algorithm",
        intent="idea",
        auto_depth=True,
    ) == "L3"
    assert task_runtime._detect_answer_depth("any prompt", intent="reading", auto_depth=False) == "L2"


def test_detect_answer_output_mode_marks_anchor_grounded_fact_answer():
    from kb import task_runtime

    mode = task_runtime._detect_answer_output_mode(
        "Does the paper explicitly state the formula in Eq. 3?",
        paper_guide_mode=True,
        intent="reading",
        anchor_grounded=True,
    )
    assert mode == "fact_answer"


def test_detect_answer_output_mode_keeps_generic_problem_question_out_of_critical_review():
    from kb import task_runtime

    mode = task_runtime._detect_answer_output_mode(
        "What problem does this paper solve, and what are its core contributions?",
        paper_guide_mode=True,
        intent="reading",
        anchor_grounded=False,
    )
    assert mode == "reading_guide"


def test_apply_answer_contract_with_hits():
    from kb import task_runtime

    raw = (
        "This method can reduce reconstruction noise in low-light capture.\n\n"
        "The retrieved snippet reports lower MAE and higher PSNR than baseline [1].\n\n"
        "Another paragraph with details [2]."
    )
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="what is the key contribution?",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert "Conclusion:" in out
    assert "Evidence:" in out
    assert "Next Steps:" in out
    assert "[1]" in out or "[2]" in out


def test_answer_contract_enabled_respects_paper_guide_toggle():
    from kb import task_runtime

    assert task_runtime._answer_contract_enabled({"paper_guide_mode": True, "answer_contract_v1": False}) is False
    assert task_runtime._answer_contract_enabled({"paper_guide_mode": True, "answer_contract_v1": True}) is True


def test_build_paper_guide_grounding_rules_respects_contract_toggle():
    from kb import task_runtime

    structured = task_runtime._build_paper_guide_grounding_rules(answer_contract_v1=True)
    plain = task_runtime._build_paper_guide_grounding_rules(answer_contract_v1=False)

    assert "3-4 sections" in structured
    assert "under Evidence" in structured
    assert "3-4 sections" not in plain
    assert "under Evidence" not in plain
    assert "do not force a fixed section template" in plain


def test_build_paper_guide_grounding_rules_fact_answer_discourages_generic_advice():
    from kb import task_runtime

    rules = task_runtime._build_paper_guide_grounding_rules(
        answer_contract_v1=True,
        output_mode="fact_answer",
    )

    assert "answer the exact paper-grounded question first" in rules
    assert "avoid generic reading advice" in rules


def test_build_paper_guide_grounding_rules_adds_abstract_and_figure_family_guards():
    from kb import task_runtime

    abstract_rules = task_runtime._build_paper_guide_grounding_rules(
        answer_contract_v1=False,
        prompt_family="abstract",
    )
    figure_rules = task_runtime._build_paper_guide_grounding_rules(
        answer_contract_v1=False,
        prompt_family="figure_walkthrough",
    )

    assert "abstract span itself" in abstract_rules
    assert "Conclusion/Evidence/Next Steps" in abstract_rules
    assert "panel letters" in figure_rules


def test_build_paper_guide_grounding_rules_adds_method_family_guard():
    from kb import task_runtime

    rules = task_runtime._build_paper_guide_grounding_rules(
        answer_contract_v1=False,
        prompt_family="method",
    )

    assert "phase correlation" in rules
    assert "generic mechanism paraphrase" in rules


def test_build_paper_guide_grounding_rules_adds_citation_lookup_guard():
    from kb import task_runtime

    rules = task_runtime._build_paper_guide_grounding_rules(
        answer_contract_v1=False,
        prompt_family="citation_lookup",
    )

    assert "exact in-paper reference numbers" in rules
    assert "do not answer 'not stated'" in rules


def test_build_paper_guide_grounding_rules_mentions_doc_scoped_candidate_refs():
    from kb import task_runtime

    rules = task_runtime._build_paper_guide_grounding_rules(answer_contract_v1=True)

    assert "Paper-guide support slots" in rules
    assert "support_example marker" in rules
    assert "[[SUPPORT:" in rules
    assert "Paper-guide citation grounding hints" in rules
    assert "same DOC-k line" in rules
    assert "cite_example" in rules


def test_apply_answer_contract_without_hits_adds_limits():
    from kb import task_runtime

    raw = "No direct paper snippet is available for this query. Here is a general answer."
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="how to start",
        has_hits=False,
        intent="reading",
        depth="L2",
    )
    assert "Conclusion:" in out
    assert "Limits:" in out
    assert "general guidance" in out
    assert "Next Steps:" in out


def test_apply_answer_contract_fact_answer_skips_default_next_steps():
    from kb import task_runtime

    raw = (
        "The paper defines the loss exactly in Eq. 3.\n\n"
        "The retrieved snippet quotes the equation and derivation context [1]."
    )
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="What is the exact loss in Eq. 3?",
        has_hits=True,
        intent="reading",
        depth="L2",
        output_mode="fact_answer",
    )
    assert "Conclusion:" in out
    assert "Evidence:" in out
    assert "Next Steps:" not in out


def test_apply_answer_contract_is_idempotent_on_structured_text():
    from kb import task_runtime

    raw = "Conclusion: done.\n\nEvidence:\n1. snippet [1]\n\nNext Steps:\n1. verify"
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="q",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert out == raw


def test_apply_answer_contract_repairs_partial_structured_text():
    from kb import task_runtime

    raw = "Next Steps:\n1. Check the cited section."
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="what should I do next?",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert "Conclusion:" in out
    assert out.count("Next Steps:") == 1


def test_apply_answer_contract_repairs_structured_text_missing_evidence_when_hits():
    from kb import task_runtime

    raw = "Conclusion: done.\n\nNext Steps:\n1. verify"
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="explain",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert "Evidence:" in out
    assert "Next Steps:" in out


def test_apply_answer_contract_fallback_evidence_has_no_hardcoded_numeric_citation():
    from kb import task_runtime

    raw = "Conclusion: this is supported by retrieved snippets.\n\nNext Steps:\n1. verify"
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="explain the evidence",
        has_hits=True,
        intent="reading",
        depth="L2",
    )
    assert "Evidence:" in out
    assert "citation [1]" not in out
    assert "引用 [1]" not in out


def test_apply_answer_contract_uses_chinese_locale_for_chinese_prompt():
    from kb import task_runtime

    raw = "这项方法在低照度下重建质量更高。"
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="这篇论文核心贡献是什么？",
        has_hits=False,
        intent="reading",
        depth="L2",
    )
    assert "结论:" in out
    assert "下一步:" in out
    assert "general guidance" not in out


def test_apply_answer_contract_avoids_over_shrinking_long_answer():
    from kb import task_runtime

    raw = (
        "This work proposes a robust reconstruction strategy for low-light imaging.\n\n"
        + "\n\n".join(
            [
                "It details assumptions, optimization steps, and implementation notes with practical caveats."
                " The paragraph includes enough context to represent real long-form output."
                for _ in range(10)
            ]
        )
    )
    out = task_runtime._apply_answer_contract_v1(
        raw,
        prompt="please summarize with practical recommendations",
        has_hits=False,
        intent="reading",
        depth="L2",
    )
    assert "Conclusion:" in out
    assert len(out) >= int(len(raw) * 0.65)


def test_enhance_kb_miss_fallback_appends_next_steps():
    from kb import task_runtime

    raw = "未命中知识库片段。\n\n这是一个通用说明。"
    out = task_runtime._enhance_kb_miss_fallback(
        raw,
        has_hits=False,
        intent="reading",
        depth="L2",
        contract_enabled=True,
    )
    assert "未命中知识库片段" in out
    assert "下一步建议" in out
    assert "1." in out


def test_enhance_kb_miss_fallback_does_not_duplicate_next_steps():
    from kb import task_runtime

    raw = "未命中知识库片段。\n\n说明。\n\nNext Steps:\n1. keep"
    out = task_runtime._enhance_kb_miss_fallback(
        raw,
        has_hits=False,
        intent="reading",
        depth="L2",
        contract_enabled=True,
    )
    assert out.count("Next Steps:") == 1


def test_enhance_kb_miss_fallback_respects_contract_disabled():
    from kb import task_runtime

    raw = "Missed library snippets.\n\nHere is a general answer."
    out = task_runtime._enhance_kb_miss_fallback(
        raw,
        has_hits=False,
        intent="reading",
        depth="L2",
        contract_enabled=False,
    )
    assert out == raw


def test_enhance_kb_miss_fallback_fact_answer_skips_default_next_steps():
    from kb import task_runtime

    raw = "Missed library snippets.\n\nHere is a general answer."
    out = task_runtime._enhance_kb_miss_fallback(
        raw,
        has_hits=False,
        intent="reading",
        depth="L2",
        contract_enabled=True,
        output_mode="fact_answer",
    )
    assert out == raw


def test_sanitize_structured_tokens_removes_sid_markers():
    from kb import task_runtime

    raw = "[SID:s50f9c165] text\nDOC-1 [SID:s50f9c165] source | section\nanswer body"
    out = task_runtime._sanitize_structured_cite_tokens(raw)
    assert "[SID:" not in out
    assert "answer body" in out


def test_sanitize_structured_tokens_drops_non_numeric_cite_markers():
    from kb import task_runtime

    raw = "See [[CITE:s50f9c165:DOC-1]] and [[CITE:s50f9c165:24]]."
    out = task_runtime._sanitize_structured_cite_tokens(raw)
    assert "[[CITE:s50f9c165:DOC-1]]" not in out
    assert "[[CITE:s50f9c165:24]]" in out


def test_promote_paper_guide_numeric_reference_citations_rewrites_reference_label():
    from kb import task_runtime

    out = task_runtime._promote_paper_guide_numeric_reference_citations(
        "Consult reference [35] for the implementation details.",
        locked_source={"sid": "s50f9c165"},
    )
    assert out == "Consult reference [[CITE:s50f9c165:35]] for the implementation details."


def test_promote_paper_guide_numeric_reference_citations_leaves_plain_numeric_brackets():
    from kb import task_runtime

    out = task_runtime._promote_paper_guide_numeric_reference_citations(
        "The measured width is [35] nm in this notation.",
        locked_source={"sid": "s50f9c165"},
    )
    assert out == "The measured width is [35] nm in this notation."


def test_build_answer_quality_probe_with_hits():
    from kb import task_runtime

    answer = "Conclusion: ok\n\nEvidence:\n1. from snippet [1]\n\nNext Steps:\n1. verify"
    probe = task_runtime._build_answer_quality_probe(
        answer,
        has_hits=True,
        contract_enabled=True,
        intent="reading",
        depth="L2",
    )
    assert probe["has_conclusion"] is True
    assert probe["has_evidence"] is True
    assert probe["has_next_steps"] is True
    assert probe["has_citations"] is True
    assert probe["evidence_ok"] is True
    assert probe["minimum_ok"] is True


def test_build_answer_quality_probe_fact_answer_allows_no_next_steps():
    from kb import task_runtime

    answer = "Conclusion: yes\n\nEvidence:\n1. exact snippet [1]"
    probe = task_runtime._build_answer_quality_probe(
        answer,
        has_hits=True,
        contract_enabled=True,
        intent="reading",
        depth="L2",
        output_mode="fact_answer",
    )
    assert probe["output_mode"] == "fact_answer"
    assert probe["next_steps_required"] is False
    assert probe["minimum_ok"] is True


def test_build_answer_quality_probe_paper_guide_abstract_rejects_section_shells():
    from kb import task_runtime

    answer = "Conclusion: Here is the abstract.\n\nEvidence:\n1. quoted text"
    probe = task_runtime._build_answer_quality_probe(
        answer,
        has_hits=True,
        contract_enabled=False,
        intent="reading",
        depth="L2",
        paper_guide_mode=True,
        prompt_family="abstract",
    )
    assert probe["evidence_required"] is False
    assert probe["abstract_style_ok"] is False
    assert probe["minimum_ok"] is False


def test_build_answer_quality_probe_paper_guide_plain_answer_without_contract_is_ok():
    from kb import task_runtime

    answer = (
        "The paper introduces interferometric image scanning microscopy for label-free live-cell imaging.\n\n"
        "It combines ISM-style scanning with interferometric detection to improve resolution and contrast."
    )
    probe = task_runtime._build_answer_quality_probe(
        answer,
        has_hits=True,
        contract_enabled=False,
        intent="reading",
        depth="L2",
        paper_guide_mode=True,
        prompt_family="overview",
    )
    assert probe["evidence_required"] is False
    assert probe["next_steps_required"] is False
    assert probe["minimum_ok"] is True


def test_build_answer_quality_probe_paper_guide_figure_tracks_locate_hint():
    from kb import task_runtime

    answer = "Conclusion: Figure 1 compares the pinhole conditions.\n\nEvidence:\n1. Figure 1d/e reports the resolution and CNR.\n\nNext Steps:\n1. Re-open Figure 1."
    probe = task_runtime._build_answer_quality_probe(
        answer,
        has_hits=True,
        contract_enabled=True,
        intent="reading",
        depth="L2",
        paper_guide_mode=True,
        prompt_family="figure_walkthrough",
    )
    assert probe["locate_required"] is True
    assert probe["has_locate_hint"] is True
    assert probe["locate_ok"] is True
    assert probe["minimum_ok"] is True


def test_build_answer_quality_probe_without_hits_allows_no_evidence():
    from kb import task_runtime

    answer = "结论: 可以先按通用路径尝试。\n\n下一步:\n1. 补充目标论文。"
    probe = task_runtime._build_answer_quality_probe(
        answer,
        has_hits=False,
        contract_enabled=True,
        intent="reading",
        depth="L2",
    )
    assert probe["has_conclusion"] is True
    assert probe["has_next_steps"] is True
    assert probe["evidence_required"] is False
    assert probe["evidence_ok"] is True
    assert probe["minimum_ok"] is True


def test_build_answer_quality_probe_accepts_evidence_alias_header():
    from kb import task_runtime

    answer = "结论：可行。\n\n证据：\n1. 命中文段 [1]\n\n下一步：\n1. 复核原文"
    probe = task_runtime._build_answer_quality_probe(
        answer,
        has_hits=True,
        contract_enabled=True,
        intent="reading",
        depth="L2",
    )
    assert probe["has_evidence"] is True
    assert probe["minimum_ok"] is True


def test_pick_locked_citation_source_prefers_single_source():
    from kb import task_runtime

    hits = [
        {"meta": {"source_path": r"db\doc\paper.en.md", "source_sha1": "abc"}},
        {"meta": {"source_path": r"db\doc\paper.en.md", "source_sha1": "abc"}},
    ]
    locked = task_runtime._pick_locked_citation_source(hits)
    assert isinstance(locked, dict)
    assert locked["sid"] == task_runtime._cite_source_id(r"db\doc\paper.en.md")
    assert locked["lock_reason"] == "single_source"


def test_validate_structured_citations_rewrites_to_locked_source(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) == source_path and int(ref_num) == 24:
            return {"ref": {"raw": "[24] Demo ref"}}
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        "This follows prior work [[CITE:sdeadbeef:24]].",
        answer_hits=[{"meta": {"source_path": source_path, "source_sha1": "abc"}}],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
    )
    assert f"[[CITE:{locked_sid}:24]]" in answer
    assert "[[CITE:sdeadbeef:24]]" not in answer
    assert stats["rewritten"] == 1


def test_validate_structured_citations_in_paper_guide_rewrites_to_evidence_candidate(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) == 1:
            return {"ref": {"raw": "[1] Wrong ref", "authors": "Smith et al.", "year": "2020", "title": "Wrong Ref"}}
        if int(ref_num) == 24:
            return {"ref": {"raw": "[24] Gehm et al. Demo. 2007.", "authors": "Gehm et al.", "year": "2007", "title": "Correct Ref"}}
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        "Gehm et al. (2007) support this claim [[CITE:sdeadbeef:1]].",
        answer_hits=[{"text": "This follows prior work [24].", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
    )
    assert f"[[CITE:{locked_sid}:24]]" in answer
    assert f"[[CITE:{locked_sid}:1]]" not in answer
    assert stats["rewritten"] == 1


def test_validate_structured_citations_in_paper_guide_drops_suspicious_ref_num(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) in {1, 24, 25}:
            return {"ref": {"raw": f"[{int(ref_num)}] Demo ref {int(ref_num)}", "authors": "Demo et al.", "year": "2024", "title": f"Ref {int(ref_num)}"}}
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        "This sentence has no grounded ref number [[CITE:sdeadbeef:1]].",
        answer_hits=[{"text": "Evidence mentions [24] and [25].", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
    )
    assert "[[CITE:" not in answer
    assert stats["dropped"] == 1


def test_validate_structured_citations_in_paper_guide_accepts_focus_candidate_refs(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) == 35:
            return {"ref": {"raw": "[35] Demo APR ref", "authors": "Demo et al.", "year": "2023", "title": "APR Tool"}}
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        f"Implementation detail: APR uses phase correlation [[CITE:{locked_sid}:35]].",
        answer_hits=[{"text": "Broad summary with no in-paper citations.", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
        paper_guide_candidate_refs_by_source={source_path: [35]},
    )
    assert f"[[CITE:{locked_sid}:35]]" in answer
    assert stats["kept"] == 1


def test_validate_structured_citations_in_paper_guide_drops_visible_cite_for_locate_only_slot(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) == 25:
            return {"ref": {"raw": "[25] Demo figure ref", "authors": "Demo et al.", "year": "2021", "title": "Figure Ref"}}
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        f"Figure 2 overview [[CITE:{locked_sid}:25]].",
        answer_hits=[{"text": "Broad Figure 2 evidence mentions [25].", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
        paper_guide_candidate_refs_by_source={source_path: [25]},
        paper_guide_support_slots=[
            {
                "doc_idx": 1,
                "support_id": "DOC-1-S1",
                "sid": locked_sid,
                "source_path": source_path,
                "heading_path": "Results / Figure 2",
                "snippet": "Figure 2. APR workflow.",
                "locate_anchor": "Figure 2. APR workflow.",
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "candidate_refs": [],
                "ref_spans": [],
            }
        ],
    )
    assert "[[CITE:" not in answer
    assert stats["dropped"] == 1


def test_validate_structured_citations_in_paper_guide_prefers_support_resolution_for_locate_only(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) == 25:
            return {"ref": {"raw": "[25] Demo figure ref", "authors": "Demo et al.", "year": "2021", "title": "Figure Ref"}}
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    raw = task_runtime._drop_paper_guide_locate_only_line_citations(
        f"Figure 2 overview [[CITE:{locked_sid}:25]].",
        support_resolution=[
            {
                "line_index": 0,
                "doc_idx": 1,
                "support_id": "DOC-1-S1",
                "sid": locked_sid,
                "source_path": source_path,
                "block_id": "blk_fig2",
                "heading_path": "Results / Figure 2",
                "locate_anchor": "Figure 2. APR workflow.",
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "candidate_refs": [],
                "resolved_ref_num": 0,
                "citation_resolution_mode": "locate_only",
            }
        ],
    )

    answer, stats = task_runtime._validate_structured_citations(
        raw,
        answer_hits=[{"text": "Broad Figure 2 evidence mentions [25].", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
        paper_guide_candidate_refs_by_source={source_path: [25]},
        paper_guide_support_slots=[
            {
                "doc_idx": 1,
                "support_id": "DOC-1-S1",
                "sid": locked_sid,
                "source_path": source_path,
                "heading_path": "Methods / APR",
                "snippet": "APR uses phase correlation [25].",
                "locate_anchor": "APR uses phase correlation.",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "candidate_refs": [25],
                "ref_spans": [{"text": "phase correlation [25]", "nums": [25], "scope": "same_sentence"}],
            }
        ],
    )
    assert "[[CITE:" not in answer
    assert stats["raw_count"] == 0


def test_validate_structured_citations_in_paper_guide_prefers_local_support_slot_ref(monkeypatch, tmp_path):
    from kb import task_runtime

    source_path = r"db\doc\paper.en.md"
    locked_sid = task_runtime._cite_source_id(source_path)

    monkeypatch.setattr(task_runtime, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path:
            return None
        if int(ref_num) == 25:
            return {"ref": {"raw": "[25] Wrong ref", "authors": "Wrong et al.", "year": "2021", "title": "Wrong Ref"}}
        if int(ref_num) == 35:
            return {"ref": {"raw": "[35] APR library ref", "authors": "Demo et al.", "year": "2023", "title": "APR Tool"}}
        return None

    monkeypatch.setattr(task_runtime, "resolve_reference_entry", fake_resolve)

    answer, stats = task_runtime._validate_structured_citations(
        f"Implementation detail: APR uses phase correlation [[CITE:{locked_sid}:25]].",
        answer_hits=[{"text": "Broad evidence mentions [25] and [35].", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        db_dir=tmp_path,
        locked_source={
            "sid": locked_sid,
            "source_path": source_path,
            "source_sha1": "abc",
        },
        paper_guide_mode=True,
        paper_guide_candidate_refs_by_source={source_path: [25, 35]},
        paper_guide_support_slots=[
            {
                "doc_idx": 1,
                "support_id": "DOC-1-S1",
                "sid": locked_sid,
                "source_path": source_path,
                "heading_path": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "locate_anchor": "APR was performed using image registration based on phase correlation.",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "candidate_refs": [35],
                "ref_spans": [{"text": "phase correlation [35]", "nums": [35], "scope": "same_sentence"}],
            }
        ],
    )
    assert f"[[CITE:{locked_sid}:35]]" in answer
    assert f"[[CITE:{locked_sid}:25]]" not in answer
    assert stats["rewritten"] == 1


def test_gen_answer_quality_summary_aggregates_rates(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime.RUNTIME,
        "GEN_QUALITY_EVENTS",
        [
            {
                "intent": "reading",
                "depth": "L2",
                "has_conclusion": True,
                "has_evidence": True,
                "has_next_steps": True,
                "evidence_required": True,
                "minimum_ok": True,
                "core_section_coverage": 1.0,
                "char_count": 120,
            },
            {
                "intent": "reading",
                "depth": "L2",
                "has_conclusion": True,
                "has_evidence": False,
                "has_next_steps": True,
                "evidence_required": True,
                "minimum_ok": False,
                "core_section_coverage": 0.667,
                "char_count": 90,
            },
            {
                "intent": "idea",
                "depth": "L3",
                "has_conclusion": True,
                "has_evidence": False,
                "has_next_steps": True,
                "evidence_required": False,
                "minimum_ok": True,
                "core_section_coverage": 0.667,
                "char_count": 210,
            },
        ],
        raising=False,
    )
    out = task_runtime._gen_answer_quality_summary(limit=200)
    assert out["total"] == 3
    assert out["filters"]["intent"] == ""
    assert out["filters"]["depth"] == ""
    assert out["filters"]["only_failed"] is False
    assert out["structure_complete_rate"] == 1.0
    assert out["evidence_coverage_rate"] == 0.667
    assert out["next_steps_coverage_rate"] == 1.0
    assert out["minimum_ok_rate"] == 0.667
    assert out["failed_count"] == 1
    assert out["failed_rate"] == 0.333
    assert out["avg_core_section_coverage"] == 0.778
    assert out["by_intent"]["reading"]["count"] == 2
    assert out["by_intent"]["reading"]["minimum_ok_rate"] == 0.5
    assert out["by_intent"]["idea"]["minimum_ok_rate"] == 1.0
    assert out["by_depth"]["L2"]["count"] == 2
    assert out["by_depth"]["L2"]["minimum_ok_rate"] == 0.5
    assert out["by_depth"]["L2"]["avg_char_count"] == 105.0
    assert out["by_depth"]["L3"]["avg_char_count"] == 210.0
    assert out["fail_reasons"]["missing_evidence"] == 1


def test_gen_record_answer_quality_respects_ring_limit(monkeypatch):
    from kb import task_runtime

    monkeypatch.setenv("KB_ANSWER_QUALITY_KEEP", "100")
    monkeypatch.setattr(task_runtime.RUNTIME, "GEN_QUALITY_EVENTS", [], raising=False)
    for i in range(130):
        task_runtime._gen_record_answer_quality(
            session_id=f"s{i}",
            task_id=f"t{i}",
            conv_id=f"c{i}",
            answer_quality={
                "intent": "reading",
                "depth": "L2",
                "contract_enabled": True,
                "has_hits": True,
                "has_conclusion": True,
                "has_evidence": True,
                "has_next_steps": True,
                "evidence_required": True,
                "evidence_ok": True,
                "minimum_ok": True,
                "core_section_coverage": 1.0,
                "char_count": 120,
            },
        )
    events = list(getattr(task_runtime.RUNTIME, "GEN_QUALITY_EVENTS", []))
    assert len(events) == 100
    assert events[0]["task_id"] == "t30"
    assert events[-1]["task_id"] == "t129"


def test_gen_answer_quality_summary_supports_filters(monkeypatch):
    from kb import task_runtime

    monkeypatch.setattr(
        task_runtime.RUNTIME,
        "GEN_QUALITY_EVENTS",
        [
            {"intent": "reading", "depth": "L2", "minimum_ok": True, "has_conclusion": True, "has_evidence": True, "has_next_steps": True, "evidence_required": True, "core_section_coverage": 1.0, "char_count": 100},
            {"intent": "reading", "depth": "L1", "minimum_ok": False, "has_conclusion": True, "has_evidence": False, "has_next_steps": True, "evidence_required": True, "core_section_coverage": 0.667, "char_count": 60},
            {"intent": "idea", "depth": "L3", "minimum_ok": False, "has_conclusion": False, "has_evidence": False, "has_next_steps": True, "evidence_required": False, "core_section_coverage": 0.333, "char_count": 180},
        ],
        raising=False,
    )
    out = task_runtime._gen_answer_quality_summary(limit=50, intent="reading", depth="l1", only_failed=True)
    assert out["filters"]["intent"] == "reading"
    assert out["filters"]["depth"] == "L1"
    assert out["filters"]["only_failed"] is True
    assert out["total"] == 1
    assert out["failed_count"] == 1
    assert out["failed_rate"] == 1.0
    assert out["by_intent"]["reading"]["count"] == 1
    assert out["by_depth"]["L1"]["count"] == 1
