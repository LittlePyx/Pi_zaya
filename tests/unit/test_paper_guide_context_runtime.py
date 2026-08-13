import kb.paper_guide_context_runtime as context_runtime


def test_build_paper_guide_context_records_prefers_ref_show_snippets_and_builds_card():
    hit = {
        "text": "Fallback body with citation [35].",
        "meta": {
            "source_path": r"db\demo\paper.en.md",
            "heading_path": "Materials and Methods / Adaptive pixel-reassignment (APR)",
            "page_start": 10,
            "page_end": 11,
            "ref_show_snippets": [
                "APR was performed using image registration based on phase correlation [35].",
                "APR was performed using image registration based on phase correlation [35].",
            ],
        },
    }

    out = context_runtime._build_paper_guide_context_records(
        [hit],
        paper_guide_mode=True,
    )

    assert len(out["ctx_parts"]) == 1
    assert "candidate refs: 35" in out["ctx_parts"][0]
    assert "pages: 10-11" in out["ctx_parts"][0]
    assert "phase correlation [35]" in out["ctx_parts"][0]
    assert len(out["paper_guide_evidence_cards"]) == 1
    assert out["paper_guide_evidence_cards"][0]["candidate_refs"] == [35]
    assert out["paper_guide_evidence_cards"][0]["snippet"].count("phase correlation") == 1
    primary = out["paper_guide_evidence_cards"][0]["primary_evidence"]
    assert primary["source_path"] == r"db\demo\paper.en.md"
    assert primary["source_name"] == "paper.pdf"
    assert primary["heading_path"] == "Materials and Methods / Adaptive pixel-reassignment (APR)"
    assert primary["selection_reason"] == "answer_hit_top"
    assert "phase correlation [35]" in primary["snippet"]


def test_apply_paper_guide_deepread_context_updates_card_snippet_for_abstract():
    card = {"snippet": "old", "deepread_texts": []}
    out = context_runtime._apply_paper_guide_deepread_context(
        ctx_parts=["DOC-1 [SID:s123] demo\nold"],
        doc_first_idx={r"db\demo\paper.en.md": 1},
        paper_guide_card_by_doc_idx={1: card},
        prompt="Give me the abstract",
        retrieval_prompt="abstract",
        used_query="abstract",
        prompt_family="abstract",
        deep_read=True,
        answer_hits=[{"meta": {"source_path": r"db\demo\paper.en.md"}}],
        deep_read_fn=lambda *_args, **_kwargs: [{"text": "unused"}],
        select_extras_fn=lambda extras, **_kwargs: ["# Abstract\nHere we introduce a new method."],
        merge_context_fn=lambda base, extra, **_kwargs: base + "\n\n" + extra,
        allows_citeless_answer_fn=lambda _family: True,
    )

    assert out["deep_docs"] == 1
    assert out["deep_added"] == 1
    assert "# Abstract\nHere we introduce a new method." in out["ctx_parts"][0]
    assert card["snippet"] == "# Abstract\nHere we introduce a new method."
    assert card["deepread_texts"] == ["# Abstract\nHere we introduce a new method."]


def test_apply_deepread_covers_six_named_papers_for_per_paper_evidence():
    paths = [rf"db\demo\paper-{idx}.en.md" for idx in range(1, 7)]
    cards = {
        idx: {"snippet": f"base-{idx}", "deepread_texts": []}
        for idx in range(1, 7)
    }
    deep_calls: list[str] = []
    overview_calls: list[str] = []

    def _deep_read(path, query, **_kwargs):
        deep_calls.append(str(path))
        return [
            {
                "text": f"Conclusion evidence for {path.name}",
                "meta": {"heading_path": "5 Conclusion and Future Work"},
            }
        ]

    def _overview(path, **_kwargs):
        overview_calls.append(str(path))
        return [f"Overview evidence for {path.name}"]

    answer_hits = [{"meta": {"source_path": path}} for path in paths]
    out = context_runtime._apply_paper_guide_deepread_context(
        ctx_parts=[f"DOC-{idx}\nbase-{idx}" for idx in range(1, 7)],
        doc_first_idx={path: idx for idx, path in enumerate(paths, start=1)},
        paper_guide_card_by_doc_idx=cards,
        prompt="Compare all six methods; each paper must include locatable evidence.",
        retrieval_prompt="compare six methods",
        used_query="compare six methods",
        prompt_family="strength_limits",
        deep_read=True,
        answer_hits=answer_hits,
        deep_read_fn=_deep_read,
        overview_fn=_overview,
        select_extras_fn=lambda extras, **_kwargs: [item["text"] for item in extras],
        merge_context_fn=lambda base, extra, **_kwargs: base + "\n\n" + extra,
        allows_citeless_answer_fn=lambda _family: False,
    )

    assert out["deep_docs"] == 6
    assert out["deep_added"] == 12
    assert len(deep_calls) == 6
    assert len(overview_calls) == 6
    assert all("Overview evidence" in item for item in out["ctx_parts"])
    assert all("Conclusion evidence" in item for item in out["ctx_parts"])
    assert all(len(card["deepread_texts"]) == 2 for card in cards.values())
    assert all(
        len(hit["meta"]["source_deepread_evidence_quotes"]) == 2
        for hit in answer_hits
    )


def test_build_paper_guide_context_records_builds_primary_evidence_card_even_when_not_paper_guide():
    hit = {
        "text": "Section 2.2 discusses Fourier single-pixel imaging and compares it with Hadamard sampling.",
        "meta": {
            "source_path": r"db\demo\oe2017.en.md",
            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
            "block_id": "blk_22",
            "anchor_id": "a_22",
        },
    }

    out = context_runtime._build_paper_guide_context_records(
        [hit],
        paper_guide_mode=False,
    )

    assert len(out["paper_guide_evidence_cards"]) == 1
    card = out["paper_guide_evidence_cards"][0]
    assert card["candidate_refs"] == []
    assert card["heading"] == "2. Comparison of theory"
    assert card["primary_evidence"]["heading_path"] == "2. Comparison of theory / 2.2 Basis patterns generation"
    assert card["primary_evidence"]["block_id"] == "blk_22"
    assert card["primary_evidence"]["anchor_id"] == "a_22"


def test_prepare_paper_guide_prompt_context_builds_blocks_and_candidate_refs(monkeypatch):
    captured = {}

    def _support_slots(*_args, **kwargs):
        captured["prompt"] = kwargs.get("prompt", "")
        return [{"support_example": "[[SUPPORT:DOC-1]]"}]

    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots", _support_slots)
    monkeypatch.setattr(context_runtime, "_build_paper_guide_evidence_cards_block", lambda *args, **kwargs: "EVIDENCE BLOCK")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots_block", lambda *args, **kwargs: "SUPPORT BLOCK")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_special_focus_block", lambda *args, **kwargs: "FOCUS BLOCK")
    monkeypatch.setattr(context_runtime, "_collect_paper_guide_candidate_refs_by_source", lambda *args, **kwargs: {r"db\demo\paper.en.md": [35]})
    monkeypatch.setattr(context_runtime, "_build_paper_guide_citation_grounding_block", lambda *args, **kwargs: "GROUNDING BLOCK")

    answer_hits = [{"meta": {"source_path": r"db\demo\paper.en.md"}}]
    evidence_cards = [
        {
            "doc_idx": 1,
            "sid": "s12345678",
            "source_path": r"db\demo\paper.en.md",
            "heading": "Results / Figure 1",
            "candidate_refs": [35],
            "cue": "phase correlation [35]",
            "snippet": "APR was performed using image registration [35].",
            "deepread_texts": [],
            "primary_evidence": {
                "source_path": r"db\demo\paper.en.md",
                "source_name": "paper.pdf",
                "heading_path": "Results / Figure 1",
                "snippet": "APR was performed using image registration [35].",
                "selection_reason": "answer_hit_top",
            },
        }
    ]

    out = context_runtime._prepare_paper_guide_prompt_context(
        paper_guide_mode=True,
        paper_guide_bound_source_ready=True,
        answer_hits=answer_hits,
        paper_guide_evidence_cards=evidence_cards,
        prompt="How is APR grounded?",
        retrieval_prompt="APR grounding",
        used_query="APR grounding",
        prompt_family="method",
        paper_guide_bound_source_path=r"db\demo\paper.en.md",
        db_dir="db",
        query_variants=["APR phase correlation image registration"],
    )

    assert out["paper_guide_evidence_cards_block"] == "EVIDENCE BLOCK"
    assert out["paper_guide_support_slots_block"] == "SUPPORT BLOCK"
    assert out["paper_guide_special_focus_block"] == "FOCUS BLOCK"
    assert out["paper_guide_citation_grounding_block"] == "GROUNDING BLOCK"
    assert out["paper_guide_candidate_refs_by_source"] == {r"db\demo\paper.en.md": [35]}
    assert out["paper_guide_support_slots"] == [{"support_example": "[[SUPPORT:DOC-1]]"}]
    assert out["paper_guide_target_scope"]["prompt_family"] == "method"
    assert out["paper_guide_focus_source_path"] == r"db\demo\paper.en.md"
    assert "How is APR grounded?" in captured["prompt"]
    assert "APR phase correlation image registration" in captured["prompt"]
    seed = out["paper_guide_contracts_seed"]
    assert seed["version"] == 1
    assert seed["intent"]["family"] == "method"
    assert seed["retrieval_bundle"]["prompt_family"] == "method"
    assert seed["retrieval_bundle"]["candidate_refs_by_source"] == {r"db\demo\paper.en.md": [35]}
    assert seed["retrieval_bundle"]["evidence_cards"][0]["sid"] == "s12345678"
    assert seed["retrieval_bundle"]["evidence_cards"][0]["candidate_refs"] == [35]
    assert seed["retrieval_bundle"]["evidence_cards"][0]["primary_evidence"]["heading_path"] == "Results / Figure 1"
    assert seed["support_pack"]["family"] == "method"
    assert seed["support_pack"]["support_records"][0]["support_example"] == "[[SUPPORT:DOC-1]]"
    assert seed["prompt_context"]["target_scope"]["prompt_family"] == "method"
    assert seed["prompt_context"]["focus_source_path"] == r"db\demo\paper.en.md"
    assert seed["prompt_context"]["bound_source_path"] == r"db\demo\paper.en.md"
    assert seed["primary_evidence"]["source_path"] == r"db\demo\paper.en.md"
    assert seed["primary_evidence"]["heading_path"] == "Results / Figure 1"


def test_prepare_paper_guide_prompt_context_builds_reference_opportunity_block(monkeypatch):
    source_path = r"db\demo\scinerf.en.md"
    support_slot = {
        "source_path": source_path,
        "sid": "s1234abcd",
        "heading_path": "2. Related Work",
        "snippet": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
        "candidate_refs": [4],
        "claim_type": "prior_work",
        "cite_policy": "prefer_ref",
    }

    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots", lambda *args, **kwargs: [support_slot])
    monkeypatch.setattr(context_runtime, "_build_paper_guide_evidence_cards_block", lambda *args, **kwargs: "EVIDENCE BLOCK")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots_block", lambda *args, **kwargs: "SUPPORT BLOCK")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_special_focus_block", lambda *args, **kwargs: "FOCUS BLOCK")
    monkeypatch.setattr(context_runtime, "_collect_paper_guide_candidate_refs_by_source", lambda *args, **kwargs: {})
    monkeypatch.setattr(context_runtime, "_build_paper_guide_citation_grounding_block", lambda *args, **kwargs: "")

    out = context_runtime._prepare_paper_guide_prompt_context(
        paper_guide_mode=True,
        paper_guide_bound_source_ready=True,
        answer_hits=[{"meta": {"source_path": source_path}}],
        paper_guide_evidence_cards=[],
        prompt="I am new to this. Is ADMM original to this paper?",
        retrieval_prompt="ADMM origin",
        used_query="ADMM origin",
        prompt_family="overview",
        paper_guide_bound_source_path=source_path,
        db_dir="db",
    )

    assert "Upstream reference opportunities:" in out["paper_guide_reference_opportunities_block"]
    assert "cite_example=[[CITE:s1234abcd:4]]" in out["paper_guide_reference_opportunities_block"]
    assert out["paper_guide_candidate_refs_by_source"] == {source_path: [4]}
    assert out["paper_guide_contracts_seed"]["reference_opportunities"][0]["ref_num"] == 4


def test_prepare_author_profile_context_skips_upstream_reference_scan(monkeypatch):
    source_path = r"db\demo\paper.en.md"
    biography_text = (
        "Kai Song received his B.S. degree in 2019. "
        "Yaoxing Bian received his B.S. degree in 2017. "
        "Liantuan Xiao received his B.S. degree in 1989."
    )
    support_slot = {
        "source_path": source_path,
        "heading_path": "Author Biography",
        "text": biography_text,
        "evidence_quote": biography_text,
        "snippet": biography_text,
    }
    monkeypatch.setattr(
        context_runtime,
        "_build_paper_guide_support_slots",
        lambda *_args, **_kwargs: [support_slot],
    )
    monkeypatch.setattr(
        context_runtime,
        "detect_paper_guide_reference_opportunities",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("author profiles must not scan upstream references")
        ),
    )
    monkeypatch.setattr(context_runtime, "_build_paper_guide_evidence_cards_block", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots_block", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_special_focus_block", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(context_runtime, "_collect_paper_guide_candidate_refs_by_source", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(context_runtime, "_build_paper_guide_citation_grounding_block", lambda *_args, **_kwargs: "")

    out = context_runtime._prepare_paper_guide_prompt_context(
        paper_guide_mode=True,
        paper_guide_bound_source_ready=True,
        answer_hits=[
            {
                "text": biography_text,
                "meta": {"source_path": source_path, "heading_path": "Author Biography"},
            }
        ],
        paper_guide_evidence_cards=[],
        prompt=(
            "请根据作者简介分别概括 Kai Song、Yaoxing Bian 和 Liantuan Xiao 的教育经历，"
            "并逐人给出原文证据。"
        ),
        retrieval_prompt="Author Biography Kai Song Yaoxing Bian Liantuan Xiao",
        used_query="Author Biography",
        prompt_family="overview",
        paper_guide_bound_source_path=source_path,
        db_dir="db",
    )

    assert out["paper_guide_reference_opportunities"] == []
    assert out["citation_plan"]["intent"] == "beginner_overview"
    assert out["citation_plan"]["system_b_enabled"] is False
    assert out["citation_plan"]["coverage_mode"] == "per_entity"


def test_prepare_ordinary_lineage_context_builds_grounded_reference_plan(monkeypatch):
    source_path = r"db\demo\scinerf.en.md"
    opportunity = {
        "source_path": source_path,
        "sid": "s1234abcd",
        "ref_num": 50,
        "label": "Snapshot compressive imaging",
        "heading_path": "Introduction",
        "evidence_quote": "video Snapshot Compressive Imaging (SCI) [50] system has emerged",
    }
    monkeypatch.setattr(
        context_runtime,
        "detect_text_reference_opportunities",
        lambda **_kwargs: [opportunity],
    )

    out = context_runtime._prepare_paper_guide_prompt_context(
        paper_guide_mode=False,
        paper_guide_bound_source_ready=False,
        answer_hits=[
            {
                "text": "SCINeRF connects SCI to a 3D representation.",
                "meta": {"source_path": source_path, "heading_path": "Introduction"},
            }
        ],
        paper_guide_evidence_cards=[],
        prompt="SCI 这条线是怎么从光谱成像走到 3D 场景重建的？",
        retrieval_prompt="SCI lineage",
        used_query="SCI lineage",
        prompt_family="overview",
        paper_guide_bound_source_path="",
        db_dir="db",
    )

    plan = out["citation_plan"]
    assert plan["intent"] == "origin_lookup"
    assert plan["system_b_enabled"] is True
    assert "cite_example=[[CITE:s1234abcd:50]]" in out["citation_plan_block"]
    assert "cite_example=[10001]" in out["citation_plan_block"]
    assert "Upstream reference opportunities:" in out["paper_guide_reference_opportunities_block"]


def test_prepare_paper_guide_prompt_context_keeps_bound_focus_when_first_hit_is_external(monkeypatch):
    captured: dict[str, str] = {}

    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots", lambda *args, **kwargs: [])
    monkeypatch.setattr(context_runtime, "_build_paper_guide_evidence_cards_block", lambda *args, **kwargs: "")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots_block", lambda *args, **kwargs: "")
    monkeypatch.setattr(context_runtime, "_build_paper_guide_citation_grounding_block", lambda *args, **kwargs: "")

    def _focus_block(*_args, **kwargs):
        captured["source_path"] = kwargs.get("source_path", "")
        return "FOCUS"

    monkeypatch.setattr(context_runtime, "_build_paper_guide_special_focus_block", _focus_block)
    monkeypatch.setattr(context_runtime, "_collect_paper_guide_candidate_refs_by_source", lambda *args, **kwargs: {})

    bound = r"db\bound\paper.en.md"
    out = context_runtime._prepare_paper_guide_prompt_context(
        paper_guide_mode=True,
        paper_guide_bound_source_ready=True,
        answer_hits=[
            {"meta": {"source_path": r"db\external\other.en.md"}},
            {"meta": {"source_path": bound}},
        ],
        paper_guide_evidence_cards=[],
        prompt="Where did this idea come from?",
        retrieval_prompt="source trace",
        used_query="source trace",
        prompt_family="citation_lookup",
        paper_guide_bound_source_path=bound,
        db_dir="db",
    )

    assert out["paper_guide_focus_source_path"] == bound
    assert captured["source_path"] == bound
    assert out["paper_guide_contracts_seed"]["prompt_context"]["focus_source_path"] == bound
