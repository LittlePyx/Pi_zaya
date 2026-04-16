import kb.paper_guide_context_runtime as context_runtime


def test_build_paper_guide_context_records_prefers_ref_show_snippets_and_builds_card():
    hit = {
        "text": "Fallback body with citation [35].",
        "meta": {
            "source_path": r"db\demo\paper.en.md",
            "heading_path": "Materials and Methods / Adaptive pixel-reassignment (APR)",
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
    monkeypatch.setattr(context_runtime, "_build_paper_guide_support_slots", lambda *args, **kwargs: [{"support_example": "[[SUPPORT:DOC-1]]"}])
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
    )

    assert out["paper_guide_evidence_cards_block"] == "EVIDENCE BLOCK"
    assert out["paper_guide_support_slots_block"] == "SUPPORT BLOCK"
    assert out["paper_guide_special_focus_block"] == "FOCUS BLOCK"
    assert out["paper_guide_citation_grounding_block"] == "GROUNDING BLOCK"
    assert out["paper_guide_candidate_refs_by_source"] == {r"db\demo\paper.en.md": [35]}
    assert out["paper_guide_support_slots"] == [{"support_example": "[[SUPPORT:DOC-1]]"}]
    assert out["paper_guide_target_scope"]["prompt_family"] == "method"
    assert out["paper_guide_focus_source_path"] == r"db\demo\paper.en.md"
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
