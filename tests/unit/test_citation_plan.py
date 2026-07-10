from __future__ import annotations

from kb.citation_plan import (
    build_citation_plan,
    build_citation_plan_prompt_block,
    citation_plan_prefers_system_b,
)


def test_origin_question_builds_system_b_first_plan():
    plan = build_citation_plan(
        prompt="ADMM 是怎么来的？作者是不是借鉴了前人的方法？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "SCINeRF uses ADMM in the optimization loop.",
                "meta": {"source_path": "scinerf.en.md", "heading_path": "Related Work"},
            }
        ],
        support_slots=[],
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 4,
                "label": "ADMM",
                "source_path": "scinerf.en.md",
                "heading_path": "2. Related Work",
                "evidence_quote": "Most existing methods employ ADMM [4].",
            }
        ],
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["budget"] == {"system_a": 1, "system_b": 1}
    assert plan["system_b_enabled"] is True
    assert plan["slots"][0]["preferred_system"] == "system_b"
    assert plan["slots"][0]["candidate_cite_examples"] == ["[[CITE:s1234abcd:4]]"]
    assert citation_plan_prefers_system_b(plan, context="ADMM comes from prior work.", ref_num=4)

    block = build_citation_plan_prompt_block(plan)
    assert "Citation plan" in block
    assert "preferred_system=system_b" in block
    assert "[[CITE:s1234abcd:4]]" in block


def test_comparison_plan_disables_system_b_budget():
    plan = build_citation_plan(
        prompt="这两篇方法有什么区别？",
        prompt_family="compare",
        answer_hits=[
            {
                "text": "Hadamard and Fourier patterns differ in sampling basis.",
                "meta": {"source_path": "oe2017.en.md", "heading_path": "Comparison"},
            }
        ],
        reference_opportunities=[
            {
                "sid": "s9999abcd",
                "ref_num": 2,
                "label": "Fourier basis",
                "source_path": "oe2017.en.md",
            }
        ],
    )

    assert plan["intent"] == "comparison"
    assert plan["budget"]["system_b"] == 0
    assert plan["system_b_enabled"] is False
    assert not citation_plan_prefers_system_b(plan, context="Fourier basis [2].", ref_num=2)


def test_multi_paper_source_marker_request_keeps_system_a_slots_for_every_requested_paper():
    hits = [
        {
            "text": f"Evidence for paper {idx}",
            "meta": {
                "source_path": f"paper-{idx}.en.md",
                "heading_path": f"Paper {idx} / Results",
            },
        }
        for idx in range(1, 7)
    ]

    plan = build_citation_plan(
        prompt="请只用最相关的 4 篇论文做阅读路线，并用来源编号标出可点回原文的依据。",
        prompt_family="method",
        answer_hits=hits,
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 11,
                "label": "upstream work",
                "source_path": "paper-1.en.md",
            }
        ],
    )

    assert plan["intent"] == "method_explain"
    assert plan["budget"]["system_a"] == 4
    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert [slot["candidate_hits"] for slot in system_a_slots] == [[1], [2], [3], [4]]


def test_previous_answer_audit_uses_every_authoritative_source_without_system_b():
    hits = [
        {
            "text": f"Evidence for paper {idx}",
            "meta": {
                "source_path": f"paper-{idx}.en.md",
                "heading_path": f"Paper {idx} / Results",
            },
        }
        for idx in range(1, 5)
    ]

    plan = build_citation_plan(
        prompt="Audit the previous answer and verify that its four titles match their evidence.",
        prompt_family="overview",
        answer_hits=hits,
        reference_opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 11,
                "label": "upstream work",
                "source_path": "paper-1.en.md",
            }
        ],
    )

    assert plan["intent"] == "answer_audit"
    assert plan["budget"] == {"system_a": 4, "system_b": 0}
    assert plan["system_b_enabled"] is False
    system_a_slots = [slot for slot in plan["slots"] if slot["preferred_system"] == "system_a"]
    assert [slot["candidate_hits"] for slot in system_a_slots] == [[1], [2], [3], [4]]
