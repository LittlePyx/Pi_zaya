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
    assert all(slot["preferred_system"] != "system_b" for slot in plan["slots"])
    assert not citation_plan_prefers_system_b(plan, context="Fourier basis [2].", ref_num=2)


def test_scope_boundary_plan_uses_one_direct_paper_evidence_citation():
    plan = build_citation_plan(
        prompt="这篇 perovskite laser 和我的单像素成像主线关系大吗？值得一起读吗？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "We demonstrate lasing from a dual-cavity perovskite device.",
                "meta": {"source_path": "perovskite.en.md", "heading_path": "Abstract"},
            }
        ],
        reference_opportunities=[],
    )

    assert plan["intent"] == "scope_boundary"
    assert plan["budget"] == {"system_a": 1, "system_b": 0}


def test_non_origin_reading_questions_do_not_budget_system_b() -> None:
    opportunity = {
        "sid": "s1234abcd",
        "ref_num": 17,
        "label": "single-pixel imaging background",
        "source_path": "spi-review.en.md",
    }
    beginner = build_citation_plan(
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？",
        prompt_family="overview",
        answer_hits=[
            {
                "text": "Single-pixel imaging principles and prospects.",
                "meta": {"source_path": "spi-review.en.md", "heading_path": "Abstract"},
            }
        ],
        reference_opportunities=[opportunity],
    )
    method = build_citation_plan(
        prompt="这个系统具体怎么实现重聚焦？",
        prompt_family="method",
        answer_hits=[
            {
                "text": "Digital refocusing uses ray tracing and wave propagation.",
                "meta": {"source_path": "qclfm.en.md", "heading_path": "Concept"},
            }
        ],
        reference_opportunities=[opportunity],
    )

    assert beginner["budget"]["system_b"] == 0
    assert beginner["system_b_enabled"] is False
    assert method["budget"]["system_b"] == 0
    assert method["system_b_enabled"] is False


def test_comparison_with_clickable_source_markers_stays_system_a_only():
    plan = build_citation_plan(
        prompt=(
            "只依据本文比较 HSI 与 FSI 的采样基和重建原理，"
            "每个结论都标出可点击回原文的来源。"
        ),
        prompt_family="compare",
        answer_hits=[
            {
                "text": "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns.",
                "meta": {"source_path": "oe2017.en.md", "heading_path": "Comparison"},
            }
        ],
        reference_opportunities=[
            {"sid": "s9999abcd", "ref_num": 18, "label": "HSI", "source_path": "oe2017.en.md"}
        ],
    )

    assert plan["intent"] == "comparison"
    assert plan["system_b_enabled"] is False
    assert all(slot["preferred_system"] != "system_b" for slot in plan["slots"])


def test_comparison_rejecting_upstream_recommendations_stays_system_a_only():
    plan = build_citation_plan(
        prompt=(
            "只依据本文，简洁比较 HSI 与 FSI 的采样基和重建原理；"
            "每个结论给可点击原文来源，不要推荐上游文献。"
        ),
        prompt_family="compare",
        answer_hits=[
            {
                "text": "HSI uses an inverse Hadamard transform; FSI uses an inverse Fourier transform.",
                "meta": {"source_path": "oe2017.en.md", "heading_path": "Principle of HSI and FSI"},
            }
        ],
        reference_opportunities=[
            {"sid": "s9999abcd", "ref_num": 7, "label": "Computational", "source_path": "oe2017.en.md"}
        ],
    )

    assert plan["intent"] == "comparison"
    assert plan["budget"]["system_b"] == 0
    assert plan["system_b_enabled"] is False
    assert all(slot["preferred_system"] != "system_b" for slot in plan["slots"])


def test_exact_reference_lookup_keeps_system_b_while_rejecting_extra_recommendations():
    plan = build_citation_plan(
        prompt=(
            "本文参考文献中的 A single-pixel terahertz imaging system based on compressed sensing "
            "是第几条？不要推荐其他上游文献。"
        ),
        prompt_family="citation_lookup",
        answer_hits=[],
        reference_opportunities=[
            {"sid": "s9999abcd", "ref_num": 9, "label": "terahertz imaging", "source_path": "oe2017.en.md"}
        ],
    )

    assert plan["intent"] == "origin_lookup"
    assert plan["system_b_enabled"] is True
    assert plan["slots"][0]["candidate_refs"] == [9]


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
