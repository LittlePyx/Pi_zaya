from __future__ import annotations

from pathlib import Path

import kb.paper_guide_reference_opportunities as reference_opportunities
from kb.paper_guide_reference_opportunities import (
    apply_reference_opportunities_to_answer,
    append_reference_opportunity_note,
    build_reference_opportunities_prompt_block,
    detect_paper_guide_reference_opportunities,
    detect_text_reference_opportunities,
    inject_reference_opportunity_citations_inline,
    merge_reference_opportunities,
    merge_reference_opportunity_candidate_refs,
    strip_reference_opportunity_note,
)


def _admm_support_slots() -> list[dict]:
    return [
        {
            "source_path": "db/demo/scinerf.en.md",
            "sid": "s1234abcd",
            "heading_path": "SCINeRF / 2. Related Work",
            "snippet": (
                "Most existing methods employ alternating direction method of multipliers "
                "(ADMM) [4]. ADMM-Net [21] unfolds this optimization idea into a network."
            ),
            "candidate_refs": [4, 21],
            "claim_type": "prior_work",
            "cite_policy": "prefer_ref",
        }
    ]


def test_detects_upstream_refs_for_ordinary_beginner_question_before_generation() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt="I am new to this. Is ADMM original to this paper, or does it come from earlier work?",
        answer="",
        prompt_family="overview",
        source_path="db/demo/scinerf.en.md",
        support_slots=_admm_support_slots(),
        max_items=3,
    )

    assert [item["ref_num"] for item in opportunities[:2]] == [4, 21]
    assert opportunities[0]["sid"] == "s1234abcd"
    assert "Related Work" in opportunities[0]["heading_path"]


def test_detect_prioritizes_named_method_ref_over_earlier_refs_in_same_related_work_paragraph() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt="ADMM 是作者自己发明的吗？我应该把它当成这篇论文的新东西吗？",
        answer="",
        prompt_family="overview",
        source_path="db/demo/scinerf.en.md",
        support_slots=[
            {
                "source_path": "db/demo/scinerf.en.md",
                "sid": "s1234abcd",
                "heading_path": "SCINeRF / 2. Related Work / Snapshot Compressive Imaging",
                "snippet": (
                    "Early reconstruction methods use regularized optimization [18,20,47,49]. "
                    "Most of the existing methods employ alternating direction method of multipliers "
                    "(ADMM) [4], which leads to good results."
                ),
                "claim_type": "prior_work",
                "cite_policy": "prefer_ref",
            }
        ],
        max_items=3,
    )

    assert opportunities
    assert opportunities[0]["ref_num"] == 4


def test_merge_reference_opportunities_keeps_explicit_text_match_first_and_dedupes() -> None:
    merged = merge_reference_opportunities(
        [{"sid": "s1234abcd", "ref_num": 4, "label": "ADMM"}],
        [
            {"sid": "s1234abcd", "ref_num": 18, "label": "SCI"},
            {"sid": "s1234abcd", "ref_num": 4, "label": "duplicate"},
        ],
        max_items=3,
    )

    assert [(row["sid"], row["ref_num"]) for row in merged] == [
        ("s1234abcd", 4),
        ("s1234abcd", 18),
    ]


def test_detect_paper_guide_reference_opportunities_requires_explicit_evidence_ref() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt="Deep learning gives SPI what benefits and pitfalls?",
        answer="Deep learning can improve reconstruction, but needs training data.",
        prompt_family="overview",
        source_path="db/demo/spi.en.md",
        support_slots=[
            {
                "source_path": "db/demo/spi.en.md",
                "sid": "s1234abcd",
                "heading_path": "Deep learning",
                "snippet": "Deep learning improves single-pixel reconstruction speed and image quality.",
                "candidate_refs": [22],
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
            }
        ],
        max_items=3,
    )

    assert opportunities == []


def test_detect_paper_guide_reference_opportunities_accepts_ref_span_from_same_sentence() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt="Did ADMM come from earlier work?",
        answer="ADMM is earlier optimization machinery.",
        prompt_family="overview",
        source_path="db/demo/scinerf.en.md",
        support_slots=[
            {
                "source_path": "db/demo/scinerf.en.md",
                "sid": "s1234abcd",
                "heading_path": "SCINeRF / 2. Related Work",
                "snippet": "Most existing methods employ alternating direction method of multipliers.",
                "candidate_refs": [4],
                "ref_spans": [
                    {
                        "text": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                        "nums": [4],
                        "scope": "same_sentence",
                    }
                ],
                "claim_type": "prior_work",
                "cite_policy": "prefer_ref",
            }
        ],
        max_items=3,
    )

    assert [item["ref_num"] for item in opportunities] == [4]


def test_detect_text_reference_opportunities_for_normal_question(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})
    monkeypatch.setattr(
        mod,
        "load_paper_guide_reference_index",
        lambda _source_path: [
            {
                "ref_num": 4,
                "title": "Distributed optimization and statistical learning via the alternating direction method of multipliers",
                "text": "Distributed optimization and statistical learning via the alternating direction method of multipliers.",
                "first_citation_context": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                "first_citation_location": "SCINeRF / 2. Related Work",
            }
        ],
    )

    def fake_resolve(_index, source_path, ref_num, *, source_sha1=""):
        del _index, source_path, source_sha1
        if int(ref_num) == 4:
            return {
                "ref": {
                    "title": "Distributed optimization and statistical learning via the alternating direction method of multipliers",
                    "raw": "Distributed optimization and statistical learning via the alternating direction method of multipliers.",
                }
            }
        return None

    monkeypatch.setattr(mod, "resolve_reference_entry", fake_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="ADMM 是作者自己发明的吗？我应该把它当成这篇论文的新东西吗？",
        answer="ADMM 不是这篇论文的新东西，而是已有优化工具。",
        answer_hits=[
            {
                "text": "Most existing methods employ ADMM.",
                "meta": {
                    "source_path": "db/demo/scinerf.en.md",
                    "source_sha1": "abc",
                    "heading_path": "SCINeRF / 2. Related Work",
                },
            }
        ],
        db_dir="db",
    )

    assert opportunities
    assert opportunities[0]["label"] == "ADMM"
    assert opportunities[0]["ref_num"] == 4


def test_detect_text_reference_opportunities_does_not_reresolve_structured_rows(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {}})
    monkeypatch.setattr(
        mod,
        "load_paper_guide_reference_index",
        lambda _source_path: [
            {
                "ref_num": 50,
                "title": "Snapshot Compressive Imaging: Theory, Algorithms, and Applications",
                "text": "Snapshot Compressive Imaging: Theory, Algorithms, and Applications.",
                "doi": "10.1109/msp.2020.3023869",
                "first_citation_context": "Video snapshot compressive imaging (SCI) [50] records dynamic scenes.",
                "first_citation_location": "Introduction",
            }
        ],
    )

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("structured rows must not rescan the global reference index")

    monkeypatch.setattr(mod, "resolve_reference_entry", fail_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="SCI 这条路线是怎么从视频走到 3D 场景重建的？",
        answer="",
        answer_hits=[
            {
                "text": "SCI captures high-dimensional dynamic scenes.",
                "meta": {
                    "source_path": "db/demo/scinerf.en.md",
                    "source_sha1": "abc",
                    "heading_path": "Introduction",
                },
            }
        ],
        db_dir="db",
    )

    assert opportunities
    assert opportunities[0]["ref_num"] == 50


def test_reference_title_starting_with_focus_outranks_incidental_title_match() -> None:
    from kb import paper_guide_reference_opportunities as mod

    focus = "snapshot compressive imaging"
    exact_topic = {
        "title": "Snapshot Compressive Imaging: Theory, Algorithms, and Applications",
        "raw": "IEEE Signal Processing Magazine, 2021.",
    }
    incidental_match = {
        "title": "BIRNAT: Bidirectional Recurrent Networks for Video Snapshot Compressive Imaging",
        "raw": "ECCV, 2020.",
    }

    assert mod._score_reference_label_match(focus, exact_topic) > mod._score_reference_label_match(
        focus,
        incidental_match,
    )


def test_citation_lookup_opportunities_keep_only_resolved_target_reference() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt='Where is "Target work" cited?',
        answer="Target work is reference [9].",
        prompt_family="citation_lookup",
        source_path="db/demo/paper.en.md",
        support_resolution=[
            {
                "source_path": "db/demo/paper.en.md",
                "sid": "s1234abcd",
                "heading_path": "Introduction",
                "locate_anchor": "Compressive sensing [7,9,14] has been adopted to reduce measurements.",
                "ref_nums": [9],
                "claim_type": "prior_work",
                "cite_policy": "prefer_ref",
            }
        ],
        support_slots=_admm_support_slots(),
        max_items=3,
    )

    assert [item["ref_num"] for item in opportunities] == [9]


def test_detect_text_reference_opportunities_for_ordinary_reading_route(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})
    monkeypatch.setattr(
        mod,
        "load_paper_guide_reference_index",
        lambda _source_path: [
            {
                "ref_num": 4,
                "title": "Single-pixel imaging via compressive sampling",
                "text": "Duarte et al. Single-pixel imaging via compressive sampling.",
                "first_citation_context": "Single-pixel imaging via compressive sampling [4] is a foundational reading point.",
                "first_citation_location": "Principles",
            }
        ],
    )

    def fake_resolve(_index, source_path, ref_num, *, source_sha1=""):
        del _index, source_path, source_sha1
        if int(ref_num) == 4:
            return {
                "ref": {
                    "title": "Single-pixel imaging via compressive sampling",
                    "raw": "Duarte et al. Single-pixel imaging via compressive sampling.",
                }
            }
        return None

    monkeypatch.setattr(mod, "resolve_reference_entry", fake_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？",
        answer="先读 single-pixel imaging 的基础综述，再读编码选择和 deep learning SPI。",
        answer_hits=[
            {
                "text": "Principles and prospects for single-pixel imaging.",
                "meta": {
                    "source_path": "db/demo/spi-prospects.en.md",
                    "source_sha1": "abc",
                    "heading_path": "Principles",
                },
            }
        ],
        db_dir="db",
    )

    assert opportunities
    assert opportunities[0]["label"] == "single-pixel imaging"
    assert opportunities[0]["ref_num"] == 4


def test_detect_text_reference_opportunities_uses_nearest_label_before_ref(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})
    monkeypatch.setattr(
        mod,
        "load_paper_guide_reference_index",
        lambda _source_path: [
            {
                "ref_num": 46,
                "title": "Single-pixel imaging by means of Fourier spectrum acquisition",
                "text": "Single-pixel imaging by means of Fourier spectrum acquisition.",
                "first_citation_context": (
                    "Universal patterns include Hadamard basis [45], Fourier basis [46], "
                    "and Cosine transform basis [47]."
                ),
                "first_citation_location": "Introduction",
            }
        ],
    )

    def fake_resolve(_index, source_path, ref_num, *, source_sha1=""):
        del _index, source_path, source_sha1
        if int(ref_num) == 46:
            return {
                "ref": {
                    "title": "Single-pixel imaging by means of Fourier spectrum acquisition",
                    "raw": "Single-pixel imaging by means of Fourier spectrum acquisition.",
                }
            }
        return None

    monkeypatch.setattr(mod, "resolve_reference_entry", fake_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？",
        answer="Hadamard 和 Fourier 编码选择是理解 SPI 实验设计的关键。",
        answer_hits=[
            {
                "text": "Hadamard and Fourier basis patterns differ in SPI.",
                "meta": {
                    "source_path": "db/demo/dl-spi.en.md",
                    "source_sha1": "abc",
                    "heading_path": "Introduction",
                },
            }
        ],
        db_dir="db",
    )

    assert opportunities
    assert opportunities[0]["label"] == "Fourier"
    assert opportunities[0]["ref_num"] == 46


def test_detect_text_reference_opportunities_rejects_broad_label_for_unrelated_ref(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})
    monkeypatch.setattr(
        mod,
        "load_paper_guide_reference_index",
        lambda _source_path: [
            {
                "ref_num": 2,
                "title": "Optical Coherence Tomography",
                "text": "Huang et al. Optical Coherence Tomography.",
                "first_citation_context": "Single-pixel imaging is discussed alongside other optical imaging work [2].",
                "first_citation_location": "Introduction",
            }
        ],
    )

    def fake_resolve(_index, source_path, ref_num, *, source_sha1=""):
        del _index, source_path, source_sha1
        if int(ref_num) == 2:
            return {
                "ref": {
                    "title": "Optical Coherence Tomography",
                    "raw": "Huang et al. Optical Coherence Tomography.",
                }
            }
        return None

    monkeypatch.setattr(mod, "resolve_reference_entry", fake_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？",
        answer="先读 single-pixel imaging 的基础综述，再读编码方案。",
        answer_hits=[
            {
                "text": "Single-pixel imaging overview.",
                "meta": {
                    "source_path": "db/demo/hadamard-fourier.en.md",
                    "source_sha1": "abc",
                    "heading_path": "Introduction",
                },
            }
        ],
        db_dir="db",
    )

    assert opportunities == []


def test_detect_text_reference_opportunities_ignores_uncited_answer_hits(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_reference_rows(source_path):
        if "uncited" not in str(source_path):
            return []
        return [
            {
                "ref_num": 4,
                "title": "Distributed optimization and statistical learning via ADMM",
                "text": "Distributed optimization and statistical learning via ADMM.",
                "first_citation_context": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                "first_citation_location": "Related Work",
            }
        ]

    monkeypatch.setattr(mod, "load_paper_guide_reference_index", fake_reference_rows)

    def fake_resolve(_index, source_path, ref_num, *, source_sha1=""):
        del _index, source_path, source_sha1
        if int(ref_num) == 4:
            return {
                "ref": {
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "raw": "Distributed optimization and statistical learning via ADMM.",
                }
            }
        return None

    monkeypatch.setattr(mod, "resolve_reference_entry", fake_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="Where did ADMM come from?",
        answer="The answer is grounded in the first retrieved paper [1].",
        answer_hits=[
            {
                "text": "The cited paper discusses reconstruction background.",
                "meta": {"source_path": "db/demo/cited.en.md", "source_sha1": "abc"},
            },
            {
                "text": "Most existing methods employ ADMM.",
                "meta": {"source_path": "db/demo/uncited.en.md", "source_sha1": "def"},
            },
        ],
        db_dir="db",
    )

    assert opportunities == []


def test_detect_text_reference_opportunities_keeps_cited_answer_hits(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

    def fake_reference_rows(source_path):
        if "uncited" not in str(source_path):
            return []
        return [
            {
                "ref_num": 4,
                "title": "Distributed optimization and statistical learning via ADMM",
                "text": "Distributed optimization and statistical learning via ADMM.",
                "first_citation_context": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                "first_citation_location": "Related Work",
            }
        ]

    monkeypatch.setattr(mod, "load_paper_guide_reference_index", fake_reference_rows)

    def fake_resolve(_index, source_path, ref_num, *, source_sha1=""):
        del _index, source_path, source_sha1
        if int(ref_num) == 4:
            return {
                "ref": {
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "raw": "Distributed optimization and statistical learning via ADMM.",
                }
            }
        return None

    monkeypatch.setattr(mod, "resolve_reference_entry", fake_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="Where did ADMM come from?",
        answer="The answer is grounded in the second retrieved paper [2].",
        answer_hits=[
            {
                "text": "The first paper discusses reconstruction background.",
                "meta": {"source_path": "db/demo/cited.en.md", "source_sha1": "abc"},
            },
            {
                "text": "Most existing methods employ ADMM.",
                "meta": {"source_path": "db/demo/uncited.en.md", "source_sha1": "def"},
            },
        ],
        db_dir="db",
    )

    assert opportunities
    assert opportunities[0]["ref_num"] == 4
    assert opportunities[0]["source_path"] == "db/demo/uncited.en.md"


def test_detect_text_reference_opportunities_keeps_common_spad_label_for_reading_pair(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})
    monkeypatch.setattr(
        mod,
        "load_paper_guide_reference_index",
        lambda _source_path: [
            {
                "ref_num": 6,
                "title": "Confocal-based fluorescence fluctuation spectroscopy with a SPAD array detector",
                "text": "Confocal-based fluorescence lifetime spectroscopy with a SPAD array detector.",
                "first_citation_context": "The method models SPAD arrays and detector noise using prior SPAD work [6].",
                "first_citation_location": "Figure 1",
            }
        ],
    )

    def fake_resolve(_index, source_path, ref_num, *, source_sha1=""):
        del _index, source_path, source_sha1
        if int(ref_num) == 6:
            return {
                "ref": {
                    "title": "Confocal-based fluorescence fluctuation spectroscopy with a SPAD array detector",
                    "raw": "Confocal-based fluorescence lifetime spectroscopy with a SPAD array detector.",
                }
            }
        return None

    monkeypatch.setattr(mod, "resolve_reference_entry", fake_resolve)

    opportunities = detect_text_reference_opportunities(
        prompt="单光子成像里，探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？",
        answer="先读 SPAD detector 噪声背景，再看 physics-informed deep learning 如何建模 SPAD 噪声。",
        answer_hits=[
            {
                "text": "The method models SPAD arrays and noise sources.",
                "meta": {
                    "source_path": "db/demo/pidl.en.md",
                    "source_sha1": "abc",
                    "heading_path": "Figure 1",
                },
            }
        ],
        db_dir="db",
    )

    assert opportunities
    assert opportunities[0]["label"] == "physics-informed deep learning" or opportunities[0]["label"] == "SPAD"
    assert opportunities[0]["ref_num"] == 6


def test_reference_opportunities_can_be_injected_inline_without_tail_note() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt="I am new to this. Is ADMM original to this paper, or does it come from earlier work?",
        answer="ADMM is not introduced as this paper's original invention; it is prior optimization machinery.",
        prompt_family="overview",
        source_path="db/demo/scinerf.en.md",
        support_slots=_admm_support_slots(),
        max_items=3,
    )

    answer, meta = inject_reference_opportunity_citations_inline(
        "ADMM is not introduced as this paper's original invention; it is prior optimization machinery.",
        prompt="I am new to this. Is ADMM original to this paper, or does it come from earlier work?",
        opportunities=opportunities,
    )

    assert "[[CITE:s1234abcd:4]]" in answer
    assert "[[CITE:s1234abcd:21]]" not in answer
    assert "citation trail" not in answer
    assert meta["mode"] == "inline"
    assert meta["injected_refs"] == [4]


def test_reference_opportunities_convert_matching_bare_ref_to_structured_marker() -> None:
    answer, meta = inject_reference_opportunity_citations_inline(
        "ADMM is prior optimization machinery [4].",
        prompt="Is ADMM original to this paper?",
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 4,
                "label": "ADMM",
                "evidence_quote": "Most existing methods employ ADMM [4].",
            }
        ],
    )

    assert answer == "ADMM is prior optimization machinery [[CITE:s1234abcd:4]]."
    assert meta["mode"] == "inline"
    assert meta["injected_refs"] == [4]


def test_reference_opportunities_inject_on_pronoun_answer_when_prompt_names_label() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt="ADMM 是作者自己发明的吗，还是借鉴了前人的方法？",
        answer="不是，它更像是作者沿用的已有优化工具。",
        prompt_family="overview",
        source_path="db/demo/scinerf.en.md",
        support_slots=_admm_support_slots(),
        max_items=3,
    )

    answer, meta = inject_reference_opportunity_citations_inline(
        "不是，它更像是作者沿用的已有优化工具。",
        prompt="ADMM 是作者自己发明的吗，还是借鉴了前人的方法？",
        opportunities=opportunities,
    )

    assert "[[CITE:s1234abcd:4]]" in answer
    assert "citation trail" not in answer
    assert meta["mode"] == "inline"
    assert meta["injected_refs"] == [4]


def test_reference_opportunities_skip_speculative_next_step_line() -> None:
    answer, meta = inject_reference_opportunity_citations_inline(
        (
            "ADMM 的来源：它是成熟优化工具，并非这篇论文原创。\n"
            "下一步建议：如果你想了解 ADMM 的原始出处，可以查阅参考文献；它很可能引用 Boyd 的综述。"
        ),
        prompt="ADMM 是作者自己发明的吗？",
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 4,
                "label": "ADMM",
                "evidence_quote": "Most existing methods employ ADMM [4].",
                "ref_title": "Distributed Optimization and Statistical Learning via ADMM",
            }
        ],
    )

    lines = answer.splitlines()
    assert "[[CITE:s1234abcd:4]]" in lines[0]
    assert "[[CITE:s1234abcd:4]]" not in lines[1]
    assert meta["mode"] == "inline"
    assert meta["injected_refs"] == [4]


def test_reference_opportunities_do_not_attach_admm_to_admm_net_line() -> None:
    answer, meta = inject_reference_opportunity_citations_inline(
        "ADMM-Net 是把这种优化思路展开成网络的前人工作。",
        prompt="ADMM-Net 之前是谁做的？",
        opportunities=[
            {"sid": "s1234abcd", "ref_num": 4, "label": "ADMM"},
            {"sid": "s1234abcd", "ref_num": 21, "label": "ADMM-Net"},
        ],
    )

    assert "[[CITE:s1234abcd:21]]" in answer
    assert "[[CITE:s1234abcd:4]]" not in answer
    assert meta["injected_refs"] == [21]


def test_reference_opportunities_do_not_stack_multiple_auto_markers_on_one_line() -> None:
    answer, meta = inject_reference_opportunity_citations_inline(
        "ADMM is prior optimization machinery rather than a new contribution.",
        prompt="Where did ADMM come from?",
        opportunities=[
            {"sid": "s1234abcd", "ref_num": 4, "label": "ADMM"},
            {"sid": "s1234abcd", "ref_num": 7, "label": "ADMM"},
        ],
    )

    assert answer.count("[[CITE:") == 1
    assert "[[CITE:s1234abcd:4]]" in answer
    assert "[[CITE:s1234abcd:7]]" not in answer
    assert meta["injected_refs"] == [4]


def test_reference_opportunities_build_prompt_block_for_generation() -> None:
    block = build_reference_opportunities_prompt_block(
        [
            {
                "sid": "s1234abcd",
                "ref_num": 4,
                "label": "ADMM",
                "heading_path": "2. Related Work",
                "evidence_quote": "Most existing methods employ ADMM [4].",
            }
        ]
    )

    assert "Upstream reference opportunities:" in block
    assert "cite_example=[[CITE:s1234abcd:4]]" in block
    assert "Do not dump these as a separate bibliography list" in block


def test_apply_reference_opportunities_uses_tail_only_when_no_sentence_matches() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "The retrieved section is short and does not name the upstream method.",
        prompt="Where did ADMM come from?",
        opportunities=[{"sid": "s1234abcd", "ref_num": 4, "label": "ADMM"}],
    )

    assert meta["mode"] == "tail"
    assert meta["tail_used"] is True
    assert "citation trail" in answer
    assert "[[CITE:s1234abcd:4]]" in answer


def test_apply_reference_opportunities_uses_tail_for_lineage_prompt() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "\u8fd9\u6761\u6280\u672f\u8def\u7ebf\u5206\u4e3a\u538b\u7f29\u5149\u8c31\u6210\u50cf\u3001"
        "\u65f6\u95f4\u7ef4\u5ea6\u7f16\u7801\u548c 3D \u573a\u666f\u8868\u793a\u4e09\u4e2a\u9636\u6bb5\u3002",
        prompt=(
            "SCI \u6216\u538b\u7f29\u5feb\u7167\u6210\u50cf\u8fd9\u6761\u7ebf\uff0c"
            "\u662f\u600e\u4e48\u4ece\u5149\u8c31\u6210\u50cf\u8d70\u5230 3D \u573a\u666f\u91cd\u5efa\u7684\uff1f"
        ),
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 50,
                "label": "snapshot compressive imaging",
                "ref_title": "Snapshot compressive imaging: theory, algorithms, and applications",
            }
        ],
    )

    assert meta["mode"] == "tail"
    assert meta["tail_used"] is True
    assert meta["tail_refs"] == [50]
    assert "[[CITE:s1234abcd:50]]" in answer


def test_reference_opportunity_prefers_upstream_identity_sentence() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "The retrieval component is based on DPR and uses a pre-trained bi-encoder.\n"
        "Dense Passage Retrieval was prior work by Karpukhin et al., not an invention of RAG.",
        prompt=(
            "Did RAG invent Dense Passage Retrieval (DPR), or reuse the prior "
            "work by Karpukhin et al.?"
        ),
        opportunities=[
            {
                "sid": "srag",
                "ref_num": 26,
                "label": "DPR",
                "ref_title": "Dense Passage Retrieval for Open-Domain Question Answering",
                "evidence_quote": "The retrieval component in RAG is based on DPR [26].",
            }
        ],
    )

    lines = answer.splitlines()
    assert "[[CITE:srag:26]]" not in lines[0]
    assert "[[CITE:srag:26]]" in lines[1]
    assert "Karpukhin" in lines[1]
    assert meta["injected_refs"] == [26]


def test_reference_opportunity_relocates_existing_marker_to_upstream_identity() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "Upstream paper: DPR comes from Karpukhin et al., Dense Passage Retrieval "
        "for Open-Domain Question Answering.\n"
        "The DPR retriever was trained on Natural Questions [[CITE:srag:26]].",
        prompt=(
            "Did RAG invent Dense Passage Retrieval (DPR), or reuse prior work? "
            "Identify the upstream paper."
        ),
        opportunities=[
            {
                "sid": "srag",
                "ref_num": 26,
                "label": "DPR",
                "ref_title": "Dense Passage Retrieval for Open-Domain Question Answering",
                "ref_authors": "Vladimir Karpukhin et al.",
                "ref_raw": (
                    "Vladimir Karpukhin et al. Dense Passage Retrieval for "
                    "Open-Domain Question Answering."
                ),
                "evidence_quote": "The retrieval component is based on DPR [26].",
            }
        ],
    )

    lines = answer.splitlines()
    assert "[[CITE:srag:26]]" in lines[0]
    assert "[[CITE:srag:26]]" not in lines[1]
    assert answer.count("[[CITE:srag:26]]") == 1
    assert meta["mode"] == "already_present"


def test_reference_opportunity_completes_verified_upstream_identity() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "No, RAG reuses Dense Passage Retrieval (DPR) as prior work.\n"
        "Upstream paper: DPR is cited as a reference in the RAG paper.",
        prompt=(
            "Did RAG invent Dense Passage Retrieval (DPR), or reuse the prior "
            "work by Karpukhin et al.?"
        ),
        opportunities=[
            {
                "sid": "srag",
                "ref_num": 26,
                "label": "DPR",
                "ref_title": "Dense Passage Retrieval for Open-Domain Question Answering",
                "ref_authors": "karpukhin",
                "ref_year": "2020",
                "ref_raw": "Vladimir Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering.",
                "evidence_quote": "The retrieval component is based on DPR [26].",
            }
        ],
    )

    upstream = answer.splitlines()[1]
    assert "Dense Passage Retrieval for Open-Domain Question Answering" in upstream
    assert "Karpukhin et al" in upstream
    assert "2020" in upstream
    assert "[[CITE:srag:26]]" in upstream
    assert meta["injected_refs"] == [26]


def test_apply_reference_opportunities_suppresses_tail_for_broad_synthesis_question() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "Deep learning improves image quality, but it can require paired data and heavy compute.",
        prompt=(
            "\u6df1\u5ea6\u5b66\u4e60\u7ed9\u5355\u50cf\u7d20\u6210\u50cf"
            "\u5e26\u6765\u7684\u597d\u5904\u548c\u5751\u5206\u522b\u662f\u4ec0\u4e48\uff1f"
        ),
        opportunities=[{"sid": "s1234abcd", "ref_num": 1, "label": "single-pixel imaging"}],
    )

    assert answer == "Deep learning improves image quality, but it can require paired data and heavy compute."
    assert meta["tail_used"] is False
    assert meta["tail_suppressed"] is True
    assert "[[CITE:" not in answer


def test_apply_reference_opportunities_does_not_inject_on_generic_benefits_sentence() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "Deep learning brings significant benefits to single-pixel imaging, but it also introduces challenges.",
        prompt="What benefits and pitfalls does deep learning bring to single-pixel imaging?",
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 22,
                "label": "single-pixel imaging",
                "evidence_quote": "Keywords: Single-pixel imaging Information extraction network Deep learning",
            }
        ],
    )

    assert "[[CITE:" not in answer
    assert meta["tail_used"] is False
    assert meta["tail_suppressed"] is True


def test_apply_reference_opportunities_uses_inline_system_b_for_ordinary_matching_sentence() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        "\u6df1\u5ea6\u5b66\u4e60\u4e3a\u5355\u50cf\u7d20\u6210\u50cf\u5e26\u6765\u4e86\u66f4\u5feb\u7684\u91cd\u5efa\u901f\u5ea6\u3002\n"
        "SPAD \u5355\u5149\u5b50\u6210\u50cf\u5219\u66f4\u5173\u5fc3\u566a\u58f0\u5efa\u6a21\u548c\u8d85\u5206\u8fa8\u91cd\u5efa\u3002",
        prompt=(
            "\u6df1\u5ea6\u5b66\u4e60\u7ed9\u5355\u50cf\u7d20\u6210\u50cf"
            "\u5e26\u6765\u7684\u597d\u5904\u548c\u5751\u5206\u522b\u662f\u4ec0\u4e48\uff1f"
        ),
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 1,
                "label": "single-pixel imaging",
                "evidence_quote": "Single-pixel imaging via deep learning improves reconstruction speed.",
            },
            {
                "sid": "s1234abcd",
                "ref_num": 22,
                "label": "SPAD",
                "evidence_quote": "SPAD noise is important for single-photon imaging.",
            },
        ],
    )

    assert "[[CITE:s1234abcd:1]]" in answer
    assert "[[CITE:s1234abcd:22]]" in answer
    assert answer.count("[[CITE:") == 2
    assert meta["mode"] == "inline"
    assert meta["injected_refs"] == [1, 22]
    assert meta["tail_used"] is False


def test_reference_opportunities_do_not_inject_on_reading_list_title_line() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        (
            "2. **《Hadamard single-pixel imaging versus Fourier single-pixel imaging》**（2017）[1]\n"
            "- **看什么**：重点理解 Hadamard 与 Fourier 基扫描如何影响采样效率。"
        ),
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？",
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 46,
                "label": "Fourier",
                "evidence_quote": "Universal patterns include Hadamard basis [45] and Fourier basis [46].",
                "ref_title": "Single-pixel imaging by means of Fourier spectrum acquisition",
            }
        ],
    )

    lines = answer.splitlines()
    assert "[[CITE:s1234abcd:46]]" not in lines[0]
    assert "[[CITE:s1234abcd:46]]" in lines[1]
    assert meta["mode"] == "inline"


def test_reference_opportunities_do_not_inject_on_labeled_bibliography_line() -> None:
    answer, meta = apply_reference_opportunities_to_answer(
        (
            "**文献**：*Hadamard single-pixel imaging versus Fourier single-pixel imaging* [1]\n"
            "- **看什么**：重点理解 Hadamard 与 Fourier 基扫描如何影响采样效率。"
        ),
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？",
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 46,
                "label": "Fourier",
                "evidence_quote": "Universal patterns include Hadamard basis [45] and Fourier basis [46].",
                "ref_title": "Single-pixel imaging by means of Fourier spectrum acquisition",
            }
        ],
    )

    lines = answer.splitlines()
    assert "[[CITE:s1234abcd:46]]" not in lines[0]
    assert "[[CITE:s1234abcd:46]]" in lines[1]
    assert meta["mode"] == "inline"


def test_reference_opportunities_merge_candidate_refs_without_duplicates() -> None:
    merged = merge_reference_opportunity_candidate_refs(
        {"db/demo/scinerf.en.md": [4]},
        [
            {"source_path": "db/demo/scinerf.en.md", "ref_num": 4},
            {"source_path": "db/demo/scinerf.en.md", "ref_num": 21},
        ],
    )

    assert merged == {"db/demo/scinerf.en.md": [4, 21]}


def test_reference_opportunity_note_does_not_duplicate_existing_marker() -> None:
    answer = append_reference_opportunity_note(
        "Already cited [[CITE:s1234abcd:4]].",
        prompt="Where did ADMM come from?",
        opportunities=[
            {"sid": "s1234abcd", "ref_num": 4, "label": "ADMM"},
            {"sid": "s1234abcd", "ref_num": 21, "label": "ADMM-Net"},
        ],
    )

    assert answer.count("[[CITE:s1234abcd:4]]") == 1
    assert "[[CITE:s1234abcd:21]]" in answer


def test_reference_opportunity_tail_dedupes_broad_labels() -> None:
    answer = append_reference_opportunity_note(
        "Read the SPI foundations first.",
        prompt="I need a reading route for single-pixel imaging.",
        opportunities=[
            {
                "sid": "s1234abcd",
                "ref_num": 2,
                "label": "single-pixel imaging",
                "ref_title": "Principles and prospects for single-pixel imaging",
            },
            {
                "sid": "s1234abcd",
                "ref_num": 135,
                "label": "single-pixel imaging",
                "ref_title": "Super-coding resolution single-pixel imaging",
            },
        ],
    )

    assert "[[CITE:s1234abcd:2]]" in answer
    assert "[[CITE:s1234abcd:135]]" not in answer
    assert "Principles and prospects" in answer


def test_strip_reference_opportunity_note_removes_unvalidated_tail() -> None:
    answer = (
        "ADMM is prior optimization machinery.\n\n"
        "To follow the paper's citation trail, open: ADMM [[CITE:s1234abcd:4]]."
    )

    stripped = strip_reference_opportunity_note(answer)

    assert "citation trail" not in stripped
    assert stripped == "ADMM is prior optimization machinery."


def test_reference_opportunity_detector_expands_bundled_answer_passages() -> None:
    opportunities = detect_paper_guide_reference_opportunities(
        prompt=(
            "Did RAG invent Dense Passage Retrieval (DPR), or reuse the prior "
            "work by Karpukhin et al.?"
        ),
        answer="",
        prompt_family="overview",
        source_path="db/demo/rag.en.md",
        support_slots=[
            {
                "source_path": "db/demo/rag.en.md",
                "heading_path": "4 Results",
                "evidence_quote": "RAG avoids specialized salient span masking [20].",
            }
        ],
        answer_hits=[
            {
                "text": "one bundled RAG document",
                "meta": {
                    "source_path": "db/demo/rag.en.md",
                    "same_source_evidence_bundle": True,
                    "source_passages": [
                        {
                            "heading_path": "4 Results",
                            "text": "RAG avoids specialized salient span masking [20].",
                        },
                        {
                            "heading_path": "2.2 Retriever: DPR",
                            "text": (
                                "The retrieval component is based on Dense Passage "
                                "Retrieval (DPR) [26], and RAG initializes its pre-trained "
                                "bi-encoder and document index from DPR."
                            ),
                        },
                    ],
                },
            }
        ],
        max_items=1,
    )

    assert opportunities
    assert opportunities[0]["ref_num"] == 26
    assert "Retriever: DPR" in opportunities[0]["heading_path"]


def test_reference_opportunity_detector_recovers_exact_target_context_from_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "rag.en.md"
    source.write_text(
        "# RAG\n\n## 2.2 Retriever: DPR\n\n"
        "The retrieval component is based on DPR [26]. DPR follows a bi-encoder architecture.\n\n"
        "## 3 Experiments\n\nOpen-domain QA is an important testbed [20].\n\n"
        "## References\n\n[20] Retrieval-Augmented pretraining.\n"
        "[26] Dense Passage Retrieval for Open-Domain Question Answering.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reference_opportunities,
        "load_paper_guide_reference_index",
        lambda _source: [
            {
                "ref_num": 20,
                "title": "Retrieval-Augmented Language Model Pre-Training",
            },
            {
                "ref_num": 26,
                "title": "Dense Passage Retrieval for Open-Domain Question Answering",
            },
        ],
    )

    opportunities = detect_paper_guide_reference_opportunities(
        prompt=(
            "Did RAG invent Dense Passage Retrieval (DPR), or reuse the prior "
            "work by Karpukhin et al.?"
        ),
        answer="RAG reuses DPR.",
        prompt_family="overview",
        source_path=str(source),
        support_slots=[
            {
                "source_path": str(source),
                "heading_path": "3 Experiments",
                "evidence_quote": "Open-domain QA is an important testbed [20].",
            }
        ],
        max_items=1,
    )

    assert opportunities[0]["ref_num"] == 26
    assert "based on DPR [26]" in opportunities[0]["evidence_quote"]
