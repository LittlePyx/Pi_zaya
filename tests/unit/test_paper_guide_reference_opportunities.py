from __future__ import annotations

from kb.paper_guide_reference_opportunities import (
    apply_reference_opportunities_to_answer,
    append_reference_opportunity_note,
    build_reference_opportunities_prompt_block,
    detect_paper_guide_reference_opportunities,
    detect_text_reference_opportunities,
    inject_reference_opportunity_citations_inline,
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


def test_detect_text_reference_opportunities_for_normal_question(monkeypatch) -> None:
    from kb import paper_guide_reference_opportunities as mod

    monkeypatch.setattr(mod, "load_reference_index", lambda _db_dir: {"docs": {"demo": {}}})

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


def test_strip_reference_opportunity_note_removes_unvalidated_tail() -> None:
    answer = (
        "ADMM is prior optimization machinery.\n\n"
        "To follow the paper's citation trail, open: ADMM [[CITE:s1234abcd:4]]."
    )

    stripped = strip_reference_opportunity_note(answer)

    assert "citation trail" not in stripped
    assert stripped == "ADMM is prior optimization machinery."
