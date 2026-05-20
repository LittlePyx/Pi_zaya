from __future__ import annotations

from kb.paper_guide_answer_repair import repair_template_only_paper_guide_answer


def test_repair_template_only_paper_guide_answer_uses_support_quote_and_structured_ref(tmp_path) -> None:
    source_path = tmp_path / "paper.en.md"
    source_path.write_text("# Paper", encoding="utf-8")

    out, meta = repair_template_only_paper_guide_answer(
        "The paper cites [4] for this point.\n> Most existing methods employ ADMM [4].",
        prompt="ADMM 是怎么来的？作者是不是借鉴了以前的想法？",
        prompt_family="citation_lookup",
        support_resolution=[
            {
                "source_path": str(source_path),
                "heading_path": "2. Related Work",
                "locate_anchor": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                "resolved_ref_num": 4,
            }
        ],
    )

    assert meta["changed"] is True
    assert "The paper cites" not in out
    assert "不是作为本文原创发明" in out
    assert "2. Related Work" in out
    assert "ADMM" in out
    assert "[[CITE:" in out
    assert ":4]]" in out


def test_repair_template_only_paper_guide_answer_keeps_good_answer_unchanged() -> None:
    answer = (
        "The paper treats ADMM as prior optimization machinery rather than a new contribution.\n\n"
        "> Most existing methods employ ADMM [4]."
    )

    out, meta = repair_template_only_paper_guide_answer(
        answer,
        prompt="Where does ADMM come from?",
        prompt_family="citation_lookup",
        support_resolution=[],
    )

    assert out == answer
    assert meta["changed"] is False


def test_repair_template_only_paper_guide_answer_falls_back_to_quote_line() -> None:
    out, meta = repair_template_only_paper_guide_answer(
        "The paper cites [17] for this point.\n> For complex motions, higher-order spline [17] can be exploited.",
        prompt="Where did the spline idea come from?",
        prompt_family="citation_lookup",
        support_resolution=[],
        fallback_source_path="paper.en.md",
    )

    assert meta["changed"] is True
    assert "The paper cites" not in out
    assert "useful point is not just the number" not in out
    assert "higher-order spline" in out
    assert "[[CITE:" in out
    assert ":17]]" in out
