from __future__ import annotations

import pytest

from kb.citation_plan import build_citation_plan_prompt_block


CONDITION = (
    "In stark contrast to our approach, they need a perfect, explicit computational model of the corruption; "
    "our requirement of the knowledge of the appropriate statistical summary (mean, median) is significantly less limiting."
)


@pytest.mark.parametrize("intent", ["method_explain", "paper_summary", "comparison"])
def test_prompt_preserves_complete_condition_already_in_source_plan(intent):
    plan = {"intent": intent, "slots": [{
        "preferred_system": "system_a", "source_path": "noise2noise.en.md",
        "candidate_hits": [2], "heading_path": "4. Discussion", "evidence_quote": CONDITION,
    }]}
    block = build_citation_plan_prompt_block(plan)
    assert CONDITION in block
    assert "retrieved_hit=2" in block
    assert "heading=4. Discussion" in block


def test_noncomparison_prompt_keeps_a_generic_second_sentence_condition():
    lead = "The estimator can be trained from observed pairs while retaining the same population optimum. "
    quote = lead * 2 + "This conclusion requires conditional independence of the two observations."
    block = build_citation_plan_prompt_block({"intent": "method_explain", "slots": [{
        "preferred_system": "system_a", "candidate_hits": [1], "evidence_quote": quote,
    }]})
    assert quote in block


def test_prompt_never_completes_a_missing_source_condition_or_merges_slots():
    truncated = "Our requirement is knowledge of the appropriate sta..."
    other = "A different method assumes pixel-wise independent noise."
    block = build_citation_plan_prompt_block({"intent": "method_explain", "slots": [
        {"preferred_system": "system_a", "candidate_hits": [1], "evidence_quote": truncated},
        {"preferred_system": "system_a", "candidate_hits": [2], "evidence_quote": other},
    ]})
    assert "mean" not in block and "median" not in block
    rows = [row for row in block.splitlines() if "evidence=" in row]
    assert len(rows) == 2
    assert truncated in rows[0] and other not in rows[0]
    assert "retrieved_hit=1" in rows[0] and "retrieved_hit=2" in rows[1]


def test_long_prompt_quote_still_has_an_explicit_bound_and_truncation_marker():
    quote = "Bounded evidence. " * 100
    block = build_citation_plan_prompt_block({"intent": "method_explain", "slots": [{
        "preferred_system": "system_a", "candidate_hits": [1], "evidence_quote": quote,
    }]})
    rendered = block.split("evidence=", 1)[1]
    assert rendered.endswith("...")
    # The existing compactor appends its ellipsis after the bounded preview.
    assert len(rendered.removesuffix("...")) <= 720
    assert quote not in block
