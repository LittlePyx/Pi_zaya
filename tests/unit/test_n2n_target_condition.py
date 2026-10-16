from __future__ import annotations

from pathlib import Path

import pytest

from kb.direct_question_evidence import attach_direct_source_evidence, direct_question_evidence_hits
from kb.denoising_quality_boundary import complete_n2n_corruption_model_boundary
from kb.paper_guide_answer_selection import _merge_same_source_answer_hits

PROMPT = "Does this method require a likelihood or an explicit model of the corruption process? What statistical condition remains necessary?"
CONTRAST = "In stark contrast to our approach, they need a perfect, explicit computational model of the corruption; our requirement of the knowledge of the appropriate statistical summary (mean, median) is significantly less limiting."
CONDITION = ("A trivial property of L 2 minimization is that the estimate remains unchanged if we replace the targets with random numbers whose expectations match the targets. "
             "The optimal network parameters remain unchanged, if input-conditioned target distributions p ( y | x ) are replaced with arbitrary distributions that have the same conditional expected values. "
             "This implies that we can, in principle, corrupt the training targets of a neural network with zero-mean noise without changing what the network learns.")
PARTIAL = "No, the method does not require an explicit model of the corruption process. It requires the appropriate statistical summary, such as the mean or median [1]."


def _chain(tmp_path: Path, condition: str = CONDITION):
    path = tmp_path / "noise2noise.en.md"
    path.write_text("<!-- kb_page: 2 -->\n" + condition + "\n\n<!-- kb_page: 8 -->\n" + CONTRAST, encoding="utf8")
    direct = direct_question_evidence_hits(path, PROMPT)
    hits = _merge_same_source_answer_hits(direct)
    plan = {"budget": {"system_a": 2}, "slots": [{"preferred_system": "system_a", "source_path": str(path), "evidence_quote": CONTRAST, "page_start": 8, "candidate_hits": [1]}]}
    return path, direct, hits, attach_direct_source_evidence(plan, hits)


def test_both_required_facets_recalled_and_bound_to_separate_pages(tmp_path):
    path, direct, hits, plan = _chain(tmp_path)
    assert {h["meta"]["page_start"] for h in direct} == {2, 8}
    assert "same conditional expected values" in hits[0]["text"]
    assert {s["page_start"] for s in plan["slots"]} == {2, 8}
    for slot in plan["slots"]:
        assert slot["source_path"] == str(path)
        assert slot["block_id"]
        assert not (slot["page_start"] == 8 and "input-conditioned" in " ".join(slot.get("source_evidence_quotes", [])))


def test_mean_word_does_not_hide_missing_target_condition(tmp_path):
    _, _, hits, plan = _chain(tmp_path)
    result = complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=hits)
    assert "input-conditioned target distributions" in result
    assert "same conditional expected values" in result
    assert "L2" in result and "zero-mean" in result
    assert result.endswith("[1].")
    assert complete_n2n_corruption_model_boundary(result, prompt=PROMPT, citation_plan=plan, answer_hits=hits) == result


@pytest.mark.parametrize("bad", [
    CONDITION.replace("input-conditioned target", "target-conditioned input"),
    CONDITION.replace("same conditional expected values", "different conditional expected values"),
    CONDITION.replace("same conditional expected values", "same unconditional expected values"),
    CONDITION.replace("zero-mean noise", "noise with mean 2"),
    CONDITION.replace("corrupt the training targets", "corrupt the training inputs"),
    CONDITION.replace("are replaced", "are not replaced"),
    CONDITION.replace("without changing", "while changing"),
    CONDITION.replace("L 2 minimization", "L 1 minimization"),
])
def test_incomplete_or_reversed_source_cannot_complete_condition(tmp_path, bad):
    _, _, hits, plan = _chain(tmp_path, bad)
    assert complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=hits) == PARTIAL


def test_wrong_source_stale_block_and_empty_hits_cannot_supply_condition(tmp_path):
    path, _, hits, plan = _chain(tmp_path)
    path.write_text("<!-- kb_page: 2 -->\nChanged source.", encoding="utf8")
    assert complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=hits) == PARTIAL
    assert complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=[]) == PARTIAL
    wrong = [{"meta": {"source_path": str(tmp_path / "other" / path.name)}}]
    assert complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=wrong) == PARTIAL


def test_verified_condition_removes_only_the_superseded_gap_caveat(tmp_path):
    _, _, hits, plan = _chain(tmp_path)
    answer = PARTIAL + "\n\n**Not stated in the retrieved evidence**\n\nThe retrieved snippets do not specify a general formal condition beyond the summary-statistic requirement."
    result = complete_n2n_corruption_model_boundary(answer, prompt=PROMPT, citation_plan=plan, answer_hits=hits)
    assert "do not specify" not in result and "Not stated" not in result
    assert "same conditional expected values" in result


def test_long_direct_paragraph_keeps_the_condition_beyond_preview_limit(tmp_path):
    _, _, hits, _ = _chain(tmp_path, "Other observations. " * 48 + CONDITION)
    assert "same conditional expected values" in hits[0]["text"]


def test_different_sources_cannot_lend_a_condition_or_a_page(tmp_path):
    _, _, hits, plan = _chain(tmp_path)
    plan["slots"][-1]["source_path"] = str(tmp_path / "other" / "noise2noise.en.md")
    assert complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=hits) == PARTIAL
    _, _, hits, plan = _chain(tmp_path)
    plan["slots"][-1]["page_start"] = 8
    assert complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=hits) == PARTIAL


def test_existing_second_page_slot_is_bound_to_the_actual_claim_occurrence(tmp_path):
    from api.claim_passage_plan import prepare_claim_passage_plan

    path, _, hits, plan = _chain(tmp_path)
    answer = complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=hits)
    rec = {"meta": {"canonical_hit_paths": [str(path)], "paper_guide_contracts": {"citation_plan": plan}}}
    result = prepare_claim_passage_plan(rec, {"hits": hits}, answer)
    slots = result["meta"]["paper_guide_contracts"]["citation_plan"]["slots"]
    assert any(s["page_start"] == 2 and s.get("passage_bound_claims") for s in slots)


def test_generation_plan_starting_on_target_page_still_keeps_model_contrast(tmp_path):
    path, direct, hits, _ = _chain(tmp_path)
    target = next(h for h in direct if h["meta"]["page_start"] == 2)
    plan = {"budget": {"system_a": 2}, "slots": [{**target["meta"], "preferred_system": "system_a", "evidence_quote": target["text"], "candidate_hits": [1]}]}
    result = attach_direct_source_evidence(plan, hits)
    assert {s["page_start"] for s in result["slots"]} == {2, 8}
    target_slot = next(s for s in result["slots"] if s["page_start"] == 2)
    assert target_slot["evidence_quote"].startswith("input-conditioned target distributions")
    assert all(s["source_path"] == str(path) for s in result["slots"])


def test_stale_anchor_cannot_grant_a_condition(tmp_path):
    _, _, hits, plan = _chain(tmp_path)
    plan["slots"][-1]["anchor_id"] = "wrong-paragraph"
    assert complete_n2n_corruption_model_boundary(PARTIAL, prompt=PROMPT, citation_plan=plan, answer_hits=hits) == PARTIAL


def test_scoped_selection_recovers_a_direct_facet_lost_by_intermediate_scans(tmp_path):
    from kb.direct_question_evidence import prioritize_direct_question_hits

    path, direct, _, _ = _chain(tmp_path)
    target = next(h for h in direct if h["meta"]["page_start"] == 2)
    selected = [target, {"text": "Other retrieved material.", "meta": {"source_path": str(path)}}]
    refreshed = prioritize_direct_question_hits(selected, [target], limit=2, prompt=PROMPT)
    assert [h["meta"]["page_start"] for h in refreshed] == [8, 2]


def test_condition_card_keeps_both_conditional_mean_and_target_noise_clauses(tmp_path):
    from api.chat_render import _refine_system_a_cite_evidence_from_citation_plan

    path, _, _, plan = _chain(tmp_path)
    slot = next(s for s in plan["slots"] if s["page_start"] == 2)
    detail = {
        "num": 1, "citation_route": "system_a", "source_path": str(path),
        "source_name": path.name, "answer_claim": "Training targets can be corrupted with zero-mean noise without changing what the network learns.",
        "evidence_quote": "We can corrupt the training targets with zero-mean noise.",
        "block_id": slot["block_id"], "page_start": 2,
    }
    cards = _refine_system_a_cite_evidence_from_citation_plan([detail], plan, render_locale="en")
    assert cards[0]["page_start"] == 2
    assert "same conditional expected values" in cards[0]["evidence_quote"]
    assert "training targets" in cards[0]["evidence_quote"]
    assert "y | x" in cards[0]["evidence_quote"]
    assert "y | x" in cards[0]["reader_evidence_quote"]
    assert cards[0]["anchor_kind"] == "paragraph"
    from api.routers.references import _overlay_refs_payload_with_answer_citations

    payload = _overlay_refs_payload_with_answer_citations(
        store=None, conv_id="n2n-reader-source",
        payload={1: {"prompt": PROMPT, "hits": [{"meta": {"source_path": str(path)}, "ui_meta": {}}]}},
        answer_citation_state=({1: cards}, set()),
    )
    reader = payload[1]["hits"][0]["ui_meta"]["reader_open"]
    assert "y | x" in reader["snippet"]
    assert reader["anchorKind"] == "paragraph"
