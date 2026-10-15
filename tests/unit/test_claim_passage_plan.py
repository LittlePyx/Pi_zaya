from __future__ import annotations

import copy
import re

import pytest

from api import chat_render
from api.claim_passage_plan import message_citation_plan, prepare_claim_passage_plan
from kb.source_blocks import load_source_blocks_cached


DIP = (
    "DIP was almost 2 dB worse. DIP is not a learning-based solution, and as such very different from our approach, "
    "but it shares the property that neither clean examples nor an explicit model of the corruption is needed."
)
SUMMARY = (
    "In stark contrast to our approach, they need a perfect, explicit computational model of the corruption; "
    "our requirement of the knowledge of the appropriate statistical summary (mean, median) is significantly less limiting."
)
DIP_CLAIM = 'It also notes that DIP shares the property that "neither clean examples nor an explicit model of the corruption is needed" [1].'
ANSWER = (
    'No — the method does not require an explicit model of the corruption process [1]. '
    'Noise2Noise requires the appropriate statistical summary (mean or median), which is "significantly less limiting". '
    f'{DIP_CLAIM}\n\nThe appropriate statistical summary of the target distribution remains necessary [1].'
)


def _fixture(tmp_path, dip_text=DIP):
    path = tmp_path / "noise2noise.en.md"
    path.write_text(f"# Noise2Noise\n\n<!-- kb_page: 4 -->\n\n## Experiments\n\n{dip_text}\n\n"
                    f"<!-- kb_page: 8 -->\n\n## Discussion\n\n{SUMMARY}\n", encoding="utf8")
    blocks = [b for b in load_source_blocks_cached(path) if b["kind"] == "paragraph"]
    passages = [{k:b[k] for k in ("text", "block_id", "anchor_id", "heading_path", "page_start", "page_end")} for b in blocks]
    slot = {"preferred_system":"system_a", "source_path":str(path), "source_name":"Noise2Noise",
            "candidate_hits":[1], "evidence_quote":SUMMARY,
            **{k:passages[1][k] for k in ("block_id", "anchor_id", "heading_path", "page_start", "page_end")}}
    plan = {"intent":"method_explain", "budget":{"system_a":2,"system_b":0}, "slots":[slot]}
    rec = {"id":2, "role":"assistant", "content":ANSWER, "meta":{
        "canonical_hit_paths":[str(path)], "answer_quality":{"citation_plan":plan},
        "paper_guide_contracts":{"citation_plan":plan},
    }}
    pack = {"render_locale":"en", "hits":[{"text":dip_text, "meta":{
        "source_path":str(path), "source_name":"Noise2Noise", "ref_answer_citation_num":1,
        "source_passages":passages,
    }}]}
    return rec, pack


def test_claim_plan_adds_verified_page_and_keeps_number_and_input_immutable(tmp_path):
    rec, pack = _fixture(tmp_path)
    before = copy.deepcopy((rec,pack))
    result = prepare_claim_passage_plan(rec, pack, ANSWER)
    assert (rec,pack) == before
    slots = message_citation_plan(result)["slots"]
    assert len(slots) == 2
    assert [s["page_start"] for s in slots] == [8,4]
    assert slots[1]["candidate_hits"] == [1]
    assert "DIP" in slots[1]["evidence_quote"]
    assert prepare_claim_passage_plan(result,pack,ANSWER) == result


def test_two_claims_share_verified_passage_without_new_number_or_input_mutation(tmp_path):
    rec, pack = _fixture(tmp_path)
    second = "DIP is not a learning-based solution [1]."
    answer = f"{DIP_CLAIM}\n\n{second}"
    before = copy.deepcopy((rec, pack))
    result = prepare_claim_passage_plan(rec, pack, answer)
    slots = message_citation_plan(result)["slots"]
    assert len(slots) == 2
    assert slots[1]["candidate_hits"] == [1]
    assert len(slots[1]["passage_bound_claims"]) == 2
    assert second in slots[1]["passage_bound_claims"]
    assert (rec, pack) == before
    assert prepare_claim_passage_plan(result, pack, answer) == result


@pytest.mark.parametrize("second", [
    "DIP requires clean examples and an explicit model of the corruption process [1].",
    "DIP achieves 99 dB without clean examples or an explicit model [1].",
    "DIP 需要干净样本和显式污染模型 [1]。",
])
def test_second_claim_does_not_inherit_passage_binding_without_support(tmp_path, second):
    rec, pack = _fixture(tmp_path)
    result = prepare_claim_passage_plan(rec, pack, f"{DIP_CLAIM}\n\n{second}")
    slots = message_citation_plan(result)["slots"]
    assert len(slots) == 2
    assert len(slots[1]["passage_bound_claims"]) == 1
    assert second not in slots[1]["passage_bound_claims"]


@pytest.mark.parametrize("corruption", ["page", "anchor", "block", "heading", "text", "namesake", "passage_source", "negative", "number", "no_citation", "code", "math"])
def test_claim_plan_does_not_invent_or_cross_bind_evidence(tmp_path,corruption):
    rec,pack = _fixture(tmp_path)
    passage = pack["hits"][0]["meta"]["source_passages"][0]
    answer = ANSWER
    if corruption == "page":
        passage["page_start"] = passage["page_end"] = 8
    elif corruption == "anchor":
        passage["anchor_id"] = "p_99999"
    elif corruption == "block":
        passage["block_id"] = "blk_wrong"
    elif corruption == "heading":
        passage["heading_path"] = "Noise2Noise / Discussion"
    elif corruption == "text":
        passage["text"] = DIP + " It achieves 99 dB."
    elif corruption == "passage_source":
        passage["source_path"] = str(tmp_path / "other" / "noise2noise.en.md")
    elif corruption == "namesake":
        other = tmp_path / "other" / "noise2noise.en.md"
        other.parent.mkdir()
        other.write_text((tmp_path / "noise2noise.en.md").read_text(encoding="utf8"),encoding="utf8")
        pack["hits"][0]["meta"]["source_path"] = str(other)
    elif corruption == "negative":
        answer = "DIP requires clean examples and an explicit model of the corruption process [1]."
    elif corruption == "number":
        answer = "DIP achieves 99 dB without clean examples or an explicit model [1]."
    elif corruption == "no_citation":
        answer = DIP_CLAIM.replace("[1]", "")
    elif corruption == "code":
        answer = f"```text\n{DIP_CLAIM}\n```"
    elif corruption == "math":
        answer = "$x=[1]$"
    result = prepare_claim_passage_plan(rec,pack,answer)
    assert len(message_citation_plan(result)["slots"]) == 1


def test_negative_requirement_cannot_borrow_affirmative_source(tmp_path):
    rec, pack = _fixture(tmp_path, "DIP requires clean examples and an explicit model of the corruption process for reconstruction.")
    result = prepare_claim_passage_plan(rec,pack,DIP_CLAIM)
    assert len(message_citation_plan(result)["slots"]) == 1


def test_ambiguous_physical_matches_do_not_choose_by_retrieval_order(tmp_path):
    rec, pack = _fixture(tmp_path)
    path = tmp_path / "noise2noise.en.md"
    path.write_text(path.read_text(encoding="utf8")+f"\n<!-- kb_page: 6 -->\n\n## Duplicate\n\n{DIP}\n",encoding="utf8")
    duplicate = [b for b in load_source_blocks_cached(path) if b.get("page_start")==6 and b["kind"]=="paragraph"][0]
    pack["hits"][0]["meta"]["source_passages"].append(duplicate)
    result = prepare_claim_passage_plan(rec,pack,DIP_CLAIM)
    assert len(message_citation_plan(result)["slots"]) == 1


def test_stored_claim_slot_is_revalidated_after_source_change(tmp_path):
    rec,pack = _fixture(tmp_path)
    bound = prepare_claim_passage_plan(rec,pack,ANSWER)
    assert len(message_citation_plan(bound)["slots"]) == 2
    path = tmp_path / "noise2noise.en.md"
    path.write_text(path.read_text(encoding="utf8").replace(DIP,"This replacement passage contains no evidence about the requested method."),encoding="utf8")
    result = prepare_claim_passage_plan(bound,pack,ANSWER)
    assert len(message_citation_plan(result)["slots"]) == 1


@pytest.mark.parametrize("second_claim", [False, True])
def test_full_render_keeps_each_sentence_on_its_own_passage_and_cache(tmp_path, monkeypatch, second_claim):
    rec,pack = _fixture(tmp_path)
    pack["hits"][0]["meta"]["ref_rank"] = {"why":"STALE page 8 Discussion statistical summary explanation"}
    if second_claim:
        rec["content"] = ANSWER.replace("\n\n", "\n\nDIP is not a learning-based solution [1].\n\n")
    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path",lambda path:False)
    monkeypatch.setattr(chat_render,"_effective_reference_render_pack",lambda raw:raw)
    monkeypatch.setattr(chat_render,"_answer_aligned_reference_render_pack",lambda raw,*args:raw)
    monkeypatch.setattr(chat_render,"_effective_citation_render_locale",lambda raw:"en")
    messages = [{"id":1,"role":"user","content":"What statistical condition remains necessary?"},rec]
    rendered = chat_render.enrich_messages_with_reference_render(messages,{1:pack},conv_id="passage-test")[-1]
    for result in (rendered,chat_render.enrich_messages_with_reference_render([messages[0],rendered],{1:pack},conv_id="passage-test")[-1]):
        cards = result["cite_details"]
        assert {c["page_start"] for c in cards} == {4,8}
        anchors = {c["anchor"]:c for c in cards}
        body = result["rendered_body"]
        markers = list(re.finditer(r'\[1\]\(#([^\s)]+)(?:\s+"[^"]*")?\)',body))
        assert len(markers) == (4 if second_claim else 3)
        assert [anchors[m[1]]["page_start"] for m in markers] == ([8,4,4,8] if second_claim else [8,4,8])
        for card in cards:
            if card["page_start"] == 4:
                assert "DIP" in card["evidence_quote"]
                assert card["passage_bound_claims"]
                assert card["block_id"] and card["anchor_id"]
                assert "STALE" not in card["card_support_explanation"]
                assert "statistical summary" not in card["card_support_explanation"]
                stale_primary = {**card,"page_start":8,"page_end":8,"strict_locate":True}
                assert chat_render._refine_system_a_cite_locators_from_final_primary([card],stale_primary)[0]["page_start"] == 4
        assert "DIP" in result["copy_markdown"] and "[1]" in result["copy_markdown"]
