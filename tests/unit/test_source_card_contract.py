from __future__ import annotations

import copy

import pytest

from api.chat_render import _refresh_visible_citation_card_contracts
from api.chat_render import enrich_messages_with_reference_render
from api.source_card_contract import prepare_source_card_message
from kb.source_blocks import load_source_blocks_cached


FIRST = "The network inherits the properties of the loss function during neural network training."
SECOND = "The loss recovers the mean of the targets, including their conditional expected values."


def test_bold_statement_card_restores_literal_markup_and_every_evidence_projection(tmp_path):
    from api.source_card_contract import refresh_source_citation_cards
    raw = '**Lemma 9.** Assume independent observations of zero mean. The expected loss equals\n$$\nR(f)=Q(f)+V.\n$$'
    path = tmp_path / 'bold.md'
    path.write_text('<!-- kb_page: 3 -->\n## Theory\n\n' + raw + '\n', encoding='utf8')
    block = next(b for b in load_source_blocks_cached(path) if 'Lemma 9' in b['raw_text'])
    quote = ' '.join(raw.replace('**', '').split())
    card = {**block, 'citation_route':'system_a', 'source_path':str(path),
            'evidence_quote':quote, 'raw':quote, 'card_evidence':quote, 'binding_status':'grounded'}
    result = refresh_source_citation_cards([card], render_locale='en')[0]
    assert result['evidence_quote'] == raw
    assert result['card_evidence'] == raw
    assert next(s['text'] for s in result['card_view']['sections'] if s['id']=='evidence') == raw
    assert all(result[k] == block[k] for k in ('block_id','anchor_id','page_start','page_end','heading_path'))


@pytest.mark.parametrize('raw,quote', [
    ('**Lemma 9.** This method does **not** require paired observations for its training.',
     'Lemma 9. This method does require paired observations for its training.'),
    ('**Lemma 9.** The validated mathematical expression is $$R(f)=a**2+b**2$$.',
     'Lemma 9. The validated mathematical expression is $$R(f)=a 2+b 2$$.'),
    ('**Lemma 9.** The measurement variance is divided by the number of samples.',
     'Lemma 9. The measurement variance is multiplied by the number of samples.'),
])
def test_emphasis_restoration_cannot_drop_negation_math_or_change_relations(raw,quote):
    from kb.source_excerpt import literal_excerpt
    assert not literal_excerpt({'raw_text':raw,'kind':'equation'},quote)


def _card(tmp_path, quote=FIRST):
    path = tmp_path / "paper.en.md"
    path.write_text(
        f"# Paper\n\n<!-- kb_page: 2 -->\n\n## Theory\n\n{FIRST}\n\n{SECOND}\n",
        encoding="utf8",
    )
    blocks = [b for b in load_source_blocks_cached(path) if b["kind"] == "paragraph"]
    card = {
        "citation_route": "system_a",
        "source_path": str(path),
        "page_start": 2,
        "page_end": 2,
        "heading_path": blocks[0]["heading_path"],
        "evidence_quote": quote,
        "raw": quote,
        "card_evidence": quote,
        "answer_claim": FIRST,
        "binding_status": "grounded",
    }
    return card, blocks


def test_page_only_card_recovers_unique_source_block_without_mutating_input(tmp_path):
    card, blocks = _card(tmp_path)
    before = copy.deepcopy(card)
    result = _refresh_visible_citation_card_contracts([card], render_locale="en")[0]
    assert card == before
    assert result["block_id"] == blocks[0]["block_id"]
    assert result["anchor_id"] == blocks[0]["anchor_id"]
    assert result["strict_locate"]
    assert result["evidence_quote"] == FIRST
    assert (
        next(
            s["text"] for s in result["card_view"]["sections"] if s["id"] == "evidence"
        )
        == FIRST
    )


def test_omitted_text_is_restored_from_one_block_including_truncated_sentence(tmp_path):
    card, _ = _card(
        tmp_path, "The network inherits the properties … during neural network train..."
    )
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert result["evidence_quote"] == FIRST
    assert result["card_evidence"] == FIRST
    assert result["source_excerpt_previous_quote"] == card["evidence_quote"]


def test_compound_context_does_not_invent_one_locator_for_two_blocks(tmp_path):
    card, blocks = _card(tmp_path, FIRST + " " + SECOND)
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert result["evidence_quote"] == FIRST
    assert result["block_id"] == blocks[0]["block_id"]
    assert result["reader_evidence_quote"] == FIRST
    assert result["source_excerpt_previous_quote"] == FIRST + " " + SECOND


def test_empty_display_uses_original_source_excerpt_not_answer_claim(tmp_path):
    card, blocks = _card(tmp_path)
    card.update(
        evidence_quote="",
        card_evidence="",
        **{k: blocks[0][k] for k in ("block_id", "anchor_id")},
    )
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert result["evidence_quote"] == FIRST
    assert result["card_evidence"] == FIRST


@pytest.mark.parametrize(
    "change",
    [
        {"page_start": 3, "page_end": 3},
        {"heading_path": "Other / Theory"},
        {"block_id": "stale"},
        {"anchor_id": "p_99999"},
        {"source_path": "missing.md"},
        {"evidence_quote": FIRST.replace("inherits", "never inherits"), "raw": ""},
        {"is_inpaper": True},
        {"citation_route": "system_b"},
    ],
)
def test_invalid_or_upstream_card_cannot_gain_verified_source_location(
    tmp_path, change
):
    card, _ = _card(tmp_path)
    card.update(change)
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert not result.get("strict_locate")
    assert result.get("block_id", "") == card.get("block_id", "")


def test_ambiguous_same_page_quote_cannot_gain_a_block_id(tmp_path):
    card, _ = _card(tmp_path)
    with open(card["source_path"], "a", encoding="utf8") as handle:
        handle.write(f"\n{FIRST}\n")
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert not result.get("block_id")


def test_wrong_claim_cannot_borrow_a_partial_compound_passage(tmp_path):
    card, _ = _card(tmp_path, FIRST + " " + SECOND)
    card["answer_claim"] = (
        "The network requires clean targets and minimizes the median loss for all images."
    )
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert not result.get("block_id")


def test_equation_and_numeric_interval_are_copied_from_source_without_rewriting(
    tmp_path,
):
    quote = r"The noise range is $\sigma\in[0,55]$ and the receptive field is $(2d+1)\times(2d+1)$."
    card, blocks = _card(tmp_path, quote)
    path = tmp_path / "paper.en.md"
    path.write_text(
        path.read_text(encoding="utf8").replace(FIRST, quote), encoding="utf8"
    )
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert result["evidence_quote"] == quote
    assert result["card_evidence"] == quote


def test_source_change_does_not_reuse_old_verified_excerpt(tmp_path):
    card, _ = _card(tmp_path)
    assert _refresh_visible_citation_card_contracts([card])[0]["block_id"]
    path = tmp_path / "paper.en.md"
    path.write_text(
        path.read_text(encoding="utf8").replace(
            FIRST, FIRST.replace("inherits", "never inherits")
        ),
        encoding="utf8",
    )
    assert not _refresh_visible_citation_card_contracts([card])[0].get("block_id")


def test_literal_excerpt_can_extend_preview_without_adding_fake_ellipsis(tmp_path):
    middle = (
        "The training observations are independently drawn from the stated distribution. "
        * 7
    )
    source = FIRST + " " + middle + SECOND
    card, _ = _card(tmp_path, FIRST + " … " + SECOND)
    path = tmp_path / "paper.en.md"
    path.write_text(
        f"# Paper\n\n<!-- kb_page: 2 -->\n\n## Theory\n\n{source}\n", encoding="utf8"
    )
    result = _refresh_visible_citation_card_contracts([card])[0]
    assert len(source) > 520
    assert result["card_evidence"] == source
    assert result["evidence_quote"] == source


def test_incomplete_answer_keeps_only_independently_verified_page_only_evidence(
    tmp_path,
):
    card, _ = _card(tmp_path)
    source = card["source_path"]
    answer = "The properties of the loss function are inherited by the neural network during training [1]."
    slot = {**card, "preferred_system": "system_a", "candidate_hits": [1]}
    quality = {
        "minimum_ok": False,
        "has_hits": True,
        "incomplete_fragment": True,
        "contract_enabled": False,
        "claim_evidence": {"minimum_ok": True},
        "citation_plan": {"slots": [slot], "budget": {"system_a": 1, "system_b": 0}},
    }
    message = {
        "id": 2,
        "role": "assistant",
        "content": answer,
        "meta": {"canonical_hit_paths": [source], "answer_quality": quality},
    }
    before = copy.deepcopy(message)
    pack = {
        "render_locale": "en",
        "hits": [
            {
                "text": FIRST,
                "meta": {
                    "source_path": source,
                    "source_name": "Paper",
                    "heading_path": card["heading_path"],
                },
            }
        ],
    }
    result = enrich_messages_with_reference_render(
        [
            {"id": 1, "role": "user", "content": "What is inherited during training?"},
            message,
        ],
        {1: pack},
        conv_id="source-contract-test",
    )[-1]
    assert message == before
    assert result["meta"]["answer_quality"]["minimum_ok"] is False
    assert result["cite_details"]
    assert result["cite_details"][0]["block_id"]
    assert result["cite_details"][0]["evidence_quote"] == FIRST


@pytest.mark.parametrize(
    "change",
    [
        {"page_start": 3, "page_end": 3},
        {"heading_path": "Other / Theory"},
        {"block_id": "stale-block"},
        {"anchor_id": "p_99999"},
        {"preferred_system": "system_b"},
        {"evidence_quote": FIRST.replace("inherits", "never inherits")},
        {"evidence_quote": FIRST + " " + SECOND},
    ],
)
def test_incomplete_answer_does_not_infer_ids_from_stale_or_compound_plan(
    tmp_path, change
):
    card, _ = _card(tmp_path)
    slot = {**card, "preferred_system": "system_a", **change}
    message = {
        "role": "assistant",
        "meta": {
            "canonical_hit_paths": [card["source_path"]],
            "answer_quality": {
                "minimum_ok": False,
                "has_hits": True,
                "incomplete_fragment": True,
                "contract_enabled": False,
                "claim_evidence": {"minimum_ok": True},
                "citation_plan": {"slots": [slot]},
            },
        },
    }
    before = copy.deepcopy(message)
    assert prepare_source_card_message(message) == before
    assert message == before


@pytest.mark.parametrize(
    "quality_change",
    [
        {"has_hits": False},
        {"claim_evidence": {"minimum_ok": False}},
        {"contract_enabled": True},
        {"resolution_status": "insufficient_evidence"},
        {"incomplete_fragment": False},
    ],
)
def test_incomplete_source_recovery_cannot_override_quality_failures(
    tmp_path, quality_change
):
    card, _ = _card(tmp_path)
    message = {
        "role": "assistant",
        "meta": {
            "canonical_hit_paths": [card["source_path"]],
            "answer_quality": {
                "minimum_ok": False,
                "has_hits": True,
                "incomplete_fragment": True,
                "contract_enabled": False,
                "claim_evidence": {"minimum_ok": True},
                "citation_plan": {"slots": [card]},
                **quality_change,
            },
        },
    }
    assert prepare_source_card_message(message) == message


def _compound_message(card):
    slot = {**card, "preferred_system": "system_a", "compound_same_page_evidence": True}
    return {
        "role": "assistant",
        "content": FIRST + " [1]. " + SECOND + " [1].",
        "meta": {
            "canonical_hit_paths": [card["source_path"]],
            "answer_quality": {
                "minimum_ok": True,
                "has_hits": True,
                "claim_evidence": {"minimum_ok": True},
                "citation_plan": {
                    "slots": [slot],
                    "budget": {"system_a": 1, "system_b": 0},
                },
            },
        },
    }


def test_compound_plan_restores_separate_physical_sources_and_retains_original_context(
    tmp_path,
):
    card, blocks = _card(tmp_path, FIRST + " " + SECOND)
    message = _compound_message(card)
    before = copy.deepcopy(message)
    result = prepare_source_card_message(message)
    plan = result["meta"]["answer_quality"]["citation_plan"]
    assert message == before
    assert [s["block_id"] for s in plan["slots"]] == [b["block_id"] for b in blocks]
    assert [s["evidence_quote"] for s in plan["slots"]] == [FIRST, SECOND]
    assert all(s["strict_locate"] and s["passage_bound_claims"] for s in plan["slots"])
    assert (
        plan["source_compound_audit"]
        == before["meta"]["answer_quality"]["citation_plan"]["slots"]
    )
    assert plan["budget"] == before["meta"]["answer_quality"]["citation_plan"]["budget"]


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_text",
        "wrong_page",
        "wrong_source",
        "wrong_claim",
        "ambiguous",
        "quality_failure",
        "math_markers",
    ],
)
def test_compound_plan_does_not_convert_ambiguous_or_unsupported_context_to_locations(
    tmp_path, failure
):
    card, _ = _card(tmp_path, FIRST + " " + SECOND)
    message = _compound_message(card)
    slot = message["meta"]["answer_quality"]["citation_plan"]["slots"][0]
    if failure == "wrong_text":
        slot["evidence_quote"] = FIRST + " " + SECOND.replace("mean", "variance")
    if failure == "wrong_page":
        slot.update(page_start=3, page_end=3)
    if failure == "wrong_source":
        message["meta"]["canonical_hit_paths"] = [str(tmp_path / "other/paper.en.md")]
    if failure == "wrong_claim":
        message["content"] = (
            "The reconstruction requires clean training targets and changes the corruption distribution [1]."
        )
    if failure == "ambiguous":
        with open(card["source_path"], "a", encoding="utf8") as handle:
            handle.write(f"\n{FIRST}\n")
    if failure == "quality_failure":
        message["meta"]["answer_quality"].update(
            minimum_ok=False, incomplete_fragment=False
        )
    if failure == "math_markers":
        message["content"] = FIRST + " $x[1]$. " + SECOND + " `$x[1]$`."
    before = copy.deepcopy(message)
    assert prepare_source_card_message(message) == before
    assert message == before
