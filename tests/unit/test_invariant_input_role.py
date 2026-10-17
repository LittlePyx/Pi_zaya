from __future__ import annotations

import pytest

from kb.claim_evidence_runtime import audit_and_repair_claim_evidence
from kb.invariant_input_role import clarify_supported_input_role


SOURCE = (
    "In Section 4, we illustrate the optimal J-invariant denoising function. "
    "Since f has to use information from outside of each subset of dimensions J "
    "to predict the values inside of J, it cannot merely be the identity."
)
ANSWER = (
    r"For a coordinate subset $J$, a predictor $f$ is J-invariant if its output on $J$ "
    r"does not depend on the value of $x_J$ — i.e., $f(x)_J$ is computed only from "
    r"the dimensions of $x$ outside $J$ [1]."
)


def test_supported_definition_names_the_function_arguments_input_role() -> None:
    answer, audit = audit_and_repair_claim_evidence(
        ANSWER,
        [{"text": SOURCE}],
        allowed_citation_numbers={1},
        prompt="What does J-invariance require of the output on a coordinate subset J?",
    )
    assert "input value of $x_J$" in answer
    assert r"$f(x)_J$" in answer
    assert audit.get("clarified_input_roles") == 1


def test_role_clarification_is_idempotent_and_preserves_all_other_text() -> None:
    result, count = clarify_supported_input_role(ANSWER, [{"text": SOURCE}])
    assert count == 1
    assert result.replace("input value", "value") == ANSWER
    assert clarify_supported_input_role(result, [{"text": SOURCE}]) == (result, 0)


def test_other_function_argument_and_subset_use_their_own_source_relation() -> None:
    source = SOURCE.replace(" f ", " g ").replace(" J", " K")
    answer = (
        ANSWER.replace("$f$", "$g$")
        .replace("f(x)", "g(y)")
        .replace("x_J", "y_K")
        .replace("_J", "_K")
        .replace("$J$", "$K$")
        .replace("$x$", "$y$")
        .replace("J-invariant", "K-invariant")
    )
    result, count = clarify_supported_input_role(answer, [{"text": source}])
    assert count == 1 and "input value of $y_K$" in result
    assert "x_J" not in result


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("Since f has", "Since g has"),
        SOURCE.replace("inside of J", "inside of K"),
        SOURCE.replace("dimensions J", "dimensions K"),
        SOURCE.replace("has to use", "does not have to use"),
        "The function has an input and an output, with no subset dependency specified.",
    ],
)
def test_conflicting_or_incomplete_source_does_not_supply_a_role(source: str) -> None:
    assert clarify_supported_input_role(ANSWER, [{"text": source}]) == (ANSWER, 0)


@pytest.mark.parametrize(
    "answer",
    [
        ANSWER.replace("x_J", "y_J"),
        ANSWER.replace("value of $x_J$", "value of $x_K$"),
        ANSWER.replace("does not depend", "depends"),
        ANSWER.replace("[1]", ""),
        ANSWER.replace("[1]", "$[1]$"),
        "> " + ANSWER,
        "```text\n" + ANSWER + "\n```",
        "~~~text\n" + ANSWER + "\n~~~",
        "`" + ANSWER + "`",
        '"' + ANSWER + '"',
        "$$" + ANSWER.replace("[1]", "") + "$$ [1]",
    ],
)
def test_unbound_or_quoted_text_does_not_gain_a_role(answer: str) -> None:
    assert clarify_supported_input_role(answer, [{"text": SOURCE}]) == (answer, 0)


def test_only_the_cited_evidence_body_can_supply_the_relation() -> None:
    assert clarify_supported_input_role(
        ANSWER, [{"text": "Other evidence", "title": SOURCE}]
    ) == (ANSWER, 0)
    answer = ANSWER.replace("[1]", "[2]")
    assert clarify_supported_input_role(
        answer, [{"text": SOURCE}, {"text": "Other evidence"}]
    ) == (answer, 0)


def test_clarified_answer_keeps_its_source_card_locator_and_literal_quote(
    tmp_path, monkeypatch
) -> None:
    from api import chat_render
    from kb.source_blocks import load_source_blocks_cached

    source = tmp_path / "noise2self.en.md"
    source.write_text(
        "# Noise2Self\n\n<!-- kb_page: 2 -->\n\n## Introduction\n\n" + SOURCE,
        encoding="utf8",
    )
    block = next(
        b for b in load_source_blocks_cached(source) if b["kind"] == "paragraph"
    )
    slot = {
        "preferred_system": "system_a",
        "source_path": str(source),
        "source_name": "Noise2Self",
        "evidence_quote": SOURCE,
        "candidate_hits": [1],
        **{
            k: block[k]
            for k in ("block_id", "anchor_id", "heading_path", "page_start", "page_end")
        },
    }
    hit = {"text": SOURCE, "meta": {**slot, "ref_answer_citation_num": 1}}
    answer, audit = audit_and_repair_claim_evidence(
        ANSWER, [hit], allowed_citation_numbers={1}
    )
    plan = {"budget": {"system_a": 1, "system_b": 0}, "slots": [slot]}
    message = {
        "id": 2,
        "role": "assistant",
        "content": answer,
        "meta": {
            "canonical_hit_paths": [str(source)],
            "answer_quality": {
                "minimum_ok": True,
                "has_hits": True,
                "claim_evidence": audit,
                "citation_plan": plan,
            },
        },
    }
    pack = {"render_locale": "en", "hits": [hit]}
    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    monkeypatch.setattr("ui.refs_renderer._load_reference_index_cached", lambda: {})
    monkeypatch.setattr(
        chat_render, "_effective_reference_render_pack", lambda value: value
    )
    monkeypatch.setattr(
        chat_render, "_answer_aligned_reference_render_pack", lambda value, *args: value
    )
    monkeypatch.setattr(
        chat_render, "_effective_citation_render_locale", lambda _value: "en"
    )
    result = chat_render.enrich_messages_with_reference_render(
        [
            {
                "id": 1,
                "role": "user",
                "content": "Explain J-invariance and the identity predictor.",
            },
            message,
        ],
        {1: pack},
        conv_id="invariant-input-role",
    )[-1]
    assert "input value" in result["copy_markdown"]
    card = next(
        c for c in result["cite_details"] if c.get("block_id") == block["block_id"]
    )
    assert card["source_path"] == str(source)
    assert card["page_start"] == 2
    assert card["evidence_quote"] in SOURCE
    assert "input" not in card["evidence_quote"]
