from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from kb.retrieval_representation_diagnostic import analyze_chunk_representation
from tools.retrieval_eval.validate_semantic_representation_report import validate_report


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "docs" / "retrieval_semantic_representation_diagnostic_v1.json"


def _character_tokenizer(text: str, **kwargs: object) -> dict[str, list[object]]:
    max_length = int(kwargs.get("max_length") or len(text) + 2)
    offsets: list[tuple[int, int]] = [(0, 0), *[(index, index + 1) for index in range(len(text))], (0, 0)]
    if kwargs.get("truncation") and len(offsets) > max_length:
        offsets = [*offsets[: max_length - 1], (0, 0)]
    return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_chunk_representation_detects_visible_and_truncated_terms() -> None:
    visible = analyze_chunk_representation(
        text="alpha evidence omega",
        evidence_terms=["evidence"],
        tokenize=_character_tokenizer,
        max_length=30,
        passage_prefix="",
    )
    assert visible["would_truncate"] is False
    assert visible["all_evidence_inside_window"] is True

    hidden = analyze_chunk_representation(
        text="alpha evidence omega",
        evidence_terms=["omega"],
        tokenize=_character_tokenizer,
        max_length=12,
        passage_prefix="",
    )
    assert hidden["would_truncate"] is True
    assert hidden["all_evidence_inside_window"] is False

    normalized = analyze_chunk_representation(
        text="features use cross - attention here",
        evidence_terms=["cross-attention"],
        tokenize=_character_tokenizer,
        max_length=50,
        passage_prefix="",
    )
    assert normalized["resolved_term_count"] == 1
    assert normalized["all_evidence_inside_window"] is True


def test_committed_representation_diagnostic_is_valid() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert validate_report(report) == []
    assert report["population"]["label_count"] == 15
    assert report["population"]["truncated_relevant_label_count"] == 0
    assert report["focus"] == {
        "case_ids": [
            "semantic-mamba-selective-state-zh",
            "semantic-kan-edge-functions-zh",
            "semantic-timesfm-probability-zh",
            "semantic-gemma3-vision-tokens-en",
            "semantic-long-context-synthesis-zh",
        ],
        "case_count": 5,
        "label_count": 6,
        "truncated_label_count": 0,
        "evidence_outside_window_label_count": 0,
        "max_token_count": 405,
    }


def test_representation_diagnostic_rejects_rechunk_or_truncation_drift() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    changed = deepcopy(report)
    changed["decision"]["rechunk_stored_citation_units"] = True
    assert any("must not be rechunked" in error for error in validate_report(changed))

    changed = deepcopy(report)
    changed["labels"][0]["would_truncate"] = True
    assert any("no-truncation finding" in error for error in validate_report(changed))
