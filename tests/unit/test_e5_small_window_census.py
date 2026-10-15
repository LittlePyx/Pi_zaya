from __future__ import annotations

from tools.retrieval_eval.analyze_e5_small_window_census import (
    evaluate_gate,
    uncovered_token_count,
    window_spans,
)


def _gate() -> dict:
    return {
        "maximum_input_tokens_per_window": 512,
        "maximum_truncated_parent_count": 0,
        "maximum_uncovered_content_token_count": 0,
        "maximum_truncated_query_count": 0,
        "maximum_window_count_multiplier": 2.0,
        "maximum_projected_index_bytes": 10000000,
        "pass_decision": "window_census_passed_freeze_stage_1",
        "reject_decision": "e5_small_rejected_before_inference",
    }


def test_window_spans_cover_short_empty_and_long_inputs() -> None:
    assert window_spans(0, window_tokens=448, stride_tokens=384) == [(0, 0)]
    assert window_spans(448, window_tokens=448, stride_tokens=384) == [(0, 448)]
    spans = window_spans(1000, window_tokens=448, stride_tokens=384)
    assert spans == [(0, 448), (384, 832), (552, 1000)]
    assert uncovered_token_count(1000, spans) == 0
    assert min(left[1] - right[0] for left, right in zip(spans, spans[1:])) >= 64


def test_window_spans_reject_invalid_geometry() -> None:
    try:
        window_spans(10, window_tokens=0, stride_tokens=1)
    except ValueError as exc:
        assert "window and stride" in str(exc)
    else:
        raise AssertionError("invalid window geometry should fail")


def test_census_gate_passes_only_a_bounded_complete_representation() -> None:
    result = evaluate_gate(
        parent_count=100,
        window_count=150,
        maximum_input_tokens=454,
        truncated_parent_count=0,
        uncovered_content_token_count=0,
        truncated_query_count=0,
        projected_index_bytes=230400,
        gate=_gate(),
    )
    assert result["passes_all"] is True
    assert result["stage_1_allowed"] is True
    assert result["full_build_allowed"] is False


def test_census_gate_rejects_hidden_tail_loss() -> None:
    result = evaluate_gate(
        parent_count=100,
        window_count=100,
        maximum_input_tokens=454,
        truncated_parent_count=0,
        uncovered_content_token_count=1,
        truncated_query_count=0,
        projected_index_bytes=153600,
        gate=_gate(),
    )
    assert result["passes_all"] is False
    assert result["stage_1_allowed"] is False
