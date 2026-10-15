from __future__ import annotations

import json
import sys

from kb.local_semantic_acceptance import (
    build_local_semantic_acceptance_report,
    load_local_observation_events,
)
from tools.observability import build_local_semantic_acceptance_report as report_tool


def _event(name: str, attributes: dict, *, schema_version: int = 2) -> dict:
    return {
        "schema_version": schema_version,
        "name": name,
        "trace_id": "private-trace-id",
        "span_id": "private-span-id",
        "attributes": attributes,
    }


def test_acceptance_report_aggregates_lifecycle_actual_modes_and_warm_latency() -> None:
    events = [
        _event(
            "pi_zaya.local_semantic_component",
            {
                "outcome_id": "local_semantic_component.started",
                "lifecycle_operation_id": "enable",
                "consented": True,
            },
        ),
        _event(
            "pi_zaya.local_semantic_component",
            {
                "outcome_id": "local_semantic_component.ready",
                "lifecycle_operation_id": "enable",
                "bm25_available": True,
            },
        ),
        _event(
            "pi_zaya.local_semantic_component",
            {
                "outcome_id": "local_semantic_component.needs_space",
                "lifecycle_operation_id": "resume",
                "bm25_available": True,
            },
        ),
        _event(
            "pi_zaya.retrieval",
            {
                "requested_retrieval_mode_id": "hybrid_local",
                "used_retrieval_mode_id": "hybrid_local",
                "automatic_fallback": False,
                "cold_start": False,
                "index_cache_hit": True,
                "duration_ms": 10.0,
            },
        ),
        _event(
            "pi_zaya.retrieval",
            {
                "requested_retrieval_mode_id": "hybrid_local",
                "used_retrieval_mode_id": "hybrid_local",
                "automatic_fallback": False,
                "cold_start": False,
                "index_cache_hit": True,
                "duration_ms": 20.0,
            },
        ),
        _event(
            "pi_zaya.retrieval",
            {
                "requested_retrieval_mode_id": "hybrid_local",
                "used_retrieval_mode_id": "bm25",
                "automatic_fallback": True,
                "fallback_reason_id": "component_lifecycle_not_ready",
                "query": "private paper question",
            },
        ),
    ]

    report = build_local_semantic_acceptance_report(events)

    assert report["lifecycle"]["observed_opt_in_completion_rate"] == 1.0
    assert report["lifecycle"]["automatic_containment_rate"] == 1.0
    retrieval = report["retrieval"]
    assert retrieval["hybrid_request_count"] == 3
    assert retrieval["actual_hybrid_use_count"] == 2
    assert retrieval["actual_hybrid_use_rate"] == 0.6667
    assert retrieval["automatic_fallback_rate"] == 0.3333
    assert retrieval["automatic_fallback_category_counts"] == {
        "component_runtime": 1
    }
    assert retrieval["warm_hybrid_latency_ms"] == {
        "sample_count": 2,
        "p50": 15.0,
        "p95": 19.5,
        "max": 20.0,
    }
    rendered = json.dumps(report, ensure_ascii=False)
    assert "private-trace-id" not in rendered
    assert "private-span-id" not in rendered
    assert "private paper question" not in rendered
    assert "component_lifecycle_not_ready" not in rendered


def test_acceptance_report_marks_empty_samples_unavailable_and_ignores_v1() -> None:
    report = build_local_semantic_acceptance_report(
        [
            _event("pi_zaya.retrieval", {}, schema_version=1),
            {"schema_version": "private-invalid", "name": "pi_zaya.retrieval"},
            _event(
                "pi_zaya.local_semantic_component",
                {
                    "outcome_id": "local_semantic_component.started",
                    "lifecycle_operation_id": "private_operation",
                },
            ),
        ]
    )

    assert report["window"]["accepted_event_count"] == 1
    assert report["window"]["ignored_event_count"] == 2
    assert report["lifecycle"]["operation_started_counts"] == {
        "unknown_operation": 1
    }
    assert report["lifecycle"]["observed_opt_in_completion_rate"] is None
    assert report["retrieval"]["actual_hybrid_use_rate"] is None
    assert report["retrieval"]["warm_hybrid_latency_ms"]["p95"] is None
    assert report["sample_sufficiency"] == {
        "opt_in_completion_available": False,
        "actual_mode_rate_available": False,
        "warm_latency_available": False,
    }
    assert "private" not in json.dumps(report)


def test_observation_loader_counts_malformed_lines_without_copying_them(tmp_path) -> None:
    path = tmp_path / "observations.jsonl"
    path.write_text(
        json.dumps(_event("pi_zaya.retrieval", {})) + "\nnot-json\n[]\n",
        encoding="utf-8",
    )

    events, line_count = load_local_observation_events(path)
    report = build_local_semantic_acceptance_report(
        events,
        input_line_count=line_count,
    )

    assert line_count == 3
    assert len(events) == 1
    assert report["window"]["input_line_count"] == 3
    assert report["window"]["ignored_event_count"] == 2


def test_acceptance_report_cli_writes_only_aggregate_output(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "observations.jsonl"
    output = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            _event(
                "pi_zaya.retrieval",
                {
                    "requested_retrieval_mode_id": "hybrid_local",
                    "used_retrieval_mode_id": "bm25",
                    "automatic_fallback": True,
                    "fallback_reason_id": "dense_query_failed",
                    "query": "private canary query",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(report_tool.__file__), str(source), "--output", str(output)],
    )

    assert report_tool.main() == 0

    rendered = output.read_text(encoding="utf-8")
    payload = json.loads(rendered)
    assert payload["retrieval"]["automatic_fallback_category_counts"] == {
        "dense_query": 1
    }
    assert "private canary query" not in rendered
    assert "private-trace-id" not in rendered
