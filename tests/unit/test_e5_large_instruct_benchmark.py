from __future__ import annotations

import json

from tools.retrieval_eval.run_e5_large_instruct_benchmark import (
    DEFAULT_FIXTURE,
    DEFAULT_REPORT,
    _report_errors,
    _sha256_binary,
    evaluate_development_gates,
)


def test_binary_weight_hash_does_not_normalize_newlines(tmp_path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"header\r\npayload\rbytes")
    assert _sha256_binary(weight) == "ecfd088c7fe740f2d562e1dfa8e73fbd3ff9a2bffb57aeb6921d91f4410f01c4"


def _channel(recall5, recall10, mrr, ndcg10, wrong5, p95):
    mode = {
        "metrics": {
            "recall@5": recall5,
            "recall@10": recall10,
            "mrr": mrr,
            "ndcg@10": ndcg10,
            "wrong_source_rate@5": wrong5,
        },
        "latency_ms": {"p95": p95},
    }
    return {"modes": {"scoped": mode, "full_library": mode}}


def test_large_candidate_gates_require_both_scope_safe_gain() -> None:
    passing = evaluate_development_gates(
        base_dense=_channel(0.4, 0.5, 0.2, 0.25, 0.3, 30.0),
        large_dense=_channel(0.5, 0.56, 0.25, 0.3, 0.25, 60.0),
        safe_rrf=_channel(0.6, 0.66, 0.4, 0.45, 0.0, 70.0),
        product_baseline=_channel(0.6, 0.62, 0.39, 0.44, 0.0, 2.0),
        index_size_ratio=1.34,
    )
    assert passing["passes_all"] is True

    no_safe_gain = evaluate_development_gates(
        base_dense=_channel(0.4, 0.5, 0.2, 0.25, 0.3, 30.0),
        large_dense=_channel(0.5, 0.56, 0.25, 0.3, 0.25, 60.0),
        safe_rrf=_channel(0.6, 0.62, 0.4, 0.45, 0.0, 70.0),
        product_baseline=_channel(0.6, 0.62, 0.39, 0.44, 0.0, 2.0),
        index_size_ratio=1.34,
    )
    assert no_safe_gain["passes_all"] is False
    assert no_safe_gain["checks"]["safe_recall_at_10_gain_both_modes"] is False


def test_measured_development_report_is_pinned_and_rejected(tmp_path) -> None:
    assert _report_errors(DEFAULT_REPORT, DEFAULT_FIXTURE) == []
    changed = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    changed["channels"]["large_instruct_e5_single_vector"]["modes"]["scoped"][
        "metrics"
    ]["recall@10"] = 1.0
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    errors = _report_errors(changed_path, DEFAULT_FIXTURE)
    assert any("measured recall@10 changed" in error for error in errors)
    assert any("development gate arithmetic changed" in error for error in errors)
