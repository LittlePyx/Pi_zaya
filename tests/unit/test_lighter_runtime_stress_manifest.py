from __future__ import annotations

from pathlib import Path

from tools.retrieval_eval.build_lighter_runtime_stress_manifest import (
    DEFAULT_PARENT_INDEX,
    STRATUM_TARGETS,
    _percentile,
    _selection_key,
)


def test_stress_manifest_quantile_math_is_deterministic() -> None:
    assert _percentile([1, 2, 3, 100], 0.5) == 2.5
    assert round(_percentile([1, 2, 3, 100], 0.95), 3) == 85.45


def test_stress_manifest_selection_hash_has_stable_tie_break() -> None:
    left = _selection_key("doc:1")
    right = _selection_key("doc:2")
    assert left == _selection_key("doc:1")
    assert left != right
    assert left[1] == "doc:1"


def test_stress_manifest_allocates_64_items_with_all_qrels() -> None:
    assert sum(STRATUM_TARGETS) == 51
    assert 13 + sum(STRATUM_TARGETS) == 64


def test_stress_manifest_parent_index_path_points_to_retained_index() -> None:
    assert DEFAULT_PARENT_INDEX == Path(
        "test_results/retrieval_cpu_full_index_feasibility_v1/index_v1"
    ).resolve()
