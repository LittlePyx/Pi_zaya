from __future__ import annotations

from tools.retrieval_eval.lighter_runtime_e5_small_full_build_worker import (
    publish_window_index,
    read_window_index,
)


def test_e5_window_index_atomic_publish_and_readback(tmp_path) -> None:
    root = tmp_path / "index"
    vectors = [[1.0, 0.0] + [0.0] * 382, [0.0, 1.0] + [0.0] * 382]
    windows = [
        {"parent_id": "p:1", "parent_position": 0, "window_ordinal": 0},
        {"parent_id": "p:1", "parent_position": 0, "window_ordinal": 1},
    ]
    result = publish_window_index(
        index_root=root,
        vectors=vectors,
        windows=windows,
        manifest_base={"candidate_id": "test"},
    )
    assert result["atomic_readback_passed"] is True
    assert result["window_count"] == 2
    loaded = read_window_index(root)
    assert loaded["windows"] == windows
    assert loaded["vectors"][1][1] == 1.0


def test_e5_window_index_rejects_broken_parent_ordinals(tmp_path) -> None:
    root = tmp_path / "index"
    windows = [
        {"parent_id": "p:1", "parent_position": 0, "window_ordinal": 1},
    ]
    try:
        publish_window_index(
            index_root=root,
            vectors=[[1.0] + [0.0] * 383],
            windows=windows,
            manifest_base={"candidate_id": "test"},
        )
    except RuntimeError as exc:
        assert "ordinal mapping" in str(exc)
    else:
        raise AssertionError("invalid ordinal mapping should fail readback")
