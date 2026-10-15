from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_MANIFEST = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stress_manifest_v1.json"
EXPECTED_MANIFEST_SHA256 = "1e697579abcdf241d943d9caffa160ed78a301d713ad7ab48e40ef02640fbd12"


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        payload.get("schema_version") != 1
        or payload.get("manifest_id") != "retrieval_lighter_runtime_stress_manifest_v1"
        or payload.get("status") != "frozen_before_first_onnx_session_run"
    ):
        errors.append("lighter runtime stress manifest identity changed")
    if payload.get("sources") != {
        "semantic_fixture_path": "docs/retrieval_semantic_paraphrase_cases_v1.json",
        "semantic_fixture_sha256": "98417031dc54af544a717d74d70938906b87c35c29232fe35fd903fffd03bc3d",
        "base_fixture_path": "docs/retrieval_eval_unseen_v2.json",
        "base_fixture_sha256": "4062ce7f0a1ce4afc86f44fbe74fdc375bbeeeb6b2bdc934344e4e0d53d749d7",
        "corpus_tree_sha256": "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf",
        "parent_index_relative_path": "test_results/retrieval_cpu_full_index_feasibility_v1/index_v1",
        "parent_index_tree_sha256": "8c8fa74b7ce72c30687933b0f41d7e6b399615cf4d3a271313c0bfb85e169b50",
        "parent_build_id": "7138dcaa15c242bd88bc14042166e879",
    }:
        errors.append("lighter runtime stress sources changed")
    selection = payload.get("selection") or {}
    if selection != {
        "seed": "pizaya-qwen-onnx-stage1-stress-v1",
        "algorithm": "all_unique_semantic_qrel_parents_then_sha256_within_five_global_character_length_rank_strata",
        "qrel_parent_count": 13,
        "length_stratum_targets": [10, 10, 10, 10, 11],
        "sample_count": 64,
        "paper_text_stored": False,
    }:
        errors.append("lighter runtime stress selection changed")
    coverage = payload.get("coverage") or {}
    if (
        coverage.get("document_count") != 10
        or coverage.get("semantic_case_count") != 12
        or coverage.get("sample_character_count") != 58834
        or coverage.get("sample_character_p95") != 2112.8
        or coverage.get("corpus_character_count") != 1488995
        or coverage.get("corpus_character_p95") != 1924.8
    ):
        errors.append("lighter runtime stress coverage changed")

    items = list(payload.get("items") or [])
    ids = [str(item.get("chunk_id") or "") for item in items]
    groups = Counter(str(item.get("selection_group") or "") for item in items)
    if (
        len(items) != 64
        or len(set(ids)) != 64
        or [item.get("sample_position") for item in items] != list(range(64))
        or groups
        != Counter(
            {
                "qrel_parent": 13,
                "length_stratum_0": 10,
                "length_stratum_1": 10,
                "length_stratum_2": 10,
                "length_stratum_3": 10,
                "length_stratum_4": 11,
            }
        )
    ):
        errors.append("lighter runtime stress item membership changed")
    for item in items:
        if (
            set(item)
            != {
                "sample_position",
                "chunk_id",
                "document_id",
                "corpus_position",
                "parent_vector_index",
                "text_sha256",
                "character_count",
                "selection_group",
                "semantic_case_ids",
            }
            or len(str(item.get("text_sha256") or "")) != 64
            or "text" in item
        ):
            errors.append("lighter runtime stress item schema or privacy boundary changed")
            break
    if payload.get("execution") != {
        "tokenizer_loaded": False,
        "onnx_session_run_calls": 0,
        "queries_encoded": 0,
        "documents_embedded": 0,
        "vectors_created": 0,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }:
        errors.append("lighter runtime stress execution boundary changed")
    return errors


def main() -> int:
    try:
        payload = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime stress manifest cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_manifest(payload)
    if sha256_file(DEFAULT_MANIFEST) != EXPECTED_MANIFEST_SHA256:
        errors.append("lighter runtime stress manifest SHA-256 changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] 64-parent stress manifest is pinned before ONNX inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
