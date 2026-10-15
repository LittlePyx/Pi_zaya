from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from kb.store import load_all_chunks
from tools.retrieval_eval.run_cpu_full_index_feasibility import _tree_fingerprint


DEFAULT_SEMANTIC_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_BASE_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"
DEFAULT_PARENT_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_cpu_full_index_feasibility_v1"
    / "index_v1"
)
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stress_manifest_v1.json"
EXPECTED_CORPUS_TREE = "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf"
EXPECTED_PARENT_INDEX_TREE = "8c8fa74b7ce72c30687933b0f41d7e6b399615cf4d3a271313c0bfb85e169b50"
SELECTION_SEED = "pizaya-qwen-onnx-stage1-stress-v1"
STRATUM_TARGETS = (10, 10, 10, 10, 11)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _selection_key(chunk_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SELECTION_SEED}|{chunk_id}".encode()).hexdigest()
    return digest, chunk_id


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_parent_manifest(index_root: Path) -> dict[str, Any]:
    pointer = json.loads((index_root / "current.json").read_text(encoding="utf-8"))
    build_id = str(pointer.get("build_id") or "")
    manifest_path = index_root / "generations" / build_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("build_id") != build_id:
        raise RuntimeError("parent index pointer and manifest disagree")
    return manifest


def build_manifest(*, db_root: Path, index_root: Path) -> dict[str, Any]:
    if _tree_fingerprint(db_root) != EXPECTED_CORPUS_TREE:
        raise RuntimeError("stress source corpus identity changed")
    if _tree_fingerprint(index_root) != EXPECTED_PARENT_INDEX_TREE:
        raise RuntimeError("stress parent index identity changed")
    semantic = json.loads(DEFAULT_SEMANTIC_FIXTURE.read_text(encoding="utf-8"))
    base = json.loads(DEFAULT_BASE_FIXTURE.read_text(encoding="utf-8"))
    base_cases = {str(case["id"]): case for case in base["cases"]}
    chunks = load_all_chunks(db_root)
    if len(chunks) != 1673:
        raise RuntimeError("stress source corpus chunk count changed")
    by_id = {str(chunk.get("id") or ""): chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise RuntimeError("stress source corpus contains duplicate chunk ids")

    qrel_cases: dict[str, set[str]] = {}
    for semantic_case in semantic["cases"]:
        base_case_id = str(semantic_case["base_case_id"])
        for chunk_id in (base_cases[base_case_id].get("relevance") or {}):
            if chunk_id not in by_id:
                raise RuntimeError("semantic qrel chunk is absent from the corpus")
            qrel_cases.setdefault(chunk_id, set()).add(str(semantic_case["id"]))
    if len(qrel_cases) != 13:
        raise RuntimeError("semantic qrel parent count changed")

    parent = _load_parent_manifest(index_root)
    parent_positions = {
        str(row.get("chunk_id") or ""): index for index, row in enumerate(parent["chunks"])
    }
    if set(parent_positions) != set(by_id):
        raise RuntimeError("parent index chunk ids no longer match the stress corpus")
    for chunk_id, position in parent_positions.items():
        text_sha = _sha256_text(str(by_id[chunk_id].get("text") or ""))
        if parent["chunks"][position].get("text_sha256") != text_sha:
            raise RuntimeError("parent index text identity no longer matches the stress corpus")

    length_order = sorted(by_id, key=lambda chunk_id: (len(str(by_id[chunk_id].get("text") or "")), chunk_id))
    stratum_by_id = {
        chunk_id: min(4, position * 5 // len(length_order))
        for position, chunk_id in enumerate(length_order)
    }
    selected_groups: dict[str, str] = {chunk_id: "qrel_parent" for chunk_id in qrel_cases}
    for stratum, target in enumerate(STRATUM_TARGETS):
        eligible = [
            chunk_id
            for chunk_id in by_id
            if chunk_id not in selected_groups and stratum_by_id[chunk_id] == stratum
        ]
        eligible.sort(key=_selection_key)
        if len(eligible) < target:
            raise RuntimeError(f"stress stratum {stratum} does not contain {target} eligible chunks")
        for chunk_id in eligible[:target]:
            selected_groups[chunk_id] = f"length_stratum_{stratum}"
    if len(selected_groups) != 64:
        raise RuntimeError("stress sample count changed")

    group_order = {"qrel_parent": 0, **{f"length_stratum_{index}": index + 1 for index in range(5)}}
    selected_ids = sorted(
        selected_groups,
        key=lambda chunk_id: (group_order[selected_groups[chunk_id]], _selection_key(chunk_id)),
    )
    items: list[dict[str, Any]] = []
    for sample_position, chunk_id in enumerate(selected_ids):
        text = str(by_id[chunk_id].get("text") or "")
        items.append(
            {
                "sample_position": sample_position,
                "chunk_id": chunk_id,
                "document_id": chunk_id.split(":", 1)[0],
                "corpus_position": next(
                    index for index, chunk in enumerate(chunks) if str(chunk.get("id") or "") == chunk_id
                ),
                "parent_vector_index": parent_positions[chunk_id],
                "text_sha256": _sha256_text(text),
                "character_count": len(text),
                "selection_group": selected_groups[chunk_id],
                "semantic_case_ids": sorted(qrel_cases.get(chunk_id, set())),
            }
        )
    lengths = [item["character_count"] for item in items]
    corpus_lengths = [len(str(chunk.get("text") or "")) for chunk in chunks]
    document_counts: dict[str, int] = {}
    for item in items:
        document_counts[item["document_id"]] = document_counts.get(item["document_id"], 0) + 1
    return {
        "schema_version": 1,
        "manifest_id": "retrieval_lighter_runtime_stress_manifest_v1",
        "status": "frozen_before_first_onnx_session_run",
        "purpose": "Deterministic representative and evidence-bearing parent chunks for bounded CPU ONNX speed and vector-parity measurement; contains no paper text.",
        "sources": {
            "semantic_fixture_path": "docs/retrieval_semantic_paraphrase_cases_v1.json",
            "semantic_fixture_sha256": sha256_file(DEFAULT_SEMANTIC_FIXTURE),
            "base_fixture_path": "docs/retrieval_eval_unseen_v2.json",
            "base_fixture_sha256": sha256_file(DEFAULT_BASE_FIXTURE),
            "corpus_tree_sha256": EXPECTED_CORPUS_TREE,
            "parent_index_relative_path": "test_results/retrieval_cpu_full_index_feasibility_v1/index_v1",
            "parent_index_tree_sha256": EXPECTED_PARENT_INDEX_TREE,
            "parent_build_id": parent["build_id"],
        },
        "selection": {
            "seed": SELECTION_SEED,
            "algorithm": "all_unique_semantic_qrel_parents_then_sha256_within_five_global_character_length_rank_strata",
            "qrel_parent_count": len(qrel_cases),
            "length_stratum_targets": list(STRATUM_TARGETS),
            "sample_count": len(items),
            "paper_text_stored": False,
        },
        "coverage": {
            "document_count": len(document_counts),
            "document_sample_counts": dict(sorted(document_counts.items())),
            "semantic_case_count": len(semantic["cases"]),
            "sample_character_count": sum(lengths),
            "sample_character_min": min(lengths),
            "sample_character_p50": statistics.median(lengths),
            "sample_character_p95": round(_percentile(lengths, 0.95), 3),
            "sample_character_max": max(lengths),
            "corpus_character_count": sum(corpus_lengths),
            "corpus_character_p50": statistics.median(corpus_lengths),
            "corpus_character_p95": round(_percentile(corpus_lengths, 0.95), 3),
            "corpus_character_max": max(corpus_lengths),
        },
        "items": items,
        "execution": {
            "tokenizer_loaded": False,
            "onnx_session_run_calls": 0,
            "queries_encoded": 0,
            "documents_embedded": 0,
            "vectors_created": 0,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the bounded ONNX stress sample before inference")
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_PARENT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        print("[FAIL] stress manifest already exists; regeneration is not allowed")
        return 1
    try:
        payload = build_manifest(
            db_root=args.db_root.expanduser().resolve(),
            index_root=args.index_root.expanduser().resolve(),
        )
        _atomic_write(args.output, payload)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] stress manifest: {type(exc).__name__}: {exc}")
        return 1
    print(f"[OK] froze {len(payload['items'])} stress parents before ONNX inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
