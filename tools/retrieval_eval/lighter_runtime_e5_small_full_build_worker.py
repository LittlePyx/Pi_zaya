from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from array import array
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.store import load_all_chunks
from tools.retrieval_eval.analyze_e5_small_window_census import (
    uncovered_token_count,
    window_spans,
)
from tools.retrieval_eval.lighter_runtime_e5_small_worker import (
    _encode_id_rows,
    _query_ids,
    _verify_tokenizer_files,
    _vector_summary,
    build_document_window_ids,
)
from tools.retrieval_eval.lighter_runtime_onnx_worker import (
    _atomic_write,
    _pack_by_padded_token_budget,
    _peak_process_rss_mib,
    _raw_sha256,
    _tree_fingerprint,
    _work_summary,
)
from tools.retrieval_eval.prepare_lighter_runtime_candidate import _validate_environment


DEFAULT_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage2_e5_small_protocol_v1.json"
)
DEFAULT_STAGE0 = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage0_e5_small_v1.json"
DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"


def _write_float32_matrix(path: Path, vectors: Sequence[Sequence[float]]) -> None:
    values = array("f", (value for vector in vectors for value in vector))
    if sys.byteorder != "little":
        values.byteswap()
    with path.open("wb") as handle:
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())


def _read_float32_matrix(path: Path, *, count: int, dimension: int) -> list[tuple[float, ...]]:
    expected = count * dimension
    if path.stat().st_size != expected * 4:
        raise RuntimeError("E5 window vector file size changed")
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, expected)
    if sys.byteorder != "little":
        values.byteswap()
    return [
        tuple(float(value) for value in values[offset : offset + dimension])
        for offset in range(0, expected, dimension)
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _index_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def publish_window_index(
    *,
    index_root: Path,
    vectors: Sequence[Sequence[float]],
    windows: list[dict[str, Any]],
    manifest_base: dict[str, Any],
) -> dict[str, Any]:
    if index_root.exists():
        raise RuntimeError("E5 Stage-2 index root already exists")
    index_root.mkdir(parents=True)
    generations = index_root / "generations"
    generations.mkdir()
    build_id = uuid.uuid4().hex
    staging = generations / f".{build_id}.staging"
    generation = generations / build_id
    staging.mkdir()
    try:
        vectors_path = staging / "vectors.f32"
        windows_path = staging / "windows.json"
        _write_float32_matrix(vectors_path, vectors)
        _atomic_json(windows_path, windows)
        manifest = {
            **manifest_base,
            "schema_version": 1,
            "build_id": build_id,
            "window_count": len(windows),
            "embedding_dimension": 384,
            "vectors_sha256": _raw_sha256(vectors_path),
            "windows_sha256": _raw_sha256(windows_path),
        }
        manifest_path = staging / "manifest.json"
        _atomic_json(manifest_path, manifest)
        manifest_sha = _raw_sha256(manifest_path)
        os.replace(staging, generation)
        _atomic_write(
            index_root / "current.json",
            {"schema_version": 1, "build_id": build_id, "manifest_sha256": manifest_sha},
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_window_index(index_root, include_vectors=False)


def read_window_index(index_root: Path, *, include_vectors: bool = True) -> dict[str, Any]:
    pointer = json.loads((index_root / "current.json").read_text(encoding="utf-8"))
    build_id = str(pointer.get("build_id") or "")
    if len(build_id) != 32 or any(character not in "0123456789abcdef" for character in build_id):
        raise RuntimeError("E5 window index pointer changed")
    generation = index_root / "generations" / build_id
    manifest_path = generation / "manifest.json"
    windows_path = generation / "windows.json"
    vectors_path = generation / "vectors.f32"
    if _raw_sha256(manifest_path) != pointer.get("manifest_sha256"):
        raise RuntimeError("E5 window index manifest pointer mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    count = int(manifest["window_count"])
    dimension = int(manifest["embedding_dimension"])
    if (
        manifest.get("schema_version") != 1
        or len(windows) != count
        or _raw_sha256(windows_path) != manifest.get("windows_sha256")
        or _raw_sha256(vectors_path) != manifest.get("vectors_sha256")
    ):
        raise RuntimeError("E5 window index readback identity changed")
    expected_ordinals: dict[str, int] = {}
    for row in windows:
        parent_id = str(row.get("parent_id") or "")
        ordinal = int(row.get("window_ordinal", -1))
        if ordinal != expected_ordinals.get(parent_id, 0):
            raise RuntimeError("E5 window index parent ordinal mapping changed")
        expected_ordinals[parent_id] = ordinal + 1
    vectors = _read_float32_matrix(vectors_path, count=count, dimension=dimension)
    summary = _vector_summary(vectors)
    result = {
        "build_id": build_id,
        "manifest": manifest,
        "manifest_sha256": _raw_sha256(manifest_path),
        "window_count": count,
        "parent_count": len(expected_ordinals),
        "vector_summary": summary,
        "index_bytes": _index_size(index_root),
        "atomic_readback_passed": True,
        "windows": windows,
    }
    if include_vectors:
        result["vectors"] = vectors
    return result


def run_worker(
    *,
    semantic_db_root: Path,
    tokenizer_root: Path,
    index_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    installed, errors = _validate_environment()
    if errors:
        raise RuntimeError("; ".join(errors))
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    stage0 = json.loads(DEFAULT_STAGE0.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    candidate = protocol["candidate"]
    representation = protocol["representation"]
    artifact = REPO_ROOT / candidate["artifact_path"]
    source_before = _tree_fingerprint(semantic_db_root)
    artifact_before = _raw_sha256(artifact)
    tokenizer_before = _verify_tokenizer_files(tokenizer_root, stage0)
    if source_before != protocol["source_corpus"]["tree_sha256"]:
        raise RuntimeError("E5 Stage-2 source changed")
    if artifact_before != candidate["artifact_sha256"]:
        raise RuntimeError("E5 Stage-2 artifact changed")

    load_started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_root / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(artifact), sess_options=options, providers=["CPUExecutionProvider"]
    )
    provider_load_ms = (time.perf_counter() - load_started) * 1000.0
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("E5 Stage-2 provider changed")

    warmup_ids = _query_ids(
        tokenizer,
        str(representation["query_prefix"]) + "Pi_zaya local evidence retrieval warmup.",
    )
    _encode_id_rows(
        session=session,
        rows=[warmup_ids],
        pad_token_id=1,
        output_name="last_hidden_state",
        dimension=384,
    )

    build_started = time.perf_counter()
    chunks = load_all_chunks(semantic_db_root)
    prefix_ids = tokenizer.encode(
        str(representation["passage_prefix"]), add_special_tokens=False
    ).ids
    window_inputs: list[list[int]] = []
    window_metadata: list[dict[str, Any]] = []
    uncovered = 0
    truncated_parents = 0
    for parent_position, chunk in enumerate(chunks):
        content_ids = tokenizer.encode(
            str(chunk.get("text") or ""), add_special_tokens=False
        ).ids
        spans = window_spans(
            len(content_ids),
            window_tokens=int(representation["content_window_tokens"]),
            stride_tokens=int(representation["nominal_content_stride_tokens"]),
        )
        uncovered += uncovered_token_count(len(content_ids), spans)
        rows = build_document_window_ids(
            content_ids,
            prefix_ids=prefix_ids,
            bos_token_id=0,
            eos_token_id=2,
            window_tokens=int(representation["content_window_tokens"]),
            stride_tokens=int(representation["nominal_content_stride_tokens"]),
        )
        truncated_parents += int(any(len(row) > 512 for row in rows))
        for ordinal, row in enumerate(rows):
            window_inputs.append(row)
            window_metadata.append(
                {
                    "parent_id": str(chunk["id"]),
                    "parent_position": parent_position,
                    "window_ordinal": ordinal,
                }
            )
    if len(chunks) != 1673 or len(window_inputs) != 2016:
        raise RuntimeError("E5 Stage-2 exact parent/window count changed")
    lengths = [len(row) for row in window_inputs]
    batches = _pack_by_padded_token_budget(
        lengths, maximum_batch_size=8, maximum_padded_tokens=4096
    )
    vectors: list[list[float] | None] = [None] * len(window_inputs)
    for batch in batches:
        encoded = _encode_id_rows(
            session=session,
            rows=[window_inputs[index] for index in batch],
            pad_token_id=1,
            output_name="last_hidden_state",
            dimension=384,
        )
        for index, vector in zip(batch, encoded):
            vectors[index] = vector
    if any(vector is None for vector in vectors):
        raise RuntimeError("E5 Stage-2 vector matrix is incomplete")
    complete_vectors = [vector for vector in vectors if vector is not None]
    index_result = publish_window_index(
        index_root=index_root,
        vectors=complete_vectors,
        windows=window_metadata,
        manifest_base={
            "candidate_id": candidate["candidate_id"],
            "model_revision": candidate["model_revision"],
            "source_tree_sha256": source_before,
            "representation": "full_coverage_parent_collapsed_windows_v1",
        },
    )
    full_build_ms = (time.perf_counter() - build_started) * 1000.0

    query_vectors: dict[str, list[float]] = {}
    query_timings: list[dict[str, Any]] = []
    truncated_queries = 0
    for ordinal, case in enumerate(fixture["cases"], start=1):
        ids = _query_ids(
            tokenizer, str(representation["query_prefix"]) + str(case["query"])
        )
        truncated_queries += int(len(ids) > 512)
        started = time.perf_counter()
        vector = _encode_id_rows(
            session=session,
            rows=[ids],
            pad_token_id=1,
            output_name="last_hidden_state",
            dimension=384,
        )[0]
        query_vectors[str(case["id"])] = vector
        query_timings.append(
            {
                "ordinal": ordinal,
                "case_id": str(case["id"]),
                "token_count": len(ids),
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )

    source_after = _tree_fingerprint(semantic_db_root)
    artifact_after = _raw_sha256(artifact)
    tokenizer_after = _verify_tokenizer_files(tokenizer_root, stage0)
    payload = {
        "schema_version": 1,
        "worker_kind": "retrieval_lighter_runtime_e5_small_stage2_worker_v1",
        "runtime": {
            "provider_load_ms": round(provider_load_ms, 3),
            "full_build_ms": round(full_build_ms, 3),
            "parent_chunks_per_second": round(len(chunks) / (full_build_ms / 1000.0), 6),
            "peak_process_rss_mib": _peak_process_rss_mib(),
            "onnxruntime_version": ort.__version__,
            "session_providers": session.get_providers(),
            "installed_distribution_count": len(installed),
        },
        "representation": {
            "parent_count": len(chunks),
            "window_count": len(window_inputs),
            "truncated_parent_count": truncated_parents,
            "uncovered_content_token_count": uncovered,
            "maximum_input_tokens": max(lengths, default=0),
            "work": _work_summary(lengths, batches),
            "vector_summary": _vector_summary(complete_vectors),
        },
        "index": {key: value for key, value in index_result.items() if key not in {"windows", "vectors"}},
        "query": {
            "timings": query_timings,
            "truncated_count": truncated_queries,
            "vectors": query_vectors,
            "vector_summary": _vector_summary(list(query_vectors.values())),
        },
        "integrity": {
            "source_before_sha256": source_before,
            "source_after_sha256": source_after,
            "artifact_before_sha256": artifact_before,
            "artifact_after_sha256": artifact_after,
            "tokenizer_before_sha256": tokenizer_before,
            "tokenizer_after_sha256": tokenizer_after,
        },
        "execution": {
            "warmup_session_run_calls": 1,
            "document_session_run_calls": len(batches),
            "query_session_run_calls": len(query_timings),
            "full_build_count": 1,
            "quality_evaluations": 0,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    _atomic_write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the sole complete E5-small window index")
    parser.add_argument("--semantic-db-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.index_root.exists():
        print("[FAIL] E5 Stage-2 output or index already exists")
        return 1
    try:
        payload = run_worker(
            semantic_db_root=args.semantic_db_root.resolve(),
            tokenizer_root=args.tokenizer_root.resolve(),
            index_root=args.index_root.resolve(),
            output_path=args.output.resolve(),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5 Stage-2 worker: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"[OK] E5 Stage-2 worker: {payload['runtime']['full_build_ms']} ms / "
        f"{payload['index']['index_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
