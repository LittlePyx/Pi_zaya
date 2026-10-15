from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import site
import sys
import time
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
    REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_e5_small_protocol_v1.json"
)
DEFAULT_STAGE0 = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage0_e5_small_v1.json"
DEFAULT_STRESS_MANIFEST = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stress_manifest_v1.json"
DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_document_window_ids(
    content_ids: Sequence[int],
    *,
    prefix_ids: Sequence[int],
    bos_token_id: int,
    eos_token_id: int,
    window_tokens: int,
    stride_tokens: int,
) -> list[list[int]]:
    spans = window_spans(
        len(content_ids), window_tokens=window_tokens, stride_tokens=stride_tokens
    )
    if uncovered_token_count(len(content_ids), spans):
        raise RuntimeError("document window construction left uncovered content")
    return [
        [bos_token_id, *prefix_ids, *content_ids[start:end], eos_token_id]
        for start, end in spans
    ]


def _masked_mean_normalize(
    hidden_state: Any,
    attention_mask: Any,
    *,
    expected_dimension: int,
) -> list[list[float]]:
    import numpy as np

    hidden = np.asarray(hidden_state, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
        raise RuntimeError("E5 hidden-state or attention-mask shape changed")
    if hidden.shape[2] != expected_dimension:
        raise RuntimeError("E5 hidden-state dimension changed")
    denominator = np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled = (hidden * mask[:, :, None]).sum(axis=1) / denominator
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if (
        not np.isfinite(hidden).all()
        or not np.isfinite(pooled).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise RuntimeError("E5 output contains invalid vectors")
    return (pooled / norms).astype(np.float32, copy=False).tolist()


def _encode_id_rows(
    *,
    session: Any,
    rows: Sequence[Sequence[int]],
    pad_token_id: int,
    output_name: str,
    dimension: int,
) -> list[list[float]]:
    import numpy as np

    if not rows:
        return []
    maximum = max(len(row) for row in rows)
    input_ids = np.full((len(rows), maximum), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(rows), maximum), dtype=np.int64)
    token_type_ids = np.zeros((len(rows), maximum), dtype=np.int64)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = row
        attention_mask[index, : len(row)] = 1
    hidden = session.run(
        [output_name],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]
    return _masked_mean_normalize(
        hidden, attention_mask, expected_dimension=dimension
    )


def _vector_summary(vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError("E5 vectors must be a matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all():
        raise RuntimeError("E5 vectors contain non-finite values")
    digest = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    return {
        "count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "minimum_norm": round(float(norms.min()), 8) if len(norms) else 0.0,
        "maximum_norm": round(float(norms.max()), 8) if len(norms) else 0.0,
        "matrix_sha256": digest,
    }


def _verify_tokenizer_files(tokenizer_root: Path, stage0: dict[str, Any]) -> str:
    expected = list((stage0.get("tokenizer") or {}).get("files") or [])
    if len(expected) != 5:
        raise RuntimeError("E5 Stage-0 tokenizer file set changed")
    for row in expected:
        path = tokenizer_root / str(row["local_name"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or _raw_sha256(path) != str(row["sha256"])
        ):
            raise RuntimeError(f"E5 tokenizer file changed: {path.name}")
    return _tree_fingerprint(tokenizer_root)


def _query_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=True).ids)


def run_worker(
    *, semantic_db_root: Path, tokenizer_root: Path, output_path: Path
) -> dict[str, Any]:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    installed, environment_errors = _validate_environment()
    if environment_errors:
        raise RuntimeError("; ".join(environment_errors))
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    stage0 = json.loads(DEFAULT_STAGE0.read_text(encoding="utf-8"))
    stress = json.loads(DEFAULT_STRESS_MANIFEST.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    candidate = protocol["candidate"]
    tokenizer_contract = protocol["tokenizer"]
    encoding = protocol["encoding"]
    runtime = protocol["runtime"]
    measurement = protocol["measurement"]
    artifact = REPO_ROOT / candidate["artifact_path"]

    source_before = _tree_fingerprint(semantic_db_root)
    artifact_before = _raw_sha256(artifact)
    tokenizer_before = _verify_tokenizer_files(tokenizer_root, stage0)
    if source_before != protocol["source_corpus"]["tree_sha256"]:
        raise RuntimeError("E5 worker source corpus identity changed")
    if artifact_before != candidate["artifact_sha256"]:
        raise RuntimeError("E5 worker artifact identity changed")

    all_chunks = load_all_chunks(semantic_db_root)
    if len(all_chunks) != int(measurement["full_parent_count"]):
        raise RuntimeError("E5 worker parent count changed")
    chunks_by_id = {str(chunk.get("id") or ""): chunk for chunk in all_chunks}
    for item in stress["items"]:
        text = str(chunks_by_id[str(item["chunk_id"])].get("text") or "")
        if _sha256_text(text) != item["text_sha256"]:
            raise RuntimeError("E5 worker stress text identity changed")

    load_started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_root / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(runtime["intra_op_threads"])
    options.inter_op_num_threads = int(runtime["inter_op_threads"])
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(artifact),
        sess_options=options,
        providers=[str(runtime["execution_provider"])],
    )
    provider_load_ms = (time.perf_counter() - load_started) * 1000.0
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("E5 worker session provider changed")

    passage_prefix_ids = tokenizer.encode(
        str(encoding["passage_prefix"]), add_special_tokens=False
    ).ids
    all_windows: list[dict[str, Any]] = []
    truncated_parents = 0
    uncovered_tokens = 0
    for parent_position, chunk in enumerate(all_chunks):
        content_ids = tokenizer.encode(
            str(chunk.get("text") or ""), add_special_tokens=False
        ).ids
        spans = window_spans(
            len(content_ids),
            window_tokens=int(encoding["content_window_tokens"]),
            stride_tokens=int(encoding["nominal_content_stride_tokens"]),
        )
        uncovered_tokens += uncovered_token_count(len(content_ids), spans)
        for ordinal, ids in enumerate(
            build_document_window_ids(
                content_ids,
                prefix_ids=passage_prefix_ids,
                bos_token_id=int(tokenizer_contract["bos_token_id"]),
                eos_token_id=int(tokenizer_contract["eos_token_id"]),
                window_tokens=int(encoding["content_window_tokens"]),
                stride_tokens=int(encoding["nominal_content_stride_tokens"]),
            )
        ):
            if len(ids) > int(tokenizer_contract["maximum_model_tokens"]):
                truncated_parents += 1
            all_windows.append(
                {
                    "parent_id": str(chunk["id"]),
                    "parent_position": parent_position,
                    "ordinal": ordinal,
                    "ids": ids,
                }
            )
    if len(all_windows) != int(measurement["full_window_count"]):
        raise RuntimeError("E5 worker full window count changed")
    stress_ids = {str(item["chunk_id"]) for item in stress["items"]}
    stress_windows = [row for row in all_windows if row["parent_id"] in stress_ids]
    if len(stress_windows) != int(measurement["stress_window_count"]):
        raise RuntimeError("E5 worker stress window count changed")

    warmup_ids = _query_ids(
        tokenizer,
        str(encoding["query_prefix"]) + str(measurement["public_warmup_text"]),
    )
    warmup_started = time.perf_counter()
    _encode_id_rows(
        session=session,
        rows=[warmup_ids],
        pad_token_id=int(tokenizer_contract["pad_token_id"]),
        output_name=str(candidate["onnx_output"]),
        dimension=int(candidate["embedding_dimension"]),
    )
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    query_vectors: list[list[float]] = []
    query_timings: list[dict[str, Any]] = []
    truncated_queries = 0
    for ordinal, case in enumerate(fixture["cases"], start=1):
        ids = _query_ids(
            tokenizer, str(encoding["query_prefix"]) + str(case["query"])
        )
        truncated_queries += int(len(ids) > int(tokenizer_contract["maximum_model_tokens"]))
        started = time.perf_counter()
        vector = _encode_id_rows(
            session=session,
            rows=[ids],
            pad_token_id=int(tokenizer_contract["pad_token_id"]),
            output_name=str(candidate["onnx_output"]),
            dimension=int(candidate["embedding_dimension"]),
        )[0]
        query_vectors.append(vector)
        query_timings.append(
            {
                "ordinal": ordinal,
                "case_id": str(case["id"]),
                "token_count": len(ids),
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )

    maximum_batch_size = int(measurement["maximum_batch_size"])
    maximum_padded_tokens = int(measurement["maximum_padded_tokens_per_batch"])
    full_lengths = [len(row["ids"]) for row in all_windows]
    stress_lengths = [len(row["ids"]) for row in stress_windows]
    full_batches = _pack_by_padded_token_budget(
        full_lengths,
        maximum_batch_size=maximum_batch_size,
        maximum_padded_tokens=maximum_padded_tokens,
    )
    stress_batches = _pack_by_padded_token_budget(
        stress_lengths,
        maximum_batch_size=maximum_batch_size,
        maximum_padded_tokens=maximum_padded_tokens,
    )
    stress_vectors: list[list[float]] = []
    document_started = time.perf_counter()
    for batch in stress_batches:
        stress_vectors.extend(
            _encode_id_rows(
                session=session,
                rows=[stress_windows[index]["ids"] for index in batch],
                pad_token_id=int(tokenizer_contract["pad_token_id"]),
                output_name=str(candidate["onnx_output"]),
                dimension=int(candidate["embedding_dimension"]),
            )
        )
    document_encode_ms = (time.perf_counter() - document_started) * 1000.0
    full_work = _work_summary(full_lengths, full_batches)
    stress_work = _work_summary(stress_lengths, stress_batches)
    projected_ms = document_encode_ms * (
        float(full_work["attention_work_proxy"])
        / float(stress_work["attention_work_proxy"])
    )

    source_after = _tree_fingerprint(semantic_db_root)
    artifact_after = _raw_sha256(artifact)
    tokenizer_after = _verify_tokenizer_files(tokenizer_root, stage0)
    payload = {
        "schema_version": 1,
        "worker_kind": "retrieval_lighter_runtime_e5_small_stage1_worker_v1",
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "user_site_enabled": bool(site.ENABLE_USER_SITE),
            "onnxruntime_version": ort.__version__,
            "session_providers": session.get_providers(),
            "provider_load_ms": round(provider_load_ms, 3),
            "warmup_ms": round(warmup_ms, 3),
            "peak_process_rss_mib": _peak_process_rss_mib(),
            "installed_distribution_count": len(installed),
        },
        "query": {
            "timings": query_timings,
            "truncated_count": truncated_queries,
            "vectors": _vector_summary(query_vectors),
        },
        "documents": {
            "parent_count": len(all_chunks),
            "full_window_count": len(all_windows),
            "stress_parent_count": len(stress_ids),
            "stress_window_count": len(stress_windows),
            "truncated_parent_count": truncated_parents,
            "uncovered_content_token_count": uncovered_tokens,
            "maximum_input_tokens": max(full_lengths, default=0),
            "encode_ms": round(document_encode_ms, 3),
            "vectors": _vector_summary(stress_vectors),
            "stress_work": stress_work,
            "full_corpus_work": full_work,
            "projected_full_build_ms": round(projected_ms, 3),
            "projected_parent_chunks_per_second": round(
                len(all_chunks) / (projected_ms / 1000.0), 6
            ),
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
            "query_session_run_calls": len(query_timings),
            "document_session_run_calls": len(stress_batches),
            "query_vector_count": len(query_vectors),
            "stress_window_vector_count": len(stress_vectors),
            "full_corpus_windows_embedded": 0,
            "quality_evaluations": 0,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    _atomic_write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated E5-small ONNX Stage-1 worker")
    parser.add_argument("--semantic-db-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print("[FAIL] E5-small worker output already exists")
        return 1
    try:
        payload = run_worker(
            semantic_db_root=args.semantic_db_root.resolve(),
            tokenizer_root=args.tokenizer_root.resolve(),
            output_path=args.output.resolve(),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5-small worker: {type(exc).__name__}: {exc}")
        return 1
    print(
        "[OK] E5-small worker completed: "
        f"{payload['documents']['encode_ms']} ms / "
        f"{payload['runtime']['peak_process_rss_mib']} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
