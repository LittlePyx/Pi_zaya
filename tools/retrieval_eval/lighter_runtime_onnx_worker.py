from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
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
from tools.retrieval_eval.prepare_lighter_runtime_candidate import _validate_environment


DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_qwen_protocol_v1.json"
DEFAULT_STRESS_MANIFEST = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stress_manifest_v1.json"
DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _peak_process_rss_mib() -> float:
    if sys.platform == "win32":
        size_type = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", size_type),
                ("WorkingSetSize", size_type),
                ("QuotaPeakPagedPoolUsage", size_type),
                ("QuotaPagedPoolUsage", size_type),
                ("QuotaPeakNonPagedPoolUsage", size_type),
                ("QuotaNonPagedPoolUsage", size_type),
                ("PagefileUsage", size_type),
                ("PeakPagefileUsage", size_type),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            raise OSError("peak process memory is unavailable")
        return round(float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0), 3)
    import resource

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(value / (1024.0 * 1024.0), 3)
    return round(value / 1024.0, 3)


def _pack_by_padded_token_budget(
    token_lengths: Sequence[int],
    *,
    maximum_batch_size: int,
    maximum_padded_tokens: int,
) -> list[list[int]]:
    ordered = sorted(range(len(token_lengths)), key=lambda index: (token_lengths[index], index))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in ordered:
        length = int(token_lengths[index])
        proposed_max = max(current_max, length)
        proposed_size = len(current) + 1
        if current and (
            proposed_size > maximum_batch_size
            or proposed_max * proposed_size > maximum_padded_tokens
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, length)
        if length > maximum_padded_tokens:
            batches.append(current)
            current = []
            current_max = 0
    if current:
        batches.append(current)
    return batches


def _work_summary(token_lengths: Sequence[int], batches: Sequence[Sequence[int]]) -> dict[str, Any]:
    real_tokens = sum(int(value) for value in token_lengths)
    padded_tokens = 0
    attention_work = 0
    peak_padded_tokens = 0
    oversize_singletons = 0
    for batch in batches:
        maximum = max(int(token_lengths[index]) for index in batch)
        batch_padded = maximum * len(batch)
        padded_tokens += batch_padded
        attention_work += maximum * maximum * len(batch)
        peak_padded_tokens = max(peak_padded_tokens, batch_padded)
        if len(batch) == 1 and maximum > 4096:
            oversize_singletons += 1
    return {
        "item_count": len(token_lengths),
        "batch_count": len(batches),
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
        "padding_amplification": round(padded_tokens / real_tokens, 6) if real_tokens else 0.0,
        "attention_work_proxy": attention_work,
        "peak_padded_tokens": peak_padded_tokens,
        "oversize_singleton_count": oversize_singletons,
        "maximum_observed_tokens": max(token_lengths, default=0),
    }


def _normalize_rows(values: Any, *, expected_count: int, expected_dimension: int) -> list[list[float]]:
    import numpy as np

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (expected_count, expected_dimension):
        raise RuntimeError(
            f"ONNX output shape changed: expected {(expected_count, expected_dimension)}, got {matrix.shape}"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError("ONNX output contains invalid vectors")
    matrix = matrix / norms
    return matrix.astype(np.float32, copy=False).tolist()


def _token_lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return [len(encoding.ids) for encoding in tokenizer.encode_batch(list(texts), add_special_tokens=True)]


def _encode_batch(
    *,
    tokenizer: Any,
    session: Any,
    texts: Sequence[str],
    maximum_length: int,
    padding_token: str,
    padding_token_id: int,
    output_name: str,
    dimension: int,
) -> list[list[float]]:
    import numpy as np

    tokenizer.enable_truncation(max_length=maximum_length)
    tokenizer.enable_padding(
        direction="left",
        pad_id=padding_token_id,
        pad_token=padding_token,
    )
    encodings = tokenizer.encode_batch(list(texts), add_special_tokens=True)
    input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
    attention_mask = np.asarray(
        [encoding.attention_mask for encoding in encodings], dtype=np.int64
    )
    outputs = session.run(
        [output_name], {"input_ids": input_ids, "attention_mask": attention_mask}
    )[0]
    return _normalize_rows(
        outputs,
        expected_count=len(texts),
        expected_dimension=dimension,
    )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_worker(
    *,
    semantic_db_root: Path,
    tokenizer_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    installed, environment_errors = _validate_environment()
    if environment_errors:
        raise RuntimeError("; ".join(environment_errors))
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    stress = json.loads(DEFAULT_STRESS_MANIFEST.read_text(encoding="utf-8"))
    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    candidate = protocol["candidate"]
    tokenizer_contract = protocol["tokenizer"]
    runtime = protocol["runtime"]
    measurement = protocol["measurement"]
    artifact = REPO_ROOT / candidate["artifact_relative_path"]
    tokenizer_json = tokenizer_root / "tokenizer.json"
    tokenizer_config = tokenizer_root / "tokenizer_config.json"
    source_before = _tree_fingerprint(semantic_db_root)
    artifact_before = _raw_sha256(artifact)
    if source_before != protocol["source_corpus"]["tree_sha256"]:
        raise RuntimeError("worker source corpus identity changed")
    if artifact_before != candidate["artifact_sha256"]:
        raise RuntimeError("worker artifact identity changed")
    if (
        tokenizer_json.stat().st_size != tokenizer_contract["tokenizer_json_bytes"]
        or _raw_sha256(tokenizer_json) != tokenizer_contract["tokenizer_json_sha256"]
        or tokenizer_config.stat().st_size != tokenizer_contract["tokenizer_config_bytes"]
        or _raw_sha256(tokenizer_config) != tokenizer_contract["tokenizer_config_sha256"]
    ):
        raise RuntimeError("worker tokenizer identity changed")

    all_chunks = load_all_chunks(semantic_db_root)
    chunks_by_id = {str(chunk.get("id") or ""): chunk for chunk in all_chunks}
    stress_texts: list[str] = []
    for item in stress["items"]:
        chunk_id = str(item["chunk_id"])
        text = str(chunks_by_id[chunk_id].get("text") or "")
        if _sha256_text(text) != item["text_sha256"]:
            raise RuntimeError("worker stress text identity changed")
        stress_texts.append(text)

    load_started = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(runtime["intra_op_threads"])
    options.inter_op_num_threads = int(runtime["inter_op_threads"])
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(artifact), sess_options=options, providers=[runtime["execution_provider"]]
    )
    provider_load_ms = (time.perf_counter() - load_started) * 1000.0
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("worker session provider changed")

    query_prefix = str(candidate["query_prefix"])
    warmup_text = query_prefix + str(measurement["public_warmup_text"])
    warmup_started = time.perf_counter()
    _encode_batch(
        tokenizer=tokenizer,
        session=session,
        texts=[warmup_text],
        maximum_length=int(tokenizer_contract["maximum_length"]),
        padding_token=str(tokenizer_contract["padding_token"]),
        padding_token_id=int(tokenizer_contract["padding_token_id"]),
        output_name=str(candidate["onnx_output"]),
        dimension=int(candidate["dimension"]),
    )
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    query_vectors: dict[str, list[float]] = {}
    query_timings: list[dict[str, Any]] = []
    truncated_queries = 0
    for ordinal, case in enumerate(fixture["cases"], start=1):
        query_text = query_prefix + str(case["query"])
        token_count = _token_lengths(tokenizer, [query_text])[0]
        truncated_queries += int(token_count > tokenizer_contract["maximum_length"])
        started = time.perf_counter()
        vector = _encode_batch(
            tokenizer=tokenizer,
            session=session,
            texts=[query_text],
            maximum_length=int(tokenizer_contract["maximum_length"]),
            padding_token=str(tokenizer_contract["padding_token"]),
            padding_token_id=int(tokenizer_contract["padding_token_id"]),
            output_name=str(candidate["onnx_output"]),
            dimension=int(candidate["dimension"]),
        )[0]
        duration_ms = (time.perf_counter() - started) * 1000.0
        case_id = str(case["id"])
        query_vectors[case_id] = vector
        query_timings.append(
            {
                "ordinal": ordinal,
                "case_id": case_id,
                "token_count": token_count,
                "duration_ms": round(duration_ms, 3),
            }
        )

    all_texts = [str(chunk.get("text") or "") for chunk in all_chunks]
    all_lengths = _token_lengths(tokenizer, all_texts)
    if sum(all_lengths) != int(
        protocol["source_corpus"]["total_effective_qwen_tokens_from_parent_diagnostic"]
    ):
        raise RuntimeError("worker tokenizer does not reproduce the pinned corpus token count")
    stress_lengths = _token_lengths(tokenizer, stress_texts)
    maximum_batch_size = int(measurement["maximum_batch_size"])
    maximum_padded_tokens = int(measurement["maximum_padded_tokens_per_batch"])
    full_batches = _pack_by_padded_token_budget(
        all_lengths,
        maximum_batch_size=maximum_batch_size,
        maximum_padded_tokens=maximum_padded_tokens,
    )
    stress_batches = _pack_by_padded_token_budget(
        stress_lengths,
        maximum_batch_size=maximum_batch_size,
        maximum_padded_tokens=maximum_padded_tokens,
    )
    document_vectors: dict[str, list[float]] = {}
    document_started = time.perf_counter()
    for batch in stress_batches:
        vectors = _encode_batch(
            tokenizer=tokenizer,
            session=session,
            texts=[stress_texts[index] for index in batch],
            maximum_length=int(tokenizer_contract["maximum_length"]),
            padding_token=str(tokenizer_contract["padding_token"]),
            padding_token_id=int(tokenizer_contract["padding_token_id"]),
            output_name=str(candidate["onnx_output"]),
            dimension=int(candidate["dimension"]),
        )
        for index, vector in zip(batch, vectors):
            document_vectors[str(stress["items"][index]["chunk_id"])] = vector
    document_encode_ms = (time.perf_counter() - document_started) * 1000.0
    full_work = _work_summary(all_lengths, full_batches)
    stress_work = _work_summary(stress_lengths, stress_batches)
    projected_ms = (
        document_encode_ms
        * float(full_work["attention_work_proxy"])
        / float(stress_work["attention_work_proxy"])
    )

    source_after = _tree_fingerprint(semantic_db_root)
    artifact_after = _raw_sha256(artifact)
    payload = {
        "schema_version": 1,
        "worker_kind": "retrieval_lighter_runtime_qwen_stage1_worker_v1",
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
            "vectors": query_vectors,
        },
        "documents": {
            "stress_token_counts": stress_lengths,
            "truncated_stress_count": sum(
                int(length > tokenizer_contract["maximum_length"])
                for length in stress_lengths
            ),
            "encode_ms": round(document_encode_ms, 3),
            "vectors": document_vectors,
            "stress_work": stress_work,
            "full_corpus_work": full_work,
            "projected_full_build_ms": round(projected_ms, 3),
            "projected_throughput_chunks_per_second": round(
                len(all_lengths) / (projected_ms / 1000.0), 6
            ),
        },
        "integrity": {
            "source_before_sha256": source_before,
            "source_after_sha256": source_after,
            "artifact_before_sha256": artifact_before,
            "artifact_after_sha256": artifact_after,
        },
        "execution": {
            "warmup_session_run_calls": 1,
            "query_session_run_calls": len(query_timings),
            "document_session_run_calls": len(stress_batches),
            "query_vector_count": len(query_vectors),
            "document_vector_count": len(document_vectors),
            "full_corpus_documents_embedded": 0,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
    }
    _atomic_write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated Qwen ONNX Stage-1 inference worker")
    parser.add_argument("--semantic-db-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print("[FAIL] worker output already exists")
        return 1
    try:
        payload = run_worker(
            semantic_db_root=args.semantic_db_root.expanduser().resolve(),
            tokenizer_root=args.tokenizer_root.expanduser().resolve(),
            output_path=args.output.expanduser().resolve(),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime worker: {type(exc).__name__}: {exc}")
        return 1
    print(
        "[OK] lighter runtime worker completed: "
        f"{payload['documents']['encode_ms']} ms / {payload['runtime']['peak_process_rss_mib']} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
