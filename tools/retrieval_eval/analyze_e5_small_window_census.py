from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.store import load_all_chunks
from tools.retrieval_eval.validate_e5_small_window_census_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stress_manifest_v1.json"
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "test_results"
    / "retrieval_lighter_runtime_v1"
    / "artifacts"
    / "multilingual_e5_small_official_onnx_v1"
    / "tokenizer"
    / "tokenizer.json"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_e5_small_window_census_v1.json"
EXPECTED_FIXTURE_SHA256 = "98417031dc54af544a717d74d70938906b87c35c29232fe35fd903fffd03bc3d"
EXPECTED_REPORT_SHA256: str | None = (
    "bd32ae3d12bad8d67203f9493da33f2ab0b57ac8ae4616992efdf4fb9745ee25"
)
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "9ff2c8f4dc9f81ade0cbae5286ade071a1d0e8fcfeaecc9cae6ba6f33b33c3cf"
)


def _normalized_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def window_spans(
    content_token_count: int, *, window_tokens: int, stride_tokens: int
) -> list[tuple[int, int]]:
    count = max(0, int(content_token_count))
    window = int(window_tokens)
    stride = int(stride_tokens)
    if window <= 0 or stride <= 0 or stride > window:
        raise ValueError("window and stride must be positive with stride <= window")
    if count <= window:
        return [(0, count)]
    starts = [0]
    while starts[-1] + window < count:
        next_start = min(starts[-1] + stride, count - window)
        if next_start <= starts[-1]:
            raise RuntimeError("window construction failed to advance")
        starts.append(next_start)
    return [(start, min(count, start + window)) for start in starts]


def uncovered_token_count(content_token_count: int, spans: list[tuple[int, int]]) -> int:
    count = max(0, int(content_token_count))
    if count == 0:
        return 0
    cursor = 0
    uncovered = 0
    for start, end in sorted(spans):
        if start > cursor:
            uncovered += start - cursor
        cursor = max(cursor, end)
    return uncovered + max(0, count - cursor)


def _distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values, default=0),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values, default=0),
        "sum": sum(values),
    }


def evaluate_gate(
    *,
    parent_count: int,
    window_count: int,
    maximum_input_tokens: int,
    truncated_parent_count: int,
    uncovered_content_token_count: int,
    truncated_query_count: int,
    projected_index_bytes: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    multiplier = window_count / parent_count if parent_count else float("inf")
    checks = {
        "input_length": maximum_input_tokens
        <= int(gate["maximum_input_tokens_per_window"]),
        "parent_truncation": truncated_parent_count
        <= int(gate["maximum_truncated_parent_count"]),
        "content_coverage": uncovered_content_token_count
        <= int(gate["maximum_uncovered_content_token_count"]),
        "query_truncation": truncated_query_count
        <= int(gate["maximum_truncated_query_count"]),
        "window_multiplier": multiplier <= float(gate["maximum_window_count_multiplier"]),
        "projected_index_bytes": projected_index_bytes
        <= int(gate["maximum_projected_index_bytes"]),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "passes_all": passed,
        "status": gate["pass_decision"] if passed else gate["reject_decision"],
        "stage_1_allowed": passed,
        "full_build_allowed": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }


def _result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "tokenizer": report.get("tokenizer"),
        "workload": report.get("workload"),
        "coverage": report.get("coverage"),
        "decision": report.get("decision"),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_report(
    report: dict[str, Any], *, expected_result_fingerprint: str | None = None
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_e5_small_window_census_v1"
    ):
        errors.append("E5-small window-census report identity changed")
    if report.get("protocol_sha256") != _normalized_sha256(DEFAULT_PROTOCOL):
        errors.append("E5-small window-census protocol identity changed")
    source = report.get("source_corpus") or {}
    if (
        source.get("tree_sha256_before")
        != "deb60a97bdfc50b0273de07ebe1c775352ffb2c5e4adeb3e925d48f7f25cafaf"
        or source.get("tree_sha256_after") != source.get("tree_sha256_before")
        or source.get("document_count") != 10
        or source.get("parent_chunk_count") != 1673
    ):
        errors.append("E5-small window-census source evidence changed")
    tokenizer = report.get("tokenizer") or {}
    if (
        tokenizer.get("tokenizer_json_sha256")
        != "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
        or tokenizer.get("tokenizers_version") != "0.21.4"
        or tokenizer.get("bos_token_id") != 0
        or tokenizer.get("pad_token_id") != 1
        or tokenizer.get("eos_token_id") != 2
        or tokenizer.get("maximum_model_tokens") != 512
    ):
        errors.append("E5-small window-census tokenizer evidence changed")
    workload = report.get("workload") or {}
    parent_count = int(source.get("parent_chunk_count") or 0)
    window_count = int(workload.get("window_count") or 0)
    expected_multiplier = round(window_count / parent_count, 6) if parent_count else 0.0
    if (
        workload.get("parent_count") != parent_count
        or workload.get("window_count_multiplier") != expected_multiplier
        or workload.get("stress_parent_count") != 64
        or workload.get("query_count") != 12
        or workload.get("projected_raw_vector_bytes") != window_count * 384 * 4
    ):
        errors.append("E5-small window-census workload arithmetic changed")
    coverage = report.get("coverage") or {}
    if (
        coverage.get("truncated_parent_count") != 0
        or coverage.get("uncovered_content_token_count") != 0
        or coverage.get("truncated_query_count") != 0
        or coverage.get("returned_parent_chunks_remain_complete") is not True
        or coverage.get("paper_text_retained_in_report") is not False
        or coverage.get("query_text_retained_in_report") is not False
    ):
        errors.append("E5-small window-census coverage changed")
    execution = report.get("execution") or {}
    if (
        execution.get("tokenizer_load_calls") != 1
        or execution.get("model_load_calls") != 0
        or execution.get("onnx_session_calls") != 0
        or execution.get("model_inference_runs") != 0
        or execution.get("vectors_created") != 0
        or execution.get("product_enabled") is not False
    ):
        errors.append("E5-small window-census execution boundary changed")
    decision = report.get("decision") or {}
    if decision.get("passes_all") is not all((decision.get("checks") or {}).values()):
        errors.append("E5-small window-census decision arithmetic changed")
    fingerprint = _result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("E5-small window-census result fingerprint changed")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("E5-small window-census result no longer matches the pinned outcome")
    return errors


def run_census(db_root: Path, report_path: Path) -> dict[str, Any]:
    if report_path.exists():
        raise RuntimeError("E5-small window-census report already exists")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        raise RuntimeError("; ".join(protocol_errors))
    if _normalized_sha256(DEFAULT_FIXTURE) != EXPECTED_FIXTURE_SHA256:
        raise RuntimeError("semantic fixture identity changed")
    tokenizer_contract = protocol["tokenizer"]
    if (
        DEFAULT_TOKENIZER.stat().st_size != tokenizer_contract["tokenizer_json_bytes"]
        or _normalized_sha256(DEFAULT_TOKENIZER)
        != tokenizer_contract["tokenizer_json_sha256"]
    ):
        raise RuntimeError("E5-small tokenizer identity changed")
    source_before = _tree_fingerprint(db_root)
    if source_before != protocol["source_corpus"]["tree_sha256"]:
        raise RuntimeError("E5-small census source tree changed")

    from tokenizers import Tokenizer, __version__ as tokenizers_version

    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    representation = protocol["representation"]
    prefix_ids = tokenizer.encode(
        representation["passage_prefix"], add_special_tokens=False
    ).ids
    query_prefix_ids = tokenizer.encode(
        representation["query_prefix"], add_special_tokens=False
    ).ids

    chunks = load_all_chunks(db_root)
    if len(chunks) != protocol["source_corpus"]["parent_chunk_count"]:
        raise RuntimeError("E5-small census parent count changed")
    document_ids = {str(chunk["id"]).split(":", 1)[0] for chunk in chunks}
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    stress_ids = {str(item["chunk_id"]) for item in manifest["items"]}
    if len(stress_ids) != 64:
        raise RuntimeError("E5-small census stress manifest changed")

    window_tokens = int(representation["content_window_tokens"])
    stride_tokens = int(representation["nominal_content_stride_tokens"])
    special_tokens = int(representation["special_tokens_per_window"])
    content_lengths: list[int] = []
    input_lengths: list[int] = []
    windows_per_parent: list[int] = []
    stress_input_lengths: list[int] = []
    uncovered = 0
    maximum_overlap = 0
    minimum_overlap: int | None = None
    for chunk in chunks:
        content_ids = tokenizer.encode(str(chunk.get("text") or ""), add_special_tokens=False).ids
        spans = window_spans(
            len(content_ids), window_tokens=window_tokens, stride_tokens=stride_tokens
        )
        content_lengths.append(len(content_ids))
        windows_per_parent.append(len(spans))
        uncovered += uncovered_token_count(len(content_ids), spans)
        lengths = [special_tokens + len(prefix_ids) + end - start for start, end in spans]
        input_lengths.extend(lengths)
        if str(chunk["id"]) in stress_ids:
            stress_input_lengths.extend(lengths)
        for left, right in zip(spans, spans[1:]):
            overlap = left[1] - right[0]
            minimum_overlap = overlap if minimum_overlap is None else min(minimum_overlap, overlap)
            maximum_overlap = max(maximum_overlap, overlap)

    fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    query_lengths = [
        len(tokenizer.encode(representation["query_prefix"] + str(case["query"])).ids)
        for case in fixture["cases"]
    ]
    maximum_model_tokens = int(tokenizer_contract["maximum_model_tokens"])
    truncated_queries = sum(length > maximum_model_tokens for length in query_lengths)
    truncated_parents = 0
    projected_raw_vector_bytes = len(input_lengths) * 384 * 4
    decision = evaluate_gate(
        parent_count=len(chunks),
        window_count=len(input_lengths),
        maximum_input_tokens=max(input_lengths, default=0),
        truncated_parent_count=truncated_parents,
        uncovered_content_token_count=uncovered,
        truncated_query_count=truncated_queries,
        projected_index_bytes=projected_raw_vector_bytes,
        gate=protocol["gates"],
    )
    source_after = _tree_fingerprint(db_root)
    report = {
        "schema_version": 1,
        "report_kind": "retrieval_e5_small_window_census_v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _normalized_sha256(DEFAULT_PROTOCOL),
        "source_corpus": {
            "tree_sha256_before": source_before,
            "tree_sha256_after": source_after,
            "document_count": len(document_ids),
            "parent_chunk_count": len(chunks),
        },
        "tokenizer": {
            "tokenizer_json_sha256": _normalized_sha256(DEFAULT_TOKENIZER),
            "tokenizers_version": tokenizers_version,
            "passage_prefix_token_count": len(prefix_ids),
            "query_prefix_token_count": len(query_prefix_ids),
            "bos_token_id": tokenizer_contract["bos_token_id"],
            "pad_token_id": tokenizer_contract["pad_token_id"],
            "eos_token_id": tokenizer_contract["eos_token_id"],
            "maximum_model_tokens": maximum_model_tokens,
        },
        "workload": {
            "parent_count": len(chunks),
            "window_count": len(input_lengths),
            "window_count_multiplier": round(len(input_lengths) / len(chunks), 6),
            "stress_parent_count": len(stress_ids),
            "stress_window_count": len(stress_input_lengths),
            "query_count": len(query_lengths),
            "content_token_distribution": _distribution(content_lengths),
            "windows_per_parent_distribution": _distribution(windows_per_parent),
            "input_token_distribution": _distribution(input_lengths),
            "stress_input_token_distribution": _distribution(stress_input_lengths),
            "query_input_token_distribution": _distribution(query_lengths),
            "minimum_observed_overlap_tokens": minimum_overlap or 0,
            "maximum_observed_overlap_tokens": maximum_overlap,
            "projected_raw_vector_bytes": projected_raw_vector_bytes,
        },
        "coverage": {
            "truncated_parent_count": truncated_parents,
            "uncovered_content_token_count": uncovered,
            "truncated_query_count": truncated_queries,
            "returned_parent_chunks_remain_complete": True,
            "paper_text_retained_in_report": False,
            "query_text_retained_in_report": False,
        },
        "execution": {
            "python_version": platform.python_version(),
            "tokenizer_load_calls": 1,
            "model_load_calls": 0,
            "onnx_session_calls": 0,
            "model_inference_runs": 0,
            "vectors_created": 0,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
        "decision": decision,
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    errors = validate_report(report)
    if source_after != source_before:
        errors.append("E5-small census source changed during measurement")
    if errors:
        raise RuntimeError("; ".join(errors))
    _atomic_write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen tokenizer-only E5 window census")
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            errors = validate_report(
                report, expected_result_fingerprint=EXPECTED_RESULT_FINGERPRINT
            )
            if EXPECTED_REPORT_SHA256 and _normalized_sha256(args.report) != EXPECTED_REPORT_SHA256:
                errors.append("E5-small window-census report SHA-256 changed")
            if errors:
                raise RuntimeError("; ".join(errors))
        else:
            if args.db_root is None:
                raise ValueError("--db-root is required unless --validate-only is used")
            report = run_census(args.db_root.resolve(), args.report.resolve())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] E5-small window census: {type(exc).__name__}: {exc}")
        return 1
    print(f"[OK] E5-small window census: {report['decision']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
