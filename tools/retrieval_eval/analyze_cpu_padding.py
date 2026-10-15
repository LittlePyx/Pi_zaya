from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file
from kb.store import atomic_write_json, load_all_chunks
from tools.retrieval_eval.run_cpu_query_compatibility import _tree_fingerprint
from tools.retrieval_eval.validate_cpu_padding_diagnostic_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_PARENT = REPO_ROOT / "docs" / "retrieval_cpu_full_index_feasibility_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_cpu_padding_diagnostic_v1.json"
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "58ecfe98fc10f3e628320e870879d6b091f741a1e4c2069153be264467b78658"
)
EXPECTED_REPORT_SHA256: str | None = (
    "af1a6fc67929af03ba3c1afaf3fe3a3c37330796343f8d1d251df8c57596973a"
)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def make_batches(
    rows: list[tuple[str, int]], strategy: dict[str, Any]
) -> list[list[tuple[str, int]]]:
    ordered = list(rows)
    if strategy["sort_within_document"]:
        ordered.sort(key=lambda row: (row[1], row[0]))
    maximum_batch_size = int(strategy["maximum_batch_size"])
    token_budget = strategy.get("maximum_padded_tokens_per_batch")
    batches: list[list[tuple[str, int]]] = []
    pending: list[tuple[str, int]] = []
    for row in ordered:
        if token_budget is not None and row[1] > int(token_budget):
            if pending:
                batches.append(pending)
                pending = []
            if strategy.get("oversize_singleton"):
                batches.append([row])
                continue
        proposed = [*pending, row]
        proposed_padded = max(item[1] for item in proposed) * len(proposed)
        if pending and (
            len(proposed) > maximum_batch_size
            or (token_budget is not None and proposed_padded > int(token_budget))
        ):
            batches.append(pending)
            pending = [row]
        else:
            pending = proposed
    if pending:
        batches.append(pending)
    return batches


def summarize_batches(
    documents: dict[str, list[tuple[str, int]]], strategy: dict[str, Any]
) -> dict[str, Any]:
    batch_rows: list[dict[str, int]] = []
    input_chunks = 0
    effective_tokens = 0
    for rows in documents.values():
        input_chunks += len(rows)
        effective_tokens += sum(length for _chunk_id, length in rows)
        for batch in make_batches(rows, strategy):
            maximum = max(length for _chunk_id, length in batch)
            batch_rows.append(
                {
                    "size": len(batch),
                    "max_effective_tokens": maximum,
                }
            )
    padded_tokens = sum(row["size"] * row["max_effective_tokens"] for row in batch_rows)
    attention_work = sum(
        row["size"] * row["max_effective_tokens"] ** 2 for row in batch_rows
    )
    peak_padded = max(
        (row["size"] * row["max_effective_tokens"] for row in batch_rows),
        default=0,
    )
    return {
        "strategy": dict(strategy),
        "input_chunk_count": input_chunks,
        "batch_count": len(batch_rows),
        "singleton_batch_count": sum(row["size"] == 1 for row in batch_rows),
        "effective_token_count": effective_tokens,
        "padded_token_count": padded_tokens,
        "padding_token_count": padded_tokens - effective_tokens,
        "padding_amplification": round(
            padded_tokens / effective_tokens if effective_tokens else 0.0, 6
        ),
        "attention_work_proxy": attention_work,
        "peak_batch_padded_tokens": peak_padded,
        "peak_batch_sequence_tokens": max(
            (row["max_effective_tokens"] for row in batch_rows), default=0
        ),
        "batch_shapes": batch_rows,
    }


def compare_strategies(
    baseline: dict[str, Any], candidate: dict[str, Any], *, parent_build_ms: float
) -> dict[str, Any]:
    attention_speedup = baseline["attention_work_proxy"] / candidate[
        "attention_work_proxy"
    ]
    return {
        "baseline_strategy": baseline["strategy"]["id"],
        "candidate_strategy": candidate["strategy"]["id"],
        "padded_token_reduction": round(
            1.0 - candidate["padded_token_count"] / baseline["padded_token_count"],
            6,
        ),
        "attention_work_proxy_reduction": round(1.0 - 1.0 / attention_speedup, 6),
        "attention_work_proxy_speedup": round(attention_speedup, 6),
        "peak_padded_tokens_reduction": round(
            1.0
            - candidate["peak_batch_padded_tokens"]
            / baseline["peak_batch_padded_tokens"],
            6,
        ),
        "projected_build_ms_from_attention_proxy": round(
            parent_build_ms / attention_speedup, 3
        ),
    }


def evaluate_branch(comparison: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "attention_work_proxy_speedup": comparison["attention_work_proxy_speedup"]
        >= float(gate["minimum_attention_work_proxy_speedup"]),
        "peak_padded_tokens_reduction": comparison["peak_padded_tokens_reduction"]
        >= float(gate["minimum_peak_padded_tokens_reduction"]),
        "projected_build_ms": comparison[
            "projected_build_ms_from_attention_proxy"
        ]
        <= float(gate["maximum_projected_build_ms_for_bounded_followup"]),
    }
    passes = all(checks.values())
    return {
        "checks": checks,
        "passes_all": passes,
        "status": gate["pass_decision"] if passes else gate["reject_decision"],
        "full_build_authorized": False,
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }


def _result_fingerprint(report: dict[str, Any]) -> str:
    summaries = []
    for strategy in report.get("strategies") or []:
        summaries.append(
            {key: value for key, value in strategy.items() if key != "batch_shapes"}
        )
    projection = {
        "token_distribution": report.get("token_distribution"),
        "strategies": summaries,
        "comparison": report.get("comparison"),
        "decision": report.get("decision"),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_strategy(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    shapes = list(row.get("batch_shapes") or [])
    effective = int(row.get("effective_token_count") or 0)
    padded = sum(
        int(shape.get("size") or 0) * int(shape.get("max_effective_tokens") or 0)
        for shape in shapes
    )
    attention = sum(
        int(shape.get("size") or 0)
        * int(shape.get("max_effective_tokens") or 0) ** 2
        for shape in shapes
    )
    if (
        int(row.get("batch_count") or 0) != len(shapes)
        or int(row.get("input_chunk_count") or 0)
        != sum(int(shape.get("size") or 0) for shape in shapes)
        or int(row.get("padded_token_count") or 0) != padded
        or int(row.get("padding_token_count") or 0) != padded - effective
        or int(row.get("attention_work_proxy") or 0) != attention
        or int(row.get("peak_batch_padded_tokens") or 0)
        != max(
            (
                int(shape.get("size") or 0)
                * int(shape.get("max_effective_tokens") or 0)
                for shape in shapes
            ),
            default=0,
        )
    ):
        errors.append(f"CPU padding strategy arithmetic changed: {row.get('strategy', {}).get('id')}")
    return errors


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_cpu_padding_diagnostic_v1"
    ):
        errors.append("CPU padding report identity changed")
    if (
        report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("parent_report_sha256") != sha256_file(DEFAULT_PARENT)
    ):
        errors.append("CPU padding report source identity changed")
    source = report.get("source_corpus") or {}
    if (
        source.get("before_sha256") != protocol["corpus"]["tree_sha256"]
        or source.get("before_sha256") != source.get("after_sha256")
        or source.get("unchanged") is not True
    ):
        errors.append("CPU padding source corpus changed")
    tokenizer = report.get("tokenizer") or {}
    if (
        tokenizer.get("model_id") != protocol["tokenizer"]["model_id"]
        or tokenizer.get("revision") != protocol["tokenizer"]["revision"]
        or tokenizer.get("model_loaded") is not False
        or tokenizer.get("queries_encoded") != 0
        or tokenizer.get("documents_embedded") != 0
        or tokenizer.get("vectors_created") != 0
    ):
        errors.append("CPU padding tokenizer boundary changed")
    rows = list(report.get("strategies") or [])
    if [row.get("strategy") for row in rows] != protocol["strategies"]:
        errors.append("CPU padding strategies changed")
        return errors
    for row in rows:
        errors.extend(_validate_strategy(row))
    baseline = rows[0]
    candidate = next(
        row
        for row in rows
        if row["strategy"]["id"] == protocol["branch_gate"]["candidate_strategy"]
    )
    expected_comparison = compare_strategies(
        baseline,
        candidate,
        parent_build_ms=float(protocol["parent"]["full_build_ms"]),
    )
    if report.get("comparison") != expected_comparison:
        errors.append("CPU padding comparison changed")
    expected_decision = evaluate_branch(expected_comparison, protocol["branch_gate"])
    if report.get("decision") != expected_decision:
        errors.append("CPU padding decision changed")
    fingerprint = _result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("CPU padding result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned CPU padding result changed")
    return errors


def measure(*, semantic_db_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("tokenizer dependency is unavailable") from exc
    source_before = _tree_fingerprint(semantic_db_root)
    started_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["tokenizer"]["model_id"],
        revision=protocol["tokenizer"]["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    load_ms = (time.perf_counter() - started_load) * 1000.0
    chunks = load_all_chunks(semantic_db_root)
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    raw_lengths: list[int] = []
    maximum = int(protocol["tokenizer"]["maximum_effective_length"])
    started_tokenize = time.perf_counter()
    for offset in range(0, len(chunks), 128):
        batch = chunks[offset : offset + 128]
        encoded = tokenizer(
            [str(chunk.get("text") or "") for chunk in batch],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        for chunk, token_ids in zip(batch, encoded):
            chunk_id = str(chunk.get("id") or "")
            document_id = chunk_id.split(":", 1)[0]
            raw_length = len(token_ids)
            raw_lengths.append(raw_length)
            grouped[document_id].append((chunk_id, min(raw_length, maximum)))
    tokenize_ms = (time.perf_counter() - started_tokenize) * 1000.0
    strategies = [summarize_batches(grouped, row) for row in protocol["strategies"]]
    baseline = strategies[0]
    candidate = next(
        row
        for row in strategies
        if row["strategy"]["id"] == protocol["branch_gate"]["candidate_strategy"]
    )
    comparison = compare_strategies(
        baseline,
        candidate,
        parent_build_ms=float(protocol["parent"]["full_build_ms"]),
    )
    source_after = _tree_fingerprint(semantic_db_root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "retrieval_cpu_padding_diagnostic_v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "parent_report_sha256": sha256_file(DEFAULT_PARENT),
        "source_corpus": {
            "before_sha256": source_before,
            "after_sha256": source_after,
            "unchanged": source_before == source_after,
            "document_count": len(grouped),
            "chunk_count": len(chunks),
        },
        "tokenizer": {
            "model_id": protocol["tokenizer"]["model_id"],
            "revision": protocol["tokenizer"]["revision"],
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "load_ms": round(load_ms, 3),
            "tokenize_ms": round(tokenize_ms, 3),
            "model_loaded": False,
            "queries_encoded": 0,
            "documents_embedded": 0,
            "vectors_created": 0,
        },
        "token_distribution": {
            "chunk_count": len(raw_lengths),
            "effective_token_count": sum(min(value, maximum) for value in raw_lengths),
            "p50": _percentile(raw_lengths, 0.50),
            "p95": _percentile(raw_lengths, 0.95),
            "p99": _percentile(raw_lengths, 0.99),
            "maximum": max(raw_lengths, default=0),
            "above_effective_limit_count": sum(value > maximum for value in raw_lengths),
        },
        "strategies": strategies,
        "comparison": comparison,
        "decision": evaluate_branch(comparison, protocol["branch_gate"]),
    }
    report["result_fingerprint"] = _result_fingerprint(report)
    return report


def _load_protocol() -> dict[str, Any]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("CPU padding protocol invalid: " + "; ".join(errors))
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate the tokenizer-only CPU padding diagnostic."
    )
    parser.add_argument("--semantic-db-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol = _load_protocol()
    if args.validate_only:
        try:
            report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[FAIL] CPU padding report cannot be read: {type(exc).__name__}")
            return 2
        errors = validate_report(report, protocol=protocol)
        if EXPECTED_REPORT_SHA256 and sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
            errors.append("pinned CPU padding report file changed")
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(f"[OK] CPU padding diagnostic: {report['decision']['status']}")
        return 0
    if args.semantic_db_root is None:
        parser.error("--semantic-db-root is required unless --validate-only is used")
    if DEFAULT_REPORT.exists():
        parser.error("CPU padding report already exists; use --validate-only")
    report = measure(
        semantic_db_root=args.semantic_db_root.expanduser().resolve(),
        protocol=protocol,
    )
    errors = validate_report(
        report,
        protocol=protocol,
        expected_result_fingerprint=None,
    )
    if errors:
        raise ValueError("CPU padding diagnostic is invalid: " + "; ".join(errors))
    atomic_write_json(DEFAULT_REPORT, report)
    print(
        json.dumps(
            {
                "token_distribution": report["token_distribution"],
                "comparison": report["comparison"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
