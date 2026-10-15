from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.dense_retriever import DenseRetriever
from kb.embedding_index import EmbeddingIndexStore
from kb.protected_dense_tail_retriever import ProtectedDenseTailRetriever
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import atomic_write_json
from kb.transformers_embedding_provider import TransformersPoolingEmbeddingProvider
from tools.retrieval_eval.run_candidate_screening import _candidate_by_id, _provider
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_e5_large_instruct_benchmark import _release_provider
from tools.retrieval_eval.run_user_journey_fusion_v3_1_semantic_acceptance import (
    DEFAULT_REPORT as DEFAULT_PARENT_REPORT,
    evaluate_acceptance,
)
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import (
    FUSION_CHANNEL,
    PRODUCT_CHANNEL,
    compact_selection_channel,
)
from tools.retrieval_eval.validate_candidate_batch import DEFAULT_BATCH, validate_batch
from tools.retrieval_eval.validate_cpu_query_compatibility_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_user_journey_fusion_v3_1_semantic"
    / "qwen_index_v1"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_cpu_query_compatibility_v1.json"
EXPECTED_RESULT_FINGERPRINT: str | None = (
    "6063af5f27991d74db0a044218b78385f5ff4318e9deb229e27d21b564730ab4"
)
EXPECTED_REPORT_SHA256 = (
    "52ed6a51d22b0e6f0b9fc2533e15faf5c66398e1a46b0e0c358568c34015e0ad"
)


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(max(0.0, float(value)) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 3)


class TimedCachedQueryEmbedder:
    def __init__(self, embed_query) -> None:
        self._embed_query = embed_query
        self._cache: dict[str, tuple[float, ...]] = {}
        self.timings: list[dict[str, object]] = []

    def __call__(self, query: str) -> tuple[float, ...]:
        key = str(query or "")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        vector = tuple(self._embed_query(key))
        duration_ms = (time.perf_counter() - started) * 1000.0
        self._cache[key] = vector
        self.timings.append(
            {
                "ordinal": len(self.timings) + 1,
                "duration_ms": round(duration_ms, 3),
            }
        )
        return vector


def evaluate_gate(
    *,
    product: dict[str, Any],
    parent_fusion: dict[str, Any],
    cpu_fusion: dict[str, Any],
    semantic_acceptance: dict[str, Any],
    provider_load_ms: float,
    query_durations_ms: list[float],
    expected_query_count: int,
    gate: dict[str, Any],
    index_unchanged: bool,
) -> dict[str, Any]:
    mode_checks: dict[str, dict[str, bool]] = {}
    metrics = list(gate["require_no_regression_vs_parent_fusion_metrics"])
    for mode in ("scoped", "full_library"):
        product_metrics = product["modes"][mode]["metrics"]
        parent_metrics = parent_fusion["modes"][mode]["metrics"]
        cpu_metrics = cpu_fusion["modes"][mode]["metrics"]
        checks = {
            f"no_regression_{metric}": cpu_metrics[metric] >= parent_metrics[metric]
            for metric in metrics
        }
        checks["recall_at_5_equal_to_product"] = (
            cpu_metrics["recall@5"] == product_metrics["recall@5"]
        )
        checks["wrong_source_at_5_equal_to_product"] = (
            cpu_metrics["wrong_source_rate@5"]
            == product_metrics["wrong_source_rate@5"]
        )
        checks["wrong_source_at_5_no_worse_than_parent"] = (
            cpu_metrics["wrong_source_rate@5"]
            <= parent_metrics["wrong_source_rate@5"]
        )
        mode_checks[mode] = checks
    query_count = len(query_durations_ms)
    operational_checks = {
        "provider_load_ms": provider_load_ms <= float(gate["maximum_provider_load_ms"]),
        "query_completion_rate": (
            round(query_count / expected_query_count, 6)
            >= float(gate["required_query_completion_rate"])
        ),
        "warm_query_embedding_p95_ms": (
            _percentile(query_durations_ms, 0.95)
            <= float(gate["maximum_warm_query_embedding_p95_ms"])
        ),
        "index_tree_unchanged": bool(index_unchanged),
    }
    passes = bool(
        semantic_acceptance.get("passes_all")
        and all(all(checks.values()) for checks in mode_checks.values())
        and all(operational_checks.values())
    )
    return {
        "semantic_acceptance_passes": bool(semantic_acceptance.get("passes_all")),
        "mode_checks": mode_checks,
        "operational_checks": operational_checks,
        "passes_all": passes,
    }


def _result_fingerprint(report: dict[str, Any]) -> str:
    projection = {
        "channels": {
            name: _stable_channel_projection(channel)
            for name, channel in (report.get("channels") or {}).items()
        },
        "semantic_acceptance": report.get("semantic_acceptance"),
        "gate": report.get("gate"),
        "decision": report.get("decision"),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _stable_channel_projection(channel: dict[str, Any]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name, mode in (channel.get("modes") or {}).items():
        modes[mode_name] = {
            key: value for key, value in mode.items() if key != "latency_ms"
        }
    return {"channel": channel.get("channel"), "modes": modes}


def _persist_then_validate(
    report: dict[str, Any],
    *,
    report_path: Path,
    validator,
) -> list[str]:
    atomic_write_json(report_path, report)
    return list(validator(report))


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    parent: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != 1 or report.get("report_kind") != "retrieval_cpu_query_compatibility_v1":
        errors.append("CPU query report identity changed")
    if (
        report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
        or report.get("parent_report_sha256") != sha256_file(DEFAULT_PARENT_REPORT)
    ):
        errors.append("CPU query report source identity changed")
    if report.get("case_ids") != [str(case["id"]) for case in fixture["cases"]]:
        errors.append("CPU query report case membership changed")
    if report.get("candidate") != protocol["cpu_query_candidate"]:
        errors.append("CPU query report candidate changed")
    index = report.get("index") or {}
    if (
        index.get("before_sha256") != protocol["read_only_index"]["tree_sha256"]
        or index.get("before_sha256") != index.get("after_sha256")
        or index.get("unchanged") is not True
    ):
        errors.append("CPU query report changed the accepted index")
    channels = report.get("channels") or {}
    if list(channels) != [PRODUCT_CHANNEL, "cpu_float32_protected_dense_tail"]:
        errors.append("CPU query report channels changed")
        return errors
    parent_product = parent["channels"][PRODUCT_CHANNEL]
    if _stable_channel_projection(channels[PRODUCT_CHANNEL]) != _stable_channel_projection(
        parent_product
    ):
        errors.append("CPU query product control changed")
    timings = list((report.get("runtime") or {}).get("query_timings") or [])
    if [row.get("ordinal") for row in timings] != list(
        range(1, int(protocol["execution"]["unique_query_count"]) + 1)
    ) or any(set(row) != {"ordinal", "duration_ms"} for row in timings):
        errors.append("CPU query timing membership changed")
    durations = [float(row.get("duration_ms") or 0.0) for row in timings]
    runtime = report.get("runtime") or {}
    expected_summary = {
        "provider_load_ms": round(float(runtime.get("provider_load_ms") or 0.0), 3),
        "unique_query_count": len(durations),
        "query_completion_rate": round(
            len(durations) / int(protocol["execution"]["unique_query_count"]), 6
        ),
        "query_embedding_p50_ms": _percentile(durations, 0.50),
        "query_embedding_p95_ms": _percentile(durations, 0.95),
    }
    if report.get("summary") != expected_summary:
        errors.append("CPU query runtime summary is inconsistent")
    expected_acceptance = evaluate_acceptance(
        channels[PRODUCT_CHANNEL],
        channels["cpu_float32_protected_dense_tail"],
        minimum_gain=float(
            protocol["predeclared_gate"]["minimum_semantic_recall_at_10_gain_each_mode"]
        ),
    )
    if report.get("semantic_acceptance") != expected_acceptance:
        errors.append("CPU query semantic gate arithmetic changed")
    expected_gate = evaluate_gate(
        product=channels[PRODUCT_CHANNEL],
        parent_fusion=parent["channels"][FUSION_CHANNEL],
        cpu_fusion=channels["cpu_float32_protected_dense_tail"],
        semantic_acceptance=expected_acceptance,
        provider_load_ms=expected_summary["provider_load_ms"],
        query_durations_ms=durations,
        expected_query_count=int(protocol["execution"]["unique_query_count"]),
        gate=protocol["predeclared_gate"],
        index_unchanged=bool(index.get("unchanged")),
    )
    if report.get("gate") != expected_gate:
        errors.append("CPU query compatibility gate is inconsistent")
    expected_decision = {
        "status": (
            protocol["execution"]["pass_decision"]
            if expected_gate["passes_all"]
            else protocol["execution"]["reject_decision"]
        ),
        "cpu_full_index_study_allowed": bool(expected_gate["passes_all"]),
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("CPU query report decision changed")
    expected_lineage = {
        "protocol_id": protocol["protocol_id"],
        "prior_attempted_measurement_runs": int(
            protocol["execution"]["prior_attempted_measurement_runs"]
        ),
        "prior_usable_outcomes": int(protocol["execution"]["prior_usable_outcomes"]),
        "recovery_run_ordinal": 1,
        "raw_report_persisted_before_validation": True,
    }
    if report.get("execution_lineage") != expected_lineage:
        errors.append("CPU query recovery lineage changed")
    fingerprint = _result_fingerprint(report)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("CPU query result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned CPU query result changed")
    return errors


def _cpu_provider(candidate: dict[str, Any]) -> TransformersPoolingEmbeddingProvider:
    return TransformersPoolingEmbeddingProvider(
        model_id=str(candidate["model_id"]),
        revision=str(candidate["revision"]),
        dimension=1024,
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query:"
        ),
        passage_prefix="",
        max_length=int(candidate["max_length"]),
        batch_size=1,
        device="cpu",
        dtype="float32",
        allow_download=False,
        pooling=str(candidate["pooling"]),
        padding_side=str(candidate["padding_side"]),
    )


def measure(*, semantic_db_root: Path, protocol: dict[str, Any], fixture: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    batch = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    batch_errors = validate_batch(batch)
    if batch_errors:
        raise ValueError("candidate batch invalid: " + "; ".join(batch_errors))
    accepted_candidate = _candidate_by_id(batch, "qwen3_embedding_0_6b_v1")
    parent_provider = _provider(accepted_candidate)
    cpu_provider = _cpu_provider(protocol["cpu_query_candidate"])
    if cpu_provider.descriptor.identity != protocol["cpu_query_candidate"]["provider_identity"]:
        raise ValueError("CPU provider identity mismatch")
    before = _tree_fingerprint(DEFAULT_INDEX)
    try:
        loaded = EmbeddingIndexStore(semantic_db_root, index_dir=DEFAULT_INDEX).load(parent_provider)
        semantic_fixture = {**fixture, "ks": [5, 10]}
        _mapping, corpus_errors = validate_indexed_corpus(semantic_fixture, loaded.chunks)
        if corpus_errors:
            raise ValueError("semantic corpus mismatch: " + "; ".join(corpus_errors))
        cpu_provider.prepare()
        provider_load_ms = float(cpu_provider.runtime_info()["load_duration_ms"])
        timed_query = TimedCachedQueryEmbedder(cpu_provider.embed_query)
        vectors = {
            str(chunk.get("id") or ""): vector
            for chunk, vector in zip(loaded.chunks, loaded.embeddings)
        }

        def product_factory(chunks: list[dict], whole_library: bool):
            return ProductQuerySparseRetriever(chunks, whole_library=whole_library)

        def fusion_factory(chunks: list[dict], whole_library: bool):
            dense = DenseRetriever(
                chunks,
                [vectors[str(chunk.get("id") or "")] for chunk in chunks],
                timed_query,
                min_score=-1.0,
            )
            fusion = protocol["fusion"]
            return ProtectedDenseTailRetriever(
                product_factory(chunks, whole_library),
                dense,
                dense_novelty_quota=int(fusion["dense_novelty_quota"]),
                candidate_multiplier=int(fusion["candidate_multiplier"]),
                protected_sparse_prefix=int(fusion["protected_sparse_prefix"]),
                sparse_tail_window=int(fusion["sparse_tail_window"]),
            )

        product = compact_selection_channel(
            evaluate_candidate_channel(
                semantic_fixture,
                loaded.chunks,
                channel_name=PRODUCT_CHANNEL,
                mode_retriever_factory=product_factory,
            )
        )
        cpu_fusion = compact_selection_channel(
            evaluate_candidate_channel(
                semantic_fixture,
                loaded.chunks,
                channel_name="cpu_float32_protected_dense_tail",
                mode_retriever_factory=fusion_factory,
            )
        )
        after = _tree_fingerprint(DEFAULT_INDEX)
        durations = [float(row["duration_ms"]) for row in timed_query.timings]
        semantic_acceptance = evaluate_acceptance(
            product,
            cpu_fusion,
            minimum_gain=float(
                protocol["predeclared_gate"]["minimum_semantic_recall_at_10_gain_each_mode"]
            ),
        )
        gate = evaluate_gate(
            product=product,
            parent_fusion=parent["channels"][FUSION_CHANNEL],
            cpu_fusion=cpu_fusion,
            semantic_acceptance=semantic_acceptance,
            provider_load_ms=round(provider_load_ms, 3),
            query_durations_ms=durations,
            expected_query_count=int(protocol["execution"]["unique_query_count"]),
            gate=protocol["predeclared_gate"],
            index_unchanged=before == after,
        )
        report = {
            "schema_version": 1,
            "report_kind": "retrieval_cpu_query_compatibility_v1",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
            "parent_report_sha256": sha256_file(DEFAULT_PARENT_REPORT),
            "case_ids": [str(case["id"]) for case in fixture["cases"]],
            "candidate": dict(protocol["cpu_query_candidate"]),
            "execution_lineage": {
                "protocol_id": protocol["protocol_id"],
                "prior_attempted_measurement_runs": int(
                    protocol["execution"]["prior_attempted_measurement_runs"]
                ),
                "prior_usable_outcomes": int(
                    protocol["execution"]["prior_usable_outcomes"]
                ),
                "recovery_run_ordinal": 1,
                "raw_report_persisted_before_validation": True,
            },
            "index": {
                "before_sha256": before,
                "after_sha256": after,
                "unchanged": before == after,
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "provider_load_ms": round(provider_load_ms, 3),
                "query_timings": timed_query.timings,
            },
            "summary": {
                "provider_load_ms": round(provider_load_ms, 3),
                "unique_query_count": len(durations),
                "query_completion_rate": round(
                    len(durations) / int(protocol["execution"]["unique_query_count"]), 6
                ),
                "query_embedding_p50_ms": _percentile(durations, 0.50),
                "query_embedding_p95_ms": _percentile(durations, 0.95),
            },
            "channels": {
                PRODUCT_CHANNEL: product,
                "cpu_float32_protected_dense_tail": cpu_fusion,
            },
            "semantic_acceptance": semantic_acceptance,
            "gate": gate,
            "decision": {
                "status": (
                    protocol["execution"]["pass_decision"]
                    if gate["passes_all"]
                    else protocol["execution"]["reject_decision"]
                ),
                "cpu_full_index_study_allowed": bool(gate["passes_all"]),
                "product_enabled": False,
                "ordinary_user_surface_exposed": False,
            },
        }
        report["result_fingerprint"] = _result_fingerprint(report)
        return report
    finally:
        _release_provider(cpu_provider)
        _release_provider(parent_provider)


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("CPU protocol invalid: " + "; ".join(errors))
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    parent = json.loads(DEFAULT_PARENT_REPORT.read_text(encoding="utf-8"))
    return protocol, fixture, parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or validate the one-run CPU query compatibility screen.")
    parser.add_argument("--semantic-db-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol, fixture, parent = _load_contracts()
    if args.validate_only:
        try:
            report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[FAIL] CPU query report cannot be read: {type(exc).__name__}")
            return 2
        errors = validate_report(report, protocol=protocol, fixture=fixture, parent=parent)
        if sha256_file(DEFAULT_REPORT) != EXPECTED_REPORT_SHA256:
            errors.append("pinned CPU query report file changed")
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(f"[OK] CPU query compatibility: {report['decision']['status']}")
        return 0
    if args.semantic_db_root is None:
        parser.error("--semantic-db-root is required unless --validate-only is used")
    if DEFAULT_REPORT.exists():
        parser.error("CPU query report already exists; use --validate-only")
    report = measure(
        semantic_db_root=args.semantic_db_root.expanduser().resolve(),
        protocol=protocol,
        fixture=fixture,
        parent=parent,
    )
    errors = _persist_then_validate(
        report,
        report_path=DEFAULT_REPORT,
        validator=lambda value: validate_report(
            value,
            protocol=protocol,
            fixture=fixture,
            parent=parent,
            expected_result_fingerprint=None,
        ),
    )
    if errors:
        print("[FAIL] raw CPU query report preserved before validation")
        print("[FAIL] " + "; ".join(errors))
        return 1
    print(json.dumps({"summary": report["summary"], "decision": report["decision"]}, indent=2))
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
