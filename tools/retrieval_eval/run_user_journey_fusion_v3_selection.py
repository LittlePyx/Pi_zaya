from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from kb.retrieval_metrics import mean_metrics
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks
from tools.retrieval_eval.run_candidate_screening import _candidate_by_id, _provider
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    _index_size_bytes,
)
from tools.retrieval_eval.run_e5_large_instruct_benchmark import _release_provider
from tools.retrieval_eval.run_user_journey_qwen_dense_diagnostic import (
    CachedQueryEmbedder,
    DEFAULT_QWEN_INDEX,
)
from tools.retrieval_eval.validate_candidate_batch import DEFAULT_BATCH, validate_batch
from tools.retrieval_eval.validate_user_journey_fusion_v3_1_protocol import (
    DEFAULT_PROTOCOL,
    effective_protocol,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_user_journey_expansion_v1.json"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "retrieval_user_journey_bm25_baseline_v1.json"
DEFAULT_DENSE_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_user_journey_qwen_dense_protocol_v1.json"
)
DEFAULT_REPORT = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_selection.json"
)
PRODUCT_CHANNEL = "bm25_product_queries_selection_control"
FUSION_CHANNEL = "protected_tail_dense_quota_2_v3"
GATE_METRICS = (
    "mrr",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "contract_recall@10",
    "wrong_source_rate@5",
    "wrong_source_rate@10",
)
BASELINE_PROJECTION_METRICS = (
    "mrr",
    "recall@5",
    "recall@10",
    "ndcg@10",
    "contract_recall@10",
    "wrong_source_rate@5",
)
EXPECTED_RESULT_FINGERPRINT = (
    "55b89302358f0f4ea18a73b422f0abd554e75b01f3e5a68da84874d2593f1284"
)


def partition_fixture(fixture: dict[str, Any], *, partition: str) -> dict[str, Any]:
    key = f"{partition}_case_ids"
    if partition not in {"calibration", "holdout"} or key not in fixture["split"]:
        raise ValueError(f"unknown user-journey partition: {partition}")
    allowed = set(fixture["split"][key])
    other_key = (
        "holdout_case_ids" if partition == "calibration" else "calibration_case_ids"
    )
    forbidden = set(fixture["split"][other_key])
    cases = [dict(case) for case in fixture["cases"] if case["id"] in allowed]
    observed = {str(case["id"]) for case in cases}
    if observed != allowed or observed & forbidden:
        raise ValueError(f"fusion-v3 {partition} partition leaked or omitted cases")
    return {
        **fixture,
        "id": f"retrieval_user_journey_fusion_v3_{partition}",
        "curation_status": f"derived_{partition}_partition_for_frozen_fusion_v3",
        "ks": [5, 10],
        "cases": cases,
    }


def compact_selection_channel(channel: dict[str, Any]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name, mode in (channel.get("modes") or {}).items():
        modes[mode_name] = {
            "case_count": mode.get("case_count"),
            "metrics": {key: mode["metrics"][key] for key in GATE_METRICS},
            "latency_ms": mode.get("latency_ms"),
            "ks": mode.get("ks"),
            "cases": [
                {
                    "id": row["id"],
                    "first_relevant_rank": row["first_relevant_rank"],
                    "metrics": {key: row["metrics"][key] for key in GATE_METRICS},
                }
                for row in mode.get("cases") or []
            ],
        }
    return {"channel": channel.get("channel"), "modes": modes}


def validate_product_control_projection(
    product: dict[str, Any], baseline: dict[str, Any], case_ids: list[str]
) -> list[str]:
    errors: list[str] = []
    allowed = set(case_ids)
    source = baseline["channels"]["bm25_product_queries"]["modes"]
    for mode_name in ("scoped", "full_library"):
        observed = product["modes"][mode_name]
        expected_rows = {
            row["id"]: row for row in source[mode_name]["cases"] if row["id"] in allowed
        }
        observed_rows = {row["id"]: row for row in observed["cases"]}
        if set(expected_rows) != set(observed_rows):
            errors.append(f"product control membership changed: {mode_name}")
            continue
        for case_id, expected in expected_rows.items():
            expected_rank = expected["first_relevant_rank"]
            expected_rank = (
                expected_rank
                if expected_rank is not None and expected_rank <= 10
                else None
            )
            if observed_rows[case_id]["first_relevant_rank"] != expected_rank:
                errors.append(f"product control rank changed: {mode_name}/{case_id}")
            for metric in BASELINE_PROJECTION_METRICS:
                expected_metric = expected["metrics"][metric]
                if metric == "mrr" and expected_rank is None:
                    expected_metric = 0.0
                if observed_rows[case_id]["metrics"][metric] != expected_metric:
                    errors.append(
                        f"product control metric changed: {mode_name}/{case_id}/{metric}"
                    )
    return errors


def evaluate_partition_gate(
    product: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name in ("scoped", "full_library"):
        product_metrics = product["modes"][mode_name]["metrics"]
        candidate_metrics = candidate["modes"][mode_name]["metrics"]
        modes[mode_name] = {
            "recall@5_equal_to_product": candidate_metrics["recall@5"]
            == product_metrics["recall@5"],
            "wrong_source_rate@5_equal_to_product": candidate_metrics[
                "wrong_source_rate@5"
            ]
            == product_metrics["wrong_source_rate@5"],
            "recall@10_no_regression": candidate_metrics["recall@10"]
            >= product_metrics["recall@10"],
            "mrr_no_regression": candidate_metrics["mrr"] >= product_metrics["mrr"],
            "ndcg@10_no_regression": candidate_metrics["ndcg@10"]
            >= product_metrics["ndcg@10"],
            "contract_recall@10_no_regression": candidate_metrics["contract_recall@10"]
            >= product_metrics["contract_recall@10"],
            "wrong_source_rate@10_no_regression": candidate_metrics[
                "wrong_source_rate@10"
            ]
            <= product_metrics["wrong_source_rate@10"],
            "recall@10_gain": round(
                candidate_metrics["recall@10"] - product_metrics["recall@10"], 6
            ),
        }
    strict_gain = any(row["recall@10_gain"] > 0.0 for row in modes.values())
    per_mode_checks = [key for key in modes["scoped"] if key not in {"recall@10_gain"}]
    return {
        "modes": modes,
        "strict_recall@10_gain_in_at_least_one_mode": strict_gain,
        "passes_all": strict_gain
        and all(row[key] is True for row in modes.values() for key in per_mode_checks),
    }


def combine_channels(
    calibration: dict[str, Any], holdout: dict[str, Any]
) -> dict[str, Any]:
    if calibration["channel"] != holdout["channel"]:
        raise ValueError("cannot combine different retrieval channels")
    modes: dict[str, Any] = {}
    for mode_name in ("scoped", "full_library"):
        rows = [
            *calibration["modes"][mode_name]["cases"],
            *holdout["modes"][mode_name]["cases"],
        ]
        metrics = mean_metrics([dict(row["metrics"]) for row in rows])
        modes[mode_name] = {
            "case_count": len(rows),
            "metrics": {key: round(float(metrics[key]), 6) for key in GATE_METRICS},
            "ks": [5, 10],
            "cases": rows,
        }
    return {"channel": calibration["channel"], "modes": modes}


def evaluate_combined_gate(
    product: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name in ("scoped", "full_library"):
        product_metrics = product["modes"][mode_name]["metrics"]
        candidate_metrics = candidate["modes"][mode_name]["metrics"]
        gain = round(candidate_metrics["recall@10"] - product_metrics["recall@10"], 6)
        modes[mode_name] = {
            "recall@10_gain": gain,
            "non_regressing_rank_metrics": all(
                candidate_metrics[key] >= product_metrics[key]
                for key in (
                    "recall@5",
                    "recall@10",
                    "mrr",
                    "ndcg@10",
                    "contract_recall@10",
                )
            ),
            "wrong_source_rate@5_and_10_no_regression": all(
                candidate_metrics[key] <= product_metrics[key]
                for key in ("wrong_source_rate@5", "wrong_source_rate@10")
            ),
        }
    gain_pass = any(row["recall@10_gain"] >= 0.02 for row in modes.values())
    return {
        "modes": modes,
        "recall@10_minimum_gain_in_at_least_one_mode": gain_pass,
        "passes_all": gain_pass
        and all(
            row["non_regressing_rank_metrics"]
            and row["wrong_source_rate@5_and_10_no_regression"]
            for row in modes.values()
        ),
    }


def selection_decision(
    calibration_gate: dict[str, Any],
    holdout_gate: dict[str, Any] | None,
    combined_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    if not calibration_gate["passes_all"]:
        outcome = "rejected_calibration"
    elif not holdout_gate or not holdout_gate["passes_all"]:
        outcome = "rejected_holdout"
    elif not combined_gate or not combined_gate["passes_all"]:
        outcome = "rejected_combined"
    else:
        outcome = "eligible_for_semantic_acceptance"
    return {
        "outcome": outcome,
        "semantic_acceptance_allowed": outcome == "eligible_for_semantic_acceptance",
        "product_enabled": False,
    }


def _selection_projection(selection: dict[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for partition_name in ("calibration", "holdout", "combined"):
        partition = selection.get(partition_name)
        if not partition:
            projection[partition_name] = None
            continue
        projection[partition_name] = {"gate": partition["gate"], "channels": {}}
        for channel_name, channel in partition["channels"].items():
            projection[partition_name]["channels"][channel_name] = {
                mode_name: {
                    "metrics": mode["metrics"],
                    "cases": [
                        {
                            "id": row["id"],
                            "first_relevant_rank": row["first_relevant_rank"],
                            "metrics": row["metrics"],
                        }
                        for row in mode["cases"]
                    ],
                }
                for mode_name, mode in channel["modes"].items()
            }
    normalized = json.dumps(
        projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {"sha256": hashlib.sha256(normalized.encode()).hexdigest()}


def validate_selection_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    baseline: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "retrieval_user_journey_fusion_v3_1_selection"
    ):
        errors.append("fusion-v3 selection report identity changed")
    if (
        report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
        or report.get("baseline_sha256") != sha256_file(DEFAULT_BASELINE)
    ):
        errors.append("fusion-v3 selection source identity changed")
    if report.get("candidate") != protocol["frozen_candidate"]:
        errors.append("fusion-v3 measured candidate changed")
    dense_protocol = json.loads(DEFAULT_DENSE_PROTOCOL.read_text(encoding="utf-8"))
    if report.get("index") != {**dense_protocol["existing_index"], "rebuilt": False}:
        errors.append("fusion-v3 index identity changed or was rebuilt")
    selection = report.get("selection") or {}
    calibration = selection.get("calibration") or {}
    calibration_channels = calibration.get("channels") or {}
    if list(calibration_channels) != [PRODUCT_CHANNEL, FUSION_CHANNEL]:
        errors.append("fusion-v3 calibration channels changed")
        return errors
    calibration_members = set(fixture["split"]["calibration_case_ids"])
    holdout_members = set(fixture["split"]["holdout_case_ids"])
    calibration_ids = [
        str(case["id"])
        for case in fixture["cases"]
        if case["id"] in calibration_members
    ]
    holdout_ids = [
        str(case["id"]) for case in fixture["cases"] if case["id"] in holdout_members
    ]
    expected_by_partition = {"calibration": calibration_ids, "holdout": holdout_ids}
    for partition_name, expected_ids in expected_by_partition.items():
        partition = selection.get(partition_name)
        if partition is None:
            if partition_name == "calibration":
                errors.append("fusion-v3 calibration result is missing")
            continue
        channels = partition.get("channels") or {}
        if list(channels) != [PRODUCT_CHANNEL, FUSION_CHANNEL]:
            errors.append(f"fusion-v3 {partition_name} channels changed")
            continue
        for channel_name, channel in channels.items():
            for mode_name in ("scoped", "full_library"):
                mode = (channel.get("modes") or {}).get(mode_name) or {}
                observed_ids = [
                    str(row.get("id") or "") for row in mode.get("cases") or []
                ]
                if observed_ids != expected_ids:
                    errors.append(
                        f"{partition_name}/{channel_name}/{mode_name} case membership changed"
                    )
                if mode.get("ks") != [5, 10]:
                    errors.append(
                        f"{partition_name}/{channel_name}/{mode_name} depth changed"
                    )
                if any(key not in (mode.get("metrics") or {}) for key in GATE_METRICS):
                    errors.append(
                        f"{partition_name}/{channel_name}/{mode_name} metrics incomplete"
                    )
        errors.extend(
            validate_product_control_projection(
                channels[PRODUCT_CHANNEL], baseline, expected_ids
            )
        )
        expected_gate = evaluate_partition_gate(
            channels[PRODUCT_CHANNEL], channels[FUSION_CHANNEL]
        )
        if partition.get("gate") != expected_gate:
            errors.append(f"fusion-v3 {partition_name} gate arithmetic changed")
    calibration_passes = bool((calibration.get("gate") or {}).get("passes_all"))
    holdout = selection.get("holdout")
    if calibration_passes != bool(report.get("holdout_accessed")):
        errors.append("fusion-v3 conditional holdout boundary changed")
    if calibration_passes and holdout is None:
        errors.append("fusion-v3 passing calibration did not run holdout")
    if not calibration_passes and holdout is not None:
        errors.append("fusion-v3 failing calibration accessed holdout")
    combined = selection.get("combined")
    if holdout is not None:
        expected_product = combine_channels(
            calibration_channels[PRODUCT_CHANNEL], holdout["channels"][PRODUCT_CHANNEL]
        )
        expected_fusion = combine_channels(
            calibration_channels[FUSION_CHANNEL], holdout["channels"][FUSION_CHANNEL]
        )
        expected_combined_gate = evaluate_combined_gate(
            expected_product, expected_fusion
        )
        if combined != {
            "channels": {
                PRODUCT_CHANNEL: expected_product,
                FUSION_CHANNEL: expected_fusion,
            },
            "gate": expected_combined_gate,
        }:
            errors.append("fusion-v3 combined result or gate changed")
    elif combined is not None:
        errors.append("fusion-v3 combined result exists without holdout")
    holdout_gate = (holdout or {}).get("gate")
    combined_gate = (combined or {}).get("gate")
    expected_decision = selection_decision(
        calibration["gate"], holdout_gate, combined_gate
    )
    if report.get("decision") != expected_decision:
        errors.append("fusion-v3 selection decision changed")
    expected_cache_count = 18 if report.get("holdout_accessed") else 12
    if report.get("query_embedding_cache") != {
        "enabled": True,
        "unique_query_count": expected_cache_count,
        "shared_across_partitions_and_modes": True,
    }:
        errors.append("fusion-v3 query cache count changed")
    fingerprint = _selection_projection(selection)["sha256"]
    if report.get("result_fingerprint") != fingerprint:
        errors.append("fusion-v3 result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned fusion-v3 selection result changed")
    if (
        report.get("semantic_acceptance_accessed") is not False
        or report.get("product_enabled") is not False
    ):
        errors.append("fusion-v3 selection crossed semantic or product boundary")
    return errors


def _measure_partition(
    *,
    partition: str,
    fixture: dict[str, Any],
    chunks: list[dict],
    vectors: dict[str, Any],
    cached_query: CachedQueryEmbedder,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    selected_fixture = partition_fixture(fixture, partition=partition)
    _mapping, corpus_errors = validate_indexed_corpus(selected_fixture, chunks)
    if corpus_errors:
        raise ValueError(f"{partition} corpus mismatch: " + "; ".join(corpus_errors))

    def product_factory(
        selected_chunks: list[dict], whole_library: bool
    ) -> ProductQuerySparseRetriever:
        return ProductQuerySparseRetriever(
            selected_chunks, whole_library=whole_library
        )

    def dense_factory(selected_chunks: list[dict]) -> DenseRetriever:
        return DenseRetriever(
            selected_chunks,
            [vectors[str(chunk.get("id") or "")] for chunk in selected_chunks],
            cached_query,
            min_score=-1.0,
        )

    candidate = protocol["frozen_candidate"]

    def fusion_factory(
        selected_chunks: list[dict], whole_library: bool
    ) -> ProtectedDenseTailRetriever:
        return ProtectedDenseTailRetriever(
            product_factory(selected_chunks, whole_library),
            dense_factory(selected_chunks),
            dense_novelty_quota=candidate["dense_novelty_quota"],
            candidate_multiplier=candidate["candidate_multiplier"],
            protected_sparse_prefix=candidate["protected_sparse_prefix"],
            sparse_tail_window=candidate["sparse_tail_window"],
        )

    product = compact_selection_channel(
        evaluate_candidate_channel(
            selected_fixture,
            chunks,
            channel_name=PRODUCT_CHANNEL,
            mode_retriever_factory=product_factory,
        )
    )
    fusion = compact_selection_channel(
        evaluate_candidate_channel(
            selected_fixture,
            chunks,
            channel_name=FUSION_CHANNEL,
            mode_retriever_factory=fusion_factory,
        )
    )
    return {
        "channels": {PRODUCT_CHANNEL: product, FUSION_CHANNEL: fusion},
        "gate": evaluate_partition_gate(product, fusion),
    }


def measure(
    *,
    db_root: Path,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    chunks = load_all_chunks(db_root)
    batch = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    batch_errors = validate_batch(batch)
    if batch_errors:
        raise ValueError("frozen candidate batch invalid: " + "; ".join(batch_errors))
    candidate = _candidate_by_id(
        batch, protocol["frozen_representation"]["candidate_id"]
    )
    provider = _provider(candidate)
    provider.prepare()
    try:
        store = EmbeddingIndexStore(db_root, index_dir=DEFAULT_QWEN_INDEX)
        status = store.status(provider)
        if status.state != "ready":
            raise ValueError(f"existing Qwen index is not ready: {status.state}")
        loaded = store.load(provider)
        vectors = {
            str(chunk.get("id") or ""): vector
            for chunk, vector in zip(loaded.chunks, loaded.embeddings)
        }
        cached_query = CachedQueryEmbedder(provider.embed_query)
        calibration = _measure_partition(
            partition="calibration",
            fixture=fixture,
            chunks=chunks,
            vectors=vectors,
            cached_query=cached_query,
            protocol=protocol,
        )
        control_errors = validate_product_control_projection(
            calibration["channels"][PRODUCT_CHANNEL],
            baseline,
            list(fixture["split"]["calibration_case_ids"]),
        )
        if control_errors:
            raise ValueError(
                "selection product control drifted: " + "; ".join(control_errors)
            )
        holdout: dict[str, Any] | None = None
        combined: dict[str, Any] | None = None
        if calibration["gate"]["passes_all"]:
            holdout = _measure_partition(
                partition="holdout",
                fixture=fixture,
                chunks=chunks,
                vectors=vectors,
                cached_query=cached_query,
                protocol=protocol,
            )
            control_errors = validate_product_control_projection(
                holdout["channels"][PRODUCT_CHANNEL],
                baseline,
                list(fixture["split"]["holdout_case_ids"]),
            )
            if control_errors:
                raise ValueError(
                    "holdout product control drifted: " + "; ".join(control_errors)
                )
            combined_product = combine_channels(
                calibration["channels"][PRODUCT_CHANNEL],
                holdout["channels"][PRODUCT_CHANNEL],
            )
            combined_fusion = combine_channels(
                calibration["channels"][FUSION_CHANNEL],
                holdout["channels"][FUSION_CHANNEL],
            )
            combined = {
                "channels": {
                    PRODUCT_CHANNEL: combined_product,
                    FUSION_CHANNEL: combined_fusion,
                },
                "gate": evaluate_combined_gate(combined_product, combined_fusion),
            }
        selection = {
            "calibration": calibration,
            "holdout": holdout,
            "combined": combined,
        }
        runtime = provider.runtime_info()
        dense_protocol = json.loads(DEFAULT_DENSE_PROTOCOL.read_text(encoding="utf-8"))
        return {
            "schema_version": 1,
            "report_kind": "retrieval_user_journey_fusion_v3_1_selection",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
            "baseline_sha256": sha256_file(DEFAULT_BASELINE),
            "candidate": dict(protocol["frozen_candidate"]),
            "runtime": {
                key: runtime.get(key)
                for key in (
                    "model_id",
                    "revision",
                    "device",
                    "dtype",
                    "allow_download",
                    "trust_remote_code",
                )
            },
            "index": {
                **dense_protocol["existing_index"],
                "size_bytes": _index_size_bytes(DEFAULT_QWEN_INDEX),
                "rebuilt": False,
            },
            "query_embedding_cache": {
                "enabled": True,
                "unique_query_count": cached_query.query_count,
                "shared_across_partitions_and_modes": True,
            },
            "selection": selection,
            "decision": selection_decision(
                calibration["gate"],
                (holdout or {}).get("gate"),
                (combined or {}).get("gate"),
            ),
            "result_fingerprint": _selection_projection(selection)["sha256"],
            "holdout_accessed": holdout is not None,
            "semantic_acceptance_accessed": False,
            "product_enabled": False,
        }
    finally:
        _release_provider(provider)


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    replacement_protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    errors = validate_protocol(replacement_protocol)
    if errors:
        raise ValueError("frozen fusion-v3.1 protocol invalid: " + "; ".join(errors))
    protocol = effective_protocol(replacement_protocol)
    return protocol, fixture, baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate the unchanged-candidate fusion-v3.1 replacement selection."
    )
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol, fixture, baseline = _load_contracts()
    if args.validate_only:
        try:
            report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[FAIL] fusion-v3.1 selection cannot be read: {type(exc).__name__}")
            return 2
        errors = validate_selection_report(
            report, protocol=protocol, fixture=fixture, baseline=baseline
        )
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(f"[OK] frozen fusion-v3.1 selection: {report['decision']['outcome']}")
        return 0
    if args.db_root is None:
        parser.error("--db-root is required unless --validate-only is used")
    if DEFAULT_REPORT.exists():
        parser.error("fusion-v3.1 selection already exists; use --validate-only")
    report = measure(
        db_root=args.db_root.expanduser().resolve(),
        protocol=protocol,
        fixture=fixture,
        baseline=baseline,
    )
    errors = validate_selection_report(
        report, protocol=protocol, fixture=fixture, baseline=baseline
    )
    if errors:
        raise ValueError(
            "measured fusion-v3 selection is invalid: " + "; ".join(errors)
        )
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"selection": report["selection"], "decision": report["decision"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
