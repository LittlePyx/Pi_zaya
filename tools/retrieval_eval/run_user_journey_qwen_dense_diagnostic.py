from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.dense_retriever import DenseRetriever
from kb.embedding_index import EmbeddingIndexStore
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_metrics import mean_metrics
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks
from tools.retrieval_eval.run_candidate_screening import _candidate_by_id, _provider
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    _compact_channel,
    _index_size_bytes,
)
from tools.retrieval_eval.run_e5_large_instruct_benchmark import _release_provider
from tools.retrieval_eval.validate_candidate_batch import DEFAULT_BATCH, validate_batch
from tools.retrieval_eval.validate_user_journey_qwen_dense_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_user_journey_expansion_v1.json"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "retrieval_user_journey_bm25_baseline_v1.json"
DEFAULT_QWEN_REPORT = (
    REPO_ROOT / "docs" / "retrieval_candidate_screening_qwen3_embedding_0_6b_v1.json"
)
DEFAULT_QWEN_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_candidate_screening_v1"
    / "qwen3_embedding_0_6b_v1"
    / "index_v1"
)
DEFAULT_REPORT = (
    REPO_ROOT / "docs" / "retrieval_user_journey_qwen_dense_diagnostic_v1.json"
)
PRODUCT_CHANNEL = "bm25_product_queries_frozen_control"
DENSE_CHANNEL = "qwen_dense_single_vector"
METRIC_KEYS = (
    "mrr",
    "recall@5",
    "recall@10",
    "recall@20",
    "ndcg@10",
    "contract_recall@10",
    "contract_recall@20",
    "wrong_source_rate@5",
)
EXPECTED_RESULT_FINGERPRINT = (
    "bb67c5f33daf4ce17d5f9d92d8237be41453d435fd367ff51a90a81f70985cb6"
)


class CachedQueryEmbedder:
    def __init__(self, embed_query: Callable[[str], list[float]]) -> None:
        self._embed_query = embed_query
        self._cache: dict[str, list[float]] = {}

    @property
    def query_count(self) -> int:
        return len(self._cache)

    def __call__(self, query: str) -> list[float]:
        if query not in self._cache:
            self._cache[query] = self._embed_query(query)
        return self._cache[query]


def calibration_fixture(
    fixture: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    allowed = set(fixture["split"]["calibration_case_ids"])
    holdout = set(fixture["split"]["holdout_case_ids"])
    cases = [dict(case) for case in fixture["cases"] if case["id"] in allowed]
    observed = {str(case["id"]) for case in cases}
    if (
        observed != allowed
        or observed & holdout
        or len(cases) != protocol["partition"]["case_count"]
    ):
        raise ValueError("dense diagnostic did not isolate the calibration partition")
    return {
        **fixture,
        "id": "retrieval_user_journey_qwen_dense_calibration_v1",
        "curation_status": "derived_calibration_partition_for_frozen_diagnostic",
        "ks": list(protocol["diagnostic_contract"]["reported_metric_depths"]),
        "cases": cases,
    }


def frozen_product_control(
    baseline: dict[str, Any], case_ids: list[str]
) -> dict[str, Any]:
    allowed = set(case_ids)
    source = baseline["channels"]["bm25_product_queries"]
    modes: dict[str, Any] = {}
    for mode_name in ("scoped", "full_library"):
        source_mode = source["modes"][mode_name]
        rows = [dict(row) for row in source_mode["cases"] if row["id"] in allowed]
        if {str(row["id"]) for row in rows} != allowed:
            raise ValueError(
                f"frozen BM25 control omitted calibration cases: {mode_name}"
            )
        metrics = mean_metrics([dict(row["metrics"]) for row in rows])
        modes[mode_name] = {
            "case_count": len(rows),
            "metrics": {key: round(float(metrics[key]), 6) for key in METRIC_KEYS},
            "ks": [5, 10, 20],
            "cases": [
                {
                    "id": row["id"],
                    "first_relevant_rank": row["first_relevant_rank"],
                    "metrics": {key: row["metrics"][key] for key in METRIC_KEYS},
                }
                for row in rows
            ],
        }
    return {"channel": PRODUCT_CHANNEL, "modes": modes}


def _rank_bucket(rank: Any) -> str:
    if rank is None:
        return "miss_top20"
    value = int(rank)
    if value == 1:
        return "rank_1"
    if value <= 5:
        return "ranks_2_5"
    if value <= 10:
        return "ranks_6_10"
    return "ranks_11_20"


def evaluate_complementarity(
    product: dict[str, Any], dense: dict[str, Any]
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name in ("scoped", "full_library"):
        product_mode = product["modes"][mode_name]
        dense_mode = dense["modes"][mode_name]
        product_rows = {row["id"]: row for row in product_mode["cases"]}
        dense_rows = {row["id"]: row for row in dense_mode["cases"]}
        classes: dict[str, list[str]] = {
            "both_hit": [],
            "bm25_only": [],
            "dense_only": [],
            "neither": [],
        }
        for case_id in product_rows:
            product_hit = float(product_rows[case_id]["metrics"]["recall@10"]) > 0.0
            dense_hit = float(dense_rows[case_id]["metrics"]["recall@10"]) > 0.0
            category = (
                "both_hit"
                if product_hit and dense_hit
                else "bm25_only"
                if product_hit
                else "dense_only"
                if dense_hit
                else "neither"
            )
            classes[category].append(case_id)
        product_metrics = product_mode["metrics"]
        dense_metrics = dense_mode["metrics"]
        modes[mode_name] = {
            "recall@10_gain_vs_product": round(
                float(dense_metrics["recall@10"]) - float(product_metrics["recall@10"]),
                6,
            ),
            "classes_at_10": classes,
            "class_counts_at_10": {key: len(value) for key, value in classes.items()},
            "dense_first_relevant_rank_buckets": dict(
                sorted(
                    Counter(
                        _rank_bucket(row["first_relevant_rank"])
                        for row in dense_rows.values()
                    ).items()
                )
            ),
        }
    dense_only_pass = any(
        mode["class_counts_at_10"]["dense_only"] >= 2 for mode in modes.values()
    )
    recall_gain_pass = any(
        mode["recall@10_gain_vs_product"] > 0.0 for mode in modes.values()
    )
    return {
        "comparison_depth": 10,
        "modes": modes,
        "fusion_v3_design_condition": {
            "dense_only_minimum_passes": dense_only_pass,
            "strict_recall_gain_passes": recall_gain_pass,
            "passes_all": dense_only_pass and recall_gain_pass,
            "product_eligibility": False,
        },
    }


def _result_fingerprint(channels: dict[str, Any]) -> str:
    projection: dict[str, Any] = {}
    for channel_name, channel in channels.items():
        projection[channel_name] = {}
        for mode_name, mode in channel["modes"].items():
            projection[channel_name][mode_name] = {
                "metrics": {key: mode["metrics"][key] for key in METRIC_KEYS},
                "cases": [
                    {
                        "id": row["id"],
                        "first_relevant_rank": row["first_relevant_rank"],
                        "metrics": {key: row["metrics"][key] for key in METRIC_KEYS},
                    }
                    for row in mode["cases"]
                ],
            }
    normalized = json.dumps(
        projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError(
            "frozen Qwen diagnostic protocol invalid: " + "; ".join(errors)
        )
    return protocol, fixture, baseline


def validate_report(
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
        or report.get("report_kind")
        != "retrieval_user_journey_qwen_dense_diagnostic_v1"
    ):
        errors.append("Qwen dense diagnostic report identity changed")
    if (
        report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
        or report.get("baseline_sha256") != sha256_file(DEFAULT_BASELINE)
    ):
        errors.append("Qwen dense diagnostic source identity changed")
    calibration_ids = list(fixture["split"]["calibration_case_ids"])
    holdout = set(fixture["split"]["holdout_case_ids"])
    if report.get("calibration_case_ids") != calibration_ids:
        errors.append("Qwen dense diagnostic calibration membership changed")
    channels = report.get("channels") or {}
    if list(channels) != [PRODUCT_CHANNEL, DENSE_CHANNEL]:
        errors.append("Qwen dense diagnostic channel set or order changed")
        return errors
    expected_control = frozen_product_control(baseline, calibration_ids)
    if channels.get(PRODUCT_CHANNEL) != expected_control:
        errors.append("frozen product BM25 control changed")
    expected_ids = set(calibration_ids)
    for channel_name, channel in channels.items():
        for mode_name in ("scoped", "full_library"):
            mode = (channel.get("modes") or {}).get(mode_name) or {}
            observed = {str(row.get("id") or "") for row in mode.get("cases") or []}
            if observed != expected_ids or observed & holdout:
                errors.append(f"{channel_name}/{mode_name} leaked or omitted cases")
            if mode.get("ks") != [5, 10, 20]:
                errors.append(f"{channel_name}/{mode_name} reporting depth changed")
            if any(key not in (mode.get("metrics") or {}) for key in METRIC_KEYS):
                errors.append(f"{channel_name}/{mode_name} metrics incomplete")
    if report.get("frozen_representation") != protocol["frozen_representation"]:
        errors.append("measured Qwen representation changed")
    if report.get("index") != {
        **protocol["existing_index"],
        "rebuilt": False,
    }:
        errors.append("measured Qwen index identity changed or was rebuilt")
    if (report.get("query_embedding_cache") or {}) != {
        "enabled": True,
        "unique_query_count": 12,
        "shared_across_scoped_and_full_library": True,
    }:
        errors.append("Qwen query cache did not encode exactly 12 unique questions")
    measured = evaluate_complementarity(
        channels[PRODUCT_CHANNEL], channels[DENSE_CHANNEL]
    )
    if report.get("analysis") != measured:
        errors.append("Qwen dense complementarity analysis changed")
    fingerprint = _result_fingerprint(channels)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("Qwen dense result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned Qwen dense diagnostic result changed")
    if (
        report.get("holdout_accessed") is not False
        or report.get("fusion_or_reranker_executed") is not False
        or report.get("product_enabled") is not False
    ):
        errors.append("Qwen dense diagnostic crossed its safety boundary")
    return errors


def measure(
    *,
    db_root: Path,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_fixture = calibration_fixture(fixture, protocol)
    chunks = load_all_chunks(db_root)
    _mapping, corpus_errors = validate_indexed_corpus(diagnostic_fixture, chunks)
    if corpus_errors:
        raise ValueError(
            "indexed corpus does not match diagnostic fixture: "
            + "; ".join(corpus_errors)
        )
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

        def dense_factory(selected_chunks: list[dict]) -> DenseRetriever:
            return DenseRetriever(
                selected_chunks,
                [vectors[str(chunk.get("id") or "")] for chunk in selected_chunks],
                cached_query,
                min_score=-1.0,
            )

        dense = _compact_channel(
            evaluate_candidate_channel(
                diagnostic_fixture,
                chunks,
                channel_name=DENSE_CHANNEL,
                retriever_factory=dense_factory,
            )
        )
        product = frozen_product_control(
            baseline, list(fixture["split"]["calibration_case_ids"])
        )
        channels = {PRODUCT_CHANNEL: product, DENSE_CHANNEL: dense}
        runtime = provider.runtime_info()
        return {
            "schema_version": 1,
            "report_kind": "retrieval_user_journey_qwen_dense_diagnostic_v1",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
            "baseline_sha256": sha256_file(DEFAULT_BASELINE),
            "calibration_case_ids": list(fixture["split"]["calibration_case_ids"]),
            "frozen_representation": dict(protocol["frozen_representation"]),
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
                **protocol["existing_index"],
                "size_bytes": _index_size_bytes(DEFAULT_QWEN_INDEX),
                "rebuilt": False,
            },
            "query_embedding_cache": {
                "enabled": True,
                "unique_query_count": cached_query.query_count,
                "shared_across_scoped_and_full_library": True,
            },
            "channels": channels,
            "analysis": evaluate_complementarity(product, dense),
            "result_fingerprint": _result_fingerprint(channels),
            "holdout_accessed": False,
            "fusion_or_reranker_executed": False,
            "product_enabled": False,
        }
    finally:
        _release_provider(provider)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure or validate the calibration-only Qwen dense diagnostic."
    )
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol, fixture, baseline = _load_contracts()
    if args.validate_only:
        try:
            report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[FAIL] Qwen dense diagnostic cannot be read: {type(exc).__name__}")
            return 2
        errors = validate_report(
            report, protocol=protocol, fixture=fixture, baseline=baseline
        )
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print("[OK] frozen calibration-only Qwen dense diagnostic")
        return 0
    if args.db_root is None:
        parser.error("--db-root is required unless --validate-only is used")
    if DEFAULT_REPORT.exists():
        parser.error("Qwen dense diagnostic already exists; use --validate-only")
    report = measure(
        db_root=args.db_root.expanduser().resolve(),
        protocol=protocol,
        fixture=fixture,
        baseline=baseline,
    )
    errors = validate_report(
        report, protocol=protocol, fixture=fixture, baseline=baseline
    )
    if errors:
        raise ValueError(
            "measured Qwen dense diagnostic is invalid: " + "; ".join(errors)
        )
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["analysis"], ensure_ascii=False, indent=2))
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
