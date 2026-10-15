from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.dense_retriever import DenseRetriever
from kb.embedding_index import EmbeddingIndexStore
from kb.hybrid_retriever import HybridCandidateRetriever
from kb.protected_dense_tail_retriever import ProtectedDenseTailRetriever
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks
from tools.retrieval_eval.run_candidate_screening import (
    _candidate_by_id,
    _provider,
    _screening_fixture,
)
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    _compact_channel,
    _index_size_bytes,
)
from tools.retrieval_eval.run_e5_large_instruct_benchmark import _release_provider
from tools.retrieval_eval.validate_candidate_batch import DEFAULT_BATCH, validate_batch
from tools.retrieval_eval.validate_qwen_fusion_protocol_v2 import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_development_v1.json"
DEFAULT_PREDECESSOR_PROTOCOL = (
    REPO_ROOT / "docs" / "retrieval_candidate_comparison_protocol_v1.json"
)
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
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_qwen_fusion_tuning_v2.json"
PRODUCT_CHANNEL = "bm25_product_queries"
_METRIC_KEYS = ("recall@5", "recall@10", "mrr", "ndcg@10", "wrong_source_rate@5")
_EXPECTED_RESULT_FINGERPRINT = (
    "34436b0f783ec5025d62b5516432c15f4147707bd4eb85e88776b31f84264bc8"
)


class _CachedQueryEmbedder:
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


def _metrics(channel: dict[str, Any], mode: str) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in dict(
            (channel.get("modes") or {}).get(mode, {}).get("metrics") or {}
        ).items()
    }


def evaluate_tuning_gate(
    *, product: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("scoped", "full_library"):
        product_metrics = _metrics(product, mode)
        candidate_metrics = _metrics(candidate, mode)
        modes[mode] = {
            "recall_at_10_gain_vs_product": round(
                candidate_metrics.get("recall@10", 0.0)
                - product_metrics.get("recall@10", 0.0),
                6,
            ),
            "recall_at_5_equal_to_product": candidate_metrics.get("recall@5")
            == product_metrics.get("recall@5"),
            "wrong_source_rate_at_5_equal_to_product": candidate_metrics.get(
                "wrong_source_rate@5"
            )
            == product_metrics.get("wrong_source_rate@5"),
            "recall_at_10_no_regression": candidate_metrics.get("recall@10", 0.0)
            >= product_metrics.get("recall@10", 0.0),
            "mrr_no_regression": candidate_metrics.get("mrr", 0.0)
            >= product_metrics.get("mrr", 0.0),
            "ndcg_at_10_no_regression": candidate_metrics.get("ndcg@10", 0.0)
            >= product_metrics.get("ndcg@10", 0.0),
        }
    checks = {
        "recall_at_5_equal_to_product_both_modes": all(
            row["recall_at_5_equal_to_product"] for row in modes.values()
        ),
        "wrong_source_rate_at_5_equal_to_product_both_modes": all(
            row["wrong_source_rate_at_5_equal_to_product"] for row in modes.values()
        ),
        "recall_at_10_no_regression_both_modes": all(
            row["recall_at_10_no_regression"] for row in modes.values()
        ),
        "mrr_no_regression_both_modes": all(
            row["mrr_no_regression"] for row in modes.values()
        ),
        "ndcg_at_10_no_regression_both_modes": all(
            row["ndcg_at_10_no_regression"] for row in modes.values()
        ),
        "strict_recall_at_10_gain_any_mode": any(
            row["recall_at_10_gain_vs_product"] > 0.0 for row in modes.values()
        ),
    }
    return {"modes": modes, "checks": checks, "passes_all": all(checks.values())}


def _candidate_rank(
    candidate_id: str,
    *,
    channels: dict[str, dict[str, Any]],
    product: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[float, float, float, str]:
    candidate = channels[candidate_id]
    recall_gains = [
        _metrics(candidate, mode)["recall@10"] - _metrics(product, mode)["recall@10"]
        for mode in ("scoped", "full_library")
    ]
    ndcg_gains = [
        _metrics(candidate, mode)["ndcg@10"] - _metrics(product, mode)["ndcg@10"]
        for mode in ("scoped", "full_library")
    ]
    row = next(
        item
        for item in protocol["frozen_candidates"]
        if item["candidate_id"] == candidate_id
    )
    quota = float(row.get("dense_novelty_quota", float("inf")))
    return (-min(recall_gains), -(sum(ndcg_gains) / 2.0), quota, candidate_id)


def select_nominee(
    *,
    channels: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    product = channels[PRODUCT_CHANNEL]
    eligible = sorted(
        candidate_id
        for candidate_id, gate in gates.items()
        if gate.get("passes_all") is True
    )
    ranked = sorted(
        eligible,
        key=lambda candidate_id: _candidate_rank(
            candidate_id,
            channels=channels,
            product=product,
            protocol=protocol,
        ),
    )
    nominee = ranked[0] if ranked else ""
    return {
        "eligible_candidate_ids": eligible,
        "nominee_candidate_id": nominee,
        "confirmation_allowed": bool(nominee),
        "status": "nominee_selected" if nominee else "no_eligible_candidate",
    }


def _result_fingerprint(channels: dict[str, dict[str, Any]]) -> str:
    payload: dict[str, Any] = {}
    for channel_id in sorted(channels):
        modes: dict[str, Any] = {}
        for mode in ("scoped", "full_library"):
            row = (channels[channel_id].get("modes") or {}).get(mode) or {}
            modes[mode] = {
                "metrics": {
                    key: (row.get("metrics") or {}).get(key) for key in _METRIC_KEYS
                },
                "cases": [
                    {
                        "id": case.get("id"),
                        "first_relevant_rank": case.get("first_relevant_rank"),
                        "metrics": case.get("metrics"),
                    }
                    for case in row.get("cases") or []
                ],
            }
        payload[channel_id] = modes
    normalized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def _load_contracts() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    predecessor = json.loads(DEFAULT_PREDECESSOR_PROTOCOL.read_text(encoding="utf-8"))
    batch = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    qwen_report = json.loads(DEFAULT_QWEN_REPORT.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol) + validate_batch(batch)
    if errors:
        raise ValueError("frozen fusion contract invalid: " + "; ".join(errors))
    return protocol, predecessor, batch, qwen_report


def _control_result_equal(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    for mode in ("scoped", "full_library"):
        observed_mode = (observed.get("modes") or {}).get(mode) or {}
        expected_mode = (expected.get("modes") or {}).get(mode) or {}
        if any(
            (observed_mode.get("metrics") or {}).get(key)
            != (expected_mode.get("metrics") or {}).get(key)
            for key in _METRIC_KEYS
        ):
            return False
        observed_cases = [
            (case.get("id"), case.get("first_relevant_rank"), case.get("metrics"))
            for case in observed_mode.get("cases") or []
        ]
        expected_cases = [
            (case.get("id"), case.get("first_relevant_rank"), case.get("metrics"))
            for case in expected_mode.get("cases") or []
        ]
        if observed_cases != expected_cases:
            return False
    return True


def validate_tuning_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    predecessor: dict[str, Any],
    qwen_report: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != "qwen_fusion_tuning_v2"
    ):
        errors.append("fusion tuning report kind changed")
    if report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        errors.append("fusion tuning protocol identity changed")
    if report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE):
        errors.append("fusion tuning fixture identity changed")
    if report.get("qwen_predecessor_report_sha256") != sha256_file(DEFAULT_QWEN_REPORT):
        errors.append("Qwen predecessor result identity changed")
    expected_ids = list(predecessor["inner_split"]["screening_case_ids"])
    hidden_ids = set(predecessor["inner_split"]["confirmation_case_ids"])
    if report.get("tuning_case_ids") != expected_ids:
        errors.append("fusion tuning case membership changed")

    channels = (
        report.get("channels") if isinstance(report.get("channels"), dict) else {}
    )
    candidate_ids = [row["candidate_id"] for row in protocol["frozen_candidates"]]
    if list(channels) != [PRODUCT_CHANNEL, *candidate_ids]:
        errors.append("fusion report channel set or order changed")
    expected_set = set(expected_ids)
    for channel_id, channel in channels.items():
        if channel.get("channel") != channel_id:
            errors.append(f"fusion channel identity changed: {channel_id}")
        for mode in ("scoped", "full_library"):
            mode_row = (channel.get("modes") or {}).get(mode) or {}
            cases = mode_row.get("cases") or []
            observed_ids = {str(case.get("id") or "") for case in cases}
            if observed_ids != expected_set or observed_ids & hidden_ids:
                errors.append(f"{channel_id} {mode} leaked or omitted tuning cases")
            metrics = mode_row.get("metrics") or {}
            if any(key not in metrics for key in _METRIC_KEYS):
                errors.append(f"{channel_id} {mode} metrics incomplete")
    requested_depths = {
        max(
            int(value)
            for value in ((channel.get("modes") or {}).get(mode) or {}).get("ks") or []
        )
        for channel in channels.values()
        for mode in ("scoped", "full_library")
    }
    if requested_depths != {int(protocol["shared_fusion_contract"]["top_k"])}:
        errors.append(
            "fusion retriever request depth did not match the frozen top_k contract"
        )
    expected_product = (qwen_report.get("channels") or {}).get(PRODUCT_CHANNEL) or {}
    if PRODUCT_CHANNEL in channels and not _control_result_equal(
        channels[PRODUCT_CHANNEL], expected_product
    ):
        errors.append("product BM25 control result changed")

    representation = report.get("frozen_representation") or {}
    if representation != protocol["frozen_representation"]:
        errors.append("measured Qwen representation changed")
    index = report.get("index") or {}
    predecessor_index = qwen_report.get("index") or {}
    expected_index = {
        "relative_path": "test_results/retrieval_candidate_screening_v1/qwen3_embedding_0_6b_v1/index_v1",
        "size_bytes": predecessor_index.get("size_bytes"),
        "state": (predecessor_index.get("status") or {}).get("state"),
        "build_id": (predecessor_index.get("status") or {}).get("build_id"),
        "corpus_fingerprint": (predecessor_index.get("status") or {}).get(
            "corpus_fingerprint"
        ),
        "document_count": (predecessor_index.get("status") or {}).get("document_count"),
        "chunk_count": (predecessor_index.get("status") or {}).get("chunk_count"),
        "provider_identity": (predecessor_index.get("status") or {}).get(
            "provider_identity"
        ),
        "rebuilt": False,
    }
    if index != expected_index:
        errors.append("existing Qwen index identity changed or was rebuilt")
    if (report.get("query_embedding_cache") or {}).get("unique_query_count") != 29:
        errors.append("shared query cache did not encode exactly the tuning queries")

    measured_gates = {
        candidate_id: evaluate_tuning_gate(
            product=channels.get(PRODUCT_CHANNEL) or {},
            candidate=channels.get(candidate_id) or {},
        )
        for candidate_id in candidate_ids
    }
    if report.get("tuning_gates") != measured_gates:
        errors.append("fusion tuning gate arithmetic changed")
    expected_decision = select_nominee(
        channels=channels, gates=measured_gates, protocol=protocol
    )
    if report.get("nomination") != expected_decision:
        errors.append("fusion tuning nomination changed")
    fingerprint = _result_fingerprint(channels)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("fusion tuning result fingerprint is inconsistent")
    if _EXPECTED_RESULT_FINGERPRINT and fingerprint != _EXPECTED_RESULT_FINGERPRINT:
        errors.append("pinned fusion tuning result changed")
    if (
        report.get("confirmation_accessed") is not False
        or report.get("product_enabled") is not False
    ):
        errors.append("fusion tuning crossed confirmation or product boundary")
    return errors


def _measure(
    *,
    db_root: Path,
    protocol: dict[str, Any],
    predecessor: dict[str, Any],
    batch: dict[str, Any],
    qwen_report: dict[str, Any],
) -> dict[str, Any]:
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    tuning_fixture = _screening_fixture(fixture, predecessor)
    chunks = load_all_chunks(db_root)
    _mapping, corpus_errors = validate_indexed_corpus(tuning_fixture, chunks)
    if corpus_errors:
        raise ValueError(
            "indexed corpus does not match tuning fixture: " + "; ".join(corpus_errors)
        )

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
        cached_embed_query = _CachedQueryEmbedder(provider.embed_query)

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
                cached_embed_query,
                min_score=-1.0,
            )

        channels: dict[str, dict[str, Any]] = {}
        channels[PRODUCT_CHANNEL] = _compact_channel(
            evaluate_candidate_channel(
                tuning_fixture,
                chunks,
                channel_name=PRODUCT_CHANNEL,
                mode_retriever_factory=product_factory,
            )
        )
        for candidate_row in protocol["frozen_candidates"]:
            candidate_id = candidate_row["candidate_id"]

            def fusion_factory(
                selected_chunks: list[dict],
                whole_library: bool,
                *,
                frozen_row: dict[str, Any] = candidate_row,
            ) -> HybridCandidateRetriever | ProtectedDenseTailRetriever:
                sparse = product_factory(selected_chunks, whole_library)
                dense = dense_factory(selected_chunks)
                if frozen_row["algorithm"] == "rrf":
                    return HybridCandidateRetriever(
                        sparse,
                        dense,
                        sparse_weight=frozen_row["sparse_weight"],
                        dense_weight=frozen_row["dense_weight"],
                        rrf_k=frozen_row["rrf_k"],
                        candidate_multiplier=protocol["shared_fusion_contract"][
                            "candidate_multiplier"
                        ],
                        protected_sparse_prefix=protocol["shared_fusion_contract"][
                            "protected_sparse_prefix"
                        ],
                        dense_source_gate_depth=frozen_row["dense_source_gate_depth"],
                    )
                return ProtectedDenseTailRetriever(
                    sparse,
                    dense,
                    dense_novelty_quota=frozen_row["dense_novelty_quota"],
                    candidate_multiplier=protocol["shared_fusion_contract"][
                        "candidate_multiplier"
                    ],
                    protected_sparse_prefix=protocol["shared_fusion_contract"][
                        "protected_sparse_prefix"
                    ],
                    sparse_tail_window=frozen_row["sparse_tail_window"],
                )

            channels[candidate_id] = _compact_channel(
                evaluate_candidate_channel(
                    tuning_fixture,
                    chunks,
                    channel_name=candidate_id,
                    mode_retriever_factory=fusion_factory,
                )
            )
        runtime = provider.runtime_info()
        index = {
            "relative_path": "test_results/retrieval_candidate_screening_v1/qwen3_embedding_0_6b_v1/index_v1",
            "size_bytes": _index_size_bytes(DEFAULT_QWEN_INDEX),
            "state": status.state,
            "build_id": status.build_id,
            "corpus_fingerprint": status.corpus_fingerprint,
            "document_count": status.document_count,
            "chunk_count": status.chunk_count,
            "provider_identity": status.provider_identity,
            "rebuilt": False,
        }
        gates = {
            candidate_id: evaluate_tuning_gate(
                product=channels[PRODUCT_CHANNEL], candidate=channels[candidate_id]
            )
            for candidate_id in (
                row["candidate_id"] for row in protocol["frozen_candidates"]
            )
        }
        nomination = select_nominee(channels=channels, gates=gates, protocol=protocol)
        result_fingerprint = _result_fingerprint(channels)
        return {
            "schema_version": 1,
            "report_kind": "qwen_fusion_tuning_v2",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
            "qwen_predecessor_report_sha256": sha256_file(DEFAULT_QWEN_REPORT),
            "tuning_case_ids": list(predecessor["inner_split"]["screening_case_ids"]),
            "frozen_representation": dict(protocol["frozen_representation"]),
            "runtime": {
                "model_id": runtime.get("model_id"),
                "revision": runtime.get("revision"),
                "device": runtime.get("device"),
                "dtype": runtime.get("dtype"),
                "allow_download": runtime.get("allow_download"),
                "trust_remote_code": runtime.get("trust_remote_code"),
            },
            "index": index,
            "query_embedding_cache": {
                "enabled": True,
                "unique_query_count": cached_embed_query.query_count,
                "shared_across_all_fusion_candidates": True,
            },
            "channels": channels,
            "tuning_gates": gates,
            "nomination": nomination,
            "result_fingerprint": result_fingerprint,
            "confirmation_accessed": False,
            "product_enabled": False,
        }
    finally:
        _release_provider(provider)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure or validate the one-run frozen Qwen fusion v2 tuning batch."
    )
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol, predecessor, batch, qwen_report = _load_contracts()
    if args.validate_only:
        try:
            report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[FAIL] fusion tuning report cannot be read: {type(exc).__name__}")
            return 2
        errors = validate_tuning_report(
            report,
            protocol=protocol,
            predecessor=predecessor,
            qwen_report=qwen_report,
        )
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print("[OK] frozen Qwen fusion v2 tuning report")
        return 0
    if DEFAULT_REPORT.exists():
        parser.error("fusion tuning report already exists and cannot be overwritten")
    if args.db_root is None:
        parser.error("--db-root is required for the single tuning measurement")
    report = _measure(
        db_root=args.db_root.expanduser().resolve(),
        protocol=protocol,
        predecessor=predecessor,
        batch=batch,
        qwen_report=qwen_report,
    )
    errors = validate_tuning_report(
        report,
        protocol=protocol,
        predecessor=predecessor,
        qwen_report=qwen_report,
    )
    if errors:
        raise ValueError(
            "generated fusion tuning report is invalid: " + "; ".join(errors)
        )
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tuning_gates": report["tuning_gates"],
                "nomination": report["nomination"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
