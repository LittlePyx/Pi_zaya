from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.dense_retriever import DenseRetriever
from kb.embedding_index import EmbeddingIndexStore
from kb.hybrid_retriever import HybridCandidateRetriever
from kb.retrieval_benchmark import evaluate_candidate_channel, validate_indexed_corpus
from kb.retrieval_eval_fixture import sha256_file
from kb.retrieval_semantic_eval_fixture import load_retrieval_benchmark_fixture
from kb.store import load_all_chunks
from tools.retrieval_eval.product_query_sparse_retriever import (
    ProductQuerySparseRetriever,
)
from tools.retrieval_eval.run_dense_research_qa_benchmark import (
    _compact_channel,
    _index_size_bytes,
)
from tools.retrieval_eval.run_e5_large_instruct_benchmark import (
    _release_provider,
    _sha256_binary,
)
from tools.retrieval_eval.transformers_embedding_provider import (
    TransformersPoolingEmbeddingProvider,
)
from tools.retrieval_eval.validate_candidate_batch import (
    DEFAULT_BATCH,
    validate_batch,
)
from tools.retrieval_eval.validate_candidate_comparison_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_development_v1.json"
DEFAULT_BASE_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_eval_development_v1"
    / "multilingual_e5_base_rrf_index"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "test_results" / "retrieval_candidate_screening_v1"
DEFAULT_REPORT_ROOT = REPO_ROOT / "docs"
DEFAULT_NOMINATION_REPORT = REPO_ROOT / "docs" / "retrieval_candidate_nomination_v1.json"
BASE_CHANNEL = "base_e5_single_vector"
PRODUCT_CHANNEL = "bm25_product_queries"
CANDIDATE_CHANNEL = "candidate_single_vector"
SAFE_CHANNEL = "candidate_safe_rrf"
_EXPECTED_CONTROL_METRICS = {
    BASE_CHANNEL: {
        "scoped": (0.344828, 0.568966, 0.240757, 0.305873, 0.0),
        "full_library": (0.258621, 0.413793, 0.180118, 0.228751, 0.531034),
    },
    PRODUCT_CHANNEL: {
        "scoped": (0.62069, 0.741379, 0.444061, 0.512766, 0.0),
        "full_library": (0.482759, 0.637931, 0.280787, 0.361011, 0.296552),
    },
}
_EXPECTED_CANDIDATE_RESULTS = {
    "granite_embedding_278m_multilingual_v1": {
        "channels": {
            CANDIDATE_CHANNEL: {
                "scoped": (0.241379, 0.431034, 0.136569, 0.193968, 0.0),
                "full_library": (0.206897, 0.241379, 0.074578, 0.103147, 0.282759),
            },
            SAFE_CHANNEL: {
                "scoped": (0.62069, 0.758621, 0.446235, 0.516124, 0.0),
                "full_library": (0.482759, 0.568966, 0.276081, 0.33648, 0.296552),
            },
        },
        "index": (2251, 7496695, 7496677, 1.000002),
    },
    "qwen3_embedding_0_6b_v1": {
        "channels": {
            CANDIDATE_CHANNEL: {
                "scoped": (0.586207, 0.741379, 0.337397, 0.430954, 0.0),
                "full_library": (0.482759, 0.62069, 0.255049, 0.338359, 0.193103),
            },
            SAFE_CHANNEL: {
                "scoped": (0.62069, 0.758621, 0.454691, 0.519009, 0.0),
                "full_library": (0.482759, 0.62069, 0.285922, 0.356122, 0.296552),
            },
        },
        "index": (2251, 9801699, 7496677, 1.307472),
    },
}
_PINNED_METRIC_KEYS = ("recall@5", "recall@10", "mrr", "ndcg@10", "wrong_source_rate@5")


def _candidate_report_path(candidate_id: str) -> Path:
    return DEFAULT_REPORT_ROOT / f"retrieval_candidate_screening_{candidate_id}.json"


def _candidate_by_id(batch: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [row for row in batch.get("candidates") or [] if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"candidate is not in the frozen batch: {candidate_id}")
    return dict(matches[0])


def _screening_fixture(
    fixture: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    allowed = set(protocol["inner_split"]["screening_case_ids"])
    cases = [dict(case) for case in fixture.get("cases") or [] if case.get("id") in allowed]
    observed = {str(case.get("id") or "") for case in cases}
    if observed != allowed or len(cases) != protocol["inner_split"]["screening_case_count"]:
        raise ValueError("screening fixture does not exactly match the frozen partition")
    return {
        **fixture,
        "id": "research_qa_library_development_v1_screening_partition_v1",
        "curation_status": "derived_from_frozen_protocol",
        "cases": cases,
    }


def _metrics(channel: dict[str, Any], mode: str) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in dict((channel.get("modes") or {}).get(mode, {}).get("metrics") or {}).items()
    }


def _p95(channel: dict[str, Any], mode: str) -> float:
    return float((channel.get("modes") or {}).get(mode, {}).get("latency_ms", {}).get("p95") or 0.0)


def evaluate_screening_gates(
    *,
    base_dense: dict[str, Any],
    product_sparse: dict[str, Any],
    candidate_dense: dict[str, Any],
    safe_rrf: dict[str, Any],
    index_size_ratio: float,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("scoped", "full_library"):
        base = _metrics(base_dense, mode)
        product = _metrics(product_sparse, mode)
        candidate = _metrics(candidate_dense, mode)
        safe = _metrics(safe_rrf, mode)
        base_p95 = _p95(base_dense, mode)
        candidate_p95 = _p95(candidate_dense, mode)
        latency_limit = max(base_p95 * 2.5, base_p95 + 50.0)
        modes[mode] = {
            "raw_recall_at_10_gain_vs_control": round(
                candidate.get("recall@10", 0.0) - base.get("recall@10", 0.0), 6
            ),
            "raw_recall_at_10_no_regression": candidate.get("recall@10", 0.0)
            >= base.get("recall@10", 0.0),
            "safe_recall_at_10_gain_vs_product": round(
                safe.get("recall@10", 0.0) - product.get("recall@10", 0.0), 6
            ),
            "safe_preserves_product_recall_at_5": safe.get("recall@5")
            == product.get("recall@5"),
            "safe_preserves_product_wrong_source_at_5": safe.get("wrong_source_rate@5")
            == product.get("wrong_source_rate@5"),
            "safe_no_regression_vs_product": all(
                safe.get(metric, 0.0) >= product.get(metric, 0.0)
                for metric in ("recall@10", "mrr", "ndcg@10")
            ),
            "control_p95_ms": round(base_p95, 3),
            "candidate_p95_ms": round(candidate_p95, 3),
            "candidate_p95_limit_ms": round(latency_limit, 3),
            "latency_pass": candidate_p95 <= latency_limit,
        }
    checks = {
        "raw_recall_at_10_no_regression_both_modes": all(
            row["raw_recall_at_10_no_regression"] for row in modes.values()
        ),
        "safe_preserves_product_recall_at_5_both_modes": all(
            row["safe_preserves_product_recall_at_5"] for row in modes.values()
        ),
        "safe_preserves_product_wrong_source_at_5_both_modes": all(
            row["safe_preserves_product_wrong_source_at_5"] for row in modes.values()
        ),
        "safe_no_regression_vs_product_both_modes": all(
            row["safe_no_regression_vs_product"] for row in modes.values()
        ),
        "index_size_ratio_pass": index_size_ratio <= 1.5,
        "latency_pass_both_modes": all(row["latency_pass"] for row in modes.values()),
    }
    return {
        "modes": modes,
        "index_size_ratio_vs_control": round(index_size_ratio, 6),
        "checks": checks,
        "passes_all": all(checks.values()),
    }


def _candidate_rank(report: dict[str, Any]) -> tuple[float, float, float, float, str]:
    channels = report["channels"]
    safe = channels[SAFE_CHANNEL]
    product = channels[PRODUCT_CHANNEL]
    gains = [
        _metrics(safe, mode)["recall@10"] - _metrics(product, mode)["recall@10"]
        for mode in ("scoped", "full_library")
    ]
    ndcg_gains = [
        _metrics(safe, mode)["ndcg@10"] - _metrics(product, mode)["ndcg@10"]
        for mode in ("scoped", "full_library")
    ]
    wrong_source = _metrics(safe, "full_library")["wrong_source_rate@5"]
    average_p95 = sum(_p95(channels[CANDIDATE_CHANNEL], mode) for mode in ("scoped", "full_library")) / 2
    return (
        -min(gains),
        -(sum(ndcg_gains) / len(ndcg_gains)),
        wrong_source,
        average_p95,
        str(report["candidate"]["model_id"]),
    )


def select_nominee(reports: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [report for report in reports if (report.get("screening_gate") or {}).get("passes_all") is True]
    ranked = sorted(eligible, key=_candidate_rank)
    nominee = str(ranked[0]["candidate"]["candidate_id"]) if ranked else ""
    return {
        "eligible_candidate_ids": sorted(
            str(report["candidate"]["candidate_id"]) for report in eligible
        ),
        "nominee_candidate_id": nominee,
        "confirmation_allowed": bool(nominee),
        "status": "nominee_selected" if nominee else "no_eligible_candidate",
    }


def _prepare_candidates(batch: dict[str, Any], cache_dir: Path | None) -> int:
    from huggingface_hub import snapshot_download

    for candidate in batch["candidates"]:
        snapshot = Path(
            snapshot_download(
                candidate["model_id"],
                revision=candidate["revision"],
                cache_dir=str(cache_dir.expanduser().resolve()) if cache_dir else None,
                allow_patterns=[
                    "config.json",
                    "config_sentence_transformers.json",
                    "sentence_bert_config.json",
                    "modules.json",
                    "1_Pooling/config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "sentencepiece.bpe.model",
                    "vocab.json",
                    "merges.txt",
                    "model.safetensors",
                ],
            )
        )
        weight = snapshot / "model.safetensors"
        if weight.stat().st_size != int(candidate["model_safetensors_bytes"]):
            raise RuntimeError(f"{candidate['candidate_id']} safe weight byte size mismatch")
        if _sha256_binary(weight) != candidate["model_safetensors_sha256"]:
            raise RuntimeError(f"{candidate['candidate_id']} safe weight SHA-256 mismatch")
        print(f"[OK] cached and verified {candidate['candidate_id']}: {snapshot}")
    return 0


def _provider(candidate: dict[str, Any]) -> TransformersPoolingEmbeddingProvider:
    installed = Version(version("transformers"))
    minimum = Version(str(candidate["minimum_transformers_version"]))
    if installed < minimum:
        raise RuntimeError(
            f"{candidate['candidate_id']} requires transformers>={minimum}; found {installed}"
        )
    return TransformersPoolingEmbeddingProvider(
        model_id=candidate["model_id"],
        revision=candidate["revision"],
        dimension=candidate["dimension"],
        query_prefix=candidate["query_prefix"],
        passage_prefix=candidate["passage_prefix"],
        max_length=candidate["benchmark_max_length"],
        batch_size=candidate["batch_size"],
        device=candidate["device"],
        dtype=candidate["dtype"],
        pooling=candidate["pooling"],
        padding_side=candidate["padding_side"],
        allow_download=False,
    )


def validate_screening_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    batch: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if report.get("report_kind") != "dense_candidate_screening_v1":
        errors.append("screening report kind changed")
    if report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        errors.append("screening protocol identity changed")
    if report.get("candidate_batch_sha256") != sha256_file(DEFAULT_BATCH):
        errors.append("candidate batch identity changed")
    screening_ids = set(protocol["inner_split"]["screening_case_ids"])
    confirmation_ids = set(protocol["inner_split"]["confirmation_case_ids"])
    if report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE):
        errors.append("development fixture identity changed")
    if report.get("screening_case_ids") != protocol["inner_split"]["screening_case_ids"]:
        errors.append("screening report case membership changed")
    channels = report.get("channels") if isinstance(report.get("channels"), dict) else {}
    for name in (BASE_CHANNEL, PRODUCT_CHANNEL, CANDIDATE_CHANNEL, SAFE_CHANNEL):
        channel = channels.get(name) if isinstance(channels.get(name), dict) else {}
        for mode in ("scoped", "full_library"):
            cases = ((channel.get("modes") or {}).get(mode) or {}).get("cases") or []
            observed = {str(case.get("id") or "") for case in cases}
            if observed != screening_ids or observed & confirmation_ids:
                errors.append(f"{name} {mode} leaked or omitted partition cases")
            metrics = _metrics(channel, mode)
            if not all(
                metric in metrics
                for metric in _PINNED_METRIC_KEYS
            ):
                errors.append(f"{name} {mode} metrics incomplete")
    candidate_row = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    try:
        frozen = _candidate_by_id(batch, str(candidate_row.get("candidate_id") or ""))
    except ValueError as exc:
        errors.append(str(exc))
        frozen = {}
    for key in (
        "candidate_id",
        "model_id",
        "revision",
        "dimension",
        "pooling",
        "padding_side",
        "query_prefix",
        "passage_prefix",
        "max_length",
        "dtype",
    ):
        frozen_key = "benchmark_max_length" if key == "max_length" else key
        if frozen and candidate_row.get(key) != frozen.get(frozen_key):
            errors.append(f"measured candidate {key} changed")
    if frozen:
        for key in ("batch_size", "device", "model_safetensors_bytes", "model_safetensors_sha256"):
            if candidate_row.get(key) != frozen.get(key):
                errors.append(f"measured candidate {key} changed")
        if candidate_row.get("model_fingerprint") != (
            "hf-contract-sha256:" + frozen["encoding_contract_sha256"]
        ):
            errors.append("measured candidate provider fingerprint changed")
    if (
        candidate_row.get("allow_download") is not False
        or candidate_row.get("trust_remote_code") is not False
        or candidate_row.get("product_enabled") is not False
    ):
        errors.append("measured candidate execution boundary changed")
    index = report.get("index") if isinstance(report.get("index"), dict) else {}
    candidate_id = str(candidate_row.get("candidate_id") or "")
    expected_result = _EXPECTED_CANDIDATE_RESULTS.get(candidate_id)
    if expected_result is None:
        errors.append(f"{candidate_id} measured result is not pinned")
    expected_channels = dict(_EXPECTED_CONTROL_METRICS)
    if expected_result:
        expected_channels.update(expected_result["channels"])
    for name, modes in expected_channels.items():
        for mode, values in modes.items():
            metrics = _metrics(channels.get(name) or {}, mode)
            for metric, expected in zip(_PINNED_METRIC_KEYS, values):
                if metrics.get(metric) != expected:
                    errors.append(f"{candidate_id} {name} {mode} measured {metric} changed")
    if expected_result:
        observed_index = (
            index.get("chunk_count"),
            index.get("size_bytes"),
            index.get("control_size_bytes"),
            index.get("size_ratio_vs_control"),
        )
        if observed_index != expected_result["index"]:
            errors.append(f"{candidate_id} measured index contract changed")
    measured_gate = evaluate_screening_gates(
        base_dense=channels.get(BASE_CHANNEL) or {},
        product_sparse=channels.get(PRODUCT_CHANNEL) or {},
        candidate_dense=channels.get(CANDIDATE_CHANNEL) or {},
        safe_rrf=channels.get(SAFE_CHANNEL) or {},
        index_size_ratio=float(index.get("size_ratio_vs_control") or 0.0),
    )
    if report.get("screening_gate") != measured_gate:
        errors.append("screening gate arithmetic changed")
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    if decision.get("eligible_for_nomination") is not bool(measured_gate["passes_all"]):
        errors.append("screening eligibility does not match gates")
    if decision.get("confirmation_accessed") is not False or decision.get("product_enabled") is not False:
        errors.append("screening report crossed confirmation or product boundary")
    return errors


def _validate_nomination_report(
    report: dict[str, Any],
    candidate_reports: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != 1 or report.get("report_kind") != "dense_candidate_nomination_v1":
        errors.append("nomination report kind changed")
    if report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL):
        errors.append("nomination protocol identity changed")
    if report.get("candidate_batch_sha256") != sha256_file(DEFAULT_BATCH):
        errors.append("nomination candidate batch identity changed")
    expected_hashes = {
        str(candidate_report["candidate"]["candidate_id"]): sha256_file(
            _candidate_report_path(candidate_report["candidate"]["candidate_id"])
        )
        for candidate_report in candidate_reports
    }
    if report.get("candidate_report_sha256") != expected_hashes:
        errors.append("nomination candidate report identity changed")
    expected = select_nominee(candidate_reports)
    if report.get("decision") != expected:
        errors.append("nomination decision changed")
    if report.get("confirmation_accessed") is not False or report.get("product_enabled") is not False:
        errors.append("nomination crossed confirmation or product boundary")
    return errors


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    batch = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    errors = validate_protocol(protocol) + validate_batch(batch)
    if errors:
        raise ValueError("frozen screening contract invalid: " + "; ".join(errors))
    return protocol, batch


def _measure_candidate(
    *,
    candidate: dict[str, Any],
    protocol: dict[str, Any],
    batch: dict[str, Any],
    db_root: Path,
) -> dict[str, Any]:
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    screen = _screening_fixture(fixture, protocol)
    chunks = load_all_chunks(db_root)
    _mapping, corpus_errors = validate_indexed_corpus(screen, chunks)
    if corpus_errors:
        raise ValueError("indexed corpus does not match screening fixture: " + "; ".join(corpus_errors))

    base = protocol["controls"]
    base_provider = TransformersPoolingEmbeddingProvider(
        model_id=base["dense_model_id"],
        revision=base["dense_revision"],
        dimension=768,
        query_prefix=base["dense_query_prefix"],
        passage_prefix=base["dense_passage_prefix"],
        max_length=512,
        batch_size=4,
        device="cuda",
        dtype="float16",
        allow_download=False,
    )
    base_provider.prepare()
    base_store = EmbeddingIndexStore(db_root, index_dir=DEFAULT_BASE_INDEX)
    base_loaded = base_store.load(base_provider)
    base_vectors = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(base_loaded.chunks, base_loaded.embeddings)
    }

    def base_factory(selected_chunks: list[dict]) -> DenseRetriever:
        return DenseRetriever(
            selected_chunks,
            [base_vectors[str(chunk.get("id") or "")] for chunk in selected_chunks],
            base_provider.embed_query,
            min_score=-1.0,
        )

    def product_factory(
        selected_chunks: list[dict], whole_library: bool
    ) -> ProductQuerySparseRetriever:
        return ProductQuerySparseRetriever(selected_chunks, whole_library=whole_library)

    base_channel = evaluate_candidate_channel(
        screen, chunks, channel_name=BASE_CHANNEL, retriever_factory=base_factory
    )
    product_channel = evaluate_candidate_channel(
        screen,
        chunks,
        channel_name=PRODUCT_CHANNEL,
        mode_retriever_factory=product_factory,
    )
    base_size = _index_size_bytes(base_store.index_dir)
    base_runtime = base_provider.runtime_info()
    _release_provider(base_provider)

    provider = _provider(candidate)
    provider.prepare()
    index_dir = DEFAULT_OUT_ROOT / candidate["candidate_id"] / "index_v1"
    store = EmbeddingIndexStore(db_root, index_dir=index_dir)
    build = store.build(provider)
    loaded = store.load(provider)
    vectors = {
        str(chunk.get("id") or ""): vector
        for chunk, vector in zip(loaded.chunks, loaded.embeddings)
    }

    def candidate_factory(selected_chunks: list[dict]) -> DenseRetriever:
        return DenseRetriever(
            selected_chunks,
            [vectors[str(chunk.get("id") or "")] for chunk in selected_chunks],
            provider.embed_query,
            min_score=-1.0,
        )

    def safe_factory(selected_chunks: list[dict], whole_library: bool) -> HybridCandidateRetriever:
        return HybridCandidateRetriever(
            product_factory(selected_chunks, whole_library),
            candidate_factory(selected_chunks),
            sparse_weight=2.0,
            dense_weight=1.0,
            rrf_k=20.0,
            candidate_multiplier=4,
            protected_sparse_prefix=5,
            dense_source_gate_depth=5,
        )

    candidate_channel = evaluate_candidate_channel(
        screen, chunks, channel_name=CANDIDATE_CHANNEL, retriever_factory=candidate_factory
    )
    safe_channel = evaluate_candidate_channel(
        screen, chunks, channel_name=SAFE_CHANNEL, mode_retriever_factory=safe_factory
    )
    candidate_size = _index_size_bytes(store.index_dir)
    size_ratio = candidate_size / max(1, base_size)
    gate = evaluate_screening_gates(
        base_dense=base_channel,
        product_sparse=product_channel,
        candidate_dense=candidate_channel,
        safe_rrf=safe_channel,
        index_size_ratio=size_ratio,
    )
    runtime = provider.runtime_info()
    return {
        "schema_version": 1,
        "report_kind": "dense_candidate_screening_v1",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "candidate_batch_sha256": sha256_file(DEFAULT_BATCH),
        "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
        "screening_case_ids": list(protocol["inner_split"]["screening_case_ids"]),
        "candidate": {
            **runtime,
            "candidate_id": candidate["candidate_id"],
            "model_safetensors_bytes": candidate["model_safetensors_bytes"],
            "model_safetensors_sha256": candidate["model_safetensors_sha256"],
            "product_enabled": False,
        },
        "control_provider": base_runtime,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": version("transformers"),
        },
        "index": {
            **asdict(build),
            "size_bytes": candidate_size,
            "control_size_bytes": base_size,
            "size_ratio_vs_control": round(size_ratio, 6),
            "status": asdict(store.status(provider)),
        },
        "channels": {
            BASE_CHANNEL: _compact_channel(base_channel),
            PRODUCT_CHANNEL: _compact_channel(product_channel),
            CANDIDATE_CHANNEL: _compact_channel(candidate_channel),
            SAFE_CHANNEL: _compact_channel(safe_channel),
        },
        "screening_gate": gate,
        "decision": {
            "status": "eligible" if gate["passes_all"] else "rejected",
            "eligible_for_nomination": bool(gate["passes_all"]),
            "confirmation_accessed": False,
            "product_enabled": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, screen, validate, or nominate the frozen dense candidate batch."
    )
    parser.add_argument("--db-root", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--prepare-models", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--nominate", action="store_true")
    args = parser.parse_args()
    protocol, batch = _load_contracts()
    if args.prepare_models:
        if not args.allow_download:
            parser.error("--prepare-models requires explicit --allow-download")
        if args.candidate_id or args.nominate or args.validate_only:
            parser.error("model preparation cannot be combined with measurement actions")
        return _prepare_candidates(batch, args.cache_dir)
    if args.allow_download:
        parser.error("downloads are allowed only during --prepare-models")
    if args.nominate:
        reports: list[dict[str, Any]] = []
        for candidate in batch["candidates"]:
            path = _candidate_report_path(candidate["candidate_id"])
            report = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_screening_report(report, protocol=protocol, batch=batch)
            if errors:
                raise ValueError(f"invalid {path.name}: " + "; ".join(errors))
            reports.append(report)
        if args.validate_only:
            nomination = json.loads(DEFAULT_NOMINATION_REPORT.read_text(encoding="utf-8"))
            errors = _validate_nomination_report(nomination, reports)
            if errors:
                print("[FAIL] " + "; ".join(errors))
                return 1
            print("[OK] frozen candidate nomination report")
            return 0
        if DEFAULT_NOMINATION_REPORT.exists():
            parser.error("nomination report already exists and cannot be overwritten")
        nomination = {
            "schema_version": 1,
            "report_kind": "dense_candidate_nomination_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "candidate_batch_sha256": sha256_file(DEFAULT_BATCH),
            "candidate_report_sha256": {
                candidate["candidate_id"]: sha256_file(_candidate_report_path(candidate["candidate_id"]))
                for candidate in batch["candidates"]
            },
            "decision": select_nominee(reports),
            "confirmation_accessed": False,
            "product_enabled": False,
        }
        DEFAULT_NOMINATION_REPORT.write_text(
            json.dumps(nomination, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(nomination["decision"], ensure_ascii=False, indent=2))
        print(f"[OK] wrote {DEFAULT_NOMINATION_REPORT}")
        return 0
    if not args.candidate_id:
        parser.error("--candidate-id is required for screening or report validation")
    candidate = _candidate_by_id(batch, args.candidate_id)
    report_path = _candidate_report_path(candidate["candidate_id"])
    if args.validate_only:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        errors = validate_screening_report(report, protocol=protocol, batch=batch)
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(f"[OK] frozen screening report: {candidate['candidate_id']}")
        return 0
    if report_path.exists():
        parser.error("candidate screening report already exists and cannot be overwritten")
    if args.db_root is None:
        parser.error("--db-root is required for screening")
    report = _measure_candidate(
        candidate=candidate,
        protocol=protocol,
        batch=batch,
        db_root=args.db_root.expanduser().resolve(),
    )
    errors = validate_screening_report(report, protocol=protocol, batch=batch)
    if errors:
        raise ValueError("generated screening report is invalid: " + "; ".join(errors))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screening_gate": report["screening_gate"], "decision": report["decision"]}, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
