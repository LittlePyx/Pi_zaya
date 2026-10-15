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
from tools.retrieval_eval.run_user_journey_fusion_v3_selection import (
    FUSION_CHANNEL,
    PRODUCT_CHANNEL,
    compact_selection_channel,
)
from tools.retrieval_eval.run_user_journey_qwen_dense_diagnostic import (
    CachedQueryEmbedder,
    DEFAULT_QWEN_INDEX,
)
from tools.retrieval_eval.validate_candidate_batch import DEFAULT_BATCH, validate_batch
from tools.retrieval_eval.validate_user_journey_fusion_v3_1_semantic_protocol_v2 import (
    DEFAULT_PROTOCOL,
    effective_protocol,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_PREDECESSOR = (
    REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_benchmark_v1.json"
)
DEFAULT_SELECTION = (
    REPO_ROOT / "docs" / "retrieval_user_journey_fusion_v3_1_selection.json"
)
DEFAULT_SEMANTIC_INDEX = (
    REPO_ROOT
    / "test_results"
    / "retrieval_user_journey_fusion_v3_1_semantic"
    / "qwen_index_v1"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "docs"
    / "retrieval_user_journey_fusion_v3_1_semantic_acceptance_v2.json"
)
EXPECTED_RESULT_FINGERPRINT = (
    "a3c81c2805cd887011219b68a33e6bbf3ed8688f5209b8483263e1a41ce2847a"
)
EXPECTED_INDEX: dict[str, Any] = {
    "relative_path": (
        "test_results/retrieval_user_journey_fusion_v3_1_semantic/qwen_index_v1"
    ),
    "build_id": "21e9718aa59f483988f3e88571a3c01c",
    "corpus_fingerprint": (
        "19dd55b81535593219b0fdb8744b6c74cdfe51d68c3567dfd822780be927ec7a"
    ),
    "document_count": 10,
    "chunk_count": 1673,
    "embedded_document_count": 10,
    "embedded_chunk_count": 1673,
    "reused_document_count": 0,
    "reused_chunk_count": 0,
    "no_op": False,
    "provider_identity": (
        "55229348d5bbe4b1e1935aeb8ffe57cc03c527685b1d08d44f11a56a54525cb8"
    ),
    "state": "ready",
    "size_bytes": 7277202,
    "build_performed": True,
}


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def semantic_corpus_chunks(fixture: dict[str, Any], chunks: list[dict]) -> list[dict]:
    index_ids = {str(row["index_doc_id"]) for row in fixture["corpus"]["documents"]}
    selected = [
        dict(chunk)
        for chunk in chunks
        if str(chunk.get("id") or "").split(":", 1)[0] in index_ids
    ]
    if len(selected) != fixture["corpus"]["chunk_count"]:
        raise ValueError(
            f"semantic corpus has {len(selected)} chunks, expected {fixture['corpus']['chunk_count']}"
        )
    return selected


def evaluate_acceptance(
    product: dict[str, Any], candidate: dict[str, Any], *, minimum_gain: float = 0.15
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name in ("scoped", "full_library"):
        product_metrics = product["modes"][mode_name]["metrics"]
        candidate_metrics = candidate["modes"][mode_name]["metrics"]
        gain = round(candidate_metrics["recall@10"] - product_metrics["recall@10"], 6)
        modes[mode_name] = {
            "recall@10_gain": gain,
            "minimum_gain_passes": gain >= minimum_gain,
            "recall@5_equal_to_product": candidate_metrics["recall@5"]
            == product_metrics["recall@5"],
            "wrong_source_rate@5_equal_to_product": candidate_metrics[
                "wrong_source_rate@5"
            ]
            == product_metrics["wrong_source_rate@5"],
        }
    return {
        "metric": "recall@10",
        "minimum_gain_in_each_mode": minimum_gain,
        "modes": modes,
        "passes_all": all(
            row["minimum_gain_passes"]
            and row["recall@5_equal_to_product"]
            and row["wrong_source_rate@5_equal_to_product"]
            for row in modes.values()
        ),
    }


def validate_product_control(
    product: dict[str, Any], predecessor: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    previous = predecessor["channels"]["product_bm25"]
    for mode_name in ("scoped", "full_library"):
        observed = product["modes"][mode_name]["metrics"]
        for metric in ("recall@5", "recall@10", "wrong_source_rate@5"):
            if observed[metric] != previous[mode_name][metric]:
                errors.append(f"semantic product control changed: {mode_name}/{metric}")
    return errors


def _result_fingerprint(channels: dict[str, Any]) -> str:
    projection: dict[str, Any] = {}
    for channel_name, channel in channels.items():
        projection[channel_name] = {}
        for mode_name, mode in channel["modes"].items():
            projection[channel_name][mode_name] = {
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
    canonical = json.dumps(
        projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
    expected_result_fingerprint: str | None = EXPECTED_RESULT_FINGERPRINT,
    expected_index: dict[str, Any] | None = EXPECTED_INDEX,
) -> list[str]:
    errors: list[str] = []
    if (
        report.get("schema_version") != 1
        or report.get("report_kind")
        != "retrieval_user_journey_fusion_v3_1_semantic_acceptance_v2"
    ):
        errors.append("semantic acceptance report identity changed")
    if (
        report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or report.get("fixture_sha256") != sha256_file(DEFAULT_FIXTURE)
        or report.get("predecessor_sha256") != sha256_file(DEFAULT_PREDECESSOR)
        or report.get("selection_sha256") != sha256_file(DEFAULT_SELECTION)
    ):
        errors.append("semantic acceptance source identity changed")
    expected_ids = [str(case["id"]) for case in fixture["cases"]]
    if report.get("case_ids") != expected_ids:
        errors.append("semantic acceptance case membership changed")
    if report.get("candidate") != protocol["unchanged_candidate"]:
        errors.append("semantic acceptance candidate changed")
    channels = report.get("channels") or {}
    if list(channels) != [PRODUCT_CHANNEL, FUSION_CHANNEL]:
        errors.append("semantic acceptance channels changed")
        return errors
    for channel_name, channel in channels.items():
        for mode_name in ("scoped", "full_library"):
            mode = channel["modes"][mode_name]
            if mode.get("ks") != [5, 10]:
                errors.append(f"semantic {channel_name}/{mode_name} depth changed")
            if [str(row["id"]) for row in mode.get("cases") or []] != expected_ids:
                errors.append(f"semantic {channel_name}/{mode_name} cases changed")
    errors.extend(validate_product_control(channels[PRODUCT_CHANNEL], predecessor))
    expected_acceptance = evaluate_acceptance(
        channels[PRODUCT_CHANNEL],
        channels[FUSION_CHANNEL],
        minimum_gain=float(
            protocol["acceptance_gate"]["minimum_gain_vs_product_in_each_mode"]
        ),
    )
    if report.get("acceptance") != expected_acceptance:
        errors.append("semantic acceptance gate arithmetic changed")
    index = report.get("index") or {}
    if (
        index.get("relative_path") != protocol["dedicated_index"]["relative_path"]
        or index.get("provider_identity")
        != protocol["dedicated_index"]["provider_identity"]
        or index.get("document_count") != fixture["corpus"]["document_count"]
        or index.get("chunk_count") != fixture["corpus"]["chunk_count"]
        or index.get("embedded_chunk_count") != fixture["corpus"]["chunk_count"]
        or index.get("no_op") is not False
        or index.get("build_performed") is not True
        or index.get("state") != "ready"
    ):
        errors.append("semantic acceptance dedicated index contract changed")
    if expected_index and index != expected_index:
        errors.append("pinned semantic acceptance index changed")
    product_index = report.get("product_index") or {}
    if (
        not product_index.get("before_sha256")
        or product_index.get("before_sha256") != product_index.get("after_sha256")
        or product_index.get("unchanged") is not True
    ):
        errors.append("semantic acceptance changed the product Qwen index")
    if report.get("query_embedding_cache") != {
        "enabled": True,
        "unique_query_count": 12,
        "shared_across_scoped_and_full_library": True,
    }:
        errors.append("semantic acceptance query cache changed")
    fingerprint = _result_fingerprint(channels)
    if report.get("result_fingerprint") != fingerprint:
        errors.append("semantic acceptance result fingerprint is inconsistent")
    if expected_result_fingerprint and fingerprint != expected_result_fingerprint:
        errors.append("pinned semantic acceptance result changed")
    expected_decision = {
        "status": "passed_for_product_shadow"
        if expected_acceptance["passes_all"]
        else "rejected_for_product_shadow",
        "product_shadow_allowed": bool(expected_acceptance["passes_all"]),
        "product_enabled": False,
        "ordinary_user_surface_exposed": False,
    }
    if report.get("decision") != expected_decision:
        errors.append("semantic acceptance decision changed")
    return errors


def measure(
    *,
    semantic_db_root: Path,
    protocol: dict[str, Any],
    fixture: dict[str, Any],
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    all_chunks = load_all_chunks(semantic_db_root)
    chunks = semantic_corpus_chunks(fixture, all_chunks)
    semantic_fixture = {**fixture, "ks": [5, 10]}
    _mapping, corpus_errors = validate_indexed_corpus(semantic_fixture, chunks)
    if corpus_errors:
        raise ValueError("semantic corpus mismatch: " + "; ".join(corpus_errors))
    batch = json.loads(DEFAULT_BATCH.read_text(encoding="utf-8"))
    batch_errors = validate_batch(batch)
    if batch_errors:
        raise ValueError("frozen candidate batch invalid: " + "; ".join(batch_errors))
    candidate = _candidate_by_id(batch, "qwen3_embedding_0_6b_v1")
    product_index_before = _tree_fingerprint(DEFAULT_QWEN_INDEX)
    if DEFAULT_SEMANTIC_INDEX.exists():
        raise ValueError(
            "dedicated semantic Qwen index already exists before its initial build"
        )
    provider = _provider(candidate)
    provider.prepare()
    try:
        store = EmbeddingIndexStore(semantic_db_root, index_dir=DEFAULT_SEMANTIC_INDEX)
        build = store.build(provider)
        status = store.status(provider)
        if status.state != "ready":
            raise ValueError(f"existing Qwen index is not ready: {status.state}")
        loaded = store.load(provider)
        vectors = {
            str(chunk.get("id") or ""): vector
            for chunk, vector in zip(loaded.chunks, loaded.embeddings)
        }
        cached_query = CachedQueryEmbedder(provider.embed_query)

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

        frozen = protocol["unchanged_candidate"]

        def fusion_factory(
            selected_chunks: list[dict], whole_library: bool
        ) -> ProtectedDenseTailRetriever:
            return ProtectedDenseTailRetriever(
                product_factory(selected_chunks, whole_library),
                dense_factory(selected_chunks),
                dense_novelty_quota=frozen["dense_novelty_quota"],
                candidate_multiplier=frozen["candidate_multiplier"],
                protected_sparse_prefix=frozen["protected_sparse_prefix"],
                sparse_tail_window=frozen["sparse_tail_window"],
            )

        product = compact_selection_channel(
            evaluate_candidate_channel(
                semantic_fixture,
                chunks,
                channel_name=PRODUCT_CHANNEL,
                mode_retriever_factory=product_factory,
            )
        )
        control_errors = validate_product_control(product, predecessor)
        if control_errors:
            raise ValueError(
                "semantic product control drifted: " + "; ".join(control_errors)
            )
        fusion = compact_selection_channel(
            evaluate_candidate_channel(
                semantic_fixture,
                chunks,
                channel_name=FUSION_CHANNEL,
                mode_retriever_factory=fusion_factory,
            )
        )
        channels = {PRODUCT_CHANNEL: product, FUSION_CHANNEL: fusion}
        acceptance = evaluate_acceptance(
            product,
            fusion,
            minimum_gain=float(
                protocol["acceptance_gate"]["minimum_gain_vs_product_in_each_mode"]
            ),
        )
        product_index_after = _tree_fingerprint(DEFAULT_QWEN_INDEX)
        return {
            "schema_version": 1,
            "report_kind": "retrieval_user_journey_fusion_v3_1_semantic_acceptance_v2",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "fixture_sha256": sha256_file(DEFAULT_FIXTURE),
            "predecessor_sha256": sha256_file(DEFAULT_PREDECESSOR),
            "selection_sha256": sha256_file(DEFAULT_SELECTION),
            "case_ids": [str(case["id"]) for case in fixture["cases"]],
            "candidate": dict(protocol["unchanged_candidate"]),
            "index": {
                "relative_path": protocol["dedicated_index"]["relative_path"],
                "build_id": build.build_id,
                "corpus_fingerprint": build.corpus_fingerprint,
                "document_count": build.document_count,
                "chunk_count": build.chunk_count,
                "embedded_document_count": build.embedded_document_count,
                "embedded_chunk_count": build.embedded_chunk_count,
                "reused_document_count": build.reused_document_count,
                "reused_chunk_count": build.reused_chunk_count,
                "no_op": build.no_op,
                "provider_identity": status.provider_identity,
                "state": status.state,
                "size_bytes": _index_size_bytes(DEFAULT_SEMANTIC_INDEX),
                "build_performed": True,
            },
            "product_index": {
                "before_sha256": product_index_before,
                "after_sha256": product_index_after,
                "unchanged": product_index_before == product_index_after,
            },
            "query_embedding_cache": {
                "enabled": True,
                "unique_query_count": cached_query.query_count,
                "shared_across_scoped_and_full_library": True,
            },
            "channels": channels,
            "acceptance": acceptance,
            "result_fingerprint": _result_fingerprint(channels),
            "decision": {
                "status": "passed_for_product_shadow"
                if acceptance["passes_all"]
                else "rejected_for_product_shadow",
                "product_shadow_allowed": bool(acceptance["passes_all"]),
                "product_enabled": False,
                "ordinary_user_surface_exposed": False,
            },
        }
    finally:
        _release_provider(provider)


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    replacement_protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    fixture = load_retrieval_benchmark_fixture(DEFAULT_FIXTURE, repo_root=REPO_ROOT)
    predecessor = json.loads(DEFAULT_PREDECESSOR.read_text(encoding="utf-8"))
    errors = validate_protocol(replacement_protocol)
    if errors:
        raise ValueError("frozen semantic protocol invalid: " + "; ".join(errors))
    protocol = effective_protocol(replacement_protocol)
    return protocol, fixture, predecessor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or validate the one-run fusion-v3.1 semantic acceptance."
    )
    parser.add_argument("--semantic-db-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol, fixture, predecessor = _load_contracts()
    if args.validate_only:
        try:
            report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[FAIL] semantic acceptance report cannot be read: {type(exc).__name__}"
            )
            return 2
        errors = validate_report(
            report, protocol=protocol, fixture=fixture, predecessor=predecessor
        )
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(f"[OK] fusion-v3.1 semantic acceptance: {report['decision']['status']}")
        return 0
    if args.semantic_db_root is None:
        parser.error("--semantic-db-root is required unless --validate-only is used")
    if DEFAULT_REPORT.exists():
        parser.error("semantic acceptance report already exists; use --validate-only")
    report = measure(
        semantic_db_root=args.semantic_db_root.expanduser().resolve(),
        protocol=protocol,
        fixture=fixture,
        predecessor=predecessor,
    )
    errors = validate_report(
        report, protocol=protocol, fixture=fixture, predecessor=predecessor
    )
    if errors:
        raise ValueError(
            "measured semantic acceptance is invalid: " + "; ".join(errors)
        )
    DEFAULT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"acceptance": report["acceptance"], "decision": report["decision"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[OK] wrote {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
