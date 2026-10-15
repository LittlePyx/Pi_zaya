from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .candidate_retriever import CandidateRetriever
from .retrieval_eval_fixture import corpus_fingerprint, sha256_text
from .retrieval_metrics import evaluate_ranking, mean_metrics
from .retriever import BM25Retriever


SearchFunction = Callable[
    [str, CandidateRetriever, int, bool],
    tuple[list[dict[str, Any]], dict[str, Any]],
]
RetrieverFactory = Callable[[list[dict]], CandidateRetriever]
ModeRetrieverFactory = Callable[[list[dict], bool], CandidateRetriever]


def raw_bm25_search(
    query: str,
    retriever: CandidateRetriever,
    top_k: int,
    _whole_library: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return retriever.search(query, top_k=top_k), {
        "used_translation": False,
        "query_variant_count": 1,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * max(0.0, min(1.0, float(percentile)))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fixture_doc_map(payload: Mapping[str, Any]) -> dict[str, str]:
    corpus = payload.get("corpus") if isinstance(payload.get("corpus"), Mapping) else {}
    return {
        str(item.get("index_doc_id") or ""): str(item.get("doc_id") or "")
        for item in list(corpus.get("documents") or [])
        if isinstance(item, Mapping)
    }


def validate_indexed_corpus(
    payload: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    index_to_logical = _fixture_doc_map(payload)
    chunk_to_doc: dict[str, str] = {}
    chunks_by_id = {str(item.get("id") or ""): item for item in chunks}
    for chunk_id in chunks_by_id:
        index_doc_id = chunk_id.split(":", 1)[0]
        logical_doc_id = index_to_logical.get(index_doc_id, "")
        if not logical_doc_id:
            errors.append(f"unregistered corpus chunk: {chunk_id}")
            continue
        chunk_to_doc[chunk_id] = logical_doc_id

    corpus = payload.get("corpus") if isinstance(payload.get("corpus"), Mapping) else {}
    expected_count = int(corpus.get("chunk_count") or 0)
    if len(chunks) != expected_count:
        errors.append(f"chunk count is {len(chunks)}, expected {expected_count}")
    actual_fingerprint = corpus_fingerprint(chunks, chunk_to_doc)
    expected_fingerprint = str(corpus.get("fingerprint") or "")
    if actual_fingerprint != expected_fingerprint:
        errors.append(
            f"corpus fingerprint is {actual_fingerprint}, expected {expected_fingerprint}"
        )

    for case in list(payload.get("cases") or []):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case.get("id") or "<missing-id>")
        for label in list(case.get("labels") or []):
            if not isinstance(label, Mapping):
                continue
            chunk_id = str(label.get("chunk_id") or "")
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                errors.append(f"{case_id}: labelled chunk is missing: {chunk_id}")
                continue
            if sha256_text(chunk.get("text")) != str(label.get("text_sha256") or ""):
                errors.append(f"{case_id}: labelled chunk text changed: {chunk_id}")
    return chunk_to_doc, errors


def _contract_recall(
    ranked_ids: Sequence[str],
    contracts: Sequence[Mapping[str, Any]],
    k: int,
) -> float:
    if not contracts:
        return 0.0
    retrieved = {str(item) for item in ranked_ids[: max(0, int(k))]}
    covered = sum(
        1
        for contract in contracts
        if retrieved
        & {str(item) for item in list(contract.get("relevant_chunk_ids") or [])}
    )
    return covered / len(contracts)


def _wrong_source_rate(
    ranked_ids: Sequence[str],
    chunk_to_doc: Mapping[str, str],
    scope_doc_ids: set[str],
    k: int,
) -> float:
    selected = list(ranked_ids[: max(0, int(k))])
    if not selected:
        return 0.0
    wrong = sum(1 for chunk_id in selected if chunk_to_doc.get(str(chunk_id), "") not in scope_doc_ids)
    return wrong / len(selected)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    metric_rows = [
        dict(row.get("metrics") or {})
        for row in rows
        if isinstance(row.get("metrics"), Mapping)
    ]
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    metrics = mean_metrics(metric_rows)
    translation_count = sum(1 for row in rows if bool(row.get("used_translation")))
    return {
        "case_count": len(rows),
        "label_count": sum(int(row.get("label_count") or 0) for row in rows),
        "zero_hit_cases": sum(1 for row in rows if not list(row.get("ranked_chunk_ids") or [])),
        "translated_cases": translation_count,
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.5), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies, default=0.0), 3),
        },
        "ks": [int(item) for item in ks],
    }


def _group_rows(rows: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "locale": {},
        "evaluation_focus": {},
        "query_scope": {},
        "coverage": {},
    }
    for row in rows:
        for dimension in ("locale", "evaluation_focus", "query_scope"):
            value = str(row.get(dimension) or "unknown")
            grouped[dimension].setdefault(value, []).append(row)
        for value in list(row.get("coverage_tags") or []):
            grouped["coverage"].setdefault(str(value), []).append(row)
    return {
        dimension: {
            value: _aggregate_rows(group_rows, ks)
            for value, group_rows in sorted(values.items())
        }
        for dimension, values in grouped.items()
    }


def evaluate_search_channel(
    payload: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    channel_name: str,
    search: SearchFunction,
    chunk_to_doc: Mapping[str, str],
    retriever_factory: RetrieverFactory = BM25Retriever,
    mode_retriever_factory: ModeRetrieverFactory | None = None,
) -> dict[str, Any]:
    ks = tuple(max(1, int(item)) for item in list(payload.get("ks") or [1, 3, 5, 10]))
    max_k = max(ks)
    retrievers: dict[tuple[str, ...], CandidateRetriever] = {}
    rows_by_mode: dict[str, list[dict[str, Any]]] = {"scoped": [], "full_library": []}

    for raw_case in list(payload.get("cases") or []):
        if not isinstance(raw_case, Mapping):
            continue
        scope_doc_ids = {str(item or "") for item in list(raw_case.get("scope_doc_ids") or [])}
        relevance = {
            str(chunk_id): float(grade)
            for chunk_id, grade in dict(raw_case.get("relevance") or {}).items()
            if float(grade) > 0.0
        }
        contracts = [
            item
            for item in list(raw_case.get("contracts") or [])
            if isinstance(item, Mapping)
        ]
        for mode in ("scoped", "full_library"):
            scope_key = tuple(sorted(scope_doc_ids)) if mode == "scoped" else ("*",)
            retriever = retrievers.get(scope_key)
            if retriever is None:
                selected_chunks = [
                    dict(chunk)
                    for chunk in chunks
                    if mode == "full_library"
                    or chunk_to_doc.get(str(chunk.get("id") or ""), "") in scope_doc_ids
                ]
                retriever = (
                    mode_retriever_factory(selected_chunks, mode == "full_library")
                    if mode_retriever_factory is not None
                    else retriever_factory(selected_chunks)
                )
                retrievers[scope_key] = retriever
            started = time.perf_counter()
            hits, details = search(
                str(raw_case.get("query") or ""),
                retriever,
                max_k,
                mode == "full_library",
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            ranked_ids = [str(item.get("id") or "") for item in hits[:max_k]]
            metrics = evaluate_ranking(ranked_ids, relevance, ks=ks)
            for k in ks:
                metrics[f"contract_recall@{k}"] = _contract_recall(ranked_ids, contracts, k)
                metrics[f"wrong_source_rate@{k}"] = _wrong_source_rate(
                    ranked_ids,
                    chunk_to_doc,
                    scope_doc_ids,
                    k,
                )
            first_relevant_rank = next(
                (rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in relevance),
                None,
            )
            rows_by_mode[mode].append(
                {
                    "id": str(raw_case.get("id") or ""),
                    "locale": str(raw_case.get("locale") or ""),
                    "evaluation_focus": str(raw_case.get("evaluation_focus") or ""),
                    "coverage_tags": list(raw_case.get("coverage_tags") or []),
                    "query_scope": str(raw_case.get("query_scope") or ""),
                    "scope_doc_ids": sorted(scope_doc_ids),
                    "label_count": len(relevance),
                    "contract_count": len(contracts),
                    "ranked_chunk_ids": ranked_ids,
                    "ranked_doc_ids": [chunk_to_doc.get(chunk_id, "") for chunk_id in ranked_ids],
                    "first_relevant_rank": first_relevant_rank,
                    "used_translation": bool(details.get("used_translation")),
                    "query_variant_count": int(details.get("query_variant_count") or 1),
                    "latency_ms": round(elapsed_ms, 3),
                    "metrics": {key: round(value, 6) for key, value in metrics.items()},
                }
            )

    return {
        "channel": channel_name,
        "modes": {
            mode: {
                **_aggregate_rows(rows, ks),
                "groups": _group_rows(rows, ks),
                "cases": rows,
            }
            for mode, rows in rows_by_mode.items()
        },
    }


def evaluate_candidate_channel(
    payload: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    channel_name: str,
    retriever_factory: RetrieverFactory | None = None,
    mode_retriever_factory: ModeRetrieverFactory | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate retriever against the frozen scoped/full qrels."""

    chunk_to_doc, corpus_errors = validate_indexed_corpus(payload, chunks)
    if corpus_errors:
        raise ValueError(
            "indexed corpus does not match frozen qrels: " + "; ".join(corpus_errors)
        )

    def direct_search(
        query: str,
        retriever: CandidateRetriever,
        top_k: int,
        _whole_library: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return list(retriever.search(query, top_k=top_k) or []), {
            "used_translation": False,
            "query_variant_count": 1,
        }

    if retriever_factory is None and mode_retriever_factory is None:
        raise ValueError("a candidate retriever factory is required")
    return evaluate_search_channel(
        payload,
        chunks,
        channel_name=channel_name,
        search=direct_search,
        chunk_to_doc=chunk_to_doc,
        retriever_factory=retriever_factory or BM25Retriever,
        mode_retriever_factory=mode_retriever_factory,
    )


def evaluate_bm25_baseline(
    payload: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    product_search: SearchFunction,
) -> dict[str, Any]:
    chunk_to_doc, corpus_errors = validate_indexed_corpus(payload, chunks)
    if corpus_errors:
        raise ValueError("indexed corpus does not match frozen qrels: " + "; ".join(corpus_errors))
    channels = [
        evaluate_search_channel(
            payload,
            chunks,
            channel_name="bm25_raw",
            search=raw_bm25_search,
            chunk_to_doc=chunk_to_doc,
        ),
        evaluate_search_channel(
            payload,
            chunks,
            channel_name="bm25_product_queries",
            search=product_search,
            chunk_to_doc=chunk_to_doc,
        ),
    ]
    corpus = payload.get("corpus") if isinstance(payload.get("corpus"), Mapping) else {}
    return {
        "schema_version": 1,
        "fixture_id": str(payload.get("id") or ""),
        "case_count": len(list(payload.get("cases") or [])),
        "corpus": {
            "document_count": int(corpus.get("document_count") or 0),
            "chunk_count": int(corpus.get("chunk_count") or 0),
            "fingerprint": str(corpus.get("fingerprint") or ""),
        },
        "channels": {item["channel"]: item for item in channels},
    }
