from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .tokenize import tokenize


RESEARCH_QA_QRELS_SCHEMA_VERSION = 2
LABEL_POLICY_VERSION = 1
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def sha256_file(path: Path) -> str:
    # These fixtures are UTF-8 JSON tracked by Git. Hash normalized text bytes
    # so Windows CRLF checkout policy and Linux LF checkout policy share one
    # durable identity in CI and release artifacts.
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _logical_source_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\\", "/")
    return re.sub(r"/+", "/", text).casefold().strip()


def _match_surface(value: object) -> str:
    return " ".join(tokenize(unicodedata.normalize("NFKC", str(value or ""))))


def _term_match(term: str, text: str) -> tuple[bool, bool, float]:
    term_surface = _match_surface(term)
    text_surface = _match_surface(text)
    if not term_surface or not text_surface:
        return False, False, 0.0
    exact = f" {term_surface} " in f" {text_surface} "
    term_tokens = set(term_surface.split())
    text_tokens = set(text_surface.split())
    coverage = len(term_tokens & text_tokens) / len(term_tokens) if term_tokens else 0.0
    threshold = 1.0 if len(term_tokens) <= 2 else 0.8
    return exact or coverage >= threshold, exact, coverage


def _page_span(chunk: Mapping[str, Any]) -> tuple[int, int]:
    meta = chunk.get("meta") if isinstance(chunk.get("meta"), Mapping) else {}
    try:
        start = int(meta.get("page_start") or meta.get("page") or 0)
    except (TypeError, ValueError):
        start = 0
    try:
        end = int(meta.get("page_end") or start)
    except (TypeError, ValueError):
        end = start
    if start > 0 and end <= 0:
        end = start
    if start > end > 0:
        start, end = end, start
    return start, end


def _source_suffix(doc: Mapping[str, Any]) -> str:
    directory = str(doc.get("dir") or "").strip()
    stem = str(doc.get("file") or directory).strip()
    return _logical_source_key(f"/{directory}/{stem}.en.md")


def map_chunks_to_logical_docs(
    chunks: Sequence[Mapping[str, Any]],
    docs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    suffixes = {
        str(doc.get("id") or "").strip(): _source_suffix(doc)
        for doc in docs
        if str(doc.get("id") or "").strip()
    }
    chunk_to_doc: dict[str, str] = {}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in suffixes}
    for raw_chunk in chunks:
        chunk = dict(raw_chunk)
        chunk_id = str(chunk.get("id") or "").strip()
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), Mapping) else {}
        source_key = _logical_source_key(meta.get("source_path"))
        matches = [doc_id for doc_id, suffix in suffixes.items() if source_key.endswith(suffix)]
        if len(matches) != 1:
            continue
        logical_doc_id = matches[0]
        if chunk_id:
            chunk_to_doc[chunk_id] = logical_doc_id
        chunks_by_doc[logical_doc_id].append(chunk)
    return chunk_to_doc, chunks_by_doc


def corpus_fingerprint(
    chunks: Sequence[Mapping[str, Any]],
    chunk_to_doc: Mapping[str, str],
) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: str(item.get("id") or "")):
        chunk_id = str(chunk.get("id") or "").strip()
        logical_doc_id = str(chunk_to_doc.get(chunk_id) or "")
        if not chunk_id or not logical_doc_id:
            continue
        page_start, page_end = _page_span(chunk)
        record = "\x1f".join(
            [
                logical_doc_id,
                chunk_id,
                sha256_text(chunk.get("text")),
                str(page_start),
                str(page_end),
            ]
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _contract_labels(
    contract: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_chunks: int = 3,
    allow_missing_source_page: bool = False,
) -> list[dict[str, Any]]:
    contract_id = str(contract.get("id") or "").strip()
    doc_id = str(contract.get("docId") or "").strip()
    try:
        source_page = int(contract.get("sourcePage") or 0)
    except (TypeError, ValueError):
        source_page = 0
    terms = [str(item or "").strip() for item in list(contract.get("evidenceTerms") or [])]
    terms = [item for item in terms if item]
    if (
        not contract_id
        or not doc_id
        or (source_page <= 0 and not allow_missing_source_page)
        or not terms
    ):
        raise ValueError(f"invalid claim-evidence contract: {contract_id or '<missing-id>'}")

    candidates: list[dict[str, Any]] = []
    for chunk in chunks:
        page_start, page_end = _page_span(chunk)
        if source_page > 0 and not (page_start <= source_page <= page_end):
            continue
        text = str(chunk.get("text") or "")
        matched_indexes: set[int] = set()
        exact_indexes: set[int] = set()
        coverage_sum = 0.0
        for index, term in enumerate(terms):
            matched, exact, coverage = _term_match(term, text)
            if matched:
                matched_indexes.add(index)
                coverage_sum += coverage
            if exact:
                exact_indexes.add(index)
        if not matched_indexes:
            continue
        candidates.append(
            {
                "chunk": chunk,
                "matched": matched_indexes,
                "exact": exact_indexes,
                "coverage_sum": coverage_sum,
            }
        )

    if not candidates:
        raise ValueError(
            f"contract {contract_id} has no evidence-term match on {doc_id} page {source_page}"
        )

    selected: list[dict[str, Any]] = []
    covered: set[int] = set()
    remaining = list(candidates)
    while remaining and len(selected) < max(1, int(max_chunks)):
        remaining.sort(
            key=lambda item: (
                len(item["matched"] - covered),
                len(item["matched"]),
                len(item["exact"]),
                float(item["coverage_sum"]),
                -len(str(item["chunk"].get("text") or "")),
                str(item["chunk"].get("id") or ""),
            ),
            reverse=True,
        )
        best = remaining.pop(0)
        if not (best["matched"] - covered):
            break
        selected.append(best)
        covered.update(best["matched"])
        if len(covered) == len(terms):
            break

    minimum_coverage = max(1, math.ceil(len(terms) / 2))
    if len(covered) < minimum_coverage:
        raise ValueError(
            f"contract {contract_id} covers only {len(covered)}/{len(terms)} registered evidence terms"
        )

    labels: list[dict[str, Any]] = []
    for item in selected:
        chunk = item["chunk"]
        chunk_id = str(chunk.get("id") or "").strip()
        page_start, page_end = _page_span(chunk)
        matched_terms = [terms[index] for index in sorted(item["matched"])]
        matched_fraction = len(item["matched"]) / len(terms)
        grade = 3 if matched_fraction >= 0.5 or len(item["exact"]) >= 2 else 2
        labels.append(
            {
                "chunk_id": chunk_id,
                "grade": grade,
                "doc_id": doc_id,
                "page_start": page_start,
                "page_end": page_end,
                "contract_ids": [contract_id],
                "matched_evidence_terms": matched_terms,
                "text_sha256": sha256_text(chunk.get("text")),
            }
        )
    return labels


def project_research_qa_qrels(
    source_payload: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    source_fixture_path: str,
    source_fixture_sha256: str,
    allow_unmapped_chunks: bool = False,
    skip_cases_without_contracts: bool = False,
    fixture_id: str = "research_qa_unseen_corpus_v2_chunk_qrels",
    curation_status: str = "frozen_pre_registered_contract_projection",
    description: str | None = None,
    allow_missing_source_page: bool = False,
    source_grounded_only: bool = False,
) -> dict[str, Any]:
    docs = [dict(item) for item in list(source_payload.get("docs") or []) if isinstance(item, Mapping)]
    cases = [dict(item) for item in list(source_payload.get("cases") or []) if isinstance(item, Mapping)]
    benchmark = source_payload.get("benchmark") if isinstance(source_payload.get("benchmark"), Mapping) else {}
    suites = source_payload.get("suites") if isinstance(source_payload.get("suites"), Mapping) else {}
    coverage_by_case: dict[str, set[str]] = {}
    for suite in suites.values():
        if not isinstance(suite, Mapping):
            continue
        coverage = suite.get("coverage") if isinstance(suite.get("coverage"), Mapping) else {}
        for tag, tagged_case_ids in coverage.items():
            for tagged_case_id in list(tagged_case_ids or []):
                coverage_by_case.setdefault(str(tagged_case_id), set()).add(str(tag))
    corpus_papers = {
        str(item.get("id") or ""): str(item.get("sha256") or "")
        for item in list(benchmark.get("corpus") or [])
        if isinstance(item, Mapping)
    }
    chunk_to_doc, chunks_by_doc = map_chunks_to_logical_docs(chunks, docs)
    missing_docs = sorted(doc_id for doc_id in chunks_by_doc if not chunks_by_doc[doc_id])
    if missing_docs:
        raise ValueError(f"indexed corpus is missing fixture documents: {', '.join(missing_docs)}")
    unmapped = [str(item.get("id") or "") for item in chunks if str(item.get("id") or "") not in chunk_to_doc]
    if unmapped and not allow_unmapped_chunks:
        raise ValueError(f"indexed corpus has {len(unmapped)} chunks outside the frozen fixture")

    background_chunks: dict[str, list[dict[str, Any]]] = {}
    if allow_unmapped_chunks:
        for raw_chunk in chunks:
            chunk_id = str(raw_chunk.get("id") or "").strip()
            if not chunk_id or chunk_id in chunk_to_doc:
                continue
            index_doc_id = chunk_id.split(":", 1)[0]
            logical_doc_id = f"background:{index_doc_id}"
            chunk_to_doc[chunk_id] = logical_doc_id
            background_chunks.setdefault(logical_doc_id, []).append(dict(raw_chunk))

    index_doc_ids: dict[str, str] = {}
    for logical_doc_id, doc_chunks in chunks_by_doc.items():
        prefixes = {str(item.get("id") or "").split(":", 1)[0] for item in doc_chunks}
        if len(prefixes) != 1:
            raise ValueError(f"fixture document {logical_doc_id} maps to multiple index document IDs")
        index_doc_ids[logical_doc_id] = next(iter(prefixes))

    projected_cases: list[dict[str, Any]] = []
    for case in cases:
        if source_grounded_only and case.get("sourceGrounded") is not True:
            continue
        case_id = str(case.get("id") or "").strip()
        question = str(case.get("question") or "").strip()
        expected = case.get("expected") if isinstance(case.get("expected"), Mapping) else {}
        contracts = [
            dict(item)
            for item in list(expected.get("claimEvidenceContracts") or [])
            if isinstance(item, Mapping)
        ]
        if not contracts and skip_cases_without_contracts:
            continue
        if not case_id or not question or not contracts:
            raise ValueError(f"source case lacks a retrieval contract: {case_id or '<missing-id>'}")
        labels_by_id: dict[str, dict[str, Any]] = {}
        contract_qrels: list[dict[str, Any]] = []
        for contract in contracts:
            doc_id = str(contract.get("docId") or "").strip()
            contract_labels = _contract_labels(
                contract,
                chunks_by_doc.get(doc_id, []),
                allow_missing_source_page=allow_missing_source_page,
            )
            contract_qrels.append(
                {
                    "id": str(contract.get("id") or ""),
                    "doc_id": doc_id,
                    "source_page": int(contract.get("sourcePage") or 0),
                    "relevant_chunk_ids": [item["chunk_id"] for item in contract_labels],
                }
            )
            for label in contract_labels:
                previous = labels_by_id.get(label["chunk_id"])
                if previous is None:
                    labels_by_id[label["chunk_id"]] = label
                    continue
                previous["grade"] = max(int(previous["grade"]), int(label["grade"]))
                previous["contract_ids"] = sorted(
                    set(previous["contract_ids"]) | set(label["contract_ids"])
                )
                previous["matched_evidence_terms"] = list(
                    dict.fromkeys(previous["matched_evidence_terms"] + label["matched_evidence_terms"])
                )
        labels = sorted(labels_by_id.values(), key=lambda item: item["chunk_id"])
        scope_doc_ids = [str(item or "").strip() for item in list(case.get("docIds") or [])]
        projected_cases.append(
            {
                "id": case_id,
                "query": question,
                "locale": "zh" if _CJK_RE.search(question) else "en",
                "evaluation_focus": str(case.get("evaluationFocus") or ""),
                "coverage_tags": sorted(coverage_by_case.get(case_id, set())),
                "query_scope": str(case.get("queryScope") or ""),
                "scope_doc_ids": scope_doc_ids,
                "contracts": contract_qrels,
                "labels": labels,
                "relevance": {item["chunk_id"]: item["grade"] for item in labels},
            }
        )

    corpus_documents = [
        {
            "doc_id": doc_id,
            "index_doc_id": index_doc_ids[doc_id],
            "paper_sha256": corpus_papers.get(doc_id, ""),
            "chunk_count": len(chunks_by_doc[doc_id]),
        }
        for doc_id in sorted(chunks_by_doc)
    ]
    for logical_doc_id, doc_chunks in sorted(background_chunks.items()):
        corpus_documents.append(
            {
                "doc_id": logical_doc_id,
                "index_doc_id": logical_doc_id.split(":", 1)[1],
                "paper_sha256": "",
                "chunk_count": len(doc_chunks),
                "role": "background_negative",
            }
        )

    return {
        "schema_version": RESEARCH_QA_QRELS_SCHEMA_VERSION,
        "id": fixture_id,
        "curation_status": curation_status,
        "description": description
        or (
            "Chunk-level retrieval qrels mechanically projected from the pre-registered "
            "Research QA v2 source/page/evidence contracts before embedding model selection."
        ),
        "source_fixture": {
            "path": source_fixture_path.replace("\\", "/"),
            "sha256": source_fixture_sha256,
            "sha256_newlines": "normalized_lf",
            "benchmark_id": str(benchmark.get("id") or ""),
        },
        "label_policy": {
            "version": LABEL_POLICY_VERSION,
            "unit": "indexed_chunk",
            "candidate_boundary": (
                "registered document; registered source page when available"
                if allow_missing_source_page
                else "registered document and source page"
            ),
            "selection": (
                "greedy evidence-term coverage independent of retriever ranking; "
                "at most three chunks per claim-evidence contract"
            ),
            "grades": {"3": "direct high-coverage evidence", "2": "direct partial evidence"},
        },
        "ks": [1, 3, 5, 10, 20],
        "corpus": {
            "document_count": len(corpus_documents),
            "chunk_count": len(chunks),
            "fingerprint_algorithm": "sha256(logical_doc_id,chunk_id,text_sha256,page_span)",
            "fingerprint": corpus_fingerprint(chunks, chunk_to_doc),
            "documents": corpus_documents,
        },
        "cases": projected_cases,
    }


def load_qrels_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_qrels_fixture(payload, repo_root=path.resolve().parents[1])
    if errors:
        raise ValueError("invalid retrieval qrels fixture: " + "; ".join(errors))
    return payload


def validate_qrels_fixture(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    if int(payload.get("schema_version") or 0) != RESEARCH_QA_QRELS_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RESEARCH_QA_QRELS_SCHEMA_VERSION}")
    label_policy = payload.get("label_policy") if isinstance(payload.get("label_policy"), Mapping) else {}
    if int(label_policy.get("version") or 0) != LABEL_POLICY_VERSION:
        errors.append(f"label_policy.version must be {LABEL_POLICY_VERSION}")
    cases = [item for item in list(payload.get("cases") or []) if isinstance(item, Mapping)]
    if not cases:
        errors.append("cases cannot be empty")
    case_ids = [str(item.get("id") or "").strip() for item in cases]
    if any(not item for item in case_ids) or len(case_ids) != len(set(case_ids)):
        errors.append("case IDs must be non-empty and unique")
    for case in cases:
        case_id = str(case.get("id") or "<missing-id>")
        scope_doc_ids = {str(item or "") for item in list(case.get("scope_doc_ids") or [])}
        labels = [item for item in list(case.get("labels") or []) if isinstance(item, Mapping)]
        relevance = case.get("relevance") if isinstance(case.get("relevance"), Mapping) else {}
        contracts = [item for item in list(case.get("contracts") or []) if isinstance(item, Mapping)]
        label_ids = [str(item.get("chunk_id") or "") for item in labels]
        if not str(case.get("query") or "").strip() or not scope_doc_ids:
            errors.append(f"{case_id}: query and scope_doc_ids are required")
        if not labels or not contracts:
            errors.append(f"{case_id}: labels and contracts are required")
        if len(label_ids) != len(set(label_ids)):
            errors.append(f"{case_id}: duplicate chunk labels")
        if set(label_ids) != {str(item) for item in relevance}:
            errors.append(f"{case_id}: relevance keys must equal label chunk IDs")
        if any(str(item.get("doc_id") or "") not in scope_doc_ids for item in labels):
            errors.append(f"{case_id}: label is outside the registered scope")
        contract_ids = {str(item.get("id") or "") for item in contracts}
        bound_contract_ids = {
            str(contract_id)
            for label in labels
            for contract_id in list(label.get("contract_ids") or [])
        }
        if not contract_ids or not contract_ids.issubset(bound_contract_ids):
            errors.append(f"{case_id}: every contract must have a bound chunk label")

    corpus = payload.get("corpus") if isinstance(payload.get("corpus"), Mapping) else {}
    if int(corpus.get("document_count") or 0) <= 0 or int(corpus.get("chunk_count") or 0) <= 0:
        errors.append("corpus document_count and chunk_count must be positive")
    fingerprint = str(corpus.get("fingerprint") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        errors.append("corpus fingerprint must be lowercase SHA-256")

    source = payload.get("source_fixture") if isinstance(payload.get("source_fixture"), Mapping) else {}
    source_hash = str(source.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        errors.append("source_fixture.sha256 must be lowercase SHA-256")
    if repo_root is not None and str(source.get("path") or "").strip():
        source_path = repo_root / str(source.get("path"))
        if not source_path.is_file():
            errors.append(f"source fixture does not exist: {source.get('path')}")
        elif sha256_file(source_path) != source_hash:
            errors.append("source fixture SHA-256 does not match")
    return errors
