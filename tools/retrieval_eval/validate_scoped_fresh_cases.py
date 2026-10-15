from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_benchmark import validate_indexed_corpus
from kb.retrieval_eval_fixture import (
    sha256_file,
    sha256_text,
    validate_qrels_fixture,
)
from tools.retrieval_eval.validate_scoped_fresh_authoring_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_scoped_fresh_cases_v1.json"
BASE_FIXTURE = REPO_ROOT / "docs" / "retrieval_user_journey_expansion_v1.json"
DEFAULT_CHUNK_PROJECTION = (
    REPO_ROOT / "docs" / "retrieval_scoped_fresh_chunk_projection_v1.json"
)
EXPECTED_FIXTURE_SHA256 = "3472826090e032739903db91dd8c359adbc68bff6fcbe40ec9f136bb945b1d27"
EXPECTED_CORPUS_FINGERPRINT = "5d65f3a033d62e15c3b90b7c32314452cb49334652d1db0f0e74a55a2b125317"
EXPECTED_DOCUMENTS = [
    ("3d-sp-video", "a34b2661f87a68d2", 31),
    ("dl-spi-review", "5527e0d96aeb608a", 82),
    ("fdm-metamaterials", "6b1a3c7c4aa0aaa9", 32),
    ("foveated-spi", "aa60f6f8d0cbcaa6", 55),
    ("qclfm", "831f8c559642fe39", 45),
    ("rt-spi-supp", "3c702d1a300625d2", 4),
    ("scigs", "32490abae33df8c9", 104),
    ("scinerf", "7f6b9404bccea454", 104),
    ("seq-cs", "6117def890d573f9", 25),
    ("spd-review", "ba919188ff49c763", 110),
    ("sph-biological", "11b5152564f8b679", 73),
    ("spi-prospects", "7846a2dbb8e31f4a", 37),
]
PRODUCT_BOUNDARY = {
    "library_papers_only": True,
    "evidence_retrieval_only": True,
    "writing_or_submission_workflow_allowed": False,
    "product_enabled": False,
    "ordinary_user_surface_exposed": False,
}
_FORBIDDEN_RESULT_KEYS = {
    "candidate_results",
    "dense_results",
    "latency_ms",
    "metrics",
    "ranked_chunk_ids",
    "score",
    "scores",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_selected_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for _logical_id, index_id, _count in EXPECTED_DOCUMENTS:
        path = REPO_ROOT / "db" / "chunks" / f"{index_id}.jsonl"
        if not path.exists():
            projection = _load_json(DEFAULT_CHUNK_PROJECTION)
            if (
                projection.get("schema_version") != 1
                or projection.get("status") != "frozen_scoped_fresh_corpus_projection"
                or projection.get("chunk_count") != 702
                or not isinstance(projection.get("chunks"), list)
            ):
                raise ValueError("scoped fresh chunk projection identity is invalid")
            return [dict(chunk) for chunk in projection["chunks"]]
        chunks.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return chunks


def _projected_corpus_fingerprint(
    chunks: Sequence[Mapping[str, Any]],
    chunk_to_doc: Mapping[str, str],
) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: str(item.get("id") or "")):
        chunk_id = str(chunk.get("id") or "").strip()
        logical_doc_id = str(chunk_to_doc.get(chunk_id) or "")
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), Mapping) else {}
        record = "\x1f".join(
            [
                logical_doc_id,
                chunk_id,
                str(chunk.get("text_sha256") or ""),
                str(int(meta.get("page_start") or 0)),
                str(int(meta.get("page_end") or 0)),
            ]
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _validate_projected_corpus(
    payload: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
) -> list[str]:
    corpus = payload.get("corpus") if isinstance(payload.get("corpus"), Mapping) else {}
    index_to_logical = {
        str(item.get("index_doc_id") or ""): str(item.get("doc_id") or "")
        for item in list(corpus.get("documents") or [])
        if isinstance(item, Mapping)
    }
    chunk_to_doc: dict[str, str] = {}
    errors: list[str] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        logical_doc_id = index_to_logical.get(chunk_id.split(":", 1)[0], "")
        if not logical_doc_id:
            errors.append(f"unregistered corpus chunk: {chunk_id}")
            continue
        chunk_to_doc[chunk_id] = logical_doc_id
    expected_count = int(corpus.get("chunk_count") or 0)
    if len(chunks) != expected_count:
        errors.append(f"chunk count is {len(chunks)}, expected {expected_count}")
    actual_fingerprint = _projected_corpus_fingerprint(chunks, chunk_to_doc)
    expected_fingerprint = str(corpus.get("fingerprint") or "")
    if actual_fingerprint != expected_fingerprint:
        errors.append(
            f"corpus fingerprint is {actual_fingerprint}, expected {expected_fingerprint}"
        )
    chunks_by_id = {str(chunk.get("id") or ""): chunk for chunk in chunks}
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
            elif str(chunk.get("text_sha256") or "") != str(
                label.get("text_sha256") or ""
            ):
                errors.append(f"{case_id}: labelled chunk text changed: {chunk_id}")
    return errors


def _prior_evidence_ids() -> set[str]:
    payload = _load_json(BASE_FIXTURE)
    return {
        str(label.get("chunk_id") or "")
        for case in list(payload.get("cases") or [])
        if isinstance(case, Mapping)
        for label in list(case.get("labels") or [])
        if isinstance(label, Mapping)
    }


def _contains_forbidden_result_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in _FORBIDDEN_RESULT_KEYS
            or _contains_forbidden_result_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_result_key(child) for child in value)
    return False


def validate_fixture(
    payload: Mapping[str, Any],
    *,
    chunks: Sequence[Mapping[str, Any]] | None = None,
    prior_evidence_ids: set[str] | None = None,
) -> list[str]:
    errors = validate_qrels_fixture(payload, repo_root=REPO_ROOT)
    if (
        payload.get("id") != "retrieval_scoped_fresh_cases_v1_chunk_qrels"
        or payload.get("curation_status")
        != "frozen_after_source_only_authoring_before_candidate_evaluation"
    ):
        errors.append("scoped fresh fixture identity changed")

    source = payload.get("source_fixture") if isinstance(payload.get("source_fixture"), Mapping) else {}
    if (
        source.get("path") != "docs/retrieval_scoped_fresh_authoring_protocol_v1.json"
        or source.get("benchmark_id") != "retrieval_scoped_fresh_authoring_v1"
        or source.get("sha256") != sha256_file(DEFAULT_PROTOCOL)
    ):
        errors.append("scoped fresh authoring lineage changed")

    corpus = payload.get("corpus") if isinstance(payload.get("corpus"), Mapping) else {}
    actual_documents = [
        (
            str(item.get("doc_id") or ""),
            str(item.get("index_doc_id") or ""),
            int(item.get("chunk_count") or 0),
        )
        for item in list(corpus.get("documents") or [])
        if isinstance(item, Mapping)
    ]
    if (
        corpus.get("document_count") != 12
        or corpus.get("chunk_count") != 702
        or corpus.get("fingerprint") != EXPECTED_CORPUS_FINGERPRINT
        or actual_documents != EXPECTED_DOCUMENTS
    ):
        errors.append("scoped fresh corpus registration changed")

    cases = [item for item in list(payload.get("cases") or []) if isinstance(item, Mapping)]
    primary_docs = [str(case.get("primary_doc_id") or "") for case in cases]
    expected_docs = [item[0] for item in EXPECTED_DOCUMENTS]
    locales = Counter(str(case.get("locale") or "") for case in cases)
    if len(cases) != 12 or primary_docs != expected_docs:
        errors.append("fixture must contain exactly one ordered case per fresh document")
    if locales != Counter({"zh": 6, "en": 6}):
        errors.append("fixture locale quota changed")

    prior_ids = _prior_evidence_ids() if prior_evidence_ids is None else prior_evidence_ids
    chunk_rows = list(_load_selected_chunks() if chunks is None else chunks)
    chunks_by_id = {str(chunk.get("id") or ""): chunk for chunk in chunk_rows}
    label_count = 0
    for case in cases:
        case_id = str(case.get("id") or "<missing-id>")
        primary_doc = str(case.get("primary_doc_id") or "")
        if list(case.get("scope_doc_ids") or []) != [primary_doc]:
            errors.append(f"{case_id}: scope is not the single registered document")
        labels = [item for item in list(case.get("labels") or []) if isinstance(item, Mapping)]
        label_count += len(labels)
        if not 1 <= len(labels) <= 2:
            errors.append(f"{case_id}: label count is outside the frozen 1..2 boundary")
        if any(int(label.get("grade") or 0) not in {2, 3} for label in labels):
            errors.append(f"{case_id}: label grade is outside the frozen policy")
        for label in labels:
            chunk_id = str(label.get("chunk_id") or "")
            if chunk_id in prior_ids:
                errors.append(f"{case_id}: evidence overlaps the previous journey suite: {chunk_id}")
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            meta = chunk.get("meta") if isinstance(chunk.get("meta"), Mapping) else {}
            chunk_text_sha256 = str(chunk.get("text_sha256") or "") or sha256_text(
                chunk.get("text")
            )
            if (
                int(label.get("page_start") or 0) != int(meta.get("page_start") or 0)
                or int(label.get("page_end") or 0) != int(meta.get("page_end") or 0)
                or str(label.get("text_sha256") or "") != chunk_text_sha256
            ):
                errors.append(f"{case_id}: evidence page or source-text identity changed: {chunk_id}")
        label_ids = {str(label.get("chunk_id") or "") for label in labels}
        for contract in list(case.get("contracts") or []):
            if not isinstance(contract, Mapping):
                continue
            if (
                str(contract.get("doc_id") or "") != primary_doc
                or int(contract.get("source_page") or 0) <= 0
                or not set(str(item) for item in list(contract.get("relevant_chunk_ids") or []))
                <= label_ids
            ):
                errors.append(f"{case_id}: evidence contract escaped its source labels")
    if label_count != 21:
        errors.append("scoped fresh label count changed")

    if payload.get("retriever_executed_during_authoring_or_labeling") is not False:
        errors.append("retriever execution was introduced into source-only authoring")
    if payload.get("answer_generation_evaluated") is not False:
        errors.append("answer generation was introduced into retrieval authoring")
    if payload.get("product_boundary") != PRODUCT_BOUNDARY:
        errors.append("scoped fresh product boundary changed")
    if _contains_forbidden_result_key(payload):
        errors.append("retrieval result data leaked into the sealed authoring fixture")

    if chunk_rows and all("text_sha256" in chunk for chunk in chunk_rows):
        corpus_errors = _validate_projected_corpus(payload, chunk_rows)
    else:
        _mapping, corpus_errors = validate_indexed_corpus(payload, chunk_rows)
    errors.extend(corpus_errors)
    return errors


def main() -> int:
    try:
        protocol = _load_json(DEFAULT_PROTOCOL)
        payload = _load_json(DEFAULT_FIXTURE)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] scoped fresh fixture unreadable: {type(exc).__name__}")
        return 1
    protocol_errors = validate_protocol(protocol)
    if protocol_errors:
        for error in protocol_errors:
            print(f"[FAIL] {error}")
        return 1
    if sha256_file(DEFAULT_FIXTURE) != EXPECTED_FIXTURE_SHA256:
        print("[FAIL] scoped fresh fixture changed after pre-evaluation sealing")
        return 1
    errors = validate_fixture(payload)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("[OK] 12-case scoped fresh fixture is source-grounded and sealed before evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
