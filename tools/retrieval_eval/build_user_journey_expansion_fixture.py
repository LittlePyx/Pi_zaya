from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file, sha256_text, validate_qrels_fixture
from kb.store import load_all_chunks
from tools.retrieval_eval.validate_user_journey_authoring_manifest import (
    DEFAULT_MANIFEST,
    validate_manifest,
)
from tools.retrieval_eval.validate_user_journey_expansion_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


DEFAULT_REGISTRY = REPO_ROOT / "docs" / "retrieval_user_journey_qrel_registry_v1.json"
DEFAULT_BASE_FIXTURE = REPO_ROOT / "docs" / "retrieval_eval_development_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "retrieval_user_journey_expansion_v1.json"
DEFAULT_DB_ROOT = REPO_ROOT / "db"
DEFAULT_CHUNK_PROJECTION = (
    REPO_ROOT / "docs" / "retrieval_user_journey_chunk_projection_v1.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_registry(
    registry: dict[str, Any],
    *,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if (
        registry.get("schema_version") != 1
        or registry.get("status")
        != "qrels_curated_from_registered_local_chunks_without_retrieval"
    ):
        errors.append("qrel registry identity changed")
    if (
        registry.get("protocol_path")
        != "docs/retrieval_user_journey_expansion_protocol_v1.json"
        or registry.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or registry.get("authoring_manifest_path")
        != "docs/retrieval_user_journey_authoring_manifest_v1.json"
        or registry.get("authoring_manifest_sha256") != sha256_file(DEFAULT_MANIFEST)
    ):
        errors.append("qrel registry source identity changed")
    if (
        registry.get("retriever_executed") is not False
        or registry.get("labels_selected_without_rank") is not True
    ):
        errors.append("qrel registry rank-independence boundary changed")
    cases = registry.get("cases") or []
    manifest_ids = {str(case.get("id") or "") for case in manifest.get("cases") or []}
    authoring_ids = [str(case.get("authoring_case_id") or "") for case in cases]
    if (
        len(cases) != 18
        or set(authoring_ids) != manifest_ids
        or len(set(authoring_ids)) != 18
    ):
        errors.append("qrel registry does not cover the authoring manifest exactly")
    final_ids: list[str] = []
    replacement_count = 0
    for case in cases:
        replacement = (
            case.get("replacement") if isinstance(case.get("replacement"), dict) else {}
        )
        final_id = str(replacement.get("id") or case.get("authoring_case_id") or "")
        final_ids.append(final_id)
        if replacement:
            replacement_count += 1
            if any(
                not str(replacement.get(key) or "").strip()
                for key in ("id", "question", "reading_intent", "reason")
            ):
                errors.append(f"qrel replacement lineage incomplete: {final_id}")
        contracts = case.get("contracts") or []
        if not contracts:
            errors.append(f"qrel case has no contracts: {final_id}")
        contract_ids: list[str] = []
        label_ids: list[str] = []
        for contract in contracts:
            contract_ids.append(str(contract.get("id") or ""))
            labels = contract.get("labels") or []
            if (
                not 1
                <= len(labels)
                <= protocol["label_policy"]["maximum_chunks_per_evidence_contract"]
            ):
                errors.append(f"contract label count invalid: {contract.get('id')}")
            for label in labels:
                label_ids.append(str(label.get("chunk_id") or ""))
                if (
                    int(label.get("grade") or 0) not in {2, 3}
                    or not str(label.get("evidence_role") or "").strip()
                ):
                    errors.append(f"qrel label grade or role invalid: {final_id}")
        if len(contract_ids) != len(set(contract_ids)) or any(
            not value for value in contract_ids
        ):
            errors.append(f"qrel contract IDs missing or duplicated: {final_id}")
        if len(label_ids) != len(set(label_ids)) or not 1 <= len(label_ids) <= 6:
            errors.append(
                f"qrel chunk IDs missing, duplicated, or out of bounds: {final_id}"
            )
    if len(set(final_ids)) != 18:
        errors.append("final qrel case IDs are duplicated")
    if replacement_count != 4:
        errors.append("ambiguous-question replacement count changed")
    return errors


def _deterministic_split(
    cases: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    split = protocol["deterministic_split"]
    holdout: list[str] = []
    for locale, quota in split["holdout_locale_quota"].items():
        locale_ids = [str(case["id"]) for case in cases if case["locale"] == locale]
        ranked = sorted(
            locale_ids,
            key=lambda case_id: hashlib.sha256(
                f"{split['seed']}:{locale}:{case_id}".encode()
            ).hexdigest(),
        )
        holdout.extend(ranked[: int(quota)])
    holdout = sorted(holdout)
    all_ids = {str(case["id"]) for case in cases}
    return {
        "algorithm": split["algorithm"],
        "seed": split["seed"],
        "calibration_case_ids": sorted(all_ids - set(holdout)),
        "holdout_case_ids": holdout,
        "calibration_case_count": len(all_ids - set(holdout)),
        "holdout_case_count": len(holdout),
        "holdout_allowed_for_tuning": False,
    }


def build_fixture(
    *,
    db_root: Path = DEFAULT_DB_ROOT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    manifest_path: Path = DEFAULT_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    base_fixture_path: Path = DEFAULT_BASE_FIXTURE,
    chunk_projection_path: Path = DEFAULT_CHUNK_PROJECTION,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    manifest = _load_json(manifest_path)
    registry = _load_json(registry_path)
    protocol_errors = validate_protocol(protocol)
    manifest_errors = validate_manifest(manifest)
    registry_errors = validate_registry(registry, protocol=protocol, manifest=manifest)
    errors = protocol_errors + manifest_errors + registry_errors
    if errors:
        raise ValueError("user-journey source contract invalid: " + "; ".join(errors))

    base = _load_json(base_fixture_path)
    corpus = dict(base["corpus"])
    index_to_logical = {
        str(row["index_doc_id"]): str(row["doc_id"]) for row in corpus["documents"]
    }
    chunks = load_all_chunks(db_root)
    if not chunks:
        projection = _load_json(chunk_projection_path)
        if (
            projection.get("schema_version") != 1
            or projection.get("status") != "frozen_qrel_source_projection"
            or not isinstance(projection.get("chunks"), list)
        ):
            raise ValueError("frozen qrel source projection identity is invalid")
        chunks = [dict(chunk) for chunk in projection["chunks"]]
    chunks_by_id = {str(chunk.get("id") or ""): dict(chunk) for chunk in chunks}
    manifest_by_id = {str(case["id"]): dict(case) for case in manifest["cases"]}
    output_cases: list[dict[str, Any]] = []

    for registry_case in registry["cases"]:
        authoring = manifest_by_id[registry_case["authoring_case_id"]]
        replacement = (
            dict(registry_case["replacement"])
            if isinstance(registry_case.get("replacement"), dict)
            else {}
        )
        case_id = str(replacement.get("id") or authoring["id"])
        query = str(replacement.get("question") or authoring["question"])
        reading_intent = str(
            replacement.get("reading_intent") or authoring["reading_intent"]
        )
        labels: list[dict[str, Any]] = []
        contracts: list[dict[str, Any]] = []
        relevant_docs: set[str] = set()
        for contract in registry_case["contracts"]:
            doc_id = str(contract["doc_id"])
            contract_labels: list[dict[str, Any]] = []
            for source_label in contract["labels"]:
                chunk_id = str(source_label["chunk_id"])
                chunk = chunks_by_id.get(chunk_id)
                if chunk is None:
                    raise ValueError(f"registered qrel chunk is missing: {chunk_id}")
                index_doc_id = chunk_id.split(":", 1)[0]
                if index_to_logical.get(index_doc_id) != doc_id:
                    raise ValueError(
                        f"qrel chunk {chunk_id} does not belong to {doc_id}"
                    )
                meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
                if meta.get("evidence_ready") is not True:
                    raise ValueError(f"qrel chunk is not evidence-ready: {chunk_id}")
                page_start = int(meta.get("page_start") or meta.get("page") or 0)
                page_end = int(meta.get("page_end") or page_start)
                label = {
                    "chunk_id": chunk_id,
                    "grade": int(source_label["grade"]),
                    "doc_id": doc_id,
                    "page_start": page_start,
                    "page_end": page_end,
                    "contract_ids": [str(contract["id"])],
                    "text_sha256": sha256_text(chunk.get("text")),
                    "evidence_role": str(source_label["evidence_role"]),
                }
                labels.append(label)
                contract_labels.append(label)
                relevant_docs.add(doc_id)
            contracts.append(
                {
                    "id": str(contract["id"]),
                    "doc_id": doc_id,
                    "source_page": min(
                        label["page_start"] for label in contract_labels
                    ),
                    "relevant_chunk_ids": [
                        label["chunk_id"] for label in contract_labels
                    ],
                }
            )
        scope = [str(value) for value in authoring["scope_doc_ids"]]
        if not relevant_docs.issubset(set(scope)):
            raise ValueError(f"qrel documents escape scope: {case_id}")
        if authoring[
            "journey_type"
        ] == "cross_paper_evidence_comparison" and relevant_docs != set(scope):
            raise ValueError(
                f"cross-paper qrels do not cover every compared paper: {case_id}"
            )
        labels.sort(key=lambda row: row["chunk_id"])
        output_cases.append(
            {
                "id": case_id,
                "query": query,
                "locale": authoring["locale"],
                "evaluation_focus": authoring["journey_type"],
                "coverage_tags": ["user_journey_expansion", authoring["journey_type"]],
                "query_scope": reading_intent,
                "primary_doc_id": authoring["primary_doc_id"],
                "scope_doc_ids": scope,
                "authoring_lineage": {
                    "authoring_case_id": authoring["id"],
                    "replacement_reason": str(replacement.get("reason") or ""),
                },
                "contracts": contracts,
                "labels": labels,
                "relevance": {label["chunk_id"]: label["grade"] for label in labels},
            }
        )

    fixture = {
        "schema_version": 2,
        "id": "retrieval_user_journey_expansion_v1_chunk_qrels",
        "curation_status": "frozen_before_any_retrieval_baseline_or_fusion_v3_design",
        "description": "New library-reading evidence-location qrels authored and labeled from registered local chunks without running a retriever; not an answer-writing evaluation.",
        "source_fixture": {
            "path": registry_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(registry_path),
            "sha256_newlines": "normalized_lf",
            "benchmark_id": protocol["id"],
        },
        "authoring_manifest": {
            "path": manifest_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "label_policy": {
            "version": 1,
            "unit": "indexed_chunk",
            "candidate_boundary": "registered local library documents and evidence-ready chunks",
            "selection": "direct source inspection independent of retriever ranking; at most three chunks per evidence contract",
            "grades": protocol["label_policy"]["grades"],
            "required_audit_fields": protocol["label_policy"]["required_label_fields"],
        },
        "ks": [1, 3, 5, 10, 20],
        "corpus": corpus,
        "split": _deterministic_split(output_cases, protocol),
        "retriever_executed_during_authoring_or_labeling": False,
        "answer_generation_evaluated": False,
        "cases": output_cases,
    }
    qrel_errors = validate_qrels_fixture(fixture, repo_root=REPO_ROOT)
    if qrel_errors:
        raise ValueError(
            "generated user-journey fixture is invalid: " + "; ".join(qrel_errors)
        )
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or check the user-journey qrel fixture."
    )
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fixture = build_fixture(db_root=args.db_root.expanduser().resolve())
    output = args.output.expanduser().resolve()
    serialized = json.dumps(fixture, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != serialized:
            print("[FAIL] user-journey qrel fixture differs from frozen local sources")
            return 1
        print("[OK] user-journey qrel fixture matches frozen local sources")
        return 0
    if output.exists():
        parser.error("user-journey qrel fixture already exists; use --check")
    output.write_text(serialized, encoding="utf-8")
    print(f"[OK] wrote {output}: {len(fixture['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
