from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.local_semantic_component import (
    LocalSemanticComponentCancelled,
    LocalSemanticComponentManifest,
    LocalSemanticComponentStore,
    local_semantic_component_root,
)
from kb.local_semantic_acceptance import (
    build_local_semantic_acceptance_report,
    load_local_observation_events,
)
from kb.local_semantic_component_download import (
    ComponentDownloadProgress,
    ComponentDownloadResult,
)
from kb.local_semantic_component_lifecycle import LocalSemanticComponentLifecycleService
from kb.local_semantic_component_service import (
    LOCAL_SEMANTIC_COMPONENT_ALLOWED_HOSTS,
    LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV,
    get_local_semantic_component_service,
)
import kb.local_semantic_component_service as component_service
from kb.portable_local_semantic_runtime import PortableLocalSemanticRuntimeAdapter
from kb.store import atomic_write_json, load_all_chunks, save_docs_index, write_doc_chunks
from kb.task_retrieval_observation import TaskRetrievalObservation


_OBSERVATION_PATH_ENV = "KB_LOCAL_OBSERVABILITY_PATH"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_fixture(db_dir: Path) -> None:
    source_a = "C:/fixture/selected-paper.md"
    source_b = "C:/fixture/out-of-scope-paper.md"
    texts_a = [
        f"Control calibration observation {index} for an unrelated spectrometer."
        for index in range(10)
    ] + [
        "Photovoltaic cells convert sunlight into electrical energy using semiconductor p n junctions.",
        "The experiment reports the solar conversion mechanism in the selected paper.",
    ]
    texts_b = [
        "Photovoltaic cells convert sunlight into electricity, but this paper is outside the selected scope."
    ]
    write_doc_chunks(
        db_dir,
        "doc-a",
        [
            {
                "text": text,
                "meta": {"source_path": source_a, "evidence_ready": True},
            }
            for text in texts_a
        ],
    )
    write_doc_chunks(
        db_dir,
        "doc-b",
        [
            {
                "text": text,
                "meta": {"source_path": source_b, "evidence_ready": True},
            }
            for text in texts_b
        ],
    )
    save_docs_index(
        db_dir,
        {
            "doc-a": {
                "doc_id": "doc-a",
                "path": source_a,
                "num_chunks": len(texts_a),
                "index_status": "ready",
            },
            "doc-b": {
                "doc_id": "doc-b",
                "path": source_b,
                "num_chunks": len(texts_b),
                "index_status": "ready",
            },
        },
    )


class _StagedArchiveDownloader:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path

    def download(
        self,
        manifest,
        _transport,
        *,
        should_cancel,
        on_progress,
    ) -> ComponentDownloadResult:
        if should_cancel():
            raise LocalSemanticComponentCancelled()
        on_progress(
            ComponentDownloadProgress(
                completed_bytes=manifest.archive_bytes,
                total_bytes=manifest.archive_bytes,
                resumed_bytes=0,
                eta_seconds=0,
            )
        )
        return ComponentDownloadResult(
            archive_path=self.archive_path,
            downloaded_bytes=manifest.archive_bytes,
            resumed_bytes=0,
            reused=True,
        )


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    missing = object()
    previous = {key: os.environ.get(key, missing) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def _clear_product_runtime_caches() -> None:
    with component_service._SERVICES_LOCK:
        component_service._SERVICES.clear()
        component_service._RUNTIMES.clear()


def _select(
    *,
    db_dir: Path,
    sparse_hits: list[dict],
    scope: str,
    mode: str,
    source_paths: list[str] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    published: list[dict[str, object]] = []
    observation = TaskRetrievalObservation()
    observation.begin(time.perf_counter(), scope)
    query = (
        "Photovoltaic cells convert sunlight into electrical energy using "
        "semiconductor p n junctions."
    )
    hits, _scores = observation.select_and_publish(
        query=query,
        sparse_hits=sparse_hits,
        db_dir=db_dir,
        retrieval_mode=mode,
        source_paths=source_paths,
        publish=lambda **payload: published.append(payload),
    )
    if len(published) != 1:
        raise RuntimeError("product retrieval trace was not published exactly once")
    observation.record(hits=hits, query_variants=[query])
    return hits, dict(observation.local_trace)


def validate_product_path(
    *,
    archive_path: Path,
    manifest_path: Path,
    work_root: Path,
) -> dict[str, object]:
    archive = archive_path.resolve(strict=True)
    manifest_file = manifest_path.resolve(strict=True)
    root = work_root.resolve(strict=False)
    if root.exists():
        raise RuntimeError("product validation work root already exists")
    root.mkdir(parents=True)
    db_dir = root / "db"
    _publish_fixture(db_dir)
    library_before = {
        "docs": _sha256(db_dir / "docs.json"),
        "doc_a": _sha256(db_dir / "chunks" / "doc-a.jsonl"),
        "doc_b": _sha256(db_dir / "chunks" / "doc-b.jsonl"),
    }

    raw_manifest = manifest_file.read_text(encoding="utf-8").strip()
    payload = json.loads(raw_manifest)
    manifest = LocalSemanticComponentManifest.from_payload(
        payload,
        allowed_hosts=LOCAL_SEMANTIC_COMPONENT_ALLOWED_HOSTS,
    )
    if archive.stat().st_size != manifest.archive_bytes or _sha256(archive) != manifest.archive_sha256:
        raise RuntimeError("product validation archive identity mismatch")

    observation_path = root / "local-observations.jsonl"
    started = time.perf_counter()
    with _temporary_environment(
        {
            LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV: raw_manifest,
            _OBSERVATION_PATH_ENV: str(observation_path),
            "LOCALAPPDATA": str(root / "app-data"),
        }
    ):
        _clear_product_runtime_caches()
        try:
            store = LocalSemanticComponentStore(local_semantic_component_root(db_dir))
            runtime = PortableLocalSemanticRuntimeAdapter(
                db_dir,
                component_store=store,
            )
            lifecycle = LocalSemanticComponentLifecycleService(
                db_dir,
                manifest,
                transport_factory=lambda: object(),
                component_store=store,
                downloader=_StagedArchiveDownloader(archive),
                health_check=runtime.health_check,
                index_ready=runtime.index_ready,
                prepare_index=runtime.prepare_index,
                remove_index=runtime.remove_index,
                release_runtime_handles=runtime.release_runtime_handles,
                runner=lambda callback: callback(),
                retry_wait=lambda _seconds: None,
                owner_session="release-product-validation",
            )
            ready, created = lifecycle.enable(consent=True)
            if not created or ready.state != "ready":
                raise RuntimeError("product validation lifecycle did not reach ready")
            _clear_product_runtime_caches()

            sparse = [
                hit
                for hit in load_all_chunks(db_dir)
                if str(hit.get("id") or "").startswith("doc-a:")
            ][:10]
            for index, hit in enumerate(sparse):
                hit["score"] = float(100 - index)
            original_ids = [str(hit.get("id") or "") for hit in sparse]
            selected, selected_trace = _select(
                db_dir=db_dir,
                sparse_hits=sparse,
                scope="current_paper",
                mode="hybrid_local",
                source_paths=["C:/fixture/selected-paper.md"],
            )
            warm_selected, warm_trace = _select(
                db_dir=db_dir,
                sparse_hits=sparse,
                scope="current_paper",
                mode="hybrid_local",
                source_paths=["C:/fixture/selected-paper.md"],
            )
            selected_ids = [str(hit.get("id") or "") for hit in selected]
            warm_ids = [str(hit.get("id") or "") for hit in warm_selected]
            if (
                selected_trace.get("used_mode") != "hybrid_local"
                or selected_trace.get("fallback_reason_id")
                or selected_trace.get("cold_start") is not True
                or selected_trace.get("index_cache_hit") is not False
                or warm_trace.get("used_mode") != "hybrid_local"
                or warm_trace.get("cold_start") is not False
                or warm_trace.get("index_cache_hit") is not True
                or selected_ids[:5] != original_ids[:5]
                or warm_ids != selected_ids
                or "doc-a:10" not in selected_ids[:10]
                or any(not item.startswith("doc-a:") for item in selected_ids)
            ):
                raise RuntimeError("selected-paper product hybrid contract failed")

            library_hits, library_trace = _select(
                db_dir=db_dir,
                sparse_hits=sparse,
                scope="library",
                mode="hybrid_local",
            )
            bm25_hits, bm25_trace = _select(
                db_dir=db_dir,
                sparse_hits=sparse,
                scope="current_paper",
                mode="bm25",
            )
            if (
                [str(hit.get("id") or "") for hit in library_hits] != original_ids
                or library_trace.get("used_mode") != "bm25"
                or library_trace.get("fallback_reason_id")
                != "local_semantic_scope_not_eligible"
                or [str(hit.get("id") or "") for hit in bm25_hits] != original_ids
                or bm25_trace.get("used_mode") != "bm25"
            ):
                raise RuntimeError("BM25 default or full-library fallback contract failed")

            service = get_local_semantic_component_service(db_dir)
            removal, _created = service.remove()
            deadline = time.monotonic() + 30.0
            while removal.state != "not_installed" and time.monotonic() < deadline:
                time.sleep(0.05)
                removal = service.status()
            events, line_count = load_local_observation_events(observation_path)
            acceptance = build_local_semantic_acceptance_report(
                events,
                input_line_count=line_count,
            )
        finally:
            _clear_product_runtime_caches()
            observation_path.unlink(missing_ok=True)
    library_after = {
        "docs": _sha256(db_dir / "docs.json"),
        "doc_a": _sha256(db_dir / "chunks" / "doc-a.jsonl"),
        "doc_b": _sha256(db_dir / "chunks" / "doc-b.jsonl"),
    }
    if removal.state != "not_installed" or library_after != library_before:
        raise RuntimeError("component removal changed the fixture library")
    lifecycle_metrics = acceptance["lifecycle"]
    retrieval_metrics = acceptance["retrieval"]
    if (
        lifecycle_metrics["opt_in_started_count"] != 1
        or lifecycle_metrics["opt_in_ready_count"] != 1
        or lifecycle_metrics["observed_opt_in_completion_rate"] != 1.0
        or lifecycle_metrics["terminal_outcome_counts"][
            "local_semantic_component.removed"
        ]
        != 1
        or retrieval_metrics["hybrid_request_count"] != 3
        or retrieval_metrics["actual_hybrid_use_count"] != 2
        or retrieval_metrics["automatic_fallback_count"] != 1
        or retrieval_metrics["automatic_fallback_category_counts"]
        != {"source_safety_boundary": 1}
        or retrieval_metrics["warm_hybrid_latency_ms"]["sample_count"] != 1
        or acceptance["privacy"].get("raw_content_included") is not False
        or acceptance["window"].get("window_boundary_incomplete") is not False
    ):
        raise RuntimeError("local semantic aggregate observation contract failed")

    result = {
        "schema_version": 2,
        "result_kind": "local_semantic_product_path_release_validation",
        "archive_sha256": manifest.archive_sha256,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "selected_paper": {
            "trace": selected_trace,
            "warm_trace": warm_trace,
            "target_in_top_10": True,
            "source_escape_count": 0,
            "protected_prefix_preserved": True,
        },
        "full_library": {"trace": library_trace, "bm25_unchanged": True},
        "default_mode": {"trace": bm25_trace, "bm25_unchanged": True},
        "removal": {
            "state": removal.state,
            "library_unchanged": library_after == library_before,
        },
        "local_acceptance": acceptance,
        "network_used": False,
        "external_web_or_model_knowledge_used": False,
        "passes_all": True,
    }
    atomic_write_json(root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the release component through the promoted product selector."
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()
    result = validate_product_path(
        archive_path=args.archive,
        manifest_path=args.manifest,
        work_root=args.work_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
