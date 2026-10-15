from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .e5_small_embedding_runtime import (
    EMBEDDING_DIMENSION,
    E5EmbeddingCancelled,
    E5EmbeddingProgress,
    E5ParentWindow,
    E5SmallOnnxExecutor,
)
from .embedding_index import _snapshot_from_db as load_embedding_corpus_snapshot
from .embedding_vectors import normalize_embedding_batch
from .local_semantic_artifacts import CANDIDATE_ID, MODEL_REVISION
from .parent_collapsed_dense_retriever import ParentCollapsedDenseRetriever
from .store import atomic_write_json, db_write_lock


INDEX_SCHEMA_VERSION = 1
REPRESENTATION_ID = "e5_small_full_coverage_parent_collapsed_windows_v1"
_BUILD_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class LocalSemanticIndexError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(str(code or "local_semantic_index_failed"))
        self.code = str(code or "local_semantic_index_failed")


@dataclass(frozen=True)
class LocalSemanticIndexStatus:
    state: str
    reason_id: str
    build_id: str = ""
    document_count: int = 0
    parent_count: int = 0
    window_count: int = 0


@dataclass(frozen=True)
class LocalSemanticIndexBuildProgress:
    completed_windows: int
    total_windows: int
    reused_windows: int
    embedded_windows: int


@dataclass(frozen=True)
class LocalSemanticIndexBuildResult:
    build_id: str
    document_count: int
    parent_count: int
    window_count: int
    reused_parent_count: int
    reused_window_count: int
    embedded_parent_count: int
    embedded_window_count: int
    no_op: bool
    duration_ms: float


@dataclass(frozen=True)
class LoadedLocalSemanticIndex:
    chunks: list[dict]
    windows: list[dict[str, Any]]
    vectors: list[tuple[float, ...]]
    manifest: dict[str, Any]

    def as_retriever(
        self,
        executor: E5SmallOnnxExecutor,
    ) -> ParentCollapsedDenseRetriever:
        parent_vectors: list[tuple[float, ...] | None] = [None] * len(self.chunks)
        child_vectors: dict[str, list[tuple[int, tuple[float, ...]]]] = {}
        for row, vector in zip(self.windows, self.vectors):
            position = int(row["parent_position"])
            ordinal = int(row["window_ordinal"])
            parent_id = str(row["parent_id"])
            if ordinal == 0:
                parent_vectors[position] = vector
            else:
                child_vectors.setdefault(parent_id, []).append((ordinal, vector))
        if any(vector is None for vector in parent_vectors):
            raise LocalSemanticIndexError("parent_vector_missing")
        return ParentCollapsedDenseRetriever(
            self.chunks,
            [vector for vector in parent_vectors if vector is not None],
            child_vectors,
            executor.embed_query,
        )


def local_semantic_index_dir(db_dir: Path | str) -> Path:
    return (
        Path(db_dir).expanduser().resolve(strict=False)
        / "local_semantic_index"
        / "e5_small_v1"
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_vectors(path: Path, vectors: Sequence[Sequence[float]]) -> None:
    values = array("f", (value for vector in vectors for value in vector))
    if sys.byteorder != "little":
        values.byteswap()
    with path.open("wb") as handle:
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())


def _read_vectors(
    path: Path,
    *,
    count: int,
    dimension: int,
) -> list[tuple[float, ...]]:
    expected_values = count * dimension
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LocalSemanticIndexError("corrupt_vectors") from exc
    if size != expected_values * 4:
        raise LocalSemanticIndexError("corrupt_vectors")
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, expected_values)
    if sys.byteorder != "little":
        values.byteswap()
    vectors = [
        tuple(float(value) for value in values[offset : offset + dimension])
        for offset in range(0, expected_values, dimension)
    ]
    try:
        return normalize_embedding_batch(
            vectors,
            expected_count=count,
            expected_dim=dimension,
        )
    except ValueError as exc:
        raise LocalSemanticIndexError("corrupt_vectors") from exc


def _ordered_parent_fingerprint(snapshot: Any) -> str:
    return _sha256_bytes(_canonical_json(snapshot.chunk_records))


def _document_count(snapshot: Any) -> int:
    return len({str(chunk.get("id") or "").split(":", 1)[0] for chunk in snapshot.chunks})


class LocalSemanticWindowIndexStore:
    """Atomic, incremental, text-free index for the accepted E5 window runtime."""

    def __init__(self, db_dir: Path | str, *, index_dir: Path | str | None = None) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        self.index_dir = (
            Path(index_dir).expanduser().resolve(strict=False)
            if index_dir is not None
            else local_semantic_index_dir(self.db_dir)
        )
        self.generations_dir = self.index_dir / "generations"
        self.current_path = self.index_dir / "current.json"

    def _read_published(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[float, ...]]]:
        try:
            pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LocalSemanticIndexError("missing") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalSemanticIndexError("corrupt_pointer") from exc
        build_id = str(pointer.get("build_id") or "")
        if not _BUILD_ID_RE.fullmatch(build_id):
            raise LocalSemanticIndexError("corrupt_pointer")
        generation = (self.generations_dir / build_id).resolve(strict=False)
        if self.generations_dir.resolve(strict=False) not in generation.parents:
            raise LocalSemanticIndexError("corrupt_pointer")
        manifest_path = generation / "manifest.json"
        windows_path = generation / "windows.json"
        vectors_path = generation / "vectors.f32"
        try:
            if _sha256_file(manifest_path) != str(pointer.get("manifest_sha256") or ""):
                raise LocalSemanticIndexError("corrupt_manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            windows = json.loads(windows_path.read_text(encoding="utf-8"))
        except LocalSemanticIndexError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalSemanticIndexError("corrupt_manifest") from exc
        if not isinstance(manifest, dict) or not isinstance(windows, list):
            raise LocalSemanticIndexError("corrupt_manifest")
        if (
            manifest.get("schema_version") != INDEX_SCHEMA_VERSION
            or manifest.get("build_id") != build_id
            or manifest.get("candidate_id") != CANDIDATE_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("representation_id") != REPRESENTATION_ID
            or manifest.get("embedding_dimension") != EMBEDDING_DIMENSION
        ):
            raise LocalSemanticIndexError("provider_mismatch")
        count = int(manifest.get("window_count") or -1)
        if count < 0 or len(windows) != count:
            raise LocalSemanticIndexError("corrupt_mapping")
        if (
            _sha256_file(windows_path) != str(manifest.get("windows_sha256") or "")
            or _sha256_file(vectors_path) != str(manifest.get("vectors_sha256") or "")
        ):
            raise LocalSemanticIndexError("corrupt_vectors")
        vectors = _read_vectors(
            vectors_path,
            count=count,
            dimension=EMBEDDING_DIMENSION,
        )
        self._validate_mapping(windows, manifest)
        return manifest, windows, vectors

    @staticmethod
    def _validate_mapping(windows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
        expected_position = 0
        offset = 0
        parents = manifest.get("parents") or []
        if not isinstance(parents, list):
            raise LocalSemanticIndexError("corrupt_mapping")
        for parent in parents:
            if not isinstance(parent, dict):
                raise LocalSemanticIndexError("corrupt_mapping")
            position = int(parent.get("parent_position", -1))
            count = int(parent.get("window_count", 0))
            parent_id = str(parent.get("parent_id") or "")
            if position != expected_position or count <= 0 or not parent_id:
                raise LocalSemanticIndexError("corrupt_mapping")
            rows = windows[offset : offset + count]
            if len(rows) != count or any(
                str(row.get("parent_id") or "") != parent_id
                or int(row.get("parent_position", -1)) != position
                or int(row.get("window_ordinal", -1)) != ordinal
                for ordinal, row in enumerate(rows)
            ):
                raise LocalSemanticIndexError("corrupt_mapping")
            expected_position += 1
            offset += count
        if offset != len(windows) or expected_position != int(manifest.get("parent_count") or -1):
            raise LocalSemanticIndexError("corrupt_mapping")

    @staticmethod
    def _validate_snapshot(manifest: dict[str, Any], snapshot: Any) -> None:
        if (
            manifest.get("committed_docs_sha256") != snapshot.committed_docs_sha256
            or manifest.get("corpus_fingerprint") != snapshot.corpus_fingerprint
            or manifest.get("ordered_parent_fingerprint")
            != _ordered_parent_fingerprint(snapshot)
            or int(manifest.get("parent_count") or -1) != len(snapshot.chunks)
        ):
            raise LocalSemanticIndexError("stale")
        parents = manifest.get("parents") or []
        if len(parents) != len(snapshot.chunks) or any(
            str(parent.get("parent_id") or "") != str(chunk.get("id") or "")
            or str(parent.get("text_sha256") or "")
            != snapshot.chunk_records[position]["text_sha256"]
            for position, (parent, chunk) in enumerate(zip(parents, snapshot.chunks))
        ):
            raise LocalSemanticIndexError("stale")

    def status(self) -> LocalSemanticIndexStatus:
        try:
            with db_write_lock(self.db_dir):
                snapshot = load_embedding_corpus_snapshot(self.db_dir)
                manifest, _windows, _vectors = self._read_published()
                self._validate_snapshot(manifest, snapshot)
        except LocalSemanticIndexError as exc:
            state = exc.code if exc.code in {"missing", "stale", "provider_mismatch"} else "corrupt"
            return LocalSemanticIndexStatus(state=state, reason_id=f"index_{state}")
        return LocalSemanticIndexStatus(
            state="ready",
            reason_id="ready",
            build_id=str(manifest["build_id"]),
            document_count=_document_count(snapshot),
            parent_count=len(snapshot.chunks),
            window_count=int(manifest["window_count"]),
        )

    def load(self, executor: E5SmallOnnxExecutor) -> LoadedLocalSemanticIndex:
        with db_write_lock(self.db_dir):
            snapshot = load_embedding_corpus_snapshot(self.db_dir)
            manifest, windows, vectors = self._read_published()
            self._validate_snapshot(manifest, snapshot)
        return LoadedLocalSemanticIndex(
            chunks=snapshot.chunks,
            windows=windows,
            vectors=vectors,
            manifest=manifest,
        )

    def build(
        self,
        executor: E5SmallOnnxExecutor,
        *,
        should_cancel: Callable[[], bool] = lambda: False,
        on_progress: Callable[[LocalSemanticIndexBuildProgress], None] = lambda _item: None,
        _before_commit: Callable[[Path], None] | None = None,
    ) -> LocalSemanticIndexBuildResult:
        started = time.perf_counter()
        with db_write_lock(self.index_dir):
            with db_write_lock(self.db_dir):
                snapshot = load_embedding_corpus_snapshot(self.db_dir)
            previous_manifest: dict[str, Any] = {}
            previous_vectors: list[tuple[float, ...]] = []
            try:
                previous_manifest, _previous_windows, previous_vectors = self._read_published()
                self._validate_snapshot(previous_manifest, snapshot)
                return LocalSemanticIndexBuildResult(
                    build_id=str(previous_manifest["build_id"]),
                    document_count=_document_count(snapshot),
                    parent_count=len(snapshot.chunks),
                    window_count=int(previous_manifest["window_count"]),
                    reused_parent_count=len(snapshot.chunks),
                    reused_window_count=int(previous_manifest["window_count"]),
                    embedded_parent_count=0,
                    embedded_window_count=0,
                    no_op=True,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )
            except LocalSemanticIndexError:
                try:
                    previous_manifest, _previous_windows, previous_vectors = self._read_published()
                except LocalSemanticIndexError:
                    previous_manifest = {}
                    previous_vectors = []
            if should_cancel():
                raise LocalSemanticIndexError("cancelled_by_user")
            try:
                windows = executor.build_windows(
                    snapshot.chunks,
                    should_cancel=should_cancel,
                )
            except E5EmbeddingCancelled as exc:
                raise LocalSemanticIndexError("cancelled_by_user") from exc
            parent_rows = self._parent_rows(snapshot, windows)
            reusable = self._reusable_vectors(previous_manifest, previous_vectors)
            vectors: list[tuple[float, ...] | None] = [None] * len(windows)
            missing: list[E5ParentWindow] = []
            missing_indices: list[int] = []
            reused_parents = 0
            reused_windows = 0
            for parent in parent_rows:
                start = int(parent["window_start"])
                count = int(parent["window_count"])
                cached = reusable.get(str(parent["text_sha256"]))
                if cached is not None and len(cached) == count:
                    reused_parents += 1
                    reused_windows += count
                    vectors[start : start + count] = cached
                else:
                    missing.extend(windows[start : start + count])
                    missing_indices.extend(range(start, start + count))
            if should_cancel():
                raise LocalSemanticIndexError("cancelled_by_user")

            def embedding_progress(item: E5EmbeddingProgress) -> None:
                on_progress(
                    LocalSemanticIndexBuildProgress(
                        completed_windows=reused_windows + item.completed_windows,
                        total_windows=len(windows),
                        reused_windows=reused_windows,
                        embedded_windows=item.completed_windows,
                    )
                )

            try:
                embedded = executor.embed_windows(
                    missing,
                    should_cancel=should_cancel,
                    on_progress=embedding_progress,
                )
            except E5EmbeddingCancelled as exc:
                raise LocalSemanticIndexError("cancelled_by_user") from exc
            for index, vector in zip(missing_indices, embedded):
                vectors[index] = vector
            if any(vector is None for vector in vectors):
                raise LocalSemanticIndexError("embedding_matrix_incomplete")
            if should_cancel():
                raise LocalSemanticIndexError("cancelled_by_user")
            with db_write_lock(self.db_dir):
                current = load_embedding_corpus_snapshot(self.db_dir)
            if (
                current.committed_docs_sha256 != snapshot.committed_docs_sha256
                or current.corpus_fingerprint != snapshot.corpus_fingerprint
                or current.chunk_records != snapshot.chunk_records
            ):
                raise LocalSemanticIndexError("corpus_changed")
            complete_vectors = [vector for vector in vectors if vector is not None]
            build_id = self._publish(
                snapshot=snapshot,
                parent_rows=parent_rows,
                windows=windows,
                vectors=complete_vectors,
                should_cancel=should_cancel,
                before_commit=_before_commit,
            )
        embedded_parent_ids = {window.parent_id for window in missing}
        return LocalSemanticIndexBuildResult(
            build_id=build_id,
            document_count=_document_count(snapshot),
            parent_count=len(snapshot.chunks),
            window_count=len(windows),
            reused_parent_count=reused_parents,
            reused_window_count=reused_windows,
            embedded_parent_count=len(embedded_parent_ids),
            embedded_window_count=len(embedded),
            no_op=False,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _parent_rows(snapshot: Any, windows: Sequence[E5ParentWindow]) -> list[dict[str, Any]]:
        grouped: list[list[E5ParentWindow]] = [[] for _chunk in snapshot.chunks]
        for window in windows:
            if window.parent_position < 0 or window.parent_position >= len(grouped):
                raise LocalSemanticIndexError("window_parent_invalid")
            grouped[window.parent_position].append(window)
        rows: list[dict[str, Any]] = []
        offset = 0
        for position, (chunk, parent_windows) in enumerate(zip(snapshot.chunks, grouped)):
            if not parent_windows or any(
                window.parent_id != str(chunk.get("id") or "")
                or window.window_ordinal != ordinal
                for ordinal, window in enumerate(parent_windows)
            ):
                raise LocalSemanticIndexError("window_mapping_invalid")
            rows.append(
                {
                    "parent_id": str(chunk["id"]),
                    "parent_position": position,
                    "text_sha256": snapshot.chunk_records[position]["text_sha256"],
                    "window_start": offset,
                    "window_count": len(parent_windows),
                }
            )
            offset += len(parent_windows)
        if offset != len(windows):
            raise LocalSemanticIndexError("window_mapping_invalid")
        return rows

    @staticmethod
    def _reusable_vectors(
        manifest: dict[str, Any],
        vectors: Sequence[tuple[float, ...]],
    ) -> dict[str, list[tuple[float, ...]]]:
        if (
            manifest.get("candidate_id") != CANDIDATE_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("representation_id") != REPRESENTATION_ID
        ):
            return {}
        reusable: dict[str, list[tuple[float, ...]]] = {}
        for parent in manifest.get("parents") or []:
            try:
                start = int(parent["window_start"])
                count = int(parent["window_count"])
                values = list(vectors[start : start + count])
            except (KeyError, TypeError, ValueError):
                continue
            signature = str(parent.get("text_sha256") or "")
            if signature and len(values) == count:
                reusable.setdefault(signature, values)
        return reusable

    def _publish(
        self,
        *,
        snapshot: Any,
        parent_rows: list[dict[str, Any]],
        windows: Sequence[E5ParentWindow],
        vectors: Sequence[Sequence[float]],
        should_cancel: Callable[[], bool],
        before_commit: Callable[[Path], None] | None,
    ) -> str:
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        build_id = uuid.uuid4().hex
        staging = Path(
            tempfile.mkdtemp(prefix=f".{build_id}.", suffix=".staging", dir=self.generations_dir)
        ).resolve(strict=False)
        generation = (self.generations_dir / build_id).resolve(strict=False)
        committed = False
        try:
            vectors_path = staging / "vectors.f32"
            windows_path = staging / "windows.json"
            _write_vectors(vectors_path, vectors)
            atomic_write_json(
                windows_path,
                [window.public_metadata() for window in windows],
            )
            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "build_id": build_id,
                "candidate_id": CANDIDATE_ID,
                "model_revision": MODEL_REVISION,
                "representation_id": REPRESENTATION_ID,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "committed_docs_sha256": snapshot.committed_docs_sha256,
                "corpus_fingerprint": snapshot.corpus_fingerprint,
                "ordered_parent_fingerprint": _ordered_parent_fingerprint(snapshot),
                "document_count": _document_count(snapshot),
                "parent_count": len(snapshot.chunks),
                "window_count": len(windows),
                "vectors_sha256": _sha256_file(vectors_path),
                "windows_sha256": _sha256_file(windows_path),
                "parents": parent_rows,
            }
            manifest_path = staging / "manifest.json"
            atomic_write_json(manifest_path, manifest)
            manifest_sha256 = _sha256_file(manifest_path)
            os.replace(staging, generation)
            if should_cancel():
                raise LocalSemanticIndexError("cancelled_by_user")
            if before_commit is not None:
                before_commit(generation)
            if should_cancel():
                raise LocalSemanticIndexError("cancelled_by_user")
            atomic_write_json(
                self.current_path,
                {
                    "schema_version": INDEX_SCHEMA_VERSION,
                    "build_id": build_id,
                    "manifest_sha256": manifest_sha256,
                },
            )
            committed = True
            return build_id
        except Exception as exc:
            if isinstance(exc, LocalSemanticIndexError):
                raise
            raise LocalSemanticIndexError("commit_failed") from exc
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if not committed and generation.exists():
                shutil.rmtree(generation, ignore_errors=True)
