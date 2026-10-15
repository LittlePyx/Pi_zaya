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
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import CHUNK_SCHEMA_VERSION
from .dense_retriever import DenseRetriever
from .embedding_provider import EmbeddingProvider, EmbeddingProviderDescriptor
from .embedding_vectors import normalize_embedding_batch
from .store import atomic_write_json, db_write_lock, load_all_chunks


EMBEDDING_INDEX_SCHEMA_VERSION = 1
_VECTOR_FORMAT = "float32-le-l2"
_CURRENT_FILE = "current.json"
_BUILD_STATUS_FILE = "last_build.json"
_BUILD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_PATH_META_KEYS = {
    "path",
    "source_path",
    "pdf_path",
    "md_path",
    "markdown_path",
    "asset_path",
}


class EmbeddingIndexError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "embedding_index_error")


class EmbeddingIndexBuildError(EmbeddingIndexError):
    pass


@dataclass(frozen=True)
class EmbeddingIndexBuildResult:
    build_id: str
    corpus_fingerprint: str
    document_count: int
    chunk_count: int
    embedded_document_count: int
    embedded_chunk_count: int
    reused_document_count: int
    reused_chunk_count: int
    no_op: bool
    duration_ms: float


@dataclass(frozen=True)
class EmbeddingIndexStatus:
    state: str
    reason: str
    build_id: str = ""
    corpus_fingerprint: str = ""
    document_count: int = 0
    chunk_count: int = 0
    provider_identity: str = ""
    last_build_state: str = ""
    last_error_category: str = ""


@dataclass(frozen=True)
class LoadedEmbeddingIndex:
    chunks: list[dict]
    embeddings: list[tuple[float, ...]]
    manifest: dict[str, Any]

    def as_retriever(
        self,
        provider: EmbeddingProvider,
        *,
        min_score: float = 0.0,
    ) -> DenseRetriever:
        descriptor = _provider_descriptor(provider)
        expected = str(self.manifest.get("provider_identity") or "")
        if descriptor.identity != expected:
            raise EmbeddingIndexError(
                "provider_mismatch",
                "embedding query provider does not match the committed index",
            )
        return DenseRetriever(
            self.chunks,
            self.embeddings,
            provider.embed_query,
            min_score=min_score,
        )


@dataclass(frozen=True)
class _DocumentSpec:
    token: str
    positions: tuple[int, ...]
    signature: str
    embedding_signature: str


@dataclass(frozen=True)
class _CorpusSnapshot:
    chunks: list[dict]
    committed_docs_sha256: str
    corpus_fingerprint: str
    chunk_records: list[dict[str, str]]
    documents: tuple[_DocumentSpec, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _required_int(
    payload: dict[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    try:
        value = int(payload.get(key))
    except (TypeError, ValueError) as exc:
        raise EmbeddingIndexError(
            "corrupt_manifest",
            f"embedding manifest {key} is invalid",
        ) from exc
    if value < minimum:
        raise EmbeddingIndexError(
            "corrupt_manifest",
            f"embedding manifest {key} is invalid",
        )
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _path_independent_meta(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _path_independent_meta(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _PATH_META_KEYS
            and not str(key).lower().endswith("_path")
        }
    if isinstance(value, (list, tuple)):
        return [_path_independent_meta(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _source_document_id(chunk: dict, position: int) -> str:
    chunk_id = str(chunk.get("id") or "").strip()
    if not chunk_id:
        raise EmbeddingIndexBuildError(
            "invalid_chunk",
            f"chunk at position {position} has no stable id",
        )
    if ":" not in chunk_id:
        raise EmbeddingIndexBuildError(
            "invalid_chunk",
            f"chunk id {chunk_id!r} does not contain a document prefix",
        )
    return chunk_id.split(":", 1)[0]


def _snapshot_from_db(db_dir: Path) -> _CorpusSnapshot:
    docs_path = db_dir / "docs.json"
    docs_bytes = docs_path.read_bytes() if docs_path.exists() else b""
    chunks = load_all_chunks(db_dir)
    seen_chunk_ids: set[str] = set()
    grouped_positions: dict[str, list[int]] = defaultdict(list)
    chunk_records: list[dict[str, str]] = []

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise EmbeddingIndexBuildError(
                "invalid_chunk",
                f"chunk at position {position} is not an object",
            )
        chunk_id = str(chunk.get("id") or "").strip()
        if chunk_id in seen_chunk_ids:
            raise EmbeddingIndexBuildError(
                "duplicate_chunk_id",
                f"duplicate chunk id: {chunk_id}",
            )
        seen_chunk_ids.add(chunk_id)
        document_id = _source_document_id(chunk, position)
        grouped_positions[document_id].append(position)
        text_sha256 = _sha256_bytes(str(chunk.get("text") or "").encode("utf-8"))
        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
        metadata_sha256 = _sha256_bytes(_canonical_json(_path_independent_meta(meta)))
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "text_sha256": text_sha256,
                "metadata_sha256": metadata_sha256,
            }
        )

    pending_documents: list[tuple[str, tuple[int, ...], str, str]] = []
    for source_document_id, raw_positions in grouped_positions.items():
        positions = tuple(raw_positions)
        text_hashes = [chunk_records[position]["text_sha256"] for position in positions]
        content_rows = [
            {
                "text_sha256": chunk_records[position]["text_sha256"],
                "metadata_sha256": chunk_records[position]["metadata_sha256"],
            }
            for position in positions
        ]
        embedding_signature = _sha256_bytes(_canonical_json(text_hashes))
        signature = _sha256_bytes(_canonical_json(content_rows))
        pending_documents.append(
            (source_document_id, positions, signature, embedding_signature)
        )

    occurrences: dict[str, int] = defaultdict(int)
    documents: list[_DocumentSpec] = []
    for _source_id, positions, signature, embedding_signature in sorted(
        pending_documents,
        key=lambda item: (item[2], item[3], item[0]),
    ):
        occurrence = occurrences[signature]
        occurrences[signature] += 1
        documents.append(
            _DocumentSpec(
                token=f"{signature[:20]}-{occurrence}",
                positions=positions,
                signature=signature,
                embedding_signature=embedding_signature,
            )
        )

    corpus_fingerprint = _sha256_bytes(
        _canonical_json(
            {
                "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                "document_signatures": sorted(
                    document.signature for document in documents
                ),
            }
        )
    )
    return _CorpusSnapshot(
        chunks=chunks,
        committed_docs_sha256=_sha256_bytes(docs_bytes),
        corpus_fingerprint=corpus_fingerprint,
        chunk_records=chunk_records,
        documents=tuple(documents),
    )
def _provider_descriptor(provider: EmbeddingProvider) -> EmbeddingProviderDescriptor:
    descriptor = getattr(provider, "descriptor", None)
    if not isinstance(descriptor, EmbeddingProviderDescriptor):
        raise TypeError("embedding provider must expose an EmbeddingProviderDescriptor")
    return descriptor


def _write_float32_matrix(
    path: Path,
    vectors: Sequence[Sequence[float]],
) -> None:
    flat = array("f", (value for vector in vectors for value in vector))
    if sys.byteorder != "little":
        flat.byteswap()
    with path.open("wb") as handle:
        flat.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())


def _read_float32_matrix(
    path: Path,
    *,
    count: int,
    dimension: int,
) -> list[tuple[float, ...]]:
    expected_values = count * dimension
    expected_bytes = expected_values * 4
    if path.stat().st_size != expected_bytes:
        raise EmbeddingIndexError(
            "corrupt_vectors",
            "embedding vector file size does not match its manifest",
        )
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, expected_values)
    if sys.byteorder != "little":
        values.byteswap()
    return [
        tuple(float(value) for value in values[offset : offset + dimension])
        for offset in range(0, expected_values, dimension)
    ]


class EmbeddingIndexStore:
    """Atomic exact-vector snapshots for the committed local chunk corpus.

    Builds only embed documents whose text sequence changed. A complete new
    matrix is published through one atomic pointer replacement, so readers
    never observe a partial generation. This store is not connected to the
    product retrieval path until explicit mode, consent, and quality gates land.
    """

    def __init__(self, db_dir: Path, *, index_dir: Path | None = None) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)
        target = index_dir if index_dir is not None else self.db_dir / "dense_index"
        self.index_dir = Path(target).expanduser().resolve(strict=False)
        self.generations_dir = self.index_dir / "generations"
        self.current_path = self.index_dir / _CURRENT_FILE
        self.build_status_path = self.index_dir / _BUILD_STATUS_FILE

    def _read_last_build(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.build_status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_build_status(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.build_status_path, payload)

    def _read_pointer(self) -> dict[str, Any]:
        if not self.current_path.exists():
            raise EmbeddingIndexError("missing", "no embedding index has been committed")
        try:
            payload = json.loads(self.current_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise EmbeddingIndexError(
                "corrupt_pointer",
                "embedding index pointer cannot be read",
            ) from exc
        build_id = str(payload.get("build_id") or "") if isinstance(payload, dict) else ""
        if not _BUILD_ID_RE.fullmatch(build_id):
            raise EmbeddingIndexError(
                "corrupt_pointer",
                "embedding index pointer contains an invalid generation",
            )
        return payload

    def _load_published(self) -> tuple[dict[str, Any], list[tuple[float, ...]]]:
        pointer = self._read_pointer()
        build_id = str(pointer["build_id"])
        generation_dir = self.generations_dir / build_id
        manifest_path = generation_dir / "manifest.json"
        vectors_path = generation_dir / "vectors.f32"
        if not manifest_path.is_file() or not vectors_path.is_file():
            raise EmbeddingIndexError(
                "missing_generation",
                "committed embedding generation is incomplete",
            )
        expected_manifest_sha = str(pointer.get("manifest_sha256") or "")
        if not expected_manifest_sha or _sha256_file(manifest_path) != expected_manifest_sha:
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding manifest checksum does not match the commit pointer",
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding manifest cannot be read",
            ) from exc
        if not isinstance(manifest, dict):
            raise EmbeddingIndexError("corrupt_manifest", "embedding manifest is not an object")
        if _required_int(manifest, "schema_version", minimum=1) != EMBEDDING_INDEX_SCHEMA_VERSION:
            raise EmbeddingIndexError(
                "schema_mismatch",
                "embedding index schema is unsupported",
            )
        if str(manifest.get("build_id") or "") != build_id:
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding manifest generation does not match its pointer",
            )
        if str(manifest.get("vector_format") or "") != _VECTOR_FORMAT:
            raise EmbeddingIndexError(
                "format_mismatch",
                "embedding vector format is unsupported",
            )
        dimension = _required_int(manifest, "dimension", minimum=1)
        count = _required_int(manifest, "chunk_count", minimum=0)
        chunks = manifest.get("chunks")
        documents = manifest.get("documents")
        if not isinstance(chunks, list) or len(chunks) != count:
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding manifest chunk count is inconsistent",
            )
        document_count = _required_int(manifest, "document_count", minimum=0)
        if not isinstance(documents, list) or len(documents) != document_count:
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding manifest document count is inconsistent",
            )
        covered_indices: list[int] = []
        for document in documents:
            if not isinstance(document, dict) or document.get("status") != "ready":
                raise EmbeddingIndexError(
                    "corrupt_manifest",
                    "embedding manifest contains an incomplete document",
                )
            try:
                indices = [int(index) for index in document.get("vector_indices") or []]
            except (TypeError, ValueError) as exc:
                raise EmbeddingIndexError(
                    "corrupt_manifest",
                    "embedding manifest contains invalid vector indices",
                ) from exc
            if len(indices) != _required_int(document, "chunk_count", minimum=0):
                raise EmbeddingIndexError(
                    "corrupt_manifest",
                    "embedding document chunk count is inconsistent",
                )
            covered_indices.extend(indices)
        if sorted(covered_indices) != list(range(count)):
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding document vectors do not cover the matrix exactly once",
            )
        raw_provider = manifest.get("provider")
        try:
            stored_descriptor = EmbeddingProviderDescriptor(**raw_provider)
        except (TypeError, ValueError) as exc:
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding provider descriptor is invalid",
            ) from exc
        if (
            stored_descriptor.identity != str(manifest.get("provider_identity") or "")
            or stored_descriptor.dimension != dimension
            or stored_descriptor.normalization != str(manifest.get("normalization") or "")
        ):
            raise EmbeddingIndexError(
                "corrupt_manifest",
                "embedding provider descriptor is inconsistent",
            )
        if _sha256_file(vectors_path) != str(manifest.get("vectors_sha256") or ""):
            raise EmbeddingIndexError(
                "corrupt_vectors",
                "embedding vector checksum does not match its manifest",
            )
        vectors = _read_float32_matrix(
            vectors_path,
            count=count,
            dimension=dimension,
        )
        vectors = normalize_embedding_batch(
            vectors,
            expected_count=count,
            expected_dim=dimension,
        )
        return manifest, vectors

    @staticmethod
    def _validate_snapshot(
        manifest: dict[str, Any],
        snapshot: _CorpusSnapshot,
    ) -> None:
        if _required_int(manifest, "chunk_schema_version", minimum=1) != CHUNK_SCHEMA_VERSION:
            raise EmbeddingIndexError(
                "chunk_schema_mismatch",
                "embedding index uses a different chunk schema",
            )
        if str(manifest.get("committed_docs_sha256") or "") != snapshot.committed_docs_sha256:
            raise EmbeddingIndexError(
                "stale",
                "embedding index does not match the committed document index",
            )
        if str(manifest.get("corpus_fingerprint") or "") != snapshot.corpus_fingerprint:
            raise EmbeddingIndexError(
                "stale",
                "embedding index does not match the current path-independent corpus",
            )
        if manifest.get("chunks") != snapshot.chunk_records:
            raise EmbeddingIndexError(
                "stale",
                "embedding index chunk order or metadata has changed",
            )

    @staticmethod
    def _validate_provider(
        manifest: dict[str, Any],
        descriptor: EmbeddingProviderDescriptor,
    ) -> None:
        if str(manifest.get("provider_identity") or "") != descriptor.identity:
            raise EmbeddingIndexError(
                "provider_mismatch",
                "embedding index was built by a different model or revision",
            )
        if _required_int(manifest, "dimension", minimum=1) != descriptor.dimension:
            raise EmbeddingIndexError(
                "dimension_mismatch",
                "embedding index dimension does not match the provider",
            )

    def load(self, provider: EmbeddingProvider) -> LoadedEmbeddingIndex:
        descriptor = _provider_descriptor(provider)
        with db_write_lock(self.db_dir):
            snapshot = _snapshot_from_db(self.db_dir)
            manifest, vectors = self._load_published()
            self._validate_provider(manifest, descriptor)
            self._validate_snapshot(manifest, snapshot)
        return LoadedEmbeddingIndex(
            chunks=snapshot.chunks,
            embeddings=vectors,
            manifest=manifest,
        )

    def status(self, provider: EmbeddingProvider | None = None) -> EmbeddingIndexStatus:
        last_build = self._read_last_build()
        common = {
            "last_build_state": str(last_build.get("state") or ""),
            "last_error_category": str(last_build.get("error_category") or ""),
        }
        try:
            with db_write_lock(self.db_dir):
                snapshot = _snapshot_from_db(self.db_dir)
                manifest, _vectors = self._load_published()
                if provider is not None:
                    self._validate_provider(manifest, _provider_descriptor(provider))
                self._validate_snapshot(manifest, snapshot)
        except EmbeddingIndexError as exc:
            state = exc.code if exc.code in {"missing", "stale", "provider_mismatch"} else "corrupt"
            return EmbeddingIndexStatus(
                state=state,
                reason=str(exc),
                **common,
            )
        return EmbeddingIndexStatus(
            state="ready",
            reason="embedding index matches the committed corpus and provider",
            build_id=str(manifest.get("build_id") or ""),
            corpus_fingerprint=str(manifest.get("corpus_fingerprint") or ""),
            document_count=int(manifest.get("document_count") or 0),
            chunk_count=int(manifest.get("chunk_count") or 0),
            provider_identity=str(manifest.get("provider_identity") or ""),
            **common,
        )

    def build(
        self,
        provider: EmbeddingProvider,
        *,
        _before_commit: Callable[[Path], None] | None = None,
    ) -> EmbeddingIndexBuildResult:
        descriptor = _provider_descriptor(provider)
        started = time.perf_counter()
        attempt_id = uuid.uuid4().hex
        with db_write_lock(self.index_dir):
            with db_write_lock(self.db_dir):
                snapshot = _snapshot_from_db(self.db_dir)

            previous_manifest: dict[str, Any] = {}
            previous_vectors: list[tuple[float, ...]] = []
            try:
                previous_manifest, previous_vectors = self._load_published()
            except EmbeddingIndexError:
                pass

            if (
                previous_manifest
                and str(previous_manifest.get("provider_identity") or "") == descriptor.identity
                and str(previous_manifest.get("committed_docs_sha256") or "")
                == snapshot.committed_docs_sha256
                and str(previous_manifest.get("corpus_fingerprint") or "")
                == snapshot.corpus_fingerprint
                and previous_manifest.get("chunks") == snapshot.chunk_records
            ):
                return EmbeddingIndexBuildResult(
                    build_id=str(previous_manifest.get("build_id") or ""),
                    corpus_fingerprint=snapshot.corpus_fingerprint,
                    document_count=len(snapshot.documents),
                    chunk_count=len(snapshot.chunks),
                    embedded_document_count=0,
                    embedded_chunk_count=0,
                    reused_document_count=len(snapshot.documents),
                    reused_chunk_count=len(snapshot.chunks),
                    no_op=True,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )

            document_status = [
                {
                    "document_token": document.token,
                    "signature": document.signature,
                    "embedding_signature": document.embedding_signature,
                    "chunk_count": len(document.positions),
                    "state": "pending",
                }
                for document in snapshot.documents
            ]
            build_status: dict[str, Any] = {
                "schema_version": EMBEDDING_INDEX_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "state": "building",
                "started_at": _utc_now(),
                "provider_identity": descriptor.identity,
                "corpus_fingerprint": snapshot.corpus_fingerprint,
                "committed_docs_sha256": snapshot.committed_docs_sha256,
                "documents": document_status,
            }
            self._write_build_status(build_status)

            reusable: dict[str, list[tuple[float, ...]]] = {}
            if (
                previous_manifest
                and str(previous_manifest.get("provider_identity") or "") == descriptor.identity
                and len(previous_vectors) == int(previous_manifest.get("chunk_count") or 0)
            ):
                for document in previous_manifest.get("documents") or []:
                    if not isinstance(document, dict):
                        continue
                    signature = str(document.get("embedding_signature") or "")
                    raw_indices = document.get("vector_indices") or []
                    try:
                        indices = [int(index) for index in raw_indices]
                        vectors = [previous_vectors[index] for index in indices]
                    except (IndexError, TypeError, ValueError):
                        continue
                    if signature and vectors:
                        reusable.setdefault(signature, vectors)

            vectors_by_position: list[tuple[float, ...] | None] = [None] * len(snapshot.chunks)
            embedded_documents = 0
            embedded_chunks = 0
            reused_documents = 0
            reused_chunks = 0
            try:
                for document_index, document in enumerate(snapshot.documents):
                    cached = reusable.get(document.embedding_signature)
                    if cached is not None and len(cached) == len(document.positions):
                        normalized = cached
                        action = "reused"
                        reused_documents += 1
                        reused_chunks += len(document.positions)
                    else:
                        texts = [
                            str(snapshot.chunks[position].get("text") or "")
                            for position in document.positions
                        ]
                        normalized = normalize_embedding_batch(
                            provider.embed_documents(texts),
                            expected_count=len(texts),
                            expected_dim=descriptor.dimension,
                        )
                        action = "embedded"
                        embedded_documents += 1
                        embedded_chunks += len(document.positions)
                    for position, vector in zip(document.positions, normalized):
                        vectors_by_position[position] = vector
                    document_status[document_index]["state"] = action
                    self._write_build_status(build_status)
            except Exception as exc:
                document_status[document_index]["state"] = "error"
                build_status["state"] = "failed"
                build_status["completed_at"] = _utc_now()
                build_status["error_category"] = type(exc).__name__
                self._write_build_status(build_status)
                raise EmbeddingIndexBuildError(
                    "provider_failed",
                    "embedding provider failed; the previous committed index was preserved",
                ) from exc

            if any(vector is None for vector in vectors_by_position):
                raise EmbeddingIndexBuildError(
                    "incomplete_build",
                    "embedding build did not produce every chunk vector",
                )
            vectors = [vector for vector in vectors_by_position if vector is not None]

            with db_write_lock(self.db_dir):
                current_snapshot = _snapshot_from_db(self.db_dir)
            if (
                current_snapshot.committed_docs_sha256 != snapshot.committed_docs_sha256
                or current_snapshot.corpus_fingerprint != snapshot.corpus_fingerprint
                or current_snapshot.chunk_records != snapshot.chunk_records
            ):
                build_status["state"] = "failed"
                build_status["completed_at"] = _utc_now()
                build_status["error_category"] = "corpus_changed"
                self._write_build_status(build_status)
                raise EmbeddingIndexBuildError(
                    "corpus_changed",
                    "the committed corpus changed during embedding; no index was published",
                )

            build_id = uuid.uuid4().hex
            manifest: dict[str, Any] = {
                "schema_version": EMBEDDING_INDEX_SCHEMA_VERSION,
                "build_id": build_id,
                "created_at": _utc_now(),
                "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                "committed_docs_sha256": snapshot.committed_docs_sha256,
                "corpus_fingerprint": snapshot.corpus_fingerprint,
                "provider": descriptor.to_dict(),
                "provider_identity": descriptor.identity,
                "normalization": descriptor.normalization,
                "vector_format": _VECTOR_FORMAT,
                "dimension": descriptor.dimension,
                "document_count": len(snapshot.documents),
                "chunk_count": len(snapshot.chunks),
                "chunks": snapshot.chunk_records,
                "documents": [
                    {
                        "document_token": document.token,
                        "signature": document.signature,
                        "embedding_signature": document.embedding_signature,
                        "chunk_count": len(document.positions),
                        "vector_indices": list(document.positions),
                        "status": "ready",
                        "build_action": document_status[index]["state"],
                    }
                    for index, document in enumerate(snapshot.documents)
                ],
            }
            try:
                generation_dir = self._publish_generation(
                    manifest,
                    vectors,
                    _before_commit=_before_commit,
                )
            except Exception as exc:
                build_status["state"] = "failed"
                build_status["completed_at"] = _utc_now()
                build_status["error_category"] = "commit_failed"
                self._write_build_status(build_status)
                raise EmbeddingIndexBuildError(
                    "commit_failed",
                    "embedding generation could not be committed; the previous index was preserved",
                ) from exc
            build_status["state"] = "ready"
            build_status["completed_at"] = _utc_now()
            build_status["build_id"] = build_id
            build_status["generation"] = generation_dir.name
            try:
                self._write_build_status(build_status)
            except OSError:
                # The atomic search commit already succeeded. A diagnostic
                # status-file failure must not turn a usable index into a
                # reported build failure.
                pass
            return EmbeddingIndexBuildResult(
                build_id=build_id,
                corpus_fingerprint=snapshot.corpus_fingerprint,
                document_count=len(snapshot.documents),
                chunk_count=len(snapshot.chunks),
                embedded_document_count=embedded_documents,
                embedded_chunk_count=embedded_chunks,
                reused_document_count=reused_documents,
                reused_chunk_count=reused_chunks,
                no_op=False,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

    def _publish_generation(
        self,
        manifest: dict[str, Any],
        vectors: Sequence[Sequence[float]],
        *,
        _before_commit: Callable[[Path], None] | None,
    ) -> Path:
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(prefix=".tmp-", dir=str(self.generations_dir))
        ).resolve(strict=False)
        final_dir = (self.generations_dir / str(manifest["build_id"])).resolve(strict=False)
        try:
            vectors_path = temp_dir / "vectors.f32"
            _write_float32_matrix(vectors_path, vectors)
            manifest["vectors_sha256"] = _sha256_file(vectors_path)
            manifest_path = temp_dir / "manifest.json"
            atomic_write_json(manifest_path, manifest)
            manifest_sha256 = _sha256_file(manifest_path)
            os.replace(temp_dir, final_dir)
            _fsync_directory(self.generations_dir)
            if _before_commit is not None:
                _before_commit(final_dir)
            atomic_write_json(
                self.current_path,
                {
                    "schema_version": EMBEDDING_INDEX_SCHEMA_VERSION,
                    "build_id": str(manifest["build_id"]),
                    "manifest_sha256": manifest_sha256,
                    "committed_at": _utc_now(),
                },
            )
            return final_dir
        except Exception:
            if temp_dir.parent == self.generations_dir.resolve(strict=False) and temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
