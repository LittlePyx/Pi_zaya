from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol
from urllib.parse import quote, urlparse

from .store import atomic_write_json, db_write_lock


MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "03415a4be176a1620747c692ed433219fabc3def"
CANDIDATE_ID = "multilingual_e5_small_official_onnx_v1"
TOTAL_ARTIFACT_BYTES = 492421554


@dataclass(frozen=True)
class LocalSemanticArtifact:
    repository_path: str
    local_name: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        path = quote(self.repository_path, safe="/")
        return f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{path}"


ARTIFACTS = (
    LocalSemanticArtifact(
        "onnx/model.onnx",
        "model.onnx",
        470268510,
        "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    ),
    LocalSemanticArtifact(
        "onnx/config.json",
        "tokenizer/config.json",
        653,
        "bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6",
    ),
    LocalSemanticArtifact(
        "onnx/sentencepiece.bpe.model",
        "tokenizer/sentencepiece.bpe.model",
        5069051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    LocalSemanticArtifact(
        "onnx/special_tokens_map.json",
        "tokenizer/special_tokens_map.json",
        167,
        "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    ),
    LocalSemanticArtifact(
        "onnx/tokenizer.json",
        "tokenizer/tokenizer.json",
        17082730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
    LocalSemanticArtifact(
        "onnx/tokenizer_config.json",
        "tokenizer/tokenizer_config.json",
        443,
        "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    ),
)


class ArtifactTransfer(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_bytes(self) -> Iterable[bytes]:
        ...

    def close(self) -> None:
        ...


class ArtifactTransport(Protocol):
    def open(self, artifact: LocalSemanticArtifact, *, start: int) -> ArtifactTransfer:
        ...


class _RequestsTransfer:
    def __init__(self, response: object) -> None:
        self._response = response
        self.status_code = int(getattr(response, "status_code", 0) or 0)
        self.headers = dict(getattr(response, "headers", {}) or {})

    def iter_bytes(self) -> Iterable[bytes]:
        iterator = getattr(self._response, "iter_content")
        return iterator(chunk_size=1024 * 1024)

    def close(self) -> None:
        close = getattr(self._response, "close", None)
        if callable(close):
            close()


class RequestsArtifactTransport:
    """Bounded HTTPS transport; construction itself never starts a download."""

    def __init__(
        self,
        *,
        request_get: Callable[..., object] | None = None,
        connect_timeout_seconds: float = 15.0,
        read_timeout_seconds: float = 120.0,
    ) -> None:
        if request_get is None:
            import requests

            request_get = requests.get
        self._request_get = request_get
        self._timeout = (
            max(1.0, float(connect_timeout_seconds)),
            max(1.0, float(read_timeout_seconds)),
        )

    def open(self, artifact: LocalSemanticArtifact, *, start: int) -> ArtifactTransfer:
        _validate_source(artifact)
        headers = {"Accept-Encoding": "identity"}
        if int(start) > 0:
            headers["Range"] = f"bytes={int(start)}-"
        try:
            response = self._request_get(
                artifact.url,
                headers=headers,
                stream=True,
                timeout=self._timeout,
                allow_redirects=True,
            )
        except Exception as exc:
            raise ArtifactPreparationError("artifact_transport_failed") from exc
        return _RequestsTransfer(response)


@dataclass(frozen=True)
class ArtifactProgress:
    phase: str
    completed_units: int
    total_units: int
    completed_bytes: int
    total_bytes: int
    eta_seconds: int | None


@dataclass(frozen=True)
class ArtifactPreparationResult:
    downloaded_bytes: int
    reused_bytes: int
    resumed_bytes: int
    file_count: int


class ArtifactPreparationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(str(code or "artifact_preparation_failed"))
        self.code = str(code or "artifact_preparation_failed")


class ArtifactPreparationCancelled(ArtifactPreparationError):
    def __init__(self) -> None:
        super().__init__("cancelled_by_user")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_file(path: Path, artifact: LocalSemanticArtifact) -> bool:
    try:
        return path.stat().st_size == artifact.bytes and _sha256_file(path) == artifact.sha256
    except OSError:
        return False


def _safe_destination(root: Path, artifact: LocalSemanticArtifact) -> Path:
    destination = (root / artifact.local_name).resolve(strict=False)
    if destination != root and root not in destination.parents:
        raise ArtifactPreparationError("artifact_path_invalid")
    return destination


def _validate_source(artifact: LocalSemanticArtifact) -> None:
    parsed = urlparse(artifact.url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ArtifactPreparationError("artifact_source_not_allowed")
    if f"/resolve/{MODEL_REVISION}/" not in parsed.path:
        raise ArtifactPreparationError("artifact_revision_not_immutable")


def _valid_resume_response(
    transfer: ArtifactTransfer,
    *,
    start: int,
    expected_size: int,
) -> bool:
    if start <= 0:
        return int(transfer.status_code) == 200
    if int(transfer.status_code) != 206:
        return False
    raw = str(transfer.headers.get("Content-Range") or "").strip().lower()
    return raw == f"bytes {start}-{expected_size - 1}/{expected_size}"


class VerifiedLocalSemanticArtifacts:
    """Prepare exact immutable model files; never publish unverified bytes."""

    def __init__(
        self,
        root: Path | str,
        *,
        artifacts: tuple[LocalSemanticArtifact, ...] = ARTIFACTS,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.artifacts = artifacts
        self.ready_path = self.root / "ready.json"

    def is_ready(self) -> bool:
        return all(
            _valid_file(_safe_destination(self.root, artifact), artifact)
            for artifact in self.artifacts
        )

    def quick_ready(self) -> bool:
        """Cheap polling check; full hashes are still required before execution."""

        try:
            payload = json.loads(self.ready_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        expected = [
            {
                "local_name": artifact.local_name,
                "bytes": artifact.bytes,
                "sha256": artifact.sha256,
            }
            for artifact in self.artifacts
        ]
        if (
            payload.get("schema_version") != 1
            or payload.get("candidate_id") != CANDIDATE_ID
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("files") != expected
        ):
            return False
        try:
            return all(
                _safe_destination(self.root, artifact).stat().st_size == artifact.bytes
                for artifact in self.artifacts
            )
        except OSError:
            return False

    def prepare(
        self,
        transport: ArtifactTransport,
        *,
        should_cancel: Callable[[], bool] = lambda: False,
        on_progress: Callable[[ArtifactProgress], None] = lambda _progress: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> ArtifactPreparationResult:
        total_bytes = sum(artifact.bytes for artifact in self.artifacts)
        if total_bytes > TOTAL_ARTIFACT_BYTES:
            raise ArtifactPreparationError("artifact_size_budget_exceeded")
        started = clock()
        downloaded_bytes = 0
        reused_bytes = 0
        resumed_bytes = 0
        completed_bytes = 0
        with db_write_lock(self.root):
            for index, artifact in enumerate(self.artifacts):
                if should_cancel():
                    raise ArtifactPreparationCancelled()
                _validate_source(artifact)
                destination = _safe_destination(self.root, artifact)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if _valid_file(destination, artifact):
                    reused_bytes += artifact.bytes
                    completed_bytes += artifact.bytes
                    self._emit_progress(
                        on_progress,
                        started=started,
                        now=clock(),
                        completed_units=index + 1,
                        completed_bytes=completed_bytes,
                        total_bytes=total_bytes,
                    )
                    continue
                partial = destination.with_name(destination.name + ".part")
                try:
                    partial_size = partial.stat().st_size
                except OSError:
                    partial_size = 0
                if partial_size < 0 or partial_size > artifact.bytes:
                    partial.unlink(missing_ok=True)
                    partial_size = 0
                try:
                    transfer = transport.open(artifact, start=partial_size)
                except ArtifactPreparationError:
                    raise
                except Exception as exc:
                    raise ArtifactPreparationError("artifact_transport_failed") from exc
                if not _valid_resume_response(
                    transfer,
                    start=partial_size,
                    expected_size=artifact.bytes,
                ):
                    transfer.close()
                    if partial_size <= 0:
                        raise ArtifactPreparationError("artifact_response_invalid")
                    partial.unlink(missing_ok=True)
                    partial_size = 0
                    try:
                        transfer = transport.open(artifact, start=0)
                    except ArtifactPreparationError:
                        raise
                    except Exception as exc:
                        raise ArtifactPreparationError("artifact_transport_failed") from exc
                    if not _valid_resume_response(
                        transfer,
                        start=0,
                        expected_size=artifact.bytes,
                    ):
                        transfer.close()
                        raise ArtifactPreparationError("artifact_resume_response_invalid")
                resumed_bytes += partial_size
                completed_bytes += partial_size
                mode = "ab" if partial_size else "wb"
                try:
                    with partial.open(mode) as handle:
                        for block in transfer.iter_bytes():
                            if should_cancel():
                                handle.flush()
                                os.fsync(handle.fileno())
                                raise ArtifactPreparationCancelled()
                            if not isinstance(block, bytes) or not block:
                                continue
                            if handle.tell() + len(block) > artifact.bytes:
                                raise ArtifactPreparationError("artifact_response_too_large")
                            handle.write(block)
                            downloaded_bytes += len(block)
                            completed_bytes += len(block)
                            self._emit_progress(
                                on_progress,
                                started=started,
                                now=clock(),
                                completed_units=index,
                                completed_bytes=completed_bytes,
                                total_bytes=total_bytes,
                            )
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    transfer.close()
                if not _valid_file(partial, artifact):
                    raise ArtifactPreparationError("artifact_integrity_mismatch")
                os.replace(partial, destination)
                self._emit_progress(
                    on_progress,
                    started=started,
                    now=clock(),
                    completed_units=index + 1,
                    completed_bytes=completed_bytes,
                    total_bytes=total_bytes,
                )
            atomic_write_json(
                self.ready_path,
                {
                    "schema_version": 1,
                    "candidate_id": CANDIDATE_ID,
                    "model_revision": MODEL_REVISION,
                    "files": [
                        {
                            "local_name": artifact.local_name,
                            "bytes": artifact.bytes,
                            "sha256": artifact.sha256,
                        }
                        for artifact in self.artifacts
                    ],
                },
            )
        return ArtifactPreparationResult(
            downloaded_bytes=downloaded_bytes,
            reused_bytes=reused_bytes,
            resumed_bytes=resumed_bytes,
            file_count=len(self.artifacts),
        )

    def _emit_progress(
        self,
        callback: Callable[[ArtifactProgress], None],
        *,
        started: float,
        now: float,
        completed_units: int,
        completed_bytes: int,
        total_bytes: int,
    ) -> None:
        elapsed = max(0.0, now - started)
        eta: int | None = None
        if completed_bytes > 0 and elapsed > 0:
            eta = max(0, round((total_bytes - completed_bytes) * elapsed / completed_bytes))
        callback(
            ArtifactProgress(
                phase="artifacts",
                completed_units=completed_units,
                total_units=len(self.artifacts),
                completed_bytes=completed_bytes,
                total_bytes=total_bytes,
                eta_seconds=eta,
            )
        )
