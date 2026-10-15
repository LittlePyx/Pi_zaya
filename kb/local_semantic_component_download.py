from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol
from urllib.parse import urlparse

from .local_semantic_component import (
    LocalSemanticComponentCancelled,
    LocalSemanticComponentError,
    LocalSemanticComponentManifest,
)
from .store import db_write_lock


class ComponentArchiveTransfer(Protocol):
    status_code: int
    headers: Mapping[str, str]
    final_url: str

    def iter_bytes(self) -> Iterable[bytes]:
        ...

    def close(self) -> None:
        ...


class ComponentArchiveTransport(Protocol):
    def open(
        self,
        manifest: LocalSemanticComponentManifest,
        *,
        start: int,
    ) -> ComponentArchiveTransfer:
        ...


class _RequestsTransfer:
    def __init__(self, response: object) -> None:
        self._response = response
        self.status_code = int(getattr(response, "status_code", 0) or 0)
        self.headers = dict(getattr(response, "headers", {}) or {})
        self.final_url = str(getattr(response, "url", "") or "")

    def iter_bytes(self) -> Iterable[bytes]:
        iterator = getattr(self._response, "iter_content")
        return iterator(chunk_size=1024 * 1024)

    def close(self) -> None:
        close = getattr(self._response, "close", None)
        if callable(close):
            close()


class RequestsComponentArchiveTransport:
    """Open a bounded component transfer; construction never performs I/O."""

    def __init__(
        self,
        *,
        allowed_hosts: set[str] | frozenset[str],
        request_get: Callable[..., object] | None = None,
        connect_timeout_seconds: float = 15.0,
        read_timeout_seconds: float = 120.0,
    ) -> None:
        self.allowed_hosts = {str(item).strip().lower() for item in allowed_hosts}
        if not self.allowed_hosts:
            raise LocalSemanticComponentError("component_download_host_not_allowed")
        if request_get is None:
            import requests

            request_get = requests.get
        self._request_get = request_get
        self._timeout = (
            max(1.0, float(connect_timeout_seconds)),
            max(1.0, float(read_timeout_seconds)),
        )

    def open(
        self,
        manifest: LocalSemanticComponentManifest,
        *,
        start: int,
    ) -> ComponentArchiveTransfer:
        headers = {"Accept-Encoding": "identity"}
        if int(start) > 0:
            headers["Range"] = f"bytes={int(start)}-"
        try:
            response = self._request_get(
                manifest.download_url,
                headers=headers,
                stream=True,
                timeout=self._timeout,
                allow_redirects=True,
            )
        except Exception as exc:
            raise LocalSemanticComponentError("component_download_transport_failed") from exc
        transfer = _RequestsTransfer(response)
        final = urlparse(transfer.final_url)
        if (
            final.scheme != "https"
            or str(final.hostname or "").lower() not in self.allowed_hosts
        ):
            transfer.close()
            raise LocalSemanticComponentError("component_download_redirect_not_allowed")
        return transfer


@dataclass(frozen=True)
class ComponentDownloadProgress:
    completed_bytes: int
    total_bytes: int
    resumed_bytes: int
    eta_seconds: int | None


@dataclass(frozen=True)
class ComponentDownloadResult:
    archive_path: Path
    downloaded_bytes: int
    resumed_bytes: int
    reused: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_range_response(
    transfer: ComponentArchiveTransfer,
    *,
    start: int,
    expected_size: int,
) -> bool:
    if start <= 0:
        return int(transfer.status_code) == 200
    if int(transfer.status_code) != 206:
        return False
    content_range = str(transfer.headers.get("Content-Range") or "").strip().lower()
    return content_range == f"bytes {start}-{expected_size - 1}/{expected_size}"


class LocalSemanticComponentDownloader:
    """Download one immutable component archive with validated range resume."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)

    def download(
        self,
        manifest: LocalSemanticComponentManifest,
        transport: ComponentArchiveTransport,
        *,
        should_cancel: Callable[[], bool] = lambda: False,
        on_progress: Callable[[ComponentDownloadProgress], None] = lambda _item: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> ComponentDownloadResult:
        archive_path = self.root / f"{manifest.archive_sha256}.zip"
        partial_path = archive_path.with_name(archive_path.name + ".part")
        started = clock()
        with db_write_lock(self.root):
            self.root.mkdir(parents=True, exist_ok=True)
            if self._is_ready(archive_path, manifest):
                return ComponentDownloadResult(
                    archive_path=archive_path,
                    downloaded_bytes=0,
                    resumed_bytes=0,
                    reused=True,
                )
            archive_path.unlink(missing_ok=True)
            partial_size = self._partial_size(partial_path, manifest)
            resumed_bytes = partial_size
            transfer = self._open_validated(
                transport,
                manifest,
                partial_path=partial_path,
                start=partial_size,
            )
            if partial_size and not _valid_range_response(
                transfer,
                start=partial_size,
                expected_size=manifest.archive_bytes,
            ):
                transfer.close()
                partial_path.unlink(missing_ok=True)
                partial_size = 0
                resumed_bytes = 0
                transfer = self._open_validated(
                    transport,
                    manifest,
                    partial_path=partial_path,
                    start=0,
                )
            if not _valid_range_response(
                transfer,
                start=partial_size,
                expected_size=manifest.archive_bytes,
            ):
                transfer.close()
                raise LocalSemanticComponentError("component_download_response_invalid")

            downloaded_bytes = 0
            mode = "ab" if partial_size else "wb"
            try:
                with partial_path.open(mode) as output:
                    completed = partial_size
                    for block in transfer.iter_bytes():
                        if should_cancel():
                            raise LocalSemanticComponentCancelled()
                        if not block:
                            continue
                        completed += len(block)
                        downloaded_bytes += len(block)
                        if completed > manifest.archive_bytes:
                            raise LocalSemanticComponentError(
                                "component_download_response_too_large"
                            )
                        output.write(block)
                        elapsed = max(0.0, clock() - started)
                        eta = None
                        if completed > 0 and elapsed > 0:
                            eta = max(
                                0,
                                round(
                                    (manifest.archive_bytes - completed)
                                    * elapsed
                                    / completed
                                ),
                            )
                        on_progress(
                            ComponentDownloadProgress(
                                completed_bytes=completed,
                                total_bytes=manifest.archive_bytes,
                                resumed_bytes=resumed_bytes,
                                eta_seconds=eta,
                            )
                        )
                    output.flush()
                    os.fsync(output.fileno())
            except LocalSemanticComponentError as exc:
                if exc.code == "component_download_response_too_large":
                    partial_path.unlink(missing_ok=True)
                raise
            except Exception as exc:
                raise LocalSemanticComponentError(
                    "component_download_transport_failed"
                ) from exc
            finally:
                transfer.close()
            if partial_path.stat().st_size != manifest.archive_bytes:
                raise LocalSemanticComponentError("component_download_incomplete")
            if _sha256_file(partial_path) != manifest.archive_sha256:
                partial_path.unlink(missing_ok=True)
                raise LocalSemanticComponentError("component_archive_hash_mismatch")
            os.replace(partial_path, archive_path)
            return ComponentDownloadResult(
                archive_path=archive_path,
                downloaded_bytes=downloaded_bytes,
                resumed_bytes=resumed_bytes,
                reused=False,
            )

    @staticmethod
    def _open_validated(
        transport: ComponentArchiveTransport,
        manifest: LocalSemanticComponentManifest,
        *,
        partial_path: Path,
        start: int,
    ) -> ComponentArchiveTransfer:
        if start < 0 or start > manifest.archive_bytes:
            partial_path.unlink(missing_ok=True)
            start = 0
        try:
            return transport.open(manifest, start=start)
        except LocalSemanticComponentError:
            raise
        except Exception as exc:
            raise LocalSemanticComponentError("component_download_transport_failed") from exc

    @staticmethod
    def _partial_size(
        partial_path: Path,
        manifest: LocalSemanticComponentManifest,
    ) -> int:
        try:
            size = int(partial_path.stat().st_size)
        except OSError:
            return 0
        if size < 0 or size >= manifest.archive_bytes:
            partial_path.unlink(missing_ok=True)
            return 0
        return size

    @staticmethod
    def _is_ready(
        archive_path: Path,
        manifest: LocalSemanticComponentManifest,
    ) -> bool:
        try:
            return (
                archive_path.stat().st_size == manifest.archive_bytes
                and _sha256_file(archive_path) == manifest.archive_sha256
            )
        except OSError:
            return False
