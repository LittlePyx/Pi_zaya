from __future__ import annotations

from pathlib import Path

from .local_semantic_component import LocalSemanticComponentError, LocalSemanticComponentManifest


class _BundledTransfer:
    def __init__(self, archive: Path, manifest: LocalSemanticComponentManifest, start: int) -> None:
        self._handle = archive.open("rb")
        self._handle.seek(start)
        self.status_code = 206 if start else 200
        self.headers = {"Content-Length": str(manifest.archive_bytes - start)}
        if start:
            self.headers["Content-Range"] = f"bytes {start}-{manifest.archive_bytes - 1}/{manifest.archive_bytes}"
        self.final_url = "bundled:" + manifest.archive_name

    def iter_bytes(self):
        while block := self._handle.read(1024 * 1024):
            yield block

    def close(self) -> None:
        self._handle.close()


class BundledComponentArchiveTransport:
    """Copy a shipped archive through the existing cancellable hash/size gate."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def open(self, manifest: LocalSemanticComponentManifest, *, start: int):
        archive = self.directory / manifest.archive_name
        try:
            unsafe = any(path.is_symlink() or int(getattr(path.stat(), "st_file_attributes", 0)) & 0x400 for path in (self.directory, archive))
            if unsafe or archive.resolve().parent != self.directory.resolve() or archive.stat().st_size != manifest.archive_bytes:
                raise LocalSemanticComponentError("component_bundled_archive_invalid")
            return _BundledTransfer(archive, manifest, start)
        except OSError as exc:
            raise LocalSemanticComponentError("component_bundled_archive_unavailable") from exc
