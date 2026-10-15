from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .store import atomic_write_json, db_write_lock


CANDIDATE_ID = "multilingual_e5_small_official_onnx_v1"
MODEL_REVISION = "03415a4be176a1620747c692ed433219fabc3def"
COMPONENT_ARCHIVE_BYTES = 296_365_646
COMPONENT_ARCHIVE_SHA256 = (
    "54d0f4fbd1c07913f5915dafbab224cf5510072fd767fcffd62da0e0b8ac3ad3"
)
COMPONENT_PAYLOAD_BYTES = 541_333_859
COMPONENT_PAYLOAD_FILE_COUNT = 680
COMPONENT_PAYLOAD_SHA256 = (
    "568ecdc099cacb4c99943f5d333a8661f4194975ebebafa0caedae0379fc54ef"
)
COMPONENT_SAFETY_MARGIN_BYTES = 128 * 1024 * 1024
FIRST_INSTALL_REQUIRED_FREE_BYTES = (
    COMPONENT_ARCHIVE_BYTES
    + COMPONENT_PAYLOAD_BYTES
    + COMPONENT_SAFETY_MARGIN_BYTES
)
WINDOWS_SAFE_COMPONENT_PATH_CHARS = 259
MAX_COMPONENT_RELATIVE_PATH_CHARS = 146
_GENERATION_RE = re.compile(r"^(?:[0-9a-f]{16}|[0-9a-f]{32})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_REPARSE_POINT = 0x400


class LocalSemanticComponentError(RuntimeError):
    def __init__(self, code: str, *, required_additional_bytes: int = 0) -> None:
        super().__init__(str(code or "semantic_component_failed"))
        self.code = str(code or "semantic_component_failed")
        self.required_additional_bytes = max(0, int(required_additional_bytes))


class LocalSemanticComponentCancelled(LocalSemanticComponentError):
    def __init__(self) -> None:
        super().__init__("semantic_component_cancelled")


@dataclass(frozen=True)
class LocalSemanticComponentManifest:
    component_id: str
    release_id: str
    download_url: str
    archive_name: str
    archive_root: str
    archive_bytes: int
    archive_sha256: str
    payload_bytes: int
    payload_file_count: int
    payload_sha256: str
    candidate_id: str = CANDIDATE_ID
    model_revision: str = MODEL_REVISION

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        *,
        allowed_hosts: set[str] | frozenset[str],
    ) -> LocalSemanticComponentManifest:
        try:
            manifest = cls(
                component_id=str(payload.get("component_id") or ""),
                release_id=str(payload.get("release_id") or ""),
                download_url=str(payload.get("download_url") or ""),
                archive_name=str(payload.get("archive_name") or ""),
                archive_root=str(payload.get("archive_root") or ""),
                archive_bytes=int(payload.get("archive_bytes") or 0),
                archive_sha256=str(payload.get("archive_sha256") or "").lower(),
                payload_bytes=int(payload.get("payload_bytes") or 0),
                payload_file_count=int(payload.get("payload_file_count") or 0),
                payload_sha256=str(payload.get("payload_sha256") or "").lower(),
                candidate_id=str(payload.get("candidate_id") or ""),
                model_revision=str(payload.get("model_revision") or ""),
            )
        except (TypeError, ValueError) as exc:
            raise LocalSemanticComponentError("component_manifest_invalid") from exc
        manifest.validate(allowed_hosts=allowed_hosts)
        return manifest

    def validate(self, *, allowed_hosts: set[str] | frozenset[str]) -> None:
        parsed = urlparse(self.download_url)
        host = str(parsed.hostname or "").lower()
        path_parts = tuple(unquote(part) for part in PurePosixPath(parsed.path).parts)
        if (
            not self.component_id
            or not self.release_id
            or parsed.scheme != "https"
            or not host
            or host not in {str(item).lower() for item in allowed_hosts}
            or parsed.query
            or parsed.fragment
            or self.release_id not in path_parts
            or Path(unquote(parsed.path)).name != self.archive_name
            or Path(self.archive_name).name != self.archive_name
            or "/" in self.archive_name
            or "\\" in self.archive_name
            or not self.archive_name.lower().endswith(".zip")
            or not self.archive_root
            or "/" in self.archive_root
            or "\\" in self.archive_root
            or self.archive_root in {".", ".."}
            or self.archive_bytes <= 0
            or self.payload_bytes <= 0
            or self.payload_file_count <= 0
            or not _SHA256_RE.fullmatch(self.archive_sha256)
            or not _SHA256_RE.fullmatch(self.payload_sha256)
            or self.candidate_id != CANDIDATE_ID
            or self.model_revision != MODEL_REVISION
        ):
            raise LocalSemanticComponentError("component_manifest_invalid")


@dataclass(frozen=True)
class LocalSemanticCapacity:
    required_bytes: int
    available_bytes: int

    @property
    def sufficient(self) -> bool:
        return self.available_bytes >= self.required_bytes

    @property
    def required_additional_bytes(self) -> int:
        return max(0, self.required_bytes - self.available_bytes)


@dataclass(frozen=True)
class InstalledLocalSemanticComponent:
    generation_id: str
    root: Path
    component_id: str
    release_id: str
    archive_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class LocalSemanticComponentRemovalResult:
    removed_generation_count: int
    cleanup_pending: bool


def local_semantic_component_root(db_dir: Path | str) -> Path:
    root = Path(db_dir).expanduser().resolve(strict=False)
    preferred = (root.parent / "_c" / "e5").resolve(strict=False)
    if os.name != "nt" or _default_layout_fits_windows(preferred):
        return preferred
    local_app_data = str(os.environ.get("LOCALAPPDATA") or "").strip()
    if local_app_data:
        fallback = (Path(local_app_data) / "Pi_zaya" / "c" / "e5").resolve(
            strict=False
        )
        if _default_layout_fits_windows(fallback):
            return fallback
    return preferred


def _default_layout_fits_windows(root: Path) -> bool:
    staging = root / "generations" / (".s-" + "0" * 16)
    return (
        len(str(staging)) + 1 + MAX_COMPONENT_RELATIVE_PATH_CHARS
        <= WINDOWS_SAFE_COMPONENT_PATH_CHARS
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_existing_parent(path: Path) -> Path:
    current = path.resolve(strict=False)
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def capacity_for_download(
    root: Path | str,
    manifest: LocalSemanticComponentManifest,
    *,
    current_generation_bytes: int = 0,
    free_bytes_probe: Callable[[Path], int] | None = None,
) -> LocalSemanticCapacity:
    destination = _nearest_existing_parent(Path(root))
    available = (
        int(free_bytes_probe(destination))
        if free_bytes_probe is not None
        else int(shutil.disk_usage(destination).free)
    )
    required = (
        manifest.archive_bytes
        + manifest.payload_bytes
        + COMPONENT_SAFETY_MARGIN_BYTES
        + max(0, int(current_generation_bytes))
    )
    return LocalSemanticCapacity(required_bytes=required, available_bytes=available)


def require_capacity(capacity: LocalSemanticCapacity) -> None:
    if not capacity.sufficient:
        raise LocalSemanticComponentError(
            "semantic_component_needs_space",
            required_additional_bytes=capacity.required_additional_bytes,
        )


def _safe_relative_path(info: zipfile.ZipInfo, *, archive_root: str) -> str | None:
    raw = str(info.filename or "")
    if not raw or "\\" in raw or raw.startswith("/"):
        raise LocalSemanticComponentError("component_archive_entry_unsafe")
    path = PurePosixPath(raw)
    parts = path.parts
    if not parts or parts[0] != archive_root or any(part in {"", ".", ".."} for part in parts):
        raise LocalSemanticComponentError("component_archive_entry_unsafe")
    mode = int(info.external_attr) >> 16
    file_type = stat.S_IFMT(mode)
    if (
        stat.S_ISLNK(mode)
        or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}
        or int(info.external_attr) & _WINDOWS_REPARSE_POINT
        or int(info.flag_bits) & 0x1
    ):
        raise LocalSemanticComponentError("component_archive_entry_unsafe")
    if info.is_dir():
        return None
    relative_parts = parts[1:]
    if not relative_parts:
        raise LocalSemanticComponentError("component_archive_entry_unsafe")
    return PurePosixPath(*relative_parts).as_posix()


def _require_windows_path_budget(
    archive_path: Path,
    *,
    archive_root: str,
    staging: Path,
    generation: Path,
) -> int:
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise LocalSemanticComponentError("component_archive_invalid") from exc
    maximum = 0
    with archive:
        for info in archive.infolist():
            relative = _safe_relative_path(info, archive_root=archive_root)
            if relative is None:
                continue
            parts = PurePosixPath(relative).parts
            maximum = max(
                maximum,
                len(str(staging.joinpath(*parts))),
                len(str(generation.joinpath(*parts))),
            )
    if os.name == "nt" and maximum > WINDOWS_SAFE_COMPONENT_PATH_CHARS:
        raise LocalSemanticComponentError("component_storage_path_too_long")
    return maximum


def _remove_owned_tree(root: Path, target: Path) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise LocalSemanticComponentError("component_owned_path_invalid")
    if resolved_target.exists():
        shutil.rmtree(resolved_target)


class LocalSemanticComponentStore:
    """Verify and atomically publish a downloaded portable semantic component."""

    def __init__(
        self,
        root: Path | str,
        *,
        free_bytes_probe: Callable[[Path], int] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.generations_dir = self.root / "generations"
        self.current_path = self.root / "current.json"
        self.free_bytes_probe = free_bytes_probe

    def preflight_download(
        self,
        manifest: LocalSemanticComponentManifest,
        *,
        current_generation_bytes: int = 0,
    ) -> LocalSemanticCapacity:
        result = capacity_for_download(
            self.root,
            manifest,
            current_generation_bytes=current_generation_bytes,
            free_bytes_probe=self.free_bytes_probe,
        )
        require_capacity(result)
        return result

    def _preflight_extraction(
        self,
        manifest: LocalSemanticComponentManifest,
    ) -> LocalSemanticCapacity:
        destination = _nearest_existing_parent(self.root)
        available = (
            int(self.free_bytes_probe(destination))
            if self.free_bytes_probe is not None
            else int(shutil.disk_usage(destination).free)
        )
        result = LocalSemanticCapacity(
            required_bytes=manifest.payload_bytes + COMPONENT_SAFETY_MARGIN_BYTES,
            available_bytes=available,
        )
        require_capacity(result)
        return result

    def install_from_archive(
        self,
        archive_path: Path | str,
        manifest: LocalSemanticComponentManifest,
        *,
        should_cancel: Callable[[], bool] = lambda: False,
        health_check: Callable[[Path], bool] = lambda _root: True,
    ) -> InstalledLocalSemanticComponent:
        archive = Path(archive_path).expanduser().resolve(strict=False)
        try:
            if archive.stat().st_size != manifest.archive_bytes:
                raise LocalSemanticComponentError("component_archive_size_mismatch")
        except OSError as exc:
            raise LocalSemanticComponentError("component_archive_unavailable") from exc
        if _sha256_file(archive) != manifest.archive_sha256:
            raise LocalSemanticComponentError("component_archive_hash_mismatch")
        self._preflight_extraction(manifest)
        if should_cancel():
            raise LocalSemanticComponentCancelled()

        generation_id = uuid.uuid4().hex[:16]
        staging = self.generations_dir / f".s-{generation_id}"
        generation = self.generations_dir / generation_id
        _require_windows_path_budget(
            archive,
            archive_root=manifest.archive_root,
            staging=staging,
            generation=generation,
        )
        with db_write_lock(self.root):
            self.generations_dir.mkdir(parents=True, exist_ok=True)
            if staging.exists() or generation.exists():
                raise LocalSemanticComponentError("component_generation_collision")
            staging.mkdir()
            try:
                records = self._extract_verified(
                    archive,
                    staging,
                    manifest,
                    should_cancel=should_cancel,
                )
                atomic_write_json(
                    staging / "install.json",
                    {
                        "schema_version": 1,
                        "generation_id": generation_id,
                        "component_id": manifest.component_id,
                        "release_id": manifest.release_id,
                        "candidate_id": manifest.candidate_id,
                        "model_revision": manifest.model_revision,
                        "archive_sha256": manifest.archive_sha256,
                        "payload_sha256": manifest.payload_sha256,
                        "payload_file_count": records["file_count"],
                        "payload_bytes": records["bytes"],
                    },
                )
                os.replace(staging, generation)
                if should_cancel():
                    raise LocalSemanticComponentCancelled()
                try:
                    healthy = bool(health_check(generation))
                except Exception as exc:
                    raise LocalSemanticComponentError(
                        "component_install_health_failed"
                    ) from exc
                if not healthy:
                    raise LocalSemanticComponentError("component_install_health_failed")
                atomic_write_json(
                    self.current_path,
                    {
                        "schema_version": 1,
                        "generation_id": generation_id,
                        "component_id": manifest.component_id,
                        "release_id": manifest.release_id,
                        "archive_sha256": manifest.archive_sha256,
                        "payload_sha256": manifest.payload_sha256,
                    },
                )
            except Exception:
                if staging.exists():
                    _remove_owned_tree(self.generations_dir, staging)
                if generation.exists():
                    _remove_owned_tree(self.generations_dir, generation)
                raise
        return self.load_current()

    def _extract_verified(
        self,
        archive_path: Path,
        staging: Path,
        manifest: LocalSemanticComponentManifest,
        *,
        should_cancel: Callable[[], bool],
    ) -> dict[str, int | str]:
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise LocalSemanticComponentError("component_archive_invalid") from exc
        with archive:
            entries: list[tuple[str, zipfile.ZipInfo]] = []
            observed_paths: set[str] = set()
            for info in archive.infolist():
                relative = _safe_relative_path(info, archive_root=manifest.archive_root)
                if relative is None:
                    continue
                folded = relative.casefold()
                if folded in observed_paths:
                    raise LocalSemanticComponentError(
                        "component_archive_duplicate_path"
                    )
                observed_paths.add(folded)
                entries.append((relative, info))
            entries.sort(key=lambda item: item[0])
            if len(entries) != manifest.payload_file_count:
                raise LocalSemanticComponentError("component_payload_file_count_mismatch")
            if sum(int(info.file_size) for _relative, info in entries) != manifest.payload_bytes:
                raise LocalSemanticComponentError("component_payload_size_mismatch")

            payload_digest = hashlib.sha256()
            total_bytes = 0
            for relative, info in entries:
                if should_cancel():
                    raise LocalSemanticComponentCancelled()
                destination = staging.joinpath(*PurePosixPath(relative).parts)
                resolved = destination.resolve(strict=False)
                if staging.resolve(strict=False) not in resolved.parents:
                    raise LocalSemanticComponentError("component_archive_entry_unsafe")
                destination.parent.mkdir(parents=True, exist_ok=True)
                file_digest = hashlib.sha256()
                written = 0
                try:
                    source = archive.open(info)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise LocalSemanticComponentError("component_archive_invalid") from exc
                try:
                    with source, destination.open("xb") as output:
                        while True:
                            if should_cancel():
                                raise LocalSemanticComponentCancelled()
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            written += len(block)
                            total_bytes += len(block)
                            if written > int(info.file_size) or total_bytes > manifest.payload_bytes:
                                raise LocalSemanticComponentError(
                                    "component_payload_size_mismatch"
                                )
                            output.write(block)
                            file_digest.update(block)
                        output.flush()
                        os.fsync(output.fileno())
                except LocalSemanticComponentError:
                    raise
                except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise LocalSemanticComponentError("component_archive_invalid") from exc
                if written != int(info.file_size):
                    raise LocalSemanticComponentError("component_payload_size_mismatch")
                payload_digest.update(
                    f"{relative}\0{written}\0{file_digest.hexdigest()}\n".encode(
                        "utf-8", "surrogatepass"
                    )
                )
            if total_bytes != manifest.payload_bytes:
                raise LocalSemanticComponentError("component_payload_size_mismatch")
            if payload_digest.hexdigest() != manifest.payload_sha256:
                raise LocalSemanticComponentError("component_payload_hash_mismatch")
            return {
                "sha256": payload_digest.hexdigest(),
                "file_count": len(entries),
                "bytes": total_bytes,
            }

    def load_current(self) -> InstalledLocalSemanticComponent:
        try:
            payload = json.loads(self.current_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LocalSemanticComponentError("component_not_installed") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalSemanticComponentError("component_pointer_corrupt") from exc
        generation_id = str(payload.get("generation_id") or "")
        if not _GENERATION_RE.fullmatch(generation_id):
            raise LocalSemanticComponentError("component_pointer_corrupt")
        generation = (self.generations_dir / generation_id).resolve(strict=False)
        if self.generations_dir.resolve(strict=False) not in generation.parents:
            raise LocalSemanticComponentError("component_pointer_corrupt")
        try:
            install = json.loads((generation / "install.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalSemanticComponentError("component_install_corrupt") from exc
        expected = {
            "generation_id": generation_id,
            "component_id": str(payload.get("component_id") or ""),
            "release_id": str(payload.get("release_id") or ""),
            "archive_sha256": str(payload.get("archive_sha256") or ""),
            "payload_sha256": str(payload.get("payload_sha256") or ""),
        }
        if any(str(install.get(key) or "") != value for key, value in expected.items()):
            raise LocalSemanticComponentError("component_install_corrupt")
        return InstalledLocalSemanticComponent(root=generation, **expected)

    def remove_all(self) -> LocalSemanticComponentRemovalResult:
        """Disable the component first, then delete only owned component data."""

        removed = 0
        pending: list[str] = []
        tombstone = self.root / f".disabled-{uuid.uuid4().hex}.json"
        with db_write_lock(self.root):
            self.root.mkdir(parents=True, exist_ok=True)
            if self.current_path.exists():
                os.replace(self.current_path, tombstone)
            if self.generations_dir.is_dir():
                for child in tuple(self.generations_dir.iterdir()):
                    if not (
                        _GENERATION_RE.fullmatch(child.name)
                        or child.name.startswith((".staging-", ".s-"))
                    ):
                        continue
                    try:
                        _remove_owned_tree(self.generations_dir, child)
                        removed += 1
                    except OSError:
                        pending.append(f"generations/{child.name}")
            downloads = self.root / "downloads"
            if downloads.is_dir():
                try:
                    _remove_owned_tree(self.root, downloads)
                except OSError:
                    pending.append("downloads")
            tombstone.unlink(missing_ok=True)
            cleanup_path = self.root / "pending_cleanup.json"
            if pending:
                atomic_write_json(
                    cleanup_path,
                    {"schema_version": 1, "owned_relative_paths": sorted(pending)},
                )
            else:
                cleanup_path.unlink(missing_ok=True)
        return LocalSemanticComponentRemovalResult(
            removed_generation_count=removed,
            cleanup_pending=bool(pending),
        )

    def retry_pending_cleanup(self) -> bool:
        cleanup_path = self.root / "pending_cleanup.json"
        try:
            payload = json.loads(cleanup_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return True
        except (OSError, json.JSONDecodeError):
            return False
        pending: list[str] = []
        with db_write_lock(self.root):
            for raw in payload.get("owned_relative_paths") or []:
                relative = str(raw or "")
                if relative == "downloads":
                    target = self.root / "downloads"
                    owner = self.root
                elif relative.startswith("generations/"):
                    name = relative.removeprefix("generations/")
                    if not (
                        _GENERATION_RE.fullmatch(name)
                        or name.startswith((".staging-", ".s-"))
                    ):
                        return False
                    target = self.generations_dir / name
                    owner = self.generations_dir
                else:
                    return False
                try:
                    _remove_owned_tree(owner, target)
                except OSError:
                    pending.append(relative)
            if pending:
                atomic_write_json(
                    cleanup_path,
                    {"schema_version": 1, "owned_relative_paths": sorted(pending)},
                )
            else:
                cleanup_path.unlink(missing_ok=True)
        return not pending
