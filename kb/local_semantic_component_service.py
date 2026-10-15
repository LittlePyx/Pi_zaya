from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Protocol

from .local_semantic_component import (
    COMPONENT_ARCHIVE_BYTES,
    COMPONENT_PAYLOAD_BYTES,
    LocalSemanticComponentManifest,
    LocalSemanticComponentStore,
    local_semantic_component_root,
)
from .local_semantic_component_download import RequestsComponentArchiveTransport
from .bundled_component_transport import BundledComponentArchiveTransport
from .local_semantic_component_lifecycle import (
    LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED,
    LocalSemanticComponentLifecycleError,
    LocalSemanticComponentLifecycleService,
)
from .local_semantic_component_lifecycle_contract import (
    LocalSemanticComponentLifecycleRecord,
)
from .local_semantic_component_lifecycle_store import (
    LocalSemanticComponentLifecycleStore,
)
from .portable_local_semantic_runtime import (
    PortableLocalSemanticRuntime,
    PortableLocalSemanticRuntimeAdapter,
    PortableLocalSemanticRuntimeError,
)


LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV = "KB_LOCAL_SEMANTIC_COMPONENT_MANIFEST_JSON"
LOCAL_SEMANTIC_COMPONENT_BUNDLED_MANIFEST_PATH = Path(__file__).with_name(
    "local_semantic_component_manifest.json"
)
LOCAL_SEMANTIC_COMPONENT_BUNDLED_ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "components"
LOCAL_SEMANTIC_COMPONENT_CONSENT_SCOPE = (
    "install_local_semantic_retrieval_component"
)
LOCAL_SEMANTIC_COMPONENT_ALLOWED_HOSTS = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)


class LocalSemanticComponentService(Protocol):
    def status(self) -> LocalSemanticComponentLifecycleRecord:
        ...

    def enable(
        self,
        *,
        consent: bool,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        ...

    def resume(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        ...

    def cancel(self, task_id: str) -> LocalSemanticComponentLifecycleRecord:
        ...

    def remove(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        ...


def local_semantic_component_disclosure() -> dict[str, object]:
    manifest, _identity = _configured_manifest()
    return {
        "consent_scope": LOCAL_SEMANTIC_COMPONENT_CONSENT_SCOPE,
        "archive_bytes": (
            manifest.archive_bytes if manifest is not None else COMPONENT_ARCHIVE_BYTES
        ),
        "installed_bytes": (
            manifest.payload_bytes if manifest is not None else COMPONENT_PAYLOAD_BYTES
        ),
        "network_required_for_install": not _has_bundled_archive(manifest),
        "library_content_uploaded": False,
        "external_content_or_model_knowledge_added": False,
        "bm25_remains_available": True,
    }


def _has_bundled_archive(manifest: LocalSemanticComponentManifest | None) -> bool:
    return manifest is not None and (LOCAL_SEMANTIC_COMPONENT_BUNDLED_ARCHIVE_DIR / manifest.archive_name).is_file()


def _archive_transport(manifest: LocalSemanticComponentManifest):
    if _has_bundled_archive(manifest):
        return BundledComponentArchiveTransport(LOCAL_SEMANTIC_COMPONENT_BUNDLED_ARCHIVE_DIR)
    return RequestsComponentArchiveTransport(allowed_hosts=LOCAL_SEMANTIC_COMPONENT_ALLOWED_HOSTS)


def _configured_manifest() -> tuple[LocalSemanticComponentManifest | None, str]:
    raw = str(os.environ.get(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV) or "").strip()
    if not raw:
        try:
            raw = LOCAL_SEMANTIC_COMPONENT_BUNDLED_MANIFEST_PATH.read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            return None, ""
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None, ""
        manifest = LocalSemanticComponentManifest.from_payload(
            payload,
            allowed_hosts=LOCAL_SEMANTIC_COMPONENT_ALLOWED_HOSTS,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, ""
    except Exception:
        return None, ""
    identity = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return manifest, identity


class _UnavailableLocalSemanticComponentService:
    def __init__(self, db_dir: Path | str) -> None:
        self.db_dir = Path(db_dir).expanduser().resolve(strict=False)

    def status(self) -> LocalSemanticComponentLifecycleRecord:
        ledger = self.db_dir / "local_semantic_component_lifecycle.sqlite3"
        if ledger.is_file():
            current = LocalSemanticComponentLifecycleStore(self.db_dir).current()
            if current.state == "removing":
                return current
        return LocalSemanticComponentLifecycleRecord(
            state="unavailable_in_build",
            reason_id="component_manifest_unavailable",
        )

    def enable(
        self,
        *,
        consent: bool,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        del consent
        raise LocalSemanticComponentLifecycleError(
            "semantic_component_requires_app_update"
        )

    def resume(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        raise LocalSemanticComponentLifecycleError(
            "semantic_component_requires_app_update"
        )

    def cancel(self, task_id: str) -> LocalSemanticComponentLifecycleRecord:
        del task_id
        raise LocalSemanticComponentLifecycleError(
            "semantic_component_task_not_cancellable"
        )

    def remove(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        component_store = LocalSemanticComponentStore(
            local_semantic_component_root(self.db_dir)
        )
        runtime = PortableLocalSemanticRuntimeAdapter(
            self.db_dir,
            component_store=component_store,
        )
        component_store.remove_all()
        runtime.remove_index()
        lifecycle = LocalSemanticComponentLifecycleStore(self.db_dir)
        current = lifecycle.current()
        created = False
        if current.state in {
            "checking_capacity",
            "downloading",
            "verifying_archive",
            "installing",
            "health_checking",
            "ready_to_index",
            "indexing",
            "cancelling",
            "recovering",
            "removing",
        }:
            task_id = current.task_id
        else:
            current, created = lifecycle.start_or_get(
                operation="remove",
                initial_state="removing",
                manifest_identity=current.manifest_identity,
                owner_session="manifest-unavailable-removal",
                consented=False,
            )
            task_id = current.task_id
        lifecycle.transition(
            task_id,
            state="not_installed",
            reason_id="component_removed",
            completed_bytes=0,
            total_bytes=0,
            eta_seconds=None,
            required_additional_bytes=0,
            consented=False,
            finished=True,
        )
        return lifecycle.current(), created


_SERVICES_LOCK = threading.Lock()
_SERVICES: dict[tuple[str, str], LocalSemanticComponentLifecycleService] = {}
_RUNTIMES: dict[tuple[str, str], PortableLocalSemanticRuntimeAdapter] = {}


def _cached_service(root: Path) -> LocalSemanticComponentLifecycleService | None:
    root_key = str(root).casefold()
    with _SERVICES_LOCK:
        return next(
            (
                service
                for (service_root, _identity), service in _SERVICES.items()
                if service_root == root_key
            ),
            None,
        )


def _build_service(
    root: Path,
    manifest: LocalSemanticComponentManifest,
    manifest_config_identity: str,
) -> LocalSemanticComponentLifecycleService:
    key = (str(root).casefold(), manifest_config_identity)
    with _SERVICES_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            component_store = LocalSemanticComponentStore(
                local_semantic_component_root(root)
            )
            runtime = PortableLocalSemanticRuntimeAdapter(
                root,
                component_store=component_store,
            )
            service = LocalSemanticComponentLifecycleService(
                root,
                manifest,
                transport_factory=lambda: _archive_transport(manifest),
                component_store=component_store,
                health_check=runtime.health_check,
                index_ready=runtime.index_ready,
                prepare_index=runtime.prepare_index,
                remove_index=runtime.remove_index,
                release_runtime_handles=runtime.release_runtime_handles,
            )
            _SERVICES[key] = service
            _RUNTIMES[key] = runtime
        return service


def load_local_semantic_component_runtime(
    db_dir: Path | str,
) -> PortableLocalSemanticRuntime:
    """Load only the runtime owned by the current ready lifecycle generation."""

    root = Path(db_dir).expanduser().resolve(strict=False)
    manifest, manifest_config_identity = _configured_manifest()
    if manifest is None:
        raise PortableLocalSemanticRuntimeError("component_manifest_unavailable")
    service = _build_service(root, manifest, manifest_config_identity)
    if service.status().state != "ready":
        raise PortableLocalSemanticRuntimeError("component_lifecycle_not_ready")
    key = (str(root).casefold(), manifest_config_identity)
    with _SERVICES_LOCK:
        runtime = _RUNTIMES.get(key)
    if runtime is None:
        raise PortableLocalSemanticRuntimeError("component_runtime_unavailable")
    return runtime.load_ready(root)


class _LazyLocalSemanticComponentService:
    """Keep first status inspection read-only until explicit install consent."""

    def __init__(
        self,
        db_dir: Path,
        manifest: LocalSemanticComponentManifest,
        manifest_config_identity: str,
    ) -> None:
        self.db_dir = db_dir
        self.manifest = manifest
        self.manifest_config_identity = manifest_config_identity

    def _service(self) -> LocalSemanticComponentLifecycleService:
        return _build_service(
            self.db_dir,
            self.manifest,
            self.manifest_config_identity,
        )

    def _has_prior_state(self) -> bool:
        return any(
            path.is_file()
            for path in (
                self.db_dir / "local_semantic_component_lifecycle.sqlite3",
                local_semantic_component_root(self.db_dir) / "current.json",
                self.db_dir
                / "local_semantic_index"
                / "e5_small_v1"
                / "current.json",
            )
        )

    def status(self) -> LocalSemanticComponentLifecycleRecord:
        existing = _cached_service(self.db_dir)
        if existing is not None:
            return existing.status()
        if self._has_prior_state():
            return self._service().status()
        return LocalSemanticComponentLifecycleRecord(
            state="not_installed",
            reason_id="component_not_installed",
        )

    def enable(
        self,
        *,
        consent: bool,
    ) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        if not consent:
            raise LocalSemanticComponentLifecycleError(
                "semantic_component_consent_required"
            )
        return self._service().enable(consent=True)

    def resume(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        return self._service().resume()

    def cancel(self, task_id: str) -> LocalSemanticComponentLifecycleRecord:
        return self._service().cancel(task_id)

    def remove(self) -> tuple[LocalSemanticComponentLifecycleRecord, bool]:
        return self._service().remove()


def get_local_semantic_component_service(
    db_dir: Path | str,
) -> LocalSemanticComponentService:
    root = Path(db_dir).expanduser().resolve(strict=False)
    manifest, manifest_config_identity = _configured_manifest()
    if manifest is None:
        existing = _cached_service(root)
        if existing is not None:
            return existing
        return _UnavailableLocalSemanticComponentService(root)
    existing = _cached_service(root)
    if existing is not None and existing.manifest_identity == (
        f"{manifest.release_id}:{manifest.archive_sha256}"
    ):
        return existing
    return _LazyLocalSemanticComponentService(
        root,
        manifest,
        manifest_config_identity,
    )


def local_semantic_component_management_available() -> bool:
    manifest, _identity = _configured_manifest()
    return manifest is not None


__all__ = [
    "LOCAL_SEMANTIC_COMPONENT_CONSENT_SCOPE",
    "LOCAL_SEMANTIC_COMPONENT_BUNDLED_MANIFEST_PATH",
    "LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV",
    "LOCAL_SEMANTIC_COMPONENT_PRODUCT_ENABLED",
    "LocalSemanticComponentService",
    "get_local_semantic_component_service",
    "load_local_semantic_component_runtime",
    "local_semantic_component_disclosure",
    "local_semantic_component_management_available",
]
