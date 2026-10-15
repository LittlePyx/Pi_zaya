from __future__ import annotations

import hashlib
import json

import pytest

import kb.local_semantic_component_service as component_service
from kb.local_semantic_component import CANDIDATE_ID, MODEL_REVISION
from kb.local_semantic_component_lifecycle import (
    LocalSemanticComponentLifecycleError,
)
from kb.local_semantic_component_lifecycle_contract import (
    LocalSemanticComponentLifecycleRecord,
)
from kb.local_semantic_component_service import (
    LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV,
    get_local_semantic_component_service,
    load_local_semantic_component_runtime,
    local_semantic_component_disclosure,
    local_semantic_component_management_available,
)


def _manifest_json() -> str:
    return json.dumps(
        {
            "component_id": "component",
            "release_id": "component-v1",
            "download_url": (
                "https://github.com/release/component-v1/component.zip"
            ),
            "archive_name": "component.zip",
            "archive_root": "c",
            "archive_bytes": 8,
            "archive_sha256": hashlib.sha256(b"archive").hexdigest(),
            "payload_bytes": 10,
            "payload_file_count": 2,
            "payload_sha256": hashlib.sha256(b"payload").hexdigest(),
            "candidate_id": CANDIDATE_ID,
            "model_revision": MODEL_REVISION,
        }
    )


def test_configured_status_and_declined_consent_are_read_only(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, _manifest_json())
    db_dir = tmp_path / "db"
    service = get_local_semantic_component_service(db_dir)

    assert service.status().state == "not_installed"
    with pytest.raises(LocalSemanticComponentLifecycleError) as captured:
        service.enable(consent=False)

    assert captured.value.code == "semantic_component_consent_required"
    assert not db_dir.exists()
    assert local_semantic_component_management_available() is True


def test_invalid_manifest_is_unavailable_without_creating_state(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, "{not-json")
    db_dir = tmp_path / "db"

    status = get_local_semantic_component_service(db_dir).status()

    assert status.state == "unavailable_in_build"
    assert status.public_payload()["action_id"] == "update_app"
    assert local_semantic_component_management_available() is False
    assert not db_dir.exists()


def test_bundled_manifest_enables_lazy_management_without_disk_write(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, raising=False)
    bundled = tmp_path / "local_semantic_component_manifest.json"
    bundled.write_text(_manifest_json(), encoding="utf-8")
    monkeypatch.setattr(
        component_service,
        "LOCAL_SEMANTIC_COMPONENT_BUNDLED_MANIFEST_PATH",
        bundled,
    )
    db_dir = tmp_path / "db"

    status = get_local_semantic_component_service(db_dir).status()
    disclosure = local_semantic_component_disclosure()

    assert status.state == "not_installed"
    assert local_semantic_component_management_available() is True
    assert disclosure["archive_bytes"] == 8
    assert disclosure["installed_bytes"] == 10
    assert not db_dir.exists()


def test_disclosure_keeps_external_knowledge_consent_separate() -> None:
    disclosure = local_semantic_component_disclosure()

    assert disclosure["network_required_for_install"] is True
    assert disclosure["library_content_uploaded"] is False
    assert disclosure["external_content_or_model_knowledge_added"] is False
    assert disclosure["bm25_remains_available"] is True


def test_product_runtime_requires_ready_lifecycle(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, _manifest_json())
    monkeypatch.setattr(
        component_service,
        "_build_service",
        lambda *_args: type(
            "Service",
            (),
            {"status": lambda _self: LocalSemanticComponentLifecycleRecord(
                state="indexing"
            )},
        )(),
    )

    with pytest.raises(component_service.PortableLocalSemanticRuntimeError) as captured:
        load_local_semantic_component_runtime(tmp_path / "db")

    assert captured.value.code == "component_lifecycle_not_ready"


def test_product_runtime_uses_lifecycle_owned_adapter(
    monkeypatch,
    tmp_path,
) -> None:
    raw_manifest = _manifest_json()
    monkeypatch.setenv(LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, raw_manifest)
    db_dir = (tmp_path / "db").resolve()
    identity = hashlib.sha256(raw_manifest.encode("utf-8")).hexdigest()
    result = object()

    class Runtime:
        def load_ready(self, requested_db_dir):
            assert requested_db_dir == db_dir
            return result

    monkeypatch.setattr(component_service, "_RUNTIMES", {
        (str(db_dir).casefold(), identity): Runtime(),
    })
    monkeypatch.setattr(
        component_service,
        "_build_service",
        lambda *_args: type(
            "Service",
            (),
            {"status": lambda _self: LocalSemanticComponentLifecycleRecord(
                state="ready"
            )},
        )(),
    )

    assert load_local_semantic_component_runtime(db_dir) is result
