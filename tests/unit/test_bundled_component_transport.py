from __future__ import annotations

import hashlib
import json

import pytest

import kb.local_semantic_component_service as service
from kb.bundled_component_transport import BundledComponentArchiveTransport
from kb.local_semantic_component import CANDIDATE_ID, MODEL_REVISION, LocalSemanticComponentError, LocalSemanticComponentManifest
from kb.local_semantic_component_download import LocalSemanticComponentDownloader
from kb.local_semantic_component_lifecycle import LocalSemanticComponentLifecycleError


def _manifest(data):
    return LocalSemanticComponentManifest(
        component_id="test", release_id="v1", download_url="https://github.com/test/v1/component.zip",
        archive_name="component.zip", archive_root="c", archive_bytes=len(data),
        archive_sha256=hashlib.sha256(data).hexdigest(), payload_bytes=10, payload_file_count=1,
        payload_sha256=hashlib.sha256(b"payload").hexdigest(), candidate_id=CANDIDATE_ID, model_revision=MODEL_REVISION,
    )


def test_bundled_copy_resumes_and_keeps_existing_hash_gate(tmp_path):
    data = b"immutable component archive"
    manifest = _manifest(data)
    bundled = tmp_path / "components"
    bundled.mkdir()
    (bundled / manifest.archive_name).write_bytes(data)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / (manifest.archive_sha256 + ".zip.part")).write_bytes(data[:7])
    result = LocalSemanticComponentDownloader(cache).download(manifest, BundledComponentArchiveTransport(bundled))
    assert result.resumed_bytes == 7
    assert result.archive_path.read_bytes() == data


def test_corrupted_bundled_archive_is_rejected_without_network_fallback(tmp_path):
    data = b"immutable component archive"
    manifest = _manifest(data)
    (tmp_path / manifest.archive_name).write_bytes(b"x" * len(data))
    with pytest.raises(LocalSemanticComponentError, match="component_archive_hash_mismatch"):
        LocalSemanticComponentDownloader(tmp_path / "cache").download(manifest, BundledComponentArchiveTransport(tmp_path))


def test_bundled_wrong_size_is_rejected(tmp_path):
    manifest = _manifest(b"expected")
    (tmp_path / manifest.archive_name).write_bytes(b"short")
    with pytest.raises(LocalSemanticComponentError, match="component_bundled_archive_invalid"):
        BundledComponentArchiveTransport(tmp_path).open(manifest, start=0)


def test_bundled_status_and_declined_consent_remain_read_only(monkeypatch, tmp_path):
    manifest = _manifest(b"archive")
    monkeypatch.setenv(service.LOCAL_SEMANTIC_COMPONENT_MANIFEST_ENV, json.dumps(manifest.__dict__))
    monkeypatch.setattr(service, "LOCAL_SEMANTIC_COMPONENT_BUNDLED_ARCHIVE_DIR", tmp_path)
    (tmp_path / manifest.archive_name).write_bytes(b"archive")
    monkeypatch.setattr(service, "_build_service", lambda *_args: pytest.fail("installation before consent"))
    db = tmp_path / "db"
    local = service.get_local_semantic_component_service(db)
    assert local.status().state == "not_installed"
    assert service.local_semantic_component_disclosure()["network_required_for_install"] is False
    with pytest.raises(LocalSemanticComponentLifecycleError, match="consent_required"):
        local.enable(consent=False)
    assert not db.exists()


def test_bundled_transport_selection_never_constructs_network_transport(monkeypatch, tmp_path):
    manifest = _manifest(b"archive")
    monkeypatch.setattr(service, "LOCAL_SEMANTIC_COMPONENT_BUNDLED_ARCHIVE_DIR", tmp_path)
    (tmp_path / manifest.archive_name).write_bytes(b"archive")
    monkeypatch.setattr(service, "RequestsComponentArchiveTransport", lambda **_kwargs: pytest.fail("unexpected network transport"))
    assert isinstance(service._archive_transport(manifest), BundledComponentArchiveTransport)
