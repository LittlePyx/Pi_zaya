from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

import kb.local_semantic_component as component_module
from kb.local_semantic_component import (
    CANDIDATE_ID,
    COMPONENT_SAFETY_MARGIN_BYTES,
    MODEL_REVISION,
    LocalSemanticComponentError,
    LocalSemanticComponentManifest,
    LocalSemanticComponentStore,
    capacity_for_download,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, payload in sorted(files.items()):
        digest.update(
            f"{relative}\0{len(payload)}\0{hashlib.sha256(payload).hexdigest()}\n".encode()
        )
    return digest.hexdigest()


def _archive(tmp_path: Path, files: dict[str, bytes], *, name: str = "component.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, payload in files.items():
            archive.writestr(f"c/{relative}", payload)
    return path


def _manifest(archive: Path, files: dict[str, bytes], *, release_id: str = "component-v1"):
    return LocalSemanticComponentManifest.from_payload(
        {
            "component_id": "pizaya_local_semantic_e5_small_windows_x64_v1",
            "release_id": release_id,
            "download_url": (
                "https://github.com/LittlePyx/Pi_zaya/releases/download/"
                f"{release_id}/{archive.name}"
            ),
            "archive_name": archive.name,
            "archive_root": "c",
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": _sha256(archive),
            "payload_bytes": sum(len(payload) for payload in files.values()),
            "payload_file_count": len(files),
            "payload_sha256": _payload_sha(files),
            "candidate_id": CANDIDATE_ID,
            "model_revision": MODEL_REVISION,
        },
        allowed_hosts={"github.com"},
    )


def test_manifest_requires_https_allowlisted_immutable_release_url(tmp_path: Path) -> None:
    archive = _archive(tmp_path, {"a.bin": b"a"})
    payload = {
        "component_id": "component",
        "release_id": "component-v1",
        "download_url": "http://example.test/latest/component.zip",
        "archive_name": archive.name,
        "archive_root": "c",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "payload_bytes": 1,
        "payload_file_count": 1,
        "payload_sha256": "0" * 64,
        "candidate_id": CANDIDATE_ID,
        "model_revision": MODEL_REVISION,
    }
    with pytest.raises(LocalSemanticComponentError) as captured:
        LocalSemanticComponentManifest.from_payload(
            payload,
            allowed_hosts={"example.test"},
        )
    assert captured.value.code == "component_manifest_invalid"


def test_capacity_includes_archive_payload_margin_and_current_generation(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path, {"a.bin": b"abc"})
    manifest = _manifest(archive, {"a.bin": b"abc"})
    capacity = capacity_for_download(
        tmp_path,
        manifest,
        current_generation_bytes=17,
        free_bytes_probe=lambda _path: 10,
    )
    assert capacity.required_bytes == (
        manifest.archive_bytes
        + manifest.payload_bytes
        + COMPONENT_SAFETY_MARGIN_BYTES
        + 17
    )
    assert capacity.required_additional_bytes == capacity.required_bytes - 10


def test_install_verifies_payload_health_and_atomically_publishes(tmp_path: Path) -> None:
    files = {
        "artifacts/model.onnx": b"model",
        "artifacts/tokenizer/tokenizer.json": b"tokenizer",
        "python/site-packages/onnxruntime/__init__.py": b"version='1.23.2'",
    }
    archive = _archive(tmp_path, files)
    manifest = _manifest(archive, files)
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )

    installed = store.install_from_archive(
        archive,
        manifest,
        health_check=lambda root: (root / "artifacts" / "model.onnx").read_bytes()
        == b"model",
    )

    assert installed == store.load_current()
    assert len(installed.generation_id) == 16
    assert installed.payload_sha256 == manifest.payload_sha256
    assert (installed.root / "artifacts" / "tokenizer" / "tokenizer.json").is_file()
    assert not list(store.generations_dir.glob(".staging-*"))


def test_low_disk_is_reported_before_extraction(tmp_path: Path) -> None:
    files = {"a.bin": b"abc"}
    archive = _archive(tmp_path, files)
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 0,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(archive, _manifest(archive, files))
    assert captured.value.code == "semantic_component_needs_space"
    assert captured.value.required_additional_bytes > 0
    assert not store.current_path.exists()


def test_path_escape_is_rejected_without_writing_outside_component_root(
    tmp_path: Path,
) -> None:
    files = {"../../escape.txt": b"escape"}
    archive = _archive(tmp_path, files)
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(archive, _manifest(archive, files))
    assert captured.value.code == "component_archive_entry_unsafe"
    assert not (tmp_path / "escape.txt").exists()
    assert not store.current_path.exists()


def test_symlink_entry_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("c/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, "target")
    files = {"link": b"target"}
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(archive, _manifest(archive, files))
    assert captured.value.code == "component_archive_entry_unsafe"
    assert not store.current_path.exists()


def test_case_insensitive_duplicate_path_is_rejected(tmp_path: Path) -> None:
    files = {"A.bin": b"one", "a.bin": b"two"}
    archive = _archive(tmp_path, files)
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(archive, _manifest(archive, files))
    assert captured.value.code == "component_archive_duplicate_path"
    assert not store.current_path.exists()


def test_archive_hash_mismatch_stops_before_generation_creation(tmp_path: Path) -> None:
    files = {"a.bin": b"abc"}
    archive = _archive(tmp_path, files)
    valid = _manifest(archive, files)
    manifest = LocalSemanticComponentManifest(
        **{**valid.__dict__, "archive_sha256": "0" * 64}
    )
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(archive, manifest)
    assert captured.value.code == "component_archive_hash_mismatch"
    assert not store.generations_dir.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows path budget")
def test_overlong_storage_layout_is_rejected_before_staging_creation(
    tmp_path: Path,
) -> None:
    files = {f"deep/{'x' * 170}.bin": b"abc"}
    archive = _archive(tmp_path, files)
    root = tmp_path / ("component-root-" + "x" * 90)
    store = LocalSemanticComponentStore(
        root,
        free_bytes_probe=lambda _path: 10**12,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(archive, _manifest(archive, files))
    assert captured.value.code == "component_storage_path_too_long"
    assert not store.generations_dir.exists()


def test_payload_identity_mismatch_never_publishes_generation(tmp_path: Path) -> None:
    files = {"a.bin": b"abc"}
    archive = _archive(tmp_path, files)
    valid = _manifest(archive, files)
    manifest = LocalSemanticComponentManifest(
        **{**valid.__dict__, "payload_sha256": "0" * 64}
    )
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(archive, manifest)
    assert captured.value.code == "component_payload_hash_mismatch"
    assert not store.current_path.exists()
    assert not list(store.generations_dir.glob("*"))


def test_failed_upgrade_health_check_preserves_previous_current_generation(
    tmp_path: Path,
) -> None:
    first_files = {"version.txt": b"one"}
    first_archive = _archive(tmp_path, first_files, name="one.zip")
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    first = store.install_from_archive(first_archive, _manifest(first_archive, first_files))

    second_files = {"version.txt": b"two"}
    second_archive = _archive(tmp_path, second_files, name="two.zip")
    second_manifest = _manifest(
        second_archive,
        second_files,
        release_id="component-v2",
    )
    with pytest.raises(LocalSemanticComponentError) as captured:
        store.install_from_archive(
            second_archive,
            second_manifest,
            health_check=lambda _root: False,
        )

    assert captured.value.code == "component_install_health_failed"
    assert store.load_current() == first
    assert (first.root / "version.txt").read_bytes() == b"one"
    pointer = json.loads(store.current_path.read_text(encoding="utf-8"))
    assert pointer["generation_id"] == first.generation_id


def test_remove_is_idempotent_and_disables_before_owned_cleanup(tmp_path: Path) -> None:
    files = {"version.txt": b"one"}
    archive = _archive(tmp_path, files)
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    installed = store.install_from_archive(archive, _manifest(archive, files))
    downloads = store.root / "downloads"
    downloads.mkdir()
    (downloads / "owned.zip.part").write_bytes(b"partial")

    first = store.remove_all()
    second = store.remove_all()

    assert first.removed_generation_count == 1
    assert first.cleanup_pending is False
    assert second.removed_generation_count == 0
    assert second.cleanup_pending is False
    assert not store.current_path.exists()
    assert not installed.root.exists()
    assert not downloads.exists()


def test_locked_generation_is_disabled_then_cleaned_on_next_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    files = {"version.txt": b"one"}
    archive = _archive(tmp_path, files)
    store = LocalSemanticComponentStore(
        tmp_path / "components",
        free_bytes_probe=lambda _path: 10**12,
    )
    installed = store.install_from_archive(archive, _manifest(archive, files))
    original_remove = component_module._remove_owned_tree
    blocked = {"value": True}

    def locked_once(root: Path, target: Path) -> None:
        if target == installed.root and blocked["value"]:
            blocked["value"] = False
            raise PermissionError("simulated Windows lock")
        original_remove(root, target)

    monkeypatch.setattr(component_module, "_remove_owned_tree", locked_once)
    result = store.remove_all()
    assert result.cleanup_pending is True
    assert not store.current_path.exists()
    assert installed.root.exists()
    assert (store.root / "pending_cleanup.json").is_file()

    assert store.retry_pending_cleanup() is True
    assert not installed.root.exists()
    assert not (store.root / "pending_cleanup.json").exists()
