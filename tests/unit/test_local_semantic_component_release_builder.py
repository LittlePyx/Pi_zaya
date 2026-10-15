from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from kb.local_semantic_component import (
    LocalSemanticComponentManifest,
    LocalSemanticComponentStore,
)
from tools.release.build_local_semantic_component import (
    ARCHIVE_NAME,
    ComponentBuildError,
    _tree_fingerprint,
    build_component,
)
from tools.release.validate_local_semantic_component_release import (
    validate_component_release,
)


def _sources(tmp_path: Path) -> tuple[Path, Path, str]:
    artifacts = tmp_path / "artifacts"
    (artifacts / "tokenizer").mkdir(parents=True)
    (artifacts / "model.onnx").write_bytes(b"model")
    (artifacts / "tokenizer" / "tokenizer.json").write_text(
        "{}",
        encoding="utf-8",
    )
    site = tmp_path / "site-packages"
    for package in ("onnxruntime", "tokenizers"):
        root = site / package
        (root / "__pycache__").mkdir(parents=True)
        (root / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
        (root / "__pycache__" / "ignored.pyc").write_bytes(b"unstable")
    for name, version in (
        ("onnxruntime-1.23.2.dist-info", "1.23.2"),
        ("tokenizers-0.21.4.dist-info", "0.21.4"),
    ):
        root = site / name
        root.mkdir(parents=True)
        (root / "METADATA").write_text(
            f"Name: {name.partition('-')[0]}\nVersion: {version}\n",
            encoding="utf-8",
        )
    return artifacts, site, _tree_fingerprint(artifacts)


def test_release_component_is_deterministic_and_installable(tmp_path: Path) -> None:
    artifacts, site, artifact_sha = _sources(tmp_path)
    first = build_component(
        artifact_source=artifacts,
        site_packages=site,
        output_dir=tmp_path / "release-one",
        release_tag="v0.1.0-beta.14",
        expected_artifact_tree_sha256=artifact_sha,
    )
    second = build_component(
        artifact_source=artifacts,
        site_packages=site,
        output_dir=tmp_path / "release-two",
        release_tag="v0.1.0-beta.14",
        expected_artifact_tree_sha256=artifact_sha,
    )

    assert first["manifest"] == second["manifest"]
    assert first["manifest"]["archive_sha256"] == second["manifest"][
        "archive_sha256"
    ]
    assert first["archive_path"].name == ARCHIVE_NAME
    assert first["checksum_path"].read_text(encoding="ascii").endswith(
        f"  {ARCHIVE_NAME}\n"
    )
    with zipfile.ZipFile(first["archive_path"]) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)

    manifest = LocalSemanticComponentManifest.from_payload(
        first["manifest"],
        allowed_hosts={"github.com"},
    )
    store = LocalSemanticComponentStore(
        tmp_path / "installed",
        free_bytes_probe=lambda _path: 10**9,
    )
    installed = store.install_from_archive(
        first["archive_path"],
        manifest,
        health_check=lambda root: (root / "component-manifest.json").is_file(),
    )
    assert installed.payload_sha256 == first["manifest"]["payload_sha256"]
    assert not any(installed.root.rglob("*.pyc"))
    _payload, errors = validate_component_release(
        first["archive_path"],
        first["manifest_path"],
    )
    assert errors == []


def test_release_component_rejects_artifact_drift(tmp_path: Path) -> None:
    artifacts, site, _artifact_sha = _sources(tmp_path)

    with pytest.raises(ComponentBuildError, match="artifact identity"):
        build_component(
            artifact_source=artifacts,
            site_packages=site,
            output_dir=tmp_path / "release",
            release_tag="v0.1.0-beta.14",
            expected_artifact_tree_sha256="0" * 64,
        )


def test_release_component_rejects_runtime_version_drift(tmp_path: Path) -> None:
    artifacts, site, artifact_sha = _sources(tmp_path)
    (site / "tokenizers-0.21.4.dist-info" / "METADATA").write_text(
        "Name: tokenizers\nVersion: 0.21.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ComponentBuildError, match="runtime version"):
        build_component(
            artifact_source=artifacts,
            site_packages=site,
            output_dir=tmp_path / "release",
            release_tag="v0.1.0-beta.14",
            expected_artifact_tree_sha256=artifact_sha,
        )


def test_release_component_refuses_overwrite_and_invalid_tag(tmp_path: Path) -> None:
    artifacts, site, artifact_sha = _sources(tmp_path)
    output = tmp_path / "release"
    output.mkdir()
    (output / ARCHIVE_NAME).write_bytes(b"existing")

    with pytest.raises(ComponentBuildError, match="already exists"):
        build_component(
            artifact_source=artifacts,
            site_packages=site,
            output_dir=output,
            release_tag="v0.1.0-beta.14",
            expected_artifact_tree_sha256=artifact_sha,
        )
    with pytest.raises(ComponentBuildError, match="tag is invalid"):
        build_component(
            artifact_source=artifacts,
            site_packages=site,
            output_dir=tmp_path / "other",
            release_tag="latest",
            expected_artifact_tree_sha256=artifact_sha,
        )


def test_release_manifest_has_no_local_paths(tmp_path: Path) -> None:
    artifacts, site, artifact_sha = _sources(tmp_path)
    result = build_component(
        artifact_source=artifacts,
        site_packages=site,
        output_dir=tmp_path / "release",
        release_tag="v0.1.0-beta.14",
        expected_artifact_tree_sha256=artifact_sha,
    )

    serialized = json.dumps(result["manifest"])
    assert str(tmp_path) not in serialized
    assert result["manifest"]["contains_user_library_data"] is False
    assert result["manifest"]["product_enabled_by_installation"] is False


def test_release_validator_rejects_manifest_or_archive_drift(tmp_path: Path) -> None:
    artifacts, site, artifact_sha = _sources(tmp_path)
    result = build_component(
        artifact_source=artifacts,
        site_packages=site,
        output_dir=tmp_path / "release",
        release_tag="v0.1.0-beta.14",
        expected_artifact_tree_sha256=artifact_sha,
    )
    changed = dict(result["manifest"])
    changed["release_id"] = "v0.1.0-beta.15"
    result["manifest_path"].write_text(json.dumps(changed), encoding="utf-8")
    _payload, manifest_errors = validate_component_release(
        result["archive_path"],
        result["manifest_path"],
    )
    assert "component release manifest contract failed" in manifest_errors

    result["manifest_path"].write_text(
        json.dumps(result["manifest"]),
        encoding="utf-8",
    )
    with result["archive_path"].open("ab") as handle:
        handle.write(b"drift")
    _payload, archive_errors = validate_component_release(
        result["archive_path"],
        result["manifest_path"],
    )
    assert "component release archive identity changed" in archive_errors
