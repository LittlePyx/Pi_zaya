from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


COMPONENT_ID = "pizaya_local_semantic_e5_small_windows_x64_v1"
CANDIDATE_ID = "multilingual_e5_small_official_onnx_v1"
MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "03415a4be176a1620747c692ed433219fabc3def"
PYTHON_VERSION = "3.10.11"
ARCHIVE_NAME = "Pi_zaya-local-semantic-e5-small-v1-windows-x64.zip"
ARCHIVE_ROOT = "c"
EXPECTED_ARTIFACT_TREE_SHA256 = (
    "5671f8ed0e55e080c2bfb535acd2059d29cd371b510640048bef56eae66f3e01"
)
RUNTIME_ENTRIES = (
    ("onnxruntime", "1.23.2"),
    ("onnxruntime-1.23.2.dist-info", "1.23.2"),
    ("tokenizers", "0.21.4"),
    ("tokenizers-0.21.4.dist-info", "0.21.4"),
)
MAX_COMPONENT_RELATIVE_PATH_CHARS = 146
_RELEASE_TAG_RE = re.compile(
    r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_WINDOWS_REPARSE_POINT = 0x400


class ComponentBuildError(RuntimeError):
    pass


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _reject_unsafe_source(root: Path) -> None:
    if not root.is_dir():
        raise ComponentBuildError("component source directory is missing")
    for path in root.rglob("*"):
        try:
            attributes = int(getattr(path.stat(), "st_file_attributes", 0) or 0)
        except OSError as exc:
            raise ComponentBuildError("component source cannot be inspected") from exc
        if path.is_symlink() or attributes & _WINDOWS_REPARSE_POINT:
            raise ComponentBuildError("component source contains a link or reparse point")


def _copy_filtered_tree(source: Path, destination: Path) -> None:
    _reject_unsafe_source(source)
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    for path in sorted(
        source.rglob("*"),
        key=lambda item: item.relative_to(source).as_posix(),
    ):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        folded = relative.as_posix().casefold()
        if folded in observed:
            raise ComponentBuildError("component source has case-insensitive path collision")
        observed.add(folded)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _metadata_version(dist_info: Path) -> str:
    metadata = dist_info / "METADATA"
    try:
        lines = metadata.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ComponentBuildError("runtime distribution metadata is missing") from exc
    return next(
        (line.partition(":")[2].strip() for line in lines if line.startswith("Version:")),
        "",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _payload_identity(component_root: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    observed: set[str] = set()
    for path in sorted(
        (item for item in component_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(component_root).as_posix(),
    ):
        relative = path.relative_to(component_root).as_posix()
        folded = relative.casefold()
        if (
            folded in observed
            or len(relative) > MAX_COMPONENT_RELATIVE_PATH_CHARS
            or ".." in Path(relative).parts
        ):
            raise ComponentBuildError("component payload path contract failed")
        observed.add(folded)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{_raw_sha256(path)}\n".encode("utf-8"))
        count += 1
        total += size
    return {"file_count": count, "bytes": total, "sha256": digest.hexdigest()}


def _zip_deterministic(component_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as output:
        for path in sorted(
            (item for item in component_root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(component_root).as_posix(),
        ):
            relative = path.relative_to(component_root).as_posix()
            info = zipfile.ZipInfo(
                filename=f"{ARCHIVE_ROOT}/{relative}",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            with path.open("rb") as source, output.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)


def build_component(
    *,
    artifact_source: Path,
    site_packages: Path,
    output_dir: Path,
    release_tag: str,
    expected_artifact_tree_sha256: str = EXPECTED_ARTIFACT_TREE_SHA256,
) -> dict[str, Any]:
    if not _RELEASE_TAG_RE.fullmatch(release_tag):
        raise ComponentBuildError("component release tag is invalid")
    artifact_source = artifact_source.resolve(strict=False)
    site_packages = site_packages.resolve(strict=False)
    output_dir = output_dir.resolve(strict=False)
    archive_path = output_dir / ARCHIVE_NAME
    manifest_path = output_dir / f"{Path(ARCHIVE_NAME).stem}.manifest.json"
    checksum_path = archive_path.with_name(archive_path.name + ".sha256")
    if any(path.exists() for path in (archive_path, manifest_path, checksum_path)):
        raise ComponentBuildError("component release output already exists")
    _reject_unsafe_source(artifact_source)
    if _tree_fingerprint(artifact_source) != expected_artifact_tree_sha256:
        raise ComponentBuildError("component model artifact identity changed")
    _reject_unsafe_source(site_packages)
    for name, expected_version in RUNTIME_ENTRIES:
        source = site_packages / name
        if not source.exists():
            raise ComponentBuildError("component runtime entry is missing")
        if name.endswith(".dist-info") and _metadata_version(source) != expected_version:
            raise ComponentBuildError("component runtime version changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix="pizaya-semantic-", dir=output_dir))
    component_root = staging_parent / ARCHIVE_ROOT
    try:
        component_root.mkdir()
        _copy_filtered_tree(artifact_source, component_root / "artifacts")
        target_site = component_root / "python" / "site-packages"
        target_site.mkdir(parents=True)
        runtime_rows: list[dict[str, int | str]] = []
        for name, version in RUNTIME_ENTRIES:
            target = target_site / name
            source = site_packages / name
            if source.is_dir():
                _copy_filtered_tree(source, target)
            else:
                shutil.copyfile(source, target)
            runtime_rows.append(
                {
                    "name": name,
                    "version": version,
                    "bytes": (
                        _directory_bytes(target) if target.is_dir() else target.stat().st_size
                    ),
                    "tree_sha256": (
                        _tree_fingerprint(target) if target.is_dir() else _raw_sha256(target)
                    ),
                }
            )
        _write_json(
            component_root / "component-manifest.json",
            {
                "schema_version": 1,
                "component_id": COMPONENT_ID,
                "platform": "windows-x64",
                "python_version": PYTHON_VERSION,
                "candidate_id": CANDIDATE_ID,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "runtime_entries": runtime_rows,
                "artifact_tree_sha256": expected_artifact_tree_sha256,
                "contains_user_library_data": False,
                "product_enabled_by_installation": False,
            },
        )
        payload = _payload_identity(component_root)
        _zip_deterministic(component_root, archive_path)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

    archive_bytes = archive_path.stat().st_size
    archive_sha256 = _raw_sha256(archive_path)
    download_url = (
        "https://github.com/LittlePyx/Pi_zaya/releases/download/"
        f"{release_tag}/{ARCHIVE_NAME}"
    )
    manifest = {
        "schema_version": 1,
        "component_id": COMPONENT_ID,
        "release_id": release_tag,
        "download_url": download_url,
        "archive_name": ARCHIVE_NAME,
        "archive_root": ARCHIVE_ROOT,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "payload_bytes": payload["bytes"],
        "payload_file_count": payload["file_count"],
        "payload_sha256": payload["sha256"],
        "candidate_id": CANDIDATE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "python_version": PYTHON_VERSION,
        "contains_user_library_data": False,
        "product_enabled_by_installation": False,
    }
    _write_json(manifest_path, manifest)
    with checksum_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{archive_sha256}  {ARCHIVE_NAME}\n")
    return {
        "archive_path": archive_path,
        "manifest_path": manifest_path,
        "checksum_path": checksum_path,
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-source", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    try:
        result = build_component(
            artifact_source=args.artifact_source,
            site_packages=args.site_packages,
            output_dir=args.output_dir,
            release_tag=args.release_tag,
        )
    except (ComponentBuildError, OSError, zipfile.BadZipFile) as exc:
        print(f"[FAIL] local semantic component build: {type(exc).__name__}: {exc}")
        return 1
    manifest = result["manifest"]
    print(
        "[OK] local semantic component built: "
        f"{manifest['archive_bytes']} bytes / {manifest['payload_file_count']} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
