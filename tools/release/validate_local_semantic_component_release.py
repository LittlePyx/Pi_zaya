from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.local_semantic_component import (
    LocalSemanticComponentManifest,
    _safe_relative_path,
)
from kb.local_semantic_component_service import (
    LOCAL_SEMANTIC_COMPONENT_ALLOWED_HOSTS,
)
from tools.release.build_local_semantic_component import (
    ARCHIVE_NAME,
    CANDIDATE_ID,
    COMPONENT_ID,
    MAX_COMPONENT_RELATIVE_PATH_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    PYTHON_VERSION,
    _RELEASE_TAG_RE,
    _raw_sha256,
)


def validate_component_release(
    archive_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["component release manifest is unreadable"]
    try:
        manifest = LocalSemanticComponentManifest.from_payload(
            payload,
            allowed_hosts=LOCAL_SEMANTIC_COMPONENT_ALLOWED_HOSTS,
        )
    except Exception:
        return payload, ["component release manifest contract failed"]
    if (
        manifest_path.name != f"{Path(ARCHIVE_NAME).stem}.manifest.json"
        or manifest.archive_name != ARCHIVE_NAME
        or manifest.component_id != COMPONENT_ID
        or manifest.candidate_id != CANDIDATE_ID
        or manifest.model_revision != MODEL_REVISION
        or not _RELEASE_TAG_RE.fullmatch(manifest.release_id)
        or payload.get("schema_version") != 1
        or payload.get("model_id") != MODEL_ID
        or payload.get("python_version") != PYTHON_VERSION
        or payload.get("contains_user_library_data") is not False
        or payload.get("product_enabled_by_installation") is not False
    ):
        errors.append("component release identity changed")
    try:
        if (
            archive_path.name != ARCHIVE_NAME
            or archive_path.stat().st_size != manifest.archive_bytes
            or _raw_sha256(archive_path) != manifest.archive_sha256
        ):
            errors.append("component release archive identity changed")
    except OSError:
        return payload, errors + ["component release archive is unavailable"]

    digest = hashlib.sha256()
    observed_paths: set[str] = set()
    relative_names: list[str] = []
    entries: list[tuple[str, zipfile.ZipInfo]] = []
    count = 0
    total = 0
    inner_manifest: dict[str, Any] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                try:
                    relative = _safe_relative_path(
                        info,
                        archive_root=manifest.archive_root,
                    )
                except Exception:
                    errors.append("component release archive entry is unsafe")
                    break
                if relative is None:
                    continue
                folded = relative.casefold()
                if folded in observed_paths:
                    errors.append("component release archive path collision")
                    break
                observed_paths.add(folded)
                relative_names.append(relative)
                entries.append((relative, info))
                if (
                    len(relative) > MAX_COMPONENT_RELATIVE_PATH_CHARS
                    or "__pycache__" in Path(relative).parts
                    or relative.lower().endswith((".pyc", ".pyo"))
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                ):
                    errors.append("component release deterministic path contract failed")
                    break
            if relative_names != sorted(relative_names):
                errors.append("component release archive ordering changed")
            for relative, info in sorted(entries, key=lambda item: item[0]):
                file_digest = hashlib.sha256()
                written = 0
                with archive.open(info) as source:
                    while True:
                        block = source.read(4 * 1024 * 1024)
                        if not block:
                            break
                        file_digest.update(block)
                        written += len(block)
                digest.update(
                    f"{relative}\0{written}\0{file_digest.hexdigest()}\n".encode(
                        "utf-8"
                    )
                )
                count += 1
                total += written
                if relative == "component-manifest.json":
                    inner_manifest = json.loads(archive.read(info).decode("utf-8"))
    except (OSError, EOFError, RuntimeError, ValueError, zipfile.BadZipFile):
        errors.append("component release ZIP is invalid")

    if (
        count != manifest.payload_file_count
        or total != manifest.payload_bytes
        or digest.hexdigest() != manifest.payload_sha256
    ):
        errors.append("component release payload identity changed")
    if (
        inner_manifest.get("schema_version") != 1
        or inner_manifest.get("component_id") != COMPONENT_ID
        or inner_manifest.get("candidate_id") != CANDIDATE_ID
        or inner_manifest.get("model_id") != MODEL_ID
        or inner_manifest.get("model_revision") != MODEL_REVISION
        or inner_manifest.get("python_version") != PYTHON_VERSION
        or inner_manifest.get("contains_user_library_data") is not False
        or inner_manifest.get("product_enabled_by_installation") is not False
    ):
        errors.append("component release inner manifest changed")
    return payload, list(dict.fromkeys(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload, errors = validate_component_release(
        args.archive.resolve(strict=False),
        args.manifest.resolve(strict=False),
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(
        "[OK] local semantic component release verified: "
        f"{payload['archive_bytes']} bytes / {payload['payload_file_count']} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
