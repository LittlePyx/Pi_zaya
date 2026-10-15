from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import site
import sys
import time
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPO_ROOT / "docs" / "retrieval_lighter_runtime_screen_protocol_v1.json"
DEFAULT_REQUIREMENTS = REPO_ROOT / "tools" / "retrieval_eval" / "lighter_runtime_requirements_v1.txt"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "test_results" / "retrieval_lighter_runtime_v1" / "artifacts"
DEFAULT_REPORT = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage0_qwen_v1.json"
ALLOWED_BOOTSTRAP_DISTRIBUTIONS = {"pip", "setuptools"}


def _normalized_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _locked_requirements(path: Path = DEFAULT_REQUIREMENTS) -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise ValueError("lighter runtime requirement must use an exact == pin")
        normalized = _normalize_distribution_name(name)
        if normalized in locked:
            raise ValueError("lighter runtime requirement names must be unique")
        locked[normalized] = version
    return locked


def _installed_distributions() -> dict[str, str]:
    return {
        _normalize_distribution_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }


def _validate_environment() -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if version != "3.10.11":
        errors.append(f"expected Python 3.10.11, found {version}")
    if sys.prefix == sys.base_prefix:
        errors.append("dedicated virtual environment is required")
    if site.ENABLE_USER_SITE:
        errors.append("user site packages must be disabled")

    locked = _locked_requirements()
    installed = _installed_distributions()
    for name, expected_version in locked.items():
        if installed.get(name) != expected_version:
            errors.append(f"{name} must be {expected_version}, found {installed.get(name) or 'missing'}")
    extras = sorted(set(installed) - set(locked) - ALLOWED_BOOTSTRAP_DISTRIBUTIONS)
    if extras:
        errors.append("unexpected distributions: " + ", ".join(extras))
    for forbidden in ("torch", "transformers", "optimum", "sentence-transformers"):
        if forbidden in installed:
            errors.append(f"forbidden distribution installed: {forbidden}")
    return installed, errors


def _distribution_bytes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = _normalize_distribution_name(name)
        if normalized in ALLOWED_BOOTSTRAP_DISTRIBUTIONS:
            continue
        files = distribution.files or []
        total = 0
        seen: set[Path] = set()
        for entry in files:
            candidate = Path(distribution.locate_file(entry))
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
        rows.append({"name": normalized, "version": distribution.version, "installed_bytes": total})
    return sorted(rows, key=lambda row: row["name"])


def _load_protocol() -> dict[str, Any]:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "retrieval_lighter_runtime_screen_v1":
        raise RuntimeError("lighter runtime protocol identity changed")
    if protocol.get("status") != "frozen_before_artifact_download":
        raise RuntimeError("lighter runtime protocol is no longer frozen before download")
    return protocol


def _select_candidate(protocol: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = list(protocol.get("candidates") or [])
    for candidate in candidates:
        if candidate.get("candidate_id") == candidate_id:
            if candidate.get("evaluation_order") == 2:
                predecessor = REPO_ROOT / "docs" / "retrieval_lighter_runtime_stage1_qwen_v1.json"
                if not predecessor.is_file():
                    raise RuntimeError("candidate 2 is blocked until the retained candidate-1 rejection exists")
                predecessor_payload = json.loads(predecessor.read_text(encoding="utf-8"))
                if (predecessor_payload.get("decision") or {}).get("status") != "rejected_stage_1":
                    raise RuntimeError("candidate 2 is blocked because candidate 1 was not rejected")
            return candidate
    raise ValueError(f"unknown lighter runtime candidate: {candidate_id}")


def _download_verified_artifact(
    candidate: dict[str, Any], artifact_root: Path
) -> tuple[Path, int, bool, int]:
    import requests

    artifact = candidate["primary_artifact"]
    candidate_root = (artifact_root / candidate["candidate_id"]).resolve()
    resolved_root = artifact_root.resolve()
    if resolved_root not in candidate_root.parents:
        raise RuntimeError("candidate artifact path escaped the configured artifact root")
    candidate_root.mkdir(parents=True, exist_ok=True)
    destination = candidate_root / Path(artifact["path"]).name
    expected_size = int(artifact["bytes"])
    expected_sha = str(artifact["sha256"])
    if destination.is_file():
        if destination.stat().st_size == expected_size and _raw_sha256(destination) == expected_sha:
            return destination, 0, True, 0
        raise RuntimeError("existing final artifact failed the frozen size or SHA-256 check")

    partial = destination.with_suffix(destination.suffix + ".partial")
    initial_bytes = partial.stat().st_size if partial.is_file() else 0
    if initial_bytes >= expected_size:
        raise RuntimeError("partial artifact is not smaller than the frozen final artifact")
    transferred_bytes = 0
    attempt_count = 0
    maximum_attempts = 12
    while (partial.stat().st_size if partial.is_file() else 0) < expected_size:
        attempt_count += 1
        if attempt_count > maximum_attempts:
            raise RuntimeError("artifact download exhausted bounded retry attempts")
        existing_bytes = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "Pi_zaya-retrieval-runtime-preflight/1"}
        mode = "wb"
        if existing_bytes:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"
        try:
            with requests.get(
                str(artifact["url"]), headers=headers, stream=True, timeout=(15, 120)
            ) as response:
                response.raise_for_status()
                if mode == "ab" and response.status_code != 206:
                    mode = "wb"
                    existing_bytes = 0
                downloaded = existing_bytes
                next_progress = ((downloaded // (64 * 1024 * 1024)) + 1) * (
                    64 * 1024 * 1024
                )
                with partial.open(mode) as handle:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        transferred_bytes += len(block)
                        downloaded += len(block)
                        if downloaded >= next_progress:
                            print(f"[download] {downloaded}/{expected_size} bytes")
                            next_progress += 64 * 1024 * 1024
                    handle.flush()
                    os.fsync(handle.fileno())
        except (requests.RequestException, OSError) as exc:
            retained = partial.stat().st_size if partial.is_file() else 0
            print(
                f"[download-retry] attempt={attempt_count} retained={retained} "
                f"error={type(exc).__name__}"
            )
            time.sleep(min(attempt_count, 5))
    if partial.stat().st_size != expected_size:
        raise RuntimeError("downloaded artifact size does not match the frozen size")
    digest = _raw_sha256(partial)
    if digest != expected_sha:
        raise RuntimeError("downloaded artifact SHA-256 does not match the frozen digest")
    os.replace(partial, destination)
    return destination, transferred_bytes, False, attempt_count


def _download_revision_file(url: str, destination: Path) -> tuple[int, bool]:
    import requests

    if destination.is_file():
        return 0, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    transferred_bytes = 0
    for attempt in range(1, 6):
        try:
            with requests.get(
                url,
                headers={"User-Agent": "Pi_zaya-retrieval-runtime-preflight/1"},
                stream=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        transferred_bytes += len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(partial, destination)
            return transferred_bytes, False
        except (requests.RequestException, OSError):
            partial.unlink(missing_ok=True)
            if attempt == 5:
                raise RuntimeError(
                    f"revision file download exhausted bounded retries: {destination.name}"
                )
            time.sleep(attempt)
    raise AssertionError("unreachable revision download state")


def _prepare_candidate_tokenizer(
    candidate: dict[str, Any], artifact_root: Path
) -> list[dict[str, Any]]:
    tokenizer = candidate.get("tokenizer") or {}
    required_files = list(tokenizer.get("required_files") or [])
    if not required_files:
        return []
    revision = str(tokenizer.get("revision") or candidate["model_revision"])
    subdirectory = str(tokenizer.get("repository_subdirectory") or "").strip("/")
    candidate_root = (artifact_root / candidate["candidate_id"]).resolve()
    tokenizer_root = (candidate_root / "tokenizer").resolve()
    if candidate_root not in tokenizer_root.parents:
        raise RuntimeError("candidate tokenizer path escaped its artifact root")

    rows: list[dict[str, Any]] = []
    for filename in required_files:
        relative_path = "/".join(part for part in (subdirectory, str(filename)) if part)
        url = (
            f"https://huggingface.co/{quote(str(candidate['model_id']), safe='/')}"
            f"/resolve/{quote(revision, safe='')}/{quote(relative_path, safe='/')}"
        )
        destination = tokenizer_root / Path(str(filename)).name
        downloaded_bytes, reused = _download_revision_file(url, destination)
        rows.append(
            {
                "repository_path": relative_path,
                "local_name": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": _raw_sha256(destination),
                "downloaded_bytes_this_run": downloaded_bytes,
                "reused": reused,
            }
        )
    return rows


def _inspect_onnx(artifact: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    checker_started = time.perf_counter()
    onnx.checker.check_model(str(artifact), full_check=False)
    checker_ms = (time.perf_counter() - checker_started) * 1000.0

    options = ort.SessionOptions()
    runtime = protocol["runtime_contract"]
    options.intra_op_num_threads = int(runtime["intra_op_threads"])
    options.inter_op_num_threads = int(runtime["inter_op_threads"])
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_started = time.perf_counter()
    session = ort.InferenceSession(
        str(artifact),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    session_ms = (time.perf_counter() - session_started) * 1000.0

    def describe(value: Any) -> dict[str, Any]:
        return {"name": value.name, "type": value.type, "shape": value.shape}

    return {
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "session_providers": session.get_providers(),
        "checker_ms": round(checker_ms, 3),
        "session_create_ms": round(session_ms, 3),
        "inputs": [describe(value) for value in session.get_inputs()],
        "outputs": [describe(value) for value in session.get_outputs()],
        "session_run_calls": 0,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(candidate_id: str, artifact_root: Path, report_path: Path) -> dict[str, Any]:
    if report_path.exists():
        raise RuntimeError("stage-0 report already exists; repeated preparation is not allowed")
    protocol = _load_protocol()
    candidate = _select_candidate(protocol, candidate_id)
    installed, environment_errors = _validate_environment()
    if environment_errors:
        raise RuntimeError("; ".join(environment_errors))

    artifact_path, downloaded_bytes, reused, download_attempts = _download_verified_artifact(
        candidate, artifact_root
    )
    tokenizer_files = _prepare_candidate_tokenizer(candidate, artifact_root)
    inspection = _inspect_onnx(artifact_path, protocol)
    if inspection["session_providers"] != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX session did not use only CPUExecutionProvider")
    distributions = _distribution_bytes()
    payload = {
        "schema_version": 1,
        "report_kind": "retrieval_lighter_runtime_stage0_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _normalized_sha256(DEFAULT_PROTOCOL),
        "requirements_sha256": _normalized_sha256(DEFAULT_REQUIREMENTS),
        "candidate": {
            "candidate_id": candidate_id,
            "evaluation_order": candidate["evaluation_order"],
            "model_id": candidate["model_id"],
            "model_revision": candidate["model_revision"],
            "artifact_path": candidate["primary_artifact"]["path"],
            "artifact_bytes": artifact_path.stat().st_size,
            "artifact_sha256": _raw_sha256(artifact_path),
            "artifact_reused": reused,
            "downloaded_bytes_this_run": downloaded_bytes,
            "download_attempts_this_run": download_attempts,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "virtual_environment": sys.prefix != sys.base_prefix,
            "user_site_enabled": bool(site.ENABLE_USER_SITE),
            "locked_distribution_count": len(_locked_requirements()),
            "installed_distribution_count_excluding_bootstrap": len(
                set(installed) - ALLOWED_BOOTSTRAP_DISTRIBUTIONS
            ),
            "distributions": distributions,
            "installed_distribution_bytes": sum(row["installed_bytes"] for row in distributions),
        },
        "onnx": inspection,
        "tokenizer": {
            "repository_revision": str(
                (candidate.get("tokenizer") or {}).get("revision")
                or candidate["model_revision"]
            ),
            "files": tokenizer_files,
            "tokenizer_load_calls": 0,
        },
        "execution": {
            "paper_text_accessed": False,
            "queries_encoded": 0,
            "documents_embedded": 0,
            "vectors_created": 0,
            "model_inference_runs": 0,
            "product_requirements_modified": False,
            "product_enabled": False,
            "ordinary_user_surface_exposed": False,
        },
        "decision": {
            "status": "passed_stage_0",
            "stage_1_allowed": True,
            "product_enabled": False,
        },
    }
    _atomic_write_json(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and inspect one frozen lighter CPU runtime candidate")
    parser.add_argument(
        "--candidate-id",
        default="qwen3_embedding_0_6b_official_quint8_avx2_onnx_v1",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args()
    try:
        protocol = _load_protocol()
        _select_candidate(protocol, args.candidate_id)
        installed, errors = _validate_environment()
        if errors:
            raise RuntimeError("; ".join(errors))
        if args.environment_only:
            print(
                "[OK] isolated lighter runtime environment matches "
                f"{len(_locked_requirements())} exact pins; {len(installed)} distributions observed"
            )
            return 0
        payload = prepare(args.candidate_id, args.artifact_root, args.report)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[FAIL] lighter runtime stage 0: {type(exc).__name__}: {exc}")
        return 1
    print(
        "[OK] lighter runtime stage 0 passed: "
        f"{payload['candidate']['candidate_id']} / {payload['candidate']['artifact_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
