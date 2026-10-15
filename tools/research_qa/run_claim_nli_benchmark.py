from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.claim_entailment_eval_fixture import validate_claim_entailment_fixture
from kb.claim_entailment_nli_benchmark import (
    evaluate_nli_acceptance,
    evaluate_nli_predictions,
)
from kb.retrieval_eval_fixture import sha256_file
from tools.research_qa.transformers_nli_provider import TransformersNliProvider
from tools.research_qa.validate_claim_nli_candidate import validate_registry


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "claim_entailment_eval_v1.json"
DEFAULT_QRELS = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "claim_entailment_nli_candidate_v1.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "claim_entailment_nli_baseline_v1.json"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _load_inputs(
    fixture_path: Path,
    qrels_path: Path,
    registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    errors = validate_claim_entailment_fixture(
        fixture,
        qrels,
        qrels_sha256=sha256_file(qrels_path),
    )
    errors.extend(validate_registry(registry))
    if errors:
        raise ValueError("invalid frozen NLI inputs: " + "; ".join(errors))
    return fixture, registry


def _evidence_texts(fixture: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("id") or ""): " ".join(
            str(segment.get("quote") or "").strip() for segment in row.get("segments") or []
        )
        for row in fixture.get("evidence") or []
    }


def _acceptance_from_report(
    report: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    runtime = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    return evaluate_nli_acceptance(
        report.get("result") or {},
        registry.get("acceptance") or {},
        truncated_pair_count=int(runtime.get("truncated_pair_count") or 0),
    )


def validate_report(
    report: dict[str, Any],
    fixture_path: Path,
    registry_path: Path,
    fixture: dict[str, Any],
    registry: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if report.get("report_kind") != "multilingual_nli_claim_baseline":
        errors.append("report kind does not match")
    source = report.get("fixture") if isinstance(report.get("fixture"), dict) else {}
    if source.get("sha256") != sha256_file(fixture_path):
        errors.append("fixture SHA-256 does not match")
    selection = report.get("registry") if isinstance(report.get("registry"), dict) else {}
    if selection.get("sha256") != sha256_file(registry_path):
        errors.append("candidate registry SHA-256 does not match")
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    frozen = registry.get("candidate") or {}
    for key in ("model_id", "revision", "label_order", "max_length", "truncation", "precision"):
        if candidate.get(key) != frozen.get(key):
            errors.append(f"candidate {key} does not match the frozen registry")
    if candidate.get("allow_download") is not False:
        errors.append("frozen measurement must be cache-only")
    if candidate.get("trust_remote_code") is not False or candidate.get("product_enabled") is not False:
        errors.append("measured candidate must remain product-disabled without remote code")
    if candidate.get("safe_weights_required") is not True:
        errors.append("safe weight requirement is missing")
    if int(candidate.get("pair_count") or 0) != len(fixture.get("cases") or []):
        errors.append("candidate pair count does not cover the fixture")

    result = report.get("result") if isinstance(report.get("result"), dict) else {}
    stored_cases = result.get("cases") if isinstance(result.get("cases"), list) else []
    try:
        recomputed = evaluate_nli_predictions(fixture, stored_cases)
    except ValueError as exc:
        errors.append(str(exc))
        recomputed = None
    if recomputed is not None and result != recomputed:
        errors.append("stored NLI metrics or case decisions are stale")
    expected_acceptance = _acceptance_from_report(report, registry)
    if report.get("acceptance") != expected_acceptance:
        errors.append("NLI acceptance decision is stale")
    expected_status = (
        "eligible_for_shadow_design"
        if expected_acceptance["passes_all"]
        else "rejected_for_shadow_design"
    )
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    if decision.get("status") != expected_status or decision.get("product_enabled") is not False:
        errors.append("NLI decision or product activation is stale")
    return errors


def _prepare_model(registry: dict[str, Any], cache_dir: Path | None) -> int:
    from huggingface_hub import snapshot_download

    candidate = registry["candidate"]
    snapshot = Path(
        snapshot_download(
            candidate["model_id"],
            revision=candidate["revision"],
            cache_dir=str(cache_dir.resolve()) if cache_dir else None,
            allow_patterns=[
                "config.json",
                "spm.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    )
    _download_weight(
        model_id=str(candidate["model_id"]),
        revision=str(candidate["revision"]),
        target=snapshot / "model.safetensors",
        expected_size=int(candidate["model_safetensors_bytes"]),
        expected_sha256=str(candidate["model_safetensors_sha256"]),
    )
    size = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
    print(f"[OK] cached pinned NLI candidate: {snapshot} ({size} bytes)")
    return 0


def _download_weight(
    *,
    model_id: str,
    revision: str,
    target: Path,
    expected_size: int,
    expected_sha256: str,
) -> None:
    import requests

    if target.is_file() and _file_sha256(target) == expected_sha256:
        print(f"[OK] pinned safe weight already cached: {target}")
        return
    partial = target.with_suffix(f"{target.suffix}.part")
    if partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors"
    attempts = 0
    last_announced = -1
    while not partial.is_file() or partial.stat().st_size < expected_size:
        attempts += 1
        if attempts > 50:
            raise RuntimeError("safe-weight download did not complete after 50 resumable attempts")
        current = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={current}-"} if current else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 60),
            ) as response:
                response.raise_for_status()
                if current and response.status_code != 206:
                    current = 0
                    mode = "wb"
                else:
                    mode = "ab" if current else "wb"
                with partial.open(mode) as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        handle.flush()
                        current += len(block)
                        announced = current // (32 * 1024 * 1024)
                        if announced != last_announced:
                            last_announced = announced
                            print(f"[download] {current}/{expected_size} bytes")
        except requests.RequestException as exc:
            persisted = partial.stat().st_size if partial.is_file() else 0
            print(
                f"[retry] safe-weight transfer interrupted at {persisted}/{expected_size}: "
                f"{type(exc).__name__}"
            )
            continue
        actual_size = partial.stat().st_size if partial.is_file() else 0
        if actual_size > expected_size:
            raise RuntimeError(
                f"safe-weight size exceeds pinned value: {actual_size} > {expected_size}"
            )
    if partial.stat().st_size != expected_size:
        raise RuntimeError("safe-weight size does not match the frozen registry")
    actual_sha256 = _file_sha256(partial)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"safe-weight SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    partial.replace(target)
    print(f"[OK] safe weight verified: {expected_size} bytes, sha256={expected_sha256}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, measure, or validate the pinned local multilingual NLI candidate."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    fixture_path = args.fixture.resolve()
    qrels_path = args.qrels.resolve()
    registry_path = args.registry.resolve()
    report_path = args.report.resolve()
    fixture, registry = _load_inputs(fixture_path, qrels_path, registry_path)
    if args.prepare_model:
        if not args.allow_download:
            parser.error("--prepare-model requires explicit --allow-download")
        return _prepare_model(registry, args.cache_dir)
    if args.allow_download:
        parser.error("downloads are allowed only during --prepare-model, never during measurement")
    if args.validate_only:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        errors = validate_report(
            report,
            fixture_path,
            registry_path,
            fixture,
            registry,
        )
        if errors:
            print("[FAIL] " + "; ".join(errors))
            return 1
        print(
            "[OK] pinned NLI report: "
            f"accuracy={report['result']['metrics']['three_way_accuracy']:.3f}, "
            f"contradiction false-support="
            f"{report['result']['metrics']['contradiction_false_support_rate']:.3f}, "
            f"decision={report['decision']['status']}"
        )
        return 0

    candidate = registry["candidate"]
    provider = TransformersNliProvider(
        model_id=candidate["model_id"],
        revision=candidate["revision"],
        max_length=int(candidate["max_length"]),
        batch_size=args.batch_size,
        device=args.device,
        allow_download=False,
        cache_dir=args.cache_dir,
    )
    provider.prepare()
    evidence = _evidence_texts(fixture)
    cases = fixture.get("cases") or []
    pairs = [
        (evidence[str(case.get("evidence_id") or "")], str(case.get("claim") or ""))
        for case in cases
    ]
    predictions = provider.predict(pairs)
    for case, prediction in zip(cases, predictions):
        prediction["id"] = str(case.get("id") or "")
    result = evaluate_nli_predictions(fixture, predictions)
    runtime = provider.runtime_info()
    acceptance = evaluate_nli_acceptance(
        result,
        registry["acceptance"],
        truncated_pair_count=int(runtime["truncated_pair_count"]),
    )
    status = "eligible_for_shadow_design" if acceptance["passes_all"] else "rejected_for_shadow_design"
    report = {
        "schema_version": 1,
        "report_kind": "multilingual_nli_claim_baseline",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "fixture": {
            "path": fixture_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(fixture_path),
        },
        "registry": {
            "path": registry_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(registry_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": _package_version("transformers"),
            "safetensors": _package_version("safetensors"),
            "sentencepiece": _package_version("sentencepiece"),
        },
        "candidate": runtime,
        "result": result,
        "acceptance": acceptance,
        "decision": {
            "status": status,
            "product_enabled": False,
            "reason": (
                "All pre-registered gates passed; this permits only a shadow arbitration design."
                if acceptance["passes_all"]
                else "One or more pre-registered semantic safety gates failed; do not integrate."
            ),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[OK] measured NLI candidate once: accuracy={result['metrics']['three_way_accuracy']:.3f}, "
        f"contradiction false-support={result['metrics']['contradiction_false_support_rate']:.3f}, "
        f"decision={status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
