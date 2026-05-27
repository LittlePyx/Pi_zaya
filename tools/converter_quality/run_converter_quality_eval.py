from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.converter.quality_acceptance import evaluate_conversion_quality, load_quality_manifest


DEFAULT_MANIFEST = Path("tools/manual_regression/manifests/converter_markdown_quality_v1.json")
DEFAULT_OUT_DIR = Path("test_results/converter_quality_eval")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _selected_cases(manifest: dict[str, Any], case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    selected = [item for item in _as_list(manifest.get("cases")) if isinstance(item, dict)]
    wanted = {str(item) for item in list(case_ids or []) if str(item or "").strip()}
    if wanted:
        selected = [item for item in selected if str(item.get("id") or "") in wanted]
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("id") or "").strip()
    md_path = Path(str(case.get("_md_abspath") or case.get("md_path") or ""))
    row: dict[str, Any] = {
        "id": case_id,
        "title": str(case.get("title") or case_id),
        "md_path": str(md_path),
    }
    if not md_path.exists():
        row.update(
            {
                "ok": False,
                "status": "MISSING",
                "metrics": {},
                "failures": [f"missing_markdown:{md_path}"],
            }
        )
        return row

    quality = evaluate_conversion_quality(md_path, checks=case.get("checks") or {})
    row.update(
        {
            "ok": bool(quality.get("ok")),
            "status": "PASS" if bool(quality.get("ok")) else "FAIL",
            "metrics": quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {},
            "failures": _as_list(quality.get("failures")),
            "quality": quality,
        }
    )
    return row


def evaluate_manifest(
    manifest: dict[str, Any],
    *,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    rows = [evaluate_case(case) for case in _selected_cases(manifest, case_ids=case_ids)]
    passed = [row for row in rows if row.get("status") == "PASS"]
    failed = [row for row in rows if row.get("status") == "FAIL"]
    missing = [row for row in rows if row.get("status") == "MISSING"]
    return {
        "suite_id": manifest.get("suite_id") or "converter_markdown_quality_v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(manifest.get("_manifest_path") or DEFAULT_MANIFEST),
        "case_count": len(rows),
        "pass_count": len(passed),
        "fail_count": len(failed),
        "missing_count": len(missing),
        "overall_status": "PASS" if not failed and not missing else "FAIL",
        "results": rows,
    }


def _metric(row: dict[str, Any], key: str) -> Any:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return metrics.get(key, 0)


def build_report(summary: dict[str, Any], *, output_dir: Path) -> str:
    rows = [item for item in _as_list(summary.get("results")) if isinstance(item, dict)]
    failed = [row for row in rows if row.get("status") != "PASS"]
    lines = [
        "# Converter Markdown Quality Eval",
        "",
        f"- Time: {summary.get('generated_at')}",
        f"- Suite: `{summary.get('suite_id')}`",
        f"- Manifest: `{summary.get('manifest')}`",
        f"- Output: `{output_dir}`",
        f"- Cases: {summary.get('case_count')}",
        f"- Passed: {summary.get('pass_count')}",
        f"- Failed: {summary.get('fail_count')}",
        f"- Missing: {summary.get('missing_count')}",
        f"- Overall: {summary.get('overall_status')}",
        "",
        "## Failures",
        "",
    ]
    if not failed:
        lines.append("- None")
    for row in failed:
        failures = ", ".join(str(item) for item in _as_list(row.get("failures"))) or "unknown"
        lines.append(f"- `{row.get('id')}` [{row.get('status')}]: {failures}")

    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| Case | Status | Chars | Headings | Images | Missing Images | Captions | Tables | Math | Refs | Body Cites | Errors | Warnings |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('id')} | "
            f"{row.get('status')} | "
            f"{_metric(row, 'chars')} | "
            f"{_metric(row, 'heading_count')} | "
            f"{_metric(row, 'image_count')} | "
            f"{_metric(row, 'missing_image_count')} | "
            f"{_metric(row, 'caption_count')} | "
            f"{_metric(row, 'table_block_count')} | "
            f"{_metric(row, 'display_math_block_count')} | "
            f"{_metric(row, 'extracted_reference_count')} | "
            f"{_metric(row, 'body_citation_expanded_index_count')} | "
            f"{_metric(row, 'analyzer_error_count')} | "
            f"{_metric(row, 'analyzer_warning_count')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate converted Markdown quality against structural research-paper acceptance gates.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Converter quality manifest JSON.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory root.")
    parser.add_argument("--case-id", action="append", default=[], help="Run one or more case ids.")
    parser.add_argument("--dry-run", action="store_true", help="Load manifest and print planned cases without reading markdown.")
    parser.add_argument("--fail-on-quality", action="store_true", help="Exit 1 when any quality check fails.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = Path(str(args.manifest or DEFAULT_MANIFEST))
    manifest = load_quality_manifest(manifest_path, repo_root=REPO_ROOT)
    selected = _selected_cases(manifest, case_ids=args.case_id)
    if not selected:
        print("[ERROR] no converter quality cases selected", file=sys.stderr)
        return 2

    if bool(args.dry_run):
        existing = sum(1 for case in selected if bool(case.get("_exists")))
        print(f"[OK] manifest: {manifest.get('_manifest_path')}")
        print(f"[OK] suite: {manifest.get('suite_id')}")
        print(f"[OK] cases: {len(selected)}")
        print(f"[OK] existing markdown: {existing}/{len(selected)}")
        for idx, case in enumerate(selected, start=1):
            status = "exists" if bool(case.get("_exists")) else "missing"
            print(f"{idx:02d}. {case.get('id')} [{status}] {case.get('md_path')}")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / Path(str(args.out_dir or DEFAULT_OUT_DIR)) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_manifest(manifest, case_ids=args.case_id)
    _write_json(output_dir / "summary.json", {k: v for k, v in summary.items() if k != "results"})
    _write_json(output_dir / "raw_results.json", summary.get("results") or [])
    (output_dir / "report.md").write_text(build_report(summary, output_dir=output_dir), encoding="utf-8")

    for row in _as_list(summary.get("results")):
        if not isinstance(row, dict):
            continue
        suffix = ""
        if row.get("failures"):
            suffix = " failures=" + ", ".join(str(item) for item in _as_list(row.get("failures"))[:4])
        print(f"[{row.get('status')}] {row.get('id')}{suffix}")
    print(
        f"overall={summary['overall_status']} pass={summary['pass_count']} "
        f"fail={summary['fail_count']} missing={summary['missing_count']} cases={summary['case_count']}"
    )
    print(f"wrote={output_dir}")
    if bool(args.fail_on_quality) and summary["overall_status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
