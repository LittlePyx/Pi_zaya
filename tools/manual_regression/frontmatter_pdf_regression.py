from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.config import load_settings
from kb.converter.config import ConvertConfig, LlmConfig
from kb.converter.pipeline import PDFConverter

DEFAULT_MANIFEST_PATH = ROOT / "tools" / "manual_regression" / "manifests" / "converter_frontmatter_pdf_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "test_results" / "frontmatter_pdf_regression"


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_existing_path(path_str: str | Path) -> Path:
    resolved = _resolve_repo_path(path_str)
    if resolved.exists():
        return resolved
    path = Path(path_str)
    if path.is_absolute():
        return resolved
    pattern = str(path).replace("\\", "/")
    candidates = sorted(ROOT.glob(pattern))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(f"Ambiguous frontmatter regression path pattern: {path_str} -> {candidates}")
    return resolved


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))


def _extract_headings(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("#")]


def _find_sequence_in_order(sequence: list[str], items: list[str]) -> tuple[list[str], dict[str, int]]:
    positions: dict[str, int] = {}
    missing: list[str] = []
    cursor = -1
    for item in items:
        pos = -1
        for idx in range(cursor + 1, len(sequence)):
            if sequence[idx] == item:
                pos = idx
                break
        if pos < 0:
            missing.append(item)
            continue
        positions[item] = pos
        cursor = pos
    return missing, positions


def _build_llm_config_if_available(*, requested_speed_mode: str) -> tuple[LlmConfig | None, str]:
    speed_mode = str(requested_speed_mode or "auto").strip().lower()
    if speed_mode in {"", "auto"}:
        settings = load_settings()
        if settings.api_key:
            return (
                LlmConfig(
                    api_key=str(settings.api_key),
                    base_url=str(settings.base_url),
                    model=str(settings.model),
                    temperature=0.0,
                    max_tokens=4096,
                    request_sleep_s=0.0,
                    timeout_s=float(settings.timeout_s),
                    max_retries=int(settings.max_retries),
                ),
                "normal",
            )
        return None, "no_llm"

    if speed_mode == "no_llm":
        return None, "no_llm"

    settings = load_settings()
    if not settings.api_key:
        return None, "no_llm"
    return (
        LlmConfig(
            api_key=str(settings.api_key),
            base_url=str(settings.base_url),
            model=str(settings.model),
            temperature=0.0,
            max_tokens=4096,
            request_sleep_s=0.0,
            timeout_s=float(settings.timeout_s),
            max_retries=int(settings.max_retries),
        ),
        speed_mode,
    )


def load_suite(manifest_path: str | Path | None = None) -> dict[str, Any]:
    suite = _load_json(manifest_path or DEFAULT_MANIFEST_PATH)
    defaults = dict(suite.get("defaults") or {})
    for case in suite.get("cases") or []:
        pdf_path = _resolve_existing_path(case["pdf_path"])
        case["_pdf_abspath"] = str(pdf_path)
        merged = dict(defaults)
        merged.update(case.get("checks") or {})
        case["_checks"] = merged
    return suite


def evaluate_output_text(
    output_text: str,
    *,
    case: dict[str, Any],
    requested_speed_mode: str,
    actual_speed_mode: str,
    output_path: str,
) -> dict[str, Any]:
    headings = _extract_headings(output_text)
    checks = dict(case.get("_checks") or case.get("checks") or {})
    must_contain = list(checks.get("must_contain") or [])
    must_not_contain = list(checks.get("must_not_contain") or [])
    ordered = list(checks.get("ordered") or [])
    must_contain_text = list(checks.get("must_contain_text") or [])
    must_not_contain_text = list(checks.get("must_not_contain_text") or [])
    ordered_text = list(checks.get("ordered_text") or [])
    must_start_with = list(checks.get("must_start_with") or [])
    must_not_start_with = list(checks.get("must_not_start_with") or [])

    missing = [item for item in must_contain if item not in headings]
    leaked = [item for item in must_not_contain if item in headings]
    order_missing, ordered_positions = _find_sequence_in_order(headings, ordered)
    missing_text = [item for item in must_contain_text if item not in output_text]
    leaked_text = [item for item in must_not_contain_text if item in output_text]
    ordered_text_missing: list[str] = []
    ordered_text_positions: dict[str, int] = {}
    cursor = -1
    for item in ordered_text:
        pos = output_text.find(item, cursor + 1)
        if pos < 0:
            ordered_text_missing.append(item)
            continue
        ordered_text_positions[item] = pos
        cursor = pos

    stripped = output_text.lstrip()
    missing_prefix = [item for item in must_start_with if not stripped.startswith(item)]
    leaked_prefix = [item for item in must_not_start_with if stripped.startswith(item)]

    reasons: list[str] = []
    reasons.extend(f"missing:{item}" for item in missing)
    reasons.extend(f"forbidden_present:{item}" for item in leaked)
    reasons.extend(f"ordered_missing:{item}" for item in order_missing)
    reasons.extend(f"missing_text:{item}" for item in missing_text)
    reasons.extend(f"forbidden_text_present:{item}" for item in leaked_text)
    reasons.extend(f"ordered_text_missing:{item}" for item in ordered_text_missing)
    reasons.extend(f"missing_prefix:{item}" for item in missing_prefix)
    reasons.extend(f"forbidden_prefix_present:{item}" for item in leaked_prefix)

    return {
        "id": case["id"],
        "title": case.get("title") or case["id"],
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "requested_speed_mode": requested_speed_mode,
        "actual_speed_mode": actual_speed_mode,
        "output_path": output_path,
        "after_headings": headings[:16],
        "ordered_positions": ordered_positions,
        "ordered_text_positions": ordered_text_positions,
    }


def run_case(
    case: dict[str, Any],
    *,
    out_root: str | Path | None = None,
    requested_speed_mode: str = "auto",
    show_converter_output: bool = False,
) -> dict[str, Any]:
    pdf_path = _resolve_existing_path(case.get("_pdf_abspath") or case["pdf_path"])
    page_limit = int(case.get("page_limit") or 2)
    llm_cfg, actual_speed_mode = _build_llm_config_if_available(requested_speed_mode=requested_speed_mode)

    root = _resolve_repo_path(out_root or DEFAULT_OUTPUT_ROOT)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = root / stamp / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    cfg = ConvertConfig(
        pdf_path=pdf_path,
        out_dir=case_dir,
        translate_zh=False,
        start_page=0,
        end_page=page_limit,
        skip_existing=False,
        keep_debug=False,
        llm=llm_cfg,
        llm_workers=1,
        workers=1,
        speed_mode=actual_speed_mode,
    )

    convert_error = ""
    try:
        converter = PDFConverter(cfg)
        converter.convert(str(pdf_path), str(case_dir))
    except Exception as exc:
        convert_error = f"{type(exc).__name__}: {exc}"

    output_path = case_dir / "output.md"
    output_text = ""
    if output_path.exists():
        output_text = output_path.read_text(encoding="utf-8", errors="replace")

    result = evaluate_output_text(
        output_text,
        case=case,
        requested_speed_mode=requested_speed_mode,
        actual_speed_mode=actual_speed_mode,
        output_path=str(output_path),
    )
    result["pdf_path"] = str(pdf_path)
    result["case_dir"] = str(case_dir)
    result["page_limit"] = page_limit
    if convert_error:
        result["status"] = "FAIL"
        result["reasons"] = [f"convert_error:{convert_error}"] + list(result.get("reasons") or [])
    if show_converter_output and output_text:
        print(f"[frontmatter_pdf_regression] {case['id']} output_path={output_path}", flush=True)
    return result


def evaluate_suite(
    suite: dict[str, Any],
    *,
    case_id: str | None = None,
    out_root: str | Path | None = None,
    requested_speed_mode: str = "auto",
    show_converter_output: bool = False,
) -> dict[str, Any]:
    selected_cases = list(suite.get("cases") or [])
    if case_id:
        selected_cases = [case for case in selected_cases if case.get("id") == case_id]
        if not selected_cases:
            raise KeyError(f"Unknown frontmatter PDF regression case id: {case_id}")
    results = [
        run_case(
            case,
            out_root=out_root,
            requested_speed_mode=requested_speed_mode,
            show_converter_output=show_converter_output,
        )
        for case in selected_cases
    ]
    pass_count = sum(1 for item in results if item["status"] == "PASS")
    fail_count = len(results) - pass_count
    return {
        "suite_id": suite.get("suite_id") or "converter_frontmatter_pdf_v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_speed_mode": requested_speed_mode,
        "case_count": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "overall_status": "PASS" if fail_count == 0 else "FAIL",
        "results": results,
    }


def _build_report(suite: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['suite_id']}",
        "",
        suite.get("description") or "",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Requested speed mode: {summary['requested_speed_mode']}",
        f"- Cases: {summary['case_count']}",
        f"- Passed: {summary['pass_count']}",
        f"- Failed: {summary['fail_count']}",
        f"- Overall: {summary['overall_status']}",
        "",
    ]
    for result in summary["results"]:
        lines.append(f"## {result['id']} - {result['status']}")
        lines.append("")
        lines.append(f"- Title: {result['title']}")
        lines.append(f"- PDF: `{result['pdf_path']}`")
        lines.append(f"- Actual speed mode: `{result['actual_speed_mode']}`")
        lines.append(f"- Page limit: `{result['page_limit']}`")
        lines.append(f"- Output: `{result['output_path']}`")
        lines.append(f"- Headings: {', '.join(result.get('after_headings') or []) or '(none)'}")
        if result["reasons"]:
            lines.append(f"- Reasons: {', '.join(result['reasons'])}")
        else:
            lines.append("- Reasons: none")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_report(
    suite: dict[str, Any],
    summary: dict[str, Any],
    *,
    out_root: str | Path | None = None,
) -> Path:
    root = _resolve_repo_path(out_root or DEFAULT_OUTPUT_ROOT)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_build_report(suite, summary), encoding="utf-8")
    return out_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-PDF converter regression checks for first-page frontmatter recovery.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH.relative_to(ROOT)),
        help="Path to regression manifest JSON",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(ROOT)),
        help="Output root for generated reports and case dirs",
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="Optional single case id to run",
    )
    parser.add_argument(
        "--speed-mode",
        default="auto",
        choices=["auto", "no_llm", "normal", "ultra_fast"],
        help="Converter speed mode. auto prefers normal if LLM credentials are available.",
    )
    parser.add_argument(
        "--show-converter-output",
        action="store_true",
        help="Print per-case converter output path while running.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    suite = load_suite(args.manifest)
    summary = evaluate_suite(
        suite,
        case_id=(args.case_id or None),
        out_root=args.out_root,
        requested_speed_mode=args.speed_mode,
        show_converter_output=bool(args.show_converter_output),
    )
    out_dir = write_report(suite, summary, out_root=args.out_root)
    for result in summary["results"]:
        prefix = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
        suffix = f" headings={', '.join(result.get('after_headings') or [])[:240]}"
        if result["reasons"]:
            suffix += f" reasons={', '.join(result['reasons'])}"
        print(f"{prefix} {result['id']}: speed={result['actual_speed_mode']}{suffix}")
    print(
        f"overall={summary['overall_status']} pass={summary['pass_count']} fail={summary['fail_count']} cases={summary['case_count']}"
    )
    print(f"wrote={out_dir}")
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
