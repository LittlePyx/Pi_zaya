from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.converter.post_processing import postprocess_markdown

DEFAULT_MANIFEST_PATH = ROOT / "tools" / "manual_regression" / "manifests" / "paper_guide_implicit_sections_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "test_results" / "implicit_section_regression"
HEADING_RE = re.compile(r"^#{1,6}\s+\S")


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
        raise FileNotFoundError(f"Ambiguous implicit-section regression path pattern: {path_str} -> {candidates}")
    return resolved


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))


def _read_source_excerpt(path: Path, *, head_line_limit: int, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if head_line_limit > 0:
        text = "\n".join(text.splitlines()[:head_line_limit])
    if max_chars > 0:
        text = text[:max_chars]
    return text


def _extract_headings(text: str, *, limit: int = 12) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if HEADING_RE.match(stripped):
            headings.append(stripped)
            if limit > 0 and len(headings) >= limit:
                break
    return headings


def _heading_title(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line.strip())


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


def load_suite(manifest_path: str | Path | None = None) -> dict[str, Any]:
    suite = _load_json(manifest_path or DEFAULT_MANIFEST_PATH)
    defaults = dict(suite.get("defaults") or {})
    for case in suite.get("cases") or []:
        source_path = _resolve_existing_path(case["source_path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source markdown for implicit-section regression case: {source_path}")
        case["_source_abspath"] = str(source_path)
        case["_pdf_path"] = str(case.get("pdf_path") or "")
        merged = dict(defaults)
        merged.update(case.get("checks") or {})
        case["_checks"] = merged
    return suite


def evaluate_case(case: dict[str, Any], *, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    merged_defaults = dict(defaults or {})
    head_line_limit = int(case.get("head_line_limit") or merged_defaults.get("head_line_limit") or 80)
    max_chars = int(case.get("max_chars") or merged_defaults.get("max_chars") or 24000)
    source_path = _resolve_existing_path(case.get("_source_abspath") or case["source_path"])
    raw_excerpt = _read_source_excerpt(
        source_path,
        head_line_limit=head_line_limit,
        max_chars=max_chars,
    )
    processed_excerpt = postprocess_markdown(raw_excerpt)
    before_headings = _extract_headings(raw_excerpt)
    after_headings = _extract_headings(processed_excerpt, limit=0)
    after_titles = [_heading_title(line) for line in after_headings]
    checks = dict(merged_defaults)
    checks.update(case.get("_checks") or case.get("checks") or {})
    must_contain = list(checks.get("must_contain") or [])
    must_not_contain = list(checks.get("must_not_contain") or [])
    ordered = list(checks.get("ordered") or [])
    must_contain_titles = list(checks.get("must_contain_titles") or [])
    must_not_contain_titles = list(checks.get("must_not_contain_titles") or [])
    ordered_titles = list(checks.get("ordered_titles") or [])
    must_contain_text = list(checks.get("must_contain_text") or [])
    must_not_contain_text = list(checks.get("must_not_contain_text") or [])
    ordered_text = list(checks.get("ordered_text") or [])

    missing = [needle for needle in must_contain if needle not in after_headings]
    leaked = [needle for needle in must_not_contain if needle in after_headings]
    order_missing, ordered_positions = _find_sequence_in_order(after_headings, ordered)
    missing_titles = [needle for needle in must_contain_titles if needle not in after_titles]
    leaked_titles = [needle for needle in must_not_contain_titles if needle in after_titles]
    ordered_title_missing, ordered_title_positions = _find_sequence_in_order(after_titles, ordered_titles)
    missing_text = [needle for needle in must_contain_text if needle not in processed_excerpt]
    leaked_text = [needle for needle in must_not_contain_text if needle in processed_excerpt]
    ordered_text_missing: list[str] = []
    ordered_text_positions: dict[str, int] = {}
    cursor = -1
    for item in ordered_text:
        pos = processed_excerpt.find(item, cursor + 1)
        if pos < 0:
            ordered_text_missing.append(item)
            continue
        ordered_text_positions[item] = pos
        cursor = pos

    reasons: list[str] = []
    reasons.extend(f"missing:{item}" for item in missing)
    reasons.extend(f"forbidden_present:{item}" for item in leaked)
    reasons.extend(f"ordered_missing:{item}" for item in order_missing)
    reasons.extend(f"missing_title:{item}" for item in missing_titles)
    reasons.extend(f"forbidden_title_present:{item}" for item in leaked_titles)
    reasons.extend(f"ordered_title_missing:{item}" for item in ordered_title_missing)
    reasons.extend(f"missing_text:{item}" for item in missing_text)
    reasons.extend(f"forbidden_text_present:{item}" for item in leaked_text)
    reasons.extend(f"ordered_text_missing:{item}" for item in ordered_text_missing)

    return {
        "id": case["id"],
        "title": case.get("title") or case["id"],
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "source_path": str(source_path),
        "pdf_path": str(case.get("_pdf_path") or case.get("pdf_path") or ""),
        "head_line_limit": head_line_limit,
        "before_headings": before_headings[:12],
        "after_headings": after_headings[:12],
        "after_heading_titles": after_titles[:12],
        "required_headings": must_contain,
        "forbidden_headings": must_not_contain,
        "ordered_headings": ordered,
        "ordered_positions": ordered_positions,
        "required_heading_titles": must_contain_titles,
        "forbidden_heading_titles": must_not_contain_titles,
        "ordered_heading_titles": ordered_titles,
        "ordered_title_positions": ordered_title_positions,
        "required_text": must_contain_text,
        "forbidden_text": must_not_contain_text,
        "ordered_text": ordered_text,
        "ordered_text_positions": ordered_text_positions,
    }


def evaluate_suite(suite: dict[str, Any], *, case_id: str | None = None) -> dict[str, Any]:
    defaults = dict(suite.get("defaults") or {})
    selected_cases = list(suite.get("cases") or [])
    if case_id:
        selected_cases = [case for case in selected_cases if case.get("id") == case_id]
        if not selected_cases:
            raise KeyError(f"Unknown implicit-section regression case id: {case_id}")
    results = [evaluate_case(case, defaults=defaults) for case in selected_cases]
    pass_count = sum(1 for item in results if item["status"] == "PASS")
    fail_count = len(results) - pass_count
    return {
        "suite_id": suite.get("suite_id") or "implicit_section_regression",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "overall_status": "PASS" if fail_count == 0 else "FAIL",
        "results": results,
    }


def _build_report(suite: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        f"# {suite.get('suite_id') or 'implicit_section_regression'}",
        "",
        suite.get("description") or "",
        "",
        f"- Generated at: {summary['generated_at']}",
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
        lines.append(f"- Source: `{result['source_path']}`")
        if result.get("pdf_path"):
            lines.append(f"- PDF: `{result['pdf_path']}`")
        lines.append(f"- Before headings: {', '.join(result['before_headings']) or '(none)'}")
        lines.append(f"- After headings: {', '.join(result['after_headings']) or '(none)'}")
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
        description="Run converter regression checks for implicit Abstract and Introduction recovery.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH.relative_to(ROOT)),
        help="Manifest path relative to the repo root or an absolute path.",
    )
    parser.add_argument("--case-id", help="Optional single case id to run.")
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(ROOT)),
        help="Directory used to write summary.json and report.md.",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not write output files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    suite = load_suite(args.manifest)
    summary = evaluate_suite(suite, case_id=args.case_id)
    for result in summary["results"]:
        print(f"[{result['status']}] {result['id']}: {', '.join(result['after_headings']) or '(no headings)'}")
    print(
        f"overall={summary['overall_status']} "
        f"pass={summary['pass_count']} fail={summary['fail_count']} cases={summary['case_count']}"
    )
    if not args.no_write:
        out_dir = write_report(suite, summary, out_root=args.out_root)
        print(f"wrote={out_dir}")
    return 0 if summary["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
