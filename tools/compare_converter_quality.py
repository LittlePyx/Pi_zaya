from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.converter.quality_compare import compare_markdown_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two converter markdown outputs")
    parser.add_argument("baseline", help="Baseline markdown path")
    parser.add_argument("candidate", help="Candidate markdown path")
    parser.add_argument("--out-md", default="", help="Optional markdown report output path")
    parser.add_argument("--out-json", default="", help="Optional JSON report output path")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    comparison = compare_markdown_files(
        base_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
    )
    report_md = str(comparison.get("report_markdown") or "")
    print(report_md, end="")

    if str(args.out_md or "").strip():
        Path(args.out_md).write_text(report_md, encoding="utf-8")
    if str(args.out_json or "").strip():
        Path(args.out_json).write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
