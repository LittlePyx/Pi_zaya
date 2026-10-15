from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.local_semantic_acceptance import (
    build_local_semantic_acceptance_report,
    load_local_observation_events,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an aggregate-only local semantic acceptance report."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="Local observation JSONL; defaults to KB_LOCAL_OBSERVABILITY_PATH.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args()
    source = args.source
    if source is None:
        configured = str(os.environ.get("KB_LOCAL_OBSERVABILITY_PATH") or "").strip()
        source = Path(configured) if configured else None
    if source is None or not source.expanduser().resolve().is_file():
        print("No readable local observation JSONL was provided.", file=sys.stderr)
        return 2

    events, line_count = load_local_observation_events(source)
    report = build_local_semantic_acceptance_report(
        events,
        input_line_count=line_count,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Aggregate local semantic acceptance report written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
