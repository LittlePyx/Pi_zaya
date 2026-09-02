from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import project_research_qa_qrels, sha256_file
from kb.store import load_all_chunks


DEFAULT_SOURCE_FIXTURE = REPO_ROOT / "docs" / "research_qa_unseen_corpus_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"


def build_fixture(source_fixture: Path, db_root: Path) -> dict:
    source_payload = json.loads(source_fixture.read_text(encoding="utf-8"))
    chunks = load_all_chunks(db_root)
    if not chunks:
        raise ValueError(f"no ready chunks found under {db_root}")
    try:
        relative_source = source_fixture.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        relative_source = source_fixture.name
    return project_research_qa_qrels(
        source_payload,
        chunks,
        source_fixture_path=relative_source,
        source_fixture_sha256=sha256_file(source_fixture),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze Research QA source/page/evidence contracts as chunk retrieval qrels."
    )
    parser.add_argument("--source-fixture", type=Path, default=DEFAULT_SOURCE_FIXTURE)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed output differs; do not write the fixture.",
    )
    args = parser.parse_args()

    payload = build_fixture(args.source_fixture.resolve(), args.db_root.resolve())
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"[FAIL] projected qrels differ from {output}")
            return 1
        print(
            f"[OK] qrels projection is current: {len(payload['cases'])} cases, "
            f"{sum(len(case['contracts']) for case in payload['cases'])} contracts, "
            f"{sum(len(case['labels']) for case in payload['cases'])} labels"
        )
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        f"[OK] wrote {output}: {len(payload['cases'])} cases, "
        f"{sum(len(case['contracts']) for case in payload['cases'])} contracts, "
        f"{sum(len(case['labels']) for case in payload['cases'])} labels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
