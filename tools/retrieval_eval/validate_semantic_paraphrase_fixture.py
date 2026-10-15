from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file, validate_qrels_fixture
from kb.retrieval_semantic_eval_fixture import build_semantic_paraphrase_qrels


DEFAULT_REGISTRY = REPO_ROOT / "docs" / "retrieval_semantic_paraphrase_cases_v1.json"
DEFAULT_BASE = REPO_ROOT / "docs" / "retrieval_eval_unseen_v2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the pre-registered semantic/paraphrase retrieval slice."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--base-fixture", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    base_path = args.base_fixture.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    payload = build_semantic_paraphrase_qrels(
        registry,
        base,
        registry_path=registry_path.relative_to(REPO_ROOT).as_posix(),
        registry_sha256=sha256_file(registry_path),
        base_fixture_path=base_path.relative_to(REPO_ROOT).as_posix(),
        base_fixture_sha256=sha256_file(base_path),
    )
    errors = validate_qrels_fixture(payload, repo_root=REPO_ROOT)
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    contract = payload["slice_contract"]
    print(
        f"[OK] semantic/paraphrase fixture: {len(payload['cases'])} cases, "
        f"locales={contract['locale_counts']}, "
        f"cross_lingual={contract['transformation_counts'].get('cross_lingual', 0)}, "
        f"corpus={payload['corpus']['chunk_count']} chunks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
