from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import kb.product_query_policy as query_policy


REPO_ROOT = Path(__file__).resolve().parents[2]
_MOVED_PRIVATE_NAMES = {
    "_deterministic_query_variants",
    "_expand_query_via_llm",
    "_merge_expanded_results",
    "_search_hits_with_fallback",
    "_translate_query_for_search",
}


def test_product_query_policy_has_only_public_moved_entrypoints() -> None:
    assert callable(query_policy.deterministic_query_variants)
    assert callable(query_policy.expand_query_via_llm)
    assert callable(query_policy.merge_expanded_results)
    assert callable(query_policy.search_hits_with_fallback)
    assert callable(query_policy.translate_query_for_search)
    assert not (_MOVED_PRIVATE_NAMES & set(vars(query_policy)))


def test_product_query_adapter_import_is_independent_of_engine_and_llm() -> None:
    script = (
        "import sys; "
        "import tools.retrieval_eval.product_query_sparse_retriever; "
        "blocked={'kb.retrieval_engine', 'kb.llm'} & set(sys.modules); "
        "raise SystemExit(','.join(sorted(blocked)) if blocked else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_production_and_evaluation_code_do_not_import_moved_engine_names() -> None:
    violations: list[str] = []
    for root_name in ("api", "kb", "tools"):
        for path in sorted((REPO_ROOT / root_name).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module != "kb.retrieval_engine":
                    continue
                moved = sorted(alias.name for alias in node.names if alias.name in _MOVED_PRIVATE_NAMES)
                if moved:
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{','.join(moved)}")
    assert violations == []
