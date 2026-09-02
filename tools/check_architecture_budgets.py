from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Ratchet, not an endorsement of these sizes. Existing oversized files may
# shrink without updating this table; any growth requires moving logic behind a
# smaller domain boundary instead of raising the limit.
LINE_BUDGETS = {
    "api/chat_render.py": 17_380,
    "api/reference_ui.py": 11_208,
    "kb/generation_answer_finalize_runtime.py": 10_721,
    "kb/task_runtime.py": 10_529,
    "api/routers/library.py": 9_156,
    "api/routers/references.py": 7_883,
    "web/src/styles/index.css": 6_591,
    "kb/converter/quality_repair.py": 6_362,
    "kb/citation_plan.py": 6_092,
    "kb/paper_guide_provenance.py": 5_994,
    "kb/retrieval_engine.py": 5_099,
    "kb/reference_index.py": 4_928,
    "web/src/stores/chatStore.ts": 3_149,
    "web/src/components/chat/MessageList.tsx": 3_007,
    "web/src/pages/ResearchNotesPage.tsx": 3_003,
    "web/src/styles/citeShelf.css": 2_713,
    "web/src/components/chat/MarkdownRenderer.tsx": 2_643,
    "web/src/styles/library.css": 2_479,
    "web/src/components/chat/citationState.ts": 2_455,
    "web/src/components/chat/CiteShelf.tsx": 2_424,
    "web/src/api/chat.ts": 2_199,
}


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def main() -> int:
    failures: list[str] = []
    for relative_path, maximum in LINE_BUDGETS.items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing ratcheted file: {relative_path}")
            continue
        actual = _line_count(path)
        if actual > maximum:
            failures.append(f"{relative_path}: {actual} lines exceeds frozen budget {maximum}")
        else:
            print(f"[architecture-budget] {relative_path}: {actual}/{maximum}")
    if failures:
        for failure in failures:
            print(f"[architecture-budget] FAIL {failure}")
        return 1
    print("[architecture-budget] PASS oversized files did not grow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
