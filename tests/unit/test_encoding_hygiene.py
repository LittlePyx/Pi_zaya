from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOTS = (
    "api",
    "kb",
    "web/src",
)

SOURCE_EXTS = {".py", ".ts", ".tsx"}

EXTRA_SOURCE_FILES = (
    "tests/unit/test_task_runtime_answer_contract.py",
    "tests/unit/test_chat_render_reference_notes.py",
)

FORBIDDEN_FRAGMENTS = (
    "\ufffd",
    "????????",
    "\u95c1\u7a3f\u6d5a",
    "\u95c1\u641e\u5133",
    "\u95b8\u5fe3\u5259",
    "\u95b8\u30e7\u6597",
    "\u934b\u6ec4",
    "\u59dd\uff45\u6e6a\u9359\u6828\u79f7",
    "\u95b8\u5b2b\u7c8d",
    "\u6769\u6b11\u7612",
    "\u95c7\u20ac\u7455",
    "\u6769\u6b11\u4edc",
    "\u7481\u5757\u68f6",
    "\u8292\u9227\ue0e0?",
    "\u9225?",
    "浣犳槸",
    "鍙傝€",
    "鏂囩尞",
    "闂",
    "鎻愬嚭",
    "缁撴灉",
    "銆俓n",
    "锛?",
    "鏃堕棿绾",
    "绾跨",
    "鍚姩",
    "澶辫触",
    "鏉ヨ嚜",
)

# These files intentionally mention mojibake/replacement markers while detecting
# and repairing corrupted generated text.
ALLOWED_FILE_FRAGMENTS = {
    "api/chat_render.py": {"鍙傝€"},
    "kb/converter/llm_general_cleanup.py": {"\ufffd"},
    "kb/converter/llm_math_cleanup.py": {"\ufffd"},
    "kb/converter/post_processing.py": {"\ufffd"},
}


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for rel_root in SOURCE_ROOTS:
        root = ROOT / rel_root
        for path in root.rglob("*"):
            if path.suffix.lower() in SOURCE_EXTS:
                files.append(path)
    for rel_path in EXTRA_SOURCE_FILES:
        path = ROOT / rel_path
        if path.exists():
            files.append(path)
    return sorted(files)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_allowed(rel_path: str, fragment: str) -> bool:
    return fragment in ALLOWED_FILE_FRAGMENTS.get(rel_path, set())


def test_user_visible_source_files_do_not_contain_mojibake_or_replacement_chars():
    failures = []
    for path in _iter_source_files():
        rel_path = _rel(path)
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            for fragment in FORBIDDEN_FRAGMENTS:
                if fragment in line and not _is_allowed(rel_path, fragment):
                    escaped = fragment.encode("unicode_escape").decode("ascii")
                    failures.append(f"{rel_path}:{line_no}: contains {escaped}")

    assert not failures, "\n".join(failures)
