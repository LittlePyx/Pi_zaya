from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOTS = (
    "api",
    "kb",
    "web/src",
)

SOURCE_EXTS = {".py", ".ts", ".tsx"}

FORBIDDEN_FRAGMENTS = (
    "\ufffd",
    "????????",
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
