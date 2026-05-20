from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

KEY_USER_VISIBLE_FILES = (
    "api/reference_ui.py",
    "kb/retrieval_engine.py",
    "ui/refs_renderer.py",
    "web/src/i18n/zh.ts",
    "web/src/components/refs/RefsPanel.tsx",
    "web/tests/e2e/paper-guide-locate-flow.spec.ts",
    "web/tests/e2e/chat-refs-perf.spec.ts",
    "tests/unit/test_reference_ui_score_calibration.py",
)

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
)


def test_reference_ui_user_visible_files_do_not_contain_mojibake_or_replacement_chars():
    failures = []
    for rel_path in KEY_USER_VISIBLE_FILES:
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            for fragment in FORBIDDEN_FRAGMENTS:
                if fragment in line:
                    failures.append(f"{rel_path}:{line_no}: contains {fragment.encode('unicode_escape').decode('ascii')}")

    assert not failures, "\n".join(failures)
