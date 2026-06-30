from __future__ import annotations

from pathlib import Path


def test_localized_strings_adapter_preserves_table():
    from kb.localized_strings import S as backend_strings
    from ui.strings import S as legacy_strings

    assert backend_strings is legacy_strings
    assert "llm_fail" in backend_strings


def test_backend_modules_do_not_import_ui_strings_directly():
    root = Path(__file__).resolve().parents[2]
    allowed = {root / "kb" / "localized_strings.py"}
    offenders: list[str] = []
    for dirname in ("api", "kb"):
        for path in (root / dirname).rglob("*.py"):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            if "ui.strings" in text:
                offenders.append(str(path.relative_to(root)).replace("\\", "/"))

    assert offenders == []
