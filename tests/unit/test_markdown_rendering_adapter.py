from __future__ import annotations

from pathlib import Path


def test_markdown_rendering_adapter_preserves_math_normalizer():
    from kb import markdown_rendering
    from ui import chat_widgets

    sample = r"| Metric | Value |\n| --- | --- |\n| $a|b$ | ok |"

    assert markdown_rendering._normalize_math_markdown(sample) == chat_widgets._normalize_math_markdown(sample)


def test_markdown_rendering_adapter_preserves_plain_text_conversion():
    from kb import markdown_rendering
    from ui import chat_widgets

    sample = "**Result** with [[CITE:abcd1234:2]]"

    assert markdown_rendering._md_to_plain_text(sample) == chat_widgets._md_to_plain_text(sample)


def test_backend_modules_do_not_import_chat_widgets_directly():
    root = Path(__file__).resolve().parents[2]
    allowed = {
        root / "kb" / "markdown_rendering.py",
    }
    offenders: list[str] = []
    for dirname in ("api", "kb"):
        for path in (root / dirname).rglob("*.py"):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            if "ui.chat_widgets" in text:
                offenders.append(str(path.relative_to(root)).replace("\\", "/"))

    assert offenders == []
