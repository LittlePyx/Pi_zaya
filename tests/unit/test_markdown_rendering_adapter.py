from __future__ import annotations

from pathlib import Path


def test_markdown_rendering_adapter_preserves_math_normalizer():
    from kb import markdown_rendering
    from ui import chat_widgets

    sample = r"| Metric | Value |\n| --- | --- |\n| $a|b$ | ok |"

    assert markdown_rendering._normalize_math_markdown(sample) == chat_widgets._normalize_math_markdown(sample)


def test_markdown_rendering_adapter_removes_empty_display_math_blocks_only():
    from kb import markdown_rendering

    sample = (
        "Before\n"
        "$$\n   \n$$\n"
        "After\n\n"
        "$$\nE = mc^2\n$$\n"
        r"Escaped citation \[24\]." "\n"
        "```markdown\n$$\n$$\n```\n"
    )

    assert markdown_rendering._normalize_math_markdown(sample) == (
        "Before\n"
        "\n"
        "After\n\n"
        "$$\nE = mc^2\n$$\n"
        r"Escaped citation \[24\]." "\n"
        "```markdown\n$$\n$$\n```\n"
    )


def test_markdown_rendering_adapter_preserves_plain_text_conversion():
    from kb import markdown_rendering
    from ui import chat_widgets

    sample = "**Result** with [[CITE:abcd1234:2]]"

    assert markdown_rendering._md_to_plain_text(sample) == chat_widgets._md_to_plain_text(sample)


def test_signed_binary_vectors_do_not_collide_with_numeric_citations():
    from kb.markdown_rendering import normalize_signed_binary_vectors

    sample = "BPSK encodes [1, -1] and [-1，1], with source support [2]."

    assert normalize_signed_binary_vectors(sample) == (
        "BPSK encodes (+1, -1) and (-1, +1), with source support [2]."
    )


def test_signed_binary_vector_normalization_skips_code_and_citation_lists():
    from kb.markdown_rendering import normalize_signed_binary_vectors

    sample = (
        "Prose [1, -1], citations [1, 2], and inline `[1, -1]`.\n"
        "```text\n[1, -1]\n```\n"
    )

    assert normalize_signed_binary_vectors(sample) == (
        "Prose (+1, -1), citations [1, 2], and inline `[1, -1]`.\n"
        "```text\n[1, -1]\n```\n"
    )


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
