from __future__ import annotations

from kb.converter.post_processing import postprocess_markdown


def test_inline_osa_footer_is_removed_without_dropping_body_text() -> None:
    source = (
        "In the past decade there have been a number of "
        "#85145 - $15.00 USD Received 10 Jul 2007; published 11 Oct 2007 "
        "(C) 2007 OSA 17 October 2007 / Vol. 15 / OPTICS EXPRESS 14014"
    )

    repaired = postprocess_markdown(source)

    assert repaired == "In the past decade there have been a number of"
    assert "USD" not in repaired


def test_standalone_osa_footer_and_running_header_are_removed() -> None:
    source = "\n".join(
        [
            "Body before.",
            r"#85145 - \$15.00 USD (C) 2007 OSA",
            "## 17 October 2007 / Vol. 15, No. 21 / OPTICS EXPRESS 14021",
            "Body after.",
        ]
    )

    repaired = postprocess_markdown(source)

    assert "Body before." in repaired
    assert "Body after." in repaired
    assert "85145" not in repaired
    assert "OPTICS EXPRESS" not in repaired
