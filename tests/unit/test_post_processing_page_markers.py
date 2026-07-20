from __future__ import annotations

from kb.converter.post_processing import postprocess_markdown


def test_postprocess_keeps_page_marker_and_following_image_as_standalone_blocks():
    source = "\n".join(
        [
            "# Demo",
            "",
            "A prose paragraph immediately before the next physical page.",
            "<!-- kb_page: 10 -->",
            "![Figure 5](./assets/page_10_fig_1.png)",
            "",
            "**Figure 5.** Example result.",
        ]
    )

    out = postprocess_markdown(source)

    assert (
        "A prose paragraph immediately before the next physical page.\n\n"
        "<!-- kb_page: 10 -->\n\n"
        "![Figure 5](./assets/page_10_fig_1.png)"
    ) in out
    assert postprocess_markdown(out) == out
