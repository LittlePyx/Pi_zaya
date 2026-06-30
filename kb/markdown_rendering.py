from __future__ import annotations

"""
Backend-facing markdown rendering helpers.

The current implementations are shared with the retired Streamlit UI module.
Keep backend code depending on this module so those helpers can be moved here
without changing API or generation code again.
"""

from ui.chat_widgets import (
    _md_to_plain_text,
    _normalize_copy_citation_links,
    _normalize_math_markdown,
)

__all__ = [
    "_md_to_plain_text",
    "_normalize_copy_citation_links",
    "_normalize_math_markdown",
]
