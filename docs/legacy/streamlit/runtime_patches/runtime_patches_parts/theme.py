from __future__ import annotations

"""Thin theme facade.

The historical implementation grew very large and is now moved to
``theme_legacy.py``. Keep this module small and stable as the import surface
for the rest of the app.
"""

from ui.runtime_patches_parts.theme_legacy import (
    _init_theme_css as _init_theme_css_legacy,
    _sync_theme_with_browser_preference as _sync_theme_with_browser_preference_legacy,
)


def _init_theme_css(theme_mode: str = "dark") -> None:
    _init_theme_css_legacy(theme_mode)


def _sync_theme_with_browser_preference() -> None:
    _sync_theme_with_browser_preference_legacy()


__all__ = [
    "_init_theme_css",
    "_sync_theme_with_browser_preference",
]

