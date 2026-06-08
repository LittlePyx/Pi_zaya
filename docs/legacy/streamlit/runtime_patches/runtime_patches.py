from __future__ import annotations

"""Compatibility facade for runtime UI patches.

Implementation is split into smaller modules under ``ui.runtime_patches_parts``
to keep this import path stable for the app while making the codebase easier
to maintain.
"""

from ui.runtime_patches_parts.chat_runtime import (
    _inject_auto_rerun_once,
    _inject_chat_dock_runtime,
    _remember_scroll_for_next_rerun,
    _restore_scroll_after_rerun_if_needed,
    _set_live_streaming_mode,
    _teardown_chat_dock_runtime,
)
from ui.runtime_patches_parts.copy_hooks import _inject_copy_js
from ui.runtime_patches_parts.runtime_ui import _inject_runtime_ui_fixes
from ui.runtime_patches_parts.theme import (
    _init_theme_css,
    _sync_theme_with_browser_preference,
)

__all__ = [
    "_init_theme_css",
    "_sync_theme_with_browser_preference",
    "_inject_copy_js",
    "_inject_runtime_ui_fixes",
    "_teardown_chat_dock_runtime",
    "_set_live_streaming_mode",
    "_remember_scroll_for_next_rerun",
    "_restore_scroll_after_rerun_if_needed",
    "_inject_chat_dock_runtime",
    "_inject_auto_rerun_once",
]
