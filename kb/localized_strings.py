from __future__ import annotations

"""
Backend-facing localized string table.

The legacy UI module still owns the current dictionary. Backend code imports
through this module so UI-string storage can move without touching runtime code.
"""

from ui.strings import S

__all__ = ["S"]
