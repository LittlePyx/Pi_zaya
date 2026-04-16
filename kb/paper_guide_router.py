"""Paper Guide router (compat import path).

The package-oriented implementation now lives in :mod:`kb.paper_guide.router`.
This flat-module path remains for compatibility with older imports.
"""

from __future__ import annotations

from .paper_guide import router as _router

PaperGuideExactSkillDeps = _router.PaperGuideExactSkillDeps
PaperGuideIntentModel = _router.PaperGuideIntentModel
PaperGuideSkillResult = _router.PaperGuideSkillResult


def __getattr__(name: str):  # pragma: no cover
    return getattr(_router, name)


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(dir(_router)))


__all__ = [name for name in dir(_router) if not name.startswith("__")]
