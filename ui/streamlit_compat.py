from __future__ import annotations

from typing import Any


class _MissingStreamlit:
    _is_running_with_streamlit = False
    session_state: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            "Streamlit is only available for the legacy UI. "
            "Install requirements-legacy.txt before using Streamlit modules."
        )


try:
    import streamlit as st  # type: ignore[import-not-found]
except ModuleNotFoundError:
    st = _MissingStreamlit()
