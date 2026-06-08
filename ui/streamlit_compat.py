from __future__ import annotations

from typing import Any


class _MissingStreamlit:
    _is_running_with_streamlit = False
    session_state: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            "Streamlit is only available for the legacy UI. "
            "The product runtime is FastAPI + React; archived legacy notes live under docs/legacy/."
        )


try:
    import streamlit as st  # type: ignore[import-not-found]
except ModuleNotFoundError:
    st = _MissingStreamlit()
