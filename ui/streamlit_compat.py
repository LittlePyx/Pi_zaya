from __future__ import annotations

from importlib import import_module
from typing import Any


class _MissingStreamlit:
    _is_running_with_streamlit = False

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            "Streamlit is only available for the legacy UI. "
            "The product runtime is FastAPI + React; archived legacy notes live under docs/legacy/."
        )


def _get_script_run_ctx() -> Any:
    for module_name in (
        "streamlit.runtime.scriptrunner",
        "streamlit.runtime.scriptrunner.script_run_context",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ):
        try:
            module = import_module(module_name)
        except Exception:
            continue
        func = getattr(module, "get_script_run_ctx", None)
        if not callable(func):
            continue
        try:
            return func(suppress_warning=True)
        except TypeError:
            try:
                return func()
            except Exception:
                return None
        except Exception:
            return None
    return None


def _running_with_streamlit(real_st: Any) -> bool:
    if bool(getattr(real_st, "_is_running_with_streamlit", False)):
        return True
    return _get_script_run_ctx() is not None


try:
    import streamlit as _streamlit  # type: ignore[import-not-found]
except ModuleNotFoundError:
    st = _MissingStreamlit()
else:
    if _running_with_streamlit(_streamlit):
        st = _streamlit
        setattr(st, "_is_running_with_streamlit", True)
    else:
        st = _MissingStreamlit()
