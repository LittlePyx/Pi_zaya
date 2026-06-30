from __future__ import annotations

import logging

import pytest

from ui import streamlit_compat


def _skip_if_real_streamlit() -> None:
    if bool(getattr(streamlit_compat.st, "_is_running_with_streamlit", False)):
        pytest.skip("running inside an active Streamlit script context")


def test_session_state_uses_local_fallback_outside_streamlit(caplog: pytest.LogCaptureFixture):
    _skip_if_real_streamlit()

    caplog.set_level(logging.WARNING, logger="streamlit")

    streamlit_compat.st.session_state["_compat_probe"] = "ok"

    assert streamlit_compat.st.session_state.get("_compat_probe") == "ok"
    assert not [record for record in caplog.records if record.name.startswith("streamlit")]


def test_missing_streamlit_methods_explain_current_product_runtime():
    _skip_if_real_streamlit()

    with pytest.raises(RuntimeError, match=r"FastAPI \+ React"):
        streamlit_compat.st.markdown("x")
