from __future__ import annotations

from tools.check_production_readiness import exit_code_for_status, format_readiness, join_api_url


def test_join_api_url_normalizes_slashes() -> None:
    assert join_api_url("http://127.0.0.1:8000/", "/api/readiness") == "http://127.0.0.1:8000/api/readiness"
    assert join_api_url("http://127.0.0.1:8000", "api/readiness") == "http://127.0.0.1:8000/api/readiness"


def test_exit_code_for_status_blocks_errors_and_optionally_allows_warnings() -> None:
    assert exit_code_for_status("ok") == 0
    assert exit_code_for_status("warning") == 1
    assert exit_code_for_status("warning", allow_warning=True) == 0
    assert exit_code_for_status("error") == 2
    assert exit_code_for_status("") == 2


def test_format_readiness_lists_severity_detail_and_action() -> None:
    text = format_readiness({
        "status": "error",
        "env": "production",
        "production": True,
        "items": [
            {"severity": "error", "label": "CORS origins", "detail": "Wildcard CORS", "action": "set_allowed_origins"},
        ],
    })

    assert "Pi-zaya readiness: ERROR (production, production)" in text
    assert "[ERROR] CORS origins - Wildcard CORS [set_allowed_origins]" in text
