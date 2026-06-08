from __future__ import annotations

import server


def test_startup_readiness_lines_show_blockers_and_cli_hint() -> None:
    payload = {
        "status": "error",
        "env": "production",
        "production": True,
        "items": [
            {
                "key": "text_llm",
                "severity": "error",
                "label": "Text model",
                "detail": "missing_api_key",
                "action": "configure_text_api_key",
            },
            {
                "key": "api_auth",
                "severity": "ok",
                "label": "API access protection",
                "detail": "Enabled",
            },
        ],
    }

    lines = server._startup_readiness_lines(payload, host="127.0.0.1", port=8000)

    assert lines[0] == "Pi-zaya startup preflight: ERROR (production, production)"
    assert "Text model - missing_api_key [configure_text_api_key]" in "\n".join(lines)
    assert "tools\\check_production_readiness.py --base-url http://127.0.0.1:8000" in "\n".join(lines)


def test_startup_preflight_exit_code_only_blocks_in_strict_mode() -> None:
    payload = {"status": "error"}

    assert server._startup_preflight_exit_code(payload, strict=False) == 0
    assert server._startup_preflight_exit_code(payload, strict=True) == 2
    assert server._startup_preflight_exit_code({"status": "warning"}, strict=True) == 0
