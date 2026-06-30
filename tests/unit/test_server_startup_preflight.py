from __future__ import annotations

import api.main as api_main
import server


def test_api_cors_exposes_frontend_observable_diagnostics_headers() -> None:
    exposed = {item.lower() for item in api_main._CORS_EXPOSE_HEADERS}

    assert "server-timing" in exposed
    assert "x-kb-refs-mode" in exposed
    assert "x-kb-refs-counts" in exposed
    assert "content-disposition" in exposed


def test_startup_readiness_lines_show_blockers_and_cli_hint() -> None:
    payload = {
        "status": "error",
        "env": "production",
        "production": True,
        "auth_required": False,
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
    assert "--token <access-token>" not in "\n".join(lines)


def test_startup_readiness_lines_show_token_hint_only_when_auth_required() -> None:
    payload = {
        "status": "ok",
        "env": "production",
        "production": True,
        "auth_required": True,
        "items": [],
    }

    lines = server._startup_readiness_lines(payload, host="0.0.0.0", port=9000)

    assert "All startup readiness checks passed." in "\n".join(lines)
    assert (
        "tools\\check_production_readiness.py --base-url http://127.0.0.1:9000 --token <access-token>"
        in "\n".join(lines)
    )


def test_startup_readiness_lines_format_reachable_base_url_for_wildcard_hosts() -> None:
    payload = {"status": "ok", "env": "production", "production": True, "auth_required": False, "items": []}

    ipv4_lines = "\n".join(server._startup_readiness_lines(payload, host="0.0.0.0", port=9000))
    ipv6_lines = "\n".join(server._startup_readiness_lines(payload, host="::", port=9000))

    assert "--base-url http://127.0.0.1:9000" in ipv4_lines
    assert "http://0.0.0.0:9000" not in ipv4_lines
    assert "--base-url http://127.0.0.1:9000" in ipv6_lines


def test_startup_readiness_base_url_formats_ipv6_loopback() -> None:
    assert server._readiness_base_url(host="::1", port=8000) == "http://[::1]:8000"
    assert server._readiness_base_url(host="[::1]", port=8000) == "http://[::1]:8000"


def test_server_port_env_parser_accepts_valid_port(monkeypatch) -> None:
    monkeypatch.setenv("KB_SERVER_PORT", "9001")

    assert server._env_int("KB_SERVER_PORT", 8000, min_value=1, max_value=65535) == 9001


def test_server_port_env_parser_uses_default_for_missing_or_blank_port(monkeypatch) -> None:
    monkeypatch.delenv("KB_SERVER_PORT", raising=False)
    assert server._env_int("KB_SERVER_PORT", 8000, min_value=1, max_value=65535) == 8000

    monkeypatch.setenv("KB_SERVER_PORT", "   ")
    assert server._env_int("KB_SERVER_PORT", 8000, min_value=1, max_value=65535) == 8000


def test_server_port_env_parser_rejects_invalid_port(monkeypatch) -> None:
    for raw in ["not-a-number", "0", "70000"]:
        monkeypatch.setenv("KB_SERVER_PORT", raw)
        try:
            server._env_int("KB_SERVER_PORT", 8000, min_value=1, max_value=65535)
        except SystemExit as exc:
            assert "KB_SERVER_PORT must be an integer between 1 and 65535" in str(exc)
        else:
            raise AssertionError(f"expected SystemExit for {raw!r}")


def test_startup_preflight_exit_code_only_blocks_in_strict_mode() -> None:
    payload = {"status": "error"}

    assert server._startup_preflight_exit_code(payload, strict=False) == 0
    assert server._startup_preflight_exit_code(payload, strict=True) == 2
    assert server._startup_preflight_exit_code({"status": "warning"}, strict=True) == 0
