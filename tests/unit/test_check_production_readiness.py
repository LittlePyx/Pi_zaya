from __future__ import annotations

import json

from tools import check_production_readiness as readiness
from tools.check_production_readiness import (
    exit_code_for_status,
    format_readiness,
    is_local_base_url,
    join_api_url,
    request_json,
    resolve_access_token,
)


def test_join_api_url_normalizes_slashes() -> None:
    assert join_api_url("http://127.0.0.1:8000/", "/api/readiness") == "http://127.0.0.1:8000/api/readiness"
    assert join_api_url("http://127.0.0.1:8000", "api/readiness") == "http://127.0.0.1:8000/api/readiness"


def test_is_local_base_url_accepts_only_loopback_hosts() -> None:
    assert is_local_base_url("http://127.0.0.1:8000")
    assert is_local_base_url("http://localhost:8000")
    assert is_local_base_url("http://[::1]:8000")
    assert is_local_base_url("http://0.0.0.0:8000")
    assert is_local_base_url("http://127.42.0.9:8000")

    assert not is_local_base_url("https://example.com")
    assert not is_local_base_url("https://192.168.1.20:8000")


def test_resolve_access_token_uses_env_only_for_local_base_url(monkeypatch) -> None:
    monkeypatch.setenv("KB_ACCESS_TOKEN", "env-token")

    assert resolve_access_token("http://127.0.0.1:8000") == "env-token"
    assert resolve_access_token("https://collector.example") == ""
    assert resolve_access_token("https://collector.example", explicit_token=" explicit ") == "explicit"


def test_request_json_attaches_token_headers_only_when_token_is_resolved(monkeypatch) -> None:
    observed: list[dict[str, str]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"status": "ok"}).encode("utf-8")

    def fake_urlopen(req, timeout):
        observed.append(dict(req.header_items()))
        return FakeResponse()

    monkeypatch.setattr(readiness.urllib.request, "urlopen", fake_urlopen)

    assert request_json("http://127.0.0.1:8000/api/readiness", token="", timeout_s=1) == {"status": "ok"}
    assert request_json("http://127.0.0.1:8000/api/readiness", token="secret-token", timeout_s=1) == {"status": "ok"}

    first = {key.lower(): value for key, value in observed[0].items()}
    assert "x-kb-access-token" not in first
    assert "authorization" not in first
    second = {key.lower(): value for key, value in observed[1].items()}
    assert second["x-kb-access-token"] == "secret-token"
    assert second["authorization"] == "Bearer secret-token"


def test_main_does_not_send_env_token_to_remote_base_url(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("KB_ACCESS_TOKEN", "env-token")

    def fake_request_json(url: str, *, token: str = "", timeout_s: float = 8.0):
        observed.update({"url": url, "token": token, "timeout_s": timeout_s})
        return {"status": "ok", "env": "production", "production": True, "items": []}

    monkeypatch.setattr(readiness, "request_json", fake_request_json)

    assert readiness.main(["--base-url", "https://example.com", "--timeout", "3"]) == 0

    assert observed == {
        "url": "https://example.com/api/readiness",
        "token": "",
        "timeout_s": 3.0,
    }
    assert "Pi-zaya readiness: OK" in capsys.readouterr().out


def test_main_uses_explicit_token_for_remote_base_url(monkeypatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("KB_ACCESS_TOKEN", "env-token")

    def fake_request_json(url: str, *, token: str = "", timeout_s: float = 8.0):
        observed.update({"url": url, "token": token, "timeout_s": timeout_s})
        return {"status": "ok", "env": "production", "production": True, "items": []}

    monkeypatch.setattr(readiness, "request_json", fake_request_json)

    assert readiness.main(["--base-url", "https://example.com", "--token", "explicit-token"]) == 0

    assert observed["token"] == "explicit-token"


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
