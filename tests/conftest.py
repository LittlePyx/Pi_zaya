import os
import sys

import pytest


# Add project root to sys.path so tests can import `kb`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def _default_api_auth_disabled(monkeypatch):
    """Keep route tests independent from a developer machine's auth env."""
    monkeypatch.setenv("KB_ENABLE_AUTH_GATE", "0")
    monkeypatch.setenv("KB_PRIVATE_INSTANCE_AUTH", "0")
    monkeypatch.setenv("KB_REQUIRE_AUTH", "0")
    monkeypatch.setenv("KB_ALLOW_LOCAL_AUTH_GATE", "0")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_ENABLED", "0")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_URL", "")
    monkeypatch.setenv("KB_USER_ISSUES_REMOTE_TOKEN", "")
    monkeypatch.setenv("KB_USER_ISSUES_ALLOW_UNAUTHENTICATED_REMOTE", "0")
    monkeypatch.setenv("KB_USER_ISSUES_ALLOW_LOCAL_REMOTE", "0")
    monkeypatch.setenv("KB_USER_ISSUES_CLIENT_ID", "")

