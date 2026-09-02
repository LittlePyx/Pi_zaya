import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


# Add project root to sys.path so tests can import `kb`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Establish isolated mutable paths before test modules import the FastAPI app.
# This prevents route and lifespan tests from ever opening a developer's normal
# chat/library databases or document directories.
_PYTEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="pi-zaya-pytest-")).resolve()
_PYTEST_ALLOWED_ROOT = Path(tempfile.gettempdir()).resolve()
os.environ["KB_TEST_ISOLATED"] = "1"
# Individual tests legitimately redirect stores into pytest's own tmp_path,
# which is a sibling of the session directory.  Limit every mutable path to
# the operating-system temp tree while keeping the default stores in a unique
# per-run directory.
os.environ["KB_TEST_DATA_ROOT"] = str(_PYTEST_ALLOWED_ROOT)
for _key, _relative in {
    "KB_DB_DIR": "db",
    "KB_CHAT_DB": "chat.sqlite3",
    "KB_LIBRARY_DB": "library.sqlite3",
    "KB_USER_ISSUES_DB": "user_issues.sqlite3",
    "KB_USER_PREFS_PATH": "user_prefs.json",
    "KB_PDF_DIR": "pdfs",
    "KB_MD_DIR": "markdown",
}.items():
    os.environ[_key] = str(_PYTEST_DATA_ROOT / _relative)
for _directory in ("db", "pdfs", "markdown"):
    (_PYTEST_DATA_ROOT / _directory).mkdir(parents=True, exist_ok=True)


def pytest_unconfigure(config):
    del config
    shutil.rmtree(_PYTEST_DATA_ROOT, ignore_errors=True)


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

