import os

import pytest


def test_imports():
    """Smoke test: critical modules should import cleanly."""
    try:
        from kb.chat_store import ChatStore  # noqa: F401
        from kb.library_store import LibraryStore  # noqa: F401
        from kb.llm import DeepSeekChat  # noqa: F401
        import app  # noqa: F401
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during import: {e}")


def test_database_init(tmp_path):
    """ChatStore should initialize a temporary database path."""
    from kb.chat_store import ChatStore

    db_path = tmp_path / "test_chat.sqlite3"
    try:
        ChatStore(str(db_path))
        assert os.path.exists(str(db_path))
    except Exception as e:
        pytest.fail(f"Failed to initialize ChatStore: {e}")

