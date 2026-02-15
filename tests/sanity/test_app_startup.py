
import pytest
import sys
import os

def test_imports():
    """
    Smoke test to ensure critical modules can be imported without error.
    This catches syntax errors, missing dependencies, or circular imports.
    """
    try:
        from kb.chat_store import ChatStore
        from kb.library_store import LibraryStore
        from kb.llm import DeepSeekChat
        import app
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during import: {e}")

def test_database_init(tmp_path):
    """
    Test that we can initialize a ChatStore with a temporary database.
    """
    from kb.chat_store import ChatStore
    
    db_path = tmp_path / "test_chat.sqlite3"
    try:
        store = ChatStore(str(db_path))
        # Verify it created tables or at least didn't crash
        assert os.path.exists(str(db_path))
    except Exception as e:
        pytest.fail(f"Failed to initialize ChatStore: {e}")
