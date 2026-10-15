"""Isolated production API + SQLite bridge for browser persistence regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with tempfile.TemporaryDirectory(prefix="pi-zaya-shelf-e2e-") as directory:
        root = Path(directory)
        os.environ["KB_TEST_ISOLATED"] = "1"
        os.environ["KB_TEST_DATA_ROOT"] = directory
        os.environ["KB_APP_DATA_DIR"] = directory
        for name, relative in {"KB_CHAT_DB": "chat.sqlite3", "KB_LIBRARY_DB": "library.sqlite3",
                               "KB_USER_ISSUES_DB": "issues.sqlite3", "KB_USER_PREFS_PATH": "preferences.json",
                               "KB_DB_DIR": "db", "KB_MD_DIR": "md", "KB_PDF_DIR": "pdf"}.items():
            os.environ[name] = str(root / relative)
        from api.main import app
        from api.routers import chat as router
        from kb.chat_store import ChatStore
        import uvicorn

        store = ChatStore(root / "chat.sqlite3")
        router.get_chat_store = lambda: store
        project = store.create_project("shelf browser regression")
        conversation = store.create_conversation("shelf browser regression", project_id=project)
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            def stop():
                sys.stdin.readline()
                server.should_exit = True
            threading.Thread(target=stop, daemon=True).start()
            print("SHELF_TEST_READY " + json.dumps({"url": f"http://127.0.0.1:{port}", "project": project, "conversation": conversation}), flush=True)
            server.run(sockets=[listener])


if __name__ == "__main__":
    main()
