from __future__ import annotations

import threading
from typing import Optional


BG_LOCK = threading.Lock()
BG_STATE = {
    "queue": [],
    "running": False,
    "done": 0,
    "total": 0,
    "current": "",
    "cur_page_done": 0,
    "cur_page_total": 0,
    "cur_page_msg": "",
    "cancel": False,
    "last": "",
}
BG_THREAD: Optional[threading.Thread] = None


GEN_LOCK = threading.Lock()
GEN_TASKS: dict[str, dict] = {}


# Citation metadata async worker state (used by ui/refs_renderer.py).
CITATION_LOCK = threading.Lock()
CITATION_TASKS: dict[str, dict] = {}


CACHE_LOCK = threading.Lock()
CACHE: dict[str, dict] = {
    "file_text": {},
    "deep_read": {},
    "trans": {},
    "rerank": {},
    "refs_pack": {},
}
