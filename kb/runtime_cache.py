from __future__ import annotations

from typing import Any

from kb import runtime_state as RUNTIME


def cache_get(bucket: str, key: str) -> Any:
    b = str(bucket or "").strip()
    k = str(key or "")
    if not b or not k:
        return None
    with RUNTIME.CACHE_LOCK:
        store = RUNTIME.CACHE.get(b)
        if not isinstance(store, dict):
            return None
        return store.get(k)


def cache_set(bucket: str, key: str, value: Any, *, max_items: int = 600) -> None:
    b = str(bucket or "").strip()
    k = str(key or "")
    if not b or not k:
        return
    try:
        cap = max(1, int(max_items or 600))
    except Exception:
        cap = 600
    with RUNTIME.CACHE_LOCK:
        store = RUNTIME.CACHE.setdefault(b, {})
        if not isinstance(store, dict):
            store = {}
            RUNTIME.CACHE[b] = store
        if len(store) >= cap and k not in store:
            overflow = len(store) - cap + 1
            for old_key in list(store.keys())[: max(1, overflow)]:
                store.pop(old_key, None)
        store[k] = value
