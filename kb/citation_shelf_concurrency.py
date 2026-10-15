from __future__ import annotations

import sqlite3
import re


def item_identity(item: dict) -> str:
    def title(value):
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").strip().lower()).strip()
    if item.get("shelfItemKind") == "reader_selection":
        source = str(item.get("sourcePath") or item.get("sourceName") or "").strip().lower()
        anchor = str(item.get("blockId") or item.get("anchorId") or item.get("anchor") or "").strip().lower()
        excerpt = title(item.get("shelfExcerpt") or item.get("evidenceQuote") or item.get("raw") or item.get("main"))[:160]
        return f"reader-selection:{source}|{anchor}|{item.get('startOffset', '')}|{item.get('endOffset', '')}|{excerpt}"
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(item.get("doi") or item.get("doiUrl") or "").strip().lower())
    doi = doi.strip(" \t\r\n\"'`([{<)]}>.,;:")
    if doi:
        return f"doi:{doi}"
    name = title(item.get("title") or item.get("main"))
    year = str(item.get("year") or "").strip()
    year = year if re.fullmatch(r"\d{4}", year) else ""
    return f"title:{name}|{year}" if name else f"key:{str(item.get('key') or '').strip()}"


def pending_operations(conn, scope, scope_id, operations):
    return [op for op in operations if not conn.execute(
        "SELECT 1 FROM citation_shelf_write_receipts WHERE scope=? AND scope_id=? AND mutation_id=?",
        (scope, scope_id, op["id"]),
    ).fetchone()]


def apply_operations(items: list[dict], opened: bool, operations: list[dict], normalize) -> tuple[list[dict], bool]:
    for op in operations:
        upserts = normalize(op.get("upserts", []))
        keys = set(op.get("removedKeys", [])) | {item["key"] for item in upserts}
        identities = set(op.get("removedIdentities", [])) | {item_identity(item) for item in upserts}
        retained = [] if op.get("clear") else [item for item in items
            if item["key"] not in keys and item_identity(item) not in identities]
        order = {key: index for index, key in enumerate(op.get("order", []))}
        items = normalize(sorted(upserts + retained, key=lambda item: order.get(item["key"], 120)))
        if isinstance(op.get("open"), bool):
            opened = op["open"]
    return items, opened


class CitationShelfConflict(ValueError):
    """A snapshot may only replace the shelf revision it was read from."""


def initialize(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS citation_shelf_write_receipts (
        scope TEXT NOT NULL, scope_id TEXT NOT NULL, mutation_id TEXT NOT NULL,
        PRIMARY KEY (scope, scope_id, mutation_id))""")


def check_write(
    conn: sqlite3.Connection, scope: str, scope_id: str, revision: int,
    expected_revision: int | None, mutation_id: str = "",
) -> bool:
    """Run inside the shelf write transaction. True means already committed.

    The empty receipt permanently upgrades this scope to guarded writes. Legacy
    clients can still read or atomically append, but cannot replace or delete it.
    Receipts survive empty shelves and process restarts.
    """
    if mutation_id and expected_revision is None:
        raise CitationShelfConflict("A shelf mutation requires its read revision")
    if mutation_id and conn.execute(
        "SELECT 1 FROM citation_shelf_write_receipts WHERE scope=? AND scope_id=? AND mutation_id=?",
        (scope, scope_id, mutation_id),
    ).fetchone():
        return True
    guarded = conn.execute(
        "SELECT 1 FROM citation_shelf_write_receipts WHERE scope=? AND scope_id=? AND mutation_id=''",
        (scope, scope_id),
    ).fetchone()
    if (expected_revision is None and guarded) or (
        expected_revision is not None and expected_revision != revision
    ):
        raise CitationShelfConflict("Shelf changed; reload and reapply the pending operation")
    return False


def acknowledge(conn: sqlite3.Connection, scope: str, scope_id: str, mutation_id: str) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO citation_shelf_write_receipts VALUES (?, ?, ?)",
        [(scope, scope_id, ""), (scope, scope_id, mutation_id)],
    )
