"""
Diagnostic: Check if [n] and [[CITE:...]] markers survive through the pipeline.
Usage: python -m tools.diagnose_citations [chat_db_path]
"""

import json
import re
import sqlite3
import sys
from pathlib import Path


def _has_n_markers(content: str) -> list[str]:
    return sorted(set(re.findall(r"\[(\d{1,4})\]", content)), key=int)


def _has_cite_markers(content: str) -> list[tuple[str, str]]:
    return re.findall(r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]+)\s*:\s*(\d+)\s*\]\]", content, re.IGNORECASE)


def _has_canonical_hit_paths(meta_dict: dict) -> list[str]:
    return list(meta_dict.get("canonical_hit_paths") or [])


def main():
    default_dbs = [
        Path("~/kb/chat.sqlite3").expanduser(),
        Path("~/kb/chat_default.sqlite3").expanduser(),
        Path(".").resolve() / "chat.sqlite3",
    ]
    db_path = None
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        for p in default_dbs:
            if p.exists():
                db_path = p
                break
    if not db_path or not db_path.exists():
        print("Chat DB not found. Provide path: python -m tools.diagnose_citations <path>")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    msgs = conn.execute(
        """SELECT id, role, content, meta_json, conv_id
           FROM messages
           WHERE role = 'assistant' AND content NOT LIKE '%_LIVE_%'
           ORDER BY id DESC LIMIT 10"""
    ).fetchall()

    if not msgs:
        print("No assistant messages found")
        return

    for msg in msgs:
        mid = msg["id"]
        conv = msg["conv_id"]
        content = msg["content"] or ""
        meta_raw = msg["meta_json"] or "{}"

        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
        except Exception:
            meta = {}

        aq = meta.get("answer_quality", {}) or {}
        n_markers = _has_n_markers(content)
        cite_markers = _has_cite_markers(content)
        canon_paths = _has_canonical_hit_paths(meta)

        status_lines = []
        if n_markers:
            status_lines.append(f"OK [n] markers: {n_markers[:6]}")
        else:
            status_lines.append("MISS [n] markers: NONE")

        if cite_markers:
            status_lines.append(f"OK [[CITE:...]]: {len(cite_markers)} tokens")
        else:
            status_lines.append("MISS [[CITE:...]]: NONE")

        if canon_paths:
            status_lines.append(f"OK canonical_hit_paths: {len(canon_paths)} entries")
        else:
            status_lines.append("MISS canonical_hit_paths: NONE")

        # Answer quality summary
        aq_summary = {
            "hits": aq.get("has_hits", "?"),
            "cites": aq.get("has_citations", "?"),
            "family": aq.get("prompt_family", "?"),
            "mode": aq.get("output_mode", "?"),
            "pg_mode": aq.get("paper_guide_mode", "?"),
        }

        print(f"msg#{mid:>5} conv={conv[:12]} | {' | '.join(status_lines)}")
        print(f"       q={aq_summary} | preview={content[:100]}...")
        print()

    # Check message_refs for hit availability
    print("--- Message refs (last 5) ---")
    refs = conn.execute(
        """SELECT user_msg_id, conv_id, render_status, hits_json
           FROM message_refs
           ORDER BY user_msg_id DESC LIMIT 5"""
    ).fetchall()
    for r in refs:
        uid = r["user_msg_id"]
        conv = r["conv_id"]
        render = r["render_status"] or "(pending)"
        hits_str = r["hits_json"] or "[]"
        try:
            hit_count = len(json.loads(hits_str))
        except Exception:
            hit_count = -1
        print(f"  ref#{uid:<5} conv={conv[:12]} hits={hit_count} render={render}")

    conn.close()


if __name__ == "__main__":
    main()
