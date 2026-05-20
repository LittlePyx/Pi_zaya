from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from kb.chat_store import ChatStore
    from kb.config import load_settings
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from kb.chat_store import ChatStore  # type: ignore[no-redef]
    from kb.config import load_settings  # type: ignore[no-redef]

# ── embedded regex + parser (mirrors ui/refs_renderer.py) ──────────────

_INPAPER_CITE_ANY_RE = re.compile(
    r"\[(\d{1,4}(?:\s*(?:-|–|—|,)\s*\d{1,4})*)\]",
)


def _parse_int_set(spec: str) -> list[int]:
    """Parse "45,46-49  52" -> [45,46,47,48,49,52] (mirrors ui/refs_renderer)."""
    s = (spec or "").strip()
    if not s:
        return []
    s = s.replace("　", ",").replace("、", ",").replace(";", ",")
    parts = re.split(r"[,\s]+", s)
    out: set[int] = set()
    for p in parts:
        t = (p or "").strip()
        if not t:
            continue
        t = t.replace("–", "-").replace("—", "-")
        if "-" in t:
            a, b = t.split("-", 1)
            a = a.strip()
            b = b.strip()
            try:
                x = int(a)
                y = int(b)
            except Exception:
                continue
            if x <= 0 or y <= 0:
                continue
            if x > y:
                x, y = y, x
            if (y - x) > 500:
                continue
            for k in range(x, y + 1):
                out.add(k)
        else:
            try:
                out.add(int(t))
            except Exception:
                continue
    return sorted(n for n in out if n > 0)


def _display_source_name(source_path: str) -> str:
    """Short display name from a source_path (mirrors ui/refs_renderer)."""
    return str(Path(str(source_path or "")).name or source_path)


def _truncate(text: str, max_len: int = 180) -> str:
    s = str(text or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# ── data model ────────────────────────────────────────────────────────

ResolutionStatus = str  # "HIT" | "FALLBACK" | "UNRESOLVED"


class CitationRecord:
    __slots__ = (
        "num",
        "status",
        "method",
        "source_path",
        "source_name",
        "heading",
        "snippet",
        "canonical_match",
    )

    def __init__(
        self,
        num: int,
        status: ResolutionStatus,
        method: str = "",
        source_path: str = "",
        source_name: str = "",
        heading: str = "",
        snippet: str = "",
        canonical_match: bool | None = None,
    ) -> None:
        self.num = num
        self.status = status
        self.method = method
        self.source_path = source_path
        self.source_name = source_name
        self.heading = heading
        self.snippet = snippet
        self.canonical_match = canonical_match


class MessageReport:
    __slots__ = (
        "msg_id",
        "user_msg_id",
        "content",
        "citations",
        "answer_quality",
        "has_canonical_paths",
        "canonical_path_count",
        "has_refs",
        "hit_count",
    )

    def __init__(
        self,
        msg_id: int,
        user_msg_id: int | None,
        content: str,
        citations: list[CitationRecord],
        answer_quality: dict | None = None,
        has_canonical_paths: bool = False,
        canonical_path_count: int = 0,
        has_refs: bool = False,
        hit_count: int = 0,
    ) -> None:
        self.msg_id = msg_id
        self.user_msg_id = user_msg_id
        self.content = content
        self.citations = citations
        self.answer_quality = answer_quality or {}
        self.has_canonical_paths = has_canonical_paths
        self.canonical_path_count = canonical_path_count
        self.has_refs = has_refs
        self.hit_count = hit_count


# ── resolution logic ──────────────────────────────────────────────────


def _resolve_n_from_hits_simulated(
    n: int,
    hits: list[dict],
    canonical_paths: list[str] | None,
) -> tuple[dict | None, str, bool | None]:
    """Simulates ui/refs_renderer._resolve_n_from_hits.

    Returns (detail_dict_or_None, method_label, canonical_match_or_None).
    """
    idx = n - 1

    # Phase 1: canonical_paths-based lookup
    if isinstance(canonical_paths, list) and canonical_paths and 0 <= idx < len(canonical_paths):
        target = (canonical_paths[idx] or "").strip().lower()
        if target:
            for hit in hits or []:
                meta_h = (hit.get("meta") or {}) or {}
                sp = str(meta_h.get("source_path") or "").strip().lower()
                if sp == target:
                    src_name = _display_source_name(str(meta_h.get("source_path") or ""))
                    heading = str(
                        meta_h.get("heading_path") or meta_h.get("ref_best_heading_path") or ""
                    ).strip()
                    snippet = str(hit.get("text") or "")[:280]
                    return (
                        {
                            "source_path": str(meta_h.get("source_path") or ""),
                            "source_name": src_name,
                            "heading": heading,
                            "snippet": snippet,
                        },
                        f"canonical_paths[{idx}] match",
                        True,
                    )
            # canonical path found but no hit with that source_path in refs
            return (
                None,
                f"canonical_paths[{idx}]={target} NOT FOUND in stored hits",
                True,
            )

    # Phase 2: positional fallback
    if 0 <= idx < len(hits):
        hit = hits[idx]
        meta_h = (hit.get("meta") or {}) or {}
        sp = str(meta_h.get("source_path") or "").strip()
        if sp:
            src_name = _display_source_name(sp)
            heading = str(
                meta_h.get("heading_path") or meta_h.get("ref_best_heading_path") or ""
            ).strip()
            snippet = str(hit.get("text") or "")[:280]
            return (
                {
                    "source_path": sp,
                    "source_name": src_name,
                    "heading": heading,
                    "snippet": snippet,
                },
                f"positional hits[{idx}] (no canonical_paths)",
                None,
            )
        return (None, f"hits[{idx}] has empty source_path", None)

    # Phase 3: out of range
    return (None, f"n={n} out of range (hits={len(hits)}, canonical_paths={len(canonical_paths or [])})", None)


# ── conversation analysis ─────────────────────────────────────────────


_FALLBACK_SOURCE_PATH_PLACEHOLDER = "__fallback_ref_index__"


def _load_reference_index(db_dir: Path) -> dict | None:
    """Try loading references_index.json for fallback resolution info."""
    p = Path(str(db_dir or "")).expanduser().resolve() / "references_index.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _try_fallback_reference(
    n: int,
    ref_index: dict | None,
    hits: list[dict],
) -> tuple[dict | None, str]:
    """Try to resolve [n] via reference index (bibliography)."""
    if not isinstance(ref_index, dict):
        return None, "reference index not available"

    # Scan all sources in the ref index for an entry matching ref_num=n
    best: dict | None = None
    for source_path, source_data in ref_index.items():
        if isinstance(source_data, list):
            for entry in source_data:
                if isinstance(entry, dict) and entry.get("ref_num") == n:
                    ref = entry.get("ref") or {}
                    best = {
                        "source_path": str(entry.get("source_path") or source_path),
                        "source_name": _display_source_name(
                            str(entry.get("source_path") or source_path)
                        ),
                        "heading": "",
                        "snippet": str(ref.get("raw") or ref.get("title") or "")[:280],
                    }
                    break
        elif isinstance(source_data, dict) and source_data.get("ref_num") == n:
            ref = source_data.get("ref") or {}
            best = {
                "source_path": str(source_data.get("source_path") or source_path),
                "source_name": _display_source_name(
                    str(source_data.get("source_path") or source_path)
                ),
                "heading": "",
                "snippet": str(ref.get("raw") or ref.get("title") or "")[:280],
            }
            break
        if best:
            break

    if best:
        return best, "reference index resolve"
    return None, "reference index: no match"


def _extract_citations(content: str) -> list[int]:
    """Extract and deduplicate all [n] numbers from content."""
    nums: set[int] = set()
    for m in _INPAPER_CITE_ANY_RE.finditer(str(content or "")):
        spec = (m.group(1) or "").strip()
        for n in _parse_int_set(spec):
            if n > 0:
                nums.add(n)
    return sorted(nums)


def verify_conv(
    conv_id: str,
    chat_store: ChatStore,
    db_dir: Path | None = None,
    *,
    verbose: bool = False,
) -> list[MessageReport]:
    """Analyze every assistant message in a conversation for citation resolution."""
    messages = chat_store.get_messages(conv_id)
    if not messages:
        return []

    refs_by_user = chat_store.list_message_refs(conv_id)  # {user_msg_id: ref_pack}
    ref_index = _load_reference_index(db_dir) if db_dir else None

    reports: list[MessageReport] = []
    last_user_msg_id = 0

    for msg in messages:
        role = str(msg.get("role") or "")
        msg_id = int(msg.get("id") or 0)

        if role == "user":
            if msg_id > 0:
                last_user_msg_id = msg_id
            continue

        if role != "assistant":
            continue

        content = str(msg.get("content") or "")
        meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
        quality = meta.get("answer_quality") if isinstance(meta.get("answer_quality"), dict) else {}
        canonical_paths_raw = meta.get("canonical_hit_paths")
        canonical_paths: list[str] | None = (
            list(canonical_paths_raw)
            if isinstance(canonical_paths_raw, list) and canonical_paths_raw
            else None
        )

        # Get refs for preceding user message
        ref_pack = refs_by_user.get(last_user_msg_id) if isinstance(refs_by_user, dict) else None
        hits = list((ref_pack or {}).get("hits") or []) if isinstance(ref_pack, dict) else []

        nums = _extract_citations(content)
        if not nums:
            continue

        records: list[CitationRecord] = []
        for n in nums:
            # Step A: hit-based resolution
            detail, method, canonical_match = _resolve_n_from_hits_simulated(
                n, hits, canonical_paths
            )
            if detail:
                records.append(
                    CitationRecord(
                        num=n,
                        status="HIT",
                        method=method,
                        source_path=str(detail.get("source_path") or ""),
                        source_name=str(detail.get("source_name") or ""),
                        heading=str(detail.get("heading") or ""),
                        snippet=str(detail.get("snippet") or ""),
                        canonical_match=canonical_match,
                    )
                )
                continue

            # Step B: fallback to reference index
            fallback_detail, fallback_method = _try_fallback_reference(n, ref_index, hits)
            if fallback_detail:
                records.append(
                    CitationRecord(
                        num=n,
                        status="FALLBACK",
                        method=fallback_method,
                        source_path=str(fallback_detail.get("source_path") or ""),
                        source_name=str(fallback_detail.get("source_name") or ""),
                        heading=str(fallback_detail.get("heading") or ""),
                        snippet=str(fallback_detail.get("snippet") or ""),
                        canonical_match=False,
                    )
                )
                continue

            # Unresolved
            records.append(
                CitationRecord(
                    num=n,
                    status="UNRESOLVED",
                    method=method,
                )
            )

        reports.append(
            MessageReport(
                msg_id=msg_id,
                user_msg_id=last_user_msg_id or None,
                content=content,
                citations=records,
                answer_quality=quality,
                has_canonical_paths=canonical_paths is not None,
                canonical_path_count=len(canonical_paths) if canonical_paths else 0,
                has_refs=bool(hits),
                hit_count=len(hits),
            )
        )

    return reports


# ── output formatting ─────────────────────────────────────────────────


_SEP = "─" * 60


def _format_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _print_header(conv: dict | None, verbose: bool = False) -> None:
    if not conv:
        return
    print()
    print(f"╔{'=' * 60}╗")
    title = str(conv.get("title") or "(no title)")
    mid = str(conv.get("id") or "")
    mode = str(conv.get("mode") or "normal")
    updated = _format_timestamp(float(conv.get("updated_at") or 0))
    print(f"║  Citation Trace Report  ──  conv: {mid}")
    print(f"║  Title: {title}")
    print(f"║  Mode: {mode}  |  Updated: {updated}")
    print(f"╚{'=' * 60}╝")
    print()


def _print_message_report(report: MessageReport, *, verbose: bool = False) -> None:
    print(f"── Message #{report.msg_id} (assistant)  ──")
    print(f"   Preceding user msg: #{report.user_msg_id or '?'}")
    print(f"   canonical_hit_paths: {'PRESENT' if report.has_canonical_paths else 'ABSENT'} "
          f"({report.canonical_path_count} paths)")
    print(f"   Stored hits count: {report.hit_count}")
    print(f"   [n] markers found: {len(report.citations)} distinct")

    if report.answer_quality:
        pf = str(report.answer_quality.get("prompt_family") or "")
        om = str(report.answer_quality.get("output_mode") or "")
        it = str(report.answer_quality.get("intent") or "")
        if pf or om or it:
            print(f"   Quality meta: intent={it}, family={pf}, output_mode={om}")

    if not report.citations:
        print("   (no [n] citations to trace)")
        print()
        return

    print()
    for rec in report.citations:
        if rec.status == "HIT":
            icon = "✅"
        elif rec.status == "FALLBACK":
            icon = "⚠️"
        else:
            icon = "❌"

        match_str = ""
        if rec.canonical_match is True:
            match_str = " ✅ canonical match"
        elif rec.canonical_match is False:
            match_str = " ❌ canonical MISMATCH"
        elif rec.canonical_match is None:
            match_str = " (positional, no canonical_paths)"

        print(f"  [{rec.num}] {icon} {rec.status} -- {rec.method}{match_str}")
        print(f"       source_name: {rec.source_name or '-'}")
        print(f"       source_path: {rec.source_path or '-'}")
        if rec.heading:
            print(f"       heading:     \"{rec.heading}\"")
        if rec.snippet:
            print(f"       snippet:     {_truncate(rec.snippet, 200)}")
        if verbose and rec.status == "UNRESOLVED":
            print(f"       Will be stripped in UI (strict mode)")
        print(f"       {'─' * 45}")

    print()


SUMM = "═" * 60


def _print_final_verdict(reports: list[MessageReport]) -> None:
    print(SUMM)
    if not reports:
        print("No assistant messages with [n] citations found.")
        print("VERDICT: ℹ️ INFO (no citations to verify)")
        print(SUMM)
        return

    total_cites = 0
    hit_count = 0
    fallback_count = 0
    unresolved_count = 0
    msgs_with_issues = 0

    for r in reports:
        for c in r.citations:
            total_cites += 1
            if c.status == "HIT":
                hit_count += 1
            elif c.status == "FALLBACK":
                fallback_count += 1
            else:
                unresolved_count += 1
        has_issue = any(c.status in ("FALLBACK", "UNRESOLVED") or c.canonical_match is False for c in r.citations)
        if has_issue:
            msgs_with_issues += 1

    hit_pct = (hit_count / total_cites * 100) if total_cites else 0
    fb_pct = (fallback_count / total_cites * 100) if total_cites else 0
    un_pct = (unresolved_count / total_cites * 100) if total_cites else 0

    print(f"  Conversations verified: 1")
    print(f"  Assistant msgs w/ cites: {len(reports)}")
    print(f"  Total [n] markers:       {total_cites}")
    print(f"  Hit-resolved:            {hit_count} ({hit_pct:.1f}%)  ✅")
    print(f"  Fallback-resolved:       {fallback_count} ({fb_pct:.1f}%)  ⚠️")
    print(f"  Unresolved:              {unresolved_count} ({un_pct:.1f}%)  ❌")

    if msgs_with_issues > 0:
        print(f"  Messages with issues:    {msgs_with_issues}")
        print()

    if unresolved_count > 0:
        verdict = "❌ FAIL"
    elif fallback_count > 0:
        verdict = "⚠️ WARNING"
    else:
        verdict = "✅ PASS"

    print(f"  OVERALL: {verdict}")
    print(SUMM)
    print()


# ── CLI ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Trace [n] citation resolution end-to-end for a conversation.",
    )
    ap.add_argument("conv_id", nargs="?", default=None, help="Conversation ID to verify")
    ap.add_argument(
        "--db",
        type=str,
        default=None,
        help="Override chat DB path (default: from settings)",
    )
    ap.add_argument(
        "--list-convs",
        action="store_true",
        help="List recent conversations with their IDs and exit",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show extra metadata for each [n]",
    )
    return ap


def _list_convs(chat_store: ChatStore) -> None:
    convs = chat_store.list_conversations(limit=50)
    if not convs:
        print("No conversations found.")
        return
    print(f"{'ID':<40} {'Title':<40} {'Updated':<20} {'Mode':<12}")
    print(f"{'─' * 40} {'─' * 40} {'─' * 20} {'─' * 12}")
    for c in convs:
        cid = str(c.get("id") or "")
        title = _truncate(str(c.get("title") or ""), 38)
        updated = _format_timestamp(float(c.get("updated_at") or 0))
        mode = str(c.get("mode") or "normal")
        print(f"{cid:<40} {title:<40} {updated:<20} {mode:<12}")
    print(f"\nTotal: {len(convs)} conversations")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings()
    db_path = (
        Path(args.db).expanduser().resolve()
        if args.db
        else getattr(settings, "chat_db_path", Path("./chat.sqlite3"))
    )
    db_dir = getattr(settings, "db_dir", None)

    if not db_path.exists():
        print(f"Chat DB not found: {db_path}", file=sys.stderr)
        return 1

    chat_store = ChatStore(db_path)

    if args.list_convs:
        _list_convs(chat_store)
        return 0

    conv_id = (args.conv_id or "").strip()
    if not conv_id:
        parser.print_help()
        return 1

    conv = chat_store.get_conversation(conv_id)
    if not conv:
        print(f"Conversation not found: {conv_id}", file=sys.stderr)
        return 1

    reports = verify_conv(
        conv_id,
        chat_store,
        db_dir=db_dir,
        verbose=args.verbose,
    )
    _print_header(conv)
    for report in reports:
        _print_message_report(report, verbose=args.verbose)
    _print_final_verdict(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
