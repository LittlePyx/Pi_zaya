from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import parse, request


DEFAULT_FIXTURE = Path("web/src/testing/researchQaData.json")
DEFAULT_OUT_DIR = Path("test_results/research_qa_eval")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class ResearchQaFixture:
    db_root: str
    docs: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    forbidden_phrases: list[str]

    @property
    def docs_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item.get("id") or ""): item for item in self.docs}


def _post_json(base_url: str, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")


def _get_json(base_url: str, path: str, timeout_s: float) -> Any:
    req = request.Request(f"{base_url}{path}", method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw or "{}")


def _stream_generation(base_url: str, session_id: str, timeout_s: float) -> dict[str, Any]:
    req = request.Request(f"{base_url}/api/generate/{parse.quote(session_id)}/stream", method="GET")
    final_payload: dict[str, Any] = {}
    with request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data_txt = line[len("data:") :].strip()
            if not data_txt:
                continue
            try:
                payload = json.loads(data_txt)
            except Exception:
                continue
            if isinstance(payload, dict):
                final_payload = payload
                if payload.get("done"):
                    break
    return final_payload


def load_fixture(path: Path | str = DEFAULT_FIXTURE) -> ResearchQaFixture:
    fixture_path = Path(path)
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    docs = data.get("docs") if isinstance(data, dict) else []
    cases = data.get("cases") if isinstance(data, dict) else []
    forbidden = data.get("forbiddenPhrases") if isinstance(data, dict) else []
    return ResearchQaFixture(
        db_root=str(data.get("dbRoot") or "") if isinstance(data, dict) else "",
        docs=[item for item in list(docs or []) if isinstance(item, dict)],
        cases=[item for item in list(cases or []) if isinstance(item, dict)],
        forbidden_phrases=[str(item) for item in list(forbidden or []) if str(item or "").strip()],
    )


def source_path_for_doc(fixture: ResearchQaFixture, doc_id: str) -> str:
    doc = fixture.docs_by_id.get(str(doc_id or ""))
    if not isinstance(doc, dict):
        return ""
    directory = str(doc.get("dir") or "").strip()
    file_stem = str(doc.get("file") or directory).strip()
    if not directory or not file_stem:
        return ""
    return f"{fixture.db_root.rstrip('/')}/{directory}/{file_stem}.en.md"


def _norm(value: Any) -> str:
    text = str(value or "").replace("\\", "/").lower()
    return " ".join(text.split())


def _walk_strings(value: Any, *, max_depth: int = 8) -> list[str]:
    if max_depth <= 0:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            out.extend(_walk_strings(key, max_depth=max_depth - 1))
            out.extend(_walk_strings(child, max_depth=max_depth - 1))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_walk_strings(child, max_depth=max_depth - 1))
        return out
    return []


def _payload_text(value: Any) -> str:
    return "\n".join(_walk_strings(value))


def _contains_term(haystack: Any, term: str) -> bool:
    hay = _norm(haystack)
    for needle in _term_aliases(term):
        needle_norm = _norm(needle)
        if needle_norm and needle_norm in hay:
            return True
    return False


def _term_aliases(term: str) -> list[str]:
    raw = str(term or "").strip()
    aliases = {
        "已有": ["已有", "现成", "成熟", "经典", "既有", "existing", "prior", "previous", "not new", "background"],
        "不是": ["不是", "不大", "关系不大", "并非", "不属于", "not", "not a", "different"],
        "foveated": ["foveated", "foveation", "注视", "焦点", "重要区域", "重点区域", "感兴趣区域", "空间变分辨率", "自适应采样"],
        "重要": ["重要", "重点", "重点区域", "关键", "变化剧烈", "边缘", "纹理", "salient", "important"],
        "wave": ["wave", "wave optics", "波动", "波前", "光场", "角度信息"],
        "重聚焦": ["重聚焦", "重新对焦", "重对焦", "refocus", "refocusing"],
        "resolution": ["resolution", "分辨率"],
        "noise": ["noise", "噪声"],
        "ray tracing": ["ray tracing", "ray-tracing", "ray transfer matrix", "射线追踪"],
        "SNR": ["SNR", "signal-to-noise", "signal to noise", "信噪比"],
        "optical sectioning": ["optical sectioning", "sectioning", "光学层切", "光学切片"],
        "perovskite": ["perovskite", "钙钛矿"],
    }
    extra_aliases = {
        "wave": ["\u6ce2\u52a8", "\u6ce2\u52a8\u5149\u5b66", "\u884d\u5c04", "\u4f20\u64ad"],
        "ray tracing": [
            "\u5149\u7ebf\u8ffd\u8ff9",
            "\u5149\u7ebf\u8ffd\u8e2a",
            "\u5c04\u7ebf\u8ffd\u8ff9",
            "\u5c04\u7ebf\u8ffd\u8e2a",
            "\u5149\u7ebf\u4f20\u9012\u77e9\u9635",
        ],
        "refocus": ["\u91cd\u805a\u7126", "\u91cd\u65b0\u5bf9\u7126", "\u91cd\u5bf9\u7126"],
    }
    out = [raw]
    out.extend(aliases.get(raw, []))
    out.extend(extra_aliases.get(raw, []))
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        key = str(item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(item))
    return deduped


def _doc_match_candidates(fixture: ResearchQaFixture, doc_id: str) -> list[str]:
    doc = fixture.docs_by_id.get(str(doc_id or "")) or {}
    return [
        str(doc.get("id") or ""),
        str(doc.get("title") or ""),
        str(doc.get("shortLabel") or ""),
        str(doc.get("dir") or ""),
        source_path_for_doc(fixture, doc_id),
    ]


def _doc_matches_payload(fixture: ResearchQaFixture, doc_id: str, payload: Any) -> bool:
    text = _norm(_payload_text(payload))
    return any(_contains_term(text, candidate) for candidate in _doc_match_candidates(fixture, doc_id))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _messages_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("messages", "items", "data"):
            child = payload.get(key)
            if isinstance(child, list):
                return [item for item in child if isinstance(item, dict)]
    return []


def _latest_assistant_message(messages_payload: Any) -> dict[str, Any]:
    messages = _messages_list(messages_payload)
    for item in reversed(messages):
        if str(item.get("role") or "").lower() == "assistant":
            return item
    return {}


def _answer_text(result: dict[str, Any]) -> str:
    message = result.get("assistant_message")
    if isinstance(message, dict):
        for key in ("rendered_body", "content", "copy_markdown", "copy_text"):
            text = str(message.get(key) or "").strip()
            if text:
                return text
    final_payload = result.get("final_payload")
    if isinstance(final_payload, dict):
        return str(final_payload.get("answer") or "").strip()
    return str(result.get("answer") or "").strip()


def _citation_details(result: dict[str, Any]) -> list[dict[str, Any]]:
    message = result.get("assistant_message")
    details: list[Any] = []
    if isinstance(message, dict):
        details.extend(_as_list(message.get("cite_details")))
        meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
        packet = meta.get("paper_guide_contracts") if isinstance(meta, dict) else {}
        if isinstance(packet, dict):
            render_packet = packet.get("render_packet") if isinstance(packet.get("render_packet"), dict) else {}
            details.extend(_as_list(render_packet.get("cite_details")))
    final_payload = result.get("final_payload")
    if isinstance(final_payload, dict):
        details.extend(_as_list(final_payload.get("cite_details")))
    return [item for item in details if isinstance(item, dict)]


def _extract_ref_packs(refs_payload: Any, user_msg_id: int | str | None = None) -> list[dict[str, Any]]:
    if not isinstance(refs_payload, dict):
        return []
    if user_msg_id is not None:
        for key in (user_msg_id, str(user_msg_id)):
            pack = refs_payload.get(key)
            if isinstance(pack, dict):
                return [pack]
    if "hits" in refs_payload and isinstance(refs_payload.get("hits"), list):
        return [refs_payload]
    return [item for item in refs_payload.values() if isinstance(item, dict)]


def _extract_ref_hits(refs_payload: Any, user_msg_id: int | str | None = None) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for pack in _extract_ref_packs(refs_payload, user_msg_id=user_msg_id):
        hits.extend([item for item in _as_list(pack.get("hits")) if isinstance(item, dict)])
    return hits


def _extract_primary_evidence_payloads(refs_payload: Any, user_msg_id: int | str | None = None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for pack in _extract_ref_packs(refs_payload, user_msg_id=user_msg_id):
        if isinstance(pack.get("primary_evidence"), dict):
            payloads.append(pack["primary_evidence"])
        if isinstance(pack.get("primary_evidence_alignment"), dict):
            payloads.append(pack["primary_evidence_alignment"])
        for hit in _as_list(pack.get("hits")):
            if not isinstance(hit, dict):
                continue
            ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
            if isinstance(ui_meta.get("primary_evidence"), dict):
                payloads.append(ui_meta["primary_evidence"])
            reader_open = ui_meta.get("reader_open") if isinstance(ui_meta.get("reader_open"), dict) else {}
            if isinstance(reader_open.get("primaryEvidence"), dict):
                payloads.append(reader_open["primaryEvidence"])
            if isinstance(reader_open.get("locateTarget"), dict):
                payloads.append(reader_open["locateTarget"])
    return payloads


def _ref_card_quality_failures(refs_payload: Any, forbidden_phrases: list[str], user_msg_id: int | str | None = None) -> list[str]:
    failures: list[str] = []
    hits = _extract_ref_hits(refs_payload, user_msg_id=user_msg_id)
    for idx, hit in enumerate(hits[:5], start=1):
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        summary = str(ui_meta.get("summary_line") or "").strip()
        why = str(ui_meta.get("why_line") or "").strip()
        if len(summary) < 12:
            failures.append(f"ref_card_{idx}_summary_too_short")
        if len(why) < 12:
            failures.append(f"ref_card_{idx}_why_too_short")
        card_text = f"{summary}\n{why}\n{_payload_text(hit)}"
        for phrase in forbidden_phrases:
            if _contains_term(card_text, phrase):
                failures.append(f"ref_card_{idx}_forbidden_phrase:{phrase}")
    return failures


def validate_case(
    case: dict[str, Any],
    fixture: ResearchQaFixture,
    result: dict[str, Any],
) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    answer = _answer_text(result)
    refs_payload = result.get("refs_payload")
    user_msg_id = result.get("user_msg_id")
    citation_details = _citation_details(result)
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    status = str(result.get("status") or "").strip().lower()
    add_check("generation_done", bool(result.get("done")) and status not in {"error", "canceled"}, status)

    forbidden_hits = [phrase for phrase in fixture.forbidden_phrases if _contains_term(answer, phrase)]
    add_check("answer_no_template_phrase", not forbidden_hits, forbidden_hits)

    direct_starts = ("the paper cites", "this is stated in", "该文在", "文献中提到")
    add_check("answer_directly_addresses_question", not _norm(answer).startswith(direct_starts), answer[:120])

    required_terms = [str(item) for item in _as_list(expected.get("requiredAnswerTerms")) if str(item or "").strip()]
    missing_answer_terms = [term for term in required_terms if not _contains_term(answer, term)]
    add_check("answer_contains_required_terms", not missing_answer_terms, missing_answer_terms)

    required_ref_doc_ids = [str(item) for item in _as_list(expected.get("requiredRefDocIds")) if str(item or "").strip()]
    missing_ref_docs = [doc_id for doc_id in required_ref_doc_ids if not _doc_matches_payload(fixture, doc_id, refs_payload)]
    add_check("refs_include_required_docs", not missing_ref_docs, missing_ref_docs)

    required_citation_doc_ids = [
        str(item) for item in _as_list(expected.get("requiredCitationDocIds")) if str(item or "").strip()
    ]
    missing_citation_docs = [
        doc_id for doc_id in required_citation_doc_ids if not _doc_matches_payload(fixture, doc_id, citation_details)
    ]
    add_check("citations_include_required_docs", not missing_citation_docs, missing_citation_docs)

    inpaper_details = [item for item in citation_details if bool(item.get("is_inpaper"))]
    if bool(expected.get("requireSystemB")):
        required_system_b_terms = [
            str(item) for item in _as_list(expected.get("requiredSystemBTerms")) if str(item or "").strip()
        ]
        missing_system_b_terms = [term for term in required_system_b_terms if not _contains_term(inpaper_details, term)]
        add_check("system_b_present", bool(inpaper_details), len(inpaper_details))
        add_check("system_b_contains_required_terms", not missing_system_b_terms, missing_system_b_terms)

    card_failures = _ref_card_quality_failures(
        refs_payload,
        fixture.forbidden_phrases,
        user_msg_id=user_msg_id,
    )
    add_check("refs_card_copy_quality", not card_failures, card_failures)

    required_primary_terms = [
        str(item) for item in _as_list(expected.get("requiredPrimaryEvidenceTerms")) if str(item or "").strip()
    ]
    if required_primary_terms:
        primary_payloads = _extract_primary_evidence_payloads(refs_payload, user_msg_id=user_msg_id)
        missing_primary_terms = [term for term in required_primary_terms if not _contains_term(primary_payloads, term)]
        add_check("primary_evidence_contains_required_terms", not missing_primary_terms, missing_primary_terms)

    forbidden_primary_terms = [
        str(item) for item in _as_list(expected.get("forbiddenPrimaryEvidenceTerms")) if str(item or "").strip()
    ]
    if forbidden_primary_terms:
        primary_payloads = _extract_primary_evidence_payloads(refs_payload, user_msg_id=user_msg_id)
        present_forbidden_primary_terms = [term for term in forbidden_primary_terms if _contains_term(primary_payloads, term)]
        add_check("primary_evidence_avoids_forbidden_terms", not present_forbidden_primary_terms, present_forbidden_primary_terms)

    failures = [item for item in checks if not item.get("ok")]
    return {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "answer_preview": answer[:360],
        "citation_count": len(citation_details),
        "system_b_count": len(inpaper_details),
        "ref_hit_count": len(_extract_ref_hits(refs_payload, user_msg_id=user_msg_id)),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _build_report(rows: list[dict[str, Any]], *, fixture_path: Path, base_url: str, output_dir: Path) -> str:
    passed = [row for row in rows if bool((row.get("quality") or {}).get("ok"))]
    failed = [row for row in rows if not bool((row.get("quality") or {}).get("ok"))]
    lines = [
        "# Research QA Eval Report",
        "",
        f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Base URL: `{base_url}`",
        f"- Fixture: `{fixture_path}`",
        f"- Output: `{output_dir}`",
        f"- Cases: {len(rows)}",
        f"- Passed: {len(passed)}",
        f"- Failed: {len(failed)}",
        "",
        "## Failures",
        "",
    ]
    if not failed:
        lines.append("- None")
    for row in failed:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        failure_names = [str(item.get("name") or "") for item in _as_list(quality.get("failures")) if isinstance(item, dict)]
        lines.append(f"- `{row.get('id')}`: {', '.join(failure_names) or 'unknown'}")
    return "\n".join(lines) + "\n"


def run_case(
    *,
    base_url: str,
    fixture: ResearchQaFixture,
    case: dict[str, Any],
    timeout_s: float,
    top_k: int,
    max_tokens: int,
) -> dict[str, Any]:
    case_id = str(case.get("id") or "").strip()
    question = str(case.get("question") or "").strip()
    preferred_sources = [source_path_for_doc(fixture, str(doc_id)) for doc_id in _as_list(case.get("docIds"))]
    preferred_sources = [item for item in preferred_sources if item]
    started = time.perf_counter()
    conv = _post_json(
        base_url,
        "/api/conversations",
        {"title": f"research-qa-{case_id}", "project_id": None},
        timeout_s=timeout_s,
    )
    conv_id = str(conv.get("id") or "").strip()
    if not conv_id:
        raise RuntimeError("conversation creation returned no id")

    gen = _post_json(
        base_url,
        "/api/generate",
        {
            "conv_id": conv_id,
            "prompt": question,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "preferred_sources": preferred_sources,
        },
        timeout_s=timeout_s,
    )
    session_id = str(gen.get("session_id") or "").strip()
    if not session_id:
        raise RuntimeError("generation returned no session_id")
    final_payload = _stream_generation(base_url, session_id, timeout_s=timeout_s)
    messages = _get_json(base_url, f"/api/conversations/{parse.quote(conv_id)}/messages?render_packet_only=1", timeout_s)
    refs_payload = _get_json(base_url, f"/api/references/conversation/{parse.quote(conv_id)}", timeout_s)
    assistant_message = _latest_assistant_message(messages)
    user_msg_id = gen.get("user_msg_id")
    row = {
        "id": case_id,
        "conv_id": conv_id,
        "user_msg_id": user_msg_id,
        "assistant_msg_id": gen.get("assistant_msg_id"),
        "status": str(final_payload.get("status") or ""),
        "done": bool(final_payload.get("done")),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "question": question,
        "expected": case.get("expected") if isinstance(case.get("expected"), dict) else {},
        "final_payload": final_payload,
        "assistant_message": assistant_message,
        "refs_payload": refs_payload,
    }
    row["quality"] = validate_case(case, fixture, row)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real research-user QA checks against the local API.")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Shared research QA fixture JSON.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory root.")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="HTTP timeout seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit.")
    parser.add_argument("--case-id", action="append", default=[], help="Run one or more case ids.")
    parser.add_argument("--top-k", type=int, default=6, help="Retrieval top_k sent to /api/generate.")
    parser.add_argument("--max-tokens", type=int, default=1800, help="Max tokens sent to /api/generate.")
    parser.add_argument("--dry-run", action="store_true", help="Load fixture and print planned cases without calling API.")
    parser.add_argument("--fail-on-quality", action="store_true", help="Exit 1 when any quality check fails.")
    args = parser.parse_args(argv)

    fixture_path = Path(args.fixture)
    fixture = load_fixture(fixture_path)
    selected_cases = list(fixture.cases)
    wanted_ids = {str(item) for item in list(args.case_id or []) if str(item or "").strip()}
    if wanted_ids:
        selected_cases = [item for item in selected_cases if str(item.get("id") or "") in wanted_ids]
    if int(args.limit or 0) > 0:
        selected_cases = selected_cases[: int(args.limit)]
    if not selected_cases:
        print("[ERROR] no cases selected", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[OK] fixture: {fixture_path}")
        print(f"[OK] docs: {len(fixture.docs)}")
        print(f"[OK] cases: {len(selected_cases)}")
        for idx, case in enumerate(selected_cases, start=1):
            doc_ids = ", ".join(str(item) for item in _as_list(case.get("docIds")))
            print(f"{idx:02d}. {case.get('id')} [{doc_ids}] {case.get('question')}")
        return 0

    base_url = str(args.base_url or DEFAULT_BASE_URL).rstrip("/")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(selected_cases, start=1):
        case_id = str(case.get("id") or f"case-{idx}")
        try:
            row = run_case(
                base_url=base_url,
                fixture=fixture,
                case=case,
                timeout_s=float(args.timeout_s),
                top_k=int(args.top_k),
                max_tokens=int(args.max_tokens),
            )
        except Exception as exc:
            row = {
                "id": case_id,
                "status": "error",
                "done": True,
                "error": str(exc),
                "quality": {"ok": False, "failures": [{"name": "runner_error", "detail": str(exc)}]},
            }
        rows.append(row)
        ok = bool((row.get("quality") or {}).get("ok")) if isinstance(row.get("quality"), dict) else False
        print(f"[{idx}/{len(selected_cases)}] {case_id} -> {'pass' if ok else 'fail'} ({row.get('latency_ms', 0)} ms)")

    summary = {
        "total": len(rows),
        "passed": sum(1 for row in rows if bool((row.get("quality") or {}).get("ok"))),
        "failed": sum(1 for row in rows if not bool((row.get("quality") or {}).get("ok"))),
        "base_url": base_url,
        "fixture": str(fixture_path),
        "output_dir": str(output_dir),
    }
    _write_jsonl(output_dir / "raw_results.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        _build_report(rows, fixture_path=fixture_path, base_url=base_url, output_dir=output_dir),
        encoding="utf-8",
    )
    print(f"[OK] research QA eval finished: {output_dir}")
    if bool(args.fail_on_quality) and summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
