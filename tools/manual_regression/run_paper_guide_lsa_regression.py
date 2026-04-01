from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SOURCE_PATH = (
    r"F:\research-papers\research-paper-pyx\LSA-2026-Interferometric Image Scanning..."
    r"lateral resolution inside live cells.pdf"
)
DEFAULT_PREF_PATCH = {
    "answer_contract_v1": True,
    "answer_depth_auto": True,
    "answer_mode_hint": "reading",
    "answer_output_mode": "reading_guide",
}
DEFAULT_GENERATE_BODY = {
    "top_k": 6,
    "deep_read": True,
    "max_tokens": 2600,
    "temperature": 0.15,
}
LSA_CASES: list[dict[str, Any]] = [
    {
        "id": "Q1",
        "title": "overview",
        "prompt": "What problem does this paper solve, and what are its core contributions?",
        "cite_required": False,
        "jump_required": False,
    },
    {
        "id": "Q2",
        "title": "method_apr",
        "prompt": "Explain how this method works, especially what APR does in the pipeline and why it matters.",
        "cite_required": False,
        "jump_required": False,
    },
    {
        "id": "Q3",
        "title": "compare_open_closed_pinhole",
        "prompt": "In reading-guide mode, explain the trade-offs they report between open-pinhole and closed-pinhole confocal iSCAT, especially resolution, noise floor, and CNR.",
        "cite_required": False,
        "jump_required": False,
    },
    {
        "id": "Q4",
        "title": "abstract_translate",
        "prompt": "Give the abstract text and translate it into Chinese.",
        "cite_required": False,
        "jump_required": False,
    },
    {
        "id": "Q5",
        "title": "figure1_walkthrough",
        "prompt": "In reading-guide mode, walk me through what Figure 1 demonstrates and which parts of the paper support that explanation.",
        "cite_required": False,
        "jump_required": True,
    },
]


def _now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_json(resp: requests.Response) -> dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_json(base_url: str, path: str, *, timeout_s: float) -> dict[str, Any]:
    resp = requests.get(f"{base_url}{path}", timeout=timeout_s)
    resp.raise_for_status()
    return _safe_json(resp)


def _patch_settings(base_url: str, patch: dict[str, Any], *, timeout_s: float) -> None:
    if not patch:
        return
    resp = requests.patch(f"{base_url}/api/settings", json=patch, timeout=timeout_s)
    resp.raise_for_status()


def _create_conversation(base_url: str, *, source_path: str, timeout_s: float) -> str:
    source = str(source_path or "").strip()
    if not source:
        raise RuntimeError("empty source_path")
    body = {
        "title": f"paper-guide-lsa-regression-{uuid.uuid4().hex[:8]}",
        "mode": "paper_guide",
        "bound_source_path": source,
        "bound_source_name": Path(source).name,
        "bound_source_ready": True,
    }
    resp = requests.post(f"{base_url}/api/conversations", json=body, timeout=timeout_s)
    resp.raise_for_status()
    payload = _safe_json(resp)
    conv_id = str(payload.get("id") or "").strip()
    if not conv_id:
        raise RuntimeError("failed to create conversation")
    return conv_id


def _start_generation(base_url: str, *, conv_id: str, prompt: str, timeout_s: float) -> str:
    body = dict(DEFAULT_GENERATE_BODY)
    body.update({"conv_id": conv_id, "prompt": prompt})
    resp = requests.post(f"{base_url}/api/generate", json=body, timeout=timeout_s)
    resp.raise_for_status()
    payload = _safe_json(resp)
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        raise RuntimeError("failed to start generation")
    return session_id


def _stream_generation(base_url: str, *, session_id: str, timeout_s: float) -> dict[str, Any]:
    final_payload: dict[str, Any] = {}
    deadline = time.time() + timeout_s
    with requests.get(f"{base_url}/api/generate/{session_id}/stream", stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if time.time() >= deadline:
                raise RuntimeError("generation stream timeout")
            if not raw:
                continue
            line = str(raw).strip()
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
                if bool(payload.get("done")):
                    return final_payload
    return final_payload


def _list_messages(base_url: str, *, conv_id: str, timeout_s: float) -> list[dict[str, Any]]:
    resp = requests.get(f"{base_url}/api/conversations/{conv_id}/messages", timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else []


def _max_message_id(messages: list[dict[str, Any]]) -> int:
    best = 0
    for msg in messages:
        try:
            mid = int((msg or {}).get("id") or 0)
        except Exception:
            mid = 0
        if mid > best:
            best = mid
    return best


def _is_live_task_text(text: str) -> bool:
    return str(text or "").strip().startswith("__KB_LIVE_TASK__:")


def _last_assistant_message(messages: list[dict[str, Any]], *, min_message_id: int = 0, allow_live: bool = False) -> dict[str, Any]:
    fallback: dict[str, Any] = {}
    for msg in reversed(messages):
        if str((msg or {}).get("role") or "").strip() != "assistant":
            continue
        try:
            mid = int((msg or {}).get("id") or 0)
        except Exception:
            mid = 0
        if mid <= int(min_message_id or 0):
            continue
        if not fallback:
            fallback = msg
        if allow_live or (not _is_live_task_text(str((msg or {}).get("content") or ""))):
            return msg
    return fallback


def _answer_text(message: dict[str, Any], payload: dict[str, Any]) -> str:
    for cand in (
        message.get("content"),
        payload.get("answer"),
        payload.get("partial"),
    ):
        if isinstance(cand, str) and cand.strip():
            text = cand.strip()
            if text.startswith("__KB_LIVE_TASK__:"):
                continue
            return text
    return ""


def _fetch_answer_bundle(
    base_url: str,
    *,
    conv_id: str,
    payload: dict[str, Any],
    timeout_s: float,
    min_message_id: int = 0,
    attempts: int = 20,
    delay_s: float = 1.0,
) -> tuple[dict[str, Any], str]:
    last_message: dict[str, Any] = {}
    answer_text = ""
    tries = max(1, int(attempts))
    for idx in range(tries):
        messages = _list_messages(base_url, conv_id=conv_id, timeout_s=timeout_s)
        last_message = _last_assistant_message(messages, min_message_id=min_message_id)
        if not last_message:
            last_message = _last_assistant_message(messages, min_message_id=min_message_id, allow_live=True)
        answer_text = _answer_text(last_message, payload)
        if answer_text.strip():
            return last_message, answer_text
        if idx + 1 < tries:
            time.sleep(max(0.1, float(delay_s)))
    return last_message, answer_text


def _render_cache(message: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    for cand in (message.get("render_cache"), payload.get("render_cache")):
        if isinstance(cand, dict):
            return cand
    return {}


def _provenance(message: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    for cand in (message.get("provenance"), payload.get("provenance")):
        if isinstance(cand, dict):
            return cand
    return {}


def _contains_any(text: str, needles: list[str]) -> bool:
    hay = str(text or "").lower()
    return any(str(needle or "").lower() in hay for needle in needles)


def _regex_count(text: str, patterns: list[str]) -> int:
    hay = str(text or "")
    return sum(1 for pattern in patterns if re.search(pattern, hay, flags=re.IGNORECASE))


def _has_jump_asset(answer_text: str, provenance: dict[str, Any]) -> bool:
    if "/api/references/asset?" in str(answer_text or ""):
        return True
    source_path = str(provenance.get("source_path") or "").strip()
    segments = provenance.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            for value in (seg.get("display_markdown"), seg.get("raw_markdown"), seg.get("text")):
                if "/api/references/asset?" in str(value or ""):
                    return True
            if not source_path:
                continue
            if str(seg.get("evidence_mode") or "").strip().lower() != "direct":
                continue
            if str(seg.get("locate_policy") or "").strip().lower() == "hidden":
                continue
            block_ids = [
                str(seg.get("primary_block_id") or "").strip(),
                *[
                    str(item or "").strip()
                    for item in list(seg.get("support_block_ids") or [])
                ],
                *[
                    str(item or "").strip()
                    for item in list(seg.get("evidence_block_ids") or [])
                ],
            ]
            has_block = any(block_ids)
            has_anchor = bool(
                str(seg.get("anchor_text") or "").strip()
                or str(seg.get("support_locate_anchor") or "").strip()
                or str(seg.get("primary_heading_path") or "").strip()
            )
            if has_block and has_anchor:
                return True
    return False


def _mentions_figure_panel(answer_text: str, panel: str) -> bool:
    token = str(panel or "").strip().lower()
    if not token:
        return False
    return bool(
        re.search(
            rf"(?:panel[s]?\s*\({re.escape(token)}\)|\({re.escape(token)}\)|\b{re.escape(token)}\s*[/,&]\s*[a-z]\b|\b[a-z]\s*[/,&]\s*{re.escape(token)}\b|panel\s*\({re.escape(token)},\s*[a-z]\)|panel\s*\([a-z],\s*{re.escape(token)}\))",
            str(answer_text or ""),
            flags=re.IGNORECASE,
        )
    )


def _rendered_citation_count(message: dict[str, Any]) -> int:
    rendered = str(message.get("rendered_body") or message.get("rendered_content") or "")
    if not rendered:
        return 0
    anchors = set(re.findall(r"#kb-cite-[A-Za-z0-9_-]+", rendered))
    return len(anchors)


def _raw_structured_citation_count(answer_text: str) -> int:
    text = str(answer_text or "")
    if not text:
        return 0
    markers = set(re.findall(r"\[\[CITE:[^:\]]+:\d+\]\]", text, flags=re.IGNORECASE))
    return len(markers)


def _minimum_ok_label(answer_quality: dict[str, Any] | None) -> str:
    if not isinstance(answer_quality, dict):
        return "n/a"
    if "minimum_ok" not in answer_quality:
        return "n/a"
    return str(bool(answer_quality.get("minimum_ok"))).lower()


def _evaluate_case(
    case: dict[str, Any],
    *,
    answer_text: str,
    answer_quality: dict[str, Any],
    render_cache: dict[str, Any],
    provenance: dict[str, Any],
    message: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    answer = str(answer_text or "").strip()
    low = answer.lower()
    raw_structured_cite_count = max(
        int(render_cache.get("raw_structured_cite_count") or 0),
        _raw_structured_citation_count(answer),
    )
    structured_cite_count = max(
        int(render_cache.get("cite_count") or 0),
        raw_structured_cite_count,
    )
    rendered_cite_count = max(
        int(len(message.get("cite_details") or [])) if isinstance(message.get("cite_details"), list) else 0,
        _rendered_citation_count(message),
        structured_cite_count,
    )
    jump_found = _has_jump_asset(answer, provenance)

    case_id = str(case.get("id") or "").strip().upper()
    if case_id == "Q1":
        if not _contains_any(low, ["interferometric image scanning", "iism", "image scanning microscopy"]):
            reasons.append("missing the paper's core method identity")
        if not _contains_any(low, ["live-cell", "live cell", "live cells", "label-free"]):
            reasons.append("missing live-cell/label-free application scope")
    elif case_id == "Q2":
        if not re.search(r"(phase correlation|image registration)", answer, flags=re.IGNORECASE):
            reasons.append("APR answer missed phase-correlation/image-registration grounding")
        if re.search(r"(phase gradient|wavefront curvature)", answer, flags=re.IGNORECASE):
            reasons.append("APR answer introduced unsupported mechanism wording")
    elif case_id == "Q3":
        if not (_contains_any(low, ["open-pinhole", "open pinhole"]) and _contains_any(low, ["closed-pinhole", "closed pinhole"])):
            reasons.append("missing explicit open/closed pinhole comparison")
        if _regex_count(answer, [r"122\s*nm", r"\b0\.03\b", r"\b10(?:\.0)?\b", r"\b0\.011\b", r"\b14(?:\.2)?\b"]) < 3:
            reasons.append("missing enough figure-1 quantitative comparisons")
    elif case_id == "Q4":
        if "Light microscopy remains indispensable" not in answer:
            reasons.append("abstract did not start from the true abstract text")
        if re.search(r"(^|\n)\s*(Conclusion|Evidence|Next Steps|结论|依据|下一步)\s*[:：]", answer):
            reasons.append("abstract answer leaked reading-guide shell sections")
        if "Interferometric Image Scanning Microscopy" in answer:
            reasons.append("abstract answer leaked the title instead of only the abstract body")
        if ("�" in answer) or ("锟" in answer):
            reasons.append("abstract translation contains mojibake")
    elif case_id == "Q5":
        if not re.search(r"(?:^|\b)(?:figure|fig\.?)\s*1", answer, flags=re.IGNORECASE):
            reasons.append("figure walkthrough did not anchor itself to Figure 1")
        has_de = bool(
            _mentions_figure_panel(answer, "d")
            and _mentions_figure_panel(answer, "e")
            and re.search(r"(open[- ]?pinhole|closed[- ]?pinhole|开针孔|闭针孔|开放针孔|关闭针孔)", answer, flags=re.IGNORECASE)
        )
        if not has_de:
            reasons.append("figure walkthrough missed the d/e open-vs-closed pinhole mapping")
        has_fg = bool(
            _mentions_figure_panel(answer, "f")
            and _mentions_figure_panel(answer, "g")
            and re.search(r"(apr|line profile|line profiles|线轮廓|重分配)", answer, flags=re.IGNORECASE)
        )
        if not has_fg:
            reasons.append("figure walkthrough missed the f/g APR-line-profile mapping")
        if not jump_found:
            reasons.append("figure walkthrough did not expose a jump asset")

    citation_status = "PASS"
    if bool(case.get("cite_required")) and rendered_cite_count <= 0:
        citation_status = "FAIL"
        reasons.append("missing rendered in-paper citations where this case expects them")
    elif rendered_cite_count <= 0:
        citation_status = "WARN"

    jump_status = "PASS"
    if bool(case.get("jump_required")) and (not jump_found):
        jump_status = "FAIL"
    elif not bool(case.get("jump_required")) and (not jump_found):
        jump_status = "WARN"

    return {
        "id": case_id,
        "title": str(case.get("title") or ""),
        "prompt": str(case.get("prompt") or ""),
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "answer_quality": answer_quality,
        "cite_count": rendered_cite_count,
        "structured_cite_count": structured_cite_count,
        "raw_structured_cite_count": raw_structured_cite_count,
        "citation_status": citation_status,
        "jump_found": bool(jump_found),
        "jump_status": jump_status,
        "answer_preview": answer[:800],
    }


def _render_report(
    *,
    base_url: str,
    source_path: str,
    results: list[dict[str, Any]],
    started_at: str,
) -> str:
    lines: list[str] = []
    lines.append("# Paper Guide LSA Regression Report")
    lines.append("")
    lines.append(f"- Time: `{started_at}`")
    lines.append(f"- Base URL: `{base_url}`")
    lines.append(f"- Source: `{source_path}`")
    lines.append("")
    lines.append("| Case | Status | minimum_ok | cites | jump |")
    lines.append("|---|---|---:|---:|---|")
    for item in results:
        aq = item.get("answer_quality") if isinstance(item.get("answer_quality"), dict) else {}
        lines.append(
            f"| {item.get('id')} | {item.get('status')} | "
            f"{_minimum_ok_label(aq)} | {int(item.get('cite_count') or 0)} | {item.get('jump_status')} |"
        )
    lines.append("")
    lines.append("## Findings")
    for item in results:
        lines.append(f"### {item.get('id')} {item.get('title')}")
        lines.append(f"- Status: **{item.get('status')}**")
        lines.append(f"- Prompt: `{item.get('prompt')}`")
        lines.append(f"- minimum_ok: `{_minimum_ok_label(item.get('answer_quality') if isinstance(item.get('answer_quality'), dict) else None)}`")
        lines.append(f"- cite_count: `{int(item.get('cite_count') or 0)}`")
        lines.append(f"- structured_cite_count: `{int(item.get('structured_cite_count') or 0)}`")
        lines.append(f"- raw_structured_cite_count: `{int(item.get('raw_structured_cite_count') or 0)}`")
        lines.append(f"- jump_status: `{item.get('jump_status')}`")
        reasons = list(item.get("reasons") or [])
        if reasons:
            for reason in reasons:
                lines.append(f"- fail_reason: `{reason}`")
        else:
            lines.append("- fail_reason: `none`")
        preview = str(item.get("answer_preview") or "").strip()
        if preview:
            lines.append("- answer_preview:")
            lines.append("")
            lines.append("```text")
            lines.append(preview)
            lines.append("```")
        lines.append("")
    failed = sum(1 for item in results if str(item.get("status") or "") != "PASS")
    lines.append("## Summary")
    lines.append(f"- failed_cases: `{failed}`")
    lines.append(f"- overall: **{'PASS' if failed == 0 else 'FAIL'}**")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LSA paper-guide regression against the local API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL.")
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH, help="Bound PDF source path.")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="HTTP timeout seconds.")
    parser.add_argument("--out-dir", default="test_results/paper_guide_regression", help="Output directory.")
    parser.add_argument("--no-restore-prefs", action="store_true", help="Do not restore answer settings after the run.")
    args = parser.parse_args()

    base_url = str(args.base_url or "").rstrip("/")
    source_path = str(args.source_path or "").strip()
    if not source_path:
        print("[ERROR] empty source path", file=sys.stderr)
        return 2

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.out_dir) / _now_str()
    out_dir.mkdir(parents=True, exist_ok=True)

    backup_prefs: dict[str, Any] | None = None
    try:
        cur = _get_json(base_url, "/api/settings", timeout_s=float(args.timeout_s))
        prefs = cur.get("prefs") if isinstance(cur, dict) else {}
        prefs = prefs if isinstance(prefs, dict) else {}
        backup_prefs = {
            "answer_contract_v1": prefs.get("answer_contract_v1"),
            "answer_depth_auto": prefs.get("answer_depth_auto"),
            "answer_mode_hint": prefs.get("answer_mode_hint"),
            "answer_output_mode": prefs.get("answer_output_mode"),
        }
    except Exception as exc:
        print(f"[WARN] failed to read current settings: {exc}", file=sys.stderr)

    conv_id = ""
    results: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    try:
        _patch_settings(base_url, DEFAULT_PREF_PATCH, timeout_s=float(args.timeout_s))
        conv_id = _create_conversation(base_url, source_path=source_path, timeout_s=float(args.timeout_s))
        for case in LSA_CASES:
            before_messages = _list_messages(base_url, conv_id=conv_id, timeout_s=float(args.timeout_s))
            min_message_id = _max_message_id(before_messages)
            session_id = _start_generation(
                base_url,
                conv_id=conv_id,
                prompt=str(case.get("prompt") or ""),
                timeout_s=float(args.timeout_s),
            )
            payload = _stream_generation(base_url, session_id=session_id, timeout_s=float(args.timeout_s))
            time.sleep(0.6)
            message, answer_text = _fetch_answer_bundle(
                base_url,
                conv_id=conv_id,
                payload=payload,
                timeout_s=float(args.timeout_s),
                min_message_id=min_message_id,
            )
            answer_quality = payload.get("answer_quality") if isinstance(payload.get("answer_quality"), dict) else {}
            render_cache = _render_cache(message, payload)
            provenance = _provenance(message, payload)
            result = _evaluate_case(
                case,
                answer_text=answer_text,
                answer_quality=answer_quality,
                render_cache=render_cache,
                provenance=provenance,
                message=message,
            )
            results.append(result)
            raw_rows.append(
                {
                    "case": case,
                    "payload": payload,
                    "message": message,
                    "evaluation": result,
                }
            )
            print(
                f"[{result['id']}] {result['status']} | minimum_ok={bool((answer_quality or {}).get('minimum_ok'))} "
                f"| cites={result['cite_count']} | jump={result['jump_status']}"
            )
    finally:
        if backup_prefs is not None and (not bool(args.no_restore_prefs)):
            try:
                _patch_settings(base_url, backup_prefs, timeout_s=float(args.timeout_s))
            except Exception as exc:
                print(f"[WARN] failed to restore settings: {exc}", file=sys.stderr)

    report_md = _render_report(
        base_url=base_url,
        source_path=source_path,
        results=results,
        started_at=started_at,
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "raw_results.json").write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written: {out_dir / 'report.md'}")
    print(f"Raw JSON written: {out_dir / 'raw_results.json'}")

    failed = any(str(item.get("status") or "") != "PASS" for item in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
