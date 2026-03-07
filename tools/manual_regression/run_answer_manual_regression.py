from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests


def _now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _auto_check_health(base_url: str) -> dict[str, Any]:
    url = f"{base_url}/api/health"
    t0 = time.perf_counter()
    try:
        resp = requests.get(url, timeout=6)
        elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
        ok = bool(resp.ok)
        payload = _safe_json(resp)
        return {
            "name": "health",
            "ok": ok and str(payload.get("status") or "").lower() in {"ok", "healthy", "ready"},
            "status_code": resp.status_code,
            "elapsed_ms": elapsed,
            "payload": payload,
            "error": "",
        }
    except Exception as exc:
        return {
            "name": "health",
            "ok": False,
            "status_code": 0,
            "elapsed_ms": 0.0,
            "payload": {},
            "error": str(exc),
        }


def _auto_check_quality_summary(base_url: str) -> dict[str, Any]:
    url = f"{base_url}/api/generate/quality/summary?limit=50"
    try:
        resp = requests.get(url, timeout=8)
        payload = _safe_json(resp)
        required_keys = {
            "total",
            "minimum_ok_rate",
            "structure_complete_rate",
            "evidence_coverage_rate",
            "next_steps_coverage_rate",
            "failed_rate",
            "by_intent",
            "by_depth",
            "fail_reasons",
        }
        ok = bool(resp.ok) and required_keys.issubset(set(payload.keys()))
        return {
            "name": "quality_summary",
            "ok": ok,
            "status_code": resp.status_code,
            "payload": payload,
            "error": "",
        }
    except Exception as exc:
        return {
            "name": "quality_summary",
            "ok": False,
            "status_code": 0,
            "payload": {},
            "error": str(exc),
        }


def _parse_sse_done_line(resp: requests.Response, timeout_s: float = 45.0) -> tuple[bool, dict[str, Any], str]:
    latest: dict[str, Any] = {}
    t0 = time.time()
    for raw in resp.iter_lines(decode_unicode=True):
        if (time.time() - t0) > timeout_s:
            return False, latest, "stream timeout"
        if not raw:
            continue
        line = str(raw).strip()
        if not line.startswith("data: "):
            continue
        payload_txt = line[6:].strip()
        try:
            payload = json.loads(payload_txt)
            if isinstance(payload, dict):
                latest = payload
                if bool(payload.get("done")):
                    return True, latest, ""
        except Exception:
            continue
    return False, latest, "stream closed before done"


def _auto_check_generation(base_url: str, conv_id: str) -> dict[str, Any]:
    start_url = f"{base_url}/api/generate"
    stream_url_tmpl = f"{base_url}/api/generate/{{session_id}}/stream"
    prompt = "请用2句话总结：为什么回答需要‘结论+依据+下一步’结构？"
    try:
        create_resp = requests.post(
            start_url,
            json={
                "conv_id": conv_id,
                "prompt": prompt,
                "top_k": 4,
                "temperature": 0.2,
                "max_tokens": 512,
            },
            timeout=15,
        )
        create_payload = _safe_json(create_resp)
        sid = str(create_payload.get("session_id") or "").strip()
        if (not create_resp.ok) or (not sid):
            return {
                "name": "generation_stream",
                "ok": False,
                "status_code": create_resp.status_code,
                "payload": create_payload,
                "error": "failed to start generation",
            }

        with requests.get(stream_url_tmpl.format(session_id=sid), stream=True, timeout=60) as stream_resp:
            if not stream_resp.ok:
                return {
                    "name": "generation_stream",
                    "ok": False,
                    "status_code": stream_resp.status_code,
                    "payload": {},
                    "error": "stream http error",
                }
            done, latest, err = _parse_sse_done_line(stream_resp, timeout_s=55.0)

        if not done:
            return {
                "name": "generation_stream",
                "ok": False,
                "status_code": 200,
                "payload": latest,
                "error": err or "stream not done",
            }

        q = latest.get("answer_quality")
        has_probe = isinstance(q, dict) and ("minimum_ok" in q)
        has_fields = all(k in latest for k in ("answer_intent", "answer_depth", "answer_contract_v1"))
        ok = bool(latest.get("status") in {"done", "error", "canceled"}) and has_probe and has_fields
        return {
            "name": "generation_stream",
            "ok": ok,
            "status_code": 200,
            "payload": latest,
            "error": "" if ok else "missing probe fields",
        }
    except Exception as exc:
        return {
            "name": "generation_stream",
            "ok": False,
            "status_code": 0,
            "payload": {},
            "error": str(exc),
        }


def _manual_cases() -> list[dict[str, str]]:
    return [
        {
            "id": "M1",
            "title": "引用弹窗字段完整",
            "step": "点击回答里的一个引用，检查弹窗是否包含标题/source/DOI或指标，且无 [SID]/[[CITE]] 原始 token。",
        },
        {
            "id": "M2",
            "title": "文献篮跨页面保持",
            "step": "加入2条文献后切换到文献管理页并返回，确认文献篮不丢失、不降级。",
        },
        {
            "id": "M3",
            "title": "文献篮跨会话隔离",
            "step": "切到其他会话再切回，确认条目不串会话；刷新页面后仍保持。",
        },
        {
            "id": "M4",
            "title": "流式展示稳定",
            "step": "发起一个较长回答，观察流式过程中无明显闪烁、重复段落或错位。",
        },
        {
            "id": "M5",
            "title": "复制与展示一致",
            "step": "对同一回答执行“复制文本/复制Markdown”，核对语义与页面展示一致。",
        },
    ]


def _ask_manual_result(case: dict[str, str]) -> dict[str, str]:
    print(f"\n[{case['id']}] {case['title']}")
    print(f"步骤: {case['step']}")
    while True:
        ans = input("结果 (p=pass / f=fail / s=skip): ").strip().lower()
        if ans in {"p", "f", "s"}:
            break
    notes = input("备注(可空): ").strip()
    status = {"p": "PASS", "f": "FAIL", "s": "SKIP"}[ans]
    return {
        "id": case["id"],
        "title": case["title"],
        "status": status,
        "notes": notes,
    }


def _render_report(
    *,
    base_url: str,
    auto_results: list[dict[str, Any]],
    manual_results: list[dict[str, str]],
    started_at: str,
) -> str:
    lines: list[str] = []
    lines.append("# 回答优化手工回归报告")
    lines.append("")
    lines.append(f"- 时间: {started_at}")
    lines.append(f"- 环境: `{base_url}`")
    lines.append("")
    lines.append("## 自动检查")
    for item in auto_results:
        mark = "PASS" if item.get("ok") else "FAIL"
        lines.append(f"- `{item.get('name')}`: **{mark}**")
        if item.get("status_code"):
            lines.append(f"  - status_code: `{item.get('status_code')}`")
        if item.get("elapsed_ms"):
            lines.append(f"  - elapsed_ms: `{item.get('elapsed_ms')}`")
        err = str(item.get("error") or "").strip()
        if err:
            lines.append(f"  - error: `{err}`")
    lines.append("")
    lines.append("## 手工检查")
    if not manual_results:
        lines.append("- 未执行（`--non-interactive`）")
    else:
        for item in manual_results:
            lines.append(f"- `{item['id']}` {item['title']}: **{item['status']}**")
            if item.get("notes"):
                lines.append(f"  - 备注: {item['notes']}")
    lines.append("")
    auto_fail = sum(1 for x in auto_results if not bool(x.get("ok")))
    manual_fail = sum(1 for x in manual_results if x.get("status") == "FAIL")
    lines.append("## 结论")
    lines.append(f"- 自动失败数: `{auto_fail}`")
    lines.append(f"- 手工失败数: `{manual_fail}`")
    lines.append(f"- 总体: **{'PASS' if (auto_fail + manual_fail) == 0 else 'FAIL'}**")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run answer optimization manual regression checklist.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--conv-id", default=f"manual-regression-{uuid.uuid4().hex[:8]}", help="Conversation id used by generation check.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip /api/generate stream check.")
    parser.add_argument("--non-interactive", action="store_true", help="Do not ask manual pass/fail prompts.")
    parser.add_argument("--output", default="", help="Output markdown report path.")
    args = parser.parse_args()

    base_url = str(args.base_url or "").rstrip("/")
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    auto_results: list[dict[str, Any]] = []
    auto_results.append(_auto_check_health(base_url))
    auto_results.append(_auto_check_quality_summary(base_url))
    if not bool(args.skip_generation):
        auto_results.append(_auto_check_generation(base_url, conv_id=str(args.conv_id)))

    manual_results: list[dict[str, str]] = []
    if not bool(args.non_interactive):
        print("\n=== 手工检查开始 ===")
        for case in _manual_cases():
            manual_results.append(_ask_manual_result(case))

    out_path = Path(args.output).expanduser() if str(args.output or "").strip() else (
        Path("test_results") / "manual_regression" / f"answer_manual_regression_{_now_str()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(
        base_url=base_url,
        auto_results=auto_results,
        manual_results=manual_results,
        started_at=started_at,
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written: {out_path}")

    auto_fail = any(not bool(x.get("ok")) for x in auto_results)
    manual_fail = any(x.get("status") == "FAIL" for x in manual_results)
    return 1 if (auto_fail or manual_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())

