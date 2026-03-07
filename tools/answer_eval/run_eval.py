from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import request


def _post_json(base_url: str, path: str, payload: dict, timeout_s: float) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")


def _get_json(base_url: str, path: str, timeout_s: float) -> dict:
    req = request.Request(f"{base_url}{path}", method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")


def _patch_settings(base_url: str, patch: dict, timeout_s: float) -> None:
    if not isinstance(patch, dict) or (not patch):
        return
    data = json.dumps(patch, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{base_url}/api/settings",
        data=data,
        headers={"Content-Type": "application/json"},
        method="PATCH",
    )
    with request.urlopen(req, timeout=timeout_s):
        return


def _stream_generation(base_url: str, session_id: str, timeout_s: float) -> dict:
    req = request.Request(f"{base_url}/api/generate/{session_id}/stream", method="GET")
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


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (not line) or line.startswith("#"):
            continue
        rec = json.loads(line)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    path.write_text(text, encoding="utf-8")


@dataclass
class EvalAggregate:
    total: int
    done: int
    error: int
    canceled: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_char_count: float
    minimum_ok_rate: float
    core_section_coverage_avg: float
    evidence_required_count: int
    evidence_ok_rate: float


def _aggregate(rows: list[dict]) -> EvalAggregate:
    total = len(rows)
    done = sum(1 for r in rows if str(r.get("status") or "") == "done")
    error = sum(1 for r in rows if str(r.get("status") or "") == "error")
    canceled = sum(1 for r in rows if str(r.get("status") or "") == "canceled")
    latencies = [float(r.get("latency_ms") or 0.0) for r in rows if float(r.get("latency_ms") or 0.0) > 0]
    avg_latency_ms = round(float(statistics.mean(latencies)) if latencies else 0.0, 2)
    if latencies:
        lat_sorted = sorted(latencies)
        p95_idx = max(0, min(len(lat_sorted) - 1, int(math.ceil(len(lat_sorted) * 0.95)) - 1))
        p95_latency_ms = round(float(lat_sorted[p95_idx]), 2)
    else:
        p95_latency_ms = 0.0

    char_counts = [
        float((r.get("answer_quality") or {}).get("char_count") or 0.0)
        for r in rows
        if isinstance(r.get("answer_quality"), dict)
    ]
    avg_char_count = round(float(statistics.mean(char_counts)) if char_counts else 0.0, 2)

    probes = [r.get("answer_quality") for r in rows if isinstance(r.get("answer_quality"), dict)]
    minimum_ok = [bool(p.get("minimum_ok")) for p in probes]
    minimum_ok_rate = round((sum(1 for x in minimum_ok if x) / len(minimum_ok)) if minimum_ok else 0.0, 4)

    core_cov_values = [float(p.get("core_section_coverage") or 0.0) for p in probes]
    core_section_coverage_avg = round(float(statistics.mean(core_cov_values)) if core_cov_values else 0.0, 4)

    evidence_required = [p for p in probes if bool(p.get("evidence_required"))]
    evidence_required_count = len(evidence_required)
    if evidence_required:
        evidence_ok_rate = round(sum(1 for p in evidence_required if bool(p.get("evidence_ok"))) / len(evidence_required), 4)
    else:
        evidence_ok_rate = 0.0

    return EvalAggregate(
        total=total,
        done=done,
        error=error,
        canceled=canceled,
        avg_latency_ms=avg_latency_ms,
        p95_latency_ms=p95_latency_ms,
        avg_char_count=avg_char_count,
        minimum_ok_rate=minimum_ok_rate,
        core_section_coverage_avg=core_section_coverage_avg,
        evidence_required_count=evidence_required_count,
        evidence_ok_rate=evidence_ok_rate,
    )


def _build_report_md(*, summary: EvalAggregate, base_url: str, dataset_path: Path, output_dir: Path) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "\n".join(
        [
            "# Answer Optimization Eval Report",
            "",
            f"- Time: {now}",
            f"- Base URL: `{base_url}`",
            f"- Dataset: `{dataset_path}`",
            f"- Output: `{output_dir}`",
            "",
            "## KPI",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total cases | {summary.total} |",
            f"| Done | {summary.done} |",
            f"| Error | {summary.error} |",
            f"| Canceled | {summary.canceled} |",
            f"| Avg latency (ms) | {summary.avg_latency_ms} |",
            f"| P95 latency (ms) | {summary.p95_latency_ms} |",
            f"| Avg answer chars | {summary.avg_char_count} |",
            f"| Minimum-ok rate | {summary.minimum_ok_rate:.2%} |",
            f"| Core section coverage avg | {summary.core_section_coverage_avg:.2%} |",
            f"| Evidence-required count | {summary.evidence_required_count} |",
            f"| Evidence-ok rate | {summary.evidence_ok_rate:.2%} |",
            "",
            "## Go/No-Go Suggestion",
            "",
            "- `go` if minimum-ok rate >= 95% and evidence-ok rate >= 90% and error == 0.",
            "- otherwise `no-go`, inspect `raw_results.jsonl` and fix top failure buckets first.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run answer optimization eval against API endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        default="docs/answer_eval_dataset_v1.jsonl",
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--out-dir",
        default="test_results/answer_eval",
        help="Output directory root",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP timeout seconds")
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for quick smoke")
    parser.add_argument(
        "--answer-contract",
        choices=["on", "off", "keep"],
        default="on",
        help="Force answer_contract_v1 during eval (default: on)",
    )
    parser.add_argument(
        "--answer-depth-auto",
        choices=["on", "off", "keep"],
        default="on",
        help="Force answer_depth_auto during eval (default: on)",
    )
    parser.add_argument(
        "--answer-mode-hint",
        default="",
        help="Force answer_mode_hint during eval (default: empty string)",
    )
    parser.set_defaults(freeze_answer_prefs=True, restore_prefs=True)
    parser.add_argument(
        "--no-freeze-answer-prefs",
        action="store_false",
        dest="freeze_answer_prefs",
        help="Do not force answer prefs before each case",
    )
    parser.add_argument(
        "--no-restore-prefs",
        action="store_false",
        dest="restore_prefs",
        help="Do not restore original answer prefs after eval",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    cases = _read_jsonl(dataset_path)
    if args.limit > 0:
        cases = cases[: int(args.limit)]
    if not cases:
        print("[ERROR] no cases loaded", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    base_url = str(args.base_url).rstrip("/")
    forced_patch: dict[str, Any] = {}
    if str(args.answer_contract) != "keep":
        forced_patch["answer_contract_v1"] = str(args.answer_contract) == "on"
    if str(args.answer_depth_auto) != "keep":
        forced_patch["answer_depth_auto"] = str(args.answer_depth_auto) == "on"
    forced_patch["answer_mode_hint"] = str(args.answer_mode_hint or "").strip()[:32]

    backup_answer_prefs: dict[str, Any] | None = None
    if bool(args.freeze_answer_prefs):
        try:
            cur = _get_json(base_url, "/api/settings", timeout_s=float(args.timeout_s))
            prefs = cur.get("prefs") if isinstance(cur, dict) else {}
            prefs = prefs if isinstance(prefs, dict) else {}
            backup_answer_prefs = {
                "answer_contract_v1": prefs.get("answer_contract_v1"),
                "answer_depth_auto": prefs.get("answer_depth_auto"),
                "answer_mode_hint": prefs.get("answer_mode_hint"),
            }
            _patch_settings(base_url, forced_patch, timeout_s=float(args.timeout_s))
        except Exception as exc:
            print(f"[WARN] failed to freeze answer prefs before eval: {exc}")

    try:
        for idx, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or f"case_{idx:03d}")
            prompt = str(case.get("prompt") or "").strip()
            if not prompt:
                rows.append(
                    {
                        "id": case_id,
                        "status": "error",
                        "error": "empty prompt in dataset",
                        "answer_quality": {},
                    }
                )
                continue
            top_k = int(case.get("top_k") or 6)
            max_tokens = int(case.get("max_tokens") or 1216)
            deep_read = bool(case.get("deep_read") or False)
            record = dict(case)
            t0 = time.perf_counter()
            try:
                if bool(args.freeze_answer_prefs):
                    try:
                        _patch_settings(base_url, forced_patch, timeout_s=float(args.timeout_s))
                    except Exception as exc:
                        print(f"[WARN] failed to freeze answer prefs for case {case_id}: {exc}")
                conv = _post_json(
                    base_url,
                    "/api/conversations",
                    {"title": f"eval-{case_id}", "project_id": None},
                    timeout_s=float(args.timeout_s),
                )
                conv_id = str(conv.get("id") or "").strip()
                if not conv_id:
                    raise RuntimeError("failed to create conversation")

                gen = _post_json(
                    base_url,
                    "/api/generate",
                    {
                        "conv_id": conv_id,
                        "prompt": prompt,
                        "top_k": top_k,
                        "max_tokens": max_tokens,
                        "deep_read": deep_read,
                    },
                    timeout_s=float(args.timeout_s),
                )
                session_id = str(gen.get("session_id") or "").strip()
                if not session_id:
                    raise RuntimeError("missing session_id")
                final = _stream_generation(base_url, session_id, timeout_s=float(args.timeout_s))
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                record.update(
                    {
                        "status": str(final.get("status") or ""),
                        "done": bool(final.get("done")),
                        "latency_ms": latency_ms,
                        "answer_intent": str(final.get("answer_intent") or ""),
                        "answer_depth": str(final.get("answer_depth") or ""),
                        "answer_contract_v1": bool(final.get("answer_contract_v1", False)),
                        "answer_quality": final.get("answer_quality") if isinstance(final.get("answer_quality"), dict) else {},
                        "answer_preview": str(final.get("answer") or "")[:280],
                    }
                )
            except Exception as exc:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                record.update(
                    {
                        "status": "error",
                        "done": True,
                        "latency_ms": latency_ms,
                        "error": str(exc),
                        "answer_quality": {},
                    }
                )
            rows.append(record)
            print(f"[{idx}/{len(cases)}] {case_id} -> {record.get('status')} ({record.get('latency_ms')} ms)")
    finally:
        if bool(args.freeze_answer_prefs) and bool(args.restore_prefs) and isinstance(backup_answer_prefs, dict):
            try:
                restore_patch = {
                    "answer_contract_v1": bool(backup_answer_prefs.get("answer_contract_v1", False)),
                    "answer_depth_auto": bool(backup_answer_prefs.get("answer_depth_auto", True)),
                    "answer_mode_hint": str(backup_answer_prefs.get("answer_mode_hint") or ""),
                }
                _patch_settings(base_url, restore_patch, timeout_s=float(args.timeout_s))
            except Exception as exc:
                print(f"[WARN] failed to restore answer prefs after eval: {exc}")

    summary = _aggregate(rows)
    summary_dict = {
        "total": summary.total,
        "done": summary.done,
        "error": summary.error,
        "canceled": summary.canceled,
        "avg_latency_ms": summary.avg_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "avg_char_count": summary.avg_char_count,
        "minimum_ok_rate": summary.minimum_ok_rate,
        "core_section_coverage_avg": summary.core_section_coverage_avg,
        "evidence_required_count": summary.evidence_required_count,
        "evidence_ok_rate": summary.evidence_ok_rate,
    }

    _write_jsonl(output_dir / "raw_results.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        _build_report_md(summary=summary, base_url=base_url, dataset_path=dataset_path, output_dir=output_dir),
        encoding="utf-8",
    )
    print(f"[OK] eval finished: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
