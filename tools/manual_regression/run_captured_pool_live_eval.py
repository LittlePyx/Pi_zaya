from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

try:
    from tools.manual_regression.score_captured_replay_pool_rubric import _case_family, _score_case  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Allow running as a script without installing the repo as a package.
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from tools.manual_regression.score_captured_replay_pool_rubric import _case_family, _score_case  # type: ignore


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


def _stream_done(base_url: str, session_id: str, timeout_s: float) -> dict[str, Any]:
    req = request.Request(f"{base_url}/api/generate/{session_id}/stream", method="GET")
    final_payload: dict[str, Any] = {}
    t0 = time.time()
    with request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            if (time.time() - t0) > timeout_s:
                break
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
                if bool(payload.get("done")):
                    break
    return final_payload


@dataclass
class LiveEvalRow:
    pool_id: str
    family: str
    prompt: str
    source_lock_path: str
    answer: str
    score_overall: float


def _family_for_case(rec: dict[str, Any]) -> str:
    return str(_case_family(rec) or "").strip() or "overview"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-run captured pool prompts against live API and rubric-score answers.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8016")
    ap.add_argument("--pool", default="tests/replay/paper_guide_failure_pool_captured.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--out", default="test_results/captured_pool_live_eval.json")
    args = ap.parse_args(argv)

    base_url = str(args.base_url).strip().rstrip("/")
    pool_path = Path(str(args.pool)).expanduser().resolve()
    rows = [json.loads(l) for l in pool_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    out_rows: list[dict[str, Any]] = []
    limit = max(1, int(args.limit))
    for rec in rows[:limit]:
        if not isinstance(rec, dict):
            continue
        prompt = str(rec.get("prompt") or "").strip()
        prov = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {}
        src = str(prov.get("source_path") or "").strip()
        if (not prompt) or (not src):
            continue
        family = _family_for_case(rec)
        conv_id = f"live-eval::{family}::{int(time.time())}"
        create = _post_json(
            base_url,
            "/api/generate",
            {
                "conv_id": conv_id,
                "prompt": prompt,
                "top_k": 6,
                "deep_read": True,
                "max_tokens": 1400,
                "temperature": 0.15,
                "source_lock_path": src,
                "source_lock_name": str(prov.get("source_name") or ""),
            },
            timeout_s=float(args.timeout_s),
        )
        session_id = str(create.get("session_id") or "").strip()
        if not session_id:
            continue
        final = _stream_done(base_url, session_id, timeout_s=float(args.timeout_s))
        ans = str(final.get("answer") or "").strip()
        scored_case = dict(rec)
        scored_case["assistant_content"] = ans
        s = _score_case(scored_case)
        out_rows.append(
            {
                "pool_id": rec.get("id"),
                "family": family,
                "prompt": prompt,
                "source_lock_path": src,
                "answer": ans,
                "score": {
                    "overall_100": s.overall_100,
                    "question_hit": s.question_hit,
                    "evidence_consistency": s.evidence_consistency,
                    "locate_first_click": s.locate_first_click,
                    "uncertainty_handling": s.uncertainty_handling,
                    "readability": s.readability,
                },
            }
        )

    out_path = Path(str(args.out)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    out_rows.sort(key=lambda r: float(((r.get("score") or {}).get("overall_100") or 0.0)))
    worst = out_rows[: min(8, len(out_rows))]
    print(f"wrote {out_path} n={len(out_rows)}")
    for r in worst:
        sc = r.get("score") or {}
        print(f"[{float(sc.get('overall_100') or 0.0):>5.1f}] {r.get('family')} {str(r.get('pool_id') or '')}")
        print(f"prompt: {str(r.get('prompt') or '')[:140]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
