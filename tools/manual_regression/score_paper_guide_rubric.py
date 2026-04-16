from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RAW_SUPPORT_RE = re.compile(r"\[\[SUPPORT:[^\]]+\]\]")
_RAW_CITE_RE = re.compile(r"\[\[CITE:[^\]]+\]\]")


def _to_int(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _to_float(v: Any) -> float:
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _extract_gate(evaluation: dict[str, Any], gate: str) -> dict[str, Any]:
    gates = evaluation.get("gates")
    if isinstance(gates, dict) and isinstance(gates.get(gate), dict):
        return dict(gates.get(gate) or {})
    gate_results = evaluation.get("gate_results")
    if isinstance(gate_results, dict) and isinstance(gate_results.get(gate), dict):
        return dict(gate_results.get(gate) or {})
    # Back-compat: some outputs inline gate info, but we can still degrade gracefully.
    return {}


def _primary_hit_level(evaluation: dict[str, Any]) -> str:
    raw = str(evaluation.get("primary_hit_level") or "").strip().lower()
    if raw:
        return raw
    # Older benchmark outputs did not record primary_hit_level. We'll infer later from provenance.
    return ""


def _infer_primary_hit_level_from_provenance(message: dict[str, Any]) -> str:
    prov = message.get("provenance") if isinstance(message.get("provenance"), dict) else {}
    segs = prov.get("segments") if isinstance(prov.get("segments"), list) else []
    visible_direct: list[dict[str, Any]] = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("evidence_mode") or "").strip().lower() != "direct":
            continue
        # mimic "visible" semantics: hidden locate_policy usually shouldn't be primary.
        if str(seg.get("locate_policy") or "").strip().lower() == "hidden" and not bool(seg.get("must_locate")):
            continue
        visible_direct.append(seg)
    if not visible_direct:
        return "none"
    hit = str(visible_direct[0].get("hit_level") or "").strip().lower()
    return hit if hit in {"exact", "block", "heading", "none"} else "none"


def _minimum_ok(payload: dict[str, Any]) -> bool:
    aq = payload.get("answer_quality")
    if isinstance(aq, dict) and ("minimum_ok" in aq):
        return bool(aq.get("minimum_ok"))
    return False


def _has_uncertainty_disclaimer(text: str) -> bool:
    t = (text or "").lower()
    return any(
        p in t
        for p in [
            "not stated",
            "not specified",
            "i can't find",
            "i cannot find",
            "i couldn't find",
            "not found in",
            "insufficient context",
            "cannot confirm",
            "unclear from the paper",
            "not mentioned",
        ]
    )


def _readability_score(text: str) -> float:
    t = str(text or "").strip()
    if not t:
        return 0.0
    # Simple heuristics: structure + not too short + not obviously full of raw markers.
    has_lines = "\n" in t
    has_list = bool(re.search(r"^\s*(?:-|\*|\d+\.)\s+", t, flags=re.M))
    raw_markers = _RAW_SUPPORT_RE.search(t) or _RAW_CITE_RE.search(t)
    length = len(t)
    score = 2.2
    if length >= 220:
        score += 1.2
    if length >= 600:
        score += 0.6
    if has_lines:
        score += 0.5
    if has_list:
        score += 0.5
    if raw_markers:
        score -= 0.8
    return float(_clamp(score, 0.0, 5.0))


@dataclass
class RubricScore:
    question_hit: float
    evidence_consistency: float
    locate_first_click: float
    uncertainty_handling: float
    readability: float

    @property
    def overall_100(self) -> float:
        return round((self.question_hit + self.evidence_consistency + self.locate_first_click + self.uncertainty_handling + self.readability) / 25.0 * 100.0, 1)


def _score_row(row: dict[str, Any]) -> tuple[RubricScore, dict[str, Any]]:
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    message = row.get("message") if isinstance(row.get("message"), dict) else {}
    answer = str(payload.get("answer") or message.get("content") or "").strip()

    min_ok = _minimum_ok(payload)
    hit = _primary_hit_level(evaluation) or _infer_primary_hit_level_from_provenance(message)
    cite_gate = _extract_gate(evaluation, "citation")
    locate_gate = _extract_gate(evaluation, "locate")
    structured_gate = _extract_gate(evaluation, "structured_markers")

    # 1) 命中问题: proxy with minimum_ok + answer gate
    answer_gate = _extract_gate(evaluation, "answer")
    answer_gate_ok = (str(answer_gate.get("status") or "").upper() != "FAIL") if answer_gate else True
    question_hit = 5.0 if (min_ok and answer_gate_ok and len(answer) >= 80) else (3.2 if len(answer) >= 80 else 1.0)

    # 2) 证据一致性: citation gate + no raw markers
    cite_ok = (str(cite_gate.get("status") or "").upper() != "FAIL") if cite_gate else True
    raw_support = _RAW_SUPPORT_RE.findall(answer)
    raw_cite = _RAW_CITE_RE.findall(answer)
    raw_marker_penalty = 0.8 if (raw_support or raw_cite) else 0.0
    evidence_consistency = 5.0 if (cite_ok and raw_marker_penalty == 0.0) else (3.0 if cite_ok else 1.6)
    evidence_consistency = float(_clamp(evidence_consistency - raw_marker_penalty, 0.0, 5.0))

    # 3) 可定位性(一跳): primary_hit_level
    locate_first_click = {"exact": 5.0, "block": 4.0, "heading": 2.5, "none": 0.8}.get(hit, 1.2)
    if locate_gate and str(locate_gate.get("status") or "").upper() == "FAIL":
        locate_first_click = min(locate_first_click, 2.0)

    # 4) 不确定性处理: if locate weak, prefer explicit disclaimers; otherwise neutral.
    if hit in {"heading", "none"}:
        uncertainty_handling = 4.5 if _has_uncertainty_disclaimer(answer) else 2.0
    else:
        uncertainty_handling = 4.0 if not _has_uncertainty_disclaimer(answer) else 3.4

    # 5) 可读性/结构
    readability = _readability_score(answer)

    notes = {
        "min_ok": bool(min_ok),
        "primary_hit_level": hit,
        "cite_gate": str(cite_gate.get("status") or "").upper() or "UNKNOWN",
        "locate_gate": str(locate_gate.get("status") or "").upper() or "UNKNOWN",
        "raw_support_markers": len(raw_support),
        "raw_cite_markers": len(raw_cite),
        "answer_chars": len(answer),
        "structured_gate": str(structured_gate.get("status") or "").upper() or "UNKNOWN",
    }
    return RubricScore(
        question_hit=round(question_hit, 2),
        evidence_consistency=round(evidence_consistency, 2),
        locate_first_click=round(locate_first_click, 2),
        uncertainty_handling=round(uncertainty_handling, 2),
        readability=round(readability, 2),
    ), notes


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    raise SystemExit("Expected a JSON list (raw_results.json).")


def _render_md(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Paper Guide Rubric Score")
    lines.append("")
    lines.append("| Case | Status | Overall | Hit | min_ok | cite | locate | Readability |")
    lines.append("|---|---|---:|---|---:|---|---|---:|")
    for item in rows:
        case = item.get("case") if isinstance(item.get("case"), dict) else {}
        evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), dict) else {}
        notes = item.get("_rubric_notes") if isinstance(item.get("_rubric_notes"), dict) else {}
        score = item.get("_rubric_score") if isinstance(item.get("_rubric_score"), dict) else {}
        overall = _to_float(score.get("overall_100"))
        read = _to_float(score.get("readability"))
        title = str(case.get("id") or case.get("title") or "")[:42]
        status = str(evaluation.get("status") or "").strip().upper() or "UNKNOWN"
        hit = str(notes.get("primary_hit_level") or "")
        min_ok = "1" if bool(notes.get("min_ok")) else "0"
        cite = str(notes.get("cite_gate") or "")
        loc = str(notes.get("locate_gate") or "")
        lines.append(f"| `{title}` | {status} | {overall:.1f} | {hit} | {min_ok} | {cite} | {loc} | {read:.1f} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rubric-based (LLM-like) scoring for paper_guide benchmark raw_results.json.")
    ap.add_argument("--raw", required=True, help="Path to raw_results.json written by paper_guide_benchmark.py")
    ap.add_argument("--out", default="", help="Optional output .md path (default: none)")
    ap.add_argument("--write-json", action="store_true", help="Write a sidecar JSON with rubric scores next to --raw")
    args = ap.parse_args(argv)

    raw_path = Path(str(args.raw)).expanduser().resolve()
    rows = _load_rows(raw_path)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        score, notes = _score_row(row)
        out = dict(row)
        out["_rubric_score"] = {
            "question_hit": score.question_hit,
            "evidence_consistency": score.evidence_consistency,
            "locate_first_click": score.locate_first_click,
            "uncertainty_handling": score.uncertainty_handling,
            "readability": score.readability,
            "overall_100": score.overall_100,
        }
        out["_rubric_notes"] = dict(notes)
        enriched.append(out)

    md = _render_md(enriched)
    print(md)

    if bool(args.write_json):
        sidecar = raw_path.with_name(raw_path.stem + ".rubric.json")
        sidecar.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {sidecar}")

    if str(args.out or "").strip():
        out_path = Path(str(args.out)).expanduser().resolve()
        out_path.write_text(md, encoding="utf-8")
        print(f"\nwrote {out_path}")

    # Non-failing tool: always return 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
