from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SAMPLES_PATH = Path("test_results/research_agent_answer_samples.jsonl")
DEFAULT_LABELS_PATH = Path("test_results/research_agent_answer_labels.jsonl")
DEFAULT_REVIEWED_PATH = Path("test_results/research_agent_answer_reviewed.jsonl")

VALID_SOURCE_BLENDS = {
    "local_grounded",
    "hybrid_local_external",
    "external_academic",
    "general_llm",
}

VALID_REVIEW_STATUSES = {"needs_review", "accepted", "rejected", "skip"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        item["_line_no"] = line_no
        rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            cleaned = {key: value for key, value in row.items() if key != "_line_no"}
            fh.write(json.dumps(cleaned, ensure_ascii=False, sort_keys=True) + "\n")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clip(value: Any, max_len: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def _source_blend(case: dict[str, Any]) -> str:
    blend = str(case.get("source_blend") or case.get("answer_source_blend") or "").strip()
    if blend in VALID_SOURCE_BLENDS:
        return blend
    trace = case.get("agent_trace")
    if isinstance(trace, dict):
        summary = trace.get("summary")
        if isinstance(summary, dict):
            blend = str(summary.get("answer_source_blend") or summary.get("source_blend") or "").strip()
            if blend in VALID_SOURCE_BLENDS:
                return blend
    return ""


def _expected_notice(source_blend: str) -> str:
    if source_blend == "hybrid_local_external":
        return "hybrid_notice"
    if source_blend == "external_academic":
        return "external_not_local"
    return "none"


def _evidence_preview(case: dict[str, Any], *, max_items: int, max_text_chars: int) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for hit in _as_list(case.get("evidence_hits"))[:max_items]:
        if not isinstance(hit, dict):
            continue
        meta = _as_dict(hit.get("meta"))
        preview.append(
            {
                "source_name": str(meta.get("source_name") or ""),
                "source_path": str(meta.get("source_path") or ""),
                "heading_path": meta.get("heading_path") if isinstance(meta.get("heading_path"), list) else [],
                "text_preview": _clip(hit.get("text"), max_text_chars),
            }
        )
    return preview


def _labels_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("id") or "").strip()
        if case_id:
            out[case_id] = row
    return out


def prepare_review_labels(
    *,
    samples_path: Path,
    labels_path: Path,
    max_answer_chars: int = 1200,
    max_evidence_chars: int = 500,
) -> dict[str, Any]:
    samples = _load_jsonl(Path(samples_path))
    existing = _labels_by_id(_load_jsonl(Path(labels_path)))
    review_rows: list[dict[str, Any]] = []

    for sample in samples:
        case_id = str(sample.get("id") or "").strip()
        if not case_id:
            continue
        prior = existing.get(case_id, {})
        observed_blend = _source_blend(sample)
        expected_blend = str(prior.get("expected_source_blend") or "").strip()
        expected_notice = str(prior.get("expected_user_notice") or sample.get("expected_user_notice") or "").strip()
        if not expected_notice:
            expected_notice = _expected_notice(expected_blend or observed_blend)
        status = str(prior.get("review_status") or "needs_review").strip()
        if status not in VALID_REVIEW_STATUSES:
            status = "needs_review"
        review_rows.append(
            {
                "id": case_id,
                "review_status": status,
                "query": str(sample.get("query") or ""),
                "answer_preview": _clip(sample.get("answer"), max_answer_chars),
                "evidence_preview": _evidence_preview(
                    sample,
                    max_items=5,
                    max_text_chars=max_evidence_chars,
                ),
                "source_blend_observed": observed_blend,
                "expected_source_blend": expected_blend,
                "expected_answer_points": _as_list(prior.get("expected_answer_points")),
                "expected_source_keywords": _as_list(prior.get("expected_source_keywords")),
                "expected_user_notice": expected_notice,
                "expected_retrieval_hit": prior.get(
                    "expected_retrieval_hit",
                    bool(sample.get("evidence_hits")),
                ),
                "should_use_local_evidence": prior.get("should_use_local_evidence"),
                "external_fallback_allowed": prior.get(
                    "external_fallback_allowed",
                    sample.get("external_fallback_allowed"),
                ),
                "review_notes": str(prior.get("review_notes") or ""),
            }
        )

    _write_jsonl(Path(labels_path), review_rows)
    return {
        "ok": True,
        "samples_path": str(samples_path),
        "labels_path": str(labels_path),
        "sample_count": len(samples),
        "label_count": len(review_rows),
        "preserved_label_count": len([row for row in review_rows if row["id"] in existing]),
    }


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _validate_label(
    *,
    case_id: str,
    label: dict[str, Any],
    allow_empty_points: bool,
) -> list[str]:
    errors: list[str] = []
    status = str(label.get("review_status") or "").strip()
    if status not in VALID_REVIEW_STATUSES:
        errors.append(f"{case_id}: review_status must be one of {', '.join(sorted(VALID_REVIEW_STATUSES))}")
    expected_blend = str(label.get("expected_source_blend") or "").strip()
    if not expected_blend:
        errors.append(f"{case_id}: accepted label requires expected_source_blend")
    elif expected_blend not in VALID_SOURCE_BLENDS:
        errors.append(f"{case_id}: expected_source_blend must be one of {', '.join(sorted(VALID_SOURCE_BLENDS))}")
    points = [str(item or "").strip() for item in _as_list(label.get("expected_answer_points")) if str(item or "").strip()]
    if not allow_empty_points and not points:
        errors.append(f"{case_id}: accepted label requires at least one expected_answer_points item")
    if not isinstance(label.get("expected_source_keywords", []), list):
        errors.append(f"{case_id}: expected_source_keywords must be a list")
    return errors


def merge_review_labels(
    *,
    samples_path: Path,
    labels_path: Path,
    out_path: Path,
    allow_empty_points: bool = False,
) -> dict[str, Any]:
    samples = _load_jsonl(Path(samples_path))
    labels = _labels_by_id(_load_jsonl(Path(labels_path)))
    errors: list[str] = []
    reviewed: list[dict[str, Any]] = []
    missing_label_count = 0
    skipped_label_count = 0

    for sample in samples:
        case_id = str(sample.get("id") or "").strip()
        if not case_id:
            continue
        label = labels.get(case_id)
        if not label:
            missing_label_count += 1
            continue
        status = str(label.get("review_status") or "").strip()
        if status != "accepted":
            skipped_label_count += 1
            continue
        label_errors = _validate_label(case_id=case_id, label=label, allow_empty_points=allow_empty_points)
        if label_errors:
            errors.extend(label_errors)
            continue

        expected_blend = str(label.get("expected_source_blend") or "").strip()
        expected_points = [
            str(item or "").strip()
            for item in _as_list(label.get("expected_answer_points"))
            if str(item or "").strip()
        ]
        expected_sources = [
            str(item or "").strip()
            for item in _as_list(label.get("expected_source_keywords"))
            if str(item or "").strip()
        ]
        notice = str(label.get("expected_user_notice") or "").strip() or _expected_notice(expected_blend)
        has_evidence = bool(sample.get("evidence_hits"))
        should_use_local = _bool_or_default(
            label.get("should_use_local_evidence"),
            expected_blend in {"local_grounded", "hybrid_local_external"} and has_evidence,
        )

        row = dict(sample)
        row.update(
            {
                "sample_kind": "real_chat_reviewed",
                "replay_unlabeled": False,
                "review_status": "accepted",
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "review_notes": str(label.get("review_notes") or ""),
                "expected_source_blend": expected_blend,
                "expected_answer_points": expected_points,
                "expected_source_keywords": expected_sources,
                "expected_user_notice": notice,
                "expected_retrieval_hit": _bool_or_default(
                    label.get("expected_retrieval_hit"),
                    has_evidence,
                ),
                "should_use_local_evidence": should_use_local,
                "external_fallback_allowed": _bool_or_default(
                    label.get("external_fallback_allowed"),
                    expected_blend in {"hybrid_local_external", "external_academic", "general_llm"},
                ),
            }
        )
        reviewed.append(row)

    _write_jsonl(Path(out_path), reviewed)
    return {
        "ok": not errors,
        "samples_path": str(samples_path),
        "labels_path": str(labels_path),
        "out_path": str(out_path),
        "sample_count": len(samples),
        "reviewed_case_count": len(reviewed),
        "missing_label_count": missing_label_count,
        "skipped_label_count": skipped_label_count,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and merge human labels for real Research Agent replay samples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create or refresh a reviewer labels JSONL file.")
    prepare.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    prepare.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    prepare.add_argument("--max-answer-chars", type=int, default=1200)
    prepare.add_argument("--max-evidence-chars", type=int, default=500)

    merge = subparsers.add_parser("merge", help="Merge accepted labels into an eval-ready reviewed JSONL.")
    merge.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    merge.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    merge.add_argument("--out", type=Path, default=DEFAULT_REVIEWED_PATH)
    merge.add_argument(
        "--allow-empty-points",
        action="store_true",
        help="Allow accepted labels without expected_answer_points.",
    )

    args = parser.parse_args()
    if args.command == "prepare":
        summary = prepare_review_labels(
            samples_path=args.samples,
            labels_path=args.labels,
            max_answer_chars=args.max_answer_chars,
            max_evidence_chars=args.max_evidence_chars,
        )
    else:
        summary = merge_review_labels(
            samples_path=args.samples,
            labels_path=args.labels,
            out_path=args.out,
            allow_empty_points=args.allow_empty_points,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
