from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kb.retrieval_eval_fixture import sha256_file


DEFAULT_FIXTURE = REPO_ROOT / "docs" / "retrieval_user_journey_expansion_v1.json"
DEFAULT_BASELINE = REPO_ROOT / "docs" / "retrieval_user_journey_bm25_baseline_v1.json"
EXPECTED_BASELINE_SHA256 = (
    "28748a9d590a637dcd8e2847f2ba9efc437b069ee4769feb95f3821986576fca"
)
EXPECTED_CHANNELS = ("bm25_raw", "bm25_product_queries")
EXPECTED_MODES = ("scoped", "full_library")
EXPECTED_KS = [1, 3, 5, 10, 20]


def _rounded_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def validate_baseline(payload: dict[str, Any], fixture: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cases = list(fixture.get("cases") or [])
    expected_ids = [str(case.get("id") or "") for case in cases]
    corpus = fixture.get("corpus") or {}
    if (
        payload.get("schema_version") != 1
        or payload.get("report_kind") != "committed_baseline_summary"
        or payload.get("fixture_id") != fixture.get("id")
        or payload.get("case_count") != len(cases)
    ):
        errors.append("BM25 baseline identity changed")
    if payload.get("fixture") != {
        "path": "docs/retrieval_user_journey_expansion_v1.json",
        "sha256": sha256_file(DEFAULT_FIXTURE),
        "sha256_newlines": "normalized_lf",
    }:
        errors.append("BM25 baseline fixture identity changed")
    if payload.get("corpus") != {
        "document_count": corpus.get("document_count"),
        "chunk_count": corpus.get("chunk_count"),
        "fingerprint": corpus.get("fingerprint"),
    }:
        errors.append("BM25 baseline corpus identity changed")

    channels = payload.get("channels") or {}
    if tuple(channels) != EXPECTED_CHANNELS:
        errors.append("BM25 baseline channel set or order changed")
        return errors
    for channel_name in EXPECTED_CHANNELS:
        channel = channels.get(channel_name) or {}
        if channel.get("channel") != channel_name:
            errors.append(f"BM25 channel identity changed: {channel_name}")
        modes = channel.get("modes") or {}
        if tuple(modes) != EXPECTED_MODES:
            errors.append(f"BM25 mode set or order changed: {channel_name}")
            continue
        for mode_name in EXPECTED_MODES:
            mode = modes.get(mode_name) or {}
            rows = list(mode.get("cases") or [])
            row_ids = [str(row.get("id") or "") for row in rows]
            if row_ids != expected_ids or mode.get("case_count") != len(cases):
                errors.append(
                    f"BM25 case membership changed: {channel_name}/{mode_name}"
                )
                continue
            if mode.get("ks") != EXPECTED_KS:
                errors.append(
                    f"BM25 reporting depths changed: {channel_name}/{mode_name}"
                )
            if mode.get("translated_cases") != sum(
                bool(row.get("used_translation")) for row in rows
            ):
                errors.append(
                    f"BM25 translation count changed: {channel_name}/{mode_name}"
                )
            metric_keys = sorted(
                {str(key) for row in rows for key in dict(row.get("metrics") or {})}
            )
            aggregate = mode.get("metrics") or {}
            for key in metric_keys:
                expected = _rounded_mean(
                    [float((row.get("metrics") or {})[key]) for row in rows]
                )
                if aggregate.get(key) != expected:
                    errors.append(
                        f"BM25 metric arithmetic changed: {channel_name}/{mode_name}/{key}"
                    )
            if mode_name == "scoped" and any(
                float((row.get("metrics") or {}).get("wrong_source_rate@5") or 0.0)
                != 0.0
                for row in rows
            ):
                errors.append(f"scoped BM25 leaked another source: {channel_name}")
    serialized = json.dumps(payload, ensure_ascii=False)
    if "ranked_chunk_ids" in serialized or "ranked_doc_ids" in serialized:
        errors.append("committed BM25 baseline contains result-list payloads")
    return errors


def main() -> int:
    try:
        fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
        baseline = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] user-journey BM25 baseline cannot be read: {type(exc).__name__}")
        return 2
    errors = validate_baseline(baseline, fixture)
    if sha256_file(DEFAULT_BASELINE) != EXPECTED_BASELINE_SHA256:
        errors.append("frozen BM25 baseline file hash changed")
    if errors:
        print("[FAIL] " + "; ".join(errors))
        return 1
    print("[OK] frozen 18-case product/raw BM25 baseline and metric arithmetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
