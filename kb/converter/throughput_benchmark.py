from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Optional

try:
    import fitz
except ImportError:
    fitz = None

from kb.pdf_tools import run_pdf_to_md

from .benchmark import parse_converter_log_metrics, temporary_env, write_csv
from .global_inflight import load_global_inflight_snapshot


_TIMEOUT_EVENT_RE = re.compile(
    r"(?:TimeoutError|timed\s+out|slots\s+saturated|time\s+budget\s+exhausted)",
    flags=re.IGNORECASE,
)
_CRITICAL_QUALITY_FIELDS = (
    "marker_mismatch_count",
    "missing_image_count",
    "table_break_count",
    "math_error_count",
    "mojibake_count",
    "analyzer_error_count",
    "missing_source_page_count",
    "source_page_text_corruption_count",
    "source_page_prose_omission_count",
)


@dataclass(frozen=True)
class ThroughputConfig:
    global_inflight: int = 8
    workers: int = 4
    llm_workers: int = 3
    parallel_documents: int = 2
    repeat: int = 3
    min_throughput_improvement_pct: float = 25.0
    max_per_doc_p95_slowdown_pct: float = 15.0
    dynamic_global_inflight: bool = True


def _pdf_page_count(pdf_path: Path) -> int:
    if fitz is None:
        return 0
    doc = None
    try:
        doc = fitz.open(pdf_path)
        return int(len(doc))
    except Exception:
        return 0
    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass


def _percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    fraction = pos - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _safe_int(mapping: dict, key: str) -> int:
    try:
        return int(mapping.get(key) or 0)
    except Exception:
        return 0


def _load_quality_metrics(output_dir: Path, *, pdf_pages: int) -> dict:
    quality_path = output_dir / "conversion_quality_result.json"
    if not quality_path.exists():
        return {
            "quality_result_present": False,
            "needs_reconvert": True,
            "recommended_action": "missing",
            "issue_codes": ["missing_quality_result"],
            "marker_mismatch_count": 1,
        }
    try:
        payload = json.loads(quality_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {
            "quality_result_present": False,
            "needs_reconvert": True,
            "recommended_action": "invalid",
            "issue_codes": ["invalid_quality_result"],
            "marker_mismatch_count": 1,
        }

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    source = payload.get("source_quality") if isinstance(payload.get("source_quality"), dict) else {}
    repair_plan = payload.get("repair_plan") if isinstance(payload.get("repair_plan"), dict) else {}
    marker_count = _safe_int(metrics, "page_marker_count")
    marker_gaps = _safe_int(metrics, "page_marker_gap_count")
    marker_mismatch = int(marker_gaps > 0 or (int(pdf_pages or 0) > 0 and marker_count != int(pdf_pages)))
    return {
        "quality_result_present": True,
        "needs_reconvert": bool(payload.get("needs_reconvert")),
        "recommended_action": str(payload.get("recommended_action") or "none"),
        "issue_codes": [str(code) for code in list(repair_plan.get("issue_codes") or []) if str(code)],
        "page_marker_count": marker_count,
        "marker_mismatch_count": marker_mismatch,
        "missing_image_count": _safe_int(metrics, "missing_image_count"),
        "table_break_count": sum(
            _safe_int(metrics, key)
            for key in (
                "table_literal_break_count",
                "collapsed_table_row_count",
                "ambiguous_table_break_row_count",
                "fragmented_table_column_count",
                "fragmented_table_duplicate_count",
            )
        ),
        "math_error_count": sum(
            _safe_int(metrics, key)
            for key in (
                "unclosed_display_math_block_count",
                "prose_dominant_display_math_block_count",
                "display_math_markdown_link_count",
            )
        ),
        "mojibake_count": _safe_int(metrics, "mojibake_count"),
        "analyzer_error_count": _safe_int(metrics, "analyzer_error_count"),
        "analyzer_warning_count": _safe_int(metrics, "analyzer_warning_count"),
        "missing_source_page_count": _safe_int(source, "missing_source_page_count"),
        "source_page_text_corruption_count": _safe_int(source, "source_page_text_corruption_count"),
        "source_page_prose_omission_count": _safe_int(source, "source_page_prose_omission_count"),
        "min_source_page_coverage": float(source.get("min_source_page_coverage") or 0.0),
    }


def _run_document(
    *,
    pdf_path: Path,
    case_root: Path,
    case_id: str,
    mode: str,
    repeat_index: int,
    max_active_conversions: int,
    global_inflight_coordinator: Path | None = None,
    global_inflight_limit: int | None = None,
    runner: Callable = run_pdf_to_md,
) -> tuple[dict, list[dict]]:
    # Keep the harness-owned path short. The product converter appends the full
    # PDF stem to ``out_root`` and long paper titles otherwise exceed legacy
    # Windows path limits before conversion starts or while writing sidecars.
    case_dir = case_root / str(case_id)
    if case_dir.exists() and any(case_dir.iterdir()):
        raise FileExistsError(f"benchmark case already exists: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "converter.log"
    output_root = case_dir / "output"
    log_lock = threading.Lock()

    def _progress(_done: int, _total: int, message: str = "") -> None:
        text = str(message or "").strip()
        if not text:
            return
        with log_lock:
            with log_path.open("a", encoding="utf-8", errors="replace") as fp:
                fp.write(text + "\n")

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    ok = False
    output_value: object = ""
    error = ""
    try:
        ok, output_value = runner(
            pdf_path=pdf_path,
            out_root=output_root,
            no_llm=False,
            keep_debug=False,
            eq_image_fallback=False,
            progress_cb=_progress,
            speed_mode="normal",
            max_active_conversions=max_active_conversions,
            global_inflight_coordinator=global_inflight_coordinator,
            global_inflight_owner=f"{mode}-{repeat_index}-{case_id}",
            global_inflight_limit=global_inflight_limit,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_s = time.perf_counter() - t0

    output_dir = Path(output_value) if ok and str(output_value or "").strip() else output_root / pdf_path.stem
    pdf_pages = _pdf_page_count(pdf_path)
    log_metrics, page_metrics = parse_converter_log_metrics(log_path)
    quality = _load_quality_metrics(output_dir, pdf_pages=pdf_pages)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        log_text = ""
    timeout_event_count = len(_TIMEOUT_EVENT_RE.findall(log_text))

    result = {
        "mode": str(mode),
        "repeat": int(repeat_index),
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "pdf_pages": int(pdf_pages),
        "ok": bool(ok),
        "elapsed_s": round(float(elapsed_s), 4),
        "started_at": started_at,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "error": error if error else ("" if ok else str(output_value or "conversion failed")),
        "max_active_conversions": int(max_active_conversions),
        "dynamic_global_inflight": bool(global_inflight_coordinator),
        "timeout_event_count": int(timeout_event_count),
    }
    result.update(log_metrics)
    result.update(quality)
    for page_row in page_metrics:
        page_row.update(
            {
                "mode": str(mode),
                "repeat": int(repeat_index),
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
            }
        )
    return result, page_metrics


def _quality_regressions(serial_run: dict, parallel_run: dict) -> list[str]:
    reasons: list[str] = []
    for key in _CRITICAL_QUALITY_FIELDS:
        if _safe_int(parallel_run, key) > _safe_int(serial_run, key):
            reasons.append(key)
    action_rank = {"none": 0, "autofix": 1, "review": 1, "reconvert": 2, "missing": 3, "invalid": 3}
    serial_action = action_rank.get(str(serial_run.get("recommended_action") or "none").lower(), 2)
    parallel_action = action_rank.get(str(parallel_run.get("recommended_action") or "none").lower(), 2)
    if parallel_action > serial_action:
        reasons.append("recommended_action")
    return reasons


def _summarize(
    *,
    experiments: list[dict],
    document_runs: list[dict],
    config: ThroughputConfig,
    pdf_paths: list[Path],
) -> dict:
    experiment_map = {
        (str(item.get("mode")), int(item.get("repeat") or 0)): item
        for item in experiments
    }
    run_map = {
        (str(item.get("mode")), int(item.get("repeat") or 0), str(item.get("pdf_path") or "")): item
        for item in document_runs
    }
    paired_experiments: list[dict] = []
    throughput_improvements: list[float] = []
    quality_regression_pairs: list[dict] = []
    for repeat_index in range(1, int(config.repeat) + 1):
        serial = experiment_map.get(("serial", repeat_index))
        parallel = experiment_map.get(("parallel", repeat_index))
        if not serial or not parallel:
            continue
        serial_elapsed = float(serial.get("elapsed_s") or 0.0)
        parallel_elapsed = float(parallel.get("elapsed_s") or 0.0)
        improvement = (
            100.0 * (serial_elapsed - parallel_elapsed) / serial_elapsed
            if serial_elapsed > 0
            else 0.0
        )
        throughput_improvements.append(improvement)
        paired_experiments.append(
            {
                "repeat": repeat_index,
                "serial_elapsed_s": round(serial_elapsed, 4),
                "parallel_elapsed_s": round(parallel_elapsed, 4),
                "throughput_improvement_pct": round(improvement, 4),
            }
        )

    per_pdf: list[dict] = []
    per_doc_p95_slowdowns: list[float] = []
    for pdf_path in pdf_paths:
        serial_values: list[float] = []
        parallel_values: list[float] = []
        paired_slowdowns: list[float] = []
        for repeat_index in range(1, int(config.repeat) + 1):
            serial = run_map.get(("serial", repeat_index, str(pdf_path)))
            parallel = run_map.get(("parallel", repeat_index, str(pdf_path)))
            if not serial or not parallel:
                continue
            serial_elapsed = float(serial.get("elapsed_s") or 0.0)
            parallel_elapsed = float(parallel.get("elapsed_s") or 0.0)
            serial_values.append(serial_elapsed)
            parallel_values.append(parallel_elapsed)
            paired_slowdowns.append(
                100.0 * (parallel_elapsed - serial_elapsed) / serial_elapsed
                if serial_elapsed > 0
                else 0.0
            )
            regressions = _quality_regressions(serial, parallel)
            if regressions:
                quality_regression_pairs.append(
                    {
                        "repeat": repeat_index,
                        "pdf_name": pdf_path.name,
                        "reasons": regressions,
                    }
                )
        serial_p95 = _percentile(serial_values, 0.95)
        parallel_p95 = _percentile(parallel_values, 0.95)
        p95_slowdown = (
            100.0 * (parallel_p95 - serial_p95) / serial_p95
            if serial_p95 > 0
            else 0.0
        )
        per_doc_p95_slowdowns.append(p95_slowdown)
        per_pdf.append(
            {
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "pairs": len(paired_slowdowns),
                "serial_median_s": round(float(median(serial_values)), 4) if serial_values else 0.0,
                "parallel_median_s": round(float(median(parallel_values)), 4) if parallel_values else 0.0,
                "paired_median_slowdown_pct": round(float(median(paired_slowdowns)), 4) if paired_slowdowns else 0.0,
                "serial_p95_s": round(serial_p95, 4),
                "parallel_p95_s": round(parallel_p95, 4),
                "p95_slowdown_pct": round(p95_slowdown, 4),
            }
        )

    serial_runs = [item for item in document_runs if str(item.get("mode")) == "serial"]
    parallel_runs = [item for item in document_runs if str(item.get("mode")) == "parallel"]
    serial_timeouts = sum(_safe_int(item, "timeout_event_count") for item in serial_runs)
    parallel_timeouts = sum(_safe_int(item, "timeout_event_count") for item in parallel_runs)
    serial_retries = sum(
        _safe_int(item, "empty_retry_count") + _safe_int(item, "math_retry_count")
        for item in serial_runs
    )
    parallel_retries = sum(
        _safe_int(item, "empty_retry_count") + _safe_int(item, "math_retry_count")
        for item in parallel_runs
    )
    serial_fallbacks = sum(_safe_int(item, "fallback_count") for item in serial_runs)
    parallel_fallbacks = sum(_safe_int(item, "fallback_count") for item in parallel_runs)
    median_improvement = float(median(throughput_improvements)) if throughput_improvements else 0.0
    worst_p95_slowdown = max(per_doc_p95_slowdowns) if per_doc_p95_slowdowns else 0.0
    all_runs_ok = len(document_runs) == int(config.repeat) * len(pdf_paths) * 2 and all(
        bool(item.get("ok")) for item in document_runs
    )
    all_markers_exact = all(_safe_int(item, "marker_mismatch_count") == 0 for item in document_runs)
    gate_checks = {
        "all_runs_ok": bool(all_runs_ok),
        "throughput_improvement": median_improvement >= float(config.min_throughput_improvement_pct),
        "per_doc_p95_slowdown": worst_p95_slowdown <= float(config.max_per_doc_p95_slowdown_pct),
        "timeouts_not_increased": parallel_timeouts <= serial_timeouts,
        "retries_not_increased": parallel_retries <= serial_retries,
        "fallbacks_not_increased": parallel_fallbacks <= serial_fallbacks,
        "page_markers_exact": bool(all_markers_exact),
        "no_quality_regression": not quality_regression_pairs,
    }
    return {
        "paired_experiments": paired_experiments,
        "per_pdf": per_pdf,
        "median_throughput_improvement_pct": round(median_improvement, 4),
        "worst_per_doc_p95_slowdown_pct": round(worst_p95_slowdown, 4),
        "serial_timeout_events": serial_timeouts,
        "parallel_timeout_events": parallel_timeouts,
        "serial_retry_events": serial_retries,
        "parallel_retry_events": parallel_retries,
        "serial_fallbacks": serial_fallbacks,
        "parallel_fallbacks": parallel_fallbacks,
        "quality_regression_pairs": quality_regression_pairs,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }


def _write_payload(out_root: Path, payload: dict) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "throughput_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_root / "throughput_experiments.csv", payload.get("experiments") or [])
    write_csv(out_root / "throughput_document_runs.csv", payload.get("document_runs") or [])
    write_csv(out_root / "throughput_page_metrics.csv", payload.get("page_metrics") or [])
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    write_csv(out_root / "throughput_paired_experiments.csv", summary.get("paired_experiments") or [])
    write_csv(out_root / "throughput_summary_by_pdf.csv", summary.get("per_pdf") or [])


def run_throughput_suite(
    *,
    pdf_paths: list[Path],
    out_root: Path,
    config: ThroughputConfig,
    runner: Callable = run_pdf_to_md,
    fail_fast: bool = False,
) -> dict:
    if len(pdf_paths) < 2:
        raise ValueError("throughput benchmark requires at least two PDFs")
    resolved_pdfs = [Path(path).expanduser().resolve() for path in pdf_paths]
    if any(not path.is_file() for path in resolved_pdfs):
        raise FileNotFoundError("one or more throughput benchmark PDFs do not exist")
    parallel_documents = max(2, min(4, int(config.parallel_documents or 2)))
    if parallel_documents > len(resolved_pdfs):
        raise ValueError("parallel document count cannot exceed the number of benchmark PDFs")

    experiments: list[dict] = []
    document_runs: list[dict] = []
    page_metrics: list[dict] = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    suite_t0 = time.perf_counter()
    overrides = {
        "KB_LLM_MAX_INFLIGHT": str(max(1, min(32, int(config.global_inflight)))),
        "KB_PDF_WORKERS": str(max(1, int(config.workers))),
        "KB_PDF_LLM_WORKERS": str(max(1, int(config.llm_workers))),
        "KB_PDF_STAGE_TIMINGS": "1",
    }

    with temporary_env(overrides):
        for repeat_index in range(1, int(config.repeat) + 1):
            ordered_modes = ["serial", "parallel"] if repeat_index % 2 else ["parallel", "serial"]
            for mode in ordered_modes:
                print(f"[THROUGHPUT] mode={mode} run={repeat_index}/{config.repeat}", flush=True)
                experiment_root = out_root / mode / f"run_{repeat_index:02d}"
                global_coordinator = (
                    experiment_root / "_global_inflight"
                    if bool(config.dynamic_global_inflight)
                    else None
                )
                experiment_t0 = time.perf_counter()
                current_results: list[tuple[dict, list[dict]]] = []
                if mode == "serial":
                    for document_index, pdf_path in enumerate(resolved_pdfs, start=1):
                        current_results.append(
                            _run_document(
                                pdf_path=pdf_path,
                                case_root=experiment_root,
                                case_id=f"doc_{document_index:02d}",
                                mode=mode,
                                repeat_index=repeat_index,
                                max_active_conversions=1,
                                global_inflight_coordinator=global_coordinator,
                                global_inflight_limit=int(config.global_inflight),
                                runner=runner,
                            )
                        )
                else:
                    with ThreadPoolExecutor(max_workers=parallel_documents) as executor:
                        futures = [
                            executor.submit(
                                _run_document,
                                pdf_path=pdf_path,
                                case_root=experiment_root,
                                case_id=f"doc_{document_index:02d}",
                                mode=mode,
                                repeat_index=repeat_index,
                                max_active_conversions=parallel_documents,
                                global_inflight_coordinator=global_coordinator,
                                global_inflight_limit=int(config.global_inflight),
                                runner=runner,
                            )
                            for document_index, pdf_path in enumerate(resolved_pdfs, start=1)
                        ]
                        for future in as_completed(futures):
                            current_results.append(future.result())
                experiment_elapsed = time.perf_counter() - experiment_t0
                current_runs = [item[0] for item in current_results]
                current_runs.sort(key=lambda item: str(item.get("pdf_name") or "").lower())
                document_runs.extend(current_runs)
                for _result, rows in current_results:
                    page_metrics.extend(rows)
                experiment = {
                    "mode": mode,
                    "repeat": repeat_index,
                    "elapsed_s": round(float(experiment_elapsed), 4),
                    "ok": all(bool(item.get("ok")) for item in current_runs),
                    "document_count": len(current_runs),
                    "sum_document_elapsed_s": round(sum(float(item.get("elapsed_s") or 0.0) for item in current_runs), 4),
                    "max_document_elapsed_s": round(max((float(item.get("elapsed_s") or 0.0) for item in current_runs), default=0.0), 4),
                    "parallel_documents": 1 if mode == "serial" else parallel_documents,
                    "dynamic_global_inflight": bool(global_coordinator),
                }
                global_snapshot = (
                    load_global_inflight_snapshot(global_coordinator)
                    if global_coordinator is not None
                    else {}
                )
                if global_snapshot:
                    experiment.update(
                        {
                            "global_configured_limit": _safe_int(global_snapshot, "configured_limit"),
                            "global_effective_limit": _safe_int(global_snapshot, "effective_limit"),
                            "global_min_effective_limit": _safe_int(global_snapshot, "min_effective_limit"),
                            "global_pressure_events": _safe_int(global_snapshot, "pressure_events"),
                            "global_rate_limited_events": _safe_int(global_snapshot, "rate_limited_events"),
                            "global_timeout_events": _safe_int(global_snapshot, "timeout_events"),
                            "global_limit_reductions": _safe_int(global_snapshot, "limit_reductions"),
                            "global_limit_recoveries": _safe_int(global_snapshot, "limit_recoveries"),
                        }
                    )
                experiments.append(experiment)
                checkpoint = {
                    "started_at": started_at,
                    "elapsed_s": round(time.perf_counter() - suite_t0, 4),
                    "pdfs": [str(path) for path in resolved_pdfs],
                    "config": config.__dict__,
                    "experiments": experiments,
                    "document_runs": document_runs,
                    "page_metrics": page_metrics,
                    "summary": _summarize(
                        experiments=experiments,
                        document_runs=document_runs,
                        config=config,
                        pdf_paths=resolved_pdfs,
                    ),
                }
                _write_payload(out_root, checkpoint)
                print(
                    f"[THROUGHPUT] {'OK' if experiment['ok'] else 'FAIL'} "
                    f"mode={mode} elapsed={experiment_elapsed:.2f}s",
                    flush=True,
                )
                if fail_fast and not bool(experiment["ok"]):
                    raise RuntimeError(f"throughput experiment failed: mode={mode} repeat={repeat_index}")

    payload = {
        "started_at": started_at,
        "elapsed_s": round(time.perf_counter() - suite_t0, 4),
        "pdfs": [str(path) for path in resolved_pdfs],
        "config": config.__dict__,
        "experiments": experiments,
        "document_runs": document_runs,
        "page_metrics": page_metrics,
        "summary": _summarize(
            experiments=experiments,
            document_runs=document_runs,
            config=config,
            pdf_paths=resolved_pdfs,
        ),
    }
    _write_payload(out_root, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark serial vs bounded multi-PDF product-path conversion throughput")
    parser.add_argument("pdfs", nargs="+", help="Two or more fixed PDF files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--global-inflight", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--llm-workers", type=int, default=3)
    parser.add_argument("--parallel-documents", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--min-throughput-improvement-pct", type=float, default=25.0)
    parser.add_argument("--max-per-doc-p95-slowdown-pct", type=float, default=15.0)
    parser.add_argument(
        "--static-budget-split",
        action="store_true",
        help="Use the legacy per-document static split instead of the dynamic global coordinator",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ThroughputConfig(
        global_inflight=max(1, min(32, int(args.global_inflight))),
        workers=max(1, int(args.workers)),
        llm_workers=max(1, int(args.llm_workers)),
        parallel_documents=max(2, min(4, int(args.parallel_documents))),
        repeat=max(1, int(args.repeat)),
        min_throughput_improvement_pct=float(args.min_throughput_improvement_pct),
        max_per_doc_p95_slowdown_pct=float(args.max_per_doc_p95_slowdown_pct),
        dynamic_global_inflight=not bool(args.static_budget_split),
    )
    payload = run_throughput_suite(
        pdf_paths=[Path(value) for value in args.pdfs],
        out_root=Path(args.out_dir).expanduser().resolve(),
        config=config,
        fail_fast=bool(args.fail_fast),
    )
    summary = payload["summary"]
    print(
        json.dumps(
            {
                "result": str(Path(args.out_dir).expanduser().resolve() / "throughput_results.json"),
                "gate_passed": bool(summary.get("gate_passed")),
                "median_throughput_improvement_pct": summary.get("median_throughput_improvement_pct"),
                "worst_per_doc_p95_slowdown_pct": summary.get("worst_per_doc_p95_slowdown_pct"),
                "gate_checks": summary.get("gate_checks"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if bool(args.fail_fast) and not bool(summary.get("gate_passed")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
