from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

try:
    import fitz
except ImportError:
    fitz = None

from kb.config import load_settings

from .config import ConvertConfig, LlmConfig
from .llm_worker import LLMWorker
from .pipeline import PDFConverter


VALID_SPEED_MODES = {"no_llm", "normal", "ultra_fast"}

_STEP_TIMING_RE = re.compile(
    r"^\s*\[Page\s+(?P<page>\d+)\]\s+Step\s+(?P<step>\d+)\s+\((?P<label>[^)]+)\):\s+"
    r"(?P<seconds>\d+(?:\.\d+)?)s(?:,\s*(?P<extra>.*))?$"
)
_PAGE_TOTAL_RE = re.compile(
    r"^\s*\[Page\s+(?P<page>\d+)\]\s+TOTAL:\s+(?P<seconds>\d+(?:\.\d+)?)s\s*$"
)
_REFS_FLAG_RE = re.compile(r"(?:^|,\s*)references=(?P<flag>[01])(?:\b|$)")
_REFS_COLUMN_MODE_RE = re.compile(
    r"^\[VISION_DIRECT\]\[REFS\]\s+page\s+(?P<page>\d+):\s+column mode enabled\b",
    flags=re.IGNORECASE,
)
_REFS_CROP_DONE_RE = re.compile(
    r"^\[VISION_DIRECT\]\[REFS\]\s+page\s+(?P<page>\d+)\s+crop\s+\d+/\d+\s+done\s+\((?P<seconds>\d+(?:\.\d+)?)s,",
    flags=re.IGNORECASE,
)
_LAYOUT_MODE_RE = re.compile(
    r"^\[VISION_DIRECT\]\[LAYOUT\]\s+page\s+(?P<page>\d+):\s+structured crop mode enabled\b",
    flags=re.IGNORECASE,
)
_LAYOUT_CROP_DONE_RE = re.compile(
    r"^\[VISION_DIRECT\]\[LAYOUT\]\s+page\s+(?P<page>\d+)\s+crop\s+\d+/\d+\s+\([^)]+\)\s+done\s+\((?P<seconds>\d+(?:\.\d+)?)s,",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    speed_mode: str
    no_llm_workers: int | None = None
    llm_page_workers: int | None = None
    llm_workers: int | None = None
    max_inflight: int | None = None
    vision_dpi: int | None = None
    vision_compress: int | None = None
    stage_timings: bool = False


def _parse_bool(value: str) -> bool:
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _parse_optional_int(fields: dict[str, str], key: str) -> int | None:
    raw = fields.get(key, "")
    if not str(raw).strip():
        return None
    value = int(str(raw).strip())
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _slugify(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    out = out.strip(".-")
    return out or "item"


def _stage_label_key(step_no: int, label: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", str(label or "").strip().lower()).strip("_")
    if not base:
        base = "unknown"
    return f"step_{int(step_no)}_{base}_s"


def _new_page_metric(page_no: int) -> dict:
    return {
        "page": int(page_no),
        "is_references_page": 0,
        "uses_refs_column_mode": 0,
        "uses_layout_crop_mode": 0,
    }


def _percentile(values: list[float], pct: float) -> float:
    vals = sorted(float(v) for v in values if float(v) >= 0.0)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    rank = max(0.0, min(1.0, float(pct))) * (len(vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    frac = rank - lo
    return float(vals[lo] + ((vals[hi] - vals[lo]) * frac))


def parse_converter_log_metrics(log_path: Path) -> tuple[dict, list[dict]]:
    page_rows: dict[int, dict] = {}
    refs_column_pages: set[int] = set()
    layout_crop_pages: set[int] = set()
    refs_crop_times: list[float] = []
    layout_crop_times: list[float] = []
    empty_retry_count = 0
    math_retry_count = 0
    fallback_count = 0

    if not Path(log_path).exists():
        return {
            "page_metric_count": 0,
            "page_timed_pages": 0,
            "page_total_avg_s": 0.0,
            "page_total_p50_s": 0.0,
            "page_total_p90_s": 0.0,
            "vision_step6_pages": 0,
            "vision_step6_avg_s": 0.0,
            "vision_step6_p50_s": 0.0,
            "vision_step6_p90_s": 0.0,
            "references_pages": 0,
            "refs_column_pages": 0,
            "refs_crop_call_count": 0,
            "refs_crop_avg_s": 0.0,
            "layout_crop_pages": 0,
            "layout_crop_call_count": 0,
            "layout_crop_avg_s": 0.0,
            "empty_retry_count": 0,
            "math_retry_count": 0,
            "fallback_count": 0,
        }, []

    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        lines = []

    for raw_line in lines:
        line = str(raw_line or "").rstrip()
        if not line:
            continue

        match = _STEP_TIMING_RE.match(line)
        if match:
            page_no = int(match.group("page"))
            row = page_rows.setdefault(page_no, _new_page_metric(page_no))
            step_no = int(match.group("step"))
            label = str(match.group("label") or "")
            seconds = round(float(match.group("seconds")), 4)
            row[_stage_label_key(step_no, label)] = seconds
            if step_no == 1:
                extra = str(match.group("extra") or "")
                ref_match = _REFS_FLAG_RE.search(extra)
                if ref_match:
                    row["is_references_page"] = int(ref_match.group("flag") == "1")
            continue

        match = _PAGE_TOTAL_RE.match(line)
        if match:
            page_no = int(match.group("page"))
            row = page_rows.setdefault(page_no, _new_page_metric(page_no))
            row["page_total_s"] = round(float(match.group("seconds")), 4)
            continue

        match = _REFS_COLUMN_MODE_RE.match(line)
        if match:
            page_no = int(match.group("page"))
            refs_column_pages.add(page_no)
            row = page_rows.setdefault(page_no, _new_page_metric(page_no))
            row["uses_refs_column_mode"] = 1
            continue

        match = _LAYOUT_MODE_RE.match(line)
        if match:
            page_no = int(match.group("page"))
            layout_crop_pages.add(page_no)
            row = page_rows.setdefault(page_no, _new_page_metric(page_no))
            row["uses_layout_crop_mode"] = 1
            continue

        match = _REFS_CROP_DONE_RE.match(line)
        if match:
            page_no = int(match.group("page"))
            refs_crop_times.append(float(match.group("seconds")))
            row = page_rows.setdefault(page_no, _new_page_metric(page_no))
            row["uses_refs_column_mode"] = 1
            continue

        match = _LAYOUT_CROP_DONE_RE.match(line)
        if match:
            page_no = int(match.group("page"))
            layout_crop_times.append(float(match.group("seconds")))
            row = page_rows.setdefault(page_no, _new_page_metric(page_no))
            row["uses_layout_crop_mode"] = 1
            continue

        if "[VISION_DIRECT] VL empty on page" in line and ", retry " in line:
            empty_retry_count += 1
            continue

        if "[VISION_DIRECT] fragmented math detected on page" in line:
            math_retry_count += 1
            continue

        if ("falling back to extraction pipeline" in line) or ("using extraction fallback" in line):
            fallback_count += 1
            continue

    page_metrics: list[dict] = []
    for page_no in sorted(page_rows.keys()):
        row = dict(page_rows[page_no])
        row["uses_refs_column_mode"] = int(page_no in refs_column_pages or bool(row.get("uses_refs_column_mode")))
        row["uses_layout_crop_mode"] = int(page_no in layout_crop_pages or bool(row.get("uses_layout_crop_mode")))
        page_metrics.append(row)

    page_total_values = [float(row["page_total_s"]) for row in page_metrics if "page_total_s" in row]
    vision_step6_values = [float(row["step_6_vision_convert_s"]) for row in page_metrics if "step_6_vision_convert_s" in row]

    run_metrics = {
        "page_metric_count": int(len(page_metrics)),
        "page_timed_pages": int(len(page_total_values)),
        "page_total_avg_s": round(sum(page_total_values) / max(1, len(page_total_values)), 4) if page_total_values else 0.0,
        "page_total_p50_s": round(_percentile(page_total_values, 0.50), 4) if page_total_values else 0.0,
        "page_total_p90_s": round(_percentile(page_total_values, 0.90), 4) if page_total_values else 0.0,
        "vision_step6_pages": int(len(vision_step6_values)),
        "vision_step6_avg_s": round(sum(vision_step6_values) / max(1, len(vision_step6_values)), 4) if vision_step6_values else 0.0,
        "vision_step6_p50_s": round(_percentile(vision_step6_values, 0.50), 4) if vision_step6_values else 0.0,
        "vision_step6_p90_s": round(_percentile(vision_step6_values, 0.90), 4) if vision_step6_values else 0.0,
        "references_pages": int(sum(int(bool(row.get("is_references_page"))) for row in page_metrics)),
        "refs_column_pages": int(len(refs_column_pages)),
        "refs_crop_call_count": int(len(refs_crop_times)),
        "refs_crop_avg_s": round(sum(refs_crop_times) / max(1, len(refs_crop_times)), 4) if refs_crop_times else 0.0,
        "layout_crop_pages": int(len(layout_crop_pages)),
        "layout_crop_call_count": int(len(layout_crop_times)),
        "layout_crop_avg_s": round(sum(layout_crop_times) / max(1, len(layout_crop_times)), 4) if layout_crop_times else 0.0,
        "empty_retry_count": int(empty_retry_count),
        "math_retry_count": int(math_retry_count),
        "fallback_count": int(fallback_count),
    }
    return run_metrics, page_metrics


def parse_profile_spec(spec: str) -> BenchmarkProfile:
    fields: dict[str, str] = {}
    for chunk in str(spec or "").split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid profile field {part!r}; expected key=value")
        key, value = part.split("=", 1)
        fields[str(key).strip().lower()] = str(value).strip()

    speed_mode = str(fields.get("speed_mode") or fields.get("speed") or "").strip().lower()
    if speed_mode not in VALID_SPEED_MODES:
        raise ValueError(f"profile requires speed_mode in {sorted(VALID_SPEED_MODES)}")

    name = str(fields.get("name") or fields.get("label") or "").strip()
    no_llm_workers = _parse_optional_int(fields, "no_llm_workers")
    llm_page_workers = _parse_optional_int(fields, "llm_page_workers")
    llm_workers = _parse_optional_int(fields, "llm_workers")
    max_inflight = _parse_optional_int(fields, "max_inflight")
    vision_dpi = _parse_optional_int(fields, "vision_dpi")
    vision_compress = _parse_optional_int(fields, "vision_compress")
    stage_timings = _parse_bool(fields["stage_timings"]) if "stage_timings" in fields else False

    if not name:
        parts = [speed_mode]
        if no_llm_workers is not None:
            parts.append(f"no{no_llm_workers}")
        if llm_page_workers is not None:
            parts.append(f"page{llm_page_workers}")
        if max_inflight is not None:
            parts.append(f"inflight{max_inflight}")
        if vision_dpi is not None:
            parts.append(f"dpi{vision_dpi}")
        name = "-".join(parts)

    return BenchmarkProfile(
        name=_slugify(name),
        speed_mode=speed_mode,
        no_llm_workers=no_llm_workers,
        llm_page_workers=llm_page_workers,
        llm_workers=llm_workers,
        max_inflight=max_inflight,
        vision_dpi=vision_dpi,
        vision_compress=vision_compress,
        stage_timings=bool(stage_timings),
    )


def discover_pdf_paths(inputs: Iterable[str | Path], *, recursive: bool = False, pattern: str = "*.pdf") -> list[Path]:
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_file():
            if path.suffix.lower() != ".pdf":
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                pdfs.append(resolved)
            continue
        if not path.is_dir():
            continue
        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        for candidate in iterator:
            if not candidate.is_file() or candidate.suffix.lower() != ".pdf":
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            pdfs.append(resolved)
    pdfs.sort(key=lambda p: (p.name.lower(), str(p).lower()))
    return pdfs


def profile_env_overrides(profile: BenchmarkProfile) -> dict[str, str | None]:
    return {
        "KB_PDF_NO_LLM_PAGE_WORKERS": str(profile.no_llm_workers) if profile.no_llm_workers is not None else None,
        "KB_PDF_LLM_PAGE_WORKERS": str(profile.llm_page_workers) if profile.llm_page_workers is not None else None,
        "KB_LLM_MAX_INFLIGHT": str(profile.max_inflight) if profile.max_inflight is not None else None,
        "KB_PDF_VISION_DPI": str(profile.vision_dpi) if profile.vision_dpi is not None else None,
        "KB_PDF_VISION_COMPRESS": str(profile.vision_compress) if profile.vision_compress is not None else None,
        "KB_PDF_STAGE_TIMINGS": "1" if profile.stage_timings else "0",
    }


@contextmanager
def temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    old_values: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            old_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


class _TeeWriter:
    def __init__(self, *writers):
        self._writers = writers

    def write(self, text: str) -> int:
        for writer in self._writers:
            writer.write(text)
        return len(text)

    def flush(self) -> None:
        for writer in self._writers:
            writer.flush()


@contextmanager
def redirect_output(log_path: Path, *, show_converter_output: bool = False) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fp:
        writer = _TeeWriter(sys.stdout, log_fp) if show_converter_output else log_fp
        err_writer = _TeeWriter(sys.stderr, log_fp) if show_converter_output else log_fp
        with redirect_stdout(writer), redirect_stderr(err_writer):
            yield


def build_llm_config(*, required: bool) -> LlmConfig | None:
    if not required:
        return None
    settings = load_settings()
    if not settings.api_key:
        raise RuntimeError(
            "LLM benchmark profile requires API credentials. "
            "Set QWEN_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY first."
        )
    return LlmConfig(
        api_key=str(settings.api_key),
        base_url=str(settings.base_url),
        model=str(settings.model),
        temperature=0.0,
        max_tokens=4096,
        request_sleep_s=0.0,
        timeout_s=float(settings.timeout_s),
        max_retries=int(settings.max_retries),
    )


def build_convert_config(*, pdf_path: Path, out_dir: Path, profile: BenchmarkProfile) -> ConvertConfig:
    llm_cfg = build_llm_config(required=(profile.speed_mode != "no_llm"))
    llm_workers = int(profile.llm_workers or 1)
    return ConvertConfig(
        pdf_path=pdf_path,
        out_dir=out_dir,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=llm_cfg,
        llm_workers=llm_workers,
        workers=int(profile.no_llm_workers or profile.llm_page_workers or 1),
        speed_mode=profile.speed_mode,
    )


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


def run_single_benchmark(
    *,
    pdf_path: Path,
    profile: BenchmarkProfile,
    repeat_index: int,
    out_root: Path,
    show_converter_output: bool = False,
    clear_page_cache: bool = True,
) -> tuple[dict, list[dict]]:
    pdf_name = pdf_path.name
    case_dir = out_root / _slugify(profile.name) / _slugify(pdf_path.stem) / f"run_{repeat_index:02d}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    log_path = case_dir / "converter.log"
    config = build_convert_config(pdf_path=pdf_path, out_dir=case_dir, profile=profile)
    page_count = _pdf_page_count(pdf_path)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    ok = False
    error = ""

    try:
        if clear_page_cache:
            LLMWorker.clear_shared_page_ocr_cache()
        with temporary_env(profile_env_overrides(profile)):
            with redirect_output(log_path, show_converter_output=show_converter_output):
                converter = PDFConverter(config)
                converter.convert(str(pdf_path), str(case_dir))
        ok = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with log_path.open("a", encoding="utf-8", errors="replace") as log_fp:
            traceback.print_exc(file=log_fp)

    elapsed_s = time.perf_counter() - t0
    md_path = case_dir / "output.md"
    md_chars = 0
    md_bytes = 0
    if md_path.exists():
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
            md_chars = len(text)
            md_bytes = len(text.encode("utf-8"))
        except Exception:
            try:
                md_bytes = md_path.stat().st_size
            except Exception:
                md_bytes = 0

    asset_count = 0
    assets_dir = case_dir / "assets"
    if assets_dir.exists():
        try:
            asset_count = sum(1 for child in assets_dir.iterdir() if child.is_file())
        except Exception:
            asset_count = 0

    log_metrics, page_metrics = parse_converter_log_metrics(log_path)

    result = {
        "profile": profile.name,
        "speed_mode": profile.speed_mode,
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "pdf_pages": page_count,
        "repeat": int(repeat_index),
        "ok": bool(ok),
        "elapsed_s": round(float(elapsed_s), 4),
        "started_at": started_at,
        "output_dir": str(case_dir),
        "log_path": str(log_path),
        "output_md_chars": int(md_chars),
        "output_md_bytes": int(md_bytes),
        "asset_count": int(asset_count),
        "error": error,
        "no_llm_workers": profile.no_llm_workers,
        "llm_page_workers": profile.llm_page_workers,
        "llm_workers": profile.llm_workers,
        "max_inflight": profile.max_inflight,
        "vision_dpi": profile.vision_dpi,
        "vision_compress": profile.vision_compress,
        "stage_timings": bool(profile.stage_timings),
    }
    result.update(log_metrics)
    meta_path = case_dir / "benchmark_run.json"
    meta_payload = dict(result)
    meta_payload["page_metrics"] = page_metrics
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, page_metrics


def summarize_runs_by_case(runs: Iterable[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        key = (str(run.get("profile") or ""), str(run.get("pdf_path") or ""))
        groups.setdefault(key, []).append(run)

    rows: list[dict] = []
    for (profile, pdf_path), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        elapsed = [float(item.get("elapsed_s") or 0.0) for item in items]
        md_chars = [int(item.get("output_md_chars") or 0) for item in items]
        page_total_avg = [float(item.get("page_total_avg_s") or 0.0) for item in items]
        vision_step6_avg = [float(item.get("vision_step6_avg_s") or 0.0) for item in items]
        references_pages = [float(item.get("references_pages") or 0.0) for item in items]
        empty_retries = [float(item.get("empty_retry_count") or 0.0) for item in items]
        math_retries = [float(item.get("math_retry_count") or 0.0) for item in items]
        fallbacks = [float(item.get("fallback_count") or 0.0) for item in items]
        ok_count = sum(1 for item in items if bool(item.get("ok")))
        fail_count = len(items) - ok_count
        rows.append(
            {
                "profile": profile,
                "pdf_name": str(items[0].get("pdf_name") or Path(pdf_path).name),
                "pdf_path": pdf_path,
                "runs": len(items),
                "ok_runs": ok_count,
                "fail_runs": fail_count,
                "avg_elapsed_s": round(sum(elapsed) / max(1, len(elapsed)), 4),
                "min_elapsed_s": round(min(elapsed), 4) if elapsed else 0.0,
                "max_elapsed_s": round(max(elapsed), 4) if elapsed else 0.0,
                "avg_output_md_chars": round(sum(md_chars) / max(1, len(md_chars)), 1),
                "avg_page_total_s": round(sum(page_total_avg) / max(1, len(page_total_avg)), 4),
                "avg_vision_step6_s": round(sum(vision_step6_avg) / max(1, len(vision_step6_avg)), 4),
                "avg_references_pages": round(sum(references_pages) / max(1, len(references_pages)), 4),
                "avg_empty_retries": round(sum(empty_retries) / max(1, len(empty_retries)), 4),
                "avg_math_retries": round(sum(math_retries) / max(1, len(math_retries)), 4),
                "avg_fallbacks": round(sum(fallbacks) / max(1, len(fallbacks)), 4),
            }
        )
    return rows


def summarize_runs_by_profile(runs: Iterable[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(str(run.get("profile") or ""), []).append(run)

    rows: list[dict] = []
    for profile, items in sorted(groups.items(), key=lambda x: x[0]):
        elapsed = [float(item.get("elapsed_s") or 0.0) for item in items]
        page_total_avg = [float(item.get("page_total_avg_s") or 0.0) for item in items]
        vision_step6_avg = [float(item.get("vision_step6_avg_s") or 0.0) for item in items]
        references_pages = [float(item.get("references_pages") or 0.0) for item in items]
        empty_retries = [float(item.get("empty_retry_count") or 0.0) for item in items]
        math_retries = [float(item.get("math_retry_count") or 0.0) for item in items]
        fallbacks = [float(item.get("fallback_count") or 0.0) for item in items]
        ok_count = sum(1 for item in items if bool(item.get("ok")))
        unique_pdfs = sorted({str(item.get("pdf_path") or "") for item in items if str(item.get("pdf_path") or "")})
        rows.append(
            {
                "profile": profile,
                "runs": len(items),
                "pdfs": len(unique_pdfs),
                "ok_runs": ok_count,
                "fail_runs": len(items) - ok_count,
                "avg_elapsed_s": round(sum(elapsed) / max(1, len(elapsed)), 4),
                "min_elapsed_s": round(min(elapsed), 4) if elapsed else 0.0,
                "max_elapsed_s": round(max(elapsed), 4) if elapsed else 0.0,
                "avg_page_total_s": round(sum(page_total_avg) / max(1, len(page_total_avg)), 4),
                "avg_vision_step6_s": round(sum(vision_step6_avg) / max(1, len(vision_step6_avg)), 4),
                "avg_references_pages": round(sum(references_pages) / max(1, len(references_pages)), 4),
                "avg_empty_retries": round(sum(empty_retries) / max(1, len(empty_retries)), 4),
                "avg_math_retries": round(sum(math_retries) / max(1, len(math_retries)), 4),
                "avg_fallbacks": round(sum(fallbacks) / max(1, len(fallbacks)), 4),
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in row_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def default_profiles(*, stage_timings: bool = False) -> list[BenchmarkProfile]:
    profiles = [
        BenchmarkProfile(name="no-llm-1", speed_mode="no_llm", no_llm_workers=1, stage_timings=stage_timings),
        BenchmarkProfile(name="no-llm-4", speed_mode="no_llm", no_llm_workers=4, stage_timings=stage_timings),
    ]
    try:
        settings = load_settings()
        has_llm = bool(settings.api_key)
    except Exception:
        has_llm = False
    if has_llm:
        profiles.extend(
            [
                BenchmarkProfile(
                    name="normal-1",
                    speed_mode="normal",
                    llm_page_workers=1,
                    llm_workers=1,
                    max_inflight=1,
                    stage_timings=stage_timings,
                ),
                BenchmarkProfile(
                    name="normal-4",
                    speed_mode="normal",
                    llm_page_workers=4,
                    llm_workers=1,
                    max_inflight=4,
                    stage_timings=stage_timings,
                ),
            ]
        )
    return profiles


def run_benchmark_suite(
    *,
    pdf_paths: list[Path],
    profiles: list[BenchmarkProfile],
    out_root: Path,
    repeat: int = 1,
    show_converter_output: bool = False,
    fail_fast: bool = False,
    clear_page_cache: bool = True,
) -> dict:
    runs: list[dict] = []
    page_metrics: list[dict] = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    suite_t0 = time.perf_counter()

    for profile in profiles:
        for pdf_path in pdf_paths:
            for run_no in range(1, int(repeat) + 1):
                print(
                    f"[BENCH] profile={profile.name} pdf={pdf_path.name} run={run_no}/{repeat}",
                    flush=True,
                )
                result, run_page_metrics = run_single_benchmark(
                    pdf_path=pdf_path,
                    profile=profile,
                    repeat_index=run_no,
                    out_root=out_root,
                    show_converter_output=show_converter_output,
                    clear_page_cache=clear_page_cache,
                )
                runs.append(result)
                for page_row in run_page_metrics:
                    row = dict(page_row)
                    row["profile"] = profile.name
                    row["speed_mode"] = profile.speed_mode
                    row["pdf_name"] = pdf_path.name
                    row["pdf_path"] = str(pdf_path)
                    row["repeat"] = int(run_no)
                    page_metrics.append(row)
                status = "OK" if result["ok"] else "FAIL"
                print(
                    f"[BENCH] {status} elapsed={result['elapsed_s']:.2f}s md_chars={result['output_md_chars']} "
                    f"log={result['log_path']}",
                    flush=True,
                )
                if fail_fast and (not result["ok"]):
                    raise RuntimeError(f"benchmark failed: {result['profile']} | {result['pdf_name']} | {result['error']}")

    payload = {
        "started_at": started_at,
        "elapsed_s": round(time.perf_counter() - suite_t0, 4),
        "pdfs": [str(p) for p in pdf_paths],
        "profiles": [asdict(p) for p in profiles],
        "runs": runs,
        "page_metrics": page_metrics,
        "summary_by_case": summarize_runs_by_case(runs),
        "summary_by_profile": summarize_runs_by_profile(runs),
    }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the PDF converter across modes and worker profiles")
    parser.add_argument("inputs", nargs="*", help="PDF files or directories")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories")
    parser.add_argument("--glob", default="*.pdf", help="Glob pattern for PDFs inside directories")
    parser.add_argument("--profile", action="append", default=[], help="Profile spec: key=value pairs separated by commas")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count per profile/PDF")
    parser.add_argument("--out-dir", default="", help="Benchmark output root (default: tmp/benchmarks/<timestamp>)")
    parser.add_argument("--max-pdfs", type=int, default=0, help="Limit number of discovered PDFs")
    parser.add_argument("--stage-timings", action="store_true", help="Enable KB_PDF_STAGE_TIMINGS=1 for all profiles")
    parser.add_argument("--show-converter-output", action="store_true", help="Tee converter stdout/stderr to console")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed benchmark run")
    parser.add_argument("--warm-cache", action="store_true", help="Keep shared page-OCR cache between runs")
    parser.add_argument("--list-default-profiles", action="store_true", help="Print default profiles and exit")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    profiles = [parse_profile_spec(spec) for spec in args.profile]
    if not profiles:
        profiles = default_profiles(stage_timings=bool(args.stage_timings))
    elif args.stage_timings:
        profiles = [replace(profile, stage_timings=True) for profile in profiles]

    if args.list_default_profiles:
        for profile in default_profiles(stage_timings=bool(args.stage_timings)):
            print(json.dumps(asdict(profile), ensure_ascii=False))
        return 0

    if not args.inputs:
        raise SystemExit("Please provide at least one PDF or directory input.")

    pdf_paths = discover_pdf_paths(args.inputs, recursive=bool(args.recursive), pattern=str(args.glob))
    if int(args.max_pdfs or 0) > 0:
        pdf_paths = pdf_paths[: int(args.max_pdfs)]
    if not pdf_paths:
        raise SystemExit("No PDF files found for benchmark.")

    out_root = Path(args.out_dir).expanduser() if str(args.out_dir).strip() else Path("tmp") / "benchmarks" / time.strftime("%Y%m%d_%H%M%S")
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    payload = run_benchmark_suite(
        pdf_paths=pdf_paths,
        profiles=profiles,
        out_root=out_root,
        repeat=max(1, int(args.repeat)),
        show_converter_output=bool(args.show_converter_output),
        fail_fast=bool(args.fail_fast),
        clear_page_cache=not bool(args.warm_cache),
    )

    json_path = out_root / "benchmark_results.json"
    runs_csv_path = out_root / "benchmark_runs.csv"
    page_metrics_csv_path = out_root / "benchmark_page_metrics.csv"
    case_csv_path = out_root / "benchmark_summary_by_case.csv"
    profile_csv_path = out_root / "benchmark_summary_by_profile.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(runs_csv_path, payload["runs"])
    write_csv(page_metrics_csv_path, payload["page_metrics"])
    write_csv(case_csv_path, payload["summary_by_case"])
    write_csv(profile_csv_path, payload["summary_by_profile"])

    print(f"[BENCH] wrote {json_path}", flush=True)
    print(f"[BENCH] wrote {runs_csv_path}", flush=True)
    print(f"[BENCH] wrote {page_metrics_csv_path}", flush=True)
    print(f"[BENCH] wrote {case_csv_path}", flush=True)
    print(f"[BENCH] wrote {profile_csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
