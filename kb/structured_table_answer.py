from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


_HIGH_EXTREME_RE = re.compile(
    r"(?:最高|最大|最好|最佳|并列第一|\bhighest\b|\bmaximum\b|\blargest\b|\bbest\b|\btop\b)",
    flags=re.I,
)
_LOW_EXTREME_RE = re.compile(
    r"(?:最低|最小|\blowest\b|\bminimum\b|\bsmallest\b)",
    flags=re.I,
)
_KNOWN_METRIC_RE = re.compile(r"\b(PSNR|SSIM|LPIPS|FID|FPS|MAE|MSE|RMSE|AP|MAP|IOU)\b", flags=re.I)
_VALUE_PAIR_RE = re.compile(
    r"(?P<name>[A-Za-z][^;:=\n]{0,120}?)\s*=\s*(?P<value>[+-]?\d+(?:\.\d+)?)"
)


def _clean_method_name(value: str) -> str:
    text = re.sub(r"\s*\[\d+(?:\s*[,;-]\s*\d+)*\]\s*", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip(" .,:;-|")
    return text


def _paper_title_from_source(source_path: str) -> str:
    title = Path(str(source_path or "")).stem.strip()
    title = re.sub(r"(?i)\.(?:en|zh|cn)$", "", title).strip()
    return title


def _metric_direction(prompt: str, meta: dict) -> str:
    text = str(prompt or "")
    wants_high = bool(_HIGH_EXTREME_RE.search(text))
    wants_low = bool(_LOW_EXTREME_RE.search(text))
    if wants_high == wants_low:
        return ""
    explicit = "max" if wants_high else "min"
    if re.search(r"(?:最好|最佳|\bbest\b|\btop\b)", text, flags=re.I):
        direction = str((meta or {}).get("table_metric_direction") or "").strip().lower()
        if direction in {"down", "lower", "min", "minimum", "asc", "ascending"}:
            return "min"
        if direction in {"up", "higher", "max", "maximum", "desc", "descending"}:
            return "max"
    return explicit


def _metric_series_text(hit: dict, metric_label: str) -> str:
    meta = hit.get("meta", {}) if isinstance(hit.get("meta"), dict) else {}
    candidates = [
        str(hit.get("text") or "").strip(),
        *[str(value or "").strip() for value in list(meta.get("ref_show_snippets") or [])],
        *[str(value or "").strip() for value in list(meta.get("ref_snippets") or [])],
    ]
    candidates = [value for value in candidates if value]
    if not candidates:
        return ""
    best = max(candidates, key=lambda value: (value.count("="), len(value)))
    if metric_label:
        match = re.search(rf"(?i){re.escape(metric_label)}\s*:\s*", best)
        if match:
            return best[match.end() :].strip()
    marker = re.search(r"(?i)\b(?:PSNR|SSIM|LPIPS|FID|FPS|MAE|MSE|RMSE|AP|MAP|IOU)\b\s*:\s*", best)
    return best[marker.end() :].strip() if marker else best


def build_structured_table_extreme_answer(prompt: str, answer_hits: list[dict] | None) -> str:
    """Answer an exact max/min table lookup without another model round trip.

    The fast path is intentionally narrow: it only accepts a structured metric
    series emitted by the table index, an explicit extrema request, a matching
    metric label, and at least two parseable method/value pairs. Broader
    comparison or explanation questions continue through the normal LLM path.
    """

    question = str(prompt or "").strip()
    if not question:
        return ""
    structured_hits: list[dict] = []
    for hit in list(answer_hits or []):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) if isinstance(hit.get("meta"), dict) else {}
        if str(meta.get("structured_kind") or "").strip().lower() != "table_metric":
            continue
        if not str(meta.get("source_path") or "").strip():
            continue
        if not str(meta.get("table_metric_label") or meta.get("table_metric") or "").strip():
            continue
        structured_hits.append(hit)
    if not structured_hits:
        return ""

    hit = structured_hits[0]
    meta = hit.get("meta", {}) if isinstance(hit.get("meta"), dict) else {}
    direction = _metric_direction(question, meta)
    if not direction:
        return ""

    metric_label = str(meta.get("table_metric_label") or meta.get("table_metric") or "").strip()
    question_metrics = {match.group(1).upper() for match in _KNOWN_METRIC_RE.finditer(question)}
    label_metrics = {match.group(1).upper() for match in _KNOWN_METRIC_RE.finditer(metric_label)}
    if question_metrics and not question_metrics.intersection(label_metrics):
        return ""

    series_text = _metric_series_text(hit, metric_label)
    values: list[tuple[str, Decimal, str]] = []
    seen_names: set[str] = set()
    for match in _VALUE_PAIR_RE.finditer(series_text):
        name = _clean_method_name(match.group("name"))
        raw_value = str(match.group("value") or "").strip()
        if not name or name.casefold() in seen_names:
            continue
        try:
            number = Decimal(raw_value)
        except InvalidOperation:
            continue
        seen_names.add(name.casefold())
        values.append((name, number, raw_value))
    if len(values) < 2:
        return ""

    extreme = (max if direction == "max" else min)(number for _name, number, _raw in values)
    winners = [(name, raw) for name, number, raw in values if number == extreme]
    if not winners:
        return ""

    prefer_zh = bool(re.search(r"[\u4e00-\u9fff]", question))
    names = [name for name, _raw in winners]
    if prefer_zh:
        joined_names = "、".join(names[:-1]) + (" 和 " if len(names) > 1 else "") + names[-1]
    else:
        joined_names = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
    raw_extreme = winners[0][1]
    source_path = str(meta.get("source_path") or "").strip()
    title = _paper_title_from_source(source_path)
    table_number = str(meta.get("table_number") or "").strip()
    metric = metric_label or next(iter(question_metrics), "metric")

    if prefer_zh:
        location = f"《{title}》" if title else "该论文"
        prefix = f"在{location}"
        if table_number:
            prefix += f"的表 {table_number} 中"
        else:
            prefix += "中"
        extreme_label = "最高值" if direction == "max" else "最低值"
        tie = "并列取得" if len(winners) > 1 else "由"
        if len(winners) > 1:
            return f"{prefix}，{metric} 的{extreme_label}为 {raw_extreme}，由 {joined_names} {tie}。"
        return f"{prefix}，{metric} 的{extreme_label}为 {raw_extreme}，由 {joined_names} 取得。"

    location = f"Table {table_number} of {title}" if table_number and title else (title or "the paper")
    extreme_label = "highest" if direction == "max" else "lowest"
    tie = "tie at" if len(winners) > 1 else "achieves"
    return f"In {location}, {joined_names} {tie} the {extreme_label} {metric} value of {raw_extreme}."
