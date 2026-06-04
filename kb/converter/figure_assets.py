from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import fitz
except ImportError:  # pragma: no cover - PyMuPDF is required in normal conversion.
    fitz = None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_PAGE_FIG_RE = re.compile(r"page_(\d+)_fig_(\d+)", flags=re.IGNORECASE)
_FIG_ALIAS_RE = re.compile(r"fig_\d+[A-Za-z]?\.[A-Za-z0-9]+$", flags=re.IGNORECASE)
_MD_ASSET_RE = re.compile(
    r"!\[[^\]]*\]\((?:\.?/)?assets/([^)\s#?]+)(?:#[^)]*)?\)",
    flags=re.IGNORECASE,
)


def _env_figure_dpi() -> int | None:
    raw = str(os.environ.get("KB_PDF_FIGURE_DPI", "") or "").strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except Exception:
        return None
    return value if value > 0 else None


def resolve_target_figure_asset_dpi(
    *,
    base_dpi: int | float | None = None,
    figure_dpi: int | float | None = None,
) -> int:
    """Return the target DPI for user-visible figure PNG assets."""
    try:
        base = int(float(base_dpi if base_dpi is not None else 200) or 200)
    except Exception:
        base = 200

    requested = _env_figure_dpi()
    if requested is None:
        try:
            requested = int(float(figure_dpi or 0)) or None
        except Exception:
            requested = None

    dpi = requested if requested is not None else max(base, 320)
    if dpi <= 0:
        dpi = base
    return max(144, min(600, int(dpi)))


def resolve_figure_asset_dpi(converter, *, base_dpi: int | float | None = None) -> int:
    """Return the DPI used for user-visible figure PNG assets."""
    try:
        base = int(float(base_dpi if base_dpi is not None else getattr(converter, "dpi", 200) or 200))
    except Exception:
        base = 200
    figure_dpi: int | float | None = None
    try:
        figure_dpi = getattr(converter, "figure_dpi", None)
    except Exception:
        figure_dpi = None
    if figure_dpi is None:
        cfg = getattr(converter, "cfg", None)
        try:
            figure_dpi = getattr(cfg, "figure_dpi", None)
        except Exception:
            figure_dpi = None
    return resolve_target_figure_asset_dpi(base_dpi=base, figure_dpi=figure_dpi)


def figure_asset_needs_refresh(path: Path, *, clip_rect, dpi: int | float) -> bool:
    """Detect old low-DPI figure assets so reconversion can upgrade them in place."""
    try:
        if (not path.exists()) or path.stat().st_size < 256:
            return True
    except Exception:
        return True

    if fitz is None:
        return False

    try:
        expected_w = max(1, int(round(float(clip_rect.width) * float(dpi) / 72.0)))
        expected_h = max(1, int(round(float(clip_rect.height) * float(dpi) / 72.0)))
    except Exception:
        return False

    try:
        pix = fitz.Pixmap(str(path))
        actual_w = int(getattr(pix, "width", 0) or 0)
        actual_h = int(getattr(pix, "height", 0) or 0)
    except Exception:
        return False

    return actual_w < expected_w * 0.9 or actual_h < expected_h * 0.9


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean_asset_name(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    return Path(raw).name


def _asset_page_and_index(name: str) -> tuple[int, int]:
    match = _PAGE_FIG_RE.search(str(name or ""))
    if not match:
        return 0, 0
    try:
        page = int(match.group(1))
    except Exception:
        page = 0
    try:
        idx = int(match.group(2))
    except Exception:
        idx = 0
    return page, idx


def _is_alias_asset(name: str) -> bool:
    return bool(_FIG_ALIAS_RE.fullmatch(_clean_asset_name(name)))


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    out: list[float] = []
    for raw in list(value)[:4]:
        try:
            out.append(float(raw))
        except Exception:
            return []
    return out


def _rect_width_height(value: Any) -> tuple[float, float]:
    vals = _float_list(value)
    if not vals:
        return 0.0, 0.0
    return max(0.0, vals[2] - vals[0]), max(0.0, vals[3] - vals[1])


def _merge_row(prev: dict[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(prev)
    for key, value in dict(row).items():
        if key.startswith("_"):
            merged[key] = value
            continue
        if value in (None, "", [], {}):
            continue
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
        elif key in {"crop_bbox", "bbox", "caption_bbox"}:
            merged[key] = value
    return merged


def _add_row(rows_by_asset: dict[str, dict[str, Any]], row: Mapping[str, Any]) -> None:
    asset_name = _clean_asset_name(
        row.get("asset_name")
        or row.get("asset_name_raw")
        or row.get("asset_name_alias")
    )
    if not asset_name:
        return
    rec = dict(row)
    rec["asset_name"] = asset_name
    page, idx = _asset_page_and_index(asset_name)
    try:
        rec["page"] = int(rec.get("page") or page or 0)
    except Exception:
        rec["page"] = page
    try:
        rec["index"] = int(rec.get("index") or idx or 0)
    except Exception:
        rec["index"] = idx
    key = asset_name.lower()
    rows_by_asset[key] = _merge_row(rows_by_asset.get(key, {}), rec)


def _load_figure_rows(assets_dir: Path) -> list[dict[str, Any]]:
    rows_by_asset: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(assets_dir.glob("page_*_fig_index.json")):
        payload = _load_json(meta_path)
        figures = payload.get("figures") if isinstance(payload, dict) else None
        if not isinstance(figures, list):
            continue
        for idx, item in enumerate(figures, start=1):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            row.setdefault("page", (payload or {}).get("page"))
            row.setdefault("index", idx)
            _add_row(rows_by_asset, row)

    payload = _load_json(assets_dir / "figure_index.json")
    figures = payload.get("figures") if isinstance(payload, dict) else None
    if isinstance(figures, list):
        for item in figures:
            if isinstance(item, Mapping):
                _add_row(rows_by_asset, item)

    for meta_path in sorted(assets_dir.glob("page_*_fig_*.meta.json")):
        payload = _load_json(meta_path)
        if isinstance(payload, Mapping):
            row = dict(payload)
            row.setdefault("asset_name", f"{meta_path.name.removesuffix('.meta.json')}.png")
            _add_row(rows_by_asset, row)

    return sorted(
        rows_by_asset.values(),
        key=lambda row: (
            int(row.get("page") or 0),
            int(row.get("index") or 0),
            str(row.get("asset_name") or "").lower(),
        ),
    )


def _markdown_asset_refs(md_path: Path) -> list[str]:
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    out: list[str] = []
    for match in _MD_ASSET_RE.finditer(text):
        name = _clean_asset_name(match.group(1))
        if name and name not in out:
            out.append(name)
    return out


def _asset_rows_from_markdown(md_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_asset = {str(row.get("asset_name") or "").lower(): dict(row) for row in rows}
    for name in _markdown_asset_refs(md_path):
        key = name.lower()
        if key in rows_by_asset:
            continue
        page, idx = _asset_page_and_index(name)
        rows_by_asset[key] = {
            "asset_name": name,
            "page": page,
            "index": idx,
            "source": "markdown",
        }
    return sorted(
        rows_by_asset.values(),
        key=lambda row: (
            int(row.get("page") or 0),
            int(row.get("index") or 0),
            str(row.get("asset_name") or "").lower(),
        ),
    )


def _fallback_asset_rows(assets_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_asset = {str(row.get("asset_name") or "").lower(): dict(row) for row in rows}
    for path in sorted(assets_dir.glob("page_*_fig_*.*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        key = path.name.lower()
        if key in rows_by_asset:
            continue
        page, idx = _asset_page_and_index(path.name)
        rows_by_asset[key] = {
            "asset_name": path.name,
            "page": page,
            "index": idx,
            "source": "assets",
        }
    return sorted(
        rows_by_asset.values(),
        key=lambda row: (
            int(row.get("page") or 0),
            int(row.get("index") or 0),
            str(row.get("asset_name") or "").lower(),
        ),
    )


def _image_dimensions(path: Path) -> tuple[int, int]:
    if fitz is not None:
        try:
            pix = fitz.Pixmap(str(path))
            return int(getattr(pix, "width", 0) or 0), int(getattr(pix, "height", 0) or 0)
        except Exception:
            pass
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return 0, 0


def _image_content_stats(path: Path) -> dict[str, float]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            gray = img.convert("L")
            gray.thumbnail((96, 96))
            pixels = list(gray.getdata())
    except Exception:
        return {}
    if not pixels:
        return {}
    total = float(len(pixels))
    non_white = sum(1 for value in pixels if int(value) < 245)
    dark = sum(1 for value in pixels if int(value) < 220)
    return {
        "non_white_ratio": float(non_white) / total,
        "dark_ratio": float(dark) / total,
    }


def _sha1_file(path: Path) -> str:
    try:
        h = hashlib.sha1()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 256)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _figure_number(row: Mapping[str, Any]) -> int:
    for key in ("paper_figure_number", "fig_no", "number"):
        try:
            value = int(row.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _issue(
    *,
    code: str,
    severity: str,
    row: Mapping[str, Any],
    message: str,
    actual: tuple[int, int] = (0, 0),
    expected: tuple[int, int] = (0, 0),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    asset_name = _clean_asset_name(row.get("asset_name") or row.get("asset_name_raw") or row.get("asset_name_alias"))
    payload: dict[str, Any] = {
        "code": str(code or ""),
        "severity": str(severity or "warning"),
        "asset_name": asset_name,
        "page": int(row.get("page") or 0),
        "figure_number": _figure_number(row),
        "message": str(message or ""),
        "actual_width": int(actual[0] or 0),
        "actual_height": int(actual[1] or 0),
        "expected_width": int(expected[0] or 0),
        "expected_height": int(expected[1] or 0),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _expected_pixels(row: Mapping[str, Any], *, target_dpi: int) -> tuple[int, int]:
    crop_w, crop_h = _rect_width_height(row.get("crop_bbox") or row.get("bbox"))
    if crop_w <= 0.0 or crop_h <= 0.0:
        return 0, 0
    return (
        max(1, int(round(float(crop_w) * float(target_dpi) / 72.0))),
        max(1, int(round(float(crop_h) * float(target_dpi) / 72.0))),
    )


def _is_suspicious_crop(
    row: Mapping[str, Any],
    *,
    actual: tuple[int, int],
    content_stats: Mapping[str, float],
) -> tuple[bool, str, dict[str, Any]]:
    crop_w, crop_h = _rect_width_height(row.get("crop_bbox") or row.get("bbox"))
    bbox_w, bbox_h = _rect_width_height(row.get("bbox"))
    actual_w, actual_h = int(actual[0] or 0), int(actual[1] or 0)
    extra: dict[str, Any] = {}

    if crop_w <= 0.0 or crop_h <= 0.0:
        return False, "", extra

    extra.update(
        {
            "crop_width_pt": round(crop_w, 2),
            "crop_height_pt": round(crop_h, 2),
        }
    )
    if bbox_w > 0.0 and bbox_h > 0.0:
        extra.update(
            {
                "bbox_width_pt": round(bbox_w, 2),
                "bbox_height_pt": round(bbox_h, 2),
            }
        )
        crop_area = crop_w * crop_h
        bbox_area = bbox_w * bbox_h
        if bbox_area > 0.0 and crop_area < bbox_area * 0.70:
            return True, "saved crop is much smaller than the detected visual figure box", extra

    if crop_w < 16.0 or crop_h < 16.0:
        return True, "figure crop is too small to be a readable visual asset", extra

    aspect = crop_w / max(1.0, crop_h)
    extra["crop_aspect_ratio"] = round(aspect, 4)
    if (aspect > 14.0 or aspect < 0.071) and max(crop_w, crop_h) >= 72.0:
        return True, "figure crop has an extreme aspect ratio", extra

    if min(actual_w, actual_h) < 64 and max(actual_w, actual_h) >= 192:
        return True, "rendered figure image is a thin strip", extra

    non_white_ratio = float(content_stats.get("non_white_ratio") or 0.0)
    dark_ratio = float(content_stats.get("dark_ratio") or 0.0)
    if content_stats:
        extra.update(
            {
                "non_white_ratio": round(non_white_ratio, 6),
                "dark_ratio": round(dark_ratio, 6),
            }
        )
    if actual_w * actual_h >= 4096 and content_stats and non_white_ratio < 0.002:
        return True, "figure crop is nearly blank", extra

    return False, "", extra


def scan_figure_asset_quality(
    md_path: Path | str,
    *,
    source_pdf_path: Path | str | None = None,
    target_dpi: int | None = None,
) -> dict[str, Any]:
    """Scan one converted Markdown folder for figure asset defects."""
    md = Path(md_path).expanduser()
    dpi = resolve_target_figure_asset_dpi(figure_dpi=target_dpi) if target_dpi else resolve_target_figure_asset_dpi()
    assets_dir = md.parent / "assets"
    issues: list[dict[str, Any]] = []
    asset_details: list[dict[str, Any]] = []

    if not md.exists() or not md.is_file():
        return {
            "ok": False,
            "status": "error",
            "md_path": str(md),
            "assets_dir": str(assets_dir),
            "source_pdf_path": str(source_pdf_path or ""),
            "source_pdf_available": bool(source_pdf_path and Path(source_pdf_path).expanduser().is_file()),
            "target_dpi": dpi,
            "figures": 0,
            "issue_count": 1,
            "issue_counts": {"missing_markdown": 1},
            "refresh_recommended": False,
            "issues": [
                {
                    "code": "missing_markdown",
                    "severity": "error",
                    "asset_name": "",
                    "page": 0,
                    "figure_number": 0,
                    "message": "converted Markdown file is missing",
                }
            ],
            "assets": [],
        }

    rows = _load_figure_rows(assets_dir) if assets_dir.exists() else []
    rows = _asset_rows_from_markdown(md, rows)
    rows = _fallback_asset_rows(assets_dir, rows) if assets_dir.exists() else rows

    hash_groups: dict[str, list[dict[str, Any]]] = {}
    seen_issue_keys: set[tuple[str, str]] = set()

    def add_issue(issue: dict[str, Any]) -> None:
        key = (str(issue.get("asset_name") or "").lower(), str(issue.get("code") or "").lower())
        if key in seen_issue_keys:
            return
        seen_issue_keys.add(key)
        issues.append(issue)

    for row in rows:
        asset_name = _clean_asset_name(row.get("asset_name") or row.get("asset_name_raw") or row.get("asset_name_alias"))
        if not asset_name:
            continue
        asset_path = assets_dir / asset_name
        detail: dict[str, Any] = {
            "asset_name": asset_name,
            "page": int(row.get("page") or 0),
            "figure_number": _figure_number(row),
            "exists": False,
            "width": 0,
            "height": 0,
            "expected_width": 0,
            "expected_height": 0,
            "issue_codes": [],
        }

        try:
            file_size = asset_path.stat().st_size if asset_path.exists() else 0
        except Exception:
            file_size = 0
        detail["file_size"] = int(file_size)
        if (not asset_path.exists()) or file_size < 256:
            add_issue(
                _issue(
                    code="missing_asset",
                    severity="error",
                    row=row,
                    message="figure asset is missing or too small to be a valid image",
                )
            )
            detail["issue_codes"].append("missing_asset")
            asset_details.append(detail)
            continue

        actual = _image_dimensions(asset_path)
        detail["exists"] = True
        detail["width"] = int(actual[0] or 0)
        detail["height"] = int(actual[1] or 0)
        if actual[0] <= 0 or actual[1] <= 0:
            add_issue(
                _issue(
                    code="invalid_image",
                    severity="error",
                    row=row,
                    message="figure asset exists but cannot be decoded as an image",
                    actual=actual,
                )
            )
            detail["issue_codes"].append("invalid_image")
            asset_details.append(detail)
            continue

        expected = _expected_pixels(row, target_dpi=dpi)
        detail["expected_width"] = int(expected[0] or 0)
        detail["expected_height"] = int(expected[1] or 0)
        if expected[0] > 0 and expected[1] > 0:
            dpi_x = float(actual[0]) * 72.0 / max(1.0, _rect_width_height(row.get("crop_bbox") or row.get("bbox"))[0])
            dpi_y = float(actual[1]) * 72.0 / max(1.0, _rect_width_height(row.get("crop_bbox") or row.get("bbox"))[1])
            detail["estimated_dpi"] = int(round(min(dpi_x, dpi_y)))
            if actual[0] < expected[0] * 0.88 or actual[1] < expected[1] * 0.88:
                add_issue(
                    _issue(
                        code="low_resolution",
                        severity="warning",
                        row=row,
                        message="figure asset was rendered below the configured figure DPI",
                        actual=actual,
                        expected=expected,
                        extra={"estimated_dpi": detail["estimated_dpi"]},
                    )
                )
                detail["issue_codes"].append("low_resolution")

        content_stats = _image_content_stats(asset_path)
        suspicious, reason, extra = _is_suspicious_crop(row, actual=actual, content_stats=content_stats)
        if suspicious:
            add_issue(
                _issue(
                    code="suspicious_crop",
                    severity="warning",
                    row=row,
                    message=reason,
                    actual=actual,
                    expected=expected,
                    extra=extra,
                )
            )
            detail["issue_codes"].append("suspicious_crop")

        if not _is_alias_asset(asset_name):
            digest = _sha1_file(asset_path)
            if digest:
                hash_groups.setdefault(digest, []).append({"row": row, "actual": actual, "expected": expected})

        asset_details.append(detail)

    for group in hash_groups.values():
        unique_assets = {
            _clean_asset_name((item.get("row") or {}).get("asset_name"))
            for item in group
            if _clean_asset_name((item.get("row") or {}).get("asset_name"))
        }
        if len(unique_assets) <= 1:
            continue
        duplicate_names = sorted(unique_assets)
        for item in group:
            row = item.get("row") if isinstance(item.get("row"), Mapping) else {}
            add_issue(
                _issue(
                    code="duplicate_asset",
                    severity="warning",
                    row=row,
                    message="multiple figure assets have identical image bytes",
                    actual=item.get("actual") or (0, 0),
                    expected=item.get("expected") or (0, 0),
                    extra={"duplicates": duplicate_names[:12]},
                )
            )

    issue_counts = Counter(str(item.get("code") or "") for item in issues if str(item.get("code") or ""))
    severity_counts = Counter(str(item.get("severity") or "warning") for item in issues)
    status = "good"
    if issues:
        status = "error" if int(severity_counts.get("error") or 0) > 0 else "warning"
    refresh_codes = {"missing_asset", "invalid_image", "low_resolution", "duplicate_asset", "suspicious_crop"}
    source_pdf = Path(source_pdf_path).expanduser() if source_pdf_path else None
    return {
        "ok": True,
        "status": status,
        "md_path": str(md),
        "assets_dir": str(assets_dir),
        "source_pdf_path": str(source_pdf or ""),
        "source_pdf_available": bool(source_pdf and source_pdf.exists() and source_pdf.is_file()),
        "target_dpi": dpi,
        "figures": len(rows),
        "issue_count": len(issues),
        "issue_counts": dict(issue_counts),
        "severity_counts": dict(severity_counts),
        "refresh_recommended": bool(any(str(issue.get("code") or "") in refresh_codes for issue in issues)),
        "issues": issues,
        "assets": asset_details,
    }


def summarize_figure_asset_quality_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = [dict(report) for report in reports if isinstance(report, Mapping)]
    issue_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    docs_with_issues = 0
    refresh_recommended = 0
    figures = 0
    for report in items:
        figures += int(report.get("figures") or 0)
        if int(report.get("issue_count") or 0) > 0:
            docs_with_issues += 1
        if bool(report.get("refresh_recommended")):
            refresh_recommended += 1
        for code, count in dict(report.get("issue_counts") or {}).items():
            issue_counts[str(code)] += int(count or 0)
        for code, count in dict(report.get("severity_counts") or {}).items():
            severity_counts[str(code)] += int(count or 0)
    status = "good"
    if int(severity_counts.get("error") or 0) > 0:
        status = "error"
    elif sum(issue_counts.values()) > 0:
        status = "warning"
    return {
        "status": status,
        "scanned": len(items),
        "figures": int(figures),
        "docs_with_issues": int(docs_with_issues),
        "refresh_recommended": int(refresh_recommended),
        "issue_counts": dict(issue_counts),
        "severity_counts": dict(severity_counts),
    }
