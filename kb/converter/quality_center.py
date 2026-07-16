from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .quality_repair import (
    conversion_quality_result_path,
    conversion_repair_strategy_for_issue,
    load_conversion_quality_result,
    repair_markdown_quality,
    write_conversion_quality_result,
)
from .structured_index_batch import iter_markdown_files
from .structured_indices import rebuild_structured_indices_for_markdown


_SOURCE_ACTION_LABELS = {
    "none": "可用",
    "autofix": "可自动修复",
    "reconvert": "需要重转",
    "review": "需要源头处理",
}

_ISSUE_LABELS = {
    "missing_markdown": "缺少 Markdown",
    "missing_images": "图片资产缺失",
    "mojibake": "编码异常",
    "weak_structure": "结构过弱",
    "missing_references": "参考文献缺失",
    "source_text_loss": "正文疑似缺失",
    "missing_source_pages": "源 PDF 页面正文缺失",
    "reference_index_truncated": "参考文献索引不完整",
    "references_before_body": "参考文献位置异常",
    "analyzer_errors": "Markdown 解析错误",
    "analyzer_warnings": "Markdown 结构警告",
    "missing_abstract": "摘要标题缺失",
    "missing_page_markers": "页码锚点缺失",
    "page_marker_gaps": "页码锚点不连续",
    "source_page_marker_alignment": "源 PDF 页码锚点错位",
    "missing_captions": "图题缺失",
    "unclosed_display_math": "公式块未闭合",
    "heading_level_jumps": "标题层级跳跃",
    "collapsed_heading_hierarchy": "Heading hierarchy collapsed",
    "stray_inline_math": "Stray inline math markup",
    "quality_scan_failed": "质量扫描失败",
}

_DOC_TYPE_LABELS = {
    "research_article": "研究论文",
    "review": "综述",
    "supplementary": "补充材料",
}

_PAGE_ALIGNMENT_LABELS = {
    "high": "页码定位稳定",
    "medium": "页码定位可用",
    "low": "页码定位偏弱",
    "missing": "缺页码锚点",
    "unknown": "未连接源 PDF",
}


def _clean_strings(values: Iterable[Any], *, limit: int = 30) -> list[str]:
    out: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if value and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def _report_issue_codes(report: Mapping[str, Any] | None) -> list[str]:
    data = report if isinstance(report, Mapping) else {}
    plan = data.get("repair_plan") if isinstance(data.get("repair_plan"), Mapping) else {}
    codes = _clean_strings(list((plan or {}).get("issue_codes") or []))
    if codes:
        return [code.lower() for code in codes]
    repair = data.get("auto_repair") if isinstance(data.get("auto_repair"), Mapping) else {}
    return [code.lower() for code in _clean_strings(list((repair or {}).get("remaining_issue_codes") or []))]


def _report_action(report: Mapping[str, Any] | None) -> str:
    data = report if isinstance(report, Mapping) else {}
    plan = data.get("repair_plan") if isinstance(data.get("repair_plan"), Mapping) else {}
    action = str((plan or {}).get("action") or data.get("recommended_action") or "").strip().lower()
    return action if action in _SOURCE_ACTION_LABELS else ("none" if not _report_issue_codes(data) else "review")


def _issue_label(code: str) -> str:
    clean = str(code or "").strip().lower()
    if clean in _ISSUE_LABELS:
        return _ISSUE_LABELS[clean]
    strategy = conversion_repair_strategy_for_issue(clean)
    label = str(strategy.get("label") or "").strip()
    return label or clean.replace("_", " ")


def _source_pdf_candidates(md_path: Path, pdf_root: Path | None) -> list[Path]:
    names: list[str] = []
    parent_name = str(md_path.parent.name or "").strip()
    if parent_name:
        names.append(parent_name)
    stem = str(md_path.stem or "").strip()
    if stem.lower().endswith(".en"):
        stem = stem[:-3]
    if stem:
        names.append(stem)

    roots: list[Path] = []
    if pdf_root is not None:
        roots.append(Path(pdf_root).expanduser())
    roots.append(md_path.parent)
    roots.append(md_path.parent.parent)

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for name in names:
            for suffix in (".pdf", ".PDF"):
                candidate = root / f"{name}{suffix}"
                key = str(candidate).lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
    return candidates


def source_pdf_for_markdown(md_path: Path | str, pdf_root: Path | str | None = None) -> Path | None:
    path = Path(md_path).expanduser()
    root = Path(pdf_root).expanduser() if pdf_root is not None else None
    for candidate in _source_pdf_candidates(path, root):
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def discover_quality_markdown_files(
    md_root: Path | str,
    *,
    glob: str = "*.en.md",
    limit: int = 1000,
) -> list[Path]:
    root = Path(md_root).expanduser()
    try:
        files = iter_markdown_files(root, glob=glob)
        if not files and glob == "*.en.md":
            files = iter_markdown_files(root, glob="*.md")
    except Exception:
        files = []
    if limit > 0:
        files = files[: int(limit)]
    return files


def quality_center_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, Mapping) or not report:
        return {
            "available": False,
            "status": "unknown",
            "severity": "warning",
            "action": "scan",
            "action_label": "尚未扫描",
            "message": "尚未生成转换质量报告。",
            "badges": [],
            "issue_labels": [],
            "issue_codes": [],
        }

    action = _report_action(report)
    issue_codes = _report_issue_codes(report)
    source = report.get("source_quality") if isinstance(report.get("source_quality"), Mapping) else {}
    auto_repair = report.get("auto_repair") if isinstance(report.get("auto_repair"), Mapping) else {}
    doc_type = str((source or {}).get("document_type") or "").strip()
    align = str((source or {}).get("page_alignment_confidence") or "").strip().lower()
    applied = _clean_strings(list((auto_repair or {}).get("applied") or []), limit=8)

    status = action if action in {"none", "autofix", "reconvert", "review"} else "review"
    if status == "none":
        status = "ready"
    severity = "ok" if status == "ready" else ("error" if status == "reconvert" else "warning")

    badges: list[str] = []
    doc_label = _DOC_TYPE_LABELS.get(doc_type)
    if doc_label:
        badges.append(doc_label)
    if bool((source or {}).get("abstract_not_applicable")):
        badges.append("摘要不适用")
    if bool((source or {}).get("source_pdf_available")):
        pages = int((source or {}).get("pdf_page_count") or 0)
        badges.append(f"源 PDF {pages} 页" if pages > 0 else "源 PDF 已连接")
    elif str((source or {}).get("source_pdf_path") or "").strip():
        badges.append("源 PDF 未连接")
    if align:
        badges.append(_PAGE_ALIGNMENT_LABELS.get(align, f"页码定位 {align}"))
    if applied:
        badges.append("已修复 " + ", ".join(applied[:3]))

    issue_labels = [_issue_label(code) for code in issue_codes]
    if bool((source or {}).get("source_text_loss")):
        message = "正文疑似没有完整进入 Markdown，建议用普通模式重新转换源 PDF。"
    elif status == "reconvert":
        message = "剩余问题来自源转换阶段，建议重新转换以恢复正文、图片或参考文献。"
    elif status == "autofix":
        message = "剩余问题可由质量中心安全修复，并同步重建结构索引。"
    elif status == "review":
        message = "质量中心暂时没有安全自动修复策略，建议回到源转换链路处理。"
    elif bool((source or {}).get("abstract_not_applicable")):
        message = "该文档按补充材料处理，缺独立摘要不再计为质量问题。"
    else:
        message = "转换质量可用于索引、引用定位和文献卡片。"

    return {
        "available": True,
        "status": status,
        "severity": severity,
        "action": action,
        "action_label": _SOURCE_ACTION_LABELS.get(action, _SOURCE_ACTION_LABELS.get(status, status)),
        "message": message,
        "badges": badges[:8],
        "issue_labels": issue_labels[:8],
        "issue_codes": issue_codes[:30],
        "source_quality": dict(source or {}),
        "report_path": str(report.get("md_path") or ""),
    }


def load_quality_center_card(md_path: Path | str, *, refresh: bool = False, pdf_root: Path | str | None = None) -> dict[str, Any]:
    path = Path(md_path).expanduser()
    report: dict[str, Any] = {}
    if refresh:
        try:
            source_pdf = source_pdf_for_markdown(path, pdf_root)
            report = write_conversion_quality_result(path, source_pdf_path=source_pdf)
        except Exception:
            report = {}
    else:
        report = load_conversion_quality_result(path)
    summary = quality_center_summary(report)
    summary["path"] = str(path)
    summary["quality_report_path"] = str(conversion_quality_result_path(path))
    return summary


@dataclass
class QualityCenterBatchStats:
    scanned: int = 0
    repaired: int = 0
    changed: int = 0
    rebuilt: int = 0
    ready: int = 0
    autofix: int = 0
    reconvert: int = 0
    review: int = 0
    unknown: int = 0
    failed: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    reconvert_paths: list[str] = field(default_factory=list)
    review_paths: list[str] = field(default_factory=list)

    def add_summary(self, md_path: Path, summary: Mapping[str, Any]) -> None:
        status = str((summary or {}).get("status") or "unknown").strip().lower()
        if status == "ready":
            self.ready += 1
        elif status == "autofix":
            self.autofix += 1
        elif status == "reconvert":
            self.reconvert += 1
            if len(self.reconvert_paths) < 20:
                self.reconvert_paths.append(str(md_path))
        elif status == "review":
            self.review += 1
            if len(self.review_paths) < 20:
                self.review_paths.append(str(md_path))
        else:
            self.unknown += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_error(stats: QualityCenterBatchStats, md_path: Path, exc: Exception, *, max_errors: int) -> None:
    stats.failed += 1
    if len(stats.errors) < max(0, int(max_errors)):
        stats.errors.append({"path": str(md_path), "error": str(exc)[:400]})


def scan_quality_targets(
    md_paths: Iterable[Path | str],
    *,
    pdf_root: Path | str | None = None,
    max_errors: int = 20,
) -> dict[str, Any]:
    stats = QualityCenterBatchStats()
    for raw_path in md_paths:
        md_path = Path(raw_path).expanduser()
        stats.scanned += 1
        try:
            source_pdf = source_pdf_for_markdown(md_path, pdf_root)
            report = write_conversion_quality_result(md_path, source_pdf_path=source_pdf)
            stats.add_summary(md_path, quality_center_summary(report))
        except Exception as exc:
            _record_error(stats, md_path, exc, max_errors=max_errors)
    return stats.to_dict()


def repair_quality_targets(
    md_paths: Iterable[Path | str],
    *,
    pdf_root: Path | str | None = None,
    rebuild_indices: bool = True,
    max_errors: int = 20,
) -> dict[str, Any]:
    stats = QualityCenterBatchStats()
    for raw_path in md_paths:
        md_path = Path(raw_path).expanduser()
        stats.scanned += 1
        try:
            source_pdf = source_pdf_for_markdown(md_path, pdf_root)
            report = write_conversion_quality_result(md_path, source_pdf_path=source_pdf)
            plan = report.get("repair_plan") if isinstance(report.get("repair_plan"), Mapping) else {}
            action = str((plan or {}).get("action") or report.get("recommended_action") or "").strip().lower()
            autofix_issue_codes = _clean_strings(list((plan or {}).get("autofix_issue_codes") or []))
            if action == "autofix" and not autofix_issue_codes:
                autofix_issue_codes = _clean_strings(list((plan or {}).get("issue_codes") or []))
            if autofix_issue_codes:
                repair_result = repair_markdown_quality(
                    md_path,
                    issue_codes=autofix_issue_codes,
                    source_pdf_path=source_pdf,
                )
                report = write_conversion_quality_result(
                    md_path,
                    auto_repair_result=repair_result,
                    source_pdf_path=source_pdf,
                )
                if bool(repair_result.get("changed")):
                    stats.repaired += 1
                    stats.changed += 1
                    if len(stats.changed_paths) < 20:
                        stats.changed_paths.append(str(md_path))
                    if bool(rebuild_indices):
                        rebuild_structured_indices_for_markdown(md_path, assets_dir=md_path.parent / "assets")
                        stats.rebuilt += 1
            stats.add_summary(md_path, quality_center_summary(report))
        except Exception as exc:
            _record_error(stats, md_path, exc, max_errors=max_errors)
    return stats.to_dict()
