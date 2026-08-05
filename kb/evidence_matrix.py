from __future__ import annotations

import csv
import hashlib
import io
import re
from pathlib import Path
from typing import Any

from kb.research_brief import research_brief_context
from kb.retriever import BM25Retriever
from kb.store import load_all_chunks
from kb.tokenize import tokenize


MATRIX_CELL_FIELDS = (
    "method",
    "dataset_or_experiment",
    "metric",
    "key_result",
    "limitation",
)
_MAX_MATRIX_SOURCES = 8
_MAX_CELL_VALUE = 520
_MAX_EVIDENCE_QUOTE = 1_200
_FIELD_SPECS: dict[str, tuple[str, re.Pattern[str]]] = {
    "method": (
        "method methodology approach architecture algorithm model framework pipeline implementation",
        re.compile(
            r"\b(?:method|methodology|approach|architecture|algorithm|model|framework|pipeline|implementation|network|techniques?)\b"
            r"|(?:方法|算法|模型|架构|框架|流程|网络|实现)",
            flags=re.IGNORECASE,
        ),
    ),
    "dataset_or_experiment": (
        "dataset benchmark experimental setup evaluation protocol training test samples simulation hardware",
        re.compile(
            r"\b(?:dataset|benchmark|experiment|experimental|evaluation|protocol|training set|test set|simulation|hardware|samples?)\b"
            r"|(?:数据集|基准|实验|评估|训练集|测试集|仿真|硬件|样本)",
            flags=re.IGNORECASE,
        ),
    ),
    "metric": (
        "metric quantitative evaluation accuracy PSNR SSIM LPIPS RMSE F1 AUC IoU latency FPS runtime",
        re.compile(
            r"\b(?:metric|accuracy|precision|recall|PSNR|SSIM|LPIPS|RMSE|NMSE|MSE|MAE|SNR|FID|F1|AUC|IoU|Dice|FPS|latency|runtime|FLOPs|parameters?|mean average precision)\b"
            r"|(?:指标|准确率|精度|召回率|延迟|运行时间|参数量)",
            flags=re.IGNORECASE,
        ),
    ),
    "key_result": (
        "result results performance achieves improves outperforms quantitative comparison ablation",
        re.compile(
            r"\b(?:result|results|performance|achiev(?:e|es|ed)|improv(?:e|es|ed)|outperform(?:s|ed)?|comparison|ablation)\b"
            r"|(?:结果|性能|达到|提升|优于|超过|对比|消融)",
            flags=re.IGNORECASE,
        ),
    ),
    "limitation": (
        "limitation limitations challenge failure weakness drawback future work remains however although",
        re.compile(
            r"\b(?:limit|limitation|limitations|challenge|failure|weakness|drawback|future work|remain|however|although)\b"
            r"|(?:局限|限制|挑战|失败|不足|缺点|未来工作|仍然|然而)",
            flags=re.IGNORECASE,
        ),
    ),
}
_NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*(?:%|dB|ms|s|fps|Hz|GB|MB)?", re.IGNORECASE)
_STRONG_METRIC_RE = re.compile(
    r"\b(?:PSNR|SSIM|LPIPS|RMSE|NMSE|MSE|MAE|SNR|FID|F1|AUC|IoU|Dice|FPS|FLOPs)\b"
    r"|\bmAP(?:@[.\d:]+)?\b",
)
_REFERENCE_HEADING_RE = re.compile(
    r"(?:^|[/ >])(?:references?|bibliography|literature cited|works cited)(?:$|[/ >])",
    re.IGNORECASE,
)
_METHOD_SIGNAL_RE = re.compile(
    r"\b(?:uses?|using|based on|consists? of|combines?|comprises?|architecture|algorithm|framework|pipeline|network|techniques?)\b"
    r"|\b(?:we|this (?:work|paper|study))\s+(?:propose|present|introduce|develop|design|use|employ|combine|implement|adopt|build|construct)\b"
    r"|(?:采用|使用|基于|结合|由.+组成|提出|设计|实现)",
    re.IGNORECASE,
)
_METHOD_MECHANISM_RE = re.compile(
    r"\b(?:based on|consists? of|combines?|comprises?|architecture|algorithm|framework|pipeline|network|techniques?|implementation|"
    r"sampling scheme|transformation|rasterization|reconstruction method|imaging method)\b"
    r"|\b(?:we|this (?:work|paper|study))\s+(?:propose|present|introduce|develop|design|implement)\b"
    r"|(?:基于|结合|由.+组成|架构|算法|框架|网络|技术|实现|采样方案|变换|重建方法|成像方法|提出|设计)",
    re.IGNORECASE,
)
_EXPERIMENT_SIGNAL_RE = re.compile(
    r"\b(?:dataset|benchmark|experimental setup|evaluation protocol|training set|test set|simulation|hardware|videos?|phantoms?|subjects?)\b"
    r"|\b\d[\d, ]*\s+samples?\b"
    r"|\b(?:comparison )?experiments?\s+(?:use|uses|used|were|are|include|includes|compare|compares)\b"
    r"|\bwe\s+(?:conduct|evaluate|test|train)\b"
    r"|(?:数据集|基准|实验设置|评估协议|训练集|测试集|仿真|硬件|样本|场景|测量)",
    re.IGNORECASE,
)
_RESULT_SIGNAL_RE = re.compile(
    r"\b(?:achiev(?:e|es|ed)|improv(?:e|es|ed)|outperform(?:s|ed)?|surpass(?:es|ed)?|exceed(?:s|ed)?|"
    r"reduc(?:e|es|ed|tion)|increas(?:e|es|ed)|higher|lower|better|superior|state-of-the-art|SOTA|"
    r"demonstrat(?:e|es|ed) that|show(?:s|ed) that)\b"
    r"|(?:达到|提升|优于|超过|降低|减少|增加|更高|更低|结果表明|实验表明)",
    re.IGNORECASE,
)
_LIMITATION_SIGNAL_RE = re.compile(
    r"\b(?:limitation|limitations|challenge|failure|weakness|drawback|trade[- ]?off|at the cost|suffer(?:s|ed)?|"
    r"constrain(?:s|ed|t)?|bottleneck|future work|cannot|unable|fails? to)\b"
    r"|\b(?:however|although|but)\b"
    r"|\bremain(?:s|ed)?\s+(?:limited|challenging|unclear|unknown|difficult)\b"
    r"|(?:局限|限制|挑战|失败|不足|缺点|代价|权衡|瓶颈|未来工作|无法|不能|仍不明确)",
    re.IGNORECASE,
)
_NEGATIVE_LIMIT_SIGNAL_RE = re.compile(
    r"\b(?:limit|limitation|challenge|failure|weakness|drawback|trade[- ]?off|cost|suffer|constraint|bottleneck|"
    r"cannot|unable|fails?|not|difficult|slow|lack|degrad|sacrific|incompatib|unclear|unknown|dependent)\w*\b"
    r"|(?:局限|限制|挑战|失败|不足|缺点|代价|权衡|瓶颈|无法|不能|困难|缓慢|缺少|牺牲|不兼容|不明确|依赖)",
    re.IGNORECASE,
)
_PRIOR_METHOD_RE = re.compile(
    r"\b(?:these|such|existing|previous|prior|conventional|traditional|current|other)\b.{0,60}\b(?:methods?|approaches?|algorithms?|models?|networks?|techniques?|systems?)\b",
    re.IGNORECASE,
)
_CURRENT_WORK_RE = re.compile(
    r"\b(?:we|our|this (?:work|paper|study|method|approach|algorithm|model|network|system)|proposed|presented|introduced)\b",
    re.IGNORECASE,
)
_CURRENT_METHOD_PROPOSAL_RE = re.compile(
    r"\b(?:we|this (?:work|paper|study))\s+(?:propose|present|introduce|develop|design|implement)\b"
    r"|\b(?:our|the proposed|this)\s+(?:method|approach|algorithm|model|network|framework)\b",
    re.IGNORECASE,
)
_RESULT_META_RE = re.compile(
    r"\b(?:results?|values?)\s+(?:are|were)\s+(?:computed|shown|presented|listed|reported)\b"
    r"|\bwe compare\b.{0,100}\b(?:shown?|presented|listed|reported)\s+in\s+(?:table|fig(?:ure)?)\b",
    re.IGNORECASE,
)
_METHOD_META_RE = re.compile(
    r"\b(?:describe|discuss|outline)\b.{0,80}\b(?:how|method|algorithm|implementation)\b",
    re.IGNORECASE,
)
_METHOD_NEGATIVE_RE = re.compile(
    r"\b(?:unrealistic|cannot|unable|not feasible|infeasible|struggle|suffer|limitation)\w*\b",
    re.IGNORECASE,
)
_OTHER_METHOD_LIMIT_RE = re.compile(
    r"\b(?:these|such|this type of|current|existing|previous|prior|other)\b.{0,60}\b(?:methods?|approaches?|algorithms?|models?|networks?|techniques?|systems?)\b"
    r"|\[[0-9,;\-\s]+\]\s+(?:shows?|reports?|finds?)\b"
    r"|\bsome works?\b.{0,40}\b(?:claim|show|report|find)s?\b"
    r"|\b[A-Z][A-Za-z0-9-]{2,}\s+(?:struggles?|suffers?|fails?|has limitations?)\b",
    re.IGNORECASE,
)
_LIMITATION_RESOLUTION_RE = re.compile(
    r"\b(?:address|overcome|resolve|alleviate|mitigate)\w*\b.{0,60}\b(?:limit|limitation|challenge|drawback)s?\b"
    r"|\b(?:limit|limitation|challenge|drawback)s?\b.{0,60}\b(?:are|is|were|was)?\s*(?:addressed|overcome|resolved|alleviated|mitigated)\b",
    re.IGNORECASE,
)
_NON_LIMITATION_CHALLENGE_RE = re.compile(
    r"\bchallenge\s+(?:track\d*|dataset|benchmark|competition)\b",
    re.IGNORECASE,
)
_DATASET_META_RE = re.compile(
    r"\b(?:images? for the dataset|dataset|results?)\b.{0,50}\b(?:shown|presented|listed)\s+in\s+(?:fig(?:ure)?|table)\b"
    r"|\b(?:images? for the )?dataset\b.{0,50}\bsee\b"
    r"|\bsee\s+(?:the\s+)?supplementary\b"
    r"|\bdetails\b.{0,100}\b(?:found|provided|described)\b",
    re.IGNORECASE,
)
_CAPTION_ONLY_RE = re.compile(
    r"^(?:\([a-z]\)\s+|!\[)?(?:\*+)?\s*(?:the\s+)?(?:modified\s+)?(?:figure|fig\.?|table|visualization|experimental setup)\s*\d*\b",
    re.IGNORECASE,
)
_FIELD_HEADING_RE: dict[str, re.Pattern[str]] = {
    "method": re.compile(r"\b(?:method|methodology|approach|architecture|model|framework|implementation|abstract)\b|(?:方法|模型|架构|实现|摘要)", re.I),
    "dataset_or_experiment": re.compile(r"\b(?:experiment|evaluation|dataset|setup|implementation|results?)\b|(?:实验|评估|数据集|设置|实现|结果)", re.I),
    "metric": re.compile(r"\b(?:metric|quantitative|evaluation|experiment|results?)\b|(?:指标|定量|评估|实验|结果)", re.I),
    "key_result": re.compile(r"\b(?:result|evaluation|experiment|comparison|conclusion|abstract)\b|(?:结果|评估|实验|对比|结论|摘要)", re.I),
    "limitation": re.compile(r"\b(?:limitation|discussion|conclusion|future|outlook)\b|(?:局限|讨论|结论|未来|展望)", re.I),
}


def _text(value: object, *, limit: int = 2_000, multiline: bool = False) -> str:
    text = str(value or "").replace("\x00", " ")
    if multiline:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    else:
        text = re.sub(r"\s+", " ", text).strip()
    return text[: max(0, int(limit))]


def _source_variants(value: object) -> set[str]:
    raw = _text(value, limit=1_200).replace("\\", "/")
    if not raw:
        return set()
    variants = {raw.lower()}
    try:
        path = Path(raw)
        variants.update(
            {
                path.name.lower(),
                path.stem.lower(),
                str(path.expanduser().resolve(strict=False)).replace("\\", "/").lower(),
            }
        )
    except Exception:
        pass
    return {item for item in variants if item}


def _meta(hit: dict[str, Any]) -> dict[str, Any]:
    value = hit.get("meta")
    return value if isinstance(value, dict) else {}


def _source_path(hit: dict[str, Any]) -> str:
    return _text(_meta(hit).get("source_path"), limit=1_200)


def _source_chunks(chunks: list[dict[str, Any]], source_path: str) -> list[dict[str, Any]]:
    expected = _source_variants(source_path)
    if not expected:
        return []
    return [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict) and expected & _source_variants(_source_path(chunk))
    ]


def _sentence_candidates(value: object) -> list[str]:
    raw = _text(value, limit=8_000, multiline=True)
    if not raw:
        return []
    raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.DOTALL)
    candidates: list[str] = []
    for line in raw.splitlines():
        if line.lstrip().startswith("#"):
            continue
        clean_line = re.sub(r"^\s*(?:#{1,6}|[-*+]|\d+[.)])\s*", "", line).strip()
        if not clean_line:
            continue
        pieces = re.split(r"(?<=[.!?。！？；;])\s+", clean_line)
        for piece in pieces:
            clean = _text(piece, limit=_MAX_CELL_VALUE)
            if len(clean) >= 20:
                candidates.append(clean)
    if not candidates:
        fallback = _text(raw, limit=_MAX_CELL_VALUE)
        if fallback:
            candidates.append(fallback)
    return candidates[:24]


def _candidate_score(
    sentence: str,
    *,
    field: str,
    pattern: re.Pattern[str],
    objective_tokens: set[str],
    heading: str,
) -> float:
    if not pattern.search(sentence) and not (field == "metric" and _STRONG_METRIC_RE.search(sentence)):
        return -1.0
    numeric_surface = re.sub(r"\[[0-9,;\-\s]+\]", " ", sentence)
    if field == "metric" and not (_NUMERIC_RE.search(numeric_surface) or _STRONG_METRIC_RE.search(sentence)):
        return -1.0
    if field == "metric" and sentence.count("·") >= 3:
        return -1.0
    if field == "method" and (
        not _METHOD_SIGNAL_RE.search(sentence)
        or not _METHOD_MECHANISM_RE.search(sentence)
    ):
        return -1.0
    if field == "method" and _PRIOR_METHOD_RE.search(sentence) and not _CURRENT_METHOD_PROPOSAL_RE.search(sentence):
        return -1.0
    if field == "method" and (_METHOD_META_RE.search(sentence) or _METHOD_NEGATIVE_RE.search(sentence)):
        return -1.0
    if field == "dataset_or_experiment" and (
        len(sentence) < 36 or not _EXPERIMENT_SIGNAL_RE.search(sentence)
    ):
        return -1.0
    if field == "dataset_or_experiment" and _DATASET_META_RE.search(sentence):
        return -1.0
    if field == "key_result" and not _RESULT_SIGNAL_RE.search(sentence):
        return -1.0
    if field == "key_result" and _RESULT_META_RE.search(sentence):
        return -1.0
    if field == "limitation" and (
        not _LIMITATION_SIGNAL_RE.search(sentence)
        or not _NEGATIVE_LIMIT_SIGNAL_RE.search(sentence)
    ):
        return -1.0
    if field == "limitation" and _OTHER_METHOD_LIMIT_RE.search(sentence) and not _CURRENT_WORK_RE.search(sentence):
        return -1.0
    if field == "limitation" and _LIMITATION_RESOLUTION_RE.search(sentence):
        return -1.0
    if field == "limitation" and _NON_LIMITATION_CHALLENGE_RE.search(sentence):
        return -1.0
    sentence_tokens = {str(token).lower() for token in tokenize(sentence) if str(token).strip()}
    objective_weight = 0.8 if field == "method" else 0.45
    score = 3.0 + min(3.0, float(len(sentence_tokens & objective_tokens)) * objective_weight)
    if field in {"metric", "key_result"} and _NUMERIC_RE.search(numeric_surface):
        score += 2.0
    if field == "limitation" and re.search(r"\b(?:however|although|but|remain)\b|(?:然而|但是|仍然)", sentence, re.I):
        score += 0.5
    if _FIELD_HEADING_RE[field].search(heading):
        score += 1.0
    score += min(len(sentence), 320) / 1_000.0
    return score


def _best_cell_hit(
    chunks: list[dict[str, Any]],
    *,
    field: str,
    objective: str,
    retriever: BM25Retriever | None = None,
    excluded_quotes: set[str] | None = None,
) -> tuple[dict[str, Any], str] | None:
    if not chunks:
        return None
    query_terms, pattern = _FIELD_SPECS[field]
    query = query_terms
    active_retriever = retriever or BM25Retriever(chunks)
    ranked = active_retriever.search(query, top_k=min(20, max(8, len(chunks))))
    if field == "method" and objective:
        contextual = active_retriever.search(
            f"{objective} {query_terms}",
            top_k=min(20, max(8, len(chunks))),
        )
        seen_ranked = {
            str(item.get("id") or _meta(item).get("chunk_id") or id(item))
            for item in ranked
        }
        for item in contextual:
            identity = str(item.get("id") or _meta(item).get("chunk_id") or id(item))
            if identity not in seen_ranked:
                ranked.append(item)
                seen_ranked.add(identity)
    if not ranked:
        ranked = chunks[:10]
    objective_tokens = {
        str(token).lower()
        for token in tokenize(objective)
        if len(str(token).strip()) >= 2
    }
    best: tuple[float, dict[str, Any], str] | None = None
    for rank, hit in enumerate(ranked[:32]):
        meta = _meta(hit)
        heading = _text(meta.get("heading_path") or meta.get("top_heading"), limit=800)
        if _REFERENCE_HEADING_RE.search(heading):
            continue
        for sentence in _sentence_candidates(hit.get("text")):
            if _CAPTION_ONLY_RE.search(sentence):
                continue
            if _normal(sentence) in set(excluded_quotes or set()):
                continue
            score = _candidate_score(
                sentence,
                field=field,
                pattern=pattern,
                objective_tokens=objective_tokens,
                heading=heading,
            )
            if score < 0:
                continue
            if pattern.search(heading):
                score += 0.6
            score += max(0.0, 0.25 - rank * 0.025)
            if best is None or score > best[0]:
                best = (score, hit, sentence)
    return (best[1], best[2]) if best else None


def _evidence_id(source_path: str, field: str, hit: dict[str, Any], quote: str) -> str:
    meta = _meta(hit)
    seed = "|".join(
        (
            source_path.replace("\\", "/").lower(),
            field,
            _text(hit.get("id") or meta.get("chunk_id"), limit=300),
            _text(meta.get("block_id") or meta.get("anchor_id"), limit=300),
            quote,
        )
    )
    return f"ev_{hashlib.sha1(seed.encode('utf-8', errors='ignore')).hexdigest()[:16]}"


def _empty_cell(field: str) -> dict[str, Any]:
    return {
        "field": field,
        "value": "",
        "support_status": "missing",
        "evidence_ids": [],
        "manual_override": False,
    }


def _normal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _source_identity(value: object) -> str:
    variants = _source_variants(value)
    return sorted(variants, key=lambda item: (-len(item), item))[0] if variants else ""


def _comparison_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_rows = [row for row in rows if str(row.get("source_status") or "active") == "active"]
    flags: list[dict[str, Any]] = []
    for field, code, message in (
        (
            "dataset_or_experiment",
            "experimental_conditions_differ",
            "Experimental conditions differ across sources; compare quantitative results cautiously.",
        ),
        (
            "metric",
            "metrics_differ",
            "Reported metrics differ across sources; values are not directly comparable.",
        ),
    ):
        values = {
            _normal((row.get("cells") or {}).get(field, {}).get("value"))
            for row in active_rows
            if isinstance(row.get("cells"), dict)
            and isinstance((row.get("cells") or {}).get(field), dict)
            and _normal((row.get("cells") or {}).get(field, {}).get("value"))
        }
        if len(values) > 1:
            flags.append({"code": code, "severity": "info", "message": message, "field": field})
    return flags


def build_project_evidence_matrix(
    selected_items: list[dict[str, Any]],
    *,
    objective: str,
    db_dir: str | Path,
    existing_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    context_items = list(research_brief_context(selected_items).get("items") or [])[:_MAX_MATRIX_SOURCES]
    chunks = [item for item in load_all_chunks(Path(db_dir)) if isinstance(item, dict)]
    prior_by_source = {
        _source_identity(row.get("source_path")): row
        for row in list(existing_rows or [])
        if isinstance(row, dict) and _source_identity(row.get("source_path"))
    }
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    evidence_seen: set[str] = set()
    for source in context_items:
        if not isinstance(source, dict):
            continue
        source_path = _text(source.get("sourcePath"), limit=1_200)
        source_name = _text(source.get("sourceName") or source.get("title"), limit=500)
        source_title = _text(source.get("title") or source_name, limit=800)
        source_key = source_path.replace("\\", "/").lower()
        row_id = f"row_{hashlib.sha1(source_key.encode('utf-8', errors='ignore')).hexdigest()[:16]}"
        previous = prior_by_source.get(_source_identity(source_path), {})
        row = {
            "id": row_id,
            "source_item_key": _text(source.get("key"), limit=500),
            "paper": source_title or source_name or Path(source_path).stem,
            "source_name": source_name or Path(source_path).name,
            "source_path": source_path,
            "authors": _text(source.get("authors"), limit=800),
            "year": _text(source.get("year"), limit=40),
            "doi": _text(source.get("doi"), limit=400),
            "notes": _text(previous.get("notes"), limit=4_000, multiline=True),
            "source_status": "active",
            "cells": {},
        }
        source_chunks = _source_chunks(chunks, source_path)
        source_retriever = BM25Retriever(source_chunks)
        used_quotes: set[str] = set()
        for field in MATRIX_CELL_FIELDS:
            selected = _best_cell_hit(
                source_chunks,
                field=field,
                objective=objective,
                retriever=source_retriever,
                excluded_quotes=used_quotes,
            )
            if not selected:
                row["cells"][field] = _empty_cell(field)
                continue
            hit, sentence = selected
            used_quotes.add(_normal(sentence))
            meta = _meta(hit)
            evidence_id = _evidence_id(source_path, field, hit, sentence)
            row["cells"][field] = {
                "field": field,
                "value": _text(sentence, limit=_MAX_CELL_VALUE),
                "support_status": "grounded",
                "evidence_ids": [evidence_id],
                "manual_override": False,
            }
            if evidence_id in evidence_seen:
                continue
            evidence_seen.add(evidence_id)
            evidence.append(
                {
                    "id": evidence_id,
                    "field": field,
                    "source_item_key": row["source_item_key"],
                    "source_path": source_path,
                    "source_name": row["source_name"],
                    "title": source_title,
                    "heading_path": _text(meta.get("heading_path") or meta.get("top_heading"), limit=800),
                    "location_label": _text(meta.get("location_label") or meta.get("heading_path"), limit=500),
                    "page_start": meta.get("page_start") or meta.get("page") or None,
                    "page_end": meta.get("page_end") or meta.get("page") or None,
                    "block_id": _text(meta.get("block_id"), limit=500),
                    "anchor_id": _text(meta.get("anchor_id") or meta.get("anchor"), limit=500),
                    "evidence_quote": _text(sentence, limit=_MAX_EVIDENCE_QUOTE),
                    "score": float(hit.get("score") or 0.0),
                }
            )
        rows.append(row)
    return rows, evidence, _comparison_flags(rows)


def evidence_matrix_quality(
    *,
    rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    selected_items: list[dict[str, Any]],
    comparison_flags: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    expected_items = list(research_brief_context(selected_items).get("items") or [])[:_MAX_MATRIX_SOURCES]
    expected_sources = {
        _source_identity(item.get("sourcePath"))
        for item in expected_items
        if isinstance(item, dict) and _source_identity(item.get("sourcePath"))
    }
    row_sources = {
        _source_identity(row.get("source_path"))
        for row in rows
        if isinstance(row, dict) and str(row.get("source_status") or "active") == "active"
    }
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    missing_cells: list[dict[str, str]] = []
    unsupported_cells: list[dict[str, str]] = []
    populated_cells = 0
    supported_cells = 0
    source_evidence_counts: dict[str, int] = {source: 0 for source in expected_sources}
    for row in rows:
        if not isinstance(row, dict) or str(row.get("source_status") or "active") != "active":
            continue
        row_source = _source_identity(row.get("source_path"))
        cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        for field in MATRIX_CELL_FIELDS:
            cell = cells.get(field) if isinstance(cells.get(field), dict) else {}
            value = _text(cell.get("value"), limit=_MAX_CELL_VALUE)
            if not value:
                missing_cells.append({"row_id": str(row.get("id") or ""), "field": field})
                continue
            populated_cells += 1
            evidence_ids = [str(item or "") for item in list(cell.get("evidence_ids") or []) if str(item or "")]
            matched = [evidence_by_id[item] for item in evidence_ids if item in evidence_by_id]
            supported = bool(matched) and all(
                _source_identity(item.get("source_path")) == row_source
                and _normal(value) in _normal(item.get("evidence_quote"))
                for item in matched
            )
            supported = supported and str(cell.get("support_status") or "") == "grounded"
            supported = supported and not bool(cell.get("manual_override"))
            if supported:
                supported_cells += 1
                source_evidence_counts[row_source] = source_evidence_counts.get(row_source, 0) + 1
            else:
                unsupported_cells.append({"row_id": str(row.get("id") or ""), "field": field})
    unexpected_sources = sorted(source for source in row_sources if source and source not in expected_sources)
    missing_sources = sorted(source for source in expected_sources if source not in row_sources)
    sources_without_evidence = sorted(
        source for source in expected_sources if source_evidence_counts.get(source, 0) <= 0
    )
    unexpected_evidence = sorted(
        {
            _source_identity(item.get("source_path"))
            for item in evidence
            if isinstance(item, dict)
            and _source_identity(item.get("source_path"))
            and _source_identity(item.get("source_path")) not in expected_sources
        }
    )
    reasons: list[str] = []
    if not rows:
        reasons.append("no_rows")
    if missing_sources:
        reasons.append("selected_sources_without_rows")
    if sources_without_evidence:
        reasons.append("selected_sources_without_evidence")
    if unexpected_sources or unexpected_evidence:
        reasons.append("unexpected_sources")
    if unsupported_cells:
        reasons.append("unsupported_cells")
    if populated_cells <= 0:
        reasons.append("no_supported_cells")
    completeness = (
        populated_cells / max(1, len(expected_sources) * len(MATRIX_CELL_FIELDS))
        if expected_sources
        else 0.0
    )
    flags = [item for item in list(comparison_flags or []) if isinstance(item, dict)]
    quality = {
        "contract_version": 1,
        "generation_mode": "extractive",
        "selected_source_count": len(expected_sources),
        "row_count": len(rows),
        "covered_source_count": len(expected_sources) - len(sources_without_evidence),
        "populated_cell_count": populated_cells,
        "supported_cell_count": supported_cells,
        "unsupported_cell_count": len(unsupported_cells),
        "missing_cell_count": len(missing_cells),
        "completeness": round(max(0.0, min(1.0, completeness)), 4),
        "missing_cells": missing_cells[:80],
        "unsupported_cells": unsupported_cells[:80],
        "missing_sources": missing_sources,
        "sources_without_evidence": sources_without_evidence,
        "unexpected_sources": sorted(set(unexpected_sources + unexpected_evidence)),
        "comparison_flags": flags,
        "confirmed_conflicts": [],
        "reasons": reasons,
        "warnings": ["missing_cells"] if missing_cells else [],
        "edited_after_verification": False,
    }
    return ("verified" if not reasons else "needs_review"), quality


def evidence_matrix_hits(record: dict[str, Any], *, limit: int = 20) -> list[dict[str, Any]]:
    rows = [item for item in list(record.get("rows") or []) if isinstance(item, dict)]
    evidence = [item for item in list(record.get("evidence") or []) if isinstance(item, dict)]
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence if str(item.get("id") or "")}
    selected: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for field in MATRIX_CELL_FIELDS:
        for row in rows:
            cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
            cell = cells.get(field) if isinstance(cells.get(field), dict) else {}
            if str(cell.get("support_status") or "") != "grounded" or bool(cell.get("manual_override")):
                continue
            for evidence_id in list(cell.get("evidence_ids") or []):
                key = str(evidence_id or "")
                item = evidence_by_id.get(key)
                if not item or key in seen_evidence:
                    continue
                seen_evidence.add(key)
                selected.append(item)
                break
            if len(selected) >= max(1, int(limit)):
                break
        if len(selected) >= max(1, int(limit)):
            break
    hits: list[dict[str, Any]] = []
    for item in selected[: max(1, int(limit))]:
        hits.append(
            {
                "id": str(item.get("id") or ""),
                "text": _text(item.get("evidence_quote"), limit=_MAX_EVIDENCE_QUOTE),
                "score": float(item.get("score") or 0.0),
                "meta": {
                    "source_path": _text(item.get("source_path"), limit=1_200),
                    "source_name": _text(item.get("source_name"), limit=500),
                    "title": _text(item.get("title"), limit=800),
                    "heading_path": _text(item.get("heading_path"), limit=800),
                    "location_label": _text(item.get("location_label"), limit=500),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "block_id": _text(item.get("block_id"), limit=500),
                    "anchor_id": _text(item.get("anchor_id"), limit=500),
                    "matrix_field": _text(item.get("field"), limit=80),
                },
            }
        )
    return hits


def _cell_value(row: dict[str, Any], field: str) -> str:
    cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
    cell = cells.get(field) if isinstance(cells.get(field), dict) else {}
    return _text(cell.get("value"), limit=_MAX_CELL_VALUE)


def _tabular_value(value: object, *, limit: int) -> str:
    text = _text(value, limit=limit, multiline=True)
    return f"'{text}" if text.startswith(("=", "+", "-", "@")) else text


def evidence_matrix_markdown(record: dict[str, Any]) -> str:
    title = _text(record.get("title"), limit=240) or "Evidence matrix"
    objective = _text(record.get("objective"), limit=4_000, multiline=True)
    status = _text(record.get("quality_status"), limit=40) or "draft"
    revision = max(1, int(record.get("revision") or 1))
    lines = [f"# {title}", "", f"> Evidence status: {status}; revision: {revision}."]
    if objective:
        lines.extend(["", "## Research objective", "", objective])
    headers = ["Paper", "Method", "Dataset / experiment", "Metric", "Key result", "Limitation", "Notes"]
    lines.extend(["", "## Matrix", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    for row in list(record.get("rows") or []):
        if not isinstance(row, dict):
            continue
        values = [
            _text(row.get("paper") or row.get("source_name"), limit=500),
            *[_cell_value(row, field) for field in MATRIX_CELL_FIELDS],
            _text(row.get("notes"), limit=4_000, multiline=True),
        ]
        escaped = [value.replace("|", "\\|").replace("\n", "<br>") or "—" for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    evidence = [item for item in list(record.get("evidence") or []) if isinstance(item, dict)]
    if evidence:
        lines.extend(["", "## Evidence appendix"])
        for item in evidence:
            label = _text(item.get("source_name") or item.get("source_path"), limit=500) or "Source"
            field = _text(item.get("field"), limit=80)
            locator = _text(item.get("heading_path") or item.get("location_label"), limit=800)
            lines.extend(["", f"### {label} — {field}{f' — {locator}' if locator else ''}"])
            quote = _text(item.get("evidence_quote"), limit=_MAX_EVIDENCE_QUOTE, multiline=True)
            if quote:
                lines.extend(["", f"> {quote.replace(chr(10), chr(10) + '> ')}"])
    return "\n".join(lines).strip() + "\n"


def evidence_matrix_csv(record: dict[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "paper",
            "source_path",
            "method",
            "dataset_or_experiment",
            "metric",
            "key_result",
            "limitation",
            "notes",
        ]
    )
    for row in list(record.get("rows") or []):
        if not isinstance(row, dict):
            continue
        writer.writerow(
            [
                _tabular_value(row.get("paper") or row.get("source_name"), limit=500),
                _tabular_value(row.get("source_path"), limit=1_200),
                *[_tabular_value(_cell_value(row, field), limit=_MAX_CELL_VALUE) for field in MATRIX_CELL_FIELDS],
                _tabular_value(row.get("notes"), limit=4_000),
            ]
        )
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def evidence_matrix_xlsx(record: dict[str, Any]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Evidence Matrix"
    headers = [
        "Paper",
        "Source path",
        "Method",
        "Dataset / experiment",
        "Metric",
        "Key result",
        "Limitation",
        "Notes",
    ]
    sheet.append(headers)
    for row in list(record.get("rows") or []):
        if not isinstance(row, dict):
            continue
        sheet.append(
            [
                _tabular_value(row.get("paper") or row.get("source_name"), limit=500),
                _tabular_value(row.get("source_path"), limit=1_200),
                *[_tabular_value(_cell_value(row, field), limit=_MAX_CELL_VALUE) for field in MATRIX_CELL_FIELDS],
                _tabular_value(row.get("notes"), limit=4_000),
            ]
        )
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    widths = [28, 42, 48, 42, 34, 48, 42, 36]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[chr(64 + index)].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    evidence_sheet = workbook.create_sheet("Evidence")
    evidence_headers = ["ID", "Paper", "Field", "Locator", "Page", "Evidence quote", "Source path"]
    evidence_sheet.append(evidence_headers)
    for item in list(record.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        evidence_sheet.append(
            [
                _tabular_value(item.get("id"), limit=120),
                _tabular_value(item.get("source_name"), limit=500),
                _tabular_value(item.get("field"), limit=80),
                _tabular_value(item.get("heading_path") or item.get("location_label"), limit=800),
                item.get("page_start") or "",
                _tabular_value(item.get("evidence_quote"), limit=_MAX_EVIDENCE_QUOTE),
                _tabular_value(item.get("source_path"), limit=1_200),
            ]
        )
    for cell in evidence_sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
    for index, width in enumerate([22, 28, 18, 36, 10, 72, 42], start=1):
        evidence_sheet.column_dimensions[chr(64 + index)].width = width
    for row in evidence_sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    evidence_sheet.freeze_panes = "A2"
    evidence_sheet.auto_filter.ref = evidence_sheet.dimensions
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()
