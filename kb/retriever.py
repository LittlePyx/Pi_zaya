from __future__ import annotations

import copy
from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi

from .tokenize import tokenize


_TABLE_QUERY_RE = re.compile(
    r"(?:\b(?:table|tabular|highest|lowest|best|worst|maximum|minimum|compare|comparison|"
    r"PSNR|SSIM|LPIPS|RMSE|NMSE|MSE|MAE|SNR|FID|mAP|F1|AUC|IoU)\b|"
    r"表格|表中|最高|最低|最好|最差|最大|最小|比较|对比)",
    flags=re.I,
)
_TABLE_COMPARISON_RE = re.compile(
    r"(?:\b(?:highest|lowest|best|worst|maximum|minimum|compare|comparison|rank|ranking)\b|"
    r"最高|最低|最好|最差|最大|最小|比较|对比|排名)",
    flags=re.I,
)
_TABLE_METHOD_QUERY_RE = re.compile(
    r"\b(?:method|model|network|algorithm|architecture)\b|方法|模型|网络|算法|架构",
    flags=re.I,
)
_TABLE_BENCHMARK_QUERY_RE = re.compile(
    r"\b(?:benchmark|leaderboard|test\s+set|dataset)\b|基准|榜单|测试集|数据集",
    flags=re.I,
)
_TABLE_VARIANT_QUERY_RE = re.compile(
    r"\b(?:ablation|blocks?|layers?|depth|width|variant|setting|configuration|component|activation|sigma|"
    r"sampling\s+ratio|CS\s+ratio|SR|patch(?:es)?)\b|"
    r"消融|块数|层数|深度|宽度|变体|设置|配置|组件|激活|采样率",
    flags=re.I,
)


@dataclass
class Hit:
    score: float
    text: str
    meta: dict
    chunk_id: str


class BM25Retriever:
    def __init__(self, chunks: list[dict]) -> None:
        # New users may run the app before ingesting any Markdown into the DB.
        # BM25Okapi cannot be initialized with an empty corpus, so we treat empty DB as "no hits".
        self._chunks = list(chunks or [])
        self._corpus_tokens = [tokenize(c.get("text", "")) for c in self._chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        adjusted_scores = [float(score) for score in scores]
        for idx, chunk in enumerate(self._chunks):
            meta = chunk.get("meta") or {}
            if meta.get("evidence_ready") is False:
                adjusted_scores[idx] = float("-inf")
        if _TABLE_QUERY_RE.search(str(query or "")):
            comparison_intent = bool(_TABLE_COMPARISON_RE.search(str(query or "")))
            method_intent = bool(_TABLE_METHOD_QUERY_RE.search(str(query or "")))
            benchmark_intent = bool(_TABLE_BENCHMARK_QUERY_RE.search(str(query or "")))
            variant_intent = bool(_TABLE_VARIANT_QUERY_RE.search(str(query or "")))
            query_tokens = {str(token or "").lower() for token in q if str(token or "").strip()}
            for idx, chunk in enumerate(self._chunks):
                meta = chunk.get("meta") or {}
                if meta.get("evidence_ready") is False:
                    continue
                kind = str(meta.get("structured_kind") or "").strip().lower()
                if kind == "table_metric":
                    label_tokens = {
                        str(token or "").lower()
                        for token in tokenize(str(meta.get("table_metric_label") or ""))
                        if str(token or "").strip()
                    }
                    overlap_ratio = (
                        len(label_tokens.intersection(query_tokens)) / len(label_tokens)
                        if label_tokens
                        else 0.0
                    )
                    if adjusted_scores[idx] <= 0.0 and overlap_ratio <= 0.0:
                        continue
                    adjusted_scores[idx] = adjusted_scores[idx] * 1.35 + (0.9 if comparison_intent else 0.08)
                    if label_tokens:
                        adjusted_scores[idx] += 3.5 * overlap_ratio
                        if overlap_ratio >= 0.999:
                            adjusted_scores[idx] += 0.5 + (0.8 * min(4, len(label_tokens)))
                    subject_kind = str(meta.get("table_subject_kind") or "").strip().lower()
                    if subject_kind == "method" and not variant_intent:
                        adjusted_scores[idx] += (
                            7.0
                            if method_intent or benchmark_intent
                            else (2.5 if comparison_intent else 0.0)
                        )
                    elif subject_kind == "variant":
                        if variant_intent:
                            adjusted_scores[idx] += 4.0
                        elif method_intent or benchmark_intent:
                            adjusted_scores[idx] -= 3.0
                        elif comparison_intent:
                            adjusted_scores[idx] -= 1.0
                elif kind == "table_row":
                    if adjusted_scores[idx] <= 0.0:
                        continue
                    adjusted_scores[idx] = adjusted_scores[idx] * 1.18 + (0.02 if comparison_intent else 0.04)
        try:
            best = float(max(adjusted_scores)) if adjusted_scores else 0.0
        except Exception:
            best = 0.0
        # If nothing matches (common for cross-lingual queries), don't return arbitrary documents.
        if best <= 0.0:
            return []
        eligible_idxs = [
            idx
            for idx, _score in enumerate(adjusted_scores)
            if (self._chunks[idx].get("meta") or {}).get("evidence_ready") is not False
        ]
        idxs = sorted(eligible_idxs, key=lambda i: adjusted_scores[i], reverse=True)[: max(1, top_k)]
        hits: list[dict] = []
        for i in idxs:
            c = self._chunks[i]
            hits.append(
                {
                    "score": float(adjusted_scores[i]),
                    "id": c.get("id", str(i)),
                    "text": c.get("text", ""),
                    # Callers enrich hit metadata in place.  Keep the cached
                    # corpus immutable so one conversation cannot leak state
                    # into a later retrieval.
                    "meta": copy.deepcopy(c.get("meta", {})),
                }
            )
        return hits
