from __future__ import annotations

from api.reference_value_utils import _non_negative_float, _positive_int


def _anchor_kind_prefix(kind: str) -> str:
    k = str(kind or "").strip().lower()
    if k == "figure":
        return "图示语义命中"
    if k == "equation":
        return "公式语义命中"
    if k == "table":
        return "表格语义命中"
    if k == "theorem":
        return "定理语义命中"
    if k == "lemma":
        return "引理语义命中"
    if k == "definition":
        return "定义语义命中"
    return "锚点语义命中"


def _anchor_kind_label(kind: str, number: int) -> str:
    k = str(kind or "").strip().lower()
    n = _positive_int(number)
    if (not k) or n <= 0:
        return ""
    if k == "figure":
        return f"图{n}"
    if k == "equation":
        return f"公式{n}"
    if k == "table":
        return f"表{n}"
    if k == "theorem":
        return f"定理{n}"
    if k == "lemma":
        return f"引理{n}"
    if k == "definition":
        return f"定义{n}"
    return f"{k} {n}"


def _build_semantic_badges(
    *,
    anchor_target_kind: str,
    anchor_target_number: int,
    anchor_match_score: float,
    explicit_doc_match_score: float,
) -> list[dict]:
    badges: list[dict] = []
    anchor_label = _anchor_kind_label(anchor_target_kind, anchor_target_number)
    if anchor_label:
        badges.append(
            {
                "text": f"{_anchor_kind_prefix(anchor_target_kind)} {anchor_label}",
                "score": _non_negative_float(anchor_match_score),
            }
        )
        return badges
    if _non_negative_float(explicit_doc_match_score) >= 6.0:
        badges.append({"text": "文档语义直连", "score": _non_negative_float(explicit_doc_match_score)})
    return badges


__all__ = [
    "_anchor_kind_label",
    "_anchor_kind_prefix",
    "_build_semantic_badges",
]
