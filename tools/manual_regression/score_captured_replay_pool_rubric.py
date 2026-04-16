from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RAW_SUPPORT_RE = re.compile(r"\[\[SUPPORT:[^\]]+\]\]")
_RAW_CITE_RE = re.compile(r"\[\[CITE:[^\]]+\]\]")
_ROLE_EXPLANATION_RE = re.compile(r"\bwhat\s+(?:is|are)\b.{0,80}\bdoing here\b", flags=re.I)
_PROMPT_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
_ROLE_ACTION_HINTS = (
    "convert",
    "converts",
    "map",
    "maps",
    "registration",
    "register",
    "phase-correlation",
    "phase correlation",
    "shift vector",
    "shift vectors",
    "estimate",
    "estimates",
    "apply",
    "applies",
    "align",
    "alignment",
    "summation",
    "sum",
    "robust",
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _is_beginner_component_role_prompt(prompt: str) -> bool:
    p = _normalize_text(prompt)
    if not p:
        return False
    if any(
        token in p
        for token in (
            "figure",
            "fig.",
            "panel",
            "reference",
            "citation",
            "图",
            "子图",
            "引用",
            "参考文献",
            "equation",
            "eq.",
            "eq(",
            "公式",
            "式",
        )
    ):
        return False
    has_beginner = any(
        token in p
        for token in (
            "beginner",
            "new to this paper",
            "just getting started",
            "getting started",
            "in simple terms",
            "通俗",
            "入门",
            "小白",
        )
    )
    has_role = bool(_ROLE_EXPLANATION_RE.search(p)) or "what role do they play" in p or "what role does it play" in p
    return bool(has_beginner and has_role)


def _is_discussion_only_prompt(prompt: str) -> bool:
    p = _normalize_text(prompt)
    if not p:
        return False
    return bool(
        re.search(r"\b(?:from\s+the\s+)?discussion(?:\s+section)?\s+only\b", p, flags=re.IGNORECASE)
        or re.search(r"\b(?:from\s+the\s+)?future(?:\s+work)?(?:\s+section)?\s+only\b", p, flags=re.IGNORECASE)
        or ("discussion section" in p and "future" in p)
        or ("discussion" in p and "future direction" in p)
        or ("discussion" in p and "future work" in p)
        or ("future work section" in p and "only" in p)
    )


def _is_strength_limits_prompt(prompt: str) -> bool:
    p = _normalize_text(prompt)
    if not p:
        return False
    return bool(
        re.search(
            r"\b(?:limitation|limitations|weakness|weaknesses|trade[\s-]?off|trade[\s-]?offs|what is missing|strongest evidence)\b",
            p,
            flags=re.IGNORECASE,
        )
        or any(token in p for token in ("局限", "不足", "缺点", "薄弱", "证据"))
    )


def _has_uncertainty_disclaimer(text: str) -> bool:
    t = (text or "").lower()
    return any(
        p in t
        for p in [
            "not stated",
            "not specified",
            "does not specify",
            "does not mention",
            "cannot be determined",
            "i can't find",
            "i cannot find",
            "unclear",
            "no further explanation",
            "the only mention",
            "only mention",
            "only mentions",
            "not explained",
        ]
    )


def _readability_score(text: str) -> float:
    t = str(text or "").strip()
    if not t:
        return 0.0
    has_lines = "\n" in t
    has_list = bool(re.search(r"^\s*(?:-|\*|\d+\.)\s+", t, flags=re.M))
    raw_markers = _RAW_SUPPORT_RE.search(t) or _RAW_CITE_RE.search(t)
    length = len(t)
    score = 2.0
    if length >= 220:
        score += 1.2
    if length >= 600:
        score += 0.6
    if has_lines:
        score += 0.5
    if has_list:
        score += 0.5
    if raw_markers:
        score -= 1.0
    return float(_clamp(score, 0.0, 5.0))


def _visible_direct_segments(prov: dict[str, Any]) -> list[dict[str, Any]]:
    segs = prov.get("segments") if isinstance(prov.get("segments"), list) else []
    out: list[dict[str, Any]] = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        if str(seg.get("evidence_mode") or "").strip().lower() != "direct":
            continue
        if str(seg.get("locate_policy") or "").strip().lower() == "hidden" and not bool(seg.get("must_locate")):
            continue
        out.append(seg)
    return out


def _infer_primary_hit_level_from_provenance(prov: dict[str, Any]) -> str:
    visible_direct = _visible_direct_segments(prov)
    if not visible_direct:
        return "none"
    hit = str(visible_direct[0].get("hit_level") or "").strip().lower()
    return hit if hit in {"exact", "block", "heading", "none"} else "none"


def _case_family(case: dict[str, Any]) -> str:
    prompt = str(case.get("prompt") or "").strip()
    prompt_norm = _normalize_text(prompt)
    raw = str(case.get("intent_family") or case.get("tag") or "").strip().lower()
    answer_norm = _normalize_text(case.get("assistant_content") or "")
    if _is_beginner_component_role_prompt(prompt):
        return "overview"
    if "discussion" in raw:
        return "discussion_only"
    if "strength" in raw or "limitation" in raw:
        return "strength_limits"
    if "citation" in raw:
        return "citation_lookup"
    if "equation" in raw:
        return "equation"
    if "figure" in raw:
        return "figure"
    if raw in {"method", "reproduce"}:
        return "method"
    if _is_discussion_only_prompt(prompt):
        return "discussion_only"
    if _is_strength_limits_prompt(prompt):
        return "strength_limits"
    if any(token in prompt_norm for token in ("equation", "eq.", "eq(", "式", "公式")):
        return "equation"
    if any(token in prompt_norm for token in ("figure", "fig.", "panel", "图", "子图")):
        return "figure"
    if any(token in prompt_norm for token in ("reference", "citation", "引用", "参考文献")):
        return "citation_lookup"
    if any(token in prompt_norm for token in ("method", "pipeline", "训练", "实现")):
        return "method"
    if "[[cite:" in answer_norm or "[cite:" in answer_norm:
        return "citation_lookup"
    return "overview"


def _prompt_component_terms(prompt: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in _PROMPT_ACRONYM_RE.findall(str(prompt or "")):
        norm = term.upper()
        if len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _count_prompt_term_hits(answer: str, prompt_terms: list[str]) -> int:
    ans_lower = str(answer or "").lower()
    return sum(1 for term in prompt_terms if term.lower() in ans_lower)


def _has_grounded_component_role_answer(prompt: str, answer: str, prov: dict[str, Any]) -> bool:
    prompt_terms = _prompt_component_terms(prompt)
    if len(prompt_terms) >= 2 and _count_prompt_term_hits(answer, prompt_terms) < 2:
        return False
    ans_lower = str(answer or "").lower()
    action_hits = sum(1 for token in _ROLE_ACTION_HINTS if token in ans_lower)
    if action_hits < 2:
        return False
    if _has_uncertainty_disclaimer(answer):
        return False
    return len(_visible_direct_segments(prov)) >= 2


@dataclass
class Score:
    overall_100: float
    question_hit: float
    evidence_consistency: float
    locate_first_click: float
    uncertainty_handling: float
    readability: float


def _score_case(case: dict[str, Any]) -> Score:
    prompt = str(case.get("prompt") or "").strip()
    answer = str(case.get("assistant_content") or "").strip()
    prov = case.get("provenance") if isinstance(case.get("provenance"), dict) else {}
    is_role_prompt = _is_beginner_component_role_prompt(prompt)
    grounded_role_answer = _has_grounded_component_role_answer(prompt, answer, prov)

    hit = _infer_primary_hit_level_from_provenance(prov)
    has_raw = bool(_RAW_SUPPORT_RE.search(answer) or _RAW_CITE_RE.search(answer))

    # 1) 命中问题: minimal proxy by length + some overlap with prompt nouns (very rough).
    question_hit = 3.5 if len(answer) >= 160 else 2.0
    if prompt and answer and (prompt.split(" ", 1)[0].lower() in answer.lower()):
        question_hit += 0.4
    if is_role_prompt and grounded_role_answer:
        question_hit = max(question_hit, 5.0)
    question_hit = float(_clamp(question_hit - (0.6 if has_raw else 0.0), 0.0, 5.0))

    # 2) 证据一致性: penalize raw markers; reward having provenance segments.
    segs = prov.get("segments") if isinstance(prov.get("segments"), list) else []
    evidence_consistency = 3.8 if segs else 2.4
    if is_role_prompt and grounded_role_answer and not has_raw:
        evidence_consistency = max(evidence_consistency, 5.0)
    evidence_consistency = float(_clamp(evidence_consistency - (1.2 if has_raw else 0.0), 0.0, 5.0))

    # 3) 可定位性(一跳)
    locate_first_click = {"exact": 5.0, "block": 4.0, "heading": 2.5, "none": 1.0}.get(hit, 1.0)

    # 4) 不确定性处理
    if hit in {"heading", "none"}:
        uncertainty_handling = 4.5 if _has_uncertainty_disclaimer(answer) else 2.0
    else:
        uncertainty_handling = 4.0 if not _has_uncertainty_disclaimer(answer) else 3.4
    if is_role_prompt and grounded_role_answer:
        uncertainty_handling = max(uncertainty_handling, 4.6)

    # 5) 可读性
    readability = _readability_score(answer)
    if is_role_prompt and grounded_role_answer and bool(re.search(r"^\s*-\s+", answer, flags=re.M)):
        readability = max(readability, 4.6)

    overall = round((question_hit + evidence_consistency + locate_first_click + uncertainty_handling + readability) / 25.0 * 100.0, 1)
    return Score(
        overall_100=overall,
        question_hit=round(question_hit, 2),
        evidence_consistency=round(evidence_consistency, 2),
        locate_first_click=round(locate_first_click, 2),
        uncertainty_handling=round(uncertainty_handling, 2),
        readability=round(readability, 2),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rubric scoring for captured paper_guide replay pool jsonl.")
    ap.add_argument("--pool", default="tests/replay/paper_guide_failure_pool_captured.jsonl", help="Input jsonl")
    ap.add_argument("--top-bad", type=int, default=10, help="Show worst N cases")
    args = ap.parse_args(argv)

    pool_path = Path(str(args.pool)).expanduser().resolve()
    rows = [json.loads(l) for l in pool_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    enriched: list[tuple[float, dict[str, Any], Score]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        s = _score_case(r)
        enriched.append((s.overall_100, r, s))
    enriched.sort(key=lambda t: t[0])

    n = max(1, int(args.top_bad))
    worst = enriched[:n]
    print(f"n={len(enriched)} showing_worst={len(worst)} pool={pool_path}")
    for overall, rec, s in worst:
        rid = str(rec.get("id") or "")
        fam = _case_family(rec)
        prompt = str(rec.get("prompt") or "").strip().replace("\n", " ")[:140]
        leak = ("raw_marker" if ("[[CITE:" in str(rec.get("assistant_content") or "") or "[[SUPPORT:" in str(rec.get("assistant_content") or "")) else "")
        print(f"\n[{overall:>5.1f}] {rid} fam={fam} {leak}")
        print(f"prompt: {prompt}")
        print(f"subscores: hit={s.question_hit} ev={s.evidence_consistency} locate={s.locate_first_click} uncertain={s.uncertainty_handling} read={s.readability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
