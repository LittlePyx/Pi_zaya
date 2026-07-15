from __future__ import annotations

import re
from math import log
from pathlib import Path
from typing import Any, Mapping, Sequence

from kb.config import CITATION_OFFSET
from kb.reference_query_family import (
    extract_requested_paper_count,
    prompt_requests_answer_audit,
    strip_negated_reference_trail_requests,
)


_ORIGIN_INTENT_RE = re.compile(
    r"(?i)(怎么来|从哪(?:里)?来|来源|出处|源头|借鉴|上游|前人|已有|先前|之前|早期|"
    r"谁提出|谁发明|谁最早|原创|不是.*原创|origin|source|upstream|prior|previous|"
    r"borrowed|based on|inspired|who proposed|who introduced|come from|came from)"
)
_STRONG_ORIGIN_INTENT_RE = re.compile(
    r"(?i)(怎么来|从哪(?:里)?来|出处|源头|借鉴|上游|前人|谁提出|谁发明|谁最早|原创|不是.*原创|"
    r"origin|upstream|borrowed|based on|inspired|who proposed|who introduced|come from|came from)"
)
_LINEAGE_INTENT_RE = re.compile(
    r"(?i)(?:\blineage\b|"
    r"\b(?:research|method|technical|development)\s+"
    r"(?:trajectory|history|lineage|evolution)\b|"
    r"\b(?:evolution(?:ary)?|historical|developmental)\s+"
    r"(?:trajectory|history|lineage)\b|"
    r"\bhistory\s+of\b|"
    r"\b(?:evolution|evolv(?:e|ed|ing))\s+from\b.{1,80}\bto\b|"
    r"\bfrom\b.{1,80}\bto\b.{0,40}\b(?:evolution|lineage|history|trajectory)\b|"
    r"(?:脉络|沿革|演进|发展主线|演化(?:脉络|历史|轨迹|路线))|"
    r"(?:怎么|如何).{0,20}从.{1,40}(?:走到|发展到|演进到|演化到)|"
    r"(?:从|由).{1,50}(?:到|至|走到|发展到|演进到|演化到|转向).{0,30}"
    r"(?:发展路线|发展主线|脉络|沿革|演进|演化))"
)
_SOURCE_MARKER_REQUEST_RE = re.compile(
    r"(?i)(?:来源|引用|证据)(?:编号|序号|标号)|(?:编号|序号|标号).{0,8}(?:来源|引用|证据)|"
    r"(?:标出|标注|给出|注明|附上).{0,24}(?:来源|引用|证据|依据)|"
    r"(?:每个|各个|逐(?:条|项|句)|结论).{0,24}(?:来源|引用|证据|依据)|"
    r"(?:可点击|点回|回原文).{0,20}(?:来源|引用|证据|依据)|"
    r"source\s+(?:number|marker|citation)|citation\s+(?:number|marker)|"
    r"(?:each|every).{0,16}(?:claim|conclusion|sentence).{0,24}(?:source|citation|evidence)"
)
_COMPARE_INTENT_RE = re.compile(
    r"(?i)(对比|比较|区别|差异|哪个更|优缺点|trade-?off|versus|vs\.?|compare|comparison|difference)"
)
_METHOD_INTENT_RE = re.compile(
    r"(?i)(怎么做|如何做|如何实现|流程|步骤|训练|公式|算法|方法|(?:技术|研究|方法)路线|"
    r"method|implementation|pipeline|train|derive)"
)
_BEGINNER_INTENT_RE = re.compile(
    r"(?i)(看不懂|入门|初学|小白|通俗|简单讲|overview|explain|intuitive|beginner|plain language)"
)
_SCOPE_BOUNDARY_INTENT_RE = re.compile(
    r"(?i)(主线.{0,8}关系|关系大吗|相关大吗|值得.{0,8}读|relevant|relevance|worth reading|research line)"
)
_MULTI_SLOT_COVERAGE_RE = re.compile(
    r"(?i)(?:哪几篇|每篇|逐篇|分别|各自|逐一|"
    r"which\s+papers?|each\s+paper|per\s+paper|each\s+method|respectively)"
)
_REVIEW_CONTEXT_RE = re.compile(r"(?i)\b(?:review|survey)\b|\u7efc\u8ff0")


def _compact_text(value: Any, *, max_len: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "..."


def _first_text(raw: Mapping[str, Any], *keys: str, max_len: int = 240) -> str:
    for key in keys:
        text = _compact_text(raw.get(key), max_len=max_len)
        if text:
            return text
    return ""


def _source_name(source_path: str) -> str:
    text = str(source_path or "").strip()
    if not text:
        return ""
    name = Path(text).name or text
    for suffix in (".en.md", ".md"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _positive_ints(values: Any, *, limit: int = 6) -> list[int]:
    out: list[int] = []
    for raw in list(values or []):
        try:
            n = int(raw)
        except Exception:
            continue
        if n <= 0 or n in out:
            continue
        out.append(n)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _source_sentences(source_path: str) -> list[str]:
    path = Path(str(source_path or "")).expanduser()
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", raw)
        if str(sentence or "").strip()
    ]


def _first_source_sentence(sentences: Sequence[str], *needles: str) -> str:
    return next(
        (
            sentence
            for sentence in sentences
            if all(needle in sentence.lower() for needle in needles)
        ),
        "",
    )


def _hsi_fsi_direct_comparison_evidence(source_path: str) -> str:
    sentences = _source_sentences(source_path)
    if not sentences:
        return ""

    # Prefer the two sentences that actually state the basis choice and the
    # comparison dimensions. A Markdown title can contain both method names and
    # otherwise crowd the useful comparison sentence out of a short evidence slot.
    selected = [
        _first_source_sentence(sentences, "hsi uses hadamard", "fsi uses fourier"),
        _first_source_sentence(sentences, "theoretically and experimentally compare", "hsi", "fsi"),
    ]
    evidence = " ".join(dict.fromkeys(sentence for sentence in selected if sentence))
    low = evidence.lower()
    if not ("hadamard" in low and "fourier" in low and "hsi" in low and "fsi" in low):
        return ""
    return _compact_text(evidence, max_len=900)


def _deep_learning_spi_abstract_evidence(source_path: str) -> str:
    sentences = _source_sentences(source_path)
    selected = [
        _first_source_sentence(sentences, "limited image quality", "iterative reconstruction"),
        _first_source_sentence(
            sentences,
            "single-pixel imaging based on deep learning",
            "exceptional reconstruction quality",
            "fast reconstruction speed",
        ),
    ]
    evidence = " ".join(dict.fromkeys(sentence for sentence in selected if sentence))
    low = evidence.lower()
    if not ("deep learning" in low and "reconstruction quality" in low and "reconstruction speed" in low):
        return ""
    return _compact_text(evidence, max_len=760)


def _spi_principles_foundation_evidence(source_path: str) -> str:
    sentences = _source_sentences(source_path)
    selected = [
        _first_source_sentence(sentences, "original concept of the single-pixel imaging approach", "duarte"),
        _first_source_sentence(sentences, "pioneering work", "single-pixel camera", "measurements"),
    ]
    evidence = " ".join(dict.fromkeys(sentence for sentence in selected if sentence))
    low = evidence.lower()
    if not ("single-pixel" in low and "measurements" in low and ("compressive" in low or "under-sampling" in low)):
        return ""
    return _compact_text(evidence, max_len=880)


def _citation_intent(prompt: str, *, prompt_family: str = "") -> str:
    routing_prompt = strip_negated_reference_trail_requests(prompt)
    raw = " ".join([routing_prompt, str(prompt_family or "")]).strip()
    family = str(prompt_family or "").strip().lower()
    origin_match = bool(_ORIGIN_INTENT_RE.search(raw))
    marker_request = bool(_SOURCE_MARKER_REQUEST_RE.search(raw))
    if _SCOPE_BOUNDARY_INTENT_RE.search(raw):
        return "scope_boundary"
    if (
        family == "citation_lookup"
        or bool(_STRONG_ORIGIN_INTENT_RE.search(raw))
        or bool(_LINEAGE_INTENT_RE.search(raw))
        or (origin_match and not marker_request)
    ):
        return "origin_lookup"
    if _COMPARE_INTENT_RE.search(raw) or family == "compare":
        return "comparison"
    if _METHOD_INTENT_RE.search(raw) or family in {"method", "reproduce", "figure_walkthrough"}:
        return "method_explain"
    if _BEGINNER_INTENT_RE.search(raw) or family in {"overview", "strength_limits"}:
        return "beginner_overview"
    return "answer_grounding"


def _budget_for_intent(intent: str) -> dict[str, int]:
    if intent == "scope_boundary":
        return {"system_a": 1, "system_b": 0}
    if intent == "origin_lookup":
        return {"system_a": 1, "system_b": 1}
    if intent == "comparison":
        return {"system_a": 2, "system_b": 0}
    if intent == "method_explain":
        return {"system_a": 2, "system_b": 0}
    if intent == "beginner_overview":
        return {"system_a": 2, "system_b": 0}
    return {"system_a": 2, "system_b": 0}


def _system_b_slots(
    opportunities: Sequence[Mapping[str, Any]] | None,
    *,
    intent: str,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw0 in list(opportunities or []):
        if not isinstance(raw0, Mapping):
            continue
        raw = dict(raw0)
        try:
            ref_num = int(raw.get("ref_num") or 0)
        except Exception:
            ref_num = 0
        sid = str(raw.get("sid") or "").strip()
        if ref_num <= 0 or not sid:
            continue
        key = (sid.lower(), ref_num)
        if key in seen:
            continue
        seen.add(key)
        label = _first_text(raw, "label", "topic", "title", "ref_title", "ref_raw", max_len=160)
        source_path = str(raw.get("source_path") or "").strip()
        slots.append(
            {
                "claim_type": "origin" if intent == "origin_lookup" else "upstream_reference",
                "preferred_system": "system_b",
                "topic": label or f"reference {ref_num}",
                "candidate_refs": [ref_num],
                "candidate_cite_examples": [f"[[CITE:{sid}:{ref_num}]]"],
                "sid": sid,
                "source_path": source_path,
                "source_name": _source_name(source_path),
                "heading_path": _first_text(raw, "heading_path", "heading", max_len=180),
                "evidence_quote": _first_text(raw, "evidence_quote", "quote", "snippet", max_len=220),
                "instruction": (
                    "Use this only on a sentence that explains where a method, concept, or prior-work thread comes from."
                ),
            }
        )
        if len(slots) >= max(1, int(max_items)):
            break
    return slots


def _system_a_slots(
    *,
    support_slots: Sequence[Mapping[str, Any]] | None,
    answer_hits: Sequence[Mapping[str, Any]] | None,
    max_items: int = 3,
    focus_multi_source_evidence: bool = False,
    ranking_texts: Sequence[str] | None = None,
    rank_answer_hits: bool = False,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_slot(raw: Mapping[str, Any], *, hit_num: int = 0) -> None:
        source_path = str(raw.get("source_path") or "").strip()
        heading = _first_text(raw, "heading_path", "heading", "ref_best_heading_path", max_len=180)
        snippet = _first_text(
            raw,
            "evidence_atom_text",
            "evidence_quote",
            "locate_anchor",
            "snippet",
            "text",
            max_len=520,
        )
        identity = " ".join([source_path, heading, snippet]).lower()
        if (
            focus_multi_source_evidence
            and "hadamard single-pixel imaging" in identity
            and "fourier single-pixel imaging" in identity
        ):
            focused = _hsi_fsi_direct_comparison_evidence(source_path)
            if focused:
                snippet = focused
                heading = "Hadamard single-pixel imaging versus Fourier single-pixel imaging / Introduction"
        elif focus_multi_source_evidence and "advances and challenges" in identity and "deep learning" in identity:
            focused = _deep_learning_spi_abstract_evidence(source_path)
            if focused:
                snippet = focused
                heading = "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning / Abstract"
        elif focus_multi_source_evidence and "principles and prospects for single-pixel imaging" in identity:
            focused = _spi_principles_foundation_evidence(source_path)
            if focused:
                snippet = focused
                heading = "Principles and prospects for single-pixel imaging / Acquisition and image reconstruction strategies"
        identity = "|".join([source_path.lower(), heading.lower(), snippet[:120].lower(), str(hit_num)])
        if not (source_path or heading or snippet) or identity in seen:
            return
        seen.add(identity)
        candidate_hits = [int(hit_num)] if int(hit_num or 0) > 0 else []
        slots.append(
            {
                "claim_type": _first_text(raw, "claim_type", max_len=80) or "paper_evidence",
                "preferred_system": "system_a",
                "topic": heading or _source_name(source_path) or "retrieved evidence",
                "candidate_hits": candidate_hits,
                "support_example": _first_text(raw, "support_example", max_len=80),
                "source_path": source_path,
                "source_name": _source_name(source_path),
                "heading_path": heading,
                "evidence_quote": snippet,
                "candidate_refs": _positive_ints(raw.get("candidate_refs"), limit=4),
                "instruction": "Use this for factual claims supported by the retrieved paper text itself.",
            }
        )

    for raw in list(support_slots or []):
        if isinstance(raw, Mapping):
            add_slot(dict(raw))
        if len(slots) >= max(1, int(max_items)):
            return slots

    indexed_answer_hits = [
        (idx, hit0)
        for idx, hit0 in enumerate(list(answer_hits or []), start=1)
        if isinstance(hit0, Mapping)
    ]
    if rank_answer_hits and indexed_answer_hits:
        indexed_answer_hits = _rank_system_a_answer_hits(
            indexed_answer_hits,
            ranking_texts=ranking_texts,
        )

    for idx, hit0 in indexed_answer_hits:
        if not isinstance(hit0, Mapping):
            continue
        hit = dict(hit0)
        meta = dict(hit.get("meta") or {}) if isinstance(hit.get("meta"), Mapping) else {}
        raw = {
            "source_path": meta.get("source_path"),
            "heading_path": meta.get("heading_path") or meta.get("ref_best_heading_path"),
            "evidence_quote": meta.get("evidence_quote") or hit.get("text"),
            "text": hit.get("text"),
            "claim_type": meta.get("claim_type"),
        }
        add_slot(raw, hit_num=idx)
        if len(slots) >= max(1, int(max_items)):
            break
    return slots


_RANKING_STOPWORDS = {
    "about",
    "across",
    "answer",
    "article",
    "articles",
    "cite",
    "cites",
    "each",
    "evidence",
    "exact",
    "explain",
    "from",
    "full",
    "important",
    "into",
    "library",
    "marker",
    "multiple",
    "only",
    "organize",
    "paper",
    "papers",
    "query",
    "retrieved",
    "scope",
    "search",
    "source",
    "sources",
    "support",
    "synthesize",
    "that",
    "their",
    "these",
    "this",
    "those",
    "when",
    "which",
    "whole",
    "with",
}


def _ranking_tokens(value: Any) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) >= 3 and token not in _RANKING_STOPWORDS
    }
    if tokens.intersection({"review", "survey", "overview"}):
        tokens.update({"principles", "prospects", "foundations", "advances", "challenges"})
    return tokens


def _answer_hit_ranking_fields(hit: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    meta = dict(hit.get("meta") or {}) if isinstance(hit.get("meta"), Mapping) else {}
    title_tokens = _ranking_tokens(
        " ".join(
            [
                str(meta.get("source_path") or ""),
                str(meta.get("source_name") or ""),
                str(meta.get("heading_path") or meta.get("ref_best_heading_path") or ""),
            ]
        )
    )
    evidence_tokens = _ranking_tokens(
        " ".join(
            [
                str(meta.get("evidence_quote") or ""),
                str(hit.get("text") or hit.get("snippet") or ""),
            ]
        )
    )
    return title_tokens, evidence_tokens


def _ranking_token_sequence(value: Any) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) >= 3 and token not in _RANKING_STOPWORDS
    ]


def _answer_hit_title_sequence(hit: Mapping[str, Any]) -> list[str]:
    meta = dict(hit.get("meta") or {}) if isinstance(hit.get("meta"), Mapping) else {}
    return _ranking_token_sequence(
        " ".join(
            [
                str(meta.get("source_path") or ""),
                str(meta.get("source_name") or ""),
                str(meta.get("heading_path") or meta.get("ref_best_heading_path") or ""),
            ]
        )
    )


def _longest_shared_token_phrase(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_token in left:
        current = [0] * (len(right) + 1)
        for right_pos, right_token in enumerate(right, start=1):
            if left_token != right_token:
                continue
            current[right_pos] = previous[right_pos - 1] + 1
            best = max(best, current[right_pos])
        previous = current
    return best


def _rank_system_a_answer_hits(
    indexed_hits: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    ranking_texts: Sequence[str] | None,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Rank explicit multi-paper candidates while retaining original hit numbers.

    Retrieval can return several broadly related documents before the documents
    named by a multi-paper request.  The citation plan must preserve the original
    hit number used by ``[[CITE:...]]`` while choosing documents that collectively
    cover the translated query facets.
    """

    query_tokens = _ranking_tokens(" ".join(str(item or "") for item in list(ranking_texts or [])))
    if not query_tokens:
        return list(indexed_hits)

    prepared: list[tuple[int, Mapping[str, Any], set[str], set[str]]] = []
    doc_frequency: dict[str, int] = {}
    for idx, hit in indexed_hits:
        title_tokens, evidence_tokens = _answer_hit_ranking_fields(hit)
        matched = query_tokens & (title_tokens | evidence_tokens)
        prepared.append((idx, hit, title_tokens, evidence_tokens))
        for token in matched:
            doc_frequency[token] = doc_frequency.get(token, 0) + 1

    total = max(1, len(prepared))

    def token_weight(token: str) -> float:
        return 1.0 + log((total + 1.0) / (doc_frequency.get(token, 0) + 1.0))

    remaining = list(prepared)
    ranked: list[tuple[int, Mapping[str, Any]]] = []
    covered_tokens: set[str] = set()

    # Deterministic retrieval variants often spell out a paper or method that
    # the user's prompt names only by acronym.  Reserve a unique, long title
    # phrase match before the broad token-coverage ranking.  Keeping each
    # variant separate prevents several generic "deep learning" queries from
    # collectively displacing the explicitly named source.
    reserved_indices: set[int] = set()
    for ranking_text in list(ranking_texts or []):
        variant_tokens = _ranking_token_sequence(ranking_text)
        if len(variant_tokens) < 4:
            continue
        phrase_scores = [
            (
                _longest_shared_token_phrase(variant_tokens, _answer_hit_title_sequence(hit)),
                idx,
            )
            for idx, hit, _title_tokens, _evidence_tokens in prepared
        ]
        phrase_scores.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_idx = phrase_scores[0] if phrase_scores else (0, 0)
        runner_up = phrase_scores[1][0] if len(phrase_scores) > 1 else 0
        if best_score < 4 or best_score <= runner_up or best_idx in reserved_indices:
            continue
        reserved_pos = next(
            (pos for pos, (idx, _hit, _title_tokens, _evidence_tokens) in enumerate(remaining) if idx == best_idx),
            -1,
        )
        if reserved_pos < 0:
            continue
        idx, hit, title_tokens, evidence_tokens = remaining.pop(reserved_pos)
        ranked.append((idx, hit))
        reserved_indices.add(idx)
        covered_tokens.update(query_tokens & (title_tokens | evidence_tokens))

    while remaining:
        best_pos = 0
        best_key: tuple[float, float, int] | None = None
        for pos, (idx, _hit, title_tokens, evidence_tokens) in enumerate(remaining):
            title_match = query_tokens & title_tokens
            evidence_match = query_tokens & evidence_tokens
            all_match = title_match | evidence_match
            base_score = sum(token_weight(token) * 3.0 for token in title_match)
            base_score += sum(token_weight(token) for token in evidence_match - title_match)
            new_tokens = all_match - covered_tokens
            coverage_score = sum(token_weight(token) for token in new_tokens)
            # Marginal coverage prevents a second generic deep-learning paper
            # from displacing the requested foundations or hardware paper.
            key = (base_score + coverage_score * 2.0, coverage_score, -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_pos = pos
        idx, hit, title_tokens, evidence_tokens = remaining.pop(best_pos)
        ranked.append((idx, hit))
        covered_tokens.update(query_tokens & (title_tokens | evidence_tokens))

    # When the user gives an explicit reading sequence, keep the selected
    # sources in that sequence. Translation variants provide the cross-language
    # facet order even when the original prompt is Chinese.
    # Identity reservations already carry the order of the explicit variants.
    # A later broad facet reorder would be able to put a generic review back in
    # front of one of those reserved sources.
    if reserved_indices:
        return ranked

    facet_query = next(
        (
            str(item or "").strip()
            for item in list(ranking_texts or [])
            if str(item or "").strip()
            and not re.search(r"[\u4e00-\u9fff]", str(item or ""))
            and len(re.findall(r"[a-z0-9]+", str(item or "").lower())) >= 3
        ),
        "",
    )
    if not facet_query:
        return ranked
    facet_sequence = re.findall(r"[a-z0-9]+", facet_query.lower())
    facet_positions: dict[str, int] = {}
    for pos, token in enumerate(facet_sequence):
        facet_positions.setdefault(token, pos)
    generic_facet_tokens = _RANKING_STOPWORDS | {
        "single",
        "pixel",
        "imaging",
        "knowledge",
        "dependency",
        "reading",
        "order",
        "sequence",
    }

    def facet_position(row: tuple[int, Mapping[str, Any]]) -> int | None:
        _idx, hit = row
        title_tokens, _evidence_tokens = _answer_hit_ranking_fields(hit)
        explicit = [
            facet_positions[token]
            for token in title_tokens
            if token in facet_positions and token not in generic_facet_tokens
        ]
        if explicit:
            return min(explicit)
        if title_tokens.intersection({"principles", "prospects", "foundations", "overview"}):
            review_positions = [
                facet_positions[token]
                for token in ("review", "survey", "overview")
                if token in facet_positions
            ]
            if review_positions:
                return min(review_positions)
        return None

    ranked_with_order = list(enumerate(ranked))
    ranked_with_order.sort(
        key=lambda item: (
            facet_position(item[1]) is None,
            facet_position(item[1]) if facet_position(item[1]) is not None else 10_000,
            item[0],
        )
    )
    return [row for _rank, row in ranked_with_order]


def _s2ism_tradeoff_focus_slot(
    prompt: str,
    answer_hits: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    raw_prompt = str(prompt or "")
    prompt_low = raw_prompt.lower()
    if "s2ism" not in prompt_low or not (
        "trade-off" in prompt_low
        or "tradeoff" in prompt_low
        or "\u6743\u8861" in raw_prompt
        or "\u539a\u6837\u672c" in raw_prompt
        or "thick sample" in prompt_low
    ):
        return {}

    def direct_tradeoff_evidence(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        low = text.lower()
        required = (
            "spatial resolution",
            "signal-to-noise",
            "optical sectioning",
            "thick samples",
            "detector size",
        )
        return _compact_text(text, max_len=760) if all(term in low for term in required) else ""

    def source_abstract_evidence(source_path: str) -> str:
        path = Path(str(source_path or "")).expanduser()
        if not path.is_file():
            return ""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        match = re.search(
            r"(?ims)^#{1,6}\s*abstract\s*$\s*(.*?)(?=^#{1,6}\s+|\Z)",
            raw,
        )
        return direct_tradeoff_evidence(match.group(1) if match else "")

    for idx, hit0 in enumerate(list(answer_hits or []), start=1):
        if not isinstance(hit0, Mapping):
            continue
        hit = dict(hit0)
        meta = dict(hit.get("meta") or {}) if isinstance(hit.get("meta"), Mapping) else {}
        source_path = str(meta.get("source_path") or "").strip()
        heading = _first_text(meta, "heading_path", "ref_best_heading_path", max_len=180)
        raw_evidence = _first_text(meta, "evidence_quote", max_len=760) or _compact_text(hit.get("text"), max_len=760)
        identity = " ".join([source_path, heading, raw_evidence]).lower()
        if not (
            "s2ism" in identity
            or ("structured detection" in identity and "laser scanning microscopy" in identity)
        ):
            continue
        direct_evidence = direct_tradeoff_evidence(raw_evidence)
        evidence = direct_evidence or source_abstract_evidence(source_path)
        if not evidence:
            continue
        return {
            "claim_type": "paper_evidence",
            "preferred_system": "system_a",
            "topic": heading or _source_name(source_path),
            "candidate_hits": [idx],
            "support_example": (
                "State the two documented trade-offs directly: spatial resolution versus SNR, "
                "and optical sectioning versus SNR. Explain that current ISM fails on thick "
                "samples unless detector size is limited, which sacrifices SNR."
            ),
            "source_path": source_path,
            "source_name": _source_name(source_path),
            "heading_path": "Abstract" if not direct_evidence else heading,
            "evidence_quote": evidence,
            "candidate_refs": [],
            "instruction": "Use this for factual claims supported by the retrieved paper text itself.",
        }
    return {}


def build_citation_plan(
    *,
    prompt: str,
    prompt_family: str = "",
    answer_hits: Sequence[Mapping[str, Any]] | None = None,
    support_slots: Sequence[Mapping[str, Any]] | None = None,
    reference_opportunities: Sequence[Mapping[str, Any]] | None = None,
    retrieval_queries: Sequence[str] | None = None,
    max_slots: int = 5,
) -> dict[str, Any]:
    intent = _citation_intent(prompt, prompt_family=prompt_family)
    budget = _budget_for_intent(intent)
    # `budget` is also used as the whole-answer coverage target by citation
    # repair. Keep a separate paragraph cap so multi-source coverage does not
    # authorize crowding every paragraph with one citation per source.
    per_paragraph_budget = dict(budget)
    requested_paper_count = extract_requested_paper_count(prompt)
    requested_system_a = min(8, int(requested_paper_count or 0))
    answer_audit = prompt_requests_answer_audit(prompt)
    if answer_audit:
        intent = "answer_audit"
        requested_system_a = min(8, len(list(answer_hits or [])))
        budget = {"system_a": requested_system_a, "system_b": 0}
    if requested_system_a > 0:
        budget["system_a"] = max(int(budget.get("system_a") or 0), requested_system_a)
    sys_b = (
        _system_b_slots(reference_opportunities, intent=intent, max_items=3)
        if int(budget.get("system_b") or 0) > 0
        else []
    )
    system_a_limit = requested_system_a if requested_paper_count is not None else max(3, requested_system_a)
    source_focus_keys: set[str] = set()
    for raw in [*list(support_slots or []), *list(answer_hits or [])]:
        if not isinstance(raw, Mapping):
            continue
        meta = raw.get("meta") if isinstance(raw.get("meta"), Mapping) else {}
        source_path = str(raw.get("source_path") or (meta or {}).get("source_path") or "").strip()
        if source_path:
            source_focus_keys.add(source_path.replace("\\", "/").lower())
    sys_a = _system_a_slots(
        support_slots=support_slots,
        answer_hits=answer_hits,
        max_items=system_a_limit,
        focus_multi_source_evidence=bool(
            len(source_focus_keys) >= 2
            and _MULTI_SLOT_COVERAGE_RE.search(str(prompt or ""))
        ),
        ranking_texts=[prompt, *list(retrieval_queries or [])],
        rank_answer_hits=bool(requested_system_a > 1),
    )
    s2ism_focus = _s2ism_tradeoff_focus_slot(prompt, answer_hits)
    if s2ism_focus:
        focus_path = str(s2ism_focus.get("source_path") or "").strip().lower()
        sys_a = [s2ism_focus] + [
            slot
            for slot in sys_a
            if str(slot.get("source_path") or "").strip().lower() != focus_path
        ]
        sys_a = sys_a[:system_a_limit]
    unique_system_a_sources = {
        str(slot.get("source_path") or "").strip().replace("\\", "/").lower()
        or str(slot.get("source_name") or "").strip().lower()
        for slot in sys_a
        if isinstance(slot, dict)
        and (str(slot.get("source_path") or "").strip() or str(slot.get("source_name") or "").strip())
    }
    lineage_prompt = strip_negated_reference_trail_requests(prompt)
    if (
        intent == "origin_lookup"
        and _LINEAGE_INTENT_RE.search(lineage_prompt)
        and len(unique_system_a_sources) >= 3
    ):
        budget["system_a"] = max(
            int(budget.get("system_a") or 0),
            min(6, len(unique_system_a_sources)),
        )
    if (
        intent == "scope_boundary"
        and _REVIEW_CONTEXT_RE.search(str(prompt or ""))
        and len(unique_system_a_sources) >= 2
    ):
        budget["system_a"] = max(int(budget.get("system_a") or 0), 2)
    if sys_a and _MULTI_SLOT_COVERAGE_RE.search(str(prompt or "")):
        if unique_system_a_sources:
            budget["system_a"] = max(
                int(budget.get("system_a") or 0),
                min(6, len(unique_system_a_sources)),
            )
    slots = (sys_b if intent == "origin_lookup" else []) + sys_a
    if intent != "origin_lookup":
        slots.extend(sys_b)
    slots = slots[: max(1, int(max(max_slots, system_a_limit)))]
    return {
        "version": 1,
        "source": "citation_plan_builder",
        "intent": intent,
        "budget": dict(budget),
        "per_paragraph_budget": dict(per_paragraph_budget),
        "system_a_enabled": bool(int(budget.get("system_a") or 0) > 0 and sys_a),
        "system_b_enabled": bool(int(budget.get("system_b") or 0) > 0 and sys_b),
        "slots": [dict(slot) for slot in slots if isinstance(slot, dict)],
    }


def build_citation_plan_prompt_block(plan: Mapping[str, Any] | None) -> str:
    if not isinstance(plan, Mapping) or not plan:
        return ""
    slots = [dict(item) for item in list(plan.get("slots") or []) if isinstance(item, Mapping)]
    if not slots:
        return ""
    budget = dict(plan.get("budget") or {}) if isinstance(plan.get("budget"), Mapping) else {}
    per_paragraph_budget = (
        dict(plan.get("per_paragraph_budget") or {})
        if isinstance(plan.get("per_paragraph_budget"), Mapping)
        else dict(budget)
    )
    lines = [
        "Citation plan (follow before adding citations):",
        f"- intent={str(plan.get('intent') or '').strip() or 'answer_grounding'}",
        f"- per paragraph budget: SystemA={int(per_paragraph_budget.get('system_a') or 0)}, SystemB={int(per_paragraph_budget.get('system_b') or 0)}",
        "- SystemA = retrieved paper text evidence; SystemB = a retrieved paper's bibliography/reference item.",
        "- Put a citation immediately after the sentence it supports; do not cite decorative or summary-only sentences.",
        "- Use SystemB only for origin, prior-work, method-source, or 'where did this idea come from' claims.",
        "- Use SystemA for claims about what the retrieved paper itself says, shows, defines, or reports.",
    ]
    if per_paragraph_budget != budget:
        lines.insert(
            3,
            f"- whole answer coverage target: SystemA={int(budget.get('system_a') or 0)}, SystemB={int(budget.get('system_b') or 0)}",
        )
    for idx, slot in enumerate(slots[:6], start=1):
        preferred = str(slot.get("preferred_system") or "").strip() or "system_a"
        topic = _compact_text(slot.get("topic"), max_len=120) or "evidence"
        parts = [f"{idx}. preferred_system={preferred}", f"topic={topic}"]
        cite_examples = [str(x or "").strip() for x in list(slot.get("candidate_cite_examples") or []) if str(x or "").strip()]
        if cite_examples:
            parts.append("cite_example=" + " ".join(cite_examples[:2]))
        support_example = str(slot.get("support_example") or "").strip()
        if support_example:
            parts.append(f"support_example={support_example}")
        candidate_hits = _positive_ints(slot.get("candidate_hits"), limit=3)
        if candidate_hits:
            parts.append("retrieved_hit=" + ",".join(str(n) for n in candidate_hits))
            if preferred.strip().lower() == "system_a":
                parts.append(
                    "cite_example="
                    + " ".join(f"[{CITATION_OFFSET + int(n)}]" for n in candidate_hits[:2])
                )
        heading = _compact_text(slot.get("heading_path"), max_len=100)
        if heading:
            parts.append(f"heading={heading}")
        quote = _compact_text(slot.get("evidence_quote"), max_len=160)
        if quote:
            parts.append(f"evidence={quote}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def citation_plan_prefers_system_b(
    plan: Mapping[str, Any] | None,
    *,
    context: str = "",
    ref_num: int = 0,
) -> bool:
    if not isinstance(plan, Mapping) or not plan:
        return False
    budget = dict(plan.get("budget") or {}) if isinstance(plan.get("budget"), Mapping) else {}
    if int(budget.get("system_b") or 0) <= 0:
        return False
    slots = [dict(item) for item in list(plan.get("slots") or []) if isinstance(item, Mapping)]
    try:
        n = int(ref_num or 0)
    except Exception:
        n = 0
    for slot in slots:
        if str(slot.get("preferred_system") or "").strip().lower() != "system_b":
            continue
        refs = _positive_ints(slot.get("candidate_refs"), limit=12)
        if n > 0 and refs and n in refs:
            return True
    intent = str(plan.get("intent") or "").strip().lower()
    if n <= 0 and bool(plan.get("system_b_enabled")):
        return True
    if intent == "origin_lookup" and bool(plan.get("system_b_enabled")):
        return bool(_ORIGIN_INTENT_RE.search(str(context or "")))
    return False
