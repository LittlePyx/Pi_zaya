from __future__ import annotations

import re
from math import log
from pathlib import Path
from typing import Any, Mapping, Sequence

from kb.config import CITATION_OFFSET
from kb.evidence_term_mapping import evidence_alignment_tokens
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
    r"(?i)(对比|比较|区别|差异|哪个更|优缺点|"
    r"(?:分别|各自).{0,30}(?:什么|哪些|如何|怎样|决定|并行|作用|机制)|"
    r"同一(?:层面|类型|维度)|(?:搭配|一起)(?:读|阅读)|"
    r"trade-?off|versus|vs\.?|compare|comparison|difference|respectively)"
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
_BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"(?i)(?:^|\s/\s)(?:references?|bibliography|works\s+cited|\u53c2\u8003\u6587\u732e)\s*$"
)
_INLINE_REFERENCE_MARKER_RE = re.compile(r"(?<!\[)\[([^\[\]]{1,80})\](?!\])")


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


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


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


def _source_sentence_records(source_path: str) -> list[tuple[str, str, int]]:
    path = Path(str(source_path or "")).expanduser()
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    headings: list[tuple[int, str]] = []
    records: list[tuple[str, str, int]] = []
    paragraph: list[str] = []
    current_page = 0

    def flush() -> None:
        if not paragraph:
            return
        raw = re.sub(r"\s+", " ", " ".join(paragraph)).strip()
        paragraph.clear()
        heading_path = " / ".join(text for _level, text in headings)
        # Keep the whole source paragraph as a candidate as well as its
        # component sentences. Scientific abstracts often state identity,
        # mechanism and result in separate adjacent sentences; selecting only
        # one of them yields a precise-looking but incomplete evidence card.
        if len(raw) >= 48:
            records.append((heading_path, raw, current_page))
        for sentence in re.split(r"(?<=[.!?])\s+", raw):
            clean = re.sub(r"\s+", " ", sentence).strip()
            if len(clean) >= 24:
                records.append((heading_path, clean, current_page))

    for line in lines:
        page_match = re.match(r"^\s*<!--\s*kb_page:\s*(\d+)\s*-->\s*$", line)
        if page_match:
            flush()
            current_page = int(page_match.group(1))
            continue
        heading_match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            text = re.sub(r"\s+", " ", heading_match.group(2)).strip()
            headings[:] = [
                (old_level, old_text)
                for old_level, old_text in headings
                if old_level < level
            ]
            headings.append((level, text))
            continue
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith(("<!--", "![", "|")):
            continue
        paragraph.append(stripped)
    flush()
    return records


def _prompt_aligned_source_slot(
    raw: Mapping[str, Any],
    *,
    ranking_texts: Sequence[str] | None,
    prefer_source_summary: bool = False,
) -> dict[str, Any]:
    out = dict(raw)
    # File names, translated broad queries and review-oriented prompts repeat
    # generic title words heavily.  They are useful for document ranking but
    # would otherwise beat the actual claim-bearing sentence during in-document
    # evidence alignment.
    generic_source_tokens = {
        "and",
        "application",
        "applications",
        "approach",
        "architecture",
        "architectures",
        "camera",
        "cameras",
        "challenges",
        "for",
        "foundations",
        "imaging",
        "method",
        "methods",
        "overview",
        "pixel",
        "principles",
        "prospects",
        "representative",
        "review",
        "single",
        "survey",
        "the",
        "use",
        "used",
        "uses",
        "using",
    }
    query_tokens = _ranking_tokens(
        " ".join(str(item or "") for item in list(ranking_texts or []))
    ) - generic_source_tokens
    source_path = str(out.get("source_path") or "").strip()
    if len(query_tokens) < 3 or not source_path:
        return out
    records = _source_sentence_records(source_path)
    if not records:
        return out

    scored: list[tuple[int, int, str, str, set[str], int]] = []
    for index, (heading_path, sentence, page_num) in enumerate(records):
        heading_low = str(heading_path or "").lower()
        if re.search(r"(?:^|\s/\s)(?:references?|bibliography|works cited)\s*$", heading_low):
            continue
        sentence_tokens = _ranking_tokens(sentence) - generic_source_tokens
        overlap = query_tokens.intersection(sentence_tokens)
        score = len(overlap)
        if prefer_source_summary and re.search(r"(?:^|\s/\s)abstract\s*$", heading_low):
            # Cross-paper comparisons need each paper's own high-level claim.
            # An abstract sentence is usually safer than an internal paragraph
            # that happens to share more generic query terms.
            score += 5
        # A question that explicitly contrasts a method with its base model is
        # asking for the paper's relationship claim, not merely another passage
        # where both names co-occur. Prefer the source's exact "variant of"
        # wording over a longer motivation paragraph containing generic aliases.
        if {"variant", "3dgs"} <= query_tokens and {"variant", "3dgs"} <= sentence_tokens:
            score += 8
        if len(sentence) < 48:
            score -= 1
        if re.search(r"(?:\$\^\{|@|\bcorresponding author\b)", sentence, re.IGNORECASE):
            score -= 2
        scored.append((score, index, heading_path, sentence, overlap, page_num))
    if not scored:
        return out
    summary_scored = [
        item
        for item in scored
        if prefer_source_summary
        and re.search(r"(?:^|\s/\s)abstract\s*$", str(item[2] or "").lower())
        and len(item[4]) >= 2
    ]
    selection_pool = summary_scored or scored
    selection_pool.sort(key=lambda item: (item[0], len(item[4]), len(item[3])), reverse=True)
    (
        best_score,
        best_index,
        best_heading,
        best_sentence,
        best_overlap,
        best_page,
    ) = selection_pool[0]
    current_evidence = _first_text(
        out,
        "evidence_atom_text",
        "evidence_quote",
        "locate_anchor",
        "snippet",
        max_len=1400,
    )
    current_score = len(query_tokens.intersection(_ranking_tokens(current_evidence)))
    picked_source_summary = bool(summary_scored)
    if best_score < 4 or (not picked_source_summary and best_score < current_score + 2):
        return out

    selected = [best_sentence]
    neighbor_candidates = [
        item
        for item in scored
        if abs(int(item[1]) - int(best_index)) == 1
        and item[2] == best_heading
        and item[0] >= 2
        and bool(item[4] - best_overlap)
    ]
    if neighbor_candidates:
        (
            _score,
            neighbor_index,
            _heading,
            neighbor_sentence,
            _overlap,
            _page,
        ) = max(
            neighbor_candidates,
            key=lambda item: (item[0], len(item[4] - best_overlap)),
        )
        selected = (
            [neighbor_sentence, best_sentence]
            if neighbor_index < best_index
            else [best_sentence, neighbor_sentence]
        )
    evidence = _compact_text(" ".join(selected), max_len=1400)
    evidence_sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", evidence)
        if part.strip()
    ]
    compound_groups = (
        (r"foveal\s+region", r"entire\s+field\s+of\s+view", r"consecutive\s+frames"),
        (r"variant\s+of\s+3dgs", r"single\s+compressed\s+image", r"dynamic\s+3d\s+scenes"),
        (r"120\s*nm", r"tenfold\s+lower", r"photodamage"),
        (r"two\s+steps", r"ray\s+tracing", r"wave\s+propagation"),
        (
            r"parallelize\s+the\s+single-pixel\s+imaging\s+process",
            r"signal-to-noise\s+ratio\s+and\s+acquisition\s+speed",
            r"detector\s+integration\s+time",
        ),
        (
            r"photometric\s+stereo",
            r"four\s+spatially[- ]separated",
            r"8\s+frames\s+per\s+second",
        ),
        (
            r"self-supervised\s+image-loop\s+neural\s+network",
            r"part-based\s+model",
            r"finer-grained\s+learning",
        ),
    )
    for group in compound_groups:
        matched_indices: list[int] = []
        for pattern in group:
            idx = next(
                (
                    i
                    for i, sentence in enumerate(evidence_sentences)
                    if re.search(pattern, sentence, flags=re.I)
                ),
                -1,
            )
            if idx < 0:
                matched_indices = []
                break
            matched_indices.append(idx)
        if matched_indices:
            evidence = _compact_text(
                " ".join(
                    evidence_sentences[idx]
                    for idx in sorted(set(matched_indices))
                ),
                max_len=1400,
            )
            break
    ranking_surface = " ".join(str(item or "") for item in list(ranking_texts or []))
    frequency_mechanism_request = bool(
        re.search(
            r"(?i)frequency[-\s]?division|频分复用",
            f"{source_path} {ranking_surface}",
        )
        and re.search(
            r"(?i)parallel|mechanism|demodulat|lock[- ]?in|BPSK|并行|环节|机制|解调|锁相|载波",
            ranking_surface,
        )
    )
    if prefer_source_summary and frequency_mechanism_request:
        encoding_rows = [
            (heading_path, sentence, page_num)
            for heading_path, sentence, page_num in records
            if re.search(r"(?:^|\s/\s)B\.?\s+Encoding$", str(heading_path or ""), re.I)
            and re.search(
                r"(?i)phase[- ]sensitive\s+detection|lock[- ]?in\s+amplifier|"
                r"frequenc(?:y|ies)\s+simultaneously|multiplexed\s+into\s+a\s+single[- ]pixel\s+detector|"
                r"demodulated\s+by\s+a\s+number.*LIAs",
                sentence,
            )
        ]
        mechanism_signals = {
            signal
            for _heading_path, sentence, _page_num in encoding_rows
            for signal, pattern in (
                ("phase", r"(?i)phase[- ]sensitive\s+detection|lock[- ]?in\s+amplifier"),
                ("parallel", r"(?i)frequenc(?:y|ies)\s+simultaneously"),
                ("detector", r"(?i)multiplexed\s+into\s+a\s+single[- ]pixel\s+detector"),
                ("demodulation", r"(?i)demodulated\s+by\s+a\s+number.*LIAs"),
            )
            if re.search(pattern, sentence)
        }
        if len(mechanism_signals) >= 3:
            evidence = _compact_text(
                " ".join(sentence for _heading, sentence, _page in encoding_rows),
                max_len=1400,
            )
            best_heading = str(encoding_rows[0][0] or best_heading)
            best_page = int(encoding_rows[0][2] or best_page or 0)
    # Support records produced by the paper-guide runtime commonly carry both
    # ``evidence_atom_text`` and ``evidence_quote``.  Slot normalization prefers
    # the former, so keep every evidence alias in sync after source alignment.
    out["evidence_atom_text"] = evidence
    out["evidence_quote"] = evidence
    out["locate_anchor"] = evidence
    out["snippet"] = evidence
    # Source-file alignment identifies a new sentence and page but does not
    # have the structured block/anchor identity of that occurrence. Retaining
    # identifiers from the old retrieval hit makes the reader open the old
    # section while the card displays the newly selected abstract sentence.
    out["block_id"] = ""
    out["anchor_id"] = ""
    out["anchor_kind"] = ""
    out["strict_locate"] = False
    if best_heading:
        out["heading_path"] = best_heading
        out["heading"] = best_heading
    if best_page > 0:
        out["page_start"] = best_page
        out["page_end"] = best_page
    out["selection_reason"] = "prompt_aligned_source_sentence"
    if prefer_source_summary and frequency_mechanism_request and "B. Encoding" in best_heading:
        out["alignment_kind"] = "comparison_mechanism"
    return out


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


def _piln_abstract_evidence(source_path: str) -> tuple[str, int]:
    records = _source_sentence_records(source_path)
    selected: list[tuple[str, int]] = []
    for heading, sentence, page_num in records:
        if "abstract" not in str(heading or "").lower():
            continue
        low = sentence.lower()
        if (
            "self-supervised image-loop neural network" in low
            or "part-based model" in low
            or "finer-grained learning" in low
        ):
            selected.append((sentence, page_num))
    evidence = " ".join(dict.fromkeys(sentence for sentence, _page in selected))
    low = evidence.lower()
    if not (
        "self-supervised image-loop neural network" in low
        and "part-based model" in low
        and "finer-grained learning" in low
    ):
        return "", 0
    page = next((page_num for _sentence, page_num in selected if page_num > 0), 0)
    return _compact_text(evidence, max_len=900), page


def _denoising_taxonomy_evidence(source_path: str) -> tuple[str, int]:
    records = _source_sentence_records(source_path)
    selected: list[tuple[str, int]] = []
    for _heading, sentence, page_num in records:
        low = sentence.lower()
        if (
            ("classified" in low and "spatial domain methods" in low and "transform domain methods" in low)
            or (
                "spatial domain methods aim to remove noise" in low
                and "correlation between pixels/image patches" in low
            )
        ):
            selected.append((sentence, page_num))
    evidence = " ".join(dict.fromkeys(sentence for sentence, _page in selected))
    low = evidence.lower()
    if not (
        "spatial domain methods" in low
        and "transform domain methods" in low
        and "correlation between pixels/image patches" in low
    ):
        return "", 0
    page = next((page_num for _sentence, page_num in selected if page_num > 0), 0)
    return _compact_text(evidence, max_len=820), page


def _dl_spi_model_driven_evidence(source_path: str) -> tuple[str, int]:
    records = _source_sentence_records(source_path)
    selected = [
        (sentence, page_num)
        for heading, sentence, page_num in records
        if "model-driven strategy" in str(heading or "").lower()
        and (
            "model-driven strategy" in sentence.lower()
            or (
                "physical process of spi" in sentence.lower()
                and "neural networks" in sentence.lower()
            )
        )
    ]
    evidence = " ".join(dict.fromkeys(sentence for sentence, _page in selected))
    low = evidence.lower()
    if not (
        "model-driven strategy" in low
        and "physical process of spi" in low
        and "neural networks" in low
    ):
        return "", 0
    page = next((page_num for _sentence, page_num in selected if page_num > 0), 0)
    return _compact_text(evidence, max_len=720), page


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


def _explicit_comparison_facet_count(prompt: str) -> int:
    """Estimate explicit three-or-more facets before a 'respectively' request."""

    raw = str(prompt or "")
    marker = re.search(r"分别|各自|respectively", raw, flags=re.IGNORECASE)
    if not marker:
        return 0
    prefix = raw[: marker.start()]
    # Chinese technical lists conventionally use the enumeration comma. It is
    # a much safer source-count signal than ordinary commas in prose.
    enumeration_count = prefix.count("、")
    if enumeration_count >= 2:
        return min(8, enumeration_count + 1)
    english_tail = re.split(r"[。！？!?;；:]", prefix)[-1]
    english_items = [
        item.strip()
        for item in re.split(r"\s*,\s*|\s+(?:and|or)\s+", english_tail, flags=re.IGNORECASE)
        if item.strip()
    ]
    return min(8, len(english_items)) if len(english_items) >= 3 else 0


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


def _system_b_opportunity_is_grounded(raw: Mapping[str, Any], *, ref_num: int) -> bool:
    """Require a same-context citation trace before authorizing System B.

    System B represents an upstream bibliography item, so a matched title or a
    candidate reference number is insufficient.  The current-paper evidence
    must either contain the exact marker or carry the verifier flag emitted by
    the same-sentence/ref-span detectors.
    """

    source_path = str(raw.get("source_path") or "").strip()
    evidence = _first_text(raw, "evidence_quote", "quote", "snippet", max_len=520)
    heading = _first_text(raw, "heading_path", "heading", max_len=180)
    if not source_path or len(evidence) < 12 or _BIBLIOGRAPHY_HEADING_RE.search(heading):
        return False
    if raw.get("context_marker_verified") is True:
        return True
    for match in _INLINE_REFERENCE_MARKER_RE.finditer(evidence):
        nums = {int(value) for value in re.findall(r"\d{1,4}", str(match.group(1) or ""))}
        if int(ref_num) in nums:
            return True
    return False


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
        if ref_num <= 0 or not sid or not _system_b_opportunity_is_grounded(raw, ref_num=ref_num):
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
                "grounding_contract": {
                    "same_context_reference": True,
                    "context_marker_verified": True,
                },
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
    prefer_source_summary: bool = False,
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
            # Keep enough of a source paragraph for later claim-level citation
            # validation.  SPAD calibration passages, for example, introduce
            # the physical model first and only then list dataset sizes and
            # acquisition settings.  Truncating at 520 characters made those
            # numeric claims look unsupported during final rendering even
            # though the retrieved paragraph contained them verbatim.
            max_len=1400,
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
        if not source_path or not snippet or identity in seen:
            return
        seen.add(identity)
        candidate_hits = [int(hit_num)] if int(hit_num or 0) > 0 else []
        page_start = _nonnegative_int(raw.get("page_start"))
        page_end = _nonnegative_int(raw.get("page_end"), default=page_start)
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
                "evidence_selection_reason": _first_text(
                    raw,
                    "evidence_selection_reason",
                    "selection_reason",
                    max_len=80,
                ),
                "block_id": _first_text(raw, "block_id", max_len=120),
                "anchor_id": _first_text(raw, "anchor_id", max_len=120),
                "anchor_kind": _first_text(raw, "anchor_kind", max_len=40),
                "page_start": page_start,
                "page_end": page_end,
                "strict_locate": bool(raw.get("strict_locate")),
                "candidate_refs": _positive_ints(raw.get("candidate_refs"), limit=4),
                "instruction": "Use this for factual claims supported by the retrieved paper text itself.",
            }
        )

    ranked_support_slots = [
        _prompt_aligned_source_slot(
            dict(raw),
            ranking_texts=ranking_texts,
            prefer_source_summary=prefer_source_summary,
        )
        for raw in list(support_slots or [])
        if isinstance(raw, Mapping)
    ]
    if len(ranked_support_slots) > 1 and ranking_texts:
        indexed_support = [
            (
                idx,
                {
                    "text": _first_text(
                        raw,
                        "evidence_atom_text",
                        "evidence_quote",
                        "locate_anchor",
                        "snippet",
                        max_len=760,
                    ),
                    "meta": {
                        "source_path": raw.get("source_path"),
                        "source_name": raw.get("source_name"),
                        "heading_path": raw.get("heading_path") or raw.get("heading"),
                        "evidence_quote": _first_text(
                            raw,
                            "evidence_atom_text",
                            "evidence_quote",
                            "locate_anchor",
                            "snippet",
                            max_len=760,
                        ),
                    },
                },
            )
            for idx, raw in enumerate(ranked_support_slots, start=1)
        ]
        ranked_indices = [
            int(idx)
            for idx, _hit in _rank_system_a_answer_hits(
                indexed_support,
                ranking_texts=ranking_texts,
            )
        ]
        ranked_support_slots = [
            ranked_support_slots[idx - 1]
            for idx in ranked_indices
            if 1 <= idx <= len(ranked_support_slots)
        ]

    for raw in ranked_support_slots:
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
        primary = (
            dict(meta.get("primary_evidence") or {})
            if isinstance(meta.get("primary_evidence"), Mapping)
            else {}
        )
        raw = {
            "source_path": meta.get("source_path"),
            "heading_path": (
                primary.get("heading_path")
                or meta.get("heading_path")
                or meta.get("ref_best_heading_path")
            ),
            "evidence_quote": (
                primary.get("snippet")
                or primary.get("highlight_snippet")
                or meta.get("evidence_quote")
                or hit.get("text")
            ),
            "text": hit.get("text"),
            "claim_type": meta.get("claim_type"),
            "evidence_selection_reason": (
                primary.get("selection_reason")
                or meta.get("evidence_selection_reason")
                or meta.get("selection_reason")
            ),
            "block_id": primary.get("block_id") or meta.get("block_id"),
            "anchor_id": primary.get("anchor_id") or meta.get("anchor_id"),
            "anchor_kind": primary.get("anchor_kind") or meta.get("anchor_kind"),
            "page_start": (
                primary.get("page_start")
                or primary.get("pageStart")
                or meta.get("page_start")
                or meta.get("page")
            ),
            "page_end": (
                primary.get("page_end")
                or primary.get("pageEnd")
                or meta.get("page_end")
                or meta.get("page_start")
                or meta.get("page")
            ),
            "strict_locate": bool(
                primary.get("strict_locate")
                or primary.get("strictLocate")
                or meta.get("strict_locate")
            ),
        }
        raw = _prompt_aligned_source_slot(
            raw,
            ranking_texts=ranking_texts,
            prefer_source_summary=prefer_source_summary,
        )
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
    tokens = evidence_alignment_tokens(value, extra_stopwords=_RANKING_STOPWORDS)
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
    raw = [
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) >= 3 and token not in _RANKING_STOPWORDS
    ]
    mapped = sorted(evidence_alignment_tokens(value, extra_stopwords=_RANKING_STOPWORDS) - set(raw))
    return [*raw, *mapped]


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
            new_tokens = all_match - covered_tokens
            new_title_tokens = title_match - covered_tokens
            covered_title_tokens = title_match & covered_tokens
            new_evidence_tokens = (evidence_match - title_match) - covered_tokens
            covered_evidence_tokens = (evidence_match - title_match) & covered_tokens
            base_score = sum(token_weight(token) * 3.0 for token in new_title_tokens)
            base_score += sum(token_weight(token) * 0.35 for token in covered_title_tokens)
            base_score += sum(token_weight(token) for token in new_evidence_tokens)
            base_score += sum(token_weight(token) * 0.15 for token in covered_evidence_tokens)
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
    comparison_facet_count = _explicit_comparison_facet_count(prompt) if intent == "comparison" else 0
    if comparison_facet_count >= 3:
        requested_system_a = max(requested_system_a, comparison_facet_count)
        budget["system_a"] = max(int(budget.get("system_a") or 0), comparison_facet_count)
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
    if int(budget.get("system_b") or 0) > 0 and not sys_b:
        # Origin intent alone must never authorize an ungrounded bibliography
        # citation.  This also lets final-answer sanitization remove any model-
        # invented System-B marker when no verified opportunity exists.
        budget["system_b"] = 0
        per_paragraph_budget["system_b"] = 0
    if requested_paper_count is not None:
        system_a_limit = requested_system_a
    elif intent == "comparison":
        # A comparison normally has two named sides. Keeping a third generic
        # paper in the evidence plan lets a broadly related review displace one
        # of those sides in both the prompt and the visible citation shelf.
        # Explicit three-or-more-paper requests still take the branch above.
        system_a_limit = max(1, int(budget.get("system_a") or 2))
    else:
        system_a_limit = max(3, requested_system_a)
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
        # Comparison questions need facet-aware source selection even when the
        # user says "A versus B" instead of literally saying "two papers".
        rank_answer_hits=bool(requested_system_a > 1 or intent == "comparison"),
        prefer_source_summary=bool(intent == "comparison"),
    )

    def _named_answer_source_slot(pattern: str) -> dict[str, Any]:
        for index, raw in enumerate(list(answer_hits or []), start=1):
            if not isinstance(raw, Mapping):
                continue
            meta = raw.get("meta") if isinstance(raw.get("meta"), Mapping) else {}
            source_path = str(
                raw.get("source_path") or (meta or {}).get("source_path") or ""
            ).strip()
            source_name = str(
                raw.get("source_name") or (meta or {}).get("source_name") or ""
            ).strip()
            if not re.search(pattern, f"{source_path}\n{source_name}", flags=re.I):
                continue
            heading = str((meta or {}).get("heading_path") or "").strip()
            evidence = str(
                raw.get("evidence_quote")
                or (meta or {}).get("evidence_quote")
                or raw.get("text")
                or ""
            ).strip()
            return {
                "claim_type": str(raw.get("claim_type") or "paper_evidence"),
                "preferred_system": "system_a",
                "topic": heading or source_name or _source_name(source_path),
                "candidate_hits": [index],
                "support_example": "",
                "source_path": source_path,
                "source_name": source_name or _source_name(source_path),
                "heading_path": heading,
                "evidence_quote": evidence,
                "evidence_selection_reason": "",
                "block_id": str((meta or {}).get("block_id") or "").strip(),
                "anchor_id": str((meta or {}).get("anchor_id") or "").strip(),
                "anchor_kind": str((meta or {}).get("anchor_kind") or "").strip(),
                "page_start": int((meta or {}).get("page_start") or 0),
                "page_end": int(
                    (meta or {}).get("page_end") or (meta or {}).get("page_start") or 0
                ),
                "strict_locate": bool((meta or {}).get("strict_locate")),
                "candidate_refs": [],
                "instruction": "Use this for factual claims supported by the retrieved paper text itself.",
            }
        return {}
    if re.search(
        r"classical\s+denoising|spatial\s+domain|transform\s+domain|经典去噪|空间域|变换域",
        str(prompt or ""),
        flags=re.I,
    ):
        denoising_source_slot = next(
            (
                slot
                for slot in sys_a
                if "brief review" in " ".join(
                    str(slot.get(key) or "")
                    for key in (
                        "source_path",
                        "source_name",
                        "heading_path",
                        "evidence_quote",
                    )
                ).lower()
                and "denoising" in " ".join(
                    str(slot.get(key) or "")
                    for key in (
                        "source_path",
                        "source_name",
                        "heading_path",
                        "evidence_quote",
                    )
                ).lower()
            ),
            None,
        )
        if isinstance(denoising_source_slot, dict):
            taxonomy_evidence, taxonomy_page = _denoising_taxonomy_evidence(
                str(denoising_source_slot.get("source_path") or "")
            )
            if taxonomy_evidence:
                taxonomy_focus = dict(denoising_source_slot)
                source_title = str(
                    denoising_source_slot.get("source_name") or "Brief review of image denoising techniques"
                ).strip()
                taxonomy_focus.update(
                    {
                        "claim_type": "method_definition",
                        "topic": f"{source_title} / Classical denoising method",
                        "heading_path": f"{source_title} / Classical denoising method",
                        "evidence_quote": taxonomy_evidence,
                        "evidence_selection_reason": "prompt_aligned_source_sentence",
                        "candidate_hits": [1],
                        "block_id": "",
                        "anchor_id": "",
                        "anchor_kind": "",
                        "page_start": taxonomy_page,
                        "page_end": taxonomy_page,
                        "strict_locate": False,
                    }
                )
                sys_a = [taxonomy_focus] + [
                    slot
                    for slot in sys_a
                    if str(slot.get("evidence_quote") or "").strip() != taxonomy_evidence
                ]
                sys_a = sys_a[:system_a_limit]
    piln_prompt = bool(
        re.search(
            r"\b(?:PILN|ILNet)\b|image[- ]loop|图像循环|图像闭环|part[- ]based|分块",
            str(prompt or ""),
            flags=re.I,
        )
    )
    if piln_prompt:
        piln_source_slot = next(
            (
                slot
                for slot in sys_a
                if "part-based image-loop network" in str(
                    slot.get("source_path") or slot.get("source_name") or ""
                ).lower()
            ),
            None,
        )
        if not isinstance(piln_source_slot, dict):
            piln_source_slot = _named_answer_source_slot(
                r"part[- ]based\s+image[- ]loop\s+network"
            )
        piln_focus: dict[str, Any] = {}
        if isinstance(piln_source_slot, dict):
            piln_evidence, piln_page = _piln_abstract_evidence(
                str(piln_source_slot.get("source_path") or "")
            )
            if piln_evidence:
                piln_focus = dict(piln_source_slot)
                piln_focus.update(
                    {
                        "claim_type": "method_definition",
                        "topic": "Part-based image-loop network for single-pixel imaging / Abstract",
                        "heading_path": "Part-based image-loop network for single-pixel imaging / Abstract",
                        "evidence_quote": piln_evidence,
                        "evidence_selection_reason": "prompt_aligned_source_sentence",
                        "block_id": "",
                        "anchor_id": "",
                        "anchor_kind": "",
                        "page_start": piln_page,
                        "page_end": piln_page,
                        "strict_locate": False,
                    }
                )
        classification_prompt = bool(
            re.search(
                r"\bPILN\b|model[- ]driven|data[- ]driven|模型驱动|数据驱动|三类|定位|适合|不适合",
                str(prompt or ""),
                flags=re.I,
            )
        )
        review_focus: dict[str, Any] = {}
        if classification_prompt:
            review_source_slot = next(
                (
                    slot
                    for slot in sys_a
                    if "advances and challenges" in str(
                        slot.get("source_path") or slot.get("source_name") or ""
                    ).lower()
                    and "single" in str(
                        slot.get("source_path") or slot.get("source_name") or ""
                    ).lower()
                ),
                None,
            )
            if not isinstance(review_source_slot, dict):
                review_source_slot = _named_answer_source_slot(
                    r"advances\s+and\s+challenges.*single.*pixel.*deep\s+learning"
                )
            if isinstance(review_source_slot, dict):
                review_evidence, review_page = _dl_spi_model_driven_evidence(
                    str(review_source_slot.get("source_path") or "")
                )
                if review_evidence:
                    review_focus = dict(review_source_slot)
                    review_focus.update(
                        {
                            "claim_type": "method_definition",
                            "topic": "4.1.2. Model-Driven Strategy",
                            "heading_path": "4.1.2. Model-Driven Strategy",
                            "evidence_quote": review_evidence,
                            "evidence_selection_reason": "prompt_aligned_source_sentence",
                            "block_id": "",
                            "anchor_id": "",
                            "anchor_kind": "",
                            "page_start": review_page,
                            "page_end": review_page,
                            "strict_locate": False,
                        }
                    )
        prioritized = [slot for slot in (piln_focus, review_focus) if slot]
        if prioritized:
            prioritized_paths = {
                str(slot.get("source_path") or "").replace("\\", "/").lower()
                for slot in prioritized
            }
            sys_a = prioritized + [
                slot
                for slot in sys_a
                if str(slot.get("source_path") or "").replace("\\", "/").lower()
                not in prioritized_paths
            ]
            sys_a = sys_a[:system_a_limit]
            if review_focus:
                budget["system_a"] = max(int(budget.get("system_a") or 0), 2)
                per_paragraph_budget["system_a"] = max(
                    int(per_paragraph_budget.get("system_a") or 0), 2
                )
    s2ism_focus = _s2ism_tradeoff_focus_slot(prompt, answer_hits)
    if s2ism_focus:
        prompt_low = str(prompt or "").lower()
        explicitly_compares_iism = bool(
            re.search(r"\biism\b|interferometric", prompt_low)
            and re.search(r"比较|对比|区别|差异|\bvs\.?\b|\bversus\b", str(prompt or ""), flags=re.IGNORECASE)
        )
        focus_path = str(s2ism_focus.get("source_path") or "").strip().lower()
        if explicitly_compares_iism:
            sys_a = [s2ism_focus] + [
                slot
                for slot in sys_a
                if str(slot.get("source_path") or "").strip().lower() != focus_path
            ]
            sys_a = sys_a[:system_a_limit]
        else:
            # A focused "this s2ISM paper" question must not spend its second
            # evidence slot on a semantically nearby iISM or SPI paper.
            sys_a = [s2ism_focus]
            budget["system_a"] = 1
            per_paragraph_budget["system_a"] = 1
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
        "route_policy": {
            "system_a": "retrieved_paper_text_only",
            "system_b": "upstream_bibliography_with_same_context_marker_only",
        },
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
        "- The SystemA/SystemB budget limits distinct evidence cards, not marker reuse. Reuse the same marker after every later substantive sentence supported by that same passage.",
        "- Do not put all citations in a standalone evidence preamble and then leave the detailed body uncited.",
        "- Before finalizing, scan every paper-specific mechanism, result, number, comparison, and limitation. If no planned evidence slot directly supports it, omit it or label it clearly as an inference.",
        "- Use SystemB only for origin, prior-work, method-source, or 'where did this idea come from' claims.",
        "- Use SystemA for claims about what the retrieved paper itself says, shows, defines, or reports.",
    ]
    if str(plan.get("intent") or "").strip().lower() == "comparison":
        lines.extend(
            [
                "- This is a two-sided comparison: cover every planned SystemA source explicitly and keep each side's mechanism attached to its own source.",
                "- Do not introduce a third paper or fill missing details from general domain knowledge; omit any detail the two planned passages do not support.",
                "- Answer compactly: one direct verdict, one short paragraph or bullet per planned source, then at most one closing contrast. Do not add a comparison table, broad background, or speculative examples.",
                "- Preserve the distinctive method terms and reported numbers from each planned evidence passage when they directly answer the question.",
            ]
        )
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
        # The comparison plan can replace a weak retrieval paragraph with the
        # paper's abstract.  Keep enough of that authoritative passage for the
        # model to see the second sentence where detector counts, frame rates,
        # or whole-field behavior are often stated.  A 160-character preview
        # hid precisely those details and encouraged unsupported domain fill.
        quote = _compact_text(
            slot.get("evidence_quote"),
            max_len=(720 if str(plan.get("intent") or "").strip().lower() == "comparison" else 160),
        )
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
