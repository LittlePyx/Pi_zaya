from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from kb.citation_evidence_pack import build_system_a_evidence_pack, build_system_b_evidence_pack
from kb.evidence_text import clean_display_text, finish_evidence_text

def _clean_text(value: Any, *, max_len: int = 520) -> str:
    return clean_display_text(value, max_len=max_len)


def _loose_tokens(value: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", str(value or ""))]


def _source_title_candidate(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = Path(text).name or text
    name = re.sub(r"\.(?:pdf|md)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.en$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^[A-Za-z]{2,12}-\d{4}-", "", name)
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _strip_token_prefix(text: str, candidate: str) -> str:
    tokens = _loose_tokens(candidate)
    if len(tokens) < 4:
        return text
    matches = list(re.finditer(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text))
    if len(matches) < len(tokens):
        return text
    limit = min(len(tokens), len(matches))
    matched = 0
    for idx in range(limit):
        if matches[idx].group(0).lower() != tokens[idx]:
            break
        matched += 1
    required = min(8, len(tokens))
    if matched < required:
        return text
    return text[matches[matched - 1].end() :].lstrip(" ,.;:，。；：-")


def _looks_author_metadata_prefix(prefix: str) -> bool:
    text = str(prefix or "").strip()
    if len(text) < 16:
        return False
    comma_count = text.count(",") + text.count("，")
    name_pairs = len(re.findall(r"\b[A-Z][a-zA-Z'`-]+\s+[A-Z][a-zA-Z'`-]+\b", text))
    tokens = _loose_tokens(text)
    if comma_count >= 2 or name_pairs >= 2:
        return True
    return len(tokens) >= 8 and bool(re.search(r"[*\\]", text))


_CONTENT_SENTENCE_START_RE = re.compile(
    r"\b(?:"
    r"single[-\s]?pixel imaging|"
    r"deep learning|"
    r"snapshot compressive|"
    r"compressive imaging|"
    r"neural radiance|"
    r"this paper|this work|this study|"
    r"in this (?:paper|work|study)|"
    r"we\s+|however,?|recent(?:ly)?|the proposed|our\s+"
    r")\b",
    re.IGNORECASE,
)


def _strip_system_a_metadata_prefix(text: str, *, source: str = "", title: str = "") -> str:
    out = str(text or "").strip()
    if not out:
        return ""

    for raw_candidate in (source, title):
        candidate = _source_title_candidate(raw_candidate)
        if len(candidate) >= 18:
            stripped = _strip_token_prefix(out, candidate)
            if stripped != out:
                out = stripped
                break

    for match in _CONTENT_SENTENCE_START_RE.finditer(out):
        idx = match.start()
        if idx <= 0:
            break
        if idx > 320:
            break
        prefix = out[:idx]
        if _looks_author_metadata_prefix(prefix):
            out = out[idx:].lstrip(" ,.;:，。；：-")
        break

    return re.sub(r"\s+", " ", out).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\.])\s+", str(text or "").strip())
    return [part.strip() for part in parts if part.strip()]


_FRAGMENT_LEAD_OK_RE = re.compile(
    r"^(?:a|an|the|this|these|those|we|our|in|on|for|by|with|when|where|while|because|however|therefore|thus|as|if|to)\b",
    re.IGNORECASE,
)


def _looks_fragmentary_sentence(sentence: str) -> bool:
    text = str(sentence or "").strip()
    if not text:
        return True
    if re.match(r"^[a-z]{2,}\b", text) and not _FRAGMENT_LEAD_OK_RE.match(text):
        return True
    if re.match(r"^(?:and|or|of|that|which|from|into|onto|within|without|using|used|measured|allowing)\b", text, re.IGNORECASE):
        return True
    if len(text) > 80 and re.search(r"\b(?:and|or|of|to|with|by|from|into|onto)$", text, re.IGNORECASE):
        return True
    if len(text) > 120 and not re.search(r"[。！？!?\.]$", text):
        return True
    return False


def _looks_caption_heading(sentence: str) -> bool:
    text = str(sentence or "").strip()
    if re.match(r"^(?:fig(?:ure)?|table)\s*\d+[.:]?\s*$", text, re.IGNORECASE):
        return True
    tokens = _loose_tokens(text)
    if re.match(r"^[a-z]\s*,\s*", text, re.IGNORECASE):
        return True
    return len(tokens) <= 5 and bool(re.search(r"\b(?:configuration|configurations|overview|pipeline|results?|figure)\b", text, re.IGNORECASE))


def _usable_evidence_sentence(sentence: str) -> bool:
    text = str(sentence or "").strip()
    if _looks_fragmentary_sentence(text) or _looks_caption_heading(text):
        return False
    tokens = _loose_tokens(text)
    return len(tokens) >= 5


def _sentence_quality(sentence: str, *, claim: str = "", heading: str = "") -> float:
    text = str(sentence or "").strip()
    if not text:
        return -10.0
    tokens = _loose_tokens(text)
    score = 0.0
    if _looks_fragmentary_sentence(text):
        score -= 5.0
    if _looks_caption_heading(text):
        score -= 2.0
    if 8 <= len(tokens) <= 90:
        score += 2.0
    elif len(tokens) < 5:
        score -= 2.0
    if _looks_author_metadata_prefix(text[:180]):
        score -= 3.0
    if re.search(r"\b(?:is|are|can|uses?|proposes?|shows?|demonstrates?|improves?|captures?|reconstructs?|实现|提出|表明|说明|用于|能够)\b", text, re.IGNORECASE):
        score += 1.0
    context_tokens = set(_loose_tokens(f"{claim} {heading}"))
    if context_tokens:
        overlap = len(set(tokens) & context_tokens)
        score += min(2.0, overlap * 0.3)
    if re.search(r"\b(?:single[-\s]?pixel|imaging|deep learning|compressive|neural|reconstruction|sampling)\b", text, re.IGNORECASE):
        score += 1.0
    return score


def _join_evidence_window(sentences: list[str], *, center_idx: int, claim: str, heading: str, max_len: int) -> str:
    if not sentences:
        return ""
    usable = [
        idx
        for idx, sentence in enumerate(sentences)
        if _usable_evidence_sentence(sentence)
    ]
    if center_idx not in usable:
        return ""

    chosen = [center_idx]
    center_score = _sentence_quality(sentences[center_idx], claim=claim, heading=heading)

    prev_idx = center_idx - 1
    if prev_idx >= 0 and _usable_evidence_sentence(sentences[prev_idx]):
        prev_score = _sentence_quality(sentences[prev_idx], claim=claim, heading=heading)
        if prev_score >= 1.0 or center_score < 2.5:
            chosen.insert(0, prev_idx)

    for next_idx in range(center_idx + 1, min(len(sentences), center_idx + 3)):
        if len(chosen) >= 3:
            break
        if not _usable_evidence_sentence(sentences[next_idx]):
            continue
        next_score = _sentence_quality(sentences[next_idx], claim=claim, heading=heading)
        if next_score < 0.5 and len(chosen) > 1:
            continue
        chosen.append(next_idx)

    out: list[str] = []
    for idx in sorted(set(chosen)):
        candidate = " ".join([*out, sentences[idx]]).strip()
        if out and len(candidate) > max_len:
            continue
        out.append(sentences[idx])
    return " ".join(out).strip()


def _first_text(rec: Mapping[str, Any], *keys: str, max_len: int = 520) -> str:
    for key in keys:
        value = _clean_text(rec.get(key), max_len=max_len)
        if value:
            return value
    return ""


def _first_raw_value(rec: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = rec.get(key)
        if str(value or "").strip():
            return value
    return ""


_CARD_TEXT_LIMITS = {
    "card_title": 220,
    "card_subtitle": 220,
    "card_takeaway_label": 80,
    "card_takeaway": 140,
    "card_claim_label": 80,
    "card_claim": 420,
    "card_locator_label": 80,
    "card_locator": 260,
    "card_evidence_label": 80,
    "card_evidence": 520,
    "card_reference_label": 80,
    "card_reference_entry": 900,
    "card_support_label": 80,
    "card_support_explanation": 420,
    "card_quality_label": 80,
    "card_warning": 360,
}


def _finalize_card_output(card: dict[str, Any], *, route: str) -> dict[str, Any]:
    out = dict(card)
    for key, limit in _CARD_TEXT_LIMITS.items():
        if key == "card_evidence":
            out[key] = finish_evidence_text(out.get(key), max_len=limit)
        else:
            out[key] = _clean_text(out.get(key), max_len=limit)

    evidence = str(out.get("card_evidence") or "").strip()
    claim = str(out.get("card_claim") or "").strip()
    takeaway = str(out.get("card_takeaway") or "").strip()
    support = str(out.get("card_support_explanation") or "").strip()

    if takeaway and (
        _sameish(takeaway, evidence)
        or _sameish(takeaway, claim)
        or _looks_low_value_takeaway(takeaway)
    ):
        out["card_takeaway"] = ""
        takeaway = ""

    if route == "system_b":
        if claim and evidence and _sameish(claim, evidence):
            out["card_claim"] = ""
            claim = ""
        if support and (
            _sameish(support, takeaway)
            or _sameish(support, evidence)
            or _sameish(support, claim)
            or _looks_generic_system_b_text(support)
        ):
            out["card_support_explanation"] = ""
    elif support and (
        _sameish(support, evidence)
        or _sameish(support, claim)
        or _sameish(support, takeaway)
    ):
        out["card_support_explanation"] = ""

    if not str(out.get("card_evidence") or "").strip():
        out["card_evidence_label"] = _clean_text(out.get("card_evidence_label"), max_len=80)
    return out


def _clean_reference_entry(value: Any, *, max_len: int = 900) -> str:
    text = _clean_text(value, max_len=max_len)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _looks_reference_author_segment(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    comma_count = text.count(",")
    amp_or_and = bool(re.search(r"\b(?:and|et al)\b|&", text, re.IGNORECASE))
    initials = len(re.findall(r"\b[A-Z]\.?\b", text))
    surnames = len(re.findall(r"\b[A-Z][A-Za-z'`-]{2,}\b", text))
    if comma_count >= 1 and initials >= 2 and surnames >= 2:
        return True
    if comma_count >= 2 and (initials >= 2 or amp_or_and):
        return True
    return comma_count >= 3 and surnames >= 3


def _looks_reference_venue_segment(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lower = text.lower()
    if re.search(r"\b(?:18|19|20)\d{2}\b", lower):
        return True
    if re.search(r"\b\d{1,4}\s*,\s*\d{1,6}(?:[-–]\d{1,6})?\b", lower):
        return True
    venue_tokens = (
        "journal",
        "transactions",
        "proceedings",
        "conference",
        "letters",
        "express",
        "optics",
        "photonics",
        "physical review",
        "phys. rev",
        "ieee",
        "acm",
        "springer",
        "elsevier",
        "nature",
        "science",
        "arxiv",
    )
    return len(lower.split()) <= 12 and any(token in lower for token in venue_tokens)


def _looks_reference_title_segment(value: str) -> bool:
    text = str(value or "").strip(" .;:,")
    if not text:
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'`-]*|[\u4e00-\u9fff]+", text)
    if len(text) < 8 or len(text) > 260:
        return False
    if len(words) < 3 or len(words) > 32:
        return False
    if re.search(r"\b(?:doi|arxiv)\b", text, re.IGNORECASE):
        return False
    if _looks_reference_author_segment(text) or _looks_reference_venue_segment(text):
        return False
    if len(re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)) < 2:
        return False
    return True


def _fallback_system_b_title_from_raw_reference(raw: str) -> str:
    text = _clean_reference_entry(raw, max_len=900)
    if not text:
        return ""
    has_reference_shape = bool(
        re.search(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)])\s+", text)
        or re.search(r"\b(?:18|19|20)\d{2}\b", text)
        or re.search(r"\bdoi\s*:?\s*10\.", text, re.IGNORECASE)
        or text.count(",") >= 2
    )
    if not has_reference_shape:
        return ""
    text = re.sub(r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)])\s*", "", text)
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoi\s*:?\s*10\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\barxiv\s*:?\s*\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .;:,")
    if not text:
        return ""

    quoted = re.search(r"[\"“”]([^\"“”]{8,260})[\"“”]", text)
    if quoted:
        title = str(quoted.group(1) or "").strip(" .;:,")
        if _looks_reference_title_segment(title):
            return _clean_text(title, max_len=220)

    year_match = re.search(r"\((?:18|19|20)\d{2}\)\s*([^.]{8,260})\.", text)
    if year_match:
        title = str(year_match.group(1) or "").strip(" .;:,")
        if _looks_reference_title_segment(title):
            return _clean_text(title, max_len=220)

    segments = [
        part.strip(" .;:,")
        for part in re.split(r"\.\s+(?=[A-Z][A-Za-z0-9])", text)
        if part.strip(" .;:,")
    ]
    if not segments:
        return ""

    for idx, segment in enumerate(segments):
        if idx == 0 and _looks_reference_author_segment(segment):
            continue
        if _looks_reference_title_segment(segment):
            return _clean_text(segment, max_len=220)
    return ""


def _source_name(source_path: str) -> str:
    text = str(source_path or "").strip()
    if not text:
        return ""
    name = Path(text).name or text
    low = name.lower()
    if low.endswith(".en.md"):
        return name[:-6] + ".pdf"
    if low.endswith(".md"):
        return name[:-3] + ".pdf"
    return name


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _page_label(start: Any, end: Any) -> str:
    p0 = _safe_int(start)
    p1 = _safe_int(end)
    if p0 <= 0:
        return ""
    if p1 <= 0 or p1 == p0:
        return f"p. {p0}"
    return f"pp. {min(p0, p1)}-{max(p0, p1)}"


def _anchor_kind_label(value: str) -> str:
    key = str(value or "").strip().lower()
    return {
        "sentence": "句子",
        "paragraph": "段落",
        "equation": "公式",
        "figure": "图",
        "table": "表",
    }.get(key, str(value or "").strip())


def _sameish(left: str, right: str) -> bool:
    a = re.sub(r"\s+", " ", str(left or "")).strip().lower()
    b = re.sub(r"\s+", " ", str(right or "")).strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 36 and a in b:
        return True
    if len(b) >= 36 and b in a:
        return True
    at = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", a))
    bt = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", b))
    if len(at) < 5 or len(bt) < 5:
        return False
    return len(at & bt) / max(1, min(len(at), len(bt))) >= 0.82


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))


def _looks_low_value_takeaway(value: str) -> bool:
    text = _clean_text(value, max_len=360)
    if not text:
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z\s-]{2,48}\s+\d{1,3}", text):
        return True
    if re.search(r"(?:这条证据|该证据|this evidence|the evidence).{0,12}(?:支持|支撑|supports?)", text, re.IGNORECASE):
        return True
    tokens = _loose_tokens(text)
    if _has_cjk(text):
        return len(text) < 12 and not re.search(r"[：:，,。；;]", text)
    return len(tokens) <= 6


def _trim_takeaway(value: str, *, max_len: int = 96) -> str:
    text = _clean_text(value, max_len=max_len + 20)
    text = re.sub(r"^\s*(?:这条证据说明|证据说明|它说明|说明)[:：]\s*", "", text)
    text = text.strip(" \t\r\n。；;")
    if len(text) > max_len:
        text = text[: max(0, max_len - 1)].rstrip(" ，,；;:：") + "..."
    if text and _has_cjk(text) and not text.endswith(("。", "！", "？", "...")):
        text += "。"
    return text


def _takeaway_from_english_evidence(evidence: str) -> str:
    text = str(evidence or "")
    low = text.lower()
    if "dmd" in low and ("spatially filter" in low or "single-pixel camera configuration" in low):
        return "DMD 可以作为单像素相机中的空间调制器，通过选择性重定向光束来完成采样和成像配置。"
    if "single-pixel imaging technology can capture images at wavelengths outside" in low:
        return "单像素成像可以覆盖传统焦平面阵列探测器难以触达的波段，但实用性仍受图像质量和计算时间限制。"
    if "structured detection" in low and "optical sectioning" in low:
        return "结构化检测用于在激光扫描显微中同时改善层切、分辨率和信噪比。"
    if "deep learning" in low and "single-pixel" in low and re.search(r"\b(?:quality|speed|reconstruction)\b", low):
        return "深度学习方法主要用于提升单像素成像的重建质量、速度或采样效率。"
    if "snapshot compressive imaging" in low and ("recover" in low or "reconstruct" in low):
        return "快照压缩成像通过一次压缩观测恢复场景信息，是该回答所说成像任务的直接背景。"
    return ""


def _system_a_takeaway(*, claim: str, evidence: str, heading: str) -> str:
    claim_clean = _trim_takeaway(claim, max_len=110)
    if claim_clean and _has_cjk(claim_clean) and not _looks_low_value_takeaway(claim_clean):
        return claim_clean

    evidence_takeaway = _trim_takeaway(_takeaway_from_english_evidence(evidence), max_len=110)
    if evidence_takeaway and not _looks_low_value_takeaway(evidence_takeaway):
        return evidence_takeaway

    heading_clean = _clean_text(heading, max_len=120)
    if _has_cjk(heading_clean) and evidence:
        candidate = f"这条证据对应“{heading_clean}”这一部分的关键表述。"
        if not _looks_low_value_takeaway(candidate):
            return candidate
    return ""


def _looks_generic_system_b_text(value: str) -> bool:
    text = _clean_text(value, max_len=360).lower()
    if not text:
        return True
    generic_patterns = [
        r"这条链接把回答中的说法追溯到",
        r"这条参考是当前论文给出的上游来源",
        r"这篇上游文献条目",
        r"the user is asking about the evidence",
        r"upstream paper to open next",
        r"cited prior work or background source",
        r"trace the upstream origin",
        r"this reference is the cited prior work",
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in generic_patterns):
        return True
    tokens = _loose_tokens(text)
    return len(tokens) <= 5


def _system_b_explicit_takeaway(*, role: str, relation: str) -> str:
    for value in (role, relation):
        text = _trim_takeaway(value, max_len=118)
        if not text or not _has_cjk(text) or _looks_generic_system_b_text(text):
            continue
        text = re.sub(r"^用户问[“\"].+?[”\"，,；;]\s*", "", text)
        text = re.sub(r"^这条参考(?:正好)?说明", "这篇上游文献说明", text)
        text = re.sub(r"^它说明", "这篇上游文献说明", text)
        return _trim_takeaway(text, max_len=118)
    return ""


def _system_b_takeaway(*, title: str, claim: str, context: str, role: str, relation: str) -> str:
    explicit = _system_b_explicit_takeaway(role=role, relation=relation)
    if explicit:
        return explicit

    combined = " ".join(str(part or "") for part in (title, claim, context, role, relation)).lower()
    if "admm-net" in combined or "unfold" in combined or "unrolled" in combined:
        return "这篇上游文献提供把迭代优化思想展开成可训练网络的前人线索。"
    if "admm" in combined or "alternating direction method" in combined:
        return "这篇上游文献提供 ADMM 优化框架背景，用来判断当前论文是在借鉴既有方法。"
    if "single-shot compressive spectral imaging" in combined:
        return "这篇上游文献提供单次压缩光谱成像的前人背景，是回答中相关概念的来源线索。"
    if "single-pixel imaging via compressive sampling" in combined or (
        "single-pixel" in combined and "compressive sampling" in combined
    ):
        return "这篇上游文献是单像素压缩采样路线的经典来源，适合用来补上“单个探测器如何靠调制与重建成像”的基础背景。"
    if re.search(r"\b(?:baseline|compare|compared|comparison|against)\b", combined):
        return "这篇上游文献在当前论文中主要作为对比基线或相关方法参照。"
    if re.search(r"\b(?:dataset|benchmark|evaluation|experiment)\b", combined):
        return "这篇上游文献提供实验数据、评测场景或 benchmark 线索。"
    if re.search(r"\b(?:architecture|network|model|module)\b", combined):
        return "这篇上游文献提供模型结构或方法设计上的前人参考。"
    if re.search(r"\b(?:background|prior work|related work|origin|source)\b", combined):
        return "这篇上游文献提供当前说法的相关工作背景和来源线索。"
    return ""


def _quality_label(score: float, *, route: str) -> str:
    if route == "system_b":
        if score >= 0.78:
            return "上游来源清楚"
        if score >= 0.58:
            return "可追溯来源"
        return "需要核对来源"
    if score >= 0.78:
        return "证据匹配"
    if score >= 0.52:
        return "候选依据"
    return "需要核对"


def _locator(rec: Mapping[str, Any]) -> str:
    loc = _first_text(rec, "location_label", max_len=260)
    if loc:
        return loc
    heading = _first_text(rec, "heading_path", max_len=180)
    page = _page_label(rec.get("page_start"), rec.get("page_end"))
    kind = _anchor_kind_label(str(rec.get("anchor_kind") or ""))
    return " · ".join(part for part in (heading, page, kind) if part)


def _compose_system_a(rec: dict[str, Any]) -> dict[str, Any]:
    source = _first_text(rec, "source_name", max_len=180) or _source_name(str(rec.get("source_path") or ""))
    heading = _first_text(rec, "heading_path", "title", max_len=180)
    title = source or heading or "答案依据"
    claim_raw = _first_text(rec, "answer_claim", max_len=420)
    evidence_raw = _first_text(rec, "evidence_quote", "summary_line", "raw", "cite_fmt", max_len=1400)
    evidence_raw_for_pack = _first_raw_value(rec, "evidence_quote", "summary_line", "raw", "cite_fmt") or evidence_raw
    locator = _locator(rec)
    support_hint = _first_text(rec, "support_relation", "binding_reason", "why_line", max_len=420)
    pack = build_system_a_evidence_pack(
        answer_claim=claim_raw,
        evidence_raw=evidence_raw_for_pack,
        source=source,
        title=_first_text(rec, "title", max_len=240),
        heading=heading,
        location_label=locator,
        support_hint=support_hint,
    )
    claim = pack.answer_claim
    evidence = pack.evidence_quote
    takeaway = _system_a_takeaway(claim=claim, evidence=evidence, heading=heading)
    if not takeaway:
        takeaway = pack.evidence_focus
    if takeaway and (_sameish(takeaway, evidence) or _sameish(takeaway, claim)):
        takeaway = ""
    subtitle = locator or (heading if heading and heading != title else "")
    binding_status = str(rec.get("binding_status") or "").strip().lower()
    binding_confidence = _safe_float(rec.get("binding_confidence"), 0.0)
    support = pack.support_explanation or support_hint

    ranked_score = min(0.76, max(0.42, _safe_float(rec.get("score"), 0.0) / 10.0))
    score = max(binding_confidence, ranked_score) if binding_confidence else ranked_score
    score += pack.score_delta
    flags: list[str] = list(pack.flags)
    if not claim:
        flags.append("missing_answer_claim")
        score -= 0.08
    if not evidence:
        flags.append("missing_evidence_quote")
        score -= 0.16
    if not locator:
        flags.append("missing_precise_location")
        score -= 0.08
    if binding_status == "mismatch":
        flags.append("binding_mismatch")
        score = min(score, 0.25)
    elif binding_status == "candidate":
        flags.append("candidate_binding")
        score = min(score, 0.58)
    if claim and evidence and _sameish(claim, evidence):
        flags.append("claim_duplicates_evidence")
    if bool(rec.get("occurrence_specific")):
        flags.append("occurrence_specific_claim")
    score = max(0.0, min(1.0, score))

    needs_review = bool(binding_status in {"candidate", "mismatch"} or score < 0.55)
    support_label = ""
    support_text = ""
    if needs_review:
        support_label = "这条依据的可靠度"
        support_text = support or "这条引用只能作为候选依据；请打开原文核对答案句和命中片段是否真正对应。"
    warning = ""
    if "binding_mismatch" in flags:
        warning = "答案句和命中片段术语冲突，已尽量抑制链接；如果仍看到这张卡，请优先打开原文核对。"
    elif "candidate_binding" in flags or score < 0.55:
        warning = "这条链接只是候选依据，建议打开原文确认语境。"

    return _finalize_card_output({
        "card_kind": "answer_evidence",
        "card_title": title,
        "card_subtitle": subtitle,
        "card_takeaway_label": "证据重点",
        "card_takeaway": takeaway,
        "card_claim_label": "答案中的话",
        "card_claim": claim,
        "card_locator_label": pack.location_label_name or "原文位置",
        "card_locator": pack.location_label or locator,
        "card_evidence_label": pack.evidence_label or "原文证据",
        "card_evidence": evidence,
        "card_support_label": support_label,
        "card_support_explanation": support_text,
        "card_quality_label": _quality_label(score, route="system_a"),
        "card_quality_score": round(score, 3),
        "card_quality_flags": flags,
        "card_warning": warning,
        "card_flow": [],
    }, route="system_a")


def _compose_system_b(rec: dict[str, Any]) -> dict[str, Any]:
    source = _first_text(rec, "source_name", max_len=180) or _source_name(str(rec.get("source_path") or ""))
    raw_reference = _clean_reference_entry(rec.get("raw") or rec.get("cite_fmt"), max_len=900)
    explicit_title = _first_text(rec, "title", max_len=220)
    parsed_title = _fallback_system_b_title_from_raw_reference(raw_reference)
    title = explicit_title or parsed_title or "上游参考文献"
    subtitle = " · ".join(
        part
        for part in (
            _first_text(rec, "authors", max_len=160),
            _first_text(rec, "venue", max_len=80),
            _first_text(rec, "year", max_len=16),
        )
        if part
    )
    claim_raw = _first_text(rec, "answer_claim", max_len=420)
    context_raw = _first_text(rec, "citation_context", "evidence_quote", "summary_line", max_len=1400)
    locator = _locator(rec) or source
    role = _first_text(rec, "upstream_work_role", "why_line", max_len=420)
    relation = _first_text(rec, "user_question_relation", "support_relation", max_len=420)
    pack = build_system_b_evidence_pack(
        answer_claim=claim_raw,
        citation_context_raw=context_raw,
        citation_context_source=str(rec.get("citation_context_source") or rec.get("evidence_source") or ""),
        source=source,
        title=title,
        heading=_first_text(rec, "heading_path", "location_label", max_len=180),
        location_label=locator,
        raw_reference=raw_reference,
        role_hint=role,
        relation_hint=relation,
    )
    claim = pack.answer_claim
    context = pack.evidence_quote
    takeaway = _system_b_takeaway(title=title, claim=claim, context=context, role=role, relation=relation)
    support = pack.support_explanation

    score = 0.72 + pack.score_delta
    flags: list[str] = list(pack.flags)
    if not explicit_title and not parsed_title:
        flags.append("missing_reference_title")
        score -= 0.16
    if not source:
        flags.append("missing_citing_source")
        score -= 0.12
    if not locator:
        flags.append("missing_citing_location")
        score -= 0.1
    if not takeaway:
        flags.append("missing_takeaway")
        score -= 0.08
    score = max(0.0, min(1.0, score))

    evidence_label = pack.evidence_label or "引用语境"
    warning = pack.warning
    if not warning and score < 0.58:
        warning = "这条上游参考信息不完整，建议打开引用语境确认。"

    return _finalize_card_output({
        "card_kind": "upstream_reference",
        "card_title": title,
        "card_subtitle": subtitle,
        "card_takeaway_label": "上游作用",
        "card_takeaway": takeaway,
        "card_claim_label": "答案里的这句话",
        "card_claim": claim,
        "card_locator_label": pack.location_label_name or "引用出现位置",
        "card_locator": pack.location_label or locator,
        "card_evidence_label": evidence_label,
        "card_evidence": context,
        "card_reference_label": pack.reference_label,
        "card_reference_entry": pack.reference_entry,
        "card_support_label": "",
        "card_support_explanation": support,
        "card_quality_label": _quality_label(score, route="system_b"),
        "card_quality_score": round(score, 3),
        "card_quality_flags": flags,
        "card_warning": warning,
        "card_flow": [],
    }, route="system_b")


def compose_citation_card(detail: Mapping[str, Any] | None) -> dict[str, Any]:
    rec = dict(detail or {}) if isinstance(detail, Mapping) else {}
    if not rec:
        return {}
    card = _compose_system_b(rec) if bool(rec.get("is_inpaper")) else _compose_system_a(rec)
    rec.update(card)
    return rec
