from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
import re

from kb.citation_meta import fetch_best_crossref_for_reference, fetch_best_crossref_meta
from kb.file_naming import citation_meta_display_pdf_name
from kb.library_store import LibraryStore
from kb.source_filters import is_excluded_source_path
from ui.refs_renderer import (
    _enrich_bibliometrics,
    _has_metrics_payload,
    _infer_title_from_source_text,
    _normalize_reference_for_popup,
    _parse_filename_meta,
    _build_ref_navigation,
    _fallback_why_line_ui,
    _is_non_navigational_heading_ui,
    _looks_like_doc_title_heading_ui,
    _open_pdf_at,
    _resolve_pdf_for_source,
    _safe_page_range,
    _sanitize_heading_path_ui,
    _score_tier,
    _split_section_subsection,
    _top_heading,
    fetch_crossref_meta,
)

_MIN_REF_UI_SCORE = 5.2
_MAX_REF_UI_GAP = 1.8


def _source_filename(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return ""
    parts = re.split(r"[\\/]+", s)
    return str(parts[-1] or "").strip() if parts else s


def _clamp_ui_score(score: float) -> float:
    try:
        v = float(score)
    except Exception:
        v = 0.0
    return max(0.0, min(10.0, v))


def _calibrated_ui_score(meta: dict, rank: dict) -> float | None:
    try:
        llm_score = float(rank.get("llm", 0.0) or 0.0)
    except Exception:
        llm_score = 0.0
    if llm_score <= 0:
        return None

    try:
        bm25 = float(rank.get("bm25", 0.0) or 0.0)
    except Exception:
        bm25 = 0.0
    try:
        deep = float(rank.get("deep", 0.0) or 0.0)
    except Exception:
        deep = 0.0
    try:
        term_bonus = float(rank.get("term_bonus", 0.0) or 0.0)
    except Exception:
        term_bonus = 0.0
    try:
        semantic_score = float(rank.get("semantic_score", 0.0) or 0.0)
    except Exception:
        semantic_score = 0.0

    ui = llm_score / 10.0
    if semantic_score > 0:
        ui = min(ui, semantic_score)

    if term_bonus < 0:
        ui += 0.60 * term_bonus
    elif term_bonus > 0:
        ui += min(0.30, 0.12 * term_bonus)

    if bm25 < 1.0:
        ui -= 1.15
    elif bm25 < 2.0:
        ui -= 0.75
    elif bm25 < 3.5:
        ui -= 0.35

    if deep <= 0:
        ui -= 0.15

    section = str(
        meta.get("ref_section")
        or ((meta.get("ref_pack") or {}).get("section") if isinstance(meta.get("ref_pack"), dict) else "")
        or meta.get("ref_best_heading_path")
        or meta.get("heading_path")
        or ""
    ).strip()
    loc_quality = str(meta.get("ref_loc_quality") or "").strip().lower()
    if not section:
        ui -= 0.70
    elif loc_quality != "high":
        ui -= 0.25

    # Do not allow weak lexical evidence to surface as "high relevance"
    # just because the LLM was optimistic.
    if term_bonus <= 0.0 and bm25 < 2.0:
        ui = min(ui, 6.4)
    if term_bonus <= 0.0 and bm25 < 1.0:
        ui = min(ui, 5.8)
    if term_bonus <= 0.0 and (not section):
        ui = min(ui, 5.6)

    return _clamp_ui_score(ui)

def _effective_ui_score(hit: dict) -> tuple[float | None, bool]:
    meta = (hit or {}).get("meta", {}) or {}
    pack_state = str(meta.get("ref_pack_state") or "").strip().lower()
    rank = meta.get("ref_rank") if isinstance(meta.get("ref_rank"), dict) else {}
    if pack_state == "ready":
        calibrated = _calibrated_ui_score(meta, rank)
        if calibrated is not None:
            return calibrated, False
    return None, pack_state == "pending"


def _should_force_keep_ref_hit(hit: dict) -> bool:
    meta = (hit or {}).get("meta", {}) or {}
    if not isinstance(meta, dict):
        return False
    if str(meta.get("ref_pack_state") or "").strip().lower() == "pending":
        return True
    try:
        explicit_doc = float(meta.get("explicit_doc_match_score") or 0.0)
    except Exception:
        explicit_doc = 0.0
    if explicit_doc >= 6.0:
        return True
    if str(meta.get("anchor_target_kind") or "").strip():
        try:
            anchor_score = float(meta.get("anchor_match_score") or 0.0)
        except Exception:
            anchor_score = 0.0
        if anchor_score > 0.0:
            return True
    return False


def _display_source_name(source_path: str, pdf_path: Path | None, lib_store: LibraryStore | None) -> str:
    try:
        if pdf_path is not None and lib_store is not None:
            meta = lib_store.get_citation_meta(pdf_path)
            full_name = citation_meta_display_pdf_name(meta)
            if full_name:
                return full_name
    except Exception:
        pass

    name = _source_filename(source_path) or str(source_path or "")
    low = name.lower()
    if low.endswith(".en.md"):
        name = name[:-6] + ".pdf"
    elif low.endswith(".md"):
        name = name[:-3] + ".pdf"
    return name or "unknown.pdf"


def _positive_int(x) -> int:
    try:
        v = int(x)
    except Exception:
        return 0
    return v if v > 0 else 0


def _non_negative_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if v > 0.0 else 0.0


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


def build_hit_ui_meta(
    hit: dict,
    *,
    prompt: str,
    pdf_root: Path | None,
    lib_store: LibraryStore | None,
    preloaded_citation_meta: dict[str, dict] | None = None,
) -> dict:
    meta = (hit or {}).get("meta", {}) or {}
    source_path = str(meta.get("source_path") or "").strip()
    heading_path = _sanitize_heading_path_ui(
        str(meta.get("ref_best_heading_path") or meta.get("heading_path") or "").strip(),
        prompt=prompt,
        source_path=source_path,
    )
    heading = str(
        meta.get("top_heading")
        or _top_heading(heading_path)
        or _top_heading(str(meta.get("heading_path") or ""))
        or ""
    ).strip()
    if heading and _is_non_navigational_heading_ui(heading, prompt=prompt, source_path=source_path):
        heading = ""
    if heading and _looks_like_doc_title_heading_ui(heading, source_path):
        heading = ""

    section_label = str(meta.get("ref_section") or "").strip()
    subsection_label = str(meta.get("ref_subsection") or "").strip()
    if section_label and _is_non_navigational_heading_ui(section_label, prompt=prompt, source_path=source_path):
        section_label = ""
    if subsection_label and _is_non_navigational_heading_ui(subsection_label, prompt=prompt, source_path=source_path):
        subsection_label = ""
    if (not section_label) and heading_path:
        section_label, subsection_label = _split_section_subsection(heading_path)
    if section_label and _looks_like_doc_title_heading_ui(section_label, source_path):
        section_label = ""
        subsection_label = ""

    p0, p1 = _safe_page_range(meta)
    score, score_pending = _effective_ui_score(hit)
    anchor_target_kind = str(meta.get("anchor_target_kind") or "").strip().lower()
    anchor_target_number = _positive_int(meta.get("anchor_target_number"))
    anchor_match_score = _non_negative_float(meta.get("anchor_match_score"))
    explicit_doc_match_score = _non_negative_float(meta.get("explicit_doc_match_score"))
    semantic_badges = _build_semantic_badges(
        anchor_target_kind=anchor_target_kind,
        anchor_target_number=anchor_target_number,
        anchor_match_score=anchor_match_score,
        explicit_doc_match_score=explicit_doc_match_score,
    )
    nav = _build_ref_navigation(meta, prompt=prompt, heading_fallback=heading)
    summary_line = str(nav.get("summary_line") or nav.get("what") or "").strip()
    why_line = str(nav.get("why") or "").strip()
    if summary_line and (not why_line):
        why_line = _fallback_why_line_ui(
            prompt=prompt,
            heading_label=heading_path or heading,
            section_label=section_label,
            subsection_label=subsection_label,
            find_terms=list(nav.get("find") or []),
        )

    pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
    display_name = _display_source_name(source_path, pdf_path, lib_store)
    citation_meta = {}
    preload_map = preloaded_citation_meta if isinstance(preloaded_citation_meta, dict) else {}
    preload_meta = preload_map.get(source_path) if source_path else None
    if isinstance(preload_meta, dict) and preload_meta:
        citation_meta = dict(preload_meta)
    if pdf_path is not None and lib_store is not None:
        try:
            if not citation_meta:
                citation_meta = lib_store.get_citation_meta(pdf_path) or {}
        except Exception:
            if not citation_meta:
                citation_meta = {}

    return {
        "display_name": display_name,
        "heading_path": heading_path or heading,
        "section_label": section_label,
        "subsection_label": subsection_label,
        "page_start": p0,
        "page_end": p1,
        "score": score,
        "score_pending": bool(score_pending),
        "score_tier": _score_tier(score or 0.0) if score is not None else "",
        "summary_line": summary_line,
        "why_line": why_line,
        "anchor_target_kind": anchor_target_kind,
        "anchor_target_number": anchor_target_number,
        "anchor_match_score": anchor_match_score,
        "explicit_doc_match_score": explicit_doc_match_score,
        "semantic_badges": semantic_badges,
        "can_open": bool(pdf_path),
        "citation_meta": citation_meta if isinstance(citation_meta, dict) else {},
        "source_path": source_path,
    }


def _should_prefetch_citation_meta(meta: dict | None) -> bool:
    if not isinstance(meta, dict) or (not meta):
        return True
    if not _has_metrics_payload(meta):
        return True
    if not str(meta.get("doi") or meta.get("doi_url") or "").strip():
        return True
    if not str(meta.get("venue") or meta.get("conference_name") or "").strip():
        return True
    if not str(meta.get("year") or "").strip():
        return True
    return False


def _prefetch_refs_citation_meta(
    hits: list[dict],
    *,
    pdf_root: Path | None,
    md_root: Path | None,
    lib_store: LibraryStore | None,
) -> dict[str, dict]:
    tasks: dict[str, tuple[str, dict]] = {}
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        source_path = str((meta or {}).get("source_path") or (ui_meta or {}).get("source_path") or "").strip()
        if (not source_path) or is_excluded_source_path(source_path):
            continue
        existing = (ui_meta or {}).get("citation_meta")
        existing_meta = existing if isinstance(existing, dict) else {}
        if (source_path in tasks) or (not _should_prefetch_citation_meta(existing_meta)):
            continue
        tasks[source_path] = (source_path, existing_meta)
    if not tasks:
        return {}

    out: dict[str, dict] = {}

    def _one(source_path: str) -> tuple[str, dict]:
        meta = ensure_source_citation_meta(
            source_path=source_path,
            pdf_root=pdf_root,
            md_root=md_root,
            lib_store=lib_store,
        )
        return source_path, (meta if isinstance(meta, dict) else {})

    max_workers = max(1, min(4, len(tasks)))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one, source_path) for source_path in tasks.keys()]
            for fu in as_completed(futs):
                try:
                    source_path, meta = fu.result()
                except Exception:
                    continue
                out[source_path] = meta
    except Exception:
        for source_path in tasks.keys():
            try:
                source_path2, meta = _one(source_path)
            except Exception:
                continue
            out[source_path2] = meta
    return out


def enrich_refs_payload(
    refs_by_user: dict[int, dict],
    *,
    pdf_root: Path | None,
    md_root: Path | None,
    lib_store: LibraryStore | None,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for user_msg_id, pack in (refs_by_user or {}).items():
        if not isinstance(pack, dict):
            continue
        prompt = str(pack.get("prompt") or "").strip()
        raw_hits = []
        scored_ready: list[float] = []
        for hit in list(pack.get("hits") or []):
            if not isinstance(hit, dict):
                continue
            meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
            source_path = str((meta or {}).get("source_path") or "").strip()
            if is_excluded_source_path(source_path):
                continue
            raw_hits.append(dict(hit))
            score, score_pending = _effective_ui_score(hit)
            if (not score_pending) and (score is not None):
                scored_ready.append(float(score))
        best_ready = max(scored_ready) if scored_ready else None
        dyn_keep_min = max(_MIN_REF_UI_SCORE, (best_ready - _MAX_REF_UI_GAP)) if best_ready is not None else _MIN_REF_UI_SCORE
        has_pending = any(
            str((((hit.get("meta") if isinstance(hit.get("meta"), dict) else {}) or {}).get("ref_pack_state") or "")).strip().lower() == "pending"
            for hit in raw_hits
        )

        hits = []
        for hit2 in raw_hits:
            score, score_pending = _effective_ui_score(hit2)
            force_keep = _should_force_keep_ref_hit(hit2)
            if has_pending:
                hits.append(hit2)
                continue
            if force_keep:
                hits.append(hit2)
                continue
            if (score is None) and (not score_pending):
                continue
            if (not score_pending) and (score is not None) and score < dyn_keep_min:
                continue
            hits.append(hit2)
        if (not hits) and raw_hits:
            fallback_hit = next((hit for hit in raw_hits if _should_force_keep_ref_hit(hit)), None)
            if fallback_hit is not None:
                hits = [fallback_hit]
        preloaded_citation_meta = _prefetch_refs_citation_meta(
            hits,
            pdf_root=pdf_root,
            md_root=md_root,
            lib_store=lib_store,
        ) if hits else {}
        for hit2 in hits:
            hit2["ui_meta"] = build_hit_ui_meta(
                hit2,
                prompt=prompt,
                pdf_root=pdf_root,
                lib_store=lib_store,
                preloaded_citation_meta=preloaded_citation_meta,
            )
        pack2 = dict(pack)
        pack2["hits"] = hits
        out[int(user_msg_id)] = pack2
    return out


def open_reference_source(*, source_path: str, pdf_root: Path | None, page: int | None = None) -> tuple[bool, str]:
    pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
    if pdf_path is None:
        return False, "PDF not found"
    return _open_pdf_at(pdf_path, page=page)


def build_doi_url(doi_or_url: str) -> str:
    raw = str(doi_or_url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return "https://doi.org/" + quote(raw, safe="/:;._-()")


def _is_weak_meta_value(key: str, value: str) -> bool:
    s = str(value or "").strip()
    if not s:
        return True
    if key == "title":
        if len(s) <= 4:
            return True
        if len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", s)) <= 1:
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z.\s&-]{1,40}\(\d{4}\)\.?", s):
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z.\s&-]{1,40}\d{4}\.?", s):
            return True
    if key == "authors":
        if len(s) <= 3:
            return True
        if len(re.findall(r"[A-Za-z\u4e00-\u9fff]+", s)) <= 1:
            return True
    if key == "venue":
        if len(s) <= 1:
            return True
    return False


def _merge_meta_prefer_richer(base: dict, incoming: dict) -> dict:
    out = dict(base or {})
    for key, raw_value in (incoming or {}).items():
        if raw_value in (None, "", [], {}):
            continue
        value = raw_value
        if not isinstance(value, str):
            out[key] = value
            continue
        cur = str(out.get(key) or "").strip()
        new = str(value or "").strip()
        if not cur:
            out[key] = new
            continue
        if key in {
            "doi",
            "doi_url",
            "citation_count",
            "citation_source",
            "journal_if",
            "journal_quartile",
            "journal_if_source",
            "conference_tier",
            "conference_rank_source",
            "conference_ccf",
            "conference_ccf_source",
            "venue_kind",
            "openalex_venue",
            "conference_name",
            "conference_acronym",
            "bibliometrics_checked",
        }:
            out[key] = value
            continue
        cur_weak = _is_weak_meta_value(key, cur)
        new_weak = _is_weak_meta_value(key, new)
        if cur_weak and (not new_weak):
            out[key] = new
            continue
        if (not cur_weak) and new_weak:
            continue
        if len(new) > len(cur) + 12:
            out[key] = new
    return out


def ensure_source_citation_meta(*, source_path: str, pdf_root: Path | None, md_root: Path | None, lib_store: LibraryStore | None) -> dict:
    pdf_path = _resolve_pdf_for_source(pdf_root, source_path)
    meta: dict = {}
    if pdf_path is not None and lib_store is not None:
        try:
            stored = lib_store.get_citation_meta(pdf_path)
            if isinstance(stored, dict):
                meta = dict(stored)
        except Exception:
            meta = {}

    if _has_metrics_payload(meta):
        return meta

    venue_hint, year_hint, _ = _parse_filename_meta(source_path)
    fallback_title = _source_filename(source_path) or str(source_path or "")
    if fallback_title.lower().endswith(".pdf"):
        fallback_title = fallback_title[:-4]
    fallback_title = re.sub(r"\.en\.md$", "", fallback_title, flags=re.I)
    fallback_title = re.sub(r"\.md$", "", fallback_title, flags=re.I)
    search_title = _infer_title_from_source_text(
        source_path,
        fallback_title,
        md_root_hint=str(md_root or ""),
    )
    if search_title:
        meta.setdefault("title", search_title)
    if venue_hint:
        meta.setdefault("venue", venue_hint)
    if year_hint:
        meta.setdefault("year", year_hint)

    fetched = fetch_crossref_meta(
        search_title,
        source_path=source_path,
        expected_venue=venue_hint,
        expected_year=year_hint,
        md_root_hint=str(md_root or ""),
    )
    if isinstance(fetched, dict):
        meta = _merge_meta_prefer_richer(
            meta,
            {k: v for k, v in fetched.items() if v not in (None, "", [], {})},
        )

    enriched = _enrich_bibliometrics(meta or {})
    if isinstance(enriched, dict):
        meta = enriched

    if pdf_path is not None and lib_store is not None and isinstance(meta, dict) and meta:
        try:
            lib_store.set_citation_meta(pdf_path, meta)
        except Exception:
            pass
    return meta if isinstance(meta, dict) else {}


def enrich_citation_detail_meta(detail: dict) -> dict:
    meta = _normalize_reference_for_popup(detail or {}) or dict(detail or {})
    raw0 = str(meta.get("cite_fmt") or meta.get("raw") or "").strip()

    def _fallback_parse_raw_reference(raw: str) -> dict:
        s = str(raw or "").strip()
        s = re.sub(r"^\s*(?:\[\s*\d+\s*\]\s*)+", "", s)
        s = s.replace("*", "")
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return {}

        out: dict[str, str] = {}
        arxiv_m = re.search(r"\barXiv:(\d{4}\.\d{4,5})(?:v\d+)?\b", s, flags=re.I)
        if arxiv_m:
            out["doi_url"] = f"https://arxiv.org/abs/{arxiv_m.group(1)}"
            out["doi"] = f"arXiv:{arxiv_m.group(1)}"

        year_m = re.search(r"\((19|20)\d{2}\)", s)
        if year_m:
            out["year"] = year_m.group(0).strip("()")
        else:
            year2 = re.search(r"\b(19|20)\d{2}\b", s)
            if year2:
                out["year"] = year2.group(0)

        etal_match = re.match(r"^(?P<authors>.+?\bet al\.)\s+(?P<title>.+?)\.\s+(?P<venue>.+)$", s, flags=re.I)
        if etal_match:
            out.setdefault("authors", etal_match.group("authors").strip(" ."))
            out.setdefault("title", etal_match.group("title").strip(" ."))
            out.setdefault("venue", etal_match.group("venue").strip(" ."))
            return out

        parts = [p.strip(" .") for p in re.split(r"\.\s+", s) if p.strip(" .")]
        if len(parts) >= 3:
            out.setdefault("authors", parts[0])
            out.setdefault("title", parts[1])
            out.setdefault("venue", parts[2])
        elif len(parts) == 2:
            out.setdefault("authors", parts[0])
            out.setdefault("title", parts[1])
        return out

    if raw0:
        parsed0 = _fallback_parse_raw_reference(raw0)
        for key, value in parsed0.items():
            if value and not str(meta.get(key) or "").strip():
                meta[key] = value

    title = str(meta.get("title") or "").strip()
    raw = str(meta.get("cite_fmt") or meta.get("raw") or "").strip()
    venue = str(meta.get("venue") or "").strip()
    year = str(meta.get("year") or "").strip()
    doi = str(meta.get("doi") or "").strip()
    doi_url = str(meta.get("doi_url") or "").strip()
    if doi and not doi_url:
        meta["doi_url"] = build_doi_url(doi)
    if doi:
        try:
            canonical = fetch_best_crossref_meta(
                query_title="" if _is_weak_meta_value("title", title) else title,
                doi_hint=doi,
                expected_year=year,
                expected_venue=venue,
                min_score=0.90,
                allow_title_only=False,
            )
        except Exception:
            canonical = None
        if isinstance(canonical, dict):
            meta = _merge_meta_prefer_richer(meta, canonical)
            if str(meta.get("doi") or "").strip() and not str(meta.get("doi_url") or "").strip():
                meta["doi_url"] = build_doi_url(str(meta.get("doi") or "").strip())
    if not doi:
        fetched_ref = None
        if raw:
            try:
                fetched_ref = fetch_best_crossref_for_reference(reference_text=raw, min_score=0.62)
            except Exception:
                fetched_ref = None
        if isinstance(fetched_ref, dict):
            meta = _merge_meta_prefer_richer(
                meta,
                {k: v for k, v in fetched_ref.items() if v not in (None, "", [], {})},
            )
            doi = str(meta.get("doi") or "").strip()
            if doi and not str(meta.get("doi_url") or "").strip():
                meta["doi_url"] = build_doi_url(doi)
        if doi:
            try:
                canonical = fetch_best_crossref_meta(
                    query_title="" if _is_weak_meta_value("title", str(meta.get("title") or title).strip()) else str(meta.get("title") or title).strip(),
                    doi_hint=doi,
                    expected_year=str(meta.get("year") or year).strip(),
                    expected_venue=str(meta.get("venue") or venue).strip(),
                    min_score=0.90,
                    allow_title_only=False,
                )
            except Exception:
                canonical = None
            if isinstance(canonical, dict):
                meta = _merge_meta_prefer_richer(meta, canonical)
                if str(meta.get("doi") or "").strip() and not str(meta.get("doi_url") or "").strip():
                    meta["doi_url"] = build_doi_url(str(meta.get("doi") or "").strip())
            enriched = _enrich_bibliometrics(meta)
            return enriched if isinstance(enriched, dict) else meta

        search_title = title
        if not search_title:
            raw2 = re.sub(r"^\s*(?:\[\s*\d+\s*\]\s*)+", "", raw).strip()
            search_title = raw2[:220]
        fetched = fetch_crossref_meta(
            search_title,
            source_path="",
            expected_venue=venue,
            expected_year=year,
            md_root_hint="",
        )
        if isinstance(fetched, dict):
            meta = _merge_meta_prefer_richer(
                meta,
                {k: v for k, v in fetched.items() if v not in (None, "", [], {})},
            )
            doi = str(meta.get("doi") or "").strip()
            if doi and not str(meta.get("doi_url") or "").strip():
                meta["doi_url"] = build_doi_url(doi)
    enriched = _enrich_bibliometrics(meta)
    return enriched if isinstance(enriched, dict) else meta
