from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .chunking import chunk_markdown
from .llm import DeepSeekChat
from .retrieval_heuristics import (
    _aspects_from_snippets,
    _clean_snippet_for_display,
    _doc_term_bonus,
    _has_cjk,
    _has_latin,
    _is_noise_snippet_text,
    _is_probably_bad_heading,
    _norm_text_for_match,
    _normalize_heading,
    _pick_best_heading_for_doc,
    _preferred_section_keys,
    _query_term_profile,
    _score_tokens,
)
from .retriever import BM25Retriever
from .tokenize import tokenize

# These callbacks are injected by app.py so this module can reuse the shared runtime cache.
_CACHE_GET: Callable[[str, str], Any] = lambda _bucket, _key: None
_CACHE_SET: Callable[..., None] = lambda _bucket, _key, _val, **_kw: None


def configure_cache(cache_get: Callable[[str, str], Any], cache_set: Callable[..., None]) -> None:
    global _CACHE_GET, _CACHE_SET
    _CACHE_GET = cache_get
    _CACHE_SET = cache_set


def _cache_get(bucket: str, key: str):
    return _CACHE_GET(bucket, key)


def _cache_set(bucket: str, key: str, val, *, max_items: int = 600) -> None:
    _CACHE_SET(bucket, key, val, max_items=max_items)


def _is_temp_source_path(source_path: str) -> bool:
    s = (source_path or "").strip()
    if not s:
        return True
    p = Path(s)
    low_parts = [str(x).strip().lower() for x in p.parts]
    low_name = p.name.lower()
    low_stem = p.stem.lower()
    if any(x in {"temp", "__pycache__"} for x in low_parts):
        return True
    if any(x.startswith("__upload__") or x.startswith("_tmp_") or x.startswith("tmp_") for x in low_parts):
        return True
    if low_name.startswith("__upload__") or low_stem.startswith("__upload__"):
        return True
    if low_name.startswith("_tmp_") or low_stem.startswith("_tmp_"):
        return True
    if low_name.startswith("tmp_") or low_stem.startswith("tmp_"):
        return True
    return False

def _top_heading(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    return hp.split(" / ", 1)[0].strip()


def _normalize_heading_path_for_display(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    parts: list[str] = []
    for raw in hp.split(" / "):
        t = _normalize_heading(str(raw).strip())
        if not t:
            continue
        parts.append(t)
    return " / ".join(parts)


def _split_heading_path_levels(heading_path: str) -> tuple[str, str]:
    hp = _normalize_heading_path_for_display(heading_path)
    if not hp:
        return "", ""
    parts = [p.strip() for p in hp.split(" / ") if p.strip()]
    if not parts:
        return "", ""
    return parts[0], " / ".join(parts[1:]).strip()


def _page_range_from_meta(meta: dict) -> tuple[int | None, int | None]:
    def _to_pos_int(x) -> int | None:
        try:
            v = int(x)
        except Exception:
            return None
        return v if v > 0 else None

    p0 = _to_pos_int(meta.get("page_start"))
    p1 = _to_pos_int(meta.get("page_end"))
    if p0 is None:
        p0 = _to_pos_int(meta.get("page"))
    if p0 is None:
        p0 = _to_pos_int(meta.get("page_num"))
    if p0 is None:
        p0 = _to_pos_int(meta.get("page_idx"))
    if p1 is None:
        p1 = p0
    if (p0 is not None) and (p1 is not None) and p1 < p0:
        p0, p1 = p1, p0
    return p0, p1

def _translate_query_for_search(settings, prompt_text: str) -> str | None:
    """
    Translate a CJK-only query to a compact English search query (keywords),
    so BM25 can match English papers.
    """
    q = (prompt_text or "").strip()
    if not q:
        return None
    if not _has_cjk(q) or _has_latin(q):
        return None
    if not getattr(settings, "api_key", None):
        return None

    key = hashlib.sha1((str(getattr(settings, "api_key", None)) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    cached = _cache_get("trans", key)
    if isinstance(cached, str) and cached.strip():
        return cached.strip()

    # Fast dictionary-based fallback for common KB terms.
    # This keeps retrieval responsive even when translation LLM is slow/unavailable.
    terms: list[str] = []
    mapping = [
        ("\u5355\u50cf\u7d20", "single-pixel"),
        ("\u5355\u5149\u5b50", "single-photon"),
        ("\u5355\u66dd\u5149", "single-shot"),
        ("\u5355\u6b21\u66dd\u5149", "single-shot"),
        ("\u538b\u7f29\u6210\u50cf", "compressive imaging"),
        ("\u538b\u7f29\u611f\u77e5", "compressed sensing"),
        ("\u5c0f\u6ce2", "wavelet"),
        ("\u6210\u50cf", "imaging"),
        ("\u91cd\u5efa", "reconstruction"),
        ("\u91c7\u6837", "sampling"),
        ("\u63a9\u819c", "mask pattern"),
        ("\u7f16\u7801", "coding"),
        ("\u5149\u8c31", "spectral"),
        ("\u8d85\u6750\u6599", "metamaterial"),
        ("\u8d85\u8868\u9762", "metasurface"),
        ("\u590d\u7528", "multiplexing"),
        ("\u9891\u5206", "frequency-division"),
    ]
    for zh, en_term in mapping:
        if zh in q:
            terms.append(en_term)
    if terms:
        uniq: list[str] = []
        seen = set()
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        heuristic = " ".join(uniq[:8]).strip()
        if heuristic:
            _cache_set("trans", key, heuristic, max_items=500)
            return heuristic

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=0,
        )
    except Exception:
        settings_fast = settings
    ds = DeepSeekChat(settings_fast)
    system = (
        "You translate Chinese research questions into an English search query for academic retrieval.\n"
        "Rules:\n"
        "- Output ONLY the English search query (no quotes, no explanations).\n"
        "- Prefer keywords and key phrases; include useful synonyms.\n"
        "- Keep it compact (8-18 tokens).\n"
        "- DO NOT confuse terms:\n"
        "  - 鍗曟洕鍏?鍗曟鏇濆厜 -> single-shot, single exposure, snapshot\n"
        "  - 鍗曞儚绱?-> single-pixel\n"
        "  - 鍘嬬缉鎴愬儚 -> compressive imaging\n"
        "  - 鍏夎氨鎴愬儚 -> spectral imaging\n"
        "- If the user didn't mention 鍗曞儚绱? avoid adding single-pixel.\n"
    )
    user = f"Question: {q}\n\nEnglish search keywords:"
    try:
        out = (ds.chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.0, max_tokens=80) or "").strip()
    except Exception:
        out = ""
    out = " ".join(out.split())
    _cache_set("trans", key, out, max_items=500)
    return out or None

def _llm_semantic_rerank_score(settings, *, question: str, doc_headings: list[str], snippets: list[str]) -> tuple[float, str]:
    """
    LLM-based semantic relevance scoring, grounded on snippets (not filenames).

    Returns: (score_0_100, short_reason)
    """
    if not settings or (not getattr(settings, "api_key", None)):
        return 0.0, ""

    q = (question or "").strip()
    if not q:
        return 0.0, ""

    hs = [h for h in (doc_headings or []) if isinstance(h, str)]
    hs = [h for h in hs if h and (not _is_probably_bad_heading(h))][:25]

    sn = []
    for s in (snippets or [])[:4]:
        s2 = " ".join((s or "").strip().split())
        if len(s2) > 420:
            s2 = s2[:420].rstrip() + "..."
        if s2:
            sn.append(s2)
    if not sn:
        return 0.0, ""

    cache_key = hashlib.sha1(("\n".join(sn) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    v0 = _cache_get("rerank", cache_key)
    if isinstance(v0, dict):
        try:
            return float(v0.get("score", 0.0) or 0.0), str(v0.get("why", "") or "")
        except Exception:
            return 0.0, ""

    ds = DeepSeekChat(settings)
    en = _has_latin(q) and (not _has_cjk(q))
    if en:
        sys = (
            "You are a strict academic retriever reranker.\n"
            "Output JSON ONLY: {\"score\":number,\"why\":string}.\n"
            "Rules:\n"
            "- score is 0..100 and MUST reflect how directly the provided snippets answer the question.\n"
            "- Penalize false friends (e.g., single-shot vs single-pixel) when mismatched.\n"
            "- Use only the snippets/headings; do NOT use filenames.\n"
            "- why: <= 18 words.\n"
        )
    else:
        sys = (
            "浣犳槸涓ユ牸鐨勫鏈绱㈤噸鎺掑櫒銆俓n"
            "鍙兘杈撳嚭 JSON锛歿\"score\":number,\"why\":string}銆俓n"
            "瑙勫垯锛歕n"
            "- score 涓?0..100锛屽繀椤诲弽鏄犫€滆繖浜涚墖娈垫槸鍚︾洿鎺ュ洖绛旂敤鎴烽棶棰樷€濄€俓n"
            "- 閬囧埌鏈鍋囨湅鍙嬭鎵ｅ垎锛堝 single-shot vs single-pixel锛夈€俓n"
            "- 鍙兘鏍规嵁 snippets/headings 鍒ゆ柇锛屼笉鑳芥牴鎹枃浠跺悕鍒ゆ柇銆俓n"
            "- why锛?= 18 涓瓧锛屽啓娓呮涓轰粈涔堛€俓n"
        )

    user = (
        f"Question: {q}\n\n"
        "Available headings:\n- " + "\n- ".join(hs) + "\n\n"
        "Snippets:\n- " + "\n- ".join(sn) + "\n"
    )
    try:
        out = (ds.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=160) or "").strip()
    except Exception:
        out = ""

    if out.startswith("```"):
        out = out.strip().strip("`")
        out = out.replace("json", "", 1).strip()

    try:
        data = json.loads(out)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return 0.0, ""

    try:
        score = float(data.get("score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    score = max(0.0, min(100.0, score))
    why = str(data.get("why") or "").strip()
    _cache_set("rerank", cache_key, {"score": score, "why": why}, max_items=600)
    return score, why

def _search_hits_with_fallback(
    prompt_text: str,
    retriever: BM25Retriever,
    top_k: int,
    settings,
    *,
    allow_translate: bool = True,
) -> tuple[list[dict], list[float], str, bool]:
    """
    Returns: (hits_raw, scores, used_query, used_translation)
    """
    q1 = (prompt_text or "").strip()
    hits1 = retriever.search(q1, top_k=max(10, top_k * 6)) if q1 else []
    hits1 = [h for h in (hits1 or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
    scores1 = [float(h.get("score", 0.0) or 0.0) for h in hits1]
    best1 = float(max(scores1) if scores1 else 0.0)

    # If the query is CJK-only, BM25 over English corpora can return all-zeros (but still returns arbitrary docs).
    # Try translating to English to get meaningful retrieval.
    used_trans = False
    q2 = _translate_query_for_search(settings, q1) if bool(allow_translate) else None
    if q2:
        hits2 = retriever.search(q2, top_k=max(10, top_k * 6))
        hits2 = [h for h in (hits2 or []) if not _is_temp_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
        scores2 = [float(h.get("score", 0.0) or 0.0) for h in hits2]
        best2 = float(max(scores2) if scores2 else 0.0)
        # Prefer translated retrieval when it yields a more meaningful signal.
        if (not hits1 and hits2) or (best1 <= 0.0 and best2 > 0.0) or (best2 > (best1 * 1.25 + 0.10)):
            return hits2, scores2, q2, True

    return hits1, scores1, q1, used_trans

def _group_hits_by_doc_for_refs(
    hits_raw: list[dict],
    prompt_text: str,
    top_k_docs: int,
    *,
    deep_query: str = "",
    deep_read: bool = False,
    llm_rerank: bool = False,
    settings=None,
) -> list[dict]:
    """
    Merge hits from the same markdown doc into a single ref entry.
    """
    by_doc: dict[str, list[dict]] = {}
    for h in hits_raw or []:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        if (not src) or _is_temp_source_path(src):
            continue
        by_doc.setdefault(src, []).append(h)

    # Pre-sort docs by best lexical hit; later stages can override with deep-read/LLM semantics.
    doc_order: list[tuple[float, str]] = []
    for src, hs in by_doc.items():
        try:
            best_score = float(max(float(h.get("score", 0.0) or 0.0) for h in hs))
        except Exception:
            best_score = 0.0
        doc_order.append((best_score, src))
    doc_order.sort(key=lambda x: x[0], reverse=True)

    docs: list[dict] = []
    profile = _query_term_profile(prompt_text, deep_query or "")
    # Bound work: only consider a limited number of candidate docs.
    max_docs_consider = max(int(top_k_docs) * 2, 12)
    # Quality-first refs: if deep_read is enabled, expand more candidate docs than before.
    deep_expand_docs = min(max_docs_consider, max(int(top_k_docs) * 2, 6)) if deep_read else 0
    for _best, src in doc_order[:max_docs_consider]:
        hs = by_doc.get(src) or []
        hs2 = sorted(hs, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        best_score = float(hs2[0].get("score", 0.0) or 0.0) if hs2 else 0.0
        # Candidate headings: (score, top_heading)
        cand: list[tuple[float, str]] = []
        snippets: list[str] = []
        locs_full: list[dict] = []
        for h in hs2[:6]:
            meta = h.get("meta", {}) or {}
            sc_h = float(h.get("score", 0.0) or 0.0)
            top = (meta.get("top_heading") or _top_heading(meta.get("heading_path", "")) or "").strip()
            if top:
                cand.append((sc_h, top))
                if not _is_probably_bad_heading(top):
                    hp = _normalize_heading_path_for_display(str(meta.get("heading_path") or ""))
                    p0, p1 = _page_range_from_meta(meta)
                    locs_full.append(
                        {
                            "heading_path": hp or top,
                            "heading": _normalize_heading(top) or top,
                            "score": sc_h,
                            "page_start": p0,
                            "page_end": p1,
                            "source": "hit",
                        }
                    )
            t = (h.get("text") or "").strip()
            if t:
                if t not in snippets:
                    snippets.append(t)

        # Optional deep-read for better section targeting + aspects + ranking.
        deep_best = 0.0
        if deep_read and deep_query and (len(docs) < deep_expand_docs):
            try:
                extra = _deep_read_md_for_context(Path(src), deep_query, max_snippets=3, snippet_chars=1600)
            except Exception:
                extra = []
            for ex in extra or []:
                meta_ex = ex.get("meta", {}) or {}
                sc_ex = float(ex.get("score", 0.0) or 0.0)
                hp2_raw = str(meta_ex.get("heading_path", "") or "").strip()
                hp2 = _normalize_heading_path_for_display(hp2_raw)
                top2 = _top_heading(hp2 or hp2_raw)
                if top2:
                    cand.append((sc_ex + 0.2, top2))
                    if not _is_probably_bad_heading(top2):
                        p0, p1 = _page_range_from_meta(meta_ex)
                        locs_full.append(
                            {
                                "heading_path": hp2 or top2,
                                "heading": _normalize_heading(top2) or top2,
                                "score": sc_ex + 0.2,
                                "page_start": p0,
                                "page_end": p1,
                                "source": "deep",
                            }
                        )
                tx = (ex.get("text") or "").strip()
                if tx:
                    if tx not in snippets:
                        snippets.append(tx)
                try:
                    deep_best = max(deep_best, float(ex.get("score", 0.0) or 0.0))
                except Exception:
                    pass

        # Final heading MUST be a real heading from this doc.
        # Prefer headings already attached to hits (grounded), but also read real md headings for navigation.
        best_heading = _pick_best_heading_for_doc(cand, prompt_text)
        headings_for_pack: list[str] = []
        try:
            prefer = _preferred_section_keys(prompt_text)
            picked = _pick_heading_from_md(Path(src), deep_query or prompt_text, prefer=prefer)
            if picked:
                best_heading = picked
        except Exception:
            pass
        try:
            headings_for_pack = _extract_md_headings(Path(src), max_n=40)
        except Exception:
            headings_for_pack = []
        if not headings_for_pack:
            # Minimal heading set: only what we have seen from hits/deep-read.
            seen_h2: list[str] = []
            for _sc2, hh in sorted(cand, key=lambda x: x[0], reverse=True):
                hh2 = _normalize_heading(hh)
                if not hh2 or _is_probably_bad_heading(hh2) or hh2 in seen_h2:
                    continue
                seen_h2.append(hh2)
                if len(seen_h2) >= 22:
                    break
            headings_for_pack = seen_h2
        aspects = _aspects_from_snippets(snippets[:3], prompt_text)

        # Build display snippets: pick the most relevant, non-noise snippets.
        q_for_pick = (deep_query or prompt_text or "").strip()
        q_tokens = [t for t in tokenize(q_for_pick) if len(t) >= 3]
        scored_snips: list[tuple[float, str]] = []
        for s in snippets:
            s2 = (s or "").strip()
            if not s2:
                continue
            if _is_noise_snippet_text(s2):
                continue
            try:
                sc = _score_tokens(s2, q_tokens) if q_tokens else 0.0
            except Exception:
                sc = 0.0
            # Prefer snippets that literally contain key phrases for single-shot/single-pixel disambiguation.
            low = _norm_text_for_match(s2)
            if profile.get("wants_single_shot") and any(k in low for k in ["single-shot", "single shot", "single exposure", "snapshot"]):
                sc += 3.0
            if profile.get("wants_single_shot") and any(k in low for k in ["single-pixel", "single pixel"]):
                sc -= 3.0
            scored_snips.append((float(sc), s2))
        scored_snips.sort(key=lambda x: x[0], reverse=True)
        show_snips = [_clean_snippet_for_display(s, max_chars=900) for _, s in scored_snips[:2]]

        # Best location candidates (prefer deep-read heading_path with subsection detail)
        locs_full.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        locs2 = []
        seen_h = set()
        for loc in locs_full:
            hh_path = _normalize_heading_path_for_display(str(loc.get("heading_path") or ""))
            if not hh_path:
                hh_path = _normalize_heading(str(loc.get("heading") or ""))
            if not hh_path:
                continue
            top_h, sub_h = _split_heading_path_levels(hh_path)
            hh_key = hh_path.lower()
            if not top_h or _is_probably_bad_heading(top_h) or hh_key in seen_h:
                continue
            seen_h.add(hh_key)
            ent = {
                "heading": top_h,
                "heading_path": hh_path,
                "score": float(loc.get("score", 0.0) or 0.0),
                "source": str(loc.get("source") or ""),
            }
            if sub_h:
                ent["subsection"] = sub_h
            try:
                p0 = int(loc.get("page_start")) if loc.get("page_start") is not None else None
            except Exception:
                p0 = None
            try:
                p1 = int(loc.get("page_end")) if loc.get("page_end") is not None else None
            except Exception:
                p1 = None
            if p0 is not None and p0 > 0:
                ent["page_start"] = p0
            if p1 is not None and p1 > 0:
                ent["page_end"] = p1
            locs2.append(ent)
            if len(locs2) >= 3:
                break

        # Heuristic base score: BM25 + small deep-read signal + term mismatch penalties.
        doc_name = Path(src).name
        term_bonus = _doc_term_bonus(profile, doc_name, snippets[:3])
        deep_scaled = 1.6 * (deep_best ** 0.6) if deep_best > 0 else 0.0
        combined = (0.75 * best_score) + (0.25 * deep_scaled) + term_bonus

        meta_out = {"source_path": src}
        if best_heading:
            meta_out["top_heading"] = best_heading
        meta_out["ref_aspects"] = aspects
        meta_out["ref_snippets"] = snippets[:3]
        meta_out["ref_show_snippets"] = show_snips[:3]
        meta_out["ref_locs"] = locs2
        meta_out["ref_headings"] = headings_for_pack
        if locs2:
            loc0 = locs2[0]
            hp0 = str(loc0.get("heading_path") or "").strip()
            sec0, sub0 = _split_heading_path_levels(hp0 or str(loc0.get("heading") or ""))
            if hp0:
                meta_out["ref_best_heading_path"] = hp0
            if sec0:
                meta_out["ref_section"] = sec0
            if sub0:
                meta_out["ref_subsection"] = sub0
            if loc0.get("page_start") is not None:
                meta_out["page_start"] = int(loc0.get("page_start"))
            if loc0.get("page_end") is not None:
                meta_out["page_end"] = int(loc0.get("page_end"))
        meta_out["ref_rank"] = {"bm25": best_score, "deep": deep_best, "term_bonus": term_bonus, "llm": 0.0, "why": "", "score": combined}

        docs.append(
            {
                "score": float(combined),
                "id": f"doc:{hashlib.sha1(src.encode('utf-8','ignore')).hexdigest()[:12]}",
                "text": snippets[0] if snippets else "",
                "meta": meta_out,
            }
        )

    # Optional LLM pack: one-shot semantic rerank + strong directional one-liner pieces (grounded on snippets/headings).
    if llm_rerank and settings and docs:
        pack = _llm_refs_pack(settings, question=(prompt_text or deep_query or ""), docs=docs)
        if isinstance(pack, dict) and pack:
            for i, d in enumerate(docs, start=1):
                meta = d.get("meta", {}) or {}
                pr = pack.get(i) or {}
                if not isinstance(pr, dict):
                    continue
                try:
                    llm_score = float(pr.get("score", 0.0) or 0.0)
                except Exception:
                    llm_score = 0.0
                llm_score = max(0.0, min(100.0, llm_score))
                llm_why = str(pr.get("why") or "").strip()

                meta["ref_pack"] = {
                    "score": llm_score,
                    "why": llm_why,
                    "what": str(pr.get("what") or "").strip(),
                    "start": str(pr.get("start") or "").strip(),
                    "gain": str(pr.get("gain") or "").strip(),
                    "find": [str(x).strip() for x in (pr.get("find") or []) if str(x).strip()][:4] if isinstance(pr.get("find"), list) else [],
                    "section": str(pr.get("section") or "").strip(),
                }

                # Use the packed section as the final "top_heading" when present (it is forced to be a real heading).
                sec = str(pr.get("section") or "").strip()
                if sec and (not _is_probably_bad_heading(sec)):
                    meta["top_heading"] = sec
                    meta["ref_section"] = sec
                    # If we already have a detailed path under this section, keep it; else fall back to section only.
                    locs_meta = meta.get("ref_locs") or []
                    if isinstance(locs_meta, list):
                        for loc in locs_meta:
                            if not isinstance(loc, dict):
                                continue
                            hp = str(loc.get("heading_path") or "").strip()
                            top_h, sub_h = _split_heading_path_levels(hp or str(loc.get("heading") or ""))
                            if top_h and (top_h.lower() == sec.lower()):
                                if hp:
                                    meta["ref_best_heading_path"] = hp
                                if sub_h:
                                    meta["ref_subsection"] = sub_h
                                break

                # Recompute combined score with semantic signal.
                r = meta.get("ref_rank") or {}
                try:
                    bm25 = float(r.get("bm25", 0.0) or 0.0)
                except Exception:
                    bm25 = 0.0
                try:
                    deep_best = float(r.get("deep", 0.0) or 0.0)
                except Exception:
                    deep_best = 0.0
                try:
                    term_bonus = float(r.get("term_bonus", 0.0) or 0.0)
                except Exception:
                    term_bonus = 0.0
                deep_scaled = 1.6 * (deep_best ** 0.6) if deep_best > 0 else 0.0
                combined2 = (llm_score / 10.0) + (0.25 * deep_scaled) + (0.10 * bm25) + (0.50 * term_bonus)
                d["score"] = float(combined2)
                meta["ref_rank"] = {"bm25": bm25, "deep": deep_best, "term_bonus": term_bonus, "llm": llm_score, "why": llm_why, "score": combined2}
                d["meta"] = meta

    docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

    # Precision-first filtering: when semantic rerank is available, drop weakly related docs
    # instead of filling the list with lexical look-alikes.
    if llm_rerank and docs:
        llm_scores: list[float] = []
        for d in docs:
            meta = d.get("meta", {}) or {}
            rank = meta.get("ref_rank") or {}
            try:
                llm_sc = float(rank.get("llm", 0.0) or 0.0)
            except Exception:
                llm_sc = 0.0
            if llm_sc > 0:
                llm_scores.append(llm_sc)
        if llm_scores:
            best_llm = max(llm_scores)
            sem_keep_min = max(28.0, best_llm - 35.0)
            filtered: list[dict] = []
            for d in docs:
                meta = d.get("meta", {}) or {}
                rank = meta.get("ref_rank") or {}
                try:
                    llm_sc = float(rank.get("llm", 0.0) or 0.0)
                except Exception:
                    llm_sc = 0.0
                if llm_sc >= sem_keep_min:
                    filtered.append(d)
            if filtered:
                docs = filtered

    return docs[: max(1, int(top_k_docs))]

def _group_hits_by_doc_for_refs_fast(hits_raw: list[dict], top_k_docs: int) -> list[dict]:
    """
    Fast fallback for background QA worker:
    - no full-md deep read
    - no LLM rerank
    - no heading extraction from file
    """
    by_doc: dict[str, dict] = {}
    for h in hits_raw or []:
        meta = h.get("meta", {}) or {}
        src = (meta.get("source_path") or "").strip()
        if (not src) or _is_temp_source_path(src):
            continue
        cur = by_doc.get(src)
        sc = float(h.get("score", 0.0) or 0.0)
        top = (meta.get("top_heading") or _top_heading(meta.get("heading_path", "")) or "").strip()
        txt = (h.get("text") or "").strip()
        if (cur is None) or (sc > float(cur.get("score", 0.0) or 0.0)):
            by_doc[src] = {
                "score": sc,
                "id": f"doc:{hashlib.sha1(src.encode('utf-8','ignore')).hexdigest()[:12]}",
                "text": txt,
                "meta": {
                    "source_path": src,
                    "top_heading": ("" if _is_probably_bad_heading(top) else top),
                    "ref_snippets": [txt] if txt else [],
                    "ref_show_snippets": [_clean_snippet_for_display(txt, max_chars=900)] if txt else [],
                    "ref_locs": ([{"heading": top, "score": sc}] if top and (not _is_probably_bad_heading(top)) else []),
                    "ref_headings": ([top] if top and (not _is_probably_bad_heading(top)) else []),
                    "ref_aspects": [],
                    "ref_rank": {"bm25": sc, "deep": 0.0, "term_bonus": 0.0, "llm": 0.0, "why": "", "score": sc},
                },
            }
        elif txt:
            m = by_doc[src].get("meta") or {}
            arr = list(m.get("ref_snippets") or [])
            if txt not in arr and len(arr) < 2:
                arr.append(txt)
                m["ref_snippets"] = arr
                m["ref_show_snippets"] = [_clean_snippet_for_display(x, max_chars=900) for x in arr]
                by_doc[src]["meta"] = m
    docs = list(by_doc.values())
    docs.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return docs[: max(1, int(top_k_docs))]

def _read_text_cached(path: Path) -> str:
    p = Path(path)
    try:
        mtime = float(p.stat().st_mtime)
    except Exception:
        mtime = 0.0
    key = f"{str(p)}|{mtime}"
    v0 = _cache_get("file_text", key)
    if isinstance(v0, str):
        return v0
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""
    _cache_set("file_text", key, text, max_items=220)
    return text

def _extract_md_headings(md_path: Path, *, max_n: int = 80) -> list[str]:
    """
    Extract real headings from the markdown file (ground truth for navigation).
    Returns plain heading titles (without leading #), preserving numbering if present.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    text = _read_text_cached(md_path)
    if not text:
        return []
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        level = len(s) - len(s.lstrip("#"))
        if level <= 0 or level > 4:
            continue
        title = s[level:].strip()
        title = _normalize_heading(title)
        if not title or _is_probably_bad_heading(title):
            continue
        if title not in out:
            out.append(title)
        if len(out) >= max_n:
            break
    return out

def _pick_heading_from_md(md_path: Path, query: str, *, prefer: list[str]) -> str:
    """
    Pick a heading that most likely contains the answer, from real md headings only.
    """
    hs = _extract_md_headings(md_path)
    if not hs:
        return ""
    q = (query or "").strip()
    if not q:
        return hs[0]
    q_toks = [t for t in tokenize(q) if len(t) >= 3]

    def score(h: str) -> float:
        low = h.lower()
        # Keyword overlap
        base = 0.0
        if q_toks:
            ht = tokenize(h)
            ct = Counter(ht)
            base += float(sum(ct.get(t, 0) for t in q_toks))
        # Preference boost (method/results/intro etc)
        bonus = 0.0
        for i, k in enumerate(prefer):
            if k in low:
                bonus += 3.0 - i * 0.25
                break
        # Slightly prefer medium-length headings
        bonus += max(0.0, (50 - abs(len(h) - 38)) / 200.0)
        return base + bonus

    best = max(hs, key=score)
    # Avoid pointing to REFERENCES unless asked.
    wants_refs = bool(re.search(r"(鍙傝€冩枃鐚畖寮曠敤|cite|citation|reference)", q, flags=re.I))
    if ("references" in best.lower() or "bibliography" in best.lower()) and not wants_refs:
        for h in hs:
            if ("references" in h.lower() or "bibliography" in h.lower()):
                continue
            if not _is_probably_bad_heading(h):
                return h
        return ""
    return best

def _deep_read_md_for_context(md_path: Path, query: str, *, max_snippets: int = 3, snippet_chars: int = 1400) -> list[dict]:
    """
    Read the full .md and extract the most relevant snippets (by token overlap),
    then return in the same dict shape as retriever hits.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        return []
    text = _read_text_cached(md_path)
    if not text.strip():
        return []

    q_tokens = [t for t in tokenize(query or "") if len(t) >= 3]
    if not q_tokens:
        return []

    # Cache per (file mtime, query) to avoid repeated full-doc scans across reruns / background tasks.
    try:
        mtime = float(md_path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    cache_key = hashlib.sha1((str(md_path) + "|" + str(mtime) + "|" + (query or "")).encode("utf-8", "ignore")).hexdigest()[:16]
    v0 = _cache_get("deep_read", cache_key)
    if isinstance(v0, list):
        try:
            return list(v0)
        except Exception:
            return []

    chunks = chunk_markdown(text, source_path=str(md_path), chunk_size=900, overlap=0)
    scored: list[tuple[float, dict]] = []
    for c in chunks:
        body = (c.get("text") or "").strip()
        if len(body) < 80:
            continue
        s = _score_tokens(body, q_tokens)
        if s <= 0.0:
            continue
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for rank, (s, c) in enumerate(scored[: max(1, int(max_snippets))], start=1):
        meta = dict((c.get("meta") or {}))
        meta.setdefault("source_path", str(md_path))
        meta["deep_read"] = True
        body = (c.get("text") or "").strip()
        if len(body) > snippet_chars:
            body = body[:snippet_chars].rstrip() + "..."
        out.append({"score": float(s), "id": f"deep:{hashlib.sha1((str(md_path)+'|'+str(rank)).encode('utf-8','ignore')).hexdigest()[:12]}", "text": body, "meta": meta})
    _cache_set("deep_read", cache_key, out, max_items=320)
    return out

def _llm_refs_pack(settings, *, question: str, docs: list[dict]) -> dict[int, dict]:
    """
    One-shot LLM pack for refs:
    - semantic relevance score (0..100) for reranking
    - concise summary + relevance + reading-start + expected gain (grounded on snippets/headings)

    Returns: {idx -> {"score":float, "why":str, "what":str, "start":str, "gain":str, "find":[str], "section":str}}
    """
    if not settings or (not getattr(settings, "api_key", None)):
        return {}
    q = (question or "").strip()
    if not q or not docs:
        return {}

    items = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("meta", {}) or {}
        headings = [h for h in (meta.get("ref_headings") or []) if isinstance(h, str)]
        headings = [h for h in headings if h and (not _is_probably_bad_heading(h))][:20]
        locs_payload: list[dict] = []
        raw_locs = meta.get("ref_locs")
        if isinstance(raw_locs, list):
            for loc in raw_locs[:3]:
                if not isinstance(loc, dict):
                    continue
                hp = str(loc.get("heading_path") or loc.get("heading") or "").strip()
                if not hp:
                    continue
                rec = {"heading_path": hp}
                try:
                    p0 = int(loc.get("page_start")) if loc.get("page_start") is not None else None
                except Exception:
                    p0 = None
                try:
                    p1 = int(loc.get("page_end")) if loc.get("page_end") is not None else None
                except Exception:
                    p1 = None
                if p0 is not None and p0 > 0:
                    rec["page_start"] = p0
                if p1 is not None and p1 > 0:
                    rec["page_end"] = p1
                locs_payload.append(rec)
        snippets = []
        for s in (meta.get("ref_show_snippets") or [])[:2]:
            s2 = " ".join(str(s).strip().split())
            if len(s2) > 360:
                s2 = s2[:360].rstrip() + "..."
            if s2:
                snippets.append(s2)
        if not snippets:
            # fallback to existing text field
            s = " ".join((d.get("text") or "").strip().split())
            if len(s) > 360:
                s = s[:360].rstrip() + "..."
            if s:
                snippets.append(s)
        items.append({"i": i, "headings": headings, "locs": locs_payload, "snippets": snippets})

    try:
        # Include mtimes so updates invalidate cache.
        sig_parts = ["refs_pack_v2", q]
        for d in docs:
            src = str((d.get("meta", {}) or {}).get("source_path") or "")
            try:
                mtime = float(Path(src).stat().st_mtime) if src else 0.0
            except Exception:
                mtime = 0.0
            sig_parts.append(src + "|" + str(mtime))
        sig = "|".join(sig_parts)
        cache_key = hashlib.sha1(sig.encode("utf-8", "ignore")).hexdigest()[:16]
    except Exception:
        cache_key = hashlib.sha1(q.encode("utf-8", "ignore")).hexdigest()[:16]

    v0 = _cache_get("refs_pack", cache_key)
    if isinstance(v0, dict):
        return v0

    ds = DeepSeekChat(settings)
    en = _has_latin(q) and (not _has_cjk(q))
    # Use an English instruction body to avoid encoding issues in long-lived terminals,
    # but require the model to match the user's language.
    _ = en  # language heuristic kept for potential future tuning
    sys = (
        "You are a strict academic retriever reranker and reading guide generator.\n"
        "Output JSON ONLY: "
        "{\"items\":[{\"i\":int,\"score\":number,\"why\":string,\"what\":string,\"start\":string,\"gain\":string,\"find\":[string],\"section\":string}]}.\n"
        "Rules:\n"
        "- score: 0..100, based on how directly snippets answer the question.\n"
        "- Penalize false-friend term mismatch (e.g., single-shot vs single-pixel).\n"
        "- Use ONLY snippets/headings; DO NOT use filenames.\n"
        "- section MUST be chosen from provided headings; otherwise empty string.\n"
        "- Prefer using candidate locs.heading_path when writing the start field.\n"
        "- Match the user's language (Chinese question -> Chinese output).\n"
        "- what: one-sentence summary of what this paper contributes for this question (specific, not generic).\n"
        "- why: why this paper is strongly relevant to the current question (point to evidence in snippets).\n"
        "- start: where to start reading (section/subsection + what to look for first).\n"
        "- gain: what the user can extract from this paper that helps answer the question.\n"
        "- find: 2-4 concrete items to look for (methods, settings, formulas, results, ablations, etc.).\n"
        "- Keep each field concise but informative. Avoid boilerplate and avoid repeating the same words across fields.\n"
        "- If evidence is weak or only partial, reduce score and state the limitation in why/gain.\n"
    )

    # Keep payload small
    payload = {"question": q, "docs": items}
    user = json.dumps(payload, ensure_ascii=False)
    try:
        out = (ds.chat(messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.0, max_tokens=900) or "").strip()
    except Exception:
        out = ""
    if out.startswith("```"):
        out = out.strip().strip("`")
        out = out.replace("json", "", 1).strip()

    try:
        data = json.loads(out)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {}
    arr = data.get("items") or []
    if not isinstance(arr, list):
        return {}

    result: dict[int, dict] = {}
    for it in arr:
        if not isinstance(it, dict):
            continue
        try:
            i = int(it.get("i"))
        except Exception:
            continue
        try:
            sc = float(it.get("score", 0.0) or 0.0)
        except Exception:
            sc = 0.0
        sc = max(0.0, min(100.0, sc))
        result[i] = {
            "score": sc,
            "why": str(it.get("why") or "").strip(),
            "what": str(it.get("what") or "").strip(),
            "start": str(it.get("start") or "").strip(),
            "gain": str(it.get("gain") or "").strip(),
            "find": [str(x).strip() for x in (it.get("find") or []) if str(x).strip()][:4] if isinstance(it.get("find"), list) else [],
            "section": str(it.get("section") or "").strip(),
        }

    _cache_set("refs_pack", cache_key, result, max_items=260)
    return result
