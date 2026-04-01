from __future__ import annotations

import re
from pathlib import Path

from kb.paper_guide_contracts import (
    _normalize_paper_guide_support_resolution,
    _normalize_paper_guide_support_slot,
)


def _source_refs_from_index(
    index_data: dict,
    source_path: str,
    *,
    source_sha1: str = "",
    norm_source_key_local,
) -> dict[int, dict]:
    if not isinstance(index_data, dict):
        return {}
    docs = index_data.get("docs")
    if not isinstance(docs, dict) or not docs:
        return {}
    src = str(source_path or "").strip()
    if not src:
        return {}

    target_norm = norm_source_key_local(src)
    target_sha = str(source_sha1 or "").strip().lower()
    want_name = Path(src).name.lower()
    want_stem = Path(src).stem.lower()

    doc = docs.get(target_norm) if target_norm else None
    if not isinstance(doc, dict):
        for cand in docs.values():
            if not isinstance(cand, dict):
                continue
            cand_sha = str(cand.get("sha1") or "").strip().lower()
            if target_sha and cand_sha and cand_sha == target_sha:
                doc = cand
                break
        if not isinstance(doc, dict):
            for cand in docs.values():
                if not isinstance(cand, dict):
                    continue
                cand_norm = norm_source_key_local(str(cand.get("path") or ""))
                if cand_norm and target_norm and cand_norm == target_norm:
                    doc = cand
                    break
        if not isinstance(doc, dict):
            for cand in docs.values():
                if not isinstance(cand, dict):
                    continue
                cand_name = str(cand.get("name") or "").strip().lower()
                cand_stem = str(cand.get("stem") or "").strip().lower()
                if want_name and cand_name and want_name == cand_name:
                    doc = cand
                    break
                if want_stem and cand_stem and want_stem == cand_stem:
                    doc = cand
                    break

    if not isinstance(doc, dict):
        return {}
    refs = doc.get("refs")
    if not isinstance(refs, dict):
        return {}
    out: dict[int, dict] = {}
    for k, v in refs.items():
        if not isinstance(v, dict):
            continue
        try:
            n = int(k)
        except Exception:
            continue
        if n <= 0:
            continue
        out[n] = v
    return out


def _validate_structured_citations(
    answer: str,
    *,
    answer_hits: list[dict],
    db_dir: Path | None,
    locked_source: dict | None = None,
    paper_guide_mode: bool = False,
    paper_guide_candidate_refs_by_source: dict[str, list[int]] | None = None,
    paper_guide_support_slots: list[dict] | None = None,
    paper_guide_support_resolution: list[dict] | None = None,
    sanitize_structured_cite_tokens,
    cite_canon_re: re.Pattern[str],
    cite_source_id,
    hit_source_path,
    load_reference_index,
    resolve_reference_entry,
    source_refs_from_index,
    extract_candidate_ref_nums_from_hits,
    extract_citation_context_hints,
    has_explicit_reference_conflict,
    select_support_slot_for_context,
    reference_alignment_score,
) -> tuple[str, dict]:
    text = str(answer or "")
    if ("[[CITE:" not in text) and ("[CITE:" not in text):
        return text, {
            "raw_count": 0,
            "kept": 0,
            "rewritten": 0,
            "dropped": 0,
            "locked_sid": str((locked_source or {}).get("sid") or ""),
        }

    cleaned = sanitize_structured_cite_tokens(text)
    raw_tokens = list(cite_canon_re.finditer(cleaned))
    if not raw_tokens:
        return cleaned, {
            "raw_count": 0,
            "kept": 0,
            "rewritten": 0,
            "dropped": 0,
            "locked_sid": str((locked_source or {}).get("sid") or ""),
        }

    sid_to_source: dict[str, str] = {}
    sha_by_source: dict[str, str] = {}
    for hit in answer_hits or []:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if not src:
            continue
        sid = cite_source_id(src).lower()
        sid_to_source[sid] = src
        sha1 = str(meta.get("source_sha1") or "").strip().lower()
        if sha1 and (src not in sha_by_source):
            sha_by_source[src] = sha1

    try:
        index_data = load_reference_index(Path(db_dir).expanduser()) if db_dir else {}
    except Exception:
        index_data = {}

    locked_sid = str((locked_source or {}).get("sid") or "").strip().lower()
    locked_source_path = str((locked_source or {}).get("source_path") or "").strip()
    locked_source_sha1 = str((locked_source or {}).get("source_sha1") or "").strip().lower()
    if locked_sid and locked_source_path and (locked_sid not in sid_to_source):
        sid_to_source[locked_sid] = locked_source_path
        if locked_source_sha1 and (locked_source_path not in sha_by_source):
            sha_by_source[locked_source_path] = locked_source_sha1

    candidate_ref_nums_by_source: dict[str, list[int]] = {}
    if paper_guide_mode:
        for src in list(dict.fromkeys([hit_source_path(h) for h in answer_hits or []])):
            src_norm = str(src or "").strip()
            if not src_norm:
                continue
            nums = extract_candidate_ref_nums_from_hits(
                answer_hits,
                source_path=src_norm,
                max_candidates=48,
            )
            if nums:
                candidate_ref_nums_by_source[src_norm] = nums
        for src, nums in (paper_guide_candidate_refs_by_source or {}).items():
            src_norm = str(src or "").strip()
            if not src_norm:
                continue
            merged = list(candidate_ref_nums_by_source.get(src_norm) or [])
            seen = set(int(n) for n in merged if int(n) > 0)
            for item in list(nums or []):
                try:
                    n = int(item)
                except Exception:
                    continue
                if n <= 0 or n in seen:
                    continue
                seen.add(n)
                merged.append(n)
            if merged:
                candidate_ref_nums_by_source[src_norm] = merged

    support_slots_by_sid: dict[str, list[dict]] = {}
    if paper_guide_mode:
        for raw_slot in list(paper_guide_support_slots or []):
            if not isinstance(raw_slot, dict):
                continue
            slot = _normalize_paper_guide_support_slot(raw_slot)
            slot_src = str(slot.get("source_path") or "").strip()
            slot_sid = cite_source_id(slot_src).lower()
            if not slot_sid:
                continue
            support_slots_by_sid.setdefault(slot_sid, []).append(slot)
    support_resolution_by_line: dict[int, list[dict]] = {}
    support_locate_only_line_indexes: set[int] = set()
    if paper_guide_mode:
        for raw_rec in list(paper_guide_support_resolution or []):
            if not isinstance(raw_rec, dict):
                continue
            rec = _normalize_paper_guide_support_resolution(raw_rec)
            try:
                line_index_raw = rec.get("line_index")
                line_index = int(line_index_raw) if str(line_index_raw).strip() else -1
            except Exception:
                line_index = -1
            if line_index < 0:
                continue
            support_resolution_by_line.setdefault(line_index, []).append(rec)
            try:
                resolved_local = int(rec.get("resolved_ref_num") or 0)
            except Exception:
                resolved_local = 0
            if str(rec.get("cite_policy") or "").strip().lower() == "locate_only" and resolved_local <= 0:
                support_locate_only_line_indexes.add(int(line_index))

    resolved_ref_cache: dict[tuple[str, int], dict | None] = {}
    source_refs_cache: dict[str, dict[int, dict]] = {}

    def _resolves(sid: str, ref_num: int) -> bool:
        sp = sid_to_source.get(str(sid or "").strip().lower())
        if (not sp) or (int(ref_num) <= 0):
            return False
        try:
            got = resolve_reference_entry(
                index_data,
                sp,
                int(ref_num),
                source_sha1=sha_by_source.get(sp, ""),
            )
        except Exception:
            got = None
        return bool(isinstance(got, dict) and isinstance(got.get("ref"), dict))

    def _resolve_ref(sp: str, ref_num: int) -> dict | None:
        src = str(sp or "").strip()
        try:
            n = int(ref_num)
        except Exception:
            return None
        if (not src) or (n <= 0):
            return None
        key = (src, n)
        if key in resolved_ref_cache:
            return resolved_ref_cache[key]
        try:
            got = resolve_reference_entry(
                index_data,
                src,
                n,
                source_sha1=sha_by_source.get(src, ""),
            )
        except Exception:
            got = None
        ref = got.get("ref") if isinstance(got, dict) and isinstance(got.get("ref"), dict) else None
        resolved_ref_cache[key] = ref
        return ref

    def _source_refs(sp: str) -> dict[int, dict]:
        src = str(sp or "").strip()
        if not src:
            return {}
        cached = source_refs_cache.get(src)
        if isinstance(cached, dict):
            return cached
        refs = source_refs_from_index(index_data, src, source_sha1=sha_by_source.get(src, ""))
        source_refs_cache[src] = refs
        return refs

    stats = {
        "raw_count": int(len(raw_tokens)),
        "kept": 0,
        "rewritten": 0,
        "dropped": 0,
        "locked_sid": locked_sid,
    }

    def _citation_context_line(*, token_start: int, token_end: int) -> str:
        try:
            start = max(0, int(token_start))
            end = max(start, int(token_end))
        except Exception:
            return ""
        left = cleaned.rfind("\n", 0, start)
        left = 0 if left < 0 else left + 1
        right = cleaned.find("\n", end)
        if right < 0:
            right = len(cleaned)
        return str(cleaned[left:right] or "").strip()

    def _citation_line_index(*, token_start: int) -> int:
        try:
            start = max(0, int(token_start))
        except Exception:
            return -1
        return int(cleaned.count("\n", 0, start))

    def _pick_grounded_ref_num(*, source_path: str, current_ref_num: int, token_start: int, token_end: int) -> int | None:
        src = str(source_path or "").strip()
        if (not src) or int(current_ref_num) <= 0:
            return None
        if not paper_guide_mode:
            return int(current_ref_num) if isinstance(_resolve_ref(src, int(current_ref_num)), dict) else None

        current_ref = _resolve_ref(src, int(current_ref_num))
        candidate_nums = list(candidate_ref_nums_by_source.get(src) or [])
        context_line = _citation_context_line(token_start=token_start, token_end=token_end)
        line_index = _citation_line_index(token_start=token_start)
        hints = extract_citation_context_hints(cleaned, token_start=token_start, token_end=token_end)
        has_strong_hints = bool(
            str(hints.get("doi") or "").strip()
            or (str(hints.get("author") or "").strip() and str(hints.get("year") or "").strip())
        )
        current_conflict = bool(current_ref and has_explicit_reference_conflict(current_ref, hints))
        local_resolution: dict | None = None
        resolution_rows = [
            dict(item)
            for item in list(support_resolution_by_line.get(line_index, []) or [])
            if isinstance(item, dict)
        ]
        if resolution_rows:
            resolution_rows.sort(
                key=lambda item: (
                    1 if int(item.get("resolved_ref_num") or 0) > 0 else 0,
                    1 if str(item.get("block_id") or "").strip() else 0,
                    len(str(item.get("locate_anchor") or "").strip()),
                ),
                reverse=True,
            )
            local_resolution = resolution_rows[0]
        slot_sid = cite_source_id(src).lower()
        local_slot = select_support_slot_for_context(
            support_slots_by_sid.get(slot_sid, []),
            context_text=context_line,
        )
        local_candidate_nums: list[int] = []
        local_seen: set[int] = set()
        local_cite_policy = ""
        if isinstance(local_resolution, dict):
            local_cite_policy = str(local_resolution.get("cite_policy") or "").strip().lower()
            for n0 in list(local_resolution.get("candidate_refs") or []):
                try:
                    n = int(n0)
                except Exception:
                    continue
                if n <= 0 or n in local_seen:
                    continue
                local_seen.add(n)
                local_candidate_nums.append(n)
            for span in list(local_resolution.get("ref_spans") or []):
                if not isinstance(span, dict):
                    continue
                for n0 in list(span.get("nums") or []):
                    try:
                        n = int(n0)
                    except Exception:
                        continue
                    if n <= 0 or n in local_seen:
                        continue
                    local_seen.add(n)
                    local_candidate_nums.append(n)
            try:
                resolved_local = int(local_resolution.get("resolved_ref_num") or 0)
            except Exception:
                resolved_local = 0
            if resolved_local > 0 and resolved_local not in local_seen:
                local_seen.add(resolved_local)
                local_candidate_nums.insert(0, resolved_local)
            if local_cite_policy == "locate_only" and (not has_strong_hints):
                return None
        if isinstance(local_slot, dict):
            if not local_cite_policy:
                local_cite_policy = str(local_slot.get("cite_policy") or "").strip().lower()
            for n0 in list(local_slot.get("candidate_refs") or []):
                try:
                    n = int(n0)
                except Exception:
                    continue
                if n <= 0 or n in local_seen:
                    continue
                local_seen.add(n)
                local_candidate_nums.append(n)
            for span in list(local_slot.get("ref_spans") or []):
                if not isinstance(span, dict):
                    continue
                for n0 in list(span.get("nums") or []):
                    try:
                        n = int(n0)
                    except Exception:
                        continue
                    if n <= 0 or n in local_seen:
                        continue
                    local_seen.add(n)
                    local_candidate_nums.append(n)
        if local_candidate_nums:
            candidate_nums = list(local_candidate_nums)
        elif local_cite_policy == "locate_only" and (not has_strong_hints):
            return None

        if current_ref and (not candidate_nums) and has_strong_hints and (not current_conflict):
            return int(current_ref_num)
        if current_ref and candidate_nums and (int(current_ref_num) in candidate_nums) and (not current_conflict):
            return int(current_ref_num)

        pool: list[int] = []
        seen_pool: set[int] = set()
        for n0 in candidate_nums:
            try:
                n = int(n0)
            except Exception:
                continue
            if n <= 0 or n in seen_pool:
                continue
            seen_pool.add(n)
            pool.append(n)
        if int(current_ref_num) > 0 and int(current_ref_num) not in seen_pool:
            seen_pool.add(int(current_ref_num))
            pool.append(int(current_ref_num))
        if has_strong_hints:
            for n in _source_refs(src).keys():
                if n in seen_pool:
                    continue
                seen_pool.add(n)
                pool.append(n)

        best_num: int | None = None
        best_score = float("-inf")
        for n in pool:
            ref = _resolve_ref(src, n)
            if not isinstance(ref, dict):
                ref = _source_refs(src).get(int(n))
            if not isinstance(ref, dict):
                continue
            score = float(reference_alignment_score(ref, hints))
            if n == int(current_ref_num):
                score += 0.1
            if score > best_score:
                best_score = score
                best_num = int(n)

        if has_strong_hints:
            if best_num is None:
                return None
            if str(hints.get("doi") or "").strip():
                return best_num if best_score >= 6.0 else None
            return best_num if best_score >= 3.5 else None

        if not candidate_nums:
            return None

        if candidate_nums:
            if len(candidate_nums) == 1:
                only = int(candidate_nums[0])
                return only if isinstance(_resolve_ref(src, only), dict) or isinstance(_source_refs(src).get(only), dict) else None
            return int(current_ref_num) if (int(current_ref_num) in candidate_nums and isinstance(current_ref, dict)) else None

        return None

    def _repl(m: re.Match[str]) -> str:
        sid = str(m.group(1) or "").strip().lower()
        try:
            n = int(m.group(2) or 0)
        except Exception:
            n = 0
        line_index = _citation_line_index(token_start=int(m.start()))
        if int(line_index) in support_locate_only_line_indexes:
            stats["dropped"] = int(stats["dropped"]) + 1
            return ""
        if n <= 0:
            stats["dropped"] = int(stats["dropped"]) + 1
            return ""

        if locked_sid:
            grounded_n = _pick_grounded_ref_num(
                source_path=locked_source_path,
                current_ref_num=n,
                token_start=int(m.start()),
                token_end=int(m.end()),
            )
            if grounded_n is not None and _resolves(locked_sid, grounded_n):
                if sid == locked_sid and grounded_n == n:
                    stats["kept"] = int(stats["kept"]) + 1
                    return f"[[CITE:{locked_sid}:{grounded_n}]]"
                stats["rewritten"] = int(stats["rewritten"]) + 1
                return f"[[CITE:{locked_sid}:{grounded_n}]]"
            if sid:
                sid_source_path = sid_to_source.get(sid, "")
                sid_grounded_n = _pick_grounded_ref_num(
                    source_path=sid_source_path,
                    current_ref_num=n,
                    token_start=int(m.start()),
                    token_end=int(m.end()),
                )
            else:
                sid_grounded_n = None
            if sid and sid_grounded_n is not None and _resolves(sid, sid_grounded_n):
                stats["kept"] = int(stats["kept"]) + 1
                return f"[[CITE:{sid}:{sid_grounded_n}]]"
            stats["dropped"] = int(stats["dropped"]) + 1
            return ""

        if sid:
            source_path = sid_to_source.get(sid, "")
            grounded_n = _pick_grounded_ref_num(
                source_path=source_path,
                current_ref_num=n,
                token_start=int(m.start()),
                token_end=int(m.end()),
            )
        else:
            grounded_n = None
        if sid and grounded_n is not None and _resolves(sid, grounded_n):
            stats["kept"] = int(stats["kept"]) + 1
            return f"[[CITE:{sid}:{grounded_n}]]"
        stats["dropped"] = int(stats["dropped"]) + 1
        return ""

    out = cite_canon_re.sub(_repl, cleaned)
    return out, stats
