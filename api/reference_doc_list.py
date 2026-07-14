from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
import re

from api.reference_value_utils import _non_negative_float, _positive_int


def _doc_list_copy_key(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").strip().lower())


def _doc_list_copy_is_duplicate(left: object, right: object) -> bool:
    left_key = _doc_list_copy_key(left)
    right_key = _doc_list_copy_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    return len(shorter) >= 24 and shorter in longer and (len(shorter) / max(1, len(longer))) >= 0.72


def _dedupe_doc_list_card_copy(*, raw_item: dict, ui_meta: dict | None) -> dict:
    out = dict(ui_meta or {}) if isinstance(ui_meta, dict) else {}
    summary_line = str(out.get("summary_line") or "").strip()
    why_line = str(out.get("why_line") or "").strip()
    if not _doc_list_copy_is_duplicate(summary_line, why_line):
        return out

    authoritative_summary = str(raw_item.get("summary_line") or "").strip()
    if authoritative_summary and not _doc_list_copy_is_duplicate(authoritative_summary, why_line):
        out["summary_line"] = authoritative_summary
        out["summary_generation"] = str(raw_item.get("summary_generation") or "").strip() or "section_grounded"
        out["summary_basis"] = "authoritative document-list evidence"
        out["summary_source"] = "doc_list_authoritative_fast"
        return out

    out.pop("why_line", None)
    out.pop("why_generation", None)
    out.pop("why_basis", None)
    return out


def _collect_doc_list_ref_text_candidates(
    *,
    raw_item: dict,
    primary_evidence: dict,
    clean_refs_evidence_snippet: Callable[..., str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        text = clean_refs_evidence_snippet(
            str(value or "").strip(),
            prompt="",
            source_path=str(raw_item.get("source_path") or primary_evidence.get("source_path") or "").strip(),
            display_name=str(raw_item.get("source_name") or primary_evidence.get("source_name") or "").strip(),
            heading_path=str(raw_item.get("heading_path") or primary_evidence.get("heading_path") or "").strip(),
            max_len=460,
        )
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(text)

    _push(str(primary_evidence.get("highlight_snippet") or "").strip())
    _push(str(primary_evidence.get("snippet") or "").strip())
    _push(str(raw_item.get("summary_line") or "").strip())
    for alt in list(primary_evidence.get("alternatives") or []):
        if not isinstance(alt, dict):
            continue
        _push(str(alt.get("highlight_snippet") or "").strip())
        _push(str(alt.get("snippet") or "").strip())
    return out


def _primary_ref_evidence_summary_seed(
    primary_evidence: dict | None,
    *,
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    clean_refs_evidence_snippet: Callable[..., str],
) -> str:
    primary = normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return ""
    return clean_refs_evidence_snippet(
        str(primary.get("highlight_snippet") or primary.get("snippet") or "").strip(),
        prompt="",
        source_path=str(primary.get("source_path") or "").strip(),
        display_name=str(primary.get("source_name") or "").strip(),
        heading_path=str(primary.get("heading_path") or "").strip(),
        max_len=360,
    )


def _primary_ref_evidence_points_to_same_surface(
    left_primary: dict | None,
    right_primary: dict | None,
    *,
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    primary_ref_evidence_summary_seed: Callable[[dict | None], str],
    same_source_identity: Callable[[str, str], bool],
    ref_summary_surfaces_match: Callable[[str, str], bool],
) -> bool:
    left = normalize_primary_ref_evidence_payload(left_primary if isinstance(left_primary, dict) else {})
    right = normalize_primary_ref_evidence_payload(right_primary if isinstance(right_primary, dict) else {})
    if (not left) or (not right):
        return False

    left_source = str(left.get("source_path") or "").strip()
    right_source = str(right.get("source_path") or "").strip()
    if left_source and right_source and (not same_source_identity(left_source, right_source)):
        return False

    left_block = str(left.get("block_id") or "").strip()
    right_block = str(right.get("block_id") or "").strip()
    if left_block or right_block:
        return bool(left_block and right_block and left_block == right_block)

    left_anchor = str(left.get("anchor_id") or "").strip()
    right_anchor = str(right.get("anchor_id") or "").strip()
    if left_anchor or right_anchor:
        return bool(left_anchor and right_anchor and left_anchor == right_anchor)

    left_heading = str(left.get("heading_path") or "").strip()
    right_heading = str(right.get("heading_path") or "").strip()
    if left_heading and right_heading and left_heading != right_heading:
        return False

    left_summary = primary_ref_evidence_summary_seed(left)
    right_summary = primary_ref_evidence_summary_seed(right)
    if left_summary and right_summary:
        return ref_summary_surfaces_match(left_summary, right_summary)
    if left_heading and right_heading:
        return True
    return False


def _doc_list_authoritative_primary_is_upgradeable(
    primary_evidence: dict | None,
    *,
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
) -> bool:
    primary = normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return True
    if bool(primary.get("strict_locate")):
        return False
    if str(primary.get("block_id") or "").strip():
        return False
    if str(primary.get("anchor_id") or "").strip():
        return False
    reason = str(primary.get("selection_reason") or "").strip().lower()
    return reason in {"", "answer_hit_top", "pending_section_seed", "section_intent_rescue", "alternative_rescue"}


def _primary_ref_evidence_summary_is_usable(
    primary_evidence: dict | None,
    *,
    prompt: str,
    display_name: str,
    primary_ref_evidence_summary_seed: Callable[[dict | None], str],
    looks_bibliographic_source_block_text: Callable[[str], bool],
    summary_line_needs_polish: Callable[..., bool],
) -> bool:
    summary_seed = primary_ref_evidence_summary_seed(primary_evidence)
    return bool(
        summary_seed
        and (not looks_bibliographic_source_block_text(summary_seed))
        and (not summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=summary_seed,
        ))
    )


def _primary_ref_evidence_precision_score(
    *,
    primary_evidence: dict | None,
    prompt: str,
    display_name: str,
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    sanitize_heading_path: Callable[..., str],
    primary_ref_evidence_summary_seed: Callable[[dict | None], str],
    primary_ref_evidence_summary_is_usable: Callable[..., bool],
) -> tuple[int, int, int, int, int, int, int]:
    primary = normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return (0, 0, 0, 0, 0, 0, 0)
    reason = str(primary.get("selection_reason") or "").strip().lower()
    reason_rank = {
        "prompt_aligned_block": 8,
        "prompt_aligned": 7,
        "navigation": 6,
        "alternative_rescue": 5,
        "fallback": 4,
        "reader_open": 4,
        "strict_locate": 4,
        "shared_refs_pack": 4,
        "section_intent_rescue": 1,
        "answer_hit_top": 0,
        "pending_section_seed": 0,
    }.get(reason, 3 if reason else 0)
    heading_path = sanitize_heading_path(
        str(primary.get("heading_path") or "").strip(),
        prompt=prompt,
        source_path=str(primary.get("source_path") or "").strip(),
    )
    summary_seed = primary_ref_evidence_summary_seed(primary)
    summary_seed_usable = primary_ref_evidence_summary_is_usable(
        primary,
        prompt=prompt,
        display_name=display_name,
    )
    return (
        reason_rank,
        1 if bool(primary.get("strict_locate")) else 0,
        1 if str(primary.get("block_id") or "").strip() else 0,
        1 if str(primary.get("anchor_id") or "").strip() else 0,
        1 if heading_path else 0,
        1 if summary_seed_usable else 0,
        1 if summary_seed else 0,
    )


def _upgrade_primary_ref_evidence_from_alternatives(
    primary_evidence: dict | None,
    *,
    prompt: str,
    display_name: str,
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    primary_ref_evidence_summary_is_usable: Callable[..., bool],
    primary_ref_evidence_precision_score: Callable[..., tuple[int, int, int, int, int, int, int]],
) -> dict:
    primary = normalize_primary_ref_evidence_payload(primary_evidence if isinstance(primary_evidence, dict) else {})
    if not primary:
        return {}
    if primary_ref_evidence_summary_is_usable(primary, prompt=prompt, display_name=display_name):
        return primary
    alternatives = [item for item in list(primary.get("alternatives") or []) if isinstance(item, dict)]
    best: dict = {}
    best_score: tuple[int, int, int, int, int, int, int] | None = None
    for raw_alt in alternatives[:8]:
        alt_raw = {
            "source_path": primary.get("source_path"),
            "source_name": primary.get("source_name"),
            **raw_alt,
        }
        alt = normalize_primary_ref_evidence_payload(alt_raw)
        if not alt or not primary_ref_evidence_summary_is_usable(alt, prompt=prompt, display_name=display_name):
            continue
        score = primary_ref_evidence_precision_score(
            primary_evidence=alt,
            prompt=prompt,
            display_name=display_name,
        )
        if best_score is None or score > best_score:
            best = alt
            best_score = score
    if not best:
        return primary
    upgraded = dict(primary)
    for key, value in best.items():
        if value not in (None, "", [], {}):
            upgraded[key] = value
    upgraded["selection_reason"] = "alternative_rescue"
    upgraded["strict_locate"] = bool(best.get("strict_locate"))
    upgraded["alternatives"] = alternatives
    return upgraded


def _select_doc_list_effective_primary_evidence(
    *,
    prompt: str,
    display_name: str,
    authoritative_primary_evidence: dict | None,
    synthesized_primary_evidence: dict | None,
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    upgrade_primary_ref_evidence_from_alternatives: Callable[..., dict],
    primary_ref_evidence_points_to_same_surface: Callable[[dict | None, dict | None], bool],
    doc_list_authoritative_primary_is_upgradeable: Callable[[dict | None], bool],
    primary_ref_evidence_precision_score: Callable[..., tuple[int, int, int, int, int, int, int]],
) -> tuple[dict, str]:
    authoritative = normalize_primary_ref_evidence_payload(
        authoritative_primary_evidence if isinstance(authoritative_primary_evidence, dict) else {}
    )
    synthesized = normalize_primary_ref_evidence_payload(
        synthesized_primary_evidence if isinstance(synthesized_primary_evidence, dict) else {}
    )
    authoritative = upgrade_primary_ref_evidence_from_alternatives(
        authoritative,
        prompt=prompt,
        display_name=display_name,
    )
    synthesized = upgrade_primary_ref_evidence_from_alternatives(
        synthesized,
        prompt=prompt,
        display_name=display_name,
    )
    if not authoritative:
        return synthesized, "synthesized"
    if not synthesized:
        return authoritative, "authoritative"
    if primary_ref_evidence_points_to_same_surface(authoritative, synthesized):
        authoritative_score = primary_ref_evidence_precision_score(
            primary_evidence=authoritative,
            prompt=prompt,
            display_name=display_name,
        )
        synthesized_score = primary_ref_evidence_precision_score(
            primary_evidence=synthesized,
            prompt=prompt,
            display_name=display_name,
        )
        return (
            (synthesized, "synthesized")
            if synthesized_score > authoritative_score
            else (authoritative, "authoritative")
        )
    if not doc_list_authoritative_primary_is_upgradeable(authoritative):
        return authoritative, "authoritative"

    authoritative_score = primary_ref_evidence_precision_score(
        primary_evidence=authoritative,
        prompt=prompt,
        display_name=display_name,
    )
    synthesized_score = primary_ref_evidence_precision_score(
        primary_evidence=synthesized,
        prompt=prompt,
        display_name=display_name,
    )
    if synthesized_score > authoritative_score:
        return synthesized, "synthesized"
    if authoritative_score > synthesized_score:
        return authoritative, "authoritative"

    auth_reason = str(authoritative.get("selection_reason") or "").strip().lower()
    synth_reason = str(synthesized.get("selection_reason") or "").strip().lower()
    if bool(synthesized.get("strict_locate")) and (not bool(authoritative.get("strict_locate"))):
        return synthesized, "synthesized"
    if synth_reason in {"prompt_aligned_block", "prompt_aligned"} and auth_reason in {"", "answer_hit_top", "pending_section_seed"}:
        return synthesized, "synthesized"
    return authoritative, "authoritative"


def _apply_doc_list_effective_primary_evidence(
    *,
    prompt: str,
    display_name: str,
    fallback_heading_path: str,
    ui_meta: dict | None,
    authoritative_primary_evidence: dict | None,
    authoritative_summary_line: str = "",
    authoritative_summary_generation: str = "",
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    select_doc_list_effective_primary_evidence: Callable[..., tuple[dict, str]],
    primary_ref_evidence_summary_seed: Callable[[dict | None], str],
    compact_reader_open_text: Callable[..., str],
    summary_line_needs_polish: Callable[..., bool],
    primary_ref_evidence_points_to_same_surface: Callable[[dict | None, dict | None], bool],
    build_ref_summary_basis_meta: Callable[..., dict],
) -> tuple[dict, dict]:
    ui_out = dict(ui_meta or {}) if isinstance(ui_meta, dict) else {}
    synthesized_primary = normalize_primary_ref_evidence_payload(
        ui_out.get("primary_evidence") if isinstance(ui_out.get("primary_evidence"), dict) else {}
    )
    authoritative_primary = normalize_primary_ref_evidence_payload(
        authoritative_primary_evidence if isinstance(authoritative_primary_evidence, dict) else {}
    )
    effective_primary, selected_source = select_doc_list_effective_primary_evidence(
        prompt=prompt,
        display_name=display_name,
        authoritative_primary_evidence=authoritative_primary,
        synthesized_primary_evidence=synthesized_primary,
    )
    effective_heading_path = str(
        effective_primary.get("heading_path")
        or ui_out.get("heading_path")
        or fallback_heading_path
        or ""
    ).strip()
    if effective_heading_path and (
        (not str(ui_out.get("heading_path") or "").strip())
        or selected_source == "authoritative"
    ):
        ui_out["heading_path"] = effective_heading_path

    current_summary_generation = str(ui_out.get("summary_generation") or "").strip().lower()
    current_summary_is_llm = current_summary_generation in {"llm_grounded", "llm_pack"}
    effective_summary_seed = primary_ref_evidence_summary_seed(effective_primary)
    authoritative_summary_seed = compact_reader_open_text(str(authoritative_summary_line or "").strip())
    authoritative_summary_generation_norm = str(authoritative_summary_generation or "").strip().lower()
    authoritative_summary_is_llm = authoritative_summary_generation_norm in {"llm_grounded", "llm_pack"}
    if authoritative_summary_seed and (not authoritative_summary_is_llm) and summary_line_needs_polish(
        prompt=prompt,
        title=display_name,
        summary_line=authoritative_summary_seed,
    ):
        authoritative_summary_seed = ""
    if (not authoritative_summary_seed) and authoritative_primary:
        authoritative_summary_seed = primary_ref_evidence_summary_seed(authoritative_primary)
        if authoritative_summary_seed and summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=authoritative_summary_seed,
        ):
            authoritative_summary_seed = ""
    authoritative_conflicts_with_synthesized = bool(
        selected_source == "authoritative"
        and authoritative_primary
        and synthesized_primary
        and (not primary_ref_evidence_points_to_same_surface(authoritative_primary, synthesized_primary))
    )
    if authoritative_conflicts_with_synthesized and authoritative_summary_seed:
        ui_out["summary_line"] = authoritative_summary_seed
        if authoritative_summary_is_llm:
            summary_basis_meta = build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind="guide",
                summary_generation=authoritative_summary_generation_norm,
                summary_line=authoritative_summary_seed,
            )
            ui_out["summary_generation"] = str(
                summary_basis_meta.get("summary_generation") or authoritative_summary_generation_norm
            )
            ui_out["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    if effective_summary_seed and (
        (not str(ui_out.get("summary_line") or "").strip())
        or (
            (not current_summary_is_llm)
            and summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=str(ui_out.get("summary_line") or "").strip(),
            )
            and (not summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=effective_summary_seed,
            ))
        )
    ):
        ui_out["summary_line"] = effective_summary_seed

    if effective_primary:
        ui_out["primary_evidence"] = dict(effective_primary)
        ui_out["primary_evidence_heading_path"] = effective_heading_path
        effective_source = str(
            effective_primary.get("selection_reason")
            or ui_out.get("primary_evidence_source")
            or ("doc_list_authoritative" if selected_source == "authoritative" else "")
        ).strip()
        if effective_source:
            ui_out["primary_evidence_source"] = effective_source
    if authoritative_primary_evidence:
        ui_out["authoritative_primary_evidence"] = dict(
            normalize_primary_ref_evidence_payload(
                authoritative_primary_evidence if isinstance(authoritative_primary_evidence, dict) else {}
            )
        )
        ui_out["primary_evidence_authority"] = "doc_list_authoritative"
    return ui_out, effective_primary


def _build_doc_list_hit_ui_seed(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
    build_doc_list_ref_hit: Callable[..., dict],
    source_filename: Callable[[str], str],
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    compact_reader_open_text: Callable[..., str],
    normalize_ref_copy_text: Callable[[str], str],
    resolve_ref_ui_heading_context: Callable[..., dict],
    top_heading: Callable[[str], str],
    primary_ref_evidence_summary_seed: Callable[[dict | None], str],
    build_ref_summary_basis_meta: Callable[..., dict],
    build_prompt_aligned_ref_why_line: Callable[..., str],
    doc_list_ref_why_line: Callable[..., str],
    prefer_zh_ref_card_locale: Callable[..., bool],
    build_ref_why_basis_meta: Callable[..., dict],
    summary_label: str,
    summary_title: str,
) -> tuple[dict, dict, dict]:
    hit = build_doc_list_ref_hit(raw_item=raw_item, idx=idx)
    meta = hit.get("meta") if isinstance(hit.get("meta"), dict) else {}
    source_path = str((meta or {}).get("source_path") or raw_item.get("source_path") or "").strip()
    source_name = str((meta or {}).get("source_name") or raw_item.get("source_name") or "").strip() or source_filename(source_path) or f"Reference {idx}"
    primary_evidence = normalize_primary_ref_evidence_payload(
        raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
    )
    authoritative_summary_line = compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    authoritative_summary_generation = (
        str(raw_item.get("summary_generation") or "").strip().lower()
        if authoritative_summary_line
        else ""
    )
    authoritative_why_line = normalize_ref_copy_text(str(raw_item.get("why_line") or "").strip())
    authoritative_why_generation = (
        str(raw_item.get("why_generation") or "").strip().lower()
        if authoritative_why_line
        else ""
    )
    heading_context = resolve_ref_ui_heading_context(
        prompt=prompt,
        source_path=source_path,
        heading_path=str((meta or {}).get("ref_best_heading_path") or raw_item.get("heading_path") or "").strip(),
        heading_fallback=str(
            (meta or {}).get("top_heading")
            or top_heading(str(raw_item.get("heading_path") or ""))
            or ""
        ).strip(),
        section_label=str((meta or {}).get("ref_section") or "").strip(),
        subsection_label=str((meta or {}).get("ref_subsection") or "").strip(),
    )
    heading_path = str(heading_context.get("heading_path") or raw_item.get("heading_path") or "").strip()
    heading = str(heading_context.get("heading") or "").strip()
    section_label = str(heading_context.get("section_label") or "").strip()
    subsection_label = str(heading_context.get("subsection_label") or "").strip()
    summary_seed = authoritative_summary_line or compact_reader_open_text(
        str(
            primary_ref_evidence_summary_seed(primary_evidence)
            or primary_evidence.get("highlight_snippet")
            or primary_evidence.get("snippet")
            or ""
        ).strip()
    )
    summary_generation = authoritative_summary_generation or "section_grounded"
    summary_basis_meta = (
        build_ref_summary_basis_meta(
            prompt=prompt,
            summary_kind="guide",
            summary_generation=summary_generation,
            summary_line=summary_seed,
        )
        if summary_seed
        else {}
    )
    why_seed = authoritative_why_line or build_prompt_aligned_ref_why_line(
        prompt=prompt,
        display_name=source_name,
        heading_path=heading_path,
        summary_line=summary_seed,
        why_line="",
    )
    if not why_seed:
        why_seed = doc_list_ref_why_line(
            prompt=prompt,
            heading_path=heading_path,
            prefer_zh=bool(prefer_zh_ref_card_locale(prompt, source_name)),
        )
    why_generation = authoritative_why_generation or "deterministic_grounded"
    why_basis_meta = (
        build_ref_why_basis_meta(
            prompt=prompt,
            why_generation=why_generation,
            why_line=why_seed,
        )
        if why_seed
        else {}
    )
    ui_meta = {
        "display_name": source_name,
        "heading_path": heading_path,
        "heading": heading,
        "section_label": section_label,
        "subsection_label": subsection_label,
        "page_start": None,
        "page_end": None,
        "summary_kind": "guide",
        "summary_label": summary_label,
        "summary_title": summary_title,
        "source_path": source_path,
        "citation_meta": {},
    }
    if summary_seed:
        ui_meta["summary_line"] = summary_seed
        ui_meta["summary_generation"] = str(summary_basis_meta.get("summary_generation") or summary_generation)
        ui_meta["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    if why_seed:
        ui_meta["why_line"] = why_seed
        ui_meta["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
        ui_meta["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    anchor_target_kind = str((meta or {}).get("anchor_target_kind") or "").strip().lower()
    anchor_target_number = _positive_int((meta or {}).get("anchor_target_number"))
    anchor_match_score = _non_negative_float((meta or {}).get("anchor_match_score"))
    explicit_doc_match_score = _non_negative_float((meta or {}).get("explicit_doc_match_score"))
    if anchor_target_kind:
        ui_meta["anchor_target_kind"] = anchor_target_kind
    if anchor_target_number > 0:
        ui_meta["anchor_target_number"] = anchor_target_number
    if anchor_match_score > 0.0:
        ui_meta["anchor_match_score"] = anchor_match_score
    if explicit_doc_match_score > 0.0:
        ui_meta["explicit_doc_match_score"] = explicit_doc_match_score
    return hit, ui_meta, primary_evidence


def _apply_doc_list_summary_fallbacks(
    *,
    raw_item: dict,
    prompt: str,
    source_name: str,
    heading_path: str,
    ui_meta: dict | None,
    primary_evidence: dict,
    effective_primary_evidence: dict,
    summary_source: str,
    summary_line_needs_polish: Callable[..., bool],
    looks_like_title_echo: Callable[[str, str], bool],
    looks_why_like_ref_summary: Callable[[str], bool],
    pick_ref_card_summary_fallback: Callable[..., str],
    collect_doc_list_ref_text_candidates: Callable[..., list[str]],
    build_ref_summary_basis_meta: Callable[..., dict],
    looks_fragmentary_ref_summary: Callable[[str], bool],
    looks_surface_like_ref_summary: Callable[[str], bool],
    looks_formula_heavy_ref_text: Callable[[str], bool],
    build_prompt_aligned_ref_summary_fallback: Callable[..., str],
    compact_reader_open_text: Callable[..., str],
    primary_ref_evidence_summary_seed: Callable[[dict | None], str],
) -> tuple[dict, str]:
    ui_out = dict(ui_meta or {}) if isinstance(ui_meta, dict) else {}
    summary_source_out = str(summary_source or "").strip()
    current_summary = str(ui_out.get("summary_line") or "").strip()
    current_summary_generation = str(ui_out.get("summary_generation") or "").strip().lower()
    current_summary_is_llm = current_summary_generation in {"llm_grounded", "llm_pack"}
    display_name = str(ui_out.get("display_name") or source_name).strip()
    if (not current_summary_is_llm) and current_summary and (
        summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=current_summary,
        )
        or looks_like_title_echo(current_summary, display_name)
        or looks_why_like_ref_summary(current_summary)
    ):
        fallback_summary = pick_ref_card_summary_fallback(
            prompt=prompt,
            title=display_name,
            candidates=collect_doc_list_ref_text_candidates(
                raw_item=raw_item,
                primary_evidence=effective_primary_evidence or primary_evidence,
            ),
        )
        if fallback_summary and (not looks_like_title_echo(fallback_summary, display_name)):
            summary_basis_meta = build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=str(ui_out.get("summary_kind") or "guide"),
                summary_generation="deterministic_grounded",
                summary_line=fallback_summary,
            )
            ui_out["summary_line"] = fallback_summary
            ui_out["summary_generation"] = str(
                summary_basis_meta.get("summary_generation") or "deterministic_grounded"
            )
            ui_out["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
            summary_source_out = "doc_list_fallback"
    current_summary = str(ui_out.get("summary_line") or "").strip()
    current_summary_generation = str(ui_out.get("summary_generation") or "").strip().lower()
    current_summary_is_llm = current_summary_generation in {"llm_grounded", "llm_pack"}
    if (not current_summary_is_llm) and (
        (not current_summary)
        or looks_like_title_echo(current_summary, display_name)
        or looks_why_like_ref_summary(current_summary)
        or looks_fragmentary_ref_summary(current_summary)
        or looks_surface_like_ref_summary(current_summary)
        or looks_formula_heavy_ref_text(current_summary)
    ):
        template_summary = build_prompt_aligned_ref_summary_fallback(
            prompt=prompt,
            display_name=display_name,
            heading_path=str(ui_out.get("heading_path") or heading_path),
            summary_line=current_summary,
            why_line=str(ui_out.get("why_line") or ""),
        )
        if template_summary and (not summary_line_needs_polish(
            prompt=prompt,
            title=display_name,
            summary_line=template_summary,
        )):
            summary_basis_meta = build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=str(ui_out.get("summary_kind") or "guide"),
                summary_generation="deterministic_grounded",
                summary_line=template_summary,
            )
            ui_out["summary_line"] = template_summary
            ui_out["summary_generation"] = str(
                summary_basis_meta.get("summary_generation") or "deterministic_grounded"
            )
            ui_out["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
            if summary_source_out != "doc_list_fallback":
                summary_source_out = "doc_list_prompt_aligned"
    if not str(ui_out.get("summary_line") or "").strip():
        summary_seed = compact_reader_open_text(
            str(
                raw_item.get("summary_line")
                or primary_ref_evidence_summary_seed(effective_primary_evidence)
                or primary_evidence.get("highlight_snippet")
                or primary_evidence.get("snippet")
                or ""
            ).strip()
        )
        if summary_seed:
            ui_out["summary_line"] = summary_seed
            summary_source_out = "doc_list_ultimate_seed"
    if not str(ui_out.get("summary_line") or "").strip():
        raw_snippets: list[str] = []
        for h in list(raw_item.get("hits") or []):
            txt = str(h.get("text") or h.get("snippet") or "").strip()
            if txt and len(txt) > 30:
                raw_snippets.append(txt)
        if not raw_snippets:
            for alt_key in ("summary_line", "highlight_snippet", "snippet", "text"):
                txt = str(raw_item.get(alt_key) or "").strip()
                if txt and len(txt) > 30:
                    raw_snippets.append(txt)
        if raw_snippets:
            fallback_raw = raw_snippets[0][:200].rsplit(" ", 1)[0] if len(raw_snippets[0]) > 200 else raw_snippets[0]
            ui_out["summary_line"] = fallback_raw
            ui_out["summary_generation"] = "raw_fallback"
            summary_source_out = "doc_list_raw_fallback"
    current_summary = str(ui_out.get("summary_line") or "").strip()
    authoritative_summary = compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    if (
        authoritative_summary
        and looks_why_like_ref_summary(current_summary)
        and not looks_like_title_echo(authoritative_summary, display_name)
        and not looks_why_like_ref_summary(authoritative_summary)
        and not looks_fragmentary_ref_summary(authoritative_summary)
        and not looks_surface_like_ref_summary(authoritative_summary)
        and not looks_formula_heavy_ref_text(authoritative_summary)
    ):
        summary_generation = str(raw_item.get("summary_generation") or "").strip().lower() or "section_grounded"
        summary_basis_meta = build_ref_summary_basis_meta(
            prompt=prompt,
            summary_kind=str(ui_out.get("summary_kind") or "guide"),
            summary_generation=summary_generation,
            summary_line=authoritative_summary,
        )
        ui_out["summary_line"] = authoritative_summary
        ui_out["summary_generation"] = str(summary_basis_meta.get("summary_generation") or summary_generation)
        ui_out["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
        summary_source_out = "doc_list_authoritative_fast"
    return ui_out, summary_source_out


def _apply_doc_list_why_fallback(
    *,
    prompt: str,
    source_name: str,
    heading_path: str,
    ui_meta: dict | None,
    why_line_needs_polish: Callable[..., bool],
    build_prompt_aligned_ref_why_line: Callable[..., str],
    doc_list_ref_why_line: Callable[..., str],
    prefer_zh_ref_card_locale: Callable[..., bool],
    build_ref_why_basis_meta: Callable[..., dict],
) -> dict:
    ui_out = dict(ui_meta or {}) if isinstance(ui_meta, dict) else {}
    if why_line_needs_polish(
        prompt=prompt,
        display_name=str(ui_out.get("display_name") or source_name),
        heading_path=str(ui_out.get("heading_path") or heading_path),
        summary_line=str(ui_out.get("summary_line") or ""),
        why_line=str(ui_out.get("why_line") or ""),
    ):
        fallback_why = build_prompt_aligned_ref_why_line(
            prompt=prompt,
            display_name=str(ui_out.get("display_name") or source_name),
            heading_path=str(ui_out.get("heading_path") or heading_path),
            summary_line=str(ui_out.get("summary_line") or ""),
            why_line=str(ui_out.get("why_line") or ""),
        )
        if not fallback_why:
            fallback_why = doc_list_ref_why_line(
                prompt=prompt,
                heading_path=str(ui_out.get("heading_path") or heading_path),
                prefer_zh=bool(prefer_zh_ref_card_locale(prompt, source_name)),
            )
        if fallback_why:
            why_basis_meta = build_ref_why_basis_meta(
                prompt=prompt,
                why_generation="deterministic_grounded",
                why_line=fallback_why,
            )
            ui_out["why_line"] = fallback_why
            ui_out["why_generation"] = str(why_basis_meta.get("why_generation") or "deterministic_grounded")
            ui_out["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    return ui_out


def _finalize_doc_list_hit_ui_meta(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
    source_path: str,
    source_name: str,
    heading_path: str,
    ui_meta: dict | None,
    primary_evidence: dict,
    effective_primary_evidence: dict,
    summary_source: str,
    allow_expensive_llm: bool,
    align_ref_card_copy_to_user_locale: Callable[..., tuple[str, str]],
    build_ref_summary_surface_meta: Callable[..., dict],
    build_ref_summary_basis_meta: Callable[..., dict],
    build_ref_why_basis_meta: Callable[..., dict],
    score_tier: Callable[[float], str],
    build_doc_list_reader_open_payload: Callable[..., dict],
) -> dict:
    ui_out = dict(ui_meta or {}) if isinstance(ui_meta, dict) else {}
    aligned_summary_line, aligned_why_line = align_ref_card_copy_to_user_locale(
        prompt=prompt,
        display_name=str(ui_out.get("display_name") or source_name),
        heading_path=str(ui_out.get("heading_path") or heading_path),
        summary_line=str(ui_out.get("summary_line") or ""),
        why_line=str(ui_out.get("why_line") or ""),
        summary_kind=str(ui_out.get("summary_kind") or "guide"),
        allow_llm_translate=bool(allow_expensive_llm),
    )
    if aligned_summary_line:
        ui_out["summary_line"] = aligned_summary_line
    if aligned_why_line:
        ui_out["why_line"] = aligned_why_line
    summary_surface = build_ref_summary_surface_meta(
        prompt=prompt,
        summary_kind=str(ui_out.get("summary_kind") or "guide"),
        summary_line=str(ui_out.get("summary_line") or ""),
    )
    ui_out["summary_kind"] = str(summary_surface.get("summary_kind") or ui_out.get("summary_kind") or "guide")
    ui_out["summary_label"] = str(summary_surface.get("summary_label") or "")
    ui_out["summary_title"] = str(summary_surface.get("summary_title") or "")
    summary_generation = str(ui_out.get("summary_generation") or "").strip().lower() or "deterministic_grounded"
    why_generation = str(ui_out.get("why_generation") or "").strip().lower() or "deterministic_grounded"
    if str(ui_out.get("summary_line") or "").strip():
        summary_basis_meta = build_ref_summary_basis_meta(
            prompt=prompt,
            summary_kind=str(ui_out.get("summary_kind") or "guide"),
            summary_generation=summary_generation,
            summary_line=str(ui_out.get("summary_line") or ""),
        )
        ui_out["summary_generation"] = str(summary_basis_meta.get("summary_generation") or summary_generation)
        ui_out["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    if str(ui_out.get("why_line") or "").strip():
        why_basis_meta = build_ref_why_basis_meta(
            prompt=prompt,
            why_generation=why_generation,
            why_line=str(ui_out.get("why_line") or ""),
        )
        ui_out["why_generation"] = str(why_basis_meta.get("why_generation") or why_generation)
        ui_out["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    score = max(7.8, round(9.55 - (idx - 1) * 0.18, 2))
    ui_out["score"] = score
    ui_out["score_pending"] = False
    ui_out["score_tier"] = score_tier(score)
    ui_out["source_path"] = source_path
    reader_open = build_doc_list_reader_open_payload(
        source_path=source_path,
        source_name=source_name,
        heading_path=str(ui_out.get("heading_path") or heading_path),
        summary_line=str(ui_out.get("summary_line") or ""),
        primary_evidence=effective_primary_evidence or primary_evidence,
        reader_open=ui_out.get("reader_open") if isinstance(ui_out.get("reader_open"), dict) else {},
    )
    if reader_open:
        ui_out["reader_open"] = reader_open
    if effective_primary_evidence:
        ui_out["primary_evidence"] = dict(effective_primary_evidence)
        ui_out["primary_evidence_heading_path"] = str(
            effective_primary_evidence.get("heading_path")
            or ui_out.get("heading_path")
            or heading_path
            or ""
        ).strip()
    elif primary_evidence:
        ui_out["primary_evidence"] = dict(primary_evidence)
        ui_out["primary_evidence_heading_path"] = str(primary_evidence.get("heading_path") or heading_path or "").strip()
        ui_out["primary_evidence_source"] = "doc_list_authoritative"
    topic_match_kind = str(raw_item.get("topic_match_kind") or "").strip().lower()
    if topic_match_kind:
        ui_out["topic_match_kind"] = topic_match_kind
    ui_out["summary_source"] = str(summary_source or "").strip()
    return ui_out


def _build_doc_list_hit_ui_meta(
    *,
    raw_item: dict,
    idx: int,
    prompt: str,
    allow_expensive_llm: bool,
    allow_exact_locate: bool,
    source_filename: Callable[[str], str],
    compact_reader_open_text: Callable[..., str],
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    build_doc_list_ref_hit: Callable[..., dict],
    build_hit_ui_meta: Callable[..., dict],
    build_doc_list_hit_ui_seed: Callable[..., tuple[dict, dict, dict]],
    apply_doc_list_effective_primary_evidence: Callable[..., tuple[dict, dict]],
    apply_doc_list_summary_fallbacks: Callable[..., tuple[dict, str]],
    apply_doc_list_why_fallback: Callable[..., dict],
    finalize_doc_list_hit_ui_meta: Callable[..., dict],
    preloaded_citation_meta: dict[str, dict] | None = None,
) -> dict:
    source_path = str(raw_item.get("source_path") or "").strip()
    source_name = str(raw_item.get("source_name") or "").strip() or source_filename(source_path) or f"Reference {idx}"
    authoritative_summary_line = compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    primary_evidence = normalize_primary_ref_evidence_payload(
        raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
    )
    auth_reason = str(primary_evidence.get("selection_reason") or "").strip().lower()
    authoritative_primary_weak = bool(
        primary_evidence
        and (not str(primary_evidence.get("snippet") or primary_evidence.get("highlight_snippet") or "").strip())
        and (not str(primary_evidence.get("block_id") or "").strip())
        and auth_reason in {"", "answer_hit_top", "pending_section_seed"}
    )
    summary_source = ""
    if authoritative_primary_weak:
        hit = build_doc_list_ref_hit(raw_item=raw_item, idx=idx)
        ui_meta = dict(
            build_hit_ui_meta(
                hit,
                prompt=prompt,
                pdf_root=None,
                lib_store=None,
                allow_expensive_llm=bool(allow_expensive_llm),
                allow_exact_locate=bool(allow_exact_locate),
            )
            or {}
        )
        summary_source = str(ui_meta.get("summary_source") or "").strip()
    else:
        _hit, ui_meta, primary_evidence = build_doc_list_hit_ui_seed(
            raw_item=raw_item,
            idx=idx,
            prompt=prompt,
        )
        summary_source = "doc_list_seed"
    cached_citation_meta = (
        (preloaded_citation_meta or {}).get(source_path)
        if source_path and isinstance(preloaded_citation_meta, dict)
        else None
    )
    if isinstance(cached_citation_meta, dict) and cached_citation_meta:
        ui_meta["citation_meta"] = dict(cached_citation_meta)
    heading_path = (
        str(ui_meta.get("heading_path") or "").strip()
        or str(raw_item.get("heading_path") or "").strip()
        or str(primary_evidence.get("heading_path") or "").strip()
    )
    if not str(ui_meta.get("display_name") or "").strip():
        ui_meta["display_name"] = source_name
    ui_meta, effective_primary_evidence = apply_doc_list_effective_primary_evidence(
        prompt=prompt,
        display_name=str(ui_meta.get("display_name") or source_name),
        fallback_heading_path=heading_path,
        ui_meta=ui_meta,
        authoritative_primary_evidence=primary_evidence,
        authoritative_summary_line=authoritative_summary_line,
        authoritative_summary_generation=str(raw_item.get("summary_generation") or "").strip(),
    )
    if not str(ui_meta.get("heading_path") or "").strip() and heading_path:
        ui_meta["heading_path"] = heading_path
    ui_meta, summary_source = apply_doc_list_summary_fallbacks(
        raw_item=raw_item,
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        ui_meta=ui_meta,
        primary_evidence=primary_evidence,
        effective_primary_evidence=effective_primary_evidence,
        summary_source=summary_source,
    )
    ui_meta = apply_doc_list_why_fallback(
        prompt=prompt,
        source_name=source_name,
        heading_path=heading_path,
        ui_meta=ui_meta,
    )
    return finalize_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=idx,
        prompt=prompt,
        source_path=source_path,
        source_name=source_name,
        heading_path=heading_path,
        ui_meta=ui_meta,
        primary_evidence=primary_evidence,
        effective_primary_evidence=effective_primary_evidence,
        summary_source=summary_source,
        allow_expensive_llm=bool(allow_expensive_llm),
    )


def _doc_list_topic_match_why_line(
    *,
    prompt: str,
    heading_path: str,
    match_kind: str,
    prefer_zh_ref_card_locale: Callable[..., bool],
) -> str:
    kind = str(match_kind or "").strip().lower()
    if not kind:
        return ""
    prefer_zh = bool(prefer_zh_ref_card_locale(prompt, heading_path))
    loc = " / ".join(part for part in str(heading_path or "").split(" / ") if part).strip()
    zh_fallback_loc = "\u76f8\u5173\u6bb5\u843d"
    en_fallback_loc = "the matched section"
    if kind == "sci_related_predecessor":
        if prefer_zh:
            return "\u8be5\u6587\u8ba8\u8bba\u7684\u662f single-shot compressive spectral imaging\uff0c\u53ef\u4f5c\u4e3a\u4e0e SCI \u76f8\u5173\u7684\u65e9\u671f\u524d\u8eab\u5de5\u4f5c\uff0c\u4f46\u4e0d\u662f\u4e25\u683c\u7684 SCI \u672f\u8bed\u547d\u4e2d\u3002"
        return "This paper is better treated as an early related predecessor: it discusses single-shot compressive spectral imaging, which is SCI-adjacent rather than an exact SCI term match."
    if kind == "explicit_sci_mention":
        if prefer_zh:
            return f"\u8be5\u6587\u5728\u201c{loc or heading_path or zh_fallback_loc}\u201d\u5904\u660e\u786e\u63d0\u5230 Snapshot Compressive Imaging (SCI)\uff0c\u76f4\u63a5\u5bf9\u5e94\u8fd9\u7c7b SCI \u5b9a\u4f4d\u95ee\u9898\u3002"
        return f"The paper explicitly mentions Snapshot Compressive Imaging (SCI) in '{loc or heading_path or en_fallback_loc}', so it is a direct match for this SCI lookup."
    return ""


def _apply_doc_list_topic_match_hints(
    *,
    prompt: str,
    raw_item: dict,
    ui_meta: dict,
    doc_list_topic_match_why_line: Callable[..., str],
    is_llm_ref_why_generation: Callable[[str], bool],
    why_line_needs_polish: Callable[..., bool],
    why_line_explicitly_names_focus_term: Callable[[str, str], bool],
    build_ref_why_basis_meta: Callable[..., dict],
    compact_reader_open_text: Callable[..., str],
    is_llm_ref_summary_generation: Callable[[str], bool],
    summary_line_needs_polish: Callable[..., bool],
    looks_like_title_echo: Callable[[str, str], bool],
    build_ref_summary_basis_meta: Callable[..., dict],
) -> dict:
    ui = dict(ui_meta or {})
    match_kind = str(raw_item.get("topic_match_kind") or ui.get("topic_match_kind") or "").strip().lower()
    if not match_kind:
        return ui
    ui["topic_match_kind"] = match_kind
    note = doc_list_topic_match_why_line(
        prompt=prompt,
        heading_path=str(ui.get("heading_path") or raw_item.get("heading_path") or "").strip(),
        match_kind=match_kind,
    )
    current_why = str(ui.get("why_line") or "").strip()
    require_llm_copy = True
    current_why_is_llm = is_llm_ref_why_generation(str(ui.get("why_generation") or ""))
    should_override = bool(
        note
        and (not (require_llm_copy and current_why_is_llm))
        and (
            match_kind == "sci_related_predecessor"
            or (not current_why)
            or why_line_needs_polish(
                prompt=prompt,
                display_name=str(ui.get("display_name") or raw_item.get("source_name") or "").strip(),
                heading_path=str(ui.get("heading_path") or raw_item.get("heading_path") or "").strip(),
                summary_line=str(ui.get("summary_line") or raw_item.get("summary_line") or "").strip(),
                why_line=current_why,
            )
            or (not why_line_explicitly_names_focus_term(prompt, current_why))
        )
    )
    if should_override:
        why_basis_meta = build_ref_why_basis_meta(
            prompt=prompt,
            why_generation="deterministic_grounded",
            why_line=note,
        )
        ui["why_line"] = note
        ui["why_generation"] = str(why_basis_meta.get("why_generation") or "deterministic_grounded")
        ui["why_basis"] = str(why_basis_meta.get("why_basis") or "")
    if match_kind == "sci_related_predecessor":
        fallback_summary = compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
        current_summary = str(ui.get("summary_line") or "").strip()
        display_name = str(ui.get("display_name") or raw_item.get("source_name") or "").strip()
        current_summary_is_llm = is_llm_ref_summary_generation(str(ui.get("summary_generation") or ""))
        if fallback_summary and (
            not (require_llm_copy and current_summary_is_llm)
        ) and (
            (not current_summary)
            or summary_line_needs_polish(
                prompt=prompt,
                title=display_name,
                summary_line=current_summary,
            )
            or bool(re.match(r"^[a-z][a-z0-9 -]{8,60}:\s", current_summary.lower()))
            or looks_like_title_echo(current_summary, display_name)
        ):
            summary_basis_meta = build_ref_summary_basis_meta(
                prompt=prompt,
                summary_kind=str(ui.get("summary_kind") or "guide"),
                summary_generation="deterministic_grounded",
                summary_line=fallback_summary,
            )
            ui["summary_line"] = fallback_summary
            ui["summary_generation"] = str(summary_basis_meta.get("summary_generation") or "deterministic_grounded")
            ui["summary_basis"] = str(summary_basis_meta.get("summary_basis") or "")
    return ui


def _filter_doc_list_rows_for_guide(
    *,
    doc_rows: list[dict] | None,
    guide_mode: bool,
    guide_source_path: str,
    guide_source_name: str,
    filter_bound_source: bool = False,
    source_filename: Callable[[str], str],
    hit_matches_guide_source: Callable[..., bool],
) -> tuple[list[dict], int]:
    rows = [dict(item) for item in list(doc_rows or []) if isinstance(item, dict)]
    guide_path = str(guide_source_path or "").strip()
    guide_name = str(guide_source_name or "").strip()
    guide_active = bool(guide_mode and filter_bound_source and (guide_path or guide_name))
    if not guide_active:
        return rows, 0
    out: list[dict] = []
    filtered_self = 0
    for raw_item in rows:
        source_path = str(raw_item.get("source_path") or "").strip()
        source_name = str(raw_item.get("source_name") or "").strip() or source_filename(source_path)
        if hit_matches_guide_source(
            {
                "source_path": source_path,
                "source_name": source_name,
                "display_name": source_name,
            },
            guide_source_path=guide_path,
            guide_source_name=guide_name,
        ):
            filtered_self += 1
            continue
        out.append(raw_item)
    return out, filtered_self


def _build_doc_list_payload_hits(
    *,
    doc_rows: list[dict] | None,
    prompt: str,
    allow_expensive_llm: bool,
    allow_exact_locate: bool,
    build_doc_list_hit_ui_meta: Callable[..., dict],
    normalize_ref_copy_ui_meta: Callable[[dict | None], dict],
    apply_doc_list_topic_match_hints: Callable[..., dict],
    preloaded_citation_meta: dict[str, dict] | None = None,
) -> list[dict]:
    hits: list[dict] = []
    for idx, raw_item in enumerate(list(doc_rows or []), start=1):
        if not isinstance(raw_item, dict):
            continue
        source_path = str(raw_item.get("source_path") or "").strip()
        if not source_path:
            continue
        hit_ui_kwargs = {
            "raw_item": raw_item,
            "idx": idx,
            "prompt": prompt,
            "allow_expensive_llm": bool(allow_expensive_llm),
            "allow_exact_locate": bool(allow_exact_locate),
        }
        if preloaded_citation_meta is not None:
            hit_ui_kwargs["preloaded_citation_meta"] = preloaded_citation_meta
        ui_meta = build_doc_list_hit_ui_meta(
            **hit_ui_kwargs,
        )
        ui_meta = normalize_ref_copy_ui_meta(ui_meta)
        ui_meta = apply_doc_list_topic_match_hints(
            prompt=prompt,
            raw_item=raw_item,
            ui_meta=ui_meta,
        )
        ui_meta = _dedupe_doc_list_card_copy(raw_item=raw_item, ui_meta=ui_meta)
        hits.append(
            {
                "text": str(ui_meta.get("summary_line") or ui_meta.get("why_line") or source_path).strip(),
                "meta": {
                    "source_path": source_path,
                    "ref_pack_state": "ready",
                    "ref_best_heading_path": str(ui_meta.get("heading_path") or "").strip(),
                },
                "ui_meta": ui_meta,
            }
        )
    return hits


def _polish_doc_list_payload_hits(
    *,
    prompt: str,
    doc_rows: list[dict] | None,
    hits: list[dict],
    allow_expensive_llm: bool,
    normalize_ref_copy_ui_meta: Callable[[dict | None], dict],
    maybe_polish_single_ref_hit_card: Callable[..., dict],
    apply_doc_list_topic_match_hints: Callable[..., dict],
    batch_polish_doc_list_ref_hit_cards: Callable[..., dict],
    ref_card_has_llm_copy: Callable[[dict | None], bool],
    refs_card_polish_max_workers: Callable[[int], int],
) -> list[dict]:
    polished_hits: list[dict] = list(hits)
    rows = list(doc_rows or [])
    jobs: list[tuple[int, dict, dict]] = []
    for idx, hit in enumerate(hits):
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        if not isinstance(ui_meta, dict):
            continue
        jobs.append((idx, hit, ui_meta))

    def _polish_one(idx: int, hit: dict, ui_meta: dict) -> tuple[int, dict]:
        polished_ui = normalize_ref_copy_ui_meta(
            maybe_polish_single_ref_hit_card(
                prompt=prompt,
                hit=hit,
                ui_meta=ui_meta,
                allow_expensive_llm=bool(allow_expensive_llm),
            )
        )
        polished_ui = apply_doc_list_topic_match_hints(
            prompt=prompt,
            raw_item=rows[idx],
            ui_meta=polished_ui,
        )
        return idx, polished_ui

    batch_polished_raw = (
        batch_polish_doc_list_ref_hit_cards(
            prompt=prompt,
            jobs=jobs,
        )
        if bool(allow_expensive_llm)
        else {}
    )
    batch_polished = {
        int(idx): apply_doc_list_topic_match_hints(
            prompt=prompt,
            raw_item=rows[int(idx)],
            ui_meta=dict(ui_meta or {}),
        )
        for idx, ui_meta in dict(batch_polished_raw or {}).items()
        if str(idx).isdigit() or isinstance(idx, int)
    }
    leftover_jobs = [
        (idx, hit, ui_meta)
        for idx, hit, ui_meta in jobs
        if (
            idx not in batch_polished
            or (
                bool(allow_expensive_llm)
                and True
                and not ref_card_has_llm_copy(batch_polished.get(idx))
            )
        )
    ]
    for idx, polished_ui in batch_polished.items():
        hit2 = dict(hits[idx])
        hit2["ui_meta"] = polished_ui
        polished_hits[idx] = hit2

    max_workers = refs_card_polish_max_workers(len(leftover_jobs))
    if max_workers <= 1:
        for idx, hit, ui_meta in leftover_jobs:
            _, polished_ui = _polish_one(idx, hit, ui_meta)
            hit2 = dict(hit)
            hit2["ui_meta"] = polished_ui
            polished_hits[idx] = hit2
    else:
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_polish_one, idx, hit, ui_meta) for idx, hit, ui_meta in leftover_jobs]
                for fu in as_completed(futs):
                    try:
                        idx, polished_ui = fu.result()
                    except Exception:
                        continue
                    hit2 = dict(hits[idx])
                    hit2["ui_meta"] = polished_ui
                    polished_hits[idx] = hit2
        except Exception:
            for idx, hit, ui_meta in leftover_jobs:
                _, polished_ui = _polish_one(idx, hit, ui_meta)
                hit2 = dict(hit)
                hit2["ui_meta"] = polished_ui
                polished_hits[idx] = hit2
    for idx, hit in enumerate(polished_hits):
        if not isinstance(hit, dict):
            continue
        row = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
        hit_out = dict(hit)
        hit_out["ui_meta"] = _dedupe_doc_list_card_copy(
            raw_item=row,
            ui_meta=hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {},
        )
        polished_hits[idx] = hit_out
    return polished_hits


def _finalize_doc_list_payload_pack(
    *,
    user_msg_id: int | str,
    pack_src: dict | None,
    hits: list[dict],
    guide_active: bool,
    guide_source_path_norm: str,
    guide_source_name_norm: str,
    prompt_cross_paper_refs: bool,
    filtered_self_doc_count: int,
    allow_expensive_llm: bool,
    refs_hits_have_llm_copy: Callable[[list[dict]], bool],
    source_filename: Callable[[str], str],
    attach_pack_display_contract: Callable[[dict], dict],
) -> dict:
    pack_out = dict(pack_src or {}) if isinstance(pack_src, dict) else {}
    pack_out["user_msg_id"] = int(user_msg_id) if str(user_msg_id).isdigit() else user_msg_id
    pack_out["hits"] = hits
    pipeline_debug = dict(pack_out.get("pipeline_debug") or {}) if isinstance(pack_out.get("pipeline_debug"), dict) else {}
    pipeline_debug["doc_list_authoritative"] = True
    pipeline_debug["guide_active"] = bool(guide_active)
    pipeline_debug["final_hit_count"] = int(len(hits))
    pipeline_debug["raw_hit_count"] = int(len(hits))
    pipeline_debug["post_score_gate_hit_count"] = int(len(hits))
    pipeline_debug["post_focus_filter_hit_count"] = int(len(hits))
    pipeline_debug["post_llm_filter_hit_count"] = int(len(hits))
    pipeline_debug["filtered_self_hit_count"] = int(filtered_self_doc_count)
    pipeline_debug["prompt_likely_cross_paper_refs"] = bool(prompt_cross_paper_refs)
    pipeline_debug["copy_polish_allow_expensive_llm"] = bool(allow_expensive_llm)
    pipeline_debug["copy_polish_llm_required"] = True
    pipeline_debug["copy_polish_llm_complete"] = bool(refs_hits_have_llm_copy(hits))
    raw_qv = pack_src.get("query_variants") if isinstance(pack_src, dict) else []
    if raw_qv:
        pipeline_debug["query_variants"] = list(raw_qv)
    pack_out["pipeline_debug"] = pipeline_debug
    if guide_active:
        hidden_self_source = bool(prompt_cross_paper_refs and (filtered_self_doc_count > 0 or not hits))
        pack_out["guide_filter"] = {
            "active": True,
            "hidden_self_source": hidden_self_source,
            "filtered_hit_count": int(filtered_self_doc_count),
            "guide_source_path": guide_source_path_norm,
            "guide_source_name": guide_source_name_norm or source_filename(guide_source_path_norm),
        }
    pack_out["payload_mode"] = "full"
    return attach_pack_display_contract(pack_out)


def _finalize_legacy_doc_list_payload_pack(
    *,
    user_msg_id: int | str,
    pack_src: dict | None,
    hits: list[dict],
    guide_active: bool,
    guide_source_path_norm: str,
    guide_source_name_norm: str,
    prompt_cross_paper_refs: bool,
    source_filename: Callable[[str], str],
    attach_pack_display_contract: Callable[[dict], dict],
) -> dict:
    pack_out = dict(pack_src or {}) if isinstance(pack_src, dict) else {}
    pack_out["user_msg_id"] = int(user_msg_id) if str(user_msg_id).isdigit() else user_msg_id
    pack_out["hits"] = hits
    pipeline_debug = dict(pack_out.get("pipeline_debug") or {}) if isinstance(pack_out.get("pipeline_debug"), dict) else {}
    pipeline_debug["doc_list_authoritative"] = True
    pipeline_debug["guide_active"] = bool(guide_active)
    pipeline_debug["final_hit_count"] = int(len(hits))
    if "raw_hit_count" not in pipeline_debug:
        pipeline_debug["raw_hit_count"] = int(len(hits))
    if "post_score_gate_hit_count" not in pipeline_debug:
        pipeline_debug["post_score_gate_hit_count"] = int(len(hits))
    if "post_focus_filter_hit_count" not in pipeline_debug:
        pipeline_debug["post_focus_filter_hit_count"] = int(len(hits))
    if "post_llm_filter_hit_count" not in pipeline_debug:
        pipeline_debug["post_llm_filter_hit_count"] = int(len(hits))
    if "filtered_self_hit_count" not in pipeline_debug:
        pipeline_debug["filtered_self_hit_count"] = 0
    pipeline_debug["prompt_likely_cross_paper_refs"] = bool(prompt_cross_paper_refs)
    pack_out["pipeline_debug"] = pipeline_debug
    if guide_active:
        hidden_self_source = bool(prompt_cross_paper_refs)
        pack_out["guide_filter"] = {
            "active": True,
            "hidden_self_source": hidden_self_source,
            "filtered_hit_count": 0,
            "guide_source_path": guide_source_path_norm,
            "guide_source_name": guide_source_name_norm or source_filename(guide_source_path_norm),
        }
    pack_out["payload_mode"] = "full"
    return attach_pack_display_contract(pack_out)


def _build_legacy_doc_list_payload_hits(
    *,
    doc_list: list[dict] | None,
    prompt: str,
    prefer_zh: bool,
    source_filename: Callable[[str], str],
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    compact_reader_open_text: Callable[..., str],
    doc_list_ref_why_line: Callable[..., str],
    score_tier: Callable[[float], str],
) -> list[dict]:
    hits: list[dict] = []
    for idx, raw_item in enumerate(list(doc_list or []), start=1):
        if not isinstance(raw_item, dict):
            continue
        source_path = str(raw_item.get("source_path") or "").strip()
        if not source_path:
            continue
        source_name = str(raw_item.get("source_name") or "").strip() or source_filename(source_path) or f"Reference {idx}"
        heading_path = str(raw_item.get("heading_path") or "").strip()
        primary_evidence = normalize_primary_ref_evidence_payload(
            raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
        )
        summary_line = compact_reader_open_text(
            str(
                raw_item.get("summary_line")
                or primary_evidence.get("highlight_snippet")
                or primary_evidence.get("snippet")
                or ""
            ).strip()
        )
        why_line = doc_list_ref_why_line(
            prompt=prompt,
            heading_path=heading_path or str(primary_evidence.get("heading_path") or "").strip(),
            prefer_zh=prefer_zh,
        )
        reader_open = {
            "sourcePath": source_path,
            "sourceName": source_name,
            "headingPath": heading_path or str(primary_evidence.get("heading_path") or "").strip() or None,
            "snippet": summary_line or None,
            "highlightSnippet": summary_line or None,
            "strictLocate": bool(primary_evidence.get("strict_locate")),
            "blockId": str(primary_evidence.get("block_id") or "").strip() or None,
            "anchorId": str(primary_evidence.get("anchor_id") or "").strip() or None,
        }
        if primary_evidence:
            reader_open["primaryEvidence"] = dict(primary_evidence)
        score = max(6.6, round(9.6 - (idx - 1) * 0.18, 2))
        ui_meta = {
            "display_name": source_name,
            "heading_path": heading_path,
            "score": score,
            "score_pending": False,
            "score_tier": score_tier(score),
            "summary_line": summary_line,
            "summary_kind": "guide",
            "summary_label": "\u5bfc\u8bfb" if prefer_zh else "Guide",
            "summary_title": "\u8fd9\u6761\u8bc1\u636e\u8bf4\u660e\u4ec0\u4e48" if prefer_zh else "What This Evidence Shows",
            "summary_generation": "doc_list_contract",
            "summary_basis": "\u57fa\u4e8e\u5171\u4eab\u591a\u7bc7\u6587\u732e\u5217\u8868 contract \u7684\u5c55\u793a\u6458\u8981" if prefer_zh else "Display summary sourced from the shared multi-paper document list contract",
            "why_line": why_line,
            "why_generation": "doc_list_contract",
            "why_basis": "\u57fa\u4e8e\u5171\u4eab\u591a\u7bc7\u6587\u732e\u5217\u8868 contract \u7684\u4fdd\u7559\u7406\u7531" if prefer_zh else "Retention reason sourced from the shared multi-paper document list contract",
            "semantic_badges": [],
            "can_open": True,
            "citation_meta": {},
            "source_path": source_path,
            "reader_open": {key: value for key, value in reader_open.items() if value not in (None, "", [], {})},
        }
        if primary_evidence:
            ui_meta["primary_evidence"] = dict(primary_evidence)
            if not str(ui_meta.get("heading_path") or "").strip():
                ui_meta["heading_path"] = str(primary_evidence.get("heading_path") or "").strip()
        hits.append(
            {
                "text": summary_line or why_line,
                "meta": {
                    "source_path": source_path,
                    "ref_pack_state": "ready",
                    "ref_best_heading_path": str(ui_meta.get("heading_path") or "").strip(),
                },
                "ui_meta": ui_meta,
            }
        )
    return hits


def _build_doc_list_refs_payload(
    *,
    user_msg_id: int | str,
    pack: dict | None,
    doc_list: list[dict] | None,
    allow_expensive_llm: bool = False,
    allow_exact_locate: bool = True,
    apply_copy_polish: bool = True,
    guide_mode: bool = False,
    guide_source_path: str = "",
    guide_source_name: str = "",
    prompt_likely_cross_paper_refs: Callable[[str], bool],
    filter_doc_list_rows_for_guide: Callable[..., tuple[list[dict], int]],
    build_doc_list_payload_hits: Callable[..., list[dict]],
    polish_doc_list_payload_hits: Callable[..., list[dict]],
    suppress_non_llm_ref_card_copy_hits: Callable[..., list[dict]],
    finalize_doc_list_payload_pack: Callable[..., dict],
    prefer_zh_ref_card_locale: Callable[..., bool],
    build_legacy_doc_list_payload_hits: Callable[..., list[dict]],
    finalize_legacy_doc_list_payload_pack: Callable[..., dict],
) -> dict:
    pack_src = dict(pack or {}) if isinstance(pack, dict) else {}
    prompt = str(pack_src.get("prompt") or "").strip()
    guide_source_path_norm = str(guide_source_path or "").strip()
    guide_source_name_norm = str(guide_source_name or "").strip()
    guide_active = bool(guide_mode and (guide_source_path_norm or guide_source_name_norm))
    prompt_cross_paper_refs = bool(prompt_likely_cross_paper_refs(prompt))
    doc_rows_all = [dict(item) for item in list(doc_list or []) if isinstance(item, dict)]
    doc_rows, filtered_self_doc_count = filter_doc_list_rows_for_guide(
        doc_rows=doc_rows_all,
        guide_mode=guide_active,
        guide_source_path=guide_source_path_norm,
        guide_source_name=guide_source_name_norm,
        filter_bound_source=prompt_cross_paper_refs,
    )
    if doc_rows_all:
        hits = build_doc_list_payload_hits(
            doc_rows=doc_rows,
            prompt=prompt,
            allow_expensive_llm=bool(allow_expensive_llm),
            allow_exact_locate=bool(allow_exact_locate),
        )
        if apply_copy_polish and hits:
            hits = polish_doc_list_payload_hits(
                prompt=prompt,
                doc_rows=doc_rows,
                hits=hits,
                allow_expensive_llm=bool(allow_expensive_llm),
            )
        if bool(allow_expensive_llm) and True:
            hits = suppress_non_llm_ref_card_copy_hits(prompt=prompt, hits=hits)
        return finalize_doc_list_payload_pack(
            user_msg_id=user_msg_id,
            pack_src=pack_src,
            hits=hits,
            guide_active=guide_active,
            guide_source_path_norm=guide_source_path_norm,
            guide_source_name_norm=guide_source_name_norm,
            prompt_cross_paper_refs=prompt_cross_paper_refs,
            filtered_self_doc_count=filtered_self_doc_count,
            allow_expensive_llm=bool(allow_expensive_llm),
        )
    prefer_zh = bool(prefer_zh_ref_card_locale(prompt))
    hits = build_legacy_doc_list_payload_hits(
        doc_list=doc_list,
        prompt=prompt,
        prefer_zh=prefer_zh,
    )
    return finalize_legacy_doc_list_payload_pack(
        user_msg_id=user_msg_id,
        pack_src=pack_src,
        hits=hits,
        guide_active=guide_active,
        guide_source_path_norm=guide_source_path_norm,
        guide_source_name_norm=guide_source_name_norm,
        prompt_cross_paper_refs=prompt_cross_paper_refs,
    )


def _build_doc_list_ref_locs(
    *,
    heading_path: str,
    primary_evidence: dict,
    clean_refs_evidence_snippet: Callable[..., str],
    top_heading: Callable[[str], str],
) -> list[dict]:
    locs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _push(candidate: dict, *, source: str) -> None:
        if not isinstance(candidate, dict):
            return
        loc_heading = str(candidate.get("heading_path") or heading_path or "").strip()
        snippet = clean_refs_evidence_snippet(
            str(candidate.get("highlight_snippet") or candidate.get("snippet") or "").strip(),
            prompt="",
            source_path=str(candidate.get("source_path") or "").strip(),
            heading_path=loc_heading,
            max_len=360,
        )
        if (not loc_heading) and (not snippet):
            return
        key = (loc_heading, snippet)
        if key in seen:
            return
        seen.add(key)
        loc = {
            "heading_path": loc_heading or None,
            "heading": top_heading(loc_heading) or None,
            "snippet": snippet or None,
            "text": snippet or None,
            "quote": snippet or None,
            "quality": "high" if (loc_heading or snippet) else "medium",
            "source": source,
            "score": 96.0 - (len(locs) * 0.5),
        }
        locs.append({key: value for key, value in loc.items() if value not in (None, "", [], {})})

    _push(primary_evidence, source="doc_list_primary")
    for alt in list(primary_evidence.get("alternatives") or []):
        _push(alt if isinstance(alt, dict) else {}, source="doc_list_alternative")
        if len(locs) >= 4:
            break
    return locs


def _build_doc_list_ref_hit(
    *,
    raw_item: dict,
    idx: int,
    source_filename: Callable[[str], str],
    normalize_primary_ref_evidence_payload: Callable[[dict | None], dict],
    compact_reader_open_text: Callable[..., str],
    split_section_subsection: Callable[[str], tuple[str, str]],
    top_heading: Callable[[str], str],
    clean_refs_evidence_snippet: Callable[..., str],
) -> dict:
    source_path = str(raw_item.get("source_path") or "").strip()
    source_name = str(raw_item.get("source_name") or "").strip() or source_filename(source_path) or f"Reference {idx}"
    primary_evidence = normalize_primary_ref_evidence_payload(
        raw_item.get("primary_evidence") if isinstance(raw_item.get("primary_evidence"), dict) else {}
    )
    authoritative_summary_line = compact_reader_open_text(str(raw_item.get("summary_line") or "").strip())
    heading_path = (
        str(raw_item.get("heading_path") or "").strip()
        or str(primary_evidence.get("heading_path") or "").strip()
    )
    section_label, subsection_label = split_section_subsection(heading_path) if heading_path else ("", "")
    text_candidates = _collect_doc_list_ref_text_candidates(
        raw_item=raw_item,
        primary_evidence=primary_evidence,
        clean_refs_evidence_snippet=clean_refs_evidence_snippet,
    )
    anchor_kind = str(primary_evidence.get("anchor_kind") or "").strip().lower()
    anchor_number = _positive_int(primary_evidence.get("anchor_number"))
    rank_llm = max(72.0, 92.0 - float(max(0, idx - 1)) * 2.0)
    rank_bm25 = max(6.0, 9.4 - float(max(0, idx - 1)) * 0.4)
    meta = {
        "source_path": source_path,
        "source_name": source_name,
        "display_name": source_name,
        "ref_pack_state": "ready",
        "heading_path": heading_path,
        "top_heading": top_heading(heading_path) or section_label or heading_path,
        "ref_best_heading_path": heading_path,
        "ref_section": section_label or top_heading(heading_path) or "",
        "ref_subsection": subsection_label or "",
        "ref_loc_quality": "high" if heading_path else "medium",
        "ref_locs": _build_doc_list_ref_locs(
            heading_path=heading_path,
            primary_evidence=primary_evidence,
            clean_refs_evidence_snippet=clean_refs_evidence_snippet,
            top_heading=top_heading,
        ),
        "ref_show_snippets": list(text_candidates[:3]),
        "ref_snippets": list(text_candidates[:3]),
        "ref_overview_snippets": list(text_candidates[:2]),
        "explicit_doc_match_score": 12.0,
        "ref_rank": {
            "llm": rank_llm,
            "bm25": rank_bm25,
            "deep": 2.8,
            "term_bonus": 2.4,
            "semantic_score": 8.8,
            "score": rank_llm,
            "display_score": rank_llm,
        },
    }
    if anchor_kind:
        meta["anchor_target_kind"] = anchor_kind
    if anchor_number > 0:
        meta["anchor_target_number"] = anchor_number
        meta["anchor_match_score"] = 10.0
    if primary_evidence:
        meta["authoritative_primary_evidence"] = dict(primary_evidence)
    return {
        "text": str(text_candidates[0] if text_candidates else (source_name or source_path)).strip(),
        "meta": meta,
    }
