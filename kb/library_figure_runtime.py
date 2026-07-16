from __future__ import annotations

import re
import threading
from pathlib import Path
from urllib.parse import quote

from kb.paper_guide_structured_index_runtime import (
    extract_figure_scope_from_text,
    figure_key_for_scope,
    filter_figure_index_rows,
    normalize_figure_scope,
)

_DOC_FIGURE_CACHE_LOCK = threading.Lock()
_DOC_FIGURE_CACHE: dict[str, tuple[float, list[dict]]] = {}
_DOC_FIGURE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_MD_IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _resolve_doc_image_path(md_path: Path, raw_ref: str) -> Path | None:
    ref = str(raw_ref or "").strip().strip("'").strip('"')
    if not ref:
        return None
    low = ref.lower()
    if low.startswith(("http://", "https://", "data:")):
        return None
    if "?" in ref:
        ref = ref.split("?", 1)[0]
    if "#" in ref:
        ref = ref.split("#", 1)[0]
    ref = ref.replace("\\", "/")
    cand = Path(ref)
    if not cand.is_absolute():
        cand = (md_path.parent / cand).resolve()
    else:
        cand = cand.resolve()
    if (not cand.exists()) or (not cand.is_file()):
        return None
    if cand.suffix.lower() not in _DOC_FIGURE_IMAGE_EXTS:
        return None
    return cand


def _collect_doc_figure_assets(
    md_path: Path,
    *,
    extract_figure_number,
) -> list[dict]:
    path = Path(md_path).expanduser()
    if (not path.exists()) or (not path.is_file()):
        return []
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    key = str(path.resolve())
    with _DOC_FIGURE_CACHE_LOCK:
        cached = _DOC_FIGURE_CACHE.get(key)
        if isinstance(cached, tuple) and len(cached) == 2:
            old_mtime, old_items = cached
            if float(old_mtime) == mtime:
                return [dict(x) for x in (old_items or [])]

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = text.splitlines()
    out: list[dict] = []
    seen_paths: set[str] = set()

    for i, line in enumerate(lines):
        for match in _MD_IMAGE_LINK_RE.finditer(line):
            alt = str(match.group(1) or "").strip()
            raw_img = str(match.group(2) or "").strip()
            img_path = _resolve_doc_image_path(path, raw_img)
            if img_path is None:
                continue
            sp = str(img_path)
            if sp in seen_paths:
                continue
            seen_paths.add(sp)
            caption = ""
            alt_number = extract_figure_number(alt) or extract_figure_number(raw_img)
            # Converters commonly leave a blank line between an image and its
            # caption (and Extended Data often places the caption above it).
            # Inspect a small symmetric window instead of treating the image
            # alt text as the semantic figure identity.
            for distance in range(1, 5):
                nearby: list[str] = []
                if (i + distance) < len(lines):
                    nearby.append(str(lines[i + distance] or "").strip())
                if (i - distance) >= 0:
                    nearby.append(str(lines[i - distance] or "").strip())
                caption = next(
                    (
                        candidate
                        for candidate in nearby
                        if extract_figure_number(candidate) > 0
                        and (
                            int(alt_number or 0) <= 0
                            or extract_figure_number(candidate) == int(alt_number)
                        )
                    ),
                    "",
                )
                if caption:
                    break
            number = extract_figure_number(caption) or alt_number
            label = caption or alt or img_path.name
            figure_scope = extract_figure_scope_from_text(label, default_main=bool(number))
            out.append(
                {
                    "path": sp,
                    "number": int(number or 0),
                    "label": str(label or "").strip(),
                    "figure_scope": figure_scope,
                    "figure_key": figure_key_for_scope(figure_scope, int(number or 0)),
                }
            )

    with _DOC_FIGURE_CACHE_LOCK:
        _DOC_FIGURE_CACHE[key] = (mtime, [dict(x) for x in out])
        if len(_DOC_FIGURE_CACHE) > 512:
            try:
                for k in list(_DOC_FIGURE_CACHE.keys())[:128]:
                    _DOC_FIGURE_CACHE.pop(k, None)
            except Exception:
                pass
    return out


def _build_doc_figure_card(
    *,
    source_path: str,
    figure_num: int,
    figure_scope: str = "",
    collect_doc_figure_assets,
    source_name_from_md_path,
) -> dict | None:
    src = Path(str(source_path or "").strip())
    if (not src.exists()) or (not src.is_file()):
        return None
    items = collect_doc_figure_assets(src)
    if not items:
        return None
    matching_items = filter_figure_index_rows(
        items,
        figure_number=figure_num,
        figure_scope=figure_scope,
    )
    selected = matching_items[0] if matching_items else None
    if selected is None:
        return None
    img_path = str(selected.get("path") or "").strip()
    if not img_path:
        return None
    src_name = source_name_from_md_path(str(source_path or ""))
    label = str(selected.get("label") or "").strip()
    if len(label) > 140:
        label = label[:140].rstrip() + "..."
    return {
        "source_name": src_name,
        "figure_num": int(figure_num),
        "figure_scope": normalize_figure_scope(selected.get("figure_scope") or figure_scope),
        "figure_key": str(selected.get("figure_key") or figure_key_for_scope(figure_scope, figure_num)),
        "label": label,
        "url": f"/api/references/asset?path={quote(img_path, safe='')}",
    }


def _figure_display_label(figure_scope: str, figure_num: int) -> str:
    scope = normalize_figure_scope(figure_scope)
    if scope == "extended_data":
        return f"Extended Data Fig. {int(figure_num)}"
    if scope == "supplementary":
        return f"Supplementary Fig. {int(figure_num)}"
    return f"Fig. {int(figure_num)}"


def _call_figure_card_builder(build_doc_figure_card, *, source_path: str, figure_num: int, figure_scope: str) -> dict | None:
    """Pass semantic scope while keeping older injected test/build callables compatible."""
    try:
        return build_doc_figure_card(
            source_path=source_path,
            figure_num=figure_num,
            figure_scope=figure_scope,
        )
    except TypeError as exc:
        if "figure_scope" not in str(exc):
            raise
        return build_doc_figure_card(source_path=source_path, figure_num=figure_num)


def _score_figure_card_source_binding(*, prompt: str, meta: dict, figure_num: int, source_path: str, source_name_from_md_path) -> float:
    query = str(prompt or "").strip().lower()
    meta_map = meta if isinstance(meta, dict) else {}
    src = str(source_path or "").strip()
    src_name = source_name_from_md_path(src).lower()
    src_stem = Path(src_name).stem.lower()

    score = 0.0
    try:
        score += 2.0 * float(meta_map.get("explicit_doc_match_score") or 0.0)
    except Exception:
        pass

    kind = str(meta_map.get("anchor_target_kind") or "").strip().lower()
    try:
        n0 = int(meta_map.get("anchor_target_number") or 0)
    except Exception:
        n0 = 0
    try:
        a0 = float(meta_map.get("anchor_match_score") or 0.0)
    except Exception:
        a0 = 0.0
    if kind == "figure" and n0 > 0:
        if int(figure_num) == int(n0):
            score += 40.0 + max(0.0, a0)
        else:
            score -= 16.0
    elif kind and kind != "figure":
        score -= 10.0

    if query:
        if src_name and src_name in query:
            score += 36.0
        if src_stem and src_stem in query:
            score += 26.0
        if src_stem:
            tokens = [t for t in re.split(r"[^a-z0-9]+", src_stem) if len(t) >= 4]
            if tokens:
                overlap = sum(1 for t in set(tokens) if t in query)
                score += min(18.0, 4.0 * float(overlap))

    return float(score)


def _maybe_append_library_figure_markdown(
    answer: str,
    *,
    prompt: str,
    answer_hits: list[dict],
    bound_source_path: str = "",
    requested_figure_number,
    build_doc_figure_card,
    score_figure_card_source_binding,
) -> str:
    base = str(answer or "").rstrip()
    if (not base) or (not answer_hits):
        return base
    if "/api/references/asset?path=" in base:
        return base
    target_num = requested_figure_number(prompt, answer_hits)
    if target_num <= 0:
        return base
    target_scope = extract_figure_scope_from_text(prompt, default_main=True)

    cards_scored: list[tuple[float, dict]] = []
    seen_src: set[str] = set()
    preferred_src = str(bound_source_path or "").strip()
    if preferred_src:
        preferred_card = _call_figure_card_builder(
            build_doc_figure_card,
            source_path=preferred_src,
            figure_num=target_num,
            figure_scope=target_scope,
        )
        if preferred_card is not None:
            cards_scored.append((1000.0, preferred_card))
            seen_src.add(preferred_src)
    for hit in answer_hits:
        if not isinstance(hit, dict):
            continue
        meta = hit.get("meta", {}) or {}
        src = str(meta.get("source_path") or "").strip()
        if (not src) or (src in seen_src):
            continue
        seen_src.add(src)
        card = _call_figure_card_builder(
            build_doc_figure_card,
            source_path=src,
            figure_num=target_num,
            figure_scope=target_scope,
        )
        if card is None:
            continue
        score = score_figure_card_source_binding(
            prompt=prompt,
            meta=meta,
            figure_num=target_num,
            source_path=src,
        )
        cards_scored.append((score, card))

    if not cards_scored:
        return base

    cards_scored.sort(key=lambda x: float(x[0]), reverse=True)
    cards = [cards_scored[0][1]]

    lines: list[str] = ["### Library Figure"]
    for card in cards:
        src_name = str(card.get("source_name") or "unknown-source")
        fig_num = int(card.get("figure_num") or target_num)
        fig_scope = normalize_figure_scope(card.get("figure_scope") or target_scope)
        figure_label = _figure_display_label(fig_scope, fig_num)
        url = str(card.get("url") or "").strip()
        label = str(card.get("label") or "").strip()
        alt = f"{src_name} {figure_label}"
        lines.append(f"![{alt}]({url})")
        if label:
            lines.append(f"*Source: {src_name}, {figure_label}. {label}*")
        else:
            lines.append(f"*Source: {src_name}, {figure_label} (library asset)*")

    return f"{base}\n\n" + "\n\n".join(lines)
