from __future__ import annotations

import re
import threading
from pathlib import Path
from urllib.parse import quote

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
            next_line = str(lines[i + 1] or "").strip() if (i + 1) < len(lines) else ""
            prev_line = str(lines[i - 1] or "").strip() if i > 0 else ""
            caption = next_line if extract_figure_number(next_line) > 0 else ""
            if (not caption) and (extract_figure_number(prev_line) > 0):
                caption = prev_line
            number = extract_figure_number(caption) or extract_figure_number(alt) or extract_figure_number(raw_img)
            label = caption or alt or img_path.name
            out.append({"path": sp, "number": int(number or 0), "label": str(label or "").strip()})

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
    collect_doc_figure_assets,
    source_name_from_md_path,
) -> dict | None:
    src = Path(str(source_path or "").strip())
    if (not src.exists()) or (not src.is_file()):
        return None
    items = collect_doc_figure_assets(src)
    if not items:
        return None
    selected = next((it for it in items if int(it.get("number") or 0) == int(figure_num)), None)
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
        "label": label,
        "url": f"/api/references/asset?path={quote(img_path, safe='')}",
    }


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

    cards_scored: list[tuple[float, dict]] = []
    seen_src: set[str] = set()
    preferred_src = str(bound_source_path or "").strip()
    if preferred_src:
        preferred_card = build_doc_figure_card(source_path=preferred_src, figure_num=target_num)
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
        card = build_doc_figure_card(source_path=src, figure_num=target_num)
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
        url = str(card.get("url") or "").strip()
        label = str(card.get("label") or "").strip()
        alt = f"{src_name} Fig. {fig_num}"
        lines.append(f"![{alt}]({url})")
        if label:
            lines.append(f"*Source: {src_name}, Fig. {fig_num}. {label}*")
        else:
            lines.append(f"*Source: {src_name}, Fig. {fig_num} (library asset)*")

    return f"{base}\n\n" + "\n\n".join(lines)
