from __future__ import annotations

import re

import fitz


def styled_span_text(span: dict, *, inline: bool = False) -> str:
    """Preserve known TeX symbol alphabets; ordinary italic letters stay ordinary."""
    text = str(span.get("text") or "")
    if not text:
        text = "".join(str(ch.get("c") or "") for ch in span.get("chars", []))
    font = re.sub(r"^[A-Z]{6}\+", "", str(span.get("font") or ""))
    alphabet = "mathcal" if re.fullmatch(r"CMSY\d+", font) else ""
    if not alphabet:
        return text
    def replace(match):
        value = "\\" + alphabet + "{" + match.group() + "}"
        return "$" + value + "$" if inline else value
    return re.sub(r"[A-Z]", replace, text)


def coalesce_baseline_lines(lines: list[dict]) -> list[dict]:
    """Join adjacent fragments of one physical line within a raw PDF block."""
    result: list[dict] = []
    for line in lines:
        spans = line.get("spans") or []
        origins = [s["origin"][1] for s in spans if s.get("origin") and s.get("text", "").strip()]
        if result and origins:
            prev = result[-1]
            previous = [s["origin"][1] for s in prev.get("spans", [])
                        if s.get("origin") and s.get("text", "").strip()]
            size = max((float(s.get("size") or 0) for s in spans), default=0)
            a, b = fitz.Rect(prev["bbox"]), fitz.Rect(line["bbox"])
            if (previous and size > 0 and max(origins + previous) - min(origins + previous) <= size * .12
                    and 0 <= b.x0 - a.x1 <= size * 3):
                result[-1] = {**prev, "bbox": tuple(a | b), "spans": prev["spans"] + spans}
                continue
        result.append(line)
    return result


def _linear_script_math(spans: list[dict]) -> str:
    """Decode a single baseline and its smaller scripts; reject multiline layouts.

    Origins, rather than font bounding-box centers, distinguish scripts across
    fonts with different ascenders. Recursion retains a complement on a subscript.
    Fractions and multiple display rows keep the existing conservative fallback.
    """
    if not spans or any(not sp.get("origin") for sp in spans):
        return ""
    size = max(float(sp.get("size") or 0) for sp in spans)
    if size <= 0:
        return ""
    bases = [sp for sp in spans if float(sp.get("size") or 0) >= size * .9]
    baseline = sum(sp["origin"][1] for sp in bases) / len(bases)
    if max(abs(sp["origin"][1] - baseline) for sp in bases) > size * .15:
        return ""
    bases.sort(key=lambda sp: sp["bbox"][0])
    scripts = [sp for sp in spans if sp not in bases]
    if not scripts:
        return ""
    if any(re.search(r"[A-Za-z]{3,}", str(sp.get("text") or "")) for sp in spans):
        return ""  # Preserve prose spacing and tail detection in the existing fallback.
    attached: dict[int, dict[str, list[dict]]] = {}
    for sp in sorted(scripts, key=lambda sp: (-float(sp.get("size") or 0), sp["bbox"][0])):
        # A smaller glyph may itself carry a superscript (e.g. J^c in x_{J^c}).
        parents = [candidate for candidate in scripts if candidate is not sp
                   and float(candidate.get("size") or 0) >= float(sp.get("size") or 0) / .88
                   and candidate["bbox"][0] <= sp["bbox"][0]
                   and -size * .2 <= sp["bbox"][0] - candidate["bbox"][2] <= size * .5]
        if parents:
            parent = max(parents, key=lambda candidate: candidate["bbox"][0])
        else:
            candidates = [candidate for candidate in bases if candidate["bbox"][0] <= sp["bbox"][0]]
            if not candidates:
                return ""
            parent = candidates[-1]
        delta = sp["origin"][1] - parent["origin"][1]
        if sp["bbox"][0] - parent["bbox"][2] > size * .7 or abs(delta) > size:
            return ""
        if abs(delta) < float(parent.get("size") or size) * .08:
            return ""  # Small text on the baseline is not necessarily a script.
        kind = "^" if delta < 0 else "_"
        attached.setdefault(id(parent), {}).setdefault(kind, []).append(sp)

    def render(sp):
        value = styled_span_text(sp).strip().replace("∗", r"\ast")
        for kind in ("^", "_"):
            group = attached.get(id(sp), {}).get(kind, [])
            if group:
                value += kind + "{" + "".join(render(item) for item in sorted(group, key=lambda item: item["bbox"][0])) + "}"
        return value
    return "".join(render(sp) for sp in bases)


def extract_math_raw(page, bbox: tuple[float, float, float, float]) -> str:
    """
    Extract math-like text from `page` inside `bbox` using span-level geometry (rawdict),
    attempting to preserve superscript/subscript structure.

    This is critical for full_llm quality: `get_text('dict')` often loses layout cues,
    producing garbled math that even an LLM can't reliably reconstruct.
    """
    try:
        clip = fitz.Rect(bbox)
        if clip.width <= 2 or clip.height <= 2:
            return ""
    except Exception:
        return ""

    try:
        d = page.get_text("rawdict")
    except Exception:
        try:
            d = page.get_text("dict")
        except Exception:
            return ""

    spans: list[tuple[float, float, float, float, float, str]] = []
    selected: list[dict] = []
    try:
        for b0 in (d.get("blocks") or []):
            for ln in (b0.get("lines") or []):
                for sp in (ln.get("spans") or []):
                    txt = str(sp.get("text") or "")
                    if not txt.strip():
                        # Some PDFs store glyphs only in `chars` with empty `text`.
                        chars = sp.get("chars") or []
                        if chars:
                            try:
                                txt = "".join(str(ch.get("c") or "") for ch in chars)
                            except Exception:
                                txt = ""
                    if not txt.strip():
                        continue
                    sb = sp.get("bbox")
                    if not sb:
                        continue
                    try:
                        r = fitz.Rect(tuple(float(x) for x in sb))
                    except Exception:
                        continue
                    # Tall symbol boxes in the next prose row can intersect by
                    # a sliver. Require the majority of each glyph/span inside.
                    intersection = r & clip
                    if not r.intersects(clip) or intersection.width * intersection.height < r.width * r.height * .6:
                        continue
                    size = float(sp.get("size") or 0.0)
                    selected.append({**sp, "text": txt})
                    txt = styled_span_text({**sp, "text": txt})
                    spans.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1), size, txt))
    except Exception:
        spans = []

    if not spans:
        return ""

    linear = _linear_script_math(selected)
    if linear:
        return linear

    # Sort top-to-bottom, left-to-right
    spans.sort(key=lambda x: (x[1], x[0]))

    # Group spans into lines by y proximity
    lines: list[list[tuple[float, float, float, float, float, str]]] = []
    cur: list[tuple[float, float, float, float, float, str]] = []
    cur_y = None
    for sp in spans:
        y0, y1, size = sp[1], sp[3], sp[4]
        cy = (y0 + y1) / 2.0
        if cur_y is None:
            cur_y = cy
            cur = [sp]
            continue
        tol = max(2.0, (size or 10.0) * 0.65)
        if abs(cy - cur_y) <= tol:
            cur.append(sp)
            # update running center y
            cur_y = (cur_y * 0.7) + (cy * 0.3)
        else:
            lines.append(cur)
            cur = [sp]
            cur_y = cy
    if cur:
        lines.append(cur)

    def _median(xs: list[float]) -> float:
        if not xs:
            return 0.0
        xs2 = sorted(xs)
        mid = len(xs2) // 2
        return xs2[mid] if (len(xs2) % 2 == 1) else (xs2[mid - 1] + xs2[mid]) / 2.0

    out_lines: list[str] = []
    for ln in lines:
        ln.sort(key=lambda x: x[0])
        centers = [((x[1] + x[3]) / 2.0) for x in ln]
        sizes = [float(x[4] or 0.0) for x in ln]
        base_c = _median(centers)
        base_s = _median([s for s in sizes if s > 0.0]) or 0.0
        # Estimate line height
        heights = [max(1.0, float(x[3] - x[1])) for x in ln]
        lh = _median(heights) or 10.0

        parts: list[str] = []
        prev_x1 = None
        for x0, y0, x1, y1, size, txt in ln:
            t = str(txt)
            # Normalize some common math glyphs early
            t = (
                t.replace("\u2299", r"\odot")
                .replace("\u2208", r"\in")
                .replace("\u00d7", r"\times")
                .replace("\u00b7", r"\cdot")
            )
            t = re.sub(r"\s+", " ", t).strip()
            if not t:
                continue
            c = (y0 + y1) / 2.0
            is_small = (base_s > 0.0) and (size > 0.0) and (size <= base_s * 0.88)
            # Sup/sub classification by relative center y
            super_th = base_c - (lh * 0.22)
            sub_th = base_c + (lh * 0.22)
            if is_small and c < super_th:
                t = "^{" + t + "}"
            elif is_small and c > sub_th:
                t = "_{" + t + "}"

            # Insert a space if there is a large horizontal gap between spans
            if prev_x1 is not None:
                gap = float(x0) - float(prev_x1)
                if gap > max(2.0, (base_s or 10.0) * 0.4):
                    parts.append(" ")
            parts.append(t)
            prev_x1 = x1

        line_s = "".join(parts).strip()
        if line_s:
            out_lines.append(line_s)

    return "\n".join(out_lines).strip()
