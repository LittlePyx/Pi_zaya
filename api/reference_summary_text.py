from __future__ import annotations

import difflib
import html
import json
import re


def _clean_summary_line(text: str) -> str:
    s = html.unescape(str(text or ""))
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\[[0-9,\-\s]{1,24}\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(?:abstract|\u6458\u8981)\s*[:\uff1a-]?\s*", "", s, flags=re.I).strip()
    if len(s) < 20:
        return ""
    return s


def _first_summary_sentence(text: str, *, max_len: int = 220) -> str:
    s = _clean_summary_line(text)
    if not s:
        return ""
    parts = re.split(r"(?<=[\u3002\uff01\uff1f!?\.])\s+", s)
    for part in parts:
        cand = str(part or "").strip()
        if len(cand) < 20:
            continue
        if len(cand) > max_len:
            cand = cand[:max_len].rstrip(" ,;:") + "..."
        return cand
    if len(s) > max_len:
        return s[:max_len].rstrip(" ,;:") + "..."
    return s


def _summary_excerpt(text: str, *, max_sentences: int = 3, max_len: int = 520) -> str:
    s = _clean_summary_line(text)
    if not s:
        return ""
    parts = re.split(r"(?<=[\u3002\uff01\uff1f!?\.])\s+", s)
    picked: list[str] = []
    total = 0
    for part in parts:
        cand = str(part or "").strip()
        if len(cand) < 18:
            continue
        if (total + len(cand)) > max_len:
            remain = max_len - total
            if remain >= 30:
                picked.append(cand[:remain].rstrip(" ,;:") + "...")
            break
        picked.append(cand)
        total += len(cand)
        if len(picked) >= max_sentences:
            break
    if picked:
        return " ".join(picked).strip()
    if len(s) > max_len:
        return s[:max_len].rstrip(" ,;:") + "..."
    return s


def _openalex_abstract_text(work: dict) -> str:
    if not isinstance(work, dict):
        return ""
    raw_abs = str(work.get("abstract") or "").strip()
    if raw_abs:
        return raw_abs
    inv = work.get("abstract_inverted_index")
    if not isinstance(inv, dict):
        return ""
    words: list[tuple[int, str]] = []
    for token, positions in inv.items():
        if not isinstance(token, str):
            continue
        if not isinstance(positions, list):
            continue
        for p in positions:
            try:
                pos = int(p)
            except Exception:
                continue
            if pos < 0:
                continue
            words.append((pos, token))
    if not words:
        return ""
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words).strip()


def _html_meta_content(page: str, names: tuple[str, ...]) -> str:
    html_text = str(page or "")
    if not html_text:
        return ""
    name_set = {name.lower() for name in names}
    for match in re.finditer(r"<meta\b[^>]*>", html_text, flags=re.I):
        tag = match.group(0)
        key_match = re.search(r"\b(?:name|property)\s*=\s*(['\"])(.*?)\1", tag, flags=re.I | re.S)
        if not key_match:
            continue
        key = html.unescape(str(key_match.group(2) or "").strip().lower())
        if key not in name_set:
            continue
        content_match = re.search(r"\bcontent\s*=\s*(['\"])(.*?)\1", tag, flags=re.I | re.S)
        if not content_match:
            continue
        value = html.unescape(str(content_match.group(2) or "")).strip()
        if value:
            return value
    return ""


def _jsonld_description_from_html(page: str) -> str:
    html_text = str(page or "")
    if not html_text:
        return ""
    for match in re.finditer(
        r"<script\b[^>]*type\s*=\s*(['\"])application/ld\+json\1[^>]*>(.*?)</script>",
        html_text,
        flags=re.I | re.S,
    ):
        raw = html.unescape(str(match.group(2) or "")).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        queue = data if isinstance(data, list) else [data]
        for item in queue:
            if not isinstance(item, dict):
                continue
            for key in ("abstract", "description"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def _looks_like_title_echo(summary_line: str, title: str) -> bool:
    s = _clean_summary_line(summary_line).lower()
    t = _clean_summary_line(title).lower()
    if (not s) or (not t):
        return False
    s_norm = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", s))
    t_norm = "".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+", t))
    if (not s_norm) or (not t_norm):
        return False
    if (t_norm in s_norm) and (len(s_norm) <= len(t_norm) + 36):
        return True
    if (s_norm in t_norm) and (len(s_norm) >= max(24, int(0.68 * len(t_norm)))):
        return True
    s_tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", s)
    t_tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", t)
    if (len(t_tokens) >= 4) and s_tokens:
        common = len(set(s_tokens) & set(t_tokens))
        if common >= max(3, int(0.85 * len(set(t_tokens)))) and len(set(s_tokens)) <= len(set(t_tokens)) + 3:
            return True
        if common >= max(2, int(0.50 * len(set(t_tokens)))):
            s_seq = " ".join(s_tokens)
            t_seq = " ".join(t_tokens)
            ratio = difflib.SequenceMatcher(None, s_seq, t_seq).ratio()
            if ratio >= 0.72:
                return True
    return False


def _has_cjk_text(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _has_latin_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text or "")))
