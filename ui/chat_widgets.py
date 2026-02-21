from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import os
import re
import time
from pathlib import Path

import streamlit as st

from ui.strings import S

_STRUCT_CITE_CANON_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\]\]",
    re.IGNORECASE,
)
_STRUCT_CITE_SINGLE_RE = re.compile(
    r"(?<!\[)\[\s*CITE\s*:\s*([A-Za-z0-9_-]{4,24})\s*:\s*(\d{1,4})\s*\](?!\])",
    re.IGNORECASE,
)
_STRUCT_CITE_SID_ONLY_RE = re.compile(
    r"\[\[\s*CITE\s*:\s*[A-Za-z0-9_-]{4,24}\s*\]\]",
    re.IGNORECASE,
)
_STRUCT_CITE_GARBAGE_RE = re.compile(r"\[\[?\s*CITE\s*:[^\]\n]*\]?\]", re.IGNORECASE)
_CITE_ANCHOR_LINK_RE = re.compile(r"\[(\d{1,4})\]\(#([^\s)\"']+)(?:\s+\"[^\"]*\")?\)")


def _source_cite_id(source_path: str) -> str:
    s = str(source_path or "").strip()
    if not s:
        return "s0000000"
    return "s" + hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:8]


def _load_sid_num_doi_map_cached() -> dict[tuple[str, int], str]:
    db_dir_str = str(st.session_state.get("db_dir") or "").strip()
    if not db_dir_str:
        return {}
    idx_path = Path(db_dir_str) / "references_index.json"
    if not idx_path.exists():
        return {}

    try:
        st_sig = idx_path.stat()
        sig = f"{str(idx_path.resolve())}|{int(st_sig.st_mtime)}|{int(st_sig.st_size)}"
    except Exception:
        sig = str(idx_path)

    cache_key = "_kb_copy_sid_num_doi_cache_v1"
    cache = st.session_state.get(cache_key)
    if isinstance(cache, dict) and str(cache.get("sig") or "") == sig and isinstance(cache.get("data"), dict):
        return cache.get("data") or {}

    out: dict[tuple[str, int], str] = {}
    try:
        data = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    docs = data.get("docs") if isinstance(data, dict) else {}
    if isinstance(docs, dict):
        for dk, dv in docs.items():
            if not isinstance(dv, dict):
                continue
            source_path = str(dv.get("path") or "").strip() or str(dk or "").strip()
            sid = _source_cite_id(source_path).lower()
            refs = dv.get("refs")
            if not isinstance(refs, dict):
                continue
            for rk, rv in refs.items():
                if not isinstance(rv, dict):
                    continue
                try:
                    n = int(rk)
                except Exception:
                    continue
                if n <= 0:
                    continue
                doi_url = str(rv.get("doi_url") or "").strip()
                if (not doi_url) and str(rv.get("doi") or "").strip():
                    doi_url = f"https://doi.org/{str(rv.get('doi') or '').strip()}"
                if doi_url:
                    out[(sid, n)] = doi_url

    st.session_state[cache_key] = {"sig": sig, "data": out}
    return out


def _normalize_copy_citation_links(md: str, cite_details: list[dict] | None = None) -> str:
    s = str(md or "")
    if not s:
        return s

    # DOI hints from rendered citation details (if provided by caller).
    anchor_to_doi: dict[str, str] = {}
    num_to_doi: dict[int, str] = {}
    for rec in cite_details or []:
        if not isinstance(rec, dict):
            continue
        try:
            n = int(rec.get("num") or 0)
        except Exception:
            n = 0
        doi_url = str(rec.get("doi_url") or "").strip()
        if (not doi_url) and str(rec.get("doi") or "").strip():
            doi_url = f"https://doi.org/{str(rec.get('doi') or '').strip()}"
        if not doi_url:
            continue
        a = str(rec.get("anchor") or "").strip()
        if a:
            anchor_to_doi[a] = doi_url
        if (n > 0) and (n not in num_to_doi):
            num_to_doi[n] = doi_url

    sid_num_to_doi = _load_sid_num_doi_map_cached()

    # 1) Normalize structured cite tokens, preserving DOI links when resolvable.
    def _repl_struct_token(m: re.Match) -> str:
        sid = str(m.group(1) or "").strip().lower()
        try:
            n = int(m.group(2))
        except Exception:
            return ""
        doi_url = sid_num_to_doi.get((sid, n)) or num_to_doi.get(n)
        if doi_url:
            return f"[{n}]({doi_url})"
        return f"[{n}]"

    s = _STRUCT_CITE_CANON_RE.sub(_repl_struct_token, s)
    s = _STRUCT_CITE_SINGLE_RE.sub(_repl_struct_token, s)
    s = _STRUCT_CITE_SID_ONLY_RE.sub("", s)
    s = _STRUCT_CITE_GARBAGE_RE.sub("", s)

    # 2) Upgrade internal popup anchors to DOI links when metadata exists.
    def _anchor_repl(m: re.Match) -> str:
        try:
            n = int(m.group(1))
        except Exception:
            return str(m.group(0) or "")
        a = str(m.group(2) or "").strip()
        doi_url = anchor_to_doi.get(a) or num_to_doi.get(n)
        if not doi_url:
            return f"[{n}]"
        return f"[{n}]({doi_url})"

    s = _CITE_ANCHOR_LINK_RE.sub(_anchor_repl, s)
    return s


def _resolve_sidebar_logo_path() -> Path | None:
    base = Path(__file__).resolve().parent
    repo_root = base.parent
    cwd = Path.cwd()
    env_logo = (os.environ.get("KB_SIDEBAR_LOGO") or "").strip().strip("'\"")
    candidates: list[Path] = []
    if env_logo:
        p = Path(env_logo).expanduser()
        if not p.is_absolute():
            candidates.append((base / p).resolve())
            candidates.append((repo_root / p).resolve())
            candidates.append((cwd / p).resolve())
        candidates.append(p.resolve())
    candidates.extend(
        [
            base / "team_logo.png",
            base / "assets" / "team_logo.png",
            base / "assets" / "team_logo.jpg",
            base / "assets" / "team_logo.jpeg",
            base / "assets" / "team_logo.webp",
            repo_root / "team_logo.png",
            repo_root / "assets" / "team_logo.png",
            repo_root / "assets" / "team_logo.jpg",
            repo_root / "assets" / "team_logo.jpeg",
            repo_root / "assets" / "team_logo.webp",
            cwd / "team_logo.png",
            cwd / "assets" / "team_logo.png",
            cwd / "assets" / "team_logo.jpg",
            cwd / "assets" / "team_logo.jpeg",
            cwd / "assets" / "team_logo.webp",
        ]
    )
    seen: set[str] = set()
    for p in candidates:
        try:
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None

def _sidebar_logo_data_uri(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
        if not raw:
            return None
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _resolve_ai_inline_logo_data_uri() -> str | None:
    cache_key = "_kb_ai_inline_logo_data_uri"
    miss_key = "__MISS__"
    cached = st.session_state.get(cache_key, None)
    if cached == miss_key:
        return None
    if isinstance(cached, str) and cached.startswith("data:"):
        return cached

    base = Path(__file__).resolve().parent
    env_logo = (os.environ.get("KB_INLINE_AI_LOGO") or "").strip().strip("'\"")
    candidates: list[Path] = []
    if env_logo:
        p = Path(env_logo).expanduser()
        if not p.is_absolute():
            candidates.append((base / p).resolve())
        candidates.append(p.resolve())
    candidates.extend(
        [
            base / "assets" / "pi_logo.png",
            base / "pi_logo.png",
            base / "assets" / "team_logo.png",
        ]
    )
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                uri = _sidebar_logo_data_uri(p)
                if uri:
                    st.session_state[cache_key] = uri
                    return uri
        except Exception:
            continue

    st.session_state[cache_key] = miss_key
    return None

def _render_app_title() -> None:
    title = str(S.get("title") or "").strip()
    if not title:
        return
    safe_title = html.escape(title)
    if bool(st.session_state.get("_hero_title_typed_once")):
        st.markdown(f"<h1 class='kb-hero-title'>{safe_title}</h1>", unsafe_allow_html=True)
        return

    holder = st.empty()
    acc: list[str] = []
    for ch in title:
        acc.append(ch)
        live = html.escape("".join(acc))
        holder.markdown(
            f"<h1 class='kb-hero-title'>{live}<span class='kb-title-caret'>▌</span></h1>",
            unsafe_allow_html=True,
        )
        time.sleep(0.020 if ord(ch) < 128 else 0.028)

    holder.markdown(f"<h1 class='kb-hero-title'>{safe_title}</h1>", unsafe_allow_html=True)
    st.session_state["_hero_title_typed_once"] = True

def _normalize_math_markdown(text: str) -> str:
    """
    Make math rendering more stable in Streamlit markdown.

    Goals:
    - Inline math: $...$
    - Display math: $$...$$
    - Avoid code spans wrapping math (backticks break KaTeX/MathJax).
    """
    if not text:
        return text

    import re

    s = text

    # Prefer $...$ and $$...$$ over \( \) and \[ \], but do it conservatively.
    # Avoid touching escaped citation brackets like \[24\].
    def _inline_math_repl(m: re.Match) -> str:
        inner = str(m.group(1) or "")
        return f"${inner}$"

    def _display_math_repl(m: re.Match) -> str:
        inner = str(m.group(1) or "")
        probe = inner.strip()
        # Keep citation-like escaped brackets untouched.
        if re.fullmatch(r"\d{1,4}(?:\s*[,;，、-]\s*\d{1,4})*", probe):
            return m.group(0)
        # Convert only when it reasonably looks like math/display content.
        looks_math = bool(
            ("\n" in inner)
            or re.search(
                r"[=^_{}]|\\(?:frac|sum|int|prod|sqrt|mathbf|mathbb|left|right|begin|end|alpha|beta|gamma|theta|lambda|cdot|times)",
                inner,
            )
        )
        if not looks_math:
            return m.group(0)
        return "$$" + inner + "$$"

    s = re.sub(r"\\\((.+?)\\\)", _inline_math_repl, s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.+?)\\\]", _display_math_repl, s, flags=re.DOTALL)

    # Unwrap math that was mistakenly put in code spans.
    s = re.sub(r"`(\$[^`]+?\$)`", r"\1", s)
    s = re.sub(r"`(\$\$[\s\S]+?\$\$)`", r"\1", s)

    return s

def _md_to_plain_text(md: str) -> str:
    """
    Best-effort Markdown -> plain text for clipboard copy.
    Keeps formulas as their LaTeX source ($$...$$ / $...$).
    """
    if not md:
        return ""

    s = md
    # Remove code fences but keep their content.
    s = re.sub(r"```[^\n]*\n", "", s)
    s = s.replace("```", "")
    # Remove inline code backticks.
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Links: keep citation marker as [n], others as plain label text.
    def _link_to_text(m: re.Match) -> str:
        label = str(m.group(1) or "").strip()
        if re.fullmatch(r"\d{1,4}", label):
            return f"[{label}]"
        return label
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", _link_to_text, s)
    # Images: ![alt](url) -> alt
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)
    # Basic emphasis markers
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Headings/list markers
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", s)
    s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)
    # Collapse extra blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def _render_answer_copy_bar(answer_md: str, *, key_ns: str, cite_details: list[dict] | None = None) -> None:
    md = _normalize_math_markdown(answer_md or "")
    md = _normalize_copy_citation_links(md, cite_details)
    txt = _md_to_plain_text(md)
    md_id = f"{key_ns}_md"
    txt_id = f"{key_ns}_txt"
    ai_logo_uri = _resolve_ai_inline_logo_data_uri()
    logo_html = (
        f'<img class="kb-ai-inline-logo" src="{html.escape(ai_logo_uri)}" alt="AI"/>'
        if ai_logo_uri
        else ""
    )

    # Hidden payloads (large text lives here, buttons reference them by id).
    st.markdown(
        f"""
<textarea id="{html.escape(md_id)}" style="display:none">{html.escape(md)}</textarea>
<textarea id="{html.escape(txt_id)}" style="display:none">{html.escape(txt)}</textarea>
<div class="kb-copybar">
  {logo_html}
  <button class="kb-copybtn" type="button" data-target="{html.escape(txt_id)}">\u590d\u5236\u6587\u672c</button>
  <button class="kb-copybtn" type="button" data-target="{html.escape(md_id)}">\u590d\u5236 Markdown</button>
</div>
        """,
        unsafe_allow_html=True,
    )

def _render_ai_live_header(stage: str = "") -> None:
    logo_uri = _resolve_ai_inline_logo_data_uri()
    safe_stage = html.escape((stage or "").strip())
    logo_html = (
        f'<img class="kb-ai-live-logo" src="{html.escape(logo_uri)}" alt="AI"/>'
        if logo_uri
        else '<span class="msg-meta">AI</span>'
    )
    stage_html = f'<span class="kb-ai-live-stage">阶段：{safe_stage}</span>' if safe_stage else ""
    st.markdown(
        f"""
<div class="kb-ai-livebar">
  {logo_html}
  <span class="kb-ai-live-pill">生成中</span>
  {stage_html}
</div>
        """,
        unsafe_allow_html=True,
    )
