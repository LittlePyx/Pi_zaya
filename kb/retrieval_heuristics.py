from __future__ import annotations

import re
from collections import Counter

from .tokenize import tokenize

def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))

def _has_latin(text: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in (text or ""))

def _extract_keywords_for_desc(text: str, *, max_n: int = 4) -> list[str]:
    """
    Human-readable keywords for the relevance one-liner.
    Prefer CJK phrases (>=2) if present; otherwise use latin tokens.
    """
    t = (text or "").strip()
    if not t:
        return []
    # CJK phrases (clean question suffixes to keep it readable)
    cjk_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", t)
    cleaned: list[str] = []
    for p in cjk_phrases:
        p = (p or "").strip()
        if not p:
            continue
        for suf in ("是什么", "是啥", "是什么意思", "怎么做", "怎么", "如何", "为什么", "是否", "可以吗", "可以", "以及", "原理", "原因"):
            if p.endswith(suf) and len(p) > len(suf):
                p = p[: -len(suf)].strip()
        # Split by common glue words to get shorter, clearer phrases
        parts = re.split(r"[的与和及以及并而或，。；、]", p)
        parts = [x.strip() for x in parts if x and len(x.strip()) >= 2]
        cleaned.extend(parts if parts else [p])
    cjk_phrases = cleaned
    if cjk_phrases:
        out: list[str] = []
        seen = set()
        for p in cjk_phrases:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if len(out) >= max_n:
                break
        return out

    # Latin tokens
    toks = tokenize(t)
    out2: list[str] = []
    seen2 = set()
    for w in toks:
        w = (w or "").strip()
        if not w or len(w) <= 2:
            continue
        if w in seen2:
            continue
        seen2.add(w)
        out2.append(w)
        if len(out2) >= max_n:
            break
    return out2

def _quick_answer_for_prompt(prompt_text: str) -> str | None:
    """
    Fast local answers for trivial identity prompts.
    This avoids unnecessary queue stalls for basic "who are you" questions.
    """
    q = (prompt_text or "").strip().lower()
    if not q:
        return None
    keys = [
        "你是谁",
        "你叫什么",
        "你是谁开发的",
        "你是哪家开发的",
        "介绍一下你自己",
        "自我介绍",
        "who are you",
        "what is your name",
        "who made you",
        "who developed you",
    ]
    if any(k in q for k in keys):
        return "我是 P&I Lab 开发的 π-zaya。"
    return None

def _norm_text_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\s_]+", " ", s)
    return s


def _should_bypass_kb_retrieval(prompt_text: str) -> bool:
    """
    Skip academic retrieval for generic coding/programming prompts that are
    clearly not asking about the literature corpus.
    """
    q = (prompt_text or "").strip().lower()
    if not q:
        return False

    code_like = bool(re.search(
        r"(\bhello\s*world\b|\brandom\b|\brandom\s+number\b|\bcode\b|\bpython\b|\bjavascript\b|\bjs\b|\bjava\b|\bc\+\+\b|\bcpp\b|\bfunction\b|\bscript\b|\bdebug\b|\bbug\b)|"
        r"(\u5199.*\u4ee3\u7801|\u751f\u6210.*\u968f\u673a\u6570|\u968f\u673a\u6570|\u51fd\u6570|\u811a\u672c|\u4ee3\u7801|\u8c03\u8bd5|\u62a5\u9519|\u793a\u4f8b\u4ee3\u7801)",
        q,
        flags=re.I,
    ))
    if not code_like:
        return False

    academic_or_domain = bool(re.search(
        r"(\breference\b|\bcitation\b|\bpaper\b|\bpdf\b|\bdoi\b|\bequation\b|\bderive\b|\bgaussian splatting\b|\b3dgs\b|\bsingle[- ]pixel\b|\bsingle[- ]photon\b|\bcompressive\b|\bspectral\b|\boptics?\b|\bimaging\b|\bdetector\b)|"
        r"(\u53c2\u8003\u6587\u732e|\u5f15\u7528|\u8bba\u6587|\u6587\u732e|\u516c\u5f0f|\u63a8\u5bfc|\u5355\u50cf\u7d20|\u5355\u5149\u5b50|\u538b\u7f29|\u5149\u8c31|\u5149\u5b66|\u6210\u50cf|\u63a2\u6d4b\u5668|\u9ad8\u65af\u6cfc\u6ea2)",
        q,
        flags=re.I,
    ))
    return not academic_or_domain


def _should_prioritize_attached_image(prompt_text: str) -> bool:
    """
    Detect prompts that are mainly asking about the currently attached image itself,
    where KB retrieval is more likely to distract than help.
    """
    q = (prompt_text or "").strip().lower()
    if not q:
        return True

    image_focus = bool(re.search(
        r"(\bthis image\b|\bthis figure\b|\bthe image\b|\bthe figure\b|\blook at (the )?image\b|\blook at (the )?figure\b|"
        r"\bdescribe (this )?(image|figure)\b|\bwhat is in (this )?(image|figure)\b|\bwhat does (this )?(image|figure) say\b|"
        r"\bexplain (this )?(image|figure)\b|\bcaption (this )?(image|figure)\b)|"
        r"(\u8fd9\u5f20\u56fe|\u8fd9\u5f20\u56fe\u7247|\u8fd9\u4e2a\u56fe|\u8fd9\u5e45\u56fe|\u770b\u56fe|\u770b\u4e0b\u56fe|\u89e3\u91ca.*\u56fe|"
        r"\u8fd9\u56fe\u5199\u4e86\u5565|\u8fd9\u56fe\u5199\u4e86\u4ec0\u4e48|\u8fd9\u56fe\u4ec0\u4e48\u610f\u601d|"
        r"\u8fd9\u5f20\u56fe\u662f\u4ec0\u4e48|\u8fd9\u5f20\u56fe\u8bb2\u4e86\u4ec0\u4e48|\u8fd9\u5f20\u56fe\u5199\u4e86\u4ec0\u4e48|"
        r"\u63cf\u8ff0.*\u56fe|\u6982\u62ec.*\u56fe|\u56fe\u91cc\u5199\u4e86\u4ec0\u4e48)",
        q,
        flags=re.I,
    ))
    if not image_focus:
        return False

    explicit_doc_link = bool(re.search(
        r"(\bpaper\b|\bpdf\b|\bsection\b|\bchapter\b|\bcitation\b|\breference\b|\bwhich paper\b|\bfrom which paper\b)|"
        r"(\u8bba\u6587|\u6587\u732e|\u7ae0\u8282|\u5f15\u7528|\u53c2\u8003\u6587\u732e|\u51fa\u81ea\u54ea\u7bc7|\u54ea\u7bc7\u8bba\u6587|\u54ea\u4e2a\u7ae0\u8282)",
        q,
        flags=re.I,
    ))
    return not explicit_doc_link

def _query_term_profile(prompt_text: str, used_query: str) -> dict[str, bool]:
    """
    Identify key intent terms to help rerank documents.
    """
    zh = (prompt_text or "")
    en = _norm_text_for_match(used_query or "")
    p = {
        "wants_single_shot": ("单曝光" in zh) or ("单次曝光" in zh) or ("single-shot" in en) or ("single shot" in en) or ("single exposure" in en) or ("snapshot" in en),
        "wants_single_pixel": ("单像素" in zh) or ("single-pixel" in en) or ("single pixel" in en),
        "wants_single_photon": ("单光子" in zh) or ("single photon" in en) or ("spad" in en) or ("sns" in en) or ("nanowire" in en),
        "wants_compressive": ("压缩" in zh) or ("compressive" in en),
        "wants_spectral": ("光谱" in zh) or ("spectral" in en),
    }
    return p

def _doc_term_bonus(profile: dict[str, bool], doc_name: str, snippets: list[str]) -> float:
    """
    Small but decisive boosts/penalties for term mismatches (e.g. single-shot vs single-pixel).
    """
    hay = _norm_text_for_match(doc_name or "") + "\n" + _norm_text_for_match("\n".join(snippets or []))
    has_single_shot = any(k in hay for k in ["single-shot", "single shot", "single exposure", "snapshot"])
    has_single_pixel = any(k in hay for k in ["single-pixel", "single pixel"])
    has_single_photon = any(k in hay for k in ["single-photon", "single photon", "spad", "sns", "snspd", "nanowire"])
    has_spectral = "spectral" in hay
    has_compressive = "compressive" in hay

    bonus = 0.0
    if profile.get("wants_single_shot"):
        if has_single_shot:
            bonus += 2.4
        if has_single_pixel and (not has_single_shot):
            bonus -= 2.8
    if profile.get("wants_single_pixel"):
        if has_single_pixel:
            bonus += 2.2
        if has_single_shot and (not has_single_pixel):
            bonus -= 1.6
        if has_single_photon and (not has_single_pixel):
            bonus -= 2.0
    if profile.get("wants_single_photon"):
        if has_single_photon:
            bonus += 2.0
        if has_single_pixel and (not has_single_photon):
            bonus -= 1.6
    if profile.get("wants_spectral"):
        bonus += 0.9 if has_spectral else -0.6
    if profile.get("wants_compressive"):
        bonus += 0.6 if has_compressive else -0.3
    return bonus

def _is_probably_bad_heading(h: str) -> bool:
    s = " ".join((h or "").strip().split())
    if not s:
        return True
    if len(s) > 90:
        return True
    low = s.lower()
    # Metadata / copyright / pricing lines often leak into markdown as headings.
    bad_sub = (
        "received",
        "revised",
        "accepted",
        "publication date",
        "©",
        "copyright",
        "all rights reserved",
        "usd",
        "$",
        "doi:",
        "issn",
        "arxiv:",
        "download",
    )
    if any(x in low for x in bad_sub):
        return True
    # Too many digits usually means page labels / ids.
    digits = sum(ch.isdigit() for ch in s)
    if digits >= 10:
        return True
    return False

def _normalize_heading(h: str) -> str:
    s = " ".join((h or "").strip().split())
    if not s:
        return ""
    # Common "1 INTRODUCTION" -> keep as-is (it’s useful for users).
    return s

def _is_noise_snippet_text(t: str) -> bool:
    s = " ".join((t or "").strip().split())
    if not s:
        return True
    low = s.lower()
    # Boilerplate / metadata / copyright / submission timelines
    bad = (
        "received",
        "revised",
        "accepted",
        "publication date",
        "copyright",
        "all rights reserved",
        "creativecommons",
        "license",
        "doi:",
        "issn",
        "arxiv:",
        "usd",
    )
    if any(x in low for x in bad):
        return True
    # "$" is common in LaTeX math (e.g. "$\\mu$m", "$r_{\\min}$", "$9 \\times 9$") and must NOT be treated as noise.
    # Only treat it as noise when it looks like a *lone* currency token (no closing "$" nearby).
    if ("$" in s) and (s.count("$") < 2) and re.search(r"(?<!\\\\)\\$\\s*\\d", s):
        return True
    # Affiliations / author lists (comma-heavy lines)
    if s.count(",") >= 6 and len(s) > 120:
        # Abstract/Methods prose can be comma-heavy; only drop when it doesn't look like sentences.
        # Author/affiliation blocks are typically comma-separated with few sentence terminators.
        if s.count(".") <= 1 and s.count("?") == 0 and s.count("!") == 0:
            return True
    if re.search(r"university|institute|department|laboratory|school of|college of", low) and len(s) > 80:
        return True
    return False

def _clean_snippet_for_display(t: str, *, max_chars: int = 900) -> str:
    """
    Display as plain text (no markdown rendering); keep line breaks but trim.
    """
    s = (t or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # Drop pure image lines (common in converted markdown).
    try:
        s = "\n".join([ln for ln in s.split("\n") if not re.match(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$", ln)])
    except Exception:
        pass
    # Remove very long runs of whitespace
    s = re.sub(r"[ \t]{3,}", "  ", s)
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"
    return s

def _preferred_section_keys(prompt_text: str) -> list[str]:
    # Prefer a stable set of section names to match.
    if re.search(r"(参考文献|引用|cite|citation|reference)", prompt_text or "", flags=re.I):
        return ["references", "bibliography"]
    if re.search(r"(是什么|定义|含义|概念|what is|definition)", prompt_text or "", flags=re.I):
        return ["abstract", "introduction", "overview", "background"]
    if re.search(r"(编码|掩膜|码|pattern|mask|sampling|coded|coding|hadamard|fourier|dmd|metamaterial|multiplex)", prompt_text or "", flags=re.I):
        return ["method", "approach", "model", "coding", "encoding", "pattern", "mask", "sampling", "implementation"]
    if re.search(r"(公式|推导|证明|算法|方法|实现|derive|equation|method|approach|model)", prompt_text or "", flags=re.I):
        return ["method", "approach", "model", "algorithm", "theory"]
    if re.search(r"(实验|结果|指标|性能|对比|消融|experiment|result|evaluation|ablation|baseline|compare)", prompt_text or "", flags=re.I):
        return ["experiment", "results", "evaluation", "implementation", "setup"]
    return ["method", "approach", "model", "experiment", "results", "introduction"]

def _pick_best_heading_for_doc(headings: list[tuple[float, str]], prompt_text: str) -> str:
    """
    Choose a high-confidence heading that is most likely where the user can find the answer.
    headings: (score, heading_text)
    """
    prefs = _preferred_section_keys(prompt_text)
    wants_refs = bool(re.search(r"(参考文献|引用|cite|citation|reference)", prompt_text or "", flags=re.I))
    best: tuple[float, str] | None = None
    for score, h in headings:
        hh = _normalize_heading(h)
        if _is_probably_bad_heading(hh):
            continue
        low = hh.lower()
        if ("references" in low or "bibliography" in low) and (not wants_refs):
            # Don't point users to References unless they explicitly asked for citations.
            continue
        bonus = 0.0
        for i, k in enumerate(prefs):
            if k in low:
                bonus += 1.5 - (i * 0.15)
                break
        # Slightly prefer shorter headings.
        bonus += max(0.0, (40 - len(hh)) / 200.0)
        v = float(score) + bonus
        if best is None or v > best[0]:
            best = (v, hh)
    return best[1] if best else ""

def _aspects_from_snippets(snippets: list[str], prompt_text: str) -> list[str]:
    """
    Extract a few concrete 'what you can find here' aspects based on matched text (not hallucinated).
    """
    text = "\n".join([s for s in snippets if s]).lower()
    out: list[str] = []

    def add(label: str, *keys: str) -> None:
        if label in out:
            return
        if any(k in text for k in keys):
            out.append(label)

    q = (prompt_text or "")
    wants_encoding = bool(re.search(r"(编码|掩膜|pattern|mask|sampling|hadamard|fourier|dmd|metamaterial|multiplex|coded|coding)", q, flags=re.I))
    wants_detector = bool(re.search(r"(单光子|探测器|spad|snspd|nanowire|dark count|jitter|quantum efficiency|detector)", q, flags=re.I))
    wants_compare = bool(re.search(r"(对比|比较|baseline|compare|ablation|消融)", q, flags=re.I))
    wants_metrics = bool(re.search(r"(指标|性能|效率|灵敏度|暗计数|jitter|psnr|ssim|efficiency|dcr)", q, flags=re.I))

    # If user clearly asks about detectors, prioritize detector aspects.
    if wants_detector:
        add("探测原理/工作机制", "spad", "avalanche", "geiger", "nanowire", "snspd", "superconduct", "transition edge", "tes")
        add("器件结构/材料体系", "semiconductor", "bulk", "low-dimensional", "perovskite", "superconduct", "nanowire", "material")
        add("关键指标与权衡", "dark count", "dcr", "time jitter", "jitter", "efficiency", "quantum efficiency", "snr", "time resolution")
        add("应用场景", "application", "lidar", "imaging", "communication", "quantum", "bio")
        if wants_compare:
            add("对比/优缺点总结", "compare", "comparison", "advantage", "disadvantage", "trade-off", "baseline")
        # Add a couple of anchor terms (makes the hint less template-like).
        anchors = []
        for k, lab in [
            ("spad", "SPAD"),
            ("snspd", "SNSPD"),
            ("nanowire", "纳米线"),
            ("transition edge", "TES"),
            ("perovskite", "钙钛矿"),
        ]:
            if k in text and lab not in anchors:
                anchors.append(lab)
        if anchors and out:
            out[0] = out[0] + "（" + "、".join(anchors[:2]) + "）"
        return out[:4]

    # Imaging / compressive sensing / single-pixel etc.
    if wants_encoding:
        add("编码/采样策略", "hadamard", "fourier", "pattern", "mask", "dmd", "coded", "coding", "sampling", "multiplex", "frequency-division", "metamaterial", "metasurface", "speckle")

    add("测量/前向模型", "forward model", "measurement model", "measurement", "sensing", "compressive", "snapshot", "ghost imaging", "single-pixel", "single pixel", "spi")
    add("光学/硬件架构", "hardware", "optical", "disperser", "camera", "mask", "sensor", "system", "metamaterial", "metasurface")
    add("重建/反演算法", "reconstruction", "inverse", "recover", "optimization", "solver", "iter", "algorithm", "unrolling", "tv", "l1")
    add("训练设置/实现细节", "implementation", "training", "hyperparameter", "batch", "lr", "learning rate")
    add("损失/正则项", "loss", "objective", "regular", "prior", "constraint")
    add("实验设置/数据集", "dataset", "benchmark", "setup", "scene", "real data", "simulated")
    add("评价指标/对比结果", "psnr", "ssim", "metric", "performance", "compare", "baseline", "ablation", "result")
    add("局限/失败案例", "limitation", "failure", "future work", "discussion")
    add("可追溯参考文献", "references", "bibliography")

    # Make aspects more query-driven: if user asked for metrics/compare, prioritize those.
    if wants_metrics and ("评价指标/对比结果" in out):
        out.remove("评价指标/对比结果")
        out.insert(0, "评价指标/对比结果")
    if wants_compare and ("评价指标/对比结果" in out) and out[0] != "评价指标/对比结果":
        try:
            out.remove("评价指标/对比结果")
        except Exception:
            pass
        out.insert(0, "评价指标/对比结果")

    # Add anchor terms to the encoding aspect for better specificity.
    if any(x.startswith("编码/采样策略") for x in out):
        anchors = []
        for k, lab in [
            ("hadamard", "Hadamard"),
            ("fourier", "Fourier"),
            ("dmd", "DMD"),
            ("frequency-division", "频分复用"),
            ("multiplex", "复用"),
            ("metamaterial", "超材料"),
            ("metasurface", "超表面"),
            ("speckle", "散斑"),
        ]:
            if k in text and lab not in anchors:
                anchors.append(lab)
        if anchors:
            for i, v in enumerate(out):
                if v.startswith("编码/采样策略"):
                    out[i] = "编码/采样策略（" + "、".join(anchors[:2]) + "）"
                    break

    # If user asks definition-like questions, ensure '概念定义' appears when possible.
    if re.search(r"(是什么|定义|含义|概念|what is|definition)", prompt_text or "", flags=re.I):
        if "概念定义/问题设定" not in out:
            out.insert(0, "概念定义/问题设定")

    return out[:4]

def _score_tokens(text: str, query_tokens: list[str]) -> float:
    toks = tokenize(text or "")
    if not toks:
        return 0.0
    if not query_tokens:
        return 0.0
    ct = Counter(toks)
    return float(sum(ct.get(t, 0) for t in query_tokens))
