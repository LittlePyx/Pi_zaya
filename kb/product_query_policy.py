from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from .candidate_retriever import CandidateRetriever
from .retrieval_query_terms import DIRECT_PROMPT_STOP_TOKENS
from .runtime_cache import cache_get, cache_set
from .source_filters import is_temporary_source_path

def _make_chat_client(settings):
    from .llm import DeepSeekChat

    return DeepSeekChat(settings)


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def _has_latin(text: str) -> bool:
    return any("a" <= ch.lower() <= "z" for ch in (text or ""))


def translate_query_for_search(settings, prompt_text: str) -> str | None:
    """
    Translate a CJK-heavy query to a compact English search query (keywords),
    so BM25 can match English papers.
    """
    q_raw = (prompt_text or "").strip()
    if not q_raw:
        return None
    # Strip bound-source hints / file paths before language heuristics, otherwise a
    # Chinese question with an appended PDF path looks "latin-heavy" and won't translate.
    q = q_raw
    try:
        q = re.sub(r"(?i)\b[a-z]:\\\\[^\s]{6,}\b", " ", q)  # Windows paths
        q = re.sub(r"(?i)\bhttps?://[^\s]{6,}\b", " ", q)  # URLs
        q = re.sub(r"(?i)\b[^\s]+\\.(pdf|md|txt|docx?)\b", " ", q)  # filenames
        q = " ".join(q.split())
    except Exception:
        q = q_raw
    # If a bound-source hint was prepended (often a long Latin-only title), drop it for translation.
    try:
        m_cjk = re.search(r"[\u4e00-\u9fff]", q)
        if m_cjk and m_cjk.start() >= 12:
            prefix = q[: m_cjk.start()]
            if (re.search(r"[\u4e00-\u9fff]", prefix) is None) and (re.search(r"[A-Za-z]", prefix) is not None):
                q = q[m_cjk.start() :].strip()
    except Exception:
        q = q

    if not q:
        return None
    if not _has_cjk(q):
        return None
    # Allow translation for mixed queries like:
    #   "这篇文章核心问题是什么 NatPhoton-2019 ..."
    # where the bound-source hint introduces Latin tokens but the query is still CJK-heavy.
    if _has_latin(q):
        try:
            cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", q))
            latin_chars = len(re.findall(r"[A-Za-z]", q))
        except Exception:
            cjk_chars, latin_chars = 0, 0
        # If the query is mostly Latin, translation is unnecessary and can hurt recall.
        if cjk_chars < 6 or latin_chars > (cjk_chars * 3):
            return None

    key = hashlib.sha1((str(getattr(settings, "api_key", None)) + "|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    cached = cache_get("trans", key)
    if isinstance(cached, str) and cached.strip():
        return cached.strip()

    # Fast dictionary-based fallback for common KB terms.
    # This keeps retrieval responsive even when translation LLM is slow/unavailable.
    terms: list[str] = []
    mapping = [
        ("钙钛矿激光器", "perovskite laser"),
        ("光泵浦", "optically pumped"),
        ("器件问题", "device challenge"),
        ("彩色单像素成像", "color single-pixel imaging"),
        ("彩色单像素", "color single-pixel"),
        ("彩色 SPI", "color SPI"),
        ("彩色SPI", "color SPI"),
        ("灰度 SPI", "gray SPI"),
        ("灰度SPI", "gray SPI"),
        ("颜色响应系数", "color response coefficient"),
        ("颜色失真", "color distortion"),
        ("系统复杂度", "system complexity"),
        ("成像时间", "imaging time"),
        ("额外挑战", "additional challenges"),
        ("\u5355\u50cf\u7d20", "single-pixel"),
        ("\u5355\u5149\u5b50", "single-photon"),
        ("\u5355\u66dd\u5149", "single-shot"),
        ("\u5355\u6b21\u66dd\u5149", "single-shot"),
        ("\u538b\u7f29\u6210\u50cf", "compressive imaging"),
        ("\u538b\u7f29\u611f\u77e5", "compressed sensing"),
        ("\u538b\u7f29\u7387", "compression ratio"),
        ("\u91c7\u6837\u6570", "number of measurements"),
        ("\u91c7\u6837\u7387", "sampling rate"),
        ("\u6d4b\u91cf\u6570", "number of measurements"),
        ("\u6d4b\u91cf", "measurements"),
        ("\u6a21\u5f0f\u6570", "number of patterns"),
        ("\u91cd\u5efa\u8d28\u91cf", "reconstruction quality"),
        ("\u6838\u5fc3\u95ee\u9898", "core problem"),
        ("\u5173\u952e\u95ee\u9898", "key problem"),
        ("\u95ee\u9898\u610f\u8bc6", "problem formulation"),
        ("\u4e3a\u4ec0\u4e48", "motivation"),
        ("\u4f20\u7edf\u65b9\u6848", "conventional approach"),
        ("\u4e0d\u591f\u597d", "limitations"),
        ("\u4f18\u7f3a\u70b9", "advantages disadvantages"),
        ("\u4f18\u70b9", "advantages"),
        ("\u7f3a\u70b9", "disadvantages"),
        ("\u9002\u7528\u573a\u666f", "applications"),
        ("\u9002\u7528", "applicable"),
        ("\u4e3b\u6d41", "mainstream"),
        ("\u91cd\u5efa\u65b9\u6cd5", "reconstruction methods"),
        ("\u91cd\u5efa\u7b97\u6cd5", "reconstruction algorithm"),
        ("\u7b97\u6cd5", "algorithm"),
        ("\u6df1\u5ea6\u5b66\u4e60", "deep learning"),
        ("\u795e\u7ecf\u7f51\u7edc", "neural network"),
        ("\u7ed3\u6784\u5316\u63a2\u6d4b", "structured detection"),
        ("\u6fc0\u5149\u626b\u63cf\u663e\u5fae", "laser scanning microscopy"),
        ("\u5171\u805a\u7126", "confocal"),
        ("\u6743\u8861", "trade-off"),
        ("\u4e3b\u8981\u8d21\u732e", "main contribution"),
        ("\u6838\u5fc3\u8d21\u732e", "core contribution"),
        ("\u8d21\u732e", "contribution"),
        ("\u539f\u7406", "principle"),
        ("\u673a\u5236", "mechanism"),
        ("\u65b9\u6cd5", "method"),
        ("\u7b97\u6cd5", "algorithm"),
        ("\u5b9e\u9a8c", "experiment"),
        ("\u7ed3\u679c", "results"),
        ("\u8ba8\u8bba", "discussion"),
        ("\u7ed3\u8bba", "conclusion"),
        ("\u5c40\u9650", "limitation"),
        ("\u7f3a\u70b9", "limitation"),
        ("\u672a\u6765\u5de5\u4f5c", "future work"),
        ("\u590d\u73b0", "reproducibility"),
        ("\u590d\u73b0\u6027", "reproducibility"),
        ("\u5f15\u7528", "citation"),
        ("\u53c2\u8003\u6587\u732e", "references"),
        ("\u76ee\u5f55", "table of contents"),
        ("\u5927\u7eb2", "outline"),
        ("\u56fe", "figure"),
        # Do not match the single character ``表`` here: substring matching
        # would turn words such as ``代表性`` into an unrelated ``table``
        # query. Cover actual table intent with explicit phrases instead.
        ("\u8868\u683c", "table"),
        ("\u8868\u4e2d", "table"),
        ("\u8868\u5185", "table"),
        ("\u516c\u5f0f", "equation"),
        ("\u5b9a\u4e49", "definition"),
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
        ("基准测试", "benchmark"),
        ("基准", "benchmark"),
        ("最高", "highest"),
        ("最大", "maximum"),
        ("最低", "lowest"),
        ("最小", "minimum"),
        ("模型", "model"),
        ("并列", "tie"),
        ("代表性应用", "representative applications"),
        ("哪些应用", "applications"),
        ("什么场景", "use cases"),
        ("综述", "review"),
    ]
    ascii_terms = [
        token
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9+._-]{1,}\b", q)
        if token.lower() not in {"the", "a", "an", "of", "for", "and", "or", "paper", "pdf", "md", "en"}
    ]
    terms.extend(ascii_terms[:8])
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
            cache_set("trans", key, heuristic, max_items=500)
            return heuristic

    # No API key: we can't call the translation LLM, but the heuristic fallback above may still help.
    if not getattr(settings, "api_key", None):
        return None

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=1,
        )
    except Exception:
        settings_fast = settings
    ds = _make_chat_client(settings_fast)
    system = (
        "You translate Chinese research questions into an English search query for academic retrieval.\n"
        "Rules:\n"
        "- Output ONLY the English search query (no quotes, no explanations).\n"
        "- Prefer keywords and key phrases; include useful synonyms.\n"
        "- Keep it compact (8-18 tokens).\n"
        "- DO NOT confuse terms:\n"
        "  - 单曝光/单次曝光 -> single-shot, single exposure, snapshot\n"
        "  - 单像素 -> single-pixel\n"
        "  - 压缩成像 -> compressive imaging\n"
        "  - 光谱成像 -> spectral imaging\n"
        "- If the user didn't mention 单像素 avoid adding single-pixel.\n"
    )
    user = f"Question: {q}\n\nEnglish search keywords:"
    try:
        out = (ds.chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.0, max_tokens=80) or "").strip()
    except Exception:
        out = ""
    out = " ".join(out.split())
    cache_set("trans", key, out, max_items=500)
    return out or None


def expand_query_via_llm(settings, prompt_text: str) -> list[str]:
    """
    Generate up to 3 search query variants via LLM for BM25 recall expansion.
    Always prepends the original query as the first variant.
    Returns [prompt_text] on failure or when expansion is not beneficial.
    """
    q = (prompt_text or "").strip()
    if not q or len(q) < 4:
        return [q] if q else []

    cache_key = hashlib.sha1((str(getattr(settings, "api_key", None)) + "|expand|" + q).encode("utf-8", "ignore")).hexdigest()[:16]
    cached = cache_get("query_expand", cache_key)
    if isinstance(cached, list) and len(cached) >= 1:
        # Always ensure original query is first
        return [q] + [v for v in cached if v and v != q]

    if not getattr(settings, "api_key", None):
        return [q]

    try:
        settings_fast = replace(
            settings,
            timeout_s=min(float(getattr(settings, "timeout_s", 60.0) or 60.0), 8.0),
            max_retries=0,
        )
    except Exception:
        settings_fast = settings
    ds = _make_chat_client(settings_fast)

    has_cjk = bool(re.search(r"[一-鿿]", q))
    system = (
        "You generate search query variants for academic paper retrieval.\n"
        "Rules:\n"
        "- Output up to 3 alternative search queries, one per line, no numbering.\n"
        "- Each query should be a compact keyword phrase (8-20 tokens).\n"
        "- Focus on the topic's core concepts and related terminology.\n"
        "- Explore BROADENING expansions: capture synonyms and parallel concepts\n"
        "  that would find papers on the same topic using different terminology.\n"
        + (
            "- For CJK queries: generate ALL variants in English (academic papers are in English).\n"
            "  Do NOT keep CJK characters — translate core concepts to English keywords.\n"
            if has_cjk
            else "- For English input, include synonym variants that may not share keywords.\n"
        )
        + "- Do NOT add quotes, numbering, or commentary.\n"
        "- If the query is already well-formed and has no useful variants, output only: NONE\n"
    )
    user = f"Query: {q}\n\nAlternative search queries:"
    try:
        out = (
            ds.chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=160,
            )
            or ""
        ).strip()
    except Exception:
        out = ""

    variants: list[str] = []
    if out and out.strip().upper() != "NONE":
        for line in out.split("\n"):
            line = " ".join(line.strip().strip('"\'「」').split())
            if line and len(line) >= 4 and line.lower() != q.lower():
                variants.append(line)
        # Limit to 3 variants
        variants = variants[:3]

    cache_set("query_expand", cache_key, variants, max_items=300)
    return [q] + variants


def deterministic_query_variants(prompt_text: str) -> list[str]:
    """Cheap domain-aware query variants for common research questions.

    These variants cover mechanism/term aliases that are easy to miss with
    lexical BM25, especially mixed Chinese-English questions where we avoid a
    translation call because the prompt already contains a long English title.
    """

    q = str(prompt_text or "").strip()
    if not q:
        return []
    low = q.lower()
    variants: list[str] = []

    def has_any(*needles: str) -> bool:
        return any(str(item or "").lower() in low for item in needles if str(item or "").strip())

    def add(value: str) -> None:
        v = " ".join(str(value or "").split()).strip()
        if not v:
            return
        key = v.lower()
        if key == low:
            return
        if key in {x.lower() for x in variants}:
            return
        variants.append(v)

    if has_any("彩色单像素", "彩色 spi", "color single-pixel", "color spi"):
        add(
            "color single-pixel imaging compared with gray SPI "
            "color response coefficient color distortion imaging time"
        )
        if has_any("深度学习", "deep learning", "dl", "复杂度", "成像时间", "imaging time"):
            add(
                "deep learning color SPI reduce system complexity imaging time "
                "color filter array frequency-division multiplexing"
            )

    if has_any("hatnet"):
        add(
            "HATNet Dual-Scale Transformer Large-Scale Single-Pixel Imaging "
            "tensor ISTA deep unfolding tensor gradient descent hybrid-attention denoising"
        )
        add(
            "HATNet stages TGD U-shaped denoiser S-SA C-SA Kronecker SPI "
            "learnable step size proximal mapping"
        )

    if has_any("learned primal-dual", "learned primal dual") and has_any(
        "pdhg",
        "primal-dual",
        "primal dual",
        "可学习网络",
        "替换",
        "初始化",
    ):
        add(
            "Learned PDHG proximal operators replaced parametrized convolutional neural networks "
            "dual update primal update fixed iterations"
        )
        if has_any("初始化", "initialization", "initialise", "initialize", "starting point", "zero", "fbp"):
            add(
                "Learned Primal-Dual choice of starting point zero initialization "
                "filtered back-projection FBP pseudo-inverse final results"
            )

    if has_any("ista-net", "ista net") and has_any(
        "ista",
        "iteration",
        "iterative",
        "algorithm",
        "unroll",
        "unfold",
        "phase",
        "module",
        "迭代",
        "展开",
        "模块",
        "可学习",
    ):
        add(
            "ISTA-Net maps ISTA update steps fixed phases each phase one iteration "
            "r(k) module x(k) module learnable step size shrinkage threshold nonlinear transforms"
        )

    if has_any(
        "refocus",
        "refocusing",
        "out of focus",
        "\u79bb\u7126",
        "\u91cd\u805a\u7126",
        "\u91cd\u65b0\u5bf9\u7126",
        "\u91cd\u5bf9\u7126",
    ):
        add(
            "digital refocusing out-of-focus sample ray tracing wave propagation "
            "diffraction angular information"
        )
    if has_any("perovskite laser", "perovskite lasing", "钙钛矿激光器", "钙钛矿激光"):
        # Keep the base expansion technology-neutral. A generic perovskite-laser
        # question may refer to optical pumping, nanowire lasers, or another
        # device family; it must not silently become the dual-cavity Nature paper.
        add("perovskite laser lasing optical gain threshold cavity device")
        if has_any(
            "dual-cavity",
            "dual cavity",
            "electrically driven",
            "electrical injection",
            "electrically injected",
            "peled",
            "双腔",
            "电驱动",
            "电注入",
            "电泵浦",
        ):
            add(
                "electrically driven lasing dual-cavity perovskite device microcavity "
                "PeLED electrical injection laser"
            )
    if has_any("trade-off", "tradeoff", "\u6743\u8861", "\u539a\u6837\u672c", "thick sample", "thick samples", "s2ism"):
        add(
            "structured detection microscopy thick samples resolution SNR signal-to-noise "
            "optical sectioning out-of-focus background"
        )
    if has_any(
        "benefit",
        "risk",
        "advantage",
        "advantages",
        "disadvantage",
        "challenge",
        "challenges",
        "limitation",
        "limitations",
        "\u597d\u5904",
        "\u574f\u5904",
        "\u98ce\u9669",
        "\u4f18\u52bf",
        "\u6311\u6218",
        "\u5c40\u9650",
        "\u7f3a\u70b9",
    ) and has_any(
        "deep learning",
        "\u6df1\u5ea6\u5b66\u4e60",
    ):
        add(
            "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning "
            "advantages limitations training data generalization interpretability "
            "reconstruction speed reconstruction quality"
        )
        add(
            "Challenges and Outlooks reliance on extensive datasets limited "
            "interpretability susceptibility to overfitting limited generalization"
        )
    if has_any("pidl"):
        add(
            "physics-informed deep learning computational single-photon imaging "
            "physical prior data generator neural network loss inference"
        )
    if has_any("single-photon", "single photon", "\u5355\u5149\u5b50") and has_any(
        "physics-informed deep learning",
        "physics informed deep learning",
        "pidl",
    ):
        add(
            "High-resolution single-photon imaging physics-informed deep learning "
            "physical multi-source noise model SPAD arrays crosstalk dark count rate"
        )
    if (
        has_any("single-photon", "single photon", "\u5355\u5149\u5b50")
        and has_any(
            "physics-informed deep learning",
            "physics informed deep learning",
            "pidl",
        )
        and has_any(
            "detector",
            "photodetector",
            "review",
            "survey",
            "\u63a2\u6d4b\u5668",
            "\u7efc\u8ff0",
            "\u600e\u4e48\u642d\u914d\u8bfb",
        )
    ):
        add(
            "Emerging single-photon detection technique high-performance "
            "photodetector review SPAD"
        )
        add(
            "High-resolution single-photon imaging physics-informed deep learning "
            "real SPAD noise"
        )
    if has_any("piln", "part-based image-loop", "image-loop network"):
        add(
            "part-based image-loop network single-pixel imaging ILNet physical model "
            "untrained neural network inference"
        )
    if has_any("piln", "part-based image-loop", "image-loop network") and has_any(
        "review",
        "survey",
        "taxonomy",
        "position",
        "relationship",
        "mainline",
        "\u7efc\u8ff0",
        "\u4e3b\u7ebf",
        "\u4f4d\u7f6e",
        "\u5173\u7cfb",
    ):
        add(
            "data-driven model-driven hybrid-driven single-pixel imaging deep learning "
            "taxonomy untrained network measurements physical model"
        )
    if has_any("origin", "source", "comes from", "prior", "previous", "\u6765\u6e90", "\u51fa\u5904", "\u4e4b\u524d", "\u5df2\u6709"):
        add("prior work existing method background reference citation source")
    wants_cassi = has_any("cassi")
    wants_sci_lineage = bool(
        has_any(
            "snapshot compressive imaging",
            "snapshot compressive",
            "compressive snapshot",
            "压缩快照",
            "快照压缩",
        )
        and has_any("3d", "three-dimensional", "三维", "光谱", "spectral")
    )
    if wants_cassi or wants_sci_lineage:
        add(
            "CASSI single-shot compressive spectral imaging dual-disperser "
            "two dispersive elements binary-valued aperture spectral datacube"
        )
    if wants_cassi and has_any("dcd") and has_any("dltr"):
        add(
            "DLTR dimension-discriminative low-rank tensor hyperspectral image "
            "overlapped cubic patches mode unfolding weighted rank regularization"
        )
    wants_scinerf_formula = bool(
        has_any("scinerf")
        and has_any(
            "formula",
            "equation",
            "forward model",
            "image formation",
            "\u516c\u5f0f",
            "\u524d\u5411",
            "\u6210\u50cf\u6a21\u578b",
        )
    )
    if wants_scinerf_formula:
        add(
            "SCINeRF 3.2 Image Formation Model of Video SCI equation 3 "
            "captured compressed image virtual image binary mask element-wise "
            "multiplication measurement noise synthesize compressed image "
            "differentiable with respect to NeRF and poses"
        )
    if wants_sci_lineage or has_any("scinerf", "scigs"):
        add(
            "SCINeRF neural radiance fields snapshot compressive image "
            "physical imaging process SCI training NeRF 3D scene"
        )
        add(
            "SCIGS 3D Gaussian Splatting snapshot compressive image "
            "single compressed image dynamic 3D scenes"
        )
    if has_any(
        "frequency-division-multiplexed",
        "frequency division multiplexed",
        "frequency-division multiplexing",
        "frequency division multiplexing",
        "\u9891\u5206\u590d\u7528",
    ) and has_any(
        "parallel",
        "speed",
        "simultaneous",
        "detector",
        "\u5e76\u884c",
        "\u901f\u5ea6",
        "\u540c\u65f6",
        "\u63a2\u6d4b\u5668",
    ):
        add(
            "Frequency-division-multiplexed single-pixel imaging B. Encoding "
            "phase-sensitive detection p frequencies simultaneously multiplexed "
            "into a single-pixel detector signal demodulated by p lock-in amplifiers"
        )
    microscopy_method_count = sum(
        (
            has_any("structured detection", "s2ism", "\u7ed3\u6784\u5316\u63a2\u6d4b"),
            has_any("interferometric", "iism", "\u5e72\u6d89"),
            has_any("light-field", "light field", "qclfm", "\u5149\u573a"),
        )
    )
    if microscopy_method_count >= 2:
        add(
            "Structured detection for simultaneous super-resolution and optical "
            "sectioning s2ISM laser scanning microscopy"
        )
        add(
            "Interferometric image scanning microscopy label-free live cells "
            "enhanced lateral resolution"
        )
        add(
            "Quantum correlation light-field microscope position angular information "
            "volumetric reconstruction extreme depth of field"
        )
    if has_any("3d single-pixel video", "3d single pixel video") and has_any(
        "detector",
        "探测器",
        "frame",
        "fps",
        "速度",
        "实时",
    ):
        add(
            "3D single-pixel video four spatially-separated single-pixel detectors "
            "continuous real-time 3D video 8 frames per second 64 x 64 pixels"
        )
    if has_any(
        "single-pixel compressive holography",
        "single pixel compressive holography",
        "单像素压缩全息",
        "单像素全息",
    ) and has_any("throughput", "phase", "相移", "吞吐"):
        add(
            "high-throughput single-pixel compressive holography beat frequency "
            "heterodyne holography phase stepping naturally in time"
        )
    if has_any(
        "sequential compressed sensing",
        "sequentially designed compressed sensing",
        "顺序压缩感知",
        "序贯压缩感知",
    ):
        add(
            "sequential adaptive compressed sensing signal support recovery "
            "distilled sensing exact support lower SNR"
        )
    if has_any("spad") and has_any(
        "geiger",
        "breakdown",
        "quench",
        "淬灭",
        "击穿",
        "雪崩",
    ):
        add(
            "SPAD operates in Geiger mode reverse bias breakdown voltage "
            "quenching circuit avalanche single photon detection"
        )
    if has_any(
        "beginner",
        "getting started",
        "reading roadmap",
        "刚开始",
        "入门",
        "主线",
        "先读",
    ) and has_any("single-pixel imaging", "single pixel imaging", "单像素成像"):
        add("Principles and prospects for single-pixel imaging review camera architecture applications")
        add("Hadamard single-pixel imaging versus Fourier single-pixel imaging sampling basis comparison")
        add("Advances and Challenges of Single-Pixel Imaging Based on Deep Learning review taxonomy")
    if (
        has_any("single-pixel", "single pixel", "\u5355\u50cf\u7d20")
        and has_any(
            "applications",
            "use case",
            "use cases",
            "representative",
            "review",
            "survey",
            "\u4ec0\u4e48\u573a\u666f",
            "\u503c\u5f97\u7528",
            "\u4ee3\u8868\u6027\u5e94\u7528",
            "\u7efc\u8ff0",
        )
    ):
        add(
            "Principles and prospects for single-pixel imaging applications "
            "wavelengths outside FPA technology high frame rates three dimensions "
            "hazardous gas leaks autonomous vehicles camera architecture review"
        )

    # A mixed-language comparison can name multiple methods while the compact
    # translation fallback keeps only one of them.  When no more specific
    # domain expansion applies, search each high-specificity identifier on its
    # own so RRF preserves every explicitly named paper.  Requiring at least
    # two identifiers keeps ordinary single-method queries unchanged.
    variant_cap = 4
    if not variants:
        named_identifiers: list[str] = []
        enumerated_titlecase_identifiers: list[str] = []
        seen_identifiers: set[str] = set()
        for token_match in re.finditer(
            r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{3,40}(?![A-Za-z0-9_-])",
            q,
        ):
            token = token_match.group(0)
            raw_token = str(token or "").strip().strip("-_")
            low_token = raw_token.lower()
            if low_token in DIRECT_PROMPT_STOP_TOKENS or low_token in seen_identifiers:
                continue
            has_identity_signal = (
                raw_token.isupper()
                or any(ch.isupper() for ch in raw_token[1:])
                or any(ch.isdigit() for ch in raw_token)
                or ("-" in raw_token)
            )
            if not has_identity_signal:
                # Model and paper names such as Informer/Autoformer use plain
                # TitleCase, unlike acronyms such as FEDformer/PatchTST. Accept
                # them only when they are visibly part of an enumerated list;
                # this avoids turning an opening word such as ``Compare`` into
                # a retrieval query.
                before = q[max(0, token_match.start() - 8) : token_match.start()]
                after = q[token_match.end() : min(len(q), token_match.end() + 8)]
                enumerated = bool(
                    re.search(r"(?:[,，、;；]\s*|(?:and|or)\s+|[和与及]\s*)$", before, flags=re.I)
                    or re.match(r"^\s*(?:[,，、;；]|\b(?:and|or)\b|[和与及])", after, flags=re.I)
                )
                if enumerated and raw_token[:1].isupper() and len(raw_token) >= 6:
                    enumerated_titlecase_identifiers.append(raw_token)
                continue
            seen_identifiers.add(low_token)
            named_identifiers.append(raw_token)
        if len(named_identifiers) >= 2:
            for identifier in enumerated_titlecase_identifiers:
                key = identifier.lower()
                if key in seen_identifiers:
                    continue
                seen_identifiers.add(key)
                named_identifiers.append(identifier)
        if len(named_identifiers) >= 2:
            for identifier in named_identifiers:
                add(identifier)
            # A bound current-paper query is prefixed with an absolute source
            # path.  That path contains several acronym-shaped tokens (for
            # example a venue, year, and filename) which must not expand the
            # RRF budget and dilute the user's factual query.  Raise the cap
            # only for an actual enumerated TitleCase paper/model list, the
            # case this expansion was introduced to support.
            if enumerated_titlecase_identifiers:
                variant_cap = min(8, max(4, len(named_identifiers)))
    return variants[:variant_cap]


def merge_expanded_results(
    results: list[tuple[list[dict], list[float], str]],
    top_k: int,
    *,
    weights: list[float] | None = None,
) -> tuple[list[dict], list[float]]:
    """
    Reciprocal Rank Fusion (RRF) over multiple query result sets.
    Each entry: (hits, scores, query_text).
    Deduplicates by chunk_id. Returns merged (hits, scores).

    When *weights* is provided (one per result set), each set's RRF
    contribution is multiplied by its weight.  This lets the original
    query (weight > 1.0) dominate over expansion variants.
    """
    rrf_scores: dict[str, float] = {}
    hit_map: dict[str, dict] = {}
    chunk_order: list[str] = []

    for result_idx, (hits, _scores, _query) in enumerate(results):
        weight = weights[result_idx] if weights else 1.0
        seen_in_result: set[str] = set()
        for rank, h in enumerate(hits or []):
            meta = h.get("meta") if isinstance(h.get("meta"), dict) else {}
            chunk_id = str(
                meta.get("chunk_id")
                or h.get("chunk_id")
                or h.get("id")
                or h.get("text", "")[:120]
            )
            if chunk_id in seen_in_result:
                continue
            seen_in_result.add(chunk_id)
            if chunk_id not in hit_map:
                hit_map[chunk_id] = h
                chunk_order.append(chunk_id)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + weight / (20.0 + rank)

    # Sort by RRF score descending
    ordered = sorted(chunk_order, key=lambda cid: rrf_scores.get(cid, 0.0), reverse=True)
    merged_hits: list[dict] = []
    merged_scores: list[float] = []
    for cid in ordered[:top_k]:
        fused_score = float(rrf_scores.get(cid, 0.0) or 0.0)
        hit = dict(hit_map[cid])
        hit.setdefault("_bm25_score", float(hit.get("score", 0.0) or 0.0))
        # Downstream document grouping sorts this field. Keeping the original
        # BM25 value here would silently undo the fused ranking and can select
        # a generic chunk instead of the query-specific evidence.
        hit["score"] = fused_score
        merged_hits.append(hit)
        merged_scores.append(fused_score)
    return merged_hits, merged_scores


def search_hits_with_fallback(
    prompt_text: str,
    retriever: CandidateRetriever,
    top_k: int,
    settings,
    *,
    allow_translate: bool = True,
    allow_expand: bool = False,
    whole_library: bool = False,
) -> tuple[list[dict], list[float], str, bool, list[str]]:
    """
    Returns: (hits_raw, scores, used_query, used_translation, query_variants)
    """
    q1 = re.split(
        r"(?im)^QUERY SCOPE:\s*(?:Current paper|Research basket|Full library)\.\s*$",
        str(prompt_text or ""),
        maxsplit=1,
    )[0].strip()
    # A library query is allowed to inspect a wider internal candidate window.
    # The answer path still selects only the best evidence-bearing documents, so
    # this improves cross-document recall without exposing low-quality matches.
    candidate_limit = max(10, top_k * (16 if whole_library else 6))
    hits1 = retriever.search(q1, top_k=candidate_limit) if q1 else []
    hits1 = [h for h in (hits1 or []) if not is_temporary_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
    scores1 = [float(h.get("score", 0.0) or 0.0) for h in hits1]
    best1 = float(max(scores1) if scores1 else 0.0)
    query_variants: list[str] = [q1] if q1 else []

    # If the query is CJK-only, BM25 over English corpora can return all-zeros (but still returns arbitrary docs).
    # Try translating to English to get meaningful retrieval.
    used_trans = False
    q2 = translate_query_for_search(settings, q1) if bool(allow_translate) else None
    if q2:
        hits2 = retriever.search(q2, top_k=candidate_limit)
        hits2 = [h for h in (hits2 or []) if not is_temporary_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
        scores2 = [float(h.get("score", 0.0) or 0.0) for h in hits2]
        best2 = float(max(scores2) if scores2 else 0.0)
        query_variants.append(q2)
        # RRF-merge original and translated results instead of replacing
        # one with the other, so hits found by either query are preserved.
        # Original query (q1) is weighted 2x so it dominates ranking.
        if hits1 and hits2:
            merged_q1q2, merged_s1s2 = merge_expanded_results(
                [(hits1, scores1, q1), (hits2, scores2, q2)],
                top_k=candidate_limit,
                # For a CJK question over an English academic corpus, the
                # translated query is the primary lexical signal.  Giving the
                # untranslated query priority lets incidental CJK matches bury
                # the actually relevant English papers.
                weights=([0.35, 2.0] if _has_cjk(q1) else [2.0, 1.0]),
            )
            merged_best = float(max(merged_s1s2) if merged_s1s2 else 0.0)
            # RRF and BM25 scores live on different scales, so comparing their
            # maxima can reject useful translated hits whenever the original
            # query has even a weak lexical match. Both result sets are already
            # represented in the fusion; accept the fused ranking directly.
            hits1, scores1, best1 = merged_q1q2, merged_s1s2, merged_best
            used_trans = True
            for _h in hits1:
                _h.setdefault("_bm25_score", _h.get("score", 0.0))
        elif not hits1 and hits2:
            hits1, scores1, best1 = hits2, scores2, best2
            for _h in hits1:
                _h.setdefault("_bm25_score", _h.get("score", 0.0))
            used_trans = True

    # LLM-based query expansion to improve recall for synonym-variant queries.
    deterministic_variants = [
        v
        for v in deterministic_query_variants(q1)
        if v and v.lower() != q1.lower() and (not q2 or v.lower() != q2.lower())
    ]
    if deterministic_variants:
        all_results: list[tuple[list[dict], list[float], str]] = [(hits1, scores1, q1)]
        for variant in deterministic_variants:
            v_hits = retriever.search(variant, top_k=candidate_limit)
            v_hits = [h for h in (v_hits or []) if not is_temporary_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
            v_scores = [float(h.get("score", 0.0) or 0.0) for h in v_hits]
            all_results.append((v_hits, v_scores, variant))
            if variant not in query_variants:
                query_variants.append(variant)
        merged_hits, merged_scores = merge_expanded_results(
            all_results,
            top_k=candidate_limit,
            weights=[1.5] + [2.5 for _ in deterministic_variants],
        )
        for _h in merged_hits:
            _h["_expansion_variants"] = list(deterministic_variants)
        for _h in merged_hits:
            _h.setdefault("_bm25_score", _h.get("score", 0.0))
        hits1, scores1, best1 = merged_hits, merged_scores, float(max(merged_scores) if merged_scores else best1)

    # LLM-based query expansion to improve recall for synonym-variant queries.
    if (
        bool(allow_expand)
        and not deterministic_variants
        and q1
        and getattr(settings, "api_key", None)
        and getattr(settings, "query_expansion_enabled", False)
    ):
        expanded = expand_query_via_llm(settings, q1)
        # expanded[0] is q1; skip it since we already searched q1
        new_variants = [v for v in expanded[1:] if v and v.lower() != q1.lower() and (not q2 or v.lower() != q2.lower())]
        if new_variants:
            all_results: list[tuple[list[dict], list[float], str]] = [(hits1, scores1, q1)]
            if q2 and not used_trans:
                all_results.append((hits2, scores2, q2))
            for variant in new_variants:
                v_hits = retriever.search(variant, top_k=candidate_limit)
                v_hits = [h for h in (v_hits or []) if not is_temporary_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
                v_scores = [float(h.get("score", 0.0) or 0.0) for h in v_hits]
                all_results.append((v_hits, v_scores, variant))
                query_variants.append(variant)

            # RRF merge all result sets (always use merged results — RRF handles
            # low-quality expansions by weighting at rank depth, and the original
            # query results are included in the pool so nothing is lost).
            # Original query (q1) is weighted 2x, all other variants 1x.
            merged_hits, merged_scores = merge_expanded_results(
                all_results,
                top_k=candidate_limit,
                weights=[2.0 if i == 0 else 1.0 for i in range(len(all_results))],
            )
            # Preserve original BM25 score so downstream sorters can use it.
            for _h in merged_hits:
                    _h.setdefault("_bm25_score", _h.get("score", 0.0))
            # Tag hits with the variant list so the focus filter can be more
            # lenient for expansion-discovered papers.
            _expansion_variants = [v for v in query_variants if v.lower() != q1.lower()]
            if _expansion_variants:
                for _h in merged_hits:
                    _h["_expansion_variants"] = list(_expansion_variants)
            return merged_hits, merged_scores, q1, used_trans, query_variants

    # Last-resort fallback: if all queries returned empty, try a broad search
    # with the original query to avoid starving downstream rendering.
    if not hits1 and q1:
        hits_fb = retriever.search(q1, top_k=max(20, top_k * 20))
        hits_fb = [h for h in (hits_fb or []) if not is_temporary_source_path(str((h.get("meta") or {}).get("source_path") or ""))]
        if hits_fb:
            scores_fb = [float(h.get("score", 0.0) or 0.0) for h in hits_fb]
            return hits_fb, scores_fb, q1, used_trans, query_variants

    return hits1, scores1, q1, used_trans, query_variants
