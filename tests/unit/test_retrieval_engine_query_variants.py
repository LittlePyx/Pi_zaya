from __future__ import annotations

from types import SimpleNamespace

from kb.retrieval_engine import (
    _anchor_text_bonus,
    _deterministic_query_variants,
    _extract_explicit_anchor_hint,
    _group_hits_by_doc_for_refs,
    _search_hits_with_fallback,
    _source_prompt_match_score,
    _translate_query_for_search,
)
from kb.retriever import BM25Retriever


def test_explicit_decimal_section_hint_is_preserved() -> None:
    hint = _extract_explicit_anchor_hint("请只依据第 5.2 节解释 MsGAN")

    assert hint["kind"] == "section"
    assert hint["number"] == 5
    assert hint["number_text"] == "5.2"
    assert _anchor_text_bonus("### 5.2. Imaging Through Scattering Media", hint) >= 25.0


def test_explicit_section_focus_selects_the_requested_passage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KB_DB_DIR", str(tmp_path))
    source = tmp_path / "LPR-2025-Advances and Challenges.en.md"
    source.write_text(
        "# Advances and Challenges\n\n"
        "## Abstract\nGeneral deep-learning overview.\n\n"
        "### 5.2. Imaging Through Scattering Media\n"
        "MsGAN clears low-quality computational ghost images under turbulence.\n\n"
        "### 5.3. Imaging at Photon-Level\nSingle-photon detector arrays.\n",
        encoding="utf-8",
    )
    chunks = [
        {
            "id": "abstract",
            "score": 40.0,
            "text": "General deep-learning overview.",
            "meta": {"source_path": str(source), "heading_path": "Advances and Challenges / Abstract", "page_start": 1},
        },
        {
            "id": "section",
            "score": 18.0,
            "text": "MsGAN clears low-quality computational ghost images under turbulence.",
            "meta": {
                "source_path": str(source),
                "heading_path": "Advances and Challenges / 5.2. Imaging Through Scattering Media",
                "page_start": 10,
            },
        },
    ]

    docs = _group_hits_by_doc_for_refs(
        chunks,
        "In LPR-2025-Advances and Challenges section 5.2, how does MsGAN handle turbulence?",
        2,
        deep_query="MsGAN turbulence section 5.2",
        deep_read=True,
        llm_rerank=False,
        settings=SimpleNamespace(api_key=None),
    )

    assert "MsGAN" in docs[0]["text"]
    assert docs[0]["meta"]["page_start"] == 10
    assert docs[0]["meta"]["anchor_target_label"] == "5.2"


def test_deterministic_query_variants_expand_refocus_mechanism_terms() -> None:
    variants = _deterministic_query_variants(
        "这个 quantum correlation light-field microscope 是怎么把离焦样品重新对焦的？"
    )
    joined = " ".join(variants).lower()

    assert "digital refocusing" in joined
    assert "ray tracing" in joined
    assert "wave propagation" in joined


def test_deterministic_query_variants_expand_learned_primal_dual_method_terms() -> None:
    variants = _deterministic_query_variants(
        "Learned Primal-Dual 如何把传统 PDHG 变成可学习网络？"
        "哪些更新被替换，为什么不需要 FBP 初始化？"
    )
    joined = " ".join(variants).lower()

    assert "proximal operators" in joined
    assert "dual update" in joined
    assert "zero initialization" in joined
    assert "pseudo-inverse" in joined


def test_deterministic_query_variants_cover_both_named_unrolled_networks() -> None:
    variants = _deterministic_query_variants(
        "比较 Learned Primal-Dual 与 ISTA-Net：各自把什么迭代步骤变成可学习模块？"
    )
    joined = "\n".join(variants).lower()

    assert "learned pdhg proximal operators" in joined
    assert "ista-net maps ista update steps" in joined
    assert "r(k) module" in joined
    assert "shrinkage threshold" in joined


def test_deterministic_query_variants_expand_thick_sample_tradeoff_terms() -> None:
    variants = _deterministic_query_variants("s2ISM 这篇说的 trade-off 是什么？为什么厚样本会麻烦？")
    joined = " ".join(variants).lower()

    assert "snr" in joined
    assert "optical sectioning" in joined
    assert "out-of-focus background" in joined


def test_query_translation_does_not_treat_representative_as_table_intent() -> None:
    translated = _translate_query_for_search(
        SimpleNamespace(api_key=None),
        "什么场景真的值得用单像素相机？这篇综述给了哪些代表性应用？",
    )

    assert translated
    assert "single-pixel" in translated
    assert "representative applications" in translated
    assert "table" not in translated.split()


def test_query_translation_keeps_explicit_table_intent() -> None:
    translated = _translate_query_for_search(
        SimpleNamespace(api_key=None),
        "表格中哪种方法的 PSNR 最高？",
    )

    assert translated
    assert "table" in translated.split()


def test_deterministic_query_variants_split_multi_method_microscopy_question() -> None:
    variants = _deterministic_query_variants(
        "显微成像这些 structured detection、interferometric、light-field 方法分别在解决什么麻烦？"
    )
    joined = "\n".join(variants).lower()

    assert "structured detection" in joined
    assert "interferometric image scanning microscopy" in joined
    assert "quantum correlation light-field microscope" in joined


def test_deterministic_query_variants_pair_detector_review_with_pidl() -> None:
    variants = _deterministic_query_variants(
        "单光子成像里，探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？"
    )
    joined = "\n".join(variants).lower()

    assert "photodetector review" in joined
    assert "physics-informed deep learning" in joined
    assert "real spad noise" in joined


def test_deterministic_query_variants_focus_single_photon_pidl_on_physical_noise_model() -> None:
    variants = _deterministic_query_variants(
        "physics-informed deep learning 在单光子成像里到底帮了什么？"
    )
    joined = "\n".join(variants).lower()

    assert "high-resolution single-photon imaging" in joined
    assert "physical multi-source noise model" in joined
    assert "spad arrays" in joined


def test_deterministic_query_variants_expand_single_pixel_application_review() -> None:
    variants = _deterministic_query_variants(
        "什么场景真的值得用单像素相机，而不是普通面阵相机？这篇综述给了哪些代表性应用？"
    )

    assert any(
        "principles and prospects for single-pixel imaging applications" in variant.lower()
        for variant in variants
    )
    joined = " ".join(variants).lower()
    assert "wavelengths outside fpa technology" in joined
    assert "high frame rates" in joined
    assert "three dimensions" in joined


def test_deterministic_query_variants_expand_piln_taxonomy_terms() -> None:
    variants = _deterministic_query_variants(
        "PILN 和综述里的深度学习单像素成像主线是什么关系？"
    )
    joined = " ".join(variants).lower()

    assert "data-driven" in joined
    assert "model-driven" in joined
    assert "hybrid-driven" in joined
    assert "taxonomy" in joined


def test_deterministic_query_variants_resolve_pidl_and_piln_as_distinct_papers() -> None:
    variants = _deterministic_query_variants(
        "比较 PIDL 与 PILN：训练阶段分别需要哪些物理先验，推理阶段输入输出是什么？"
    )

    assert any(
        "physics-informed deep learning computational single-photon imaging" in value.lower()
        for value in variants
    )
    assert any(
        "part-based image-loop network single-pixel imaging" in value.lower()
        for value in variants
    )


def test_pidl_piln_aliases_keep_both_named_sources_through_focus_matching() -> None:
    question = "比较 PIDL 与 PILN 的训练先验和推理输入输出"
    pidl_source = (
        "library/NatCommun-2023-Physics-informed deep learning for computational "
        "single-photon imaging.en.md"
    )
    piln_source = (
        "library/Optics-2024-Part-based image-loop network for single-pixel imaging.en.md"
    )

    assert _source_prompt_match_score(question, pidl_source) >= 7.5
    assert _source_prompt_match_score(question, piln_source) >= 7.5
    assert _source_prompt_match_score(question, "library/SCINeRF.en.md") == 0.0


def test_cassi_alias_matches_dual_disperser_title_without_literal_acronym() -> None:
    question = "CASSI 的双色散结构具体怎么摆，为什么中间要放二值孔径？"
    cassi_source = (
        "library/OE-2007-Single-shot compressive spectral imaging with "
        "a dual-disperser architecture.en.md"
    )

    assert _source_prompt_match_score(question, cassi_source) >= 7.5
    assert _source_prompt_match_score(
        question,
        "library/SCIGS-3D-Gaussians-Splatting-from-a-Snapshot-Compressive-Image.en.md",
    ) == 0.0


def test_deterministic_query_variants_split_cassi_dcd_and_dltr_facets() -> None:
    variants = _deterministic_query_variants(
        "CASSI 与 DCD 的观测模型有什么区别？DLTR 如何利用三维张量低秩性？"
    )
    joined = "\n".join(variants).lower()

    assert "dual-disperser" in joined
    assert "dimension-discriminative low-rank tensor" in joined
    assert "mode unfolding" in joined


def test_hatnet_alias_and_variants_resolve_dual_scale_transformer_source() -> None:
    question = (
        "ISTA-Net 和 HATNet 的深度展开架构如何把迭代算法变成可学习网络？"
    )
    source = (
        "library/CVPR-2024-Dual-Scale Transformer for Large-Scale "
        "Single-Pixel Imaging.en.md"
    )

    assert _source_prompt_match_score(question, source) >= 7.5
    variants = " ".join(_deterministic_query_variants(question)).lower()
    assert "dual-scale transformer" in variants
    assert "tensor gradient descent" in variants
    assert "s-sa" in variants and "c-sa" in variants


def test_deterministic_query_variants_expand_snapshot_compressive_3d_lineage() -> None:
    variants = _deterministic_query_variants(
        "SCI 或压缩快照成像这条线，是怎么从光谱成像走到 3D 场景重建的？"
    )
    joined = "\n".join(variants).lower()

    assert "dual-disperser" in joined
    assert "binary-valued aperture" in joined
    assert "scinerf" in joined
    assert "physical imaging process" in joined
    assert "scigs" in joined
    assert "dynamic 3d scenes" in joined


def test_snapshot_compressive_3d_lineage_retrieves_all_named_stages_without_an_llm() -> None:
    cassi_source = (
        "library/OE-2007-Single-shot compressive spectral imaging with "
        "a dual-disperser architecture.en.md"
    )
    scinerf_source = (
        "library/CVPR-2024-SCINeRF-Neural-Radiance-Fields-from-a-"
        "Snapshot-Compressive-Image.en.md"
    )
    scigs_source = (
        "library/ICIP-2025-SCIGS-3D-Gaussians-Splatting-from-a-"
        "Snapshot-Compressive-Image.en.md"
    )
    retriever = BM25Retriever(
        [
            {
                "id": "cassi",
                "text": (
                    "A single-shot compressive spectral imager uses two dispersive "
                    "elements with a binary-valued aperture between them."
                ),
                "meta": {"source_path": cassi_source},
            },
            {
                "id": "scinerf",
                "text": (
                    "SCINeRF formulates the physical imaging process of snapshot "
                    "compressive imaging as part of NeRF training for a 3D scene."
                ),
                "meta": {"source_path": scinerf_source},
            },
            {
                "id": "scigs",
                "text": (
                    "SCIGS adapts 3D Gaussian Splatting to recover dynamic 3D scenes "
                    "from a single compressed image."
                ),
                "meta": {"source_path": scigs_source},
            },
            {
                "id": "unrelated",
                "text": "Deep learning for classical Japanese literature recognition.",
                "meta": {"source_path": "library/Classical-Japanese-Literature.en.md"},
            },
        ]
    )

    hits, _scores, _used_query, _used_translation, _variants = _search_hits_with_fallback(
        "SCI 或压缩快照成像这条线，是怎么从光谱成像走到 3D 场景重建的？",
        retriever,
        3,
        SimpleNamespace(api_key=None, query_expansion_enabled=False),
        allow_translate=False,
    )

    assert {hit["meta"]["source_path"] for hit in hits[:3]} == {
        cassi_source,
        scinerf_source,
        scigs_source,
    }


def test_deterministic_query_variants_expand_exact_mechanism_terms() -> None:
    cases = {
        "这篇 3D single-pixel video 用了几个探测器，速度是多少？": (
            "four spatially-separated single-pixel detectors",
            "8 frames per second",
        ),
        "单像素压缩全息怎么提高吞吐量，为什么不再主动相移？": (
            "beat frequency",
            "phase stepping naturally in time",
        ),
        "Sequential compressed sensing 多利用了什么信息？": (
            "signal support recovery",
            "distilled sensing",
        ),
        "SPAD 雪崩之后为什么需要淬灭电路？": (
            "geiger mode",
            "quenching circuit",
        ),
    }

    for question, required in cases.items():
        joined = " ".join(_deterministic_query_variants(question)).lower()
        assert all(term in joined for term in required)


def test_pidl_piln_comparison_retrieves_both_named_sources_without_an_llm() -> None:
    pidl_source = (
        "library/NatCommun-2023-Physics-informed deep learning for computational "
        "single-photon imaging.en.md"
    )
    piln_source = (
        "library/Optics-2024-Part-based image-loop network for single-pixel imaging.en.md"
    )
    retriever = BM25Retriever(
        [
            {
                "id": "pidl",
                "text": (
                    "Physics-informed deep learning for computational single-photon imaging "
                    "uses a physical data generator for supervised training."
                ),
                "meta": {"source_path": pidl_source},
            },
            {
                "id": "piln",
                "text": (
                    "Part-based image-loop network for single-pixel imaging embeds the "
                    "physical measurement model in an untrained neural network."
                ),
                "meta": {"source_path": piln_source},
            },
            {
                "id": "unrelated",
                "text": "SCINeRF optimizes a neural radiance field from snapshot measurements.",
                "meta": {"source_path": "library/SCINeRF.en.md"},
            },
        ]
    )

    hits, _scores, _used_query, _used_translation, variants = _search_hits_with_fallback(
        "比较 PIDL 与 PILN 的训练先验和推理输入输出",
        retriever,
        2,
        SimpleNamespace(api_key=None, query_expansion_enabled=False),
        allow_translate=False,
    )

    assert {hit["meta"]["source_path"] for hit in hits[:2]} == {pidl_source, piln_source}
    assert len(variants) == 3


def test_deterministic_query_variants_keep_generic_perovskite_laser_query_neutral() -> None:
    variants = _deterministic_query_variants(
        "钙钛矿激光器这篇和我的单像素成像主线关系大吗？"
        "请直接说明它解决的是什么器件问题。"
    )
    joined = " ".join(variants).lower()

    assert "perovskite laser" in joined
    assert "lasing" in joined
    assert "electrically driven" not in joined
    assert "dual-cavity" not in joined
    assert "peled" not in joined


def test_deterministic_query_variants_add_device_terms_only_for_explicit_clues() -> None:
    variants = _deterministic_query_variants(
        "这篇电驱动双腔钙钛矿激光器如何实现电注入？"
    )
    joined = " ".join(variants).lower()

    assert "electrically driven lasing" in joined
    assert "dual-cavity perovskite device" in joined
    assert "peled" in joined


def test_perovskite_laser_alias_matches_lasing_title_not_detector_material() -> None:
    question = (
        "钙钛矿激光器这篇和我的单像素成像主线关系大吗？"
        "请直接说明它解决的是什么器件问题。"
    )

    assert _source_prompt_match_score(
        question,
        "library/Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.en.md",
    ) >= 7.5
    assert _source_prompt_match_score(
        question,
        "library/Perovskite photodetector material for single-pixel imaging.en.md",
    ) == 0.0


def test_perovskite_laser_query_ranks_dual_cavity_device_over_detector_material() -> None:
    laser_source = "library/Nature-2025-Electrically driven lasing from a dual-cavity perovskite device.en.md"
    detector_source = "library/Perovskite photodetector material for single-pixel imaging.en.md"
    retriever = BM25Retriever(
        [
            {
                "id": "laser",
                "text": (
                    "Electrically driven lasing from a dual-cavity perovskite device. "
                    "A high-power microcavity PeLED drives a low-threshold perovskite microcavity."
                ),
                "meta": {"source_path": laser_source},
            },
            {
                "id": "detector",
                "text": (
                    "A perovskite photodetector material improves single-pixel imaging "
                    "sensitivity and reconstruction."
                ),
                "meta": {"source_path": detector_source},
            },
            {
                "id": "spi-review",
                "text": "A review of computational single-pixel imaging methods.",
                "meta": {"source_path": "library/Single-pixel imaging review.en.md"},
            },
            {
                "id": "microscopy",
                "text": "Structured illumination microscopy and optical sectioning.",
                "meta": {"source_path": "library/Structured illumination microscopy.en.md"},
            },
        ]
    )
    question = (
        "钙钛矿激光器这篇和我的单像素成像主线关系大吗？"
        "请直接说明它解决的是什么器件问题。"
    )

    hits, _scores, _used_query, _used_translation, variants = _search_hits_with_fallback(
        question,
        retriever,
        2,
        SimpleNamespace(api_key=None, query_expansion_enabled=False),
    )

    joined = " ".join(variants).lower()
    assert "perovskite laser" in joined
    assert "dual-cavity" not in joined
    assert hits[0]["meta"]["source_path"] == laser_source


def test_optically_pumped_perovskite_laser_query_does_not_promote_dual_cavity_paper() -> None:
    optical_source = "library/Optically pumped perovskite nanowire lasers.en.md"
    dual_cavity_source = (
        "library/Nature-2025-Electrically driven lasing from a dual-cavity "
        "perovskite device.en.md"
    )
    retriever = BM25Retriever(
        [
            {
                "id": "optical",
                "text": (
                    "Optically pumped perovskite nanowire lasers exhibit low-threshold "
                    "lasing and strong optical gain in a Fabry-Perot cavity."
                ),
                "meta": {"source_path": optical_source},
            },
            {
                "id": "dual-cavity",
                "text": (
                    "Electrically driven lasing from a dual-cavity perovskite device. "
                    "A microcavity PeLED provides electrical injection."
                ),
                "meta": {"source_path": dual_cavity_source},
            },
            {
                "id": "spi-review",
                "text": "A review of computational single-pixel imaging methods.",
                "meta": {"source_path": "library/Single-pixel imaging review.en.md"},
            },
            {
                "id": "microscopy",
                "text": "Structured illumination microscopy and optical sectioning.",
                "meta": {"source_path": "library/Structured illumination microscopy.en.md"},
            },
        ]
    )

    hits, _scores, _used_query, _used_translation, variants = _search_hits_with_fallback(
        "帮我找那篇光泵浦钙钛矿激光器论文",
        retriever,
        2,
        SimpleNamespace(api_key=None, query_expansion_enabled=False),
    )

    joined = " ".join(variants).lower()
    assert "optically pumped" in joined
    assert "dual-cavity" not in joined
    assert "electrically driven" not in joined
    assert hits[0]["meta"]["source_path"] == optical_source
