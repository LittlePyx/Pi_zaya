from __future__ import annotations

from pathlib import Path

import kb.retrieval_engine as retrieval_engine
from kb.retrieval_engine import (
    _anchor_text_bonus,
    _extract_explicit_anchor_hint,
    _find_anchor_snippets_in_md,
    _group_hits_by_doc_for_refs,
    _postprocess_refs_pack,
)


def test_extract_explicit_anchor_hint_supports_figure_equation_and_theorem():
    fig = _extract_explicit_anchor_hint("LPR-2025.pdf这篇文章的第三张图讲了啥")
    assert fig["kind"] == "figure"
    assert fig["number"] == 3

    fig_direct = _extract_explicit_anchor_hint("SCINeRF的真实硬件实验装置，请对应到原文图3或实验设置")
    assert fig_direct["kind"] == "figure"
    assert fig_direct["number"] == 3

    eq = _extract_explicit_anchor_hint("请解释这篇论文里的公式(11)")
    assert eq["kind"] == "equation"
    assert eq["number"] == 11

    thm = _extract_explicit_anchor_hint("what does theorem 2 mean in this paper")
    assert thm["kind"] == "theorem"
    assert thm["number"] == 2


def test_extract_explicit_anchor_hint_distinguishes_figure_scopes():
    main = _extract_explicit_anchor_hint("Explain Figure 5 panel a")
    extended = _extract_explicit_anchor_hint("Explain Extended Data Figure 5 panel a")
    supplementary = _extract_explicit_anchor_hint("Explain Fig. S5 panel a")

    assert (main["figure_scope"], main["figure_key"]) == ("main", "main:5")
    assert (extended["figure_scope"], extended["figure_key"]) == ("extended_data", "extended_data:5")
    assert (supplementary["figure_scope"], supplementary["figure_key"]) == ("supplementary", "supplementary:5")


def test_find_anchor_snippets_keeps_same_number_figure_scopes_separate(tmp_path: Path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("# Demo\n\nFigure index fixture.", encoding="utf-8")
    monkeypatch.setattr(
        retrieval_engine,
        "_load_anchor_index_cached",
        lambda _path: {
            "figures": [
                {
                    "number": 5,
                    "figure_scope": "main",
                    "figure_key": "main:5",
                    "caption_text": "Figure 5. Main FLIM result.",
                    "block_id": "main-caption",
                    "heading_path": "Results / Figure 5",
                },
                {
                    "number": 5,
                    "figure_scope": "extended_data",
                    "figure_key": "extended_data:5",
                    "caption_text": "Extended Data Figure 5. Live-cell mitochondria at 25 seconds per frame.",
                    "block_id": "extended-caption",
                    "heading_path": "Extended Data / Extended Data Figure 5",
                },
            ]
        },
    )

    main = _find_anchor_snippets_in_md(md_path, _extract_explicit_anchor_hint("Figure 5"))
    extended = _find_anchor_snippets_in_md(md_path, _extract_explicit_anchor_hint("Extended Data Figure 5"))

    assert [item["id"] for item in main] == ["main-caption"]
    assert [item["id"] for item in extended] == ["extended-caption"]
    assert extended[0]["meta"]["figure_scope"] == "extended_data"
    assert extended[0]["meta"]["figure_key"] == "extended_data:5"


def test_find_anchor_snippets_uses_only_unscoped_legacy_row_as_fallback(tmp_path: Path, monkeypatch):
    md_path = tmp_path / "legacy.en.md"
    md_path.write_text("# Demo\n\nLegacy figure index fixture.", encoding="utf-8")
    monkeypatch.setattr(
        retrieval_engine,
        "_load_anchor_index_cached",
        lambda _path: {
            "figures": [
                {
                    "number": 5,
                    "figure_scope": "main",
                    "figure_key": "main:5",
                    "caption_text": "Figure 5. Explicit main row must not satisfy an extended request.",
                    "block_id": "main-caption",
                },
                {
                    "number": 5,
                    "caption_text": "Legacy Figure 5 row without semantic scope.",
                    "block_id": "legacy-caption",
                },
            ]
        },
    )

    extended = _find_anchor_snippets_in_md(md_path, _extract_explicit_anchor_hint("Extended Data Figure 5"))

    assert [item["id"] for item in extended] == ["legacy-caption"]
    assert extended[0]["meta"]["figure_scope"] == "extended_data"


def test_anchor_text_bonus_prefers_direct_caption_at_snippet_start():
    hint = {"kind": "figure", "number": 3}
    direct = "![Figure 3](./fig3.png)\n**Figure 3.** Experimental setup with a CCD camera and DMD."
    delayed = (
        "The real-world paragraph says Fig. 3 shows the setup. "
        "Several unrelated sentences appear before the actual caption.\n"
        "![Figure 3](./fig3.png)\n**Figure 3.** Experimental setup with a CCD camera and DMD."
    )
    indirect = "**Figure 5.** Qualitative examples captured by our system in Fig. 3."

    assert _anchor_text_bonus(direct, hint) > _anchor_text_bonus(delayed, hint)
    assert _anchor_text_bonus(direct, hint) > _anchor_text_bonus(indirect, hint)


def test_group_hits_by_doc_for_refs_prioritizes_anchor_snippet_for_explicit_doc(tmp_path: Path, monkeypatch):
    md = tmp_path / "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.md"
    md.write_text(
        "\n".join(
            [
                "# LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "",
                "## 1. Introduction",
                "This survey reviews single-pixel imaging with deep learning and summarizes the overall motivation, scope, and background for the field.",
                "",
                "## 3. Fundamentals of Deep Learning",
                "![Figure 3](./assets/page_5_fig_2.png)",
                "**Figure 3.** The basic principles of neural networks. a) ANN. b) Convolution operation. c) Contraction network. d) Encoder-Decoder network. e) RNN. f) GAN. g) Transformer.",
                "The figure summarizes the neural-network building blocks referenced later in the survey and is the key visual explanation for this section.",
                "",
                "Equation (11) defines the reconstruction loss used by the optimization-based method.",
                "Theorem 2 gives the sufficient condition for convergence in the iterative reconstruction setting.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 9.0,
            "id": "h1",
            "text": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "meta": {
                "source_path": str(md),
                "heading_path": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            },
        },
        {
            "score": 8.5,
            "id": "h2",
            "text": "This survey reviews single-pixel imaging with deep learning and summarizes the overall motivation, scope, and background for the field.",
            "meta": {
                "source_path": str(md),
                "heading_path": "1. Introduction",
            },
        },
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf这篇文章的第三张图讲了啥",
        top_k_docs=3,
        deep_query="",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 1
    doc = docs[0]
    assert "Figure 3" in str(doc.get("text") or "")
    meta = doc.get("meta", {}) or {}
    assert float(meta.get("explicit_doc_match_score") or 0.0) >= 6.0
    assert meta.get("anchor_target_kind") == "figure"
    assert meta.get("anchor_target_number") == 3
    show_snips = meta.get("ref_show_snippets") or []
    assert show_snips
    assert "Figure 3" in str(show_snips[0])


def test_group_hits_by_doc_for_refs_rescues_cn_direct_figure_anchor_for_short_title(tmp_path: Path, monkeypatch):
    md = tmp_path / "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    md.write_text(
        "\n".join(
            [
                "# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "",
                "## 4. Experiments",
                "This section contains general reconstruction results.",
                "",
                "**Real-world datasets.** The setup consists of an iRAYPLE A5402MU90 camera and a FLDISCOVERY F4110 DMD. Fig. 3 shows the experimental setup used to collect the real dataset.",
                "",
                "![Figure 3](./assets/fig3.png)",
                "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.",
                "",
                "![Figure 5](./assets/fig5.png)",
                "**Figure 5.** Additional synthetic reconstruction comparisons.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 8.0,
            "id": "h1",
            "text": "SCINeRF reports reconstruction experiments and synthetic comparisons.",
            "meta": {
                "source_path": str(md),
                "heading_path": "4. Experiments",
            },
        }
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="SCINeRF的真实硬件实验装置包含哪些部件？请对应到原文图3或实验设置。",
        top_k_docs=3,
        deep_query="",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 1
    doc = docs[0]
    text = str(doc.get("text") or "")
    meta = doc.get("meta", {}) or {}
    assert "Figure 3" in text
    assert "CCD camera" in text
    assert meta.get("anchor_target_kind") == "figure"
    assert meta.get("anchor_target_number") == 3
    assert float(meta.get("explicit_doc_match_score") or 0.0) >= 6.0
    show_snips = [str(item or "") for item in (meta.get("ref_show_snippets") or [])]
    assert any("FLDISCOVERY F4110 DMD" in item for item in show_snips)


def test_group_hits_by_doc_for_refs_prefers_direct_figure_caption_over_later_reference(tmp_path: Path, monkeypatch):
    md = tmp_path / "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    md.write_text(
        "\n".join(
            [
                "# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "",
                "## 4. Experiments",
                "### 4.1. Experimental Setup",
                "![Figure 3](./assets/fig3.png)",
                "**Figure 3.** Experimental setup for real dataset collection. This SCI imaging system contains a CCD camera to record snapshot measurement, primary and relay lens, and a DMD to modulate input frames.",
                "**Implementation details.** We use PyTorch [48], NeRF [26], and Adam [16] after the caption.",
                "",
                "### 4.2. Additional Study",
                "![Figure 5](./assets/fig5.png)",
                "**Figure 5.** Qualitative evaluations on the real dataset captured by our system in Fig. 3.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 18.0,
            "id": "h-fig5",
            "text": "**Figure 5.** Qualitative evaluations on the real dataset captured by our system in Fig. 3.",
            "meta": {
                "source_path": str(md),
                "heading_path": "4. Experiments / 4.2. Additional Study / Figure 5",
            },
        }
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="SCINeRF的真实硬件实验装置包含哪些部件？请对应到原文图3或实验设置。",
        top_k_docs=3,
        deep_query="",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    doc = docs[0]
    text = str(doc.get("text") or "")
    meta = doc.get("meta", {}) or {}
    assert "Figure 3" in text
    assert "CCD camera" in text
    assert "Figure 5" not in text
    assert str(meta.get("ref_best_heading_path") or "").endswith("4.1. Experimental Setup")


def test_group_hits_by_doc_for_refs_supports_latex_tagged_equation_anchor(tmp_path: Path, monkeypatch):
    md = tmp_path / "NatPhoton-2019-Principles and prospects for single-pixel imaging.md"
    md.write_text(
        "\n".join(
            [
                "# Principles and prospects for single-pixel imaging",
                "",
                "## Box 1 | The maths behind single-pixel imaging",
                "$$",
                r"\mathbf{I}_{\text{TC}} = \sum_{i=1}^N \left( \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} \right) \tag{8}",
                "$$",
                "Equation (8) defines the total-curvature objective used in the reconstruction problem.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 8.8,
            "id": "h1",
            "text": "Principles and prospects for single-pixel imaging",
            "meta": {
                "source_path": str(md),
                "heading_path": "Principles and prospects for single-pixel imaging",
            },
        }
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf里的公式8写的是什么",
        top_k_docs=3,
        deep_query="",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 1
    doc = docs[0]
    text = str(doc.get("text") or "")
    assert "\\tag{8}" in text
    meta = doc.get("meta", {}) or {}
    assert meta.get("anchor_target_kind") == "equation"
    assert meta.get("anchor_target_number") == 8


def test_group_hits_by_doc_for_refs_boosts_exact_focus_doc_over_higher_bm25_noise(tmp_path: Path, monkeypatch):
    target_md = tmp_path / "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.en.md"
    target_md.write_text(
        "\n".join(
            [
                "# Adaptive foveated single-pixel imaging with dynamic supersampling",
                "",
                "## INTRODUCTION",
                "Spatially variant digital supersampling is introduced as a dynamic supersampling strategy for adaptive single-pixel imaging.",
            ]
        ),
        encoding="utf-8",
    )
    noise_md = tmp_path / "Psychological Review-1954-Some informational aspects of visual perception.en.md"
    noise_md.write_text(
        "\n".join(
            [
                "# Some informational aspects of visual perception",
                "",
                "## INFORMATIONAL ASPECTS OF VISUAL PERCEPTION",
                "This transformation saves information by using a relatively simple transformation under dynamic viewing conditions.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 24.4,
            "id": "noise-hit",
            "text": "This transformation saves information by using a relatively simple transformation under dynamic viewing conditions.",
            "meta": {
                "source_path": str(noise_md),
                "heading_path": "INFORMATIONAL ASPECTS OF VISUAL PERCEPTION",
            },
        },
        {
            "score": 20.1,
            "id": "target-hit",
            "text": "Spatially variant digital supersampling is introduced as a dynamic supersampling strategy for adaptive single-pixel imaging.",
            "meta": {
                "source_path": str(target_md),
                "heading_path": "INTRODUCTION",
            },
        },
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="Which paper in my library most directly defines dynamic supersampling? Please point me to the source section.",
        top_k_docs=2,
        deep_query="Which paper in my library most directly defines dynamic supersampling? Please point me to the source section.",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 2
    top_meta = docs[0].get("meta", {}) or {}
    assert str(top_meta.get("source_path") or "").endswith(target_md.name)
    assert float(((top_meta.get("ref_rank") or {}).get("focus_bonus") or 0.0)) > 0.0


def test_group_hits_by_doc_for_refs_promotes_user_written_technical_phrase(tmp_path: Path, monkeypatch):
    target_md = tmp_path / "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.en.md"
    target_md.write_text(
        "\n".join(
            [
                "# Adaptive foveated single-pixel imaging with dynamic supersampling",
                "",
                "## INTRODUCTION",
                "### Spatially variant digital supersampling",
                "Dynamic supersampling shifts pixel boundaries between frames so important regions receive denser spatial samples.",
            ]
        ),
        encoding="utf-8",
    )
    background_md = tmp_path / "SSP-2012-Sequential compressed sensing.en.md"
    background_md.write_text(
        "\n".join(
            [
                "# Sequential compressed sensing",
                "",
                "## I. Introduction",
                "Single-pixel imaging can use compressed sensing and dynamically adapt measurements in a general reconstruction pipeline.",
            ]
        ),
        encoding="utf-8",
    )
    mention_only_md = tmp_path / "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    mention_only_md.write_text(
        "\n".join(
            [
                "# Principles and prospects for single-pixel imaging",
                "",
                "## Adaptive strategies",
                "Recently, adaptive and smart sensing with dynamic supersampling was reported for single-pixel imaging.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 31.0,
            "id": "background",
            "text": "Single-pixel imaging can use compressed sensing and dynamically adapt measurements in a general reconstruction pipeline.",
            "meta": {
                "source_path": str(background_md),
                "heading_path": "I. Introduction",
            },
        },
        {
            "score": 28.0,
            "id": "mention-only",
            "text": "Recently, adaptive and smart sensing with dynamic supersampling was reported for single-pixel imaging.",
            "meta": {
                "source_path": str(mention_only_md),
                "heading_path": "Adaptive strategies",
            },
        },
        {
            "score": 22.0,
            "id": "target",
            "text": "Dynamic supersampling shifts pixel boundaries between frames so important regions receive denser spatial samples.",
            "meta": {
                "source_path": str(target_md),
                "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            },
        },
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="dynamic supersampling \u662f\u4e0d\u662f\u5c31\u662f\u53ea\u76ef\u7740\u753b\u9762\u91cd\u8981\u7684\u5730\u65b9\u591a\u62cd\u4e00\u70b9\uff1f",
        top_k_docs=3,
        deep_query="",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 3
    top_meta = docs[0].get("meta", {}) or {}
    assert str(top_meta.get("source_path") or "").endswith(target_md.name)
    assert float(top_meta.get("direct_prompt_match_score") or 0.0) >= 6.0
    assert "dynamic supersampling" in list(top_meta.get("direct_prompt_match_terms") or [])


def test_group_hits_by_doc_for_refs_does_not_mistake_reference_index_for_equation_anchor(tmp_path: Path, monkeypatch):
    md = tmp_path / "CVPR-2024-SCINeRF.en.md"
    md.write_text(
        "\n".join(
            [
                "# SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "",
                "## 3. Method",
                "### 3.1. Background on NeRF",
                "The whole process can be formally defined via the following equation:",
                "$$",
                r"C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})dt, \tag{1}",
                "$$",
                "where t_n and t_f are near and far bounds for volumetric rendering respectively.",
                "",
                "## References",
                "[1] Ben Mildenhall et al. NeRF: Representing scenes as neural radiance fields for view synthesis.",
                "[5] Yoonwoo Jeong et al. Self-calibrating neural radiance fields. In Proceedings of ICCV, 2021.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)

    hits_raw = [
        {
            "score": 8.1,
            "id": "h1",
            "text": "3.1. Background on NeRF",
            "meta": {
                "source_path": str(md),
                "heading_path": "3. Method / 3.1. Background on NeRF",
            },
        },
        {
            "score": 7.4,
            "id": "h2",
            "text": "[1] Ben Mildenhall et al. NeRF: Representing scenes as neural radiance fields for view synthesis.",
            "meta": {
                "source_path": str(md),
                "heading_path": "References",
            },
        },
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text=f"{md} SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image SCINeRF 的 NeRF 体渲染公式是哪条？请解释公式(1)以及后面的 where 句",
        top_k_docs=3,
        deep_query=f"{md} SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image SCINeRF 的 NeRF 体渲染公式是哪条？请解释公式(1)以及后面的 where 句",
        deep_read=True,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 1
    doc = docs[0]
    text = str(doc.get("text") or "")
    meta = doc.get("meta", {}) or {}
    assert "\\tag{1}" in text
    assert "where t_n and t_f" in text
    assert "[1] Ben Mildenhall" not in text
    assert meta.get("anchor_target_kind") == "equation"
    assert meta.get("anchor_target_number") == 1
    show_snips = [str(x or "") for x in (meta.get("ref_show_snippets") or [])]
    assert show_snips
    assert any("\\tag{1}" in item for item in show_snips)
    assert all("[1] Ben Mildenhall" not in item for item in show_snips)


def test_grouped_reference_keeps_top_structured_table_as_its_only_metric_evidence(
    tmp_path: Path,
    monkeypatch,
):
    md = tmp_path / "Simple Baselines for Image Restoration.md"
    md.write_text(
        "\n".join(
            [
                "# Simple Baselines for Image Restoration",
                "## 5 Experiments",
                "### 5.1 Ablations",
                "**Table 3.** The effect of the number of blocks.",
                "| blocks | SIDD PSNR |",
                "| --- | --- |",
                "| 36 | 39.96 |",
                "### 5.2 Applications",
                "**Table 6.** Image Denoising Results on SIDD.",
                "| Method | Baseline ours | NAFNet ours |",
                "| --- | --- | --- |",
                "| PSNR | 40.30 | 40.30 |",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval_engine, "_is_temp_source_path", lambda _src: False)
    benchmark = (
        "Table 6. Image Denoising Results on SIDD. SIDD PSNR: "
        "Baseline ours = 40.30; NAFNet ours = 40.30"
    )
    ablation = "Table 3. SIDD PSNR: 9 = 39.78; 18 = 39.90; 36 = 39.96; 72 = 39.95"
    hits_raw = [
        {
            "score": 31.2,
            "id": "benchmark",
            "text": benchmark,
            "meta": {
                "source_path": str(md),
                "heading_path": "5 Experiments / 5.2 Applications",
                "structured_kind": "table_metric",
                "table_index": 2,
                "table_number": 6,
                "table_metric": "PSNR",
                "table_metric_label": "SIDD PSNR",
                "table_subject_kind": "method",
                "block_id": "table-6",
                "anchor_id": "tb_00002",
                "page_start": 13,
                "page_end": 13,
            },
        },
        {
            "score": 27.4,
            "id": "ablation",
            "text": ablation,
            "meta": {
                "source_path": str(md),
                "heading_path": "5 Experiments / 5.1 Ablations",
                "structured_kind": "table_metric",
                "table_index": 1,
                "table_number": 3,
                "table_metric": "PSNR",
                "table_metric_label": "SIDD PSNR",
                "table_subject_kind": "variant",
                "block_id": "table-3",
                "anchor_id": "tb_00001",
                "page_start": 12,
                "page_end": 12,
            },
        },
    ]

    docs = _group_hits_by_doc_for_refs(
        hits_raw,
        prompt_text="Which model has the highest SIDD PSNR?",
        top_k_docs=1,
        deep_query="Which model has the highest SIDD PSNR?",
        deep_read=False,
        llm_rerank=False,
        settings=None,
    )

    assert len(docs) == 1
    doc = docs[0]
    meta = doc["meta"]
    assert doc["text"] == benchmark
    assert meta["ref_show_snippets"] == [benchmark]
    assert meta["ref_snippets"] == [benchmark]
    assert meta["structured_kind"] == "table_metric"
    assert meta["structured_evidence_locked"] is True
    assert meta["table_number"] == 6
    assert meta["table_subject_kind"] == "method"
    assert meta["block_id"] == "table-6"
    assert meta["anchor_id"] == "tb_00002"
    assert meta["heading_path"] == "5 Experiments / 5.2 Applications"
    assert meta["page_start"] == 13
    assert ablation not in meta["ref_show_snippets"]


def test_postprocess_refs_pack_overrides_conflicting_why_when_anchor_hit():
    docs = [
        {
            "text": "Equation content",
            "meta": {
                "source_path": "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                "anchor_target_kind": "equation",
                "anchor_target_number": 8,
                "anchor_match_score": 15.2,
                "ref_section": "Box 1 | The maths behind single-pixel imaging",
                "ref_locs": [
                    {
                        "heading": "Box 1 | The maths behind single-pixel imaging",
                        "heading_path": "Box 1 | The maths behind single-pixel imaging",
                        "score": 10.0,
                        "quality": "high",
                    }
                ],
            },
        }
    ]
    result = {
        1: {
            "score": 82.0,
            "what": "这篇文献讨论单像素成像的数学基础。",
            "why": "问题询问公式8，但文档片段中未直接给出该公式表达式，因此无法确认。",
            "start": "",
            "gain": "只能提供部分背景。",
            "find": [],
            "section": "",
        }
    }
    out = _postprocess_refs_pack(result, docs, question="NatPhoton-2019这篇文章里公式8是什么")
    why = str(out[1]["why"] or "")
    assert "未直接给出" not in why
    assert "无法确认" not in why
    assert "公式8" in why
