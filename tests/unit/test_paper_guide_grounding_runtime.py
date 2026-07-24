from __future__ import annotations

import json
from pathlib import Path

from kb.paper_guide.grounder import (
    _build_paper_guide_support_slots,
    _build_paper_guide_support_slots_block,
    _extract_inline_reference_specs,
    _inject_paper_guide_support_markers,
    _paper_guide_cue_tokens,
    _paper_guide_support_cite_policy,
    _paper_guide_support_claim_type,
    _resolve_paper_guide_support_slot_block,
    _resolve_paper_guide_support_markers,
    _resolve_paper_guide_support_ref_num,
)
from kb.paper_guide_grounding_runtime import (
    _paper_guide_refocus_support_excerpt,
    _paper_guide_support_focus_tokens,
    _score_paper_guide_evidence_atom,
    _select_grounding_figure_index_entry,
)


def test_select_grounding_figure_index_entry_hard_filters_explicit_figure_scope():
    rows = [
        {
            "paper_figure_number": 5,
            "figure_scope": "main",
            "figure_key": "main:5",
            "caption": "Figure 5. Main result with a tissue sample.",
            "caption_block_id": "main-caption",
        },
        {
            "paper_figure_number": 5,
            "figure_scope": "extended_data",
            "figure_key": "extended_data:5",
            "caption": "Extended Data Figure 5. Live-cell mitochondria.",
            "caption_block_id": "extended-caption",
        },
    ]

    main = _select_grounding_figure_index_entry(rows, figure_number=5, figure_scope="main")
    extended = _select_grounding_figure_index_entry(rows, figure_number=5, figure_scope="extended_data")

    assert main["caption_block_id"] == "main-caption"
    assert extended["caption_block_id"] == "extended-caption"


def test_extract_inline_reference_specs_supports_brackets_and_superscripts():
    out = _extract_inline_reference_specs(
        "We use RVT [34] and Hadamard$^{64,65}$ as prior work, with an additional note ^{72}."
    )

    assert out == ["34", "64,65", "72"]


def test_paper_guide_cue_tokens_keeps_domain_terms():
    out = _paper_guide_cue_tokens(
        "APR uses phase correlation image registration on the iISM dataset."
    )

    assert "apr" in out
    assert "phase" in out
    assert "correlation" in out
    assert "registration" in out
    assert "iism" in out


def test_refocus_support_excerpt_keeps_both_mechanism_steps():
    excerpt = _paper_guide_refocus_support_excerpt(
        "Digital refocusing is achieved using two steps. "
        "First, photon trajectories are reconstructed through a ray tracing operation. "
        "For microscopic samples diffraction must also be considered. "
        "The second step applies wave propagation of distance -z to bring the sample back into focus."
    )

    assert "two steps" in excerpt
    assert "ray tracing" in excerpt
    assert "wave propagation" in excerpt


def test_paper_guide_support_claim_type_and_policy_for_method_refs():
    claim_type = _paper_guide_support_claim_type(
        prompt_family="method",
        heading="Methods / APR",
        snippet="APR was performed using image registration based on phase correlation [35].",
        candidate_refs=[35],
        ref_spans=[{"text": "phase correlation [35]", "nums": [35], "scope": "same_sentence"}],
    )

    assert claim_type == "method_detail"
    assert _paper_guide_support_cite_policy(claim_type=claim_type, prompt_family="method") == "prefer_ref"


def test_paper_guide_support_claim_type_does_not_force_prior_work_for_overview_with_incidental_refs():
    claim_type = _paper_guide_support_claim_type(
        prompt_family="overview",
        heading="Applications and future potential for single-pixel imaging / Figure 3",
        snippet="Figure 3 shows the cost per megapixel across wavelength bands [64] and example application thumbnails [15].",
        candidate_refs=[64, 15],
        ref_spans=[{"text": "cost per megapixel [64]", "nums": [64], "scope": "same_sentence"}],
    )

    assert claim_type == "own_result"
    assert _paper_guide_support_cite_policy(claim_type=claim_type, prompt_family="overview") == "locate_only"


def test_paper_guide_support_claim_type_does_not_force_prior_work_for_strength_limits_refs():
    claim_type = _paper_guide_support_claim_type(
        prompt_family="strength_limits",
        heading="Acquisition and image reconstruction strategies",
        snippet="A subset strategy may use prior information [16], while optimization algorithms may use the l1-norm [60,61] or total variation [62], but reconstruction time can exceed acquisition time.",
        candidate_refs=[16, 60, 61, 62],
        ref_spans=[{"text": "prior information [16]", "nums": [16], "scope": "same_sentence"}],
    )

    assert claim_type == "own_result"
    assert _paper_guide_support_cite_policy(claim_type=claim_type, prompt_family="strength_limits") == "locate_only"


def test_score_paper_guide_evidence_atom_prefers_sentence_over_ref_span_for_own_result():
    probe = "计算开销仍不可忽略（“computational overhead is not negligible”）；"
    sentence_score, _ = _score_paper_guide_evidence_atom(
        {
            "atom_kind": "sentence",
            "text": "Still, the computational overhead is not negligible, and there is a hotbed of alternative minimization strategies.",
            "heading_path": "Understanding compressed sensing",
            "inline_refs": [],
        },
        probe=probe,
        heading="Understanding compressed sensing",
        prompt_family="strength_limits",
        claim_type="own_result",
    )
    ref_score, _ = _score_paper_guide_evidence_atom(
        {
            "atom_kind": "ref_span",
            "text": "hence depth [26,27,29,49]",
            "heading_path": "Applications and future potential / Figure 5",
            "inline_refs": [26, 27, 29, 49],
        },
        probe=probe,
        heading="Understanding compressed sensing",
        prompt_family="strength_limits",
        claim_type="own_result",
    )

    assert sentence_score > ref_score


def test_score_paper_guide_evidence_atom_prefers_exact_quality_tradeoff_sentence():
    probe = "优点：在显著欠采样下仍能获得最高图像质量和高帧率视频；适合压缩数据的离线后处理。"
    query_tokens = _paper_guide_support_focus_tokens(probe)
    exact_score, _ = _score_paper_guide_evidence_atom(
        {
            "atom_kind": "sentence",
            "text": "Nonetheless, for applications that permit post-processing offline, this strategy typically yields highest image quality and highest frame-rate video from significantly compressed data.",
            "heading_path": "Understanding compressed sensing",
            "inline_refs": [],
        },
        probe=probe,
        heading="Understanding compressed sensing",
        prompt_family="strength_limits",
        claim_type="own_result",
        query_tokens=query_tokens,
    )
    generic_score, _ = _score_paper_guide_evidence_atom(
        {
            "atom_kind": "sentence",
            "text": "When used to recover an image from data where M=N, one can invert the measurement matrix and recover the image.",
            "heading_path": "Single-pixel imaging overview",
            "inline_refs": [],
        },
        probe=probe,
        heading="Understanding compressed sensing",
        prompt_family="strength_limits",
        claim_type="own_result",
        query_tokens=query_tokens,
    )

    assert exact_score > generic_score


def test_score_paper_guide_evidence_atom_prefers_fast_low_resolution_sentence():
    probe = "适用场景：低/中分辨率成像、需快速反馈的应用（如视频流、动态目标跟踪）。"
    query_tokens = _paper_guide_support_focus_tokens(probe)
    fast_score, _ = _score_paper_guide_evidence_atom(
        {
            "atom_kind": "sentence",
            "text": "In general, a sub-sampled basis is most appropriate to applications that require low to moderate image resolutions as well as faster, or even real-time, image reconstruction.",
            "heading_path": "Acquisition and image reconstruction strategies",
            "inline_refs": [],
        },
        probe=probe,
        heading="Acquisition and image reconstruction strategies",
        prompt_family="strength_limits",
        claim_type="own_result",
        query_tokens=query_tokens,
    )
    offline_score, _ = _score_paper_guide_evidence_atom(
        {
            "atom_kind": "sentence",
            "text": "Nonetheless, for applications that permit post-processing offline, this strategy typically yields highest image quality and highest frame-rate video from significantly compressed data.",
            "heading_path": "Acquisition and image reconstruction strategies",
            "inline_refs": [],
        },
        probe=probe,
        heading="Acquisition and image reconstruction strategies",
        prompt_family="strength_limits",
        claim_type="own_result",
        query_tokens=query_tokens,
    )

    assert fast_score > offline_score


def test_build_paper_guide_support_slots_assigns_unique_markers_and_block_renders(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "Figure 1. d Open pinhole confocal iSCAT. e Closed pinhole confocal iSCAT.\n\n"
            "## Methods\n\n"
            "APR was performed using image registration based on phase correlation [35].\n"
        ),
        encoding="utf-8",
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s1",
                "source_path": str(source_pdf),
                "heading": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "candidate_refs": [35],
                "deepread_texts": [],
            },
            {
                "doc_idx": 1,
                "sid": "s1",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 1",
                "snippet": "Figure 1. d Open pinhole confocal iSCAT. e Closed pinhole confocal iSCAT.",
                "candidate_refs": [],
                "deepread_texts": [],
            },
        ],
        prompt="Walk me through Figure 1 and explain how APR is implemented.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert [slot["support_example"] for slot in slots] == ["[[SUPPORT:DOC-1-S1]]", "[[SUPPORT:DOC-1-S2]]"]
    block = _build_paper_guide_support_slots_block(slots)
    assert "Paper-guide support slots" in block
    assert "support_example=[[SUPPORT:DOC-1-S1]]" in block
    assert "cite_example=[[CITE:s1:35]]" in block


def test_build_support_slots_uses_cross_language_retrieval_terms_for_locator(tmp_path: Path):
    source_pdf = tmp_path / "SpiProspects.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "SpiProspects"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "SpiProspects.en.md"
    abstract = (
        "Modern digital cameras employ silicon focal plane array (FPA) image sensors "
        "featuring millions of pixels. As the approach suits a wide variety of detector "
        "technologies, images can be collected at wavelengths outside the reach of FPA "
        "technology or at high frame rates or in three dimensions. Promising applications "
        "include the visualization of hazardous gas leaks and 3D situation awareness for "
        "autonomous vehicles."
    )
    md_main.write_text(f"## Abstract\n\n{abstract}\n", encoding="utf-8")

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s1",
                "source_path": str(source_pdf),
                "heading": "Abstract",
                "snippet": abstract,
                "candidate_refs": [],
                "deepread_texts": [],
            }
        ],
        prompt=(
            "什么场景值得用单像素相机？\n"
            "wavelengths outside FPA technology high frame rates three dimensions "
            "hazardous gas leaks autonomous vehicles"
        ),
        prompt_family="overview",
        db_dir=tmp_path,
    )

    assert len(slots) == 1
    evidence = " ".join(
        [
            str(slots[0].get("snippet") or ""),
            str(slots[0].get("locate_anchor") or ""),
        ]
    ).lower()
    assert "wavelengths outside the reach of fpa technology" in evidence
    assert "high frame rates" in evidence
    assert "three dimensions" in evidence
    assert "hazardous gas leaks" in evidence
    assert "autonomous vehicles" in evidence


def test_build_paper_guide_support_slots_expands_targeted_panel_atoms(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "**Figure 1.** d Open pinhole confocal iSCAT. "
            "e Closed pinhole confocal iSCAT. "
            "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR), with same incident illumination power and number of detected photons. "
            "g Line profiles of the iPSF in the three configurations as indicated in d-f.\n"
        ),
        encoding="utf-8",
    )

    slots = _build_paper_guide_support_slots(
        [
            {
                "doc_idx": 1,
                "sid": "s1",
                "source_path": str(source_pdf),
                "heading": "Results / Figure 1",
                "snippet": "Figure 1. f Resulting iPSF from iISM after adaptive pixel-reassignment (APR). g Line profiles of the iPSF in the three configurations.",
                "candidate_refs": [],
                "deepread_texts": [],
            }
        ],
        prompt="Walk me through Figure 1 panels f and g.",
        prompt_family="figure_walkthrough",
        db_dir=tmp_path,
    )

    assert len(slots) == 2
    assert {tuple(slot.get("panel_letters") or []) for slot in slots} == {("f",), ("g",)}
    assert {slot["support_example"] for slot in slots} == {"[[SUPPORT:DOC-1-S1]]", "[[SUPPORT:DOC-1-S2]]"}
    assert {str(slot.get("heading_path") or "") for slot in slots} == {"Results / Figure 1"}
    anchors = {
        tuple(slot.get("panel_letters") or []): str(slot.get("locate_anchor") or "")
        for slot in slots
    }
    assert "adaptive pixel-reassignment" in anchors[("f",)].lower()
    assert "line profiles" in anchors[("g",)].lower()


def test_inject_paper_guide_support_markers_prefers_method_line():
    out = _inject_paper_guide_support_markers(
        "APR is performed using phase correlation image registration.",
        support_slots=[
            {
                "support_example": "[[SUPPORT:DOC-1]]",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "cue": "phase correlation image registration",
                "heading_path": "Methods / APR",
                "heading": "Methods / APR",
                "snippet": "APR was performed using image registration based on phase correlation [35].",
                "locate_anchor": "APR was performed using image registration based on phase correlation [35].",
                "deepread_texts": [],
            }
        ],
        prompt_family="method",
    )

    assert out.endswith("[[SUPPORT:DOC-1]]")


def test_resolve_paper_guide_support_ref_num_prefers_local_ref_span():
    ref_num, mode = _resolve_paper_guide_support_ref_num(
        {
            "cite_policy": "prefer_ref",
            "candidate_refs": [32, 34],
            "ref_spans": [{"text": "RVT [34]", "nums": [34], "scope": "same_sentence"}],
        },
        context_text="The method uses RVT [34] for the transform.",
    )

    assert ref_num == 34
    assert mode == "context_explicit_ref"


def test_resolve_paper_guide_support_markers_rewrites_to_structured_cite():
    answer, resolutions = _resolve_paper_guide_support_markers(
        "We use RVT for the transform [[SUPPORT:DOC-1]].",
        support_slots=[
            {
                "doc_idx": 1,
                "support_id": "DOC-1",
                "support_example": "[[SUPPORT:DOC-1]]",
                "sid": "s1",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Methods / RVT",
                "heading_path": "Methods / RVT",
                "snippet": "We use the radial variance transform (RVT) [34].",
                "locate_anchor": "We use the radial variance transform (RVT) [34].",
                "claim_type": "method_detail",
                "cite_policy": "prefer_ref",
                "candidate_refs": [34],
                "ref_spans": [{"text": "RVT [34]", "nums": [34], "scope": "same_sentence"}],
                "deepread_texts": [],
                "block_id": "blk1",
                "anchor_id": "a1",
            }
        ],
        prompt_family="method",
        db_dir=None,
    )

    assert "[[SUPPORT:" not in answer
    assert "[[CITE:s1:34]]" in answer
    assert len(resolutions) == 1
    assert resolutions[0]["resolved_ref_num"] == 34


def test_resolve_paper_guide_support_markers_drops_broad_summary_markers():
    answer, resolutions = _resolve_paper_guide_support_markers(
        "文中明确提到了两类主流重建策略，其优缺点与适用场景如下：[[SUPPORT:DOC-1]][[SUPPORT:DOC-2]]",
        support_slots=[
            {
                "doc_idx": 1,
                "support_id": "DOC-1",
                "support_example": "[[SUPPORT:DOC-1]]",
                "sid": "s1",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Applications",
                "heading_path": "Applications",
                "snippet": "Single-pixel cameras have also been demonstrated at terahertz frequencies.",
                "locate_anchor": "Single-pixel cameras have also been demonstrated at terahertz frequencies.",
                "claim_type": "own_result",
                "cite_policy": "locate_only",
                "candidate_refs": [],
                "ref_spans": [],
            },
            {
                "doc_idx": 2,
                "support_id": "DOC-2",
                "support_example": "[[SUPPORT:DOC-2]]",
                "sid": "s1",
                "source_path": r"db\demo\paper.en.md",
                "heading": "Randomness",
                "heading_path": "Randomness",
                "snippet": "Random number sequences are central to cryptography.",
                "locate_anchor": "Random number sequences are central to cryptography.",
                "claim_type": "own_result",
                "cite_policy": "locate_only",
                "candidate_refs": [],
                "ref_spans": [],
            },
        ],
        prompt_family="strength_limits",
        db_dir=None,
    )

    assert "[[SUPPORT:" not in answer
    assert "两类主流重建策略" in answer
    assert resolutions == []


def test_inject_paper_guide_support_markers_skips_nested_figure_color_bullets():
    out = _inject_paper_guide_support_markers(
        (
            "- **Panel (f)** shows the iPSF after APR.\n"
            "- **Panel (g)** shows line profiles of the iPSF.\n"
            " - red: iISM with APR."
        ),
        support_slots=[
            {
                "support_example": "[[SUPPORT:DOC-1-S1]]",
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "cue": "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR)",
                "heading_path": "Results / Figure 1",
                "heading": "Results / Figure 1",
                "snippet": "Figure 1. f Resulting iPSF ... g Line profiles ...",
                "locate_anchor": "f Resulting iPSF from iISM after adaptive pixel-reassignment (APR)",
                "panel_letters": ["f"],
                "deepread_texts": [],
            },
            {
                "support_example": "[[SUPPORT:DOC-1-S2]]",
                "claim_type": "figure_panel",
                "cite_policy": "locate_only",
                "cue": "g Line profiles of the iPSF",
                "heading_path": "Results / Figure 1",
                "heading": "Results / Figure 1",
                "snippet": "Figure 1. f Resulting iPSF ... g Line profiles ...",
                "locate_anchor": "g Line profiles of the iPSF in the three configurations",
                "panel_letters": ["g"],
                "deepread_texts": [],
            },
        ],
        prompt_family="figure_walkthrough",
    )

    lines = out.splitlines()
    assert lines[0].endswith("[[SUPPORT:DOC-1-S1]]")
    assert lines[1].endswith("[[SUPPORT:DOC-1-S2]]")
    assert "[[SUPPORT:" not in lines[2]


def test_inject_paper_guide_support_markers_skips_figure_like_overview_slot_for_generic_summary_line():
    out = _inject_paper_guide_support_markers(
        "This paper tackles low-light single-photon imaging and uses physics-informed deep learning to improve resolution.",
        support_slots=[
            {
                "support_example": "[[SUPPORT:DOC-1]]",
                "claim_type": "own_result",
                "cite_policy": "locate_only",
                "figure_number": 7,
                "cue": "sub-pixel convolution is applied in the reconstruction block to further upsample the feature map",
                "heading_path": "Methods / Image reconstruction",
                "heading": "Methods / Image reconstruction",
                "snippet": "The workflow of sub-pixel convolution reconstruction.",
                "locate_anchor": "In addition, sub-pixel convolution is applied in the reconstruction block to further upsample the feature map for single-photon super resolution.",
                "deepread_texts": [],
            }
        ],
        prompt_family="overview",
    )

    assert "[[SUPPORT:DOC-1]]" not in out


def test_inject_paper_guide_support_markers_keeps_figure_like_slot_when_overview_line_explicitly_mentions_figure():
    out = _inject_paper_guide_support_markers(
        "Figure 7 shows how the reconstruction block upsamples the feature map for super resolution.",
        support_slots=[
            {
                "support_example": "[[SUPPORT:DOC-1]]",
                "claim_type": "own_result",
                "cite_policy": "locate_only",
                "figure_number": 7,
                "cue": "sub-pixel convolution is applied in the reconstruction block to further upsample the feature map",
                "heading_path": "Methods / Image reconstruction / Figure 7",
                "heading": "Methods / Image reconstruction / Figure 7",
                "snippet": "Figure 7. a The workflow of sub-pixel convolution reconstruction.",
                "locate_anchor": "In addition, sub-pixel convolution is applied in the reconstruction block to further upsample the feature map for single-photon super resolution.",
                "deepread_texts": [],
            }
        ],
        prompt_family="overview",
    )

    assert out.endswith("[[SUPPORT:DOC-1]]")


def test_resolve_paper_guide_support_slot_block_prefers_ref_span_atom_for_citation_lookup(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Acquisition and image reconstruction strategies\n\n"
            "The original concept of the single-pixel imaging approach, demonstrated by Sen et al.$^{3,58}$, "
            "was developed further in conjunction with compressive sensing$^{59}$ and reported soon after in a seminal paper by Duarte et al. at Rice University$^{4}$.\n"
        ),
        encoding="utf-8",
    )

    rec = _resolve_paper_guide_support_slot_block(
        source_path=str(source_pdf),
        snippet="reported soon after in a seminal paper by Duarte et al. at Rice University.",
        heading="Acquisition and image reconstruction strategies",
        prompt_family="citation_lookup",
        claim_type="prior_work",
        db_dir=tmp_path,
    )

    assert "Duarte" in str(rec.get("locate_anchor") or "")
    assert rec.get("candidate_refs") == [4]
    assert list((rec.get("ref_spans") or [])[0].get("nums") or []) == [4]


def test_resolve_paper_guide_support_slot_block_falls_back_to_reference_index_for_citation_lookup(
    monkeypatch,
    tmp_path: Path,
):
    import kb.paper_guide_grounding_runtime as legacy_grounding_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Introduction\n\n"
            "This paragraph discusses adaptive sampling generally but does not contain inline references.\n"
        ),
        encoding="utf-8",
    )
    (assets_dir / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 31,
                        "reference_entry_id": "ref_0031",
                        "text": "[31] Adaptive foveated single-pixel imaging via supersampling. Optics Express, 2021.",
                        "doi": "",
                        "year": "2021",
                        "parse_confidence": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(legacy_grounding_runtime, "_build_paper_guide_evidence_atoms", lambda _blocks: [])
    monkeypatch.setattr(legacy_grounding_runtime, "match_source_blocks", lambda *args, **kwargs: [])

    rec = legacy_grounding_runtime._resolve_paper_guide_support_slot_block(
        source_path=str(source_pdf),
        snippet="Which paper is cited for adaptive and smart sensing with dynamic supersampling?",
        heading="Introduction",
        prompt_family="citation_lookup",
        claim_type="prior_work",
        db_dir=tmp_path,
    )

    assert rec.get("candidate_refs") == [31]
    assert str(rec.get("heading_path") or "") == "References"
    assert "supersampling" in str(rec.get("locate_anchor") or "").lower()
    assert str(rec.get("evidence_atom_kind") or "") == "reference_entry"


def test_resolve_paper_guide_support_slot_block_appends_figure_heading_for_panel_caption(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "**Figure 3.** Thumbnail images: (e) methane imaging using scanning laser; "
            "(f) methane imaging using SPC; (g) methane imaging using scanning OPO.\n"
        ),
        encoding="utf-8",
    )

    rec = _resolve_paper_guide_support_slot_block(
        source_path=str(source_pdf),
        snippet="Panel (f) corresponds to methane imaging using SPC.",
        heading="Results",
        prompt_family="figure_walkthrough",
        claim_type="figure_panel",
        db_dir=tmp_path,
        target_scope={
            "prompt_family": "figure_walkthrough",
            "target_figure_number": 3,
            "target_panel_letters": ["f"],
        },
    )

    assert str(rec.get("heading_path") or "") == "Results / Figure 3"
    assert "methane imaging using SPC" in str(rec.get("locate_anchor") or "")
    assert "f" in list(rec.get("panel_letters") or [])


def test_resolve_paper_guide_support_slot_block_prefers_figure_index_binding_for_panel_caption(
    monkeypatch,
    tmp_path: Path,
):
    import kb.paper_guide_grounding_runtime as legacy_grounding_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results\n\n"
            "![Figure 3](./assets/fig3.png)\n"
            "*Thumbnail images: (e) methane imaging using scanning laser; "
            "(f) methane imaging using SPC; (g) methane imaging using scanning OPO.*\n\n"
            "A nearby paragraph about methane imaging that should lose the binding race.\n"
        ),
        encoding="utf-8",
    )
    (assets_dir / "fig3.png").write_bytes(b"fake")

    blocks = legacy_grounding_runtime.load_source_blocks(md_main)
    figure_block = next(block for block in blocks if str(block.get("kind") or "") == "figure")
    caption_block = next(
        block
        for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "methane imaging using spc" in str(block.get("text") or "").lower()
    )
    decoy_block = next(
        block
        for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "should lose the binding race" in str(block.get("text") or "").lower()
    )
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "paper_figure_number": 3,
                        "figure_block_id": str(figure_block.get("block_id") or ""),
                        "caption_block_id": str(caption_block.get("block_id") or ""),
                        "caption_anchor_id": str(caption_block.get("anchor_id") or ""),
                        "heading_path": "Results / Figure 3",
                        "locate_anchor": "Thumbnail images: (e) methane imaging using scanning laser; (f) methane imaging using SPC; (g) methane imaging using scanning OPO.",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(legacy_grounding_runtime, "_build_paper_guide_evidence_atoms", lambda _blocks: [])
    monkeypatch.setattr(
        legacy_grounding_runtime,
        "match_source_blocks",
        lambda blocks, **kwargs: [{"score": 9.0, "block": dict(decoy_block)}],
    )

    rec = legacy_grounding_runtime._resolve_paper_guide_support_slot_block(
        source_path=str(source_pdf),
        snippet="Panel (f) corresponds to methane imaging using SPC.",
        heading="Results",
        prompt_family="figure_walkthrough",
        claim_type="figure_panel",
        db_dir=tmp_path,
        target_scope={
            "prompt_family": "figure_walkthrough",
            "target_figure_number": 3,
            "target_panel_letters": ["f"],
        },
    )

    assert str(rec.get("block_id") or "") == str(caption_block.get("block_id") or "")
    assert str(rec.get("block_id") or "") != str(decoy_block.get("block_id") or "")
    assert str(rec.get("heading_path") or "") == "Results / Figure 3"
    assert "methane imaging using spc" in str(rec.get("locate_anchor") or "").lower()
    assert list(rec.get("panel_letters") or []) == ["f"]


def test_resolve_paper_guide_support_slot_block_prefers_exact_box_sentence(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Acquisition and image reconstruction strategies\n\n"
            "**[Box 1 - The maths behind single-pixel imaging]**\n\n"
            "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$, "
            "the image in the transform domain can be reconstructed by solving an optimization problem.\n"
        ),
        encoding="utf-8",
    )

    rec = _resolve_paper_guide_support_slot_block(
        source_path=str(source_pdf),
        snippet="What condition is given for reconstructing the image in the transform domain?",
        heading="Box 1",
        prompt_family="box_only",
        claim_type="critical_fact",
        db_dir=tmp_path,
        target_scope={
            "prompt_family": "box_only",
            "requested_boxes": [1],
        },
    )

    assert str(rec.get("heading_path") or "") == "Acquisition and image reconstruction strategies / Box 1"
    assert "transform domain" in str(rec.get("locate_anchor") or "").lower()
    assert "k \\log(n/k)" in str(rec.get("locate_anchor") or "").lower()
    assert str(rec.get("evidence_atom_kind") or "") == "sentence"


def test_resolve_paper_guide_support_slot_block_prefers_original_iism_dataset_exact_phrase(tmp_path: Path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "## Results / APR\n\n"
            "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.\n\n"
            "## Materials and Methods / APR\n\n"
            "The obtained shift vectors were then applied to the original iISM pinhole stack, enabling precise alignment of the off-axis pinhole images prior to summation.\n"
        ),
        encoding="utf-8",
    )

    rec = _resolve_paper_guide_support_slot_block(
        source_path=str(source_pdf),
        snippet="Where do the authors say the shift vectors are re-applied to the original iISM dataset?",
        heading="Methods / APR",
        prompt_family="method",
        claim_type="method_detail",
        db_dir=tmp_path,
    )

    assert "original iISM dataset" in str(rec.get("locate_anchor") or "")
