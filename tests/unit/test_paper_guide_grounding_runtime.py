from __future__ import annotations

from pathlib import Path

from kb.paper_guide_grounding_runtime import (
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
