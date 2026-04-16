import json

import kb.paper_guide_answer_post_runtime as answer_post_runtime


def test_resolve_exact_method_support_from_source_combines_framework_lr_and_batch_details(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "load_source_blocks",
        lambda _path: [
            {
                "kind": "paragraph",
                "block_id": "blk_training",
                "anchor_id": "anc_training",
                "heading_path": "Workflow and structure / Network training",
                "text": (
                    "We implemented the network on Ubuntu20 operating system using the Pytorch framework. "
                    "The network was optimized by Adam with an initial learning rate of 0.0003, and the batch size was set to 24."
                ),
            }
        ],
    )

    rec = answer_post_runtime._resolve_exact_method_support_from_source(
        "demo.pdf",
        prompt=(
            "In the Network training section, what framework, optimizer, initial learning rate, "
            "and batch size did they use? Point me to the exact supporting sentence."
        ),
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "Workflow and structure / Network training"
    assert "Pytorch framework" in rec["locate_anchor"]
    assert "Adam" in rec["locate_anchor"]
    assert "0.0003" in rec["locate_anchor"]
    assert "batch size was set to 24" in rec["locate_anchor"]


def test_resolve_exact_method_support_from_source_combines_beat_frequency_and_sampling_rate(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "load_source_blocks",
        lambda _path: [
            {
                "kind": "paragraph",
                "block_id": "blk_setup",
                "anchor_id": "anc_setup",
                "heading_path": "Experimental setup",
                "text": (
                    "Thus, the beat frequency of these two beams is 62,500 Hz, indicating a temporal period of 16 us. "
                    "Using a lens with 150-mm focal length, the combined light was collected by a photodetector, which was then digitized by a data acquisition card with a sampling rate of 1.25 Ms/s."
                ),
            }
        ],
    )

    rec = answer_post_runtime._resolve_exact_method_support_from_source(
        "demo.pdf",
        prompt=(
            "In the Experimental setup section, what beat frequency do they use, "
            "and what sampling rate does the data acquisition card use? Point me to the exact supporting sentence."
        ),
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "Experimental setup"
    assert "62,500 Hz" in rec["locate_anchor"]
    assert "sampling rate of 1.25 Ms/s" in rec["locate_anchor"]


def test_resolve_exact_citation_lookup_support_from_source_scans_source_blocks_when_targeted_hits_miss(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": "[31] Adaptive foveated single-pixel imaging via supersampling.",
                "meta": {"heading_path": "References", "block_id": "blk_ref31", "anchor_id": "anc_ref31"},
            }
        ],
    )
    monkeypatch.setattr(
        answer_post_runtime,
        "load_source_blocks",
        lambda _path: [
            {
                "kind": "paragraph",
                "block_id": "blk_intro",
                "anchor_id": "anc_intro",
                "heading_path": "Introduction",
                "text": (
                    "Recently, adaptive and smart sensing with dynamic supersampling was reported to combine with "
                    "compressive sensing in SPI. Thus, it significantly shortens acquisition time without considerably "
                    "sacrificing spatial information$^{31}$."
                ),
            }
        ],
    )

    rec = answer_post_runtime._resolve_exact_citation_lookup_support_from_source(
        "demo.pdf",
        prompt="Which reference do the authors cite for adaptive and smart sensing with dynamic supersampling, and where is that stated exactly?",
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "Introduction"
    assert rec["ref_nums"] == [31]
    assert "dynamic supersampling" in rec["locate_anchor"]


def test_resolve_exact_citation_lookup_support_from_source_prefers_single_ref_and_focus_clause(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": (
                    "Drawing inspiration from Compressed Sensing (CS) [5,8], "
                    "video Snapshot Compressive Imaging (SCI) [50] system has emerged to address these limitations."
                ),
                "meta": {"heading_path": "1. Introduction", "block_id": "blk_intro", "anchor_id": "anc_intro"},
            }
        ],
    )
    monkeypatch.setattr(answer_post_runtime, "load_source_blocks", lambda _path: [])

    rec = answer_post_runtime._resolve_exact_citation_lookup_support_from_source(
        "demo.pdf",
        prompt=(
            "In the Abstract, the authors mention video Snapshot Compressive Imaging (SCI). "
            "Which reference do they cite for SCI, and where is that stated exactly?"
        ),
        db_dir=tmp_path,
    )

    assert rec["ref_nums"] == [50]
    assert "SCI) [50]" in rec["locate_anchor"]
    assert "[5,8]" not in rec["locate_anchor"]


def test_resolve_exact_citation_lookup_support_from_source_keeps_multi_refs_for_plural_prompt(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": "The transformer framework [33,34] has recently attracted increasing attention.",
                "meta": {"heading_path": "Related Work", "block_id": "blk_rel", "anchor_id": "anc_rel"},
            }
        ],
    )
    monkeypatch.setattr(answer_post_runtime, "load_source_blocks", lambda _path: [])

    rec = answer_post_runtime._resolve_exact_citation_lookup_support_from_source(
        "demo.pdf",
        prompt="Which references do the authors cite for the transformer framework, and where is that stated exactly?",
        db_dir=tmp_path,
    )

    assert set(rec["ref_nums"]) == {33, 34}


def test_resolve_exact_citation_lookup_support_from_source_falls_back_to_reference_index(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    md_path.write_text("placeholder", encoding="utf-8")
    (assets_dir / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 7,
                        "reference_entry_id": "ref_0007",
                        "text": "[7] A. Other. Unrelated sensing pipeline. Journal, 2023.",
                        "doi": "",
                        "year": "2023",
                        "parse_confidence": 0.7,
                    },
                    {
                        "ref_num": 31,
                        "reference_entry_id": "ref_0031",
                        "text": "[31] Adaptive foveated single-pixel imaging via supersampling. Optics Express, 2021.",
                        "doi": "",
                        "year": "2021",
                        "parse_confidence": 0.9,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(answer_post_runtime, "_paper_guide_targeted_source_block_hits", lambda **_kwargs: [])
    monkeypatch.setattr(answer_post_runtime, "load_source_blocks", lambda _path: [])

    rec = answer_post_runtime._resolve_exact_citation_lookup_support_from_source(
        "demo.pdf",
        prompt="Which reference do the authors cite for adaptive and smart sensing with dynamic supersampling, and where is that stated exactly?",
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "References"
    assert rec["ref_nums"] == [31]
    assert "supersampling" in rec["locate_anchor"].lower()


def test_resolve_exact_equation_support_from_source_picks_equation_and_neighbor_explanation(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "load_source_blocks",
        lambda _path: [
            {
                "kind": "paragraph",
                "block_id": "blk_intro",
                "anchor_id": "anc_intro",
                "heading_path": "3. Method / 3.1. Background on NeRF",
                "text": "The color $C(r)$ of a ray $r=o+td$ can be written as",
            },
            {
                "kind": "equation",
                "block_id": "blk_eq1",
                "anchor_id": "eq_00001",
                "heading_path": "3. Method / 3.1. Background on NeRF",
                "number": 1,
                "raw_text": "$$\nC(\\mathbf{r}) = \\int_{t_n}^{t_f} T(t)\\sigma(\\mathbf{r}(t))\\mathbf{c}(\\mathbf{r}(t),\\mathbf{d})dt, \\tag{1}\n$$",
            },
            {
                "kind": "paragraph",
                "block_id": "blk_where",
                "anchor_id": "anc_where",
                "heading_path": "3. Method / 3.1. Background on NeRF",
                "text": "where $t_n$ and $t_f$ are near and far bounds for volumetric rendering respectively, and $T(t)$ represents the accumulated transmittance.",
            },
        ],
    )

    rec = answer_post_runtime._resolve_exact_equation_support_from_source(
        "demo.pdf",
        prompt=(
            "What does Equation (1) define in this paper, and where do the authors define the variables "
            "like t_n, t_f, and T(t)? Point me to the exact supporting part."
        ),
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "3. Method / 3.1. Background on NeRF"
    assert rec["equation_number"] == 1
    assert "\\tag{1}" in rec["equation_markdown"]
    assert "The color $C(r)$" in rec["leadin_text"]
    assert "where $t_n$ and $t_f$" in rec["explanation_text"]


def test_resolve_exact_equation_support_from_source_uses_equation_index_when_blocks_missing(tmp_path, monkeypatch):
    doc_dir = tmp_path / "DemoEquation"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "DemoEquation.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    (assets_dir / "equation_index.json").write_text(
        json.dumps(
            {
                "equations": [
                    {
                        "equation_number": 3,
                        "equation_markdown": "$$Y = \\\\sum_i X_i \\\\odot M_i + Z \\\\tag{3}$$",
                        "context_before": "The SCI measurement can be written as follows.",
                        "context_after": "where Y is the measurement, X_i are frames, and M_i are masks.",
                        "block_id": "blk_eq3",
                        "anchor_id": "eq_00003",
                        "heading_path": "Method / SCI formulation",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(answer_post_runtime, "load_source_blocks", lambda _path: [])

    rec = answer_post_runtime._resolve_exact_equation_support_from_source(
        "demo.pdf",
        prompt="Where do the authors define the variables in equation (3)? Point me to the exact supporting part.",
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "Method / SCI formulation"
    assert rec["equation_block_id"] == "blk_eq3"
    assert rec["equation_anchor_id"] == "eq_00003"
    assert "\\tag{3}" in rec["equation_markdown"]
    assert "SCI measurement can be written" in rec["leadin_text"]
    assert "where Y is the measurement" in rec["explanation_text"]


def test_apply_paper_guide_answer_postprocess_runs_focus_support_and_sanitize(monkeypatch):
    calls = []

    monkeypatch.setattr(answer_post_runtime, "_repair_paper_guide_focus_answer_generic", lambda answer, **kwargs: calls.append(("repair", kwargs)) or (answer + " [repair]"))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: calls.append(("inject_support", kwargs)) or (answer + " [support]"))
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_paper_guide_support_markers",
        lambda answer, **kwargs: (calls.append(("resolve_support", kwargs)) or (answer + " [resolved]", [{"line_index": 0, "cite_policy": "locate_only"}])),
    )
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: calls.append(("focus_cite", kwargs)) or (answer + " [focuscite]"))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: calls.append(("card_cite", kwargs)) or (answer + " [cardcite]"))
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: calls.append(("drop_locate", kwargs)) or (answer + " [drop]"))
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: calls.append(("sanitize", kwargs)) or (answer + " [sanitized]"))

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "base",
        paper_guide_mode=True,
        prompt="How is APR grounded?",
        prompt_for_user="How is APR grounded?",
        prompt_family="method",
        special_focus_block="focus block",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        support_slots=[{"support_example": "[[SUPPORT:DOC-1]]"}],
        cards=[{"doc_idx": 1}],
        locked_citation_source=None,
    )

    assert out.endswith("[sanitized]")
    assert support_resolution == [{"line_index": 0, "cite_policy": "locate_only"}]
    assert [name for name, _ in calls] == [
        "repair",
        "inject_support",
        "resolve_support",
        "focus_cite",
        "card_cite",
        "drop_locate",
        "sanitize",
    ]


def test_apply_paper_guide_answer_postprocess_skips_locate_only_drop_for_overview(monkeypatch):
    calls = []

    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_paper_guide_support_markers",
        lambda answer, **kwargs: (
            answer + " [[CITE:s1234abcd:26]]",
            [{"line_index": 0, "cite_policy": "locate_only", "resolved_ref_num": 0}],
        ),
    )
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_drop_paper_guide_locate_only_line_citations",
        lambda answer, **kwargs: calls.append("drop_locate") or answer.replace("[[CITE:s1234abcd:26]]", ""),
    )
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)

    out, _support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "Overview answer",
        paper_guide_mode=True,
        prompt="What does this paper do?",
        prompt_for_user="What does this paper do?",
        prompt_family="overview",
        special_focus_block="",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert calls == []
    assert "[[CITE:s1234abcd:26]]" in out


def test_apply_paper_guide_answer_postprocess_forces_exact_equation_surface(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_exact_equation_support_from_source",
        lambda source_path, **_kwargs: {
            "source_path": source_path,
            "heading_path": "3. Method / 3.1. Background on NeRF",
            "equation_number": 1,
            "equation_markdown": "$$\nC(\\mathbf{r}) = \\int_{t_n}^{t_f} ... \\tag{1}\n$$",
            "equation_block_id": "blk_eq1",
            "equation_anchor_id": "eq_00001",
            "leadin_text": "The color $C(r)$ of a ray $r=o+td$ can be written as",
            "explanation_text": "where $t_n$ and $t_f$ are near and far bounds for volumetric rendering respectively.",
            "explanation_block_id": "blk_where",
            "explanation_anchor_id": "anc_where",
        },
    )

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "placeholder",
        paper_guide_mode=True,
        prompt=(
            "What does Equation (1) define in this paper, and where do the authors define the variables "
            "like t_n and t_f? Point me to the exact supporting part."
        ),
        prompt_for_user=(
            "What does Equation (1) define in this paper, and where do the authors define the variables "
            "like t_n and t_f? Point me to the exact supporting part."
        ),
        prompt_family="equation",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "Equation (1) is stated in 3. Method / 3.1. Background on NeRF:" in out
    assert "The lead-in sentence says" in out
    assert "The variable definitions appear immediately after the equation:" in out
    assert len(support_resolution) == 2
    assert support_resolution[0]["block_id"] == "blk_eq1"
    assert support_resolution[0]["claim_type"] == "formula_claim"
    assert support_resolution[1]["block_id"] == "blk_where"
    assert support_resolution[1]["claim_type"] == "equation_explanation_claim"


def test_apply_paper_guide_answer_postprocess_uses_figure_caption_fallback(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_requested_figure_number", lambda prompt, answer_hits: 2)
    monkeypatch.setattr(answer_post_runtime, "_extract_bound_paper_figure_caption", lambda source_path, **kwargs: "Figure 2 caption")
    monkeypatch.setattr(answer_post_runtime, "_repair_paper_guide_focus_answer_generic", lambda answer, **kwargs: answer + "\n" + kwargs["special_focus_block"])
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, []))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "base",
        paper_guide_mode=True,
        prompt="Walk me through Figure 2.",
        prompt_for_user="Walk me through Figure 2.",
        prompt_family="figure_walkthrough",
        special_focus_block="",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "Paper-guide figure focus:" in out
    assert "Figure 2 caption" in out
    assert support_resolution == []


def test_apply_paper_guide_answer_postprocess_forces_exact_figure_caption_clause(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, []))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_exact_figure_panel_caption_support_from_source",
        lambda source_path, **kwargs: {
            "source_path": source_path,
            "block_id": "blk_cap",
            "anchor_id": "p_00099",
            "heading_path": "Results / Figure 3",
            "locate_anchor": "(f) methane imaging using SPC$^{15}$",
            "claim_type": "figure_panel",
            "cite_policy": "locate_only",
            "segment_text": "(f) methane imaging using SPC$^{15}$",
            "segment_index": -1,
            "figure_number": 3,
            "panel_letters": ["f"],
        },
    )

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "base",
        paper_guide_mode=True,
        prompt="For Figure 3 panel (f), what exactly does that panel correspond to? Point me to the exact supporting caption clause.",
        prompt_for_user="For Figure 3 panel (f), what exactly does that panel correspond to? Point me to the exact supporting caption clause.",
        prompt_family="figure_walkthrough",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "Figure 3 caption for panel (f)" in out
    assert "(f) methane imaging using SPC" in out
    assert support_resolution and support_resolution[0].get("block_id") == "blk_cap"


def test_apply_paper_guide_answer_postprocess_exact_citation_sets_resolved_ref_num(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_exact_citation_lookup_support_from_source",
        lambda source_path, **kwargs: {
            "source_path": source_path,
            "block_id": "blk_rel",
            "anchor_id": "anc_rel",
            "heading_path": "2. Related Work",
            "locate_anchor": "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4].",
            "claim_type": "prior_work",
            "cite_policy": "prefer_ref",
            "segment_text": "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4].",
            "segment_index": -1,
            "ref_nums": [4],
        },
    )

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "placeholder",
        paper_guide_mode=True,
        prompt="Which reference do the authors cite for ADMM, and where is that stated exactly?",
        prompt_for_user="Which reference do the authors cite for ADMM, and where is that stated exactly?",
        prompt_family="citation_lookup",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "The paper cites [4] for this point." in out
    assert support_resolution == [
        {
            "source_path": "bound.md",
            "block_id": "blk_rel",
            "anchor_id": "anc_rel",
            "heading_path": "2. Related Work",
            "locate_anchor": "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4].",
            "claim_type": "prior_work",
            "cite_policy": "prefer_ref",
            "segment_text": "Most of the existing methods employ alternating direction method of multipliers (ADMM) [4].",
            "segment_index": -1,
            "ref_nums": [4],
            "candidate_refs": [4],
            "resolved_ref_num": 4,
        }
    ]


def test_apply_paper_guide_answer_postprocess_exact_citation_sets_resolved_ref_num_for_multi_refs(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_exact_citation_lookup_support_from_source",
        lambda source_path, **kwargs: {
            "source_path": source_path,
            "block_id": "blk_abs",
            "anchor_id": "anc_abs",
            "heading_path": "Abstract",
            "locate_anchor": "The transformer framework [33,34] has recently attracted increasing attention.",
            "claim_type": "prior_work",
            "cite_policy": "prefer_ref",
            "segment_text": "The transformer framework [33,34] has recently attracted increasing attention.",
            "segment_index": -1,
            "ref_nums": [33, 34],
        },
    )

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "placeholder",
        paper_guide_mode=True,
        prompt="Which references do the authors cite for the transformer framework, and where is that stated exactly?",
        prompt_for_user="Which references do the authors cite for the transformer framework, and where is that stated exactly?",
        prompt_family="citation_lookup",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "The paper cites [33], [34] for this point." in out
    assert support_resolution and support_resolution[0].get("candidate_refs") == [33, 34]
    assert support_resolution and int(support_resolution[0].get("resolved_ref_num") or 0) == 33


def test_apply_paper_guide_answer_postprocess_exact_citation_extracts_refs_from_anchor_when_ref_nums_missing(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_exact_citation_lookup_support_from_source",
        lambda source_path, **kwargs: {
            "source_path": source_path,
            "block_id": "blk_abs",
            "anchor_id": "anc_abs",
            "heading_path": "Abstract",
            "locate_anchor": "The transformer framework [33,34] has recently attracted increasing attention.",
            "claim_type": "prior_work",
            "cite_policy": "prefer_ref",
            "segment_text": "The transformer framework [33,34] has recently attracted increasing attention.",
            "segment_index": -1,
            # no ref_nums on purpose; should be extracted from locate_anchor
        },
    )

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "placeholder",
        paper_guide_mode=True,
        prompt="Which references do the authors cite for the transformer framework, and where is that stated exactly?",
        prompt_for_user="Which references do the authors cite for the transformer framework, and where is that stated exactly?",
        prompt_family="citation_lookup",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "The paper cites [33], [34] for this point." in out
    assert support_resolution and support_resolution[0].get("candidate_refs") == [33, 34]
    assert support_resolution and int(support_resolution[0].get("resolved_ref_num") or 0) == 33


def test_apply_paper_guide_answer_postprocess_citation_lookup_backfills_resolved_ref_from_candidate_refs(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_paper_guide_support_markers",
        lambda answer, **kwargs: (
            answer,
            [
                {
                    "heading_path": "Abstract",
                    "locate_anchor": "The transformer framework [33,34] has recently attracted increasing attention.",
                    "claim_type": "prior_work",
                    "candidate_refs": [33, 34],
                    "resolved_ref_num": 0,
                }
            ],
        ),
    )
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "The paper cites [33], [34] for this point.",
        paper_guide_mode=True,
        prompt="Which references do the authors cite for the transformer framework?",
        prompt_for_user="Which references do the authors cite for the transformer framework?",
        prompt_family="citation_lookup",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "The paper cites [33], [34] for this point." in out
    assert support_resolution and support_resolution[0].get("candidate_refs") == [33, 34]
    assert support_resolution and int(support_resolution[0].get("resolved_ref_num") or 0) == 33


def test_apply_paper_guide_answer_postprocess_citation_lookup_backfills_from_support_ref_candidates(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_paper_guide_support_markers",
        lambda answer, **kwargs: (
            answer,
            [
                {
                    "heading_path": "Abstract",
                    "locate_anchor": "The transformer framework [33,34] has recently attracted increasing attention.",
                    "claim_type": "prior_work",
                    "support_ref_candidates": [33, 34],
                    "resolved_ref_num": 0,
                }
            ],
        ),
    )
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)

    _out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "The paper cites [33], [34] for this point.",
        paper_guide_mode=True,
        prompt="Which references do the authors cite for the transformer framework?",
        prompt_for_user="Which references do the authors cite for the transformer framework?",
        prompt_family="citation_lookup",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert support_resolution and support_resolution[0].get("candidate_refs") == [33, 34]
    assert support_resolution and int(support_resolution[0].get("resolved_ref_num") or 0) == 33


def test_resolve_exact_citation_lookup_support_from_source_merges_split_voc_focus_fragments(tmp_path, monkeypatch):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "_paper_guide_targeted_source_block_hits",
        lambda **_kwargs: [
            {
                "text": (
                    "One way to acquire high-resolution images is placing another CMOS or CCD camera "
                    "to take images of the same target, which introduces additional workload [24]."
                ),
                "meta": {"heading_path": "Abstract", "block_id": "blk_24", "anchor_id": "anc_24"},
            }
        ],
    )
    monkeypatch.setattr(
        answer_post_runtime,
        "load_source_blocks",
        lambda _path: [
            {
                "kind": "paragraph",
                "block_id": "blk_voc1",
                "anchor_id": "anc_voc1",
                "heading_path": "Abstract",
                "text": "With the calibrated physical noise model, we employed off-the-shelf images from PASCAL VOC2007 [31] and",
            },
            {
                "kind": "paragraph",
                "block_id": "blk_noise",
                "anchor_id": "anc_noise",
                "heading_path": "Abstract",
                "text": "*model, a cat toy and different printed shapes) using the reported technique.*",
            },
            {
                "kind": "paragraph",
                "block_id": "blk_voc2",
                "anchor_id": "anc_voc2",
                "heading_path": "Abstract",
                "text": (
                    "VOC2012 [32] datasets) to synthesize training data. "
                    "The transformer framework [33,34] has recently attracted increasing attention."
                ),
            },
        ],
    )

    rec = answer_post_runtime._resolve_exact_citation_lookup_support_from_source(
        "demo.pdf",
        prompt=(
            "The authors mention using off-the-shelf public high-resolution images from "
            "PASCAL VOC2007/VOC2012 to synthesize training data. Which references correspond "
            "to those datasets, and where is that stated exactly?"
        ),
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "Abstract"
    assert 31 in set(rec["ref_nums"])
    assert 32 in set(rec["ref_nums"])
    assert "voc2007" in str(rec["locate_anchor"]).lower()
    assert "voc2012" in str(rec["locate_anchor"]).lower()


def test_apply_paper_guide_answer_postprocess_detects_section_by_section_doc_map_prompt(monkeypatch):
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_doc_map_records_from_source",
        lambda source_path, **kwargs: [
            {
                "source_path": source_path,
                "block_id": "blk_intro",
                "anchor_id": "p_00001",
                "heading_path": "Introduction",
                "locate_anchor": "This section introduces the motivation and core contribution.",
                "claim_type": "doc_map",
                "cite_policy": "locate_only",
                "segment_text": "This section introduces the motivation and core contribution.",
                "segment_index": -1,
            }
        ],
    )

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "placeholder",
        paper_guide_mode=True,
        prompt="First give me a section-by-section reading map of the markdown: one verbatim anchor sentence for each major section, and keep the answer navigable for locate jumps.",
        prompt_for_user="First give me a section-by-section reading map of the markdown: one verbatim anchor sentence for each major section, and keep the answer navigable for locate jumps.",
        prompt_family="overview",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert out.startswith("Doc map (verbatim anchors by section):")
    assert "1. Introduction" in out
    assert support_resolution and support_resolution[0].get("claim_type") == "doc_map"


def test_extract_caption_panel_clause_supports_plain_and_bold_markers():
    clause = answer_post_runtime._extract_caption_panel_clause(
        "Figure 6. | Illustration of the network. a The workflow and structure. b The enhancement comparison.",
        panel_letter="b",
    )
    assert clause.lower().startswith("b ")
    assert "enhancement comparison" in clause

    clause2 = answer_post_runtime._extract_caption_panel_clause(
        "**Figure 3.** | Thumbnail images: **a** X-ray; **b** visible SPC; **c** SWIR.",
        panel_letter="b",
    )
    assert "b" in clause2.lower()
    assert "visible" in clause2.lower()


def test_resolve_exact_figure_panel_caption_support_from_source_uses_caption_continuation_heading(tmp_path, monkeypatch):
    doc_dir = tmp_path / "DemoFigure"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "DemoFigure.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "## Results",
                "",
                "![Figure 6](./assets/page_10_fig_1.png)",
                "",
                "**Figure 6.** Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging.",
                "",
                "a The workflow and structure of the reported network. b The enhancement comparison between CNN-based U-net network and the reported transformer-based network.",
            ]
        ),
        encoding="utf-8",
    )
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "page": 10,
                        "index": 1,
                        "asset_name": "page_10_fig_1.png",
                        "asset_name_raw": "page_10_fig_1.png",
                        "asset_name_alias": "fig_6.png",
                        "figure_id": "fig_006",
                        "figure_ident": "6",
                        "paper_figure_number": 6,
                        "caption": "Figure 6. Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging.",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)

    rec = answer_post_runtime._resolve_exact_figure_panel_caption_support_from_source(
        "demo.pdf",
        prompt="In Figure 6 panel (b), where is the enhancement comparison between the CNN-based U-net and the reported transformer-based network stated exactly?",
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "Demo / Results / Figure 6"
    assert rec["claim_type"] == "figure_panel"
    assert rec["panel_letters"] == ["b"]
    assert "enhancement comparison" in rec["locate_anchor"].lower()


def test_resolve_exact_figure_panel_caption_support_from_source_prefers_figure_index_binding(tmp_path, monkeypatch):
    doc_dir = tmp_path / "DemoFigureIndex"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = doc_dir / "DemoFigureIndex.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "paper_figure_number": 6,
                        "figure_id": "fig_006",
                        "figure_block_id": "blk_fig6",
                        "caption_block_id": "blk_cap6",
                        "caption_anchor_id": "p_00098",
                        "anchor_id": "fg_00006",
                        "heading_path": "Demo / Results / Figure 6",
                        "caption": "Illustration of the reported deep transformer network.",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(
        answer_post_runtime,
        "load_source_blocks",
        lambda _path: [
            {
                "kind": "figure",
                "block_id": "blk_fig6",
                "anchor_id": "fg_00006",
                "heading_path": "Demo / Results / Figure 6",
                "figure_id": "fig_006",
                "paper_figure_number": 6,
                "text": "Figure 6",
            },
            {
                "kind": "paragraph",
                "block_id": "blk_cap6",
                "anchor_id": "p_00098",
                "heading_path": "Demo / Results / Figure 6",
                "figure_role": "caption",
                "linked_figure_block_id": "blk_fig6",
                "text": "Illustration of the reported deep transformer network for high-fidelity large-scale single-photon imaging.",
            },
            {
                "kind": "paragraph",
                "block_id": "blk_cap6b",
                "anchor_id": "p_00099",
                "heading_path": "Demo / Results / Figure 6",
                "figure_role": "caption_continuation",
                "linked_figure_block_id": "blk_fig6",
                "text": "a The workflow and structure of the reported network. b The enhancement comparison between CNN-based U-net network and the reported transformer-based network.",
            },
        ],
    )

    rec = answer_post_runtime._resolve_exact_figure_panel_caption_support_from_source(
        "demo.pdf",
        prompt="In Figure 6 panel (b), where is the enhancement comparison between the CNN-based U-net and the reported transformer-based network stated exactly?",
        db_dir=tmp_path,
    )

    assert rec["heading_path"] == "Demo / Results / Figure 6"
    assert rec["block_id"] == "blk_cap6"
    assert rec["anchor_id"] == "p_00098"
    assert rec["claim_type"] == "figure_panel"
    assert rec["panel_letters"] == ["b"]
    assert "enhancement comparison" in rec["locate_anchor"].lower()


def test_apply_paper_guide_answer_postprocess_promotes_numeric_cites_for_method(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, []))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_promote_paper_guide_numeric_reference_citations", lambda answer, **kwargs: answer + " [[CITE:s123:35]]")

    out, _ = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "base [35]",
        paper_guide_mode=True,
        prompt="How is APR grounded?",
        prompt_for_user="How is APR grounded?",
        prompt_family="method",
        special_focus_block="",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[{"meta": {"source_path": "focus.md"}}],
        support_slots=[],
        cards=[],
        locked_citation_source={"sid": "s123", "source_name": "demo.pdf"},
    )

    assert out.endswith("[[CITE:s123:35]]")


def test_apply_paper_guide_answer_postprocess_forces_exact_method_surface(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, []))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)

    out, _ = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "The paper states this explicitly in the retrieved method evidence.",
        paper_guide_mode=True,
        prompt="In the APR pipeline, where do the authors say the shift vectors are re-applied to the original iISM dataset? Point me to the exact supporting part of the paper.",
        prompt_for_user="In the APR pipeline, where do the authors say the shift vectors are re-applied to the original iISM dataset? Point me to the exact supporting part of the paper.",
        prompt_family="method",
        special_focus_block="",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[
            {
                "heading_path": "Results / APR",
                "locate_anchor": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset, yielding reconstructions with enhanced spatial resolution.",
            }
        ],
        cards=[],
        locked_citation_source=None,
    )

    assert "original iISM dataset" in out
    assert "Results / APR" in out


def test_apply_paper_guide_answer_postprocess_forces_exact_method_surface_from_bound_focus(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, []))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "_extract_bound_paper_method_focus",
        lambda source_path, **kwargs: "The obtained shift vectors were then applied to the original iISM pinhole stack.",
    )
    monkeypatch.setattr(
        answer_post_runtime,
        "_extract_paper_guide_method_detail_excerpt",
        lambda excerpt, **kwargs: excerpt,
    )
    monkeypatch.setattr(
        answer_post_runtime,
        "_extract_paper_guide_method_focus_terms",
        lambda prompt: ["APR", "shift vectors"],
    )
    monkeypatch.setattr(
        answer_post_runtime,
        "_resolve_exact_method_support_from_source",
        lambda source_path, **kwargs: {
            "source_path": source_path,
            "block_id": "blk_exact",
            "anchor_id": "anc_exact",
            "heading_path": "Results / APR",
            "locate_anchor": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
            "claim_type": "method_detail",
            "cite_policy": "locate_only",
            "segment_text": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
        },
    )

    out, support_resolution = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "The paper states this explicitly in the retrieved method evidence.",
        paper_guide_mode=True,
        prompt="In the APR pipeline, where do the authors say the shift vectors are re-applied to the original iISM dataset? Point me to the exact supporting part of the paper.",
        prompt_for_user="In the APR pipeline, where do the authors say the shift vectors are re-applied to the original iISM dataset? Point me to the exact supporting part of the paper.",
        prompt_family="method",
        special_focus_block="",
        focus_source_path="",
        direct_source_path="",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "original iISM dataset" in out
    assert "Results / APR" in out
    assert support_resolution == [
        {
            "source_path": "bound.md",
            "block_id": "blk_exact",
            "anchor_id": "anc_exact",
            "heading_path": "Results / APR",
            "locate_anchor": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
            "claim_type": "method_detail",
            "cite_policy": "locate_only",
            "segment_text": "Finally, these RVT-APR shift vectors were applied back to the original iISM dataset.",
            "segment_index": -1,
        }
    ]


def test_resolve_exact_method_support_from_source_appends_inline_heading_when_prompt_targets_it(tmp_path):
    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "# Methods\n\n"
            "**Experimental setup.** Thus, the beat frequency of these two beams is 62,500 Hz. "
            "The data acquisition card uses a sampling rate of 1.25 Ms/s.\n"
        ),
        encoding="utf-8",
    )

    rec = answer_post_runtime._resolve_exact_method_support_from_source(
        str(source_pdf),
        prompt="In the Experimental setup section, what beat frequency do they use, and what sampling rate does the data acquisition card use?",
        db_dir=tmp_path,
    )

    assert "Experimental setup" in str(rec.get("heading_path") or "")
    assert "62,500 Hz" in str(rec.get("locate_anchor") or "")
    assert "1.25 Ms/s" in str(rec.get("locate_anchor") or "")


def test_apply_paper_guide_answer_postprocess_surfaces_plain_text_box_formula(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, [{"locate_anchor": "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$, the image in the transform domain can be reconstructed."}]))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)

    out, _ = answer_post_runtime._apply_paper_guide_answer_postprocess(
        "From Box 1, the reconstruction condition is $$ M \\ge O(K \\log(N/K)) $$.",
        paper_guide_mode=True,
        prompt="From Box 1 only, what condition do the authors give for reconstructing the image in the transform domain?",
        prompt_for_user="From Box 1 only, what condition do the authors give for reconstructing the image in the transform domain?",
        prompt_family="box_only",
        special_focus_block="",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert "M >= O(K log(N/K))" in out


def test_apply_paper_guide_answer_postprocess_strips_contradicted_method_focus_line(monkeypatch):
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, []))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)

    out, _ = answer_post_runtime._apply_paper_guide_answer_postprocess(
        (
            "APR is not mentioned in the retrieved context.\n\n"
            "Implementation detail: APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one."
        ),
        paper_guide_mode=True,
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_for_user="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="method",
        special_focus_block="",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="bound.md",
        db_dir="db",
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    low = out.lower()
    assert "apr is not mentioned" not in low
    assert "implementation detail:" in low


def test_apply_paper_guide_answer_postprocess_adds_component_role_support_resolution(monkeypatch, tmp_path):
    md_path = tmp_path / "paper.en.md"
    md_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_md_path", lambda *_args, **_kwargs: md_path)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_support_markers", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_resolve_paper_guide_support_markers", lambda answer, **kwargs: (answer, []))
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_focus_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_inject_paper_guide_card_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_drop_paper_guide_locate_only_line_citations", lambda answer, **kwargs: answer)
    monkeypatch.setattr(answer_post_runtime, "_sanitize_paper_guide_answer_for_user", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        answer_post_runtime,
        "load_source_blocks",
        lambda _path: [
            {
                "kind": "paragraph",
                "block_id": "blk_rvt",
                "anchor_id": "anc_rvt",
                "heading_path": "Methods / Data analysis / Radial variance transform (RVT)",
                "text": (
                    "RVT computes, for each pixel, the variance of intensity values along concentric circular areas "
                    "of increasing radius and generates a new image in which pixel intensity encodes the degree of radial symmetry. "
                    "The RVT output can therefore be used for subsequent pixel-reassignment analysis."
                ),
            },
            {
                "kind": "paragraph",
                "block_id": "blk_apr",
                "anchor_id": "anc_apr",
                "heading_path": "Methods / Data analysis / Adaptive pixel-reassignment (APR)",
                "text": (
                    "APR was performed using image registration based on phase correlation of the off-axis raw images "
                    "with respect to the central one. Image registration of the RVT pinhole stack yielded shift vectors. "
                    "These vectors were then applied to the original iISM pinhole stack prior to summation."
                ),
            },
        ],
    )

    answer = (
        "From the retrieved method evidence, in simple terms:\n"
        "- RVT converts each pinhole image into a radial-symmetry map so the registration step is more robust to interferometric phase.\n"
        "- APR uses phase-correlation registration to estimate shift vectors, then applies those shifts back to the original iISM data before summation."
    )
    out, support = answer_post_runtime._apply_paper_guide_answer_postprocess(
        answer,
        paper_guide_mode=True,
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_for_user="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="overview",
        special_focus_block="",
        focus_source_path="focus.md",
        direct_source_path="direct.md",
        bound_source_path="demo.pdf",
        db_dir=tmp_path,
        answer_hits=[],
        support_slots=[],
        cards=[],
        locked_citation_source=None,
    )

    assert out == answer
    assert len(support) == 2
    by_segment = {str(item.get("segment_text") or ""): item for item in support}
    rvt_seg = by_segment[
        "RVT converts each pinhole image into a radial-symmetry map so the registration step is more robust to interferometric phase."
    ]
    apr_seg = by_segment[
        "APR uses phase-correlation registration to estimate shift vectors, then applies those shifts back to the original iISM data before summation."
    ]
    assert "RVT" in rvt_seg["heading_path"]
    assert "radial symmetry" in rvt_seg["locate_anchor"].lower()
    assert "APR" in apr_seg["heading_path"]
    assert "phase correlation" in apr_seg["locate_anchor"].lower()
