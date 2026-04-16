from pathlib import Path

from kb.paper_guide_focus import (
    _build_paper_guide_special_focus_block,
    _build_paper_guide_direct_abstract_answer,
    _extract_bound_paper_abstract,
    _extract_bound_paper_figure_caption,
    _extract_bound_paper_method_focus,
    _extract_caption_focus_fragment,
    _extract_caption_prompt_fragment,
    _extract_paper_guide_abstract_excerpt,
    _extract_paper_guide_method_detail_excerpt,
    _extract_paper_guide_special_focus_excerpt,
    _paper_guide_abstract_requests_translation,
    _repair_paper_guide_focus_answer_generic,
)


def test_extract_paper_guide_abstract_excerpt_prefers_abstract_section():
    text = """# Title

# Abstract
This is the abstract sentence one. This is the abstract sentence two.

# Introduction
This is not abstract.
"""
    out = _extract_paper_guide_abstract_excerpt(text, max_chars=400)
    assert "abstract sentence one" in out.lower()
    assert "not abstract" not in out.lower()


def test_extract_bound_paper_abstract_implicit_does_not_swallow_introduction(tmp_path: Path):
    doc_dir = tmp_path / "paper"
    doc_dir.mkdir()
    md = doc_dir / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Principles and prospects for single-pixel imaging",
                "",
                "Matthew P. Edgar$^{1,2}$, Graham M. Gibson$^{1,2}$ and Miles J. Padgett$^{1,2*}$",
                "",
                "Modern digital cameras employ silicon focal plane array (FPA) image sensors featuring millions of pixels.",
                "",
                "How many pixels does your camera have? The question that should really be asked is how many does your camera need?",
                "",
                "## How a single-pixel camera works",
                "",
                "A single-pixel camera is a technology that produces images by interrogating a scene...",
            ]
        ),
        encoding="utf-8",
    )
    out = _extract_bound_paper_abstract(str(md), db_dir=tmp_path)
    low = out.lower()
    assert "modern digital cameras employ" in low
    assert "how many pixels" not in low


def test_build_paper_guide_direct_abstract_answer_uses_titles_and_translation():
    class _FakeLLM:
        def chat(self, messages, temperature=0.0, max_tokens=0):
            return "这是一段中文翻译。"

    out = _build_paper_guide_direct_abstract_answer(
        prompt="把摘要翻译成中文",
        source_path="demo.pdf",
        db_dir=None,
        llm=_FakeLLM(),
        prefer_zh_locale=lambda _a, _b: True,
        extract_bound_paper_abstract=lambda source_path, db_dir=None: "This is the abstract.",
    )
    assert "摘要原文" in out
    assert "中文翻译" in out
    assert "This is the abstract." in out
    assert "这是一段中文翻译" in out


def test_build_paper_guide_direct_abstract_answer_adds_anchor_sentence_when_requested():
    out = _build_paper_guide_direct_abstract_answer(
        prompt="Please provide the abstract text and point me to one exact supporting sentence for locate jump.",
        source_path="demo.pdf",
        db_dir=None,
        llm=None,
        prefer_zh_locale=lambda _a, _b: False,
        extract_bound_paper_abstract=lambda source_path, db_dir=None: (
            "In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) technique "
            "for recovering the underlying 3D scene representation from a single temporal compressed image. "
            "SCI is a cost-effective method that enables recording high-dimensional data."
        ),
    )
    assert "Abstract text" in out
    assert "Anchor sentence for locate jump" in out
    assert "Snapshot Compressive Imaging" in out


def test_paper_guide_abstract_requests_translation_detects_english_and_chinese():
    assert _paper_guide_abstract_requests_translation("Translate the abstract into Chinese.")
    assert _paper_guide_abstract_requests_translation("把摘要翻译成中文")
    assert not _paper_guide_abstract_requests_translation("Show me the abstract only.")


def test_extract_bound_paper_method_focus_prefers_methods_block_with_phase_correlation(tmp_path: Path):
    doc_dir = tmp_path / "paper"
    doc_dir.mkdir()
    md = doc_dir / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Materials and Methods",
                "",
                "APR was performed using image registration based on phase correlation and these shift vectors were applied back to the original iISM dataset.",
                "",
                "# Results",
                "",
                "APR improves CNR.",
            ]
        ),
        encoding="utf-8",
    )
    out = _extract_bound_paper_method_focus(str(md), db_dir=tmp_path, focus_terms=["APR", "phase correlation"])
    low = out.lower()
    assert "phase correlation" in low
    assert "original iism dataset" in low


def test_extract_bound_paper_method_focus_combines_multiple_focus_terms_when_they_live_in_different_blocks(tmp_path: Path):
    doc_dir = tmp_path / "paper"
    doc_dir.mkdir()
    md = doc_dir / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Data analysis",
                "",
                "## Radial variance transform (RVT)",
                "",
                "Specifically, we use the radial variance transform (RVT), which converts an interferogram into an intensity-only map.",
                "",
                "## Adaptive pixel-reassignment (APR)",
                "",
                "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one.",
            ]
        ),
        encoding="utf-8",
    )
    out = _extract_bound_paper_method_focus(str(md), db_dir=tmp_path, focus_terms=["RVT", "APR"])
    low = out.lower()
    assert "radial variance transform" in low
    assert "phase correlation" in low


def test_extract_bound_paper_method_focus_prefers_term_specific_heading_for_single_focus_term(tmp_path: Path):
    doc_dir = tmp_path / "paper"
    doc_dir.mkdir()
    md = doc_dir / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Results",
                "",
                "We modified the workflow to account for phase information before APR can be applied.",
                "",
                "### Radial variance transform (RVT)",
                "",
                "Specifically, we use the radial variance transform (RVT), which converts an interferogram into an intensity-only map that reflects the local degree of symmetry.",
                "",
                "### Adaptive pixel-reassignment (APR)",
                "",
                "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one.",
            ]
        ),
        encoding="utf-8",
    )
    out = _extract_bound_paper_method_focus(str(md), db_dir=tmp_path, focus_terms=["RVT"])
    low = out.lower()
    assert "intensity-only map" in low
    assert "phase correlation" not in low


def test_extract_bound_paper_figure_caption_returns_matching_caption(tmp_path: Path):
    doc_dir = tmp_path / "paper"
    doc_dir.mkdir()
    md = doc_dir / "paper.en.md"
    md.write_text(
        "\n".join(
            [
                "# Results",
                "",
                "Figure 1. a Hardware. b Polarization response. c Line profiles.",
                "",
                "A later paragraph mentions Figure 1 again but is not the caption.",
            ]
        ),
        encoding="utf-8",
    )
    out = _extract_bound_paper_figure_caption(str(md), figure_num=1, db_dir=tmp_path)
    assert out.startswith("Figure 1.")
    assert "hardware" in out.lower()


def test_extract_caption_focus_fragment_prefers_missing_panel_clause_pair():
    excerpt = (
        "Figure 1. a Hardware. b Linear polarization iPSF. c Circular polarization iPSF. "
        "d Open pinhole. e Closed pinhole. f Interferometric iSM-APR. g Line profiles."
    )
    answer_text = "Panels (d) and (e) compare open and closed pinhole baselines."
    out = _extract_caption_focus_fragment(excerpt, answer_text=answer_text)
    low = out.lower()
    assert "f interferometric ism-apr" in low
    assert "g line profiles" in low


def test_extract_caption_prompt_fragment_targets_requested_panels():
    excerpt = (
        "Figure 1. a Hardware. b Linear polarization iPSF. c Circular polarization iPSF. "
        "d Open pinhole. e Closed pinhole. f Interferometric iSM-APR. g Line profiles."
    )
    out = _extract_caption_prompt_fragment(excerpt, prompt="Explain panels (f) and (g).")
    low = out.lower()
    assert "f interferometric ism-apr" in low
    assert "g line profiles" in low


def test_extract_paper_guide_method_detail_excerpt_prefers_applied_back_sentence():
    excerpt = (
        "APR improves the result. "
        "These RVT-APR shift vectors were applied back to the original iISM dataset. "
        "Another sentence is broader."
    )
    out = _extract_paper_guide_method_detail_excerpt(excerpt, focus_terms=["APR", "shift vectors"])
    low = out.lower()
    assert "applied back" in low
    assert "original iism dataset" in low


def test_extract_paper_guide_method_detail_excerpt_prefers_direct_apr_implementation_sentence():
    excerpt = (
        "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one. "
        "Image registration of the RVT pinhole stack yielded shift vectors for each off-axis pinhole image relative to the central one."
    )
    out = _extract_paper_guide_method_detail_excerpt(excerpt, focus_terms=["RVT", "APR"])
    low = out.lower()
    assert low.startswith("apr was performed")
    assert "phase correlation" in low


def test_extract_paper_guide_special_focus_excerpt_unwraps_focus_snippet():
    block = "Paper-guide method focus:\n- Keep names exact.\n- Focus snippet:\nAPR was performed using phase correlation."
    out = _extract_paper_guide_special_focus_excerpt(block)
    assert out == "APR was performed using phase correlation."


def test_build_paper_guide_special_focus_block_for_method_prefers_bound_focus():
    out = _build_paper_guide_special_focus_block(
        [
            {
                "heading": "Introduction",
                "snippet": "APR is the key innovation that improves CNR.",
                "deepread_texts": [],
            }
        ],
        prompt="Explain how this method works, especially what APR does in the pipeline.",
        prompt_family="method",
        source_path="demo.pdf",
        extract_bound_paper_method_focus=lambda source_path, db_dir=None, focus_terms=None: (
            "APR was performed using image registration based on phase correlation."
        ),
    )
    assert "Paper-guide method focus:" in out
    assert "phase correlation" in out


def test_build_paper_guide_special_focus_block_for_overview_role_prompt_uses_bound_method_focus():
    out = _build_paper_guide_special_focus_block(
        [],
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="overview",
        source_path="demo.pdf",
        extract_bound_paper_method_focus=lambda source_path, db_dir=None, focus_terms=None: (
            "Specifically, we use the radial variance transform (RVT), which converts an interferogram into an intensity-only map that reflects the local degree of symmetry.\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one. "
            "The obtained shift vectors were then applied back to the original iISM dataset."
        ),
    )
    low = out.lower()
    assert "paper-guide overview role focus" in low
    assert "intensity-only map" in low
    assert "phase correlation" in low


def test_build_paper_guide_special_focus_block_for_method_role_prompt_uses_overview_role_focus():
    out = _build_paper_guide_special_focus_block(
        [],
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="method",
        source_path="demo.pdf",
        extract_bound_paper_method_focus=lambda source_path, db_dir=None, focus_terms=None: (
            "Specifically, we use the radial variance transform (RVT), which converts an interferogram into an intensity-only map that reflects the local degree of symmetry.\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one. "
            "The obtained shift vectors were then applied back to the original iISM dataset."
        ),
    )
    low = out.lower()
    assert "paper-guide overview role focus" in low
    assert "intensity-only map" in low
    assert "phase correlation" in low


def test_build_paper_guide_special_focus_block_for_figure_uses_injected_caption():
    out = _build_paper_guide_special_focus_block(
        [],
        prompt="Walk me through Figure 1 panels (f) and (g).",
        prompt_family="figure_walkthrough",
        source_path="demo.pdf",
        requested_figure_number=lambda prompt, hits: 1,
        extract_bound_paper_figure_caption=lambda source_path, figure_num=0, db_dir=None: (
            "Figure 1. f Interferometric iSM-APR. g Line profiles."
        ),
    )
    assert "Requested figure: Figure 1" in out
    assert "Caption excerpt" in out


def test_repair_paper_guide_focus_answer_generic_replaces_not_stated_for_citation_lookup():
    out = _repair_paper_guide_focus_answer_generic(
        "Not stated in the retrieved evidence.",
        prompt="Which prior work is RVT attributed to in this paper?",
        prompt_family="citation_lookup",
        special_focus_block="Paper-guide citation focus:\n- Focus snippet:\nRVT was introduced in prior work [34].",
        extract_inline_reference_numbers=lambda text: [34] if "[34]" in text else [],
    )
    assert "introduced in prior work [34]" in out


def test_repair_paper_guide_focus_answer_generic_overrides_not_stated_for_exact_method_prompt():
    out = _repair_paper_guide_focus_answer_generic(
        "Not stated in the retrieved evidence.",
        prompt="Where exactly are the shift vectors applied back to the original iISM dataset?",
        prompt_family="method",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "These RVT-APR shift vectors were applied back to the original iISM dataset."
        ),
    )
    low = out.lower()
    assert "retrieved method evidence" in low
    assert "applied back to the original iism dataset" in low


def test_repair_paper_guide_focus_answer_generic_drops_contradicted_negative_focus_line():
    out = _repair_paper_guide_focus_answer_generic(
        (
            "RVT is named, but its function is not explained in the given excerpts.\n"
            "APR is not mentioned at all in the provided snippets.\n"
            "APR does not appear in the retrieved context."
        ),
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="method",
        special_focus_block=(
            "Paper-guide method focus:\n"
            "- Focus snippet:\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one."
        ),
        source_path="demo.pdf",
        db_dir=None,
        extract_bound_paper_method_focus=lambda source_path, **kwargs: (
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one."
        ),
    )
    low = out.lower()
    assert "apr is not mentioned" not in low
    assert "apr does not appear" not in low
    assert "apr is not found" not in low
    assert "apr is not referenced" not in low
    assert "implementation detail:" in low
    assert "phase correlation" in low


def test_repair_paper_guide_focus_answer_generic_replaces_overview_role_not_stated_shell():
    out = _repair_paper_guide_focus_answer_generic(
        (
            "RVT is introduced as a data-analysis step, but its purpose is not explained in the retrieved text.\n"
            "APR is not mentioned at all in the provided snippets."
        ),
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="overview",
        special_focus_block=(
            "Paper-guide overview role focus:\n"
            "- Focus snippet:\n"
            "Specifically, we use the radial variance transform (RVT), which converts an interferogram into an intensity-only map that reflects the local degree of symmetry.\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one. "
            "Image registration of the RVT pinhole stack yielded shift vectors. "
            "These vectors were then applied back to the original iISM dataset."
        ),
    )
    low = out.lower()
    assert "retrieved method evidence" in low
    assert "rvt turns the interferometric image into an intensity-only map" in low
    assert "apr uses phase-correlation registration to estimate shift vectors" in low
    assert "not explained" not in low
    assert "not mentioned" not in low


def test_repair_paper_guide_focus_answer_generic_treats_markdown_does_not_explain_as_not_stated_shell():
    out = _repair_paper_guide_focus_answer_generic(
        (
            "RVT is named explicitly in the Data analysis section, but the provided snippet does *not* explain what it does.\n"
            "Its function is not described in the given text."
        ),
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="overview",
        special_focus_block=(
            "Paper-guide overview role focus:\n"
            "- Focus snippet:\n"
            "To enable robust registration of the iISM pinhole stack independent of the interferometric phase, we applied RVT to it. "
            "RVT computes, for each pixel, the variance of intensity values along concentric circular areas of increasing radius and generates a new image in which pixel intensity encodes the degree of radial symmetry."
        ),
    )
    low = out.lower()
    assert "retrieved method evidence" in low
    assert "rvt converts each pinhole image into a radial-symmetry map" in low
    assert "does *not* explain" not in low
    assert "not described" not in low


def test_repair_paper_guide_focus_answer_generic_replaces_method_family_role_prompt_when_two_roles_are_grounded():
    out = _repair_paper_guide_focus_answer_generic(
        (
            "RVT is named but its function is not described.\n"
            "APR is not mentioned at all in the provided snippets."
        ),
        prompt="I am new to this paper. What are RVT and APR doing here, in simple terms?",
        prompt_family="method",
        special_focus_block=(
            "Paper-guide overview role focus:\n"
            "- Focus snippet:\n"
            "To enable robust registration of the iISM pinhole stack independent of the interferometric phase, we applied RVT to it. "
            "RVT computes, for each pixel, the variance of intensity values along concentric circular areas of increasing radius and generates a new image in which pixel intensity encodes the degree of radial symmetry.\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images with respect to the central one. "
            "Image registration of the RVT pinhole stack yielded shift vectors. "
            "These vectors were then applied back to the original iISM dataset."
        ),
    )
    low = out.lower()
    assert "retrieved method evidence" in low
    assert "rvt converts each pinhole image into a radial-symmetry map" in low
    assert "apr uses phase-correlation registration to estimate shift vectors" in low
