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
