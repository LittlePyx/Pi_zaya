from __future__ import annotations

from pathlib import Path


def test_full_library_scope_drops_open_reader_source_preference():
    from kb import task_runtime

    current = r"F:\papers\ICCV-2019-Computational Hyperspectral Imaging.pdf"
    other = r"F:\papers\TMI-2018-Learned Primal-Dual Reconstruction.pdf"
    out = task_runtime._filter_current_paper_preference_for_scope(
        [current, other],
        effective_query_scope="library",
        bound_source_path=current,
        bound_source_name="ICCV-2019-Computational Hyperspectral Imaging",
    )

    assert out == [other]


def test_current_paper_scope_keeps_reader_source_preference():
    from kb import task_runtime

    current = r"F:\papers\ICCV-2019-Computational Hyperspectral Imaging.pdf"
    assert task_runtime._filter_current_paper_preference_for_scope(
        [current],
        effective_query_scope="current_paper",
        bound_source_path=current,
        bound_source_name="ICCV-2019-Computational Hyperspectral Imaging",
    ) == [current]


def test_named_paper_comparison_excludes_neighboring_retrieval_sources():
    from kb import task_runtime

    def hit(path: str) -> dict:
        return {"text": path, "meta": {"source_path": path, "source_name": Path(path).name}}

    learned = hit(r"db\TMI-2018-Learned Primal-Dual Reconstruction.en.md")
    ista = hit(r"db\CVPR-2018-ISTA-Net-Interpretable Optimization-Inspired Deep Network.en.md")
    hatnet = hit(r"db\CVPR-2024-Dual-Scale Transformer for Large-Scale Single-Pixel Imaging.en.md")

    out = task_runtime._focus_answer_seed_on_prompt_named_sources(
        [learned, ista, hatnet],
        prompt="比较 Learned Primal-Dual 与 ISTA-Net，并分别给出两篇论文的证据。",
    )

    assert out == [learned, ista]


def test_named_six_paper_comparison_keeps_titlecase_and_body_alias_sources():
    from kb import task_runtime

    def hit(path: str, heading: str, text: str = "") -> dict:
        return {
            "text": text or heading,
            "meta": {"source_path": path, "source_name": Path(path).name, "heading_path": heading},
        }

    rows = [
        hit(r"db\arXiv-Informer.en.md", "Informer: Beyond Efficient Transformer"),
        hit(r"db\arXiv-Autoformer.en.md", "Autoformer: Decomposition Transformers"),
        hit(r"db\FEDformer.en.md", "FEDformer"),
        hit(r"db\A Time Series is Worth 64 Words.en.md", "Patching", "PatchTST divides a series into patches."),
        hit(r"db\TimesNet.en.md", "TimesNet"),
        hit(r"db\iTransformer.en.md", "iTransformer"),
        hit(r"db\Unrelated Transformer Survey.en.md", "Survey"),
    ]
    prompt = (
        "请比较 Informer、Autoformer、FEDformer、PatchTST、TimesNet 和 iTransformer，"
        "每篇论文都给出证据。"
    )

    out = task_runtime._focus_answer_seed_on_prompt_named_sources(rows, prompt=prompt)

    assert out == rows[:6]


def test_named_microscopy_comparison_keeps_light_field_pdf_ligature_source():
    from kb import task_runtime

    def hit(path: str, text: str = "generic abstract passage") -> dict:
        return {
            "text": text,
            "meta": {"source_path": path, "source_name": Path(path).name},
        }

    structured = hit(
        r"db\iISM\image scanning microscopy.en.md",
        "Structured detection improves image scanning microscopy.",
    )
    interferometric = hit(
        r"db\s2ISM\interferometric image scanning microscopy.en.md",
        "Interferometric detection measures weak scattering signals.",
    )
    light_field = hit(
        "db\\QCLFM\\Quantum correlation light-\ufb01eld microscope.en.md"
    )
    unrelated = hit(r"db\Forecasting\transformer survey.en.md")

    out = task_runtime._focus_answer_seed_on_prompt_named_sources(
        [structured, interferometric, light_field, unrelated],
        prompt=(
            "显微成像这些 structured detection、interferometric、light-field "
            "方法分别是在解决什么麻烦？"
        ),
    )

    assert out == [structured, interferometric, light_field]


def test_explicit_per_paper_answer_budget_scales_only_for_four_or_more_sources():
    from kb import task_runtime

    prompt = "比较六篇论文；每篇论文都必须给出可定位证据。"
    assert task_runtime._max_tokens_for_explicit_per_source_answer(
        1216,
        prompt=prompt,
        source_count=6,
    ) == 1920
    assert task_runtime._max_tokens_for_explicit_per_source_answer(
        1216,
        prompt=prompt,
        source_count=3,
    ) == 1216
    assert task_runtime._max_tokens_for_explicit_per_source_answer(
        1216,
        prompt="比较六篇论文。",
        source_count=6,
    ) == 1216


def test_needs_bound_source_hint_for_inpaper_queries():
    from kb import task_runtime

    assert task_runtime._needs_bound_source_hint("这篇文章里的公式8是什么")
    assert task_runtime._needs_bound_source_hint("explain figure 3 in this paper")
    assert not task_runtime._needs_bound_source_hint("NatPhoton-2019-xxx.pdf 公式8是什么")


def test_pick_recent_bound_source_hints():
    from kb import task_runtime

    class FakeStore:
        def list_conversation_sources(self, conv_id: str, limit: int = 2):
            return [
                {"source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf", "source_path": "/x/a.en.md"},
                {"source_name": "", "source_path": "/x/LPR-2025-Advances and Challenges.en.md"},
                {"source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf", "source_path": "/x/a.en.md"},
            ]

    hints = task_runtime._pick_recent_bound_source_hints(conv_id="conv-1", chat_store=FakeStore(), limit=3)
    assert hints[0].startswith("NatPhoton-2019")
    assert any("LPR-2025" in h for h in hints)


def test_should_apply_implicit_source_hints_skips_plain_multi_paper_list_queries():
    from kb import task_runtime

    assert task_runtime._should_apply_implicit_source_hints(
        prompt="哪几篇文章里提到了NeRF",
        paper_guide_mode=False,
    ) is False
    assert task_runtime._should_apply_implicit_source_hints(
        prompt="Besides this paper, what other papers mention ADMM?",
        paper_guide_mode=True,
    ) is True


def test_paper_guide_supplemental_scan_prompts_keeps_specific_query_expansions():
    from kb import task_runtime

    original = "请解释 Learned Primal-Dual 怎样把 PDHG 展开成网络"
    variants = [
        original,
        "Learned Primal-Dual PDHG FBP motivation",
        "Learned PDHG proximal operators dual update primal update",
        "Learned Primal-Dual zero initialization FBP pseudo-inverse final results",
    ]

    out = task_runtime._paper_guide_supplemental_scan_prompts(
        prompt=original,
        retrieval_prompt=f"{original}\nQUERY SCOPE: Current paper.",
        used_query=original,
        query_variants=variants,
    )

    assert out[0] == original
    assert "zero initialization" in out[1]
    assert any("proximal operators" in item for item in out[:3])


def test_paper_guide_supplemental_scan_does_not_turn_generic_method_keyword_into_section_lock():
    from kb import task_runtime

    out = task_runtime._paper_guide_supplemental_scan_prompts(
        prompt="CASSI 与 DCD 的观测模型有什么区别？",
        retrieval_prompt="CASSI DCD observation model formula",
        used_query="CASSI DCD observation model formula",
        query_variants=["CASSI DCD DLTR method equation spectral model"],
    )

    assert out[0] == "CASSI 与 DCD 的观测模型有什么区别？"
    assert " method " not in f" {out[1].lower()} "


def test_collect_doc_figure_assets_and_append_markdown(tmp_path: Path):
    from kb import task_runtime

    doc_dir = tmp_path / "NatPhoton-2019-Principles and prospects for single-pixel imaging"
    assets_dir = doc_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig2 = assets_dir / "page_5_fig_2.png"
    fig2.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    md_path = doc_dir / "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    md_path.write_text(
        "\n".join(
            [
                "## Box 1",
                "![Figure](./assets/page_5_fig_2.png)",
                "**Figure 2.** Single-pixel imaging setup and reconstruction principle.",
            ]
        ),
        encoding="utf-8",
    )

    assets = task_runtime._collect_doc_figure_assets(md_path)
    assert assets
    assert int(assets[0].get("number") or 0) == 2

    answer = "这是该文中的 Fig.2。"
    hits = [
        {
            "text": "Figure 2 ...",
            "meta": {
                "source_path": str(md_path),
                "anchor_target_kind": "figure",
                "anchor_target_number": 2,
                "anchor_match_score": 9.0,
            },
        }
    ]
    out = task_runtime._maybe_append_library_figure_markdown(answer, prompt="这篇文章的第二张图是什么", answer_hits=hits)
    assert "### Library Figure" in out
    assert "/api/references/asset?path=" in out
    assert "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf" in out


def test_append_library_figure_markdown_prefers_bound_source(tmp_path: Path):
    from kb import task_runtime

    doc_a = tmp_path / "NatPhoton-2019"
    doc_b = tmp_path / "LPR-2025"
    for d in (doc_a, doc_b):
        (d / "assets").mkdir(parents=True, exist_ok=True)

    (doc_a / "assets" / "page_5_fig_1.png").write_bytes(b"\x89PNG\r\n\x1a\nA")
    (doc_b / "assets" / "page_8_fig_1.png").write_bytes(b"\x89PNG\r\n\x1a\nB")

    md_a = doc_a / "NatPhoton-2019.en.md"
    md_b = doc_b / "LPR-2025.en.md"
    md_a.write_text(
        "\n".join(
            [
                "## Example",
                "![Fig. 3](./assets/page_5_fig_1.png)",
                "Fig. 3",
            ]
        ),
        encoding="utf-8",
    )
    md_b.write_text(
        "\n".join(
            [
                "## Example",
                "![Fig. 3](./assets/page_8_fig_1.png)",
                "Fig. 3",
            ]
        ),
        encoding="utf-8",
    )

    hits = [
        {
            "text": "Figure 3 ...",
            "meta": {
                "source_path": str(md_b),
                "anchor_target_kind": "figure",
                "anchor_target_number": 3,
                "anchor_match_score": 7.0,
                "explicit_doc_match_score": 0.0,
            },
        },
        {
            "text": "Figure 3 ...",
            "meta": {
                "source_path": str(md_a),
                "anchor_target_kind": "figure",
                "anchor_target_number": 3,
                "anchor_match_score": 8.0,
                "explicit_doc_match_score": 8.6,
            },
        },
    ]

    out = task_runtime._maybe_append_library_figure_markdown(
        "这是图3。",
        prompt="NatPhoton-2019.pdf 这篇文章的图3是什么",
        answer_hits=hits,
    )
    assert out.count("/api/references/asset?path=") == 1
    assert "NatPhoton-2019.pdf" in out
    assert "LPR-2025.pdf" not in out
