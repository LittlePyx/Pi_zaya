from __future__ import annotations

from api import reference_hit_context as hit_context


def test_select_hit_initial_heading_path_prefers_stronger_leading_heading() -> None:
    out = hit_context._select_hit_initial_heading_path(
        meta={"ref_best_heading_path": "Abstract"},
        hit_text="# 3. Method\nThe method text.",
        prompt="method details",
        leading_markdown_heading_from_hit_text=lambda text: "3. Method",
        refs_section_intent_heading_score=lambda prompt, heading: 2.0 if heading == "3. Method" else 0.0,
        normalize_title_identity=lambda text: str(text or "").strip().lower(),
    )

    assert out == "3. Method"


def test_resolve_hit_anchor_target_uses_prompt_figure_when_meta_anchor_missing() -> None:
    out = hit_context._resolve_hit_anchor_target(
        meta={},
        prompt="Explain Figure 7.",
        positive_int=lambda value: int(value or 0),
        extract_figure_number=lambda prompt: 7,
        extract_equation_number=lambda prompt: 0,
    )

    assert out == ("figure", 7)


def test_build_ref_hit_context_uses_preloaded_citation_meta_before_library() -> None:
    class Store:
        def get_citation_meta(self, pdf_path):
            raise AssertionError("preloaded citation meta should win")

    badge_calls: list[dict] = []

    def build_badges(**kwargs):
        badge_calls.append(kwargs)
        return ["figure:7"]

    out = hit_context._build_ref_hit_context(
        hit={
            "text": "# 3. Method\nBody.",
            "meta": {
                "source_path": "/kb/paper.md",
                "ref_pack_state": "READY",
                "ref_best_heading_path": "Abstract",
                "ref_rank": {"llm": 8.0},
                "anchor_match_score": "6.5",
            },
        },
        prompt="Explain Figure 7.",
        pdf_root="/pdfs",
        lib_store=Store(),
        preloaded_citation_meta={"/kb/paper.md": {"title": "Preloaded Paper"}},
        leading_markdown_heading_from_hit_text=lambda text: "3. Method",
        refs_section_intent_heading_score=lambda prompt, heading: 2.0 if heading == "3. Method" else 0.0,
        normalize_title_identity=lambda text: str(text or "").strip().lower(),
        resolve_ref_ui_heading_context=lambda **kwargs: {
            "heading_path": kwargs["heading_path"],
            "heading": "3. Method",
            "section_label": "3. Method",
            "subsection_label": "",
        },
        top_heading=lambda heading: str(heading).split(" / ", 1)[0],
        safe_page_range=lambda meta: (2, 3),
        effective_ui_score=lambda hit: (8.0, False),
        positive_int=lambda value: int(value or 0),
        extract_figure_number=lambda prompt: 7,
        extract_equation_number=lambda prompt: 0,
        non_negative_float=lambda value: float(value or 0.0),
        build_semantic_badges=build_badges,
        resolve_pdf_for_source=lambda pdf_root, source_path: "/pdfs/paper.pdf",
        display_source_name=lambda source_path, pdf_path, lib_store: "Paper.pdf",
    )

    assert out["source_path"] == "/kb/paper.md"
    assert out["ref_pack_state"] == "ready"
    assert out["heading_path"] == "3. Method"
    assert out["page_start"] == 2
    assert out["score"] == 8.0
    assert out["anchor_target_kind"] == "figure"
    assert out["anchor_target_number"] == 7
    assert out["semantic_badges"] == ["figure:7"]
    assert out["pdf_path"] == "/pdfs/paper.pdf"
    assert out["display_name"] == "Paper.pdf"
    assert out["citation_meta"] == {"title": "Preloaded Paper"}
    assert badge_calls[0]["anchor_match_score"] == 6.5


def test_apply_section_intent_rescue_context_overrides_heading_and_summary() -> None:
    out = hit_context._apply_section_intent_rescue_context(
        meta={
            "section_intent_rescue": True,
            "ref_best_heading_path": "4. Experiments / 4.2 Ablation",
            "ref_section": "4. Experiments",
        },
        hit_text="Ablation confirms the mask design improves reconstruction.",
        heading_path="1. Intro",
        heading="Intro",
        section_label="Intro",
        subsection_label="",
        summary_line="Old summary.",
        summary_source="fallback",
        top_heading=lambda heading: str(heading).split(" / ", 1)[0],
        summary_excerpt=lambda text, **kwargs: "Ablation confirms the mask design.",
    )

    assert out["heading_path"] == "4. Experiments / 4.2 Ablation"
    assert out["heading"] == "4.2 Ablation"
    assert out["section_label"] == "4. Experiments"
    assert out["subsection_label"] == "4.2 Ablation"
    assert out["summary_line"] == "Ablation confirms the mask design."
    assert out["summary_source"] == "section_intent_rescue"


def test_reference_ui_build_ref_hit_context_proxy_uses_hit_context_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return {"source_path": "/kb/paper.md"}

    monkeypatch.setattr(hit_context, "_build_ref_hit_context", fake_build)

    out = reference_ui._build_ref_hit_context(
        hit={"meta": {"source_path": "/kb/paper.md"}},
        prompt="method",
        pdf_root=None,
        lib_store=None,
        preloaded_citation_meta={},
    )

    assert out == {"source_path": "/kb/paper.md"}
    assert calls
    call = calls[0]
    assert call["leading_markdown_heading_from_hit_text"] is reference_ui._leading_markdown_heading_from_hit_text
    assert call["refs_section_intent_heading_score"] is reference_ui._refs_section_intent_heading_score
    assert call["resolve_ref_ui_heading_context"] is reference_ui._resolve_ref_ui_heading_context
    assert call["effective_ui_score"] is reference_ui._effective_ui_score
    assert call["build_semantic_badges"] is reference_ui._build_semantic_badges
    assert call["display_source_name"] is reference_ui._display_source_name


def test_reference_ui_apply_section_intent_rescue_context_proxy_uses_hit_context_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_rescue(**kwargs):
        calls.append(kwargs)
        return {"heading_path": "2. Method", "summary_source": "section_intent_rescue"}

    monkeypatch.setattr(hit_context, "_apply_section_intent_rescue_context", fake_rescue)

    out = reference_ui._apply_section_intent_rescue_context(
        meta={"section_intent_rescue": True},
        hit_text="text",
        heading_path="",
        heading="",
        section_label="",
        subsection_label="",
        summary_line="summary",
        summary_source="fallback",
    )

    assert out["heading_path"] == "2. Method"
    assert calls
    assert calls[0]["top_heading"] is reference_ui._top_heading
    assert calls[0]["summary_excerpt"] is reference_ui._summary_excerpt


def test_reference_ui_locked_table_card_keeps_metric_summary_and_exact_anchor() -> None:
    from api import reference_ui

    evidence = (
        "Table 6. Image Denoising Results on SIDD. SIDD PSNR: "
        "MPRNet [37] = 39.71; Restormer [39] = 40.02; "
        "Baseline ours = 40.30; NAFNet ours = 40.30"
    )
    hit = {
        "score": 14.6,
        "text": evidence,
        "meta": {
            "source_path": "paper.md",
            "heading_path": "Simple Baselines / 5 Experiments / 5.2 Applications",
            "ref_best_heading_path": "5 Experiments / 5.2 Applications",
            "ref_section": "5 Experiments",
            "ref_subsection": "5.2 Applications",
            "page_start": 13,
            "page_end": 13,
            "ref_show_snippets": [evidence],
            "ref_snippets": [evidence],
            "structured_kind": "table_metric",
            "structured_evidence_locked": True,
            "table_number": 6,
            "table_metric": "PSNR",
            "table_metric_label": "SIDD PSNR",
            "table_subject_kind": "method",
            "block_id": "table-6",
            "table_block_id": "table-6",
            "anchor_id": "tb_00006",
            "ref_rank": {"display_score": 8.8, "llm": 88.0},
        },
    }

    ui = reference_ui.build_hit_ui_meta(
        hit,
        prompt="SIDD 基准测试里 PSNR 最高的模型是谁？如果并列请全部列出。",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=True,
    )

    assert ui["summary_line"] == (
        "表 6 汇总了 SIDD PSNR 的方法对比；最高值为 40.30，"
        "由 Baseline (ours)、NAFNet (ours) 并列取得。"
    )
    assert "直接比较了问题所问的 SIDD PSNR" in ui["why_line"]
    assert ui["heading_path"].endswith("5 Experiments / 5.2 Applications")
    assert ui["page_start"] == 13
    assert ui["primary_evidence"]["block_id"] == "table-6"
    assert ui["primary_evidence"]["anchor_id"] == "tb_00006"
    assert ui["primary_evidence"]["anchor_kind"] == "table"
    assert "Baseline ours = 40.30" in ui["primary_evidence"]["snippet"]
    assert ui["reader_open"]["blockId"] == "table-6"
    assert ui["reader_open"]["anchorId"] == "tb_00006"
    assert ui["reader_open"]["anchorKind"] == "table"


def test_reference_ui_exact_support_keeps_localized_guide_and_relevance(tmp_path) -> None:
    from api import reference_ui

    source = tmp_path / "pidl.en.md"
    evidence = (
        "The underlying limitation originates from the employed single-source Poisson noise model, "
        "which deviates from real SPAD noise containing crosstalk and dark count rate. "
        "The multi-source physical noise model includes shot noise, dark count rate, and crosstalk noise."
    )
    source.write_text(f"# Introduction\n\n{evidence}\n", encoding="utf-8")
    guide = "论文先解释单源泊松模型为什么偏离真实 SPAD 噪声，再列出多源物理噪声模型的组成。"
    why = "这段原文直接回答泊松噪声为什么不够，以及模型纳入了哪些真实噪声。"
    hit = {
        "score": 1_000_000.0,
        "text": evidence,
        "meta": {
            "source_path": str(source),
            "source_name": "PIDL",
            "heading_path": "Introduction / Figure 1a",
            "top_heading": "Introduction / Figure 1a",
            "page_start": 2,
            "page_end": 2,
            "paper_guide_fast_exact": True,
            "structured_evidence_locked": True,
            "guide_line": guide,
            "why_line": why,
            "primary_evidence": {
                "source_path": str(source),
                "source_name": "PIDL",
                "heading_path": "Introduction / Figure 1a",
                "snippet": evidence,
                "highlight_snippet": evidence,
                "page_start": 2,
                "page_end": 2,
                "selection_reason": "spad_noise_model_exact_source",
                "strict_locate": True,
            },
        },
    }

    ui = reference_ui.build_hit_ui_meta(
        hit,
        prompt="为什么只用泊松噪声训练 SPAD 模型不够？模型纳入了哪些真实噪声？",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=True,
    )

    assert ui["summary_line"] == guide
    assert ui["why_line"] == why
    assert ui["heading_path"] == "Introduction / Figure 1a"
    assert ui["page_start"] == 2
    assert ui["primary_evidence"]["snippet"] == evidence
    assert ui["primary_evidence"]["strict_locate"] is True
    assert ui["reader_open"]["snippet"] == evidence
    assert ui["reader_open"]["pageStart"] == 2
