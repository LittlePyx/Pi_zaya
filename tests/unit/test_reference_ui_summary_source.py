from __future__ import annotations

import api.reference_ui as reference_ui


def test_chain_a_summary_source_navigation(monkeypatch):
    """build_hit_ui_meta — navigation path sets summary_source='navigation'."""
    monkeypatch.setattr(
        reference_ui, "_build_ref_navigation",
        lambda meta, prompt, heading_fallback="": {
            "summary_line": "The proposed method achieves high performance.",
            "what": "The proposed method achieves high performance.",
            "why": "",
            "start_from": "",
            "gain": "",
            "sem_score": 0.0,
            "section": "Method",
            "subsection": "",
            "find": [],
            "pack_pending": False,
        },
    )
    monkeypatch.setattr(reference_ui, "_choose_prompt_aligned_ref_summary_candidate", lambda *a, **kw: {})
    monkeypatch.setattr(reference_ui, "extract_figure_number", lambda prompt: 0)

    ui_meta = reference_ui.build_hit_ui_meta(
        {"meta": {"source_path": "db/paper/paper.en.md", "ref_rank": {"bm25": 5.0}}},
        prompt="What method?",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )
    assert str(ui_meta.get("summary_source") or "").strip() == "navigation"


def test_chain_a_summary_source_fallback(monkeypatch):
    """build_hit_ui_meta — no navigation summary => summary_source='fallback'."""
    monkeypatch.setattr(
        reference_ui, "_build_ref_navigation",
        lambda meta, prompt, heading_fallback="": {
            "summary_line": "",
            "what": "",
            "why": "",
            "start_from": "",
            "gain": "",
            "sem_score": 0.0,
            "section": "",
            "subsection": "",
            "find": [],
            "pack_pending": False,
        },
    )
    monkeypatch.setattr(reference_ui, "_fallback_ref_ui_summary_line", lambda *a, **kw: "Fallback summary text.")
    monkeypatch.setattr(reference_ui, "_choose_prompt_aligned_ref_summary_candidate", lambda *a, **kw: {})
    monkeypatch.setattr(reference_ui, "extract_figure_number", lambda prompt: 0)

    ui_meta = reference_ui.build_hit_ui_meta(
        {"meta": {"source_path": "db/paper/paper.en.md", "ref_rank": {"bm25": 5.0}}},
        prompt="What method?",
        pdf_root=None,
        lib_store=None,
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )
    assert str(ui_meta.get("summary_source") or "").strip() == "fallback"


def test_chain_b_summary_source_weak_primary_delegates_to_chain_a(monkeypatch):
    """_build_doc_list_hit_ui_meta — weak primary delegates to Chain A and captures its summary_source."""
    monkeypatch.setattr(
        reference_ui, "build_hit_ui_meta",
        lambda hit, **kw: {
            "summary_source": "fallback",
            "summary_line": "Some summary.",
            "heading_path": "Paper / Intro",
        },
    )
    raw_item = {
        "source_path": "db/paper/paper.en.md",
        "source_name": "Test Paper",
        "primary_evidence": {
            "selection_reason": "answer_hit_top",
            "snippet": None,
            "highlight_snippet": None,
            "block_id": None,
            "heading_path": "Paper / Intro",
        },
        "heading_path": "Paper / Intro",
    }
    ui_meta = reference_ui._build_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=1,
        prompt="What is the paper about?",
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )
    assert str(ui_meta.get("summary_source") or "").strip() == "fallback"


def test_chain_b_summary_source_seed(monkeypatch):
    """_build_doc_list_hit_ui_meta — strong primary seed sets summary_source='doc_list_seed'."""
    raw_item = {
        "source_path": "db/paper/paper.en.md",
        "source_name": "Test Paper",
        "summary_line": "A novel approach for solving the problem.",
        "primary_evidence": {
            "selection_reason": "exact_anchor",
            "snippet": "A novel approach for solving the problem.",
            "block_id": "block_001",
            "heading_path": "Paper / Method",
        },
        "heading_path": "Paper / Method",
    }
    monkeypatch.setattr(
        reference_ui, "_build_doc_list_hit_ui_seed",
        lambda **kw: (
            {"meta": {"source_path": "db/paper/paper.en.md"}},
            {
                "summary_line": "A novel approach for solving the problem.",
                "display_name": "Test Paper",
            },
            raw_item["primary_evidence"],
        ),
    )
    monkeypatch.setattr(reference_ui, "_summary_line_needs_polish", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_like_title_echo", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_why_like_ref_summary", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_fragmentary_ref_summary", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_surface_like_ref_summary", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_formula_heavy_ref_text", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_build_prompt_aligned_ref_summary_fallback", lambda *a, **kw: "")

    ui_meta = reference_ui._build_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=1,
        prompt="What method?",
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )
    summary_source = str(ui_meta.get("summary_source") or "").strip()
    assert summary_source == "doc_list_seed"


def test_chain_b_summary_source_ultimate_seed(monkeypatch):
    """_build_doc_list_hit_ui_meta — all fallback paths empty => ultimate_seed."""
    raw_item = {
        "source_path": "db/paper/paper.en.md",
        "source_name": "Test Paper",
        "summary_line": "Ultimate seed summary.",
        "primary_evidence": {
            "selection_reason": "exact_anchor",
            "snippet": None,
            "highlight_snippet": None,
            "block_id": None,
            "heading_path": "Paper / Related",
        },
        "heading_path": "Paper / Related",
    }

    def _ui_seed(**kw):
        return (
            {"meta": {"source_path": "db/paper/paper.en.md"}},
            {"summary_line": "", "display_name": "Test Paper"},
            raw_item["primary_evidence"],
        )

    monkeypatch.setattr(reference_ui, "_build_doc_list_hit_ui_seed", _ui_seed)

    # Prevent _apply_doc_list_effective_primary_evidence from filling summary_line
    def _effective_evidence(**kw):
        return (dict(kw.get("ui_meta") or {}), {})
    monkeypatch.setattr(reference_ui, "_apply_doc_list_effective_primary_evidence", _effective_evidence)

    monkeypatch.setattr(reference_ui, "_summary_line_needs_polish", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_like_title_echo", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_why_like_ref_summary", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_fragmentary_ref_summary", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_surface_like_ref_summary", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_looks_formula_heavy_ref_text", lambda *a, **kw: False)
    monkeypatch.setattr(reference_ui, "_build_prompt_aligned_ref_summary_fallback", lambda *a, **kw: "")

    ui_meta = reference_ui._build_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=1,
        prompt="What prior work?",
        allow_expensive_llm=False,
        allow_exact_locate=False,
    )
    summary_source = str(ui_meta.get("summary_source") or "").strip()
    assert summary_source == "doc_list_ultimate_seed", f"got {summary_source!r}"
