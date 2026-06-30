from __future__ import annotations

from api import reference_primary_evidence as primary_evidence


def test_primary_ref_evidence_base_summary_prefers_navigation_summary() -> None:
    calls: list[dict] = []

    def fallback(*args, **kwargs) -> str:
        calls.append({"args": args, **kwargs})
        return "fallback"

    out = primary_evidence._primary_ref_evidence_base_summary(
        meta={"title": "Paper"},
        prompt="method",
        heading="2. Method",
        citation_meta={"title": "Citation"},
        allow_llm_translate=False,
        build_ref_navigation=lambda meta, **kwargs: {
            "summary_line": "navigation summary",
            "what": "what summary",
        },
        fallback_ref_ui_summary_line=fallback,
    )

    assert out["summary_line"] == "navigation summary"
    assert out["summary_source"] == "navigation"
    assert out["used_nav_summary"] is True
    assert calls == []


def test_primary_ref_evidence_base_summary_uses_fallback_when_navigation_empty() -> None:
    calls: list[dict] = []

    def fallback(meta, **kwargs) -> str:
        calls.append({"meta": meta, **kwargs})
        return "fallback summary"

    out = primary_evidence._primary_ref_evidence_base_summary(
        meta={"title": "Paper"},
        prompt="method",
        heading="2. Method",
        citation_meta={"title": "Citation"},
        allow_llm_translate=True,
        build_ref_navigation=lambda meta, **kwargs: {"summary_line": "", "what": ""},
        fallback_ref_ui_summary_line=fallback,
    )

    assert out["summary_line"] == "fallback summary"
    assert out["summary_source"] == "fallback"
    assert out["used_nav_summary"] is False
    assert calls == [
        {
            "meta": {"title": "Paper"},
            "prompt": "method",
            "citation_meta": {"title": "Citation"},
            "allow_llm_translate": True,
        }
    ]


def test_select_primary_block_prompt_aligned_candidate_rescues_weak_fallback() -> None:
    calls: list[dict] = []

    def choose_block(**kwargs):
        calls.append(kwargs)
        return {"summary": "block summary", "source_kind": "source_block"}

    out = primary_evidence._select_primary_block_prompt_aligned_candidate(
        prompt="define dynamic supersampling",
        source_path="/kb/paper.md",
        title="Paper",
        display_name="Paper.pdf",
        summary_line="dynamic supersampling: abstract shell",
        summary_source="fallback",
        heading_path="1. Intro",
        meta_prompt_aligned_candidate={},
        anchor_target_kind="",
        anchor_target_number=0,
        allow_summary_block_rescue=True,
        allow_llm_translate=False,
        looks_focus_prefixed_ref_summary=lambda prompt, summary: True,
        summary_line_needs_polish=lambda **kwargs: False,
        sanitize_heading_path_ui=lambda raw, **kwargs: str(raw or "").strip(),
        rank_prompt_aligned_ref_summary_candidate=lambda candidate, **kwargs: (0.0,),
        choose_prompt_aligned_ref_summary_candidate_from_source_blocks=choose_block,
    )

    assert out == {"summary": "block summary", "source_kind": "source_block"}
    assert calls == [
        {
            "prompt": "define dynamic supersampling",
            "source_path": "/kb/paper.md",
            "title": "Paper",
            "anchor_target_kind": "",
            "anchor_target_number": 0,
            "allow_llm_translate": False,
        }
    ]


def test_select_primary_block_prompt_aligned_candidate_keeps_strong_meta_candidate() -> None:
    calls: list[dict] = []

    out = primary_evidence._select_primary_block_prompt_aligned_candidate(
        prompt="method",
        source_path="/kb/paper.md",
        title="Paper",
        display_name="Paper.pdf",
        summary_line="good current summary",
        summary_source="navigation",
        heading_path="2. Method",
        meta_prompt_aligned_candidate={"summary": "meta summary", "heading_path": "2. Method"},
        anchor_target_kind="",
        anchor_target_number=0,
        allow_summary_block_rescue=True,
        allow_llm_translate=True,
        looks_focus_prefixed_ref_summary=lambda prompt, summary: False,
        summary_line_needs_polish=lambda **kwargs: False,
        sanitize_heading_path_ui=lambda raw, **kwargs: str(raw or "").strip(),
        rank_prompt_aligned_ref_summary_candidate=lambda candidate, **kwargs: (3.0,),
        choose_prompt_aligned_ref_summary_candidate_from_source_blocks=lambda **kwargs: calls.append(kwargs) or {},
    )

    assert out == {}
    assert calls == []


def test_apply_primary_prompt_aligned_summary_candidate_replaces_and_rebinds_heading() -> None:
    def focus_score(**kwargs) -> float:
        return 8.0 if kwargs["text"] == "aligned summary" else 1.0

    out = primary_evidence._apply_primary_prompt_aligned_summary_candidate(
        prompt="compare Hadamard and Fourier",
        source_path="/kb/paper.md",
        title="Paper",
        display_name="Paper.pdf",
        summary_line="generic fallback",
        summary_source="fallback",
        heading_path="2.4 Efficiency",
        prompt_aligned_candidate={
            "summary": "aligned summary",
            "heading_path": "2.2 Basis patterns generation",
            "source_kind": "source_block",
        },
        anchor_target_kind="",
        anchor_target_number=0,
        allow_summary_block_rescue=False,
        sanitize_heading_path_ui=lambda raw, **kwargs: str(raw or "").strip(),
        refs_heading_anchor_number=lambda anchor_kind, heading_path: 0,
        refs_heading_paths_related=lambda left, right: False,
        infer_heading_path_for_summary_from_source_blocks=lambda **kwargs: "",
        summary_line_needs_polish=lambda **kwargs: False,
        ref_summary_focus_score=focus_score,
        matched_focus_terms_for_ref_card=lambda prompt, **kwargs: ["compare"],
        ref_summary_surfaces_match=lambda left, right: False,
    )

    assert out["summary_line"] == "aligned summary"
    assert out["summary_source"] == "prompt_aligned_block"
    assert out["used_prompt_aligned_summary"] is True
    assert out["selected_heading_path"] == "2.2 Basis patterns generation"


def test_apply_reader_anchor_summary_override_uses_stronger_reader_anchor() -> None:
    score_calls: list[str] = []

    def focus_score(**kwargs) -> float:
        score_calls.append(str(kwargs.get("text") or ""))
        return 8.0 if "Figure 2 shows" in str(kwargs.get("text") or "") else 1.0

    summary_line, summary_source = primary_evidence._apply_reader_anchor_summary_override(
        reader_open={
            "snippet": "Figure 2 shows the optical setup and detector path.",
            "headingPath": "3. Results / Figure 2",
        },
        prompt="explain Figure 2",
        source_path="/kb/paper.md",
        display_name="Paper.pdf",
        summary_line="generic figure summary",
        summary_source="fallback",
        anchor_target_kind="figure",
        anchor_target_number=2,
        refs_heading_anchor_number=lambda anchor_kind, heading_path: 2,
        ref_summary_focus_score=focus_score,
        build_evidence_backed_ref_summary_from_seed=lambda **kwargs: f"backed: {kwargs['summary_line']}",
        prefer_zh_ref_card_locale=lambda *texts: False,
        summary_excerpt=lambda text, **kwargs: f"excerpt: {text}",
        normalize_ref_copy_text=lambda text: str(text).replace("backed:", "exact:"),
    )

    assert summary_line == "exact: Figure 2 shows the optical setup and detector path."
    assert summary_source == "exact_anchor"
    assert "generic figure summary" in score_calls


def test_apply_reader_anchor_summary_override_keeps_current_when_reader_score_is_not_better() -> None:
    summary_line, summary_source = primary_evidence._apply_reader_anchor_summary_override(
        reader_open={
            "snippet": "Figure 2 shows a weakly related caption.",
            "headingPath": "3. Results / Figure 2",
        },
        prompt="explain Figure 2",
        source_path="/kb/paper.md",
        display_name="Paper.pdf",
        summary_line="current exact Figure 2 summary",
        summary_source="navigation",
        anchor_target_kind="figure",
        anchor_target_number=2,
        refs_heading_anchor_number=lambda anchor_kind, heading_path: 2,
        ref_summary_focus_score=lambda **kwargs: (
            4.1 if str(kwargs.get("text") or "").startswith("Figure 2 shows") else 4.0
        ),
        build_evidence_backed_ref_summary_from_seed=lambda **kwargs: "should not be used",
        prefer_zh_ref_card_locale=lambda *texts: False,
        summary_excerpt=lambda text, **kwargs: "should not be used",
        normalize_ref_copy_text=lambda text: str(text),
    )

    assert summary_line == "current exact Figure 2 summary"
    assert summary_source == "navigation"


def test_reference_ui_primary_summary_selection_proxy_uses_primary_evidence_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_resolve(**kwargs):
        calls.append(kwargs)
        return {
            "candidate_title": "Paper",
            "nav": {},
            "summary_line": "summary",
            "summary_source": "fallback",
            "used_nav_summary": False,
            "used_prompt_aligned_summary": False,
            "selected_heading_path": "2. Method",
            "prompt_aligned_candidate": {},
        }

    monkeypatch.setattr(primary_evidence, "_resolve_primary_ref_evidence_summary_selection", fake_resolve)

    out = reference_ui._resolve_primary_ref_evidence_summary_selection(
        meta={"title": "Paper"},
        prompt="method",
        source_path="/kb/paper.md",
        display_name="Paper.pdf",
        citation_meta={},
        heading_path="2. Method",
        heading="Method",
        anchor_target_kind="",
        anchor_target_number=0,
        allow_summary_block_rescue=True,
        allow_llm_translate=False,
    )

    assert out["summary_line"] == "summary"
    assert calls
    call = calls[0]
    assert call["build_ref_navigation"] is reference_ui._build_ref_navigation
    assert call["fallback_ref_ui_summary_line"] is reference_ui._fallback_ref_ui_summary_line
    assert call["choose_prompt_aligned_ref_summary_candidate"] is reference_ui._choose_prompt_aligned_ref_summary_candidate
    assert call["choose_prompt_aligned_ref_summary_candidate_from_source_blocks"] is (
        reference_ui._choose_prompt_aligned_ref_summary_candidate_from_source_blocks
    )
    assert call["pick_best_prompt_aligned_ref_summary_candidate"] is reference_ui._pick_best_prompt_aligned_ref_summary_candidate
    assert call["ref_summary_focus_score"] is reference_ui._ref_summary_focus_score
