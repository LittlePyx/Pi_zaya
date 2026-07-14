from __future__ import annotations

from api import reference_doc_list as doc_list
from api import reference_ui


def test_collect_doc_list_ref_text_candidates_cleans_and_dedupes() -> None:
    clean_calls: list[dict] = []

    def clean(text: str, **kwargs) -> str:
        clean_calls.append({"text": text, **kwargs})
        return str(text or "").strip()

    out = doc_list._collect_doc_list_ref_text_candidates(
        raw_item={
            "source_path": "/kb/paper.md",
            "source_name": "Paper",
            "heading_path": "2. Method",
            "summary_line": "summary",
        },
        primary_evidence={
            "highlight_snippet": "primary",
            "snippet": "primary",
            "alternatives": [
                {"snippet": "alt"},
                {"highlight_snippet": "alt"},
            ],
        },
        clean_refs_evidence_snippet=clean,
    )

    assert out == ["primary", "summary", "alt"]
    assert clean_calls[0] == {
        "text": "primary",
        "prompt": "",
        "source_path": "/kb/paper.md",
        "display_name": "Paper",
        "heading_path": "2. Method",
        "max_len": 460,
    }


def test_primary_ref_evidence_summary_seed_normalizes_and_cleans() -> None:
    out = doc_list._primary_ref_evidence_summary_seed(
        {
            "source_path": "/kb/paper.md",
            "source_name": "Paper",
            "heading_path": "2. Method",
            "snippet": "seed",
        },
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}, snippet="normalized seed"),
        clean_refs_evidence_snippet=lambda text, **kwargs: f"clean:{text}",
    )

    assert out == "clean:normalized seed"


def test_build_doc_list_ref_locs_keeps_primary_and_unique_alternatives() -> None:
    out = doc_list._build_doc_list_ref_locs(
        heading_path="2. Method / Details",
        primary_evidence={
            "highlight_snippet": "primary",
            "alternatives": [
                {"heading_path": "3. Results", "snippet": "alt"},
                {"heading_path": "3. Results", "snippet": "alt"},
            ],
        },
        clean_refs_evidence_snippet=lambda text, **kwargs: str(text or "").strip(),
        top_heading=lambda heading: str(heading).split(" / ", 1)[0],
    )

    assert [item["source"] for item in out] == ["doc_list_primary", "doc_list_alternative"]
    assert out[0]["heading"] == "2. Method"
    assert out[1]["heading_path"] == "3. Results"
    assert out[1]["score"] == 95.5


def test_build_doc_list_ref_hit_shapes_retrieval_meta() -> None:
    out = doc_list._build_doc_list_ref_hit(
        raw_item={
            "source_path": "/kb/paper.md",
            "source_name": "",
            "heading_path": "2. Method / Details",
            "summary_line": "summary",
            "primary_evidence": {
                "snippet": "primary snippet",
                "anchor_kind": "Figure",
                "anchor_number": "3",
            },
        },
        idx=2,
        source_filename=lambda source_path: "paper.pdf",
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        split_section_subsection=lambda heading: ("2. Method", "Details"),
        top_heading=lambda heading: str(heading).split(" / ", 1)[0],
        clean_refs_evidence_snippet=lambda text, **kwargs: str(text or "").strip(),
    )

    assert out["text"] == "primary snippet"
    meta = out["meta"]
    assert meta["source_name"] == "paper.pdf"
    assert meta["top_heading"] == "2. Method"
    assert meta["ref_section"] == "2. Method"
    assert meta["ref_subsection"] == "Details"
    assert meta["anchor_target_kind"] == "figure"
    assert meta["anchor_target_number"] == 3
    assert meta["anchor_match_score"] == 10.0
    assert meta["ref_rank"]["llm"] == 90.0


def test_upgrade_primary_ref_evidence_from_alternatives_promotes_usable_alt() -> None:
    out = doc_list._upgrade_primary_ref_evidence_from_alternatives(
        {
            "source_path": "/kb/paper.md",
            "source_name": "Paper",
            "snippet": "weak generated shell",
            "selection_reason": "section_intent_rescue",
            "alternatives": [
                {"heading_path": "2. Method", "snippet": "DMD evidence"},
            ],
        },
        prompt="single pixel",
        display_name="Paper",
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        primary_ref_evidence_summary_is_usable=lambda evidence, **kwargs: "DMD" in str((evidence or {}).get("snippet") or ""),
        primary_ref_evidence_precision_score=lambda **kwargs: (
            9 if "DMD" in str((kwargs.get("primary_evidence") or {}).get("snippet") or "") else 0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    )

    assert out["selection_reason"] == "alternative_rescue"
    assert out["heading_path"] == "2. Method"
    assert out["snippet"] == "DMD evidence"


def test_select_doc_list_effective_primary_evidence_prefers_stronger_synthesized() -> None:
    synthesized = {"selection_reason": "prompt_aligned_block", "block_id": "blk", "snippet": "synth"}

    out, source = doc_list._select_doc_list_effective_primary_evidence(
        prompt="method",
        display_name="Paper",
        authoritative_primary_evidence={"selection_reason": "answer_hit_top", "snippet": "auth"},
        synthesized_primary_evidence=synthesized,
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        upgrade_primary_ref_evidence_from_alternatives=lambda primary, **kwargs: dict(primary or {}),
        primary_ref_evidence_points_to_same_surface=lambda left, right: False,
        doc_list_authoritative_primary_is_upgradeable=lambda primary: True,
        primary_ref_evidence_precision_score=lambda **kwargs: (
            8 if str((kwargs.get("primary_evidence") or {}).get("selection_reason") or "") == "prompt_aligned_block" else 0,
            0,
            1 if str((kwargs.get("primary_evidence") or {}).get("block_id") or "") else 0,
            0,
            0,
            0,
            0,
        ),
    )

    assert out == synthesized
    assert source == "synthesized"


def test_apply_doc_list_effective_primary_evidence_prefers_authoritative_conflict_summary() -> None:
    authoritative = {
        "heading_path": "2. Method",
        "snippet": "authoritative snippet",
        "selection_reason": "shared_refs_pack",
    }

    out, effective = doc_list._apply_doc_list_effective_primary_evidence(
        prompt="method",
        display_name="Paper",
        fallback_heading_path="1. Intro",
        ui_meta={
            "heading_path": "old heading",
            "summary_line": "old summary",
            "summary_generation": "deterministic_grounded",
            "primary_evidence": {"heading_path": "1. Intro", "snippet": "synth"},
        },
        authoritative_primary_evidence=authoritative,
        authoritative_summary_line="authoritative summary",
        authoritative_summary_generation="llm_grounded",
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        select_doc_list_effective_primary_evidence=lambda **kwargs: (dict(kwargs["authoritative_primary_evidence"]), "authoritative"),
        primary_ref_evidence_summary_seed=lambda primary: str((primary or {}).get("snippet") or ""),
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        summary_line_needs_polish=lambda **kwargs: False,
        primary_ref_evidence_points_to_same_surface=lambda left, right: False,
        build_ref_summary_basis_meta=lambda **kwargs: {
            "summary_generation": kwargs["summary_generation"],
            "summary_basis": "basis",
        },
    )

    assert effective == authoritative
    assert out["heading_path"] == "2. Method"
    assert out["summary_line"] == "authoritative summary"
    assert out["summary_generation"] == "llm_grounded"
    assert out["summary_basis"] == "basis"
    assert out["primary_evidence"] == authoritative
    assert out["primary_evidence_heading_path"] == "2. Method"
    assert out["primary_evidence_source"] == "shared_refs_pack"
    assert out["authoritative_primary_evidence"] == authoritative
    assert out["primary_evidence_authority"] == "doc_list_authoritative"


def test_build_doc_list_hit_ui_seed_builds_seed_meta() -> None:
    raw_item = {
        "source_path": "/kb/paper.md",
        "source_name": "Paper",
        "heading_path": "2. Method",
        "summary_line": "authoritative summary",
        "summary_generation": "llm_grounded",
        "why_line": "authoritative why",
        "why_generation": "llm_pack",
        "primary_evidence": {"snippet": "primary seed"},
    }
    hit = {
        "text": "primary seed",
        "meta": {
            "source_path": "/kb/paper.md",
            "source_name": "Paper",
            "ref_best_heading_path": "2. Method",
            "top_heading": "Method",
            "ref_section": "2. Method",
            "ref_subsection": "Details",
            "anchor_target_kind": "Figure",
            "anchor_target_number": "3",
            "anchor_match_score": "9.5",
            "explicit_doc_match_score": "12",
        },
    }

    out_hit, ui_meta, primary = doc_list._build_doc_list_hit_ui_seed(
        raw_item=raw_item,
        idx=1,
        prompt="method",
        build_doc_list_ref_hit=lambda **kwargs: hit,
        source_filename=lambda source_path: "fallback.pdf",
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        normalize_ref_copy_text=lambda text: str(text or "").strip(),
        resolve_ref_ui_heading_context=lambda **kwargs: {
            "heading_path": kwargs["heading_path"],
            "heading": "Method",
            "section_label": kwargs["section_label"],
            "subsection_label": kwargs["subsection_label"],
        },
        top_heading=lambda heading: "Method",
        primary_ref_evidence_summary_seed=lambda primary: str((primary or {}).get("snippet") or ""),
        build_ref_summary_basis_meta=lambda **kwargs: {
            "summary_generation": kwargs["summary_generation"],
            "summary_basis": "summary basis",
        },
        build_prompt_aligned_ref_why_line=lambda **kwargs: "prompt why",
        doc_list_ref_why_line=lambda **kwargs: "fallback why",
        prefer_zh_ref_card_locale=lambda *args: True,
        build_ref_why_basis_meta=lambda **kwargs: {
            "why_generation": kwargs["why_generation"],
            "why_basis": "why basis",
        },
        summary_label="Guide",
        summary_title="Evidence",
    )

    assert out_hit is hit
    assert primary == {"snippet": "primary seed"}
    assert ui_meta["display_name"] == "Paper"
    assert ui_meta["heading_path"] == "2. Method"
    assert ui_meta["heading"] == "Method"
    assert ui_meta["summary_label"] == "Guide"
    assert ui_meta["summary_title"] == "Evidence"
    assert ui_meta["summary_line"] == "authoritative summary"
    assert ui_meta["summary_generation"] == "llm_grounded"
    assert ui_meta["summary_basis"] == "summary basis"
    assert ui_meta["why_line"] == "authoritative why"
    assert ui_meta["why_generation"] == "llm_pack"
    assert ui_meta["why_basis"] == "why basis"
    assert ui_meta["anchor_target_kind"] == "figure"
    assert ui_meta["anchor_target_number"] == 3
    assert ui_meta["anchor_match_score"] == 9.5
    assert ui_meta["explicit_doc_match_score"] == 12.0


def test_apply_doc_list_summary_fallbacks_uses_candidate_fallback() -> None:
    out, source = doc_list._apply_doc_list_summary_fallbacks(
        raw_item={"source_path": "/kb/paper.md"},
        prompt="method",
        source_name="Paper",
        heading_path="2. Method",
        ui_meta={
            "display_name": "Paper",
            "heading_path": "2. Method",
            "summary_kind": "guide",
            "summary_line": "Paper",
            "summary_generation": "deterministic_grounded",
        },
        primary_evidence={"snippet": "primary"},
        effective_primary_evidence={"snippet": "effective"},
        summary_source="doc_list_seed",
        summary_line_needs_polish=lambda **kwargs: kwargs["summary_line"] == "Paper",
        looks_like_title_echo=lambda summary, title: summary == title,
        looks_why_like_ref_summary=lambda summary: False,
        pick_ref_card_summary_fallback=lambda **kwargs: "fallback summary",
        collect_doc_list_ref_text_candidates=lambda **kwargs: ["effective"],
        build_ref_summary_basis_meta=lambda **kwargs: {
            "summary_generation": kwargs["summary_generation"],
            "summary_basis": "basis",
        },
        looks_fragmentary_ref_summary=lambda summary: False,
        looks_surface_like_ref_summary=lambda summary: False,
        looks_formula_heavy_ref_text=lambda summary: False,
        build_prompt_aligned_ref_summary_fallback=lambda **kwargs: "",
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        primary_ref_evidence_summary_seed=lambda primary: str((primary or {}).get("snippet") or ""),
    )

    assert out["summary_line"] == "fallback summary"
    assert out["summary_generation"] == "deterministic_grounded"
    assert out["summary_basis"] == "basis"
    assert source == "doc_list_fallback"


def test_apply_doc_list_summary_fallbacks_keeps_authoritative_evidence_over_generic_why_copy() -> None:
    authoritative = "The paper compares Hadamard and Fourier single-pixel imaging in efficiency and noise robustness."
    generic = "This section contains a source clue that can be used to verify the definition or method."
    out, source = doc_list._apply_doc_list_summary_fallbacks(
        raw_item={"source_path": "/kb/paper.md", "summary_line": authoritative},
        prompt="我应该先读哪几篇？",
        source_name="Paper",
        heading_path="2. Comparison",
        ui_meta={
            "display_name": "Paper",
            "heading_path": "2. Comparison",
            "summary_kind": "guide",
            "summary_line": generic,
            "summary_generation": "deterministic_grounded",
        },
        primary_evidence={"snippet": authoritative},
        effective_primary_evidence={"snippet": authoritative},
        summary_source="doc_list_prompt_aligned",
        summary_line_needs_polish=lambda **kwargs: False,
        looks_like_title_echo=lambda summary, title: False,
        looks_why_like_ref_summary=lambda summary: summary == generic,
        pick_ref_card_summary_fallback=lambda **kwargs: "",
        collect_doc_list_ref_text_candidates=lambda **kwargs: [authoritative],
        build_ref_summary_basis_meta=lambda **kwargs: {
            "summary_generation": kwargs["summary_generation"],
            "summary_basis": "grounded source evidence",
        },
        looks_fragmentary_ref_summary=lambda summary: False,
        looks_surface_like_ref_summary=lambda summary: False,
        looks_formula_heavy_ref_text=lambda summary: False,
        build_prompt_aligned_ref_summary_fallback=lambda **kwargs: "",
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        primary_ref_evidence_summary_seed=lambda primary: str((primary or {}).get("snippet") or ""),
    )

    assert out["summary_line"] == authoritative
    assert out["summary_generation"] == "section_grounded"
    assert out["summary_basis"] == "grounded source evidence"
    assert source == "doc_list_authoritative_fast"


def test_apply_doc_list_summary_fallbacks_uses_raw_fallback_when_empty() -> None:
    raw_text = "This raw hit snippet is long enough to serve as a final fallback summary."
    out, source = doc_list._apply_doc_list_summary_fallbacks(
        raw_item={"hits": [{"text": raw_text}]},
        prompt="method",
        source_name="Paper",
        heading_path="2. Method",
        ui_meta={"display_name": "Paper"},
        primary_evidence={},
        effective_primary_evidence={},
        summary_source="",
        summary_line_needs_polish=lambda **kwargs: False,
        looks_like_title_echo=lambda summary, title: False,
        looks_why_like_ref_summary=lambda summary: False,
        pick_ref_card_summary_fallback=lambda **kwargs: "",
        collect_doc_list_ref_text_candidates=lambda **kwargs: [],
        build_ref_summary_basis_meta=lambda **kwargs: {},
        looks_fragmentary_ref_summary=lambda summary: False,
        looks_surface_like_ref_summary=lambda summary: False,
        looks_formula_heavy_ref_text=lambda summary: False,
        build_prompt_aligned_ref_summary_fallback=lambda **kwargs: "",
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        primary_ref_evidence_summary_seed=lambda primary: "",
    )

    assert out["summary_line"] == raw_text
    assert out["summary_generation"] == "raw_fallback"
    assert source == "doc_list_raw_fallback"


def test_apply_doc_list_why_fallback_replaces_polished_line() -> None:
    out = doc_list._apply_doc_list_why_fallback(
        prompt="method",
        source_name="Paper",
        heading_path="2. Method",
        ui_meta={
            "display_name": "Paper",
            "heading_path": "2. Method",
            "summary_line": "summary",
            "why_line": "weak why",
        },
        why_line_needs_polish=lambda **kwargs: True,
        build_prompt_aligned_ref_why_line=lambda **kwargs: "better why",
        doc_list_ref_why_line=lambda **kwargs: "fallback why",
        prefer_zh_ref_card_locale=lambda *args: True,
        build_ref_why_basis_meta=lambda **kwargs: {
            "why_generation": kwargs["why_generation"],
            "why_basis": "basis",
        },
    )

    assert out["why_line"] == "better why"
    assert out["why_generation"] == "deterministic_grounded"
    assert out["why_basis"] == "basis"


def test_finalize_doc_list_hit_ui_meta_sets_reader_open_score_and_sources() -> None:
    out = doc_list._finalize_doc_list_hit_ui_meta(
        raw_item={"topic_match_kind": "Explicit_SCI_Mention"},
        idx=3,
        prompt="method",
        source_path="/kb/paper.md",
        source_name="Paper",
        heading_path="2. Method",
        ui_meta={
            "display_name": "Paper",
            "heading_path": "2. Method",
            "summary_kind": "guide",
            "summary_line": "summary",
            "summary_generation": "section_grounded",
            "why_line": "why",
            "why_generation": "deterministic_grounded",
            "reader_open": {"old": True},
        },
        primary_evidence={"heading_path": "1. Intro", "snippet": "primary"},
        effective_primary_evidence={"heading_path": "2. Method", "snippet": "effective"},
        summary_source="doc_list_seed",
        allow_expensive_llm=True,
        align_ref_card_copy_to_user_locale=lambda **kwargs: ("aligned summary", "aligned why"),
        build_ref_summary_surface_meta=lambda **kwargs: {
            "summary_kind": "guide",
            "summary_label": "Guide",
            "summary_title": "Evidence",
        },
        build_ref_summary_basis_meta=lambda **kwargs: {
            "summary_generation": kwargs["summary_generation"],
            "summary_basis": "summary basis",
        },
        build_ref_why_basis_meta=lambda **kwargs: {
            "why_generation": kwargs["why_generation"],
            "why_basis": "why basis",
        },
        score_tier=lambda score: "strong",
        build_doc_list_reader_open_payload=lambda **kwargs: {
            "sourcePath": kwargs["source_path"],
            "headingPath": kwargs["heading_path"],
            "summaryLine": kwargs["summary_line"],
            "primaryEvidence": kwargs["primary_evidence"],
        },
    )

    assert out["summary_line"] == "aligned summary"
    assert out["why_line"] == "aligned why"
    assert out["summary_label"] == "Guide"
    assert out["summary_basis"] == "summary basis"
    assert out["why_basis"] == "why basis"
    assert out["score"] == 9.19
    assert out["score_pending"] is False
    assert out["score_tier"] == "strong"
    assert out["source_path"] == "/kb/paper.md"
    assert out["reader_open"]["sourcePath"] == "/kb/paper.md"
    assert out["reader_open"]["primaryEvidence"]["snippet"] == "effective"
    assert out["primary_evidence"]["snippet"] == "effective"
    assert out["primary_evidence_heading_path"] == "2. Method"
    assert out["topic_match_kind"] == "explicit_sci_mention"
    assert out["summary_source"] == "doc_list_seed"


def test_build_doc_list_hit_ui_meta_delegates_weak_primary_to_chain_a() -> None:
    calls: list[dict] = []

    def build_hit(**kwargs):
        calls.append({"fn": "build_hit", **kwargs})
        return {"text": "chain A hit"}

    def build_hit_ui_meta(hit, **kwargs):
        calls.append({"fn": "chain_a", "hit": hit, **kwargs})
        return {
            "summary_source": "fallback",
            "summary_line": "chain A summary",
            "heading_path": "Paper / Intro",
        }

    def build_seed(**kwargs):
        raise AssertionError(f"seed branch should not run: {kwargs}")

    def apply_effective(**kwargs):
        calls.append({"fn": "effective", **kwargs})
        return kwargs["ui_meta"], {"heading_path": "Paper / Intro", "snippet": "effective"}

    def apply_summary(**kwargs):
        calls.append({"fn": "summary", **kwargs})
        return kwargs["ui_meta"], kwargs["summary_source"]

    def apply_why(**kwargs):
        calls.append({"fn": "why", **kwargs})
        return kwargs["ui_meta"]

    def finalize(**kwargs):
        calls.append({"fn": "finalize", **kwargs})
        return {
            "summary_source": kwargs["summary_source"],
            "source_name": kwargs["source_name"],
            "heading_path": kwargs["heading_path"],
            "primary_evidence": kwargs["primary_evidence"],
            "effective_primary_evidence": kwargs["effective_primary_evidence"],
            "allow_expensive_llm": kwargs["allow_expensive_llm"],
        }

    raw_item = {
        "source_path": "/kb/paper.md",
        "source_name": "Paper",
        "heading_path": "Paper / Intro",
        "primary_evidence": {
            "selection_reason": "answer_hit_top",
            "heading_path": "Paper / Intro",
        },
    }
    out = doc_list._build_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=1,
        prompt="method",
        allow_expensive_llm=True,
        allow_exact_locate=False,
        source_filename=lambda source_path: "fallback.pdf",
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        build_doc_list_ref_hit=build_hit,
        build_hit_ui_meta=build_hit_ui_meta,
        build_doc_list_hit_ui_seed=build_seed,
        apply_doc_list_effective_primary_evidence=apply_effective,
        apply_doc_list_summary_fallbacks=apply_summary,
        apply_doc_list_why_fallback=apply_why,
        finalize_doc_list_hit_ui_meta=finalize,
    )

    assert out["summary_source"] == "fallback"
    assert out["source_name"] == "Paper"
    assert out["heading_path"] == "Paper / Intro"
    assert out["effective_primary_evidence"]["snippet"] == "effective"
    assert out["allow_expensive_llm"] is True
    assert [call["fn"] for call in calls] == ["build_hit", "chain_a", "effective", "summary", "why", "finalize"]
    assert calls[1]["allow_exact_locate"] is False
    assert calls[2]["authoritative_primary_evidence"]["selection_reason"] == "answer_hit_top"
    assert calls[3]["summary_source"] == "fallback"


def test_build_doc_list_hit_ui_meta_uses_seed_for_strong_primary() -> None:
    calls: list[dict] = []

    def build_hit(**kwargs):
        raise AssertionError(f"Chain A hit builder should not run: {kwargs}")

    def build_hit_ui_meta(hit, **kwargs):
        raise AssertionError(f"Chain A UI builder should not run: {hit}, {kwargs}")

    def build_seed(**kwargs):
        calls.append({"fn": "seed", **kwargs})
        return (
            {"text": "seed hit"},
            {"summary_line": "seed summary", "display_name": "Seed Paper"},
            {
                "selection_reason": "exact_anchor",
                "block_id": "blk",
                "snippet": "strong primary",
                "heading_path": "Paper / Method",
            },
        )

    def apply_effective(**kwargs):
        calls.append({"fn": "effective", **kwargs})
        ui_meta = dict(kwargs["ui_meta"])
        ui_meta["summary_line"] = "effective summary"
        return ui_meta, {"snippet": "effective", "heading_path": "Paper / Method"}

    def apply_summary(**kwargs):
        calls.append({"fn": "summary", **kwargs})
        return kwargs["ui_meta"], kwargs["summary_source"]

    def apply_why(**kwargs):
        calls.append({"fn": "why", **kwargs})
        return kwargs["ui_meta"]

    def finalize(**kwargs):
        calls.append({"fn": "finalize", **kwargs})
        return {
            "summary_line": kwargs["ui_meta"]["summary_line"],
            "summary_source": kwargs["summary_source"],
            "source_name": kwargs["source_name"],
            "primary_evidence": kwargs["primary_evidence"],
            "effective_primary_evidence": kwargs["effective_primary_evidence"],
        }

    out = doc_list._build_doc_list_hit_ui_meta(
        raw_item={
            "source_path": "/kb/paper.md",
            "summary_line": "authoritative summary",
            "primary_evidence": {
                "selection_reason": "exact_anchor",
                "block_id": "blk",
                "snippet": "strong primary",
                "heading_path": "Paper / Method",
            },
        },
        idx=2,
        prompt="method",
        allow_expensive_llm=False,
        allow_exact_locate=True,
        source_filename=lambda source_path: "paper.pdf",
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        build_doc_list_ref_hit=build_hit,
        build_hit_ui_meta=build_hit_ui_meta,
        build_doc_list_hit_ui_seed=build_seed,
        apply_doc_list_effective_primary_evidence=apply_effective,
        apply_doc_list_summary_fallbacks=apply_summary,
        apply_doc_list_why_fallback=apply_why,
        finalize_doc_list_hit_ui_meta=finalize,
    )

    assert out["summary_source"] == "doc_list_seed"
    assert out["summary_line"] == "effective summary"
    assert out["source_name"] == "paper.pdf"
    assert out["primary_evidence"]["block_id"] == "blk"
    assert out["effective_primary_evidence"]["snippet"] == "effective"
    assert [call["fn"] for call in calls] == ["seed", "effective", "summary", "why", "finalize"]
    assert calls[1]["authoritative_summary_line"] == "authoritative summary"
    assert calls[2]["summary_source"] == "doc_list_seed"


def test_doc_list_topic_match_why_line_localizes_sci_notes() -> None:
    out_en = doc_list._doc_list_topic_match_why_line(
        prompt="Which papers mention SCI?",
        heading_path="5. Conclusions",
        match_kind="sci_related_predecessor",
        prefer_zh_ref_card_locale=lambda *args: False,
    )
    out_zh = doc_list._doc_list_topic_match_why_line(
        prompt="\u6709\u54ea\u4e9b\u6587\u732e\u63d0\u5230 SCI\uff1f",
        heading_path="2. Related Work",
        match_kind="explicit_sci_mention",
        prefer_zh_ref_card_locale=lambda *args: True,
    )

    assert "early related predecessor" in out_en
    assert "single-shot compressive spectral imaging" in out_en
    assert "Snapshot Compressive Imaging" in out_zh
    assert "\u660e\u786e\u63d0\u5230" in out_zh


def test_apply_doc_list_topic_match_hints_overrides_predecessor_copy_and_summary() -> None:
    why_basis_calls: list[dict] = []
    summary_basis_calls: list[dict] = []

    raw_summary = "The paper presents a single-shot compressive spectral imaging approach."
    out = doc_list._apply_doc_list_topic_match_hints(
        prompt="Which papers mention SCI?",
        raw_item={
            "topic_match_kind": "sci_related_predecessor",
            "heading_path": "5. Conclusions",
            "source_name": "OE-2007.pdf",
            "summary_line": raw_summary,
        },
        ui_meta={
            "display_name": "OE-2007.pdf",
            "heading_path": "5. Conclusions",
            "summary_kind": "guide",
            "summary_line": "Snapshot Compressive Imaging: generic summary.",
            "summary_generation": "section_grounded",
            "why_line": "This hit directly discusses Snapshot Compressive Imaging (SCI).",
            "why_generation": "deterministic_grounded",
        },
        doc_list_topic_match_why_line=lambda **kwargs: (
            "This paper is better treated as an early related predecessor: "
            "it discusses single-shot compressive spectral imaging."
        ),
        is_llm_ref_why_generation=lambda generation: False,
        why_line_needs_polish=lambda **kwargs: False,
        why_line_explicitly_names_focus_term=lambda prompt, why_line: True,
        build_ref_why_basis_meta=lambda **kwargs: why_basis_calls.append(kwargs)
        or {"why_generation": kwargs["why_generation"], "why_basis": "why basis"},
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        is_llm_ref_summary_generation=lambda generation: False,
        summary_line_needs_polish=lambda **kwargs: False,
        looks_like_title_echo=lambda summary, title: False,
        build_ref_summary_basis_meta=lambda **kwargs: summary_basis_calls.append(kwargs)
        or {"summary_generation": kwargs["summary_generation"], "summary_basis": "summary basis"},
    )

    assert out["topic_match_kind"] == "sci_related_predecessor"
    assert "early related predecessor" in out["why_line"]
    assert out["why_basis"] == "why basis"
    assert out["summary_line"] == raw_summary
    assert out["summary_basis"] == "summary basis"
    assert why_basis_calls[0]["why_generation"] == "deterministic_grounded"
    assert summary_basis_calls[0]["summary_line"] == raw_summary


def test_apply_doc_list_topic_match_hints_keeps_llm_why_copy() -> None:
    out = doc_list._apply_doc_list_topic_match_hints(
        prompt="Which papers mention SCI?",
        raw_item={
            "topic_match_kind": "explicit_sci_mention",
            "heading_path": "2. Related Work",
            "source_name": "SCINeRF.pdf",
        },
        ui_meta={
            "display_name": "SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "The paper discusses Snapshot Compressive Imaging.",
            "why_line": "LLM-grounded SCI explanation.",
            "why_generation": "llm_grounded",
        },
        doc_list_topic_match_why_line=lambda **kwargs: "deterministic SCI note",
        is_llm_ref_why_generation=lambda generation: True,
        why_line_needs_polish=lambda **kwargs: True,
        why_line_explicitly_names_focus_term=lambda prompt, why_line: False,
        build_ref_why_basis_meta=lambda **kwargs: (_ for _ in ()).throw(AssertionError("why basis unused")),
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        is_llm_ref_summary_generation=lambda generation: False,
        summary_line_needs_polish=lambda **kwargs: False,
        looks_like_title_echo=lambda summary, title: False,
        build_ref_summary_basis_meta=lambda **kwargs: {},
    )

    assert out["topic_match_kind"] == "explicit_sci_mention"
    assert out["why_line"] == "LLM-grounded SCI explanation."
    assert out["why_generation"] == "llm_grounded"


def test_filter_doc_list_rows_for_guide_filters_current_source_when_active() -> None:
    calls: list[dict] = []

    def hit_matches(meta: dict, **kwargs) -> bool:
        calls.append({"meta": meta, **kwargs})
        return (
            str(meta.get("source_path") or "") == kwargs["guide_source_path"]
            or str(meta.get("source_name") or "") == kwargs["guide_source_name"]
        )

    rows, filtered = doc_list._filter_doc_list_rows_for_guide(
        doc_rows=[
            {
                "source_path": "/kb/guide.md",
                "source_name": "",
                "summary_line": "self",
            },
            {
                "source_path": "/kb/other.md",
                "source_name": "Other.pdf",
                "summary_line": "other",
            },
            "not a row",
        ],
        guide_mode=True,
        guide_source_path="/kb/guide.md",
        guide_source_name="guide.pdf",
        filter_bound_source=True,
        source_filename=lambda source_path: "guide.pdf" if source_path == "/kb/guide.md" else "other.pdf",
        hit_matches_guide_source=hit_matches,
    )

    assert filtered == 1
    assert rows == [
        {
            "source_path": "/kb/other.md",
            "source_name": "Other.pdf",
            "summary_line": "other",
        }
    ]
    assert len(calls) == 2
    assert calls[0]["meta"]["source_name"] == "guide.pdf"
    assert calls[0]["guide_source_path"] == "/kb/guide.md"
    assert calls[0]["guide_source_name"] == "guide.pdf"


def test_filter_doc_list_rows_for_guide_inactive_keeps_clean_row_copies() -> None:
    original = {"source_path": "/kb/guide.md", "source_name": "Guide.pdf"}
    rows, filtered = doc_list._filter_doc_list_rows_for_guide(
        doc_rows=[original, None, {"source_path": "/kb/other.md"}],
        guide_mode=True,
        guide_source_path="/kb/guide.md",
        guide_source_name="Guide.pdf",
        filter_bound_source=False,
        source_filename=lambda source_path: "unused.pdf",
        hit_matches_guide_source=lambda **kwargs: (_ for _ in ()).throw(AssertionError("filter inactive")),
    )

    assert filtered == 0
    assert rows == [
        {"source_path": "/kb/guide.md", "source_name": "Guide.pdf"},
        {"source_path": "/kb/other.md"},
    ]
    assert rows[0] is not original


def test_build_doc_list_payload_hits_shapes_hits_without_reindexing_skipped_rows() -> None:
    calls: list[dict] = []

    def build_ui(**kwargs):
        calls.append({"fn": "build", **kwargs})
        return {
            "summary_line": f"summary {kwargs['idx']}",
            "heading_path": f"{kwargs['idx']}. Section",
        }

    def normalize(ui_meta):
        calls.append({"fn": "normalize", "ui_meta": ui_meta})
        out = dict(ui_meta or {})
        out["normalized"] = True
        return out

    def apply_hints(**kwargs):
        calls.append({"fn": "hints", **kwargs})
        out = dict(kwargs["ui_meta"])
        out["hinted_source"] = kwargs["raw_item"]["source_path"]
        return out

    hits = doc_list._build_doc_list_payload_hits(
        doc_rows=[
            {"source_path": "", "summary_line": "skip empty"},
            {"source_path": "/kb/paper-a.md", "source_name": "A.pdf"},
            "not a row",
            {"source_path": "/kb/paper-b.md", "source_name": "B.pdf"},
        ],
        prompt="method",
        allow_expensive_llm=True,
        allow_exact_locate=False,
        build_doc_list_hit_ui_meta=build_ui,
        normalize_ref_copy_ui_meta=normalize,
        apply_doc_list_topic_match_hints=apply_hints,
    )

    assert [hit["text"] for hit in hits] == ["summary 2", "summary 4"]
    assert [hit["meta"]["source_path"] for hit in hits] == ["/kb/paper-a.md", "/kb/paper-b.md"]
    assert [hit["meta"]["ref_best_heading_path"] for hit in hits] == ["2. Section", "4. Section"]
    assert [hit["ui_meta"]["normalized"] for hit in hits] == [True, True]
    assert [call["idx"] for call in calls if call["fn"] == "build"] == [2, 4]
    assert calls[0]["allow_expensive_llm"] is True
    assert calls[0]["allow_exact_locate"] is False
    assert [call["raw_item"]["source_path"] for call in calls if call["fn"] == "hints"] == [
        "/kb/paper-a.md",
        "/kb/paper-b.md",
    ]


def test_build_doc_list_payload_hits_restores_authoritative_summary_when_copy_repeats() -> None:
    authoritative = "The paper compares Hadamard and Fourier sampling under equal measurement budgets."
    repeated = "This paper helps compare sampling bases under the same measurement budget."

    hits = doc_list._build_doc_list_payload_hits(
        doc_rows=[
            {
                "source_path": "/kb/paper-a.md",
                "summary_line": authoritative,
                "summary_generation": "section_grounded",
            }
        ],
        prompt="Which paper should I read first?",
        allow_expensive_llm=False,
        allow_exact_locate=False,
        build_doc_list_hit_ui_meta=lambda **kwargs: {
            "summary_line": repeated,
            "summary_generation": "deterministic_grounded",
            "why_line": repeated,
            "why_generation": "deterministic_grounded",
        },
        normalize_ref_copy_ui_meta=lambda ui_meta: dict(ui_meta or {}),
        apply_doc_list_topic_match_hints=lambda **kwargs: dict(kwargs["ui_meta"]),
    )

    ui_meta = hits[0]["ui_meta"]
    assert ui_meta["summary_line"] == authoritative
    assert ui_meta["summary_generation"] == "section_grounded"
    assert ui_meta["why_line"] == repeated
    assert ui_meta["summary_source"] == "doc_list_authoritative_fast"


def test_build_doc_list_payload_hits_drops_repeated_why_without_distinct_summary() -> None:
    repeated = "This paper provides evidence for comparing single-pixel imaging sampling strategies."

    hits = doc_list._build_doc_list_payload_hits(
        doc_rows=[{"source_path": "/kb/paper-a.md"}],
        prompt="compare sampling strategies",
        allow_expensive_llm=False,
        allow_exact_locate=False,
        build_doc_list_hit_ui_meta=lambda **kwargs: {
            "summary_line": repeated,
            "why_line": "Initially distinct relevance explanation.",
            "why_generation": "deterministic_grounded",
            "why_basis": "same copy",
        },
        normalize_ref_copy_ui_meta=lambda ui_meta: dict(ui_meta or {}),
        apply_doc_list_topic_match_hints=lambda **kwargs: {
            **dict(kwargs["ui_meta"]),
            "why_line": repeated,
        },
    )

    ui_meta = hits[0]["ui_meta"]
    assert ui_meta["summary_line"] == repeated
    assert "why_line" not in ui_meta
    assert "why_generation" not in ui_meta
    assert "why_basis" not in ui_meta


def test_hydrate_doc_list_refs_payload_adds_local_metadata_and_sanitizes_copy(monkeypatch) -> None:
    source_path = "/kb/paper-a.md"
    authoritative = "The paper compares Hadamard and Fourier sampling under equal budgets."
    repeated = "This paper helps compare Hadamard and Fourier sampling under equal budgets."
    monkeypatch.setattr(
        reference_ui,
        "_cached_doc_list_citation_meta",
        lambda *args, **kwargs: {
            source_path: {
                "doi": "10.1364/OE.123456",
                "citation_count": 42,
                "journal_if": 3.3,
                "journal_quartile": "Q2",
            }
        },
    )

    out = reference_ui.hydrate_doc_list_refs_payload_citation_meta(
        {
            "hits": [
                {
                    "meta": {"source_path": source_path},
                    "ui_meta": {"summary_line": repeated, "why_line": repeated},
                }
            ]
        },
        doc_list=[{"source_path": source_path, "summary_line": authoritative}],
        pdf_root=None,
        lib_store=None,
    )

    ui_meta = out["hits"][0]["ui_meta"]
    assert ui_meta["citation_meta"]["doi"] == "10.1364/OE.123456"
    assert ui_meta["citation_meta"]["citation_count"] == 42
    assert ui_meta["citation_meta"]["journal_if"] == 3.3
    assert ui_meta["citation_meta"]["journal_quartile"] == "Q2"
    assert ui_meta["summary_line"] == authoritative
    assert ui_meta["why_line"] == repeated


def test_polish_doc_list_payload_hits_uses_batch_then_single_leftovers() -> None:
    calls: list[dict] = []
    doc_rows = [
        {"source_path": "/kb/paper-a.md", "source_name": "A.pdf"},
        {"source_path": "/kb/paper-b.md", "source_name": "B.pdf"},
    ]
    hits = [
        {"text": "A", "ui_meta": {"display_name": "A.pdf"}},
        {"text": "B", "ui_meta": {"display_name": "B.pdf"}},
    ]

    def normalize(ui_meta):
        calls.append({"fn": "normalize", "ui_meta": ui_meta})
        out = dict(ui_meta or {})
        out["normalized"] = True
        return out

    def single_polish(**kwargs):
        calls.append({"fn": "single", **kwargs})
        return {
            "display_name": str(kwargs["ui_meta"].get("display_name") or ""),
            "summary_line": f"single::{kwargs['hit']['text']}",
            "summary_generation": "deterministic_grounded",
        }

    def apply_hints(**kwargs):
        calls.append({"fn": "hints", **kwargs})
        out = dict(kwargs["ui_meta"])
        out["hinted_source"] = kwargs["raw_item"]["source_path"]
        return out

    def batch_polish(**kwargs):
        calls.append({"fn": "batch", **kwargs})
        return {
            0: {
                "display_name": "A.pdf",
                "summary_line": "batch::A",
                "summary_generation": "llm_grounded",
                "why_generation": "llm_grounded",
            }
        }

    out = doc_list._polish_doc_list_payload_hits(
        prompt="method",
        doc_rows=doc_rows,
        hits=hits,
        allow_expensive_llm=True,
        normalize_ref_copy_ui_meta=normalize,
        maybe_polish_single_ref_hit_card=single_polish,
        apply_doc_list_topic_match_hints=apply_hints,
        batch_polish_doc_list_ref_hit_cards=batch_polish,
        ref_card_has_llm_copy=lambda ui_meta: str((ui_meta or {}).get("summary_generation") or "") == "llm_grounded"
        and str((ui_meta or {}).get("why_generation") or "") == "llm_grounded",
        refs_card_polish_max_workers=lambda job_count: 1,
    )

    assert [hit["ui_meta"]["summary_line"] for hit in out] == ["batch::A", "single::B"]
    assert out[0]["ui_meta"]["hinted_source"] == "/kb/paper-a.md"
    assert out[1]["ui_meta"]["hinted_source"] == "/kb/paper-b.md"
    assert out[1]["ui_meta"]["normalized"] is True
    assert [call["fn"] for call in calls] == ["batch", "hints", "single", "normalize", "hints"]
    assert calls[0]["jobs"][0][0] == 0
    assert calls[0]["jobs"][1][0] == 1
    assert calls[2]["allow_expensive_llm"] is True
    assert calls[2]["ui_meta"]["display_name"] == "B.pdf"


def test_polish_doc_list_payload_hits_without_expensive_llm_skips_batch() -> None:
    calls: list[str] = []

    out = doc_list._polish_doc_list_payload_hits(
        prompt="method",
        doc_rows=[{"source_path": "/kb/paper-a.md"}],
        hits=[{"text": "A", "ui_meta": {"display_name": "A.pdf"}}],
        allow_expensive_llm=False,
        normalize_ref_copy_ui_meta=lambda ui_meta: dict(ui_meta or {}, normalized=True),
        maybe_polish_single_ref_hit_card=lambda **kwargs: calls.append("single")
        or {"summary_line": "single", "summary_generation": "deterministic_grounded"},
        apply_doc_list_topic_match_hints=lambda **kwargs: calls.append("hints") or dict(kwargs["ui_meta"]),
        batch_polish_doc_list_ref_hit_cards=lambda **kwargs: (_ for _ in ()).throw(AssertionError("batch unused")),
        ref_card_has_llm_copy=lambda ui_meta: False,
        refs_card_polish_max_workers=lambda job_count: 1,
    )

    assert out[0]["ui_meta"]["summary_line"] == "single"
    assert out[0]["ui_meta"]["normalized"] is True
    assert calls == ["single", "hints"]


def test_finalize_doc_list_payload_pack_sets_debug_guide_filter_and_contract() -> None:
    calls: list[dict] = []
    hits = [
        {
            "text": "summary",
            "ui_meta": {
                "summary_generation": "llm_grounded",
                "why_generation": "llm_grounded",
            },
        }
    ]

    def attach_contract(pack: dict) -> dict:
        calls.append(pack)
        return dict(pack, display_state="ready")

    out = doc_list._finalize_doc_list_payload_pack(
        user_msg_id="42",
        pack_src={
            "prompt": "method",
            "query_variants": ["method", "approach"],
            "pipeline_debug": {"kept": "yes"},
        },
        hits=hits,
        guide_active=True,
        guide_source_path_norm="/kb/guide.md",
        guide_source_name_norm="",
        prompt_cross_paper_refs=True,
        filtered_self_doc_count=1,
        allow_expensive_llm=True,
        refs_hits_have_llm_copy=lambda hits_arg: hits_arg is hits,
        source_filename=lambda source_path: "Guide.pdf",
        attach_pack_display_contract=attach_contract,
    )

    assert out["user_msg_id"] == 42
    assert out["hits"] is hits
    assert out["payload_mode"] == "full"
    assert out["display_state"] == "ready"
    pipeline_debug = dict(out.get("pipeline_debug") or {})
    assert pipeline_debug["kept"] == "yes"
    assert pipeline_debug["doc_list_authoritative"] is True
    assert pipeline_debug["guide_active"] is True
    assert pipeline_debug["final_hit_count"] == 1
    assert pipeline_debug["raw_hit_count"] == 1
    assert pipeline_debug["post_score_gate_hit_count"] == 1
    assert pipeline_debug["post_focus_filter_hit_count"] == 1
    assert pipeline_debug["post_llm_filter_hit_count"] == 1
    assert pipeline_debug["filtered_self_hit_count"] == 1
    assert pipeline_debug["prompt_likely_cross_paper_refs"] is True
    assert pipeline_debug["copy_polish_allow_expensive_llm"] is True
    assert pipeline_debug["copy_polish_llm_required"] is True
    assert pipeline_debug["copy_polish_llm_complete"] is True
    assert pipeline_debug["query_variants"] == ["method", "approach"]
    assert out["guide_filter"] == {
        "active": True,
        "hidden_self_source": True,
        "filtered_hit_count": 1,
        "guide_source_path": "/kb/guide.md",
        "guide_source_name": "Guide.pdf",
    }
    assert calls[0]["payload_mode"] == "full"


def test_finalize_doc_list_payload_pack_skips_guide_filter_for_regular_mode() -> None:
    out = doc_list._finalize_doc_list_payload_pack(
        user_msg_id="msg-x",
        pack_src={"pipeline_debug": {}},
        hits=[],
        guide_active=False,
        guide_source_path_norm="",
        guide_source_name_norm="",
        prompt_cross_paper_refs=True,
        filtered_self_doc_count=0,
        allow_expensive_llm=False,
        refs_hits_have_llm_copy=lambda hits_arg: False,
        source_filename=lambda source_path: "unused.pdf",
        attach_pack_display_contract=lambda pack: dict(pack, display_state="empty"),
    )

    assert out["user_msg_id"] == "msg-x"
    assert "guide_filter" not in out
    assert out["pipeline_debug"]["copy_polish_allow_expensive_llm"] is False
    assert out["pipeline_debug"]["copy_polish_llm_complete"] is False
    assert out["pipeline_debug"]["final_hit_count"] == 0


def test_finalize_legacy_doc_list_payload_pack_preserves_existing_counts_and_sets_guide_filter() -> None:
    calls: list[dict] = []
    hits = [{"text": "legacy"}]

    def attach_contract(pack: dict) -> dict:
        calls.append(pack)
        return dict(pack, display_state="hidden_by_guide")

    out = doc_list._finalize_legacy_doc_list_payload_pack(
        user_msg_id="43",
        pack_src={
            "prompt": "method",
            "pipeline_debug": {
                "raw_hit_count": 9,
                "post_score_gate_hit_count": 8,
                "post_focus_filter_hit_count": 7,
                "post_llm_filter_hit_count": 6,
                "filtered_self_hit_count": 5,
            },
        },
        hits=hits,
        guide_active=True,
        guide_source_path_norm="/kb/guide.md",
        guide_source_name_norm="",
        prompt_cross_paper_refs=True,
        source_filename=lambda source_path: "Guide.pdf",
        attach_pack_display_contract=attach_contract,
    )

    assert out["user_msg_id"] == 43
    assert out["hits"] is hits
    assert out["payload_mode"] == "full"
    assert out["display_state"] == "hidden_by_guide"
    pipeline_debug = dict(out.get("pipeline_debug") or {})
    assert pipeline_debug["doc_list_authoritative"] is True
    assert pipeline_debug["guide_active"] is True
    assert pipeline_debug["final_hit_count"] == 1
    assert pipeline_debug["raw_hit_count"] == 9
    assert pipeline_debug["post_score_gate_hit_count"] == 8
    assert pipeline_debug["post_focus_filter_hit_count"] == 7
    assert pipeline_debug["post_llm_filter_hit_count"] == 6
    assert pipeline_debug["filtered_self_hit_count"] == 5
    assert pipeline_debug["prompt_likely_cross_paper_refs"] is True
    assert out["guide_filter"] == {
        "active": True,
        "hidden_self_source": True,
        "filtered_hit_count": 0,
        "guide_source_path": "/kb/guide.md",
        "guide_source_name": "Guide.pdf",
    }
    assert calls[0]["payload_mode"] == "full"


def test_finalize_legacy_doc_list_payload_pack_fills_missing_counts_without_guide_filter() -> None:
    out = doc_list._finalize_legacy_doc_list_payload_pack(
        user_msg_id="msg-x",
        pack_src={"pipeline_debug": {}},
        hits=[{"text": "a"}, {"text": "b"}],
        guide_active=False,
        guide_source_path_norm="",
        guide_source_name_norm="",
        prompt_cross_paper_refs=False,
        source_filename=lambda source_path: "unused.pdf",
        attach_pack_display_contract=lambda pack: dict(pack, display_state="ready"),
    )

    assert out["user_msg_id"] == "msg-x"
    assert "guide_filter" not in out
    assert out["display_state"] == "ready"
    pipeline_debug = dict(out.get("pipeline_debug") or {})
    assert pipeline_debug["final_hit_count"] == 2
    assert pipeline_debug["raw_hit_count"] == 2
    assert pipeline_debug["post_score_gate_hit_count"] == 2
    assert pipeline_debug["post_focus_filter_hit_count"] == 2
    assert pipeline_debug["post_llm_filter_hit_count"] == 2
    assert pipeline_debug["filtered_self_hit_count"] == 0
    assert pipeline_debug["prompt_likely_cross_paper_refs"] is False


def test_build_legacy_doc_list_payload_hits_shapes_contract_meta_without_reindexing() -> None:
    hits = doc_list._build_legacy_doc_list_payload_hits(
        doc_list=[
            "not a row",
            {"source_path": "", "summary_line": "skip empty"},
            {
                "source_path": "/kb/paper-a.md",
                "source_name": "",
                "heading_path": "",
                "primary_evidence": {
                    "heading_path": "2. Method",
                    "highlight_snippet": "primary summary",
                    "snippet": "fallback snippet",
                    "strict_locate": True,
                    "block_id": "blk-a",
                    "anchor_id": "anc-a",
                },
            },
        ],
        prompt="method",
        prefer_zh=True,
        source_filename=lambda source_path: "Paper A.pdf",
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        doc_list_ref_why_line=lambda **kwargs: f"why::{kwargs['heading_path']}::{kwargs['prefer_zh']}",
        score_tier=lambda score: f"tier::{score}",
    )

    assert len(hits) == 1
    hit = hits[0]
    assert hit["text"] == "primary summary"
    assert hit["meta"] == {
        "source_path": "/kb/paper-a.md",
        "ref_pack_state": "ready",
        "ref_best_heading_path": "2. Method",
    }
    ui_meta = dict(hit["ui_meta"])
    assert ui_meta["display_name"] == "Paper A.pdf"
    assert ui_meta["heading_path"] == "2. Method"
    assert ui_meta["score"] == 9.24
    assert ui_meta["score_tier"] == "tier::9.24"
    assert ui_meta["summary_line"] == "primary summary"
    assert ui_meta["summary_label"] == "\u5bfc\u8bfb"
    assert ui_meta["summary_title"] == "\u8fd9\u6761\u8bc1\u636e\u8bf4\u660e\u4ec0\u4e48"
    assert ui_meta["summary_generation"] == "doc_list_contract"
    assert ui_meta["summary_basis"] == "\u57fa\u4e8e\u5171\u4eab\u591a\u7bc7\u6587\u732e\u5217\u8868 contract \u7684\u5c55\u793a\u6458\u8981"
    assert ui_meta["why_line"] == "why::2. Method::True"
    assert ui_meta["why_generation"] == "doc_list_contract"
    assert ui_meta["why_basis"] == "\u57fa\u4e8e\u5171\u4eab\u591a\u7bc7\u6587\u732e\u5217\u8868 contract \u7684\u4fdd\u7559\u7406\u7531"
    assert ui_meta["can_open"] is True
    assert ui_meta["primary_evidence"]["block_id"] == "blk-a"
    assert ui_meta["reader_open"]["sourcePath"] == "/kb/paper-a.md"
    assert ui_meta["reader_open"]["sourceName"] == "Paper A.pdf"
    assert ui_meta["reader_open"]["headingPath"] == "2. Method"
    assert ui_meta["reader_open"]["strictLocate"] is True
    assert ui_meta["reader_open"]["blockId"] == "blk-a"
    assert ui_meta["reader_open"]["anchorId"] == "anc-a"
    assert ui_meta["reader_open"]["primaryEvidence"]["highlight_snippet"] == "primary summary"


def test_build_legacy_doc_list_payload_hits_prefers_raw_summary_and_english_labels() -> None:
    hits = doc_list._build_legacy_doc_list_payload_hits(
        doc_list=[
            {
                "source_path": "/kb/paper-b.md",
                "source_name": "Paper B.pdf",
                "heading_path": "Abstract",
                "summary_line": "raw summary",
                "primary_evidence": {"snippet": "primary snippet"},
            }
        ],
        prompt="method",
        prefer_zh=False,
        source_filename=lambda source_path: "unused.pdf",
        normalize_primary_ref_evidence_payload=lambda raw: dict(raw or {}),
        compact_reader_open_text=lambda text, max_len=360: str(text or "").strip(),
        doc_list_ref_why_line=lambda **kwargs: "why",
        score_tier=lambda score: "high",
    )

    ui_meta = dict(hits[0]["ui_meta"])
    assert hits[0]["text"] == "raw summary"
    assert ui_meta["summary_label"] == "Guide"
    assert ui_meta["summary_title"] == "What This Evidence Shows"
    assert ui_meta["summary_basis"] == "Display summary sourced from the shared multi-paper document list contract"
    assert ui_meta["why_basis"] == "Retention reason sourced from the shared multi-paper document list contract"
    assert ui_meta["reader_open"]["strictLocate"] is False
    assert ui_meta["reader_open"]["primaryEvidence"]["snippet"] == "primary snippet"


def test_build_doc_list_refs_payload_full_branch_runs_pipeline_helpers_in_order() -> None:
    calls: list[dict] = []

    def filter_rows(**kwargs):
        calls.append({"fn": "filter", **kwargs})
        return [{"source_path": "/kb/other.md"}], 1

    def build_hits(**kwargs):
        calls.append({"fn": "build_hits", **kwargs})
        return [{"text": "raw", "ui_meta": {"summary_line": "raw"}}]

    def polish_hits(**kwargs):
        calls.append({"fn": "polish", **kwargs})
        return [{"text": "polished", "ui_meta": {"summary_line": "polished"}}]

    def suppress_hits(**kwargs):
        calls.append({"fn": "suppress", **kwargs})
        return [{"text": "suppressed", "ui_meta": {"summary_line": "suppressed"}}]

    def finalize(**kwargs):
        calls.append({"fn": "finalize", **kwargs})
        return {
            "hits": kwargs["hits"],
            "guide_active": kwargs["guide_active"],
            "filtered_self_doc_count": kwargs["filtered_self_doc_count"],
            "prompt_cross_paper_refs": kwargs["prompt_cross_paper_refs"],
        }

    out = doc_list._build_doc_list_refs_payload(
        user_msg_id="42",
        pack={"prompt": "Besides this paper, list related work."},
        doc_list=[{"source_path": "/kb/self.md"}],
        allow_expensive_llm=True,
        allow_exact_locate=False,
        apply_copy_polish=True,
        guide_mode=True,
        guide_source_path=" /kb/self.md ",
        guide_source_name=" Self.pdf ",
        prompt_likely_cross_paper_refs=lambda prompt: True,
        filter_doc_list_rows_for_guide=filter_rows,
        build_doc_list_payload_hits=build_hits,
        polish_doc_list_payload_hits=polish_hits,
        suppress_non_llm_ref_card_copy_hits=suppress_hits,
        finalize_doc_list_payload_pack=finalize,
        prefer_zh_ref_card_locale=lambda prompt: (_ for _ in ()).throw(AssertionError("legacy locale unused")),
        build_legacy_doc_list_payload_hits=lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy hits unused")),
        finalize_legacy_doc_list_payload_pack=lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy finalize unused")),
    )

    assert [call["fn"] for call in calls] == ["filter", "build_hits", "polish", "suppress", "finalize"]
    assert calls[0]["doc_rows"] == [{"source_path": "/kb/self.md"}]
    assert calls[0]["guide_mode"] is True
    assert calls[0]["guide_source_path"] == "/kb/self.md"
    assert calls[0]["guide_source_name"] == "Self.pdf"
    assert calls[0]["filter_bound_source"] is True
    assert calls[1]["doc_rows"] == [{"source_path": "/kb/other.md"}]
    assert calls[1]["allow_expensive_llm"] is True
    assert calls[1]["allow_exact_locate"] is False
    assert calls[2]["doc_rows"] == [{"source_path": "/kb/other.md"}]
    assert calls[3]["hits"][0]["text"] == "polished"
    assert calls[4]["hits"][0]["text"] == "suppressed"
    assert calls[4]["filtered_self_doc_count"] == 1
    assert out["guide_active"] is True
    assert out["prompt_cross_paper_refs"] is True


def test_build_doc_list_refs_payload_legacy_branch_runs_legacy_helpers() -> None:
    calls: list[dict] = []

    def filter_rows(**kwargs):
        calls.append({"fn": "filter", **kwargs})
        return [], 0

    def legacy_hits(**kwargs):
        calls.append({"fn": "legacy_hits", **kwargs})
        return [{"text": "legacy"}]

    def legacy_finalize(**kwargs):
        calls.append({"fn": "legacy_finalize", **kwargs})
        return {
            "hits": kwargs["hits"],
            "guide_active": kwargs["guide_active"],
            "prompt_cross_paper_refs": kwargs["prompt_cross_paper_refs"],
        }

    out = doc_list._build_doc_list_refs_payload(
        user_msg_id="msg-x",
        pack={"prompt": "\u8bf7\u5217\u51fa\u76f8\u5173\u6587\u732e"},
        doc_list=[],
        allow_expensive_llm=False,
        allow_exact_locate=True,
        apply_copy_polish=True,
        guide_mode=True,
        guide_source_path="/kb/self.md",
        guide_source_name="Self.pdf",
        prompt_likely_cross_paper_refs=lambda prompt: False,
        filter_doc_list_rows_for_guide=filter_rows,
        build_doc_list_payload_hits=lambda **kwargs: (_ for _ in ()).throw(AssertionError("full build unused")),
        polish_doc_list_payload_hits=lambda **kwargs: (_ for _ in ()).throw(AssertionError("full polish unused")),
        suppress_non_llm_ref_card_copy_hits=lambda **kwargs: (_ for _ in ()).throw(AssertionError("suppress unused")),
        finalize_doc_list_payload_pack=lambda **kwargs: (_ for _ in ()).throw(AssertionError("full finalize unused")),
        prefer_zh_ref_card_locale=lambda prompt: True,
        build_legacy_doc_list_payload_hits=legacy_hits,
        finalize_legacy_doc_list_payload_pack=legacy_finalize,
    )

    assert [call["fn"] for call in calls] == ["filter", "legacy_hits", "legacy_finalize"]
    assert calls[0]["doc_rows"] == []
    assert calls[0]["guide_mode"] is True
    assert calls[0]["filter_bound_source"] is False
    assert calls[1]["doc_list"] == []
    assert calls[1]["prefer_zh"] is True
    assert calls[2]["hits"] == [{"text": "legacy"}]
    assert calls[2]["guide_active"] is True
    assert calls[2]["prompt_cross_paper_refs"] is False
    assert out["hits"] == [{"text": "legacy"}]


def test_reference_ui_doc_list_helpers_use_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_collect(**kwargs):
        calls.append({"fn": "collect", **kwargs})
        return ["candidate"]

    def fake_seed(primary_evidence, **kwargs):
        calls.append({"fn": "seed", "primary_evidence": primary_evidence, **kwargs})
        return "seed"

    def fake_locs(**kwargs):
        calls.append({"fn": "locs", **kwargs})
        return [{"source": "doc_list_primary"}]

    def fake_hit(**kwargs):
        calls.append({"fn": "hit", **kwargs})
        return {"text": "hit", "meta": {}}

    monkeypatch.setattr(doc_list, "_collect_doc_list_ref_text_candidates", fake_collect)
    monkeypatch.setattr(doc_list, "_primary_ref_evidence_summary_seed", fake_seed)
    monkeypatch.setattr(doc_list, "_build_doc_list_ref_locs", fake_locs)
    monkeypatch.setattr(doc_list, "_build_doc_list_ref_hit", fake_hit)

    raw_item = {"source_path": "/kb/paper.md"}
    primary = {"snippet": "primary"}
    assert reference_ui._collect_doc_list_ref_text_candidates(raw_item=raw_item, primary_evidence=primary) == ["candidate"]
    assert reference_ui._primary_ref_evidence_summary_seed(primary) == "seed"
    assert reference_ui._build_doc_list_ref_locs(heading_path="2. Method", primary_evidence=primary) == [
        {"source": "doc_list_primary"}
    ]
    assert reference_ui._build_doc_list_ref_hit(raw_item=raw_item, idx=1) == {"text": "hit", "meta": {}}

    assert calls == [
        {
            "fn": "collect",
            "raw_item": raw_item,
            "primary_evidence": primary,
            "clean_refs_evidence_snippet": reference_ui._clean_refs_evidence_snippet,
        },
        {
            "fn": "seed",
            "primary_evidence": primary,
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "clean_refs_evidence_snippet": reference_ui._clean_refs_evidence_snippet,
        },
        {
            "fn": "locs",
            "heading_path": "2. Method",
            "primary_evidence": primary,
            "clean_refs_evidence_snippet": reference_ui._clean_refs_evidence_snippet,
            "top_heading": reference_ui._top_heading,
        },
        {
            "fn": "hit",
            "raw_item": raw_item,
            "idx": 1,
            "source_filename": reference_ui._source_filename,
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "compact_reader_open_text": reference_ui._compact_reader_open_text,
            "split_section_subsection": reference_ui._split_section_subsection,
            "top_heading": reference_ui._top_heading,
            "clean_refs_evidence_snippet": reference_ui._clean_refs_evidence_snippet,
        },
    ]


def test_reference_ui_doc_list_primary_selection_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return {"block_id": "blk"}, "synthesized"

    monkeypatch.setattr(doc_list, "_select_doc_list_effective_primary_evidence", fake_select)

    authoritative = {"snippet": "auth"}
    synthesized = {"snippet": "synth"}
    assert reference_ui._select_doc_list_effective_primary_evidence(
        prompt="method",
        display_name="Paper",
        authoritative_primary_evidence=authoritative,
        synthesized_primary_evidence=synthesized,
    ) == ({"block_id": "blk"}, "synthesized")
    assert calls == [
        {
            "prompt": "method",
            "display_name": "Paper",
            "authoritative_primary_evidence": authoritative,
            "synthesized_primary_evidence": synthesized,
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "upgrade_primary_ref_evidence_from_alternatives": reference_ui._upgrade_primary_ref_evidence_from_alternatives,
            "primary_ref_evidence_points_to_same_surface": reference_ui._primary_ref_evidence_points_to_same_surface,
            "doc_list_authoritative_primary_is_upgradeable": reference_ui._doc_list_authoritative_primary_is_upgradeable,
            "primary_ref_evidence_precision_score": reference_ui._primary_ref_evidence_precision_score,
        }
    ]


def test_reference_ui_apply_doc_list_effective_primary_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_apply(**kwargs):
        calls.append(kwargs)
        return {"summary_line": "ok"}, {"block_id": "blk"}

    monkeypatch.setattr(doc_list, "_apply_doc_list_effective_primary_evidence", fake_apply)

    ui_meta = {"summary_line": "old"}
    authoritative = {"snippet": "auth"}
    assert reference_ui._apply_doc_list_effective_primary_evidence(
        prompt="method",
        display_name="Paper",
        fallback_heading_path="2. Method",
        ui_meta=ui_meta,
        authoritative_primary_evidence=authoritative,
        authoritative_summary_line="summary",
        authoritative_summary_generation="llm_grounded",
    ) == ({"summary_line": "ok"}, {"block_id": "blk"})
    assert calls == [
        {
            "prompt": "method",
            "display_name": "Paper",
            "fallback_heading_path": "2. Method",
            "ui_meta": ui_meta,
            "authoritative_primary_evidence": authoritative,
            "authoritative_summary_line": "summary",
            "authoritative_summary_generation": "llm_grounded",
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "select_doc_list_effective_primary_evidence": reference_ui._select_doc_list_effective_primary_evidence,
            "primary_ref_evidence_summary_seed": reference_ui._primary_ref_evidence_summary_seed,
            "compact_reader_open_text": reference_ui._compact_reader_open_text,
            "summary_line_needs_polish": reference_ui._summary_line_needs_polish,
            "primary_ref_evidence_points_to_same_surface": reference_ui._primary_ref_evidence_points_to_same_surface,
            "build_ref_summary_basis_meta": reference_ui._build_ref_summary_basis_meta,
        }
    ]


def test_reference_ui_doc_list_hit_ui_seed_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_seed(**kwargs):
        calls.append(kwargs)
        return {"text": "hit"}, {"summary_line": "seed"}, {"snippet": "primary"}

    monkeypatch.setattr(doc_list, "_build_doc_list_hit_ui_seed", fake_seed)

    raw_item = {"source_path": "/kb/paper.md"}
    assert reference_ui._build_doc_list_hit_ui_seed(
        raw_item=raw_item,
        idx=2,
        prompt="method",
    ) == ({"text": "hit"}, {"summary_line": "seed"}, {"snippet": "primary"})
    assert calls == [
        {
            "raw_item": raw_item,
            "idx": 2,
            "prompt": "method",
            "build_doc_list_ref_hit": reference_ui._build_doc_list_ref_hit,
            "source_filename": reference_ui._source_filename,
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "compact_reader_open_text": reference_ui._compact_reader_open_text,
            "normalize_ref_copy_text": reference_ui._normalize_ref_copy_text,
            "resolve_ref_ui_heading_context": reference_ui._resolve_ref_ui_heading_context,
            "top_heading": reference_ui._top_heading,
            "primary_ref_evidence_summary_seed": reference_ui._primary_ref_evidence_summary_seed,
            "build_ref_summary_basis_meta": reference_ui._build_ref_summary_basis_meta,
            "build_prompt_aligned_ref_why_line": reference_ui._build_prompt_aligned_ref_why_line_v3,
            "doc_list_ref_why_line": reference_ui._doc_list_ref_why_line,
            "prefer_zh_ref_card_locale": reference_ui._prefer_zh_ref_card_locale,
            "build_ref_why_basis_meta": reference_ui._build_ref_why_basis_meta,
            "summary_label": "\u5bfc\u8bfb",
            "summary_title": "\u8fd9\u6761\u8bc1\u636e\u8bf4\u660e\u4ec0\u4e48",
        }
    ]


def test_reference_ui_doc_list_summary_fallbacks_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_summary(**kwargs):
        calls.append(kwargs)
        return {"summary_line": "fallback"}, "doc_list_fallback"

    monkeypatch.setattr(doc_list, "_apply_doc_list_summary_fallbacks", fake_summary)

    raw_item = {"source_path": "/kb/paper.md"}
    ui_meta = {"summary_line": "old"}
    primary = {"snippet": "primary"}
    effective = {"snippet": "effective"}
    assert reference_ui._apply_doc_list_summary_fallbacks(
        raw_item=raw_item,
        prompt="method",
        source_name="Paper",
        heading_path="2. Method",
        ui_meta=ui_meta,
        primary_evidence=primary,
        effective_primary_evidence=effective,
        summary_source="doc_list_seed",
    ) == ({"summary_line": "fallback"}, "doc_list_fallback")
    assert calls == [
        {
            "raw_item": raw_item,
            "prompt": "method",
            "source_name": "Paper",
            "heading_path": "2. Method",
            "ui_meta": ui_meta,
            "primary_evidence": primary,
            "effective_primary_evidence": effective,
            "summary_source": "doc_list_seed",
            "summary_line_needs_polish": reference_ui._summary_line_needs_polish,
            "looks_like_title_echo": reference_ui._looks_like_title_echo,
            "looks_why_like_ref_summary": reference_ui._looks_why_like_ref_summary,
            "pick_ref_card_summary_fallback": reference_ui._pick_ref_card_summary_fallback,
            "collect_doc_list_ref_text_candidates": reference_ui._collect_doc_list_ref_text_candidates,
            "build_ref_summary_basis_meta": reference_ui._build_ref_summary_basis_meta,
            "looks_fragmentary_ref_summary": reference_ui._looks_fragmentary_ref_summary,
            "looks_surface_like_ref_summary": reference_ui._looks_surface_like_ref_summary,
            "looks_formula_heavy_ref_text": reference_ui._looks_formula_heavy_ref_text,
            "build_prompt_aligned_ref_summary_fallback": reference_ui._build_prompt_aligned_ref_summary_fallback,
            "compact_reader_open_text": reference_ui._compact_reader_open_text,
            "primary_ref_evidence_summary_seed": reference_ui._primary_ref_evidence_summary_seed,
        }
    ]


def test_reference_ui_doc_list_why_fallback_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_why(**kwargs):
        calls.append(kwargs)
        return {"why_line": "fallback"}

    monkeypatch.setattr(doc_list, "_apply_doc_list_why_fallback", fake_why)

    ui_meta = {"why_line": "old"}
    assert reference_ui._apply_doc_list_why_fallback(
        prompt="method",
        source_name="Paper",
        heading_path="2. Method",
        ui_meta=ui_meta,
    ) == {"why_line": "fallback"}
    assert calls == [
        {
            "prompt": "method",
            "source_name": "Paper",
            "heading_path": "2. Method",
            "ui_meta": ui_meta,
            "why_line_needs_polish": reference_ui._why_line_needs_polish,
            "build_prompt_aligned_ref_why_line": reference_ui._build_prompt_aligned_ref_why_line_v3,
            "doc_list_ref_why_line": reference_ui._doc_list_ref_why_line,
            "prefer_zh_ref_card_locale": reference_ui._prefer_zh_ref_card_locale,
            "build_ref_why_basis_meta": reference_ui._build_ref_why_basis_meta,
        }
    ]


def test_reference_ui_finalize_doc_list_hit_ui_meta_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_finalize(**kwargs):
        calls.append(kwargs)
        return {"summary_source": kwargs["summary_source"]}

    monkeypatch.setattr(doc_list, "_finalize_doc_list_hit_ui_meta", fake_finalize)

    raw_item = {"source_path": "/kb/paper.md"}
    ui_meta = {"summary_line": "summary"}
    primary = {"snippet": "primary"}
    effective = {"snippet": "effective"}
    assert reference_ui._finalize_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=2,
        prompt="method",
        source_path="/kb/paper.md",
        source_name="Paper",
        heading_path="2. Method",
        ui_meta=ui_meta,
        primary_evidence=primary,
        effective_primary_evidence=effective,
        summary_source="doc_list_seed",
        allow_expensive_llm=True,
    ) == {"summary_source": "doc_list_seed"}
    assert calls == [
        {
            "raw_item": raw_item,
            "idx": 2,
            "prompt": "method",
            "source_path": "/kb/paper.md",
            "source_name": "Paper",
            "heading_path": "2. Method",
            "ui_meta": ui_meta,
            "primary_evidence": primary,
            "effective_primary_evidence": effective,
            "summary_source": "doc_list_seed",
            "allow_expensive_llm": True,
            "align_ref_card_copy_to_user_locale": reference_ui._align_ref_card_copy_to_user_locale,
            "build_ref_summary_surface_meta": reference_ui._build_ref_summary_surface_meta,
            "build_ref_summary_basis_meta": reference_ui._build_ref_summary_basis_meta,
            "build_ref_why_basis_meta": reference_ui._build_ref_why_basis_meta,
            "score_tier": reference_ui._score_tier,
            "build_doc_list_reader_open_payload": reference_ui._build_doc_list_reader_open_payload,
        }
    ]


def test_reference_ui_doc_list_hit_ui_meta_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return {"summary_source": "doc_list_seed"}

    monkeypatch.setattr(doc_list, "_build_doc_list_hit_ui_meta", fake_build)

    raw_item = {"source_path": "/kb/paper.md"}
    assert reference_ui._build_doc_list_hit_ui_meta(
        raw_item=raw_item,
        idx=3,
        prompt="method",
        allow_expensive_llm=True,
        allow_exact_locate=False,
    ) == {"summary_source": "doc_list_seed"}
    assert calls == [
        {
            "raw_item": raw_item,
            "idx": 3,
            "prompt": "method",
            "allow_expensive_llm": True,
            "allow_exact_locate": False,
            "source_filename": reference_ui._source_filename,
            "compact_reader_open_text": reference_ui._compact_reader_open_text,
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "build_doc_list_ref_hit": reference_ui._build_doc_list_ref_hit,
            "build_hit_ui_meta": reference_ui.build_hit_ui_meta,
            "build_doc_list_hit_ui_seed": reference_ui._build_doc_list_hit_ui_seed,
            "apply_doc_list_effective_primary_evidence": reference_ui._apply_doc_list_effective_primary_evidence,
            "apply_doc_list_summary_fallbacks": reference_ui._apply_doc_list_summary_fallbacks,
            "apply_doc_list_why_fallback": reference_ui._apply_doc_list_why_fallback,
            "finalize_doc_list_hit_ui_meta": reference_ui._finalize_doc_list_hit_ui_meta,
        }
    ]


def test_reference_ui_doc_list_topic_match_why_line_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_why(**kwargs):
        calls.append(kwargs)
        return "topic why"

    monkeypatch.setattr(doc_list, "_doc_list_topic_match_why_line", fake_why)

    assert reference_ui._doc_list_topic_match_why_line(
        prompt="SCI?",
        heading_path="2. Related Work",
        match_kind="explicit_sci_mention",
    ) == "topic why"
    assert calls == [
        {
            "prompt": "SCI?",
            "heading_path": "2. Related Work",
            "match_kind": "explicit_sci_mention",
            "prefer_zh_ref_card_locale": reference_ui._prefer_zh_ref_card_locale,
        }
    ]


def test_reference_ui_apply_doc_list_topic_match_hints_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_apply(**kwargs):
        calls.append(kwargs)
        return {"why_line": "topic why"}

    monkeypatch.setattr(doc_list, "_apply_doc_list_topic_match_hints", fake_apply)

    raw_item = {"topic_match_kind": "explicit_sci_mention"}
    ui_meta = {"why_line": "old"}
    assert reference_ui._apply_doc_list_topic_match_hints(
        prompt="SCI?",
        raw_item=raw_item,
        ui_meta=ui_meta,
    ) == {"why_line": "topic why"}
    assert calls == [
        {
            "prompt": "SCI?",
            "raw_item": raw_item,
            "ui_meta": ui_meta,
            "doc_list_topic_match_why_line": reference_ui._doc_list_topic_match_why_line,
            "is_llm_ref_why_generation": reference_ui._is_llm_ref_why_generation,
            "why_line_needs_polish": reference_ui._why_line_needs_polish,
            "why_line_explicitly_names_focus_term": reference_ui._why_line_explicitly_names_focus_term,
            "build_ref_why_basis_meta": reference_ui._build_ref_why_basis_meta,
            "compact_reader_open_text": reference_ui._compact_reader_open_text,
            "is_llm_ref_summary_generation": reference_ui._is_llm_ref_summary_generation,
            "summary_line_needs_polish": reference_ui._summary_line_needs_polish,
            "looks_like_title_echo": reference_ui._looks_like_title_echo,
            "build_ref_summary_basis_meta": reference_ui._build_ref_summary_basis_meta,
        }
    ]


def test_reference_ui_filter_doc_list_rows_for_guide_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_filter(**kwargs):
        calls.append(kwargs)
        return [{"source_path": "/kb/other.md"}], 1

    monkeypatch.setattr(doc_list, "_filter_doc_list_rows_for_guide", fake_filter)

    doc_rows = [{"source_path": "/kb/guide.md"}]
    assert reference_ui._filter_doc_list_rows_for_guide(
        doc_rows=doc_rows,
        guide_mode=True,
        guide_source_path="/kb/guide.md",
        guide_source_name="Guide.pdf",
        filter_bound_source=True,
    ) == ([{"source_path": "/kb/other.md"}], 1)
    assert calls == [
        {
            "doc_rows": doc_rows,
            "guide_mode": True,
            "guide_source_path": "/kb/guide.md",
            "guide_source_name": "Guide.pdf",
            "filter_bound_source": True,
            "source_filename": reference_ui._source_filename,
            "hit_matches_guide_source": reference_ui._hit_matches_guide_source,
        }
    ]


def test_reference_ui_build_doc_list_payload_hits_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_hits(**kwargs):
        calls.append(kwargs)
        return [{"text": "summary", "ui_meta": {"summary_line": "summary"}}]

    monkeypatch.setattr(doc_list, "_build_doc_list_payload_hits", fake_hits)

    doc_rows = [{"source_path": "/kb/paper.md"}]
    assert reference_ui._build_doc_list_payload_hits(
        doc_rows=doc_rows,
        prompt="method",
        allow_expensive_llm=True,
        allow_exact_locate=False,
    ) == [{"text": "summary", "ui_meta": {"summary_line": "summary"}}]
    assert calls == [
        {
            "doc_rows": doc_rows,
            "prompt": "method",
            "allow_expensive_llm": True,
            "allow_exact_locate": False,
            "build_doc_list_hit_ui_meta": reference_ui._build_doc_list_hit_ui_meta,
            "normalize_ref_copy_ui_meta": reference_ui._normalize_ref_copy_ui_meta,
            "apply_doc_list_topic_match_hints": reference_ui._apply_doc_list_topic_match_hints,
        }
    ]


def test_reference_ui_polish_doc_list_payload_hits_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_polish(**kwargs):
        calls.append(kwargs)
        return [{"text": "polished", "ui_meta": {"summary_line": "polished"}}]

    monkeypatch.setattr(doc_list, "_polish_doc_list_payload_hits", fake_polish)

    doc_rows = [{"source_path": "/kb/paper.md"}]
    hits = [{"text": "raw", "ui_meta": {"summary_line": "raw"}}]
    assert reference_ui._polish_doc_list_payload_hits(
        prompt="method",
        doc_rows=doc_rows,
        hits=hits,
        allow_expensive_llm=True,
    ) == [{"text": "polished", "ui_meta": {"summary_line": "polished"}}]
    assert calls == [
        {
            "prompt": "method",
            "doc_rows": doc_rows,
            "hits": hits,
            "allow_expensive_llm": True,
            "normalize_ref_copy_ui_meta": reference_ui._normalize_ref_copy_ui_meta,
            "maybe_polish_single_ref_hit_card": reference_ui._maybe_polish_single_ref_hit_card,
            "apply_doc_list_topic_match_hints": reference_ui._apply_doc_list_topic_match_hints,
            "batch_polish_doc_list_ref_hit_cards": reference_ui._batch_polish_doc_list_ref_hit_cards,
            "ref_card_has_llm_copy": reference_ui._ref_card_has_llm_copy,
            "refs_card_polish_max_workers": reference_ui._refs_card_polish_max_workers,
        }
    ]


def test_reference_ui_finalize_doc_list_payload_pack_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_finalize(**kwargs):
        calls.append(kwargs)
        return {"payload_mode": "full", "display_state": "ready"}

    monkeypatch.setattr(doc_list, "_finalize_doc_list_payload_pack", fake_finalize)

    pack_src = {"prompt": "method"}
    hits = [{"text": "summary"}]
    assert reference_ui._finalize_doc_list_payload_pack(
        user_msg_id="42",
        pack_src=pack_src,
        hits=hits,
        guide_active=True,
        guide_source_path_norm="/kb/guide.md",
        guide_source_name_norm="Guide.pdf",
        prompt_cross_paper_refs=True,
        filtered_self_doc_count=1,
        allow_expensive_llm=True,
    ) == {"payload_mode": "full", "display_state": "ready"}
    assert calls == [
        {
            "user_msg_id": "42",
            "pack_src": pack_src,
            "hits": hits,
            "guide_active": True,
            "guide_source_path_norm": "/kb/guide.md",
            "guide_source_name_norm": "Guide.pdf",
            "prompt_cross_paper_refs": True,
            "filtered_self_doc_count": 1,
            "allow_expensive_llm": True,
            "refs_hits_have_llm_copy": reference_ui._refs_hits_have_llm_copy,
            "source_filename": reference_ui._source_filename,
            "attach_pack_display_contract": reference_ui._attach_pack_display_contract,
        }
    ]


def test_reference_ui_build_legacy_doc_list_payload_hits_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_legacy(**kwargs):
        calls.append(kwargs)
        return [{"text": "legacy", "ui_meta": {"summary_line": "legacy"}}]

    monkeypatch.setattr(doc_list, "_build_legacy_doc_list_payload_hits", fake_legacy)

    doc_rows = [{"source_path": "/kb/paper.md"}]
    assert reference_ui._build_legacy_doc_list_payload_hits(
        doc_list=doc_rows,
        prompt="method",
        prefer_zh=True,
    ) == [{"text": "legacy", "ui_meta": {"summary_line": "legacy"}}]
    assert calls == [
        {
            "doc_list": doc_rows,
            "prompt": "method",
            "prefer_zh": True,
            "source_filename": reference_ui._source_filename,
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "compact_reader_open_text": reference_ui._compact_reader_open_text,
            "doc_list_ref_why_line": reference_ui._doc_list_ref_why_line,
            "score_tier": reference_ui._score_tier,
        }
    ]


def test_reference_ui_finalize_legacy_doc_list_payload_pack_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_finalize(**kwargs):
        calls.append(kwargs)
        return {"payload_mode": "full", "display_state": "hidden_by_guide"}

    monkeypatch.setattr(doc_list, "_finalize_legacy_doc_list_payload_pack", fake_finalize)

    pack_src = {"prompt": "method"}
    hits = [{"text": "legacy"}]
    assert reference_ui._finalize_legacy_doc_list_payload_pack(
        user_msg_id="43",
        pack_src=pack_src,
        hits=hits,
        guide_active=True,
        guide_source_path_norm="/kb/guide.md",
        guide_source_name_norm="Guide.pdf",
        prompt_cross_paper_refs=True,
    ) == {"payload_mode": "full", "display_state": "hidden_by_guide"}
    assert calls == [
        {
            "user_msg_id": "43",
            "pack_src": pack_src,
            "hits": hits,
            "guide_active": True,
            "guide_source_path_norm": "/kb/guide.md",
            "guide_source_name_norm": "Guide.pdf",
            "prompt_cross_paper_refs": True,
            "source_filename": reference_ui._source_filename,
            "attach_pack_display_contract": reference_ui._attach_pack_display_contract,
        }
    ]


def test_reference_ui_build_doc_list_refs_payload_uses_doc_list_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return {"payload_mode": "full", "display_state": "ready"}

    monkeypatch.setattr(doc_list, "_build_doc_list_refs_payload", fake_build)

    pack = {"prompt": "method"}
    rows = [{"source_path": "/kb/paper.md"}]
    assert reference_ui.build_doc_list_refs_payload(
        user_msg_id="42",
        pack=pack,
        doc_list=rows,
        allow_expensive_llm=True,
        allow_exact_locate=False,
        apply_copy_polish=False,
        guide_mode=True,
        guide_source_path="/kb/guide.md",
        guide_source_name="Guide.pdf",
    ) == {"payload_mode": "full", "display_state": "ready"}
    assert calls == [
        {
            "user_msg_id": "42",
            "pack": pack,
            "doc_list": rows,
            "allow_expensive_llm": True,
            "allow_exact_locate": False,
            "apply_copy_polish": False,
            "guide_mode": True,
            "guide_source_path": "/kb/guide.md",
            "guide_source_name": "Guide.pdf",
            "prompt_likely_cross_paper_refs": reference_ui._prompt_likely_cross_paper_refs,
            "filter_doc_list_rows_for_guide": reference_ui._filter_doc_list_rows_for_guide,
            "build_doc_list_payload_hits": reference_ui._build_doc_list_payload_hits,
            "polish_doc_list_payload_hits": reference_ui._polish_doc_list_payload_hits,
            "suppress_non_llm_ref_card_copy_hits": reference_ui._suppress_non_llm_ref_card_copy_hits,
            "finalize_doc_list_payload_pack": reference_ui._finalize_doc_list_payload_pack,
            "prefer_zh_ref_card_locale": reference_ui._prefer_zh_ref_card_locale,
            "build_legacy_doc_list_payload_hits": reference_ui._build_legacy_doc_list_payload_hits,
            "finalize_legacy_doc_list_payload_pack": reference_ui._finalize_legacy_doc_list_payload_pack,
        }
    ]


def test_build_doc_list_refs_payload_keeps_cached_library_bibliometrics(monkeypatch, tmp_path) -> None:
    from api import reference_ui

    source_path = r"db\NatPhoton-2019\NatPhoton-2019.en.md"
    pdf_path = tmp_path / "NatPhoton-2019.pdf"
    citation_meta = {
        "title": "Principles and prospects for single-pixel imaging",
        "doi": "10.1038/s41566-018-0300-7",
        "citation_count": 910,
        "citation_source": "OpenAlex",
        "journal_if": 32.9,
        "journal_quartile": "Q1",
    }

    class _LibraryStore:
        def get_citation_meta(self, requested_path):
            assert requested_path == pdf_path
            return citation_meta

    monkeypatch.setattr(reference_ui, "_resolve_pdf_for_source", lambda pdf_root, source: pdf_path)
    monkeypatch.setattr(
        reference_ui,
        "_prefetch_refs_citation_meta",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fast doc-list cards must stay local")),
    )

    out = reference_ui.build_doc_list_refs_payload(
        user_msg_id=42,
        pack={"prompt": "给出单像素成像入门路线图"},
        doc_list=[
            {
                "source_path": source_path,
                "source_name": "NatPhoton-2019.pdf",
                "heading_path": "Abstract",
                "summary_line": "该综述解释了单像素相机的基本原理和主要应用。",
                "primary_evidence": {
                    "source_path": source_path,
                    "source_name": "NatPhoton-2019.pdf",
                    "heading_path": "Abstract",
                    "snippet": "The review explains the principles and applications of single-pixel imaging.",
                    "block_id": "block-1",
                },
            }
        ],
        apply_copy_polish=False,
        pdf_root=tmp_path,
        lib_store=_LibraryStore(),
    )

    hit = list(out.get("hits") or [])[0]
    ui_meta = dict(hit.get("ui_meta") or {})
    assert ui_meta.get("citation_meta") == {
        "title": "Principles and prospects for single-pixel imaging",
        "doi": "10.1038/s41566-018-0300-7",
        "doi_url": "https://doi.org/10.1038/s41566-018-0300-7",
        "citation_count": 910,
        "journal_if": 32.9,
        "journal_quartile": "Q1",
    }
    assert "citation_source" not in ui_meta["citation_meta"]
    assert ui_meta["citation_meta"]["citation_count"] == 910
    assert ui_meta["citation_meta"]["journal_if"] == 32.9
