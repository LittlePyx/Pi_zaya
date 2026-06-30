from __future__ import annotations

from api import reference_reader_open as reader_open


def test_compact_reader_open_text_normalizes_whitespace_and_truncates() -> None:
    assert reader_open._compact_reader_open_text("  alpha\n beta\tgamma  ") == "alpha beta gamma"
    assert reader_open._compact_reader_open_text("abcdef", max_len=3) == "abc..."


def test_pick_reader_open_loc_text_prefers_first_available_field() -> None:
    loc = {"snippet": "  ", "text": "first text", "quote": "later quote"}

    assert reader_open._pick_reader_open_loc_text(loc) == "first text"
    assert reader_open._pick_reader_open_loc_text({"why": "because"}) == "because"
    assert reader_open._pick_reader_open_loc_text("bad") == ""


def test_refs_reader_open_candidate_key_includes_identity_fields() -> None:
    key = reader_open._refs_reader_open_candidate_key(
        {
            "headingPath": "2. Related Work",
            "highlightSnippet": "ADMM reconstruction is discussed.",
            "snippet": "ADMM reconstruction",
            "anchorKind": "Sentence",
            "anchorNumber": 3,
            "blockId": "blk-1",
            "anchorId": "sent-3",
        }
    )

    assert key == "2. related work::admm reconstruction is discussed.::admm reconstruction::sentence::3::blk-1::sent-3"
    assert reader_open._refs_reader_open_candidate_key({}) == ""
    assert reader_open._refs_reader_open_candidate_key("bad") == ""


def test_refs_heading_paths_related_matches_parent_child_paths() -> None:
    assert reader_open._refs_heading_paths_related("2. Related Work", "2. Related Work") is True
    assert reader_open._refs_heading_paths_related("2. Related Work / ADMM", "2. Related Work") is True
    assert reader_open._refs_heading_paths_related("3. Method", "2. Related Work") is False


def test_normalize_refs_reader_heading_path_strips_document_title_prefix() -> None:
    def sanitize(value: str, *, prompt: str, source_path: str) -> str:
        assert prompt == "compare methods"
        assert source_path == "/kb/paper.md"
        return value

    assert reader_open._normalize_refs_reader_heading_path(
        prompt="compare methods",
        source_path="/kb/paper.md",
        heading_path="Paper Title / 2. Methods / ADMM",
        sanitize_heading_path=sanitize,
        looks_like_doc_title_heading=lambda heading, source_path: heading == "Paper Title",
    ) == "2. Methods / ADMM"
    assert reader_open._normalize_refs_reader_heading_path(
        prompt="compare methods",
        source_path="/kb/paper.md",
        heading_path="Front matter / 3. Results / Ablation",
        sanitize_heading_path=sanitize,
        looks_like_doc_title_heading=lambda heading, source_path: False,
    ) == "3. Results / Ablation"


def test_refs_heading_anchor_number_delegates_by_anchor_kind() -> None:
    out_figure = reader_open._refs_heading_anchor_number(
        "figure",
        "Figure 7. Pipeline",
        extract_figure_number=lambda text: 7,
        extract_equation_number=lambda text: 0,
    )
    out_equation = reader_open._refs_heading_anchor_number(
        "equation",
        "Equation 4",
        extract_figure_number=lambda text: 0,
        extract_equation_number=lambda text: 4,
    )
    out_table = reader_open._refs_heading_anchor_number(
        "table",
        "Table 12. Results",
        extract_figure_number=lambda text: 0,
        extract_equation_number=lambda text: 0,
    )

    assert out_figure == 7
    assert out_equation == 4
    assert out_table == 12
    assert reader_open._refs_heading_anchor_number(
        "table",
        "表 3. 消融结果",
        extract_figure_number=lambda text: 0,
        extract_equation_number=lambda text: 0,
    ) == 3
    assert reader_open._refs_heading_anchor_number(
        "definition",
        "Definition 2",
        extract_figure_number=lambda text: 0,
        extract_equation_number=lambda text: 0,
    ) == 0


def test_clean_refs_evidence_snippet_uses_picker_and_fallback_cleaner() -> None:
    calls: list[tuple[str, dict]] = []

    def picker(text: str, **kwargs) -> str:
        calls.append((text, kwargs))
        return "picked snippet"

    assert reader_open._clean_refs_evidence_snippet(
        "  raw evidence text  ",
        prompt="claim",
        source_path="/kb/Paper.pdf",
        display_name="Display Title",
        heading_path="2. Method",
        max_len=120,
        pick_readable_evidence_text=picker,
        clean_evidence_display_text=lambda text, max_len: "cleaned",
    ) == "picked snippet"
    assert calls == [
        (
            "raw evidence text",
            {
                "source": "/kb/Paper.pdf",
                "title": "Display Title",
                "claim": "claim",
                "heading": "2. Method",
                "max_len": 120,
            },
        )
    ]

    assert reader_open._clean_refs_evidence_snippet(
        "fallback text",
        prompt="",
        source_path="/kb/Paper.pdf",
        pick_readable_evidence_text=lambda text, **kwargs: "",
        clean_evidence_display_text=lambda text, max_len: f"{text}:{max_len}",
    ) == "fallback text:360"


def test_build_refs_reader_open_candidate_normalizes_and_cleans_fields() -> None:
    candidate = reader_open._build_refs_reader_open_candidate(
        prompt="claim",
        source_path="/kb/paper.md",
        heading_path="Paper Title / 2. Method",
        snippet="method evidence",
        highlight_snippet="",
        anchor_kind="Figure",
        anchor_number=5,
        sanitize_heading_path=lambda value, **kwargs: value,
        looks_like_doc_title_heading=lambda heading, source_path: heading == "Paper Title",
        pick_readable_evidence_text=lambda text, **kwargs: text.upper(),
        clean_evidence_display_text=lambda text, max_len: f"clean:{text}:{max_len}",
    )

    assert candidate == {
        "headingPath": "2. Method",
        "snippet": "METHOD EVIDENCE",
        "highlightSnippet": "METHOD EVIDENCE",
        "anchorKind": "figure",
        "anchorNumber": 5,
    }
    assert reader_open._build_refs_reader_open_candidate(
        prompt="",
        source_path="",
        heading_path="",
        snippet="",
        highlight_snippet="",
        anchor_kind="",
        anchor_number=0,
        sanitize_heading_path=lambda value, **kwargs: "",
        looks_like_doc_title_heading=lambda heading, source_path: False,
        pick_readable_evidence_text=lambda text, **kwargs: "",
        clean_evidence_display_text=lambda text, max_len: "",
    ) is None


def test_infer_heading_path_for_summary_from_source_blocks_returns_first_match() -> None:
    match_calls: list[dict] = []

    def match_blocks(blocks, **kwargs):
        match_calls.append({"blocks": blocks, **kwargs})
        return [{"block": {"heading_path": "Paper Title / 4. Results"}}]

    heading = reader_open._infer_heading_path_for_summary_from_source_blocks(
        prompt="compare result",
        source_path="/kb/paper.md",
        summary_line="  result\nsummary  ",
        anchor_target_kind="table",
        anchor_target_number=3,
        resolve_source_md_path=lambda source_path: "paper.md",
        load_source_blocks=lambda md_path: [{"text": "block", "md_path": md_path}],
        match_source_blocks=match_blocks,
        sanitize_heading_path=lambda value, **kwargs: value,
        looks_like_doc_title_heading=lambda heading, source_path: heading == "Paper Title",
    )

    assert heading == "4. Results"
    assert match_calls == [
        {
            "blocks": [{"text": "block", "md_path": "paper.md"}],
            "snippet": "result summary",
            "heading_path": "",
            "prefer_kind": "table",
            "target_number": 3,
            "limit": 3,
            "score_floor": 0.24,
        }
    ]


def test_score_refs_exact_surface_applies_positive_shape_signals() -> None:
    score = reader_open._score_refs_exact_surface(
        "This method explains how the reconstruction improves noisy photon evidence in a practical pipeline.",
        prompt="",
        title="",
        block_kind="paragraph",
        anchor_target_kind="",
        looks_bibliographic_source_block_text=lambda text: False,
        looks_title_like_ref_surface=lambda text, title: False,
        looks_like_front_matter_ref_summary=lambda text: False,
        looks_prefixed_heading_shell_ref_summary=lambda text: False,
        looks_surface_like_ref_summary=lambda text: False,
        looks_fragmentary_ref_summary=lambda text: False,
        looks_why_like_ref_summary=lambda text: False,
        looks_formula_heavy_ref_text=lambda text: False,
        prompt_reference_focus_action=lambda prompt: "",
        refs_summary_focus_keyword_hit_count=lambda prompt, text: 0,
        looks_natural_language_ref_summary=lambda text: True,
        has_ref_summary_explainer_signal=lambda text: True,
        has_ref_summary_value_signal=lambda text: True,
        refs_exact_focus_match_count=lambda prompt, text: 0,
        matched_focus_terms_for_ref_card=lambda prompt, surface_text: [],
    )

    assert score == 3.0
    assert reader_open._score_refs_exact_surface(
        "",
        prompt="",
        title="",
        looks_bibliographic_source_block_text=lambda text: False,
        looks_title_like_ref_surface=lambda text, title: False,
        looks_like_front_matter_ref_summary=lambda text: False,
        looks_prefixed_heading_shell_ref_summary=lambda text: False,
        looks_surface_like_ref_summary=lambda text: False,
        looks_fragmentary_ref_summary=lambda text: False,
        looks_why_like_ref_summary=lambda text: False,
        looks_formula_heavy_ref_text=lambda text: False,
        prompt_reference_focus_action=lambda prompt: "",
        refs_summary_focus_keyword_hit_count=lambda prompt, text: 0,
        looks_natural_language_ref_summary=lambda text: False,
        has_ref_summary_explainer_signal=lambda text: False,
        has_ref_summary_value_signal=lambda text: False,
        refs_exact_focus_match_count=lambda prompt, text: 0,
        matched_focus_terms_for_ref_card=lambda prompt, surface_text: [],
    ) == -1000.0


def test_select_reader_open_exact_snippet_prefers_better_block_surface() -> None:
    calls: list[tuple[str, dict]] = []

    def score_surface(text: str, **kwargs) -> float:
        calls.append((text, kwargs))
        return 2.0 if "better" in text else 0.5

    out = reader_open._select_reader_open_exact_snippet(
        "rough seed",
        "better paragraph evidence",
        prompt="compare",
        title="Paper",
        block_kind="paragraph",
        anchor_target_kind="figure",
        score_refs_exact_surface=score_surface,
        looks_focus_prefixed_ref_summary=lambda prompt, seed: False,
        summary_line_needs_polish=lambda **kwargs: False,
    )

    assert out == ("better paragraph evidence", "better paragraph evidence")
    assert calls == [
        (
            "rough seed",
            {"prompt": "compare", "title": "Paper", "block_kind": "", "anchor_target_kind": "figure"},
        ),
        (
            "better paragraph evidence",
            {"prompt": "compare", "title": "Paper", "block_kind": "paragraph", "anchor_target_kind": "figure"},
        ),
    ]


def test_build_refs_exact_candidate_from_block_adds_locators() -> None:
    select_calls: list[dict] = []
    build_calls: list[dict] = []

    def select_snippet(seed_text: str, block_text: str, **kwargs) -> tuple[str, str]:
        select_calls.append({"seed_text": seed_text, "block_text": block_text, **kwargs})
        return "chosen snippet", "chosen highlight"

    def build_candidate(**kwargs) -> dict:
        build_calls.append(kwargs)
        return {"headingPath": kwargs["heading_path"], "anchorKind": kwargs["anchor_kind"]}

    candidate = reader_open._build_refs_exact_candidate_from_block(
        prompt="p",
        source_path="/kb/paper.md",
        title="Paper",
        block={
            "block_id": "blk-1",
            "anchor_id": "sent-2",
            "heading_path": "2. Method",
            "text": "block evidence",
            "kind": "paragraph",
            "number": 9,
        },
        seed_heading_path="fallback heading",
        seed_snippet="seed evidence",
        anchor_kind="figure",
        anchor_number=3,
        select_reader_open_exact_snippet=select_snippet,
        build_refs_reader_open_candidate=build_candidate,
    )

    assert candidate == {"headingPath": "2. Method", "anchorKind": "figure", "blockId": "blk-1", "anchorId": "sent-2"}
    assert select_calls == [
        {
            "seed_text": "seed evidence",
            "block_text": "block evidence",
            "prompt": "p",
            "title": "Paper",
            "block_kind": "paragraph",
            "anchor_target_kind": "figure",
        }
    ]
    assert build_calls == [
        {
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "heading_path": "2. Method",
            "snippet": "chosen snippet",
            "highlight_snippet": "chosen highlight",
            "anchor_kind": "figure",
            "anchor_number": 3,
        }
    ]
    assert reader_open._build_refs_exact_candidate_from_block(
        prompt="",
        source_path="",
        title="",
        block={"text": "missing id"},
        seed_heading_path="",
        seed_snippet="",
        anchor_kind="",
        anchor_number=0,
        select_reader_open_exact_snippet=select_snippet,
        build_refs_reader_open_candidate=build_candidate,
    ) is None


def test_build_preferred_refs_exact_candidate_from_source_summary_validates_source_block() -> None:
    candidate = reader_open._build_preferred_refs_exact_candidate_from_source_summary(
        prompt="p",
        source_path="/kb/paper.md",
        title="Paper",
        summary_line="summary",
        selected_heading_path="2. Method",
        anchor_target_kind="table",
        anchor_target_number=4,
        prompt_aligned_candidate={
            "source_kind": "source_block",
            "block_id": "blk-2",
            "anchor_id": "sent-5",
            "summary": "summary",
            "heading_path": "2. Method",
            "block_kind": "table",
            "block_number": 8,
            "block_text": "block text",
        },
        ref_summary_surfaces_match=lambda left, right: left == right,
        normalize_refs_reader_heading_path=lambda **kwargs: kwargs["heading_path"],
        select_reader_open_exact_snippet=lambda seed_text, block_text, **kwargs: ("snippet", "highlight"),
        build_refs_reader_open_candidate=lambda **kwargs: {"headingPath": kwargs["heading_path"], "anchorNumber": kwargs["anchor_number"]},
    )

    assert candidate == {"headingPath": "2. Method", "anchorNumber": 4, "blockId": "blk-2", "anchorId": "sent-5"}
    assert reader_open._build_preferred_refs_exact_candidate_from_source_summary(
        prompt="p",
        source_path="/kb/paper.md",
        title="Paper",
        summary_line="summary",
        selected_heading_path="2. Method",
        anchor_target_kind="table",
        anchor_target_number=4,
        prompt_aligned_candidate={"source_kind": "source_block", "block_id": "blk-2", "summary": "other"},
        ref_summary_surfaces_match=lambda left, right: False,
        normalize_refs_reader_heading_path=lambda **kwargs: kwargs["heading_path"],
        select_reader_open_exact_snippet=lambda seed_text, block_text, **kwargs: ("", ""),
        build_refs_reader_open_candidate=lambda **kwargs: {},
    ) == {}


def test_resolve_refs_exact_candidates_collects_sorts_and_allows_llm_pick() -> None:
    blocks = [
        {"block_id": "blk-a", "heading_path": "1. Intro", "kind": "paragraph", "text": "intro evidence"},
        {"block_id": "blk-b", "heading_path": "2. Method", "kind": "paragraph", "text": "method evidence"},
    ]
    match_calls: list[dict] = []
    llm_calls: list[dict] = []

    def match_blocks(_blocks, **kwargs):
        match_calls.append(kwargs)
        return [
            {"score": 0.92, "block": blocks[0]},
            {"score": 0.87, "block": blocks[1]},
        ]

    def build_from_block(**kwargs):
        block = kwargs["block"]
        return {
            "headingPath": block["heading_path"],
            "snippet": block["text"],
            "highlightSnippet": block["text"],
            "blockId": block["block_id"],
        }

    def pick_llm(**kwargs) -> int:
        llm_calls.append(kwargs)
        return 2

    out = reader_open._resolve_refs_exact_candidates(
        prompt="where is method evidence?",
        source_path="/kb/paper.md",
        display_name="Paper",
        anchor_target_kind="",
        anchor_target_number=0,
        primary_candidate={"headingPath": "1. Intro", "snippet": "intro"},
        secondary_candidates=[{"headingPath": "2. Method", "snippet": "method"}],
        allow_llm_disambiguation=True,
        resolve_source_md_path=lambda source_path: "paper.md",
        load_source_blocks=lambda md_path: blocks,
        match_source_blocks=match_blocks,
        build_refs_exact_candidate_from_block=build_from_block,
        refs_heading_paths_related=lambda left, right: left == right,
        refs_heading_anchor_number=lambda kind, heading: 0,
        score_refs_exact_surface=lambda text, **kwargs: 0.0,
        refs_exact_focus_match_count=lambda prompt, surface: 0,
        matched_focus_terms_for_ref_card=lambda prompt, surface_text: [],
        should_try_refs_locate_llm=lambda rows: True,
        llm_pick_refs_exact_candidate_index=pick_llm,
    )

    assert [item["blockId"] for item in out[:2]] == ["blk-b", "blk-a"]
    assert match_calls[0] == {
        "snippet": "intro",
        "heading_path": "1. Intro",
        "prefer_kind": "",
        "target_number": 0,
        "limit": 3,
        "score_floor": 0.52,
    }
    assert llm_calls
    assert "blk-b" not in llm_calls[0]["candidates_payload"]
    assert "2. Method" in llm_calls[0]["candidates_payload"]


def test_build_refs_reader_open_payload_keeps_pending_candidates_visible() -> None:
    def build_candidate(**kwargs) -> dict:
        return {
            "headingPath": kwargs["heading_path"],
            "snippet": kwargs["snippet"],
            "highlightSnippet": kwargs["highlight_snippet"],
            "anchorKind": kwargs["anchor_kind"],
            "anchorNumber": kwargs["anchor_number"],
        }

    out = reader_open._build_refs_reader_open_payload(
        meta={
            "ref_pack_state": "pending",
            "ref_locs": [{"heading_path": "2. Method", "snippet": "loc evidence"}],
            "ref_snippets": ["secondary evidence"],
        },
        prompt="p",
        source_path="/kb/paper.md",
        display_name="Paper",
        heading_path="1. Intro",
        heading="",
        summary_line="primary evidence",
        why_line="",
        anchor_target_kind="figure",
        anchor_target_number=2,
        build_refs_reader_open_candidate=build_candidate,
        resolve_refs_exact_candidates=lambda **kwargs: [],
        prompt_requires_explicit_focus_match=lambda prompt: False,
        allow_exact_locate=False,
    )

    assert out["strictLocate"] is False
    assert out["headingPath"] == "1. Intro"
    assert out["initialAltIndex"] == 0
    assert [item["headingPath"] for item in out["visibleAlternatives"]] == ["1. Intro", "2. Method", "1. Intro"]
    assert len(out["alternatives"]) == 2


def test_build_refs_reader_open_payload_uses_exact_candidate_and_locate_target() -> None:
    def build_candidate(**kwargs) -> dict:
        return {
            "headingPath": kwargs["heading_path"],
            "snippet": kwargs["snippet"],
            "highlightSnippet": kwargs["highlight_snippet"],
            "anchorKind": kwargs["anchor_kind"],
            "anchorNumber": kwargs["anchor_number"],
        }

    exact = {
        "headingPath": "2. Method",
        "snippet": "block snippet",
        "highlightSnippet": "block highlight",
        "blockId": "blk-1",
        "anchorId": "sent-1",
        "anchorKind": "table",
        "anchorNumber": 4,
    }
    out = reader_open._build_refs_reader_open_payload(
        meta={},
        prompt="p",
        source_path="/kb/paper.md",
        display_name="Paper",
        heading_path="1. Intro",
        heading="",
        summary_line="primary evidence",
        why_line="",
        anchor_target_kind="table",
        anchor_target_number=4,
        preferred_exact_candidate=exact,
        build_refs_reader_open_candidate=build_candidate,
        resolve_refs_exact_candidates=lambda **kwargs: [],
        prompt_requires_explicit_focus_match=lambda prompt: False,
    )

    assert out["strictLocate"] is True
    assert out["blockId"] == "blk-1"
    assert out["locateTarget"]["hitLevel"] == "block"
    assert out["locateTarget"]["relatedBlockIds"] == ["blk-1"]


def test_build_primary_ref_evidence_payload_maps_primary_and_alternatives() -> None:
    out = reader_open._build_primary_ref_evidence_payload(
        source_path="/kb/paper.md",
        display_name="Paper",
        reader_open={
            "headingPath": "2. Method",
            "snippet": "primary snippet",
            "highlightSnippet": "primary highlight",
            "blockId": "blk-1",
            "anchorId": "sent-1",
            "anchorKind": "table",
            "anchorNumber": 4,
            "strictLocate": True,
            "evidenceAlternatives": [
                {
                    "headingPath": "3. Results",
                    "snippet": "alt snippet",
                    "highlightSnippet": "alt highlight",
                    "blockId": "blk-2",
                    "anchorKind": "figure",
                    "anchorNumber": 2,
                },
            ],
        },
        selection_reason="exact",
        score=0.92,
        prompt="p",
        clean_refs_evidence_snippet=lambda text, **kwargs: f"clean:{text}",
    )

    assert out["source_path"] == "/kb/paper.md"
    assert out["source_name"] == "Paper"
    assert out["block_id"] == "blk-1"
    assert out["highlight_snippet"] == "clean:primary highlight"
    assert out["strict_locate"] is True
    assert out["score"] == 0.92
    assert out["alternatives"][0]["block_id"] == "blk-2"


def test_normalize_primary_ref_evidence_payload_maps_aliases_and_alternatives() -> None:
    def finish(text: str, *, max_len: int) -> str:
        return str(text or "").strip()[:max_len]

    out = reader_open._normalize_primary_ref_evidence_payload(
        {
            "sourcePath": " /kb/paper.md ",
            "sourceName": " Paper ",
            "blockId": " blk-1 ",
            "anchorId": " sent-1 ",
            "headingPath": " 2. Method ",
            "snippet": " primary snippet ",
            "highlightSnippet": "",
            "anchorKind": "Table",
            "anchorNumber": "4",
            "selectionReason": " exact ",
            "strictLocate": "yes",
            "score": "0.75",
            "alternatives": [
                {"blockId": f"alt-{idx}", "snippet": f" alt {idx} "}
                for idx in range(6)
            ],
        },
        finish_evidence_text=finish,
    )

    assert out["source_path"] == "/kb/paper.md"
    assert out["source_name"] == "Paper"
    assert out["block_id"] == "blk-1"
    assert out["anchor_id"] == "sent-1"
    assert out["heading_path"] == "2. Method"
    assert out["snippet"] == "primary snippet"
    assert out["highlight_snippet"] == "primary snippet"
    assert out["anchor_kind"] == "table"
    assert out["anchor_number"] == 4
    assert out["selection_reason"] == "exact"
    assert out["strict_locate"] is True
    assert out["score"] == 0.75
    assert [item["block_id"] for item in out["alternatives"]] == [
        "alt-0",
        "alt-1",
        "alt-2",
        "alt-3",
        "alt-4",
    ]


def test_build_doc_list_reader_open_payload_prefers_primary_evidence() -> None:
    clean_calls: list[dict] = []

    def clean(text: str, **kwargs) -> str:
        clean_calls.append({"text": text, **kwargs})
        return f"clean:{text}"

    out = reader_open._build_doc_list_reader_open_payload(
        source_path="/kb/paper.md",
        source_name="Paper",
        heading_path="1. Intro",
        summary_line="summary fallback",
        primary_evidence={"raw": True},
        reader_open={"headingPath": "old", "snippet": "old snippet"},
        normalize_primary_ref_evidence_payload=lambda raw: {
            "heading_path": "2. Method",
            "snippet": "primary snippet",
            "highlight_snippet": "primary highlight",
            "block_id": "blk-1",
            "anchor_id": "sent-1",
            "anchor_kind": "table",
            "anchor_number": 4,
            "strict_locate": True,
        },
        clean_refs_evidence_snippet=clean,
    )

    assert out["sourcePath"] == "/kb/paper.md"
    assert out["sourceName"] == "Paper"
    assert out["headingPath"] == "2. Method"
    assert out["snippet"] == "clean:primary snippet"
    assert out["highlightSnippet"] == "clean:primary highlight"
    assert out["blockId"] == "blk-1"
    assert out["anchorId"] == "sent-1"
    assert out["anchorKind"] == "table"
    assert out["anchorNumber"] == 4
    assert out["strictLocate"] is True
    assert out["primaryEvidence"]["block_id"] == "blk-1"
    assert clean_calls == [
        {
            "text": "primary snippet",
            "prompt": "",
            "source_path": "/kb/paper.md",
            "display_name": "Paper",
            "heading_path": "2. Method",
            "max_len": 460,
        },
        {
            "text": "primary highlight",
            "prompt": "",
            "source_path": "/kb/paper.md",
            "display_name": "Paper",
            "heading_path": "2. Method",
            "max_len": 460,
        },
    ]


def test_reference_ui_heading_anchor_number_uses_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[tuple[str, str, object, object]] = []

    def fake_anchor_number(
        anchor_kind: str,
        heading_path: str,
        *,
        extract_figure_number,
        extract_equation_number,
    ) -> int:
        calls.append((anchor_kind, heading_path, extract_figure_number, extract_equation_number))
        return 42

    monkeypatch.setattr(reader_open, "_refs_heading_anchor_number", fake_anchor_number)

    assert reference_ui._refs_heading_anchor_number("table", "Table 2. Results") == 42
    assert calls == [
        (
            "table",
            "Table 2. Results",
            reference_ui.extract_figure_number,
            reference_ui.extract_equation_number,
        )
    ]


def test_reference_ui_primary_ref_evidence_uses_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_primary(**kwargs):
        calls.append(kwargs)
        return {"block_id": "blk"}

    monkeypatch.setattr(reader_open, "_build_primary_ref_evidence_payload", fake_primary)

    reader_payload = {"blockId": "blk"}
    assert reference_ui._build_primary_ref_evidence_payload(
        source_path="/kb/paper.md",
        display_name="Paper",
        reader_open=reader_payload,
        selection_reason="exact",
        score=0.5,
        prompt="p",
    ) == {"block_id": "blk"}
    assert calls == [
        {
            "source_path": "/kb/paper.md",
            "display_name": "Paper",
            "reader_open": reader_payload,
            "selection_reason": "exact",
            "score": 0.5,
            "prompt": "p",
            "clean_refs_evidence_snippet": reference_ui._clean_refs_evidence_snippet,
        }
    ]


def test_reference_ui_normalize_primary_ref_evidence_uses_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_normalize(primary_evidence, **kwargs):
        calls.append({"primary_evidence": primary_evidence, **kwargs})
        return {"block_id": "blk"}

    monkeypatch.setattr(reader_open, "_normalize_primary_ref_evidence_payload", fake_normalize)

    raw = {"blockId": "blk"}
    assert reference_ui._normalize_primary_ref_evidence_payload(raw) == {"block_id": "blk"}
    assert calls == [
        {
            "primary_evidence": raw,
            "finish_evidence_text": reference_ui._finish_evidence_text,
        }
    ]


def test_reference_ui_doc_list_reader_open_payload_uses_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_payload(**kwargs):
        calls.append(kwargs)
        return {"sourcePath": kwargs["source_path"]}

    monkeypatch.setattr(reader_open, "_build_doc_list_reader_open_payload", fake_payload)

    primary = {"block_id": "blk"}
    reader_payload = {"headingPath": "1. Intro"}
    assert reference_ui._build_doc_list_reader_open_payload(
        source_path="/kb/paper.md",
        source_name="Paper",
        heading_path="1. Intro",
        summary_line="summary",
        primary_evidence=primary,
        reader_open=reader_payload,
    ) == {"sourcePath": "/kb/paper.md"}
    assert calls == [
        {
            "source_path": "/kb/paper.md",
            "source_name": "Paper",
            "heading_path": "1. Intro",
            "summary_line": "summary",
            "primary_evidence": primary,
            "reader_open": reader_payload,
            "normalize_primary_ref_evidence_payload": reference_ui._normalize_primary_ref_evidence_payload,
            "clean_refs_evidence_snippet": reference_ui._clean_refs_evidence_snippet,
        }
    ]


def test_reference_ui_reader_open_payload_uses_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_payload(**kwargs):
        calls.append(kwargs)
        return {"sourcePath": kwargs["source_path"]}

    monkeypatch.setattr(reader_open, "_build_refs_reader_open_payload", fake_payload)

    exact = {"blockId": "blk"}
    assert reference_ui._build_refs_reader_open_payload(
        meta={"ref_pack_state": "ready"},
        prompt="p",
        source_path="/kb/paper.md",
        display_name="Paper",
        heading_path="1. Intro",
        heading="Intro",
        summary_line="summary",
        why_line="why",
        anchor_target_kind="figure",
        anchor_target_number=2,
        preferred_exact_candidate=exact,
        allow_llm_disambiguation=False,
        allow_exact_locate=False,
    ) == {"sourcePath": "/kb/paper.md"}
    assert calls == [
        {
            "meta": {"ref_pack_state": "ready"},
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "display_name": "Paper",
            "heading_path": "1. Intro",
            "heading": "Intro",
            "summary_line": "summary",
            "why_line": "why",
            "anchor_target_kind": "figure",
            "anchor_target_number": 2,
            "build_refs_reader_open_candidate": reference_ui._build_refs_reader_open_candidate,
            "resolve_refs_exact_candidates": reference_ui._resolve_refs_exact_candidates,
            "prompt_requires_explicit_focus_match": reference_ui._prompt_requires_explicit_focus_match,
            "preferred_exact_candidate": exact,
            "allow_llm_disambiguation": False,
            "allow_exact_locate": False,
        }
    ]


def test_reference_ui_resolve_refs_exact_candidates_uses_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_resolve(**kwargs):
        calls.append(kwargs)
        return [{"blockId": "blk"}]

    monkeypatch.setattr(reader_open, "_resolve_refs_exact_candidates", fake_resolve)

    primary = {"headingPath": "1. Intro"}
    secondary = [{"headingPath": "2. Method"}]
    assert reference_ui._resolve_refs_exact_candidates(
        prompt="p",
        source_path="/kb/paper.md",
        display_name="Paper",
        anchor_target_kind="figure",
        anchor_target_number=2,
        primary_candidate=primary,
        secondary_candidates=secondary,
        allow_llm_disambiguation=False,
    ) == [{"blockId": "blk"}]
    assert calls == [
        {
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "display_name": "Paper",
            "anchor_target_kind": "figure",
            "anchor_target_number": 2,
            "primary_candidate": primary,
            "secondary_candidates": secondary,
            "allow_llm_disambiguation": False,
            "resolve_source_md_path": reference_ui._resolve_source_md_path,
            "load_source_blocks": reference_ui.load_source_blocks,
            "match_source_blocks": reference_ui.match_source_blocks,
            "build_refs_exact_candidate_from_block": reference_ui._build_refs_exact_candidate_from_block,
            "refs_heading_paths_related": reference_ui._refs_heading_paths_related,
            "refs_heading_anchor_number": reference_ui._refs_heading_anchor_number,
            "score_refs_exact_surface": reference_ui._score_refs_exact_surface,
            "refs_exact_focus_match_count": reference_ui._refs_exact_focus_match_count,
            "matched_focus_terms_for_ref_card": reference_ui._matched_focus_terms_for_ref_card,
            "should_try_refs_locate_llm": reference_ui._should_try_refs_locate_llm,
            "llm_pick_refs_exact_candidate_index": reference_ui._llm_pick_refs_exact_candidate_index,
        }
    ]


def test_reference_ui_exact_candidate_builders_use_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    block_calls: list[dict] = []
    preferred_calls: list[dict] = []

    def fake_block(**kwargs):
        block_calls.append(kwargs)
        return {"blockId": "blk"}

    def fake_preferred(**kwargs):
        preferred_calls.append(kwargs)
        return {"blockId": "pref"}

    monkeypatch.setattr(reader_open, "_build_refs_exact_candidate_from_block", fake_block)
    monkeypatch.setattr(reader_open, "_build_preferred_refs_exact_candidate_from_source_summary", fake_preferred)

    assert reference_ui._build_refs_exact_candidate_from_block(
        prompt="p",
        source_path="/kb/paper.md",
        title="Paper",
        block={"block_id": "blk"},
        seed_heading_path="2. Method",
        seed_snippet="seed",
        anchor_kind="figure",
        anchor_number=2,
    ) == {"blockId": "blk"}
    assert block_calls == [
        {
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "title": "Paper",
            "block": {"block_id": "blk"},
            "seed_heading_path": "2. Method",
            "seed_snippet": "seed",
            "anchor_kind": "figure",
            "anchor_number": 2,
            "select_reader_open_exact_snippet": reference_ui._select_reader_open_exact_snippet,
            "build_refs_reader_open_candidate": reference_ui._build_refs_reader_open_candidate,
        }
    ]

    prompt_candidate = {"source_kind": "source_block", "block_id": "pref"}
    assert reference_ui._build_preferred_refs_exact_candidate_from_source_summary(
        prompt="p",
        source_path="/kb/paper.md",
        title="Paper",
        summary_line="summary",
        selected_heading_path="2. Method",
        anchor_target_kind="table",
        anchor_target_number=3,
        prompt_aligned_candidate=prompt_candidate,
    ) == {"blockId": "pref"}
    assert preferred_calls == [
        {
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "title": "Paper",
            "summary_line": "summary",
            "selected_heading_path": "2. Method",
            "anchor_target_kind": "table",
            "anchor_target_number": 3,
            "prompt_aligned_candidate": prompt_candidate,
            "ref_summary_surfaces_match": reference_ui._ref_summary_surfaces_match,
            "normalize_refs_reader_heading_path": reference_ui._normalize_refs_reader_heading_path,
            "select_reader_open_exact_snippet": reference_ui._select_reader_open_exact_snippet,
            "build_refs_reader_open_candidate": reference_ui._build_refs_reader_open_candidate,
        }
    ]


def test_reference_ui_exact_surface_and_snippet_selection_use_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    score_calls: list[dict] = []
    select_calls: list[dict] = []

    def fake_score(text: str, **kwargs) -> float:
        score_calls.append({"text": text, **kwargs})
        return 7.5

    def fake_select(seed_text: str, block_text: str, **kwargs) -> tuple[str, str]:
        select_calls.append({"seed_text": seed_text, "block_text": block_text, **kwargs})
        return "chosen", "chosen"

    monkeypatch.setattr(reader_open, "_score_refs_exact_surface", fake_score)
    monkeypatch.setattr(reader_open, "_select_reader_open_exact_snippet", fake_select)

    assert reference_ui._score_refs_exact_surface(
        "surface",
        prompt="p",
        title="Paper",
        block_kind="paragraph",
        anchor_target_kind="table",
    ) == 7.5
    assert score_calls == [
        {
            "text": "surface",
            "prompt": "p",
            "title": "Paper",
            "block_kind": "paragraph",
            "anchor_target_kind": "table",
            "looks_bibliographic_source_block_text": reference_ui._looks_bibliographic_source_block_text,
            "looks_title_like_ref_surface": reference_ui._looks_title_like_ref_surface,
            "looks_like_front_matter_ref_summary": reference_ui._looks_like_front_matter_ref_summary,
            "looks_prefixed_heading_shell_ref_summary": reference_ui._looks_prefixed_heading_shell_ref_summary,
            "looks_surface_like_ref_summary": reference_ui._looks_surface_like_ref_summary,
            "looks_fragmentary_ref_summary": reference_ui._looks_fragmentary_ref_summary,
            "looks_why_like_ref_summary": reference_ui._looks_why_like_ref_summary,
            "looks_formula_heavy_ref_text": reference_ui._looks_formula_heavy_ref_text,
            "prompt_reference_focus_action": reference_ui._shared_prompt_reference_focus_action,
            "refs_summary_focus_keyword_hit_count": reference_ui._refs_summary_focus_keyword_hit_count,
            "looks_natural_language_ref_summary": reference_ui._looks_natural_language_ref_summary,
            "has_ref_summary_explainer_signal": reference_ui._has_ref_summary_explainer_signal,
            "has_ref_summary_value_signal": reference_ui._has_ref_summary_value_signal,
            "refs_exact_focus_match_count": reference_ui._refs_exact_focus_match_count,
            "matched_focus_terms_for_ref_card": reference_ui._matched_focus_terms_for_ref_card,
        }
    ]

    assert reference_ui._select_reader_open_exact_snippet(
        "seed",
        "block",
        prompt="p",
        title="Paper",
        block_kind="paragraph",
        anchor_target_kind="table",
    ) == ("chosen", "chosen")
    assert select_calls == [
        {
            "seed_text": "seed",
            "block_text": "block",
            "prompt": "p",
            "title": "Paper",
            "block_kind": "paragraph",
            "anchor_target_kind": "table",
            "score_refs_exact_surface": reference_ui._score_refs_exact_surface,
            "looks_focus_prefixed_ref_summary": reference_ui._looks_focus_prefixed_ref_summary,
            "summary_line_needs_polish": reference_ui._summary_line_needs_polish,
        }
    ]


def test_reference_ui_reader_heading_and_snippet_use_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    heading_calls: list[dict] = []
    snippet_calls: list[dict] = []

    def fake_heading(**kwargs) -> str:
        heading_calls.append(kwargs)
        return "2. Methods"

    def fake_snippet(raw: str, **kwargs) -> str:
        snippet_calls.append({"raw": raw, **kwargs})
        return "clean snippet"

    monkeypatch.setattr(reader_open, "_normalize_refs_reader_heading_path", fake_heading)
    monkeypatch.setattr(reader_open, "_clean_refs_evidence_snippet", fake_snippet)

    assert reference_ui._normalize_refs_reader_heading_path(
        prompt="p",
        source_path="/kb/paper.md",
        heading_path="Paper / 2. Methods",
    ) == "2. Methods"
    assert heading_calls == [
        {
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "heading_path": "Paper / 2. Methods",
            "sanitize_heading_path": reference_ui._sanitize_heading_path_ui,
            "looks_like_doc_title_heading": reference_ui._looks_like_doc_title_heading_ui,
        }
    ]

    assert reference_ui._clean_refs_evidence_snippet(
        "raw",
        prompt="p",
        source_path="/kb/paper.md",
        display_name="Paper",
        heading_path="2. Methods",
        max_len=80,
    ) == "clean snippet"
    assert snippet_calls == [
        {
            "raw": "raw",
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "display_name": "Paper",
            "heading_path": "2. Methods",
            "max_len": 80,
            "pick_readable_evidence_text": reference_ui._pick_readable_evidence_text,
            "clean_evidence_display_text": reference_ui._clean_evidence_display_text,
        }
    ]


def test_reference_ui_candidate_and_summary_heading_use_reader_open_module(monkeypatch) -> None:
    from api import reference_ui

    candidate_calls: list[dict] = []
    infer_calls: list[dict] = []

    def fake_candidate(**kwargs):
        candidate_calls.append(kwargs)
        return {"headingPath": "2. Methods"}

    def fake_infer(**kwargs) -> str:
        infer_calls.append(kwargs)
        return "4. Results"

    monkeypatch.setattr(reader_open, "_build_refs_reader_open_candidate", fake_candidate)
    monkeypatch.setattr(reader_open, "_infer_heading_path_for_summary_from_source_blocks", fake_infer)

    assert reference_ui._build_refs_reader_open_candidate(
        prompt="p",
        source_path="/kb/paper.md",
        heading_path="2. Methods",
        snippet="snippet",
        highlight_snippet="highlight",
        anchor_kind="figure",
        anchor_number=2,
    ) == {"headingPath": "2. Methods"}
    assert candidate_calls == [
        {
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "heading_path": "2. Methods",
            "snippet": "snippet",
            "highlight_snippet": "highlight",
            "anchor_kind": "figure",
            "anchor_number": 2,
            "sanitize_heading_path": reference_ui._sanitize_heading_path_ui,
            "looks_like_doc_title_heading": reference_ui._looks_like_doc_title_heading_ui,
            "pick_readable_evidence_text": reference_ui._pick_readable_evidence_text,
            "clean_evidence_display_text": reference_ui._clean_evidence_display_text,
        }
    ]

    assert reference_ui._infer_heading_path_for_summary_from_source_blocks(
        prompt="p",
        source_path="/kb/paper.md",
        summary_line="summary",
        anchor_target_kind="table",
        anchor_target_number=3,
    ) == "4. Results"
    assert infer_calls == [
        {
            "prompt": "p",
            "source_path": "/kb/paper.md",
            "summary_line": "summary",
            "anchor_target_kind": "table",
            "anchor_target_number": 3,
            "resolve_source_md_path": reference_ui._resolve_source_md_path,
            "load_source_blocks": reference_ui.load_source_blocks,
            "match_source_blocks": reference_ui.match_source_blocks,
            "sanitize_heading_path": reference_ui._sanitize_heading_path_ui,
            "looks_like_doc_title_heading": reference_ui._looks_like_doc_title_heading_ui,
        }
    ]
