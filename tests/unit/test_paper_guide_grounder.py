from pathlib import Path

import kb.paper_guide.grounder as grounder


def test_ground_paper_guide_answer_support_runs_inject_then_resolve(monkeypatch):
    calls: list[tuple[str, str]] = []

    def _fake_inject(answer: str, **_kwargs) -> str:
        calls.append(("inject", answer))
        return f"{answer} [[SUPPORT:DOC-1]]"

    def _fake_resolve(answer: str, **_kwargs):
        calls.append(("resolve", answer))
        return "cleaned answer", [{"block_id": "blk_1"}]

    monkeypatch.setattr(grounder, "_inject_paper_guide_support_markers", _fake_inject)
    monkeypatch.setattr(grounder, "_resolve_paper_guide_support_markers", _fake_resolve)

    out, support_resolution = grounder._ground_paper_guide_answer_support(
        "base answer",
        support_slots=[{"support_id": "DOC-1"}],
        prompt_family="method",
        db_dir=Path("db"),
    )

    assert calls == [
        ("inject", "base answer"),
        ("resolve", "base answer [[SUPPORT:DOC-1]]"),
    ]
    assert out == "cleaned answer"
    assert support_resolution == [{"block_id": "blk_1"}]


def test_resolve_paper_guide_panel_clause_snippet_prefers_block_lookup_clause():
    seg = {
        "claim_type": "figure_panel",
        "support_slot_panel_letters": ["f"],
        "primary_block_id": "blk_cap",
    }
    block_lookup = {
        "blk_cap": {
            "raw_text": "(e) baseline reconstruction; (f) methane imaging using SPC$^{15}$; (g) final output",
        }
    }

    snippet = grounder._resolve_paper_guide_panel_clause_snippet(
        seg,
        block_lookup=block_lookup,
        md_path="paper.en.md",
    )

    assert snippet == "(f) methane imaging using SPC$^{15}$"


def test_resolve_paper_guide_panel_clause_snippet_falls_back_to_figure_index(monkeypatch):
    seg = {
        "claim_type": "figure_panel",
        "support_slot_panel_letters": ["b"],
        "support_slot_figure_number": 3,
    }
    monkeypatch.setattr(
        grounder,
        "load_paper_guide_figure_index",
        lambda _md_path: [
            {
                "paper_figure_number": 3,
                "caption": "(a) input; (b) aligned output; (c) residual",
            }
        ],
    )

    snippet = grounder._resolve_paper_guide_panel_clause_snippet(
        seg,
        block_lookup={},
        md_path="paper.en.md",
    )

    assert snippet == "(b) aligned output"


def test_build_paper_guide_segment_locate_target_and_reader_open_use_panel_snippet():
    seg = {
        "segment_id": "seg-1",
        "primary_heading_path": "Results / Figure 3",
        "evidence_quote": "full figure caption",
        "anchor_text": "full figure caption",
        "text": "fallback text",
        "hit_level": "exact",
        "primary_block_id": "blk_cap",
        "primary_anchor_id": "p_00068",
        "anchor_kind": "figure",
        "support_slot_figure_number": 3,
        "claim_type": "figure_panel",
        "locate_policy": "required",
        "locate_surface_policy": "primary",
        "snippet_aliases": ["Figure 3"],
        "related_block_ids": ["blk_cap", "blk_fig"],
    }

    locate_target = grounder._build_paper_guide_segment_locate_target(
        seg,
        panel_clause_snippet="(f) methane imaging using SPC$^{15}$",
    )
    reader_open = grounder._build_paper_guide_segment_reader_open(
        seg,
        source_path="paper.pdf",
        source_name="paper.pdf",
        locate_target=locate_target,
        alternative_candidates=[{"blockId": "blk_fig"}],
        claim_group={"id": "cg-1"},
    )

    assert locate_target["snippet"] == "(f) methane imaging using SPC$^{15}$"
    assert locate_target["highlightSnippet"] == "(f) methane imaging using SPC$^{15}$"
    assert locate_target["anchorText"] == "(f) methane imaging using SPC$^{15}$"
    assert locate_target["anchorNumber"] == 3
    assert locate_target["relatedBlockIds"] == ["blk_cap", "blk_fig"]
    assert reader_open["blockId"] == "blk_cap"
    assert reader_open["anchorId"] == "p_00068"
    assert reader_open["anchorKind"] == "figure"
    assert reader_open["anchorNumber"] == 3
    assert reader_open["strictLocate"] is True
    assert reader_open["locateTarget"] == locate_target
    assert reader_open["claimGroup"] == {"id": "cg-1"}
