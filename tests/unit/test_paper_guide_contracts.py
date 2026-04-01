from kb.paper_guide_contracts import (
    _build_paper_guide_support_resolution,
    _normalize_paper_guide_support_slot,
)


def test_build_paper_guide_support_resolution_normalizes_lists():
    out = _build_paper_guide_support_resolution(
        doc_idx="2",
        support_id=" DOC-1 ",
        candidate_refs=["4", 4, "59", "bad", 0],
        ref_spans=[
            {"text": " Duarte et al. [4] ", "nums": ["4", 4], "scope": "same_clause"},
            {"text": "Duarte et al. [4]", "nums": [4], "scope": "same_clause"},
            {"text": "compressive sensing [59]", "nums": ["59"], "scope": "same_sentence"},
        ],
        figure_number="3",
        box_number="1",
        panel_letters=["F", "f", "g", ""],
        line_index="3",
    )

    assert out["doc_idx"] == 2
    assert out["support_id"] == "DOC-1"
    assert out["candidate_refs"] == [4, 59]
    assert out["figure_number"] == 3
    assert out["box_number"] == 1
    assert out["panel_letters"] == ["f", "g"]
    assert out["line_index"] == 3
    assert out["ref_spans"] == [
        {"text": "Duarte et al. [4]", "nums": [4], "scope": "same_clause"},
        {"text": "compressive sensing [59]", "nums": [59], "scope": "same_sentence"},
    ]


def test_normalize_paper_guide_support_slot_preserves_extra_keys():
    out = _normalize_paper_guide_support_slot(
        {
            "doc_idx": "1",
            "source_path": " demo.md ",
            "candidate_refs": ["35", 35, 0],
            "figure_number": "3",
            "box_number": "1",
            "panel_letters": ["F", "f"],
            "deepread_texts": ["  alpha  ", "", "beta"],
            "custom_flag": True,
        }
    )

    assert out["doc_idx"] == 1
    assert out["source_path"] == "demo.md"
    assert out["candidate_refs"] == [35]
    assert out["figure_number"] == 3
    assert out["box_number"] == 1
    assert out["panel_letters"] == ["f"]
    assert out["deepread_texts"] == ["alpha", "beta"]
    assert out["custom_flag"] is True
