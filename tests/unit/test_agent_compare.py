from kb.agent.tools import compare_papers


def test_compare_papers_returns_structured_source_specific_notes():
    hits = [
        {
            "text": "The method uses a transformer network with dense feature fusion.",
            "score": 8.0,
            "meta": {
                "source_name": "Paper A",
                "source_path": "paper-a.md",
                "heading_path": "Methods / Network",
            },
        },
        {
            "text": "However, the approach remains limited by training data coverage.",
            "score": 5.0,
            "meta": {
                "source_name": "Paper A",
                "source_path": "paper-a.md",
                "heading_path": "Discussion / Limitations",
            },
        },
        {
            "text": "The approach uses an optical reconstruction pipeline.",
            "score": 7.0,
            "meta": {
                "source_name": "Paper B",
                "source_path": "paper-b.md",
                "heading_path": "Approach",
            },
        },
    ]

    result = compare_papers("Compare transformer method limitations", hits)
    comparisons = result["comparisons"]

    assert len(comparisons) == 2
    first = comparisons[0]
    assert {
        "paper",
        "source_name",
        "source_path",
        "method",
        "evidence",
        "limitation",
        "relation_to_question",
        "supporting_headings",
        "hit_count",
    }.issubset(first)
    assert first["paper"] == "Paper A"
    assert "transformer network" in first["method"]
    assert "limited by training data" in first["limitation"]
    assert first["evidence"][0]["heading_path"] == "Methods / Network"
    assert first["hit_count"] == 2
