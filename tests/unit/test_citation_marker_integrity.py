from __future__ import annotations

from api.citation_marker_integrity import reconcile_citation_markers


def test_orphans_are_removed_without_assigning_them_to_another_paper():
    body = "Supported [1](#kb-cite-a). Unsupported [2]. Broken [2](#kb-cite-b)."
    result = reconcile_citation_markers(
        body, [{"num": 1, "anchor": "kb-cite-a"}], source_numbers={1, 2}
    )
    assert "[1](#kb-cite-a)" in result
    assert "[2]" not in result and "kb-cite-b" not in result
    assert "Unsupported" in result and "Broken" in result


def test_citation_reconciliation_preserves_math_code_arrays_and_years():
    body = r"Noise $\sigma\in[0,55]$; `$x[2]$`; values [1,2]; year [2024]; support [2]."
    result = reconcile_citation_markers(body, [], source_numbers={1, 2})
    assert r"$\sigma\in[0,55]$" in result
    assert "`$x[2]$`" in result and "[1,2]" in result and "[2024]" in result
    assert "support [2]" not in result


def test_blocked_cached_packet_cannot_keep_a_link_after_its_details_are_cleared():
    from api.citation_marker_integrity import reconcile_blocked_message

    text = "Claim [2](#kb-cite-old). Orphan [3]. Keep $x[2]$ and `[2]`."
    message = {
        "content": text,
        "copy_markdown": text,
        "meta": {
            "canonical_hit_paths": ["a.md", "b.md"],
            "paper_guide_contracts": {
                "render_packet": {"rendered_body": text, "cite_details": []}
            },
        },
    }
    result = reconcile_blocked_message(message)
    assert "kb-cite-old" not in result["content"]
    assert "Orphan [3]" not in result["content"]
    assert (
        "kb-cite-old"
        not in result["meta"]["paper_guide_contracts"]["render_packet"]["rendered_body"]
    )
    assert "$x[2]$" in result["content"] and "`[2]`" in result["copy_markdown"]


def test_final_packet_cannot_reintroduce_bare_numbers_from_provenance():
    from api.citation_marker_integrity import reconcile_render_packet

    packet = {
        "rendered_body": "Actual claim [2] [3]. Keep $x[2]$.",
        "copy_markdown": "Actual claim [2] [3].",
        "cite_details": [],
        "answer_markdown": "Raw claim [1].",
    }
    result = reconcile_render_packet(packet, local_answer=True)
    assert "[2]" not in result["copy_markdown"] and "[3]" not in result["rendered_body"]
    assert "$x[2]$" in result["rendered_body"]
    assert result["answer_markdown"] == packet["answer_markdown"]


def test_external_numeric_links_are_preserved_but_wrong_local_numbers_are_removed():
    body = "External [2](https://example.org). Wrong [2](#kb-cite-a)."
    result = reconcile_citation_markers(
        body, [{"num": 1, "anchor": "kb-cite-a"}], source_numbers={1, 2}
    )
    assert "[2](https://example.org)" in result
    assert "(#kb-cite-a)" not in result
