from __future__ import annotations

from api.chat_render import enrich_messages_with_reference_render


def _local_ref_pack() -> dict:
    return {
        "hits": [
            {
                "text": "The local paper reports a five decibel improvement.",
                "score": 9.0,
                "meta": {"source_path": "library/paper.md", "heading_path": "Results"},
                "ui_meta": {
                    "reader_open": {
                        "sourcePath": "library/paper.md",
                        "headingPath": "Results",
                        "highlightSnippet": "five decibel improvement",
                    }
                },
            }
        ]
    }


def test_external_answer_cannot_inherit_local_citations_or_reader_actions() -> None:
    messages = [
        {"id": 1, "role": "user", "content": "Explain an unrelated model concept."},
        {
            "id": 2,
            "role": "assistant",
            "content": "This is model background, not evidence from the open paper [1].",
            "meta": {
                "agent_source_summary": {
                    "kind": "external_not_kb",
                    "source_policy_payload": {
                        "kind": "external_not_kb",
                        "uses_local_knowledge_base": False,
                    },
                },
                "provenance": {
                    "source_path": "library/paper.md",
                    "segments": [{"evidence_mode": "direct", "locate_policy": "required"}],
                },
                "paper_guide_contracts": {
                    "primary_evidence": {"sourcePath": "library/paper.md"},
                    "render_packet": {
                        "answer_markdown": "This is model background, not evidence from the open paper [1].",
                        "cite_details": [{"num": 1, "sourcePath": "library/paper.md"}],
                        "reader_open": {"sourcePath": "library/paper.md"},
                        "locate_target": {"sourcePath": "library/paper.md"},
                    },
                },
            },
            "provenance": {
                "source_path": "library/paper.md",
                "segments": [{"evidence_mode": "direct", "locate_policy": "required"}],
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(
        messages,
        {1: _local_ref_pack()},
        conv_id="external-policy",
    )
    answer = rendered[1]

    assert answer.get("cite_details") == []
    assert "provenance" not in answer
    meta = answer["meta"]
    assert "provenance" not in meta
    packet = meta["paper_guide_contracts"]["render_packet"]
    assert packet.get("cite_details") == []
    assert packet.get("reader_open") == {}
    assert packet.get("locate_target") == {}
    assert "primary_evidence" not in meta["paper_guide_contracts"]


def test_failed_quality_answer_cannot_expose_evidence_actions() -> None:
    messages = [
        {"id": 3, "role": "user", "content": "Was this validated clinically?"},
        {
            "id": 4,
            "role": "assistant",
            "content": "Validation scope:\n\nPlease inspect the paper yourself. [1]",
            "meta": {
                "answer_quality": {"minimum_ok": False},
                "agent_source_summary": {"kind": "local_kb"},
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(
        messages,
        {3: _local_ref_pack()},
        conv_id="failed-quality-policy",
    )

    assert rendered[1].get("cite_details") == []
    assert "provenance" not in rendered[1]


def test_incomplete_answer_keeps_independently_bound_local_evidence():
    from api.chat_render import _message_local_evidence_actions_allowed
    quote = "The method trains on single noisy images without clean targets."
    message = {"meta": {"answer_quality": {"minimum_ok": False, "has_hits": True,
        "incomplete_fragment": True, "citation_plan": {"slots": [{
            "preferred_system": "system_a", "candidate_hits": [2], "source_path": "paper.md",
            "evidence_quote": quote, "block_id": "b2", "page_start": 1}]}}}}
    assert _message_local_evidence_actions_allowed(message) is not False


def test_incomplete_local_answer_renders_links_details_and_copy_together(tmp_path, monkeypatch):
    from kb.source_blocks import load_source_blocks
    import ui.refs_renderer as refs_renderer
    # This is a registered-library fixture stored in pytest's temporary folder,
    # rather than a temporary chat upload, which the product excludes on purpose.
    monkeypatch.setattr(refs_renderer, "_is_temp_source_path", lambda _path: False)
    quote = "The method trains on single noisy images without clean targets."
    source = tmp_path / "paper.md"
    source.write_text(f"<!-- kb_page: 1 -->\n## Method\n\n{quote}\n", encoding="utf-8")
    block = next(row for row in load_source_blocks(source) if row.get("kind") == "paragraph")
    path = str(source)
    slot = {"preferred_system": "system_a", "candidate_hits": [1], "source_path": path,
        "evidence_quote": quote, **{k: block[k] for k in ("block_id", "anchor_id", "page_start", "heading_path")}}
    plan = {"budget": {"system_a": 1, "system_b": 0}, "slots": [slot]}
    message = {"id": 2, "role": "assistant", "content": quote[:-1] + " [1];", "meta": {
        "canonical_hit_paths": [path], "answer_quality": {"has_hits": True, "minimum_ok": False,
        "incomplete_fragment": True, "citation_plan": plan}}}
    result = enrich_messages_with_reference_render(
        [{"id": 1, "role": "user", "content": "What training data is used?"}, message],
        {1: {"hits": [{"text": quote, "meta": {"source_path": path, **block}}]}}, conv_id="incomplete-local")[-1]
    assert len(result["cite_details"]) == 1
    assert result["cite_details"][0]["source_path"] == path
    assert "](#kb-cite-" in result["rendered_body"]
    assert "single noisy images" in result["copy_markdown"]
