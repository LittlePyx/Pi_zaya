from __future__ import annotations

import hashlib

from api.message_render_contract import (
    build_render_cache_payload,
    content_has_linkable_answer_citations,
    normalize_render_cache_payload,
    project_render_packet_to_record,
    render_payload_is_degraded_for_citations,
    strip_legacy_render_fields,
)


def test_normalize_cache_falls_back_to_render_packet_fields():
    cache = {
        "schema": 5,
        "cache_key": "abc",
        "render_packet": {
            "rendered_content": "Answer [1](#kb-cite-demo-1).",
            "copy_text": "Answer [1].",
            "cite_details": [{"num": 1, "anchor": "kb-cite-demo-1"}],
        },
    }

    payload = normalize_render_cache_payload(cache, schema=5, expected_key="abc")

    assert payload is not None
    assert payload.rendered_content == "Answer [1](#kb-cite-demo-1)."
    assert payload.copy_text == "Answer [1]."
    assert payload.cite_details == [{"num": 1, "anchor": "kb-cite-demo-1"}]


def test_degraded_numeric_cache_requires_rendered_links_when_hits_are_linkable():
    hits = [
        {"meta": {"source_path": "paper-one.md"}},
        {"meta": {"source_path": "paper-two.md"}},
    ]
    cache = build_render_cache_payload(
        schema=5,
        cache_key="abc",
        notice="",
        rendered_body="Deep learning improves SPI [1].",
        rendered_content="Deep learning improves SPI [1].",
        copy_markdown="Deep learning improves SPI [1].",
        copy_text="Deep learning improves SPI [1].",
        cite_details=[],
        refs_user_msg_id=10,
        render_packet={},
    )

    assert content_has_linkable_answer_citations("Deep learning improves SPI [1].", hits)
    assert render_payload_is_degraded_for_citations(
        cache,
        raw_content="Deep learning improves SPI [1].",
        hits=hits,
    )


def test_degraded_structured_cache_requires_rendered_links_when_hits_are_linkable():
    source_path = "paper-one.md"
    sid = "s" + hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]
    hits = [{"meta": {"source_path": source_path}}]
    raw = f"Prior work is cited as [[CITE:{sid}:35]]."
    cache = build_render_cache_payload(
        schema=5,
        cache_key="abc",
        notice="",
        rendered_body="Prior work is cited as .",
        rendered_content="Prior work is cited as .",
        copy_markdown="Prior work is cited as .",
        copy_text="Prior work is cited as .",
        cite_details=[],
        refs_user_msg_id=10,
        render_packet={},
    )

    assert content_has_linkable_answer_citations(raw, hits)
    assert render_payload_is_degraded_for_citations(
        cache,
        raw_content=raw,
        hits=hits,
    )


def test_invalid_structured_cite_sid_does_not_make_cache_degraded():
    hits = [{"meta": {"source_path": "paper-one.md"}}]
    raw = "Prior work is cited as [[CITE:sdeadbeef:35]]."
    cache = build_render_cache_payload(
        schema=5,
        cache_key="abc",
        notice="",
        rendered_body="Prior work is cited as .",
        rendered_content="Prior work is cited as .",
        copy_markdown="Prior work is cited as .",
        copy_text="Prior work is cited as .",
        cite_details=[],
        refs_user_msg_id=10,
        render_packet={},
    )

    assert not content_has_linkable_answer_citations(raw, hits)
    assert not render_payload_is_degraded_for_citations(
        cache,
        raw_content=raw,
        hits=hits,
    )


def test_render_packet_projection_and_legacy_strip_share_contract_fields():
    rec = {"id": 2, "content": "Answer [[CITE:s1234abcd:1]]."}
    packet = {
        "notice": "",
        "rendered_body": "Answer [1](#kb-cite-demo-1).",
        "rendered_content": "Answer [1](#kb-cite-demo-1).",
        "copy_markdown": "Answer [1].",
        "copy_text": "Answer [1].",
        "cite_details": [{"num": 1, "anchor": "kb-cite-demo-1"}],
    }

    assert project_render_packet_to_record(rec, packet)
    assert rec["rendered_body"] == "Answer [1](#kb-cite-demo-1)."
    assert rec["cite_details"] == [{"num": 1, "anchor": "kb-cite-demo-1"}]

    strip_legacy_render_fields(rec)

    assert "content" in rec
    assert "rendered_body" not in rec
    assert rec["cite_details"] == [{"num": 1, "anchor": "kb-cite-demo-1"}]
