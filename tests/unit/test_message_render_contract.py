from __future__ import annotations

import hashlib

from api.message_render_contract import (
    build_render_cache_payload,
    content_has_linkable_answer_citations,
    normalize_render_cache_payload,
    project_render_packet_to_record,
    render_payload_is_degraded_for_citations,
    render_payload_is_missing_planned_system_a,
    strip_legacy_render_fields,
)


def test_render_payload_rejects_missing_planned_system_a() -> None:
    plan = {
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "paper.en.md",
                "evidence_quote": "The paper reports faster reconstruction.",
            }
        ],
    }

    assert render_payload_is_missing_planned_system_a(
        {"rendered_body": "Faster reconstruction.", "cite_details": []},
        citation_plan=plan,
    )
    assert not render_payload_is_missing_planned_system_a(
        {
            "cite_details": [
                {"citation_route": "system_a", "source_path": "paper.en.md"}
            ]
        },
        citation_plan=plan,
    )


def test_render_payload_rejects_system_a_bound_to_wrong_passage() -> None:
    plan = {
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": r"F:\db\Simple Baselines\Simple Baselines.en.md",
                "heading_path": "5 Experiments / 5.2 Applications",
                "evidence_quote": (
                    "Table 6. SIDD PSNR: Restormer = 40.02; "
                    "Baseline ours = 40.30; NAFNet ours = 40.30"
                ),
            }
        ],
    }
    wrong = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/Simple Baselines/Simple Baselines.en.md",
                "heading_path": "5 Experiments / 5.1 Ablations",
                "evidence_quote": "In PSNR, LN brings 0.46 dB and 3.39 dB on SIDD and GoPro.",
                "answer_claim": "Baseline and NAFNet tie on SIDD at PSNR 40.30.",
            }
        ]
    }
    incomplete = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/Simple Baselines/Simple Baselines.en.md",
                "heading_path": "5 Experiments / 5.2 Applications",
                "evidence_quote": "The table shows SIDD PSNR results: Baseline ours = 40.30.",
                "answer_claim": "Baseline and NAFNet tie on SIDD at PSNR 40.30.",
            }
        ]
    }
    correct = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/Simple Baselines/Simple Baselines.en.md",
                "heading_path": "5 Experiments / 5.2 Applications",
                "evidence_quote": (
                    "The table shows SIDD PSNR results: "
                    "Baseline ours = 40.30, NAFNet ours = 40.30."
                ),
                "answer_claim": "Baseline and NAFNet tie on SIDD at PSNR 40.30.",
            }
        ]
    }

    assert render_payload_is_missing_planned_system_a(wrong, citation_plan=plan)
    assert render_payload_is_missing_planned_system_a(incomplete, citation_plan=plan)
    assert not render_payload_is_missing_planned_system_a(correct, citation_plan=plan)


def test_render_payload_rejects_multi_paper_cache_with_one_weak_card() -> None:
    plan = {
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": f"F:/repo/db/paper-{idx}/paper-{idx}.en.md",
                "heading_path": heading,
                "evidence_quote": evidence,
            }
            for idx, heading, evidence in (
                (1, "Abstract", "Deep learning improves reconstruction quality and speed."),
                (2, "Acquisition", "Compressed sensing recovers images from fewer measurements."),
                (3, "Introduction", "HSI uses Hadamard patterns while FSI uses Fourier patterns."),
            )
        ],
    }
    stale = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/paper-1/paper-1.en.md",
                "heading_path": "Abstract",
                "evidence_quote": "Deep learning improves reconstruction quality and speed.",
            },
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/paper-2/paper-2.en.md",
                "heading_path": "Acquisition",
                "evidence_quote": "Compressed sensing recovers images from fewer measurements.",
            },
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/paper-3/paper-3.en.md",
                "heading_path": "Experiments",
                "evidence_quote": "The target uses 4 x 4 pixel binning.",
            },
        ]
    }
    repaired = {
        "cite_details": [
            *stale["cite_details"][:2],
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/paper-3/paper-3.en.md",
                "heading_path": "Introduction",
                "evidence_quote": "HSI uses Hadamard patterns while FSI uses Fourier patterns.",
            },
        ]
    }

    assert render_payload_is_missing_planned_system_a(stale, citation_plan=plan)
    assert not render_payload_is_missing_planned_system_a(repaired, citation_plan=plan)


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
