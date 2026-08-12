from __future__ import annotations

import hashlib

from api.message_render_contract import (
    build_render_cache_payload,
    content_has_linkable_answer_citations,
    normalize_render_cache_payload,
    project_render_packet_to_record,
    render_payload_is_degraded_for_citations,
    render_payload_has_citation_links,
    render_payload_is_missing_planned_system_a,
    render_payload_is_missing_planned_system_b,
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


def test_render_payload_rejects_missing_planned_system_b() -> None:
    plan = {
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": "paper.en.md",
                "candidate_refs": [4],
                "candidate_cite_examples": ["[[CITE:s1234abcd:4]]"],
            }
        ],
    }

    assert render_payload_is_missing_planned_system_b(
        {
            "cite_details": [
                {"citation_route": "system_a", "source_path": "paper.en.md"}
            ]
        },
        citation_plan=plan,
    )
    assert not render_payload_is_missing_planned_system_b(
        {
            "render_packet": {
                "cite_details": [
                    {
                        "is_inpaper": True,
                        "source_path": "paper.en.md",
                        "num": 4,
                        "anchor": "kb-cite-upstream-4",
                    }
                ]
            }
        },
        citation_plan=plan,
    )


def test_render_payload_rejects_system_b_card_from_wrong_source_or_ref() -> None:
    plan = {
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": r"db\paper-a\paper-a.en.md",
                "candidate_refs": [4],
                "candidate_cite_examples": ["[[CITE:s1234abcd:4]]"],
            }
        ],
    }

    assert render_payload_is_missing_planned_system_b(
        {
            "cite_details": [
                {
                    "citation_route": "system_b",
                    "source_path": r"db\paper-b\paper-b.en.md",
                    "ref_num": 4,
                    "anchor": "wrong-source",
                }
            ]
        },
        citation_plan=plan,
    )
    assert render_payload_is_missing_planned_system_b(
        {
            "cite_details": [
                {
                    "citation_route": "system_b",
                    "source_path": "db/paper-a/paper-a.en.md",
                    "ref_num": 9,
                    "anchor": "wrong-ref",
                }
            ]
        },
        citation_plan=plan,
    )
    assert not render_payload_is_missing_planned_system_b(
        {
            "cite_details": [
                {
                    "citation_route": "system_b",
                    "source_path": "C:/workspace/db/paper-a/paper-a.en.md",
                    "ref_num": 4,
                    "anchor": "exact-plan-coordinate",
                }
            ]
        },
        citation_plan=plan,
    )


def test_render_payload_rejects_system_b_same_tail_from_different_roots() -> None:
    plan = {
        "budget": {"system_a": 0, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": "C:/kb-one/shared/paper.en.md",
                "candidate_refs": [4],
            }
        ],
    }
    payload = {
        "cite_details": [
            {
                "citation_route": "system_b",
                "source_path": "D:/kb-two/shared/paper.en.md",
                "ref_num": 4,
            }
        ]
    }

    assert render_payload_is_missing_planned_system_b(payload, citation_plan=plan)


def test_render_payload_keeps_same_tail_system_b_slots_as_distinct_obligations() -> None:
    plan = {
        "budget": {"system_a": 0, "system_b": 2},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": "C:/kb-one/shared/paper.en.md",
                "candidate_refs": [4],
            },
            {
                "preferred_system": "system_b",
                "source_path": "D:/kb-two/shared/paper.en.md",
                "candidate_refs": [4],
            },
        ],
    }
    payload = {
        "cite_details": [
            {
                "citation_route": "system_b",
                "source_path": "C:/kb-one/shared/paper.en.md",
                "ref_num": 4,
            }
        ]
    }

    assert render_payload_is_missing_planned_system_b(payload, citation_plan=plan)


def test_render_payload_rejects_unresolved_public_source_same_tail() -> None:
    plan = {
        "budget": {"system_a": 0, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": "kb-source/0/shared/paper.en.md",
                "candidate_refs": [4],
            }
        ],
    }
    payload = {
        "cite_details": [
            {
                "citation_route": "system_b",
                "source_path": "D:/wrong/db/shared/paper.en.md",
                "ref_num": 4,
            }
        ]
    }

    assert render_payload_is_missing_planned_system_b(payload, citation_plan=plan)


def test_render_payload_accepts_public_source_after_canonical_resolution(
    monkeypatch,
) -> None:
    from api import message_render_contract

    public_path = "kb-source/0/shared/paper.en.md"
    private_path = "C:/workspace/md_output/shared/paper.en.md"

    def fake_canonical_source_path_identity(value: str) -> str:
        if str(value or "").replace("\\", "/").casefold() in {
            public_path.casefold(),
            private_path.casefold(),
        }:
            return "c:/canonical/shared/paper.en.md"
        return str(value or "").replace("\\", "/").casefold()

    monkeypatch.setattr(
        message_render_contract,
        "_canonical_source_path_identity",
        fake_canonical_source_path_identity,
    )
    plan = {
        "budget": {"system_a": 0, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": public_path,
                "candidate_refs": [4],
            }
        ],
    }
    payload = {
        "cite_details": [
            {
                "citation_route": "system_b",
                "source_path": private_path,
                "ref_num": 4,
            }
        ]
    }

    assert not render_payload_is_missing_planned_system_b(
        payload,
        citation_plan=plan,
    )


def test_render_packet_does_not_duplicate_one_system_b_card_for_two_slots() -> None:
    source_path = "db/paper/paper.en.md"
    plan = {
        "budget": {"system_a": 0, "system_b": 2},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": source_path,
                "candidate_refs": [4, 5],
            },
            {
                "preferred_system": "system_b",
                "source_path": source_path,
                "candidate_refs": [4, 6],
            },
        ],
    }
    detail = {
        "citation_route": "system_b",
        "source_path": source_path,
        "ref_num": 4,
    }

    assert render_payload_is_missing_planned_system_b(
        {"cite_details": [detail]},
        citation_plan=plan,
    )
    assert render_payload_is_missing_planned_system_b(
        {"render_packet": {"cite_details": [detail]}},
        citation_plan=plan,
    )


def test_render_payload_ignores_ranked_system_a_fallbacks_beyond_budget() -> None:
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": f"paper-{idx}.en.md",
                "heading_path": "Abstract",
                "evidence_quote": f"Paper {idx} directly supports claim {idx}.",
            }
            for idx in (1, 2, 3)
        ],
    }
    complete = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": f"paper-{idx}.en.md",
                "heading_path": "Abstract",
                "evidence_quote": f"Paper {idx} directly supports claim {idx}.",
            }
            for idx in (1, 2)
        ]
    }
    missing_selected_source = {
        "cite_details": [complete["cite_details"][0]]
    }

    assert not render_payload_is_missing_planned_system_a(
        complete,
        citation_plan=plan,
    )
    assert render_payload_is_missing_planned_system_a(
        missing_selected_source,
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


def test_render_payload_rejects_wrong_same_paper_passage_for_compound_claim() -> None:
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": r"F:\db\PILN\PILN.en.md",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "We proposed a self-supervised image-loop neural network with a "
                    "part-based model. It divides image features to facilitate "
                    "finer-grained learning."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": r"F:\db\PILN\PILN.en.md",
                "heading_path": "2.1. Methods",
                "evidence_quote": (
                    "ILNet uses a semi-finished reconstructed image loop to replace "
                    "the input of the network."
                ),
            },
        ],
    }
    stale = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/PILN/PILN.en.md",
                "heading_path": "2.1. Methods",
                "evidence_quote": (
                    "ILNet uses a semi-finished reconstructed image loop to replace "
                    "the input of the network."
                ),
                "answer_claim": (
                    "ILNet is a self-supervised image-loop neural network whose "
                    "part-based model supports finer-grained learning."
                ),
            }
        ]
    }
    repaired = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": "kb-source/0/PILN/PILN.en.md",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "We proposed a self-supervised image-loop neural network with a "
                    "part-based model. It divides image features to facilitate "
                    "finer-grained learning."
                ),
                "answer_claim": (
                    "ILNet is a self-supervised image-loop neural network whose "
                    "part-based model supports finer-grained learning."
                ),
            }
        ]
    }

    assert render_payload_is_missing_planned_system_a(stale, citation_plan=plan)
    assert not render_payload_is_missing_planned_system_a(repaired, citation_plan=plan)


def test_render_payload_requires_each_requested_relation_bundle_passage() -> None:
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "ddpm.en.md",
                "heading_path": "3.4 Simplified training objective",
                "evidence_quote": (
                    "The simplified training objective is an unweighted version "
                    "and predicts epsilon with L_simple."
                ),
                "evidence_selection_reason": "requested_relation_bundle",
            },
            {
                "preferred_system": "system_a",
                "source_path": "ddpm.en.md",
                "heading_path": "4 Experiments / 4.1 Sample quality",
                "evidence_quote": (
                    "The true variational bound yields better codelengths, while "
                    "the simplified objective gives the best sample quality."
                ),
                "evidence_selection_reason": "requested_relation_bundle",
            },
        ],
    }
    objective = {
        "citation_route": "system_a",
        "source_path": "ddpm.en.md",
        "heading_path": "3.4 Simplified training objective",
        "evidence_quote": plan["slots"][0]["evidence_quote"],
        "answer_claim": "L_simple is an unweighted objective that predicts epsilon.",
    }
    tradeoff = {
        "citation_route": "system_a",
        "source_path": "ddpm.en.md",
        "heading_path": "4 Experiments / 4.1 Sample quality",
        "evidence_quote": plan["slots"][1]["evidence_quote"],
        "answer_claim": (
            "The true variational bound improves codelengths while the simplified "
            "objective gives the best sample quality."
        ),
    }

    assert render_payload_is_missing_planned_system_a(
        {"cite_details": [objective]},
        citation_plan=plan,
    )
    assert not render_payload_is_missing_planned_system_a(
        {"cite_details": [objective, tradeoff]},
        citation_plan=plan,
    )


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


def test_numeric_citation_detection_protects_code_arrays_ranges_and_years() -> None:
    hits = [{"meta": {"source_path": "paper-one.md"}}]

    for content in (
        "Use the empty array [].",
        "The values are [1, 2].",
        "The interval is [1-3].",
        "The paper appeared in [2024].",
        "Inline code: `[1]`.",
        "```python\nvalues = [1]\n```",
    ):
        assert not content_has_linkable_answer_citations(content, hits)

    assert content_has_linkable_answer_citations("Supported claim [1].", hits)


def test_render_packet_is_authoritative_over_conflicting_legacy_cache_fields() -> None:
    packet = {
        "notice": "packet notice",
        "rendered_body": "packet body",
        "rendered_content": "packet notice\n\npacket body",
        "copy_markdown": "packet notice\n\npacket body",
        "copy_text": "packet notice packet body",
        "cite_details": [],
    }
    cache = {
        "schema": 5,
        "cache_key": "abc",
        "notice": "stale notice",
        "rendered_body": "stale body",
        "rendered_content": "stale content",
        "copy_markdown": "stale copy",
        "copy_text": "stale text",
        "cite_details": [{"num": 9}],
        "render_packet": packet,
    }

    payload = normalize_render_cache_payload(cache, schema=5, expected_key="abc")

    assert payload is not None
    assert payload.notice == "packet notice"
    assert payload.rendered_body == "packet body"
    assert payload.copy_markdown == "packet notice\n\npacket body"
    assert payload.cite_details == []

    rec = {"rendered_body": "other stale body", "cite_details": [{"num": 7}]}
    assert project_render_packet_to_record(rec, packet)
    assert rec["rendered_body"] == "packet body"
    assert rec["cite_details"] == []


def test_build_cache_atomically_projects_packet_and_signatures() -> None:
    packet = {
        "answer_markdown": "Answer [1].",
        "notice": "Read with care.",
        "rendered_body": "Answer [1](#kb-cite-a-1).",
        "rendered_content": "Read with care.\n\nAnswer [1](#kb-cite-a-1).",
        "copy_markdown": "Read with care.\n\nAnswer [1].",
        "copy_text": "Read with care. Answer [1].",
        "cite_details": [
            {
                "num": 1,
                "anchor": "kb-cite-a-1",
                "source_path": "paper-one.md",
            }
        ],
    }

    cache = build_render_cache_payload(
        schema=48,
        cache_key="cache-key",
        notice="stale notice",
        rendered_body="stale body",
        rendered_content="stale content",
        copy_markdown="stale copy",
        copy_text="stale text",
        cite_details=[],
        refs_user_msg_id=10,
        render_packet=packet,
        answer_sig="answer-sig",
        input_ref_sig="refs-sig",
        citation_plan_sig="plan-sig",
        locale="ZH",
    )

    assert cache["rendered_body"] == packet["rendered_body"]
    assert cache["copy_markdown"].startswith("Read with care.")
    assert cache["cite_details"] == packet["cite_details"]
    assert cache["answer_sig"] == cache["render_packet"]["answer_sig"] == "answer-sig"
    assert cache["input_ref_sig"] == cache["render_packet"]["input_ref_sig"] == "refs-sig"
    assert cache["citation_plan_sig"] == cache["render_packet"]["citation_plan_sig"] == "plan-sig"
    assert cache["locale"] == cache["render_packet"]["locale"] == "zh"
    assert cache["schema"] == cache["render_packet"]["schema"] == 48


def test_citation_link_health_requires_exact_card_anchor_number_and_source() -> None:
    valid = {
        "rendered_body": "Claim [1](#kb-cite-paper-1).",
        "cite_details": [
            {
                "num": 1,
                "anchor": "kb-cite-paper-1",
                "source_path": "paper-one.md",
            }
        ],
    }
    hits = [{"meta": {"source_path": "paper-one.md"}}]

    assert render_payload_has_citation_links(valid, hits=hits)
    assert not render_payload_has_citation_links(
        {**valid, "cite_details": []},
        hits=hits,
    )
    assert not render_payload_has_citation_links(
        {
            **valid,
            "rendered_body": "Claim [1](#kb-cite-missing-1).",
        },
        hits=hits,
    )
    assert not render_payload_has_citation_links(
        {
            **valid,
            "cite_details": [
                {
                    "num": 0,
                    "anchor": "kb-cite-paper-1",
                    "source_path": "paper-one.md",
                }
            ],
        },
        hits=hits,
    )
    assert render_payload_has_citation_links(
        {
            "rendered_body": (
                "Claim [1](#kb-cite-paper-1) and repeat "
                "[1](#kb-cite-paper-duplicate)."
            ),
            "cite_details": [
                {
                    "num": 1,
                    "anchor": "kb-cite-paper-1",
                    "source_path": "paper-one.md",
                },
                {
                    "num": 1,
                    "anchor": "kb-cite-paper-duplicate",
                    "source_path": "paper-one.md",
                },
            ],
        },
        hits=hits,
    )
    assert not render_payload_has_citation_links(
        {
            "rendered_body": (
                "First [1](#kb-cite-paper-1) and conflicting "
                "[1](#kb-cite-paper-two)."
            ),
            "cite_details": [
                {
                    "num": 1,
                    "anchor": "kb-cite-paper-1",
                    "source_path": "paper-one.md",
                },
                {
                    "num": 1,
                    "anchor": "kb-cite-paper-two",
                    "source_path": "paper-two.md",
                },
            ],
        },
        hits=[
            {"meta": {"source_path": "paper-one.md"}},
            {"meta": {"source_path": "paper-two.md"}},
        ],
    )
    assert not render_payload_has_citation_links(
        {
            **valid,
            "cite_details": [
                {
                    "num": 1,
                    "anchor": "kb-cite-paper-1",
                    "source_path": "paper-two.md",
                }
            ],
        },
        hits=hits,
    )
