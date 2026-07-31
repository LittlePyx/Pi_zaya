import json
import re
from pathlib import Path

import pytest

from kb.chat_store import ChatStore
from api.citation_display_registry import (
    remap_system_a_citations_for_display,
    system_a_source_key,
)
from api.chat_render import (
    _augment_hits_with_canonical_answer_citations,
    _citation_free_answer_body,
    _collapse_adjacent_same_citation_links,
    _enrich_provenance_segments_for_display,
    _normalize_chat_markdown_for_display,
    _normalize_double_numeric_citation_markers,
    _normalize_equation_source_notes,
    _strip_freeform_numeric_citation_markers,
    enrich_messages_with_reference_render,
)
from tests._paper_guide_fixtures import build_scinerf_like_fixture


MOJIBAKE_REFERENCE_LOCATOR = "\u9359\u509d\u20ac\u51a8\u757e\u6d63"
MOJIBAKE_REFERENCE_SOURCE_PREFIX = "\u93c9\u30e8\u569c" + MOJIBAKE_REFERENCE_LOCATOR + "?#1\u951b\u6b5a"


def test_effective_citation_render_locale_prefers_current_card_preference(monkeypatch) -> None:
    from api import chat_render

    monkeypatch.setattr(
        chat_render,
        "load_prefs",
        lambda: {"refs_card_locale": "en", "ui_locale": "zh"},
    )

    assert chat_render._effective_citation_render_locale(
        {
            "render_locale": "zh",
            "rendered_payload": {"render_locale": "zh"},
        }
    ) == "en"


def test_effective_citation_render_locale_uses_ui_then_pack_fallback(monkeypatch) -> None:
    from api import chat_render

    monkeypatch.setattr(
        chat_render,
        "load_prefs",
        lambda: {"refs_card_locale": "auto", "ui_locale": "en"},
    )
    assert chat_render._effective_citation_render_locale({"render_locale": "zh"}) == "en"

    monkeypatch.setattr(
        chat_render,
        "load_prefs",
        lambda: {"refs_card_locale": "auto", "ui_locale": ""},
    )
    assert chat_render._effective_citation_render_locale({"render_locale": "zh"}) == "zh"


def test_double_numeric_citations_never_render_as_empty_brackets() -> None:
    assert _normalize_double_numeric_citation_markers("A [[4]], B [[5；2]].") == "A [4], B [5；2]."
    stripped = _strip_freeform_numeric_citation_markers("A [[4]], B [[]], C [].")
    assert stripped == "A, B [[]], C []."


def test_signed_bpsk_values_do_not_consume_the_real_source_citation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    def fake_annotate(
        markdown,
        hits,
        *,
        anchor_ns="",
        canonical_paths=None,
        citation_plan=None,
        render_locale=None,
    ):
        del hits, anchor_ns, canonical_paths, citation_plan, render_locale
        assert "(+1, -1)" in markdown
        assert "[1, -1]" not in markdown
        return (
            markdown.replace("[1]", "[1](#kb-cite-fdm-1)", 1),
            [
                {
                    "num": 1,
                    "anchor": "kb-cite-fdm-1",
                    "citation_route": "system_a",
                    "source_path": "FDM.en.md",
                    "source_name": "FDM.pdf",
                    "evidence_quote": "BPSK and parallel demodulation evidence.",
                }
            ],
        )

    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        fake_annotate,
    )
    source_file = tmp_path / "FDM" / "FDM.en.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "# FDM\n\n## B. Encoding\n\n"
        "The mask values are encoded with binary phase-shift keying, and each "
        "carrier frequency is demodulated by a lock-in amplifier.\n",
        encoding="utf-8",
    )
    source_path = str(source_file)
    messages = [
        {"id": 1, "role": "user", "content": "FDM-SPI 如何编码？"},
        {
            "id": 2,
            "role": "assistant",
            "content": (
                "Each mask uses BPSK values [1, -1], and the carrier-frequency "
                "channels are demodulated in parallel [1]."
            ),
            "meta": {"canonical_hit_paths": [source_path]},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": (
                        "The mask values are encoded with binary phase-shift keying, "
                        "and each carrier frequency is demodulated by a lock-in amplifier."
                    ),
                    "meta": {
                        "source_path": source_path,
                        "source_name": "FDM.pdf",
                        "heading_path": "B. Encoding",
                        "page_start": 2,
                        "ref_answer_citation_num": 1,
                    },
                }
            ],
            "display_state": "ready",
        }
    }

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-bpsk-vector",
    )[-1]
    body = str(rendered.get("rendered_content") or "")

    assert "(+1, -1)" in body
    assert "[1, -1]" not in body
    assert "[]" not in body
    assert "](#kb-cite-" in body
    assert len(rendered.get("cite_details") or []) == 1


def test_adjacent_same_citation_links_collapse_even_when_titles_differ() -> None:
    first = '[1](#kb-cite-author-2 "source: paper | ref 1")'
    duplicate = '[1](#kb-cite-author-2 "source: paper | ref 2")'
    other = '[1](#kb-cite-author-3 "source: paper | ref 3")'

    collapsed = _collapse_adjacent_same_citation_links(
        f"Yaoxing {first}{duplicate}; Liantuan {other}"
    )

    assert collapsed == f"Yaoxing {first}; Liantuan {other}"


def test_system_a_display_registry_remaps_answer_hit_numbers_but_keeps_system_b() -> None:
    markdown = (
        '速度证据 [4](#cite-speed "speed")；'
        '质量证据 [5](#cite-quality "quality")；'
        '文内参考 [66](#cite-system-b "reference").'
    )
    details = [
        {
            "num": 4,
            "linked_nums": [4],
            "anchor": "cite-speed",
            "citation_route": "system_a",
            "source_path": r"F:\db\Speed\Speed.en.md",
            "source_name": "Speed.pdf",
        },
        {
            "num": 5,
            "linked_nums": [5],
            "anchor": "cite-quality",
            "citation_route": "system_a",
            "source_path": r"F:\db\Quality\Quality.en.md",
            "source_name": "Quality.pdf",
        },
        {
            "num": 66,
            "anchor": "cite-system-b",
            "citation_route": "system_b",
            "source_path": r"F:\db\Review\Review.en.md",
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert '[1](#cite-speed' in rendered
    assert '[2](#cite-quality' in rendered
    assert '[66](#cite-system-b' in rendered
    assert [row["num"] for row in remapped] == [1, 2, 66]
    assert remapped[0]["answer_hit_num"] == 4
    assert remapped[1]["answer_hit_num"] == 5
    assert [row["original_nums"] for row in registry] == [[4], [5]]


def test_system_a_display_registry_skips_numbers_reserved_by_system_b() -> None:
    markdown = (
        'Direct evidence [4](#cite-direct "source: Direct.pdf | ref 4"). '
        'Upstream reference [1](#cite-upstream "source: Review.pdf | ref 1").'
    )
    details = [
        {
            "num": 4,
            "anchor": "cite-direct",
            "citation_route": "system_a",
            "source_path": r"F:\db\Direct\Direct.en.md",
            "source_name": "Direct.pdf",
        },
        {
            "num": 1,
            "linked_nums": [1],
            "anchor": "cite-upstream",
            "citation_route": "system_b",
            "is_inpaper": True,
            "source_path": r"F:\db\Review\Review.en.md",
            "inpaper_ref_num": 1,
            "title": "Upstream work",
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert '[2](#cite-direct' in rendered
    assert '[1](#cite-upstream' in rendered
    assert [row["num"] for row in remapped] == [2, 1]
    assert registry[0]["display_num"] == 2


def test_system_a_source_key_keeps_same_named_papers_from_different_roots_distinct() -> None:
    first = system_a_source_key(
        {
            "citation_route": "system_a",
            "source_path": r"F:\project-one\Paper\Paper.en.md",
        }
    )
    second = system_a_source_key(
        {
            "citation_route": "system_a",
            "source_path": r"G:\project-two\Paper\Paper.en.md",
        }
    )

    assert first.startswith("path:")
    assert second.startswith("path:")
    assert first != second
    assert "project-one" not in first
    assert "project-two" not in second

    rendered, remapped, registry = remap_system_a_citations_for_display(
        'First [4](#cite-first "first"). Second [5](#cite-second "second").',
        [
            {
                "num": 4,
                "anchor": "cite-first",
                "citation_route": "system_a",
                "source_path": r"F:\project-one\Paper\Paper.en.md",
            },
            {
                "num": 5,
                "anchor": "cite-second",
                "citation_route": "system_a",
                "source_path": r"G:\project-two\Paper\Paper.en.md",
            },
        ],
    )
    assert '[1](#cite-first' in rendered
    assert '[2](#cite-second' in rendered
    assert [row["num"] for row in remapped] == [1, 2]
    assert len(registry) == 2


def test_system_a_display_registry_maps_multiple_passages_from_one_paper_to_one_card() -> None:
    markdown = '优势 [4](#cite-benefit "benefit")，局限 [5](#cite-limit "limit").'
    details = [
        {
            "num": 4,
            "anchor": "cite-benefit",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "answer_claim": "Review benefit claim.",
        },
        {
            "num": 5,
            "anchor": "cite-limit",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "answer_claim": "Review limitation claim.",
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert '[1](#cite-benefit' in rendered
    assert '[1](#cite-limit' in rendered
    assert [row["num"] for row in remapped] == [1, 1]
    assert remapped[0]["answer_claims"] == remapped[1]["answer_claims"] == [
        "Review benefit claim.",
        "Review limitation claim.",
    ]
    assert len(registry) == 1
    assert registry[0]["original_nums"] == [4, 5]


def test_system_a_display_registry_collapses_exact_evidence_duplicates_only() -> None:
    markdown = (
        'Overview [1](#cite-overview "source: Review.pdf"). '
        'Acquisition [1](#cite-acquisition "source: Review.pdf"). '
        'Reconstruction [1](#cite-reconstruction "source: Review.pdf").'
    )
    details = [
        {
            "num": 1,
            "anchor": "cite-overview",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "source_name": "Review.pdf",
            "evidence_fingerprint": "same-evidence-12345678",
            "evidence_quote": "The same exact source passage.",
        },
        {
            "num": 1,
            "anchor": "cite-acquisition",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "source_name": "Review.pdf",
            "evidence_fingerprint": "same-evidence-12345678",
            "answer_claim": "Acquisition claim.",
            "evidence_quote": "The same exact source passage.",
        },
        {
            "num": 1,
            "anchor": "cite-reconstruction",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "source_name": "Review.pdf",
            "evidence_fingerprint": "same-evidence-12345678",
            "answer_claim": "Reconstruction claim.",
            "evidence_quote": "The same exact source passage.",
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert len(remapped) == 1
    assert rendered.count(f'](#{remapped[0]["anchor"]}') == 3
    assert "cite-overview" not in rendered
    assert set(remapped[0]["answer_claims"]) >= {
        "Acquisition claim.",
        "Reconstruction claim.",
    }
    assert len(registry) == 1


def test_system_a_display_registry_collapses_occurrences_with_shared_plan_budget_key() -> None:
    markdown = (
        'Paper title [3](#cite-title "source: HSI-FSI.pdf"). '
        'Measured comparison [3](#cite-claim "source: HSI-FSI.pdf").'
    )
    details = [
        {
            "num": 3,
            "anchor": "cite-title",
            "citation_route": "system_a",
            "source_path": r"F:\db\HSI-FSI\HSI-FSI.en.md",
            "evidence_fingerprint": "title-occurrence-12345678",
            "citation_budget_key": "plan:shared-comparison-evidence",
            "evidence_quote": "HSI and FSI are compared in imaging efficiency and noise robustness.",
        },
        {
            "num": 3,
            "anchor": "cite-claim",
            "citation_route": "system_a",
            "source_path": r"F:\db\HSI-FSI\HSI-FSI.en.md",
            "evidence_fingerprint": "claim-occurrence-87654321",
            "citation_budget_key": "plan:shared-comparison-evidence",
            "answer_claim": "The paper compares HSI and FSI directly.",
            "evidence_quote": "HSI and FSI are compared in imaging efficiency and noise robustness.",
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert len(remapped) == 1
    assert rendered.count(f'](#{remapped[0]["anchor"]}') == 2
    assert "cite-title" not in rendered or "cite-claim" not in rendered
    assert len(registry) == 1


def test_system_a_display_registry_keeps_distinct_evidence_from_same_paper() -> None:
    markdown = (
        'Benefit [1](#cite-benefit "source: Review.pdf"). '
        'Limit [1](#cite-limit "source: Review.pdf").'
    )
    details = [
        {
            "num": 1,
            "anchor": "cite-benefit",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "evidence_fingerprint": "benefit-evidence-1234",
            "evidence_quote": "The method improves reconstruction quality.",
        },
        {
            "num": 1,
            "anchor": "cite-limit",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "evidence_fingerprint": "limit-evidence-567890",
            "evidence_quote": "The method still requires extensive training data.",
        },
    ]

    rendered, remapped, _registry = remap_system_a_citations_for_display(markdown, details)

    assert len(remapped) == 2
    assert "#cite-benefit" in rendered
    assert "#cite-limit" in rendered


def test_system_a_display_registry_keeps_distinct_evidence_with_shared_plan_budget_key() -> None:
    markdown = (
        'Benefit [1](#cite-benefit "source: Review.pdf"). '
        'Limit [1](#cite-limit "source: Review.pdf").'
    )
    details = [
        {
            "num": 1,
            "anchor": "cite-benefit",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "evidence_fingerprint": "benefit-evidence-1234",
            "citation_budget_key": "plan:shared-paper-slot",
            "evidence_quote": "The method improves reconstruction quality and speed.",
        },
        {
            "num": 1,
            "anchor": "cite-limit",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "evidence_fingerprint": "limit-evidence-567890",
            "citation_budget_key": "plan:shared-paper-slot",
            "evidence_quote": "The method requires lengthy training and generalizes poorly.",
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert len(remapped) == 2
    assert "#cite-benefit" in rendered
    assert "#cite-limit" in rendered
    assert len(registry) == 1


def test_system_a_display_registry_rebinds_repeated_source_to_matching_passage() -> None:
    markdown = (
        '先做光线追迹 [1](#cite-ray "source: qCLFM.pdf")。'
        '再做波传播逆运算 [1](#cite-ray "source: qCLFM.pdf")。'
        '两步共同完成重聚焦 [1](#cite-compound "source: qCLFM.pdf")。'
    )
    details = [
        {
            "num": 1,
            "anchor": "cite-ray",
            "citation_route": "system_a",
            "source_path": r"F:\db\qCLFM\qCLFM.en.md",
            "source_name": "qCLFM.pdf",
            "evidence_quote": "The photon trajectory is reconstructed through ray tracing.",
        },
        {
            "num": 1,
            "anchor": "cite-compound",
            "citation_route": "system_a",
            "source_path": r"F:\db\qCLFM\qCLFM.en.md",
            "source_name": "qCLFM.pdf",
            "evidence_quote": (
                "First, the photon trajectory is reconstructed through ray tracing. "
                "Second, wave propagation of distance -z brings the sample back into focus."
            ),
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert rendered.count('[1](#cite-ray') == 1
    assert rendered.count('[1](#cite-compound') == 2
    assert [row["anchor"] for row in remapped] == ["cite-ray", "cite-compound"]
    assert len(registry) == 1


def test_system_a_display_registry_never_crosses_original_same_paper_occurrences() -> None:
    markdown = (
        'Kai profile [1](#cite-kai "source: LPR.pdf | ref 1"). '
        'Yaoxing is currently a lecturer [2](#cite-yaoxing "source: LPR.pdf | ref 2"). '
        'Liantuan is currently a professor at the university '
        '[3](#cite-liantuan "source: LPR.pdf | ref 3").'
    )
    details = [
        {
            "num": 1,
            "anchor": "cite-kai",
            "citation_route": "system_a",
            "source_path": r"F:\db\LPR\LPR.en.md",
            "source_name": "LPR.pdf",
            "evidence_quote": "Kai completed his degrees and is pursuing a doctorate.",
        },
        {
            "num": 2,
            "anchor": "cite-yaoxing",
            "citation_route": "system_a",
            "source_path": r"F:\db\LPR\LPR.en.md",
            "source_name": "LPR.pdf",
            "evidence_quote": (
                "Yaoxing is currently a lecturer at the university and his research "
                "interests include imaging."
            ),
        },
        {
            "num": 3,
            "anchor": "cite-liantuan",
            "citation_route": "system_a",
            "source_path": r"F:\db\LPR\LPR.en.md",
            "source_name": "LPR.pdf",
            # Exercise the real failure mode: the third occurrence's card
            # evidence can be filtered before final citation-plan refinement.
            "evidence_quote": "",
        },
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert '[1](#cite-kai' in rendered
    assert '[1](#cite-yaoxing' in rendered
    assert '[1](#cite-liantuan' in rendered
    assert [row["answer_hit_num"] for row in remapped] == [1, 2, 3]
    assert registry[0]["original_nums"] == [1, 2, 3]

    rendered_again, remapped_again, registry_again = remap_system_a_citations_for_display(
        rendered,
        remapped,
    )
    assert rendered_again == rendered
    assert [row["anchor"] for row in remapped_again] == [
        "cite-kai",
        "cite-yaoxing",
        "cite-liantuan",
    ]
    assert registry_again[0]["original_nums"] == [1, 2, 3]


def test_system_a_display_registry_is_idempotent_for_historical_render_packets() -> None:
    markdown = '速度证据 [1](#cite-speed "speed").'
    details = [
        {
            "num": 1,
            "display_num": 1,
            "linked_nums": [1],
            "answer_hit_num": 4,
            "answer_hit_linked_nums": [4],
            "original_num": 4,
            "anchor": "cite-speed",
            "citation_route": "system_a",
            "source_path": r"F:\db\Speed\Speed.en.md",
        }
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert rendered == markdown
    assert remapped[0]["num"] == 1
    assert remapped[0]["answer_hit_num"] == 4
    assert remapped[0]["answer_hit_linked_nums"] == [4]
    assert registry[0]["original_nums"] == [4]


def test_system_a_display_registry_collapses_same_source_occurrence_to_card_anchor() -> None:
    markdown = (
        '直接结论 [4](#cite-result "source: Review.pdf | ref 4")；'
        'PDF 页码 [4](#cite-page "source: Review.pdf | ref 4").'
    )
    details = [
        {
            "num": 4,
            "anchor": "cite-result",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "source_name": "Review.pdf",
        }
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert '[1](#cite-result' in rendered
    assert '#cite-page' not in rendered
    assert rendered.count('[1](#cite-result') == 2
    assert [row["anchor"] for row in remapped] == ["cite-result"]
    assert [row["num"] for row in remapped] == [1]
    assert len(registry) == 1


def test_system_a_display_registry_prefers_original_number_over_colliding_display_number() -> None:
    markdown = (
        'First paper [2](#cite-first "source: First.pdf | ref 2"). '
        'Repeated first paper [2](#cite-first-extra "source: First.pdf | ref 2"). '
        'Second paper [3](#cite-second "source: Second.pdf | ref 3").'
    )
    details = [
        {
            "num": 2,
            "anchor": "cite-first",
            "citation_route": "system_a",
            "source_path": r"F:\db\First\First.en.md",
            "source_name": "First.pdf",
        },
        {
            "num": 3,
            "anchor": "cite-second",
            "citation_route": "system_a",
            "source_path": r"F:\db\Second\Second.en.md",
            "source_name": "Second.pdf",
        },
    ]

    rendered, remapped, _registry = remap_system_a_citations_for_display(markdown, details)

    assert rendered.count('[1](#cite-first') == 2
    assert '#cite-first-extra' not in rendered
    assert '[2](#cite-second' in rendered
    assert [(row["num"], row["anchor"]) for row in remapped] == [
        (1, "cite-first"),
        (2, "cite-second"),
    ]


def test_system_a_display_registry_collects_all_linked_claims_for_one_card() -> None:
    markdown = (
        '**Structured detection** solves the confocal trade-off [2](#cite-method "source: Method.pdf"). '
        'The resulting s²ISM also provides optical sectioning [2](#cite-method "source: Method.pdf").'
    )
    details = [
        {
            "num": 2,
            "anchor": "cite-method",
            "citation_route": "system_a",
            "source_path": r"F:\db\Method\Method.en.md",
            "source_name": "Method.pdf",
        }
    ]

    _rendered, remapped, _registry = remap_system_a_citations_for_display(markdown, details)

    claims = remapped[0]["answer_claims"]
    assert any("Structured detection" in claim for claim in claims)
    assert any("s²ISM" in claim and "optical sectioning" in claim for claim in claims)


def test_system_a_display_registry_does_not_alias_system_b_reader_anchor() -> None:
    markdown = (
        'Direct evidence [4](#cite-result "source: Review.pdf | ref 4"); '
        'bibliography [4](#kb-cite-reader-external-4 "source: Review.pdf | ref 4").'
    )
    details = [
        {
            "num": 4,
            "anchor": "cite-result",
            "citation_route": "system_a",
            "source_path": r"F:\db\Review\Review.en.md",
            "source_name": "Review.pdf",
        }
    ]

    rendered, remapped, registry = remap_system_a_citations_for_display(markdown, details)

    assert '[1](#cite-result' in rendered
    assert '[4](#kb-cite-reader-external-4' in rendered
    assert [row["anchor"] for row in remapped] == ["cite-result"]
    assert len(registry) == 1


def test_system_a_backfill_rebinds_wrong_same_paper_passage_to_claim_block(tmp_path) -> None:
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source = tmp_path / "Review.en.md"
    source.write_text(
        "# Review\n\n"
        "## 5.2. Imaging Through Scattering Media\n\n"
        "<!-- kb_page: 10 -->\n\n"
        "Zhang et al. proposed a Multi-scale GAN (MsGAN). "
        "The experimental results acquired by CCD and MsGAN are shown in Figure 6a. "
        "The image quality recovered by MsGAN consistently surpasses that of CCD, "
        "and the advantage becomes notably prominent as turbulence increases.\n\n"
        "## 5.3. Imaging at Photon-Level\n\n"
        "<!-- kb_page: 11 -->\n\n"
        "Figure 7a shows the results of an unsupervised anti-noise framework.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 1,
            "anchor": "kb-cite-review-1",
            "citation_route": "system_a",
            "source_path": str(source),
            "source_name": "Review.pdf",
            "heading_path": "Review / 5.3. Imaging at Photon-Level",
            "evidence_quote": "Figure 7a shows the results of an unsupervised anti-noise framework.",
            "raw": "Figure 7a shows the results of an unsupervised anti-noise framework.",
            "page_start": 11,
            "answer_claim": "根据 LPR-2025 综述，MsGAN 恢复质量始终优于 CCD，且湍流越强优势越明显。",
        }
    ]

    out = _backfill_system_a_cite_details_from_ref_pack(details, {}, render_locale="zh")

    assert len(out) == 1
    assert "MsGAN consistently surpasses" in out[0]["evidence_quote"]
    assert "turbulence increases" in out[0]["evidence_quote"]
    assert "Figure 7a shows" not in out[0]["evidence_quote"]
    assert "5.2. Imaging Through Scattering Media" in out[0]["heading_path"]
    assert out[0]["page_start"] == 10
    assert out[0]["block_id"]


def test_inline_system_a_card_uses_claim_aligned_window_from_full_citation_plan() -> None:
    from api.chat_render import _refine_system_a_cite_evidence_from_citation_plan

    source_path = r"F:\db\FDM\FDM.en.md"
    opening = (
        "We propose and experimentally realize frequency-division-multiplexed single-pixel imaging. "
        "Our technique relies on metamaterial spatial light modulators. "
        "Earlier implementations used one encoding frequency and were sensitive to narrowband noise."
    )
    support = (
        "Here, we implement frequency-division methods to parallelize the single-pixel imaging process at 3.2 THz. "
        "Our technique enables a trade-off between signal-to-noise ratio and acquisition speed—without altering "
        "detector integration time."
    )
    details = [
        {
            "num": 1,
            "citation_route": "system_a",
            "source_path": source_path,
            "source_name": "FDM.pdf",
            "heading_path": "FDM / Abstract",
            "evidence_quote": opening,
            "summary_line": opening,
            "raw": f"{opening} Here, we implement frequency-division methods to parallelize",
            "answer_claim": "频分复用把成像过程并行化，并以信噪比换取采集速度。",
            "answer_claims": ["频分复用无需改变探测器积分时间。"],
            "block_id": "blk-abstract",
            "anchor_id": "p-abstract",
            "anchor_kind": "paragraph",
            "page_start": 1,
        }
    ]
    citation_plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "FDM.pdf",
                "heading_path": "FDM / Abstract",
                "evidence_quote": f"## Abstract {opening} {support}",
            }
        ]
    }

    out = _refine_system_a_cite_evidence_from_citation_plan(
        details,
        citation_plan,
        render_locale="zh",
    )

    quote = out[0]["evidence_quote"]
    assert quote.startswith("Here, we implement frequency-division methods")
    assert all(
        term in quote
        for term in (
            "parallelize",
            "signal-to-noise ratio",
            "acquisition speed",
            "detector integration time",
        )
    )
    evidence_section = next(
        section
        for section in out[0]["card_view"]["sections"]
        if section["id"] == "evidence"
    )
    assert evidence_section["text"] == quote


def test_inline_system_a_card_keeps_complete_multi_step_mechanism() -> None:
    from api.chat_render import _refine_system_a_cite_evidence_from_citation_plan

    source_path = r"F:\db\qCLFM\qCLFM.en.md"
    full_evidence = (
        "The operation for digital refocusing of a sample placed out of focus by a distance z "
        "can be achieved using two steps. First, using the position and angular information of "
        "each photon, and knowing the optical elements used between them, the trajectory of the "
        "photons can be reconstructed through a ray tracing operation. Thus, the second step is "
        "to reverse this diffraction by applying a wave propagation of distance -z to the image "
        "obtained after step one in order to bring the sample back into focus."
    )
    first_step_only = full_evidence.split(" Thus,")[0]
    details = [
        {
            "num": 1,
            "citation_route": "system_a",
            "source_path": source_path,
            "source_name": "qCLFM.pdf",
            "heading_path": "qCLFM / A. Concept",
            "evidence_quote": first_step_only,
            "summary_line": first_step_only,
            "raw": first_step_only,
            "answer_claim": "数字重聚焦先用 ray tracing 重建光子轨迹，再用 wave propagation 反演衍射。",
            "block_id": "blk-concept",
            "anchor_id": "p-refocus",
            "anchor_kind": "paragraph",
            "page_start": 2,
        }
    ]
    citation_plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "qCLFM.pdf",
                "heading_path": "qCLFM / A. Concept",
                "evidence_quote": full_evidence,
            }
        ]
    }

    out = _refine_system_a_cite_evidence_from_citation_plan(
        details,
        citation_plan,
        render_locale="zh",
    )

    quote = out[0]["evidence_quote"]
    assert len(quote) > 460
    assert all(term in quote for term in ("two steps", "ray tracing", "wave propagation", "distance -z"))
    assert "ray tracing" in out[0]["support_relation"]
    assert "wave propagation" in out[0]["support_relation"]
    evidence_section = next(
        section for section in out[0]["card_view"]["sections"] if section["id"] == "evidence"
    )
    assert evidence_section["text"] == quote


def test_inline_system_a_compound_card_and_reader_locator_ignore_plan_order() -> None:
    from api.chat_render import (
        _refine_system_a_cite_evidence_from_citation_plan,
        _refine_system_a_cite_locators_from_final_primary,
    )

    source_path = r"F:\db\qCLFM\qCLFM.en.md"
    framing = (
        "The operation for digital refocusing of a sample placed out of focus by a distance z "
        "can be achieved using two steps."
    )
    ray_step = (
        "First, using the position and angular information of each photon, and knowing the optical "
        "elements used between them, the trajectory of the photons can be reconstructed through a "
        "ray tracing operation."
    )
    intervening = (
        "For macroscopic samples, this first step, using ray optics, is enough to bring the sample "
        "back into focus [15], however, for microscopic samples, interference and diffraction "
        "effects from wave optics must also be taken into account. In the microscopic regime, the "
        "image obtained after this first step is, in fact, the diffraction pattern of the sample "
        "after propagating a distance z."
    )
    wave_step = (
        "Thus, the second step is to reverse this diffraction by applying a wave propagation of "
        "distance -z to the image obtained after step one in order to bring the sample back into "
        "focus."
    )
    tail = (
        "The refocusing process is illustrated in Fig.2. Details on the experimental setup and the "
        "refocusing procedure can be found in the Methods section."
    )
    compact = " ".join((framing, ray_step, wave_step))
    continuous = " ".join((framing, ray_step, intervening, wave_step, tail))
    detail = {
        "num": 1,
        "citation_route": "system_a",
        "source_path": source_path,
        "source_name": "qCLFM.pdf",
        "heading_path": "qCLFM / A. Concept",
        "evidence_quote": " ".join((framing, ray_step)),
        "summary_line": " ".join((framing, ray_step)),
        "raw": " ".join((framing, ray_step)),
        "answer_claim": (
            "Digital refocusing uses two steps: first reconstruct photon trajectories with ray "
            "tracing, then reverse diffraction with wave propagation."
        ),
    }
    compact_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "qCLFM.pdf",
        "heading_path": "qCLFM / A. Concept",
        "evidence_quote": compact,
        "page_start": 2,
    }
    located_slot = {
        **compact_slot,
        "evidence_quote": continuous,
        "block_id": "blk-concept",
        "anchor_id": "p-refocus",
        "anchor_kind": "paragraph",
    }

    outputs = []
    for slots in ([compact_slot, located_slot], [located_slot, compact_slot]):
        refined = _refine_system_a_cite_evidence_from_citation_plan(
            [detail],
            {"slots": slots},
            render_locale="zh",
        )
        outputs.append(
            _refine_system_a_cite_locators_from_final_primary(
                refined,
                {
                    "source_path": source_path,
                    "heading_path": "qCLFM / A. Concept",
                    "block_id": "blk-concept",
                    "anchor_id": "p-refocus",
                    "anchor_kind": "paragraph",
                    "page_start": 2,
                    "page_end": 2,
                    "strict_locate": True,
                },
                render_locale="zh",
            )[0]
        )

    for output in outputs:
        assert output["evidence_quote"] == compact
        assert "For macroscopic samples" not in output["evidence_quote"]
        assert output["reader_evidence_quote"] == continuous
        assert "For macroscopic samples" in output["reader_evidence_quote"]
        assert "wave propagation" in output["reader_evidence_quote"]
        assert output["block_id"] == "blk-concept"
        assert output["anchor_id"] == "p-refocus"
        assert output["page_start"] == 2
        assert output["strict_locate"] is True


def test_final_display_cleanup_removes_empty_citation_wrappers_but_keeps_task_boxes() -> None:
    value = (
        "Claim [ [[CITE:nonexistent:1]] ].\n"
        "Nested [ [ [[CITE:nonexistent:2]] ] ].\n"
        "Literal [] remains.\n"
        "- [ ] keep this task\n"
        "1. [ ] keep this numbered task"
    )

    out = _normalize_chat_markdown_for_display(value)

    assert "Claim." in out
    assert "Nested." in out
    assert "Literal [] remains." in out
    assert "- [ ] keep this task" in out
    assert "1. [ ] keep this numbered task" in out

    assert _citation_free_answer_body(value) == _citation_free_answer_body(out)


def test_legacy_canonical_citations_recover_the_actual_answer_sources(tmp_path: Path) -> None:
    paths = []
    for idx in range(1, 6):
        path = tmp_path / f"paper-{idx}.en.md"
        if idx == 2:
            body = (
                "# Robust SPI\n\n## Results\n\nIn the domain shift test, the physical degradation model "
                "enables degradation-robust representations and the best reconstruction results.\n"
            )
        elif idx == 4:
                body = (
                    "# Real-time SPI\n\n## Results\n\nThe method uses 333 patterns and yields a reconstruction frame rate of "
                "30 Hz for 128 x 128 single-pixel video.\n"
            )
        elif idx == 5:
            body = (
                "# Transformer SPI\n\n## Real data\n\nThe network generalizes in both low-light and high-light "
                "conditions and improves image resolution.\n"
            )
        else:
            body = f"# Paper {idx}\n\nUnrelated background text for paper {idx}.\n"
        path.write_text(body, encoding="utf-8")
        paths.append(str(path))

    legacy_hits = [
        {"text": "seed", "meta": {"source_path": paths[idx]}}
        for idx in range(3)
    ]
    repaired = _augment_hits_with_canonical_answer_citations(
        legacy_hits,
        canonical_paths=paths,
        answer_text=(
            "Real-time reconstruction with 333 patterns [[4]].\n\n"
            "Low-light and high-light resolution gains [[5]].\n\n"
            "Domain shift degradation-robust generalization [[2]]."
        ),
    )

    cited = {
        int((hit.get("meta") or {}).get("ref_answer_citation_num") or 0): hit
        for hit in repaired
        if int((hit.get("meta") or {}).get("ref_answer_citation_num") or 0) > 0
    }
    assert set(cited) == {2, 4, 5}
    assert "30 Hz" in cited[4]["text"]
    assert "low-light" in cited[5]["text"]
    assert "degradation-robust" in cited[2]["text"]


def test_canonical_citation_rescues_existing_same_source_hit_to_claim_specific_block(tmp_path: Path) -> None:
    source = tmp_path / "robust.en.md"
    source.write_text(
        "# Robust SPI\n\n## Abstract\n\nA general framework for robust imaging.\n\n"
        "## Results\n\nAll real-world samples involving mist, jitter, and sensor noise achieve "
        "the lowest LPIPS score with the proposed method.\n",
        encoding="utf-8",
    )
    existing = {
        "text": "A general framework for robust imaging.",
        "meta": {
            "source_path": str(source),
            "heading_path": "Abstract",
            "ref_answer_citation_num": 1,
        },
    }

    repaired = _augment_hits_with_canonical_answer_citations(
        [existing],
        canonical_paths=[str(source)],
        answer_text="真实退化样本包含雾、抖动和传感器噪声，并取得最低 LPIPS [1]。",
    )

    assert len(repaired) == 1
    assert repaired[0]["meta"]["ref_answer_citation_num"] == 1
    assert "Results" in repaired[0]["meta"]["heading_path"]
    assert "lowest LPIPS" in repaired[0]["text"]


def test_canonical_citation_reuses_converged_strict_primary_without_rescan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "paper.en.md"
    source.write_text("# Paper\n\nExact evidence.\n", encoding="utf-8")
    hit = {
        "text": "Exact evidence.",
        "meta": {
            "source_path": str(source),
            "ref_answer_citation_num": 1,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": str(source),
                "heading_path": "Results",
                "snippet": "Exact evidence.",
                "strict_locate": True,
                "selection_reason": "answer_citation_grounded",
            }
        },
    }

    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail("converged primary should not rescan source blocks"),
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [hit],
        canonical_paths=[str(source)],
        answer_text="The result is exact [1].",
    )

    assert repaired == [hit]


def test_canonical_citation_reuses_exact_lineage_primary_without_rescan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "lineage.en.md"
    source.write_text("# Paper\n\nExact lineage evidence.\n", encoding="utf-8")
    hit = {
        "text": "Exact lineage evidence.",
        "meta": {
            "source_path": str(source),
            "ref_answer_citation_num": 1,
            "citation_plan_slot": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": str(source),
                "heading_path": "Method",
                "snippet": "Exact lineage evidence.",
                "block_id": "blk-lineage",
                "strict_locate": True,
                "selection_reason": "lineage_exact_source_block",
            }
        },
    }

    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail("exact lineage evidence should not rescan source blocks"),
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [hit],
        canonical_paths=[str(source)],
        answer_text="The method follows this lineage [1].",
    )

    assert repaired == [hit]


def test_canonical_citation_reuses_numbered_plan_slot_without_rescan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "paper.en.md"
    source.write_text("# Paper\n\nExact planned evidence.\n", encoding="utf-8")
    hit = {
        "text": "Exact planned evidence.",
        "meta": {
            "source_path": str(source),
            "ref_answer_citation_num": 1,
            "citation_plan_slot": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": str(source),
                "heading_path": "Results",
                "snippet": "Exact planned evidence.",
                "strict_locate": False,
                "selection_reason": "citation_plan_slot",
            }
        },
    }

    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail("numbered plan evidence should not rescan source blocks"),
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [hit],
        canonical_paths=[str(source)],
        answer_text="The result follows from the planned passage [1].",
    )

    assert repaired == [hit]


def test_canonical_citation_reuses_source_bound_authoritative_plan_without_rescan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "prompt-aligned.en.md"
    source.write_text("# Paper\n\nPrompt-aligned source evidence.\n", encoding="utf-8")
    hit = {
        "text": "Prompt-aligned source evidence.",
        "meta": {
            "source_path": str(source),
            "ref_answer_citation_num": 1,
            "citation_plan_slot": True,
            "citation_plan_evidence_authoritative": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": str(source),
                "heading_path": "Results",
                "snippet": "Prompt-aligned source evidence.",
                "strict_locate": False,
                "selection_reason": "prompt_aligned_source_sentence",
            }
        },
    }

    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail(
            "source-bound authoritative plan evidence should not rescan source blocks"
        ),
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [hit],
        canonical_paths=[str(source)],
        answer_text="The answer uses prompt-aligned evidence [1].",
    )

    assert repaired == [hit]


def test_canonical_citation_reuses_unique_authoritative_plan_across_number_reassignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "renumbered.en.md"
    source.write_text("# Paper\n\nUnique planned evidence.\n", encoding="utf-8")
    hit = {
        "text": "Unique planned evidence for the final claim.",
        "meta": {
            "source_path": str(source),
            "ref_answer_citation_num": 1,
            "citation_plan_slot": True,
            "citation_plan_evidence_authoritative": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": str(source),
                "snippet": "Unique planned evidence for the final claim.",
                "selection_reason": "prompt_aligned_source_sentence",
            }
        },
    }
    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail("unique same-source plan evidence must be reused"),
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [hit],
        canonical_paths=["a.md", "b.md", "c.md", str(source)],
        answer_text="The final answer cites the same planned paper under its final number [4].",
    )

    assert len(repaired) == 1
    assert repaired[0]["meta"]["ref_answer_citation_num"] == 4
    assert repaired[0]["meta"]["citation_plan_original_answer_citation_num"] == 1


def test_authoritative_doc_list_complete_plan_disables_answer_source_scan() -> None:
    from api import chat_render

    source = r"db\paper\paper.en.md"
    pack = {
        "pipeline_debug": {"doc_list_authoritative": True},
        "hits": [{"meta": {"source_path": source}}],
    }
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source,
                "evidence_quote": "A complete source-bound passage that grounded the answer.",
            }
        ]
    }

    assert chat_render._authoritative_doc_list_plan_covers_pack(pack, plan) is True
    nested_pack = {
        "hits": [{"meta": {"source_path": "stale.md"}}],
        "rendered_payload": pack,
    }
    assert chat_render._authoritative_doc_list_plan_covers_pack(nested_pack, plan) is True
    plan["slots"][0]["evidence_quote"] = "too short"
    assert chat_render._authoritative_doc_list_plan_covers_pack(pack, plan) is False


def test_authoritative_system_a_plan_covers_each_visible_answer_citation() -> None:
    from api import chat_render

    source = "review.en.md"
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": source,
                "heading_path": "Review / Abstract",
                "page_start": 1,
                "evidence_quote": (
                    "Deep learning provides exceptional reconstruction quality "
                    "and fast reconstruction speed."
                ),
                "evidence_selection_reason": "single_paper_comparison_facet",
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source,
                "heading_path": "Review / Risks",
                "page_start": 2,
                "evidence_quote": (
                    "Data-driven strategies have prolonged training duration "
                    "and limited generalization."
                ),
                "evidence_selection_reason": "single_paper_comparison_facet",
            },
        ]
    }

    assert (
        chat_render._authoritative_system_a_plan_covers_answer(
            plan,
            answer_text="The risk matters [1], while the benefit is direct [2].",
            canonical_paths=[source, source],
        )
        is True
    )
    assert (
        chat_render._authoritative_system_a_plan_covers_answer(
            plan,
            answer_text="The plan does not provide this third citation [3].",
            canonical_paths=[source, source, source],
        )
        is False
    )


def test_scope_boundary_abstract_plan_is_authoritative_without_seed_hit(tmp_path: Path) -> None:
    from api import chat_render

    source = tmp_path / "scope.en.md"
    evidence = "This paper studies the reconstruction scope of the proposed imaging method."
    source.write_text(f"# Scope\n\n## Abstract\n\n{evidence}\n", encoding="utf-8")
    plan = {
        "intent": "scope_boundary",
        "budget": {"system_a": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source),
                "source_name": "scope.pdf",
                "heading_path": "Scope / Abstract",
                "evidence_quote": evidence,
            }
        ],
    }

    hits = chat_render._augment_hits_with_system_a_plan_slots([], plan)

    assert len(hits) == 1
    assert hits[0]["meta"]["citation_plan_evidence_authoritative"] is True


def test_foveated_exact_generated_plan_is_authoritative_without_source_rescan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "foveated.en.md"
    evidence = (
        "A high-resolution foveal region tracks motion while every frame delivers "
        "new spatial information from across the entire field of view."
    )
    source.write_text(f"# Foveated\n\n## Abstract\n\n{evidence}\n", encoding="utf-8")
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [3],
                "source_path": str(source),
                "source_name": "Adaptive foveated SPI",
                "heading_path": "Adaptive foveated SPI / Abstract",
                "evidence_quote": evidence,
                "page_start": 1,
                "page_end": 1,
                "evidence_selection_reason": "exact_foveated_dynamic_supersampling_source",
            }
        ]
    }
    hits = chat_render._augment_hits_with_system_a_plan_slots(
        [
            {"text": "unrelated one", "meta": {"source_path": "one.en.md"}},
            {"text": "unrelated two", "meta": {"source_path": "two.en.md"}},
            {
                "text": "broad foveated seed",
                "meta": {
                    "source_path": str(source),
                    "ref_answer_citation_num": 3,
                },
                "ui_meta": {},
            },
        ],
        plan,
        reserved_count=3,
        canonical_paths=["one.en.md", "two.en.md", str(source)],
    )

    assert hits[2]["text"] == evidence
    assert hits[2]["meta"]["citation_plan_evidence_authoritative"] is True
    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail("exact foveated plan must not rescan the paper"),
    )
    repaired = chat_render._augment_hits_with_canonical_answer_citations(
        hits,
        canonical_paths=["one.en.md", "two.en.md", str(source)],
        answer_text="The foveal region still receives whole-field information [3].",
    )
    assert repaired[2]["text"] == evidence


def test_authoritative_answer_does_not_gain_unused_system_a_plan_citations() -> None:
    from api import chat_render

    sources = ["hadamard.en.md", "overview.en.md", "foveated.en.md"]
    foveated_evidence = (
        "Every frame delivers new spatial information from across the entire "
        "field of view while a high-resolution foveal region tracks motion."
    )
    plan = {
        "budget": {"system_a": 3},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": sources[0],
                "heading_path": "Hadamard / Abstract",
                "page_start": 1,
                "evidence_quote": (
                    "Hadamard and Fourier patterns provide different basis choices "
                    "for conventional single-pixel imaging."
                ),
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": sources[1],
                "heading_path": "Overview / Introduction",
                "page_start": 1,
                "evidence_quote": (
                    "Single-pixel imaging combines structured illumination with "
                    "a single-element detector."
                ),
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [3],
                "source_path": sources[2],
                "heading_path": "Foveated / Abstract",
                "page_start": 1,
                "evidence_quote": foveated_evidence,
                "evidence_selection_reason": (
                    "exact_foveated_dynamic_supersampling_source"
                ),
            },
        ],
    }
    hits = [
        {
            "text": str(slot["evidence_quote"]),
            "meta": {
                "source_path": str(slot["source_path"]),
                "ref_answer_citation_num": index,
            },
            "ui_meta": {},
        }
        for index, slot in enumerate(plan["slots"], start=1)
    ]
    answer = (
        "不完全对。动态超采样仍从整个视场获得新信息，只把更高分辨率集中在"
        "随运动更新的焦点区域 [3]。"
    )

    scoped = chat_render._scope_citation_plan_to_cited_system_a_sources(
        plan,
        answer_text=answer,
        canonical_paths=sources,
    )
    assert [
        slot["source_path"]
        for slot in scoped["slots"]
        if slot.get("preferred_system") == "system_a"
    ] == [sources[2]]

    repaired = chat_render._reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=sources,
    )
    assert set(re.findall(r"(?<![!\\])\[(\d+)\](?!\()", repaired)) == {"3"}


def test_single_paper_comparison_facets_remain_authoritative_without_rescan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "piln.en.md"
    method_evidence = (
        "ILNet is a self-supervised image-loop neural network with a part-based "
        "model for finer-grained learning."
    )
    positioning_evidence = (
        "The method embeds the single-pixel imaging model in an untrained network "
        "and optimizes it for each measurement."
    )
    source.write_text(
        f"# PILN\n\n## Abstract\n\n{method_evidence}\n\n{positioning_evidence}\n",
        encoding="utf-8",
    )
    slots = [
        {
            "preferred_system": "system_a",
            "candidate_hits": [1],
            "source_path": str(source),
            "source_name": "PILN",
            "heading_path": "PILN / Abstract",
            "evidence_quote": method_evidence,
            "page_start": 1,
            "evidence_selection_reason": "single_paper_comparison_facet",
        },
        {
            "preferred_system": "system_a",
            "candidate_hits": [2],
            "source_path": str(source),
            "source_name": "PILN",
            "heading_path": "PILN / Introduction",
            "evidence_quote": positioning_evidence,
            "page_start": 2,
            "evidence_selection_reason": "single_paper_comparison_facet",
        },
    ]
    hits = chat_render._augment_hits_with_system_a_plan_slots(
        [
            {
                "text": "generic method seed",
                "meta": {"source_path": str(source), "ref_answer_citation_num": 1},
                "ui_meta": {},
            },
            {
                "text": "generic positioning seed",
                "meta": {"source_path": str(source), "ref_answer_citation_num": 2},
                "ui_meta": {},
            },
        ],
        {"slots": slots},
        reserved_count=2,
        canonical_paths=[str(source), str(source)],
    )

    assert [hit["text"] for hit in hits] == [method_evidence, positioning_evidence]
    assert all(
        hit["meta"]["citation_plan_evidence_authoritative"] is True
        for hit in hits
    )
    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail("comparison facets must not rescan the paper"),
    )
    repaired = chat_render._augment_hits_with_canonical_answer_citations(
        hits,
        canonical_paths=[str(source), str(source)],
        answer_text="PILN is self-supervised [1] and model-driven per measurement [2].",
    )
    assert [hit["text"] for hit in repaired] == [method_evidence, positioning_evidence]


def test_single_paper_facets_rebind_stale_same_source_occurrence_numbers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "review.en.md"
    benefit = "Deep learning provides exceptional reconstruction quality and fast reconstruction speed."
    risk = "Data-driven strategies have prolonged training duration and limited generalization."
    source.write_text(
        f"# Review\n\n## Abstract\n\n{benefit}\n\n## Risks\n\n{risk}\n",
        encoding="utf-8",
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": str(source),
                "source_name": "Review",
                "heading_path": "Review / Abstract",
                "evidence_quote": benefit,
                "page_start": 1,
                "evidence_selection_reason": "single_paper_comparison_facet",
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": str(source),
                "source_name": "Review",
                "heading_path": "Review / Risks",
                "evidence_quote": risk,
                "page_start": 2,
                "evidence_selection_reason": "single_paper_comparison_facet",
            },
        ]
    }

    hits = chat_render._augment_hits_with_system_a_plan_slots(
        [
            {
                "text": "compact seed carrying the other occurrence number",
                "meta": {
                    "source_path": str(source),
                    "ref_answer_citation_num": 2,
                },
                "ui_meta": {},
            }
        ],
        plan,
        reserved_count=2,
        canonical_paths=[str(source), str(source)],
        answer_text="The risk matters [1], while the benefit is direct [2].",
    )

    assert len(hits) == 2
    assert [hit["meta"]["ref_answer_citation_num"] for hit in hits] == [1, 2]
    assert [hit["text"] for hit in hits] == [risk, benefit]
    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit same-source occurrences must not rescan the paper"
        ),
    )
    repaired = chat_render._augment_hits_with_canonical_answer_citations(
        hits,
        canonical_paths=[str(source), str(source)],
        answer_text="The risk matters [1], while the benefit is direct [2].",
    )
    assert [hit["text"] for hit in repaired] == [risk, benefit]


def test_canonical_citation_incomplete_authoritative_plan_still_scans_and_recovers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "recovery.en.md"
    source.write_text(
        "# Recovery\n\n## Abstract\n\nBroad background.\n\n"
        "## Results\n\nThe method reconstructs single-pixel video at 30 Hz using 333 patterns.\n",
        encoding="utf-8",
    )
    other_source = tmp_path / "other.en.md"
    hit = {
        "text": "An incomplete prompt-aligned passage.",
        "meta": {
            "source_path": str(source),
            "ref_answer_citation_num": 1,
            "citation_plan_slot": True,
            "citation_plan_evidence_authoritative": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": str(other_source),
                "snippet": "An incomplete prompt-aligned passage.",
                "strict_locate": False,
                "selection_reason": "prompt_aligned_source_sentence",
            }
        },
    }
    real_load_source_blocks = chat_render.task_runtime.load_source_blocks
    calls: list[Path] = []

    def _tracked_load_source_blocks(path: Path, *args, **kwargs):
        calls.append(Path(path))
        return real_load_source_blocks(path, *args, **kwargs)

    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        _tracked_load_source_blocks,
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [hit],
        canonical_paths=[str(source)],
        answer_text="The method reconstructs at 30 Hz using 333 patterns [1].",
    )

    assert calls == [source]
    assert repaired[0]["meta"]["ref_answer_citation_num"] == 1
    assert repaired[0]["meta"]["source_path"] == str(source)
    assert repaired[0]["meta"]["ref_display_reason"] == "canonical_answer_repair"
    assert "30 Hz" in repaired[0]["text"]


def test_canonical_citation_seeds_persisted_answer_evidence_before_legacy_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "compacted-answer-source.en.md"
    source.write_text("# Paper\n\nA much larger source that should not be rescanned.\n", encoding="utf-8")
    evidence = {
        "text": "The compact answer context reports a 30 Hz reconstruction rate.",
        "meta": {
            "source_path": str(source),
            "source_name": "Compacted Answer Source.pdf",
            "heading_path": "Results / Runtime",
            "block_id": "blk-runtime",
            "anchor_id": "p-runtime",
            "page_start": 7,
        },
    }

    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: [],
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [],
        canonical_paths=[str(source)],
        canonical_evidence=[evidence],
        answer_text="The method reaches 30 Hz [1].",
    )

    assert len(repaired) == 1
    assert repaired[0]["meta"]["ref_answer_citation_num"] == 1
    assert repaired[0]["meta"]["canonical_answer_evidence"] is True
    assert repaired[0]["ui_meta"]["primary_evidence"]["strict_locate"] is True
    assert repaired[0]["ui_meta"]["primary_evidence"]["block_id"] == "blk-runtime"


def test_canonical_citation_marks_matching_plan_hit_before_legacy_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    source = tmp_path / "planned-answer-source.en.md"
    source.write_text("# Paper\n\nThe source should not be rescanned.\n", encoding="utf-8")
    existing = {
        "text": "The prompt-aligned passage explains the model-driven strategy.",
        "meta": {
            "source_path": str(source),
            "ref_answer_citation_num": 1,
            "citation_plan_slot": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": str(source),
                "snippet": "The prompt-aligned passage explains the model-driven strategy.",
                "selection_reason": "prompt_aligned_source_sentence",
                "strict_locate": False,
            }
        },
    }
    canonical = {
        "text": "The answer context also explains the model-driven strategy in detail.",
        "meta": {"source_path": str(source)},
    }

    monkeypatch.setattr(
        chat_render.task_runtime,
        "load_source_blocks",
        lambda *_args, **_kwargs: [],
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [existing],
        canonical_paths=[str(source)],
        canonical_evidence=[canonical],
        answer_text="This is a model-driven strategy [1].",
    )

    assert len(repaired) == 1
    assert repaired[0]["meta"]["canonical_answer_evidence"] is True
    assert repaired[0]["text"] == existing["text"]


def test_canonical_citation_combines_distinct_blocks_for_one_multi_signal_claim(tmp_path: Path) -> None:
    source = tmp_path / "hatnet.en.md"
    source.write_text(
        "# HATNet\n\n## Results / Optical resolution\n\n"
        "Full sampling reconstructs a 64 x 64 image, while HATNet reconstructs a 256 x 256 image at the same data throughput.\n\n"
        "## Results / Illumination robustness\n\n"
        "HATNet shows strong generalization ability in both low-light and high-light conditions.\n",
        encoding="utf-8",
    )

    repaired = _augment_hits_with_canonical_answer_citations(
        [],
        canonical_paths=[str(source)],
        answer_text=(
            "The method improves optical resolution from 64 x 64 to 256 x 256 and also "
            "generalizes in low-light and high-light conditions [1]."
        ),
    )

    assert len(repaired) == 1
    assert repaired[0]["meta"]["ref_answer_citation_num"] == 1
    assert "64 x 64" in repaired[0]["text"]
    assert "low-light" in repaired[0]["text"]


def test_equation_source_note_does_not_reference_removed_refs_ui():
    messages = [
        {"id": 1, "role": "user", "content": "NatPhoton 公式 8 是什么？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "$$\nI_{TC}=x \\tag{8}\n$$",
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Equation (8) defines the total-curvature objective.",
                    "meta": {
                        "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    body = str(rendered[-1].get("rendered_body") or "")

    assert "Open/Page" not in body
    assert MOJIBAKE_REFERENCE_LOCATOR not in body
    assert "库内文献" in body
    assert "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf" in body


def test_research_basket_synthetic_citation_uses_friendly_non_openable_detail():
    synthetic_path = "__research_basket__/item_1_deadbeef"
    messages = [
        {"id": 1, "role": "user", "content": "Use selected item"},
        {
            "id": 2,
            "role": "assistant",
            "content": "This is supported by the selected item [1].",
            "meta": {"canonical_hit_paths": [synthetic_path]},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Title: A hard to find preprint\nDOI: 10.1234/example.1\nSummary: selected metadata",
                    "score": 999.0,
                    "meta": {
                        "source_path": synthetic_path,
                        "source_name": "Research basket: A hard to find preprint",
                        "title": "A hard to find preprint",
                        "doi": "10.1234/example.1",
                        "ref_pack_state": "ready",
                        "research_basket_evidence": True,
                        "basket_source_role": "synthetic_basket_item",
                    },
                    "ui_meta": {
                        "display_name": "Research basket: A hard to find preprint",
                        "can_open": False,
                    },
                }
            ],
            "display_state": "ready",
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-basket")
    detail = rendered[-1]["cite_details"][0]

    assert detail["source_name"] == "Research basket: A hard to find preprint"
    assert detail["source_path"] == ""
    assert detail["citation_route"] == "research_basket"
    assert detail["routing_reason"] == "research_basket_evidence"
    assert detail["location_label"] == "Research basket"
    assert "item_1_deadbeef" not in json.dumps(detail, ensure_ascii=False)


def test_normalize_chat_markdown_cleans_empty_example_connectors_and_duplicate_terms():
    raw = (
        "This review (for example or this survey [2]) is a good entry point.\n\n"
        "The topic includes single-pixel imaging, single-pixel imaging."
    )

    rendered = _normalize_chat_markdown_for_display(raw)

    assert "for example or" not in rendered
    assert "single-pixel imaging, single-pixel imaging" not in rendered
    assert "This review (this survey [2]) is a good entry point." in rendered
    assert "The topic includes single-pixel imaging." in rendered


def test_equation_source_note_is_not_added_without_hits():
    messages = [
        {"id": 1, "role": "user", "content": "NatPhoton 公式 8 是什么？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "$$\nI_{TC}=x \\tag{8}\n$$",
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    body = str(rendered[-1].get("rendered_body") or "")

    assert "库内文献" not in body


def test_normalize_equation_source_notes_strips_mojibake_prefix_from_pdf_label():
    raw = (
        "*（式(1) 对应命中的库内文献："
        f"`1) {MOJIBAKE_REFERENCE_SOURCE_PREFIX}CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf`）*"
    )

    out = _normalize_equation_source_notes(raw)

    assert MOJIBAKE_REFERENCE_LOCATOR not in out
    assert "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf" in out
    assert "`1) " not in out


def test_copy_outputs_and_rendered_content_are_consistent():
    messages = [
        {"id": 1, "role": "user", "content": "请解释这个结论？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "结论见 [[CITE:s1a2b3c4:12]]，并可对比 [CITE:s1a2b3c4:13]。",
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    rendered_content = str(msg.get("rendered_content") or "")
    copy_markdown = str(msg.get("copy_markdown") or "")
    copy_text = str(msg.get("copy_text") or "")

    assert "[[CITE:" not in rendered_content
    assert "[CITE:" not in rendered_content
    assert "[[CITE:" not in copy_markdown
    assert "[CITE:" not in copy_markdown
    assert "结论见" in copy_text


def test_rendered_body_falls_back_to_content_when_no_notice():
    messages = [
        {"id": 1, "role": "user", "content": "hello"},
        {"id": 2, "role": "assistant", "content": ""},
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    assert str(msg.get("notice") or "") == ""
    assert str(msg.get("rendered_body") or "") == str(msg.get("rendered_content") or "")


def test_render_packet_contract_is_backfilled_from_rendered_message():
    messages = [
        {"id": 1, "role": "user", "content": "explain this"},
        {
            "id": 2,
            "role": "assistant",
            "content": "APR uses phase correlation [[CITE:s1234abcd:3]].",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "text": "APR uses phase correlation for registration.",
                        "locate_policy": "required",
                        "primary_heading_path": "Methods / APR",
                        "primary_block_id": "b-7",
                        "primary_anchor_id": "a-7",
                        "anchor_kind": "paragraph",
                        "claim_type": "method_claim",
                    }
                ],
            },
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "method"},
                    "render_packet": {"citation_validation": {"kept": 1}},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {"source_path": r"db\doc\doc.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert packet["answer_markdown"] == "APR uses phase correlation [[CITE:s1234abcd:3]]."
    assert packet["rendered_body"]
    assert packet["copy_text"]
    assert packet["citation_validation"] == {"kept": 1}
    assert packet["locate_target"]["segmentId"] == "seg-1"
    assert packet["reader_open"]["blockId"] == "b-7"
    assert packet["segment_ids"] == ["seg-1"]
    assert packet["visible_segment_ids"] == ["seg-1"]


def test_existing_render_packet_citation_cards_are_refreshed() -> None:
    messages = [
        {"id": 1, "role": "user", "content": "How should I read these papers?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "A reading route is available.",
            "meta": {
                "answer_quality": {"output_mode": "citation"},
                "paper_guide_contracts": {
                    "version": 1,
                    "render_packet": {
                        "answer_markdown": "A reading route is available.",
                        "rendered_body": "A reading route is available.",
                        "copy_markdown": "A reading route is available.",
                        "copy_text": "A reading route is available.",
                        "cite_details": [
                            {
                                "num": 1,
                                "anchor": "roadmap-a1",
                                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                                "source_path": "hsi-fsi.en.md",
                                "heading_path": "Experiment design / Coding choice",
                                "answer_claim": (
                                    "\u518d\u8bfb\u65b9\u6cd5\u5bf9\u6bd4\uff1a"
                                    "\u300aHadamard single-pixel imaging versus Fourier single-pixel imaging\u300b "
                                    "(Optics Express, 2017)"
                                ),
                                "evidence_quote": (
                                    "Hadamard basis patterns are binary, which makes HSI naturally suitable "
                                    "for single-pixel imaging systems based on digital micromirror devices."
                                ),
                                "location_label": "Experiment design / Coding choice",
                            }
                        ],
                    },
                },
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-refresh-card")
    packet = (((rendered[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert packet["cite_details"] == []
    assert "roadmap-a1" not in str(packet.get("rendered_body") or "")


def test_non_paper_guide_message_preserves_minimal_primary_evidence_contract():
    messages = [
        {"id": 1, "role": "user", "content": "Which paper compares Hadamard and Fourier single-pixel imaging?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "OE-2017 directly compares Hadamard and Fourier single-pixel imaging.",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "primary_evidence": {
                        "source_name": "OE-2017.pdf",
                        "block_id": "blk_22",
                        "anchor_id": "a_22",
                        "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                        "snippet": "Section 2.2 explicitly compares the two methods in terms of basis pattern properties.",
                    },
                    "render_packet": {
                        "answer_markdown": "OE-2017 directly compares Hadamard and Fourier single-pixel imaging.",
                        "primary_evidence": {
                            "source_name": "OE-2017.pdf",
                            "block_id": "blk_22",
                            "anchor_id": "a_22",
                            "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                            "snippet": "Section 2.2 explicitly compares the two methods in terms of basis pattern properties.",
                        },
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-normal")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert packet["answer_markdown"] == "OE-2017 directly compares Hadamard and Fourier single-pixel imaging."
    assert packet["primary_evidence"]["block_id"] == "blk_22"
    assert packet["primary_evidence"]["heading_path"] == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_enrich_messages_uses_rendered_payload_primary_evidence_from_stored_refs_row():
    messages = [
        {"id": 1, "role": "user", "content": "Besides this paper, what other papers discuss Fourier single-pixel imaging?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "A coarse answer seeded from the bound paper.",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "primary_evidence": {
                        "source_name": "NatPhoton-2019.pdf",
                        "heading_path": "Abstract / Camera architecture",
                        "selection_reason": "answer_hit_top",
                    },
                    "render_packet": {},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [],
            "rendered_payload": {
                "hits": [
                    {
                        "ui_meta": {
                            "reader_open": {
                                "sourcePath": "oe2017.md",
                                "headingPath": "2. Comparison of theory / 2.2 Basis patterns generation",
                                "blockId": "blk_22",
                            }
                        }
                    }
                ],
                "primary_evidence": {
                    "source_path": "oe2017.md",
                    "source_name": "OE-2017.pdf",
                    "block_id": "blk_22",
                    "anchor_id": "a_22",
                    "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                    "selection_reason": "prompt_aligned",
                },
                "render_status": "full",
            },
        }
    }

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user=refs_by_user,
        conv_id="conv-cross-paper",
        render_packet_only=True,
    )
    msg = rendered[-1]
    contracts = (((msg.get("meta") or {}).get("paper_guide_contracts")) or {})
    packet = contracts.get("render_packet") or {}

    assert (contracts.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (contracts.get("primary_evidence") or {}).get("block_id") == "blk_22"
    assert (packet.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (packet.get("primary_evidence") or {}).get("heading_path") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_existing_render_packet_preserves_compat_render_fields_when_current_render_degrades():
    messages = [
        {"id": 1, "role": "user", "content": "explain this"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "created_at": 1,
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                    "render_packet": {
                        "answer_markdown": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
                        "rendered_body": "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
                        "rendered_content": "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
                        "copy_markdown": "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
                        "copy_text": "SPI relies on compressive sensing [1].",
                        "cite_details": [
                            {
                                "num": 1,
                                "anchor": "kb-cite-demo-1",
                                "source_name": "demo.pdf",
                                "source_path": "demo.md",
                                "raw": "Demo reference [1]",
                            }
                        ],
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert "#kb-cite-demo-1" not in str(msg.get("rendered_body") or "")
    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert msg.get("cite_details") == []
    assert packet.get("rendered_body") == msg.get("rendered_body")
    assert packet.get("cite_details") == []


def test_render_packet_replaces_stale_primary_jump_target_when_current_provenance_is_better():
    messages = [
        {"id": 1, "role": "user", "content": "Which ADMM citation is this?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "The paper cites [4] for this point.\n> most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "segments": [
                    {
                        "segment_id": "seg-2",
                        "text": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                        "locate_policy": "required",
                        "primary_heading_path": "Related Work / Snapshot Compressive Imaging",
                        "primary_block_id": "b-right",
                        "primary_anchor_id": "a-right",
                        "anchor_kind": "blockquote",
                        "claim_type": "prior_work",
                        "support_slot_claim_type": "prior_work",
                        "support_locate_anchor": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],",
                        "resolved_ref_num": 4,
                    }
                ],
            },
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                    "render_packet": {
                        "answer_markdown": "The paper cites [4] for this point.",
                        "rendered_body": "The paper cites [4] for this point.",
                        "rendered_content": "The paper cites [4] for this point.",
                        "copy_markdown": "The paper cites [4] for this point.",
                        "copy_text": "The paper cites [4] for this point.",
                        "locate_target": {
                            "segmentId": "seg-wrong",
                            "headingPath": "Method / Wrong Section",
                            "snippet": "A generic method sentence unrelated to this citation.",
                            "anchorText": "A generic method sentence unrelated to this citation.",
                            "blockId": "b-wrong",
                            "anchorId": "a-wrong",
                        },
                        "reader_open": {
                            "sourcePath": "demo.md",
                            "headingPath": "Method / Wrong Section",
                            "snippet": "A generic method sentence unrelated to this citation.",
                            "blockId": "b-wrong",
                            "anchorId": "a-wrong",
                            "strictLocate": True,
                        },
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert packet["locate_target"]["segmentId"] == "seg-2"
    assert packet["locate_target"]["blockId"] == "b-right"
    assert "alternating direction method of multipliers" in str(packet["locate_target"]["snippet"]).lower()
    assert packet["reader_open"]["blockId"] == "b-right"


def test_sid_markers_are_removed_from_rendered_outputs():
    messages = [
        {"id": 1, "role": "user", "content": "解释单像素成像？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "[SID:s50f9c165] 这是内部标记，不应该展示给用户。",
        },
    ]

    rendered = enrich_messages_with_reference_render(messages, refs_by_user={}, conv_id="conv-test")
    msg = rendered[-1]
    rendered_content = str(msg.get("rendered_content") or "")
    copy_markdown = str(msg.get("copy_markdown") or "")
    copy_text = str(msg.get("copy_text") or "")

    assert "[SID:" not in rendered_content
    assert "[SID:" not in copy_markdown
    assert "[SID:" not in copy_text


def test_structured_cite_fallback_does_not_relink_after_safe_downgrade(monkeypatch):
    from api import chat_render

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _md, _hits, anchor_ns, canonical_paths
        # Simulate safety downgrade result from primary annotator:
        # CITE token resolved to plain numeric marker and no details.
        return "Gehm et al. (2007) [24].", []

    def fake_fallback(*args, **kwargs):
        raise AssertionError("fallback should not run after safe downgrade")

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    monkeypatch.setattr(chat_render, "_fallback_render_structured_citations", fake_fallback)

    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Gehm et al. (2007) [[CITE:s1234abcd:24]].",
            "meta": {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    assert "[24]" in str(msg.get("rendered_body") or "")
    assert msg.get("cite_details") == []


def test_structured_cite_fallback_recovers_links_when_primary_strips_tokens(monkeypatch):
    from api import chat_render

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _md, _hits, anchor_ns, canonical_paths
        return "SPI relies on compressive sensing.", []

    def fake_fallback(_md, _hits, *, anchor_ns=""):
        del _md, _hits, anchor_ns
        return (
            "SPI relies on compressive sensing [1](#kb-cite-demo-1).",
            [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    monkeypatch.setattr(chat_render, "_fallback_render_structured_citations", fake_fallback)

    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    assert "[1](#kb-cite-demo-1)" in str(msg.get("rendered_body") or "")
    assert len(msg.get("cite_details") or []) == 1


def test_normal_answer_does_not_auto_link_freeform_numeric_markers_from_refs_hits():
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": (
                "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf [2].\n"
                "Section 2.2 compares the two methods [2]."
            ),
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    assert "[2]" in str(msg.get("rendered_body") or "")
    assert "[2]" in str(msg.get("rendered_content") or "")
    assert msg.get("cite_details") == []


def test_normal_answer_strips_structured_cite_markers_without_linking():
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": (
                "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf [[CITE:demo:2]].\n"
                "Section 2.2 compares the two methods [[CITE:demo:2]]."
            ),
            "meta": {
                "answer_quality": {
                    "prompt_family": "overview",
                    "output_mode": "reading_guide",
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {
                        "source_path": r"db\doc\doc.en.md",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]
    rendered_body = str(msg.get("rendered_body") or "")
    assert "[[CITE:" not in rendered_body
    assert "[2]" not in rendered_body
    assert msg.get("cite_details") == []


def test_normal_answer_preserves_validated_system_b_marker(monkeypatch):
    from api import chat_render

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None):
        del hits, anchor_ns, canonical_paths
        assert "[[CITE:s1234abcd:4]]" in md
        return (
            "ADMM is prior optimization machinery [4](#kb-cite-demo-4).",
            [
                {
                    "num": 4,
                    "anchor": "kb-cite-demo-4",
                    "source_name": "SCINeRF.pdf",
                    "source_path": r"db\demo\scinerf.en.md",
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "is_inpaper": True,
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    messages = [
        {"id": 1, "role": "user", "content": "ADMM 是作者自己发明的吗？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "ADMM is prior optimization machinery [[CITE:s1234abcd:4]].",
            "meta": {
                "answer_quality": {
                    "prompt_family": "overview",
                    "output_mode": "reading_guide",
                    "reference_opportunities": {"count": 1, "mode": "inline", "refs": [4]},
                    "citation_validation": {"raw_count": 1, "kept": 1, "rewritten": 0},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {"source_path": r"db\demo\scinerf.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-normal-sysb")
    msg = rendered[-1]

    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert "#kb-cite-demo-4" in str(msg.get("rendered_body") or "")
    assert (msg.get("cite_details") or [])[0]["is_inpaper"] is True


def test_normal_upstream_question_can_route_structured_marker_to_system_b_without_validation(monkeypatch):
    from api import chat_render

    calls = []

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None):
        del hits, anchor_ns, canonical_paths
        calls.append(md)
        return (
            "ADMM comes from prior optimization work [4](#kb-cite-demo-4).",
            [
                {
                    "num": 4,
                    "anchor": "kb-cite-demo-4",
                    "source_name": "SCINeRF.pdf",
                    "source_path": r"db\demo\scinerf.en.md",
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "is_inpaper": True,
                    "citation_route": "system_b",
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    messages = [
        {"id": 1, "role": "user", "content": "ADMM 是怎么来的？作者是不是借鉴了前人的方法？"},
        {
            "id": 2,
            "role": "assistant",
            "content": "ADMM comes from prior optimization work [[CITE:s1234abcd:4]].",
            "meta": {"answer_quality": {"prompt_family": "overview", "output_mode": "reading_guide"}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {"source_path": r"db\demo\scinerf.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-normal-sysb-unvalidated")
    msg = rendered[-1]

    assert calls
    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert "#kb-cite-demo-4" in str(msg.get("rendered_body") or "")
    assert (msg.get("cite_details") or [])[0]["is_inpaper"] is True


def test_citation_plan_allows_normal_question_to_render_system_b_without_validation(monkeypatch):
    from api import chat_render

    calls = []

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None, citation_plan=None):
        del hits, anchor_ns, canonical_paths
        calls.append({"md": md, "citation_plan": citation_plan})
        return (
            "ADMM is the optimization background [4](#kb-cite-demo-4).",
            [
                {
                    "num": 4,
                    "anchor": "kb-cite-demo-4",
                    "source_name": "SCINeRF.pdf",
                    "source_path": r"db\demo\scinerf.en.md",
                    "title": "Distributed optimization and statistical learning via ADMM",
                    "is_inpaper": True,
                    "citation_route": "system_b",
                    "routing_reason": "citation_plan",
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    plan = {
        "intent": "beginner_overview",
        "budget": {"system_a": 2, "system_b": 1},
        "system_b_enabled": True,
        "slots": [{"preferred_system": "system_b", "candidate_refs": [4]}],
    }
    messages = [
        {"id": 1, "role": "user", "content": "ADMM 这个东西我不太懂，简单说一下。"},
        {
            "id": 2,
            "role": "assistant",
            "content": "ADMM is the optimization background [[CITE:s1234abcd:4]].",
            "meta": {"answer_quality": {"prompt_family": "overview", "citation_plan": plan}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Most existing methods employ ADMM [4].",
                    "meta": {"source_path": r"db\demo\scinerf.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-citation-plan-sysb")
    msg = rendered[-1]

    assert calls
    assert calls[0]["citation_plan"]["intent"] == "beginner_overview"
    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert "#kb-cite-demo-4" in str(msg.get("rendered_body") or "")
    assert (msg.get("cite_details") or [])[0]["routing_reason"] == "citation_plan"


def test_named_upstream_title_is_linked_from_current_reference_index(monkeypatch):
    from api import chat_render
    from ui import refs_renderer

    source_path = r"db\paper\paper.en.md"
    source_key = chat_render._render_norm_source_key(source_path)
    index_data = {
        "docs": {
            source_key: {
                "path": source_path,
                "name": "paper.pdf",
                "sha1": "abc",
                "refs": {
                    "24": {
                        "authors": "Jiang X, Li Z, Du G",
                        "venue": "Optics Express",
                        "year": "2022",
                        "doi": "10.1364/oe.458742",
                        "title": "Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
                        "raw": (
                            "[24] Jiang X, Li Z, Du G. Fast hyperspectral single-pixel imaging via "
                            "frequency-division multiplexed illumination. Optics Express, 2022. "
                            "doi:10.1364/oe.458742"
                        ),
                    }
                },
            }
        }
    }

    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: index_data)
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: index_data)

    messages = [
        {"id": 1, "role": "user", "content": "What else should I read?"},
        {
            "id": 2,
            "role": "assistant",
            "content": (
                "You can compare against Fast hyperspectral single-pixel imaging via "
                "frequency-division multiplexed illumination for real-time SPI context."
            ),
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "The current paper cites fast hyperspectral SPI reconstruction [24].",
                    "meta": {"source_path": source_path, "source_sha1": "abc"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-title-sysb")
    msg = rendered[-1]
    body = str(msg.get("rendered_body") or "")
    details = list(msg.get("cite_details") or [])

    assert "[[CITE:" not in body
    assert "[24](#kb-cite-" in body
    assert len(details) == 1
    assert details[0]["is_inpaper"] is True
    assert details[0]["doi"] == "10.1364/oe.458742"
    assert "frequency-division multiplexed illumination" in details[0]["title"]


def test_named_upstream_title_repair_does_not_link_short_venue_mentions(monkeypatch):
    from api import chat_render

    source_path = r"db\paper\paper.en.md"
    index_data = {
        "docs": {
            chat_render._render_norm_source_key(source_path): {
                "path": source_path,
                "name": "paper.pdf",
                "refs": {
                    "24": {
                        "title": "Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
                        "raw": "[24] Jiang X. Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination. Optics Express, 2022.",
                    }
                },
            }
        }
    }
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: index_data)

    repaired, changed = chat_render._repair_named_system_b_citation_markers(
        "For comparison, the Optica 2024 work is also relevant.",
        [{"text": "hit", "meta": {"source_path": source_path}}],
        {"budget": {"system_b": 2}},
    )

    assert changed is False
    assert "[[CITE:" not in repaired


def test_named_upstream_title_repair_does_not_duplicate_a_current_library_source(monkeypatch):
    from api import chat_render

    citing_path = r"db\video\Journal of Optics-2016-3D single-pixel video.en.md"
    current_path = (
        r"db\review\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    )
    index_data = {
        "docs": {
            chat_render._render_norm_source_key(citing_path): {
                "path": citing_path,
                "name": "3D single-pixel video.pdf",
                "refs": {
                    "11": {
                        "title": "Principles and prospects for single-pixel imaging",
                        "raw": "[11] Principles and prospects for single-pixel imaging.",
                    }
                },
            }
        }
    }
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: index_data)

    repaired, changed = chat_render._repair_named_system_b_citation_markers(
        "Read Principles and prospects for single-pixel imaging first.",
        [
            {"text": "citing hit", "meta": {"source_path": citing_path}},
            {"text": "review hit", "meta": {"source_path": current_path}},
        ],
        {"budget": {"system_b": 2}},
    )

    assert changed is False
    assert "[[CITE:" not in repaired


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "zh_overview_freeform_numeric",
            "user": "这篇论文的核心方法是什么？",
            "assistant": "该方法把快照压缩成像和 NeRF 训练结合起来 [1]，用于从单帧压缩观测恢复三维表示。",
            "meta": {"answer_quality": {"prompt_family": "overview", "output_mode": "reading_guide"}},
        },
        {
            "name": "en_comparison_numeric_range",
            "user": "Which paper compares Hadamard and Fourier single-pixel imaging?",
            "assistant": "OE-2017 compares Hadamard and Fourier single-pixel imaging [2, 3], but this is a normal library answer.",
            "meta": {"answer_quality": {"prompt_family": "compare", "output_mode": "reading_guide"}},
        },
        {
            "name": "zh_method_structured_marker",
            "user": "它是怎么训练 NeRF 的？",
            "assistant": "论文把物理成像过程写进训练目标 [[CITE:s1234abcd:4]]，但普通方法问答不应保留文内参考链接。",
            "meta": {"answer_quality": {"prompt_family": "method", "output_mode": "reading_guide"}},
        },
        {
            "name": "no_meta_source_like_numeric",
            "user": "给我正常概括一下这篇文献。",
            "assistant": "The answer mentions a source-like marker [5] but has no citation intent metadata.",
            "meta": {},
        },
    ],
    ids=lambda case: str(case.get("name") or "case"),
)
def test_normal_question_variants_do_not_trigger_inpaper_reference_links(case):
    messages = [
        {"id": 1, "role": "user", "content": case["user"]},
        {
            "id": 2,
            "role": "assistant",
            "content": case["assistant"],
            "meta": case["meta"],
        },
    ]
    name = str(case.get("name") or "")
    hit_text = "retrieved evidence"
    hit_meta = {"source_path": r"db\doc\doc.en.md"}
    if name == "zh_overview_freeform_numeric":
        hit_text = (
            "Snapshot Compressive Imaging (SCI) is combined with NeRF training "
            "to recover a 3D scene representation from a compressed observation."
        )
        hit_meta = {
            "source_path": r"db\doc\scinerf.en.md",
            "heading_path": "Abstract",
            "evidence_quote": hit_text,
        }
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": hit_text,
                    "meta": hit_meta,
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id=f"conv-{name}")
    msg = rendered[-1]
    rendered_body = str(msg.get("rendered_body") or "")
    rendered_content = str(msg.get("rendered_content") or "")
    copy_markdown = str(msg.get("copy_markdown") or "")
    cite_details = list(msg.get("cite_details") or [])

    # Structured [[CITE:...]] markers are always stripped in non-paper-guide mode.
    assert "[[CITE:" not in rendered_body
    assert "[CITE:" not in rendered_body
    # Cases with resolvable [n] markers (n <= hit count) get linked.
    # Cases with unresolvable [2,3] or [5] (only 1 hit) or structured markers get stripped.
    if name == "zh_overview_freeform_numeric":
        assert "#kb-cite-" in rendered_body
        assert len(cite_details) > 0
    else:
        assert "#kb-cite-" not in rendered_body
        assert cite_details == []


@pytest.mark.parametrize(
    "meta",
    [
        {"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
        {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "reading_guide"}},
        {"answer_quality": {"prompt_family": "overview", "output_mode": "citation_lookup"}},
    ],
    ids=["contract_intent", "answer_prompt_family", "answer_output_mode"],
)
def test_citation_lookup_variants_trigger_and_preserve_inpaper_reference_links(monkeypatch, meta):
    from api import chat_render

    calls = []

    def fake_primary(md, hits, *, anchor_ns="", canonical_paths=None):
        calls.append({"md": md, "hits": hits, "anchor_ns": anchor_ns, "canonical_paths": canonical_paths})
        return (
            "SCI relies on compressive sensing [1](#kb-cite-demo-1).",
            [
                {
                    "num": 1,
                    "anchor": "kb-cite-demo-1",
                    "source_name": "demo.pdf",
                    "source_path": "demo.md",
                    "raw": "Demo reference [1]",
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)

    messages = [
        {"id": 1, "role": "user", "content": "Which in-paper reference supports SCI?"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SCI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": meta,
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "dummy",
                    "meta": {"source_path": r"db\doc\doc.en.md"},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-citation-lookup")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert calls
    assert "[1](#kb-cite-demo-1)" in str(msg.get("rendered_body") or "")
    assert "[[CITE:" not in str(msg.get("rendered_body") or "")
    assert len(msg.get("cite_details") or []) == 1
    assert packet.get("rendered_body") == msg.get("rendered_body")
    assert len(packet.get("cite_details") or []) == 1


def test_citation_lookup_rendered_link_points_to_validated_target_reference(monkeypatch):
    from ui import refs_renderer

    source_path = r"db\doc\paper.en.md"
    sid = refs_renderer._source_cite_id(source_path)

    refs = {
        1: {
            "authors": "Wrong A",
            "year": "2020",
            "doi": "10.1000/wrong",
            "title": "Wrong Reference",
            "raw": "[1] Wrong A. Wrong Reference. 2020. doi:10.1000/wrong",
        },
        24: {
            "authors": "Gehm M, Brady D",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "raw": (
                "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with "
                "a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
            ),
        },
    }

    def fake_resolve(_index_data, src, ref_num, *, source_sha1=""):
        del _index_data, source_sha1
        if str(src) != source_path:
            return None
        ref = refs.get(int(ref_num))
        return {
            "source_path": source_path,
            "source_name": "paper.pdf",
            "ref_num": int(ref_num),
            "ref": dict(ref),
        } if isinstance(ref, dict) else None

    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_display_source_name", lambda _sp: "paper.pdf")

    messages = [
        {"id": 1, "role": "user", "content": "Which in-paper reference supports this DOI?"},
        {
            "id": 2,
            "role": "assistant",
            "content": f"This follows DOI 10.1364/OE.15.014013 [[CITE:{sid}:24]].",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Evidence mentions prior work [24].",
                    "meta": {
                        "source_path": source_path,
                        "source_sha1": "abc",
                    },
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-target-ref")
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    details = list(msg.get("cite_details") or [])

    assert len(details) == 1
    detail = details[0]
    assert detail["num"] == 24
    assert detail["doi"] == "10.1364/OE.15.014013"
    assert "dual-disperser architecture" in detail["title"]
    assert "Wrong Reference" not in str(detail)
    assert f"[24](#{detail['anchor']}" in str(msg.get("rendered_body") or "")
    assert packet["rendered_body"] == msg.get("rendered_body")
    assert packet["cite_details"][0]["doi"] == "10.1364/OE.15.014013"


def test_structured_cite_fallback_uses_local_answer_line_for_system_b_context(monkeypatch, tmp_path: Path):
    from api import chat_render

    source_file = tmp_path / "paper.en.md"
    source_file.write_text(
        "\n".join(
            [
                "# Paper",
                "This body intentionally leaves mention choice to the structured asset.",
                "",
                "## References",
                "[3] Example A. Detector-array reconstruction benchmark. Journal, 2024.",
            ]
        ),
        encoding="utf-8",
    )
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "reference_index.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_num": 3,
                        "citation_mentions": [
                            {
                                "citation_context": "The introduction briefly names earlier optical sectioning work [3].",
                                "heading_path": "Paper / Introduction",
                                "location_label": "Paper / Introduction / p. 1",
                                "page_start": 1,
                                "page_end": 1,
                                "line_start": 8,
                                "line_end": 8,
                            },
                            {
                                "citation_context": "The benchmark compares detector-array reconstruction accuracy against prior work [3].",
                                "heading_path": "Paper / Benchmark",
                                "location_label": "Paper / Benchmark / p. 5",
                                "page_start": 5,
                                "page_end": 5,
                                "line_start": 88,
                                "line_end": 88,
                            },
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    source_path = str(source_file)
    sid = chat_render._source_cite_id(source_path)
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: {})

    md = f"Intro sentence without a citation.\nFor the benchmark, open prior work [[CITE:{sid}:3]]."
    rendered, details = chat_render._fallback_render_structured_citations(
        md,
        [{"text": "hit", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        anchor_ns="local-line-test",
    )

    assert "[3](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["is_inpaper"] is True
    assert detail["answer_claim"] == "For the benchmark, open prior work."
    assert detail["citation_context_source"] == "structured_reference_index"
    assert "detector-array reconstruction accuracy" in detail["citation_context"]
    assert "briefly names earlier" not in detail["citation_context"]
    assert detail["heading_path"].endswith("Benchmark")
    assert detail["page_start"] == 5


def test_structured_cite_fallback_marks_answer_context_only_when_source_context_missing(monkeypatch, tmp_path: Path):
    from api import chat_render

    source_file = tmp_path / "paper.en.md"
    source_file.write_text(
        "\n".join(
            [
                "# Paper",
                "No body mention is available for this reference.",
                "",
                "## References",
                "[6] Example B. Unlocated upstream method. 2023.",
            ]
        ),
        encoding="utf-8",
    )
    source_path = str(source_file)
    sid = chat_render._source_cite_id(source_path)
    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: {})

    md = f"This answer mentions an upstream method [[CITE:{sid}:6]]."
    rendered, details = chat_render._fallback_render_structured_citations(
        md,
        [{"text": "hit", "meta": {"source_path": source_path, "source_sha1": "abc"}}],
        anchor_ns="answer-only-test",
    )

    assert "[6](#kb-cite-" in rendered
    assert len(details) == 1
    detail = details[0]
    assert detail["citation_context_source"] == "answer_context"
    assert detail["card_evidence_label"] == "回答里的线索"
    assert "answer_context_only" in detail["card_quality_flags"]


def test_non_citation_message_does_not_preserve_stale_existing_render_packet_links():
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf [2].",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "overview"},
                    "render_packet": {
                        "rendered_body": "Existing rendered body [2](#kb-cite-demo-2).",
                        "rendered_content": "Existing rendered body [2](#kb-cite-demo-2).",
                        "copy_markdown": "Existing rendered body [2](#kb-cite-demo-2).",
                        "copy_text": "Existing rendered body [2].",
                        "cite_details": [{"num": 2, "anchor": "kb-cite-demo-2", "source_name": "demo.pdf"}],
                    },
                }
            },
        },
    ]

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})

    assert "[2](" not in str(packet.get("rendered_body") or "")
    assert "[2]" in str(packet.get("rendered_body") or "")
    assert packet.get("cite_details") == []


def test_enrich_provenance_segments_for_display_loads_md_blocks_for_quote_rebind(tmp_path: Path):
    fixture = build_scinerf_like_fixture(tmp_path)
    md_main = fixture["md_main"]
    wrong_method_block = fixture["wrong_method_block"]
    conclusion_block = fixture["conclusion_block"]

    provenance = {
        "md_path": str(md_main),
        "source_path": str(tmp_path / "dummy.pdf"),
        "source_name": "SCINeRF.pdf",
        "block_map": {
            str(wrong_method_block.get("block_id") or ""): dict(wrong_method_block),
        },
        "segments": [
            {
                "segment_id": "seg_004",
                "segment_index": 4,
                "kind": "blockquote",
                "segment_type": "evidence",
                "text": (
                    "SCINeRF exploits neural radiance fields as its underlying scene representation [...] "
                    "Physical image formation process of an SCI image is exploited to formulate the training objective "
                    "for jointly NeRF training and camera poses optimization."
                ),
                "raw_markdown": (
                    '*"SCINeRF exploits neural radiance fields as its underlying scene representation [...] '
                    "Physical image formation process of an SCI image is exploited to formulate the training objective "
                    'for jointly NeRF training and camera poses optimization."*'
                ),
                "evidence_mode": "direct",
                "claim_type": "blockquote_claim",
                "must_locate": True,
                "anchor_kind": "blockquote",
                "primary_block_id": str(wrong_method_block.get("block_id") or ""),
                "primary_anchor_id": str(wrong_method_block.get("anchor_id") or ""),
                "primary_heading_path": str(wrong_method_block.get("heading_path") or ""),
                "evidence_block_ids": [str(wrong_method_block.get("block_id") or "")],
                "support_block_ids": [],
                "anchor_text": str(wrong_method_block.get("text") or ""),
                "evidence_quote": str(wrong_method_block.get("text") or ""),
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("primary_block_id") or "") == str(conclusion_block.get("block_id") or "")
    block_map = enriched.get("block_map") or {}
    assert str(conclusion_block.get("block_id") or "") in block_map


def test_enrich_provenance_segments_for_display_preserves_figure_scope_heading():
    provenance = {
        "block_map": {},
        "segments": [
            {
                "segment_id": "seg_001",
                "segment_index": 0,
                "text": "Panel (f) corresponds to methane imaging using SPC.",
                "raw_markdown": "Panel (f) corresponds to methane imaging using SPC.",
                "evidence_mode": "direct",
                "claim_type": "figure_claim",
                "must_locate": True,
                "anchor_kind": "figure",
                "anchor_text": "(f) methane imaging using SPC$^{15}$",
                "primary_heading_path": "Applications and future potential for single-pixel imaging",
                "support_slot_claim_type": "figure_panel",
                "support_slot_figure_number": 3,
                "support_slot_panel_letters": ["f"],
                "support_locate_anchor": "(f) methane imaging using SPC$^{15}$",
                "locate_policy": "required",
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    assert str(segments[0].get("primary_heading_path") or "") == (
        "Applications and future potential for single-pixel imaging / Figure 3"
    )


def test_enrich_provenance_segments_for_display_preserves_box_only_heading():
    provenance = {
        "block_map": {},
        "segments": [
            {
                "segment_id": "seg_001",
                "segment_index": 0,
                "text": "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))...",
                "raw_markdown": "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))...",
                "evidence_mode": "direct",
                "claim_type": "own_result",
                "must_locate": False,
                "anchor_kind": "sentence",
                "anchor_text": "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))...",
                "primary_heading_path": "Acquisition and image reconstruction strategies",
                "support_slot_claim_type": "own_result",
                "support_slot_box_number": 1,
                "support_slot_panel_letters": [],
                "support_locate_anchor": "It can be shown that when the number of sampling patterns used $M \\ge O(K \\log(N/K))$...",
                "locate_policy": "hidden",
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    assert str(segments[0].get("primary_heading_path") or "") == "Box 1"
    assert str(segments[0].get("support_locate_anchor") or "") == (
        "It can be shown that when the number of sampling patterns used M >= O(K log(N/K))..."
    )
    assert str(segments[0].get("locate_policy") or "") == "required"


def test_enrich_provenance_segments_for_display_preserves_exact_method_detail_heading():
    provenance = {
        "block_map": {
            "blk_setup": {
                "block_id": "blk_setup",
                "anchor_id": "p_00035",
                "heading_path": "ARTICLE / Methods / Principle of high-throughput SPH",
                "kind": "paragraph",
                "text": (
                    "**Experimental setup.** Thus, the beat frequency of these two beams is 62,500 Hz. "
                    "The data acquisition card uses a sampling rate of 1.25 Ms/s."
                ),
                "raw_text": (
                    "**Experimental setup.** Thus, the beat frequency of these two beams is 62,500 Hz. "
                    "The data acquisition card uses a sampling rate of 1.25 Ms/s."
                ),
            }
        },
        "segments": [
            {
                "segment_id": "seg_001",
                "segment_index": 1,
                "text": "The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:",
                "raw_markdown": "The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:",
                "evidence_mode": "synthesis",
                "claim_type": "critical_fact_claim",
                "anchor_kind": "sentence",
                "anchor_text": "The paper states this explicitly in ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup:",
                "locate_policy": "hidden",
            },
            {
                "segment_id": "seg_002",
                "segment_index": 2,
                "text": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "raw_markdown": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "evidence_mode": "direct",
                "claim_type": "method_detail",
                "must_locate": True,
                "anchor_kind": "sentence",
                "anchor_text": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "primary_block_id": "blk_setup",
                "primary_anchor_id": "p_00035",
                "primary_heading_path": "ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup",
                "evidence_block_ids": ["blk_setup"],
                "support_block_ids": [],
                "support_slot_claim_type": "method_detail",
                "support_locate_anchor": "Thus, the beat frequency of these two beams is 62,500 Hz. The data acquisition card uses a sampling rate of 1.25 Ms/s.",
                "locate_policy": "required",
            },
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 2
    assert str(segments[1].get("primary_heading_path") or "") == (
        "ARTICLE / Methods / Principle of high-throughput SPH / Experimental setup"
    )
    assert str(segments[1].get("support_slot_claim_type") or "") == "method_detail"


def test_enrich_provenance_segments_for_display_rebinds_formula_claim_using_equation_index(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    formula = "$$Y = \\\\sum_{i=1}^{N} X_i \\\\odot M_i + Z \\\\tag{1}$$"
    method_line = "This paragraph explains the measurement process before the formal equation."
    md_main.write_text(
        (
            "# Method\n\n"
            f"{method_line}\n\n"
            f"{formula}\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    method_block = next(
        block for block in blocks
        if "measurement process" in str(block.get("text") or "").lower()
    )
    equation_block = next(
        block for block in blocks
        if str(block.get("kind") or "").strip().lower() == "equation"
    )
    (assets_dir / "equation_index.json").write_text(
        json.dumps(
            {
                "equations": [
                    {
                        "equation_number": 1,
                        "equation_markdown": str(equation_block.get("raw_text") or equation_block.get("text") or ""),
                        "normalized_tex": "Y = sum_i X_i odot M_i + Z tag(1)",
                        "context_before": method_line,
                        "context_after": "",
                        "block_id": str(equation_block.get("block_id") or ""),
                        "anchor_id": str(equation_block.get("anchor_id") or ""),
                        "heading_path": str(equation_block.get("heading_path") or ""),
                        "line_start": int(equation_block.get("line_start") or 0),
                        "line_end": int(equation_block.get("line_end") or 0),
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    provenance = {
        "md_path": str(md_main),
        "source_path": str(source_pdf),
        "source_name": "DemoPaper.pdf",
        "block_map": {
            str(method_block.get("block_id") or ""): dict(method_block),
        },
        "segments": [
            {
                "segment_id": "seg_formula_display_fix",
                "segment_index": 1,
                "kind": "paragraph",
                "segment_type": "equation",
                "text": "Equation (1) defines the coded measurement.",
                "raw_markdown": formula,
                "evidence_mode": "direct",
                "claim_type": "formula_claim",
                "must_locate": True,
                "anchor_kind": "equation",
                "anchor_text": formula,
                "equation_number": 1,
                "primary_block_id": str(method_block.get("block_id") or ""),
                "primary_anchor_id": str(method_block.get("anchor_id") or ""),
                "primary_heading_path": str(method_block.get("heading_path") or ""),
                "evidence_block_ids": [str(method_block.get("block_id") or "")],
                "support_block_ids": [],
                "evidence_quote": formula,
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("primary_block_id") or "") == str(equation_block.get("block_id") or "")
    assert str(seg.get("primary_anchor_id") or "") == str(equation_block.get("anchor_id") or "")
    assert str(seg.get("anchor_kind") or "") == "equation"
    assert str(seg.get("hit_level") or "") == "exact"
    locate_target = seg.get("locate_target") or {}
    assert str(locate_target.get("blockId") or "") == str(equation_block.get("block_id") or "")
    assert str(locate_target.get("anchorId") or "") == str(equation_block.get("anchor_id") or "")
    assert str(locate_target.get("anchorKind") or "") == "equation"
    assert str(locate_target.get("hitLevel") or "") == "exact"
    reader_open = seg.get("reader_open") or {}
    assert str(reader_open.get("sourcePath") or "") == str(source_pdf)
    assert str(reader_open.get("blockId") or "") == str(equation_block.get("block_id") or "")
    assert str(reader_open.get("anchorId") or "") == str(equation_block.get("anchor_id") or "")
    assert bool(reader_open.get("strictLocate")) is True
    assert str(((reader_open.get("locateTarget") or {}).get("anchorKind")) or "") == "equation"
    block_map = enriched.get("block_map") or {}
    assert str(equation_block.get("block_id") or "") in block_map


def test_enrich_provenance_segments_for_display_backfills_anchor_only_segment_using_anchor_index(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "DemoPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "DemoPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_main = md_dir / "DemoPaper.en.md"
    md_main.write_text(
        (
            "# Abstract\n\n"
            "APR improves coherent reconstruction quality.\n\n"
            "# Methods\n\n"
            "APR was performed using image registration based on phase correlation of the off-axis raw images.\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    method_block = next(
        block for block in blocks
        if "phase correlation" in str(block.get("text") or "").lower()
    )
    (assets_dir / "anchor_index.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {
                        "anchor_id": str(method_block.get("anchor_id") or ""),
                        "block_id": str(method_block.get("block_id") or ""),
                        "kind": str(method_block.get("kind") or ""),
                        "heading_path": str(method_block.get("heading_path") or ""),
                        "order_index": int(method_block.get("order_index") or 0),
                        "line_start": int(method_block.get("line_start") or 0),
                        "line_end": int(method_block.get("line_end") or 0),
                        "text": str(method_block.get("text") or ""),
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    provenance = {
        "md_path": str(md_main),
        "source_path": str(source_pdf),
        "source_name": "DemoPaper.pdf",
        "block_map": {},
        "segments": [
            {
                "segment_id": "seg_anchor_only_display_fix",
                "segment_index": 1,
                "kind": "paragraph",
                "segment_type": "evidence",
                "text": "APR uses phase correlation to align the off-axis raw images.",
                "raw_markdown": "APR uses phase correlation to align the off-axis raw images.",
                "evidence_mode": "direct",
                "claim_type": "method_detail",
                "must_locate": True,
                "locate_policy": "required",
                "primary_block_id": "",
                "primary_anchor_id": str(method_block.get("anchor_id") or ""),
                "primary_heading_path": str(method_block.get("heading_path") or ""),
                "evidence_block_ids": [],
                "support_block_ids": [],
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    assert str(seg.get("primary_block_id") or "") == str(method_block.get("block_id") or "")
    assert str(seg.get("primary_anchor_id") or "") == str(method_block.get("anchor_id") or "")
    assert str(seg.get("primary_heading_path") or "") == str(method_block.get("heading_path") or "")
    assert str(seg.get("anchor_kind") or "") == "sentence"
    assert str(seg.get("hit_level") or "") == "exact"
    locate_target = seg.get("locate_target") or {}
    assert str(locate_target.get("blockId") or "") == str(method_block.get("block_id") or "")
    assert str(locate_target.get("anchorId") or "") == str(method_block.get("anchor_id") or "")
    assert str(locate_target.get("anchorKind") or "") == "sentence"
    assert str(locate_target.get("hitLevel") or "") == "exact"
    reader_open = seg.get("reader_open") or {}
    assert str(reader_open.get("sourcePath") or "") == str(source_pdf)
    assert str(reader_open.get("blockId") or "") == str(method_block.get("block_id") or "")
    assert str(reader_open.get("anchorId") or "") == str(method_block.get("anchor_id") or "")
    assert str(((reader_open.get("locateTarget") or {}).get("anchorKind")) or "") == "sentence"
    block_map = enriched.get("block_map") or {}
    assert str(method_block.get("block_id") or "") in block_map


def test_enrich_provenance_segments_for_display_rebinds_figure_claim_using_figure_index(tmp_path: Path):
    from kb import task_runtime

    source_pdf = tmp_path / "VisionPaper.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")
    md_dir = tmp_path / "VisionPaper"
    assets_dir = md_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "fig1.png").write_bytes(b"fake")
    md_main = md_dir / "VisionPaper.en.md"
    figure_caption = (
        "Figure 1. Given a single snapshot compressed image, our method is able to recover "
        "the underlying 3D scene representation."
    )
    method_para = (
        "Our method takes a single compressed image and encoding masks as input, and recovers "
        "the underlying 3D scene representation as well as camera poses."
    )
    md_main.write_text(
        (
            "# VisionPaper\n\n"
            "![Figure 1](./assets/fig1.png)\n"
            f"*{figure_caption}*\n\n"
            "## Method\n\n"
            f"{method_para}\n"
        ),
        encoding="utf-8",
    )

    blocks = task_runtime.load_source_blocks(md_main)
    figure_block = next(block for block in blocks if str(block.get("kind") or "") == "figure")
    caption_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "single snapshot compressed image" in str(block.get("text") or "").lower()
    )
    method_block = next(
        block for block in blocks
        if str(block.get("kind") or "") == "paragraph"
        and "encoding masks as input" in str(block.get("text") or "").lower()
    )
    (assets_dir / "figure_index.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "paper_figure_number": 1,
                        "figure_id": str(figure_block.get("figure_id") or ""),
                        "figure_block_id": str(figure_block.get("block_id") or ""),
                        "caption_block_id": str(caption_block.get("block_id") or ""),
                        "caption_anchor_id": str(caption_block.get("anchor_id") or ""),
                        "anchor_id": str(figure_block.get("anchor_id") or ""),
                        "heading_path": str(figure_block.get("heading_path") or ""),
                        "caption": figure_caption,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    provenance = {
        "md_path": str(md_main),
        "source_path": str(source_pdf),
        "source_name": "VisionPaper.pdf",
        "block_map": {
            str(method_block.get("block_id") or ""): dict(method_block),
        },
        "segments": [
            {
                "segment_id": "seg_figure_display_fix",
                "segment_index": 1,
                "kind": "paragraph",
                "segment_type": "evidence",
                "text": "Figure 1 shows recovery from a single snapshot compressed image.",
                "raw_markdown": "Figure 1 shows recovery from a single snapshot compressed image.",
                "evidence_mode": "direct",
                "claim_type": "figure_claim",
                "must_locate": True,
                "anchor_kind": "figure",
                "anchor_text": "Figure 1",
                "support_slot_figure_number": 1,
                "primary_block_id": str(method_block.get("block_id") or ""),
                "primary_anchor_id": str(method_block.get("anchor_id") or ""),
                "primary_heading_path": str(method_block.get("heading_path") or ""),
                "evidence_block_ids": [str(method_block.get("block_id") or "")],
                "support_block_ids": [],
                "evidence_quote": "Given a single snapshot compressed image, our method is able to recover the underlying 3D scene representation.",
            }
        ],
    }

    enriched = _enrich_provenance_segments_for_display(provenance, hits=[], anchor_ns="conv:test")

    assert isinstance(enriched, dict)
    segments = enriched.get("segments") or []
    assert len(segments) == 1
    seg = segments[0]
    # For figure claims, prefer landing on the caption when it exists (more informative than the figure placeholder).
    assert str(seg.get("primary_block_id") or "") == str(caption_block.get("block_id") or "")
    assert str(seg.get("primary_anchor_id") or "") == str(caption_block.get("anchor_id") or "")
    assert str(seg.get("primary_heading_path") or "") == str(figure_block.get("heading_path") or "")
    assert str(seg.get("anchor_kind") or "") == "figure"
    assert str(seg.get("hit_level") or "") == "exact"
    assert str(caption_block.get("block_id") or "") in list(seg.get("evidence_block_ids") or [])
    locate_target = seg.get("locate_target") or {}
    assert str(locate_target.get("blockId") or "") == str(caption_block.get("block_id") or "")
    assert str(locate_target.get("anchorId") or "") == str(caption_block.get("anchor_id") or "")
    assert str(locate_target.get("anchorKind") or "") == "figure"
    assert int(locate_target.get("anchorNumber") or 0) == 1
    reader_open = seg.get("reader_open") or {}
    assert str(reader_open.get("sourcePath") or "") == str(source_pdf)
    assert str(reader_open.get("blockId") or "") == str(caption_block.get("block_id") or "")
    assert str(reader_open.get("anchorId") or "") == str(caption_block.get("anchor_id") or "")
    assert int(reader_open.get("anchorNumber") or 0) == 1
    assert str(((reader_open.get("locateTarget") or {}).get("anchorKind")) or "") == "figure"
    alternatives = list(reader_open.get("alternatives") or [])
    assert len(alternatives) >= 1
    assert isinstance(alternatives[0], dict)
    assert list(reader_open.get("visibleAlternatives") or []) == alternatives
    assert list(reader_open.get("evidenceAlternatives") or []) == alternatives
    block_map = enriched.get("block_map") or {}
    assert str(figure_block.get("block_id") or "") in block_map


def test_enrich_messages_reuses_persisted_render_cache(monkeypatch, tmp_path: Path):
    from api import chat_render

    calls = {"primary": 0, "merge": 0}

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _hits, anchor_ns, canonical_paths
        calls["primary"] += 1
        return (
            f"cached::{_md}",
            [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)
    original_merge = chat_render._merge_render_packet_contract_meta

    def counted_merge(**kwargs):
        calls["merge"] += 1
        return original_merge(**kwargs)

    monkeypatch.setattr(chat_render, "_merge_render_packet_contract_meta", counted_merge)

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("cache test")
    user_id = store.append_message(conv_id, "user", "test")
    assistant_id = store.append_message(conv_id, "assistant", "SPI relies on compressive sensing [[CITE:s1234abcd:1]].")
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "test",
            "used_translation": False,
            "hits": [
                {
                    "text": "dummy",
                    "meta": {"source_path": r"db\doc\doc.en.md"},
                }
            ],
        }
    }

    store.merge_message_meta(
        assistant_id,
        {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
    )

    first = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)
    second = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)
    persisted = store.get_messages(conv_id)[-1]
    render_cache = ((persisted.get("meta") or {}).get("render_cache") or {})

    assert calls["primary"] == 1
    assert calls["merge"] == 1
    assert str(first[-1].get("rendered_content") or "") == str(second[-1].get("rendered_content") or "")
    assert str(second[-1].get("copy_text") or "").strip()
    assert isinstance(render_cache.get("render_packet"), dict)


def test_cached_render_packet_does_not_refresh_for_subset_primary_projection():
    from api import chat_render

    packet_primary = {
        "source_path": "db/paper/paper.en.md",
        "heading_path": "Abstract",
        "snippet": "The method parallelizes acquisition without changing integration time.",
        "block_id": "blk-1",
        "anchor_id": "p-1",
        "page_start": 1,
        "selection_reason": "answer_aligned_block",
        "strict_locate": True,
    }
    cached = {"render_packet": {"primary_evidence": packet_primary}}
    projected_pack = {
        "primary_evidence": {
            "source_path": "db/paper/paper.en.md",
            "heading_path": "Abstract",
            "snippet": packet_primary["snippet"],
        }
    }

    assert not chat_render._cached_render_packet_needs_contract_refresh(
        cached,
        enriched_provenance=None,
        ref_pack=projected_pack,
    )


def test_cached_render_packet_refreshes_for_new_primary_identity_or_provenance():
    from api import chat_render

    cached = {
        "render_packet": {
            "primary_evidence": {
                "source_path": "db/paper-a/paper-a.en.md",
                "heading_path": "Abstract",
                "snippet": "Existing evidence.",
            }
        }
    }
    changed_pack = {
        "primary_evidence": {
            "source_path": "db/paper-b/paper-b.en.md",
            "heading_path": "Methods",
            "snippet": "New evidence.",
        }
    }

    assert chat_render._cached_render_packet_needs_contract_refresh(
        cached,
        enriched_provenance=None,
        ref_pack=changed_pack,
    )
    assert chat_render._cached_render_packet_needs_contract_refresh(
        cached,
        enriched_provenance={"segments": [{"segment_id": "seg-1"}]},
        ref_pack=None,
    )


def test_provenance_display_enrichment_reuses_exact_cached_inputs(monkeypatch):
    from api import chat_render

    calls = {"count": 0}

    def fake_uncached(provenance, hits, *, anchor_ns):
        calls["count"] += 1
        return {
            **dict(provenance or {}),
            "segments": [{"segment_id": "seg-1", "anchor_ns": anchor_ns}],
            "hit_count": len(hits or []),
        }

    chat_render._enrich_provenance_segments_for_display_cached.cache_clear()
    monkeypatch.setattr(
        chat_render,
        "_enrich_provenance_segments_for_display_uncached",
        fake_uncached,
    )
    provenance = {"segments": [{"segment_id": "seg-1"}]}
    hits = [{"text": "evidence", "meta": {"source_path": "paper.en.md"}}]

    first = chat_render._enrich_provenance_segments_for_display(
        provenance,
        hits,
        anchor_ns="conv:1",
    )
    first["segments"][0]["anchor_ns"] = "mutated"
    second = chat_render._enrich_provenance_segments_for_display(
        provenance,
        hits,
        anchor_ns="conv:1",
    )

    assert calls["count"] == 1
    assert second["segments"][0]["anchor_ns"] == "conv:1"


def test_historical_render_cache_reuse_requires_same_answer_refs_and_plan():
    from api import chat_render

    answer = "The method reports a concrete reconstruction result."
    input_ref_sig = "refs-current"
    citation_plan_sig = "plan-current"
    answer_sig = chat_render._answer_render_signature(answer)
    render_packet = {
        "answer_markdown": answer,
        "notice": "",
        "rendered_body": answer,
        "rendered_content": answer,
        "copy_markdown": answer,
        "copy_text": answer,
        "cite_details": [],
    }
    cache = chat_render._build_render_cache_payload(
        cache_key="old-reference-card-key",
        notice="",
        rendered_body=answer,
        rendered_content=answer,
        copy_markdown=answer,
        copy_text=answer,
        cite_details=[],
        refs_user_msg_id=1,
        render_packet=render_packet,
        answer_sig=answer_sig,
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        locale="en",
    )

    reused = chat_render._extract_compatible_historical_render_cache(
        {"render_cache": cache},
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        raw_content=answer,
        hits=[],
        answer_sig=answer_sig,
        locale="en",
    )
    rejected = chat_render._extract_compatible_historical_render_cache(
        {"render_cache": cache},
        input_ref_sig="refs-changed",
        citation_plan_sig=citation_plan_sig,
        raw_content=answer,
        hits=[],
        answer_sig=answer_sig,
        locale="en",
    )
    rejected_plan = chat_render._extract_compatible_historical_render_cache(
        {"render_cache": cache},
        input_ref_sig=input_ref_sig,
        citation_plan_sig="plan-changed",
        raw_content=answer,
        hits=[],
        answer_sig=answer_sig,
        locale="en",
    )
    rejected_answer = chat_render._extract_compatible_historical_render_cache(
        {"render_cache": cache},
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        raw_content=answer,
        hits=[],
        answer_sig="answer-changed",
        locale="en",
    )
    rejected_locale = chat_render._extract_compatible_historical_render_cache(
        {"render_cache": cache},
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        raw_content=answer,
        hits=[],
        answer_sig=answer_sig,
        locale="zh",
    )

    assert reused is not None
    assert reused["rendered_content"] == answer
    assert rejected is None
    assert rejected_plan is None
    assert rejected_answer is None
    assert rejected_locale is None

    stale_packet = {
        **render_packet,
        "rendered_body": "A stale streaming draft that mentions a different paper.",
        "rendered_content": "A stale streaming draft that mentions a different paper.",
        "copy_markdown": "A stale streaming draft that mentions a different paper.",
        "copy_text": "A stale streaming draft that mentions a different paper.",
    }
    stale_cache = chat_render._build_render_cache_payload(
        cache_key="stale-prose-key",
        notice="",
        rendered_body=stale_packet["rendered_body"],
        rendered_content=stale_packet["rendered_content"],
        copy_markdown=stale_packet["copy_markdown"],
        copy_text=stale_packet["copy_text"],
        cite_details=[],
        refs_user_msg_id=1,
        render_packet=stale_packet,
        answer_sig=answer_sig,
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        locale="en",
    )

    assert chat_render._extract_compatible_historical_render_cache(
        {"render_cache": stale_cache},
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        raw_content=answer,
        hits=[],
        answer_sig=answer_sig,
        locale="en",
    ) is None
    assert chat_render._extract_render_cache(
        {"render_cache": stale_cache},
        expected_key="stale-prose-key",
        raw_content=answer,
        hits=[],
        answer_sig=answer_sig,
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        locale="en",
    ) is None


def test_enrich_rejects_scigs_lineage_whole_answer_rewrite(monkeypatch):
    from api import chat_render

    original = "SCIGS reconstructs an explicit dynamic 3D scene from one compressed image [1]."
    rewritten = (
        "### 从编码测量到 3D 表示\n\n"
        "- CASSI encodes a spectral cube.\n"
        "- Video SCI provides the upstream route.\n"
        "- SCINeRF uses an implicit NeRF.\n"
        "- SCIGS uses explicit 3DGS [1]."
    )
    annotate_inputs: list[str] = []

    def fake_annotate(md, _hits, **_kwargs):
        annotate_inputs.append(str(md))
        rendered = str(md).replace("[1]", "[1](#kb-cite-scigs-1)")
        return rendered, [
            {
                "num": 1,
                "anchor": "kb-cite-scigs-1",
                "source_name": "SCIGS.pdf",
                "source_path": r"db\scigs\scigs.en.md",
                "citation_route": "system_a",
                "is_inpaper": False,
            }
        ]

    monkeypatch.setattr(
        chat_render,
        "_answer_aligned_reference_render_pack",
        lambda pack, _answer: dict(pack or {}),
    )
    monkeypatch.setattr(
        chat_render,
        "_should_link_inpaper_citations_for_message",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        chat_render,
        "_reading_guide_repair_missing_system_a_citations",
        lambda *_args, **_kwargs: rewritten,
    )
    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        fake_annotate,
    )

    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": r"db\scigs\scigs.en.md",
                "source_name": "SCIGS.pdf",
                "evidence_quote": "SCIGS reconstructs an explicit dynamic 3D scene.",
            }
        ],
    }
    messages = [
        {"id": 1, "role": "user", "content": "Compare the SCI lineage."},
        {
            "id": 2,
            "role": "assistant",
            "content": original,
            "meta": {
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "citation_plan": plan,
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "SCIGS reconstructs an explicit dynamic 3D scene.",
                    "meta": {
                        "source_path": r"db\scigs\scigs.en.md",
                        "ref_answer_citation_num": 1,
                    },
                }
            ]
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="scigs-render-guard",
    )[-1]
    rendered_body = str(rendered.get("rendered_body") or "")

    assert annotate_inputs
    assert all(value == original for value in annotate_inputs)
    assert "从编码测量到 3D 表示" not in rendered_body
    assert "SCIGS reconstructs an explicit dynamic 3D scene" in rendered_body
    assert "](#kb-cite-scigs-1)" in rendered_body
    assert chat_render._rendered_body_preserves_answer_body(
        answer_body=original,
        rendered_body=rendered_body,
    )


def test_comparison_plan_allows_typed_three_source_evidence_repair() -> None:
    from api import chat_render

    original = "Compare structured, interferometric, and light-field microscopy [1]."
    repaired = (
        "Structured detection provides super-resolution and optical sectioning [1].\n\n"
        "Interferometric imaging reaches 120 nm lateral resolution [2].\n\n"
        "Light-field microscopy records position and angular information [3]."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [index],
                    "source_path": f"db/method-{index}/method-{index}.en.md",
                    "evidence_quote": evidence,
            }
            for index, evidence in enumerate(
                (
                    "Structured detection provides super-resolution and optical sectioning.",
                    "Interferometric imaging reaches 120 nm lateral resolution.",
                    "Light-field microscopy records position and angular information.",
                ),
                start=1,
            )
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_comparison_plan_does_not_authorize_single_token_hallucination() -> None:
    from api import chat_render

    original = "Compare AlphaNet and BetaNet [1]."
    repaired = (
        "AlphaNet cures cancer and guarantees perfect safety [1].\n\n"
        "BetaNet reconstructs images [2]."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "db/alphanet/alphanet.en.md",
                "evidence_quote": "AlphaNet reconstructs grayscale images.",
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": "db/betanet/betanet.en.md",
                "evidence_quote": "BetaNet reconstructs images.",
            },
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == original


def test_comparison_plan_rejects_repair_with_unplanned_citation_numbers() -> None:
    from api import chat_render

    original = "Compare AlphaNet and BetaNet [1]."
    repaired = (
        "AlphaNet reconstructs grayscale images [1].\n\n"
        "BetaNet reconstructs color images [2]."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [5],
                "source_path": "db/alphanet/alphanet.en.md",
                "evidence_quote": "AlphaNet reconstructs grayscale images.",
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [6],
                "source_path": "db/betanet/betanet.en.md",
                "evidence_quote": "BetaNet reconstructs color images.",
            },
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == original


def test_comparison_plan_does_not_authorize_unrelated_render_rewrite() -> None:
    from api import chat_render

    original = "Compare structured and interferometric microscopy [1]."
    repaired = "The sky is green and unrelated to either method [1]."
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "db/structured/structured.en.md",
                "evidence_quote": "Structured detection provides optical sectioning.",
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": "db/interferometric/interferometric.en.md",
                "evidence_quote": "Interferometric detection improves lateral resolution.",
            },
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == original


def test_enrich_keeps_normal_citation_decoration_when_body_is_unchanged(monkeypatch):
    from api import chat_render

    original = "SCIGS reconstructs an explicit dynamic 3D scene [1]."
    calls = {"annotate": 0}

    def fake_annotate(md, _hits, **_kwargs):
        calls["annotate"] += 1
        return str(md).replace("[1]", "[1](#kb-cite-scigs-1)"), [
            {
                "num": 1,
                "anchor": "kb-cite-scigs-1",
                "source_name": "SCIGS.pdf",
                "source_path": r"db\scigs\scigs.en.md",
                "citation_route": "system_a",
                "is_inpaper": False,
            }
        ]

    monkeypatch.setattr(
        chat_render,
        "_answer_aligned_reference_render_pack",
        lambda pack, _answer: dict(pack or {}),
    )
    monkeypatch.setattr(
        chat_render,
        "_should_link_inpaper_citations_for_message",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        fake_annotate,
    )
    messages = [
        {"id": 1, "role": "user", "content": "What does SCIGS do?"},
        {"id": 2, "role": "assistant", "content": original},
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "SCIGS reconstructs an explicit dynamic 3D scene.",
                    "meta": {
                        "source_path": r"db\scigs\scigs.en.md",
                        "ref_answer_citation_num": 1,
                    },
                }
            ]
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="citation-decoration-guard",
    )[-1]
    rendered_body = str(rendered.get("rendered_body") or "")

    assert calls["annotate"] == 1
    assert "](#kb-cite-scigs-1)" in rendered_body
    assert chat_render._rendered_body_preserves_answer_body(
        answer_body=original,
        rendered_body=rendered_body,
    )


def test_enrich_rejects_semantic_change_from_final_markdown_cleanup(monkeypatch):
    from api import chat_render

    original = "The final answer prose must remain unchanged."
    monkeypatch.setattr(
        chat_render,
        "_normalize_chat_markdown_for_display",
        lambda _md: "A cleanup stage replaced the answer.",
    )
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {"id": 2, "role": "assistant", "content": original},
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        {},
        conv_id="final-cleanup-render-guard",
    )[-1]

    assert rendered["rendered_body"] == original
    assert rendered["rendered_content"] == original


def test_merge_render_packet_does_not_restore_semantically_drifted_body(monkeypatch):
    from api import chat_render

    original = "The stored answer remains the source of truth."
    drifted = "A historical render packet replaced the answer with a different summary."
    monkeypatch.setattr(
        chat_render,
        "_should_link_inpaper_citations_for_message",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        lambda md, _hits, **_kwargs: (str(md), []),
    )
    messages = [
        {"id": 1, "role": "user", "content": "test"},
        {
            "id": 2,
            "role": "assistant",
            "content": original,
            "meta": {
                "paper_guide_contracts": {
                    "render_packet": {
                        "answer_markdown": original,
                        "rendered_body": drifted,
                        "rendered_content": drifted,
                        "copy_markdown": drifted,
                        "copy_text": drifted,
                        "cite_details": [
                            {
                                "num": 1,
                                "anchor": "kb-cite-stale-1",
                                "source_name": "stale.pdf",
                                "source_path": r"db\stale\stale.en.md",
                                "citation_route": "system_a",
                                "is_inpaper": False,
                            }
                        ],
                    }
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Evidence.",
                    "meta": {"source_path": r"db\fresh\fresh.en.md"},
                }
            ]
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="render-packet-body-guard",
    )[-1]
    packet = (
        ((rendered.get("meta") or {}).get("paper_guide_contracts") or {}).get(
            "render_packet"
        )
        or {}
    )

    assert rendered["rendered_body"] == original
    assert packet.get("rendered_body") == original
    assert drifted not in str(packet.get("rendered_content") or "")


def test_historical_cache_signature_mismatch_rerenders_old_assistant(monkeypatch):
    from api import chat_render

    historical_answer = "Historical evidence remains linked [1]."
    stale_rendered = "Historical evidence remains linked [1](#kb-cite-stale-1)."
    cache = chat_render._build_render_cache_payload(
        cache_key="stale-historical-key",
        notice="",
        rendered_body=stale_rendered,
        rendered_content=stale_rendered,
        copy_markdown=stale_rendered,
        copy_text="Historical evidence remains linked [1].",
        cite_details=[
            {
                "num": 1,
                "anchor": "kb-cite-stale-1",
                "source_name": "stale.pdf",
                "source_path": r"db\stale\stale.en.md",
                "citation_route": "system_a",
                "is_inpaper": False,
            }
        ],
        refs_user_msg_id=1,
        render_packet={
            "answer_markdown": historical_answer,
            "rendered_content": stale_rendered,
        },
    )
    cache["input_ref_sig"] = "stale-ref-signature"
    cache["citation_plan_sig"] = "stale-plan-signature"
    annotate_inputs: list[str] = []

    def fake_annotate(md, _hits, **_kwargs):
        annotate_inputs.append(str(md))
        if "[1]" not in str(md):
            return str(md), []
        return str(md).replace("[1]", "[1](#kb-cite-fresh-1)"), [
            {
                "num": 1,
                "anchor": "kb-cite-fresh-1",
                "source_name": "fresh.pdf",
                "source_path": r"db\fresh\fresh.en.md",
                "citation_route": "system_a",
                "is_inpaper": False,
            }
        ]

    monkeypatch.setattr(
        chat_render,
        "_answer_aligned_reference_render_pack",
        lambda pack, _answer: dict(pack or {}),
    )
    monkeypatch.setattr(
        chat_render,
        "_should_link_inpaper_citations_for_message",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        fake_annotate,
    )
    messages = [
        {"id": 1, "role": "user", "content": "Show the evidence."},
        {
            "id": 2,
            "role": "assistant",
            "content": historical_answer,
            "meta": {"render_cache": cache},
        },
        {"id": 3, "role": "assistant", "content": "Latest answer without a citation."},
    ]
    refs_by_user = {
        1: {
            "prompt_sig": "fresh-prompt",
            "hits": [
                {
                    "text": "Fresh evidence.",
                    "meta": {
                        "source_path": r"db\fresh\fresh.en.md",
                        "ref_answer_citation_num": 1,
                    },
                }
            ],
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="historical-cache-signature-guard",
    )
    historical_rendered = str(rendered[1].get("rendered_body") or "")

    assert historical_answer in annotate_inputs
    assert "#kb-cite-fresh-1" in historical_rendered
    assert "#kb-cite-stale-1" not in historical_rendered
    assert cache["input_ref_sig"] == "stale-ref-signature"
    assert cache["citation_plan_sig"] == "stale-plan-signature"


def test_render_cache_persists_render_packet_when_contracts_present(monkeypatch, tmp_path: Path):
    from api import chat_render

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _hits, anchor_ns, canonical_paths
        return (
            f"cached::{_md}",
            [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("cache contract test")
    user_id = store.append_message(conv_id, "user", "test")
    store.append_message(
        conv_id,
        "assistant",
        "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
        meta={"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
    )
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "test",
            "used_translation": False,
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    enrich_messages_with_reference_render(store.get_messages(conv_id), refs_by_user, conv_id=conv_id, chat_store=store)
    persisted = store.get_messages(conv_id)[-1]
    render_cache = ((persisted.get("meta") or {}).get("render_cache") or {})
    render_packet = render_cache.get("render_packet")

    assert isinstance(render_packet, dict)
    assert str(render_packet.get("rendered_content") or "").strip()


def test_enrich_messages_rebuilds_degraded_numeric_citation_cache(tmp_path: Path):
    from api import chat_render

    content = (
        "成像质量提升：深度学习能够改善单像素成像的重建质量 [1]。\n\n"
        "降低采样率：端到端模型可以在更少测量下恢复目标图像 [2]。"
    )
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("degraded citation cache")
    user_id = store.append_message(conv_id, "user", "深度学习对单像素成像有什么好处？")
    assistant_id = store.append_message(conv_id, "assistant", content)
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "single pixel imaging deep learning",
            "used_translation": False,
            "hits": [
                {
                    "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md",
                        "heading_path": "Benefits / Image quality",
                    },
                },
                {
                    "text": "Learning based image reconstruction can reduce the sampling ratio.",
                    "meta": {
                        "source_path": r"db\Optics-2024-Part-based image-loop network for single-pixel imaging.en.md",
                        "heading_path": "Method / Sampling ratio",
                    },
                },
            ],
        }
    }
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body=content,
                rendered_content=content,
                copy_markdown=content,
                copy_text=content,
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet={"rendered_content": content, "cite_details": []},
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )
    msg = rendered[-1]
    persisted = store.get_messages(conv_id)[-1]
    persisted_cache = ((persisted.get("meta") or {}).get("render_cache") or {})

    assert "](#kb-cite-" in str(msg.get("rendered_content") or "")
    assert len(msg.get("cite_details") or []) == 2
    assert all(item.get("is_inpaper") is False for item in (msg.get("cite_details") or []))
    assert len(persisted_cache.get("cite_details") or []) == 2


def test_enrich_messages_rebuilds_matching_empty_cache_after_fast_refs_arrive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    content = (
        "探测器综述总结了 PMT 和 SPAD 探测器限制 [2]。\n\n"
        "PIDL 使用校准后的 SPAD 物理噪声模型（图 1a 及对应证据 [1]）训练网络。"
    )
    pidl_path = r"db\pidl.en.md"
    detector_path = r"db\detector-review.en.md"
    citation_plan: dict = {}
    refs_pack = {
        "prompt_sig": "fast-refs-sig",
        "used_query": "single photon detector review physics informed learning",
        "used_translation": False,
        "render_status": "",
        "render_attempts": 0,
        "hits": [
            {
                "text": "PIDL 使用校准后的 SPAD 物理噪声模型训练网络。",
                "meta": {
                    "source_path": pidl_path,
                    "source_name": "Physics-informed deep learning.pdf",
                    "heading_path": "Introduction",
                    "ref_answer_citation_num": 1,
                },
            },
            {
                "text": "探测器综述总结了 PMT 和 SPAD 探测器限制。",
                "meta": {
                    "source_path": detector_path,
                    "source_name": "Detector review.pdf",
                    "heading_path": "Abstract",
                    "ref_answer_citation_num": 2,
                },
            },
        ],
    }
    monkeypatch.setattr(
        chat_render,
        "_answer_aligned_reference_render_pack",
        lambda raw_pack, _answer_text: dict(raw_pack or {}),
    )
    monkeypatch.setattr(
        chat_render,
        "_effective_citation_render_locale",
        lambda _ref_pack=None: "zh",
    )

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("late fast refs cache")
    user_id = store.append_message(conv_id, "user", "这两篇应该怎样搭配阅读？")
    assistant_id = store.append_message(
        conv_id,
        "assistant",
        content,
        meta={"answer_quality": {"citation_plan": citation_plan}},
    )
    input_ref_sig = chat_render._raw_reference_render_cache_input_signature(refs_pack)
    answer_sig = chat_render._answer_render_signature(content)
    citation_plan_sig = chat_render._stable_json_hash(citation_plan)
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=refs_pack,
        provenance=None,
        citation_plan=citation_plan,
        render_locale="zh",
    )
    empty_cache = chat_render._build_render_cache_payload(
        cache_key=cache_key,
        notice="",
        rendered_body=content,
        rendered_content=content,
        copy_markdown=content,
        copy_text=content,
        cite_details=[],
        refs_user_msg_id=user_id,
        render_packet={
            "answer_markdown": content,
            "rendered_body": content,
            "rendered_content": content,
            "copy_markdown": content,
            "copy_text": content,
            "cite_details": [],
        },
        answer_sig=answer_sig,
        input_ref_sig=input_ref_sig,
        citation_plan_sig=citation_plan_sig,
        locale="zh",
    )
    assert empty_cache["schema"] == 54
    assert empty_cache["cache_key"] == cache_key
    assert empty_cache["answer_sig"] == answer_sig
    assert empty_cache["input_ref_sig"] == input_ref_sig
    assert empty_cache["citation_plan_sig"] == citation_plan_sig
    assert empty_cache["locale"] == "zh"
    store.merge_message_meta(assistant_id, {"render_cache": empty_cache})

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        {user_id: refs_pack},
        conv_id=conv_id,
        chat_store=store,
    )
    message = rendered[-1]
    rendered_content = str(message.get("rendered_content") or "")
    details = [
        item for item in list(message.get("cite_details") or []) if isinstance(item, dict)
    ]
    persisted = store.get_messages(conv_id)[-1]
    persisted_cache = ((persisted.get("meta") or {}).get("render_cache") or {})

    assert rendered_content.count("](#kb-cite-") >= 2
    assert {str(item.get("source_path") or "") for item in details} == {
        pidl_path,
        detector_path,
    }
    assert persisted_cache.get("cite_details")
    assert (persisted_cache.get("render_packet") or {}).get("cite_details")


def test_render_cache_key_changes_when_citation_plan_evidence_changes():
    from api import chat_render

    common = {
        "conv_id": "conv-plan-cache",
        "msg_id": 10,
        "role": "assistant",
        "content": "s2ISM provides super-resolution and optical sectioning [1].",
        "refs_user_msg_id": 9,
        "ref_pack": {"hits": []},
        "provenance": None,
    }
    broad_key = chat_render._build_message_render_cache_key(
        **common,
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "evidence_quote": "Current methods do not provide optical sectioning.",
                }
            ]
        },
    )
    direct_key = chat_render._build_message_render_cache_key(
        **common,
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "evidence_quote": (
                        "We reconstruct an image with super-resolution and enhanced optical sectioning."
                    ),
                    "evidence_selection_reason": "microscopy_direct",
                }
            ]
        },
    )

    assert broad_key != direct_key


def test_render_cache_rejects_stale_microscopy_evidence_after_plan_refinement():
    from api import chat_render

    source_path = "db/s2ism/s2ism.en.md"
    cache = {
        "cite_details": [
            {
                "citation_route": "system_a",
                "source_path": source_path,
                "evidence_quote": (
                    "Current image scanning microscopy approaches do not provide optical sectioning "
                    "in thick samples."
                ),
            }
        ]
    }
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "evidence_quote": (
                    "We reconstruct an image with digital and optical super-resolution, high "
                    "signal-to-noise ratio and enhanced optical sectioning."
                ),
                "evidence_selection_reason": "microscopy_direct",
            }
        ]
    }

    assert chat_render._render_cache_missing_authoritative_plan_evidence(cache, plan) is True
    plan["slots"][0].pop("evidence_selection_reason")
    assert chat_render._render_cache_missing_authoritative_plan_evidence(cache, plan) is True
    cache["cite_details"][0]["evidence_quote"] = plan["slots"][0]["evidence_quote"]
    assert chat_render._render_cache_missing_authoritative_plan_evidence(cache, plan) is False


def test_render_cache_rejects_scope_boundary_citation_bound_only_to_metric():
    from api import chat_render

    source_path = "db/perovskite/perovskite.en.md"
    plan = {
        "intent": "scope_boundary",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": (
                    "We demonstrate electrically driven lasing from a dual-cavity "
                    "perovskite device."
                ),
            }
        ],
    }
    cache = {
        "render_packet": {
            "cite_details": [
                {
                    "citation_route": "system_a",
                    "source_path": source_path,
                    "answer_claim": "最低激射阈值为 92 A cm-2。",
                    "evidence_quote": (
                        "The device shows a minimum lasing threshold of 92 A cm-2."
                    ),
                }
            ]
        }
    }

    assert chat_render._render_cache_missing_authoritative_plan_evidence(cache, plan) is True
    cache["render_packet"]["cite_details"][0].update(
        {
            "answer_claim": (
                "这是一项 dual-cavity perovskite 激光器件研究，"
                "而不是单像素成像方法。"
            ),
            "evidence_quote": plan["slots"][0]["evidence_quote"],
        }
    )
    assert chat_render._render_cache_missing_authoritative_plan_evidence(cache, plan) is False


def test_enrich_messages_rebuilds_degraded_structured_citation_cache(monkeypatch, tmp_path: Path):
    from api import chat_render
    from ui import refs_renderer

    source_path = r"db\paper-one.en.md"
    sid = chat_render._source_cite_id(source_path)
    content = f"Prior work is cited as [[CITE:{sid}:35]]."
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("structured citation cache")
    user_id = store.append_message(conv_id, "user", "which prior work is cited?")
    assistant_id = store.append_message(conv_id, "assistant", content)
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-structured-cache",
            "updated_at": 1.0,
            "used_query": "prior work cited",
            "used_translation": False,
            "hits": [
                {
                    "text": "The paper cites compressive sensing as prior work [35].",
                    "meta": {
                        "source_path": source_path,
                        "heading_path": "Related work",
                    },
                }
            ],
        }
    }

    def fake_resolve(_index, src, ref_num, *, source_sha1=""):
        del _index, source_sha1
        if str(src) != source_path or int(ref_num) != 35:
            return None
        return {
            "source_path": source_path,
            "source_name": "paper-one.pdf",
            "ref_num": 35,
            "ref": {
                "raw": "[35] Candes et al. Compressive sensing. 2006.",
                "title": "Compressive sensing",
                "authors": "Candes et al.",
                "year": "2006",
            },
        }

    monkeypatch.setattr(chat_render, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(chat_render, "resolve_reference_entry", fake_resolve)
    monkeypatch.setattr(refs_renderer, "_load_reference_index_cached", lambda: {})
    monkeypatch.setattr(refs_renderer, "_resolve_reference_entry_from_index", fake_resolve)

    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body="Prior work is cited as .",
                rendered_content="Prior work is cited as .",
                copy_markdown="Prior work is cited as .",
                copy_text="Prior work is cited as .",
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet={"rendered_content": "Prior work is cited as .", "cite_details": []},
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )
    msg = rendered[-1]
    persisted_cache = ((store.get_messages(conv_id)[-1].get("meta") or {}).get("render_cache") or {})

    assert "[35](#kb-cite-" in str(msg.get("rendered_content") or "")
    assert len(msg.get("cite_details") or []) == 1
    assert (msg.get("cite_details") or [{}])[0].get("is_inpaper") is True
    assert len(persisted_cache.get("cite_details") or []) == 1


def test_enrich_messages_ignores_previous_schema_render_cache(tmp_path: Path):
    from api import chat_render

    content = "Learning-based SPI improves reconstruction quality [1]."
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("previous schema cache")
    user_id = store.append_message(conv_id, "user", "what helps SPI reconstruction?")
    assistant_id = store.append_message(conv_id, "assistant", content)
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-prev-cache",
            "updated_at": 1.0,
            "used_query": "SPI reconstruction",
            "used_translation": False,
            "hits": [
                {
                    "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\LPR-2025\LPR-2025.en.md",
                        "heading_path": "Benefits / Reconstruction quality",
                    },
                }
            ],
        }
    }
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    old_cache = chat_render._build_render_cache_payload(
        cache_key=cache_key,
        notice="",
        rendered_body="stale plain [1]",
        rendered_content="stale plain [1]",
        copy_markdown="stale plain [1]",
        copy_text="stale plain [1]",
        cite_details=[],
        refs_user_msg_id=user_id,
        render_packet={"rendered_content": "stale plain [1]", "cite_details": []},
    )
    old_cache["schema"] = int(chat_render._RENDER_CACHE_SCHEMA_VERSION) - 1
    store.merge_message_meta(assistant_id, {"render_cache": old_cache})

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )
    msg = rendered[-1]
    persisted_cache = ((store.get_messages(conv_id)[-1].get("meta") or {}).get("render_cache") or {})

    assert str(msg.get("rendered_content") or "") != "stale plain [1]"
    assert "](#kb-cite-" in str(msg.get("rendered_content") or "")
    assert int(persisted_cache.get("schema") or 0) == int(chat_render._RENDER_CACHE_SCHEMA_VERSION)
    assert len(persisted_cache.get("cite_details") or []) == 1


def test_enrich_assistant_only_slice_recovers_reference_owner_without_empty_overwrite(
    tmp_path: Path,
) -> None:
    content = "Learning-based SPI improves reconstruction quality [1]."
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("assistant-only page")
    user_id = store.append_message(conv_id, "user", "what helps SPI reconstruction?")
    store.append_message(conv_id, "assistant", content)
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-assistant-only-page",
            "updated_at": 1.0,
            "used_query": "SPI reconstruction",
            "used_translation": False,
            "hits": [
                {
                    "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\LPR-2025\LPR-2025.en.md",
                        "heading_path": "Benefits / Reconstruction quality",
                    },
                }
            ],
        }
    }

    primed = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )
    assert primed[-1]["refs_user_msg_id"] == user_id
    assert len(primed[-1].get("cite_details") or []) == 1

    assistant_record = store.get_messages(conv_id)[-1]
    rendered_slice = enrich_messages_with_reference_render(
        [assistant_record],
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
    )

    assert rendered_slice[0]["refs_user_msg_id"] == user_id
    assert len(rendered_slice[0].get("cite_details") or []) == 1
    persisted = store.get_messages(conv_id)[-1]
    render_cache = ((persisted.get("meta") or {}).get("render_cache") or {})
    packet = (
        (((persisted.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet"))
        or {}
    )
    assert render_cache.get("refs_user_msg_id") == user_id
    assert len(packet.get("cite_details") or []) == 1


def test_render_packet_only_rebuilds_legacy_answer_markdown_citations_when_content_empty(tmp_path: Path):
    from api import chat_render

    answer = "Learning-based SPI improves reconstruction quality [1] and reduces sampling demand [2]."
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("legacy packet repair")
    user_id = store.append_message(conv_id, "user", "what are the benefits of deep learning for SPI?")
    assistant_id = store.append_message(
        conv_id,
        "assistant",
        "",
        meta={
            "paper_guide_contracts": {
                "version": 1,
                "intent": {"family": "overview"},
                "render_packet": {
                    "answer_markdown": answer,
                    "rendered_body": "",
                    "rendered_content": "",
                    "copy_markdown": "",
                    "copy_text": "",
                    "cite_details": [],
                },
            }
        },
    )
    refs_by_user = {
        user_id: {
            "prompt_sig": "sig-legacy",
            "updated_at": 1.0,
            "used_query": "single pixel imaging deep learning benefits",
            "used_translation": False,
            "hits": [
                {
                    "text": "Deep learning improves reconstruction quality in single-pixel imaging.",
                    "meta": {
                        "source_path": r"db\paper-a.en.md",
                        "heading_path": "Benefits / Reconstruction quality",
                    },
                },
                {
                    "text": "Learning based reconstruction can reduce sampling demand.",
                    "meta": {
                        "source_path": r"db\paper-b.en.md",
                        "heading_path": "Benefits / Sampling demand",
                    },
                },
            ],
        }
    }
    stale_packet = (((store.get_messages(conv_id)[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=answer,
        refs_user_msg_id=user_id,
        ref_pack=refs_by_user[user_id],
        provenance=None,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body="",
                rendered_content="",
                copy_markdown="",
                copy_text="",
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet=stale_packet,
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user,
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=True,
    )
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    persisted = store.get_messages(conv_id)[-1]
    persisted_packet = ((((persisted.get("meta") or {}).get("render_cache") or {}).get("render_packet") or {}))

    assert "rendered_body" not in msg
    assert len(msg.get("cite_details") or []) == 2
    assert "](#kb-cite-" in str(packet.get("rendered_body") or "")
    assert "](#kb-cite-" in str(packet.get("rendered_content") or "")
    assert len(packet.get("cite_details") or []) == 2
    assert str(packet.get("answer_markdown") or "") == answer
    assert len(persisted_packet.get("cite_details") or []) == 2


def test_enrich_provenance_surfaces_hidden_derived_formula_source_anchor():
    provenance = {
        "source_path": "paper.pdf",
        "segments": [
            {
                "segment_id": "seg-formula",
                "claim_type": "formula_claim",
                "formula_origin": "derived",
                "evidence_mode": "direct",
                "locate_policy": "hidden",
                "locate_surface_policy": "hidden",
                "primary_heading_path": "How a single-pixel camera works",
                "primary_block_id": "blk-12",
                "primary_anchor_id": "p-12",
                "support_locate_anchor": "The single-pixel camera consists of two main components.",
                "locate_target": {
                    "segmentId": "seg-formula",
                    "headingPath": "How a single-pixel camera works",
                    "snippet": "The single-pixel camera consists of two main components.",
                    "blockId": "blk-12",
                    "anchorId": "p-12",
                    "anchorKind": "equation",
                    "locatePolicy": "hidden",
                    "locateSurfacePolicy": "hidden",
                },
                "reader_open": {
                    "sourcePath": "paper.pdf",
                    "blockId": "blk-12",
                    "anchorId": "p-12",
                    "anchorKind": "equation",
                    "strictLocate": False,
                    "locateTarget": {
                        "blockId": "blk-12",
                        "anchorId": "p-12",
                        "anchorKind": "equation",
                        "locatePolicy": "hidden",
                        "locateSurfacePolicy": "hidden",
                    },
                },
            }
        ],
        "block_map": {
            "blk-12": {
                "block_id": "blk-12",
                "anchor_id": "p-12",
                "heading_path": "How a single-pixel camera works",
                "text": "The single-pixel camera consists of two main components.",
                "kind": "paragraph",
            }
        },
    }

    out = _enrich_provenance_segments_for_display(provenance, [], anchor_ns="conv:1:2:test")
    seg = (out.get("segments") or [])[0]

    assert seg.get("locate_policy") == "required"
    assert seg.get("locate_surface_policy") == "primary"
    assert seg.get("locate_target", {}).get("locatePolicy") == "required"
    assert seg.get("locate_target", {}).get("anchorKind") in {"paragraph", "sentence"}
    assert seg.get("reader_open", {}).get("strictLocate") is True


def test_enrich_messages_refreshes_stale_cached_render_packet_from_provenance(tmp_path: Path):
    from api import chat_render

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("stale cache packet test")
    user_id = store.append_message(conv_id, "user", "Explain the SPI workflow.")
    content = "Grounded answer."
    good_locate_target = {
        "segmentId": "seg_001",
        "headingPath": "Abstract / Acquisition and image reconstruction strategies.",
        "snippet": "Unlike the raster-scan strategy...",
        "blockId": "blk-good-26",
        "anchorId": "p-good-19",
        "anchorKind": "sentence",
        "hitLevel": "exact",
        "locatePolicy": "required",
        "locateSurfacePolicy": "primary",
    }
    provenance = {
        "status": "ready",
        "strict_identity_ready": True,
        "must_locate_count": 1,
        "strict_identity_count": 1,
        "segments": [
            {
                "segment_id": "seg_001",
                "source_segment_id": "seg_001",
                "claim_type": "critical_fact_claim",
                "must_locate": True,
                "locate_policy": "required",
                "locate_surface_policy": "primary",
                "evidence_mode": "direct",
                "primary_block_id": "blk-good-26",
                "primary_anchor_id": "p-good-19",
                "locate_target": good_locate_target,
                "reader_open": {
                    "sourcePath": "demo.en.md",
                    "headingPath": "Abstract / Acquisition and image reconstruction strategies.",
                    "blockId": "blk-good-26",
                    "anchorId": "p-good-19",
                    "anchorKind": "sentence",
                    "strictLocate": True,
                    "locateTarget": good_locate_target,
                },
            }
        ],
    }
    stale_packet = {
        "answer_markdown": content,
        "rendered_body": content,
        "rendered_content": content,
        "copy_markdown": content,
        "copy_text": content,
        "locate_target": {
            "segmentId": "seg_001",
            "snippet": "Grounded answer.",
            "hitLevel": "none",
            "locatePolicy": "hidden",
            "locateSurfacePolicy": "hidden",
        },
        "reader_open": {},
        "visible_segment_ids": [],
    }
    assistant_id = store.append_message(
        conv_id,
        "assistant",
        content,
        meta={
            "provenance": provenance,
            "paper_guide_contracts": {
                "version": 1,
                "intent": {"family": "paper_guide"},
                "render_packet": stale_packet,
            },
        },
    )
    cache_key = chat_render._build_message_render_cache_key(
        conv_id=conv_id,
        msg_id=assistant_id,
        role="assistant",
        content=content,
        refs_user_msg_id=user_id,
        ref_pack=None,
        provenance=provenance,
    )
    store.merge_message_meta(
        assistant_id,
        {
            "render_cache": chat_render._build_render_cache_payload(
                cache_key=cache_key,
                notice="",
                rendered_body=content,
                rendered_content=content,
                copy_markdown=content,
                copy_text=content,
                cite_details=[],
                refs_user_msg_id=user_id,
                render_packet=stale_packet,
            )
        },
    )

    rendered = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user={},
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=True,
    )
    packet = (((rendered[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    persisted = store.get_messages(conv_id)[-1]
    cache_packet = (((persisted.get("meta") or {}).get("render_cache") or {}).get("render_packet") or {})

    assert packet.get("locate_target", {}).get("blockId") == "blk-good-26"
    assert packet.get("visible_segment_ids") == ["seg_001"]
    assert cache_packet.get("locate_target", {}).get("blockId") == "blk-good-26"
    assert cache_packet.get("visible_segment_ids") == ["seg_001"]

    second = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_by_user={},
        conv_id=conv_id,
        chat_store=store,
        render_packet_only=True,
    )
    second_packet = (((second[-1].get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert second_packet.get("locate_target", {}).get("blockId") == "blk-good-26"


def test_render_packet_only_env_strips_legacy_render_fields(monkeypatch):
    from api import chat_render

    monkeypatch.setenv("KB_CHAT_RENDER_PACKET_ONLY", "1")
    messages = [
        {"id": 1, "role": "user", "content": "explain"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(messages, refs_by_user, conv_id="conv-test")
    msg = rendered[-1]

    assert "rendered_body" not in msg
    assert "rendered_content" not in msg
    assert "copy_text" not in msg
    assert "copy_markdown" not in msg
    assert isinstance(msg.get("cite_details"), list)
    assert "notice" not in msg

    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert str(packet.get("rendered_body") or "").strip()
    assert isinstance(packet.get("cite_details"), list)


def test_render_packet_only_flag_strips_legacy_render_fields(monkeypatch):
    from api import chat_render

    # No env needed; flag should be enough.
    monkeypatch.delenv("KB_CHAT_RENDER_PACKET_ONLY", raising=False)
    messages = [
        {"id": 1, "role": "user", "content": "explain"},
        {
            "id": 2,
            "role": "assistant",
            "content": "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "citation_lookup"}}},
        },
    ]
    refs_by_user = {
        1: {
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]

    assert "rendered_body" not in msg
    assert "rendered_content" not in msg
    assert "copy_text" not in msg
    assert "copy_markdown" not in msg
    assert isinstance(msg.get("cite_details"), list)
    assert "notice" not in msg


def test_figure_claim_segments_can_reach_exact_hit_level_after_required_coverage_contract():
    from api import chat_render

    messages = [
        {"id": 1, "role": "user", "content": "show me figure 6"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Figure 6 shows the pipeline.",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "md_path": "demo.md",
                "segments": [
                    {
                        "segment_id": "seg-fig-1",
                        "text": "Figure 6 shows the pipeline.",
                        "evidence_mode": "direct",
                        "claim_type": "figure_claim",
                        "must_locate": True,
                        "locate_policy": "required",
                        "locate_surface_policy": "primary",
                        "primary_heading_path": "Methods / Figure 6",
                        "primary_block_id": "blk_demo_00001",
                        "primary_anchor_id": "fg_00006",
                        # anchor_kind intentionally omitted; contract should fill it.
                    }
                ],
                "block_map": {
                    "blk_demo_00001": {
                        "block_id": "blk_demo_00001",
                        "anchor_id": "fg_00006",
                        "kind": "figure",
                        "heading_path": "Methods / Figure 6",
                        "text": "Figure 6",
                        "line_start": 1,
                        "line_end": 1,
                        "number": 6,
                    }
                },
            },
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "figure_walkthrough"}}},
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    prov = msg.get("provenance") or {}
    segs = prov.get("segments") or []
    seg = segs[0] if isinstance(segs, list) and segs else {}

    assert str(seg.get("hit_level") or "") == "exact"


def test_figure_panel_segments_can_reach_exact_hit_level_after_required_coverage_contract():
    from api import chat_render

    messages = [
        {"id": 1, "role": "user", "content": "what does panel (b) show"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Panel (b) compares the enhancement performance.",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "md_path": "demo.md",
                "segments": [
                    {
                        "segment_id": "seg-figp-1",
                        "text": "Panel (b) compares the enhancement performance.",
                        "evidence_mode": "direct",
                        "claim_type": "figure_panel",
                        "must_locate": True,
                        "locate_policy": "required",
                        "locate_surface_policy": "primary",
                        "primary_heading_path": "Methods / Figure 6",
                        "primary_block_id": "blk_demo_00002",
                        "primary_anchor_id": "p_00068",
                        # anchor_kind intentionally omitted; contract should fill it.
                        "support_slot_figure_number": 6,
                        "support_slot_panel_letters": ["b"],
                    }
                ],
                "block_map": {
                    "blk_demo_00002": {
                        "block_id": "blk_demo_00002",
                        "anchor_id": "p_00068",
                        "kind": "paragraph",
                        "heading_path": "Methods / Figure 6",
                        "text": "Figure 6 ... b The enhancement comparison ...",
                        "line_start": 1,
                        "line_end": 1,
                        "number": 0,
                        "paper_figure_number": 6,
                    }
                },
            },
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "figure_walkthrough"}}},
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    prov = msg.get("provenance") or {}
    segs = prov.get("segments") or []
    seg = segs[0] if isinstance(segs, list) and segs else {}

    assert str(seg.get("hit_level") or "") == "exact"


def test_figure_claim_prefers_caption_block_as_primary_locate_target_when_available():
    from api import chat_render

    messages = [
        {"id": 1, "role": "user", "content": "what does figure 6 show"},
        {
            "id": 2,
            "role": "assistant",
            "content": "Figure 6 shows the pipeline.",
            "provenance": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "md_path": "demo.md",
                "segments": [
                    {
                        "segment_id": "seg-fig-prim-1",
                        "text": "Figure 6 shows the pipeline.",
                        "evidence_mode": "direct",
                        "claim_type": "figure_claim",
                        "must_locate": True,
                        "locate_policy": "required",
                        "locate_surface_policy": "primary",
                        "primary_heading_path": "Methods / Figure 6",
                        "primary_block_id": "blk_demo_fig",
                        "primary_anchor_id": "fg_00006",
                        "paper_figure_number": 6,
                    }
                ],
                "block_map": {
                    "blk_demo_fig": {
                        "block_id": "blk_demo_fig",
                        "anchor_id": "fg_00006",
                        "kind": "figure",
                        "heading_path": "Methods / Figure 6",
                        "text": "Figure 6",
                        "line_start": 1,
                        "line_end": 1,
                        "number": 6,
                        "paper_figure_number": 6,
                    },
                    "blk_demo_cap": {
                        "block_id": "blk_demo_cap",
                        "anchor_id": "p_00068",
                        "kind": "paragraph",
                        "figure_role": "caption",
                        "paper_figure_number": 6,
                        "heading_path": "Methods / Figure 6",
                        "text": "**Figure 6.** Caption text for the pipeline.",
                        "raw_text": "**Figure 6.** Caption text for the pipeline.",
                        "line_start": 2,
                        "line_end": 2,
                    },
                },
            },
            "meta": {"paper_guide_contracts": {"version": 1, "intent": {"family": "figure_walkthrough"}}},
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    reader_open = packet.get("reader_open") or {}

    assert str(reader_open.get("blockId") or "") == "blk_demo_cap"
    assert str(reader_open.get("anchorId") or "") == "p_00068"


def test_render_packet_notice_is_not_dropped_when_preserving_existing_render(monkeypatch):
    from api import chat_render

    monkeypatch.delenv("KB_CHAT_RENDER_PACKET_ONLY", raising=False)
    messages = [
        {"id": 1, "role": "user", "content": "explain"},
        {
            "id": 2,
            "role": "assistant",
            # This prefix triggers _split_kb_miss_notice() and produces a non-empty notice.
            "content": "未命中知识库片段\nBody that cannot be re-rendered without hits.",
            "meta": {
                "paper_guide_contracts": {
                    "version": 1,
                    "intent": {"family": "citation_lookup"},
                    # Existing contract has cite_details but no notice; preserving existing render
                    # should still pick up the current notice extracted from content.
                    "render_packet": {
                        "notice": "",
                        "rendered_body": "Existing rendered body [1](#kb-cite-demo-1).",
                        "rendered_content": "Existing rendered body [1](#kb-cite-demo-1).",
                        "copy_markdown": "Existing rendered body [1](#kb-cite-demo-1).",
                        "copy_text": "Existing rendered body [1].",
                        "cite_details": [{"num": 1, "anchor": "kb-cite-demo-1", "source_name": "demo.pdf"}],
                    },
                }
            },
        },
    ]

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user={},
        conv_id="conv-test",
        render_packet_only=True,
    )
    msg = rendered[-1]

    assert "notice" not in msg
    packet = (((msg.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert "未命中知识库片段" in str(packet.get("notice") or "")
    assert isinstance(packet.get("cite_details"), list)


def test_merge_render_packet_contract_meta_drops_stale_negative_locate_when_no_current_identity():
    from api import chat_render

    rec = {
        "id": 2,
        "role": "assistant",
        "content": "The paper does not mention ADMM in the retrieved context.",
        "rendered_body": "The paper does not mention ADMM in the retrieved context.",
        "rendered_content": "The paper does not mention ADMM in the retrieved context.",
        "copy_markdown": "The paper does not mention ADMM in the retrieved context.",
        "copy_text": "The paper does not mention ADMM in the retrieved context.",
        "meta": {
            "paper_guide_contracts": {
                "version": 1,
                "intent": {"family": "overview"},
                "render_packet": {
                    "rendered_body": "The paper does not mention ADMM in the retrieved context.",
                    "rendered_content": "The paper does not mention ADMM in the retrieved context.",
                    "copy_markdown": "The paper does not mention ADMM in the retrieved context.",
                    "copy_text": "The paper does not mention ADMM in the retrieved context.",
                    "locate_target": {
                        "segmentId": "seg-neg",
                        "headingPath": "Discussion",
                        "snippet": "The paper does not mention ADMM in the retrieved context.",
                        "anchorText": "The paper does not mention ADMM in the retrieved context.",
                        "blockId": "b-neg",
                        "anchorId": "a-neg",
                        "anchorKind": "sentence",
                        "locatePolicy": "required",
                        "locateSurfacePolicy": "primary",
                    },
                    "reader_open": {
                        "sourcePath": "demo.md",
                        "headingPath": "Discussion",
                        "snippet": "The paper does not mention ADMM in the retrieved context.",
                        "blockId": "b-neg",
                        "anchorId": "a-neg",
                        "anchorKind": "sentence",
                        "strictLocate": True,
                    },
                },
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=2,
        enriched_provenance={"segments": []},
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert packet.get("locate_target") == {}
    assert packet.get("reader_open") == {}


def test_merge_render_packet_contract_meta_surfaces_primary_evidence_from_provenance():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_name": "demo.pdf",
                    "heading_path": "Methods / APR",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=2,
        enriched_provenance={
            "segments": [
                {
                    "segment_id": "seg-1",
                    "locate_policy": "required",
                    "locate_target": {
                        "segmentId": "seg-1",
                        "headingPath": "Methods / APR",
                        "blockId": "b-7",
                    },
                    "reader_open": {
                        "sourcePath": "demo.md",
                        "headingPath": "Methods / APR",
                        "blockId": "b-7",
                    },
                }
            ],
            "primary_evidence": {
                "source_path": "demo.md",
                "source_name": "demo.pdf",
                "block_id": "b-7",
                "anchor_id": "a-7",
                "heading_path": "Methods / APR",
                "snippet": "APR uses phase correlation for registration.",
                "selection_reason": "provenance_segment",
            },
        },
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert packet.get("primary_evidence", {}).get("block_id") == "b-7"
    assert packet.get("primary_evidence", {}).get("heading_path") == "Methods / APR"
    assert packet.get("reader_open", {}).get("blockId") == "b-7"


def test_merge_render_packet_contract_meta_prefers_shared_primary_identity_over_drifting_provenance():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_path": "oe.md",
                    "source_name": "OE-2017.pdf",
                    "block_id": "b-22",
                    "anchor_id": "a-22",
                    "heading_path": "2. Comparison / 2.2 Basis patterns generation",
                    "snippet": "Fourier basis patterns are strictly periodical.",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=3,
        enriched_provenance={
            "segments": [
                {
                    "segment_id": "seg-1",
                    "locate_policy": "required",
                    "locate_target": {
                        "segmentId": "seg-1",
                        "headingPath": "2. Comparison / 2.2 Basis patterns generation",
                        "blockId": "b-22",
                    },
                    "reader_open": {
                        "sourcePath": "oe.md",
                        "headingPath": "2. Comparison / 2.2 Basis patterns generation",
                        "blockId": "b-22",
                    },
                }
            ],
            "primary_evidence": {
                "source_path": "natphoton.md",
                "source_name": "NatPhoton-2019.pdf",
                "block_id": "b-nat",
                "anchor_id": "a-nat",
                "heading_path": "Abstract / Acquisition and image reconstruction strategies.",
                "snippet": "A broader overview paragraph.",
                "selection_reason": "provenance_segment",
            },
        },
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    assert packet.get("primary_evidence", {}).get("source_name") == "OE-2017.pdf"
    assert packet.get("primary_evidence", {}).get("block_id") == "b-22"
    assert packet.get("primary_evidence", {}).get("heading_path") == "2. Comparison / 2.2 Basis patterns generation"


def test_merge_render_packet_contract_meta_refreshes_contract_primary_from_refs_pack():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_path": "sciadv.md",
                    "source_name": "SciAdv-2017.pdf",
                    "heading_path": "INTRODUCTION",
                    "snippet": "A broad answer-hit snippet.",
                    "selection_reason": "answer_hit_top",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=4,
        enriched_provenance={"segments": []},
        ref_pack={
            "primary_evidence": {
                "source_path": "sciadv.md",
                "source_name": "SciAdv-2017.pdf",
                "block_id": "blk_30",
                "anchor_id": "a_30",
                "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
                "snippet": "dynamic supersampling is defined here.",
                "selection_reason": "prompt_aligned",
            }
        },
        chat_store=None,
    )

    contracts = ((rec.get("meta") or {}).get("paper_guide_contracts") or {})
    packet = contracts.get("render_packet") or {}
    assert (contracts.get("primary_evidence") or {}).get("block_id") == "blk_30"
    assert (contracts.get("primary_evidence") or {}).get("heading_path") == "INTRODUCTION / Spatially variant digital supersampling"
    assert (packet.get("primary_evidence") or {}).get("block_id") == "blk_30"


def test_merge_render_packet_contract_meta_backfills_system_a_card_from_ref_primary_evidence():
    from api import chat_render

    rec = {
        "content": "Light-field microscopy solves the depth-of-field trade-off [1].",
        "rendered_body": "Light-field microscopy solves the depth-of-field trade-off [1](#kb-cite-demo-1).",
        "rendered_content": "Light-field microscopy solves the depth-of-field trade-off [1](#kb-cite-demo-1).",
        "copy_markdown": "Light-field microscopy solves the depth-of-field trade-off [1].",
        "copy_text": "Light-field microscopy solves the depth-of-field trade-off [1].",
        "notice": "",
        "cite_details": [
            {
                "num": 1,
                "anchor": "kb-cite-demo-1",
                "source_path": "db/qclfm/qclfm.en.md",
                "source_name": "QCLFM.pdf",
                "citation_route": "system_a",
                "is_inpaper": False,
                "heading_path": "I. INTRODUCTION",
                "answer_claim": "Light-field microscopy solves the depth-of-field trade-off.",
                "evidence_quote": (
                    "# Quantum correlation light-field microscope with extreme depth of field\n"
                    "Yingwen Zhang,$^{1,2,*}$ Duncan England"
                ),
                "raw": (
                    "# Quantum correlation light-field microscope with extreme depth of field\n"
                    "Yingwen Zhang,$^{1,2,*}$ Duncan England"
                ),
                "card_quality_flags": ["evidence_quote_filtered", "missing_evidence_quote"],
            }
        ],
        "meta": {"paper_guide_contracts": {"render_packet": {}}},
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=6,
        enriched_provenance={"segments": []},
        ref_pack={
            "hits": [
                {
                    "text": "rough title text",
                    "meta": {"source_path": "db/qclfm/qclfm.en.md"},
                    "ui_meta": {
                        "primary_evidence": {
                            "source_path": "db/qclfm/qclfm.en.md",
                            "source_name": "QCLFM.pdf",
                            "block_id": "blk_light_field",
                            "anchor_id": "p_light_field",
                            "heading_path": "I. INTRODUCTION / Light-field microscopy",
                            "snippet": (
                                "Light-field microscopy is a 3D microscopy technique whereby volumetric "
                                "information of a sample is gained in a single shot."
                            ),
                            "highlight_snippet": (
                                "Light-field microscopy is a 3D microscopy technique whereby volumetric "
                                "information of a sample is gained in a single shot."
                            ),
                            "anchor_kind": "paragraph",
                            "selection_reason": "prompt_aligned",
                        }
                    },
                }
            ]
        },
        chat_store=None,
    )

    packet = (((rec.get("meta") or {}).get("paper_guide_contracts") or {}).get("render_packet") or {})
    details = packet.get("cite_details") or []
    assert len(details) == 1
    detail = details[0]
    assert detail["block_id"] == "blk_light_field"
    assert detail["anchor_id"] == "p_light_field"
    assert detail["heading_path"] == "I. INTRODUCTION / Light-field microscopy"
    assert "volumetric information" in detail["card_evidence"]
    assert "Yingwen Zhang" not in detail["card_evidence"]
    assert "##" not in detail["card_evidence"]


def test_chat_messages_merge_cached_reference_payload_prefers_enriched_hits():
    from api.routers import chat, references

    raw_refs = {101: {"prompt": "raw prompt", "hits": [{"text": "raw only"}]}}
    references._store_cached_conversation_refs_payload(
        conv_id="conv-cached-refs",
        signature="conversation-sig",
        refs=raw_refs,
        payload={
            101: {
                "prompt": "cached prompt",
                "hits": [
                    {
                        "text": "enriched",
                        "ui_meta": {
                            "primary_evidence": {
                                "snippet": "Precise cached evidence.",
                                "block_id": "blk_cached",
                            }
                        },
                    }
                ],
            }
        },
    )

    merged = chat._merge_cached_reference_render_payload(
        "conv-cached-refs",
        raw_refs,
    )

    assert merged[101]["prompt"] == "raw prompt"
    assert merged[101]["hits"][0]["text"] == "raw only"
    assert merged[101]["rendered_payload"]["prompt"] == "cached prompt"
    assert merged[101]["rendered_payload"]["hits"][0]["text"] == "enriched"
    assert merged[101]["rendered_payload"]["hits"][0]["ui_meta"]["primary_evidence"]["block_id"] == "blk_cached"


def test_chat_messages_ignore_stale_cached_refs_and_rerender_current_primary_evidence():
    from api.chat_render import enrich_messages_with_reference_render
    from api.routers import chat, references

    source_path = "db/paper/current.en.md"
    old_refs = {
        101: {
            "prompt": "compare the evidence",
            "prompt_sig": "prompt-sig",
            "updated_at": 10.0,
            "hits": [
                {
                    "text": "Old result evidence.",
                    "meta": {"source_path": source_path, "heading_path": "Old results"},
                }
            ],
        }
    }
    references._store_cached_conversation_refs_payload(
        conv_id="conv-stale-cached-refs",
        signature="old-conversation-sig",
        refs=old_refs,
        payload={
            101: {
                "hits": [
                    {
                        "text": "Old result evidence.",
                        "meta": {"source_path": source_path, "heading_path": "Old results"},
                        "ui_meta": {
                            "primary_evidence": {
                                "source_path": source_path,
                                "heading_path": "Old results",
                                "snippet": "Old result evidence.",
                                "block_id": "blk_old",
                                "anchor_id": "p_old",
                            }
                        },
                    }
                ]
            }
        },
    )
    current_primary = {
        "source_path": source_path,
        "source_name": "Current Paper.pdf",
        "heading_path": "Abstract",
        "snippet": "Current primary evidence supports the answer.",
        "highlight_snippet": "Current primary evidence supports the answer.",
        "block_id": "blk_current",
        "anchor_id": "p_current",
        "anchor_kind": "paragraph",
        "strict_locate": True,
    }
    current_refs = {
        101: {
            "prompt": "compare the evidence",
            "prompt_sig": "prompt-sig",
            "updated_at": 20.0,
            "hits": [
                {
                    "text": current_primary["snippet"],
                    "meta": {
                        "source_path": source_path,
                        "source_name": current_primary["source_name"],
                        "heading_path": current_primary["heading_path"],
                    },
                    "ui_meta": {"primary_evidence": current_primary},
                }
            ],
            "primary_evidence": current_primary,
        }
    }

    merged = chat._merge_cached_reference_render_payload(
        "conv-stale-cached-refs",
        current_refs,
    )

    assert "rendered_payload" not in merged[101]
    rendered = enrich_messages_with_reference_render(
        [
            {"id": 101, "role": "user", "content": "compare the evidence"},
            {
                "id": 102,
                "role": "assistant",
                "content": "The current evidence supports this claim [1].",
                "meta": {"canonical_hit_paths": [source_path]},
            },
        ],
        merged,
        conv_id="conv-stale-cached-refs",
    )[-1]
    details = list(rendered.get("cite_details") or [])
    assert len(details) == 1
    assert details[0]["block_id"] == "blk_current"
    assert details[0]["heading_path"] == "Abstract"
    assert "Current primary evidence" in details[0]["evidence_quote"]


def test_chat_messages_do_not_overlay_unverifiable_authoritative_doc_list_cache():
    from api.routers import chat, references

    raw_refs = {
        101: {
            "prompt": "compare selected papers",
            "prompt_sig": "same-prompt",
            "updated_at": 10.0,
            "hits": [{"text": "selected", "meta": {"source_path": "selected.en.md"}}],
        }
    }
    references._store_cached_conversation_refs_payload(
        conv_id="conv-authoritative-stale",
        signature="cached-conversation",
        refs=raw_refs,
        payload={
            101: {
                "hits": [
                    {"text": "selected", "meta": {"source_path": "selected.en.md"}},
                    {"text": "stale extra", "meta": {"source_path": "extra.en.md"}},
                ],
                "pipeline_debug": {"doc_list_authoritative": True},
            }
        },
    )

    merged = chat._merge_cached_reference_render_payload(
        "conv-authoritative-stale",
        raw_refs,
    )

    assert "rendered_payload" not in merged[101]
    assert [hit["meta"]["source_path"] for hit in merged[101]["hits"]] == [
        "selected.en.md"
    ]


def test_effective_reference_pack_keeps_raw_hit_order_and_exposes_enriched_hits():
    from api.chat_render import _effective_reference_render_pack

    pack = {
        "hits": [{"text": "raw generation hit", "meta": {"source_path": "raw.md"}}],
        "rendered_payload": {
            "hits": [
                {
                    "text": "enriched reference hit",
                    "ui_meta": {"primary_evidence": {"snippet": "precise"}},
                }
            ]
        },
    }

    effective = _effective_reference_render_pack(pack)

    assert effective["hits"][0]["text"] == "raw generation hit"
    assert effective["enriched_hits"][0]["text"] == "enriched reference hit"


def test_effective_reference_pack_prefers_authoritative_doc_list_hits():
    from api.chat_render import _effective_reference_render_pack

    pack = {
        "hits": [{"text": "raw generation hit", "meta": {"source_path": "paper.md"}}],
        "rendered_payload": {
            "hits": [
                {
                    "text": "authoritative card hit",
                    "meta": {"source_path": "paper.md"},
                    "ui_meta": {"citation_meta": {"doi": "10.1000/example", "journal_if": 12.3}},
                }
            ],
            "pipeline_debug": {"doc_list_authoritative": True},
        },
    }

    effective = _effective_reference_render_pack(pack)

    assert effective["hits"][0]["text"] == "authoritative card hit"
    assert effective["hits"][0]["ui_meta"]["citation_meta"]["doi"] == "10.1000/example"
    assert effective["retrieval_hits"][0]["text"] == "raw generation hit"
    assert "enriched_hits" not in effective


def test_effective_reference_pack_preserves_generation_hits_when_cards_are_answer_aligned():
    from api.chat_render import _effective_reference_render_pack

    pack = {
        "hits": [{"text": "raw generation hit", "meta": {"source_path": "paper.md"}}],
        "rendered_payload": {
            "answer_aligned_citation_cards": True,
            "hits": [
                {
                    "text": "the exact evidence quoted by the answer",
                    "meta": {
                        "source_path": "paper.md",
                        "ref_answer_citation_num": 1,
                        "answer_citation_overlay_grounded": True,
                    },
                    "ui_meta": {
                        "primary_evidence": {
                            "snippet": "the exact evidence quoted by the answer",
                            "strict_locate": True,
                            "selection_reason": "answer_citation_grounded",
                        }
                    },
                }
            ],
        },
    }

    effective = _effective_reference_render_pack(pack)

    assert effective["hits"][0]["text"] == "raw generation hit"
    assert effective["enriched_hits"][0]["text"] == "the exact evidence quoted by the answer"
    assert "retrieval_hits" not in effective


def test_effective_reference_pack_keeps_newer_top_level_authoritative_hits():
    from api.chat_render import _effective_reference_render_pack

    pack = {
        "hits": [
            {
                "text": "new authoritative hit",
                "ui_meta": {"citation_meta": {"doi": "10.1000/new"}},
            }
        ],
        "pipeline_debug": {"doc_list_authoritative": True},
        "rendered_payload": {
            "hits": [{"text": "stale nested hit", "ui_meta": {"citation_meta": {}}}],
            "pipeline_debug": {"doc_list_authoritative": True},
        },
    }

    effective = _effective_reference_render_pack(pack)

    assert effective["hits"][0]["text"] == "new authoritative hit"
    assert effective["hits"][0]["ui_meta"]["citation_meta"]["doi"] == "10.1000/new"


def test_enrich_messages_does_not_mutate_reference_pack() -> None:
    from api.chat_render import enrich_messages_with_reference_render

    refs_by_user = {
        1: {
            "hits": [{"text": "new hit", "meta": {"source_path": "new.md"}}],
            "pipeline_debug": {"doc_list_authoritative": True},
            "rendered_payload": {
                "hits": [{"text": "stale hit", "meta": {"source_path": "stale.md"}}],
                "pipeline_debug": {"doc_list_authoritative": True},
            },
        }
    }

    enrich_messages_with_reference_render(
        [
            {"id": 1, "role": "user", "content": "question"},
            {"id": 2, "role": "assistant", "content": "answer"},
        ],
        refs_by_user,
        conv_id="conv-no-ref-mutation",
    )

    assert refs_by_user[1]["hits"][0]["text"] == "new hit"
    assert refs_by_user[1]["rendered_payload"]["hits"][0]["text"] == "stale hit"


def test_answer_aligned_pack_primary_replaces_stale_precise_system_a_detail():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "scinerf.en.md"
    details = [
        {
            "num": 1,
            "anchor": "cite-a",
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "is_inpaper": False,
            "heading_path": "3. Method / 3.3. Proposed Framework",
            "title": "3. Method / 3.3. Proposed Framework",
            "block_id": "blk_method",
            "anchor_id": "p_method",
            "anchor_kind": "paragraph",
            "evidence_quote": "The camera poses cannot be estimated directly.",
            "summary_line": "The camera poses cannot be estimated directly.",
            "answer_claim": "ADMM is prior work, not an original contribution.",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "block_id": "blk_related",
            "anchor_id": "p_related",
            "anchor_kind": "paragraph",
            "snippet": "Most existing methods employ ADMM [4].",
            "highlight_snippet": "Most existing methods employ ADMM [4].",
            "selection_reason": "answer_aligned_block",
            "strict_locate": True,
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    assert out[0]["heading_path"] == "2. Related Work"
    assert out[0]["block_id"] == "blk_related"
    assert out[0]["anchor_id"] == "p_related"
    assert "existing methods employ ADMM" in out[0]["evidence_quote"]


def test_answer_aligned_primary_matches_same_path_across_different_display_names():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "F:/db/SCINeRF/SCINeRF.en.md"
    details = [
        {
            "num": 1,
            "anchor": "cite-a",
            "source_path": source_path,
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "is_inpaper": False,
            "heading_path": "3. Method",
            "block_id": "blk_method",
            "anchor_id": "p_method",
            "evidence_quote": "A stale method excerpt.",
            "answer_claim": "ADMM is prior work, not an original contribution.",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "2024 IEEE CVPR - SCINeRF.pdf",
            "heading_path": "2. Related Work",
            "block_id": "blk_related",
            "anchor_id": "p_related",
            "snippet": "Most existing methods employ ADMM [4].",
            "selection_reason": "answer_aligned_block",
            "strict_locate": True,
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    assert out[0]["heading_path"] == "2. Related Work"
    assert out[0]["block_id"] == "blk_related"


def test_system_a_primary_backfill_selects_claim_aligned_abstracts_without_relabeling_results(
    tmp_path: Path,
):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n"
        "## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a "
        "single compressed image, extending its application to dynamic 3D scenes.\n\n"
        "## 4.2 Result and Analysis\n\n"
        "The proposed method is evaluated on static datasets.\n",
        encoding="utf-8",
    )
    scinerf_path = tmp_path / "scinerf.en.md"
    scinerf_path.write_text(
        "# SCINeRF\n\n"
        "## Abstract\n\n"
        "Specifically, we formulate the physical imaging process of SCI as part of "
        "the training of NeRF, allowing recovery of complex scene structures.\n\n"
        "## 5. Conclusion\n\n"
        "SCINeRF exploits neural radiance fields as its scene representation.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 1,
            "source_path": str(scigs_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCIGS 从单张压缩图像恢复动态 3D 场景表示。",
            "evidence_quote": "Title: SCIGS",
        },
        {
            "num": 2,
            "source_path": str(scinerf_path),
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCINeRF 基于 NeRF 隐式表示。",
            "evidence_quote": "Title: SCINeRF",
        },
        {
            "num": 3,
            "source_path": str(scigs_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "answer_claim": "SCIGS 在静态数据集上的性能超过多种方法。",
            "heading_path": "4.2 Result and Analysis",
            "evidence_quote": "The proposed method is evaluated on static datasets.",
            "block_id": "blk_results",
            "anchor_id": "p_results",
        },
        {
            "num": 4,
            "source_path": str(scinerf_path),
            "source_name": "SCINeRF.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCINeRF jointly optimizes NeRF parameters and camera poses.",
            "heading_path": "3. Method / 3.3 Proposed Framework",
            "evidence_quote": "The camera poses and NeRF parameters are jointly optimized.",
            "block_id": "blk_camera_pose",
            "anchor_id": "p_camera_pose",
        },
    ]
    pack = {
        "hits": [
            {
                "text": "The proposed method is evaluated on static datasets.",
                "meta": {"source_path": str(scigs_path), "source_name": "SCIGS.pdf"},
                "ui_meta": {
                    "primary_evidence": {
                        "source_path": str(scigs_path),
                        "source_name": "SCIGS.pdf",
                        "heading_path": "4.2 Result and Analysis",
                        "snippet": "The proposed method is evaluated on static datasets.",
                    }
                },
            },
            {
                "text": "SCINeRF exploits neural radiance fields as its scene representation.",
                "meta": {"source_path": str(scinerf_path), "source_name": "SCINeRF.pdf"},
                "ui_meta": {
                    "primary_evidence": {
                        "source_path": str(scinerf_path),
                        "source_name": "SCINeRF.pdf",
                        "heading_path": "5. Conclusion",
                        "snippet": "SCINeRF exploits neural radiance fields as its scene representation.",
                    }
                },
            },
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    scigs_abstract = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scigs_path and "Abstract" in detail["heading_path"]
    )
    scinerf_abstract = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scinerf_path and "Abstract" in detail["heading_path"]
    )
    scigs_results = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scigs_path
        and detail["heading_path"] == "4.2 Result and Analysis"
    )
    scinerf_camera_pose = next(
        detail
        for detail in out
        if Path(detail["source_path"]) == scinerf_path
        and detail["heading_path"] == "3. Method / 3.3 Proposed Framework"
    )
    assert "dynamic" in scigs_abstract["evidence_quote"]
    assert "3D" in scigs_abstract["evidence_quote"]
    assert scigs_abstract["block_id"] and scigs_abstract["anchor_id"]
    assert "physical imaging process" in scinerf_abstract["evidence_quote"]
    assert "NeRF" in scinerf_abstract["evidence_quote"]
    assert scinerf_abstract["block_id"] and scinerf_abstract["anchor_id"]
    assert scigs_results["evidence_quote"] == "The proposed method is evaluated on static datasets."
    assert scinerf_camera_pose["block_id"] == "blk_camera_pose"
    assert scinerf_camera_pose["anchor_id"] == "p_camera_pose"
    assert scinerf_camera_pose["evidence_quote"] == (
        "The camera poses and NeRF parameters are jointly optimized."
    )


def test_system_a_primary_backfill_selects_direct_s2ism_capability_evidence(
    tmp_path: Path,
):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = tmp_path / "s2ism.en.md"
    source_path.write_text(
        "# Structured detection for simultaneous super-resolution and optical sectioning\n\n"
        "## Abstract\n\n"
        "From single-plane acquisition, we reconstruct an image with digital and optical "
        "super-resolution, high signal-to-noise ratio and enhanced optical sectioning.\n\n"
        "## Introduction\n\n"
        "Since super-resolution and optical sectioning are achieved simultaneously, "
        "we named our technique s$^2$ISM (super-resolution sectioning ISM).\n\n"
        "## Results\n\n"
        "More specifically, s2ISM can be applied to any LSM equipped with a detector array.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 3,
            "source_path": str(source_path),
            "source_name": "NatPhoton s2ISM.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "s2ISM 能够同时实现超分辨和光学切片。",
            "heading_path": "Results / Versatility of s2ISM",
            "evidence_quote": (
                "More specifically, s2ISM can be applied to any LSM equipped with a detector array."
            ),
            "block_id": "blk_weak",
            "anchor_id": "p_weak",
        }
    ]
    pack = {
        "hits": [
            {
                "text": "More specifically, s2ISM can be applied to any LSM equipped with a detector array.",
                "meta": {"source_path": str(source_path), "source_name": "NatPhoton s2ISM.pdf"},
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert len(out) == 1
    assert out[0]["heading_path"].endswith("Abstract")
    assert "digital and optical super-resolution" in out[0]["evidence_quote"]
    assert "enhanced optical sectioning" in out[0]["evidence_quote"]
    assert out[0]["block_id"] != "blk_weak"
    assert out[0]["anchor_id"] != "p_weak"


def test_system_a_primary_backfill_preserves_better_citation_plan_evidence() -> None:
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "db/qclfm/qclfm.en.md"
    direct_evidence = (
        "Light-field imaging captures both the position and angular information "
        "of light rays for volumetric reconstruction."
    )
    weaker_alternative = (
        "At a resolving power of 100 micrometers, QCLFM achieved a near infinite "
        "depth of field."
    )
    details = [
        {
            "num": 3,
            "source_path": source_path,
            "source_name": "QCLFM.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": (
                "Light-field records position and angular information for volumetric reconstruction."
            ),
            "heading_path": "Introduction",
            "evidence_quote": direct_evidence,
            "block_id": "blk_direct",
            "anchor_id": "p_direct",
        }
    ]
    pack = {
        "hits": [
            {
                "text": weaker_alternative,
                "meta": {"source_path": source_path, "source_name": "QCLFM.pdf"},
                "ui_meta": {
                    "primary_evidence": {
                        "source_path": source_path,
                        "heading_path": "Results",
                        "snippet": weaker_alternative,
                        "block_id": "blk_weaker",
                        "anchor_id": "p_weaker",
                        "strict_locate": True,
                        "selection_reason": "answer_aligned_block",
                    }
                },
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    assert out[0]["evidence_quote"] == direct_evidence
    assert out[0]["block_id"] == "blk_direct"
    assert out[0]["anchor_id"] == "p_direct"


def test_system_a_primary_backfill_sets_scigs_dynamic_relation_after_replacement(
    tmp_path: Path,
):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = tmp_path / "scigs.en.md"
    source_path.write_text(
        "# SCIGS\n\n## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a single "
        "compressed image, extending its application to dynamic 3D scenes.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 4,
            "source_path": str(source_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCIGS 从单张压缩图像重建动态 3D 场景。",
            "evidence_quote": "Title: SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image",
        }
    ]
    pack = {
        "hits": [
            {
                "text": "The method is evaluated on static datasets.",
                "meta": {"source_path": str(source_path), "source_name": "SCIGS.pdf"},
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert "dynamic 3D scenes" in out[0]["evidence_quote"]
    assert "SCIGS" in out[0]["support_relation"]
    assert "动态 3D" in out[0]["support_relation"]


def test_system_a_primary_relations_do_not_rewrite_unrelated_or_risk_claims():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    scigs_path = "F:/library/scigs.en.md"
    dl_spi_path = "F:/library/dl-spi-review.en.md"
    details = [
        {
            "num": 1,
            "source_path": scigs_path,
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "SCIGS 的静态数据集评测设置仍需要进一步核对。",
            "evidence_quote": "The method is evaluated on static datasets.",
            "support_relation": "保留静态评测说明。",
        },
        {
            "num": 2,
            "source_path": dl_spi_path,
            "source_name": "Deep-learning SPI review.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "深度学习单像素成像的风险是训练时间长且泛化能力有限。",
            "evidence_quote": "Data-driven methods have prolonged training and limited generalization.",
            "support_relation": "保留训练与泛化风险说明。",
        },
        {
            "num": 3,
            "source_path": dl_spi_path,
            "source_name": "Deep-learning SPI review.pdf",
            "citation_route": "system_a",
            "citation_plan_slot": True,
            "answer_claim": "深度学习单像素成像能同时提高重建质量和重建速度。",
            "evidence_quote": "Deep learning provides exceptional reconstruction quality and reconstruction speed.",
        },
    ]
    pack = {
        "hits": [
            {
                "text": "The method is evaluated on static datasets.",
                "meta": {"source_path": scigs_path, "source_name": "SCIGS.pdf"},
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Abstract",
                        "snippet": (
                            "SCIGS reconstructs a 3D explicit scene from one compressed image and "
                            "extends its application to dynamic 3D scenes."
                        ),
                    }
                },
            },
            {
                "text": "Data-driven methods have prolonged training and limited generalization.",
                "meta": {
                    "source_path": dl_spi_path,
                    "source_name": "Deep-learning SPI review.pdf",
                },
                "ui_meta": {
                    "primary_evidence": {
                        "heading_path": "Abstract",
                        "snippet": (
                            "Deep learning provides exceptional reconstruction quality and "
                            "reconstruction speed."
                        ),
                    }
                },
            },
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert out[0]["support_relation"] == "保留静态评测说明。"
    assert out[1]["support_relation"] == "保留训练与泛化风险说明。"
    assert "重建质量" in out[2]["support_relation"]
    assert "重建速度" in out[2]["support_relation"]


def test_abstract_primary_evidence_refreshes_after_markdown_repair(tmp_path: Path):
    from api.chat_render import _abstract_primary_evidence_from_source

    source_path = tmp_path / "paper.en.md"
    source_path.write_text(
        "# Paper\n\n## Abstract\n\nThe original abstract describes static 3D scenes.\n",
        encoding="utf-8",
    )
    first = _abstract_primary_evidence_from_source(str(source_path))

    source_path.write_text(
        "# Paper\n\n## Abstract\n\n"
        "The repaired abstract now describes explicit dynamic 3D scenes in detail.\n",
        encoding="utf-8",
    )
    second = _abstract_primary_evidence_from_source(str(source_path))

    assert "original abstract" in first["snippet"]
    assert "repaired abstract" in second["snippet"]
    assert first["snippet"] != second["snippet"]


def test_claim_aligned_primary_selects_exact_mechanism_block_outside_abstract(tmp_path: Path):
    from api.chat_render import _claim_aligned_abstract_primary_evidence

    source_path = tmp_path / "spad-review.en.md"
    source_path.write_text(
        "# Paper\n\n"
        "## Abstract\n\nA broad review of emerging photodetectors.\n\n"
        "<!-- kb_page: 2 -->\n"
        "## Principle of single photon detection avalanche diode\n\n"
        "A SPAD operates in Geiger mode with a bias higher than its reverse bias "
        "breakdown voltage. Excessive induced current can damage the device, so it "
        "must be supported by a quenching circuit.\n\n"
        "## Low-dimensional devices\n\nPhotogating improves material performance.\n",
        encoding="utf-8",
    )
    pack = {
        "hits": [
            {
                "meta": {
                    "source_path": str(source_path),
                    "source_name": "SPAD review.pdf",
                }
            }
        ]
    }
    detail = {
        "source_path": str(source_path),
        "source_name": "SPAD review.pdf",
        "answer_claim": (
            "SPAD 工作在 Geiger 模式并偏置在击穿电压以上，雪崩后需要淬灭电路。"
        ),
    }

    primary = _claim_aligned_abstract_primary_evidence(pack, detail)

    assert "Principle of single photon detection avalanche diode" in primary["heading_path"]
    assert primary["page_start"] == 2
    assert "quenching circuit" in primary["snippet"]
    assert "Photogating" not in primary["snippet"]


def test_system_a_plan_keeps_existing_answer_grounded_locator():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    grounded = {
        "text": "Exact answer evidence with a verified reader locator.",
        "meta": {
            "source_path": "paper.en.md",
            "ref_answer_citation_num": 1,
            "answer_citation_overlay_grounded": True,
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": "paper.en.md",
                "heading_path": "Results / Exact result",
                "snippet": "Exact answer evidence with a verified reader locator.",
                "block_id": "blk-exact",
                "anchor_id": "p-exact",
                "selection_reason": "answer_citation_grounded",
                "strict_locate": True,
            }
        },
    }
    broad_plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "paper.en.md",
                "source_name": "Paper",
                "heading_path": "Introduction",
                "evidence_quote": "A broader plan seed that has no exact locator.",
                "candidate_hits": [1],
            }
        ]
    }

    hits = _augment_hits_with_system_a_plan_slots(
        [grounded],
        broad_plan,
        reserved_count=1,
        canonical_paths=["paper.en.md"],
    )

    assert len(hits) == 1
    assert hits[0]["text"] == grounded["text"]
    assert hits[0]["ui_meta"]["primary_evidence"]["block_id"] == "blk-exact"


def test_system_a_plan_populates_reserved_canonical_padding_for_missing_third_source():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    paths = ["cassi.en.md", "scinerf.en.md", "scigs.en.md"]
    plan = {
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": paths[2],
                "candidate_refs": [42],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[0],
                "source_name": "CASSI",
                "heading_path": "Abstract",
                "evidence_quote": "Two dispersive elements surround a binary aperture.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[1],
                "source_name": "SCINeRF",
                "heading_path": "Abstract",
                "evidence_quote": "The SCI physical process is part of NeRF training.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[2],
                "source_name": "SCIGS",
                "heading_path": "Abstract",
                "evidence_quote": "SCIGS reconstructs dynamic 3D scenes.",
                "candidate_hits": [3],
            },
        ],
    }

    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "CASSI hit", "meta": {"source_path": paths[0]}},
            {"text": "SCINeRF hit", "meta": {"source_path": paths[1]}},
        ],
        plan,
        reserved_count=3,
    )

    assert hits[2]["meta"]["source_path"] == paths[2]
    assert hits[2]["meta"]["ref_answer_citation_num"] == 3
    assert hits[2]["meta"]["citation_plan_slot"] is True
    assert not hits[2]["meta"].get("citation_plan_padding")
    assert "dynamic 3D scenes" in hits[2]["text"]


def test_system_a_primary_backfill_does_not_relabel_repeated_citation_claim(tmp_path: Path):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n"
        "## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a "
        "single compressed image, extending its application to dynamic 3D scenes.\n\n"
        "## 4.2 Result and Analysis\n\n"
        "The proposed method is evaluated on static datasets.\n",
        encoding="utf-8",
    )
    details = [
        {
            "num": 4,
            "source_path": str(scigs_path),
            "source_name": "SCIGS.pdf",
            "citation_route": "system_a",
            "answer_claim": "SCIGS performs well on several static datasets.",
            "heading_path": "4.2 Result and Analysis",
            "evidence_quote": "The proposed method is evaluated on static datasets.",
        }
    ]
    pack = {
        "hits": [
            {
                "text": "The proposed method is evaluated on static datasets.",
                "meta": {"source_path": str(scigs_path), "source_name": "SCIGS.pdf"},
            }
        ]
    }
    answer_text = (
        "2. **Dynamic scenes**: SCIGS can reconstruct an explicit dynamic 3D scene "
        "from a snapshot compressive image [4].\n\n"
        "- SCIGS is competitive on static datasets [4]."
    )

    out = _backfill_system_a_cite_details_from_ref_pack(
        details,
        pack,
        render_locale="en",
        answer_text=answer_text,
    )

    assert out[0]["answer_claim"] == "SCIGS performs well on several static datasets."
    assert out[0]["heading_path"] == "4.2 Result and Analysis"
    assert out[0]["evidence_quote"] == "The proposed method is evaluated on static datasets."


def test_system_a_primary_backfill_describes_quantitative_measurement_support():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "hsi-fsi.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_path,
            "source_name": "Hadamard versus Fourier.pdf",
            "citation_route": "system_a",
            "answer_claim": "Hadamard 和 Fourier 的选择取决于实验目标。",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "Hadamard versus Fourier.pdf",
            "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
            "block_id": "blk_compare",
            "anchor_id": "p_compare",
            "snippet": (
                "The sampling ratio increases across experiments. "
                "PSNR and SSIM show that FSI converges faster than HSI."
            ),
            "selection_reason": "section_intent_rescue",
            "strict_locate": True,
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert "测量指标" in out[0]["support_relation"]
    assert "采样率" in out[0]["support_relation"]
    assert "PSNR" in out[0]["support_relation"]
    assert "SSIM" in out[0]["support_relation"]


def test_system_a_primary_backfill_describes_device_scope_boundary():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "perovskite-laser.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_path,
            "source_name": "Perovskite laser.pdf",
            "citation_route": "system_a",
            "answer_claim": "这是一篇器件论文，与单像素成像几乎没有交集。",
        }
    ]
    pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "Perovskite laser.pdf",
            "heading_path": "Abstract",
            "snippet": "We demonstrate lasing from an electrically driven dual-cavity perovskite device.",
            "selection_reason": "prompt_aligned",
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert "perovskite" in out[0]["support_relation"]
    assert "器件" in out[0]["support_relation"]
    assert "不是" in out[0]["support_relation"]


def test_reading_guide_repair_adds_missing_system_a_source_to_matching_paragraph():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "建议按以下顺序阅读：\n\n"
        "1. **先读探测器综述**：快速了解单光子探测器、SPAD、暗计数和死时间。\n\n"
        "2. **再读 Physics-informed deep learning 论文**：看它如何建立 SPAD 噪声模型 [1]。"
    )
    hits = [
        {
            "text": "High-resolution single-photon imaging with physics-informed deep learning.",
            "meta": {"source_path": "pidl.md"},
        },
        {
            "text": "Emerging single-photon detection technique for high-performance photodetector.",
            "meta": {"source_path": "spd-review.md"},
        },
    ]
    plan = {
        "slots": [
            {"preferred_system": "system_a", "candidate_hits": [1], "source_name": "physics-informed deep learning"},
            {"preferred_system": "system_a", "candidate_hits": [2], "source_name": "single-photon detection photodetector review"},
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "死时间 [2]。" in repaired
    assert "噪声模型 [1]" in repaired


def test_reading_guide_repairs_cross_language_application_evidence() -> None:
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = "spi-prospects.en.md"
    evidence = (
        "Single-pixel imaging can operate at wavelengths outside the reach of FPA "
        "technology, at high frame rates, and in three dimensions. Applications "
        "include hazardous gas leaks and autonomous vehicles."
    )
    answer = (
        "当普通面阵相机受波段限制、需要高帧率或三维测量时，单像素相机更值得考虑。\n\n"
        "代表性应用包括危险气体泄漏监测和自动驾驶。"
    )
    hits = [
        {
            "text": evidence,
            "meta": {"source_path": source_path, "heading_path": "Abstract"},
        }
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path],
    )

    assert "[1]" in repaired
    # The source supports two independent application claims, so each claim
    # keeps an explicit marker even though both markers open the same card.
    assert repaired.count("[1]") == 2


def test_spi_prospects_repair_restores_full_use_case_boundary_and_clean_evidence() -> None:
    from api.chat_render import _reading_guide_repair_spi_prospects_answer

    source_path = "spi-prospects.en.md"
    evidence = (
        "Modern cameras use focal plane arrays. As the approach suits a wide variety of "
        "detector technologies, images can be collected at wavelengths outside the reach "
        "of FPA technology or at high frame rates or in three dimensions. Promising "
        "applications include hazardous gas leaks and autonomous vehicles."
    )
    hits = [
        {
            "text": f"## Abstract {evidence}",
            "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
            "ui_meta": {
                "source_path": source_path,
                "reader_open": {"snippet": f"## Abstract {evidence}"},
            },
        }
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
                "page_start": 1,
            }
        ],
    }

    repaired = _reading_guide_repair_spi_prospects_answer(
        "代表应用包括危险气体泄漏和自动驾驶 [3]。",
        hits,
        plan,
        canonical_paths=[source_path],
    )

    assert all(term in repaired for term in ("波段", "高帧率", "三维", "危险气体", "自动驾驶"))
    assert repaired.count("[1]") == 2
    assert "[3]" not in repaired
    assert not hits[0]["ui_meta"]["reader_open"]["snippet"].startswith("##")


def test_authoritative_multi_source_path_still_repairs_spi_use_case_marker() -> None:
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    prospects_path = "spi-prospects.en.md"
    upstream_path = "duarte-comparison.md"
    evidence = (
        "Images can be collected at wavelengths outside the reach of FPA technology, "
        "at high frame rates or in three dimensions. Promising applications include "
        "hazardous gas leaks and autonomous vehicles."
    )
    answer = (
        "核心原理是压缩感知重建 [1]。在面阵无法覆盖的波段、高帧率或三维成像场景，"
        "单像素相机更值得采用。\n\n"
        "代表应用包括危险气体泄漏和自动驾驶 3D 态势感知。\n\n"
        "与逐点扫描相比仍有探测器动态范围权衡 [2]。"
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_path": prospects_path,
                "ref_answer_citation_num": 1,
            },
        },
        {
            "text": "The detailed comparison discusses detector dynamic range.",
            "meta": {
                "source_path": upstream_path,
                "ref_answer_citation_num": 2,
            },
        },
    ]
    plan = {
        "intent": "answer_grounding",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": prospects_path,
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
                "page_start": 1,
            },
            {
                "preferred_system": "system_b",
                "source_path": upstream_path,
                "candidate_hits": [2],
            },
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[prospects_path, upstream_path],
    )

    boundary_paragraph = next(
        paragraph
        for paragraph in repaired.split("\n\n")
        if all(term in paragraph for term in ("波段", "高帧率", "三维"))
    )
    assert "[1]" in boundary_paragraph
    assert "危险气体泄漏和自动驾驶 3D 态势感知 [1]" in repaired
    assert "动态范围权衡 [2]" in repaired


def test_fdm_tradeoff_promotes_complete_abstract_evidence_on_existing_hit() -> None:
    from api.chat_render import _reading_guide_promote_fdm_abstract_evidence

    source_path = "fdm.en.md"
    exact = (
        "Here, we implement frequency-division methods to parallelize the single-pixel "
        "imaging process. Our technique enables a trade-off between signal-to-noise ratio "
        "and acquisition speed—without altering detector integration time."
    )
    hits = [
        {
            "text": "Discussion reports a frame-rate increase with reduced SNR.",
            "meta": {"source_path": source_path, "heading_path": "Discussion"},
            "ui_meta": {"source_path": source_path, "heading_path": "Discussion"},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "FDM",
                "heading_path": "Paper / Abstract",
                "evidence_quote": exact,
                "page_start": 1,
            }
        ]
    }

    answer = _reading_guide_promote_fdm_abstract_evidence(
        "频分复用通过并行化提高速度，以信噪比（SNR）为代价，且不改变积分时间 [1]。",
        hits,
        plan,
    )

    assert "[1]" in answer
    assert hits[0]["text"] == exact
    assert hits[0]["meta"]["heading_path"].endswith("Abstract")
    assert "detector integration time" in hits[0]["ui_meta"]["reader_open"]["snippet"]


def test_reading_guide_budget_counts_only_bound_comparison_citations():
    source_path = "hsi-fsi.en.md"
    comparison_heading = (
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging / "
        "3. Comparison of experiment / 3.1 Numerical simulations"
    )
    comparison_evidence = (
        "The coefficients are corrected gradually as the sampling ratio increases. "
        "As indicated by the curves of PSNR, SSIM, and RMSE, the convergence of HSI "
        "is lower than that of FSI."
    )
    otf_heading = (
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging / "
        "2. Comparison of theory / 2.4 Efficiency"
    )
    otf_evidence = (
        "The optical transfer function (OTF), defined as the Fourier transform of "
        "the point spread function, shows how different spatial frequencies are "
        "handled by the system and explains the practical efficiency tradeoff "
        "between Hadamard and Fourier imaging."
    )
    answer = (
        "## 核心对比\n\n"
        "Hadamard 全采样需要 $2N^2$ 次测量，Fourier 需要 $4N^2$ 次；"
        "实验还在不同采样率下比较了 PSNR 与 SSIM。\n\n"
        "## 实用建议\n\n"
        "追求速度时选 Hadamard。需要分析系统的 OTF 和空间频率响应时选 Fourier。"
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                "evidence_quote": "Computational ghost imaging uses a bucket detector.",
                "candidate_hits": [],
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "2. Comparison of theory / 2.1 Principle of HSI and FSI",
                "evidence_quote": "The image is reconstructed by applying an inverse transform.",
                "candidate_hits": [],
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "2. Comparison of theory / 2.4 Efficiency",
                "evidence_quote": (
                    "The optical transfer function (OTF), defined as the Fourier transform "
                    "of the point spread function, shows how different spatial frequencies "
                    "are handled by the system."
                ),
                "candidate_hits": [],
            },
        ],
    }
    primary_evidence = {
        "source_path": source_path,
        "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        "heading_path": comparison_heading,
        "snippet": comparison_evidence,
        "highlight_snippet": comparison_evidence,
        "block_id": "blk_comparison",
        "anchor_id": "p_comparison",
        "anchor_kind": "paragraph",
        "strict_locate": True,
    }
    messages = [
        {"id": 1, "role": "user", "content": "Hadamard 和 Fourier 到底该怎么选？"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer,
            "meta": {
                # Reserve the model's full citation-number range. The comparison
                # rescue must use a new exact-evidence hit, not alias canonical [1].
                "canonical_hit_paths": [source_path] * 6,
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "prompt_family": "compare",
                    "citation_plan": plan,
                },
                "paper_guide_contracts": {"citation_plan": plan},
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    # This mirrors the fully enriched shape: the general hit surface
                    # favors the fluent OTF passage, while primary_evidence carries
                    # the strict quantitative Comparison/3.1 locate target.
                    "text": otf_evidence,
                    "meta": {
                        "source_path": source_path,
                        "source_name": primary_evidence["source_name"],
                        "heading_path": otf_heading,
                    },
                    "ui_meta": {
                        "display_name": primary_evidence["source_name"],
                        "heading_path": otf_heading,
                        "summary_line": "OTF and spatial-frequency efficiency comparison.",
                        "primary_evidence": primary_evidence,
                    },
                }
            ],
            "primary_evidence": primary_evidence,
        }
    }

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-hadamard-fourier",
    )[-1]

    assert rendered["content"] == answer
    assert "#kb-cite-" in rendered["rendered_content"]
    details = rendered["cite_details"]
    assert len(details) == 2
    assert all(detail["citation_route"] == "system_a" for detail in details)
    assert all("Comparison" in detail["heading_path"] for detail in details)
    comparison_detail = next(
        detail
        for detail in details
        if "PSNR" in detail["evidence_quote"] and "SSIM" in detail["evidence_quote"]
    )
    assert comparison_detail["num"] == 1
    assert comparison_detail["answer_hit_num"] > 6
    assert comparison_detail["citation_plan_slot"] is True
    assert comparison_detail["block_id"] == "blk_comparison"
    assert comparison_detail["anchor_id"] == "p_comparison"
    assert any(term in comparison_detail["answer_claim"] for term in ("测量", "采样"))


def test_comparison_rescue_does_not_append_prose_when_model_omits_all_citations():
    source_path = "hsi-fsi.en.md"
    heading = (
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging / "
        "3. Comparison of experiment / 3.1 Numerical simulations"
    )
    evidence = (
        "The coefficients are corrected gradually as the sampling ratio increases. "
        "As indicated by the curves of PSNR, SSIM, and RMSE, the convergence of HSI "
        "is lower than that of FSI."
    )
    answer = (
        "## 核心结论\n\n"
        "追求速度时选 Hadamard；追求物理可解释性时选 Fourier。\n\n"
        "## 证据支撑的权衡\n\n"
        "Hadamard 的二值图案更适合高速 DMD。Fourier 更适合分析空间频率响应。\n\n"
        "## 一句话建议\n\n"
        "快速采集选 Hadamard，需要 OTF 解释时选 Fourier。"
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
                "heading_path": "2. Comparison of theory",
                "evidence_quote": "A theoretical comparison of the two methods.",
                "candidate_hits": [],
            }
        ],
    }
    primary_evidence = {
        "source_path": source_path,
        "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
        "heading_path": heading,
        "snippet": evidence,
        "highlight_snippet": evidence,
        "block_id": "blk_comparison",
        "anchor_id": "p_comparison",
        "anchor_kind": "paragraph",
        "strict_locate": True,
    }
    messages = [
        {"id": 1, "role": "user", "content": "Hadamard 和 Fourier 到底该怎么选？"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer,
            "meta": {
                "canonical_hit_paths": [source_path] * 6,
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "prompt_family": "compare",
                    "citation_plan": plan,
                },
                "paper_guide_contracts": {"citation_plan": plan},
            },
        },
    ]
    refs_by_user = {
        1: {
            "hits": [
                {
                    "text": "Fourier OTF and spatial-frequency interpretation.",
                    "meta": {"source_path": source_path},
                    "ui_meta": {"primary_evidence": primary_evidence},
                }
            ]
        }
    }

    rendered = enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="conv-hadamard-no-model-citations",
    )[-1]

    assert rendered["content"] == answer
    assert rendered["rendered_content"] == answer
    assert "定量对比依据" not in rendered["rendered_content"]
    assert rendered["cite_details"] == []


def test_reading_guide_repairs_uncited_source_definition_from_abstract(tmp_path: Path):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n## Abstract\n\n"
        "SCIGS reconstructs a 3D explicit scene and extends the task to dynamic 3D scenes.\n",
        encoding="utf-8",
    )
    scinerf_path = tmp_path / "scinerf.en.md"
    scinerf_path.write_text(
        "# SCINeRF\n\n## Abstract\n\n"
        "We formulate the physical imaging process of SCI as part of the training of NeRF.\n",
        encoding="utf-8",
    )
    answer = (
        "1. SCIGS can recover a dynamic 3D scene [1].\n"
        "2. Both methods reconstruct a scene [2].\n\n"
        "**Representation**: SCINeRF uses an implicit NeRF representation."
    )
    hits = [
        {"text": "SCIGS title", "meta": {"source_path": str(scigs_path)}},
        {"text": "SCINeRF title", "meta": {"source_path": str(scinerf_path)}},
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(scigs_path),
                "source_name": "ICIP SCIGS",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": str(scinerf_path),
                "source_name": "CVPR SCINeRF",
                "candidate_hits": [2],
            },
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(scigs_path), str(scinerf_path)],
    )

    assert repaired.startswith("## Direct answer")
    assert "SCIGS addresses** explicit 3D reconstruction" in repaired
    assert "dynamic 3D scenes [3]" in repaired
    assert "SCI physical imaging process into NeRF training [4]" in repaired
    assert "SCIGS can recover a dynamic 3D scene [1]" not in repaired
    assert "not be inferred from these abstract passages alone" in repaired
    assert len(hits) == 4
    assert all(hit["meta"]["citation_plan_claim_abstract"] is True for hit in hits[2:])
    assert all("Abstract" in hit["meta"]["heading_path"] for hit in hits[2:])


def test_reading_guide_lineage_rebinds_cassi_and_scinerf_to_direct_evidence():
    from api.chat_render import _reading_guide_repair_lineage_scinerf_evidence

    hits = [
        {"text": "Generic CASSI conclusion.", "meta": {"source_path": "cassi.en.md"}},
        {"text": "Generic SCINeRF conclusion.", "meta": {"source_path": "scinerf.en.md"}},
        {"text": "SCIGS dynamic 3D scene.", "meta": {"source_path": "scigs.en.md"}},
    ]
    plan = {
        "intent": "origin_lookup",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "cassi.en.md",
                "source_name": "CASSI dual-disperser spectral imaging",
                "heading_path": "Abstract",
                "candidate_hits": [1],
                "evidence_quote": (
                    "The system design uses two dispersive elements arranged in opposition "
                    "around a binary-valued aperture code."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": "scinerf.en.md",
                "source_name": "SCINeRF",
                "heading_path": "Conclusion",
                "candidate_hits": [2],
                "evidence_quote": (
                    "SCINeRF learns a 3D scene representation with NeRF from a single "
                    "snapshot compressed image."
                ),
            },
        ],
    }
    answer = (
        "### 1. Dual-disperser spectral imaging\nCASSI is an early spectral system [1].\n\n"
        "### 3. Key transition\nSCINeRF uses NeRF for 3D scenes [2].\n"
        "SCIGS uses a dynamic 3D scene [3]."
    )

    repaired = _reading_guide_repair_lineage_scinerf_evidence(answer, hits, plan)

    assert "two dispersive elements around a binary-valued aperture code" in repaired
    assert "spectral projections [4]" in repaired
    assert "SCINeRF** learns a 3D scene representation" in repaired
    assert "using NeRF [5]" in repaired
    assert "[1]" not in repaired
    assert "[2]" not in repaired
    assert "SCIGS uses a dynamic 3D scene [3]" in repaired
    assert hits[3]["meta"]["citation_plan_lineage_cassi"] is True
    assert hits[4]["meta"]["citation_plan_lineage_scinerf"] is True


def test_reading_guide_lineage_completes_truncated_scigs_stage_from_plan():
    from api.chat_render import _reading_guide_repair_lineage_scinerf_evidence

    hits = [
        {"text": "Generic CASSI.", "meta": {"source_path": "cassi.en.md"}},
        {"text": "Generic SCINeRF.", "meta": {"source_path": "scinerf.en.md"}},
        {"text": "Generic SCIGS.", "meta": {"source_path": "scigs.en.md"}},
    ]
    plan = {
        "intent": "origin_lookup",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "cassi.en.md",
                "source_name": "CASSI dual-disperser spectral imaging",
                "candidate_hits": [1],
                "evidence_quote": (
                    "The system uses two dispersive elements arranged in opposition "
                    "around a binary-valued aperture code."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": "scinerf.en.md",
                "source_name": "SCINeRF",
                "candidate_hits": [2],
                "evidence_quote": (
                    "SCINeRF learns a 3D scene representation with NeRF from a single "
                    "snapshot compressed image."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": "scigs.en.md",
                "source_name": "SCIGS 3D Gaussians Splatting",
                "candidate_hits": [3],
                "evidence_quote": (
                    "The proposed SCIGS is the first to reconstruct a 3D explicit scene "
                    "from a single compressed image, extending its application to dynamic 3D scenes."
                ),
            },
        ],
    }
    answer = (
        "### 1. Dual-disperser spectral imaging\nCASSI starts the lineage [1].\n\n"
        "### 2. SCINeRF\nSCINeRF moves the task to a 3D scene [2].\n\n"
        "### 3. SCI + 3D Gaussian Splatting → SC"
    )

    repaired = _reading_guide_repair_lineage_scinerf_evidence(answer, hits, plan)

    assert "SCIGS / 3DGS" in repaired
    assert "explicit 3D scene from one compressed image" in repaired
    assert "dynamic 3D scenes [6]" in repaired
    assert len(hits) == 6
    assert not repaired.rstrip().endswith("→ SC")
    assert "[1]" not in repaired
    assert "[2]" not in repaired
    assert "[3]" not in repaired
    assert hits[5]["meta"]["citation_plan_lineage_scigs"] is True


def test_reading_guide_lineage_reuses_canonical_system_a_numbers():
    from api.chat_render import _reading_guide_repair_lineage_scinerf_evidence

    source_paths = ["cassi.en.md", "scinerf.en.md", "scigs.en.md"]
    hits = [
        {
            "text": "old",
            "meta": {
                "source_path": path,
                "ref_answer_citation_num": idx,
            },
        }
        for idx, path in enumerate(source_paths, start=1)
    ]
    plan = {
        "intent": "origin_lookup",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_paths[0],
                "source_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "The system uses two dispersive elements surrounding a "
                    "binary-valued aperture code."
                ),
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[1],
                "source_name": "SCINeRF",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "SCINeRF incorporates the physical imaging process into "
                    "NeRF training for a 3D scene."
                ),
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[2],
                "source_name": "SCIGS",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "SCIGS reconstructs an explicit 3D scene from a single "
                    "compressed image and extends to dynamic 3D scenes."
                ),
                "candidate_hits": [3],
            },
        ],
    }
    answer = (
        "## 第三阶段：3D 场景\n"
        "双色散光谱成像走向 SCINeRF [2]，再走向 SCIGS [3]。"
    )

    repaired = _reading_guide_repair_lineage_scinerf_evidence(answer, hits, plan)

    assert len(hits) == 3
    assert "[1]" in repaired
    assert "[2]" in repaired
    assert "[3]" in repaired
    assert all(
        int(hit["meta"]["ref_answer_citation_num"]) == idx
        for idx, hit in enumerate(hits, start=1)
    )
    assert hits[0]["meta"]["citation_plan_lineage_cassi"] is True
    assert hits[1]["meta"]["citation_plan_lineage_scinerf"] is True
    assert hits[2]["meta"]["citation_plan_lineage_scigs"] is True


def test_reading_guide_lineage_recovers_when_provider_stops_inside_scinerf_formula(
    tmp_path,
):
    from api.chat_render import _reading_guide_repair_lineage_scinerf_evidence

    cassi_path = tmp_path / "cassi.en.md"
    scinerf_path = tmp_path / "scinerf.en.md"
    scigs_path = tmp_path / "scigs.en.md"
    cassi_path.write_text(
        "# CASSI\n\n## Abstract\n\n"
        "The system design uses two dispersive elements arranged in opposition "
        "and surrounding a binary-valued aperture code.\n",
        encoding="utf-8",
    )
    scinerf_path.write_text(
        "# SCINeRF\n\n## Abstract\n\n"
        "SCINeRF recovers the underlying 3D scene representation from a single "
        "snapshot compressed image. We formulate the physical imaging process "
        "of SCI as part of the training of NeRF.\n",
        encoding="utf-8",
    )
    scigs_path.write_text(
        "# SCIGS\n\n## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene "
        "from a single compressed image, extending its application to dynamic 3D scenes.\n",
        encoding="utf-8",
    )
    hits = [
        {"text": "CASSI candidate", "meta": {"source_path": str(cassi_path)}},
        {"text": "SCINeRF candidate", "meta": {"source_path": str(scinerf_path)}},
        {"text": "SCIGS candidate", "meta": {"source_path": str(scigs_path)}},
    ]
    plan = {
        "intent": "origin_lookup",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(cassi_path),
                "source_name": "CASSI dual-disperser spectral imaging",
                "candidate_hits": [1],
                "evidence_quote": (
                    "The design uses two dispersive elements arranged in opposition "
                    "around a binary-valued aperture code."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": str(scinerf_path),
                "source_name": "SCINeRF",
                "candidate_hits": [2],
                "evidence_quote": "A broad method-section hit without the method name.",
            },
            {
                "preferred_system": "system_a",
                "source_path": str(scigs_path),
                "source_name": "SCIGS 3D Gaussian Splatting",
                "candidate_hits": [3],
                "evidence_quote": "A broad method-section hit without the abstract conclusion.",
            },
        ],
    }
    answer = (
        "### 1. Spectral imaging\nCASSI starts the lineage [1].\n\n"
        "### 2. SCI + NeRF → SCINeRF\n"
        "- Input: one compressed image and masks.\n"
        "- Key innovation: $Y = \\sum_t \\Phi_t \\odot R_t(\\pi_\\theta)$ "
        "（其中 $R_t$ 是 NeRF 在第 $t$ 时刻\n\n"
        "如果想顺着论文的引用链继续追，可以打开上游综述。"
    )

    repaired = _reading_guide_repair_lineage_scinerf_evidence(answer, hits, plan)

    assert all(term in repaired for term in ("SCINeRF", "物理成像", "NeRF"))
    assert "SCIGS / 3DGS" in repaired
    assert "动态 3D 场景" in repaired
    assert "（其中 $R_t$ 是 NeRF 在第 $t$ 时刻" not in repaired
    assert "如果想顺着论文的引用链继续追" in repaired


def test_authoritative_lineage_mapping_still_runs_exact_evidence_repair():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    sources = ["cassi.en.md", "scinerf.en.md", "scigs.en.md"]
    hits = [
        {
            "text": f"Generic evidence {num}.",
            "meta": {
                "source_path": source,
                "ref_answer_citation_num": num,
            },
        }
        for num, source in enumerate(sources, start=1)
    ]
    plan = {
        "intent": "origin_lookup",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": sources[0],
                "source_name": "CASSI dual-disperser spectral imaging",
                "candidate_hits": [1],
                "evidence_quote": (
                    "The design has two dispersive elements in opposition around "
                    "a binary-valued aperture code."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": sources[1],
                "source_name": "SCINeRF",
                "candidate_hits": [2],
                "evidence_quote": (
                    "SCINeRF learns a 3D scene representation using NeRF from a single "
                    "snapshot compressed image."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": sources[2],
                "source_name": "SCIGS 3D Gaussian Splatting",
                "candidate_hits": [3],
                "evidence_quote": (
                    "SCIGS reconstructs a 3D explicit scene from a single compressed image "
                    "and extends the application to dynamic 3D scenes."
                ),
            },
        ],
    }
    answer = (
        "### 1. Dual-disperser spectral imaging\nCASSI starts the lineage [1].\n\n"
        "### 2. SCINeRF\nSCINeRF moves it to a 3D scene [2].\n\n"
        "### 3. 3D Gaussian Splatting → SC"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=sources,
    )

    assert "two dispersive elements" in repaired
    assert "SCINeRF** learns a 3D scene representation" in repaired
    assert "SCIGS / 3DGS" in repaired
    assert "dynamic 3D scenes [3]" in repaired
    assert len(hits) == 3


def test_reading_guide_keeps_only_planned_system_b_marker_within_budget():
    from api.chat_render import _reading_guide_enforce_system_b_plan_budget

    sid = "s7f6b9404"
    answer = (
        f"Background [[CITE:{sid}:5]][[CITE:{sid}:8]] and selected [[CITE:{sid}:50]].\n"
        f"The selected reference is repeated here [[CITE:{sid}:50]]."
    )
    plan = {
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "candidate_refs": [50],
                "candidate_cite_examples": [f"[[CITE:{sid}:50]]"],
            }
        ],
    }

    repaired = _reading_guide_enforce_system_b_plan_budget(answer, plan)

    assert f"[[CITE:{sid}:5]]" not in repaired
    assert f"[[CITE:{sid}:8]]" not in repaired
    assert repaired.count(f"[[CITE:{sid}:50]]") == 1


def test_reading_guide_keeps_canonical_marker_when_abstract_loses_claim_alignment(tmp_path: Path):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    scigs_path = tmp_path / "scigs.en.md"
    scigs_path.write_text(
        "# SCIGS\n\n## Abstract\n\n"
        "The proposed SCIGS is the first to reconstruct a 3D explicit scene from a single "
        "compressed image, extending its application to dynamic 3D scenes.\n",
        encoding="utf-8",
    )
    answer = (
        "SCIGS extends SCI to dynamic scenes and uses 3DGS as its explicit representation [1].\n\n"
        "SCIGS reconstructs a dynamic 3D scene from one compressed image [1]."
    )
    hits = [
        {
            "text": (
                "SCIGS reconstructs dynamic 3D scenes and uses a transformation network "
                "with pre-trained 3DGS representations."
            ),
            "meta": {"source_path": str(scigs_path), "heading_path": "5. Conclusion"},
        }
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(scigs_path),
                "source_name": "ICIP-2025-SCIGS-3D Gaussians Splatting",
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(scigs_path)],
    )

    assert repaired == answer
    assert len(hits) == 1


def test_reading_guide_replaces_weak_s2ism_marker_with_claim_aligned_abstract(
    tmp_path: Path,
):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = tmp_path / "NatPhoton-Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy.en.md"
    source_path.write_text(
        "# Structured detection for simultaneous super-resolution and optical sectioning\n\n"
        "## Abstract\n\n"
        "From single-plane acquisition, we reconstruct an image with digital and optical "
        "super-resolution, high signal-to-noise ratio and enhanced optical sectioning.\n\n"
        "## Results\n\n### Versatility\n\n"
        "More specifically, the method can be applied to any LSM equipped with a detector array.\n",
        encoding="utf-8",
    )
    other_a = str(tmp_path / "iism.en.md")
    other_b = str(tmp_path / "light-field.en.md")
    answer = "s2ISM 能够同时实现超分辨和光学切片 [3]。"
    hits = [
        {"text": "iISM evidence", "meta": {"source_path": other_a}},
        {"text": "Light-field evidence", "meta": {"source_path": other_b}},
        {
            "text": "More specifically, the method can be applied to any LSM equipped with a detector array.",
            "meta": {
                "source_path": str(source_path),
                "heading_path": (
                    "Structured detection for simultaneous super-resolution and optical "
                    "sectioning in laser scanning microscopy / Results / Versatility"
                ),
            },
        },
    ]
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": (
                    "NatPhoton-2025-Structured detection for simultaneous super-resolution "
                    "and optical sectioning in laser scanning microscopy"
                ),
                "candidate_hits": [3],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[other_a, other_b, str(source_path)],
    )

    assert repaired == "s2ISM 能够同时实现超分辨和光学切片 [4]。"
    assert len(hits) == 4
    assert hits[3]["meta"]["citation_plan_claim_abstract"] is True
    assert hits[3]["meta"]["heading_path"].endswith("Abstract")
    assert "enhanced optical sectioning" in hits[3]["text"]


def test_reading_guide_repairs_s2ism_tradeoff_answer_and_binds_exact_abstract(
    tmp_path: Path,
):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations
    from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta

    source_path = tmp_path / "NatPhoton-Structured detection in laser scanning microscopy.en.md"
    source_path.write_text(
        "# Structured detection for laser scanning microscopy\n\n"
        "## Abstract\n\n"
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. However, current image scanning microscopy approaches "
        "do not provide optical sectioning and fail with thick samples unless the detector "
        "size is limited, introducing a trade-off between optical sectioning and "
        "signal-to-noise ratio.\n\n"
        "## Results\n\nThe method is versatile.\n",
        encoding="utf-8",
    )
    answer = (
        "s2ISM 的核心 trade-off 是分辨率提升与噪声放大之间的平衡。\n\n"
        "关于厚样本，算法假设光学像差可以忽略。"
    )
    hits = [
        {
            "text": "The method can be applied to any LSM equipped with a detector array.",
            "meta": {
                "source_path": str(source_path),
                "heading_path": "Results / Versatility of s2ISM",
            },
        }
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": "NatPhoton Structured detection for s2ISM",
                "heading_path": "Results / Versatility of s2ISM",
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(source_path)],
    )
    repaired_twice = _reading_guide_repair_missing_system_a_citations(
        repaired,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(source_path)],
    )
    assert "空间分辨率与 SNR" in repaired
    assert "光学切片（optical sectioning）与 SNR" in repaired
    assert "限制探测器尺寸" in repaired
    assert "迭代次数" not in repaired
    assert repaired_twice == repaired
    assert hits[-1]["meta"]["citation_plan_s2ism_tradeoff"] is True
    public_source_path = "F:/library/NatPhoton-s2ism.en.md"
    for hit in hits:
        hit["meta"]["source_path"] = public_source_path
        ui_meta = hit.get("ui_meta") if isinstance(hit.get("ui_meta"), dict) else {}
        if ui_meta:
            ui_meta["source_path"] = public_source_path
            primary = ui_meta.get("primary_evidence") if isinstance(ui_meta.get("primary_evidence"), dict) else {}
            if primary:
                primary["source_path"] = public_source_path
    plan["slots"][0]["source_path"] = public_source_path
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[public_source_path],
        citation_plan=plan,
    )
    detail = next(item for item in details if "s2ism" in str(item.get("source_path") or "").lower())
    assert detail["citation_route"] == "system_a"
    assert "Abstract" in detail["heading_path"]
    assert "thick samples" in detail["evidence_quote"]


def test_s2ism_name_detection_accepts_superscript_and_subscript_two():
    from api.chat_render import _mentions_s2ism

    assert _mentions_s2ism("s2ISM")
    assert _mentions_s2ism("s²ISM")
    assert _mentions_s2ism("s₂ISM")


def test_s2ism_tradeoff_repair_checks_correct_terms_only_in_target_paragraph(tmp_path: Path):
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = tmp_path / "s2ism-mixed-paragraphs.en.md"
    source_path.write_text(
        "# Structured detection for laser scanning microscopy\n\n"
        "## Abstract\n\n"
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. Current image scanning microscopy approaches do not "
        "provide optical sectioning and fail with thick samples unless the detector size "
        "is limited, introducing a trade-off between optical sectioning and "
        "signal-to-noise ratio.\n",
        encoding="utf-8",
    )
    answer = (
        "The main s2ISM trade-off is iteration count versus noise amplification.\n\n"
        "Spatial resolution, SNR, and optical sectioning are general microscopy terms "
        "mentioned elsewhere in this answer.\n\n"
        "Thick samples require special care."
    )
    hits = [
        {
            "text": "The method applies to laser scanning microscopy.",
            "meta": {"source_path": str(source_path), "heading_path": "Results"},
        }
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": "Structured detection in laser scanning microscopy",
                "heading_path": "Results",
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[str(source_path)],
    )

    assert "iteration count versus noise amplification" not in repaired
    assert "two coupled trade-offs" in repaired
    assert "spatial resolution versus SNR" in repaired
    assert "optical sectioning versus SNR" in repaired
    assert "iteration count versus noise amplification" not in repaired
    assert hits[-1]["meta"]["citation_plan_s2ism_tradeoff"] is True


def test_s2ism_tradeoff_uses_canonical_source_number_when_refs_are_reordered():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    iism_path = "db/iism.en.md"
    s2ism_path = "db/s2ism.en.md"
    other_path = "db/other.en.md"
    exact_evidence = (
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. Current image scanning microscopy approaches do not "
        "provide optical sectioning and fail with thick samples unless the detector size "
        "is limited, introducing a trade-off between optical sectioning and signal-to-noise ratio."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": s2ism_path,
                "source_name": "Structured detection for s2ISM",
                "heading_path": "Abstract",
                "evidence_quote": exact_evidence,
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": iism_path,
                "source_name": "Interferometric image scanning microscopy",
                "heading_path": "Methods",
                "evidence_quote": "An unrelated interferometric microscope setup.",
            },
            {
                "preferred_system": "system_a",
                "source_path": other_path,
                "source_name": "Other comparison",
                "heading_path": "Results",
                "evidence_quote": "An unrelated comparison passage.",
            },
        ],
    }
    # Refs cards are reordered with s2ISM first, while answer markers follow
    # canonical retrieval order where s2ISM is number 2.
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": exact_evidence, "meta": {"source_path": s2ism_path}},
            {"text": "iISM setup.", "meta": {"source_path": iism_path}},
            {"text": "Other passage.", "meta": {"source_path": other_path}},
        ],
        plan,
        reserved_count=6,
    )
    canonical_paths = [iism_path, s2ism_path, other_path, "extra4", "extra5", "extra6"]
    answer = (
        "## s2ISM trade-off and thick samples\n"
        "The claimed trade-off is spatial resolution versus SNR [2].\n\n"
        "Thick samples are difficult because optical sectioning is limited [2]."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=canonical_paths,
    )

    assert "limiting detector size restores sectioning only by sacrificing SNR" in repaired
    assert "[2]" in repaired
    assert repaired.count("[2]") >= 2
    assert "[7]" not in repaired
    assert "[8]" not in repaired
    assert "[9]" not in repaired
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=canonical_paths,
        citation_plan=plan,
    )
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["source_path"] == s2ism_path
    assert "thick samples" in details[0]["evidence_quote"]


def test_reading_guide_rebinds_foveated_claim_to_plan_passage():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _backfill_system_a_cite_details_from_ref_pack,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "foveated-spi.en.md"
    slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "Adaptive foveated single-pixel imaging with dynamic supersampling",
        "heading_path": "INTRODUCTION",
        "evidence_quote": (
            "This speeds up the frame rate of the vision system. Here, we demonstrate how "
            "an adaptive foveated imaging approach enhances useful data gathering."
        ),
        "candidate_hits": [1],
    }
    plan = {"intent": "answer_grounding", "slots": [slot]}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {
                "text": "Successive frames sample different subsets for dynamic supersampling.",
                "meta": {"source_path": source_path, "heading_path": "Spatially variant supersampling"},
            }
        ],
        plan,
        reserved_count=1,
    )
    answer = (
        "1. 自适应中心凹成像把更多采样资源放在重要区域，从而减少数据量并提高帧率 [1]。\n\n"
        "2. Dynamic supersampling 融合连续帧来补充外围细节 [1]。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path],
    )

    assert "提高帧率 [2]" in repaired
    assert "外围细节 [1]" in repaired


def test_comparison_rescue_reads_strict_source_block_before_async_ref_enrichment(
    tmp_path: Path,
):
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = tmp_path / "hsi-fsi.en.md"
    source_path.write_text(
        "# Hadamard versus Fourier\n\n"
        "## 2. Comparison of theory\n\n"
        "The OTF is an ideal low-pass filter.\n\n"
        "## 3. Comparison of experiment\n\n"
        "### 3.1 Numerical simulations\n\n"
        "The coefficients are corrected gradually as the sampling ratio increases. "
        "As indicated by the curves of PSNR, SSIM, and RMSE, the convergence of HSI "
        "is lower than that of FSI.\n",
        encoding="utf-8",
    )
    weak_slot = {
        "preferred_system": "system_a",
        "source_path": str(source_path),
        "source_name": "Hadamard versus Fourier.pdf",
        "heading_path": "2. Comparison of theory",
        "evidence_quote": "The OTF is an ideal low-pass filter.",
        "candidate_hits": [],
    }
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [weak_slot],
    }
    raw_hits = [
        {
            "text": "The OTF is an ideal low-pass filter.",
            "meta": {
                "source_path": str(source_path),
                "source_name": "Hadamard versus Fourier.pdf",
                "heading_path": "2. Comparison of theory",
            },
        }
    ]

    augmented = _augment_hits_with_system_a_plan_slots(
        raw_hits,
        plan,
        reserved_count=6,
    )

    rescue = augmented[6]
    assert rescue["meta"]["citation_plan_comparison_rescue"] is True
    assert "3. Comparison of experiment" in rescue["meta"]["heading_path"]
    assert "sampling ratio" in rescue["text"]
    assert "PSNR" in rescue["text"]
    assert "SSIM" in rescue["text"]
    assert rescue["meta"]["primary_block_id"]
    assert rescue["meta"]["primary_anchor_id"]


def test_comparison_rescue_does_not_select_unplanned_retrieval_source(tmp_path: Path):
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    target_path = tmp_path / "target.en.md"
    target_path.write_text(
        "# Target\n\n## 3. Comparison of experiment\n\n"
        "At each sampling ratio, PSNR and SSIM compare the two target methods.\n",
        encoding="utf-8",
    )
    extra_path = tmp_path / "extra.en.md"
    extra_path.write_text(
        "# Extra\n\n## 9. Comparison of experiment\n\n"
        "Sampling ratio, measurements, PSNR, SSIM, and RMSE describe an unrelated study.\n",
        encoding="utf-8",
    )
    plan = {
        "intent": "comparison",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(target_path),
                "source_name": "Target.pdf",
                "heading_path": "2. Comparison",
                "evidence_quote": "Target overview.",
                "candidate_hits": [],
            }
        ],
    }
    hits = [
        {"text": "Unrelated", "meta": {"source_path": str(extra_path)}},
        {"text": "Target overview", "meta": {"source_path": str(target_path)}},
    ]

    augmented = _augment_hits_with_system_a_plan_slots(hits, plan)

    rescue = next(
        hit
        for hit in augmented
        if bool((hit.get("meta") or {}).get("citation_plan_comparison_rescue"))
    )
    assert rescue["meta"]["source_path"] == str(target_path)
    assert "target methods" in rescue["text"]
    assert "unrelated study" not in rescue["text"]


def test_system_a_plan_slots_create_distinct_same_paper_evidence_hits():
    from api.chat_render import _augment_hits_with_system_a_plan_slots, _reading_slot_hit_nums

    source_path = "dl-spi-review.en.md"
    hits = [{"text": "Paper overview.", "meta": {"source_path": source_path}}]
    benefit_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL SPI review",
        "heading_path": "Abstract",
        "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
    }
    risk_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL SPI review",
        "heading_path": "4. Strategy and Advantages",
        "evidence_quote": "Data-driven training has limited generalization across imaging scenes.",
    }
    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        {"slots": [benefit_slot, risk_slot]},
    )

    assert len(augmented) == 3
    assert _reading_slot_hit_nums(benefit_slot, augmented) == [2]
    assert _reading_slot_hit_nums(risk_slot, augmented) == [3]

    reserved = _augment_hits_with_system_a_plan_slots(
        hits,
        {"slots": [benefit_slot, risk_slot]},
        reserved_count=3,
    )
    assert _reading_slot_hit_nums(benefit_slot, reserved, canonical_paths=[source_path] * 3) == [4]
    assert _reading_slot_hit_nums(risk_slot, reserved, canonical_paths=[source_path] * 3) == [5]


def test_system_a_plan_slot_marks_existing_candidate_with_authoritative_number():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "simple-baselines.en.md"
    hits = [
        {
            "text": "SIDD PSNR: Baseline ours = 40.30; NAFNet ours = 40.30",
            "meta": {"source_path": source_path},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "candidate_hits": [1],
                "evidence_quote": "SIDD PSNR: Baseline ours = 40.30; NAFNet ours = 40.30",
            }
        ]
    }

    augmented = _augment_hits_with_system_a_plan_slots(hits, plan, reserved_count=1)

    assert augmented[0]["meta"]["ref_answer_citation_num"] == 1


def test_prompt_aligned_source_slot_rebinds_single_paper_canonical_hit():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "spi-prospects.en.md"
    weak_evidence = (
        "It is worth noting that binary sampling can reduce measurement noise."
    )
    exact_evidence = (
        "Images can be collected at wavelengths outside the reach of FPA technology "
        "or at high frame rates or in three dimensions. Promising applications "
        "include hazardous gas leaks and autonomous vehicles."
    )
    hits = [
        {
            "text": weak_evidence,
            "meta": {
                "source_path": source_path,
                "heading_path": "Acquisition strategies",
            },
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": exact_evidence,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            }
        ]
    }

    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=1,
        canonical_paths=[source_path],
    )

    assert augmented[0]["text"] == exact_evidence
    assert augmented[0]["meta"]["citation_plan_slot"] is True
    assert augmented[0]["meta"]["ref_answer_citation_num"] == 1
    assert augmented[0]["ui_meta"]["primary_evidence"]["snippet"] == exact_evidence


def test_same_source_prompt_aligned_slots_do_not_overwrite_primary_visible_evidence():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "denoising-review.en.md"
    taxonomy = (
        "Image denoising methods are classified as spatial domain methods and "
        "transform domain methods."
    )
    wavelet = "The wavelet transform decomposes input data into a scale-space representation."
    hits = [
        {
            "text": "A weak same-paper retrieval passage.",
            "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
            "ui_meta": {},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Classical denoising method",
                "evidence_quote": taxonomy,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Non-data adaptive transform",
                "evidence_quote": wavelet,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
        ]
    }

    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=1,
        canonical_paths=[source_path],
    )

    assert augmented[0]["text"] == taxonomy
    assert augmented[0]["meta"]["ref_answer_citation_num"] == 1
    assert any(hit.get("text") == wavelet for hit in augmented[1:])


def test_single_source_marker_prefers_table_slot_aligned_with_numeric_answer():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "simple-baselines.en.md"
    abstract = (
        "We derive a Nonlinear Activation Free Network, namely NAFNet, from the "
        "baseline. It achieves 40.30 dB PSNR on SIDD."
    )
    table = (
        "Table 6. Image Denoising Results on SIDD. SIDD PSNR: "
        "Restormer = 40.02; Baseline ours = 40.30; NAFNet ours = 40.30"
    )
    hits = [
        {
            "text": table,
            "meta": {
                "source_path": source_path,
                "heading_path": "5.2 Applications",
                "ref_answer_citation_num": 1,
            },
            "ui_meta": {},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": abstract,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "5.2 Applications",
                "evidence_quote": table,
                "candidate_hits": [1],
                "block_id": "table-6",
                "anchor_id": "tb-6",
            },
        ]
    }
    answer = (
        "Table 6 shows that the highest SIDD PSNR is 40.30, tied by "
        "Baseline ours and NAFNet ours [1]."
    )

    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=1,
        canonical_paths=[source_path],
        answer_text=answer,
    )

    assert augmented[0]["text"] == table
    assert augmented[0]["meta"]["heading_path"] == "5.2 Applications"
    assert augmented[0]["ui_meta"]["primary_evidence"]["block_id"] == "table-6"
    assert any(hit.get("text") == abstract for hit in augmented[1:])


def test_single_source_marker_keeps_abstract_when_method_answer_matches_it_best():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "simple-baselines.en.md"
    abstract = (
        "Nonlinear activation functions can be replaced by multiplication or "
        "removed, deriving NAFNet from the baseline."
    )
    table = "Table 6. SIDD PSNR: Baseline ours = 40.30; NAFNet ours = 40.30"
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": abstract,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "5.2 Applications",
                "evidence_quote": table,
                "candidate_hits": [1],
                "block_id": "table-6",
            },
        ]
    }

    augmented = _augment_hits_with_system_a_plan_slots(
        [{"text": table, "meta": {"source_path": source_path}, "ui_meta": {}}],
        plan,
        reserved_count=1,
        canonical_paths=[source_path],
        answer_text=(
            "NAFNet is derived from the baseline by removing nonlinear activation "
            "functions and replacing them with multiplication [1]."
        ),
    )

    assert augmented[0]["text"] == abstract
    assert augmented[0]["meta"]["heading_path"] == "Abstract"


def test_prompt_aligned_source_slot_rebinds_compacted_hit_without_reserved_padding():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "foveated.en.md"
    exact_evidence = (
        "A high-resolution foveal region tracks motion, yet unlike a simple zoom, "
        "every frame delivers information from the entire field of view over "
        "several consecutive frames."
    )
    hits = [
        {
            "text": "A nearby section defines digital supersampling.",
            "meta": {
                "source_path": source_path,
                "heading_path": "Results",
                "ref_answer_citation_num": 3,
            },
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": exact_evidence,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            }
        ]
    }

    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=0,
        canonical_paths=[source_path],
    )

    assert len(augmented) == 1
    assert augmented[0]["text"] == exact_evidence
    assert augmented[0]["meta"]["heading_path"] == "Abstract"
    assert augmented[0]["meta"]["ref_answer_citation_num"] == 3


def test_broad_dl_review_slot_yields_to_two_claim_specific_candidate_rows():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "dl-spi-review.en.md"
    broad = (
        "Deep learning brings exceptional reconstruction quality and fast reconstruction speed. "
        + ("This review surveys data-driven SPI methods and their applications. " * 12)
        + "Data-driven strategies still require prolonged training and have limited generalization."
    )
    risk = (
        "Data-driven strategies have prolonged training duration and limited generalization "
        "when adapting to diverse imaging scenes."
    )
    benefit = (
        "Deep-learning single-pixel imaging provides exceptional reconstruction quality "
        "and fast reconstruction speed."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Challenges and Outlooks",
                "evidence_quote": broad,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Strategy and Advantages",
                "evidence_quote": risk,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": benefit,
                "candidate_hits": [2],
            },
        ]
    }
    hits = [
        {
            "text": "Compacted first hit.",
            "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
        }
    ]

    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=2,
        canonical_paths=[source_path, source_path],
    )

    assert augmented[0]["text"] == risk
    assert augmented[0]["meta"]["heading_path"] == "Strategy and Advantages"
    assert augmented[1]["text"] == benefit
    assert augmented[1]["meta"]["heading_path"] == "Abstract"


def test_foveated_chinese_claim_binds_to_english_abstract_evidence():
    from ui.refs_renderer import _assess_system_a_hit_binding

    claim = (
        "高分辨率中央凹区域跟踪运动；每帧仍从整个视场获取新空间信息，"
        "并通过连续多帧累积慢变区域的细节。"
    )
    evidence = (
        "A high-resolution foveal region tracks motion within the scene, yet unlike "
        "a simple zoom, every frame delivers new spatial information from across the "
        "entire field of view. The system accumulates detail over several consecutive frames."
    )

    binding = _assess_system_a_hit_binding(
        answer_claim=claim,
        hit={"text": evidence},
        meta={},
        heading="Abstract",
        evidence_quote=evidence,
        source_name="Adaptive foveated single-pixel imaging with dynamic supersampling",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "foveated" in binding["overlap_terms"]


def test_compound_plan_evidence_excerpt_keeps_scigs_relationship_and_output_claims():
    from ui.refs_renderer import _compound_plan_evidence_excerpt

    plan_text = (
        "We propose SCIGS, a variant of 3DGS, with a primitive transformation network. "
        "The network uses camera pose stamps and Gaussian coordinates. "
        "SCIGS reconstructs a dynamic 3D scene from a single compressed image."
    )
    answer_claim = (
        "SCIGS 是 3DGS 变体，可从单张压缩图像（single compressed image）"
        "重建动态 3D 场景。"
    )

    excerpt = _compound_plan_evidence_excerpt(plan_text, answer_claim)

    assert "variant of 3DGS" in excerpt
    assert "single compressed image" in excerpt
    assert "dynamic 3D scene" in excerpt
    assert "camera pose stamps" not in excerpt


def test_system_a_evidence_picker_keeps_compound_primary_evidence():
    from ui.refs_renderer import _system_a_pick_best_evidence_candidate

    evidence = (
        "We propose SCIGS, a variant of 3DGS, with a primitive transformation network. "
        "The network uses camera pose stamps and Gaussian coordinates. "
        "SCIGS reconstructs dynamic 3D scenes from a single compressed image."
    )
    primary = {
        "snippet": evidence,
        "heading_path": "Abstract",
        "selection_reason": "prompt_aligned_source_sentence",
        "strict_locate": True,
    }

    picked = _system_a_pick_best_evidence_candidate(
        hit={"text": evidence},
        meta={},
        ui_meta={},
        primary_evidence=primary,
        answer_claim=(
            "SCIGS 是 3DGS 变体，可从单张压缩图像（single compressed image）"
            "重建动态 3D 场景。"
        ),
        source_name="SCIGS.pdf",
        default_heading="Abstract",
    )

    assert picked["compound_evidence"] is True
    assert "variant of 3DGS" in picked["readable_text"]
    assert "single compressed image" in picked["readable_text"]


def test_system_a_canonical_number_matches_public_projected_source_path(monkeypatch):
    from api.chat_render import _annotate_inpaper_citations_with_hover_meta

    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    absolute_path = (
        r"F:\library\ECCV-2022-Simple Baselines for Image Restoration"
        r"\ECCV-2022-Simple Baselines for Image Restoration.en.md"
    )
    public_path = (
        "kb-source/0/ECCV-2022-Simple Baselines for Image Restoration/"
        "ECCV-2022-Simple Baselines for Image Restoration.en.md"
    )
    hits = [
        {
                "text": (
                    "Table 6. SIDD PSNR: Restormer = 40.02; "
                    "Baseline ours = 40.30; NAFNet ours = 40."
                ),
                "meta": {
                "source_path": public_path,
                "source_name": "Simple Baselines for Image Restoration",
                "heading_path": "5 Experiments / 5.2 Applications",
                    "primary_block_id": "blk-table-6",
                    "primary_anchor_id": "tb-6",
                    "ref_snippets": [
                        "Table 6. SIDD PSNR: Restormer = 40.02; "
                        "Baseline ours = 40.30; NAFNet ours = 40.30"
                    ],
                },
        }
    ]

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Restormer PSNR is 40.02 dB [1].",
        hits,
        canonical_paths=[absolute_path],
        citation_plan={"budget": {"system_a": 1, "system_b": 0}},
        render_locale="zh",
    )

    assert "[1](#" in rendered
    assert len(details) == 1
    assert details[0]["citation_route"] == "system_a"
    assert details[0]["binding_status"] == "grounded"
    assert details[0]["source_path"] == public_path
    assert "Baseline ours = 40.30" in details[0]["card_evidence"]


def test_table_backfill_preserves_same_anchor_table_label():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "kb-source/0/Simple-Baselines/Simple-Baselines.en.md"
    compact = "SIDD PSNR: Baseline ours = 40.30; NAFNet ours = 40.30."
    located = "Table 6. Image Denoising Results on SIDD. " + compact
    details = [
        {
            "num": 1,
            "citation_route": "system_a",
            "source_path": source_path,
            "source_name": "Simple Baselines.pdf",
            "heading_path": "5 Experiments / 5.2 Applications",
            "answer_claim": "Baseline ours and NAFNet ours tie at 40.30 PSNR on SIDD.",
            "evidence_quote": compact,
            "block_id": "blk-table-6",
            "anchor_id": "tb-6",
            "anchor_kind": "table",
            "page_start": 13,
            "binding_status": "grounded",
            "binding_confidence": 0.9,
            "score": 60.0,
        }
    ]
    primary = {
        "source_path": source_path,
        "heading_path": "5 Experiments / 5.2 Applications",
        "snippet": compact,
        "block_id": "blk-table-6",
        "anchor_id": "tb-6",
        "anchor_kind": "table",
        "page_start": 13,
        "selection_reason": "answer_citation_grounded",
        "strict_locate": True,
    }
    ref_pack = {
        "primary_evidence": primary,
        "hits": [
            {
                "text": compact,
                "meta": {"source_path": source_path},
                "ui_meta": {
                    "source_path": source_path,
                    "primary_evidence": primary,
                    "reader_open": {
                        "locateTarget": {
                            "sourcePath": source_path,
                            "headingPath": "5 Experiments / 5.2 Applications",
                            "snippet": located,
                            "blockId": "blk-table-6",
                            "anchorId": "tb-6",
                            "anchorKind": "table",
                            "pageStart": 13,
                            "strictLocate": True,
                        }
                    },
                },
            }
        ],
    }

    out = _backfill_system_a_cite_details_from_ref_pack(
        details,
        ref_pack,
        render_locale="en",
    )

    assert len(out) == 1
    assert "Table 6" in out[0]["card_evidence"]
    assert "Baseline ours = 40.30" in out[0]["card_evidence"]
    assert out[0]["page_start"] == 13


def test_reading_guide_deduplicates_equivalent_raw_and_normalized_table_slots():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "db/Simple-Baselines/Simple-Baselines.en.md"
    heading = "5 Experiments / 5.2 Applications"
    normalized = (
        "Table 6. SIDD PSNR: Restormer [39] = 40.02; "
        "Baseline ours = 40.30; NAFNet ours = 40.30"
    )
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": heading,
                "evidence_quote": (
                    "| Method | Restormer [39] | Baseline ours | NAFNet ours | "
                    "| --- | --- | --- | --- | | PSNR | 40.02 | 40.30 | 40.30 |"
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": heading,
                "evidence_quote": normalized,
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": normalized, "meta": {"source_path": source_path, "heading_path": heading}}],
        plan,
        reserved_count=1,
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        "Baseline (ours) 和 NAFNet (ours) 的 SIDD PSNR 均为 40.30，Restormer 为 40.02。",
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path],
    )

    assert repaired.count("[") == 1
    assert "[1]" in repaired


def test_reading_guide_does_not_add_duplicate_plan_slot_citations_to_multi_source_answer():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["review.md", "comparison.md", "frontier.md"]
    hits = [
        {"text": f"Core evidence {idx}.", "meta": {"source_path": source_path}}
        for idx, source_path in enumerate(source_paths, start=1)
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_paths[0],
                "heading_path": "Abstract",
                "evidence_quote": "The review establishes the field overview.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[1],
                "heading_path": "Principles",
                "evidence_quote": "The comparison explains the acquisition strategies.",
            },
        ],
    }
    augmented = _augment_hits_with_system_a_plan_slots(hits, plan, reserved_count=6)
    answer = "Review evidence [1].\n\nComparison evidence [2].\n\nFrontier evidence [3]."

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        augmented,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths + ["extra-4.md", "extra-5.md", "extra-6.md"],
    )

    assert repaired == answer
    assert "[7]" not in repaired
    assert "[8]" not in repaired


def test_reading_guide_roadmap_keeps_review_evidence_without_inserting_bridge():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_paths = ["dl-review.en.md", "hsi-fsi.en.md", "spi-prospects.en.md"]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_paths[0],
                "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "heading_path": "Abstract",
                "evidence_quote": "Deep learning improves reconstruction quality and speed.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[1],
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "heading_path": "Introduction",
                "evidence_quote": "The paper compares HSI and FSI in imaging efficiency and noise robustness.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_paths[2],
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Acquisition and image reconstruction strategies",
                "evidence_quote": (
                    "A single-pixel camera can recover images when the number of measurements is "
                    "fewer than the total number of unknown pixels, also known as under-sampling."
                ),
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Generic DL overview.", "meta": {"source_path": source_paths[0]}},
            {"text": "Generic HSI comparison.", "meta": {"source_path": source_paths[1]}},
            {"text": "Generic SPI review.", "meta": {"source_path": source_paths[2]}},
        ],
        plan,
        reserved_count=6,
    )
    answer = (
        "### Principles and prospects for single-pixel imaging\n"
        "Compressive sensing enables undersampled reconstruction [3].\n\n"
        "### Hadamard versus Fourier\nThe two bases are compared [2].\n\n"
        "### Deep learning review\nQuality and speed are reviewed [1]."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths + ["extra-4.md", "extra-5.md", "extra-6.md"],
    )

    assert "### Principles and prospects for single-pixel imaging [9]" in repaired
    assert "Compressive sensing enables undersampled reconstruction [3]." in repaired
    assert repaired.count("[9]") == 1
    assert "The review states that" not in repaired
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=source_paths + ["extra-4.md", "extra-5.md", "extra-6.md"],
        citation_plan=plan,
    )
    detail = next(item for item in details if int(item.get("num") or 0) == 9)
    assert "number of measurements is fewer" in detail["evidence_quote"]
    assert "unknown pixels" in detail["evidence_quote"]


def test_reading_guide_system_a_plan_enables_linking_without_existing_marker():
    from api.chat_render import _should_link_inpaper_citations_for_message

    rec = {
        "meta": {
            "answer_quality": {
                "output_mode": "reading_guide",
                "citation_plan": {
                    "slots": [
                        {
                            "preferred_system": "system_a",
                            "candidate_hits": [],
                            "source_path": "hsi-fsi.md",
                        }
                    ]
                },
            }
        }
    }

    assert _should_link_inpaper_citations_for_message(
        rec=rec,
        content="追求采集速度选 Hadamard，追求物理可解释性选 Fourier。",
        hits=[{"text": "Hadamard and Fourier comparison", "meta": {"source_path": "hsi-fsi.md"}}],
    ) is True


def test_reading_guide_repair_dedupes_system_a_slots_for_same_source():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "结论：追求采集速度选 Hadamard，追求物理可解释性选 Fourier。\n\n"
        "| 维度 | Hadamard | Fourier |\n"
        "|:---|:---|:---|\n"
        "| 采集 | 快 | 慢 |\n\n"
        "选择建议：\n"
        "1. Hadamard 适合高速 DMD。\n"
        "2. Fourier 适合分析空间频率。\n"
        "3. 两者都属于全局变换。"
    )
    hits = [
        {
            "text": "Hadamard and Fourier single-pixel imaging are compared experimentally.",
            "meta": {"source_path": "hsi-fsi.md"},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "hsi-fsi.md",
                "source_name": "Hadamard Fourier comparison",
                "evidence_quote": "Hadamard and Fourier comparison.",
            },
            {
                "preferred_system": "system_a",
                "source_path": "hsi-fsi.md",
                "source_name": "Hadamard Fourier comparison",
                "evidence_quote": "Fourier spatial frequency comparison.",
            },
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert repaired.count("[1]") == 1
    citation_line = next(line for line in repaired.splitlines() if "[1]" in line)
    assert "Hadamard" in citation_line or "Fourier" in citation_line
    assert "两者都属于全局变换 [1]" not in repaired
    assert "| [1]" not in repaired


def test_reading_guide_repair_does_not_add_ranked_sources_when_every_step_is_already_cited():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "## 1. 综述\n\n**论文：** Review paper [2]\n\n为什么读它：建立全局认识。\n\n"
        "## 2. 实时系统\n\n**论文：** Real-time paper [6]\n\n为什么读它：理解工程实现。"
    )
    hits = [
        {"text": "Unselected ranked source", "meta": {"source_path": "rank-1.md"}},
        {"text": "Review paper", "meta": {"source_path": "review.md"}},
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "rank-1.md",
                "source_name": "Unselected ranked source",
            }
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert repaired == answer
    assert "[1]" not in repaired


def test_reading_guide_repair_resolves_stale_candidate_hit_by_source_path():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "Overview: SCI moves from spectral data cubes toward 3D scene reconstruction.\n\n"
        "Stage 1: early SCI dual-disperser spectral imaging compresses a spectral data cube [1].\n\n"
        "Stage 2: SCINeRF and SCIGS extend SCI to 3D scene reconstruction [2]."
    )
    hits = [
        {"text": "SCINeRF uses snapshot compressive imaging for 3D scene representation.", "meta": {"source_path": "scinerf.md"}},
        {"text": "SCIGS reconstructs dynamic 3D scenes from snapshot compressive images.", "meta": {"source_path": "scigs.md"}},
        {"text": "Single-shot compressive spectral imaging uses a dual-disperser architecture.", "meta": {"source_path": "cassi.md"}},
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "cassi.md",
                "source_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "evidence_quote": "Single-shot compressive spectral imaging uses a dual-disperser architecture.",
            }
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "spectral data cube [1] [3]." in repaired
    assert "Overview: SCI moves from spectral data cubes toward 3D scene reconstruction. [3]" not in repaired


def test_reading_guide_repair_adds_cassi_marker_to_chinese_mechanism_answer():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "CASSI 的双色散结构使用两个色散元件反向排列，中间放置二值孔径编码。\n\n"
        "二值编码提供空间调制，色散把调制扩展到光谱维度，从而完成单次压缩测量。"
    )
    evidence = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )
    hits = [{"text": evidence, "meta": {"source_path": "cassi.en.md"}}]
    plan = {
        "budget": {"system_a": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "cassi.en.md",
                "source_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
            }
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=["cassi.en.md"],
    )

    assert repaired.count("[1]") == 1
    assert "二值孔径编码 [1]。" in repaired


def test_cassi_normalization_splits_exact_architecture_from_broader_projection_claim():
    import re

    from api.chat_render import (
        _reading_guide_normalize_cassi_architecture_terms,
        _reading_guide_repair_mechanism_marker_target,
    )

    evidence = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )
    answer = (
        "其基本思路是：通过一个物理编码掩模（例如双色散器系统 [1]），"
        "将三维数据立方体压缩投影到二维探测器上。"
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "cassi.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    normalized = _reading_guide_normalize_cassi_architecture_terms(answer, plan)
    repaired = _reading_guide_repair_mechanism_marker_target(
        normalized,
        [{"text": evidence, "meta": {"source_path": "cassi.en.md"}}],
        plan,
        canonical_paths=["cassi.en.md"],
    )

    cited_sentence = next(
        sentence
        for sentence in re.split(r"(?<=[。！？])", repaired)
        if "[1]" in sentence
    )
    assert "CASSI（编码孔径快照光谱成像）" in cited_sentence
    assert "两个相向布置的色散元件" in cited_sentence
    assert "二值编码孔径" in cited_sentence
    assert "三维数据立方体" not in cited_sentence
    assert repaired.count("[1]") == 1


def test_cassi_normalization_uses_exact_retrieval_hit_when_plan_quote_is_noisy():
    import re

    from api.chat_render import _reading_guide_normalize_cassi_architecture_terms

    answer = (
        "其基本思路是：通过一个物理编码掩模（例如双色散器系统 [1]），"
        "将三维数据立方体压缩投影到二维探测器上。"
    )
    plan = {
        "slots": [
            {
                "source_path": "cassi.en.md",
                "evidence_quote": "Table data. 2: 15 = 2; 15 = 2",
            }
        ]
    }
    hits = [
        {
            "text": (
                "The primary features of the system design are two dispersive elements, "
                "arranged in opposition and surrounding a binary-valued aperture code."
            ),
            "meta": {"source_path": "cassi.en.md", "heading_path": "Abstract"},
        }
    ]

    normalized = _reading_guide_normalize_cassi_architecture_terms(answer, plan, hits)

    cited_sentence = next(
        sentence
        for sentence in re.split(r"(?<=[。！？])", normalized)
        if "[1]" in sentence
    )
    assert "两个相向布置的色散元件" in cited_sentence
    assert "二值编码孔径" in cited_sentence
    assert "三维数据立方体" not in cited_sentence


def test_cassi_normalization_moves_variable_broad_claim_marker_to_exact_bridge():
    import re

    from api.chat_render import _reading_guide_normalize_cassi_architecture_terms

    answer = (
        "最早的快照压缩成像正是为光谱成像设计的，如 2007 年的双色散架构，"
        "通过编码掩模将高光谱立方体压缩到单次二维测量中 [1]。"
    )
    evidence = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )

    normalized = _reading_guide_normalize_cassi_architecture_terms(
        answer,
        {"slots": [{"evidence_quote": evidence}]},
    )

    cited_sentence = next(
        sentence
        for sentence in re.split(r"(?<=[。！？])", normalized)
        if "[1]" in sentence
    )
    assert "CASSI（编码孔径快照光谱成像）" in cited_sentence
    assert "两个相向布置的色散元件" in cited_sentence
    assert "二值编码孔径" in cited_sentence
    assert "高光谱立方体压缩到" not in cited_sentence
    assert normalized.count("[1]") == 1


def test_cassi_normalization_moves_marker_across_display_equation_to_exact_bridge():
    from api.chat_render import _reading_guide_normalize_cassi_architecture_terms

    evidence = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )
    answer = (
        "The 2007 dual-disperser system uses an encoded aperture and dispersive optics "
        "to compress a spectral data cube into one 2D measurement.\n"
        "$$\nB = Phi vec(I)\n$$\n"
        "The reconstruction then uses a compressive-sensing prior [1]."
    )
    plan = {
        "slots": [
            {
                "candidate_hits": [1],
                "source_path": "cassi.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    normalized = _reading_guide_normalize_cassi_architecture_terms(answer, plan)

    cited_line = next(line for line in normalized.splitlines() if "[1]" in line)
    assert "two dispersive elements" in cited_line
    assert "binary-valued aperture" in cited_line
    assert "compressive-sensing prior [1]" not in normalized
    assert normalized.count("[1]") == 1


def test_cassi_normalization_replaces_unsupported_generic_sci_origin_claim():
    from api.chat_render import _reading_guide_normalize_cassi_architecture_terms

    evidence = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )
    answer = (
        "Snapshot Compressive Imaging (SCI) 最初是为解决高维数据（如高光谱、视频）"
        "记录问题而提出的 [1]。"
    )

    normalized = _reading_guide_normalize_cassi_architecture_terms(
        answer,
        {"slots": [{"candidate_hits": [1], "evidence_quote": evidence}]},
    )

    assert "两个相向布置的色散元件" in normalized
    assert "二值编码孔径" in normalized
    assert "为解决高维数据" not in normalized
    assert normalized.count("[1]") == 1


def test_sequential_support_terms_cover_natural_chinese_recovery_wording():
    from api.chat_render import _reading_guide_normalize_sequential_support_terms

    evidence = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is "
        "proposed and analyzed based on the principle of distilled sensing."
    )
    answer = "顺序自适应压缩感知精确恢复信号的支撑集（support），并分配后续测量。"

    normalized = _reading_guide_normalize_sequential_support_terms(
        answer,
        {"slots": [{"evidence_quote": evidence}]},
    )

    assert "信号支撑集恢复（signal support recovery）" in normalized
    assert "distilled sensing" in normalized


def test_cassi_normalization_recovers_compacted_answer_hit_number():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = "db/CASSI/CASSI.en.md"
    evidence = (
        "The primary features of the system design are two dispersive elements, "
        "arranged in opposition and surrounding a binary-valued aperture code."
    )
    answer = (
        "CASSI uses a dual-disperser architecture and a coded aperture.\n"
        "$$\nB = Phi vec(I)\n$$\n"
        "A compressive-sensing reconstruction recovers the spectral cube [2]."
    )
    plan = {
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [],
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": evidence,
            }
        ],
    }
    hits = [
        {
            "text": "A compacted stale bibliography hit.",
            "meta": {
                "source_path": source_path,
                "ref_answer_citation_num": 2,
            },
        }
    ]

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path, source_path],
    )

    cited_line = next(line for line in repaired.splitlines() if "[2]" in line)
    assert "two dispersive elements" in cited_line
    assert "binary-valued aperture" in cited_line
    assert "reconstruction recovers the spectral cube [2]" not in repaired
    assert repaired.count("[2]") == 1


def test_backfill_cassi_citation_uses_architecture_block_when_claim_omits_acronym(tmp_path):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    md_path = tmp_path / "cassi.en.md"
    md_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 1 -->",
                "## Abstract",
                (
                    "The primary features of the system design are two dispersive elements, "
                    "arranged in opposition and surrounding a binary-valued aperture code."
                ),
                "<!-- kb_page: 8 -->",
                "## Conclusions",
                "The dual-disperser design creates spectral projections on the source datacube.",
            ]
        ),
        encoding="utf-8",
    )
    details = [
        {
            "citation_route": "system_a",
            "source_path": str(md_path),
            "source_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "heading_path": "Conclusions",
            "answer_claim": "这种反向配置让编码孔径的调制通过色散叠加到光谱维度。",
            "evidence_quote": "The dual-disperser design creates spectral projections on the source datacube.",
        }
    ]
    pack = {
        "hits": [
            {
                "text": details[0]["evidence_quote"],
                "meta": {"source_path": str(md_path)},
                "ui_meta": {
                    "source_path": str(md_path),
                    "display_name": details[0]["source_name"],
                },
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="zh")

    assert out[0]["heading_path"] == "Abstract"
    assert out[0]["page_start"] == 1
    assert "two dispersive elements" in out[0]["evidence_quote"]
    assert "binary-valued aperture code" in out[0]["evidence_quote"]


def test_mechanism_marker_targets_sph_beat_frequency_sentence():
    from api.chat_render import (
        _reading_guide_repair_mechanism_marker_target,
    )

    evidence = (
        "Instead of actively performing phase shifting, a beat frequency is introduced between "
        "the signal beam and the reference beam, thereby realizing phase stepping naturally in "
        "time by exploiting the framework of heterodyne holography."
    )
    answer = (
        "这篇论文通过关键设计提升单像素全息（SPH）的吞吐量，并放弃主动相移。\n"
        "本文利用两个 AOM 引入差频的拍频，使相移在时间上自然完成，并采用外差全息。"
    )
    hits = [{"text": evidence, "meta": {"source_path": "sph.en.md"}}]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "sph.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    repaired = _reading_guide_repair_mechanism_marker_target(
        answer,
        hits,
        plan,
        canonical_paths=["sph.en.md"],
    )

    assert "吞吐量，并放弃主动相移。[1]" not in repaired
    assert "采用外差全息 [1]。" in repaired


def test_sequential_support_terms_and_marker_are_normalized_from_exact_source():
    from api.chat_render import (
        _reading_guide_normalize_sequential_support_terms,
        _reading_guide_repair_mechanism_marker_target,
    )

    evidence = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is "
        "proposed and analyzed. The procedure is based on the principle of distilled sensing."
    )
    answer = (
        "顺序压缩感知（SCS）利用自适应反馈分配测量资源，这一思想源于蒸馏感知。\n\n"
        "SCS主要保证的是信号支撑（support）的精确恢复。"
    )
    hits = [{"text": evidence, "meta": {"source_path": "sequential.en.md"}}]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "sequential.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    normalized = _reading_guide_normalize_sequential_support_terms(answer, plan)
    repaired = _reading_guide_repair_mechanism_marker_target(
        normalized,
        hits,
        plan,
        canonical_paths=["sequential.en.md"],
    )

    assert "顺序自适应压缩感知" in repaired
    assert "信号支撑集恢复（signal support recovery）" in repaired
    assert "源于蒸馏感知 [1]。" in repaired


def test_s2ism_tradeoff_marker_targets_three_way_claim():
    from api.chat_render import _reading_guide_repair_mechanism_marker_target

    evidence = (
        "There is a trade-off between spatial resolution and signal-to-noise ratio. "
        "Current approaches do not provide optical sectioning and fail with thick samples."
    )
    answer = (
        "s²ISM 同时改善空间分辨率、信噪比和光学切片能力，打破三方权衡。\n\n"
        "它通过结构化检测完成重建。"
    )
    hits = [{"text": evidence, "meta": {"source_path": "s2ism.en.md"}}]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "s2ism.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    repaired = _reading_guide_repair_mechanism_marker_target(
        answer,
        hits,
        plan,
        canonical_paths=["s2ism.en.md"],
    )

    assert "打破三方权衡 [1]。" in repaired


def test_s2ism_tradeoff_repair_uses_plan_identity_when_model_omits_method_name():
    from api.chat_render import _reading_guide_repair_s2ism_tradeoff_answer

    source_path = "s2ism.en.md"
    evidence = (
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. Current approaches do not provide optical sectioning and "
        "fail with thick samples unless detector size is limited."
    )
    hits = [
        {
            "text": evidence,
            "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
        }
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Structured detection in laser scanning microscopy",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
                "support_example": "State the two trade-offs and explain failure with thick samples.",
            }
        ],
    }

    repaired = _reading_guide_repair_s2ism_tradeoff_answer(
        "普通 ISM 在厚样本中因缺乏光学切片机制而失败 [1]。",
        hits,
        plan,
        canonical_paths=[source_path],
    )

    assert all(term in repaired for term in ("s²ISM", "空间分辨率", "光学切片", "信噪比"))


def test_spad_marker_targets_complete_geiger_breakdown_quenching_claim():
    from api.chat_render import _reading_guide_repair_mechanism_marker_target

    evidence = (
        "A SPAD operates in Geiger mode above its reverse bias breakdown voltage and "
        "must be supported by a quenching circuit."
    )
    answer = (
        "SPAD 在盖革模式下高于击穿电压工作，雪崩后由淬灭电路复位。\n\n"
        "这是单光子探测器的核心工作流程。"
    )
    hits = [{"text": evidence, "meta": {"source_path": "spad.en.md"}}]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "spad.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    repaired = _reading_guide_repair_mechanism_marker_target(
        answer,
        hits,
        plan,
        canonical_paths=["spad.en.md"],
    )

    assert "淬灭电路复位 [1]。" in repaired


def test_spad_marker_adds_complete_source_backed_bridge_when_answer_splits_mechanism():
    from api.chat_render import _reading_guide_repair_mechanism_marker_target

    evidence = (
        "A SPAD operates in Geiger mode above its reverse bias breakdown voltage and "
        "must be supported by a quenching circuit."
    )
    answer = (
        "SPAD operates in Geiger mode to detect individual photons.\n\n"
        "Its bias is higher than breakdown voltage.\n\n"
        "A quenching circuit interrupts the avalanche before reset."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "spad.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    repaired = _reading_guide_repair_mechanism_marker_target(
        answer,
        [{"text": evidence, "meta": {"source_path": "spad.en.md"}}],
        plan,
        canonical_paths=["spad.en.md"],
    )

    cited_line = next(line for line in repaired.splitlines() if "[1]" in line)
    assert "Geiger mode" in cited_line
    assert "breakdown voltage" in cited_line
    assert "quenching circuit" in cited_line
    assert repaired.count("[1]") == 1


def test_sequential_english_label_gets_source_supported_precise_terms():
    from api.chat_render import _reading_guide_normalize_sequential_support_terms

    evidence = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is "
        "proposed and analyzed. The procedure is based on the principle of distilled sensing."
    )
    answer = (
        "Sequential compressed sensing uses feedback from earlier measurements.\n\n"
        "It reliably achieves support set exact recovery at a lower SNR."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "sequential.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    normalized = _reading_guide_normalize_sequential_support_terms(answer, plan)

    assert "Sequential adaptive compressed sensing" in normalized
    assert "based on distilled sensing" in normalized
    assert "signal support recovery" in normalized
    assert not any("\u4e00" <= char <= "\u9fff" for char in normalized)


def test_reading_guide_repair_prefers_canonical_number_for_source_path():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "1. Detector review covers single-photon detectors, SPAD hardware, and photodetector applications.\n\n"
        "2. Physics-informed deep learning models SPAD noise [1]."
    )
    hits = [
        {"text": "Physics-informed deep learning models SPAD noise.", "meta": {"source_path": "pidl.md"}},
        {"text": "A denoising review mentions physics-informed methods.", "meta": {"source_path": "denoise.md"}},
        {"text": "Single-photon detector review covers SPAD devices and applications.", "meta": {"source_path": "spd-review.md"}},
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": "spd-review.md",
                "source_name": "Emerging single-photon detection technique for high-performance photodetector",
                "evidence_quote": "Single-photon detector review covers SPAD devices and applications.",
            }
        ]
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=["pidl.md", "spd-review.md", "piln.md"],
    )

    assert "photodetector applications [2]." in repaired
    assert "photodetector applications [3]." not in repaired


def test_merge_render_packet_contract_meta_allows_refs_pack_to_replace_coarse_cross_paper_seed():
    from api import chat_render

    rec = {
        "content": "Grounded answer.",
        "rendered_body": "Grounded answer.",
        "rendered_content": "Grounded answer.",
        "copy_markdown": "Grounded answer.",
        "copy_text": "Grounded answer.",
        "notice": "",
        "cite_details": [],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": {
                    "source_path": "natphoton.md",
                    "source_name": "NatPhoton-2019.pdf",
                    "heading_path": "Abstract / Acquisition and image reconstruction strategies.",
                    "snippet": "A broad answer-hit snippet.",
                    "selection_reason": "answer_hit_top",
                },
                "render_packet": {},
            }
        },
    }

    chat_render._merge_render_packet_contract_meta(
        rec=rec,
        msg_id=5,
        enriched_provenance={"segments": []},
        ref_pack={
            "primary_evidence": {
                "source_path": "oe2017.md",
                "source_name": "OE-2017.pdf",
                "block_id": "blk_22",
                "anchor_id": "a_22",
                "heading_path": "2. Comparison of theory / 2.2 Basis patterns generation",
                "snippet": "Section 2.2 explicitly compares Hadamard and Fourier basis patterns.",
                "selection_reason": "prompt_aligned",
            }
        },
        chat_store=None,
    )

    contracts = ((rec.get("meta") or {}).get("paper_guide_contracts") or {})
    packet = contracts.get("render_packet") or {}
    assert (contracts.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (contracts.get("primary_evidence") or {}).get("block_id") == "blk_22"
    assert (packet.get("primary_evidence") or {}).get("source_name") == "OE-2017.pdf"
    assert (packet.get("primary_evidence") or {}).get("heading_path") == "2. Comparison of theory / 2.2 Basis patterns generation"


def test_enrich_messages_invalidates_render_cache_when_refs_change(monkeypatch, tmp_path: Path):
    from api import chat_render

    calls = {"primary": 0}

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _hits, anchor_ns, canonical_paths
        calls["primary"] += 1
        anchor = f"kb-cite-demo-{calls['primary']}"
        return (
            str(_md).replace(
                "[[CITE:s1234abcd:1]]",
                f"[1](#{anchor})",
            ),
            [
                {
                    "num": 1,
                    "anchor": anchor,
                    "source_name": "demo.pdf",
                    "is_inpaper": True,
                }
            ],
        )

    monkeypatch.setattr(chat_render, "_annotate_inpaper_citations_with_hover_meta", fake_primary)

    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("cache invalidation test")
    user_id = store.append_message(conv_id, "user", "test")
    assistant_id = store.append_message(conv_id, "assistant", "SPI relies on compressive sensing [[CITE:s1234abcd:1]].")

    refs_v1 = {
        user_id: {
            "prompt_sig": "sig-1",
            "updated_at": 1.0,
            "used_query": "test",
            "used_translation": False,
            "hits": [{"text": "dummy", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }
    refs_v2 = {
        user_id: {
            "prompt_sig": "sig-2",
            "updated_at": 2.0,
            "used_query": "test-updated",
            "used_translation": False,
            "hits": [{"text": "dummy-updated", "meta": {"source_path": r"db\doc\doc.en.md"}}],
        }
    }

    store.merge_message_meta(
        assistant_id,
        {"answer_quality": {"prompt_family": "citation_lookup", "output_mode": "citation_lookup"}},
    )

    first = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_v1, conv_id=conv_id, chat_store=store)
    second = enrich_messages_with_reference_render(store.get_messages(conv_id), refs_v2, conv_id=conv_id, chat_store=store)

    assert calls["primary"] == 2
    assert str(first[-1].get("rendered_content") or "") != str(second[-1].get("rendered_content") or "")


def test_enrich_messages_invalidates_cache_when_only_render_evidence_revision_changes(
    monkeypatch,
    tmp_path: Path,
):
    from api import chat_render

    calls = {"primary": 0}

    def fake_primary(_md, _hits, *, anchor_ns="", canonical_paths=None):
        del _hits, anchor_ns, canonical_paths
        calls["primary"] += 1
        anchor = f"kb-cite-revision-{calls['primary']}"
        return (
            str(_md).replace("[[CITE:s1234abcd:1]]", f"[1](#{anchor})"),
            [
                {
                    "num": 1,
                    "anchor": anchor,
                    "source_name": "demo.pdf",
                    "is_inpaper": True,
                }
            ],
        )

    monkeypatch.setattr(
        chat_render,
        "_annotate_inpaper_citations_with_hover_meta",
        fake_primary,
    )
    store = ChatStore(tmp_path / "chat.db")
    conv_id = store.create_conversation("render evidence revision cache test")
    user_id = store.append_message(conv_id, "user", "test")
    assistant_id = store.append_message(
        conv_id,
        "assistant",
        "SPI relies on compressive sensing [[CITE:s1234abcd:1]].",
    )
    store.merge_message_meta(
        assistant_id,
        {
            "answer_quality": {
                "prompt_family": "citation_lookup",
                "output_mode": "citation_lookup",
            }
        },
    )
    base_pack = {
        "prompt_sig": "same-prompt",
        "used_query": "same-query",
        "used_translation": False,
        "hits": [
            {"text": "same evidence", "meta": {"source_path": r"db\doc\doc.en.md"}}
        ],
    }
    refs_v1 = {
        user_id: {
            **base_pack,
            "rendered_payload_sig": "render-v1",
            "render_evidence_sig": "evidence-v1",
        }
    }
    refs_v2 = {
        user_id: {
            **base_pack,
            "rendered_payload_sig": "render-v2",
            "render_evidence_sig": "evidence-v2",
        }
    }

    first = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_v1,
        conv_id=conv_id,
        chat_store=store,
    )
    second = enrich_messages_with_reference_render(
        store.get_messages(conv_id),
        refs_v2,
        conv_id=conv_id,
        chat_store=store,
    )

    assert calls["primary"] == 2
    assert first[-1]["rendered_content"] != second[-1]["rendered_content"]


def test_unlinked_reference_candidates_find_unique_venue_year(monkeypatch):
    from api import chat_render

    monkeypatch.setattr(
        chat_render,
        "_load_reference_index_cached",
        lambda: {
            "docs": {
                "demo": {
                    "path": "current-paper.md",
                    "name": "current-paper.en.md",
                    "refs": {
                        "7": {
                            "num": 7,
                            "raw": "Smith J. Fast rotation-shearing single-pixel imaging. Optica. 2024.",
                            "title": "Fast rotation-shearing single-pixel imaging",
                            "authors": "Smith J",
                            "venue": "Optica",
                            "year": "2024",
                            "doi": "10.1364/optica.demo",
                        }
                    },
                }
            }
        },
    )

    candidates = chat_render._build_unlinked_reference_candidates(
        answer_markdown="For real-time imaging, the Optica 2024 work is a better comparison point.",
        rendered_body="",
        copy_text="",
        cite_details=[],
        ref_pack={"hits": [{"meta": {"source_path": "current-paper.md"}}]},
        provenance_segments=[],
        render_locale="en",
        anchor_ns="test",
    )

    assert len(candidates) == 1
    assert candidates[0]["match_method"] == "unique_venue_year_mention"
    assert candidates[0]["ref_num"] == 7
    assert candidates[0]["title"] == "Fast rotation-shearing single-pixel imaging"
    assert candidates[0]["cite_detail"]["citation_route"] == "system_b"
    assert "answer_context_only" in candidates[0]["cite_detail"]["card_quality_flags"]
    assert candidates[0]["cite_detail"]["system_b_trace_complete"] is False
    assert "answer_context_only" in candidates[0]["cite_detail"]["system_b_trace_flags"]


def test_unlinked_reference_candidates_respect_zero_system_b_budget(monkeypatch):
    from api import chat_render

    monkeypatch.setattr(
        chat_render,
        "_load_reference_index_cached",
        lambda: {
            "docs": {
                "demo": {
                    "path": "current-paper.md",
                    "name": "current-paper.en.md",
                    "refs": {
                        "7": {
                            "num": 7,
                            "raw": "Smith J. Fast rotation-shearing single-pixel imaging. Optica. 2024.",
                            "title": "Fast rotation-shearing single-pixel imaging",
                            "authors": "Smith J",
                            "venue": "Optica",
                            "year": "2024",
                            "doi": "10.1364/optica.demo",
                        }
                    },
                }
            }
        },
    )

    candidates = chat_render._build_unlinked_reference_candidates(
        answer_markdown="For real-time imaging, the Optica 2024 work is a better comparison point.",
        rendered_body="",
        copy_text="",
        cite_details=[],
        ref_pack={"hits": [{"meta": {"source_path": "current-paper.md"}}]},
        provenance_segments=[],
        render_locale="en",
        anchor_ns="test",
        allow_system_b=False,
    )

    assert candidates == []


def test_unlinked_reference_candidate_promotes_retrieved_library_document(monkeypatch):
    from api import chat_render

    local_source = r"F:\kb\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    parent_source = r"F:\kb\NatCommun-2021-Imaging biological tissue.en.md"
    second_parent_source = r"F:\kb\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"
    monkeypatch.setattr(
        chat_render,
        "_load_reference_index_cached",
        lambda: {
            "docs": {
                "demo": {
                    "path": parent_source,
                    "name": "NatCommun-2021-Imaging biological tissue.en.md",
                    "refs": {
                        "12": {
                            "num": 12,
                            "raw": "Zhang Z et al. Hadamard single-pixel imaging versus Fourier single-pixel imaging. Opt. Express. 2017.",
                            "title": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                            "authors": "Zhang Z, Wang X, Zheng G, et al",
                            "venue": "Opt. Express",
                            "year": "2017",
                            "doi": "10.1364/oe.25.019619",
                        }
                    },
                },
                "demo-duplicate": {
                    "path": second_parent_source,
                    "name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
                    "refs": {
                        "41": {
                            "num": 41,
                            "raw": "Zhang Z et al. Hadamard single-pixel imaging versus Fourier single-pixel imaging. Opt. Express. 2017.",
                            "title": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                            "authors": "Zhang Z, Wang X, Zheng G, et al",
                            "venue": "Opt. Express",
                            "year": "2017",
                            "doi": "10.1364/oe.25.019619-duplicate-index-row",
                        }
                    },
                },
            }
        },
    )

    candidates = chat_render._build_unlinked_reference_candidates(
        answer_markdown="The best match is Hadamard single-pixel imaging versus Fourier single-pixel imaging.",
        rendered_body="",
        copy_text="",
        cite_details=[],
        ref_pack={
            "hits": [
                {"meta": {"source_path": parent_source}},
                {"meta": {"source_path": second_parent_source}},
                {"meta": {"source_path": local_source}},
            ]
        },
        provenance_segments=[],
        render_locale="en",
        anchor_ns="test",
        allow_system_b=False,
    )

    assert len(candidates) == 1
    detail = candidates[0]["cite_detail"]
    assert candidates[0]["source_path"] == local_source
    assert candidates[0]["ref_num"] == 0
    assert detail["source_path"] == local_source
    assert detail["is_inpaper"] is False
    assert detail["citation_route"] == "system_a"
    assert detail["library_match_status"] == "in_library"
    assert detail["reference_source_path"] == parent_source
    assert detail["reference_ref_num"] == 12


def test_reading_guide_repair_binds_benefit_and_risk_evidence_to_distinct_claims():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "dl-spi-review.en.md"
    risk_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "4. Strategy and Advantages",
        "evidence_quote": "Data-driven strategies have prolonged training and limited generalization across imaging scenes.",
    }
    benefit_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Abstract",
        "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
    }
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [risk_slot, benefit_slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
    )
    answer = (
        "深度学习给单像素成像带来了更高的重建质量和更快的重建速度。\n\n"
        "主要风险：\n"
        "- 数据驱动方法训练时间长，而且泛化能力有限，难以适应多样场景。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "重建速度 [3]。" in repaired
    assert "多样场景 [2]。" in repaired


def test_reading_guide_repair_combines_adjacent_risks_supported_by_one_evidence_sentence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "dl-spi-review.en.md"
    risk_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL-SPI review",
        "heading_path": "4. Strategy and Advantages",
        "evidence_quote": (
            "Data-driven strategies have prolonged training duration and limited generalization, "
            "which makes them hard to adapt to diverse imaging scenes."
        ),
    }
    benefit_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "DL-SPI review",
        "heading_path": "Abstract",
        "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
    }
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [risk_slot, benefit_slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
        reserved_count=6,
    )
    answer = (
        "深度学习能够提高重建质量和重建速度。\n\n"
        "主要风险包括：\n"
        "- 训练时间长：数据驱动策略的训练周期较长。\n"
        "- 泛化能力有限：难以有效适应多样化的成像场景。\n"
        "- 依赖大量数据集：需要大量训练数据。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path] * 6,
    )
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[source_path] * 6,
        citation_plan=plan,
    )

    assert "数据驱动策略的直接局限是训练时间较长、泛化能力有限" in repaired
    assert "- 训练时间长" not in repaired
    assert "\n- 泛化能力有限" not in repaired
    risk_detail = next(item for item in details if int(item.get("num") or 0) == 7)
    assert "训练" in str(risk_detail.get("answer_claim") or "")
    assert "泛化" in str(risk_detail.get("answer_claim") or "")
    assert "prolonged training" in str(risk_detail.get("evidence_quote") or "")
    assert "limited generalization" in str(risk_detail.get("evidence_quote") or "")


def test_reading_guide_repair_combines_separated_data_training_and_generalization_claims():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "dl-spi-review.en.md"
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "4. Strategy and Advantages",
                "evidence_quote": (
                    "Data-driven strategies have prolonged training duration and limited generalization, "
                    "which makes them hard to adapt to diverse imaging scenes."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "Deep learning provides exceptional reconstruction quality and fast reconstruction speed."
                ),
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
        reserved_count=6,
    )
    answer = (
        "深度学习能够提高重建质量和重建速度。\n\n"
        "主要风险包括：\n"
        "- 依赖大规模数据集：训练需要大量标注数据。\n"
        "- 泛化能力有限：难以有效适应多样化的成像场景。\n\n"
        "- 可解释性差：模型的决策过程难以理解。\n"
        "- 容易过拟合：在未见过的数据上可能表现不佳。\n\n"
        "此外，数据驱动策略的训练时间较长，这也是实际应用中的挑战。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path] * 6,
    )
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[source_path] * 6,
        citation_plan=plan,
    )

    assert "数据驱动策略的直接局限是训练时间较长" in repaired
    assert "泛化能力有限" in repaired
    risk_detail = next(
        item
        for item in details
        if "limited generalization" in str(item.get("evidence_quote") or "")
    )
    assert "数据" in str(risk_detail.get("answer_claim") or "")
    assert "泛化" in str(risk_detail.get("answer_claim") or "")


def test_reading_guide_repair_combines_numbered_risks_supported_by_one_evidence_sentence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    source_path = "dl-spi-review.en.md"
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "4. Strategy and Advantages",
                "evidence_quote": (
                    "Data-driven strategies have prolonged training duration and limited generalization, "
                    "which makes them hard to adapt to diverse imaging scenes."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "DL-SPI review",
                "heading_path": "Abstract",
                "evidence_quote": (
                    "Deep learning provides exceptional reconstruction quality and fast reconstruction speed."
                ),
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
        reserved_count=6,
    )
    answer = (
        "Deep learning improves reconstruction quality and speed.\n\n"
        "Main risks:\n"
        "1. Prolonged training: data-driven strategies take a long time to train.\n"
        "2. Limited generalization: they struggle with diverse imaging scenes.\n"
        "3. Large datasets: they require substantial training data."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path] * 6,
    )
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[source_path] * 6,
        citation_plan=plan,
    )

    assert (
        "The directly supported limitation is that data-driven strategies have prolonged "
        "training and limited generalization across imaging scenes [7]."
    ) in repaired
    assert "\n2. Limited generalization" not in repaired
    risk_detail = next(item for item in details if int(item.get("num") or 0) == 7)
    assert "prolonged training" in str(risk_detail.get("answer_claim") or "").lower()
    assert "limited generalization" in str(risk_detail.get("answer_claim") or "").lower()
    assert "prolonged training" in str(risk_detail.get("evidence_quote") or "")
    assert "limited generalization" in str(risk_detail.get("evidence_quote") or "")


def test_reading_guide_rebinds_three_source_markers_to_dedicated_plan_hits():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["paper-a.en.md", "paper-b.en.md", "paper-c.en.md"]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_path,
            "source_name": f"Paper {idx}",
            "heading_path": "Abstract",
            "evidence_quote": f"Direct evidence for paper {idx}.",
        }
        for idx, source_path in enumerate(source_paths, start=1)
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": slots,
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Raw C", "meta": {"source_path": source_paths[2]}},
            {"text": "Raw B", "meta": {"source_path": source_paths[1]}},
            {"text": "Raw A", "meta": {"source_path": source_paths[0]}},
        ],
        plan,
        reserved_count=3,
    )
    answer = "1. Paper A overview [1].\n2. Paper B overview [2].\n3. Paper C overview [3]."

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert "Paper A overview [4]" in repaired
    assert "Paper B overview [5]" in repaired
    assert "Paper C overview [6]" in repaired


def test_reading_guide_adds_one_plan_citation_to_each_named_paper_heading():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    titles = [
        "Principles and prospects for single-pixel imaging",
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
        "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
    ]
    source_paths = [f"paper-{idx}.en.md" for idx in range(1, 4)]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_path,
            "source_name": title,
            "topic": f"{title} / Abstract",
            "heading_path": f"{title} / Abstract",
            "evidence_quote": f"Direct source evidence for {title}.",
        }
        for title, source_path in zip(titles, source_paths)
    ]
    plan = {"budget": {"system_a": 3, "system_b": 0}, "slots": slots}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": f"Raw evidence {idx}.", "meta": {"source_path": source_path}}
            for idx, source_path in enumerate(source_paths, start=1)
        ],
        plan,
        reserved_count=3,
    )
    answer = "\n\n".join(
        f"### {idx}. {title}\n\nMain point [{idx}]."
        for idx, title in enumerate(titles, start=1)
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    for idx, title in enumerate(titles, start=4):
        assert f"{title} [{idx}]" in repaired


def test_reading_guide_keeps_occurrence_markers_when_source_has_multiple_plan_slots():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["paper-a.en.md", "paper-b.en.md", "paper-c.en.md"]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_paths[0],
            "heading_path": "Method",
            "evidence_quote": "Paper A method evidence.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[0],
            "heading_path": "Limitations",
            "evidence_quote": "Paper A limitation evidence.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[1],
            "heading_path": "Abstract",
            "evidence_quote": "Paper B evidence.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[2],
            "heading_path": "Abstract",
            "evidence_quote": "Paper C evidence.",
        },
    ]
    plan = {"budget": {"system_a": 3, "system_b": 0}, "slots": slots}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": f"Raw {idx}.", "meta": {"source_path": source_path}}
            for idx, source_path in enumerate(source_paths, start=1)
        ],
        plan,
        reserved_count=3,
    )
    answer = "Paper A method [1]. Paper A limitation [1]. Paper B [2]. Paper C [3]."

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert repaired.count("[1]") == 2
    assert "Paper B [6]" in repaired
    assert "Paper C [7]" in repaired


def test_reading_guide_rebinds_only_the_locally_aligned_same_source_occurrence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["paper-a.en.md", "paper-b.en.md", "paper-c.en.md"]
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_path,
            "source_name": f"Paper {letter}",
            "heading_path": "Abstract",
            "evidence_quote": f"Paper {letter} directly supports its overview.",
        }
        for source_path, letter in zip(source_paths, ("A", "B", "C"))
    ]
    plan = {"budget": {"system_a": 3, "system_b": 0}, "slots": slots}
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": f"Raw {letter}.", "meta": {"source_path": source_path}}
            for source_path, letter in zip(source_paths, ("A", "B", "C"))
        ],
        plan,
        reserved_count=3,
    )
    answer = (
        "Paper A overview [1].\n"
        "A general deployment warning with no support in the selected passage [1].\n"
        "Paper B overview [2].\n"
        "Paper C overview [3]."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert "Paper A overview [4]" in repaired
    assert "deployment warning with no support in the selected passage [1]" in repaired
    assert "Paper B overview [5]" in repaired
    assert "Paper C overview [6]" in repaired


def test_reading_guide_cassi_lineage_keeps_three_system_a_cards_and_cleans_system_b_prose():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_paths = ["cassi.en.md", "scinerf.en.md", "scigs.en.md"]
    system_a_slots = [
        {
            "preferred_system": "system_a",
            "source_path": source_paths[0],
            "source_name": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "heading_path": "Abstract",
            "evidence_quote": "Two dispersive elements surround a binary aperture code.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[1],
            "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
            "heading_path": "Methods",
            "evidence_quote": "SCINeRF uses a neural radiance field and the SCI physical image formation process.",
        },
        {
            "preferred_system": "system_a",
            "source_path": source_paths[2],
            "source_name": "SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image",
            "heading_path": "Abstract",
            "evidence_quote": "SCIGS reconstructs a dynamic 3D scene from one compressed image.",
        },
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "source_path": source_paths[1],
                "topic": "snapshot compressive imaging",
            },
            *system_a_slots,
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Raw CASSI.", "meta": {"source_path": source_paths[0]}},
            {"text": "Raw SCINeRF.", "meta": {"source_path": source_paths[1]}},
            {"text": "Raw SCIGS.", "meta": {"source_path": source_paths[2]}},
        ],
        plan,
        reserved_count=3,
    )
    answer = (
        "CASSI starts with a dual-disperser architecture [1].\n"
        "Video SCI is an upstream step [ [[CITE:sid:50]] ].\n"
        "SCINeRF uses NeRF with the SCI physical image formation process.\n"
        "SCIGS reconstructs a dynamic 3D scene [3].\n"
        "如需细节，请查阅原始论文（如文献[[CITE:sid:50]]）。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=source_paths,
    )

    assert "CASSI starts with a dual-disperser architecture [1]" in repaired
    assert "SCINeRF uses NeRF with the SCI physical image formation process [2]" in repaired
    assert "SCIGS reconstructs a dynamic 3D scene [3]" in repaired
    assert all(f"[{num}]" not in repaired for num in (4, 5, 6))
    assert "[ [[CITE:sid:50]] ]" not in repaired
    assert "upstream step [[CITE:sid:50]]" in repaired
    assert "原始论文" not in repaired
    assert "上游文献或背景入口（如文献[[CITE:sid:50]]）" in repaired


def test_reading_guide_names_ilnet_and_binds_method_plus_strategy_evidence():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _backfill_system_a_cite_details_from_ref_pack,
        _reading_guide_repair_missing_system_a_citations,
    )
    from api.reference_rendering import _annotate_inpaper_citations_with_hover_meta

    method_path = "part-based-image-loop.en.md"
    review_path = "dl-spi-review.en.md"
    other_path = "unrelated.en.md"
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": method_path,
            "source_name": "Part-based image-loop network for single-pixel imaging",
            "heading_path": "Methods / ILNet architecture",
            "evidence_quote": (
                "We propose a self-supervised image-loop neural network (ILNet) with a "
                "part-based model; detector signals are labels for optimization."
            ),
        },
        {
            "preferred_system": "system_a",
            "source_path": review_path,
            "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
            "heading_path": "4.1.2 Model-Driven Strategy",
            "evidence_quote": (
                "Model-driven strategy is an unsupervised learning mode that integrates the "
                "physical process of SPI with neural networks."
            ),
        },
        {
            "preferred_system": "system_a",
            "source_path": other_path,
            "source_name": "Other SPI paper",
            "heading_path": "Methods",
            "evidence_quote": "An unrelated detector model.",
        },
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": slots,
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "Raw ILNet.", "meta": {"source_path": method_path}},
            {"text": "Raw review.", "meta": {"source_path": review_path}},
            {"text": "Raw other.", "meta": {"source_path": other_path}},
        ],
        plan,
        reserved_count=3,
    )
    answer = (
        "## PILN 与主线的关系\n\n"
        "PILN（Part-based Image-Loop Network）属于模型驱动策略，这是两条主线之一 [2]。\n\n"
        "### 深度学习单像素成像的两条主线\n\n"
        "### 不适合解决的问题\n\n"
        "| 实时成像任务 | 迭代需要大量计算时间 |\n"
        "| 高帧率视频成像 | 难以恢复高帧率图像 [5] |\n\n"
        "### 关键权衡\n\n"
        "代价是 **计算时间**，这限制了它在实时应用中的部署。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[method_path, review_path, other_path],
    )

    assert "论文原文将该方法称为 **ILNet**" in repaired
    assert "part-based model" in repaired
    assert "PILN/ILNet" in repaired
    assert "[1]" in repaired
    assert "[2]" in repaired
    assert "model-driven strategy" in repaired
    assert "两条主线之一" not in repaired
    assert "实时成像" not in repaired
    assert "高帧率" not in repaired
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[method_path, review_path, other_path],
        citation_plan=plan,
    )
    review_detail = next(item for item in details if int(item.get("num") or 0) == 2)
    assert review_detail.get("citation_plan_slot") is True
    generic_abstract = (
        "Single-pixel imaging technology can capture images at wavelengths outside conventional "
        "detectors, while deep learning improves reconstruction quality and speed."
    )
    backfilled = _backfill_system_a_cite_details_from_ref_pack(
        [review_detail],
        {
            "primary_evidence": {
                "source_path": review_path,
                "source_name": "DL-SPI review",
                "heading_path": "Abstract",
                "snippet": generic_abstract,
                "selection_reason": "pending_section_seed",
            }
        },
    )
    assert "model-driven strategy" in backfilled[0]["evidence_quote"].lower()
    assert "physical process of SPI" in backfilled[0]["evidence_quote"]
    assert backfilled[0]["heading_path"] == "4.1.2. Model-Driven Strategy"


def test_s2ism_tradeoff_whole_paragraph_rewrite_requires_focused_comparison_plan():
    from api.chat_render import _reading_guide_repair_s2ism_tradeoff_answer

    answer = (
        "This method map mentions the s2ISM trade-off in thick samples, then compares "
        "it with two unrelated microscopy methods."
    )
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "s2ism.en.md",
                "source_name": "Structured detection for s2ISM",
            }
        ],
    }

    repaired = _reading_guide_repair_s2ism_tradeoff_answer(answer, [], plan)

    assert repaired == answer


def test_s2ism_tradeoff_repair_accepts_direct_answer_grounding_plan(tmp_path: Path):
    from api.chat_render import _reading_guide_repair_s2ism_tradeoff_answer

    source_path = tmp_path / "s2ism.en.md"
    evidence = (
        "Fast detector arrays overcome the trade-off between spatial resolution and "
        "signal-to-noise ratio. However, current image scanning microscopy approaches "
        "do not provide optical sectioning and fail with thick samples unless the detector "
        "size is limited, introducing a trade-off between optical sectioning and "
        "signal-to-noise ratio."
    )
    source_path.write_text(
        f"<!-- kb_page: 1 -->\n## Abstract\n\n{evidence}\n",
        encoding="utf-8",
    )
    answer = (
        "s²ISM 打破了空间分辨率、信噪比和光学切片能力的三方权衡。\n\n"
        "普通 ISM 在厚样本里会失败。"
    )
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": "Structured detection for s2ISM",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ],
    }

    repaired = _reading_guide_repair_s2ism_tradeoff_answer(
        answer,
        [{"text": evidence, "meta": {"source_path": str(source_path)}}],
        plan,
        canonical_paths=[str(source_path)],
    )

    assert "空间分辨率与 SNR" in repaired
    assert "光学切片（optical sectioning）与 SNR" in repaired
    assert "厚样本" in repaired
    assert "[1]" in repaired


def test_s2ism_repair_continues_binding_other_planned_sources():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    s2ism_evidence = (
        "Spatial resolution and signal-to-noise trade-off. Current image scanning "
        "microscopy approaches do not provide optical sectioning in thick samples "
        "unless detector size is limited, sacrificing signal-to-noise."
    )
    method_x_evidence = "Method X improves axial resolution by phase diversity."
    hits = [
        {
            "text": s2ism_evidence,
            "meta": {"source_path": "s2ism.en.md", "heading_path": "Abstract"},
        },
        {
            "text": method_x_evidence,
            "meta": {"source_path": "method-x.en.md", "heading_path": "Results"},
        },
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "s2ism.en.md",
                "source_name": "Structured detection s2ISM",
                "heading_path": "Abstract",
                "evidence_quote": s2ism_evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "method-x.en.md",
                "source_name": "Method X",
                "heading_path": "Results",
                "evidence_quote": method_x_evidence,
                "candidate_hits": [2],
            },
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        "s2ISM trade-off in thick samples.\n\n"
        "Method X improves axial resolution by phase diversity.",
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=["s2ism.en.md", "method-x.en.md"],
    )

    assert "Method X improves axial resolution" in repaired
    assert "[1]" in repaired
    assert "[2]" in repaired


def test_normal_answer_binds_three_planned_sources_without_inserting_topic_specific_prose():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    answer = (
        "Paper Alpha establishes the measurement model.\n\n"
        "Paper Beta parallelizes the hardware acquisition.\n\n"
        "Paper Gamma adds learned reconstruction."
    )
    hits = [
        {"text": "Alpha evidence.", "meta": {"source_path": "alpha.en.md"}},
        {"text": "Beta evidence.", "meta": {"source_path": "beta.en.md"}},
        {"text": "Gamma evidence.", "meta": {"source_path": "gamma.en.md"}},
    ]
    plan = {
        "intent": "multi_source_synthesis",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "alpha.en.md",
                "source_name": "Paper Alpha",
                "evidence_quote": "Alpha evidence.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "beta.en.md",
                "source_name": "Paper Beta",
                "evidence_quote": "Beta evidence.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": "gamma.en.md",
                "source_name": "Paper Gamma",
                "evidence_quote": "Gamma evidence.",
                "candidate_hits": [3],
            },
        ],
    }

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="answer",
        canonical_paths=["alpha.en.md", "beta.en.md", "gamma.en.md"],
    )

    assert "Paper Alpha establishes the measurement model [1]." in repaired
    assert "Paper Beta parallelizes the hardware acquisition [2]." in repaired
    assert "Paper Gamma adds learned reconstruction [3]." in repaired
    assert "single-pixel camera" not in repaired


def test_comparison_binds_missing_planned_source_inside_matching_table_cell(monkeypatch):
    from api.chat_render import (
        _annotate_inpaper_citations_with_hover_meta,
        _reading_guide_repair_missing_system_a_citations,
    )

    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    monkeypatch.setattr("ui.refs_renderer._load_reference_index_cached", lambda: {})

    scigs_path = "db/SCIGS/SCIGS.en.md"
    scinerf_path = "db/SCINeRF/SCINeRF.en.md"
    scigs_evidence = (
        "SCIGS reconstructs an explicit 3D scene from a single compressed image "
        "and extends the method to dynamic 3D scenes."
    )
    scinerf_evidence = (
        "We formulate the physical imaging process of SCI as part of the training "
        "of NeRF."
    )
    hits = [
        {
            "text": "SCIGS title.",
            "meta": {"source_path": scigs_path, "ref_answer_citation_num": 1},
        },
        {
            "text": scinerf_evidence,
            "meta": {
                "source_path": scinerf_path,
                "heading_path": "Abstract",
                "ref_answer_citation_num": 2,
            },
        },
        {
            "text": "SCINeRF mask-overlap experiment.",
            "meta": {"source_path": scinerf_path, "ref_answer_citation_num": 3},
        },
        {
            "text": scigs_evidence,
            "meta": {
                "source_path": scigs_path,
                "heading_path": "Abstract",
                "ref_answer_citation_num": 4,
            },
        },
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": scinerf_path,
                "source_name": "SCINeRF",
                "heading_path": "Abstract",
                "evidence_quote": scinerf_evidence,
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": scigs_path,
                "source_name": "SCIGS",
                "heading_path": "Abstract",
                "evidence_quote": scigs_evidence,
                "candidate_hits": [4],
            },
        ],
    }
    answer = (
        "SCIGS reconstructs explicit dynamic 3D scenes [4].\n\n"
        "| \u5bf9\u6bd4\u7ef4\u5ea6 | SCIGS | SCINeRF |\n"
        "| --- | --- | --- |\n"
        "| \u6838\u5fc3\u65b9\u6cd5 | primitive transformation + 3DGS [4] | "
        "\u5c06 SCI \u7269\u7406\u6210\u50cf\u8fc7\u7a0b\u878d\u5165 NeRF \u8bad\u7ec3 |\n"
        "| SSIM | 0.9137 [4] | 0.7974 [3] |"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[scigs_path, scinerf_path, scinerf_path, scigs_path],
    )

    assert (
        "\u5c06 SCI \u7269\u7406\u6210\u50cf\u8fc7\u7a0b\u878d\u5165 NeRF \u8bad\u7ec3 [2]|"
        in repaired
    )
    assert repaired.count("[2]") == 1
    assert "0.7974 [3]" in repaired
    assert "0.7974 [2]" not in repaired

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[scigs_path, scinerf_path, scinerf_path, scigs_path],
        citation_plan=plan,
        render_locale="zh",
    )
    scinerf_card = next(detail for detail in details if detail.get("num") == 2)
    assert "[2](#" in rendered
    assert scinerf_card["citation_route"] == "system_a"
    assert scinerf_card["source_path"] == scinerf_path
    assert "physical imaging process of SCI" in scinerf_card["card_evidence"]


def test_authoritative_comparison_path_binds_scinerf_physics_claim_in_table() -> None:
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    scigs_path = "db/SCIGS/SCIGS.en.md"
    scinerf_path = "db/SCINeRF/SCINeRF.en.md"
    scigs_evidence = (
        "SCIGS is a variant of 3DGS that reconstructs an explicit 3D scene from a "
        "single compressed image and extends to dynamic 3D scenes."
    )
    scinerf_evidence = (
        "We formulate the physical imaging process of SCI as part of the training of NeRF."
    )
    hits = [
        {
            "text": scigs_evidence,
            "meta": {
                "source_path": scigs_path,
                "ref_answer_citation_num": 1,
                "canonical_answer_citation_num": 1,
            },
        },
        {
            "text": scinerf_evidence,
            "meta": {
                "source_path": scinerf_path,
                "ref_answer_citation_num": 2,
                "canonical_answer_citation_num": 2,
            },
        },
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": scigs_path,
                "source_name": "SCIGS",
                "heading_path": "Abstract",
                "evidence_quote": scigs_evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": scinerf_path,
                "source_name": "SCINeRF",
                "heading_path": "Abstract",
                "evidence_quote": scinerf_evidence,
                "candidate_hits": [2],
            },
        ],
    }
    answer = (
        "SCIGS reconstructs explicit dynamic 3D scenes [1].\n\n"
        "SCINeRF uses an implicit neural field [2].\n\n"
        "| 对比维度 | SCIGS | SCINeRF |\n"
        "| --- | --- | --- |\n"
        "| 核心训练 | 3DGS primitive transform [1] | "
        "将 SCI 物理成像过程公式化为 NeRF 训练的一部分 |"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[scigs_path, scinerf_path],
    )

    assert "将 SCI 物理成像过程公式化为 NeRF 训练的一部分 [2]|" in repaired


def test_origin_answer_binds_current_paper_evidence_beside_verified_upstream_marker():
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = "F:/db/SCINeRF/SCINeRF.en.md"
    evidence = (
        "most of the existing methods employ alternating direction method of "
        "multipliers (ADMM) [4],"
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_path": source_path,
                "heading_path": "SCINeRF / 2. Related Work",
                "ref_answer_citation_num": 1,
            },
        }
    ]
    plan = {
        "intent": "origin_lookup",
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "SCINeRF",
                "heading_path": "SCINeRF / 2. Related Work",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_b",
                "source_path": source_path,
                "source_name": "SCINeRF",
                "heading_path": "SCINeRF / 2. Related Work",
                "evidence_quote": evidence,
                "candidate_refs": [4],
                "candidate_cite_examples": ["[[CITE:s7f6b9404:4]]"],
            },
        ],
    }
    answer = (
        "不是。ADMM 在这里是当前论文引用的已有方法背景 "
        "[[CITE:s7f6b9404:4]]，不是本文原创。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path],
    )

    assert "[[CITE:s7f6b9404:4]]" in repaired
    assert repaired.count("[1]") == 1
    assert "不是本文原创 [1]。" in repaired


def test_origin_answer_does_not_share_system_a_when_upstream_context_differs():
    from api.chat_render import _reading_guide_attach_claim_level_system_a_citations

    source_path = "F:/db/SCINeRF/SCINeRF.en.md"
    system_a_evidence = "Most existing methods employ ADMM [4]."
    plan = {
        "intent": "origin_lookup",
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "evidence_quote": system_a_evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_b",
                "source_path": source_path,
                "evidence_quote": "Boyd et al. provide an ADMM tutorial.",
                "candidate_refs": [4],
                "candidate_cite_examples": ["[[CITE:s7f6b9404:4]]"],
            },
        ],
    }
    answer = (
        "ADMM is existing-method background [[CITE:s7f6b9404:4]], "
        "not original to this paper."
    )

    repaired = _reading_guide_attach_claim_level_system_a_citations(
        answer,
        [
            {
                "text": system_a_evidence,
                "meta": {
                    "source_path": source_path,
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        plan,
        canonical_paths=[source_path],
    )

    assert repaired == answer


def test_multi_source_plan_normalizes_duplicate_canonical_markers_before_budgeting():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    canonical_paths = [
        "distractor-a.en.md",
        "distractor-b.en.md",
        "review.en.md",
        "distractor-c.en.md",
        "deep-review.en.md",
        "hardware.en.md",
    ]
    hits = [
        {"text": "Hardware evidence.", "meta": {"source_path": "hardware.en.md"}},
        {"text": "Deep review evidence.", "meta": {"source_path": "deep-review.en.md"}},
        {"text": "Foundation review evidence.", "meta": {"source_path": "review.en.md"}},
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "deep-review.en.md",
                "source_name": "Deep Review",
                "evidence_quote": "Deep review evidence.",
                "candidate_hits": [5],
            },
            {
                "preferred_system": "system_a",
                "source_path": "hardware.en.md",
                "source_name": "Hardware Paper",
                "evidence_quote": "Hardware evidence.",
                "candidate_hits": [6],
            },
            {
                "preferred_system": "system_a",
                "source_path": "review.en.md",
                "source_name": "Foundation Review",
                "evidence_quote": "Foundation review evidence.",
                "candidate_hits": [3],
            },
        ],
    }
    augmented = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=len(canonical_paths),
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        "Foundation Review establishes the model.\n\n"
        "Hardware Paper accelerates acquisition [6] and improves throughput [6].\n\n"
        "Deep Review covers learned reconstruction [5] and deployment [5].",
        augmented,
        plan,
        output_mode="answer",
        canonical_paths=canonical_paths,
    )

    assert repaired.count("[5]") == 2
    assert repaired.count("[6]") == 2
    # The third canonical source is already reserved for the foundation
    # review, so the repair should use that stable answer number instead of
    # inventing an out-of-range synthetic marker.
    assert repaired.count("[3]") == 1
    assert "[7]" not in repaired
    assert "[8]" not in repaired
    assert "[9]" not in repaired


def test_microscopy_method_map_repair_preserves_unrelated_numeric_citations(
    tmp_path: Path,
    monkeypatch,
):
    from api import chat_render

    sources = [
        (
            "s2ism.en.md",
            "Structured detection for s2ISM",
            "Existing image-scanning methods retain a resolution-versus-sectioning trade-off. "
            "Structured detection provides simultaneous super-resolution, high signal-to-noise "
            "ratio, and optical sectioning.",
        ),
        (
            "iism.en.md",
            "Interferometric image scanning microscopy",
            "Interferometric detection enables live-cell imaging at 120 nm lateral resolution.",
        ),
        (
            "light-field.en.md",
            "Light-field microscopy",
            "Light-field microscopy records position and angular information for volumetric reconstruction.",
        ),
    ]
    hits: list[dict] = []
    slots: list[dict] = []
    for index, (filename, source_name, evidence) in enumerate(sources, start=1):
        source_path = tmp_path / filename
        source_path.write_text(f"# {source_name}\n\n## Abstract\n\n{evidence}\n", encoding="utf-8")
        hits.append(
            {
                "text": evidence,
                "meta": {"source_path": str(source_path), "source_name": source_name},
            }
        )
        slots.append(
            {
                "preferred_system": "system_a",
                "source_path": str(source_path),
                "source_name": source_name,
                "candidate_hits": [index],
            }
        )
    evidence_by_source = {source_name: evidence for _, source_name, evidence in sources}
    monkeypatch.setattr(
        chat_render,
        "_claim_aligned_abstract_primary_evidence",
        lambda _pack, item: {
            "snippet": evidence_by_source[str(item.get("source_name") or "")],
            "heading_path": "Abstract",
        },
    )
    hits.append(
        {
            "text": "An unrelated paper supports the acquisition-system claim.",
            "meta": {"source_path": str(tmp_path / "unrelated.en.md")},
        }
    )
    answer = (
        "s2ISM uses structured detection [1].\n\n"
        "iISM uses interferometric detection [2].\n\n"
        "Light-field microscopy records angular information [3].\n\n"
        "The acquisition system has an independently supported property [4]."
    )

    repaired = chat_render._reading_guide_repair_microscopy_method_map_evidence(
        answer,
        hits,
        {"slots": slots},
    )

    assert "s2ISM uses structured detection [1]." not in repaired
    assert "iISM uses interferometric detection [2]." not in repaired
    assert "Light-field microscopy records angular information [3]." not in repaired
    assert (
        "s2ISM addresses the difficulty of obtaining super-resolution and optical "
        "sectioning together" in repaired
    )
    assert "achieves both simultaneously while maintaining high SNR [1]" in repaired
    assert "about 120 nm lateral resolution [2]" in repaired
    assert "captures both position and angular information for volumetric reconstruction" in repaired
    assert "extreme depth of field [3]" in repaired
    assert "independently supported property [4]" in repaired
    assert all(f"[{num}]" in repaired for num in (1, 2, 3))
    assert len(hits) == 4
    assert str(hits[0]["meta"]["evidence_quote"]).startswith("Structured detection provides")
    assert str(slots[0]["evidence_quote"]).startswith("Structured detection provides")
    assert [hit["meta"]["citation_plan_microscopy_direct"] for hit in hits[:3]] == [
        "s2ism",
        "iism",
        "light_field",
    ]
    repaired_again = chat_render._reading_guide_repair_claim_aligned_abstract_citations(
        repaired,
        hits,
        {"slots": slots},
        canonical_paths=[str(tmp_path / filename) for filename, _, _ in sources],
    )
    assert repaired_again == repaired
    assert len(hits) == 4


def test_single_photon_reading_pair_keeps_facts_on_two_sources_and_uncites_reading_steps():
    from api import chat_render

    detector_evidence = (
        "Performance information of different single-photon detectors. Detector type: Si-SPAD; "
        "spectral range 400–1000 nm; 50%–92% QE; operating temperature 200–300 K."
    )
    model_evidence = (
        "Physics-informed deep learning uses a real-world physical noise model of SPAD for low bit "
        "depth. It includes dark count rate and is calibrated with 2790 images."
    )
    hits = [
        {
            "text": detector_evidence,
            "meta": {"source_path": "detector-review.en.md"},
        },
        {
            "text": model_evidence,
            "meta": {"source_path": "spad-pidl.en.md"},
        },
        {
            "text": "A single-pixel imaging review, not a SPAD-array source.",
            "meta": {"source_path": "spi-review.en.md"},
        },
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "detector-review.en.md",
                "source_name": "Emerging single-photon detector review",
                "evidence_quote": detector_evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "spad-pidl.en.md",
                "source_name": "Physics-informed deep learning for SPAD imaging",
                "evidence_quote": model_evidence,
                "candidate_hits": [2],
            },
        ]
    }

    repaired = chat_render._reading_guide_repair_single_photon_reading_pair(
        "请结合 single-photon detector 综述和 physics-informed 物理噪声模型给出阅读路线。",
        hits,
        plan,
    )

    assert repaired.count("[1]") == 1
    assert repaired.count("[2]") == 3
    assert "[3]" not in repaired
    assert "这些参数决定后续算法面对的观测质量" not in repaired
    assert "### 1. 先读探测器综述" in repaired
    assert "### 2. 再读 physics-informed deep learning" in repaired
    assert "### 3." not in repaired


def test_single_photon_pair_restores_2790_plan_evidence_to_canonical_model_hit() -> None:
    from api import chat_render

    detector_evidence = (
        "This review summarizes principles and technical challenges of mainstream "
        "single-photon detectors."
    )
    model_evidence = (
        "We studied the photon flow model of SPAD electronics and collected a real "
        "SPAD image dataset (64 x 32 pixels, 90 scenes, 10 bit depths, 3 illumination "
        "flux, 2790 images in total) to calibrate the noise model. To tackle low bit "
        "depth, low resolution, and heavy noise, we built a deep transformer network."
    )
    hits = [
        {
            "text": detector_evidence,
            "meta": {
                "source_path": "detector-review.en.md",
                "ref_answer_citation_num": 1,
            },
        },
        {
            "text": "Deep learning improves SPAD reconstruction quality.",
            "meta": {
                "source_path": "spad-pidl.en.md",
                "ref_answer_citation_num": 2,
            },
        },
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "spad-pidl.en.md",
                "source_name": "High-resolution single-photon imaging with physics-informed deep learning",
                "heading_path": "Abstract",
                "evidence_quote": model_evidence,
                "candidate_hits": [2],
                "page_start": 1,
                "page_end": 1,
            },
            {
                "preferred_system": "system_a",
                "source_path": "detector-review.en.md",
                "source_name": "Emerging single-photon detection technique review",
                "evidence_quote": detector_evidence,
                "candidate_hits": [1],
            },
        ]
    }
    answer = (
        "The physics-informed model is calibrated with 2790 real SPAD images [2]; "
        "the single-photon detector review provides the hardware background [1]."
    )

    repaired = chat_render._reading_guide_repair_single_photon_reading_pair(
        answer,
        hits,
        plan,
        canonical_paths=["detector-review.en.md", "spad-pidl.en.md"],
    )

    assert repaired == answer
    assert "2790 images" in hits[1]["text"]
    assert hits[1]["meta"]["citation_plan_evidence_authoritative"] is True
    assert (
        hits[1]["meta"]["citation_plan_evidence_selection_reason"]
        == "spad_noise_model_exact_source"
    )
    assert "2790 images" in hits[1]["ui_meta"]["primary_evidence"]["snippet"]


def test_single_photon_pair_rebinds_reranked_model_hit_by_source_identity() -> None:
    from api import chat_render

    detector_path = "detector-review.en.md"
    model_path = "spad-pidl.en.md"
    detector_evidence = (
        "Performance information of different single-photon detectors. "
        "Detector type: Si-SPAD."
    )
    model_evidence = (
        "Physics-informed deep learning uses a real-world physical noise model of "
        "SPAD arrays and collected 2790 images to calibrate it."
    )
    hits = [
        {
            "text": "Deep learning improves SPAD reconstruction quality.",
            "meta": {
                "source_path": model_path,
                "ref_answer_citation_num": 2,
            },
        },
        {
            "text": detector_evidence,
            "meta": {
                "source_path": detector_path,
                "ref_answer_citation_num": 1,
            },
        },
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": detector_path,
                "source_name": "Emerging single-photon detector review",
                "evidence_quote": detector_evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": model_path,
                "source_name": "Physics-informed deep learning for SPAD imaging",
                "evidence_quote": model_evidence,
                "candidate_hits": [2],
            },
        ]
    }

    repaired = chat_render._reading_guide_repair_single_photon_reading_pair(
        "single-photon SPAD physics-informed reading guide [1] [2]",
        hits,
        plan,
        canonical_paths=[detector_path, model_path],
    )

    assert "[1]" in repaired and "[2]" in repaired
    assert [hit["meta"]["source_path"] for hit in hits] == [model_path, detector_path]
    assert [hit["meta"]["ref_answer_citation_num"] for hit in hits] == [2, 1]
    assert "2790 images" in hits[0]["text"]
    assert hits[0]["meta"]["citation_plan_full_evidence_quote"] == model_evidence
    assert "citation_plan_full_evidence_quote" not in hits[1]["meta"]


def test_full_enrich_keeps_independent_pidl_training_chain_citation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api import chat_render

    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    monkeypatch.setattr("ui.refs_renderer._load_reference_index_cached", lambda: {})
    model_path = tmp_path / "High-resolution single-photon imaging with physics-informed deep learning.en.md"
    detector_path = tmp_path / "Emerging single-photon detection technique review.en.md"
    model_evidence = (
        "with low bit depth, low resolution and heavy noise in photon-limited scenarios, "
        "we first established a real-world physical noise model of SPAD arrays. The real "
        "physical noise sources consist of shot noise, fixed-pattern noise, dark count rate, "
        "afterpulsing, crosstalk noise, and deadtime noise. To calibrate the parameters, we "
        "collected a real-shot SPAD image dataset containing 2790 images in total, each with "
        "64 x 32 pixels. With the calibrated physical noise model under different illumination "
        "and acquisition settings, we further employed off-the-shelf public high-resolution "
        "images (collected from the PASCAL VOC2007 [31] and VOC2012 [32] datasets) to digitally "
        "synthesize a large-scale realistic single-photon image dataset containing 2.6 million "
        "image pairs. The gated fusion transformer network was trained as the deep learning "
        "reconstruction network using the above large-scale single-photon image dataset."
    )
    detector_evidence = (
        "This review summarizes mainstream single-photon detectors including PMTs, SAPDs, "
        "SNSPDs, and TES devices, together with manufacturing and low-temperature challenges."
    )
    model_path.write_text(
        "<!-- kb_page: 3 -->\n## Introduction\n" + model_evidence,
        encoding="utf-8",
    )
    detector_path.write_text(
        "<!-- kb_page: 1 -->\n## Abstract\n" + detector_evidence,
        encoding="utf-8",
    )
    answer = (
        "该单光子探测器综述给出硬件背景 [2]。\n\n"
        "physics-informed 方法先建立 SPAD 物理噪声模型 [1]。\n\n"
        "* **校准与数据合成**：通过采集 2790 张真实 SPAD 图像校准模型 [1]。"
        "然后，利用校准后的模型和公开的高分辨率图像（如 PASCAL VOC2007）"
        "合成配对数据，用于训练深度学习网络 [1]。"
    )
    plan = {
        "source": "citation_plan_builder",
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "per_paragraph_budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": str(detector_path),
                "source_name": "Emerging single-photon detection technique review",
                "heading_path": "Abstract",
                "page_start": 1,
                "page_end": 1,
                "evidence_quote": detector_evidence,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": str(model_path),
                "source_name": (
                    "High-resolution single-photon imaging with physics-informed deep learning"
                ),
                "heading_path": "Introduction",
                "page_start": 3,
                "page_end": 3,
                "evidence_quote": model_evidence,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            },
        ],
    }
    messages = [
        {"id": 1, "role": "user", "content": "单光子探测器综述和 physics-informed 论文怎么搭配读？"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer,
            "meta": {
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "citation_plan": plan,
                },
                "canonical_hit_paths": [str(model_path), str(detector_path)],
            },
        },
    ]
    refs_by_user = {
        1: {
            "prompt": messages[0]["content"],
            "hits": [
                {
                    "text": (
                        "To calibrate the noise model, we collected 2790 SPAD images. "
                        "The physical noise model includes dark count and crosstalk noise."
                    ),
                    "meta": {
                        "source_path": str(model_path),
                        "source_name": plan["slots"][1]["source_name"],
                        "heading_path": "Introduction",
                        "page_start": 3,
                        "ref_answer_citation_num": 1,
                    },
                },
                {
                    "text": detector_evidence,
                    "meta": {
                        "source_path": str(detector_path),
                        "source_name": plan["slots"][0]["source_name"],
                        "heading_path": "Abstract",
                        "page_start": 1,
                        "ref_answer_citation_num": 2,
                    },
                },
            ],
        }
    }

    rendered = chat_render.enrich_messages_with_reference_render(
        messages,
        refs_by_user,
        conv_id="full-enrich-pidl-training",
        chat_store=None,
    )[-1]

    target_line = next(
        line
        for line in str(rendered.get("rendered_content") or "").splitlines()
        if "PASCAL VOC2007" in line
    )
    matching_details = [
        detail
        for detail in list(rendered.get("cite_details") or [])
        if "PASCAL VOC2007" in str(detail.get("answer_claim") or "")
    ]
    assert "](#kb-cite-" in target_line.split("PASCAL VOC2007", 1)[1]
    assert len(matching_details) == 1
    assert "2.6 million image pairs" in matching_details[0]["evidence_quote"]
    assert "network was trained" in matching_details[0]["evidence_quote"]


def test_claim_level_citation_reuse_binds_supported_body_and_skips_unsupported_details():
    from api import chat_render

    sources = [
        (
            "s2ism.en.md",
            "Structured detection for high-SNR s2ISM in thick samples",
            "s2ISM structured detection simultaneously provides super-resolution and optical "
            "sectioning while improving signal-to-noise ratio in thick samples.",
        ),
        (
            "iism.en.md",
            "Interferometric image scanning microscopy",
            "iISM combines interferometric detection with image scanning microscopy and achieves "
            "120 nm lateral resolution in live cells.",
        ),
        (
            "light-field.en.md",
            "Light-field microscopy",
            "Light-field microscopy records position and angular information for volumetric "
            "three-dimensional reconstruction.",
        ),
    ]
    hits = [
        {
            "text": evidence,
            "meta": {"source_path": source_path, "source_name": source_name},
        }
        for source_path, source_name, evidence in sources
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": source_name,
                "evidence_quote": evidence,
                "candidate_hits": [index],
            }
            for index, (source_path, source_name, evidence) in enumerate(sources, start=1)
        ]
    }
    answer = (
        "**三条原文直接依据：**\n"
        "- s2ISM 同时实现 super-resolution 与 optical sectioning [1]。\n"
        "- iISM 结合 interferometric detection 并达到 120 nm lateral resolution [2]。\n"
        "- Light-field 同时记录 position 与 angular information [3]。\n\n"
        "s2ISM 的 structured detection 同时改善超分辨率和光学切片能力。\n"
        "s2ISM 的 structured detection 还显著提高机械稳定性。\n"
        "iISM 的 interferometric detection 达到 120 nm lateral resolution。\n"
        "iISM 的 interferometric detection 将 lateral resolution 提高两倍。\n"
        "Light-field microscopy 记录 position 和 angular information 以支持 3D reconstruction。\n"
        "光场方法结合位置信息与角度信息支持三维重建。"
    )

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        hits,
        plan,
    )

    assert "s2ISM 的 structured detection 同时改善超分辨率和光学切片能力 [1]。" in repaired
    assert "iISM 的 interferometric detection 达到 120 nm lateral resolution [2]。" in repaired
    assert "Light-field microscopy 记录 position 和 angular information 以支持 3D reconstruction [3]。" in repaired
    assert "光场方法结合位置信息与角度信息支持三维重建 [3]。" in repaired
    assert "显著提高机械稳定性 [1]" not in repaired
    assert "lateral resolution 提高两倍 [2]" not in repaired


def test_claim_level_citation_reuse_skips_a_different_named_paper():
    from api import chat_render

    source_path = "natphoton-2019-principles-and-prospects.en.md"
    source_name = "Principles and prospects for single-pixel imaging"
    evidence = (
        "Single-pixel imaging uses deep learning to improve reconstruction quality "
        "and reconstruction speed."
    )
    answer = (
        "1. **《LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning》**（LPR, 2025）\n"
        "- 这篇综述讨论深度学习单像素成像。\n"
        "- 基础综述说明单像素成像的重建质量与重建速度。"
    )

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        [{"text": evidence, "meta": {"source_path": source_path, "source_name": source_name}}],
        {
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "source_name": source_name,
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                }
            ]
        },
    )

    assert "Deep Learning》**（LPR, 2025） [1]" not in repaired
    assert repaired.splitlines()[0] == answer.splitlines()[0]
    assert "基础综述说明单像素成像的重建质量与重建速度 [1]。" in repaired


def test_claim_level_citation_reuse_skips_markdown_wrapped_different_paper_title():
    from api import chat_render

    source_path = "natphoton-2019-principles-and-prospects.en.md"
    source_name = "Principles and prospects for single-pixel imaging"
    evidence = (
        "Single-pixel imaging uses deep learning to improve reconstruction quality "
        "and reconstruction speed."
    )
    answer = (
        "- **LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep "
        "Learning** reports that deep learning improves reconstruction quality and speed."
    )

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        [{"text": evidence, "meta": {"source_path": source_path, "source_name": source_name}}],
        {
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "source_name": source_name,
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                }
            ]
        },
    )

    assert "[1]" not in repaired


def test_claim_level_citation_reuse_keeps_bold_conclusion_claims_eligible():
    from api import chat_render

    source_path = "dl-spi-review.en.md"
    source_name = "Principles and prospects for single-pixel imaging.pdf"
    evidence = (
        "Deep learning reconstruction improves reconstruction quality and reconstruction "
        "speed for single-pixel imaging."
    )
    answer = (
        "1. **Deep learning reconstruction improves reconstruction quality and "
        "reconstruction speed for single-pixel imaging.**"
    )

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        [{"text": evidence, "meta": {"source_path": source_path, "source_name": source_name}}],
        {
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "source_name": source_name,
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                }
            ]
        },
    )

    assert repaired.count("[1]") == 1
    assert "single-pixel imaging" in repaired


def test_paper_identity_line_recognizes_bold_title_with_year():
    from api import chat_render

    claim = (
        "1. **LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on "
        "Deep Learning** (LPR, 2025)"
    )

    assert chat_render._reading_claim_is_paper_identity_line(
        claim,
        "Principles and prospects for single-pixel imaging.pdf",
    ) is True


def test_claim_level_citation_reuse_keeps_bold_lead_in_claims_eligible():
    from api import chat_render

    source_path = "dl-spi-review.en.md"
    source_name = "Principles and prospects for single-pixel imaging.pdf"
    evidence = (
        "Deep learning reconstruction improves reconstruction quality and reconstruction "
        "speed for single-pixel imaging."
    )
    answer = (
        "**Main conclusion about deep learning reconstruction**: single-pixel imaging "
        "improves reconstruction quality and reconstruction speed."
    )

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        [{"text": evidence, "meta": {"source_path": source_path, "source_name": source_name}}],
        {
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "source_name": source_name,
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                }
            ]
        },
    )

    assert repaired.count("[1]") == 1


def test_claim_level_citation_reuse_does_not_cross_single_photon_pixel_modalities():
    from api import chat_render

    hits = [
        {
            "text": "A physical SPAD noise model supports single-photon image reconstruction.",
            "meta": {"source_path": "pidl.en.md"},
        },
        {
            "text": "Deep learning single-pixel imaging improves reconstruction quality and speed.",
            "meta": {"source_path": "dl-spi.en.md"},
        },
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "pidl.en.md",
                "evidence_quote": hits[0]["text"],
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "dl-spi.en.md",
                "evidence_quote": hits[1]["text"],
                "candidate_hits": [2],
            },
        ]
    }
    answer = "SPAD 单光子成像利用物理噪声模型改善重建质量 [1]。"

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        hits,
        plan,
    )

    assert repaired.count("[1]") == 1
    assert "[2]" not in repaired


def test_claim_level_citation_reuse_requires_physical_noise_model_evidence():
    from api import chat_render

    hits = [
        {
            "text": (
                "A detector review discusses SPAD hardware, dark count, photon detection, "
                "noise reduction, and imaging resolution."
            ),
            "meta": {"source_path": "detector-review.en.md"},
        },
        {
            "text": (
                "We established a real-world physical noise model of SPAD arrays with "
                "dark count rate, afterpulsing, and crosstalk noise."
            ),
            "meta": {"source_path": "pidl.en.md"},
        },
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": hit["meta"]["source_path"],
                "evidence_quote": hit["text"],
                "candidate_hits": [index],
            }
            for index, hit in enumerate(hits, start=1)
        ]
    }
    answer = "SPAD 阵列的多源物理噪声模型包含暗计数、后脉冲和串扰 [2]。"

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        hits,
        plan,
    )

    assert repaired.count("[2]") == 1
    assert "[1]" not in repaired


def test_dl_spi_benefit_repair_does_not_cross_single_photon_pixel_modalities():
    from api import chat_render

    answer = (
        "physics-informed deep learning 提高了 SPAD 单光子成像的重建质量和重建速度 [1]。"
    )
    hits = [
        {"text": "SPAD single-photon evidence.", "meta": {"source_path": "pidl.en.md"}},
        {
            "text": "Single-pixel imaging review evidence.",
            "meta": {"source_path": "dl-spi.en.md"},
        },
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "dl-spi.en.md",
                "candidate_hits": [2],
                "evidence_quote": (
                    "Deep learning single-pixel imaging has exceptional reconstruction "
                    "quality and fast reconstruction speed."
                ),
            }
        ]
    }

    repaired = chat_render._reading_guide_repair_dl_spi_benefit_marker(
        answer,
        hits,
        plan,
    )

    assert repaired == answer
    assert "[2]" not in repaired


def test_dl_spi_benefit_risk_repair_keeps_only_two_direct_markers():
    from api import chat_render

    answer = (
        "深度学习给 SPI 带来好处和挑战 [1]。\n\n"
        "- 卓越的重建质量与快速重建速度：深度学习提高重建质量和重建速度。\n"
        "- 数据驱动策略训练时间长、泛化能力有限，难以适应多样场景 [1]。\n"
        "- 深度学习的固有限制包括依赖大规模数据、容易过拟合和泛化有限 [1]。"
    )
    hits = [
        {
            "text": "Deep learning SPI evidence.",
            "meta": {"source_path": "dl-spi.en.md", "ref_answer_citation_num": 1},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "dl-spi.en.md",
                "candidate_hits": [1],
                "evidence_quote": (
                    "Deep learning offers exceptional reconstruction quality and fast reconstruction speed."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": "dl-spi.en.md",
                "candidate_hits": [1],
                "evidence_quote": (
                    "Data-driven strategies have prolonged training duration and limited generalization."
                ),
            },
        ]
    }

    repaired = chat_render._reading_guide_repair_dl_spi_benefit_marker(
        answer,
        hits,
        plan,
        canonical_paths=["dl-spi.en.md"],
    )

    assert repaired.count("[1]") == 2
    assert "重建质量和重建速度 [1]。" in repaired
    assert "训练时间长、泛化能力有限" in repaired
    assert "固有限制" not in repaired


def test_dl_spi_benefit_repair_preserves_multi_paper_roadmap_markers():
    from api import chat_render

    answer = (
        "先读基础综述 [1]。\n\n"
        "再读深度学习综述，了解重建质量、重建速度和训练泛化风险 [2]。\n\n"
        "最后比较 Hadamard 与 Fourier 编码 [3]。"
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "spi-prospects.en.md",
                "evidence_quote": "Single-pixel imaging foundation.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "dl-spi.en.md",
                "evidence_quote": (
                    "Deep learning provides exceptional reconstruction quality and "
                    "fast reconstruction speed."
                ),
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": "hsi-fsi.en.md",
                "evidence_quote": "Hadamard and Fourier basis comparison.",
                "candidate_hits": [3],
            },
        ]
    }
    hits = [
        {"text": "foundation", "meta": {"source_path": "spi-prospects.en.md"}},
        {"text": "DL review", "meta": {"source_path": "dl-spi.en.md"}},
        {"text": "comparison", "meta": {"source_path": "hsi-fsi.en.md"}},
    ]

    repaired = chat_render._reading_guide_repair_dl_spi_benefit_marker(
        answer,
        hits,
        plan,
        canonical_paths=[
            "spi-prospects.en.md",
            "dl-spi.en.md",
            "hsi-fsi.en.md",
        ],
    )

    assert repaired == answer
    assert "[2]" in repaired


def test_missing_system_a_repair_does_not_force_single_pixel_source_onto_spad_reading_tip():
    from api import chat_render

    answer = (
        "physics-informed deep learning 在 SPAD 单光子成像中建立物理噪声模型 [1]。\n\n"
        "**阅读建议**：阅读 *High-resolution single-photon imaging with physics-informed "
        "deep learning* 的 Introduction 和参数校准部分。"
    )
    hits = [
        {
            "text": "A physical noise model supports SPAD single-photon reconstruction.",
            "meta": {"source_path": "pidl.en.md", "ref_answer_citation_num": 1},
        },
        {
            "text": "Deep learning single-pixel imaging improves reconstruction quality and speed.",
            "meta": {"source_path": "dl-spi.en.md", "ref_answer_citation_num": 2},
        },
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "pidl.en.md",
                "candidate_hits": [1],
                "evidence_quote": hits[0]["text"],
            },
            {
                "preferred_system": "system_a",
                "source_path": "dl-spi.en.md",
                "candidate_hits": [2],
                "evidence_quote": hits[1]["text"],
            },
        ],
    }

    repaired = chat_render._reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=["pidl.en.md", "dl-spi.en.md"],
    )

    assert repaired.count("[1]") == 1
    assert "[2]" not in repaired


def test_claim_level_repair_does_not_add_citation_to_navigation_only_reading_advice():
    from api import chat_render

    source_path = "pidl.en.md"
    evidence = (
        "We established a real-world physical noise model of SPAD arrays and calibrated "
        "it with a real-shot dataset."
    )
    answer = (
        "SPAD 单光子成像使用真实物理噪声模型改善重建 [1]。\n\n"
        "**阅读建议**：如果想了解网络结构，可查阅原文的实验部分和结构图。"
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_path": source_path,
                "source_name": "High-resolution single-photon imaging with physics-informed deep learning",
                "ref_answer_citation_num": 1,
            },
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": hits[0]["meta"]["source_name"],
                "candidate_hits": [1],
                "evidence_quote": evidence,
            }
        ]
    }

    repaired = chat_render._reading_guide_attach_claim_level_system_a_citations(
        answer,
        hits,
        plan,
        canonical_paths=[source_path],
    )

    assert repaired == answer
    assert repaired.count("[1]") == 1


def test_missing_system_a_repair_does_not_force_detector_review_onto_cited_spad_tip():
    from api import chat_render

    answer = (
        "SPAD 的物理噪声模型支撑单光子重建 [2]。\n\n"
        "**阅读建议**：这篇 High-resolution single-photon imaging with physics-informed "
        "deep learning 论文是处理 SPAD 低信噪比数据的必读材料 [2]。"
    )
    hits = [
        {
            "text": (
                "A detector review discusses SPAD sensitivity, photon detection, low dark count, "
                "and detector performance."
            ),
            "meta": {"source_path": "detector-review.en.md", "ref_answer_citation_num": 1},
        },
        {
            "text": "A real-world physical noise model supports SPAD single-photon reconstruction.",
            "meta": {"source_path": "pidl.en.md", "ref_answer_citation_num": 2},
        },
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": hit["meta"]["source_path"],
                "candidate_hits": [index],
                "evidence_quote": hit["text"],
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }

    repaired = chat_render._reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=["detector-review.en.md", "pidl.en.md"],
    )

    assert "[1]" not in repaired
    assert repaired.count("[2]") == 2


def test_perovskite_scope_bridge_does_not_rewrite_answer_without_boundary_claim():
    from api.chat_render import _reading_guide_repair_scope_boundary_citation

    answer = "The perovskite laser uses a dual-cavity device and we should inspect its materials stack."
    plan = {
        "intent": "scope_boundary",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "perovskite.en.md",
                "evidence_quote": "We demonstrate lasing from a dual-cavity perovskite device.",
                "candidate_hits": [1],
            }
        ],
    }
    hits = [{"text": "Device evidence.", "meta": {"source_path": "perovskite.en.md"}}]

    repaired = _reading_guide_repair_scope_boundary_citation(answer, hits, plan)

    assert repaired == answer


def test_perovskite_scope_bridge_accepts_direct_relevance_is_weak_wording() -> None:
    from api.chat_render import _reading_guide_repair_scope_boundary_citation

    source_path = "perovskite.en.md"
    answer = (
        "这篇论文研究电驱动钙钛矿激光器，与单像素成像的技术领域差异较大，"
        "直接关联性不强 [1]。"
    )
    plan = {
        "intent": "scope_boundary",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "evidence_quote": (
                    "We demonstrate electrically driven lasing from a dual-cavity perovskite device."
                ),
                "candidate_hits": [1],
            }
        ],
    }
    hits = [
        {
            "text": "Device evidence.",
            "meta": {
                "source_path": source_path,
                "ref_answer_citation_num": 1,
                "citation_plan_scope_boundary": True,
            },
        }
    ]

    repaired = _reading_guide_repair_scope_boundary_citation(
        answer,
        hits,
        plan,
        canonical_paths=[source_path],
    )

    assert "不是单像素成像方法 [1]" in repaired


def test_beginner_roadmap_drops_title_marker_when_same_paper_has_claim_marker() -> None:
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    paths = ["spi-prospects.en.md", "dl-review.en.md", "hsi-fsi.en.md"]
    hits = [
        {
            "text": "Paper evidence.",
            "meta": {"source_path": path, "ref_answer_citation_num": num},
        }
        for num, path in enumerate(paths, start=1)
    ]
    plan = {
        "intent": "beginner_overview",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": path,
                "source_name": name,
                "candidate_hits": [num],
                "evidence_quote": evidence,
            }
            for num, (path, name, evidence) in enumerate(
                zip(
                    paths,
                    (
                        "Principles and prospects for single-pixel imaging",
                        "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                        "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                    ),
                    (
                        "Images can be recovered when measurements are fewer than unknown pixels.",
                        "Deep learning improves reconstruction quality and speed.",
                        "HSI and FSI are compared in efficiency and noise robustness.",
                    ),
                    ),
                start=1,
            )
        ],
    }
    answer = (
        "1. **《Principles and prospects for single-pixel imaging》** [1]\n"
        "- 先建立基础原理。\n\n"
        "2. **《Advances and Challenges of Single-Pixel Imaging Based on Deep Learning》** [2]\n"
        "- 重点看深度学习如何改善重建质量和速度 [2]。\n\n"
        "3. **《Hadamard single-pixel imaging versus Fourier single-pixel imaging》** [3]\n"
        "- 重点看两种编码的效率和噪声鲁棒性 [3]。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=paths,
    )

    assert re.findall(r"(?<![!\\])\[(\d+)\](?!\()", repaired) == ["1", "2", "3"]
    assert "imaging》** [2]" not in repaired
    assert "imaging》** [3]" not in repaired


def test_beginner_roadmap_moves_sole_title_marker_to_explanatory_claim() -> None:
    from api.chat_render import _reading_guide_drop_redundant_paper_identity_markers

    source_path = "spi-prospects.en.md"
    source_name = "Principles and prospects for single-pixel imaging"
    answer = (
        "1. **Principles and prospects for single-pixel imaging** (2019) [1]\n"
        "**What to read**: this review explains how a single-pixel camera recovers images "
        "when measurements are fewer than unknown pixels."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "source_name": source_name,
                "evidence_quote": (
                    "Images can be recovered from a single-pixel camera when the number of "
                    "measurements is fewer than the total number of unknown pixels."
                ),
            }
        ]
    }

    repaired = _reading_guide_drop_redundant_paper_identity_markers(
        answer,
        [{"text": plan["slots"][0]["evidence_quote"], "meta": {"source_path": source_path}}],
        canonical_paths=[source_path],
        citation_plan=plan,
    )

    assert "imaging** (2019) [1]" not in repaired
    assert "unknown pixels [1]." in repaired


def test_beginner_roadmap_anchors_marker_before_narrow_focus_hint() -> None:
    from api.chat_render import _reading_guide_drop_redundant_paper_identity_markers

    source_path = "F:/library/OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    answer = (
        "1. **Hadamard single-pixel imaging versus Fourier single-pixel imaging** [3]\n"
        "**看什么**：这篇直接对比了 Hadamard 基（HSI）和 Fourier 基（FSI）。"
        "重点看 Table 4，它列出了每个系数的测量次数。\n"
    )
    repaired = _reading_guide_drop_redundant_paper_identity_markers(
        answer,
        [],
        canonical_paths=["a.en.md", "b.en.md", source_path],
        citation_plan={
            "slots": [
                {
                    "source_path": source_path,
                    "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                    "candidate_hits": [3],
                    "evidence_quote": (
                        "HSI uses Hadamard basis patterns for illumination while FSI uses Fourier basis patterns. "
                        "We compare these two representative techniques."
                    ),
                }
            ]
        },
    )

    assert "imaging** [3]" not in repaired
    assert "（FSI） [3]。重点看 Table 4" in repaired


def test_preservation_gate_allows_exact_spad_mechanism_bridge() -> None:
    from api import chat_render

    original = "SPAD operates in Geiger mode to detect individual photons [1]."
    bridge = (
        "The source states the complete mechanism: a SPAD operates in Geiger mode above "
        "its reverse bias breakdown voltage and requires a quenching circuit [1]."
    )
    repaired = f"{bridge}\n\n{original.replace(' [1]', '')}"
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "spad.en.md",
                "evidence_quote": (
                    "A SPAD operates in Geiger mode above its reverse bias breakdown voltage "
                    "and must be supported by a quenching circuit."
                ),
            }
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_preservation_gate_allows_exact_single_source_s2ism_tradeoff_repair() -> None:
    from api import chat_render

    repaired = "s2ISM 同时改善空间分辨率、信噪比和光学切片能力 [1]。"
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "s2ism.en.md",
                "evidence_quote": (
                    "There is a trade-off between spatial resolution and signal-to-noise ratio. "
                    "Existing methods lack optical sectioning for thick samples, while the "
                    "achievable sectioning depends on detector size."
                ),
            }
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body="s2ISM resolves the trade-off [1].",
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_preservation_gate_allows_exact_single_source_sequential_support_repair() -> None:
    from api import chat_render

    repaired = (
        "顺序自适应压缩感知（基于 distilled sensing / 蒸馏感知）实现"
        "信号支撑集恢复（signal support recovery） [1]。"
    )
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "sequential.en.md",
                "evidence_quote": (
                    "A sequential adaptive compressed sensing procedure for signal support "
                    "recovery is based on the principle of distilled sensing."
                ),
            }
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body="Sequential sensing recovers the support [1].",
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_preservation_gate_allows_exact_single_source_spi_use_case_repair() -> None:
    from api import chat_render

    evidence = (
        "Images can be collected at wavelengths outside the reach of FPA technology or "
        "at high frame rates or in three dimensions. Promising applications include "
        "hazardous gas leaks and autonomous vehicles."
    )
    repaired = (
        "单像素相机适合面阵相机无法覆盖的波段、高帧率和三维成像；代表应用包括"
        "危险气体泄漏与自动驾驶 [1]。"
    )
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "spi-prospects.en.md",
                "evidence_quote": evidence,
            }
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body="代表应用包括危险气体泄漏和自动驾驶 [3]。",
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_preservation_gate_allows_exact_three_method_microscopy_map() -> None:
    from api import chat_render

    repaired = (
        "s2ISM uses structured detection for optical sectioning [1].\n\n"
        "iISM uses interferometric detection and reaches 120 nm [2].\n\n"
        "Light-field microscopy records position and angular information for volumetric "
        "reconstruction and digital refocusing [3]."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "s2ism.en.md",
                "evidence_quote": "Structured detection provides optical sectioning.",
            },
            {
                "preferred_system": "system_a",
                "source_path": "iism.en.md",
                "evidence_quote": "Interferometric detection reaches 120 nm lateral resolution.",
            },
            {
                "preferred_system": "system_a",
                "source_path": "light-field.en.md",
                "evidence_quote": (
                    "Light-field microscopy records position and angular information for "
                    "volumetric reconstruction and digital refocusing."
                ),
            },
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body="Compare three microscopy methods [1].",
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_reading_guide_repair_bridges_perovskite_device_scope_to_chinese_claim():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _backfill_system_a_cite_details_from_ref_pack,
        _reading_guide_repair_missing_system_a_citations,
        _should_link_inpaper_citations_for_message,
    )
    from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta

    source_path = "F:/library/perovskite-laser.en.md"
    slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Abstract",
        "evidence_quote": "We demonstrate electrically driven lasing from a dual-cavity perovskite device.",
    }
    plan = {
        "intent": "scope_boundary",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
    )
    answer = (
        "直接回答：关系不大，不是当前主线的核心文献。\n\n"
        "这篇论文研究电驱动钙钛矿激光器的器件结构。"
        "你的单像素成像主线属于计算成像，两者几乎没有交集。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "dual-cavity perovskite" in repaired
    assert "lasing 研究，而不是单像素成像方法 [1]" in repaired
    _rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        citation_plan=plan,
    )
    details = _backfill_system_a_cite_details_from_ref_pack(
        details,
        {
            "primary_evidence": {
                "source_path": source_path,
                "source_name": "Perovskite laser.pdf",
                "heading_path": "Abstract",
                "snippet": slot["evidence_quote"],
                "selection_reason": "prompt_aligned",
            }
        },
        render_locale="zh",
    )
    assert len(details) == 1
    detail = details[0]
    assert detail["citation_route"] == "system_a"
    assert detail["binding_status"] == "grounded"
    assert all(term in detail["answer_claim"] for term in ("perovskite", "器件", "不是"))
    assert all(term in detail["evidence_quote"] for term in ("dual-cavity perovskite", "lasing"))
    assert "Abstract" in detail["heading_path"]

    concise_repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="concise_answer",
    )
    rec = {
        "content": answer,
        "meta": {
            "answer_quality": {
                "output_mode": "concise_answer",
                "citation_plan": plan,
            }
        },
    }
    assert "dual-cavity perovskite" in concise_repaired
    assert _should_link_inpaper_citations_for_message(rec=rec, content=answer, hits=hits)

    omitted_identity_answer = (
        "Direct answer: this is not central to the current imaging route.\n\n"
        "The device details are outside a workflow focused on encoding and reconstruction."
    )
    omitted_identity_repaired = _reading_guide_repair_missing_system_a_citations(
        omitted_identity_answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "dual-cavity perovskite lasing device" in omitted_identity_repaired
    assert "not a single-pixel imaging method [1]" in omitted_identity_repaired


def test_beginner_roadmap_restores_omitted_foundational_paper_without_rebuilding_answer():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    paths = ["dl-review.en.md", "spi-prospects.en.md", "hsi-fsi.en.md"]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": paths[0],
                "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "heading_path": "Abstract",
                "evidence_quote": "Deep learning improves reconstruction quality and speed.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[1],
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Acquisition and image reconstruction strategies",
                "evidence_quote": "Images can be recovered when measurements are fewer than unknown pixels.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[2],
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "heading_path": "Introduction",
                "evidence_quote": "HSI and FSI are compared in efficiency and noise robustness.",
                "candidate_hits": [3],
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {"text": "DL evidence", "meta": {"source_path": paths[0]}},
            {"text": "Foundation evidence", "meta": {"source_path": paths[1]}},
            {"text": "Comparison evidence", "meta": {"source_path": paths[2]}},
        ],
        plan,
        reserved_count=3,
    )
    answer = (
        "要快速建立单像素成像的知识主线，建议重点阅读 3 篇。\n\n"
        "### 1. 深度学习综述\nLPR 提供进展与局限 [1]。\n\n"
        "### 2. 编码对比\nHadamard 与 Fourier 的区别 [3]。\n\n"
        "**总结行动建议**：按顺序阅读。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=paths,
    )

    assert "Principles and prospects for single-pixel imaging" in repaired
    assert "compressive sensing" in repaired
    assert "[2]" in repaired
    assert "[4]" not in repaired
    assert "[3]" in repaired
    assert "[6]" not in repaired
    assert "### 2. 深度学习综述" in repaired
    assert "### 3. 编码对比" in repaired


def test_beginner_roadmap_completes_empty_dl_section_and_binds_comparison_claim():
    from api.chat_render import _reading_guide_repair_beginner_roadmap_missing_paper

    paths = ["spi-prospects.en.md", "dl-review.en.md", "hsi-fsi.en.md"]
    plan = {
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": paths[0],
                "source_name": "Principles and prospects for single-pixel imaging",
                "evidence_quote": "Images can be recovered from fewer measurements.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[1],
                "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "evidence_quote": (
                    "Iterative reconstruction has limited image quality and lengthy computational "
                    "times; deep learning offers exceptional reconstruction quality and fast speed."
                ),
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[2],
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "evidence_quote": "HSI uses Hadamard patterns while FSI uses Fourier patterns.",
                "candidate_hits": [3],
            },
        ],
    }
    hits = [
        {"text": "foundation", "meta": {"source_path": paths[0], "ref_answer_citation_num": 1}},
        {"text": "DL review", "meta": {"source_path": paths[1], "ref_answer_citation_num": 2}},
        {"text": "comparison", "meta": {"source_path": paths[2], "ref_answer_citation_num": 3}},
    ]
    answer = (
        "建议按主线阅读三篇论文。\n\n"
        "### 1. 建立原理框架：Principles and prospects for single-pixel imaging\n"
        "先理解欠采样重建 [1]。\n\n"
        "### 2. 掌握主流方法对比：Hadamard single-pixel imaging versus Fourier single-pixel imaging\n"
        "- **主要看什么**：Hadamard 基（HSI）与 Fourier 基（FSI）的原理对比。\n\n"
        "### 3. 了解前沿进展：Advances and Challenges of Single-Pixel Imaging Based on Deep Learning\n\n"
        "### 阅读建议\n按顺序阅读。"
    )

    repaired = _reading_guide_repair_beginner_roadmap_missing_paper(
        answer,
        hits,
        plan,
        canonical_paths=paths,
    )

    assert "传统迭代重建的图像质量与计算耗时瓶颈" in repaired
    assert "重建质量和重建速度收益 [2]" in repaired
    assert "Hadamard 基（HSI）与 Fourier 基（FSI）的原理对比 [3]" in repaired


def test_reading_guide_repair_ignores_unrelated_same_paper_method_slot():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "qclfm.en.md"
    refocus_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "A. Concept",
        "evidence_quote": (
            "Digital refocusing uses two steps. First, ray tracing reconstructs photon trajectories. "
            "The second step applies wave propagation to reverse diffraction."
        ),
    }
    unrelated_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Figure 1",
        "evidence_quote": "Type-II spontaneous parametric down-conversion produces orthogonally polarized photon pairs.",
    }
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [refocus_slot, unrelated_slot],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [{"text": "Paper overview.", "meta": {"source_path": source_path}}],
        plan,
    )
    answer = "数字重聚焦分为两步：先用 ray tracing 重建轨迹，再用 wave propagation 反演衍射。"

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert "反演衍射 [2]。" in repaired
    assert "[3]" not in repaired


def test_system_a_render_backfills_public_bibliography_without_primary_evidence(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack
    from kb.citation_card import compose_citation_card

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {},
    )
    source_path = r"db\Nature-2024-Useful paper\Nature-2024-Useful paper.en.md"
    details = [
        compose_citation_card({
            "num": 1,
            "anchor": "kb-cite-1",
            "source_name": "Nature-2024-Useful paper.pdf",
            "source_path": source_path,
            "title": "3. Results",
            "heading_path": "3. Results",
            "is_inpaper": False,
            "citation_route": "system_a",
            "answer_claim": "The method improves reconstruction quality.",
            "evidence_quote": "The method improves reconstruction quality.",
        })
    ]
    ref_pack = {
        "hits": [
            {
                "text": "The method improves reconstruction quality.",
                "meta": {"source_path": source_path},
                "ui_meta": {
                    "citation_meta": {
                        "title": "Useful Paper",
                        "authors": "Ada Lovelace; Grace Hopper",
                        "venue": "Nature Methods",
                        "year": "2024",
                        "doi": "10.1234/useful.paper",
                        "doi_url": "https://doi.org/10.9999/wrong-url",
                        "venue_kind": "journal",
                        "metadata_quality": {"score": 99},
                    }
                },
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert len(out) == 1
    detail = out[0]
    assert detail["title"] == "Useful Paper"
    assert detail["bibliographic_title"] == "Useful Paper"
    assert detail["authors"] == "Ada Lovelace; Grace Hopper"
    assert detail["venue"] == "Nature Methods"
    assert detail["year"] == "2024"
    assert detail["doi"] == "10.1234/useful.paper"
    assert detail["doi_url"] == "https://doi.org/10.1234/useful.paper"
    assert detail["venue_kind"] == "journal"
    assert "metadata_quality" not in detail
    assert detail["metadata_export_acceptance"]["export_ready"] is True
    assert detail["metadata_export_acceptance"]["export_mode"] == "complete_with_doi"
    assert detail["heading_path"] == "3. Results"
    assert detail["card_view"]["header"]["subtitle"] == "3. Results"


def test_system_a_render_prefers_abstract_root_over_truncated_local_filename_title(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {
            "title": "Interferometric Image Scanning...inside live cells",
            "venue": "LSA",
            "year": "2026",
            "doi": "10.1038/example.iism",
        },
    )
    article_title = (
        "Interferometric Image Scanning Microscopy for label-free imaging at "
        "120 nm lateral resolution inside live cells"
    )
    detail = {
        "num": 1,
        "anchor": "kb-cite-iism",
        "source_name": "LSA-2026-Interferometric Image Scanning...inside live cells.pdf",
        "source_path": "db/iism/iism.en.md",
        "title": f"{article_title} / Abstract",
        "heading_path": f"{article_title} / Abstract",
        "citation_route": "system_a",
        "is_inpaper": False,
        "evidence_quote": "The method operates at lower incident illumination power.",
        "summary_line": "The method operates at lower incident illumination power.",
    }

    out = _backfill_system_a_cite_details_from_ref_pack(
        [detail],
        {"hits": [{"meta": {"source_path": detail["source_path"]}, "ui_meta": {}}]},
    )

    assert out[0]["title"] == article_title
    assert out[0]["bibliographic_title"] == article_title
    assert out[0]["heading_path"] == f"{article_title} / Abstract"
    assert out[0]["metadata_export_acceptance"]["export_ready"] is True


def test_system_a_prompt_contract_refines_existing_locator(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {},
    )
    source_path = "db/SPAD/SPAD.en.md"
    evidence = (
        "SPAD operates in Geiger mode above reverse breakdown voltage and "
        "must be supported by a quenching circuit."
    )
    details = [
        {
            "num": 1,
            "source_path": source_path,
            "source_name": "SPAD.pdf",
            "heading_path": "1 Introduction",
            "evidence_quote": evidence,
            "summary_line": evidence,
            "raw": evidence,
            "block_id": "blk-principle",
            "answer_claim": "SPAD needs Geiger-mode bias and a quenching circuit.",
            "citation_route": "system_a",
            "is_inpaper": False,
        }
    ]
    ref_pack = {
        "primary_evidence": {
            "source_path": source_path,
            "source_name": "SPAD.pdf",
            "heading_path": (
                "1 Introduction / Principle of single photon detection avalanche diode"
            ),
            "snippet": (
                "SPAD operates in Geiger mode … above reverse breakdown voltage … "
                "supported by a quenching circuit."
            ),
            "block_id": "blk-principle",
            "anchor_id": "p-principle",
            "page_start": 2,
            "page_end": 2,
            "selection_reason": "prompt_contract_block",
            "strict_locate": True,
        },
        "hits": [
            {
                "text": evidence,
                "meta": {"source_path": source_path},
                "ui_meta": {"source_path": source_path},
            }
        ],
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert len(out) == 1
    assert "Principle of single photon detection avalanche diode" in out[0]["heading_path"]
    assert out[0]["block_id"] == "blk-principle"
    assert out[0]["page_start"] == 2


def test_render_packet_final_primary_refines_citation_locator(monkeypatch):
    from api.chat_render import _merge_render_packet_contract_meta

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {},
    )
    source_path = "db/SPAD/SPAD.en.md"
    full_evidence = (
        "SPAD operates in Geiger mode above reverse breakdown voltage and "
        "must be supported by a quenching circuit."
    )
    final_primary = {
        "source_path": source_path,
        "source_name": "SPAD.pdf",
        "heading_path": (
            "1 Introduction / Principle of single photon detection avalanche diode"
        ),
        "snippet": (
            "SPAD operates in Geiger mode … above reverse breakdown voltage … "
            "supported by a quenching circuit."
        ),
        "block_id": "blk-principle",
        "anchor_id": "p-principle",
        "page_start": 2,
        "page_end": 2,
        "selection_reason": "prompt_contract_block",
        "strict_locate": True,
    }
    rec = {
        "content": "SPAD needs Geiger-mode bias and quenching.",
        "cite_details": [
            {
                "num": 1,
                "source_path": source_path,
                "source_name": "SPAD.pdf",
                "heading_path": "1 Introduction",
                "evidence_quote": full_evidence,
                "summary_line": full_evidence,
                "raw": full_evidence,
                "block_id": "blk-principle",
                "anchor_id": "p-principle",
                "answer_claim": "SPAD needs Geiger-mode bias and quenching.",
                "citation_route": "system_a",
                "is_inpaper": False,
            }
        ],
        "meta": {
            "paper_guide_contracts": {
                "primary_evidence": final_primary,
                "render_packet": {},
            }
        },
    }
    ref_pack = {
        "hits": [
            {
                "text": full_evidence,
                "meta": {"source_path": source_path},
                "ui_meta": {"source_path": source_path},
            }
        ]
    }

    _merge_render_packet_contract_meta(
        rec=rec,
        msg_id=1,
        enriched_provenance={},
        ref_pack=ref_pack,
        render_locale="zh",
    )

    details = rec["meta"]["paper_guide_contracts"]["render_packet"]["cite_details"]
    assert len(details) == 1
    assert "Principle of single photon detection avalanche diode" in details[0]["heading_path"]
    assert details[0]["block_id"] == "blk-principle"


def test_reading_guide_does_not_bind_piln_evidence_to_pidl_retrieval_notice():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "F:/library/Part-based image-loop network for single-pixel imaging.en.md"
    evidence = (
        "Researchers embed an untrained neural network into the physical model for "
        "single-pixel image reconstruction."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Part-based image-loop network for single-pixel imaging.pdf",
                "heading_path": "1. Introduction",
                "evidence_quote": evidence,
            }
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots(
        [
            {
                "text": "PILN is an untrained network for single-pixel imaging.",
                "meta": {"source_path": source_path},
            }
        ],
        plan,
    )
    answer = (
        "根据检索到的文献，PIDL 的相关内容未出现在本次检索结果中，"
        "因此以下比较仅基于检索到的 PILN 信息。\n\n"
        "PILN 将单像素成像物理模型嵌入未训练神经网络 [1]。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    retrieval_notice, piln_claim = repaired.split("\n\n", 1)
    assert "[2]" not in retrieval_notice
    assert "[2]" in piln_claim


def test_system_a_bibliography_priority_is_existing_then_ref_pack_then_local(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {
            "title": "Stale Local Title",
            "authors": "Local Author",
            "venue": "Local Venue",
            "year": "2020",
            "doi": "10.1000/local",
        },
    )
    source_path = "db/paper.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_path,
            "source_name": "paper.pdf",
            "citation_route": "system_a",
            "title": "Existing Detail Title",
            "bibliographic_title": "Existing Detail Title",
            "authors": "Existing Detail Author",
            "heading_path": "3. Results",
            "evidence_quote": "Grounded evidence.",
        }
    ]
    ref_pack = {
        "hits": [
            {
                "meta": {"source_path": source_path},
                "ui_meta": {
                    "citation_meta": {
                        "title": "New Ref Pack Title",
                        "authors": "Ref Pack Author",
                        "venue": "Ref Pack Venue",
                        "year": "2025",
                    }
                },
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert out[0]["title"] == "Existing Detail Title"
    assert out[0]["bibliographic_title"] == "Existing Detail Title"
    assert out[0]["authors"] == "Existing Detail Author"
    assert out[0]["venue"] == "Ref Pack Venue"
    assert out[0]["year"] == "2025"
    assert out[0]["doi"] == "10.1000/local"


def test_system_a_bibliography_keeps_same_basename_paths_and_dois_separate(monkeypatch):
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    monkeypatch.setattr(
        "api.chat_render.load_local_source_citation_meta",
        lambda *_args, **_kwargs: {},
    )
    source_a = r"db\collection-a\Repeated Paper.en.md"
    source_b = r"db\collection-b\Repeated Paper.en.md"
    details = [
        {
            "num": 1,
            "source_path": source_a,
            "source_name": "Repeated Paper.pdf",
            "citation_route": "system_a",
            "title": "Methods A",
            "heading_path": "Methods A",
            "evidence_quote": "Evidence from collection A.",
        },
        {
            "num": 2,
            "source_path": source_b,
            "source_name": "Repeated Paper.pdf",
            "citation_route": "system_a",
            "title": "Methods B",
            "heading_path": "Methods B",
            "evidence_quote": "Evidence from collection B.",
        },
        {
            "num": 3,
            "source_name": "Repeated Paper.pdf",
            "citation_route": "system_a",
            "title": "Ambiguous Methods",
            "heading_path": "Ambiguous Methods",
            "evidence_quote": "Evidence without a source path.",
        },
    ]
    ref_pack = {
        "hits": [
            {
                "meta": {"source_path": source_a},
                "ui_meta": {
                    "citation_meta": {
                        "title": "Collection A Paper",
                        "doi": "10.1234/collection-a",
                    }
                },
            },
            {
                "meta": {"source_path": source_b},
                "ui_meta": {
                    "citation_meta": {
                        "title": "Collection B Paper",
                        "doi": "10.1234/collection-b",
                    }
                },
            },
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert out[0]["bibliographic_title"] == "Collection A Paper"
    assert out[0]["doi"] == "10.1234/collection-a"
    assert out[1]["bibliographic_title"] == "Collection B Paper"
    assert out[1]["doi"] == "10.1234/collection-b"
    assert "bibliographic_title" not in out[2]
    assert "doi" not in out[2]


def test_system_a_primary_evidence_backfill_preserves_page_locator():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "db/Paper/Paper.en.md"
    details = [
        {
            "num": 1,
            "citation_route": "system_a",
            "source_path": source_path,
            "answer_claim": "SCINeRF embeds the physical imaging process in NeRF training.",
            "evidence_quote": "NeRF training.",
        }
    ]
    primary = {
        "source_path": source_path,
        "source_name": "Paper",
        "heading_path": "Abstract",
        "snippet": "We formulate the physical imaging process of SCI as part of the training of NeRF.",
        "highlight_snippet": "We formulate the physical imaging process of SCI as part of the training of NeRF.",
        "block_id": "blk-1",
        "anchor_id": "p-1",
        "anchor_kind": "paragraph",
        "page_start": 1,
        "page_end": 1,
    }
    pack = {
        "primary_evidence": primary,
        "hits": [
            {
                "text": primary["snippet"],
                "meta": {"source_path": source_path},
                "ui_meta": {"source_path": source_path, "primary_evidence": primary},
            }
        ],
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, pack, render_locale="en")

    assert out[0]["page_start"] == 1
    assert out[0]["page_end"] == 1
    assert "p. 1" in out[0]["location_label"]


def test_answer_aligned_ref_primary_becomes_page_aware_citation_plan_hit():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _citation_plan_with_ref_primary,
    )

    primary = {
        "source_path": "db/Paper/Paper.en.md",
        "source_name": "Paper",
        "heading_path": "Abstract",
        "snippet": "The supported answer evidence.",
        "block_id": "blk-2",
        "anchor_id": "p-2",
        "anchor_kind": "paragraph",
        "page_start": 2,
        "page_end": 2,
        "strict_locate": True,
    }
    plan = _citation_plan_with_ref_primary(
        {
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "db/Paper/Paper.en.md",
                    "heading_path": "Introduction",
                    "evidence_quote": "A generic same-paper passage.",
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "db/Other/Other.en.md",
                    "heading_path": "Results",
                    "evidence_quote": "A relevant passage from another paper.",
                },
            ],
        },
        {"primary_evidence": primary},
    )
    hits = _augment_hits_with_system_a_plan_slots([], plan)

    assert plan["slots"][0]["selection_reason"] == "answer_aligned_reference_primary"
    assert len(plan["slots"]) == 2
    assert plan["slots"][1]["source_path"] == "db/Other/Other.en.md"
    assert hits[0]["meta"]["page_start"] == 2
    assert hits[0]["meta"]["primary_block_id"] == "blk-2"
    assert hits[0]["ui_meta"]["primary_evidence"]["page_start"] == 2


def test_answer_aligned_ref_primary_keeps_authoritative_answer_number():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/CASSI/CASSI.en.md"
    primary = {
        "source_path": source_path,
        "source_name": "CASSI",
        "heading_path": "Abstract",
        "snippet": "Two dispersive elements surround a binary-valued aperture.",
        "block_id": "blk-cassi",
        "anchor_id": "p-cassi",
        "page_start": 1,
        "selection_reason": "prompt_contract_block",
        "strict_locate": True,
    }

    plan = _citation_plan_with_ref_primary(
        {"budget": {"system_a": 1}, "slots": []},
        {
            "primary_evidence": primary,
            "hits": [
                {
                    "text": primary["snippet"],
                    "meta": {
                        "source_path": source_path,
                        "ref_answer_citation_num": 2,
                    },
                }
            ],
        },
    )

    assert plan["slots"][0]["candidate_hits"] == [2]


def test_answer_aligned_ref_primary_preserves_multi_claim_same_paper_slots():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/DL-SPI/DL-SPI.en.md"
    benefit = "Deep learning provides exceptional reconstruction quality and fast reconstruction speed."
    risk = "Data-driven methods have prolonged training duration and limited generalization."
    original = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {"preferred_system": "system_a", "source_path": source_path, "evidence_quote": benefit},
            {"preferred_system": "system_a", "source_path": source_path, "evidence_quote": risk},
        ],
    }

    resolved = _citation_plan_with_ref_primary(
        original,
        {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "Neural Network Basics",
                "snippet": "Artificial neural networks contain input, hidden, and output layers.",
            }
        },
    )

    assert resolved["slots"] == original["slots"]


def test_prompt_contract_primary_replaces_generic_same_paper_slots():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/SPH/SPH.en.md"
    original = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": "SPH uses a single-pixel detector.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Conclusion",
                "evidence_quote": "We developed high-throughput SPH.",
            },
        ],
    }
    exact = (
        "A beat frequency realizes phase stepping naturally in time by exploiting "
        "the framework of heterodyne holography."
    )

    resolved = _citation_plan_with_ref_primary(
        original,
        {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "Introduction",
                "snippet": exact,
                "block_id": "blk-intro",
                "anchor_id": "p-intro",
                "selection_reason": "prompt_contract_block",
                "strict_locate": True,
            }
        },
    )

    assert len(resolved["slots"]) == 1
    assert resolved["slots"][0]["heading_path"] == "Introduction"
    assert resolved["slots"][0]["evidence_quote"] == exact
    assert resolved["slots"][0]["evidence_selection_reason"] == "prompt_contract_block"


def test_prompt_contract_primary_replaces_truncated_same_block_slot():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/SPAD/SPAD.en.md"
    truncated = "SPAD operates in Geiger mode above its reverse bias breakdown voltage."
    complete = truncated + " The avalanche diode must be supported by a quenching circuit."
    resolved = _citation_plan_with_ref_primary(
        {
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "heading_path": "Principle of SPAD",
                    "block_id": "blk-principle",
                    "evidence_quote": truncated,
                    "evidence_selection_reason": "prompt_aligned_source_sentence",
                }
            ]
        },
        {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "Principle of SPAD",
                "snippet": complete,
                "block_id": "blk-principle",
                "anchor_id": "p-principle",
                "selection_reason": "prompt_contract_block",
                "strict_locate": True,
            }
        },
    )

    assert resolved["slots"][0]["evidence_quote"] == complete
    assert resolved["slots"][0]["evidence_selection_reason"] == "prompt_contract_block"


def test_prompt_contract_primary_drops_generic_slots_with_incidental_noise_terms():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/SPAD/SPAD.en.md"
    complete = (
        "SPAD operates in Geiger mode above its reverse bias breakdown voltage "
        "and must be supported by a quenching circuit."
    )
    resolved = _citation_plan_with_ref_primary(
        {
            "slots": [
                {
                    "claim_type": "own_result",
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "heading_path": "Introduction",
                    "block_id": "blk-principle",
                    "evidence_quote": (
                        "SPAD operates in Geiger mode above breakdown voltage; "
                        "excess current can cause a long dead time."
                    ),
                },
                {
                    "claim_type": "paper_evidence",
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "heading_path": "Low-dimensional SPAD",
                    "block_id": "blk-materials",
                    "evidence_quote": (
                        "Low-dimensional photodetectors can also produce a large dark count."
                    ),
                },
            ]
        },
        {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "Principle",
                "snippet": complete,
                "block_id": "blk-principle",
                "anchor_id": "p-principle",
                "selection_reason": "prompt_contract_block",
                "strict_locate": True,
            }
        },
    )

    assert len(resolved["slots"]) == 1
    assert resolved["slots"][0]["evidence_quote"] == complete
    assert resolved["slots"][0]["evidence_selection_reason"] == "prompt_contract_block"


def test_prompt_aligned_slot_survives_unrelated_same_paper_section_rescue():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/PILN/PILN.en.md"
    exact = (
        "ILNet is a self-supervised image-loop neural network with a part-based model "
        "that enables finer-grained learning."
    )
    original = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": exact,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            }
        ]
    }
    resolved = _citation_plan_with_ref_primary(
        original,
        {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "1. Introduction",
                "snippet": "Large-scale datasets are necessary for data-driven reconstruction.",
                "block_id": "blk-introduction",
                "selection_reason": "section_intent_rescue",
                "strict_locate": True,
            }
        },
    )

    assert resolved["slots"] == original["slots"]


def test_prompt_contract_primary_recovers_public_source_path_from_matching_hit():
    from api.chat_render import _citation_plan_with_ref_primary

    public_path = "kb-source/0/SPAD/SPAD.en.md"
    exact = (
        "SPAD operates in Geiger mode above its reverse bias breakdown voltage "
        "and must be supported by a quenching circuit."
    )
    resolved = _citation_plan_with_ref_primary(
        {
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "F:/repo/db/SPAD/SPAD.en.md",
                    "heading_path": "Figure 2",
                    "evidence_quote": "SPAD operates in Geiger mode.",
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "F:/repo/db/SPAD/SPAD.en.md",
                    "heading_path": "Materials",
                    "evidence_quote": "SPADs use several semiconductor materials.",
                },
            ],
        },
        {
            "primary_evidence": {
                # Public API payloads intentionally omit the private path.
                "source_name": "SPAD.pdf",
                "heading_path": "Principle of SPAD",
                "snippet": exact,
                "block_id": "blk-principle",
                "anchor_id": "p-principle",
                "selection_reason": "prompt_contract_block",
                "strict_locate": True,
            },
            "hits": [
                {
                    "text": exact,
                    "meta": {
                        "source_path": public_path,
                        "primary_block_id": "blk-principle",
                    },
                    "ui_meta": {},
                }
            ],
        },
    )

    assert len(resolved["slots"]) == 1
    assert resolved["slots"][0]["source_path"] == public_path
    assert resolved["slots"][0]["heading_path"] == "Principle of SPAD"
    assert resolved["slots"][0]["evidence_quote"] == exact
    assert resolved["slots"][0]["evidence_selection_reason"] == "prompt_contract_block"


def test_prompt_contract_primary_preserves_other_same_source_claim_slots():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/SPAD/SPAD.en.md"
    noise_evidence = (
        "The SPAD noise model includes dark count, afterpulsing, and crosstalk."
    )
    resolved = _citation_plan_with_ref_primary(
        {
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "claim_type": "mechanism",
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "heading_path": "Principle",
                    "evidence_quote": "SPAD operates in Geiger mode.",
                },
                {
                    "claim_type": "noise_limitations",
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "heading_path": "Noise model",
                    "evidence_quote": noise_evidence,
                },
            ],
        },
        {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "Principle",
                "snippet": (
                    "SPAD operates in Geiger mode above breakdown voltage and "
                    "requires a quenching circuit."
                ),
                "block_id": "blk-principle",
                "anchor_id": "p-principle",
                "selection_reason": "prompt_contract_block",
                "strict_locate": True,
            }
        },
    )

    assert len(resolved["slots"]) == 2
    assert resolved["slots"][0]["evidence_selection_reason"] == "prompt_contract_block"
    assert resolved["slots"][1]["evidence_quote"] == noise_evidence
    assert resolved["slots"][1]["claim_type"] == "noise_limitations"


def test_prompt_contract_primary_reuses_matching_reserved_public_hit():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _citation_plan_with_ref_primary,
    )

    private_path = "F:/repo/db/SPH/SPH.en.md"
    exact = (
        "A beat frequency realizes phase stepping naturally in time by exploiting "
        "the framework of heterodyne holography."
    )
    plan = _citation_plan_with_ref_primary(
        {
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": private_path,
                    "heading_path": "Abstract",
                    "evidence_quote": "SPH uses a single-pixel detector.",
                }
            ],
        },
        {
            "primary_evidence": {
                "source_path": private_path,
                "source_name": "SPH",
                "heading_path": "Introduction",
                "snippet": exact,
                "block_id": "blk-intro",
                "anchor_id": "p-intro",
                "page_start": 2,
                "page_end": 2,
                "selection_reason": "prompt_contract_block",
                "strict_locate": True,
            }
        },
    )
    hits = [
        {
            "text": "A generic SPH passage.",
            "meta": {
                "source_path": "kb-source/0/SPH/SPH.en.md",
                "heading_path": "Abstract",
            },
            "ui_meta": {},
        }
    ]

    rebound = _augment_hits_with_system_a_plan_slots(hits, plan, reserved_count=1)

    assert len(rebound) == 1
    assert rebound[0]["text"] == exact
    assert rebound[0]["meta"]["ref_answer_citation_num"] == 1
    assert rebound[0]["meta"]["primary_block_id"] == "blk-intro"
    assert rebound[0]["ui_meta"]["primary_evidence"]["strict_locate"] is True


def test_prompt_contract_primary_reuses_matching_reserved_exact_source_hit():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _citation_plan_with_ref_primary,
    )

    source_path = "kb-source/0/SPH/SPH.en.md"
    exact = "A beat frequency realizes phase stepping naturally in time."
    plan = _citation_plan_with_ref_primary(
        {"budget": {"system_a": 1, "system_b": 0}, "slots": []},
        {
            "primary_evidence": {
                "source_path": source_path,
                "source_name": "SPH",
                "heading_path": "Introduction",
                "snippet": exact,
                "block_id": "blk-intro",
                "anchor_id": "p-intro",
                "selection_reason": "prompt_contract_block",
                "strict_locate": True,
            }
        },
    )
    hits = [
        {
            "text": "A broader passage from the same source.",
            "meta": {"source_path": source_path, "heading_path": "Introduction"},
            "ui_meta": {},
        }
    ]

    rebound = _augment_hits_with_system_a_plan_slots(hits, plan, reserved_count=3)

    assert len(rebound) == 3
    assert rebound[0]["text"] == exact
    assert rebound[0]["meta"]["ref_answer_citation_num"] == 1


def test_answer_aligned_ref_primary_preserves_multi_paper_reading_role_slots():
    from api.chat_render import _citation_plan_with_ref_primary

    slots = [
        {
            "preferred_system": "system_a",
            "source_path": f"db/paper-{idx}/paper-{idx}.en.md",
            "heading_path": heading,
            "evidence_quote": evidence,
        }
        for idx, heading, evidence in (
            (1, "Abstract", "Deep learning improves reconstruction quality and speed."),
            (2, "Acquisition strategies", "Compressed sensing recovers from fewer measurements."),
            (3, "Introduction", "HSI and FSI are compared in efficiency and noise robustness."),
        )
    ]
    original = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": slots,
    }

    resolved = _citation_plan_with_ref_primary(
        original,
        {
            "primary_evidence": {
                "source_path": "db/paper-3/paper-3.en.md",
                "heading_path": "3.2 Experiments",
                "snippet": "The target uses 4 x 4 pixel binning.",
                "block_id": "blk-experiment",
                "anchor_id": "p-experiment",
                "strict_locate": True,
            }
        },
    )

    assert resolved["slots"] == slots


def test_multi_paper_plan_rebinds_reordered_public_hits_by_source():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    private_paths = [
        f"F:/repo/db/paper-{idx}/paper-{idx}.en.md"
        for idx in range(1, 4)
    ]
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": private_paths[idx - 1],
                "source_name": f"paper-{idx}",
                "heading_path": f"Planned section {idx}",
                "evidence_quote": f"Planned evidence for paper {idx}.",
                "candidate_hits": [idx],
            }
            for idx in range(1, 4)
        ],
    }
    # Canonical answer alignment reordered the hits after candidate_hits were
    # recorded, and persisted only public source URLs.
    hits = [
        {
            "text": f"Weak passage from paper {idx}.",
            "meta": {
                "source_path": f"kb-source/0/paper-{idx}/paper-{idx}.en.md",
                "heading_path": "Unrelated experiment",
            },
            "ui_meta": {},
        }
        for idx in (2, 3, 1)
    ]

    rebound = _augment_hits_with_system_a_plan_slots(hits, plan, reserved_count=3)

    assert len(rebound) == 3
    for idx, hit in zip((2, 3, 1), rebound):
        assert hit["meta"]["citation_plan_slot"] is True
        assert hit["meta"]["heading_path"] == f"Planned section {idx}"
        assert hit["text"] == f"Planned evidence for paper {idx}."
        assert hit["ui_meta"]["primary_evidence"]["selection_reason"] == "citation_plan_slot"


def test_existing_special_system_a_marker_counts_toward_plan_budget():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
    )

    source_path = "db/DL-SPI/DL-SPI.en.md"
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Challenges",
                "evidence_quote": "Data-driven methods have prolonged training duration and limited generalization.",
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": "Deep learning provides exceptional reconstruction quality and fast reconstruction speed.",
            },
        ],
    }
    hits = _augment_hits_with_system_a_plan_slots([], plan)
    answer = (
        "好处：深度学习带来更高的重建质量和更快的重建速度。\n\n"
        "坑：训练时间长，而且泛化能力有限。"
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
    )

    assert repaired.count("[1]") == 1
    assert repaired.count("[2]") == 1
    assert "数据驱动策略的直接局限是训练时间较长、泛化能力有限" in repaired


def test_exact_support_citation_plan_keeps_resolved_related_work_locator():
    from api.chat_render import _citation_plan_with_ref_primary

    source_path = "db/SCINeRF/SCINeRF.en.md"
    exact_plan = {
        "source": "exact_support_preflight",
        "intent": "origin_lookup",
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "SCINeRF / 2. Related Work",
                "evidence_quote": "most existing methods employ ADMM [4]",
                "block_id": "blk-related",
            },
            {
                "preferred_system": "system_b",
                "source_path": source_path,
                "heading_path": "SCINeRF / 2. Related Work",
                "candidate_refs": [4],
            },
        ],
    }
    refs_pack = {
        "primary_evidence": {
            "source_path": source_path,
            "heading_path": "3. Method / 3.3. Proposed Framework",
            "snippet": "Most existing methods employ ADMM.",
            "block_id": "blk-duplicate-method",
            "selection_reason": "reader_open_alt",
            "strict_locate": True,
        }
    }

    resolved = _citation_plan_with_ref_primary(exact_plan, refs_pack)

    assert resolved["slots"] == exact_plan["slots"]
    assert resolved["slots"][0]["heading_path"] == "SCINeRF / 2. Related Work"


def test_exact_support_plan_gets_dedicated_hit_when_enriched_hit_reuses_text():
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_slot_hit_nums,
    )

    source_path = "db/SCINeRF/SCINeRF.en.md"
    evidence = "most existing methods employ ADMM [4]"
    plan_slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "source_name": "SCINeRF",
        "heading_path": "SCINeRF / 2. Related Work",
        "evidence_quote": evidence,
        "block_id": "blk-related",
        "anchor_id": "p-related",
        "strict_locate": True,
    }
    enriched_hit = {
        "text": evidence,
        "meta": {
            "source_path": source_path,
            "source_name": "SCINeRF",
            "heading_path": "SCINeRF / 2. Related Work",
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "3. Method / 3.3. Proposed Framework",
                "snippet": evidence,
                "block_id": "blk-duplicate-method",
            }
        },
    }

    hits = _augment_hits_with_system_a_plan_slots(
        [enriched_hit],
        {
            "source": "exact_support_preflight",
            "intent": "origin_lookup",
            "slots": [plan_slot],
        },
        reserved_count=1,
    )

    assert len(hits) == 2
    assert hits[1]["meta"]["citation_plan_slot"] is True
    assert hits[1]["ui_meta"]["primary_evidence"]["heading_path"] == "SCINeRF / 2. Related Work"
    assert _reading_slot_hit_nums(plan_slot, hits, [source_path]) == [2]


def test_exact_support_plan_rebinds_its_reserved_answer_citation():
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "db/PIDL/PIDL.en.md"
    exact_evidence = "The multi-source physical noise model includes crosstalk and dark count rate."
    plan_slot = {
        "preferred_system": "system_a",
        "candidate_hits": [1],
        "source_path": source_path,
        "source_name": "PIDL",
        "heading_path": "Introduction / Figure 1a",
        "evidence_quote": exact_evidence,
        "page_start": 2,
        "page_end": 2,
        "evidence_selection_reason": "spad_noise_model_exact_source",
        "strict_locate": True,
    }
    abstract_hit = {
        "text": "The abstract says that SPAD has multiple noise sources.",
        "meta": {
            "source_path": source_path,
            "source_name": "PIDL",
            "heading_path": "Abstract",
            "page_start": 1,
            "page_end": 1,
        },
        "ui_meta": {},
    }

    hits = _augment_hits_with_system_a_plan_slots(
        [abstract_hit],
        {
            "source": "exact_support_preflight",
            "intent": "evidence_lookup",
            "slots": [plan_slot],
        },
        reserved_count=1,
    )

    assert len(hits) == 1
    assert hits[0]["text"] == exact_evidence
    assert hits[0]["meta"]["heading_path"] == "Introduction / Figure 1a"
    assert hits[0]["meta"]["page_start"] == 2
    assert hits[0]["meta"]["citation_plan_evidence_authoritative"] is True
    assert hits[0]["meta"]["citation_plan_source"] == "exact_support_preflight"
    assert (
        hits[0]["meta"]["citation_plan_evidence_selection_reason"]
        == "spad_noise_model_exact_source"
    )
    assert hits[0]["ui_meta"]["primary_evidence"]["selection_reason"] == "spad_noise_model_exact_source"
    assert hits[0]["ui_meta"]["primary_evidence"]["strict_locate"] is True


def test_verified_abstract_quote_repairs_stale_nonabstract_locator(monkeypatch):
    from api import chat_render

    source_path = "db/Foveated/Foveated.en.md"
    evidence = (
        "Foveated single-pixel imaging improves the spatial resolution of the foveal region "
        "without sacrificing the field of view, unlike simple digital zoom."
    )
    monkeypatch.setattr(
        chat_render,
        "_abstract_primary_evidence_from_source",
        lambda _source_path: {
            "source_path": source_path,
            "heading_path": "Abstract",
            "snippet": evidence,
            "block_id": "blk-abstract",
            "anchor_id": "p-abstract",
            "anchor_kind": "paragraph",
            "page_start": 1,
            "page_end": 1,
        },
    )

    repaired = chat_render._citation_plan_with_verified_heading_locators(
        {
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                    "block_id": "blk-spatial-variant",
                    "anchor_id": "p-spatial-variant",
                    "page_start": 3,
                    "page_end": 3,
                }
            ]
        }
    )

    slot = repaired["slots"][0]
    assert slot["heading_path"] == "Abstract"
    assert slot["block_id"] == "blk-abstract"
    assert slot["anchor_id"] == "p-abstract"
    assert slot["page_start"] == 1
    assert slot["strict_locate"] is True


def test_scope_boundary_abstract_row_is_not_overwritten_by_later_plan_slot(monkeypatch):
    from api import chat_render

    source_path = "db/perovskite/perovskite.en.md"
    abstract_evidence = (
        "We demonstrate electrically driven lasing from a dual-cavity perovskite device."
    )
    monkeypatch.setattr(
        chat_render,
        "_abstract_primary_evidence_from_source",
        lambda _source_path: {
            "source_path": source_path,
            "heading_path": "Abstract",
            "snippet": abstract_evidence,
            "block_id": "blk-abstract",
            "anchor_id": "p-abstract",
            "anchor_kind": "paragraph",
            "page_start": 1,
            "page_end": 1,
        },
    )
    plan = {
        "intent": "scope_boundary",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Perovskite laser",
                "heading_path": "Results / Device performance",
                "evidence_quote": "The device reaches an external quantum efficiency of 82.7%.",
                "block_id": "blk-result",
                "page_start": 4,
                "strict_locate": True,
                "candidate_hits": [1],
            }
        ],
    }

    hits = chat_render._augment_hits_with_system_a_plan_slots(
        [
            {
                "text": "A generic paper overview.",
                "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
                "ui_meta": {},
            }
        ],
        plan,
        reserved_count=1,
        canonical_paths=[source_path] * 6,
    )

    assert len(hits) == 1
    assert hits[0]["text"] == abstract_evidence
    assert hits[0]["meta"]["citation_plan_scope_boundary"] is True
    assert hits[0]["meta"]["citation_plan_evidence_authoritative"] is True
    assert hits[0]["meta"]["primary_block_id"] == "blk-abstract"
    assert hits[0]["meta"]["ref_answer_citation_num"] == 1


def test_scope_boundary_repair_prefers_direct_abstract_over_reference_list_slot():
    from api.chat_render import _reading_guide_repair_scope_boundary_citation

    source_path = "db/perovskite/perovskite.en.md"
    distractor = (
        "Deschler et al. reported optically pumped lasing in solution-processed "
        "perovskite semiconductors and other dual-cavity studies."
    )
    direct = (
        "In this work, we demonstrate electrically driven lasing from a dual-cavity "
        "perovskite device with vertically stacked sub-units."
    )
    plan = {
        "intent": "scope_boundary",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Conclusion",
                "evidence_quote": distractor,
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": direct,
                "candidate_hits": [1],
            },
        ],
    }
    hits = [
        {
            "text": direct,
            "meta": {
                "source_path": source_path,
                "ref_answer_citation_num": 1,
                "citation_plan_scope_boundary": True,
            },
        },
        *({"text": "", "meta": {}} for _idx in range(5)),
        {
            "text": direct,
            "meta": {
                "source_path": source_path,
                "heading_path": "Abstract",
                "citation_plan_slot": True,
                "ref_answer_citation_num": 7,
            },
        },
    ]
    answer = (
        "相关性不大：它研究的是钙钛矿激光器件，而不是单像素成像主线。\n\n"
        "其双腔结构实现了电驱动激射 [1]。"
    )

    repaired = _reading_guide_repair_scope_boundary_citation(
        answer,
        hits,
        plan,
        canonical_paths=[source_path] * 6,
    )

    assert "原文摘要表明" in repaired
    assert "不是单像素成像方法 [1]" in repaired
    assert "[7]" not in repaired


def test_scope_boundary_repair_uses_visible_same_source_candidate_when_flag_moved():
    from api.chat_render import _reading_guide_repair_scope_boundary_citation

    source_path = "db/perovskite/perovskite.en.md"
    direct = (
        "In this work, we demonstrate electrically driven lasing from a dual-cavity "
        "perovskite device with vertically stacked sub-units."
    )
    plan = {
        "intent": "scope_boundary",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": direct,
                "candidate_hits": [1],
            }
        ],
    }
    hits = [
        {
            "text": "The device reaches a minimum threshold of 92 A cm-2.",
            "meta": {
                "source_path": source_path,
                "ref_answer_citation_num": 1,
            },
        },
        *({"text": "", "meta": {}} for _idx in range(4)),
        {
            "text": direct,
            "meta": {
                "source_path": source_path,
                "citation_plan_scope_boundary": True,
                "ref_answer_citation_num": 1,
            },
        },
        {
            "text": direct,
            "meta": {
                "source_path": source_path,
                "heading_path": "Abstract",
                "citation_plan_slot": True,
                "ref_answer_citation_num": 7,
            },
        },
    ]
    answer = (
        "相关性不大：它研究的是钙钛矿激光器件，而不是单像素成像主线。\n\n"
        "其双腔结构实现了电驱动激射 [1]。"
    )

    repaired = _reading_guide_repair_scope_boundary_citation(
        answer,
        hits,
        plan,
        canonical_paths=[source_path] * 6,
    )

    assert "不是单像素成像方法 [1]" in repaired
    assert "[7]" not in repaired


def test_exact_support_citation_backfill_does_not_replace_locked_passage():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "db/PIDL/PIDL.en.md"
    exact_evidence = "The multi-source physical noise model includes crosstalk and dark count rate."
    details = [
        {
            "num": 1,
            "citation_route": "system_a",
            "routing_reason": "exact_support_preflight",
            "evidence_source": "exact_support_preflight",
            "source_path": source_path,
            "source_name": "PIDL.pdf",
            "heading_path": "Introduction / Figure 1a",
            "evidence_quote": exact_evidence,
            "raw": exact_evidence,
            "page_start": 2,
            "page_end": 2,
            "selection_reason": "exact_support_preflight",
            "strict_locate": True,
        }
    ]
    ref_pack = {
        "primary_evidence": {
            "source_path": source_path,
            "heading_path": "Abstract",
            "snippet": "A broad abstract sentence about SPAD.",
            "page_start": 1,
            "page_end": 1,
            "selection_reason": "answer_aligned_block",
            "strict_locate": True,
        }
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert out[0]["heading_path"] == "Introduction / Figure 1a"
    assert out[0]["page_start"] == 2
    assert out[0]["evidence_quote"] == exact_evidence
    assert out[0]["routing_reason"] == "exact_support_preflight"


def test_microscopy_direct_citation_backfill_keeps_claim_matching_passage():
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "db/s2ism/s2ism.en.md"
    exact_evidence = (
        "We reconstruct an image with digital and optical super-resolution, high signal-to-noise "
        "ratio and enhanced optical sectioning."
    )
    details = [
        {
            "num": 1,
            "citation_route": "system_a",
            "routing_reason": "retrieval_hit",
            "source_path": source_path,
            "source_name": "Structured detection for s2ISM.pdf",
            "heading_path": "Abstract",
            "answer_claim": (
                "s2ISM structured detection simultaneously provides super-resolution and optical "
                "sectioning."
            ),
            "evidence_quote": exact_evidence,
            "raw": exact_evidence,
            "page_start": 1,
            "page_end": 1,
            "selection_reason": "microscopy_direct",
            "strict_locate": True,
        }
    ]
    ref_pack = {
        "hits": [
            {
                "text": (
                    "Fast detector arrays improve signal-to-noise ratio. Current approaches do not "
                    "provide optical sectioning in thick samples."
                ),
                "meta": {"source_path": source_path, "heading_path": "Abstract", "page_start": 1},
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack)

    assert out[0]["evidence_quote"] == exact_evidence
    assert "super-resolution" in str(out[0].get("card_evidence") or exact_evidence)
    assert "Current approaches do not provide" not in str(out[0].get("evidence_quote") or "")


def test_canonical_answer_repair_preserves_authoritative_exact_support_hit():
    from api.chat_render import _augment_hits_with_canonical_answer_citations

    source_path = "db/PIDL/PIDL.en.md"
    evidence = "The multi-source physical noise model includes crosstalk and dark count rate."
    exact_hit = {
        "text": evidence,
        "meta": {
            "source_path": source_path,
            "source_name": "PIDL",
            "heading_path": "Introduction / Figure 1a",
            "page_start": 2,
            "page_end": 2,
            "ref_answer_citation_num": 1,
            "citation_plan_slot": True,
            "citation_plan_evidence_authoritative": True,
            "citation_plan_source": "exact_support_preflight",
        },
        "ui_meta": {
            "primary_evidence": {
                "source_path": source_path,
                "heading_path": "Introduction / Figure 1a",
                "snippet": evidence,
                "page_start": 2,
                "page_end": 2,
                "selection_reason": "spad_noise_model_exact_source",
                "strict_locate": True,
            }
        },
    }

    out = _augment_hits_with_canonical_answer_citations(
        [exact_hit],
        canonical_paths=[source_path],
        answer_text="SPAD needs a multi-source physical noise model [1].",
    )

    assert len(out) == 1
    assert out[0]["text"] == evidence
    assert out[0]["meta"]["heading_path"] == "Introduction / Figure 1a"
    assert out[0]["meta"]["page_start"] == 2
    assert out[0]["meta"]["citation_plan_source"] == "exact_support_preflight"


def test_system_a_backfill_keeps_spad_exact_compound_evidence() -> None:
    from api.chat_render import _backfill_system_a_cite_details_from_ref_pack

    source_path = "db/PIDL/PIDL.en.md"
    exact_evidence = (
        "The multi-source physical noise model includes shot noise, fixed-pattern noise, "
        "dark count rate, afterpulsing, crosstalk, and deadtime noise.\n\n"
        "Single-source Poisson and Gaussian noise statistics lead to degraded imaging quality."
    )
    details = [
        {
            "num": 1,
            "anchor": "kb-cite-pidl-1",
            "citation_route": "system_a",
            "source_path": source_path,
            "source_name": "PIDL.pdf",
            "heading_path": "Introduction / Figure 1a",
            "answer_claim": "单源泊松统计遗漏多源噪声，并会导致成像质量退化。",
            "evidence_quote": exact_evidence,
            "raw": exact_evidence,
            "page_start": 2,
            "page_end": 2,
            "selection_reason": "spad_noise_model_exact_source",
            "strict_locate": True,
        }
    ]
    ref_pack = {
        "hits": [
            {
                "text": "Only a broad background sentence about single-photon imaging.",
                "meta": {"source_path": source_path, "heading_path": "Introduction"},
            }
        ]
    }

    out = _backfill_system_a_cite_details_from_ref_pack(details, ref_pack, render_locale="zh")

    assert " ".join(out[0]["evidence_quote"].split()) == " ".join(exact_evidence.split())
    assert "degraded imaging quality" in out[0]["card_evidence"]
    assert "deadtime noise" in out[0]["card_evidence"]


def test_scigs_comparison_abstract_repair_keeps_appended_citation_numbers(monkeypatch):
    from api.chat_render import (
        _annotate_inpaper_citations_with_hover_meta,
        _reading_guide_repair_scigs_scinerf_comparison_evidence,
    )

    scigs_path = "db/SCIGS/SCIGS.en.md"
    scinerf_path = "db/SCINeRF/SCINeRF.en.md"

    def abstract_primary(_pack, detail):
        source_path = str(detail.get("source_path") or "")
        if "SCIGS" in source_path:
            snippet = "SCIGS reconstructs an explicit 3D scene and extends it to dynamic 3D scenes."
        else:
            snippet = "We formulate the physical imaging process of SCI as part of training NeRF."
        return {
            "source_path": source_path,
            "heading_path": "Abstract",
            "snippet": snippet,
            "block_id": f"blk-{source_path}",
            "anchor_id": f"p-{source_path}",
            "anchor_kind": "paragraph",
            "strict_locate": True,
        }

    monkeypatch.setattr("api.chat_render._claim_aligned_abstract_primary_evidence", abstract_primary)
    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    monkeypatch.setattr("ui.refs_renderer._load_reference_index_cached", lambda: {})
    hits = [
        {
            "text": "SCIGS experiment table.",
            "meta": {
                "source_path": scigs_path,
                "source_name": "SCIGS",
                "heading_path": "4. Experiments",
                "ref_answer_citation_num": 1,
            },
        },
        {
            "text": "SCINeRF experiment table.",
            "meta": {
                "source_path": scinerf_path,
                "source_name": "SCINeRF",
                "heading_path": "4. Experiments",
                "ref_answer_citation_num": 2,
            },
        },
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {"preferred_system": "system_a", "source_path": scigs_path, "source_name": "SCIGS"},
            {"preferred_system": "system_a", "source_path": scinerf_path, "source_name": "SCINeRF"},
        ],
    }

    repaired = _reading_guide_repair_scigs_scinerf_comparison_evidence(
        "SCIGS 与 SCINeRF 的核心区别。\n\nSSIM 0.9137，且推理速度更快。",
        hits,
        plan,
    )
    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        hits,
        canonical_paths=[scigs_path, scinerf_path],
        citation_plan=plan,
    )

    assert hits[2]["meta"]["ref_answer_citation_num"] == 3
    assert hits[3]["meta"]["ref_answer_citation_num"] == 4
    assert "[3](#" in rendered
    assert "[4](#" in rendered
    assert "SSIM 0.9137" not in repaired
    assert "推理速度更快" not in repaired
    assert "不能仅凭这两段摘要下结论" in repaired
    assert [detail["heading_path"] for detail in details] == ["Abstract", "Abstract"]


def test_final_answer_alignment_runs_before_citation_repair(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    source_path = tmp_path / "three-d-video.en.md"
    source_path.write_text(
        "\n".join(
            [
                "<!-- kb_page: 2 -->",
                "## Abstract",
                (
                    "Performing high-speed structured illumination and sensing reflected light with four "
                    "spatially-separated, single-pixel detectors, our system reconstructs real-time 3D video "
                    "at 8 frames per second for image resolutions of 64 by 64 pixels."
                ),
                "<!-- kb_page: 5 -->",
                "## Methods",
                "Hadamard patterns are projected by the spatial light modulator.",
            ]
        ),
        encoding="utf-8",
    )
    answer = (
        "The system uses four spatially-separated single-pixel detectors and reconstructs real-time 3D video "
        "at 8 frames per second for 64 by 64 pixels."
    )
    messages = [
        {"id": 1, "role": "user", "content": "How many detectors are used, and what is the speed?"},
        {
            "id": 2,
            "role": "assistant",
            "content": answer,
            "meta": {
                "answer_quality": {
                    "output_mode": "reading_guide",
                    "citation_plan": {"budget": {"system_a": 1, "system_b": 0}, "slots": []},
                }
            },
        },
    ]
    refs_by_user = {
        1: {
            "prompt": messages[0]["content"],
            "hits": [
                {
                    "text": "Hadamard patterns are projected by the spatial light modulator.",
                    "meta": {"source_path": str(source_path), "heading_path": "Methods"},
                    "ui_meta": {
                        "source_path": str(source_path),
                        "display_name": "3D single-pixel video",
                        "heading_path": "Methods",
                    },
                }
            ],
        }
    }

    from api.chat_render import (
        _answer_aligned_reference_render_pack,
        _augment_hits_with_system_a_plan_slots,
        _citation_plan_with_ref_primary,
        _reading_guide_repair_missing_system_a_citations,
    )

    aligned_pack = _answer_aligned_reference_render_pack(refs_by_user[1], answer)
    assert aligned_pack["primary_evidence"]["heading_path"] == "Abstract"
    plan = _citation_plan_with_ref_primary(messages[1]["meta"]["answer_quality"]["citation_plan"], aligned_pack)
    assert plan["slots"][0]["heading_path"] == "Abstract"
    citation_hits = _augment_hits_with_system_a_plan_slots(aligned_pack["hits"], plan)
    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        citation_hits,
        plan,
        output_mode="reading_guide",
    )
    assert repaired != answer

    rendered = enrich_messages_with_reference_render(messages, refs_by_user, conv_id="answer-align")

    details = rendered[-1]["cite_details"]
    assert len(details) == 1
    assert details[0]["heading_path"] == "Abstract"
    assert details[0]["page_start"] == 2
    assert "four spatially-separated" in details[0]["evidence_quote"]
    assert "8 frames per second" in details[0]["evidence_quote"]


def test_system_a_binding_bridges_chinese_spad_noise_claim_to_english_evidence():
    from ui.refs_renderer import _assess_system_a_hit_binding

    evidence = (
        "The multi-source physical noise model of SPAD arrays consists of dark count rate, "
        "afterpulsing and crosstalk noise."
    )
    binding = _assess_system_a_hit_binding(
        answer_claim="后脉冲和串扰噪声会产生额外的虚假事件。",
        hit={"text": evidence},
        meta={},
        heading="Introduction / Figure 1",
        evidence_quote=evidence,
        source_name="High-resolution single-photon imaging with physics-informed deep learning",
    )

    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False
    assert "crosstalk noise" in binding["overlap_terms"]


def test_lineage_plan_replaces_each_system_a_slot_with_exact_source_evidence(tmp_path: Path) -> None:
    from api.chat_render import _citation_plan_with_exact_lineage_evidence

    cassi_path = tmp_path / "dual-disperser-cassi.en.md"
    scinerf_path = tmp_path / "SCINeRF.en.md"
    scigs_path = tmp_path / "SCIGS.en.md"
    cassi_path.write_text(
        "\n".join(
            [
                "# CASSI",
                "<!-- kb_page: 1 -->",
                "## Abstract",
                (
                    "The coded snapshot spectral imager uses two dispersive elements "
                    "and a binary-valued aperture to encode a hyperspectral cube."
                ),
            ]
        ),
        encoding="utf-8",
    )
    scinerf_path.write_text(
        "\n".join(
            [
                "# SCINeRF",
                "<!-- kb_page: 2 -->",
                "## Abstract",
                (
                    "SCINeRF incorporates the physical imaging process of snapshot "
                    "compressive imaging into NeRF reconstruction."
                ),
            ]
        ),
        encoding="utf-8",
    )
    scigs_path.write_text(
        "\n".join(
            [
                "# SCIGS",
                "<!-- kb_page: 3 -->",
                "## Abstract",
                "SCIGS reconstructs a dynamic 3D scene from snapshot measurements.",
            ]
        ),
        encoding="utf-8",
    )
    original_slots = [
        {
            "preferred_system": "system_a",
            "source_path": str(path),
            "heading_path": "Weak section",
            "evidence_quote": "Weak lineage evidence.",
        }
        for path in (cassi_path, scinerf_path, scigs_path)
    ]

    repaired = _citation_plan_with_exact_lineage_evidence(
        {
            "intent": "origin_lookup",
            "budget": {"system_a": 3, "system_b": 0},
            "slots": original_slots,
        }
    )

    repaired_slots = repaired["slots"]
    assert len(repaired_slots) == 3
    assert "two dispersive elements" in repaired_slots[0]["evidence_quote"]
    assert "binary-valued aperture" in repaired_slots[0]["evidence_quote"]
    assert "physical imaging process" in repaired_slots[1]["evidence_quote"]
    assert "NeRF" in repaired_slots[1]["evidence_quote"]
    assert "dynamic 3D scene" in repaired_slots[2]["evidence_quote"]
    assert {slot["page_start"] for slot in repaired_slots} == {1, 2, 3}
    assert all(slot["strict_locate"] is True for slot in repaired_slots)
    assert all(
        slot["evidence_selection_reason"] == "lineage_exact_source_block"
        for slot in repaired_slots
    )
    assert all(slot["evidence_quote"] == "Weak lineage evidence." for slot in original_slots)


def test_lineage_plan_prefers_scigs_3dgs_mechanism_over_broad_dynamic_abstract(
    tmp_path: Path,
) -> None:
    from api.chat_render import _citation_plan_with_exact_lineage_evidence
    from ui.refs_renderer import _assess_system_a_hit_binding

    cassi_path = tmp_path / "dual-disperser-cassi.en.md"
    scinerf_path = tmp_path / "SCINeRF.en.md"
    scigs_path = tmp_path / "SCIGS.en.md"
    cassi_path.write_text(
        "## Abstract\nTwo dispersive elements surround a binary-valued aperture.",
        encoding="utf-8",
    )
    scinerf_path.write_text(
        "## Abstract\nSCINeRF puts the physical imaging process into NeRF training.",
        encoding="utf-8",
    )
    scigs_path.write_text(
        "\n".join(
            [
                "## Abstract",
                "SCIGS reconstructs a dynamic 3D scene from snapshot measurements.",
                "## 3. Method",
                (
                    "SCIGS is a variant of 3DGS with a transformation network. "
                    "The method reconstructs an explicit scene from a single compressed image."
                ),
            ]
        ),
        encoding="utf-8",
    )
    slots = [
        {
            "preferred_system": "system_a",
            "source_path": str(path),
            "source_name": path.stem,
            "heading_path": "Weak section",
            "evidence_quote": "Weak lineage evidence.",
        }
        for path in (cassi_path, scinerf_path, scigs_path)
    ]

    repaired = _citation_plan_with_exact_lineage_evidence(
        {
            "intent": "origin_lookup",
            "budget": {"system_a": 3, "system_b": 0},
            "slots": slots,
        }
    )
    scigs_slot = repaired["slots"][2]
    evidence = scigs_slot["evidence_quote"]
    binding = _assess_system_a_hit_binding(
        answer_claim="SCIGS 将方法迁移到 3D 高斯泼溅（3DGS），用显式高斯替代 NeRF。",
        hit={"text": evidence},
        meta={},
        heading=scigs_slot["heading_path"],
        evidence_quote=evidence,
        source_name="SCIGS",
    )

    assert "3DGS" in evidence
    assert "transformation network" in evidence
    assert "single compressed image" in evidence
    assert binding["status"] == "grounded"
    assert binding["suppress_link"] is False


def test_lineage_canonical_paths_bind_all_three_reserved_system_a_hits(tmp_path: Path) -> None:
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_paths = [
        str(tmp_path / "CASSI" / "CASSI.en.md"),
        str(tmp_path / "SCINeRF" / "SCINeRF.en.md"),
        str(tmp_path / "SCIGS" / "SCIGS.en.md"),
    ]
    evidence = [
        "CASSI uses two dispersive elements and a binary-valued aperture.",
        "SCINeRF embeds the physical imaging process into NeRF.",
        "SCIGS reconstructs a dynamic 3D scene.",
    ]
    plan = {
        "intent": "origin_lookup",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": Path(source_path).name,
                "heading_path": "Abstract",
                "evidence_quote": quote,
                "block_id": f"block-{idx}",
                "anchor_id": f"anchor-{idx}",
                "strict_locate": True,
            }
            for idx, (source_path, quote) in enumerate(
                zip(source_paths, evidence),
                start=1,
            )
        ],
    }
    hits = [
        {
            "text": f"Stale retrieval passage {idx}.",
            "meta": {
                "source_path": f"kb-source/0/stale-{idx}/stale-{idx}.en.md",
                "citation_plan_padding": True,
            },
            "ui_meta": {},
        }
        for idx in range(1, 4)
    ]

    rebound = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=3,
        canonical_paths=source_paths,
    )

    assert len(rebound) == 3
    for idx, hit in enumerate(rebound, start=1):
        assert hit["text"] == evidence[idx - 1]
        assert hit["meta"]["source_path"] == source_paths[idx - 1]
        assert hit["meta"]["ref_answer_citation_num"] == idx
        assert hit["meta"]["citation_plan_slot"] is True
        assert "citation_plan_padding" not in hit["meta"]
        assert hit["ui_meta"]["primary_evidence"]["block_id"] == f"block-{idx}"


def test_prompt_aligned_rebind_discards_previous_source_card_fields(tmp_path: Path) -> None:
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = str(tmp_path / "FDM" / "FDM.en.md")
    evidence = (
        "Multiple frequency-division masks are projected simultaneously "
        "without increasing integration time."
    )
    stale_3d = "Four detectors reconstruct the 3D object at eight frames per second."
    hits = [
        {
            "text": "stale retrieval text",
            "meta": {
                "source_path": source_path,
                "ref_answer_citation_num": 1,
                "ref_snippets": [stale_3d],
                "ref_show_snippets": [stale_3d],
                "ref_locs": [{"snippet": stale_3d}],
                "citation_meta": {
                    "title": "Single-pixel 3D imaging",
                    "doi": "10.1000/3d",
                },
            },
            "ui_meta": {
                "source_path": source_path,
                "citation_meta": {
                    "title": "Single-pixel 3D imaging",
                    "doi": "10.1000/3d",
                },
                "reader_open": {
                    "sourcePath": "3d.en.md",
                    "evidenceAlternatives": [{"snippet": stale_3d}],
                },
                "summary_line": "Stale 3D summary.",
                "summary_generation": "deterministic_grounded",
                "why_line": "Stale 3D relevance.",
                "why_generation": "deterministic_grounded",
                "card_view": {"sections": [{"id": "guide", "body": "stale"}]},
            },
        }
    ]
    plan = {
        "source": "generation_citation_planner",
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "source_name": "FDM.pdf",
                "heading_path": "Principle",
                "evidence_quote": evidence,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
                "block_id": "blk-fdm",
                "anchor_id": "p-fdm",
                "anchor_kind": "sentence",
                "page_start": 3,
                "strict_locate": True,
            }
        ],
    }

    rebound = _augment_hits_with_system_a_plan_slots(hits, plan)
    hit = rebound[0]

    assert hit["text"] == evidence
    assert "ref_snippets" not in hit["meta"]
    assert "ref_show_snippets" not in hit["meta"]
    assert "ref_locs" not in hit["meta"]
    assert "citation_meta" not in hit["meta"]
    assert "citation_meta" not in hit["ui_meta"]
    assert hit["ui_meta"]["summary_line"] == evidence
    assert hit["ui_meta"].get("summary_generation") != "deterministic_grounded"
    assert "why_line" not in hit["ui_meta"]
    assert "why_generation" not in hit["ui_meta"]
    assert "card_view" not in hit["ui_meta"]
    assert hit["ui_meta"]["reader_open"]["sourcePath"] == source_path
    assert hit["ui_meta"]["reader_open"]["evidenceAlternatives"] == [
        {
            "headingPath": "Principle",
            "snippet": evidence,
            "highlightSnippet": evidence,
            "blockId": "blk-fdm",
            "anchorId": "p-fdm",
            "anchorKind": "sentence",
            "pageStart": 3,
            "pageEnd": 3,
        }
    ]
    assert stale_3d not in str(hit)


def test_prompt_aligned_rebind_keeps_same_source_answer_passage_as_alternative(
    tmp_path: Path,
) -> None:
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = str(tmp_path / "FDM" / "FDM.en.md")
    mechanism = (
        "The SLM pixels use BPSK at four carrier frequencies and the lock-in "
        "amplifier demodulates the corresponding mask measurements in parallel."
    )
    abstract = (
        "Frequency-division methods parallelize the single-pixel imaging process "
        "without altering detector integration time."
    )
    hits = [
        {
            "text": mechanism,
            "meta": {
                "source_path": source_path,
                "heading_path": "Encoding",
                "block_id": "blk-encoding",
                "anchor_id": "p-encoding",
                "page_start": 3,
            },
            "ui_meta": {"source_path": source_path},
        }
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "source_name": "FDM.pdf",
                "heading_path": "Abstract",
                "evidence_quote": abstract,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
            }
        ]
    }

    rebound = _augment_hits_with_system_a_plan_slots(hits, plan)
    alternatives = rebound[0]["ui_meta"]["reader_open"]["evidenceAlternatives"]

    assert any(item.get("snippet") == abstract for item in alternatives)
    assert any(item.get("snippet") == mechanism for item in alternatives)
    assert any(item.get("blockId") == "blk-encoding" for item in alternatives)


def test_unbound_prompt_aligned_slot_does_not_steal_reserved_same_source_occurrence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _refine_system_a_cite_evidence_from_citation_plan,
    )
    from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta

    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    monkeypatch.setattr("ui.refs_renderer._load_reference_index_cached", lambda: {})

    source_path = str(tmp_path / "LPR" / "LPR.en.md")
    biographies = (
        "Kai Song received his B.S. degree in 2019 and M.S. degree in 2022. "
        "Yaoxing Bian is currently a lecturer. Liantuan Xiao is currently a "
        "Changjiang professor."
    )
    unrelated = (
        "Thanks to the unsupervised learning mode, the model-driven "
        "reconstruction algorithm can be adapted to diverse imaging scenes."
    )
    hits = [
        {
            "text": biographies,
            "meta": {
                "source_path": source_path,
                "heading_path": "Author Biographies",
                "ref_answer_citation_num": 1,
                "page_start": 21,
                "page_end": 21,
            },
            "ui_meta": {
                "source_path": source_path,
                "primary_evidence": {
                    "source_path": source_path,
                    "heading_path": "Author Biographies",
                    "snippet": biographies,
                    "page_start": 21,
                    "page_end": 21,
                },
            },
        }
    ]
    plan = {
        "source": "generation_citation_planner",
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [],
                "source_path": source_path,
                "heading_path": "4.1 Strategy",
                "evidence_quote": unrelated,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
                "page_start": 8,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "heading_path": "Author Biographies",
                "evidence_quote": biographies,
                "page_start": 21,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": source_path,
                "heading_path": "4.1 Strategy",
                "evidence_quote": unrelated,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
                "page_start": 8,
            },
        ],
    }

    rebound = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=4,
        canonical_paths=[source_path] * 4,
        answer_text=(
            "Kai Song completed his degrees in 2019 and 2022. Yaoxing Bian is "
            "a lecturer. Liantuan Xiao is a Changjiang professor."
        ),
    )

    assert rebound[0]["text"] == biographies
    assert rebound[0]["meta"]["heading_path"] == "Author Biographies"
    assert rebound[0]["meta"]["page_start"] == 21
    assert rebound[1]["text"] == unrelated
    assert rebound[2]["text"] == unrelated

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        "Liantuan Xiao earned his degrees in 1989, 1997, and 2001 [1].",
        rebound,
        anchor_ns="author-biographies-occurrence",
        canonical_paths=[source_path] * 4,
        citation_plan=plan,
        render_locale="zh",
    )

    assert "[1](#" in rendered
    assert len(details) == 1
    assert details[0]["heading_path"] == "Author Biographies"
    assert details[0]["page_start"] == 21
    assert "Liantuan Xiao" in details[0]["evidence_quote"]

    refined = _refine_system_a_cite_evidence_from_citation_plan(
        details,
        plan,
        render_locale="zh",
    )
    assert refined[0]["heading_path"] == "Author Biographies"
    assert refined[0]["page_start"] == 21
    assert "Liantuan Xiao" in refined[0]["evidence_quote"]
    assert "unsupervised learning" not in refined[0]["evidence_quote"]


def test_per_author_profiles_reuse_same_grounded_source_marker_for_each_entity() -> None:
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    source_path = "db/LPR/LPR-2025-Advances-and-Challenges.en.md"
    evidence = (
        "Kai Song received his B.S. and M.S. degrees and is pursuing his Ph.D. "
        "Yaoxing Bian received his Ph.D. degree and is currently a lecturer. "
        "Liantuan Xiao received his B.S., M.S., and Ph.D. degrees and is currently "
        "a Changjiang professor."
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_path": source_path,
                "heading_path": "Author Biographies",
                "page_start": 21,
                "page_end": 21,
                "ref_answer_citation_num": 1,
            },
            "ui_meta": {"source_path": source_path},
        }
    ]
    plan = {
        "intent": "beginner_overview",
        "coverage_mode": "per_entity",
        "coverage_entity_type": "author_profile",
        "coverage_target_count": 3,
        "coverage_targets": ["Kai Song", "Yaoxing Bian", "Liantuan Xiao"],
        "budget": {"system_a": 3, "system_b": 0},
        "per_paragraph_budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "source_name": "LPR.pdf",
                "heading_path": "Author Biographies",
                "evidence_quote": evidence,
                "page_start": 21,
                "page_end": 21,
            }
        ],
    }
    answer = """### Kai Song

- Education: B.S. and M.S.; currently pursuing a Ph.D. [1]

### Yaoxing Bian

- Education: Ph.D.; current position: lecturer. [1]

### Liantuan Xiao

- Education: B.S., M.S., and Ph.D.; current position: Changjiang professor.
"""

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=[source_path],
    )

    assert repaired.count("[1]") == 3
    assert re.search(r"### Liantuan Xiao.*?\[1\]", repaired, flags=re.DOTALL)


def test_per_author_profiles_keep_distinct_same_paper_occurrence_links(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from api.chat_render import (
        _augment_hits_with_system_a_plan_slots,
        _reading_guide_repair_missing_system_a_citations,
        _refine_system_a_cite_evidence_from_citation_plan,
    )
    from ui.refs_renderer import _annotate_inpaper_citations_with_hover_meta

    monkeypatch.setattr("ui.refs_renderer._is_temp_source_path", lambda _path: False)
    monkeypatch.setattr("ui.refs_renderer._load_reference_index_cached", lambda: {})

    source_path = str(tmp_path / "LPR" / "LPR.en.md")
    profiles = [
        (
            "Kai Song",
            "Kai Song completed undergraduate and master degrees and is now pursuing "
            "a doctorate in single-pixel imaging.",
            "bio-kai",
            "sent-kai",
        ),
        (
            "Yaoxing Bian",
            "Yaoxing Bian completed a doctorate and is currently a lecturer studying "
            "random lasers and single-pixel imaging.",
            "bio-yaoxing",
            "sent-yaoxing",
        ),
        (
            "Liantuan Xiao",
            "Liantuan Xiao completed three degrees and is currently a Changjiang "
            "professor studying laser spectroscopy.",
            "bio-liantuan",
            "sent-liantuan",
        ),
    ]
    hits: list[dict] = []
    slots: list[dict] = []
    for num, (name, evidence, block_id, anchor_id) in enumerate(profiles, start=1):
        primary = {
            "source_path": source_path,
            "source_name": "LPR.pdf",
            "heading_path": "Author Biography",
            "snippet": evidence,
            "highlight_snippet": evidence,
            "block_id": block_id,
            "anchor_id": anchor_id,
            "anchor_kind": "paragraph",
            "page_start": 21,
            "page_end": 21,
            "strict_locate": True,
        }
        hits.append(
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "source_name": "LPR.pdf",
                    "heading_path": "Author Biography",
                    "ref_answer_citation_num": num,
                    "primary_block_id": block_id,
                    "primary_anchor_id": anchor_id,
                    "anchor_kind": "paragraph",
                    "page_start": 21,
                    "page_end": 21,
                },
                "ui_meta": {
                    "source_path": source_path,
                    "display_name": "LPR.pdf",
                    "heading_path": "Author Biography",
                    "summary_line": evidence,
                    "primary_evidence": primary,
                },
            }
        )
        slots.append(
            {
                "preferred_system": "system_a",
                "candidate_hits": [num],
                "coverage_target": name,
                "source_path": source_path,
                "source_name": "LPR.pdf",
                "heading_path": "Author Biography",
                "evidence_quote": evidence,
                "block_id": block_id,
                "anchor_id": anchor_id,
                "anchor_kind": "paragraph",
                "page_start": 21,
                "page_end": 21,
                "strict_locate": True,
            }
        )
    plan = {
        "coverage_mode": "per_entity",
        "coverage_entity_type": "author_profile",
        "coverage_target_count": 3,
        "coverage_targets": [profile[0] for profile in profiles],
        "budget": {"system_a": 3, "system_b": 0},
        "per_paragraph_budget": {"system_a": 3, "system_b": 0},
        "slots": slots,
    }
    answer = """### Kai Song
- Original evidence: `Kai Song completed undergraduate and master degrees and is now pursuing a doctorate in single-pixel imaging.` [1]

### Yaoxing Bian
- Original evidence: `Yaoxing Bian completed a doctorate and is currently a lecturer studying random lasers and single-pixel imaging.` [1][2]

### Liantuan Xiao
- Original evidence: `Liantuan Xiao completed three degrees and is currently a Changjiang professor studying laser spectroscopy.` [1][3]
"""
    canonical_paths = [source_path] * 3
    citation_hits = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        reserved_count=3,
        canonical_paths=canonical_paths,
        answer_text=answer,
    )
    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        citation_hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=canonical_paths,
    )

    assert re.findall(r"(?<![!\\])\[(\d+)\](?!\()", repaired) == ["1", "2", "3"]
    assert "[1][2]" not in repaired
    assert "[1][3]" not in repaired

    rendered, details = _annotate_inpaper_citations_with_hover_meta(
        repaired,
        citation_hits,
        anchor_ns="author-biography-occurrences",
        canonical_paths=canonical_paths,
        citation_plan=plan,
        render_locale="en",
    )
    rendered, details, registry = remap_system_a_citations_for_display(rendered, details)
    rendered = _collapse_adjacent_same_citation_links(rendered)
    refined = _refine_system_a_cite_evidence_from_citation_plan(
        details,
        plan,
        render_locale="en",
    )

    links = re.findall(r"\[1\]\(#([^\s)]+)", rendered)
    assert len(links) == 3
    assert len(set(links)) == 3
    assert "[2](#" not in rendered
    assert "[3](#" not in rendered
    assert "[]" not in rendered
    assert not re.search(r"\)\s*\[1\]\(#", rendered)
    assert len(registry) == 1
    assert registry[0]["display_num"] == 1
    assert registry[0]["original_nums"] == [1, 2, 3]

    details_by_anchor = {str(detail.get("anchor") or ""): detail for detail in refined}
    for link, (name, _evidence, block_id, anchor_id), answer_hit_num in zip(
        links,
        profiles,
        (1, 2, 3),
    ):
        detail = details_by_anchor[link]
        assert detail["num"] == 1
        assert detail["answer_hit_num"] == answer_hit_num
        assert detail["block_id"] == block_id
        assert detail["anchor_id"] == anchor_id
        assert name in detail["evidence_quote"]
        assert name in detail["card_evidence"]


def test_per_author_citation_repair_does_not_mark_trailing_summary_or_inference() -> None:
    from api.chat_render import _reading_guide_repair_per_entity_system_a_citations

    source_path = "db/LPR/LPR-2025-Advances-and-Challenges.en.md"
    evidence = (
        "Kai Song received his degrees. Yaoxing Bian is currently a lecturer. "
        "Liantuan Xiao is currently a Changjiang professor."
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_path": source_path,
                "heading_path": "Author Biographies",
                "ref_answer_citation_num": 1,
            },
        }
    ]
    plan = {
        "coverage_mode": "per_entity",
        "coverage_entity_type": "author_profile",
        "coverage_target_count": 3,
        "coverage_targets": ["Kai Song", "Yaoxing Bian", "Liantuan Xiao"],
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "heading_path": "Author Biographies",
                "evidence_quote": evidence,
            }
        ],
    }
    compact_answer = """- Kai Song: degree holder.
- Yaoxing Bian: current position is lecturer.
- Liantuan Xiao: current position is Changjiang professor.

Overall, this proves the team is internationally dominant.
"""

    repaired_compact = _reading_guide_repair_per_entity_system_a_citations(
        compact_answer,
        hits,
        plan,
        canonical_paths=[source_path],
    )

    assert repaired_compact.count("[1]") == 3
    assert "internationally dominant [1]" not in repaired_compact

    headed_answer = """Kai Song
- Current position: Ph.D. student.

Yaoxing Bian
- Current position: lecturer.

Liantuan Xiao
- Current position: Changjiang professor.
- Inference: his research interests probably extend to quantum imaging.
"""
    repaired_headed = _reading_guide_repair_per_entity_system_a_citations(
        headed_answer,
        hits,
        plan,
        canonical_paths=[source_path],
    )

    assert "Changjiang professor [1]." in repaired_headed
    assert "quantum imaging [1]" not in repaired_headed


def test_per_author_citation_prefers_supported_fact_over_hallucinated_later_field() -> None:
    from api.chat_render import _reading_guide_repair_per_entity_system_a_citations

    source_path = "db/LPR/LPR-2025-Advances-and-Challenges.en.md"
    evidence = (
        "Kai Song received his B.S. degree in 2019. "
        "Yaoxing Bian is currently a lecturer."
    )
    plan = {
        "coverage_mode": "per_entity",
        "coverage_entity_type": "author_profile",
        "coverage_target_count": 2,
        "coverage_targets": ["Kai Song", "Yaoxing Bian"],
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "heading_path": "Author Biographies",
                "evidence_quote": evidence,
            }
        ],
    }
    answer = """Kai Song
- Education: B.S. degree.
- Current position: professor at Stanford University.

Yaoxing Bian
- Current position: lecturer.
"""

    repaired = _reading_guide_repair_per_entity_system_a_citations(
        answer,
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Author Biographies",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        plan,
        canonical_paths=[source_path],
    )

    assert "Education: B.S. degree [1]." in repaired
    assert "Stanford University [1]" not in repaired

    wrong_facts = """Kai Song
- Education: M.S. and Ph.D. degrees in 2018.

Yaoxing Bian
- Current position: lecturer.
"""
    repaired_wrong_facts = _reading_guide_repair_per_entity_system_a_citations(
        wrong_facts,
        [
            {
                "text": evidence,
                "meta": {
                    "source_path": source_path,
                    "heading_path": "Author Biographies",
                    "ref_answer_citation_num": 1,
                },
            }
        ],
        plan,
        canonical_paths=[source_path],
    )

    assert "M.S. and Ph.D. degrees in 2018 [1]" not in repaired_wrong_facts
    assert repaired_wrong_facts.count("[1]") == 1


def test_per_author_profile_plan_rebinds_stale_same_source_abstract_hit() -> None:
    from api.chat_render import _augment_hits_with_system_a_plan_slots

    source_path = "db/LPR/LPR-2025-Advances-and-Challenges.en.md"
    abstract = "The review discusses reconstruction quality and speed."
    biographies = (
        "Kai Song received his degrees. Yaoxing Bian is currently a lecturer. "
        "Liantuan Xiao is currently a Changjiang professor."
    )
    hits = [
        {
            "text": "Yaoxing Bian is currently a lecturer.",
            "meta": {
                "source_path": source_path,
                "heading_path": "Abstract",
                "page_start": 1,
                "page_end": 1,
                "ref_answer_citation_num": 1,
            },
            "ui_meta": {
                "source_path": source_path,
                "heading_path": "Abstract",
                "primary_evidence": {
                    "source_path": source_path,
                    "heading_path": "Abstract",
                    "snippet": "Yaoxing Bian is currently a lecturer.",
                    "page_start": 1,
                    "page_end": 1,
                    "strict_locate": True,
                    "selection_reason": "answer_citation_grounded",
                },
            },
        }
    ]
    plan = {
        "source": "citation_plan_builder",
        "coverage_mode": "per_entity",
        "coverage_entity_type": "author_profile",
        "coverage_target_count": 3,
        "coverage_targets": ["Kai Song", "Yaoxing Bian", "Liantuan Xiao"],
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [],
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": abstract,
                "evidence_selection_reason": "prompt_aligned_source_sentence",
                "page_start": 1,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "heading_path": "Author Biographies",
                "evidence_quote": biographies,
                "page_start": 21,
                "page_end": 21,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": abstract,
                "page_start": 1,
            },
        ],
    }

    rebound = _augment_hits_with_system_a_plan_slots(
        hits,
        plan,
        canonical_paths=[source_path],
        answer_text=(
            "Kai Song earned his degrees. Yaoxing Bian is a lecturer. "
            "Liantuan Xiao is a Changjiang professor."
        ),
    )

    assert rebound[0]["text"] == biographies
    assert rebound[0]["meta"]["heading_path"] == "Author Biographies"
    assert rebound[0]["meta"]["page_start"] == 21
    assert rebound[0]["ui_meta"]["primary_evidence"]["page_start"] == 21


def test_lineage_system_b_retargets_same_reference_to_downstream_paper(tmp_path: Path) -> None:
    from api.chat_render import _retarget_lineage_system_b_to_downstream_source
    from api.reference_rendering import _source_cite_id

    upstream_path = tmp_path / "SCINeRF.en.md"
    cassi_path = tmp_path / "CASSI.en.md"
    downstream_path = tmp_path / "SCIGS.en.md"
    reference = (
        "X. Yuan, D. J. Brady, and A. K. Katsaggelos. Snapshot compressive imaging: "
        "Theory, algorithms, and applications. IEEE Signal Processing Magazine, 2021."
    )
    upstream_path.write_text(
        f"# SCINeRF\n\n## References\n[50] {reference}\n",
        encoding="utf-8",
    )
    cassi_path.write_text(
        "# CASSI\n\n## Abstract\nCASSI uses a coded aperture.\n",
        encoding="utf-8",
    )
    downstream_path.write_text(
        "\n".join(
            [
                "# SCIGS",
                "## Introduction",
                (
                    "Snapshot compressive imaging theory, algorithms, and applications [42] "
                    "provide the foundation for our dynamic scene method."
                ),
                "## References",
                f"[42] {reference}",
            ]
        ),
        encoding="utf-8",
    )
    old_sid = _source_cite_id(str(upstream_path))
    new_sid = _source_cite_id(str(downstream_path))
    answer = f"The lineage starts from SCI [[CITE:{old_sid}:50]]."
    plan = {
        "intent": "origin_lookup",
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(upstream_path),
                "heading_path": "Abstract",
                "evidence_quote": "SCINeRF evidence.",
            },
            {
                "preferred_system": "system_a",
                "source_path": str(cassi_path),
                "heading_path": "Abstract",
                "evidence_quote": "CASSI evidence.",
            },
            {
                "preferred_system": "system_a",
                "source_path": str(downstream_path),
                "heading_path": "Introduction",
                "evidence_quote": "SCIGS evidence.",
            },
            {
                "preferred_system": "system_b",
                "source_path": str(upstream_path),
                "source_name": "SCINeRF.pdf",
                "topic": "Snapshot compressive imaging: Theory, algorithms, and applications",
                "sid": old_sid,
                "candidate_refs": [50],
                "candidate_cite_examples": [f"[[CITE:{old_sid}:50]]"],
            },
        ],
    }

    repaired_answer, repaired_plan = _retarget_lineage_system_b_to_downstream_source(
        answer,
        plan,
    )

    system_b = repaired_plan["slots"][-1]
    assert f"[[CITE:{old_sid}:50]]" not in repaired_answer
    assert f"[[CITE:{new_sid}:42]]" in repaired_answer
    assert system_b["source_path"] == str(downstream_path)
    assert system_b["candidate_refs"] == [42]
    assert system_b["candidate_cite_examples"] == [f"[[CITE:{new_sid}:42]]"]
    assert system_b["selection_reason"] == "downstream_duplicate_reference"
    assert system_b["grounding_contract"] == {
        "same_context_reference": True,
        "context_marker_verified": True,
        "relation_context_verified": True,
        "relation_entities": ["video_sci"],
    }
    assert "Snapshot compressive imaging theory" in system_b["evidence_quote"]

    seeded_answer, seeded_plan = _retarget_lineage_system_b_to_downstream_source(
        "Video SCI connects compressed measurements to the later 3D scene route.",
        plan,
    )

    assert seeded_answer == "Video SCI connects compressed measurements to the later 3D scene route."
    assert all(
        slot.get("preferred_system") != "system_b"
        for slot in seeded_plan["slots"]
    )


def test_lineage_system_b_drops_unsupported_spectral_origin_relation(tmp_path: Path) -> None:
    from api.chat_render import _retarget_lineage_system_b_to_downstream_source
    from api.reference_rendering import _source_cite_id

    upstream_path = tmp_path / "SCINeRF.en.md"
    cassi_path = tmp_path / "CASSI.en.md"
    downstream_path = tmp_path / "SCIGS.en.md"
    reference = (
        "X. Yuan et al. Snapshot compressive imaging: Theory, algorithms, "
        "and applications. IEEE Signal Processing Magazine, 2021."
    )
    upstream_path.write_text(
        f"# SCINeRF\n\n## References\n[50] {reference}\n",
        encoding="utf-8",
    )
    cassi_path.write_text("# CASSI\n\nCoded aperture evidence.\n", encoding="utf-8")
    downstream_path.write_text(
        "\n".join(
            [
                "# SCIGS",
                "## Introduction",
                "Compressed Sensing and video SCI [42] technology has been developed.",
                "## References",
                f"[42] {reference}",
            ]
        ),
        encoding="utf-8",
    )
    old_sid = _source_cite_id(str(upstream_path))
    token = f"[[CITE:{old_sid}:50]]"
    answer = f"This method directly originates from the spectral data-cube paradigm {token}."
    plan = {
        "intent": "origin_lookup",
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": str(path),
                "evidence_quote": "Direct paper evidence.",
            }
            for path in (upstream_path, cassi_path, downstream_path)
        ]
        + [
            {
                "preferred_system": "system_b",
                "source_path": str(upstream_path),
                "topic": "Snapshot compressive imaging: Theory, algorithms, and applications",
                "sid": old_sid,
                "candidate_refs": [50],
            }
        ],
    }

    repaired_answer, repaired_plan = _retarget_lineage_system_b_to_downstream_source(
        answer,
        plan,
    )

    assert token not in repaired_answer
    assert all(
        slot.get("preferred_system") != "system_b"
        for slot in repaired_plan["slots"]
    )
    assert repaired_plan["budget"]["system_b"] == 0


def test_named_table_locator_survives_card_recomposition() -> None:
    from api.chat_render import _normalize_system_a_named_table_locators

    details = [
        {
            "num": 1,
            "anchor": "kb-cite-table-1",
            "citation_route": "system_a",
            "source_path": "db/simple-baselines.en.md",
            "source_name": "Simple Baselines.pdf",
            "heading_path": "5 Experiments / 5.2 Applications",
            "location_label": "5 Experiments / 5.2 Applications · sentence · p. 13",
            "anchor_id": "tb_00006",
            "anchor_kind": "sentence",
            "reader_evidence_quote": (
                "Table 6. Image Denoising Results on SIDD. "
                "Baseline ours = 40.30; NAFNet ours = 40.30."
            ),
            "evidence_quote": "SIDD PSNR: Baseline ours = 40.30; NAFNet ours = 40.30.",
            "answer_claim": "The highest SIDD PSNR in Table 6 is 40.30.",
        }
    ]

    out = _normalize_system_a_named_table_locators(details, render_locale="en")

    assert out[0]["anchor_kind"] == "table"
    assert "Table 6" in out[0]["location_label"]
    assert "Table 6" in out[0]["card_locator"]
    assert out[0]["card_evidence"].startswith("Table 6.")
    assert "sentence" not in out[0]["card_locator"].lower()


def test_spi_prospects_repair_does_not_replace_foveated_answer() -> None:
    from api.chat_render import _reading_guide_repair_spi_prospects_answer

    answer = (
        "Foveated dynamic supersampling tracks a high-resolution foveal region while "
        "sampling the entire field in every frame [1]."
    )
    prospects = (
        "Images can be collected at wavelengths outside the reach of FPA technology, at "
        "high frame rates, or in three dimensions. Applications include hazardous gas "
        "leaks and autonomous vehicles."
    )
    hits = [
        {"text": answer, "meta": {"source_path": "foveated.en.md", "ref_answer_citation_num": 1}},
        {"text": prospects, "meta": {"source_path": "prospects.en.md", "ref_answer_citation_num": 2}},
    ]
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "foveated.en.md",
                "evidence_quote": answer,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "prospects.en.md",
                "evidence_quote": prospects,
                "candidate_hits": [2],
            },
        ],
    }

    repaired = _reading_guide_repair_spi_prospects_answer(
        answer,
        hits,
        plan,
        canonical_paths=["foveated.en.md", "prospects.en.md"],
    )

    assert repaired == answer


def test_beginner_numbered_roadmap_keeps_hsi_fsi_evidence_marker() -> None:
    from api.chat_render import _reading_guide_repair_beginner_roadmap_missing_paper

    paths = ["prospects.en.md", "dl-review.en.md", "hsi-fsi.en.md"]
    hits = [
        {"text": "foundation", "meta": {"source_path": paths[0], "ref_answer_citation_num": 1}},
        {"text": "deep learning", "meta": {"source_path": paths[1], "ref_answer_citation_num": 2}},
        {"text": "basis comparison", "meta": {"source_path": paths[2], "ref_answer_citation_num": 3}},
    ]
    plan = {
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": paths[0],
                "source_name": "Principles and prospects for single-pixel imaging",
                "evidence_quote": "Images are recovered by compressive sensing.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[1],
                "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "evidence_quote": "Deep learning improves reconstruction quality and speed.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[2],
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "evidence_quote": (
                    "HSI uses Hadamard basis patterns and FSI uses Fourier basis patterns, "
                    "compared in principle, imaging efficiency, and noise robustness."
                ),
                "candidate_hits": [3],
            },
        ],
    }
    answer = (
        "Use this reading order:\n\n"
        "1. **Principles and prospects for single-pixel imaging** [1]\nFoundation.\n\n"
        "2. **Advances and Challenges of Single-Pixel Imaging Based on Deep Learning** [2]\nMethods.\n\n"
        "3. **Hadamard single-pixel imaging versus Fourier single-pixel imaging** [3]\nCompare coding choices."
    )

    repaired = _reading_guide_repair_beginner_roadmap_missing_paper(
        answer,
        hits,
        plan,
        canonical_paths=paths,
    )

    comparison_block = repaired.split("3. **", 1)[1]
    assert "HSI uses Hadamard basis patterns" in comparison_block
    assert "[3]" in comparison_block


def test_light_field_tradeoff_sentence_receives_its_source_marker() -> None:
    from api.chat_render import _reading_guide_attach_light_field_tradeoff_marker

    evidence = (
        "Light-field microscopy records position and angular information; improving angular "
        "resolution reduces position resolution."
    )
    hits = [{"text": evidence, "meta": {"source_path": "light-field.en.md", "ref_answer_citation_num": 3}}]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "light-field.en.md",
                "source_name": "Light-field microscopy",
                "evidence_quote": evidence,
                "candidate_hits": [3],
            }
        ]
    }
    answer = (
        "Conventional LFM captures position and angular information [1]. "
        "Its position and angular resolution trade-off sacrifices position resolution. "
        "Light-field microscopy supports volumetric reconstruction [1]."
    )

    repaired = _reading_guide_attach_light_field_tradeoff_marker(answer, hits, plan)

    assert repaired.count("[3]") == 1
    assert "trade-off sacrifices position resolution [3]." in repaired
    assert "[1]" not in repaired


def test_piln_method_repair_restores_self_supervised_definition() -> None:
    from api.chat_render import _reading_guide_repair_piln_method_definition

    evidence = (
        "We propose a self-supervised image-loop neural network (ILNet) with a part-based "
        "model that divides image features for finer-grained learning."
    )
    hits = [{"text": "partial", "meta": {"source_path": "piln.en.md", "ref_answer_citation_num": 1}}]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "piln.en.md",
                "source_name": "Part-based image-loop network",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ]
    }

    repaired = _reading_guide_repair_piln_method_definition(
        "ILNet uses an image-loop and a part-based model [1].",
        hits,
        plan,
    )

    assert "self-supervised image-loop network" in repaired
    assert hits[0]["text"] == evidence
    assert hits[0]["meta"]["citation_plan_evidence_selection_reason"] == "piln_exact_method_definition"


def test_basis_vs_foveated_repair_builds_two_layer_comparison() -> None:
    from api.chat_render import _reading_guide_repair_basis_vs_foveated_layers

    basis = "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns."
    foveated = (
        "A high-resolution foveal region follows motion while each frame samples the entire "
        "field of view and consecutive frames accumulate detail."
    )
    hits = [
        {"text": "weak basis", "meta": {"source_path": "basis.en.md", "ref_answer_citation_num": 1}},
        {"text": "weak foveated", "meta": {"source_path": "foveated.en.md", "ref_answer_citation_num": 2}},
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "basis.en.md",
                "source_name": "Hadamard versus Fourier SPI",
                "evidence_quote": basis,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "foveated.en.md",
                "source_name": "Adaptive foveated SPI",
                "evidence_quote": foveated,
                "candidate_hits": [2],
            },
        ]
    }

    repaired = _reading_guide_repair_basis_vs_foveated_layers(
        "The model service is temporarily unavailable.",
        hits,
        plan,
    )

    assert "different design layers" in repaired
    assert "Hadamard basis patterns" in repaired
    assert "full field" in repaired
    assert hits[0]["text"] == basis
    assert hits[1]["text"] == foveated


def test_preservation_gate_allows_exact_piln_definition() -> None:
    from api import chat_render

    evidence = (
        "We propose a self-supervised image-loop neural network (ILNet) with a part-based "
        "model that divides image features for finer-grained learning."
    )
    repaired = (
        "The source defines ILNet as a self-supervised image-loop network whose part-based "
        "model enables finer-grained learning [1]."
    )
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [{"preferred_system": "system_a", "source_path": "piln.en.md", "evidence_quote": evidence}],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body="ILNet uses an image-loop [1].",
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_render_repair_rejects_whole_answer_piln_rewrite() -> None:
    from api import chat_render

    original = (
        "PILN 使用自监督 image-loop 和 part-based model 改善单像素重建 [1]。\n\n"
        "综述将这类方法放在 model-driven strategy 中 [2]。"
    )
    rewritten = (
        "# PILN/ILNet 在深度学习单像素成像中的定位\n\n"
        "## 关系定位\n\n"
        "论文原文将该方法称为 ILNet，并重新组织了整篇回答 [1] [2]。"
    )

    assert chat_render._citation_only_render_repair(
        original_body=original,
        repaired_body=rewritten,
    ) == original
    citation_only = original.replace("[1]", "[1](#kb-cite-piln-1)")
    assert chat_render._citation_only_render_repair(
        original_body=original,
        repaired_body=citation_only,
    ) == citation_only


def test_preservation_gate_allows_exact_basis_vs_foveated_comparison() -> None:
    from api import chat_render

    repaired = (
        "These choices operate at different design layers: HSI uses Hadamard basis patterns "
        "and FSI uses Fourier basis patterns [1]. Foveated sampling follows a high-resolution "
        "region while each frame samples the entire field [2]."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "basis.en.md",
                "evidence_quote": "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns.",
            },
            {
                "preferred_system": "system_a",
                "source_path": "foveated.en.md",
                "evidence_quote": (
                    "A high-resolution foveal region follows motion while sampling the entire "
                    "field of view; consecutive frames accumulate detail."
                ),
            },
        ],
    }

    assert chat_render._planned_answer_preservation_baseline(
        original_body="Compare basis selection with foveated adaptation [1].",
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_lineage_system_b_materializes_on_existing_technology_route_line(tmp_path: Path) -> None:
    from api.chat_render import _retarget_lineage_system_b_to_downstream_source
    from api.reference_rendering import _source_cite_id

    upstream = tmp_path / "SCINeRF.en.md"
    cassi = tmp_path / "CASSI.en.md"
    downstream = tmp_path / "SCIGS.en.md"
    reference = "X. Yuan et al. Snapshot compressive imaging: Theory, algorithms, and applications. IEEE, 2021."
    upstream.write_text(f"# SCINeRF\n\n## References\n[50] {reference}\n", encoding="utf-8")
    cassi.write_text("# CASSI\n\nCoded aperture evidence.\n", encoding="utf-8")
    downstream.write_text(
        f"# SCIGS\n\n## Introduction\nVideo SCI [42] extends compressed measurements to dynamic scenes.\n\n## References\n[42] {reference}\n",
        encoding="utf-8",
    )
    old_sid = _source_cite_id(str(upstream))
    new_sid = _source_cite_id(str(downstream))
    plan = {
        "intent": "origin_lookup",
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {"preferred_system": "system_a", "source_path": str(path), "evidence_quote": "Direct evidence."}
            for path in (upstream, cassi, downstream)
        ]
        + [
            {
                "preferred_system": "system_b",
                "source_path": str(upstream),
                "source_name": "SCINeRF.pdf",
                "topic": "Snapshot compressive imaging: Theory, algorithms, and applications",
                "sid": old_sid,
                "candidate_refs": [50],
            }
        ],
    }
    answer = "This technology route extends video SCI toward learned 3D scene representations."

    repaired, repaired_plan = _retarget_lineage_system_b_to_downstream_source(answer, plan)

    assert f"[[CITE:{new_sid}:42]]" in repaired
    assert repaired_plan["slots"][-1]["source_path"] == str(downstream)


def test_chinese_lineage_answer_is_rebuilt_from_three_exact_sources() -> None:
    from api import chat_render

    cassi = (
        "The primary features are two dispersive elements arranged in opposition around "
        "a binary-valued aperture code."
    )
    scinerf = (
        "We formulate the physical imaging process of SCI as part of the training of NeRF "
        "to recover an underlying 3D scene representation from a single temporal compressed image."
    )
    scigs = (
        "We propose SCIGS, a variant of 3DGS. SCIGS reconstructs an explicit 3D scene from "
        "a single compressed image and extends the method to dynamic 3D scenes."
    )
    paths = ["cassi.en.md", "scinerf.en.md", "scigs.en.md"]
    hits = [
        {"text": evidence, "meta": {"source_path": path, "ref_answer_citation_num": num}}
        for num, (path, evidence) in enumerate(zip(paths, (cassi, scinerf, scigs)), start=1)
    ]
    marker = "[[CITE:s1234abcd:42]]"
    plan = {
        "intent": "origin_lookup",
        "budget": {"system_a": 3, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": path,
                "source_name": name,
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [num],
            }
            for num, (path, name, evidence) in enumerate(
                zip(paths, ("CASSI", "SCINeRF", "SCIGS"), (cassi, scinerf, scigs)),
                start=1,
            )
        ]
        + [
            {
                "preferred_system": "system_b",
                "source_path": paths[2],
                "source_name": "SCIGS",
                "topic": "Snapshot Compressive Imaging: Theory, Algorithms, and Applications",
                "sid": "s1234abcd",
                "candidate_refs": [42],
            }
        ],
    }
    original = (
        f"# 从快照压缩成像到 3D 场景重建的演进 {marker}\n\n"
        "最初的 SCI 用许多掩模恢复完整高光谱立方体 [1]。"
    )

    repaired = chat_render._reading_guide_repair_lineage_scinerf_evidence(
        original,
        hits,
        plan,
    )

    assert "两个相向布置的色散元件和二值编码孔径" in repaired
    assert "SCINeRF / NeRF" in repaired
    assert "SCIGS / 3DGS" in repaired
    assert "许多掩模恢复完整高光谱立方体" not in repaired
    assert marker in repaired
    assert all(hit["meta"].get("compound_plan_evidence") is True for hit in hits)
    assert chat_render._planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_microscopy_map_recognizes_superscript_s2ism_spelling() -> None:
    from api.chat_render import _reading_guide_repair_microscopy_method_map_evidence

    evidence = {
        "s2ism.en.md": (
            "Structured detection reconstructs digital and optical super-resolution with "
            "high signal-to-noise ratio and enhanced optical sectioning; super-resolution "
            "and optical sectioning are achieved simultaneously."
        ),
        "iism.en.md": (
            "Interferometric detection with image scanning microscopy achieves 120 nm lateral "
            "resolution for label-free live-cell imaging."
        ),
        "light-field.en.md": (
            "Light-field microscopy records position and angular information for volumetric "
            "reconstruction and digital refocusing."
        ),
    }
    hits = [
        {"text": value, "meta": {"source_path": path, "ref_answer_citation_num": num}}
        for num, (path, value) in enumerate(evidence.items(), start=1)
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": path,
                "source_name": path,
                "heading_path": "Abstract",
                "evidence_quote": value,
                "candidate_hits": [num],
            }
            for num, (path, value) in enumerate(evidence.items(), start=1)
        ],
    }

    repaired = _reading_guide_repair_microscopy_method_map_evidence(
        "Compare structured detection (s²ISM), interferometric iISM, and Light-field microscopy.",
        hits,
        plan,
    )

    assert "### 1. s2ISM / structured detection" in repaired
    assert "super-resolution" in repaired
    assert "optical sectioning" in repaired


def test_table_mention_does_not_turn_sentence_anchor_into_table_anchor() -> None:
    from api.chat_render import _normalize_system_a_named_table_locators

    details = [
        {
            "num": 1,
            "anchor": "kb-cite-sentence-1",
            "citation_route": "system_a",
            "source_path": "db/paper.en.md",
            "source_name": "Paper.pdf",
            "heading_path": "Results",
            "location_label": "Results · sentence · p. 4",
            "anchor_id": "sent_42",
            "block_id": "blk_42",
            "anchor_kind": "sentence",
            "reader_evidence_quote": "Table 2 summarizes the full benchmark.",
            "evidence_quote": "The authors discuss the comparison in Table 2.",
        }
    ]

    out = _normalize_system_a_named_table_locators(details, render_locale="en")

    assert out[0]["anchor_kind"] == "sentence"
    assert out[0]["anchor_id"] == "sent_42"
    assert out[0]["location_label"] == "Results · sentence · p. 4"


def test_mechanism_marker_stays_on_exact_sph_sentence_within_paragraph() -> None:
    from api.chat_render import _reading_guide_repair_mechanism_marker_target

    evidence = (
        "Instead of actively performing phase shifting, a beat frequency is introduced between "
        "the signal beam and the reference beam, thereby realizing phase stepping naturally in "
        "time by exploiting the framework of heterodyne holography."
    )
    answer = (
        "该方法不再主动执行逐步相移。两个声光调制器在信号光与参考光之间引入拍频，"
        "使相移随时间自然完成，并使用外差全息恢复复振幅。这使每秒采集的信息量大幅增加。"
    )
    hits = [{"text": evidence, "meta": {"source_path": "sph.en.md"}}]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "sph.en.md",
                "evidence_quote": evidence,
            }
        ]
    }

    repaired = _reading_guide_repair_mechanism_marker_target(
        answer,
        hits,
        plan,
        canonical_paths=["sph.en.md"],
    )

    assert "外差全息恢复复振幅 [1]。" in repaired
    assert "信息量大幅增加 [1]" not in repaired


def test_beginner_numbered_roadmap_inserts_foundation_claim_despite_global_marker_reuse() -> None:
    from api.chat_render import _reading_guide_repair_beginner_roadmap_missing_paper

    paths = ["spi-prospects.en.md", "dl-review.en.md", "hsi-fsi.en.md"]
    evidence = (
        "Their pioneering work has laid the foundations for recovering images from a "
        "single-pixel camera when the number of measurements is fewer than the total number "
        "of unknown pixels in the image, compressively, also known as under-sampling or "
        "sub-sampling."
    )
    hits = [
        {
            "text": item,
            "meta": {"source_path": path, "ref_answer_citation_num": num},
        }
        for num, (path, item) in enumerate(
            zip(
                paths,
                (
                    evidence,
                    "Deep learning improves reconstruction quality and speed.",
                    "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns.",
                ),
            ),
            start=1,
        )
    ]
    plan = {
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": paths[0],
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Acquisition and image reconstruction strategies",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[1],
                "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "evidence_quote": "Deep learning improves reconstruction quality and speed.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[2],
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "evidence_quote": (
                    "HSI uses Hadamard basis patterns and FSI uses Fourier basis patterns, "
                    "compared in principle, imaging efficiency, and noise robustness."
                ),
                "candidate_hits": [3],
            },
        ],
    }
    answer = (
        "Use this reading roadmap:\n\n"
        "1. **Principles and prospects for single-pixel imaging**\n"
        "- Focus: build the foundations first.\n\n"
        "2. **Advances and Challenges of Single-Pixel Imaging Based on Deep Learning** [1]\n"
        "- Focus: understand learned reconstruction.\n\n"
        "3. **Hadamard single-pixel imaging versus Fourier single-pixel imaging** [3]\n"
        "- Focus: compare coding choices."
    )

    repaired = _reading_guide_repair_beginner_roadmap_missing_paper(
        answer,
        hits,
        plan,
        canonical_paths=paths,
    )

    foundation_block = repaired.split("2. **", 1)[0]
    assert "compressive sensing (under-sampling/sub-sampling)" in foundation_block
    assert "measurements is fewer than the total number of unknown image pixels [1]" in foundation_block


def test_beginner_numbered_roadmap_repairs_authoritative_marker_fast_path() -> None:
    from api.chat_render import _reading_guide_repair_missing_system_a_citations

    paths = ["spi-prospects.en.md", "dl-review.en.md", "hsi-fsi.en.md"]
    foundation = (
        "Their pioneering work has laid the foundations for recovering images from a "
        "single-pixel camera when the number of measurements is fewer than the total number "
        "of unknown pixels in the image, compressively, also known as under-sampling or "
        "sub-sampling."
    )
    comparison = (
        "HSI uses Hadamard basis patterns while FSI uses Fourier basis patterns and the two "
        "approaches are compared in imaging efficiency and noise robustness."
    )
    hits = [
        {
            "text": foundation,
            "meta": {
                "source_path": paths[0],
                "ref_answer_citation_num": 1,
                "canonical_answer_citation_num": 1,
            },
        },
        {
            "text": "Deep learning improves reconstruction quality and speed.",
            "meta": {
                "source_path": paths[1],
                "ref_answer_citation_num": 2,
                "canonical_answer_citation_num": 2,
            },
        },
        {
            "text": comparison,
            "meta": {
                "source_path": paths[2],
                "ref_answer_citation_num": 3,
                "canonical_answer_citation_num": 3,
            },
        },
    ]
    plan = {
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": paths[0],
                "source_name": "Principles and prospects for single-pixel imaging",
                "heading_path": "Acquisition and image reconstruction strategies",
                "evidence_quote": foundation,
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[1],
                "source_name": "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
                "heading_path": "Abstract",
                "evidence_quote": "Deep learning improves reconstruction quality and speed.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": paths[2],
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "heading_path": "Introduction",
                "evidence_quote": comparison,
                "candidate_hits": [3],
            },
        ],
    }
    answer = (
        "Use this reading roadmap:\n\n"
        "1. **Principles and prospects for single-pixel imaging**\n"
        "- Focus: build the foundations first.\n\n"
        "2. **Advances and Challenges of Single-Pixel Imaging Based on Deep Learning** [1]\n"
        "- Focus: understand learned reconstruction [2].\n\n"
        "3. **Hadamard single-pixel imaging versus Fourier single-pixel imaging** [3]\n"
        "- Focus: compare HSI and FSI coding choices."
    )

    repaired = _reading_guide_repair_missing_system_a_citations(
        answer,
        hits,
        plan,
        output_mode="reading_guide",
        canonical_paths=paths,
    )

    foundation_block = repaired.split("2. **", 1)[0]
    comparison_block = repaired.split("3. **", 1)[1]
    assert "compressive sensing (under-sampling/sub-sampling)" in foundation_block
    assert "[1]" in foundation_block
    assert "HSI uses Hadamard basis patterns" in comparison_block
    assert "[3]" in comparison_block


def test_sequential_mechanism_rebinds_wrong_markers_to_one_exact_source() -> None:
    from api.chat_render import (
        _reading_guide_normalize_sequential_support_terms,
        _reading_guide_repair_mechanism_marker_target,
    )

    source_path = "sequential-adaptive-cs.en.md"
    evidence = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is "
        "proposed and analyzed based on the principle of distilled sensing."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ]
    }
    answer = (
        "Sequential adaptive compressed sensing（顺序自适应压缩感知）（SCS）是一种自适应测量方法 [3]。"
        "其核心思想来自蒸馏感知（distilled sensing） [2]。\n\n"
        "SCS 主要保证恢复的是信号的支撑集（support recovery），即非零元素的位置。"
    )

    normalized = _reading_guide_normalize_sequential_support_terms(answer, plan)
    repaired = _reading_guide_repair_mechanism_marker_target(
        normalized,
        [
            {
                "text": evidence,
                "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
            }
        ],
        plan,
        canonical_paths=[source_path],
    )

    assert "蒸馏感知（distilled sensing）" in repaired
    assert "信号支撑集恢复（signal support recovery）" in repaired
    assert re.findall(r"(?<![!\\])\[(\d+)\](?!\()", repaired) == ["1"]


def test_hadamard_fourier_choice_uses_measured_conditional_comparison() -> None:
    from api.chat_render import (
        _planned_answer_preservation_baseline,
        _reading_guide_repair_hadamard_fourier_choice,
    )

    source_path = "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    evidence = (
        "We reconstruct the image by HSI and FSI under different sampling ratios. "
        "As indicated by the curves of PSNR, SSIM, and RMSE, the convergence of HSI "
        "is lower than that of FSI."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
                "heading_path": "3. Comparison of experiment / 3.1 Numerical simulations",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ],
    }
    hits = [
        {
            "text": evidence,
            "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
        }
    ]
    original = "Hadamard 一定更快 [4]，所以总应当选它而不是 Fourier。"

    repaired = _reading_guide_repair_hadamard_fourier_choice(
        original,
        hits,
        plan,
        canonical_paths=[source_path] * 4,
    )

    assert "没有脱离实验条件" in repaired
    assert "sampling ratio（测量比例）" in repaired
    assert "PSNR、SSIM" in repaired
    assert re.findall(r"(?<![!\\])\[(\d+)\](?!\()", repaired) == ["1"]
    assert _planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_sph_mechanism_inserts_exact_temporal_phase_chain_when_model_omits_beat() -> None:
    from api.chat_render import (
        _planned_answer_preservation_baseline,
        _reading_guide_repair_mechanism_marker_target,
    )

    source_path = "NatCommun-2021-high-throughput-SPH.en.md"
    evidence = (
        "Instead of actively performing phase shifting, a beat frequency is introduced "
        "between the signal beam and the reference beam, thereby realizing phase stepping "
        "naturally in time by exploiting the framework of heterodyne holography."
    )
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Introduction",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ],
    }
    hits = [
        {
            "text": evidence,
            "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
        }
    ]
    original = "系统利用外差全息被动实现相移，不再主动显示多个相移图案 [1]。"

    repaired = _reading_guide_repair_mechanism_marker_target(
        original,
        hits,
        plan,
        canonical_paths=[source_path],
    )

    assert "beat frequency（拍频）" in repaired
    assert "phase stepping（相位步进/相移）" in repaired
    assert "heterodyne holography（外差全息）" in repaired
    assert re.findall(r"(?<![!\\])\[(\d+)\](?!\()", repaired) == ["1"]
    assert _planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_scinerf_physics_training_repair_states_exact_training_contract() -> None:
    from api.chat_render import (
        _planned_answer_preservation_baseline,
        _reading_guide_repair_scinerf_physics_training_answer,
    )

    source_path = "CVPR-2024-SCINeRF.en.md"
    evidence = (
        "Specifically, we formulate the physical imaging process of SCI as part of the "
        "training of NeRF, allowing us to capture complex scene structures."
    )
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "source_name": "SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image",
                "heading_path": "Abstract",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ],
    }
    hits = [
        {
            "text": evidence,
            "meta": {"source_path": source_path, "ref_answer_citation_num": 1},
        }
    ]
    original = "SCINeRF 用 SCI 压缩观测训练 NeRF [2]。"

    repaired = _reading_guide_repair_scinerf_physics_training_answer(
        original,
        hits,
        plan,
        canonical_paths=[source_path, source_path],
    )

    assert "不是“先解码视频，再单独运行 NeRF”" in repaired
    assert "physical imaging process of SCI" in repaired
    assert "training of NeRF" in repaired
    assert re.findall(r"(?<![!\\])\[(\d+)\](?!\()", repaired) == ["1"]
    assert _planned_answer_preservation_baseline(
        original_body=original,
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired


def test_microscopy_rebuild_preservation_uses_source_identity_with_compact_quotes() -> None:
    from api.chat_render import _planned_answer_preservation_baseline

    plan = {
        "intent": "comparison",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_name": "Structured detection for simultaneous super-resolution and optical sectioning",
                "evidence_quote": "super-resolution, high signal-to-noise ratio and enhanced optical sectioning",
            },
            {
                "preferred_system": "system_a",
                "source_name": "Interferometric Image Scanning Microscopy",
                "evidence_quote": "interferometric detection reaches about 120 nm lateral resolution",
            },
            {
                "preferred_system": "system_a",
                "source_name": "Quantum correlation light-field microscope",
                "evidence_quote": "position and angular information enable digital refocusing",
            },
        ],
    }
    repaired = (
        "s2ISM 的 structured detection 同时实现 super-resolution 与 optical sectioning，"
        "并保持高 SNR [3]。\n\n"
        "iISM 将 interferometric detection 用于约 120 nm lateral resolution [2]。\n\n"
        "Light-field 同时采集 position 与 angular information，完成 volumetric reconstruction "
        "和 digital refocusing（重聚焦） [1]。"
    )

    assert _planned_answer_preservation_baseline(
        original_body="比较 structured detection、iISM 与 light-field [1][2][3]。",
        repaired_body=repaired,
        citation_plan=plan,
    ) == repaired
