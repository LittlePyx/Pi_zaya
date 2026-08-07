from io import BytesIO

from docx import Document

from kb.research_brief import (
    _matrix_answer_passes,
    generate_research_brief_from_matrix,
    research_brief_bibliography,
    research_brief_bibtex,
    research_brief_context,
    research_brief_docx,
    research_brief_evidence,
    research_brief_markdown,
    research_brief_quality,
    research_brief_ris,
    select_research_brief_sources,
)


def test_matrix_gate_requires_every_visible_citation_to_support_its_claim() -> None:
    from kb.agent.tools import verify_answer_citations

    hits = [
        {
            "text": "Coded acquisition uses binary masks for image reconstruction.",
            "meta": {"source_name": "Paper A", "source_path": "F:/papers/a.md"},
        },
        {
            "text": "Interferometric reconstruction estimates phase from fringe measurements.",
            "meta": {"source_name": "Paper B", "source_path": "F:/papers/b.md"},
        },
    ]
    verification = verify_answer_citations(
        "Coded acquisition uses binary masks for image reconstruction [1][2].",
        hits,
        answer_mode="evidence_grounded",
    )["verification"]

    assert verification["unsupported_claims"] == 0
    assert _matrix_answer_passes(verification, {"f:/papers/a.md", "f:/papers/b.md"}) is False

    boundary_verification = verify_answer_citations(
        "Coded acquisition and interferometric reconstruction are not directly comparable [1][2].",
        hits,
        answer_mode="evidence_grounded",
    )["verification"]
    assert boundary_verification["unsupported_claims"] == 0
    assert _matrix_answer_passes(
        boundary_verification,
        {"f:/papers/a.md", "f:/papers/b.md"},
    ) is False

    year_hits = [
        {
            "text": "The coded acquisition study uses binary masks for image reconstruction.",
            "meta": {"source_name": "Journal-2017-Coded acquisition", "source_path": "F:/papers/2017-a.md"},
        }
    ]
    wrong_year_verification = verify_answer_citations(
        "The 2020 study uses coded acquisition and binary masks [1].",
        year_hits,
        answer_mode="evidence_grounded",
    )["verification"]
    assert wrong_year_verification["unsupported_claims"] == 0
    assert _matrix_answer_passes(wrong_year_verification, {"f:/papers/2017-a.md"}) is False

    oversized_verification = verify_answer_citations(
        "\n".join(
            f"Coded acquisition uses binary masks for image reconstruction variant {index} [1]."
            for index in range(9)
        ),
        year_hits,
        answer_mode="evidence_grounded",
    )["verification"]
    assert oversized_verification["unsupported_claims"] == 0
    assert _matrix_answer_passes(oversized_verification, {"f:/papers/2017-a.md"}) is False

    contrast_hits = [
        {
            "text": "Light microscopy advances toward higher spatial resolution and improved imaging depth.",
            "meta": {"source_name": "Microscopy paper", "source_path": "F:/papers/microscopy.md"},
        }
    ]
    unsupported_contrast_verification = verify_answer_citations(
        "Light microscopy advances toward higher spatial resolution, though coherent speckle limits live cells [1].",
        contrast_hits,
        answer_mode="evidence_grounded",
    )["verification"]
    assert unsupported_contrast_verification["unsupported_claims"] == 0
    assert _matrix_answer_passes(
        unsupported_contrast_verification,
        {"f:/papers/microscopy.md"},
    ) is False


def _grounded_trace() -> dict:
    return {
        "status": "done",
        "errors": [],
        "verification": {
            "total_claims": 1,
            "supported_claims": 1,
            "unsupported_claims": 0,
            "support_ratio": 1.0,
            "evidence_status": "grounded",
        },
        "summary": {"query_scope": "basket", "quality_gate_status": "passed"},
    }


def test_research_brief_sources_prefer_matched_library_fulltext() -> None:
    selected = select_research_brief_sources(
        [
            {
                "key": "reference-12",
                "shelfItemKind": "reference",
                "title": "Matched paper",
                "sourceName": "Review paper",
                "sourcePath": "db/review.md",
                "libraryMatchTitle": "Matched paper full text",
                "libraryMatchPath": "db/library/matched-paper.md",
            }
        ]
    )

    context = research_brief_context(selected)

    assert context["items"][0]["sourcePath"] == "db/library/matched-paper.md"
    assert context["items"][0]["sourceName"] == "Matched paper full text"


def test_research_brief_reference_requires_its_own_matched_fulltext() -> None:
    selected = select_research_brief_sources(
        [
            {
                "key": "reference-12",
                "shelfItemKind": "reference",
                "title": "Upstream paper",
                "sourcePath": "db/review-that-cites-it.md",
            }
        ]
    )

    assert selected == []


def test_research_brief_quality_rejects_unexpected_or_unsupported_evidence() -> None:
    selected = [{"key": "a", "title": "Paper A", "sourcePath": "F:/papers/a.md"}]
    evidence = [{"citation_number": 1, "source_name": "Paper A", "source_path": "F:/papers/a.md"}]

    status, quality = research_brief_quality(
        answer="The experiment reports a direct improvement [1].",
        agent_trace=_grounded_trace(),
        selected_items=selected,
        evidence=evidence,
    )
    assert status == "verified"
    assert quality["support_ratio"] == 1.0
    assert quality["unexpected_sources"] == []

    wrong_scope = _grounded_trace()
    wrong_scope["summary"]["query_scope"] = "full_library"
    status, quality = research_brief_quality(
        answer="The experiment reports a direct improvement [1].",
        agent_trace=wrong_scope,
        selected_items=selected,
        evidence=evidence,
    )
    assert status == "needs_review"
    assert "query_scope_not_basket" in quality["reasons"]

    bad_trace = _grounded_trace()
    bad_trace["verification"] = {
        **bad_trace["verification"],
        "unsupported_claims": 1,
        "support_ratio": 0.5,
    }
    status, quality = research_brief_quality(
        answer="Unsupported synthesis [2].",
        agent_trace=bad_trace,
        selected_items=selected,
        evidence=[{"citation_number": 1, "source_name": "Paper B", "source_path": "F:/papers/b.md"}],
    )
    assert status == "needs_review"
    assert "unsupported_claims" in quality["reasons"]
    assert "unexpected_sources" in quality["reasons"]
    assert "unresolved_citations" in quality["reasons"]


def test_research_brief_quality_parses_combined_citations_and_requires_source_coverage() -> None:
    selected = [
        {"key": "a", "title": "Paper A", "sourcePath": "F:/papers/a.md"},
        {"key": "b", "title": "Paper B", "sourcePath": "F:/papers/b.md"},
    ]
    evidence = [
        {"citation_number": 1, "source_name": "Paper A", "source_path": "F:/papers/a.md"},
        {"citation_number": 2, "source_name": "Paper B", "source_path": "F:/papers/b.md"},
    ]

    status, quality = research_brief_quality(
        answer="The methods differ in acquisition and reconstruction [1, 2].",
        agent_trace=_grounded_trace(),
        selected_items=selected,
        evidence=evidence,
    )

    assert status == "verified"
    assert quality["citation_numbers"] == [1, 2]
    assert quality["selected_sources_without_evidence"] == []
    assert quality["generation_mode"] == "model_synthesis"
    assert quality["warnings"] == []

    status, quality = research_brief_quality(
        answer="Only one method is represented [1].",
        agent_trace=_grounded_trace(),
        selected_items=selected,
        evidence=evidence,
    )

    assert status == "needs_review"
    assert quality["selected_sources_without_evidence"] == ["Paper B"]
    assert "selected_sources_without_evidence" in quality["reasons"]


def test_research_brief_exports_keep_evidence_and_reference_identity() -> None:
    shelf = [
        {
            "key": "a",
            "title": "Paper A",
            "sourcePath": "F:/papers/a.md",
            "authors": "Ada Author",
            "year": "2025",
            "venue": "Optics Letters",
            "doi": "10.1000/paper-a",
        }
    ]
    assert select_research_brief_sources(shelf, item_keys=["a", "missing"]) == shelf
    evidence = research_brief_evidence(
        [
            {
                "text": "Paper A reports the measured acquisition result.",
                "score": 9.0,
                "meta": {
                    "source_path": "F:/papers/a.md",
                    "source_name": "Paper A",
                    "heading_path": "Results / Acquisition",
                    "page": 4,
                },
            }
        ]
    )
    bibliography = research_brief_bibliography(shelf, evidence)
    record = {
        "title": "Acquisition brief",
        "objective": "Compare acquisition performance.",
        "content_markdown": "## Finding\n\nPaper A reports the measured result [1].",
        "quality_status": "verified",
        "revision": 2,
        "evidence": evidence,
        "bibliography": bibliography,
        "lineage": {
            "status": "matrix_updated",
            "source_matrix_id": "matrix-1",
            "source_matrix_title": "Acquisition matrix",
            "source_matrix_revision": 1,
            "current_matrix_revision": 2,
            "export_mode": "historical",
        },
    }

    markdown = research_brief_markdown(record)
    assert "Paper A reports the measured result [1]" in markdown
    assert "Evidence appendix" in markdown
    assert "Results / Acquisition" in markdown
    assert "10.1000/paper-a" in markdown
    assert "freshness: matrix_updated" in markdown

    bibtex = research_brief_bibtex(record)
    assert "@article" in bibtex
    assert "10.1000/paper-a" in bibtex
    assert "% Pi_zaya Matrix lineage:" in bibtex
    ris = research_brief_ris(record)
    assert "TI  - Paper A" in ris
    assert "DO  - 10.1000/paper-a" in ris
    assert "N1  - Pi_zaya Matrix lineage:" in ris

    document = Document(BytesIO(research_brief_docx(record)))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Acquisition brief" in text
    assert "Paper A reports the measured result [1]" in text
    assert "Evidence appendix" in text
    assert "freshness: matrix_updated" in text


def test_matrix_backed_brief_falls_back_when_model_omits_a_selected_source(monkeypatch) -> None:
    from kb.agent import tools as agent_tools

    source_items = [
        {"key": "a", "title": "Paper A", "sourcePath": "F:/papers/a.md"},
        {"key": "b", "title": "Paper B", "sourcePath": "F:/papers/b.md"},
    ]
    rows = []
    evidence = []
    for index, item in enumerate(source_items, start=1):
        evidence_id = f"ev-{index}"
        quote = f"Paper {chr(64 + index)} uses a measured coded acquisition method."
        rows.append(
            {
                "id": f"row-{index}",
                "paper": item["title"],
                "source_name": item["title"],
                "source_path": item["sourcePath"],
                "source_status": "active",
                "cells": {
                    "method": {
                        "value": quote,
                        "support_status": "grounded",
                        "evidence_ids": [evidence_id],
                        "manual_override": False,
                    }
                },
            }
        )
        evidence.append(
            {
                "id": evidence_id,
                "field": "method",
                "source_name": item["title"],
                "source_path": item["sourcePath"],
                "heading_path": "Method",
                "evidence_quote": quote,
                "score": 8.0,
            }
        )
    matrix = {
        "id": "matrix-a-b",
        "revision": 3,
        "quality_status": "verified",
        "rows": rows,
        "evidence": evidence,
        "source_items": source_items,
    }
    monkeypatch.setattr(
        agent_tools,
        "generate_grounded_answer",
        lambda *args, **kwargs: {
            "answer": "Only Paper A is represented [1].",
            "quality_gate": {"status": "passed", "reasons": [], "warnings": []},
        },
    )

    payload = generate_research_brief_from_matrix(
        "Compare both selected papers.",
        matrix_record=matrix,
        settings=object(),
    )

    assert payload["agent_trace"]["summary"]["quality_gate_status"] == "fallback"
    assert "[1]" in payload["answer"]
    assert "[2]" in payload["answer"]
    matrix_evidence = research_brief_evidence(payload["hits"])
    status, quality = research_brief_quality(
        answer=payload["answer"],
        agent_trace=payload["agent_trace"],
        selected_items=source_items,
        evidence=matrix_evidence,
    )
    assert status == "verified"
    assert quality["generation_mode"] == "extractive_fallback"
    assert quality["selected_sources_without_evidence"] == []


def test_matrix_backed_brief_preserves_supported_model_claims_and_repairs_only_failed_ones(monkeypatch) -> None:
    from kb.agent import tools as agent_tools

    source_items = [
        {"key": "a", "title": "Paper A", "sourcePath": "F:/papers/a.md"},
        {"key": "b", "title": "Paper B", "sourcePath": "F:/papers/b.md"},
    ]
    rows = []
    evidence = []
    for index, item in enumerate(source_items, start=1):
        evidence_id = f"ev-{index}"
        quote = f"Paper {chr(64 + index)} uses a measured coded acquisition method."
        rows.append(
            {
                "id": f"row-{index}",
                "paper": item["title"],
                "source_name": item["title"],
                "source_path": item["sourcePath"],
                "source_status": "active",
                "cells": {
                    "method": {
                        "value": quote,
                        "support_status": "grounded",
                        "evidence_ids": [evidence_id],
                        "manual_override": False,
                    }
                },
            }
        )
        evidence.append(
            {
                "id": evidence_id,
                "field": "method",
                "source_name": item["title"],
                "source_path": item["sourcePath"],
                "heading_path": "Method",
                "evidence_quote": quote,
                "score": 8.0,
            }
        )
    matrix = {
        "id": "matrix-a-b",
        "revision": 3,
        "quality_status": "verified",
        "rows": rows,
        "evidence": evidence,
        "source_items": source_items,
    }
    monkeypatch.setattr(
        agent_tools,
        "generate_grounded_answer",
        lambda *args, **kwargs: {
                "answer": (
                    "Paper A uses a measured coded acquisition method [1].\n"
                    "An unrelated acceleration method improves throughput [2]."
            ),
            "llm_used": True,
            "quality_gate": {
                "status": "failed",
                "reasons": ["unsupported_local_claim"],
                "warnings": [],
            },
        },
    )

    payload = generate_research_brief_from_matrix(
        "Compare both selected papers.",
        matrix_record=matrix,
        settings=object(),
    )

    summary = payload["agent_trace"]["summary"]
    repair = summary["claim_repair"]
    assert summary["quality_gate_status"] == "repaired"
    assert repair["preserved_model_claims"] == 1
    assert repair["removed_unsupported_claims"] == 1
    assert repair["supplemented_source_claims"] == 1
    assert "unrelated acceleration" not in payload["answer"]
    assert "Paper B uses a measured coded acquisition method [2]" in payload["answer"]
    assert summary["phase_timings_ms"]["model_synthesis"] >= 0

    matrix_evidence = research_brief_evidence(payload["hits"])
    status, quality = research_brief_quality(
        answer=payload["answer"],
        agent_trace=payload["agent_trace"],
        selected_items=source_items,
        evidence=matrix_evidence,
    )
    assert status == "verified"
    assert quality["generation_mode"] == "model_synthesis_repaired"
    assert quality["unsupported_claims"] == 0
