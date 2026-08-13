from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.research_qa.prepare_unseen_corpus import (
    load_corpus_manifest,
    sha256_file,
    verify_pdf,
)
from tools.research_qa.run_research_qa_eval import (
    load_fixture,
    summarize_suite_coverage,
    validate_fixture_contracts,
)
from kb.converter.quality_repair import _source_page_prose_omission_damage


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "docs" / "research_qa_unseen_corpus_v1.json"
FIXTURE_V2_PATH = REPO_ROOT / "docs" / "research_qa_unseen_corpus_v2.json"


def test_unseen_corpus_fixture_is_complete_and_pre_registered() -> None:
    fixture = load_fixture(FIXTURE_PATH)

    assert len(fixture.docs) == 10
    assert len(fixture.cases) == 20
    assert validate_fixture_contracts(fixture) == []
    coverage = summarize_suite_coverage(fixture, "unseen_corpus_acceptance_v1")
    assert coverage["case_count"] == 20
    assert coverage["doc_count"] == 10
    assert coverage["locales"] == ["en", "zh"]
    assert all(bool(case.get("sourceGrounded")) for case in fixture.cases)


def test_unseen_corpus_manifest_has_unique_verified_hashes() -> None:
    manifest = load_corpus_manifest(FIXTURE_PATH)

    assert len(manifest) == 10
    assert len({item["id"] for item in manifest}) == 10
    assert len({item["sha256"] for item in manifest}) == 10
    assert all(len(item["sha256"]) == 64 for item in manifest)


def test_unseen_corpus_v2_fixture_is_complete_pre_registered_and_disjoint() -> None:
    fixture = load_fixture(FIXTURE_V2_PATH)
    manifest = load_corpus_manifest(FIXTURE_V2_PATH)
    v1_manifest = load_corpus_manifest(FIXTURE_PATH)

    assert len(fixture.docs) == 10
    assert len(fixture.cases) == 22
    assert validate_fixture_contracts(fixture) == []
    coverage = summarize_suite_coverage(fixture, "unseen_corpus_acceptance_v2")
    assert coverage["case_count"] == 22
    assert coverage["doc_count"] == 10
    assert coverage["locales"] == ["en", "zh"]
    assert all(bool(case.get("sourceGrounded")) for case in fixture.cases)
    assert len({item["id"] for item in manifest}) == 10
    assert len({item["sha256"] for item in manifest}) == 10
    assert {item["arxiv"] for item in manifest}.isdisjoint(
        {item["arxiv"] for item in v1_manifest}
    )


def test_pdf_verification_rejects_non_pdf_and_hash_mismatch(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="not a PDF"):
        verify_pdf(invalid, hashlib.sha256(invalid.read_bytes()).hexdigest())

    candidate = tmp_path / "candidate.pdf"
    candidate.write_bytes(b"%PDF-1.7\nfixture")
    assert sha256_file(candidate) == hashlib.sha256(candidate.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_pdf(candidate, "0" * 64)


def test_prose_omission_check_keeps_interior_deletion_detection_on_long_pages() -> None:
    prefix = " ".join(f"prefixword{index}" for index in range(2500))
    suffix = " ".join(f"suffixword{index}" for index in range(2500))
    source = (
        "The anchored method begins with stable context words for comparison and evaluation. "
        "Researchers carefully describe the experimental setting, training procedure, model "
        "architecture, measured outcomes, relevant assumptions, and reproducible observations. "
        "This extended explanation supplies enough ordinary prose tokens to represent a source "
        "paragraph that the production quality gate would assess instead of a short caption. "
        "It then contains eight deliberately omitted important scientific evidence terms inside the paragraph. "
        "The surrounding sentences continue with detailed analysis of accuracy, efficiency, "
        "limitations, ablations, dataset construction, and comparisons against established baselines. "
        "Finally stable trailing context words close this source paragraph for reliable matching."
    )
    converted = source.replace("eight deliberately omitted important scientific evidence terms inside ", "")

    result = _source_page_prose_omission_damage([source], f"{prefix} {converted} {suffix}")

    assert result["assessed_prose_block_count"] == 1
    assert result["anchored_omitted_word_count"] == 8
    assert result["anchored_omission_group_count"] == 1
    assert result["text_omission"] is True
