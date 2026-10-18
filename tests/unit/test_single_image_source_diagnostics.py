from __future__ import annotations

import fitz
import pytest

from kb.converter.quality_repair import (
    _demote_source_proven_nonheading_numbered_headings,
    _plausible_reference_citation_indices,
)
from kb.converter.quality_gate import prepare_markdown_for_index


@pytest.mark.parametrize("prefix", [
    "the noise level range L = ",
    "its pre-trained model on L = ",
])
def test_declared_training_interval_does_not_inflate_reference_index(prefix):
    assert _plausible_reference_citation_indices(prefix + "[0,55]. Compare [42].") == {42}


@pytest.mark.parametrize("text", [
    "The noise level range is discussed in [50,55].",
    "The pre-trained model is described in [50,55].",
    "Compare noise models [50,55].",
    "L = [50,55].",
    "Noise level range L = [0,50,55].",
])
def test_real_or_ambiguous_reference_numbers_still_require_complete_bibliography(text):
    assert {50, 55} <= _plausible_reference_citation_indices(text)


def _pdf(path, *, number_size=7, number_baseline=154, duplicate_heading=False):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "1. Introduction", fontsize=12, fontname="hebo")
    page.insert_text((50, 100), "3. Method", fontsize=12, fontname="hebo")
    page.insert_text((50, number_baseline), "2", fontsize=number_size)
    page.insert_text((57, 150), " .", fontsize=10)
    if duplicate_heading:
        page.insert_text((50, 200), "2 .", fontsize=12, fontname="hebo")
    doc.save(path)
    doc.close()


def test_small_lowered_fraction_denominator_is_demoted_without_dropping_text(tmp_path):
    pdf=tmp_path/'source.pdf';_pdf(pdf)
    text="<!-- kb_page: 1 -->\n## 1. Introduction\n## 3. Method\n## 2 .\n"
    fixed,changed=_demote_source_proven_nonheading_numbered_headings(text,tmp_path/'source.en.md',pdf)
    assert changed
    assert fixed==text.replace("## 2 .", "2 .")


@pytest.mark.parametrize("number_size,number_baseline,duplicate_heading,page_marker", [
    (10,150,False,1),
    (7,150,False,1),
    (7,154,True,1),
    (7,154,False,2),
])
def test_heading_demote_requires_same_page_unambiguous_fraction_typography(tmp_path,number_size,number_baseline,duplicate_heading,page_marker):
    pdf=tmp_path/'source.pdf';_pdf(pdf,number_size=number_size,number_baseline=number_baseline,duplicate_heading=duplicate_heading)
    text=f"<!-- kb_page: {page_marker} -->\n## 1. Introduction\n## 3. Method\n## 2 .\n"
    assert _demote_source_proven_nonheading_numbered_headings(text,tmp_path/'source.en.md',pdf)==(text,False)


def test_fraction_text_in_code_is_not_rewritten(tmp_path):
    pdf=tmp_path/'source.pdf';_pdf(pdf)
    text="<!-- kb_page: 1 -->\n```md\n## 2 .\n```\n"
    assert _demote_source_proven_nonheading_numbered_headings(text,tmp_path/'source.en.md',pdf)==(text,False)


@pytest.mark.parametrize("body", ["    ## 2 .\n", "````md\n```\n## 2 .\n````\n"])
def test_indented_code_and_longer_fences_preserve_literal_heading_text(tmp_path,body):
    pdf=tmp_path/'source.pdf';_pdf(pdf)
    text="<!-- kb_page: 1 -->\n"+body
    assert _demote_source_proven_nonheading_numbered_headings(text,tmp_path/'source.en.md',pdf)==(text,False)


def test_unproven_section_displacement_stays_blocked_after_source_repair_attempt(tmp_path):
    pdf=tmp_path/'source.pdf'
    doc=fitz.open();page=doc.new_page()
    for y,text in [(50,"1. Introduction"),(100,"3. Methods"),(150,"2. Results")]:
        page.insert_text((50,y),text,fontsize=12,fontname="hebo")
    doc.save(pdf);doc.close()
    md=tmp_path/'source.en.md'
    text="<!-- kb_page: 1 -->\n# Demo\n## Abstract\nA demonstration.\n## 1. Introduction\n## 3. Methods\n## 2. Results\n"
    md.write_text(text,encoding='utf8')
    result=prepare_markdown_for_index(md,source_pdf_path=pdf)
    assert result['auto_repair']['attempted']
    assert result['indexable'] is False and result['status']=='blocked'
    assert 'out_of_order_sections' in result['blocking_issue_codes']
    assert result['override_applied'] is False
    assert md.read_text(encoding='utf8')==text
