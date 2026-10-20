from __future__ import annotations

import re

import fitz
import pytest

from kb.converter.pdf_inline_prose import recover_inline_prose_fragments
from kb.converter.pdf_image_prose import recover_pdf_prose_structure
from kb.converter.quality_repair import repair_markdown_text, write_conversion_quality_result


BEFORE = 'Our training procedure is as follows. We create'
BODY = 'a noisier version, Z := Y + M. We then'
AFTER = 'train the network to predict the observed image using L2 loss.'


def fixture(tmp_path, layout='normal'):
    pdf = tmp_path / 'source.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((50 if layout == 'no_indent' else 62, 100), BEFORE, fontsize=10)
        page.insert_text((120 if layout == 'centered' else 50, 140 if layout == 'gap' else 112), BODY, fontsize=10)
        page.insert_text((330 if layout == 'column' else 62 if layout == 'new_paragraph' else 50,
                          152 if layout == 'gap' else 124), AFTER, fontsize=10)
        if layout == 'duplicate':
            page.insert_text((50, 300), BODY, fontsize=10)
        doc.new_page()
        doc.save(pdf)
    text = '<!-- kb_page: 1 -->\n\n# Study\n\n## Method\n\n' + BEFORE + '\n\n$$\n' + BODY + '\n$$\n\n' + AFTER + '\n'
    return pdf, text


def test_inline_source_prose_is_restored_without_transcribing_any_content(tmp_path):
    pdf, text = fixture(tmp_path)
    fixed, changed = recover_inline_prose_fragments(text, pdf)
    assert changed
    assert fixed == text.replace(BEFORE + '\n\n$$\n' + BODY + '\n$$\n\n' + AFTER, BEFORE + ' ' + BODY + ' ' + AFTER)
    assert re.sub(r'\s+|\$\$', '', fixed) == re.sub(r'\s+|\$\$', '', text)
    assert '<!-- kb_page: 1 -->' in fixed and 'Z := Y + M' in fixed
    assert recover_inline_prose_fragments(fixed, pdf) == (fixed, False)
    assert recover_pdf_prose_structure(text, tmp_path / 'source.md', pdf) == (fixed, True)


@pytest.mark.parametrize('layout', ['centered', 'gap', 'column', 'duplicate', 'no_indent', 'new_paragraph'])
def test_display_equations_columns_and_paragraph_boundaries_are_not_joined(tmp_path, layout):
    pdf, text = fixture(tmp_path, layout)
    assert recover_inline_prose_fragments(text, pdf) == (text, False)


@pytest.mark.parametrize('failure', ['page', 'variable', 'operator', 'case', 'index', 'code', 'indented', 'outer_math', 'before', 'after', 'references'])
def test_no_source_identity_math_or_protected_region_substitutes(tmp_path, failure):
    pdf, text = fixture(tmp_path)
    if failure == 'page': text = text.replace('kb_page: 1', 'kb_page: 2')
    if failure == 'variable': text = text.replace('Y + M', 'Y + N')
    if failure == 'operator': text = text.replace('Y + M', 'Y - M')
    if failure == 'case': text = text.replace('Z :=', 'z :=')
    if failure == 'index': text = text.replace('Z :=', 'Z_2 :=')
    if failure == 'code': text = '```markdown\n' + text + '\n```'
    if failure == 'indented': text = text.replace('$$\n' + BODY, '    $$\n' + BODY)
    if failure == 'outer_math': text = '<!-- kb_page: 1 -->\n\n$$\n\n' + text + '\n\n$$'
    if failure == 'before': text = text.replace(BEFORE, 'A different paragraph appears here.')
    if failure == 'after': text = text.replace(AFTER, 'A different paragraph follows here.')
    if failure == 'references': text = text.replace('## Method', '## References')
    assert recover_inline_prose_fragments(text, pdf) == (text, False)


def test_fake_page_markers_inside_code_cannot_authorize_the_wrong_page(tmp_path):
    pdf, text = fixture(tmp_path)
    text = text.replace('kb_page: 1', 'kb_page: 2').replace('## Method', '```\n<!-- kb_page: 1 -->\n```\n\n## Method')
    assert recover_inline_prose_fragments(text, pdf) == (text, False)


def test_genuine_equations_and_missing_pdf_stay_unchanged(tmp_path):
    pdf, text = fixture(tmp_path)
    text = text.replace(BODY, 'Z := Y + M')
    assert recover_inline_prose_fragments(text, pdf) == (text, False)
    assert recover_inline_prose_fragments(text, tmp_path / 'missing.pdf') == (text, False)


def test_source_diagnostics_offer_the_existing_safe_repair_then_clear(tmp_path):
    pdf, text = fixture(tmp_path)
    md = tmp_path / 'source.md'
    md.write_text(text, encoding='utf8')
    quality = write_conversion_quality_result(md, source_pdf_path=pdf)
    assert quality['source_quality']['source_inline_prose_math_repair_available'] is True
    assert 'prose_dominant_display_math' in quality['repair_plan']['autofix_issue_codes']
    repaired = repair_markdown_text(md, text, issue_codes=['prose_dominant_display_math'], source_pdf_path=pdf)
    assert repaired['changed'] is True and repaired['unsafe'] is False
    assert 'unwrap_prose_display_math' in repaired['applied']
    assert 'prose_dominant_display_math' not in repaired['remaining_issue_codes']
    assert re.sub(r'\s+|\$\$', '', repaired['repaired_text']) == re.sub(r'\s+|\$\$', '', text)
