from __future__ import annotations

import fitz
import pytest

from kb.converter.pdf_image_prose import recover_equation_image_prose

PREFIX = 'Recall that U and V are i.i.d. and therefore and thus'
TAIL = 'have the same expectation given S, E[U|S] = E[V|S].'


@pytest.mark.parametrize('failure', ['', 'wrong_page', 'wrong_block', 'ambiguous', 'missing_asset', 'code', 'indented_code', 'math', 'wrong_prefix', 'plain_sentence', 'missing_formula_boundary'])
def test_image_prose_requires_unique_exact_same_page_context_and_retains_math(tmp_path, failure):
    pdf = tmp_path / 'source.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        content = PREFIX + '\n' + TAIL
        if failure == 'ambiguous':
            content += '\n' + PREFIX + '\n' + TAIL
        if failure == 'plain_sentence':
            content = PREFIX + '\nThis is an unrelated following sentence.'
        if failure == 'missing_formula_boundary':
            content = PREFIX + '\nhave the same expectation given S, without an image equation.'
        if failure == 'wrong_block':
            content = PREFIX
            page.insert_text((40, 300), TAIL, fontsize=10)
        page.insert_text((40, 60), content, fontsize=10)
        doc.new_page()
        doc.save(pdf)
    md = tmp_path / 'source.en.md'
    asset = tmp_path / 'equation.png'
    if failure != 'missing_asset':
        asset.write_bytes(b'preserved-equation-image')
    text = f'<!-- kb_page: {2 if failure == "wrong_page" else 1} -->\n\n{PREFIX}\n\n![Equation](equation.png)\n'
    if failure == 'code':
        text = '```\n' + text + '```\n'
    if failure == 'math':
        text = '<!-- kb_page: 1 -->\n$$\n' + text + '$$\n'
    if failure == 'indented_code':
        text = text.replace(PREFIX, '    ' + PREFIX)
    if failure == 'wrong_prefix':
        text = text.replace('U and V', 'U and W')
    fixed, changed = recover_equation_image_prose(text, md, pdf)
    if failure:
        assert not changed and fixed == text
    else:
        assert changed
        assert fixed == text.replace(PREFIX, PREFIX + ' have the same expectation given S,')
        assert '![Equation](equation.png)' in fixed
        assert recover_equation_image_prose(fixed, md, pdf) == (fixed, False)


def test_missing_pdf_is_noop(tmp_path):
    text = PREFIX + '\n\n![Equation](missing.png)'
    assert recover_equation_image_prose(text, tmp_path / 'source.md', tmp_path / 'missing.pdf') == (text, False)
