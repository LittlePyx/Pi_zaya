from __future__ import annotations

from kb.citation_sentences import citation_sentence_context
from kb.citation_table_context import cited_table_cell_claims


def test_each_comparison_cell_keeps_its_own_claim_and_source_coordinate():
    table='| Condition | First paper | Second paper |\n|---|---|---|\n| Noise | Independent components with zero mean [1] | U and V are i.i.d. [2] |'
    start=table.index('[2]')
    assert citation_sentence_context(table,start,start+3)=='U and V are i.i.d. [2]'
    assert cited_table_cell_claims(table)==['Independent components with zero mean [1]','U and V are i.i.d. [2]']


def test_math_pipes_and_code_cannot_create_or_borrow_comparison_cells():
    table='| Condition | Source |\n|---|---|\n| $E[Y|Z]$ retains its conditional bar [1] | Another result [2] |'
    cells=cited_table_cell_claims(table)
    assert cells[0]=='$E[Y|Z]$ retains its conditional bar [1]'
    assert not cited_table_cell_claims('```markdown\n'+table+'\n```')
    assert not cited_table_cell_claims('| no table header | unsupported [1] |')
    assert not cited_table_cell_claims('| A | B |\n|---|---|\n| mixed claims [1][2] | unbound |')
