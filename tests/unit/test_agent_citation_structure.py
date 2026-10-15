from kb.agent.citation_structure import citation_numbers, inspect_citation_structure


def test_citation_structure_expands_ranges_and_keeps_order() -> None:
    assert citation_numbers("A [2, 1-3, 2].") == [2, 1, 3]
    result = inspect_citation_structure("A [1-2].", evidence_hit_count=2)
    assert result.structurally_bound is True
    assert result.binding == "bound"
    assert result.bound_citation_numbers == [1, 2]
    assert result.structural_reason == ""


def test_citation_structure_separates_binding_from_semantic_support() -> None:
    result = inspect_citation_structure(
        "A semantically unrelated claim [1, 4].",
        evidence_hit_count=2,
    )
    assert result.structurally_bound is False
    assert result.binding == "partial"
    assert result.bound_citation_numbers == [1]
    assert result.unresolved_citation_numbers == [4]
    assert result.structural_reason == "citation_index_out_of_range"


def test_non_numeric_placeholder_is_present_but_not_bound() -> None:
    result = inspect_citation_structure("Claim [[CITE:source]].", evidence_hit_count=1)
    assert result.citation_present is True
    assert result.structurally_bound is False
    assert result.binding == "unbound"
