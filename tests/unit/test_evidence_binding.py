import pytest

import kb.evidence_binding as evidence_binding


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        (
            "SPAD inputs have low bit depth, low resolution, and heavy noise.",
            "photon-limited SPAD degradation",
        ),
        (
            "The system performs super-resolution with enhanced bit depth and imaging quality.",
            "super-resolution bit-depth outcome",
        ),
        (
            "Frequency channels are multiplexed for parallel acquisition.",
            "frequency-division multiplexing",
        ),
        (
            "Spatial resolution, SNR, and optical sectioning form a three-way tradeoff.",
            "s2ism tradeoff",
        ),
        (
            "A SPAD operates in Geiger mode above breakdown voltage and uses a quenching circuit.",
            "spad geiger quenching",
        ),
        (
            "SCINeRF models the physical imaging formation process.",
            "scinerf physical formation",
        ),
        (
            "单光子 SPAD 阵列受到低分辨率和严重噪声影响。",
            "photon-limited SPAD degradation",
        ),
    ],
)
def test_compound_domain_requirements_preserve_binding_terms(
    surface: str,
    expected: str,
) -> None:
    evidence_binding._system_a_domain_terms.cache_clear()

    assert expected in evidence_binding._system_a_domain_terms(surface)


def test_compound_domain_matching_skips_legacy_full_surface_lookaheads(
    monkeypatch,
) -> None:
    class _UnexpectedSearch:
        def search(self, _surface: str):
            raise AssertionError("legacy compound lookahead should not run")

    patched_patterns = tuple(
        (
            (name, _UnexpectedSearch())
            if name in evidence_binding._SYSTEM_A_COMPOUND_DOMAIN_NAMES
            else (name, pattern)
        )
        for name, pattern in evidence_binding._SYSTEM_A_DOMAIN_PATTERNS
    )
    monkeypatch.setattr(
        evidence_binding,
        "_SYSTEM_A_DOMAIN_PATTERNS",
        patched_patterns,
    )
    evidence_binding._system_a_domain_terms.cache_clear()

    terms = evidence_binding._system_a_domain_terms(
        "SPAD inputs have low bit depth, low resolution, and heavy noise."
    )

    assert "photon-limited SPAD degradation" in terms
    evidence_binding._system_a_domain_terms.cache_clear()


def test_review_claim_rejects_method_paper_evidence() -> None:
    binding = evidence_binding.assess_system_a_hit_binding(
        answer_claim="先读综述了解探测器类型与性能瓶颈。",
        hit={"text": "We introduce deep learning into a SPAD imaging system."},
        meta={
            "citation_plan_evidence_authoritative": True,
            "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
        },
        heading="Abstract",
        evidence_quote="We introduce deep learning into a SPAD imaging system.",
        source_name="High-resolution single-photon imaging.pdf",
    )

    assert binding["status"] == "mismatch"
    assert binding["suppress_link"] is True
    assert binding["missing_terms"] == ["review identity"]
