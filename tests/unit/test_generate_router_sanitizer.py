from __future__ import annotations

from api.routers.generate import _strip_internal_structured_markers


def test_stream_output_hides_complete_and_partial_structured_citation_tokens() -> None:
    complete = _strip_internal_structured_markers(
        "The claim is supported [[CITE:abc123:7]] by the source."
    )
    partial = _strip_internal_structured_markers(
        "The claim is supported [[CITE:abc123:"
    )

    assert "CITE" not in complete
    assert "[[" not in complete
    assert "The claim is supported" in complete
    assert "CITE" not in partial
    assert "[[" not in partial


def test_stream_output_removes_empty_bracket_shell_after_hidden_citation() -> None:
    out = _strip_internal_structured_markers("Claim [ [[CITE:abc123:7]] ].")
    nested = _strip_internal_structured_markers("Claim [[[CITE:abc123:7]]].")

    assert out == "Claim."
    assert nested == "Claim."
    assert _strip_internal_structured_markers("- [ ] task") == "- [ ] task"
    assert _strip_internal_structured_markers("[](https://example.com)") == "[](https://example.com)"
