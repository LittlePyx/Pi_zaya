from __future__ import annotations

from kb.converter.post_processing import _enforce_heading_policy


def test_enforce_heading_policy_drops_journal_running_headers():
    md = "\n".join(
        [
            "# Principles and prospects for single-pixel imaging",
            "",
            "## How a single-pixel camera works",
            "## REVIEW ARTICLE",
            "## NATURAL PHOTONICS",
            "## Understanding compressed sensing",
        ]
    )
    out = _enforce_heading_policy(md)
    assert "REVIEW ARTICLE" not in out
    assert "NATURAL PHOTONICS" not in out
    assert "How a single-pixel camera works" in out
    assert "Understanding compressed sensing" in out
