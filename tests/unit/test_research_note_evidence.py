from copy import deepcopy
from pathlib import Path

import pytest

from kb.research_note_evidence import (
    RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION,
    auto_resolve_research_note_source_state,
    build_research_note_evidence_context,
    capture_research_note_source_context_baseline,
    migrate_research_note_source_state_paths,
    record_research_note_source_baseline_check,
    repair_research_note_source_state,
    research_note_evidence_maintenance_due,
    research_note_evidence_resolution_mode,
    research_note_source_state_dependency_count,
    resolve_research_note_source_baseline_check,
    review_research_note_source_state,
    source_content_fingerprint,
)
from kb.source_blocks import load_source_blocks


def test_source_path_migration_is_exact_audited_and_idempotent() -> None:
    state = {
        "links": [
            {
                "kind": "source",
                "source_path": "file:///F:/Library/Old%20Paper.pdf?download=1#selection",
                "source_name": "Old Paper.pdf",
                "evidence_quote": "Keep the saved evidence unchanged.",
            },
            {
                "kind": "source",
                "source_path": "kb-source/0/Old Paper/assets/table.md",
                "source_name": "A stable paper title",
            },
            {"kind": "source", "source_path": "F:/Library/Other.pdf"},
            {"kind": "answer", "source_path": "F:/Library/Old Paper.pdf"},
        ]
    }

    migrated, count = migrate_research_note_source_state_paths(
        state,
        exact_path_migrations={"F:/Library/Old Paper.pdf": "F:/Library/New Paper.pdf"},
        prefix_path_migrations=[
            ("kb-source/0/Old Paper", "kb-source/0/New Paper"),
        ],
        migrated_at=1_800_000_000,
    )

    assert count == 2
    assert state["links"][0]["source_path"].startswith("file:///")
    assert migrated["links"][0]["source_path"] == "F:/Library/New Paper.pdf"
    assert migrated["links"][0]["source_name"] == "New Paper.pdf"
    assert migrated["links"][0]["evidence_quote"] == "Keep the saved evidence unchanged."
    assert migrated["links"][0]["source_path_migration_history"][-1] == {
        "version": 1,
        "migrated_at": 1_800_000_000,
        "reason": "library_source_renamed",
        "from": "file:///F:/Library/Old%20Paper.pdf?download=1#selection",
        "to": "F:/Library/New Paper.pdf",
    }
    assert migrated["links"][1]["source_path"] == "kb-source/0/New Paper/assets/table.md"
    assert migrated["links"][1]["source_name"] == "A stable paper title"
    assert migrated["links"][2]["source_path"] == "F:/Library/Other.pdf"
    assert migrated["links"][3]["source_path"] == "F:/Library/Old Paper.pdf"

    repeated, repeated_count = migrate_research_note_source_state_paths(
        migrated,
        exact_path_migrations={"F:/Library/Old Paper.pdf": "F:/Library/New Paper.pdf"},
        prefix_path_migrations=[("kb-source/0/Old Paper", "kb-source/0/New Paper")],
        migrated_at=1_800_000_001,
    )
    assert repeated_count == 0
    assert repeated == migrated


def test_source_dependency_count_is_explicit_and_skips_answer_links() -> None:
    state = {
        "links": [
            {"kind": "source", "source_path": "file:///F:/Library/Paper.pdf#page=2"},
            {"kind": "source", "source_path": "kb-source/0/Paper/assets/table.md"},
            {"kind": "source", "source_path": "kb-source/0/Paper-Other/table.md"},
            {"kind": "answer", "source_path": "F:/Library/Paper.pdf"},
        ]
    }

    assert research_note_source_state_dependency_count(
        state,
        exact_source_paths=["F:/Library/Paper.pdf"],
        prefix_source_paths=["kb-source/0/Paper"],
    ) == 2


def test_source_preservation_migration_keeps_the_paper_display_name() -> None:
    migrated, count = migrate_research_note_source_state_paths(
        {
            "links": [{
                "kind": "source",
                "source_path": "F:/Library/Paper.pdf",
                "source_name": "Paper.pdf",
            }]
        },
        exact_path_migrations={"F:/Library/Paper.pdf": "F:/Markdown/Paper/Paper.en.md"},
        reason="library_source_preserved_before_delete",
        update_source_name=False,
        migrated_at=1_800_000_002,
    )

    assert count == 1
    assert migrated["links"][0]["source_path"] == "F:/Markdown/Paper/Paper.en.md"
    assert migrated["links"][0]["source_name"] == "Paper.pdf"


def _write_source(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "paper.md"
    path.write_text(body, encoding="utf-8")
    return path


def _fixture_source(tmp_path: Path) -> tuple[Path, dict]:
    path = _write_source(
        tmp_path,
        """# Example paper

<!-- kb_page: 2 -->

## Methods

The detector was calibrated before every acquisition.

Photon arrival is a stochastic process governed by Poisson statistics.

Dark counts were measured with the shutter closed.
""",
    )
    blocks = load_source_blocks(path)
    target = next(block for block in blocks if "Photon arrival" in str(block.get("text") or ""))
    return path, target


def test_exact_block_returns_surrounding_context_and_canonical_reader_target(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        source_name="Example paper",
        heading_path="Old heading",
        evidence_quote="Photon arrival is a stochastic process",
        block_id=str(target["block_id"]),
    )

    assert result["status"] == "exact"
    assert result["reason"] == "exact_block"
    assert result["can_open_exact"] is True
    assert "Photon arrival" in result["target"]["text"]
    assert "calibrated" in result["before"][0]["text"]
    assert "Dark counts" in result["after"][0]["text"]
    assert result["page_start"] == 2
    assert result["reader_open"]["blockId"] == target["block_id"]
    assert result["reader_open"]["strictLocate"] is True
    snapshot = result["context_snapshot"]
    assert snapshot["version"] == 1
    assert snapshot["source_fingerprint"] == result["source_fingerprint"]
    assert "calibrated" in snapshot["before"][0]["text"]
    assert "Photon arrival" in snapshot["target"]["text"]
    assert "Dark counts" in snapshot["after"][0]["text"]
    assert "markdown" not in snapshot["target"]


def test_current_context_baseline_keeps_saved_evidence_and_records_audit(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)
    context = build_research_note_evidence_context(
        md_path=path,
        source_path="library/paper.pdf",
        evidence_quote=target["text"],
        block_id=target["block_id"],
        anchor_id=target["anchor_id"],
    )
    state = {
        "version": 1,
        "links": [
            {
                "kind": "source",
                "source_path": "library/paper.pdf",
                "block_id": target["block_id"],
                "anchor_id": target["anchor_id"],
                "anchor_kind": target["kind"],
                "heading_path": target["heading_path"],
                "evidence_quote": target["text"],
            }
        ],
        "unrelated": {"keep": True},
    }
    state = record_research_note_source_baseline_check(
        state,
        source_index=0,
        status="skipped",
        reason="candidate",
        source_fingerprint=context["source_fingerprint"],
        checked_at=1_799_999_900,
    )

    captured = capture_research_note_source_context_baseline(
        state,
        source_index=0,
        source_path="library/paper.pdf",
        target=context["target"],
        source_fingerprint=context["source_fingerprint"],
        context_snapshot=context["context_snapshot"],
    )

    assert "evidence_context_snapshot" not in state["links"][0]
    assert captured["unrelated"] == {"keep": True}
    link = captured["links"][0]
    assert link["block_id"] == target["block_id"]
    assert link["anchor_id"] == target["anchor_id"]
    assert link["evidence_quote"] == target["text"]
    assert link["evidence_context_snapshot"]["before"][0]["text"].startswith("The detector")
    assert link["evidence_verification"] == {
        "status": "exact",
        "reason": "baseline_backfilled",
        "verified_at": link["evidence_context_snapshot"]["captured_at"],
        "source_fingerprint": context["source_fingerprint"],
    }
    audit = link["evidence_context_history"][-1]
    assert audit["action"] == "baseline_backfilled"
    assert audit["locator"]["block_id"] == target["block_id"]
    assert "evidence_quote" not in audit["locator"]
    assert "evidence_baseline_check" not in link
    assert link["evidence_baseline_check_history"][-1]["status"] == "resolved"
    assert link["evidence_baseline_check_history"][-1]["reason"] == "baseline_created"


def test_baseline_check_audit_is_idempotent_bounded_and_resolvable() -> None:
    state = {
        "links": [{
            "kind": "source",
            "source_path": "library/example.pdf",
            "block_id": "block-1",
            "anchor_id": "anchor-1",
            "evidence_quote": "Saved evidence.",
        }]
    }
    checked = record_research_note_source_baseline_check(
        state,
        source_index=0,
        status="skipped",
        reason="source_changed",
        source_fingerprint="sha256:new",
        checked_at=1_800_000_000,
    )
    repeated = record_research_note_source_baseline_check(
        checked,
        source_index=0,
        status="skipped",
        reason="source_changed",
        source_fingerprint="sha256:new",
        checked_at=1_800_000_100,
    )

    assert repeated == checked
    assert repeated["links"][0]["evidence_baseline_check"]["checked_at"] == 1_800_000_000
    assert repeated["links"][0]["evidence_baseline_check"]["resolution_mode"] == "user_decision"
    assert len(repeated["links"][0]["evidence_baseline_check_history"]) == 1

    changed = repeated
    for index in range(14):
        changed = record_research_note_source_baseline_check(
            changed,
            source_index=0,
            status="failed" if index % 2 else "skipped",
            reason=f"outcome_{index}",
            checked_at=1_800_000_200 + index,
        )
    assert len(changed["links"][0]["evidence_baseline_check_history"]) == 12

    resolved = resolve_research_note_source_baseline_check(
        changed,
        source_index=0,
        reason="source_review_confirmed",
        source_fingerprint="sha256:current",
        resolved_at=1_800_000_300,
    )
    link = resolved["links"][0]
    assert "evidence_baseline_check" not in link
    assert len(link["evidence_baseline_check_history"]) == 12
    assert link["evidence_baseline_check_history"][-1] == {
        "version": 1,
        "resolved_at": 1_800_000_300.0,
        "status": "resolved",
        "reason": "source_review_confirmed",
        "source_fingerprint": "sha256:current",
        "previous_status": "failed",
        "previous_reason": "outcome_13",
        "locator": {
            "block_id": "block-1",
            "anchor_id": "anchor-1",
            "anchor_kind": "",
            "heading_path": "",
            "page_start": None,
            "page_end": None,
            "line_start": None,
            "line_end": None,
        },
    }


def test_evidence_maintenance_due_respects_version_and_persisted_retry_schedule() -> None:
    local_link = {
        "kind": "source",
        "source_path": "library/example.pdf",
        "block_id": "block-1",
        "evidence_quote": "Saved evidence long enough to verify.",
    }
    assert research_note_evidence_maintenance_due({"links": [local_link]}, now=100.0) is True

    decision_state = record_research_note_source_baseline_check(
        {"links": [local_link]},
        source_index=0,
        status="skipped",
        reason="candidate",
        checked_at=90.0,
        maintenance_version=RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION,
    )
    assert research_note_evidence_maintenance_due(decision_state, now=100.0) is False

    legacy_context_decision = deepcopy(decision_state)
    legacy_context_decision["links"][0]["evidence_context_snapshot"] = {
        "version": 1,
        "captured_at": 80.0,
        "source_fingerprint": "sha256:saved",
        "reason": "exact_block",
        "before": [],
        "target": {"block_id": "block-1", "text": "Saved evidence long enough to verify."},
        "after": [],
    }
    legacy_context_decision["links"][0]["evidence_baseline_check"]["maintenance_version"] = (
        RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION - 1
    )
    assert research_note_evidence_maintenance_due(legacy_context_decision, now=100.0) is True
    legacy_context_decision["links"][0]["evidence_baseline_check"]["maintenance_version"] = (
        RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION
    )
    assert research_note_evidence_maintenance_due(legacy_context_decision, now=100.0) is False
    legacy_context_decision["links"][0]["evidence_baseline_check"].update(
        {
            "status": "failed",
            "reason": "verification_failed",
            "retry_after": 120.0,
        }
    )
    assert research_note_evidence_maintenance_due(legacy_context_decision, now=100.0) is False
    assert research_note_evidence_maintenance_due(legacy_context_decision, now=120.0) is True

    retry_state = record_research_note_source_baseline_check(
        {"links": [local_link]},
        source_index=0,
        status="failed",
        reason="verification_failed",
        checked_at=90.0,
        maintenance_version=RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION,
        retry_count=2,
        retry_after=120.0,
    )
    retry_check = retry_state["links"][0]["evidence_baseline_check"]
    assert retry_check["retry_count"] == 2
    assert retry_check["retry_after"] == 120.0
    assert research_note_evidence_maintenance_due(retry_state, now=100.0) is False
    assert research_note_evidence_maintenance_due(retry_state, now=120.0) is True

    terminal_state = record_research_note_source_baseline_check(
        {"links": [local_link]},
        source_index=0,
        status="skipped",
        reason="source_lost",
        checked_at=130.0,
        maintenance_version=RESEARCH_NOTE_EVIDENCE_MAINTENANCE_VERSION,
        retry_count=6,
        terminal=True,
    )
    terminal_check = terminal_state["links"][0]["evidence_baseline_check"]
    assert terminal_check["terminal"] is True
    assert terminal_check["terminal_at"] == 130.0
    assert "retry_after" not in terminal_check
    assert research_note_evidence_maintenance_due(terminal_state, now=10_000.0) is False

    external_state = {"links": [{"kind": "source", "evidence_quote": "External evidence"}]}
    assert research_note_evidence_maintenance_due(external_state, now=100.0) is False


def test_evidence_resolution_mode_is_allowlisted_and_fails_safe_to_system() -> None:
    assert research_note_evidence_resolution_mode("candidate") == "user_decision"
    assert research_note_evidence_resolution_mode("source_changed") == "user_decision"
    assert research_note_evidence_resolution_mode("verification_failed") == "system_auto"
    assert research_note_evidence_resolution_mode("new_internal_failure") == "system_auto"
    assert research_note_evidence_resolution_mode("source_lost") == "source_boundary"
    assert research_note_evidence_resolution_mode("candidate", terminal=True) == "source_boundary"


def test_current_context_baseline_rejects_existing_changed_or_invalid_evidence(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)
    context = build_research_note_evidence_context(
        md_path=path,
        source_path="library/paper.pdf",
        evidence_quote=target["text"],
        block_id=target["block_id"],
        anchor_id=target["anchor_id"],
    )
    base_link = {
        "kind": "source",
        "source_path": "library/paper.pdf",
        "block_id": target["block_id"],
        "anchor_id": target["anchor_id"],
        "evidence_quote": target["text"],
    }
    kwargs = {
        "source_index": 0,
        "source_path": "library/paper.pdf",
        "target": context["target"],
        "source_fingerprint": context["source_fingerprint"],
        "context_snapshot": context["context_snapshot"],
    }

    with pytest.raises(ValueError, match="already exists"):
        capture_research_note_source_context_baseline(
            {"links": [{**base_link, "evidence_context_snapshot": context["context_snapshot"]}]},
            **kwargs,
        )
    with pytest.raises(ValueError, match="source changed"):
        capture_research_note_source_context_baseline(
            {
                "links": [{
                    **base_link,
                    "evidence_verification": {
                        "status": "exact",
                        "source_fingerprint": "sha256:older",
                    },
                }]
            },
            **kwargs,
        )
    with pytest.raises(ValueError, match="invalid"):
        capture_research_note_source_context_baseline(
            {"links": [{**base_link, "evidence_verification": {"status": "invalid"}}]},
            **kwargs,
        )


def test_unique_quote_can_recover_an_exact_location_after_ids_change(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        source_name="Example paper",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id="stale-block-id",
    )

    assert result["status"] == "exact"
    assert result["reason"] == "quote_relocated"
    assert result["reader_open"]["blockId"] == target["block_id"]


def test_positional_block_id_is_not_trusted_when_its_text_no_longer_matches(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)
    wrong_block = next(
        block
        for block in load_source_blocks(path)
        if "calibrated" in str(block.get("text") or "")
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id=str(wrong_block["block_id"]),
    )

    assert result["status"] == "exact"
    assert result["reason"] == "quote_relocated"
    assert result["target"]["block_id"] == target["block_id"]


def test_fuzzy_match_is_only_a_review_candidate(tmp_path: Path) -> None:
    path, _target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        source_name="Example paper",
        evidence_quote="stochastic photon arrivals follow Poisson behavior",
        block_id="stale-block-id",
    )

    assert result["status"] == "candidate"
    assert result["reason"] == "saved_target_missing"
    assert result["can_open_exact"] is False
    assert result["reader_open"] is None
    assert result["candidates"]
    assert result["candidates"][0]["reader_open"]["strictLocate"] is True


def test_fuzzy_candidates_exclude_headings_and_unrelated_section_neighbors(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Detector paper

<!-- kb_page: 4 -->

## Results

The coded exposure follows detector calibration.

SCI compresses a short video into one coded measurement.

The reconstruction is evaluated on the same sequence.
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/detector.pdf",
        source_name="Detector paper",
        heading_path="Detector paper / Results",
        evidence_quote="SCI records one compressed video measurement.",
        block_id="stale-block-id",
        anchor_kind="blockquote",
    )

    assert result["status"] == "candidate"
    assert [(candidate["kind"], candidate["text"]) for candidate in result["candidates"]] == [
        ("paragraph", "SCI compresses a short video into one coded measurement."),
    ]


def test_section_proximity_alone_cannot_create_a_candidate(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Detector paper

## Results

The coded exposure follows detector calibration.

The reconstruction is evaluated on the same sequence.
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/detector.pdf",
        heading_path="Detector paper / Results",
        evidence_quote="Quantum efficiency reached ninety percent after cooling the sensor.",
        block_id="stale-block-id",
        anchor_kind="paragraph",
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "no_match"
    assert result["candidates"] == []


@pytest.mark.parametrize(
    ("evidence_quote", "anchor_kind", "expected_text"),
    [
        ("| Photon efficiency | 90 percent |", "table", "Photon efficiency | 92 percent"),
        ("E = m c^2 + noise", "equation", "E = m c^2 + background"),
        ("Detector output", "figure", "Detector response"),
        (
            "Figure 1. Detector response improves under low light.",
            "paragraph",
            "Figure 1. Detector sensitivity improves under low light.",
        ),
    ],
)
def test_structured_and_caption_evidence_remain_reviewable_candidates(
    tmp_path: Path,
    evidence_quote: str,
    anchor_kind: str,
    expected_text: str,
) -> None:
    path = _write_source(
        tmp_path,
        """# Evidence paper

<!-- kb_page: 5 -->

## Results

Table 1. Detector measurements.

| Metric | Value |
|---|---|
| Photon efficiency | 92 percent |

$$ E = m c^2 + background \\tag{1} $$

![Detector response](assets/figure1.png)

Figure 1. Detector sensitivity improves under low light.
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/evidence.pdf",
        heading_path="Evidence paper / Results",
        evidence_quote=evidence_quote,
        block_id="stale-block-id",
        anchor_kind=anchor_kind,
    )

    assert result["status"] == "candidate"
    assert len(result["candidates"]) == 1
    assert expected_text in result["candidates"][0]["text"]
    assert result["candidates"][0]["kind"] == anchor_kind


def test_changed_short_selection_can_match_inside_a_long_evidence_block(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Evidence paper

## Results

After calibration, the cooled scientific detector measured photon arrivals under low light while the control channel remained closed and the acquisition software recorded every exposure for later verification.
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/evidence.pdf",
        heading_path="Evidence paper / Results",
        evidence_quote="The detector recorded photon arrivals under low light.",
        block_id="stale-block-id",
        anchor_kind="paragraph",
    )

    assert result["status"] == "candidate"
    assert len(result["candidates"]) == 1
    assert "measured photon arrivals under low light" in result["candidates"][0]["text"]


def test_exact_quote_relocation_ignores_a_same_text_heading_for_paragraph_evidence(
    tmp_path: Path,
) -> None:
    path = _write_source(
        tmp_path,
        """# Evidence paper

## The detector response remains stable.

The detector response remains stable.
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/evidence.pdf",
        evidence_quote="The detector response remains stable.",
        block_id="stale-block-id",
        anchor_kind="paragraph",
    )

    assert result["status"] == "exact"
    assert result["reason"] == "quote_relocated"
    assert result["target"]["kind"] == "paragraph"


def test_saved_heading_kind_keeps_heading_relocation_reviewable(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Evidence paper

## Detection findings
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/evidence.pdf",
        heading_path="Evidence paper",
        evidence_quote="Detector findings",
        block_id="stale-block-id",
        anchor_kind="heading",
    )

    assert result["status"] == "candidate"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["kind"] == "heading"
    assert result["candidates"][0]["text"] == "Detection findings"


def test_ambiguous_exact_quote_requires_candidate_review(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Repeated evidence

## First experiment

The same calibration sentence appears here verbatim.

## Second experiment

The same calibration sentence appears here verbatim.
""",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id="stale-block-id",
    )

    assert result["status"] == "candidate"
    assert result["reason"] == "ambiguous_quote"
    assert len(result["candidates"]) == 2
    assert result["reader_open"] is None


def test_saved_context_strictly_disambiguates_a_repeated_exact_quote(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Repeated evidence

<!-- kb_page: 2 -->

## First experiment

The first detector was calibrated against a stable reference before acquisition.

The same calibration sentence appears here verbatim.

The first run then measured dark counts with the shutter closed.

<!-- kb_page: 3 -->

## Second experiment

The second detector used a different reference before acquisition.

The same calibration sentence appears here verbatim.

The second run then measured saturation under bright illumination.
""",
    )
    blocks = load_source_blocks(path)
    first_target = next(
        block
        for block in blocks
        if "same calibration sentence" in str(block.get("text") or "")
        and int(block.get("page_start") or 0) == 2
    )
    saved = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id=str(first_target["block_id"]),
        anchor_id=str(first_target["anchor_id"]),
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id="stale-block-id",
        anchor_id="stale-anchor-id",
        saved_context_snapshot=saved["context_snapshot"],
    )

    assert result["status"] == "exact"
    assert result["reason"] == "context_disambiguated"
    assert result["target"]["block_id"] == first_target["block_id"]
    assert result["context_disambiguation"]["candidate_count"] == 2
    assert result["context_disambiguation"]["mode"] == "weighted_saved_context"
    assert result["context_disambiguation"]["score_version"] == 1
    assert result["context_disambiguation"]["score_gap"] >= 0.10
    assert set(result["context_disambiguation"]["matched_fields"]) == {
        "target",
        "before",
        "after",
        "heading",
        "page",
    }


def test_saved_context_tolerates_minor_neighbor_drift_when_disambiguating_exact_quote(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Repeated evidence

<!-- kb_page: 2 -->

## First experiment

The first detector was calibrated against a stable reference before acquisition.

The same calibration sentence appears here verbatim.

The first run then measured dark counts with the shutter closed.

<!-- kb_page: 3 -->

## Second experiment

The second detector used a different reference before acquisition.

The same calibration sentence appears here verbatim.

The second run then measured saturation under bright illumination.
""",
    )
    first_target = next(
        block
        for block in load_source_blocks(path)
        if "same calibration sentence" in str(block.get("text") or "")
        and int(block.get("page_start") or 0) == 2
    )
    saved = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id=str(first_target["block_id"]),
        anchor_id=str(first_target["anchor_id"]),
    )
    path.write_text(
        """# Repeated evidence

<!-- kb_page: 2 -->

## First experiment

The first detector was carefully calibrated against a stable reference before acquisition.

The same calibration sentence appears here verbatim.

The first run then carefully measured dark counts with the shutter closed.

<!-- kb_page: 3 -->

## Second experiment

The second detector used a different reference before acquisition.

The same calibration sentence appears here verbatim.

The second run then measured saturation under bright illumination.
""",
        encoding="utf-8",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id="stale-block-id",
        anchor_id="stale-anchor-id",
        saved_context_snapshot=saved["context_snapshot"],
    )

    assert result["status"] == "exact"
    assert result["reason"] == "context_disambiguated"
    assert result["target"]["page_start"] == 2
    assert result["context_disambiguation"]["mode"] == "weighted_saved_context"
    assert result["context_disambiguation"]["score_gap"] >= 0.10
    assert {"before", "after"}.issubset(result["context_disambiguation"]["matched_fields"])


def test_saved_context_keeps_a_true_repeated_context_ambiguous(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Repeated evidence

<!-- kb_page: 4 -->

## Shared experiment

The shared detector context appears immediately before the evidence.

The same calibration sentence appears here verbatim.

The shared detector context appears immediately after the evidence.

The shared detector context appears immediately before the evidence.

The same calibration sentence appears here verbatim.

The shared detector context appears immediately after the evidence.
""",
    )
    blocks = load_source_blocks(path)
    targets = [block for block in blocks if "same calibration sentence" in str(block.get("text") or "")]
    saved = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id=str(targets[0]["block_id"]),
        anchor_id=str(targets[0]["anchor_id"]),
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id="stale-block-id",
        saved_context_snapshot=saved["context_snapshot"],
    )

    assert result["status"] == "candidate"
    assert result["reason"] == "ambiguous_quote"
    assert len(result["candidates"]) == 2
    assert all(candidate["before"] and candidate["after"] for candidate in result["candidates"])
    assert all(
        set(candidate["context_match"]["matched_fields"]) == {"target", "before", "after", "heading", "page"}
        for candidate in result["candidates"]
    )
    assert all(not candidate["context_match"].get("best_match") for candidate in result["candidates"])


def test_fuzzy_candidates_are_ranked_by_saved_context_without_becoming_exact(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Detector paper

<!-- kb_page: 2 -->

## Low-light results

The detector was calibrated against the stable low-light reference before acquisition.

The detector recorded photon arrivals under low light.

Dark counts were measured with the shutter closed after acquisition.
""",
    )
    target = next(
        block
        for block in load_source_blocks(path)
        if "recorded photon arrivals" in str(block.get("text") or "")
    )
    saved = build_research_note_evidence_context(
        md_path=path,
        source_path="library/detector.pdf",
        evidence_quote="The detector recorded photon arrivals under low light.",
        block_id=str(target["block_id"]),
        anchor_id=str(target["anchor_id"]),
    )
    path.write_text(
        """# Detector paper

<!-- kb_page: 2 -->

## Low-light results

The detector was calibrated against the stable low-light reference before acquisition.

The detector measured photon arrivals under low light.

Dark counts were measured with the shutter closed after acquisition.

<!-- kb_page: 7 -->

## Bright-light controls

The instrument was prepared for the separate bright-light control.

The detector observed photon arrivals under low light.

Saturation was measured under bright illumination after acquisition.
""",
        encoding="utf-8",
    )

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/detector.pdf",
        evidence_quote="The detector recorded photon arrivals under low light.",
        block_id="stale-block-id",
        saved_context_snapshot=saved["context_snapshot"],
    )

    assert result["status"] == "candidate"
    assert result["reason"] == "saved_target_missing"
    assert result["reader_open"] is None
    assert result["can_open_exact"] is False
    assert len(result["candidates"]) >= 2
    best = result["candidates"][0]
    assert best["rank"] == 1
    assert best["page_start"] == 2
    assert best["context_match"]["best_match"] is True
    assert best["context_match"]["score_gap"] >= 0.10
    assert {"before", "after", "heading", "page"}.issubset(best["context_match"]["matched_fields"])


def test_unrelated_quote_does_not_create_a_false_location(tmp_path: Path) -> None:
    path, _target = _fixture_source(tmp_path)

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="xylophone zeppelin kumquat",
        block_id="stale-block-id",
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "no_match"
    assert result["candidates"] == []


def test_surrounding_context_does_not_cross_a_section_heading(tmp_path: Path) -> None:
    path = _write_source(
        tmp_path,
        """# Paper

## Earlier section

This paragraph belongs to the earlier section.

## Target section

This is the first evidence paragraph in the target section.
""",
    )
    blocks = load_source_blocks(path)
    target = next(block for block in blocks if "first evidence" in str(block.get("text") or ""))

    result = build_research_note_evidence_context(
        md_path=path,
        source_path="library/sections.pdf",
        evidence_quote="This is the first evidence paragraph in the target section.",
        block_id=str(target["block_id"]),
    )

    assert result["status"] == "exact"
    assert result["before"] == []


def test_source_fingerprint_marks_a_changed_conversion(tmp_path: Path) -> None:
    path, target = _fixture_source(tmp_path)
    fingerprint = source_content_fingerprint(path)

    unchanged = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id=str(target["block_id"]),
        source_fingerprint=fingerprint,
    )
    path.write_text(path.read_text(encoding="utf-8") + "\nA new appendix line.\n", encoding="utf-8")
    changed = build_research_note_evidence_context(
        md_path=path,
        source_path="library/example.pdf",
        evidence_quote="Photon arrival is a stochastic process governed by Poisson statistics.",
        block_id=str(target["block_id"]),
        source_fingerprint=fingerprint,
    )

    assert unchanged["source_fingerprint"] == fingerprint
    assert unchanged["source_changed"] is False
    assert changed["source_fingerprint"] != fingerprint
    assert changed["source_changed"] is True


def test_confirmed_candidate_repairs_one_locator_and_keeps_an_audit_trail() -> None:
    source_state = {
        "version": 1,
        "links": [
            {
                "kind": "source",
                "label": "Detector evidence",
                "source_path": "library/example.pdf",
                "capture_id": "capture-1",
                "capture_kind": "selection",
                "block_id": "old-block",
                "anchor_id": "old-anchor",
                "anchor_kind": "paragraph",
                "heading_path": "Old section",
                "location_label": "Old section · page 1",
                "page_start": 1,
                "evidence_quote": "Old detector sentence.",
                "start_offset": 3,
                "end_offset": 25,
                "occurrence": 2,
                "asset_src": "/kept.png",
            }
        ],
    }
    source_state = record_research_note_source_baseline_check(
        source_state,
        source_index=0,
        status="skipped",
        reason="candidate",
        checked_at=1_799_999_900,
    )
    target = {
        "block_id": "new-block",
        "anchor_id": "new-anchor",
        "kind": "paragraph",
        "heading_path": "Results",
        "page_start": 4,
        "page_end": 4,
        "line_start": 22,
        "line_end": 24,
        "text": "The repaired detector sentence is exact.",
    }
    repaired = repair_research_note_source_state(
        source_state,
        source_index=0,
        source_path="LIBRARY\\example.pdf",
        capture_id="capture-1",
        old_block_id="old-block",
        old_anchor_id="old-anchor",
        old_evidence_quote="Old detector sentence.",
        target=target,
        source_fingerprint="sha256:abc123",
        context_snapshot={
            "version": 1,
            "captured_at": 1_800_000_000,
            "source_fingerprint": "sha256:abc123",
            "reason": "exact_block",
            "before": [{"block_id": "before", "text": "Earlier context."}],
            "target": target,
            "after": [{"block_id": "after", "text": "Later context."}],
        },
        repaired_at=1_800_000_000,
    )

    assert source_state["links"][0]["block_id"] == "old-block"
    link = repaired["links"][0]
    assert link["capture_id"] == "capture-1"
    assert link["capture_kind"] == "selection"
    assert link["asset_src"] == "/kept.png"
    assert link["block_id"] == "new-block"
    assert link["anchor_id"] == "new-anchor"
    assert link["heading_path"] == "Results"
    assert link["page_start"] == 4
    assert link["evidence_quote"] == "The repaired detector sentence is exact."
    assert "location_label" not in link
    assert "start_offset" not in link
    assert "end_offset" not in link
    assert "occurrence" not in link
    assert link["evidence_verification"] == {
        "status": "exact",
        "reason": "confirmed_candidate",
        "verified_at": 1_800_000_000.0,
        "source_fingerprint": "sha256:abc123",
    }
    audit = link["locator_repair_history"][-1]
    assert audit["from"]["block_id"] == "old-block"
    assert audit["to"]["block_id"] == "new-block"
    assert audit["repaired_at"] == 1_800_000_000.0
    assert audit["previous_context_snapshot"] is None
    assert audit["reviewed_context_snapshot"]["target"]["block_id"] == "new-block"
    assert link["evidence_context_snapshot"]["before"][0]["text"] == "Earlier context."
    assert "evidence_baseline_check" not in link
    assert link["evidence_baseline_check_history"][-1]["reason"] == "candidate_repaired"


def test_locator_repair_rejects_a_source_that_changed_after_reader_opened() -> None:
    state = {
        "links": [
            {
                "source_path": "library/example.pdf",
                "capture_id": "newer-capture",
                "block_id": "newer-block",
                "evidence_quote": "Newer evidence.",
            }
        ]
    }

    with pytest.raises(ValueError, match="source changed"):
        repair_research_note_source_state(
            state,
            source_index=0,
            source_path="library/example.pdf",
            capture_id="older-capture",
            old_block_id="older-block",
            old_evidence_quote="Older evidence.",
            target={"block_id": "candidate", "text": "Candidate evidence."},
            source_fingerprint="sha256:abc123",
        )


def test_exact_quote_relocation_is_auto_resolved_without_rewriting_saved_quote() -> None:
    saved_quote = "The detector remained linear across the calibrated range."
    state = record_research_note_source_baseline_check(
        {
            "links": [{
                "kind": "source",
                "source_path": "library/example.pdf",
                "block_id": "old-block",
                "anchor_id": "old-anchor",
                "evidence_quote": saved_quote,
                "start_offset": 4,
                "end_offset": 59,
            }],
        },
        source_index=0,
        status="skipped",
        reason="location_changed",
        checked_at=1_799_999_900,
    )
    target = {
        "block_id": "new-block",
        "anchor_id": "new-anchor",
        "kind": "paragraph",
        "heading_path": "Results",
        "page_start": 6,
        "text": f"In repeated trials, {saved_quote} No saturation was observed.",
    }
    snapshot = {
        "version": 1,
        "captured_at": 1_800_000_000,
        "source_fingerprint": "sha256:new",
        "reason": "quote_relocated",
        "before": [],
        "target": target,
        "after": [],
    }

    resolved = auto_resolve_research_note_source_state(
        state,
        source_index=0,
        source_path="LIBRARY\\example.pdf",
        target=target,
        source_fingerprint="sha256:new",
        context_snapshot=snapshot,
        match_reason="quote_relocated",
        action="exact_quote_auto_relocated",
        resolved_at=1_800_000_000,
    )

    link = resolved["links"][0]
    assert link["block_id"] == "new-block"
    assert link["anchor_id"] == "new-anchor"
    assert link["evidence_quote"] == saved_quote
    assert "start_offset" not in link
    assert "end_offset" not in link
    assert link["evidence_verification"] == {
        "status": "exact",
        "reason": "exact_quote_auto_relocated",
        "verified_at": 1_800_000_000.0,
        "source_fingerprint": "sha256:new",
    }
    audit = link["evidence_auto_resolution_history"][-1]
    assert audit["action"] == "exact_quote_auto_relocated"
    assert audit["match_reason"] == "quote_relocated"
    assert audit["from"]["block_id"] == "old-block"
    assert audit["to"]["block_id"] == "new-block"
    assert "evidence_baseline_check" not in link
    assert link["evidence_baseline_check_history"][-1]["reason"] == "exact_quote_auto_relocated"


def test_source_change_review_can_confirm_or_invalidate_without_rewriting_evidence() -> None:
    old_snapshot = {
        "version": 1,
        "captured_at": 1_700_000_000,
        "source_fingerprint": "sha256:old",
        "reason": "exact_block",
        "before": [{"block_id": "before-1", "text": "Old preceding context."}],
        "target": {
            "block_id": "block-1",
            "anchor_id": "anchor-1",
            "kind": "paragraph",
            "text": "The saved detector evidence remains exact.",
        },
        "after": [{"block_id": "after-1", "text": "Old following context."}],
    }
    state = {
        "links": [
            {
                "kind": "source",
                "source_path": "library/example.pdf",
                "capture_id": "capture-1",
                "block_id": "block-1",
                "anchor_id": "anchor-1",
                "anchor_kind": "paragraph",
                "heading_path": "Results",
                "evidence_quote": "The saved detector evidence remains exact.",
                "evidence_verification": {
                    "status": "exact",
                    "reason": "confirmed_candidate",
                    "verified_at": 1_700_000_000.0,
                    "source_fingerprint": "sha256:old",
                },
                "evidence_context_snapshot": old_snapshot,
            }
        ]
    }
    state = record_research_note_source_baseline_check(
        state,
        source_index=0,
        status="skipped",
        reason="source_changed",
        source_fingerprint="sha256:new",
        checked_at=1_799_999_950,
    )
    target = {
        "block_id": "block-1",
        "anchor_id": "anchor-1",
        "kind": "paragraph",
        "heading_path": "Results",
        "text": "The saved detector evidence remains exact.",
    }
    current_snapshot = {
        "version": 1,
        "captured_at": 1_800_000_000,
        "source_fingerprint": "sha256:new",
        "reason": "exact_block",
        "before": [{"block_id": "before-1", "text": "Updated preceding context."}],
        "target": target,
        "after": [{"block_id": "after-1", "text": "Old following context."}],
    }

    invalidated = review_research_note_source_state(
        state,
        source_index=0,
        source_path="LIBRARY\\example.pdf",
        decision="invalid",
        capture_id="capture-1",
        old_block_id="block-1",
        old_anchor_id="anchor-1",
        old_evidence_quote="The saved detector evidence remains exact.",
        expected_source_fingerprint="sha256:old",
        target=target,
        source_fingerprint="sha256:new",
        context_snapshot=current_snapshot,
        reviewed_at=1_800_000_000,
    )

    assert state["links"][0]["evidence_verification"]["status"] == "exact"
    invalid_link = invalidated["links"][0]
    assert invalid_link["block_id"] == "block-1"
    assert invalid_link["evidence_quote"] == "The saved detector evidence remains exact."
    assert invalid_link["evidence_verification"] == {
        "status": "invalid",
        "reason": "marked_invalid",
        "verified_at": 1_800_000_000.0,
        "source_fingerprint": "sha256:new",
    }
    first_audit = invalid_link["evidence_review_history"][0]
    assert first_audit["decision"] == "invalid"
    assert first_audit["previous_verification"]["verified_at"] == 1_700_000_000.0
    assert first_audit["saved_locator"]["block_id"] == "block-1"
    assert first_audit["source_fingerprint_before"] == "sha256:old"
    assert first_audit["source_fingerprint_after"] == "sha256:new"
    assert first_audit["previous_context_snapshot"]["before"][0]["text"] == "Old preceding context."
    assert first_audit["reviewed_context_snapshot"]["before"][0]["text"] == "Updated preceding context."
    assert invalid_link["evidence_context_snapshot"] == old_snapshot
    assert "evidence_baseline_check" not in invalid_link
    assert invalid_link["evidence_baseline_check_history"][-1]["reason"] == "source_review_invalid"

    confirmed = review_research_note_source_state(
        invalidated,
        source_index=0,
        source_path="library/example.pdf",
        decision="confirmed",
        capture_id="capture-1",
        old_block_id="block-1",
        old_anchor_id="anchor-1",
        old_evidence_quote="The saved detector evidence remains exact.",
        expected_source_fingerprint="sha256:new",
        target=target,
        source_fingerprint="sha256:new",
        context_snapshot=current_snapshot,
        reviewed_at=1_800_000_100,
    )
    confirmed_link = confirmed["links"][0]
    assert confirmed_link["evidence_verification"]["status"] == "exact"
    assert confirmed_link["evidence_verification"]["reason"] == "confirmed_source_change"
    assert [row["decision"] for row in confirmed_link["evidence_review_history"]] == [
        "invalid",
        "confirmed",
    ]
    assert "previous_context_snapshot" not in confirmed_link["evidence_review_history"][0]
    assert confirmed_link["evidence_context_snapshot"]["before"][0]["text"] == "Updated preceding context."


def test_candidate_can_be_invalidated_without_adopting_its_changed_locator() -> None:
    saved_quote = "The saved detector claim is no longer supported."
    state = {
        "links": [
            {
                "source_path": "library/example.pdf",
                "capture_id": "capture-1",
                "block_id": "stale-block",
                "anchor_id": "stale-anchor",
                "evidence_quote": saved_quote,
                "evidence_context_snapshot": {
                    "version": 1,
                    "captured_at": 1_700_000_000,
                    "source_fingerprint": "sha256:old",
                    "before": [],
                    "target": {
                        "block_id": "stale-block",
                        "anchor_id": "stale-anchor",
                        "text": saved_quote,
                    },
                    "after": [],
                },
                "evidence_baseline_check": {
                    "status": "skipped",
                    "reason": "candidate",
                    "checked_at": 1_799_999_900,
                },
            }
        ]
    }
    candidate = {
        "block_id": "candidate-block",
        "anchor_id": "candidate-anchor",
        "kind": "paragraph",
        "heading_path": "Results",
        "text": "The current source now states a different detector result.",
    }

    invalidated = review_research_note_source_state(
        state,
        source_index=0,
        source_path="LIBRARY\\example.pdf",
        decision="invalid",
        capture_id="capture-1",
        old_block_id="stale-block",
        old_anchor_id="stale-anchor",
        old_evidence_quote=saved_quote,
        expected_source_fingerprint="sha256:old",
        target=candidate,
        source_fingerprint="sha256:new",
        reviewed_at=1_800_000_000,
    )

    link = invalidated["links"][0]
    assert link["block_id"] == "stale-block"
    assert link["anchor_id"] == "stale-anchor"
    assert link["evidence_quote"] == saved_quote
    assert link["evidence_verification"]["status"] == "invalid"
    assert "evidence_baseline_check" not in link
    audit = link["evidence_review_history"][-1]
    assert audit["saved_locator"]["block_id"] == "stale-block"
    assert audit["reviewed_target"]["block_id"] == "candidate-block"
    assert audit["reviewed_target"]["evidence_quote"] == candidate["text"]

    with pytest.raises(ValueError, match="repair required"):
        review_research_note_source_state(
            state,
            source_index=0,
            source_path="library/example.pdf",
            decision="confirmed",
            capture_id="capture-1",
            old_block_id="stale-block",
            old_anchor_id="stale-anchor",
            old_evidence_quote=saved_quote,
            expected_source_fingerprint="sha256:old",
            target=candidate,
            source_fingerprint="sha256:new",
        )


def test_source_change_review_ignores_malformed_legacy_context_lists() -> None:
    target = {
        "block_id": "block-1",
        "anchor_id": "anchor-1",
        "kind": "paragraph",
        "text": "Saved evidence.",
    }
    state = {
        "links": [
            {
                "source_path": "library/example.pdf",
                "block_id": "block-1",
                "anchor_id": "anchor-1",
                "evidence_quote": "Saved evidence.",
                "evidence_verification": {
                    "status": "exact",
                    "source_fingerprint": "sha256:old",
                },
                "evidence_context_snapshot": {
                    "version": 1,
                    "captured_at": 1_700_000_000,
                    "source_fingerprint": "sha256:old",
                    "before": 17,
                    "target": target,
                    "after": {"text": "not a list"},
                },
            }
        ]
    }
    current_snapshot = {
        "version": 1,
        "captured_at": 1_800_000_000,
        "source_fingerprint": "sha256:new",
        "before": [],
        "target": target,
        "after": [],
    }

    reviewed = review_research_note_source_state(
        state,
        source_index=0,
        source_path="library/example.pdf",
        decision="confirmed",
        old_block_id="block-1",
        old_anchor_id="anchor-1",
        old_evidence_quote="Saved evidence.",
        expected_source_fingerprint="sha256:old",
        target=target,
        source_fingerprint="sha256:new",
        context_snapshot=current_snapshot,
        reviewed_at=1_800_000_000,
    )

    link = reviewed["links"][0]
    assert link["evidence_review_history"][-1]["previous_context_snapshot"]["before"] == []
    assert link["evidence_review_history"][-1]["previous_context_snapshot"]["after"] == []
    assert link["evidence_context_snapshot"]["source_fingerprint"] == "sha256:new"


def test_source_change_review_rejects_stale_verification_or_relocated_target() -> None:
    state = {
        "links": [
            {
                "source_path": "library/example.pdf",
                "block_id": "saved-block",
                "evidence_quote": "Saved evidence.",
                "evidence_verification": {
                    "status": "exact",
                    "source_fingerprint": "sha256:newer",
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="verification changed"):
        review_research_note_source_state(
            state,
            source_index=0,
            source_path="library/example.pdf",
            decision="confirmed",
            old_block_id="saved-block",
            old_evidence_quote="Saved evidence.",
            expected_source_fingerprint="sha256:older",
            target={"block_id": "saved-block", "text": "Saved evidence."},
            source_fingerprint="sha256:current",
        )

    with pytest.raises(ValueError, match="repair required"):
        review_research_note_source_state(
            state,
            source_index=0,
            source_path="library/example.pdf",
            decision="confirmed",
            old_block_id="saved-block",
            old_evidence_quote="Saved evidence.",
            expected_source_fingerprint="sha256:newer",
            target={"block_id": "relocated-block", "text": "Saved evidence."},
            source_fingerprint="sha256:current",
        )
