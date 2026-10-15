from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from kb.chat_store import ChatStore


def test_chat_research_notes_api_roundtrip_and_revision_conflict(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Notes")
    conv_id = store.create_conversation("Answers", project_id=project_id)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(
        chat_router,
        "_dangerous_auto_snapshot",
        lambda *args, **kwargs: {"ok": True, "block_operation": False},
    )
    client = TestClient(app)

    created_response = client.post(
        "/api/chat/research-notes",
        json={
            "project_id": project_id,
            "source_conv_id": conv_id,
            "title": "Traceable note",
            "content_markdown": "## Finding\n\nGrounded answer [1].",
            "source_state": {
                "links": [
                    {
                        "kind": "answer",
                        "conversation_id": conv_id,
                        "message_id": 12,
                        "label": "Question",
                    }
                ]
            },
        },
    )
    assert created_response.status_code == 200
    created = created_response.json()
    note_id = created["id"]
    assert created["revision"] == 1

    listed = client.get(f"/api/chat/research-notes?project_id={project_id}")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == note_id
    assert listed.json()[0]["content_markdown"] == ""
    assert listed.json()[0]["source_state"]["links"][0]["message_id"] == 12

    fetched = client.get(f"/api/chat/research-notes/{note_id}")
    assert fetched.status_code == 200
    assert "Grounded answer" in fetched.json()["content_markdown"]

    updated = client.patch(
        f"/api/chat/research-notes/{note_id}",
        json={
            "expected_revision": 1,
            "title": "Renamed note",
            "content_markdown": "Updated content",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["title"] == "Renamed note"

    conflict = client.patch(
        f"/api/chat/research-notes/{note_id}",
        json={"expected_revision": 1, "content_markdown": "Stale"},
    )
    assert conflict.status_code == 409
    assert store.get_research_note(note_id)["content_markdown"] == "Updated content"

    deleted = client.delete(f"/api/chat/research-notes/{note_id}")
    assert deleted.status_code == 200
    assert deleted.json()["auto_backup"]["ok"] is True
    assert client.get(f"/api/chat/research-notes/{note_id}").status_code == 404


def test_chat_research_notes_api_rejects_invalid_scope_and_blank_content(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    client = TestClient(app)

    invalid_scope = client.post(
        "/api/chat/research-notes",
        json={
            "project_id": "missing-project",
            "title": "Note",
            "content_markdown": "Content",
        },
    )
    blank = client.post(
        "/api/chat/research-notes",
        json={"title": "Note", "content_markdown": "   "},
    )

    assert invalid_scope.status_code == 404
    assert blank.status_code == 422


def test_chat_research_note_evidence_context_verifies_source_without_exposing_internal_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router

    md_path = tmp_path / "converted.md"
    md_path.write_text(
        """# Paper

<!-- kb_page: 4 -->

## Results

Context immediately before the evidence.

The measured response remained linear across the tested range.

Context immediately after the evidence.
""",
        encoding="utf-8",
    )
    from kb.source_blocks import load_source_blocks

    target = next(
        block
        for block in load_source_blocks(md_path)
        if "remained linear" in str(block.get("text") or "")
    )
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda _path: str(md_path))
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", lambda *args, **kwargs: md_path)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/chat/research-note/evidence-context",
        json={
            "source_path": "library/paper.pdf",
            "source_name": "Paper",
            "heading_path": "Results",
            "evidence_quote": "The measured response remained linear across the tested range.",
            "block_id": target["block_id"],
            "page_start": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "exact"
    assert payload["page_start"] == 4
    assert payload["reader_open"]["sourcePath"] == "library/paper.pdf"
    assert str(md_path) not in str(payload)
    assert client.post(
        "/api/chat/research-note/evidence-context",
        json={"source_path": "   "},
    ).status_code == 422


def test_chat_research_note_context_baseline_batch_is_safe_audited_and_idempotent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router
    from kb.source_blocks import load_source_blocks

    md_path = tmp_path / "converted.md"
    md_path.write_text(
        """# Paper

<!-- kb_page: 4 -->

## Results

The detector was calibrated before every acquisition.

Photon arrival is a stochastic process governed by Poisson statistics.

Dark counts were measured with the shutter closed.
""",
        encoding="utf-8",
    )
    target = next(
        block
        for block in load_source_blocks(md_path)
        if "Photon arrival" in str(block.get("text") or "")
    )
    exact_link = {
        "kind": "source",
        "label": "Paper",
        "source_path": "library/paper.pdf",
        "capture_id": "capture-exact",
        "block_id": target["block_id"],
        "anchor_id": target["anchor_id"],
        "anchor_kind": target["kind"],
        "heading_path": target["heading_path"],
        "page_start": target["page_start"],
        "evidence_quote": target["text"],
    }
    store = ChatStore(tmp_path / "chat.sqlite3")
    note = store.create_research_note(
        title="Legacy evidence note",
        content_markdown="Original note body",
        source_state={
            "version": 1,
            "links": [
                exact_link,
                {
                    **exact_link,
                    "capture_id": "capture-changed",
                    "evidence_verification": {
                        "status": "exact",
                        "source_fingerprint": "sha256:" + ("0" * 64),
                    },
                },
                {
                    **exact_link,
                    "capture_id": "capture-candidate",
                    "block_id": "stale-block",
                    "anchor_id": "stale-anchor",
                    "evidence_quote": "photon arrivals follow Poisson behavior",
                },
                {
                    **exact_link,
                    "capture_id": "capture-invalid",
                    "evidence_verification": {"status": "invalid"},
                },
                {
                    **exact_link,
                    "capture_id": "capture-unavailable",
                    "source_path": "",
                },
                {
                    **exact_link,
                    "capture_id": "capture-existing",
                    "evidence_context_snapshot": {"version": 1},
                },
                {
                    **exact_link,
                    "capture_id": "capture-location-changed",
                    "block_id": "stale-block",
                },
                {"kind": "answer", "label": "Answer", "message_id": 12},
            ],
        },
    )
    assert note is not None
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda _path: str(md_path))
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", lambda *args, **kwargs: md_path)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    response = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-context-baseline",
        json={"expected_revision": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {
        "total": 7,
        "created": 3,
        "skipped": 4,
        "failed": 0,
        "reasons": {
            "baseline_created": 1,
            "source_change_auto_verified": 1,
            "candidate": 1,
            "invalid": 1,
                "source_lost": 1,
            "already_present": 1,
            "stable_locator_auto_aligned": 1,
        },
    }
    record = payload["note"]
    assert record["revision"] == 2
    assert record["content_markdown"] == "Original note body"
    captured = record["source_state"]["links"][0]
    assert captured["block_id"] == target["block_id"]
    assert captured["anchor_id"] == target["anchor_id"]
    assert captured["evidence_quote"] == target["text"]
    assert captured["evidence_verification"]["reason"] == "baseline_backfilled"
    assert captured["evidence_context_history"][-1]["action"] == "baseline_backfilled"
    assert captured["evidence_context_snapshot"]["target"]["block_id"] == target["block_id"]
    auto_verified = record["source_state"]["links"][1]
    assert auto_verified["evidence_verification"]["reason"] == "source_change_auto_verified"
    assert auto_verified["evidence_context_snapshot"]["target"]["block_id"] == target["block_id"]
    assert auto_verified["evidence_auto_resolution_history"][-1]["action"] == "source_change_auto_verified"
    auto_aligned = record["source_state"]["links"][6]
    assert auto_aligned["block_id"] == target["block_id"]
    assert auto_aligned["evidence_quote"] == target["text"]
    assert auto_aligned["evidence_verification"]["reason"] == "stable_locator_auto_aligned"
    assert auto_aligned["evidence_auto_resolution_history"][-1]["action"] == "stable_locator_auto_aligned"
    for source_index, reason in ((2, "candidate"), (4, "source_lost")):
        checked = record["source_state"]["links"][source_index]["evidence_baseline_check"]
        assert checked["status"] == "skipped"
        assert checked["reason"] == reason
        assert checked["checked_at"] > 0
        assert record["source_state"]["links"][source_index]["evidence_baseline_check_history"][-1] == checked
    for source_index in (0, 1, 3, 5, 6):
        assert "evidence_baseline_check" not in record["source_state"]["links"][source_index]
    assert all(
        item.get("checked_at", 0) > 0
        for item in payload["items"]
        if item["reason"] in {"candidate", "source_lost"}
    )

    repeated = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-context-baseline",
        json={"expected_revision": 2},
    )
    assert repeated.status_code == 200
    assert repeated.json()["summary"]["created"] == 0
    assert repeated.json()["summary"]["skipped"] == 7
    assert repeated.json()["note"]["revision"] == 2

    stale = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-context-baseline",
        json={"expected_revision": 1},
    )
    assert stale.status_code == 409
    assert store.get_research_note(note["id"])["revision"] == 2

    unresolved_note = store.create_research_note(
        title="Unavailable evidence note",
        content_markdown="Keep this note body",
        source_state={
            "version": 1,
            "links": [{
                "kind": "source",
                "label": "Unavailable paper",
                "source_path": "",
                "evidence_quote": "Saved evidence.",
            }],
        },
    )
    assert unresolved_note is not None
    unresolved = client.post(
        f"/api/chat/research-notes/{unresolved_note['id']}/evidence-context-baseline",
        json={"expected_revision": 1},
    )
    assert unresolved.status_code == 200
    unresolved_payload = unresolved.json()
    assert unresolved_payload["summary"]["created"] == 0
    assert unresolved_payload["note"]["revision"] == 2
    assert unresolved_payload["note"]["content_markdown"] == "Keep this note body"
    unresolved_check = unresolved_payload["note"]["source_state"]["links"][0]["evidence_baseline_check"]
    assert unresolved_check["reason"] == "source_lost"
    assert unresolved_check["terminal"] is True
    assert "retry_after" not in unresolved_check
    assert client.get(f"/api/chat/research-notes/{unresolved_note['id']}").json()["source_state"]["links"][0][
        "evidence_baseline_check"
    ] == unresolved_check
    unresolved_repeat = client.post(
        f"/api/chat/research-notes/{unresolved_note['id']}/evidence-context-baseline",
        json={"expected_revision": 2},
    )
    assert unresolved_repeat.status_code == 200
    assert unresolved_repeat.json()["note"]["revision"] == 2
    assert unresolved_repeat.json()["items"][0]["checked_at"] == unresolved_check["checked_at"]


def test_chat_research_note_evidence_maintenance_batches_updates_and_persists_backoff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router
    from kb.research_note_evidence import research_note_evidence_maintenance_due
    from kb.source_blocks import load_source_blocks

    md_path = tmp_path / "converted.md"
    md_path.write_text(
        """# Paper

## Results

Photon arrival is a stochastic process governed by Poisson statistics.
""",
        encoding="utf-8",
    )
    target = next(block for block in load_source_blocks(md_path) if "Photon arrival" in block["text"])
    store = ChatStore(tmp_path / "chat.sqlite3")
    exact = store.create_research_note(
        title="Exact evidence",
        content_markdown="User note body must remain unchanged",
        source_state={
            "links": [{
                "kind": "source",
                "source_path": "library/exact.pdf",
                "block_id": target["block_id"],
                "anchor_id": target["anchor_id"],
                "evidence_quote": target["text"],
            }]
        },
    )
    missing = store.create_research_note(
        title="Temporarily missing evidence",
        content_markdown="Missing-source note body",
        source_state={
            "links": [{
                "kind": "source",
                "source_path": "library/missing.pdf",
                "evidence_quote": "Saved evidence from a temporarily missing source.",
                "evidence_context_snapshot": {
                    "version": 1,
                    "captured_at": 80.0,
                    "source_fingerprint": "sha256:saved-missing",
                    "reason": "exact_block",
                    "before": [],
                    "target": {
                        "block_id": "missing-block",
                        "text": "Saved evidence from a temporarily missing source.",
                    },
                    "after": [],
                },
                "evidence_baseline_check": {
                    "version": 1,
                    "checked_at": 90.0,
                    "status": "failed",
                    "reason": "verification_failed",
                    "maintenance_version": 3,
                    "retry_count": 1,
                    "retry_after": 95.0,
                },
            }]
        },
    )
    decision = store.create_research_note(
        title="Ambiguous evidence",
        content_markdown="Ambiguous note body",
        source_state={
            "links": [{
                "kind": "source",
                "source_path": "library/exact.pdf",
                "block_id": "stale-block",
                "evidence_quote": "photon arrivals follow Poisson behavior",
            }]
        },
    )
    conflict = store.create_research_note(title="Edited elsewhere", content_markdown="Current body")
    assert exact is not None and missing is not None and decision is not None and conflict is not None
    conflict_latest, conflicted = store.update_research_note(
        conflict["id"],
        expected_revision=1,
        content_markdown="Edited body",
    )
    assert conflicted is False and conflict_latest is not None

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda path: path)
    monkeypatch.setattr(
        chat_router,
        "_resolve_paper_guide_md_path",
        lambda source_path, **_kwargs: md_path if "exact.pdf" in str(source_path) else None,
    )
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/chat/research-notes/evidence-maintenance",
        json={
            "notes": [
                {"note_id": exact["id"], "expected_revision": 1},
                {"note_id": missing["id"], "expected_revision": 1},
                {"note_id": decision["id"], "expected_revision": 1},
                {"note_id": conflict["id"], "expected_revision": 1},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {
        "requested": 4,
        "processed": 3,
        "updated": 3,
        "deferred": 0,
        "conflicted": 1,
        "not_found": 0,
        "errors": 0,
        "resolved_links": 1,
        "decision_links": 1,
        "has_more": False,
    }
    assert {item["status"] for item in payload["items"]} == {"processed", "conflicted"}
    assert all(note["content_markdown"] == "" for note in payload["notes"])

    exact_after = store.get_research_note(exact["id"])
    missing_after = store.get_research_note(missing["id"])
    decision_after = store.get_research_note(decision["id"])
    assert exact_after is not None and missing_after is not None and decision_after is not None
    assert exact_after["updated_at"] == exact["updated_at"]
    assert exact_after["content_markdown"] == "User note body must remain unchanged"
    assert exact_after["source_state"]["links"][0]["evidence_context_snapshot"]
    retry_check = missing_after["source_state"]["links"][0]["evidence_baseline_check"]
    assert retry_check["retry_count"] == 2
    assert retry_check["retry_after"] > retry_check["checked_at"]
    assert research_note_evidence_maintenance_due(missing_after["source_state"], now=retry_check["checked_at"]) is False
    decision_check = decision_after["source_state"]["links"][0]["evidence_baseline_check"]
    assert decision_check["reason"] == "candidate"
    assert decision_check["maintenance_version"] == 3

    immediate_repeat = client.post(
        "/api/chat/research-notes/evidence-maintenance",
        json={
            "notes": [
                {"note_id": missing["id"], "expected_revision": 2},
                {"note_id": decision["id"], "expected_revision": 2},
            ]
        },
    )
    assert immediate_repeat.status_code == 200
    assert immediate_repeat.json()["summary"]["processed"] == 0
    assert immediate_repeat.json()["summary"]["deferred"] == 2
    assert store.get_research_note(missing["id"])["revision"] == 2


def test_chat_research_note_evidence_health_refreshes_existing_and_tracks_system_recovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router
    from kb.source_blocks import load_source_blocks

    md_path = tmp_path / "paper.en.md"
    md_path.write_text(
        """# Paper

## Results

Photon arrival is governed by Poisson statistics.
""",
        encoding="utf-8",
    )
    target = next(block for block in load_source_blocks(md_path) if "Photon arrival" in block["text"])
    saved_context = chat_router.build_research_note_evidence_context(
        md_path=md_path,
        source_path="library/paper.pdf",
        evidence_quote=target["text"],
        block_id=target["block_id"],
        anchor_id=target["anchor_id"],
    )
    md_path.write_text(
        md_path.read_text(encoding="utf-8") + "\nConversion output was refreshed.\n",
        encoding="utf-8",
    )

    store = ChatStore(tmp_path / "chat.sqlite3")
    note = store.create_research_note(
        title="Evidence health",
        content_markdown="The user-authored reading note must not change.",
        source_state={"links": [
            {
                "kind": "source",
                "source_path": "library/paper.pdf",
                "block_id": target["block_id"],
                "anchor_id": target["anchor_id"],
                "evidence_quote": target["text"],
                "evidence_context_snapshot": saved_context["context_snapshot"],
                "evidence_verification": {
                    "status": "exact",
                    "reason": "captured",
                    "source_fingerprint": saved_context["source_fingerprint"],
                },
            },
            {
                "kind": "source",
                "source_path": "library/pending.pdf",
                "evidence_quote": "Evidence waiting for its conversion output.",
            },
        ]},
    )
    assert note is not None

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda path: path)
    monkeypatch.setattr(
        chat_router,
        "_resolve_paper_guide_md_path",
        lambda source_path, **_kwargs: md_path if "paper.pdf" in str(source_path) else None,
    )
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    response = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-health",
        json={"expected_revision": note["revision"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["created"] == 1
    assert payload["summary"]["reasons"] == {
        "source_change_auto_verified": 1,
        "conversion_pending": 1,
    }
    refreshed = payload["note"]
    assert refreshed["revision"] == 2
    assert refreshed["updated_at"] == note["updated_at"]
    assert refreshed["content_markdown"] == "The user-authored reading note must not change."
    exact_link, pending_link = refreshed["source_state"]["links"]
    assert exact_link["evidence_verification"]["reason"] == "source_change_auto_verified"
    assert exact_link["evidence_context_snapshot"]["source_fingerprint"] != saved_context["source_fingerprint"]
    assert pending_link["evidence_baseline_check"]["reason"] == "conversion_pending"
    assert pending_link["evidence_baseline_check"]["retry_after"] > pending_link["evidence_baseline_check"]["checked_at"]


def test_missing_note_source_reaches_terminal_outcome_and_recovers_when_source_returns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from fastapi import HTTPException

    from api.routers import chat as chat_router
    from kb.research_note_evidence import research_note_evidence_maintenance_due
    from kb.source_blocks import load_source_blocks

    md_path = tmp_path / "missing.en.md"
    md_path.write_text(
        """# Recovered paper

## Results

Recovered evidence remains exact after the source returns.
""",
        encoding="utf-8",
    )
    target = next(block for block in load_source_blocks(md_path) if "Recovered evidence" in block["text"])
    store = ChatStore(tmp_path / "chat.sqlite3")
    note = store.create_research_note(
        title="Missing source evidence",
        content_markdown="The reader's note body must remain unchanged.",
        source_state={"links": [{
            "kind": "source",
            "source_path": "library/missing.pdf",
            "source_name": "Missing paper",
            "block_id": target["block_id"],
            "anchor_id": target["anchor_id"],
            "evidence_quote": target["text"],
        }]},
    )
    assert note is not None
    available = {"value": False}

    def resolve_md(*_args, **_kwargs):
        return md_path if available["value"] else None

    def resolve_reader(_path: str) -> str:
        if not available["value"]:
            raise HTTPException(404, "source PDF not found")
        return str(md_path)

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", resolve_reader)
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", resolve_md)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    current = note
    for attempt in range(1, 7):
        response = client.post(
            f"/api/chat/research-notes/{note['id']}/evidence-health",
            json={"expected_revision": current["revision"]},
        )
        assert response.status_code == 200
        current = response.json()["note"]
        assert current["content_markdown"] == "The reader's note body must remain unchanged."
        check = current["source_state"]["links"][0]["evidence_baseline_check"]
        assert check["retry_count"] == attempt
        if attempt < 6:
            assert check["reason"] == "source_missing"
            assert check["retry_after"] > check["checked_at"]
            assert not check.get("terminal")
        else:
            assert check["reason"] == "source_lost"
            assert check["terminal"] is True
            assert check["terminal_at"] == check["checked_at"]
            assert "retry_after" not in check

    terminal_revision = current["revision"]
    terminal_check = current["source_state"]["links"][0]["evidence_baseline_check"]
    terminal_history_count = len(current["source_state"]["links"][0]["evidence_baseline_check_history"])
    assert research_note_evidence_maintenance_due(current["source_state"], now=10**12) is False

    repeated = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-health",
        json={"expected_revision": terminal_revision},
    )
    assert repeated.status_code == 200
    assert repeated.json()["note"]["revision"] == terminal_revision
    repeated_link = repeated.json()["note"]["source_state"]["links"][0]
    assert repeated_link["evidence_baseline_check"] == terminal_check
    assert len(repeated_link["evidence_baseline_check_history"]) == terminal_history_count

    available["value"] = True
    recovery = chat_router.maintain_research_note_evidence_for_recovered_sources(
        source_paths=("library/missing.pdf",),
        reason="conversion_ready",
    )
    assert recovery["processed"] == 1
    assert recovery["resolved_links"] == 1
    recovered = store.get_research_note(note["id"])
    assert recovered is not None
    assert recovered["revision"] == terminal_revision + 1
    assert recovered["content_markdown"] == "The reader's note body must remain unchanged."
    recovered_link = recovered["source_state"]["links"][0]
    assert "evidence_baseline_check" not in recovered_link
    assert recovered_link["evidence_verification"]["status"] == "exact"
    assert recovered_link["evidence_context_snapshot"]["target"]["block_id"] == target["block_id"]
    assert recovered_link["evidence_baseline_check_history"][-1]["status"] == "resolved"
    assert recovered_link["evidence_baseline_check_history"][-1]["previous_reason"] == "source_lost"


def test_note_evidence_uses_retained_conversion_after_pdf_disappears(tmp_path: Path) -> None:
    from api.routers.chat import _resolve_research_note_evidence_md_path

    md_root = tmp_path / "md"
    pdf_root = tmp_path / "pdf"
    db_root = tmp_path / "db"
    converted_dir = md_root / "retained-paper"
    converted_dir.mkdir(parents=True)
    pdf_root.mkdir()
    db_root.mkdir()
    retained_md = converted_dir / "retained-paper.en.md"
    retained_md.write_text("# Retained paper\n\nThe converted evidence is still readable.\n", encoding="utf-8")

    resolved, reason = _resolve_research_note_evidence_md_path(
        "retained-paper.pdf",
        md_root=md_root,
        pdf_root=pdf_root,
        db_dir=db_root,
    )

    assert resolved == retained_md.resolve()
    assert reason == ""


def test_note_evidence_uses_existing_allowed_markdown_source(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    md_root = tmp_path / "md"
    pdf_root = tmp_path / "pdf"
    db_root = tmp_path / "db"
    md_root.mkdir()
    pdf_root.mkdir()
    db_root.mkdir()
    recovered_md = md_root / "recovered-paper.md"
    recovered_md.write_text("# Recovered paper\n\nThe evidence is readable again.\n", encoding="utf-8")
    monkeypatch.setattr(
        chat_router,
        "_resolve_allowed_reader_source_path",
        lambda _path: str(recovered_md),
    )

    resolved, reason = chat_router._resolve_research_note_evidence_md_path(
        str(recovered_md),
        md_root=md_root,
        pdf_root=pdf_root,
        db_dir=db_root,
    )

    assert resolved == recovered_md
    assert reason == ""


def test_recovered_source_event_refreshes_exact_note_evidence_and_leaves_only_real_decisions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router
    from kb.research_note_maintenance import (
        enqueue_research_note_source_recovery,
        register_research_note_recovery_handler,
        unregister_research_note_recovery_handler,
        wait_for_research_note_recovery_idle,
    )
    from kb.source_blocks import load_source_blocks

    md_path = tmp_path / "paper.en.md"
    md_path.write_text(
        """# Paper

## Results

Photon arrival is a stochastic process governed by Poisson statistics.
""",
        encoding="utf-8",
    )
    target = next(block for block in load_source_blocks(md_path) if "Photon arrival" in block["text"])
    store = ChatStore(tmp_path / "chat.sqlite3")
    exact = store.create_research_note(
        title="Exact recovered evidence",
        content_markdown="User reading note remains unchanged.",
        source_state={"links": [{
            "kind": "source",
            "source_path": "library/paper.pdf",
            "block_id": target["block_id"],
            "anchor_id": target["anchor_id"],
            "evidence_quote": target["text"],
        }]},
    )
    decision = store.create_research_note(
        title="Ambiguous recovered evidence",
        content_markdown="",
        source_state={"links": [{
            "kind": "source",
            "source_path": "library/paper.pdf",
            "block_id": "stale-block",
            "anchor_id": "stale-anchor",
            "evidence_quote": "photon arrivals follow Poisson behavior",
        }]},
    )
    unrelated = store.create_research_note(
        title="Unrelated source",
        content_markdown="",
        source_state={"links": [{
            "kind": "source",
            "source_path": "library/other.pdf",
            "evidence_quote": "Other evidence.",
        }]},
    )
    assert exact is not None and decision is not None and unrelated is not None

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda _path: str(md_path))
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", lambda *args, **kwargs: md_path)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    baseline = chat_router.capture_chat_research_note_evidence_context_baseline(
        exact["id"],
        chat_router.ResearchNoteEvidenceBaselineBody(expected_revision=1),
    )
    before = baseline["note"]
    before_fingerprint = before["source_state"]["links"][0]["evidence_context_snapshot"]["source_fingerprint"]
    assert before["revision"] == 2

    md_path.write_text(
        md_path.read_text(encoding="utf-8") + "\n## Notes\n\nConversion repair restored this source.\n",
        encoding="utf-8",
    )
    handler = chat_router.maintain_research_note_evidence_for_recovered_sources
    unregister_research_note_recovery_handler()
    register_research_note_recovery_handler(handler)
    try:
        assert enqueue_research_note_source_recovery(
            (str(tmp_path / "paper.pdf"), str(md_path)),
            reason="conversion_ready",
        ) is True
        assert wait_for_research_note_recovery_idle(timeout=5.0) is True
        assert enqueue_research_note_source_recovery(
            (str(tmp_path / "paper.pdf"), str(md_path)),
            reason="file_reindex_ready",
        ) is True
        assert wait_for_research_note_recovery_idle(timeout=5.0) is True
    finally:
        unregister_research_note_recovery_handler(handler)

    exact_after = store.get_research_note(exact["id"])
    decision_after = store.get_research_note(decision["id"])
    unrelated_after = store.get_research_note(unrelated["id"])
    assert exact_after is not None and decision_after is not None and unrelated_after is not None
    exact_link = exact_after["source_state"]["links"][0]
    assert exact_after["revision"] == 3
    assert exact_after["updated_at"] == exact["updated_at"]
    assert exact_after["content_markdown"] == "User reading note remains unchanged."
    assert exact_link["evidence_context_snapshot"]["source_fingerprint"] != before_fingerprint
    assert exact_link["evidence_verification"]["reason"] == "source_change_auto_verified"
    assert exact_link["evidence_auto_resolution_history"][-1]["action"] == "source_change_auto_verified"
    assert decision_after["revision"] == 2
    assert decision_after["source_state"]["links"][0]["evidence_baseline_check"]["reason"] == "candidate"
    assert unrelated_after["revision"] == 1


def test_repeated_quote_is_auto_relocated_from_saved_context_without_user_note_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router
    from kb.source_blocks import load_source_blocks

    md_path = tmp_path / "repeated.en.md"
    md_path.write_text(
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
        encoding="utf-8",
    )
    blocks = load_source_blocks(md_path)
    first_target = next(
        block
        for block in blocks
        if "same calibration sentence" in str(block.get("text") or "")
        and int(block.get("page_start") or 0) == 2
    )
    saved = chat_router.build_research_note_evidence_context(
        md_path=md_path,
        source_path="library/repeated.pdf",
        evidence_quote="The same calibration sentence appears here verbatim.",
        block_id=str(first_target["block_id"]),
        anchor_id=str(first_target["anchor_id"]),
    )
    md_path.write_text(
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
    store = ChatStore(tmp_path / "chat.sqlite3")
    note = store.create_research_note(
        title="Repeated evidence",
        content_markdown="User reading note must remain byte-for-byte unchanged.",
        source_state={"links": [{
            "kind": "source",
            "source_path": "library/repeated.pdf",
            "block_id": "stale-block",
            "anchor_id": "stale-anchor",
            "heading_path": "First experiment",
            "page_start": 2,
            "evidence_quote": "The same calibration sentence appears here verbatim.",
            "evidence_context_snapshot": saved["context_snapshot"],
            "evidence_verification": {
                "status": "exact",
                "reason": "captured",
                "source_fingerprint": saved["source_fingerprint"],
            },
            "evidence_baseline_check": {
                "version": 1,
                "status": "skipped",
                "reason": "candidate",
                "maintenance_version": 1,
            },
        }]},
    )
    assert note is not None

    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda _path: str(md_path))
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", lambda *args, **kwargs: md_path)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    inspected = client.post(
        "/api/chat/research-note/evidence-context",
        json={
            "source_path": "library/repeated.pdf",
            "evidence_quote": "The same calibration sentence appears here verbatim.",
            "block_id": "stale-block",
            "anchor_id": "stale-anchor",
            "source_fingerprint": saved["source_fingerprint"],
            "evidence_context_snapshot": saved["context_snapshot"],
        },
    )
    assert inspected.status_code == 200
    assert inspected.json()["reason"] == "context_disambiguated"
    assert inspected.json()["target"]["page_start"] == 2
    assert inspected.json()["context_disambiguation"]["mode"] == "weighted_saved_context"
    assert inspected.json()["context_disambiguation"]["score_gap"] >= 0.10

    maintained = client.post(
        "/api/chat/research-notes/evidence-maintenance",
        json={"notes": [{"note_id": note["id"], "expected_revision": note["revision"]}]},
    )
    assert maintained.status_code == 200
    summary = maintained.json()["summary"]
    assert summary["resolved_links"] == 1
    assert summary["decision_links"] == 0

    updated = store.get_research_note(note["id"])
    assert updated is not None
    assert updated["revision"] == 2
    assert updated["updated_at"] == note["updated_at"]
    assert updated["content_markdown"] == "User reading note must remain byte-for-byte unchanged."
    link = updated["source_state"]["links"][0]
    assert link["page_start"] == 2
    assert link["evidence_verification"]["reason"] == "saved_context_auto_disambiguated"
    audit = link["evidence_auto_resolution_history"][-1]
    assert audit["action"] == "saved_context_auto_disambiguated"
    assert audit["resolution_evidence"]["mode"] == "weighted_saved_context"
    assert audit["resolution_evidence"]["score_gap"] >= 0.10
    assert "evidence_baseline_check" not in link


def test_chat_research_note_candidate_repair_is_explicit_audited_and_reopenable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router
    from kb.research_note_evidence import source_content_fingerprint

    md_path = tmp_path / "converted.md"
    md_path.write_text(
        """# Paper

<!-- kb_page: 4 -->

## Results

The detector was calibrated before every acquisition.

Photon arrival is a stochastic process governed by Poisson statistics.

Dark counts were measured with the shutter closed.
""",
        encoding="utf-8",
    )
    store = ChatStore(tmp_path / "chat.sqlite3")
    initial_fingerprint = source_content_fingerprint(md_path)
    note = store.create_research_note(
        title="Evidence note",
        content_markdown="Detector evidence",
        source_state={
            "version": 1,
            "links": [
                {
                    "kind": "source",
                    "label": "Paper",
                    "source_path": "library/paper.pdf",
                    "capture_id": "capture-1",
                    "block_id": "stale-block",
                    "anchor_id": "stale-anchor",
                    "evidence_quote": "stochastic photon arrivals follow Poisson behavior",
                    "start_offset": 2,
                    "end_offset": 54,
                    "evidence_verification": {
                        "status": "exact",
                        "reason": "legacy_saved",
                        "verified_at": 1_700_000_000,
                        "source_fingerprint": initial_fingerprint,
                    },
                }
            ],
        },
    )
    assert note is not None
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda _path: str(md_path))
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", lambda *args, **kwargs: md_path)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    reviewed = client.post(
        "/api/chat/research-note/evidence-context",
        json={
            "source_path": "library/paper.pdf",
            "source_name": "Paper",
            "evidence_quote": "stochastic photon arrivals follow Poisson behavior",
            "block_id": "stale-block",
            "anchor_id": "stale-anchor",
        },
    )
    assert reviewed.status_code == 200
    context = reviewed.json()
    assert context["status"] == "candidate"
    candidate = context["candidates"][0]

    invalidated_candidate = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-review",
        json={
            "decision": "invalid",
            "source_index": 0,
            "source_path": "library/paper.pdf",
            "capture_id": "capture-1",
            "old_block_id": "stale-block",
            "old_anchor_id": "stale-anchor",
            "old_evidence_quote": "stochastic photon arrivals follow Poisson behavior",
            "expected_source_fingerprint": initial_fingerprint,
            "current_block_id": candidate["block_id"],
            "current_anchor_id": candidate["anchor_id"],
            "current_anchor_kind": candidate["kind"],
            "current_evidence_quote": candidate["text"],
            "source_fingerprint": context["source_fingerprint"],
        },
    )
    assert invalidated_candidate.status_code == 200
    invalid_candidate_record = invalidated_candidate.json()
    assert invalid_candidate_record["revision"] == 2
    assert invalid_candidate_record["content_markdown"] == "Detector evidence"
    invalid_candidate_link = invalid_candidate_record["source_state"]["links"][0]
    assert invalid_candidate_link["block_id"] == "stale-block"
    assert invalid_candidate_link["evidence_quote"] == "stochastic photon arrivals follow Poisson behavior"
    assert invalid_candidate_link["evidence_verification"]["status"] == "invalid"
    assert invalid_candidate_link["evidence_review_history"][-1]["reviewed_target"]["block_id"] == candidate["block_id"]

    repaired = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-repair",
        json={
            "source_index": 0,
            "source_path": "library/paper.pdf",
            "capture_id": "capture-1",
            "old_block_id": "stale-block",
            "old_anchor_id": "stale-anchor",
            "old_evidence_quote": "stochastic photon arrivals follow Poisson behavior",
            "candidate_block_id": candidate["block_id"],
            "candidate_anchor_id": candidate["anchor_id"],
            "candidate_anchor_kind": candidate["kind"],
            "candidate_evidence_quote": candidate["text"],
            "source_fingerprint": context["source_fingerprint"],
        },
    )
    assert repaired.status_code == 200
    record = repaired.json()
    assert record["revision"] == 3
    link = record["source_state"]["links"][0]
    assert link["block_id"] == candidate["block_id"]
    assert link["anchor_id"] == candidate["anchor_id"]
    assert "start_offset" not in link
    assert link["locator_repair_history"][-1]["from"]["block_id"] == "stale-block"
    assert link["evidence_verification"]["source_fingerprint"] == context["source_fingerprint"]
    assert link["evidence_context_snapshot"]["source_fingerprint"] == context["source_fingerprint"]
    assert link["evidence_context_snapshot"]["target"]["block_id"] == candidate["block_id"]
    assert link["locator_repair_history"][-1]["reviewed_context_snapshot"]["target"]["block_id"] == candidate["block_id"]

    reopened = client.post(
        "/api/chat/research-note/evidence-context",
        json={
            "source_path": link["source_path"],
            "source_name": link["label"],
            "evidence_quote": link["evidence_quote"],
            "block_id": link["block_id"],
            "anchor_id": link["anchor_id"],
            "source_fingerprint": link["evidence_verification"]["source_fingerprint"],
        },
    )
    assert reopened.status_code == 200
    assert reopened.json()["status"] == "exact"
    assert reopened.json()["source_changed"] is False

    conflict = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-repair",
        json={
            "source_index": 0,
            "source_path": "library/paper.pdf",
            "capture_id": "capture-1",
            "old_block_id": candidate["block_id"],
            "old_anchor_id": candidate["anchor_id"],
            "old_evidence_quote": candidate["text"],
            "candidate_block_id": candidate["block_id"],
            "candidate_anchor_id": candidate["anchor_id"],
            "candidate_anchor_kind": candidate["kind"],
            "candidate_evidence_quote": candidate["text"],
            "source_fingerprint": "sha256:" + ("0" * 64),
        },
    )
    assert conflict.status_code == 409
    assert store.get_research_note(note["id"])["revision"] == 3

    old_fingerprint = link["evidence_verification"]["source_fingerprint"]
    md_path.write_text(
        md_path.read_text(encoding="utf-8") + "\n## Appendix\n\nAdditional conversion output.\n",
        encoding="utf-8",
    )
    changed = client.post(
        "/api/chat/research-note/evidence-context",
        json={
            "source_path": link["source_path"],
            "source_name": link["label"],
            "evidence_quote": link["evidence_quote"],
            "block_id": link["block_id"],
            "anchor_id": link["anchor_id"],
            "source_fingerprint": old_fingerprint,
        },
    )
    assert changed.status_code == 200
    changed_context = changed.json()
    assert changed_context["status"] == "exact"
    assert changed_context["source_changed"] is True
    assert changed_context["context_snapshot"]["source_fingerprint"] == changed_context["source_fingerprint"]
    current_target = changed_context["target"]
    review_body = {
        "source_index": 0,
        "source_path": link["source_path"],
        "capture_id": link["capture_id"],
        "old_block_id": link["block_id"],
        "old_anchor_id": link["anchor_id"],
        "old_evidence_quote": link["evidence_quote"],
        "expected_source_fingerprint": old_fingerprint,
        "current_block_id": current_target["block_id"],
        "current_anchor_id": current_target["anchor_id"],
        "current_anchor_kind": current_target["kind"],
        "current_evidence_quote": current_target["text"],
        "source_fingerprint": changed_context["source_fingerprint"],
    }

    invalidated = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-review",
        json={**review_body, "decision": "invalid"},
    )
    assert invalidated.status_code == 200
    invalid_record = invalidated.json()
    assert invalid_record["revision"] == 4
    invalid_link = invalid_record["source_state"]["links"][0]
    assert invalid_link["evidence_quote"] == candidate["text"]
    assert invalid_link["evidence_verification"]["status"] == "invalid"
    assert invalid_link["evidence_review_history"][-1]["previous_verification"]["source_fingerprint"] == old_fingerprint
    assert invalid_link["evidence_review_history"][-1]["saved_locator"]["block_id"] == candidate["block_id"]
    assert invalid_link["evidence_context_snapshot"]["source_fingerprint"] == old_fingerprint
    assert invalid_link["evidence_review_history"][-1]["reviewed_context_snapshot"]["source_fingerprint"] == changed_context["source_fingerprint"]

    confirmed = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-review",
        json={
            **review_body,
            "decision": "confirmed",
            "expected_source_fingerprint": changed_context["source_fingerprint"],
        },
    )
    assert confirmed.status_code == 200
    confirmed_record = confirmed.json()
    assert confirmed_record["revision"] == 5
    confirmed_link = confirmed_record["source_state"]["links"][0]
    assert confirmed_link["evidence_verification"]["status"] == "exact"
    assert confirmed_link["evidence_verification"]["reason"] == "confirmed_source_change"
    assert [row["decision"] for row in confirmed_link["evidence_review_history"]] == [
        "invalid",
        "invalid",
        "confirmed",
    ]
    assert confirmed_link["evidence_context_snapshot"]["source_fingerprint"] == changed_context["source_fingerprint"]
    assert "reviewed_context_snapshot" not in confirmed_link["evidence_review_history"][0]

    stale_review = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-review",
        json={**review_body, "decision": "invalid"},
    )
    assert stale_review.status_code == 409
    assert store.get_research_note(note["id"])["revision"] == 5


def test_chat_research_note_repair_reverifies_a_trusted_candidate_after_source_update(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from api.routers import chat as chat_router
    from kb.research_note_evidence import source_content_fingerprint

    md_path = tmp_path / "converted.md"
    md_path.write_text(
        """# Paper

## Results

Photon arrival is a stochastic process governed by Poisson statistics.
""",
        encoding="utf-8",
    )
    initial_fingerprint = source_content_fingerprint(md_path)
    store = ChatStore(tmp_path / "chat.sqlite3")
    note = store.create_research_note(
        title="Evidence note",
        content_markdown="Detector evidence",
        source_state={
            "version": 1,
            "links": [
                {
                    "kind": "source",
                    "label": "Paper",
                    "source_path": "library/paper.pdf",
                    "capture_id": "capture-1",
                    "block_id": "stale-block",
                    "anchor_id": "stale-anchor",
                    "evidence_quote": "stochastic photon arrivals follow Poisson behavior",
                    "evidence_verification": {
                        "status": "exact",
                        "reason": "legacy_saved",
                        "verified_at": 1_700_000_000,
                        "source_fingerprint": initial_fingerprint,
                    },
                }
            ],
        },
    )
    assert note is not None
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    monkeypatch.setattr(chat_router, "_resolve_allowed_reader_source_path", lambda _path: str(md_path))
    monkeypatch.setattr(chat_router, "_resolve_paper_guide_md_path", lambda *args, **kwargs: md_path)
    monkeypatch.setattr(chat_router, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(chat_router, "_pdf_dir", lambda: tmp_path)
    client = TestClient(app)

    reviewed = client.post(
        "/api/chat/research-note/evidence-context",
        json={
            "source_path": "library/paper.pdf",
            "source_name": "Paper",
            "evidence_quote": "stochastic photon arrivals follow Poisson behavior",
            "block_id": "stale-block",
            "anchor_id": "stale-anchor",
        },
    )
    assert reviewed.status_code == 200
    context = reviewed.json()
    assert context["status"] == "candidate"
    candidate = context["candidates"][0]
    assert context["source_fingerprint"] == initial_fingerprint

    md_path.write_text(
        md_path.read_text(encoding="utf-8") + "\n## Appendix\n\nAdditional conversion output.\n",
        encoding="utf-8",
    )
    current_fingerprint = source_content_fingerprint(md_path)
    assert current_fingerprint != initial_fingerprint

    repaired = client.post(
        f"/api/chat/research-notes/{note['id']}/evidence-repair",
        json={
            "source_index": 0,
            "source_path": "library/paper.pdf",
            "capture_id": "capture-1",
            "old_block_id": "stale-block",
            "old_anchor_id": "stale-anchor",
            "old_evidence_quote": "stochastic photon arrivals follow Poisson behavior",
            "candidate_block_id": candidate["block_id"],
            "candidate_anchor_id": candidate["anchor_id"],
            "candidate_anchor_kind": candidate["kind"],
            "candidate_evidence_quote": candidate["text"],
            "source_fingerprint": context["source_fingerprint"],
        },
    )

    assert repaired.status_code == 200
    link = repaired.json()["source_state"]["links"][0]
    assert link["block_id"] == candidate["block_id"]
    assert link["evidence_quote"] == candidate["text"]
    assert link["evidence_verification"]["source_fingerprint"] == current_fingerprint
    assert link["locator_repair_history"][-1]["source_fingerprint"] == current_fingerprint


def test_chat_research_notes_workspace_list_and_metadata(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    store = ChatStore(tmp_path / "chat.sqlite3")
    project_id = store.create_project("Workspace")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    client = TestClient(app)

    created = client.post(
        "/api/chat/research-notes",
        json={
            "project_id": project_id,
            "title": "Evidence note",
            "content_markdown": "A rare-search-token appears in the body.",
            "tags": ["evidence", "review"],
            "pinned": True,
        },
    )
    assert created.status_code == 200
    note_id = created.json()["id"]

    found = client.get(
        "/api/chat/research-notes",
        params={"scope": "all", "query": "rare-search-token", "archived": "active"},
    )
    assert found.status_code == 200
    assert [item["id"] for item in found.json()] == [note_id]
    assert found.json()[0]["tags"] == ["evidence", "review"]
    assert found.json()[0]["pinned"] is True

    archived = client.patch(
        f"/api/chat/research-notes/{note_id}",
        json={
            "expected_revision": 1,
            "project_id": None,
            "tags": ["final"],
            "archived": True,
        },
    )
    assert archived.status_code == 200
    assert archived.json()["project_id"] is None
    assert archived.json()["tags"] == ["final"]
    assert archived.json()["archived"] is True

    assert client.get("/api/chat/research-notes", params={"scope": "all"}).json() == []
    archived_list = client.get(
        "/api/chat/research-notes",
        params={"scope": "all", "archived": "archived"},
    )
    assert [item["id"] for item in archived_list.json()] == [note_id]

    invalid_query = client.get(
        "/api/chat/research-notes",
        params={"scope": "workspace", "archived": "later"},
    )
    assert invalid_query.status_code == 422


def test_chat_research_outline_composition_survives_store_reopen(monkeypatch, tmp_path: Path) -> None:
    from api.routers import chat as chat_router

    db_path = tmp_path / "chat.sqlite3"
    store = ChatStore(db_path)
    first = store.create_research_note(title="First source note", content_markdown="First finding")
    second = store.create_research_note(title="Second source note", content_markdown="Second finding")
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: store)
    client = TestClient(app)

    response = client.post(
        "/api/chat/research-notes",
        json={
            "title": "Persistent evidence outline",
            "content_markdown": "# Persistent evidence outline\n\n## 1. First source note",
            "source_state": {
                "version": 1,
                "links": [
                    {
                        "kind": "source",
                        "label": "Grounded source",
                        "source_path": "db/paper/source.md",
                        "block_id": "block-source",
                    }
                ],
                "composition": {
                    "kind": "research_note_outline",
                    "version": 1,
                    "source_notes": [
                        {"id": first["id"], "title": first["title"], "revision": first["revision"]},
                        {"id": second["id"], "title": second["title"], "revision": second["revision"]},
                    ],
                },
            },
            "tags": ["outline", "evidence"],
        },
    )
    assert response.status_code == 200
    outline_id = response.json()["id"]

    latest_first, conflicted = store.update_research_note(
        first["id"],
        expected_revision=1,
        content_markdown="First finding with calibrated evidence",
    )
    assert conflicted is False
    assert latest_first is not None
    refresh_response = client.patch(
        f"/api/chat/research-notes/{outline_id}",
        json={
            "expected_revision": 1,
            "content_markdown": "# Persistent evidence outline\n\n## 1. First source note\n\nFirst finding with calibrated evidence",
            "source_state": {
                "version": 1,
                "links": [
                    {
                        "kind": "source",
                        "label": "Grounded source",
                        "source_path": "db/paper/source.md",
                        "block_id": "block-source",
                    }
                ],
                "composition": {
                    "kind": "research_note_outline",
                    "version": 1,
                    "source_notes": [
                        {
                            "id": first["id"],
                            "title": first["title"],
                            "revision": 2,
                            "content_fingerprint": "fnv1a32:1234abcd:39",
                            "last_refreshed_at": 1_800_000_000,
                        },
                        {"id": second["id"], "title": second["title"], "revision": 1},
                    ],
                    "refresh_history": [
                        {
                            "source_id": first["id"],
                            "from_revision": 1,
                            "to_revision": 2,
                            "refreshed_at": 1_800_000_000,
                            "replaced_manual_edits": True,
                        }
                    ],
                },
            },
        },
    )
    assert refresh_response.status_code == 200
    assert refresh_response.json()["revision"] == 2

    reopened = ChatStore(db_path)
    monkeypatch.setattr(chat_router, "get_chat_store", lambda: reopened)
    restored = client.get(f"/api/chat/research-notes/{outline_id}")

    assert restored.status_code == 200
    record = restored.json()
    assert record["tags"] == ["outline", "evidence"]
    assert record["source_state"]["links"][0]["block_id"] == "block-source"
    assert record["source_state"]["composition"] == {
        "kind": "research_note_outline",
        "version": 1,
        "source_notes": [
            {
                "id": first["id"],
                "title": "First source note",
                "revision": 2,
                "content_fingerprint": "fnv1a32:1234abcd:39",
                "last_refreshed_at": 1_800_000_000,
            },
            {"id": second["id"], "title": "Second source note", "revision": 1},
        ],
        "refresh_history": [
            {
                "source_id": first["id"],
                "from_revision": 1,
                "to_revision": 2,
                "refreshed_at": 1_800_000_000,
                "replaced_manual_edits": True,
            }
        ],
    }
