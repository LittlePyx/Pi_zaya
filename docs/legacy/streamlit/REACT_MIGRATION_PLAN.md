# React Migration Plan (Feature Parity + Product Upgrade)

Last updated: 2026-03-05

## Goal

- Migrate core user flows from `app.py` (Streamlit) to `server.py + web` (FastAPI + React).
- Keep behavioral parity for existing users.
- Improve UX where old behavior is slow/heavy/confusing.

## Principles

- Parity first, then optimization.
- Avoid hidden mode complexity in primary workflows.
- Keep long-running tasks non-blocking and observable.
- Preserve source traceability (citations, references, file provenance).

## Status Snapshot

### Chat page

- [x] Conversation/project sidebar basics (create/select/rename/delete/move).
- [x] Message stream + partial rendering.
- [x] Citation rendering pipeline and refs panel integration.
- [x] Upload (PDF/image) queue + background ingest polling.
- [x] Auto source binding for uploaded/duplicate PDFs.
- [ ] Full parity for all history sidebar micro-interactions from legacy UI.
- [ ] Full parity for all runtime hover/detail affordances in complex citations.

### Library page

- [x] Directory settings (PDF/MD) and persisted settings wiring.
- [x] File list tabs (pending/converted/all).
- [x] Per-file convert/open/delete actions.
- [x] Batch convert pending.
- [x] Reindex + reference sync trigger.
- [x] Filename manager (scan/apply) with server-side perf optimizations.
- [x] Upload workbench (queue, inspect, save, save+convert, duplicate feedback).
- [x] Collapsible heavy panels to reduce page occupancy.
- [ ] Legacy-level operation hints and all edge-case messages in Chinese copy.
- [ ] More complete recoverability UX for partial failures (retry batch subsets, sticky error grouping).

### Backend/API

- [x] Library files API with queue/running/reconvert classification.
- [x] Upload inspect/commit split endpoints.
- [x] Rename suggestion/apply APIs with md sync support.
- [x] Reference sync runtime status stream.
- [x] Chat upload job status/retry/cancel APIs.
- [ ] Converter-level figure mapping correctness fixes (tracked separately).

## Product Improvements Applied in Current Iteration

- Set upload ingest defaults to non-fast path in React/FastAPI entry (`balanced`).
- Library upload workbench now supports explicit inspect before commit.
- Lock upload controls while conversion/reference-sync is running.
- Show reference sync live state on library page.
- Require/save directory draft before operations that depend on filesystem targets.

## Next Implementation Batches

### Batch A (High value, low risk)

- Unify and polish user-facing copy (CN) for chat/library operation states.
- Add grouped retry controls for failed upload/rename items.
- Add clearer “what changed” hints after rename/delete/reindex.

### Batch B (Parity deepening)

- Port remaining legacy sidebar/history micro-interactions.
- Port advanced citation hover/detail parity and keyboard actions.
- Port full legacy rename management affordances and diagnostics.

### Batch C (Reliability + observability)

- Add end-to-end sanity tests for full “upload -> convert -> reindex -> ask” flow.
- Add API contract checks for long-running job state transitions.
- Add perf budget checks for rename scan and upload inspect on larger batches.

## Acceptance Criteria (per batch)

- No regression in existing sanity/unit tests.
- Build passes in `web` with no TypeScript errors.
- Key user flows executable without manual backend restarts.
- New behavior documented in README or feature docs when user-visible.

## Known Outstanding Non-UI Issue

- Converter figure mapping mismatch (`NatPhoton-2019 Fig. 3`) is tracked in:
  - `docs/CONVERTER_PENDING_FIXES.md`
