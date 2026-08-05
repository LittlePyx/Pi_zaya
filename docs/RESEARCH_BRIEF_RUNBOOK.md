# Research Brief Runbook

Updated: 2026-08-06

## Purpose

Research briefs turn a project's verified evidence matrix into a durable
research artifact. A brief is editable Markdown with immutable revisions,
locatable evidence, bibliography metadata, matrix lineage, a generation trace,
and explicit quality state. It can be exported as Markdown, DOCX, BibTeX, or
RIS.

This workflow is evidence-first. It must not gain speed or a `verified` badge by
dropping a selected paper, broadening retrieval outside the basket, omitting an
unsupported claim from diagnostics, or weakening citation matching.

## User Workflow

1. Put the conversation in a project.
2. Add papers to the project's literature basket. Every selected item must have
   a matched local Markdown full-text path.
3. Open **Evidence matrix**, enter the comparison objective, and generate it.
4. Review the cell-level evidence and comparison-boundary notices. Missing facts
   remain empty; an edited factual cell must be refreshed before it is verified.
5. Choose **Create verified brief**, or open **Research brief** and select a
   verified matrix.
6. Enter a title and research objective, then generate. The brief uses only the
   selected matrix's audited evidence and records its matrix ID and revision.
7. Review the quality banner and the Evidence tab. Evidence rows open at the
   corresponding source location in the reader.
8. Edit and save as needed. Each save creates a new revision; restoring an old
   revision also creates a new revision instead of rewriting history.
9. Export the current revision as Markdown, DOCX, BibTeX, or RIS.

## Quality Contract

`quality_status` has three user-relevant states:

- `verified`: the agent completed without errors; evidence is grounded; every
  audited claim is supported; every selected source has at least one evidence
  hit; all visible citations resolve; and no evidence source is outside the
  selected basket.
- `needs_review`: generation completed, but at least one evidence requirement
  failed. The `quality.reasons` list preserves the failure reason.
- `draft`: the brief was created manually or its objective/body was edited after
  verification. Editing invalidates the prior audit until regeneration.

`quality.generation_mode` is independent of evidence status:

- `model_synthesis`: model-authored synthesis passed the generation gate.
- `extractive_fallback`: model synthesis failed the gate, so the product shows
  source-balanced, sentence-level cited evidence. The UI always discloses this
  fallback; it is not presented as synthesis.

A fallback may still be `verified` when the extractive sentences pass the full
evidence audit. Conversely, a fluent model answer remains `needs_review` when
its citations do not support its claims.

The React workflow requires a `verified` evidence matrix. The generation API
accepts `matrix_id` and rejects matrices from another project or matrices whose
quality state is not `verified`. Matrix-backed generation retrieves no evidence
outside the recorded matrix. The older request shape without `matrix_id`
remains available for API compatibility, but is not the product's default path.

## Version and Concurrency Rules

Every update, regeneration, and restore uses `expected_revision`. A stale write
returns HTTP 409 with the current revision rather than overwriting newer work.
List endpoints return lightweight metadata; the editor loads full content,
evidence, and trace only for the selected brief. Deletion first creates an
automatic snapshot and is blocked if that backup fails.

## API Surface

- `GET/POST /api/projects/{project_id}/research-briefs`
- `POST /api/projects/{project_id}/research-briefs/generate`
- `GET/PATCH/DELETE /api/research-briefs/{brief_id}`
- `GET /api/research-briefs/{brief_id}/revisions`
- `GET /api/research-briefs/{brief_id}/revisions/{revision}`
- `POST /api/research-briefs/{brief_id}/restore`
- `GET /api/research-briefs/{brief_id}/export?format=markdown|docx|bibtex|ris`

Regeneration requires `brief_id` and `expected_revision`. At most eight selected
basket keys are accepted per generation request. Matrix-backed generation sends
`matrix_id`; the saved brief quality record includes `source_matrix_id`,
`source_matrix_revision`, and `source_matrix_quality_status`.

## Release Checks

Run the focused implementation checks:

```bash
python -m pytest -q tests/unit/test_research_brief.py tests/unit/test_chat_store_research_briefs.py tests/unit/test_evidence_matrix.py tests/unit/test_chat_store_evidence_matrices.py tests/sanity/test_research_briefs_api.py tests/sanity/test_evidence_matrices_api.py
cd web
npm run lint
npm run build
npm run test:e2e:smoke
```

Then run the unchanged QA gates from `docs/RESEARCH_QA_EVAL_RUNBOOK.md`, including
the five-question paid-model smoke, 29-question live full-library suite,
29-question deterministic retrieval coverage, source validation, and reviewed
replay. A release is a no-go if any strict evidence, source, citation route,
reader locator, or citation-card check fails.
