# Evaluation Dashboard

This dashboard defines how Pi_zaya should be evaluated as an evidence-grounded
research agent. It is a tracking document, not a report of measured results.
Do not fill baseline numbers unless they come from a reproducible run or a
documented manual review.

## Evaluation Scope

| Area | Status | Notes |
|---|---|---|
| Research QA quality | Semi-automated/manual | Existing dry-run tooling and fixture replay can check harness health; answer quality still needs human review. |
| Citation/evidence quality | Semi-automated/manual | Citation rendering and grounding have unit coverage; claim-level answer support needs review on real outputs. |
| Converter quality | Semi-automated | Existing converter quality runners and unit tests cover structural regressions. |
| Agent trace quality | Semi-automated/manual | New trace schema can be checked automatically; usefulness of plan/tool observations should be reviewed manually. |

## Research QA Metrics

| Metric | Type | How to measure | Baseline |
|---|---|---|---|
| Answer correctness | Manual | Review answer against source paper(s). | TBD |
| Answer completeness | Manual | Check whether required aspects of the question are addressed. | TBD |
| Groundedness | Manual/semi-automated | Count answer claims that are supported by retrieved evidence. | TBD |
| Retrieval relevance | Semi-automated/manual | Inspect top retrieved chunks for relevance to the query. | TBD |
| Multi-paper comparison usefulness | Manual | Check whether comparison claims are source-specific and not merged vaguely. | TBD |
| Reading guide usefulness | Manual | Check whether suggested sections form a coherent reading path. | TBD |

## Citation And Evidence Metrics

| Metric | Type | How to measure | Baseline |
|---|---|---|---|
| Citation coverage | Semi-automated | `supported_claims / total_claims` from `agent_trace.verification`. | TBD |
| Unsupported claim count | Semi-automated | `unsupported_claims` from `agent_trace.verification`. | TBD |
| Evidence locate success | Manual/semi-automated | Open citation/locate targets and confirm they land near supporting text. | TBD |
| Source-card accuracy | Manual | Check title, source, DOI/reference metadata, and displayed evidence quote. | TBD |
| Reference-followup precision | Manual | For upstream/reference questions, verify cited prior work matches the answer. | TBD |

## Converter Quality Metrics

| Metric | Type | How to measure | Baseline |
|---|---|---|---|
| Page marker preservation | Automated | Unit tests and conversion output inspection for `<!-- kb_page: N -->`. | TBD |
| Heading structure quality | Semi-automated | Converter quality reports and manual spot checks. | TBD |
| Figure/table/formula preservation | Semi-automated/manual | Quality center diagnostics plus Reader inspection. | TBD |
| Markdown indexability | Automated | Ingest/index tests and dry-run rebuilds. | TBD |
| Repair safety | Automated/manual | Repair tests plus manual diff review for representative papers. | TBD |

## Agent Trace Metrics

| Metric | Type | How to measure | Baseline |
|---|---|---|---|
| Planner classification accuracy | Automated/manual | Unit tests for keyword classes; manual review on real prompts. | TBD |
| Plan execution completeness | Automated | Check every planned step reaches `done`, `error`, or `skipped`. | TBD |
| Tool observation usefulness | Manual | Review whether observations explain what happened without leaking internals. | TBD |
| Error trace availability | Automated | Simulate degraded/error paths and confirm `agent_trace.errors` is returned. | TBD |
| Trace payload size | Semi-automated | Inspect serialized message meta size; compact large claim/hit lists. | TBD |

## Baseline Tables

Use these tables when a reproducible run exists.

### Research QA Baseline

| Dataset/run | Date | Questions | Correct | Complete | Grounded | Notes |
|---|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | No measured baseline recorded yet. |

### Citation Baseline

| Dataset/run | Date | Total claims | Supported claims | Unsupported claims | Locate success | Notes |
|---|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | No measured baseline recorded yet. |

### Converter Baseline

| Dataset/run | Date | PDFs | Page markers OK | Structure OK | Repair needed | Notes |
|---|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | No measured baseline recorded yet. |

### Agent Trace Baseline

| Dataset/run | Date | Prompts | Planner OK | Complete traces | Error traces OK | Notes |
|---|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | No measured baseline recorded yet. |

## How To Run Evals

Backend unit tests:

```powershell
python -m pytest tests/unit -q
```

Research QA dry run:

```powershell
python tools\research_qa\run_research_qa_eval.py --dry-run
```

Converter quality dry run:

```powershell
python tools\converter_quality\run_converter_quality_eval.py --dry-run
```

Frontend checks:

```powershell
cd web
npm run lint
npm run build
npm run test:e2e:smoke
```

Manual Research Agent review:

1. Start the app with `.\run_new.ps1 -StopExisting`.
2. Upload or use a small known paper set.
3. Ask one question for each type: single-paper QA, comparison, reading guide, and reference followup.
4. Enable the `Agent` composer toggle.
5. Record the returned `agent_trace.question_type`, plan steps, tool calls, and verification summary.
6. Open evidence/citation cards and confirm that cited claims map to source text.

## Limitations

- The current planner is heuristic and keyword-based.
- Claim verification is sentence-level and citation/evidence-overlap based; it is not a semantic entailment model.
- Degraded mode can summarize retrieved snippets without an LLM, but it cannot synthesize a full research answer.
- Some metrics remain manual until curated gold datasets and grading rubrics are added.
- Trace usefulness should be reviewed by researchers, not judged only by schema validity.

## Future Work

- Add a curated research QA benchmark with paper IDs, expected evidence, and rubric notes.
- Add automated trace-schema validation for stored assistant messages.
- Track citation support over time in a local dashboard.
- Add source-specific comparison rubrics for multi-paper questions.
- Add regression fixtures for degraded mode with no text API key.
