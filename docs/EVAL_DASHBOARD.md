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
| Agent trace quality | Semi-automated/manual | Trace schema and scope context can be checked automatically; usefulness of plan/tool observations should be reviewed manually. |

The lightweight trace eval can write a portfolio-friendly JSON report. Metrics
that are not measured by the current dry run are emitted as `null`, not guessed.
The current answer-quality numbers are fixture-based regression checks over
recorded cases, not a live model benchmark.

## Golden Prompt Set

The first lightweight Research Agent prompt set lives at:

- `docs/research_agent_golden_v0.jsonl`

It covers single-paper QA, multi-paper comparison, reading-guide prompts,
reference-followup prompts, and an unknown/empty-query edge case. The file does
not contain scored results. It records expected planner classifications,
required tools, and manual review dimensions.

Validate the prompt set against the current planner/tool plan:

```powershell
python tools\research_qa\validate_research_agent_golden.py
```

## Answer Quality Fixture

The first recorded answer-quality fixture lives at:

- `docs/research_agent_eval_v1.jsonl`

It covers:

- local evidence-grounded answers
- hybrid local evidence plus external background
- academic no-hit external fallback disclosure
- knowledge-base-unrelated general API answers
- paper-specific insufficient-evidence disclosure

Each fixture row can also declare an `answer_profile`. Current profiles are:

- `local_evidence_grounded`: local-only answer, at least one local citation, no source-mode notice.
- `hybrid_synthesis`: local citation plus concise external-background disclosure.
- `external_academic`: no-hit academic answer with an explicit not-from-KB notice.
- `general_api`: unrelated/general API answer with no knowledge-base miss notice.
- `insufficient_local_evidence`: paper-specific no-hit answer that clearly marks the local evidence gap.

These cases are intentionally small and deterministic. They verify source
disclosure, expected answer points, citation support on local claims, and that
trace/tool/debug content stays out of the main answer. They should not be
presented as live LLM quality scores. Profile metrics check shape and product
contracts only: answer compactness, local citation presence when required, and
whether source notices are present but not repeated.

## Research QA Metrics

| Metric | Type | How to measure | Baseline |
|---|---|---|---|
| Answer correctness | Manual | Review answer against source paper(s). | TBD |
| Answer completeness | Manual | Check whether required aspects of the question are addressed. | TBD |
| Groundedness | Manual/semi-automated | Count answer claims that are supported by retrieved evidence. | TBD |
| Expected answer point coverage | Semi-automated fixture | Count expected answer points found in `docs/research_agent_eval_v1.jsonl` recorded answers. | Fixture-only |
| Answer profile accuracy | Semi-automated fixture | Check recorded answers against profile contracts for source blend, citation shape, compactness, and source-notice count. | Fixture-only |
| Answer compactness | Semi-automated fixture/manual | Check fixture answers stay within profile-specific max length; review real answers manually for relevance and readability. | Fixture-only |
| Retrieval relevance | Semi-automated/manual | Inspect top retrieved chunks for relevance to the query. | TBD |
| Multi-paper comparison usefulness | Manual | Check whether comparison claims are source-specific and not merged vaguely. | TBD |
| Structured comparison completeness | Semi-automated/manual | For comparison prompts, inspect `compare_papers` output for `paper`, `method`, `evidence`, `limitation`, and `relation_to_question`. | TBD |
| Reading guide usefulness | Manual | Check whether suggested sections form a coherent reading path. | TBD |

## Citation And Evidence Metrics

| Metric | Type | How to measure | Baseline |
|---|---|---|---|
| Citation coverage | Semi-automated | `supported_claims / total_claims` from `agent_trace.verification`. | TBD |
| Citation precision | Semi-automated fixture/manual | In the recorded quality fixture, count supported cited local claims divided by cited local claims; for live answers, review against labeled expected evidence. | Fixture-only |
| Unsupported claim count | Semi-automated | `unsupported_claims` from `agent_trace.verification`. | TBD |
| Evidence status distribution | Semi-automated | Count `grounded`, `needs_review`, `insufficient`, and `not_applicable` from `agent_trace.verification.evidence_status`. | TBD |
| Hybrid answer rate | Semi-automated | Count traces where `generate_grounded_answer.answer_mode` is `hybrid_local_external`, split by `web_search_used`. | TBD |
| No-hit external fallback rate | Semi-automated | Count no-hit traces where `generate_grounded_answer.answer_mode` is `external_academic_llm` and whether `web_search_used` is true. | TBD |
| External fallback disclosure accuracy | Semi-automated fixture/manual | Check whether external or hybrid answers visibly disclose that non-local content is not knowledge-base-grounded. | Fixture-only |
| Source notice shape accuracy | Semi-automated fixture/manual | Check that local/general answers do not show unnecessary KB-miss notices and hybrid/external answers show at most one concise source notice. | Fixture-only |
| Local citation contract accuracy | Semi-automated fixture/manual | Check that profiles requiring local evidence include at least one local citation marker. | Fixture-only |
| Unsupported claim diagnostics | Semi-automated/manual | Inspect `agent_trace.verification.claims[*].unsupported_reason` and matched evidence source summaries. | TBD |
| Evidence locate success | Manual/semi-automated | Open citation/locate targets and confirm they land near supporting text. | TBD |
| Source-card accuracy | Manual | Check title, source, DOI/reference metadata, and displayed evidence quote. | TBD |
| Reference-followup precision | Manual | For upstream/reference questions, verify cited prior work matches the answer. | TBD |
| Reference-index resolution coverage | Semi-automated/manual | For `reference_followup` traces, inspect `retrieve_references.resolved_reference_count` and spot-check that returned `ref_num/title/doi` fields match the citing paper bibliography. | TBD |

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
| Planner confidence calibration | Manual/semi-automated | Inspect `agent_trace.context.planner_intent.confidence` against reviewed task labels. | TBD |
| Evidence need routing | Automated/manual | Confirm `agent_trace.context.planner_intent.evidence_need` is high for comparison, reference, method, limitation, and experiment prompts. | TBD |
| Plan execution completeness | Automated | Check every planned step reaches `done`, `error`, or `skipped`. | TBD |
| Tool latency | Semi-automated | Use `agent_trace.steps[*].elapsed_ms` to compute P50/P95 by tool and end-to-end trace. | TBD |
| Tool observation usefulness | Manual | Review whether observations explain what happened without leaking internals. | TBD |
| Trace actionability | Automated/manual | For reference-followup traces, confirm resolved references expose reader-open and basket-add actions. | TBD |
| Trace scope reproducibility | Automated/manual | Inspect `agent_trace.context.query_scope`, `requested_query_scope`, and selected/current-source fields for scoped questions. | TBD |
| Evidence matrix completeness | Automated/manual | Inspect `agent_trace.research_run.evidence_matrix` rows for paper, method, result, limitation, evidence quote, and support status on comparison/reference prompts. | TBD |
| Pre-answer matrix use | Automated/manual | Confirm generated-answer prompts receive `agent_notes.evidence_matrix` and instruct synthesis from the matrix before free-form summarization. | TBD |
| Answer source routing | Automated/manual | Track `source_blend_accuracy`, `unnecessary_notice_rate`, and `required_notice_accuracy` for `local_grounded`, `hybrid_local_external`, `external_academic`, and `general_llm` cases. | Fixture-only |
| Compact source summary accuracy | Automated | Track `source_summary_accuracy`, `source_summary_present_rate`, and `source_summary_shape_accuracy` for the user-facing `agent_source_summary` chip payload. | Fixture/replay |
| Answer runtime self-check | Semi-automated | Inspect stored `message.meta.answer_runtime_check` or stream `answer_runtime_check` for profile/source-summary/notice-shape/clutter failures. Deterministic pre-storage repair may remove debug clutter, dedupe source notices, or add one required source notice; the check remains meta-only and should not be rendered in the main answer. | TBD |
| Answer quality repair rate | Semi-automated | Count generated answers whose quality gate status is `passed`, `repaired`, or `fallback` in tool outputs; leave rates null when no gate status is recorded. | TBD |
| Source policy disclosure | Automated/manual | Confirm `agent_trace.research_run.source_policy` distinguishes local-only, local+external background, and external-with-notice answers. | TBD |
| Trace summary readability | Automated/manual | Confirm `agent_trace.summary` exposes claim support, scope, tool-call count, and error status, while plan/tool logs remain behind the UI's Execution Details disclosure. | TBD |
| Trace audit replay | Automated/manual | Use `GET /api/messages/{message_id}/agent-trace` to confirm stored traces can be re-opened without adding default answer clutter. | TBD |
| Answer clutter guardrail | Automated/manual | Confirm plan steps, tool calls, trace JSON, and verification details stay out of the main answer body, API stream text, and stored assistant message unless the user opens the trace panel. | TBD |
| Error trace availability | Automated | Simulate degraded/error paths and confirm `agent_trace.errors` is returned. | TBD |
| Trace payload size | Semi-automated | Inspect serialized message meta size; compact large claim/hit lists. | TBD |
| Main-answer clutter rate | Semi-automated fixture/e2e | Confirm answer text does not include `agent_trace`, plan steps, tool calls, or verification statistics. | Fixture-only |

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

Research Agent golden prompt validation:

```powershell
python tools\research_qa\validate_research_agent_golden.py
```

Research Agent trace schema and scope-context validation:

```powershell
python tools\research_qa\run_agent_trace_eval.py --json-out test_results\agent_trace_eval.json
```

This command validates both `docs/research_agent_golden_v0.jsonl` and
`docs/research_agent_eval_v1.jsonl` by default. Use `--skip-quality` to run only
planner/schema validation.

Real Research Agent answer replay:

```powershell
python tools\research_qa\export_research_agent_samples.py --db chat.sqlite3 --out test_results\research_agent_answer_samples.jsonl --limit 50
python tools\research_qa\run_agent_trace_eval.py --real-samples test_results\research_agent_answer_samples.jsonl --json-out test_results\agent_trace_real_replay_eval.json
```

The exported replay file is semi-automated and unlabeled by default. It is useful
for checking source-mode disclosure, trace/tool clutter in user-facing answers,
and routing regressions on real conversations. It is not a correctness benchmark.
Use `--check-local-support` on the export command only after reviewing that the
exported snippets are suitable for claim-support checks.

Human-reviewed replay labels:

```powershell
python tools\research_qa\review_research_agent_samples.py prepare --samples test_results\research_agent_answer_samples.jsonl --labels test_results\research_agent_answer_labels.jsonl
# Edit test_results\research_agent_answer_labels.jsonl:
# - set review_status to accepted
# - fill expected_source_blend
# - add expected_answer_points
# - optionally add expected_source_keywords and should_use_local_evidence
python tools\research_qa\review_research_agent_samples.py merge --samples test_results\research_agent_answer_samples.jsonl --labels test_results\research_agent_answer_labels.jsonl --out test_results\research_agent_answer_reviewed.jsonl
python tools\research_qa\run_agent_trace_eval.py --real-samples test_results\research_agent_answer_reviewed.jsonl --json-out test_results\agent_trace_real_reviewed_eval.json
python tools\research_qa\run_reviewed_replay_eval.py
```

Reviewed replay samples become stricter eval cases because the merge step only
includes labels marked `accepted`. Accepted labels must include
`expected_source_blend` and at least one `expected_answer_points` item by default.
Keep these files local unless the prompts, answers, and evidence snippets have
been reviewed for privacy and publication suitability.

`run_reviewed_replay_eval.py` is the CI-friendly quality gate. It checks the
committed `docs/research_agent_reviewed_replay.jsonl` de-identified smoke fixture
and `test_results/research_agent_answer_reviewed.jsonl` for local private
reviewed samples. Missing local reviewed datasets are skipped successfully. If
reviewed cases exist, the gate requires the file to contain only accepted
reviewed cases and runs the strict answer-quality evaluator.
The committed fixture intentionally covers local evidence, local+external,
external academic fallback, general API answers, and paper-specific insufficient
evidence so routing and source-summary regressions are caught in CI.

To print the legacy summary shape only:

```powershell
python tools\research_qa\run_agent_trace_eval.py --summary-only
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
6. For reference-followup prompts, inspect the `retrieve_references` tool output for resolved upstream references.
7. Use trace actions to open one upstream reference in the reader and add one to the literature basket.
8. Open evidence/citation cards and confirm that cited claims map to source text.

## Limitations

- The current planner is heuristic and keyword-based.
- Claim verification is sentence-level and citation/evidence-overlap based; it is not a semantic entailment model.
- Degraded mode can summarize retrieved snippets without an LLM, but it cannot synthesize a full research answer.
- Some metrics remain manual until curated gold datasets and grading rubrics are added.
- The answer-quality fixture is a deterministic regression suite, not a live
  benchmark of model quality across arbitrary papers.
- Real chat replay samples are unlabeled unless a reviewer adds expected answer
  points and source labels to the exported JSONL.
- Trace usefulness should be reviewed by researchers, not judged only by schema validity.

## Future Work

- Add a curated research QA benchmark with paper IDs, expected evidence, and rubric notes.
- Add automated trace-schema validation for stored assistant messages.
- Track citation support over time in a local dashboard.
- Add source-specific comparison rubrics for multi-paper questions.
- Add regression fixtures for degraded mode with no text API key.
