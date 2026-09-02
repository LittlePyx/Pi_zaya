# Dense Retrieval and Architecture Control Plan

Status: `phase 1 implemented and validated; embedding product activation pending`

Started: `2026-09-02`

Baseline: `main@48228fa6` (published version remains `v0.1.0-beta.13`, tag `9c78a763`)

## 1. Objective and boundary

Improve retrieval recall, ranking quality, evaluation coverage, and observability
without expanding Pi_zaya beyond its product boundary. The system continues to
help users read papers already in their library and find, compare, verify, and
organize traceable evidence. Dense retrieval, reranking, evaluation, and tracing
must not introduce manuscript-writing or submission-preparation behavior.

This plan also prevents the new retrieval work from increasing the existing
large-file architecture debt.

## 2. Verified baseline

The pre-change repository contains approximately:

- 336,018 raw production lines across `api`, `kb`, and `web/src`;
- 226,268 Python lines in `api` and `kb`;
- 89,528 TypeScript/TSX lines and 20,222 CSS lines in `web/src`;
- 163,090 test lines.

The largest current modules include:

| File | Baseline lines |
|---|---:|
| `api/chat_render.py` | 17,380 |
| `api/reference_ui.py` | 11,208 |
| `kb/generation_answer_finalize_runtime.py` | 10,721 |
| `kb/task_runtime.py` | 10,529 |
| `api/routers/library.py` | 9,156 |
| `api/routers/references.py` | 7,883 |
| `kb/retrieval_engine.py` | 5,098 before this slice |

The current product retrieval path is BM25 plus structured indexes, query
translation, deterministic/optional LLM query expansion, RRF across query
variants, source/heading/table/page rules, and optional LLM reranking. It has no
production embedding index or dense candidate retrieval. The Research Agent's
central citation verifier is sentence-level and lexical-overlap based; separate
grounding layers add numeric, method-identity, relation, scope, and locator
checks, but do not provide general entailment/contradiction verification.

At this baseline the local development database has 31 documents and 2,251
chunks. Exact cosine search over a compact matrix is sufficient at this scale;
a hosted vector database would add operational complexity without demonstrated
benefit.

## 3. Target retrieval pipeline

```text
exact anchor / structured preflight
              |
        scope-qualified chunks
              |
      +-------+-------+
      |               |
   BM25 top-N      Dense top-N
      |               |
      +---- RRF -------+
              |
      cross-encoder top-M
              |
 current document/source/quality/locator gates
              |
       grounded answer generation
              |
 deterministic + semantic claim verification
```

Rules:

1. Exact paper, section, figure, equation, table, and citation navigation keeps
   the existing structured path; dense retrieval must not override it.
2. BM25 remains the fail-safe channel. Dense index/model/provider failures fall
   back without taking down local retrieval.
3. RRF is the first fusion method because sparse and dense scores are not
   naturally calibrated. Weighted score fusion may be evaluated later, but may
   not replace RRF without benchmark evidence.
4. Every returned item remains an existing local chunk with its document,
   page, heading, block, and evidence-quality metadata.
5. External model knowledge and web pages are not added to the dense index.

## 4. User choice and privacy contract

The intended setting is:

- `Off`: current BM25 and structured retrieval only;
- `Local`: a packaged or explicitly installed local embedding model;
- `Cloud`: an explicitly configured embedding API.

Cloud mode must explain that initial indexing sends local paper chunks to the
embedding provider and each search sends the query. This is different from
allowing external model knowledge in an answer, but it is still external data
processing and therefore requires separate consent.

No cloud endpoint, model download, telemetry exporter, or paid request may be
activated merely because code support exists. Defaults remain off until the
corresponding UI, status, recovery, and consent contracts land together.

## 5. Index and provider design

The production index layer must be separate from candidate ranking and record:

- committed `docs.json` signature;
- chunk schema version and ordered chunk IDs;
- embedding provider kind (`local` or `cloud`), model ID, revision/fingerprint,
  vector dimension, normalization, and build time;
- per-document completion and error state;
- an atomic manifest/vector commit so partially written embeddings are never
  searched.

Indexing must be incremental by document. A stale, incomplete, dimension-
mismatched, or model-mismatched index is unavailable rather than silently
mixed with current chunks. BM25 remains usable while a dense rebuild is pending.

For the current corpus size, use an on-disk numeric matrix and exact cosine
search. Reconsider HNSW/FAISS or an external vector store only after measured
corpus size or latency demonstrates the need.

## 6. Cross-encoder reranking

Cross-encoder reranking runs only on the fused top-M candidates. It must be
implemented through a model adapter so retrieval logic does not own downloads,
API credentials, or network policy. Required behavior:

- bounded candidate count and timeout;
- deterministic preservation of the original fused order on any failure;
- recorded pre-rerank and cross-encoder scores;
- no loss of source/locator metadata;
- separate local and cloud consent/readiness states.

Phase 0 supplies this adapter contract and fail-open behavior. It does not ship
a model weight or claim production cross-encoder quality.

## 7. Retrieval evaluation

### 7.1 Metrics

Every retrieval run must report:

- `Recall@K`: fraction of all relevant chunks/documents recovered in top K;
- `MRR`: reciprocal rank of the first relevant result;
- `nDCG@K`: graded relevance and ordering quality;
- wrong-source rate and exact-locator success for product-specific safety;
- p50/p95 latency per channel, fusion, and reranking;
- dense index size, freshness, build duration, and failure/fallback rate.

### 7.2 Evaluation sets

There are two tiers:

1. `docs/retrieval_eval_v1.json` is a small deterministic committed regression
   seed. It validates ranking metrics and protects exact BM25 behavior while
   retaining an explicit semantic-gap case. It is not a claim of general model
   quality.
2. The frozen unseen-paper benchmark in
   `docs/research_qa_unseen_corpus_v2.json` remains the real-paper acceptance
   source. Phase 1 projected its 22 questions and 26 source/page/evidence
   contracts into 27 fixed chunk qrels in `docs/retrieval_eval_unseen_v2.json`.
   The derived fixture pins the source-fixture SHA-256, path-independent corpus
   fingerprint, indexed chunk IDs, page spans, and labelled-text hashes. It was
   frozen before any embedding model was selected or measured.

The real set must include exact identifiers, tables/numbers, paraphrases,
bilingual queries, negation/boundaries, multi-paper comparison, and hard
same-topic wrong-paper negatives. Labels must be frozen before comparing model
candidates.

### 7.3 Initial promotion thresholds

A candidate may enter an opt-in product path only when:

- exact-query Recall@5, MRR, and nDCG@5 do not regress from BM25;
- semantic/paraphrase Recall@10 improves by at least 15 percentage points over
  the frozen BM25 baseline;
- wrong-source rate and exact-locator success do not regress;
- no existing Research QA, citation, library, conversion, or browser gate
  regresses;
- latency and package-size changes are measured and documented rather than
  hidden behind an average.

Thresholds may be tightened after the first frozen real-corpus baseline. They
must not be weakened after seeing a candidate result without a recorded reason.

## 8. Answer and citation evaluation

Retrieval metrics alone are insufficient. End-to-end comparison must report:

- answer correctness against registered answer points;
- faithfulness: factual answer claims supported by their bound local evidence;
- citation precision: cited sources/passages that support the attached claim;
- citation completeness/recall: checkable local claims with a supporting cite;
- source and page/anchor accuracy;
- contradiction and unsupported-claim rates;
- external-source disclosure accuracy when the user opted into supplementation.

The existing Research QA fixture, unseen-paper suite, grounded replay, reviewed
Agent replay, source-grounding gates, citation audit, and browser locate checks
remain authoritative. New aggregate metrics must consume those contracts rather
than replacing them with an unreviewed LLM-as-judge score.

Semantic entailment may be added as a second-stage check for high-risk or
ambiguous claims. It must distinguish `supported`, `contradicted`, and
`insufficient`. On failure, the system should retrieve again, narrow/remove the
claim, or return an evidence boundary; it must not hand an unexplained
"verification failed" state to the user.

## 9. Observability decision

Use a vendor-neutral internal span/event contract aligned with OpenTelemetry.
OpenTelemetry is the preferred first integration because it can remain local
and can export to multiple backends. Langfuse may be an optional explicit
exporter for development/evaluation, not a required product dependency.

Required spans/events:

- request/trace ID and conversation/task identity;
- query routing and selected local scope;
- BM25, dense, fusion, cross-encoder, structured lookup, and deep-read latency;
- candidate counts and anonymized chunk/document identities;
- tool names, result status, retry/fallback, and elapsed time;
- model/provider class, token counts, model latency, and error category;
- evidence/verification aggregate outcomes.

Privacy rules:

- raw questions, paper text, evidence excerpts, API keys, and full filesystem
  paths are not exported by default;
- local structured traces remain the default;
- any remote exporter is separately enabled and visibly disclosed;
- exporter failure never affects answering or retrieval.

The current `research_trace` and performance events already capture parts of
tool execution and latency. A later phase will map them to the common span
contract instead of instrumenting every large runtime independently.

## 10. Architecture control

`tools/check_architecture_budgets.py` freezes the current size of the largest
backend and frontend files. It is a ratchet, not a declaration that their sizes
are acceptable:

- a protected file may shrink without changing the budget;
- growth fails CI and the Windows release workflow;
- new retrieval code must live behind small domain modules;
- a limit must not be raised to make a feature patch pass.

Longer-term refactoring priorities:

1. Replace the 3,304-line `_gen_worker` with named orchestration stages and a
   typed generation state.
2. Establish one canonical `Claim -> EvidenceBinding -> Locator` contract.
3. Make rendering a projection of that contract rather than another semantic
   repair layer.
4. Consolidate paper-specific repair rules into test-backed domain policies or
   remove them when the canonical contract makes them redundant.
5. Split frontend state selectors, effects, and visual components without
   reimplementing backend evidence decisions in React.

## 11. Delivery phases

| Phase | Scope | Status |
|---|---|---|
| 0 | truthful README, candidate protocol, exact dense kernel, BM25+dense RRF, cross-encoder adapter, metric functions, regression seed, size ratchet | Implemented and validated in commit `ba3c3d00` |
| 1 | frozen real-paper retrieval labels and BM25 baseline report | Implemented and validated in current working tree |
| 2 | atomic embedding index, local/cloud provider adapters, indexing status and recovery | Planned |
| 3 | explicit Off/Local/Cloud user settings and opt-in hybrid product wiring | Planned |
| 4 | benchmark local/cloud embedding and cross-encoder candidates; promote only accepted models | Planned |
| 5 | semantic contradiction verifier for high-risk claims | Planned |
| 6 | OpenTelemetry-aligned local spans and optional Langfuse exporter | Planned |
| 7 | runtime/render/evidence-contract decomposition under the size ratchet | Planned, incremental |

## 12. Modification log

### 2026-09-02 — Phase 0

- Added `kb/candidate_retriever.py` with the minimal candidate-search contract.
- Added `kb/dense_retriever.py` with exact normalized cosine search over
  caller-supplied embeddings, evidence-quality exclusion, and immutable result
  metadata.
- Added `kb/hybrid_retriever.py` with weighted RRF and BM25 fail-safe behavior.
- Updated the retrieval-engine type boundary without activating dense retrieval
  in the product path.
- Added `kb/cross_encoder_reranker.py` with bounded adapter and fail-open order
  preservation.
- Added `kb/retrieval_metrics.py` for Recall@K, MRR, nDCG@K, and aggregate means.
- Added `docs/retrieval_eval_v1.json` plus
  `tools/retrieval_eval/run_retrieval_eval.py` and CI/release execution.
- Added unit coverage for dense retrieval, fusion, failure fallback,
  cross-encoder ordering, and retrieval metrics.
- Added `tools/check_architecture_budgets.py` to CI and release gates.
- Replaced README's ambiguous "Hybrid Retrieval" label with the truthful
  current `BM25 + Structured Retrieval` label and documented that dense is not
  yet product-enabled.

### 2026-09-02 — Phase 0 validation

- `git diff --check`: pass.
- Ruff over the full repository: pass.
- Architecture line-budget ratchet: 21/21 protected files within their frozen
  baseline; pass.
- Final-tree backend unit suite: 4,583 passed, 41 skipped.
- Backend sanity suite: 305 passed, 2 skipped.
- Focused Dense/Hybrid/reranker/metric tests: 7 passed; the complete unit total
  above includes them.
- Candidate retrieval seed: 11 cases; exact group (10 cases) has Recall@5,
  MRR, and nDCG@5 all equal to 1.0. The registered semantic-gap case has
  Recall@1 0.0, rank 3, MRR 0.333333, and nDCG@5 0.5, preserving a measurable
  sparse baseline for later Dense comparison.
- Research QA fixture: 56 cases and 41 source-grounded cases; pass.
- Version A/B, evidence comparison (5), comparison candidates (5), project
  status (5), and project journey fixtures: pass.
- Grounded replay: 6/6; reviewed Agent replay: 5/5; pass.
- Converter quality: 13/13 existing Markdown cases; pass.
- Frontend lint and production build: pass.
- Browser smoke: 160 passed, 3 expected build-mode skips.
- Core citation/library browser regressions: 127/127.
- Ordinary-user surface isolation: 4/4.
- Evidence-workspace release isolation: 1/1.
- Dedicated library-quality browser suite: 24/24.

No release version or immutable tag was changed. Dense retrieval and
cross-encoder model execution remain outside the product default path until the
remaining phases satisfy their own acceptance gates.

### 2026-09-02 — Phase 1 real-paper retrieval benchmark

- Added `kb/retrieval_eval_fixture.py` to project the already pre-registered
  Research QA evidence contracts into retrieval labels without consulting any
  retriever ranking. Labels are bounded by registered paper and source page,
  then selected by registered evidence-term coverage.
- Added `docs/retrieval_eval_unseen_v2.json`: 10 papers, 1,673 indexed chunks,
  22 questions, 26 claim-evidence contracts, and 27 fixed relevant-chunk labels.
- The fixture pins source fixture SHA-256
  `1d8ca4283355aa0cfafcb845e75107fab9c70ab75264ea7c6b2e61d5b5d89ae6`
  and path-independent corpus fingerprint
  `d1602fb0d3a7320806dc1cf1a9c3615d7fc1864471e93270f2135f8f44f27516`.
  It stores no paper text or machine-specific path.
- Added `kb/retrieval_benchmark.py` and
  `tools/retrieval_eval/run_research_qa_baseline.py`. The runner validates the
  exact corpus and every labelled-text hash, then reports Recall@K, contract
  Recall@K, MRR, nDCG@K, wrong-source rate, latency, locale, evaluation-focus,
  query-scope, and coverage slices.
- Added `tools/retrieval_eval/project_research_qa_labels.py --check` so the
  projection can be reproduced against the isolated corpus without silently
  rewriting the committed labels.
- Added `docs/retrieval_bm25_baseline_unseen_v2.json` with both raw BM25 and the
  current product-query path (deterministic translation/query variants), each
  measured in scope-qualified and full-library modes.
- CI and the Windows release workflow validate that the frozen qrels, source
  fixture hash, baseline fixture hash, corpus fingerprint, channels, modes, and
  case counts remain internally consistent. The copyrighted isolated corpus is
  not copied into CI.
- Added five unit tests covering page-bound projection, path-independent corpus
  identity, missing-evidence rejection, scoped/full-library metrics, and text
  drift detection.

Reproduction against the retained isolated corpus:

```powershell
python tools/retrieval_eval/project_research_qa_labels.py `
  --db-root F:\research-papers\2026\Jan\else\kb_chat_eval\unseen_v2\db `
  --check
python tools/retrieval_eval/run_research_qa_baseline.py `
  --db-root F:\research-papers\2026\Jan\else\kb_chat_eval\unseen_v2\db `
  --output docs/retrieval_bm25_baseline_unseen_v2.json `
  --compact
python tools/retrieval_eval/run_research_qa_baseline.py --validate-only
```

#### Frozen BM25 baseline

| Channel / scope | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Wrong source@5 |
|---|---:|---:|---:|---:|---:|---:|
| Raw BM25 / scope-qualified | 0.6136 | 0.8409 | 0.9091 | 0.5229 | 0.5861 | 0.0000 |
| Product-query BM25 / scope-qualified | 0.6136 | 0.8182 | 0.8636 | 0.4686 | 0.5330 | 0.0000 |
| Raw BM25 / full library | 0.5227 | 0.7045 | 0.8409 | 0.5092 | 0.5310 | 0.4273 |
| Product-query BM25 / full library | 0.5682 | 0.7273 | 0.8182 | 0.4285 | 0.4875 | 0.0818 |

Interpretation:

- Current translation/query variants cut full-library wrong-source@5 from
  42.73% to 8.18% and improve full-library Recall@5 by 4.55 percentage points.
  They nevertheless reduce MRR and nDCG, and reduce scope-qualified Recall@10
  by 2.27 points. Query fusion therefore needs measurement and calibration; it
  must not be assumed beneficial at every ranking depth.
- Product-query scope-qualified Recall@5 is 79.17% for English questions but
  only 40.00% for Chinese questions. Multilingual semantic matching is the
  clearest Dense-retrieval opportunity.
- Cross-paper synthesis has only 50.00% Recall@10. Exact-location and
  quantitative slices each have 66.67% Recall@10. Dense candidates must improve
  these without weakening exact figure, table, formula, source, or page routes.
- `sam2-streaming-memory` and `gemma3-vision-tokens` have no labelled evidence
  in the product-query top 20. `deepseek-r1-zero-boundary` first appears at rank
  13, and only one of two `segmentation-transfer-synthesis` contracts is covered
  by rank 20. These cases are retained as the first hard-negative/error-analysis
  set; none is removed or relabelled after observing the baseline.

### Priority assessment after Phase 1

| Priority | Work | User value | Dependency/risk | Decision |
|---|---|---:|---|---|
| P0.1 | Atomic, incremental, path-independent embedding index plus local provider boundary | Very high | Required before any repeatable Dense benchmark or safe recovery | Do next |
| P0.2 | Run multilingual Dense and BM25+Dense RRF over the frozen 22-case qrels; tune only on a separate development slice | Very high | Requires P0.1; avoid tuning on the acceptance report | Immediately after P0.1 |
| P0.3 | Cross-encoder candidate benchmark and bounded top-M calibration | High | Useful only after Dense raises candidate recall | After accepted Dense candidate set |
| P1.1 | Answer correctness, faithfulness, citation precision/completeness, and contradiction repair metrics derived from existing Research QA contracts | Very high | Must preserve deterministic checks and must resolve failures rather than expose status labels | Before product-wide hybrid activation |
| P1.2 | Explicit Off/Local/Cloud settings, consent, index status, rebuild, and fallback UI | High | Requires an accepted index/provider and measured privacy/latency/package cost | Gate product activation |
| P1.3 | OpenTelemetry-aligned local spans for retrieval, reranking, tools, tokens, latency, and fallback; optional Langfuse export | Medium-high | Must redact questions, paper text, keys, and paths by default | Land before opt-in rollout |
| P2 | Decompose generation/render/evidence runtimes under the size ratchet | High long-term | Broad regression surface; should proceed in small contract-preserving slices | Incremental, not on the critical retrieval path |

The immediate implementation priority is P0.1, not UI activation. A reliable
index manifest and provider boundary make model comparisons reproducible and
give BM25 a safe fallback during stale, incomplete, failed, or model-mismatched
embedding builds. Phase 2 will remain non-product-enabled until P0.2 establishes
an accepted quality delta on this frozen baseline.

### 2026-09-02 — Phase 1 validation

- Qrels projection `--check`: 22 cases, 26 contracts, 27 labels; pass.
- Frozen qrels/baseline CI metadata validation: 22 cases and 1,673 chunks; pass.
- `git diff --check`, full Ruff, and the 21-file architecture budget: pass.
- Focused retrieval-fixture tests: 5/5; pass.
- Backend unit suite: 4,588 passed, 41 skipped.
- Backend sanity suite: 305 passed, 2 skipped.
- Deterministic retrieval seed: 11/11; exact subset Recall@5, MRR, and nDCG@5
  remain 1.0; pass.
- Research QA fixture: 56 cases and 41 source-grounded cases; pass.
- Version A/B fixture: 34 expected QA cases plus three project journeys; pass.
- Evidence comparison 5, comparison candidates 5, project status 5, and
  project journey: pass.
- Grounded replay 6/6 and reviewed Agent replay 5/5: pass.
- Converter quality: 13/13; pass.
- Frontend lint and production build: pass.
- Browser smoke: 160 passed, 3 expected private/build-mode skips.
- Core citation/library browser regressions: 127/127.
- Ordinary-user surface isolation: 4/4.
- Evidence-workspace release isolation: 1/1.
- Dedicated library-quality suite: 24/24.

An exploratory attempt to run five browser servers concurrently caused four
unrelated 8-second page-readiness timeouts. The release-equivalent serial reruns
of the complete affected suites passed with no code, assertion, worker, or
timeout change; only the serial results above are treated as the gate outcome.

No release version or immutable tag was changed. Product retrieval remains on
the existing BM25/structured path; this phase adds measurement and integrity
infrastructure, not an unmeasured Dense default.
