# Dense Retrieval and Architecture Control Plan

Status: `phase 0 implemented and validated; product activation pending`

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
   `docs/research_qa_unseen_corpus_v1.json` remains the real-paper acceptance
   source. Its questions and source/page evidence contracts must be projected
   into a versioned retrieval ranking set before a dense model is selected.

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
| 0 | truthful README, candidate protocol, exact dense kernel, BM25+dense RRF, cross-encoder adapter, metric functions, regression seed, size ratchet | Implemented and validated in current working tree |
| 1 | frozen real-paper retrieval labels and BM25 baseline report | Planned |
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
