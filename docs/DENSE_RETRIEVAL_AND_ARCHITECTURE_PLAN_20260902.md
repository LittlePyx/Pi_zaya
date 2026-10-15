# Dense Retrieval and Architecture Control Plan

Status: `2026-09-07: real-backend shelf CAS and durable operation receipts accepted through final local regression gates; next is scoped dense source-boundary repair and paired end-result evaluation; published beta.13 remains unchanged`

Current decision entry: [当前优化路线](CURRENT_OPTIMIZATION_ROADMAP.md).
The historical sections below retain their original evidence and status; they are not the current priority queue.

Started: `2026-09-02`

Baseline: `main@677e2a5a` (published version remains `v0.1.0-beta.13`, tag `9c78a763`)

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

Phase 2A implements this as immutable generation directories plus one atomically
replaced `current.json` pointer. Each generation is independently checksummed.
The builder takes a short committed-corpus snapshot under the existing database
lock, releases that lock while the provider runs, then rechecks the corpus
before publishing. A concurrent library change therefore aborts the build
instead of publishing mixed vectors.

Incrementality is based on ordered per-document text hashes, not machine paths
or the current path-derived document ID. Moving/reimporting unchanged papers
can reuse their embeddings while still committing the new `docs.json` digest
and ordered chunk IDs. Metadata changes create a fresh generation but do not
recompute vectors whose input text is unchanged. Manifests and build-status
records contain hashes and opaque document tokens, not paper text or filesystem
paths.

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

There are three tiers:

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
3. `docs/retrieval_eval_development_v1.json` projects 41 established
   source-grounded library QA cases into 46 evidence contracts and 46 labelled
   chunks over all 31 local papers and 2,251 chunks. Papers without a matching
   contract remain in the corpus as background negatives. This historical
   fixture is explicitly development-only: it may select fusion parameters but
   cannot be reported as unseen acceptance evidence.

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
| 1 | frozen real-paper retrieval labels and BM25 baseline report | Implemented and validated in commit `f143b43b` |
| 2A | atomic exact-vector generations, path-independent document reuse, provider fingerprint contract, indexing status and safe failure recovery | Implemented and fully validated |
| 2B.1 | pinned local Transformers adapter and raw multilingual Dense benchmark; no product wiring | Implemented |
| 2B.2 | preregistered BM25+dense RRF measurement and error analysis | Implemented; candidate not promoted because full-library wrong-source gate regressed |
| 2B.3 | independent real-paper development qrels and source-safe product-query BM25+dense fusion | Implemented; `sparse_weight=2`, `dense_weight=1`, `rrf_k=20`, protected sparse top 5, Dense source gate at sparse top 5 selected |
| 2B.4 | one locked source-safe fusion acceptance run on unseen-v2 | Implemented; top-five safety preserved and aggregate ranking improved, but product activation remains gated |
| 2C | pinned bounded cross-encoder benchmark over the accepted safe-RRF top 20 | Implemented on development fixture and rejected before unseen acceptance; gains do not justify 493 MB cache/runtime cost |
| 2D | aggregate answer correctness, faithfulness, citation accuracy, contradiction, locator, and automatic-repair outcomes | Next |
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

### 2026-09-02 — Phase 2A atomic incremental embedding foundation

- Added `kb/embedding_provider.py` with a stable provider descriptor covering
  local/cloud processing class, model ID, revision, weights fingerprint,
  dimension, and normalization. `LocalEmbeddingProvider` accepts only an
  explicitly supplied executor; it downloads no model, reads no credential,
  and makes no network request.
- Moved shared finite/dimension/L2 validation into
  `kb/embedding_vectors.py`, used by both provider adapters and exact cosine
  retrieval.
- Added `kb/embedding_index.py` with immutable, checksummed Float32 generations
  and an atomic current pointer. Search loading rejects stale `docs.json`,
  changed chunk order/metadata, chunk-schema drift, model/revision/fingerprint
  mismatch, dimension mismatch, partial document coverage, and corrupt
  manifests or vectors.
- Embedding calls are incremental by the ordered text sequence of each paper.
  A relocated or reimported unchanged paper reuses vectors even when its
  path-derived document/chunk IDs change; the replacement generation still
  records the new committed document digest and ordered chunk IDs.
- Builds use the existing cross-process lock for single-writer publication,
  avoid holding the library lock during model execution, and revalidate the
  corpus before commit. Per-document `pending`, `embedded`, `reused`, or
  `error` states are written atomically without raw paper text or paths.
- Provider or final-pointer failures preserve the previous committed
  generation. BM25 and the current product path are untouched; no setting,
  model download, cloud endpoint, or hybrid default has been activated.
- Added nine focused tests covering descriptor/shape validation, build/load/
  search, no-op builds, one-paper incremental rebuilds, path and document-ID
  relocation reuse, provider failure, commit failure, provider mismatch, and
  vector corruption, plus a concurrent corpus mutation during provider work.
- Added the three new production modules to the architecture size ratchet; the
  largest is 840 lines, well below the existing multi-thousand-line hotspots
  and prevented from growing silently.

#### Phase 2A validation

- Focused embedding, Dense/Hybrid, and storage-transaction suites after final
  fault injection: 23 passed, 1 Windows-platform skip.
- Complete backend unit suite: 4,596 passed, 41 configuration-dependent skips.
- Backend sanity suite: 305 passed, 2 configuration-dependent skips.
- Full-repository Ruff, `git diff --check`, and the expanded 24-file
  architecture ratchet: pass.
- Research QA fixture: 56 cases and 41 source-grounded cases; pass.
- Grounded replay 6/6 and reviewed Agent replay 5/5; pass.
- Candidate retrieval seed, frozen unseen-v2 qrels/baseline integrity, version
  A/B, evidence comparison 5, comparison candidates 5, project status 5, and
  the complete project journey fixture: pass.
- Converter quality: 13/13 retained Markdown cases; pass.
- Frontend ESLint and production build: pass.
- Browser smoke: 160 passed with 3 expected private/build-mode skips; core
  citation/library regressions: 127/127; ordinary-user surface isolation: 4/4;
  evidence-workspace release isolation: 1/1; dedicated library-quality and
  automatic-recovery suite: 24/24.

All browser groups ran serially to avoid cross-server resource contention. No
release version or immutable tag changed. Dense retrieval remains unreachable
from the product runtime, so these results validate the persistence boundary
without changing current user behavior.

### Priority assessment after Phase 2A

| Priority | Work | User value | Dependency/risk | Decision |
|---|---|---:|---|---|
| P0.2a | Add reproducible concrete local-model benchmark adapters and measure raw Dense on the frozen multilingual qrels | Very high | Pinned model and encoding contract; no product dependency or default download | Completed in Phase 2B.1 |
| P0.2b | Measure the Phase-0-preregistered BM25+Dense RRF configuration; tune any future alternative only on a separate development slice | Very high | Scoped quality passed, but full-library wrong-source@5 regressed versus the current product-query channel | Completed; do not promote this configuration |
| P0.2c | Freeze a separate real-paper development qrel set from the established 41 source-grounded library cases, then develop source-safe product-query BM25+dense fusion there | Very high | Required to improve full-library source precision without tuning unseen-v2 | Do next |
| P0.3 | Benchmark bounded cross-encoder top-M reranking only after fused candidate recall improves | High | Reranking cannot recover evidence absent from the candidate pool | After an accepted hybrid candidate set |
| P1.1 | Derive answer correctness, faithfulness, citation precision/completeness, and contradiction-repair metrics from existing Research QA contracts | Very high | Failures must trigger retry/narrowing/evidence boundary, not another user-facing status label | Before product activation |
| P1.2 | Ship explicit Off/Local/Cloud consent, status, rebuild, and automatic BM25 fallback UI | High | Requires an accepted provider plus measured package, latency, and privacy cost | Activation gate, not the next coding step |
| P1.3 | Add redacted OpenTelemetry-aligned local spans and optional Langfuse export | Medium-high | Exporter failures must be invisible to retrieval and raw content stays local by default | Before opt-in rollout |
| P2 | Continue contract-preserving decomposition under the expanded size ratchet | High long-term | Avoid mixing broad runtime refactors into model-quality experiments | Incremental parallel track |

The design decision after implementation is to keep exact Float32 search and
full-generation rewrites while the corpus remains small, but reuse embedding
computation per paper. This gives atomic recovery and simple integrity checks
without a vector service. ANN or a vector database should be reconsidered only
after benchmarked memory or p95 search latency, not feature fashion, shows the
exact matrix is the bottleneck.

### 2026-09-02 — Phase 2B.1 pinned raw Dense benchmark

- Preselected `intfloat/multilingual-e5-base` at immutable Hugging Face revision
  `d128750597153bb5987e10b1c3493a34e5a4502a` before viewing its Pi_zaya
  ranking results. Its official model card specifies 768 dimensions,
  attention-masked mean pooling, `query: ` / `passage: ` prefixes, 512-token
  truncation, multilingual training including Chinese retrieval data, and an
  MIT license:
  <https://huggingface.co/intfloat/multilingual-e5-base>.
- Chose this first candidate because it runs on the repository's already
  available Transformers version without `trust_remote_code`. The benchmark
  does not install Sentence Transformers, FlagEmbedding, a vector database, or
  any product dependency.
- Added a benchmark-only local adapter in
  `tools/retrieval_eval/transformers_embedding_provider.py`. Cache-only is the
  default; downloading requires explicit `--allow-download`. Revision,
  prefixes, pooling, max length, and execution dtype are part of the provider
  fingerprint, so incompatible vectors cannot be reused.
- Generalized `kb/retrieval_benchmark.py` to construct any candidate retriever
  over the exact scoped/full-library chunk subset while keeping the existing
  BM25 report unchanged.
- Added `tools/retrieval_eval/run_dense_research_qa_benchmark.py` and committed
  `docs/retrieval_dense_multilingual_e5_base_unseen_v2.json`. CI and the Windows
  release workflow validate the fixture hash, corpus fingerprint, pinned model
  and encoding identity, cache-only/no-remote-code/no-product-activation flags,
  index readiness, and all 22 scoped/full-library case IDs without loading the
  model or needing the private corpus.
- The first permitted download was an operator benchmark action only. The
  committed measurement then rebuilt a fresh index in cache-only mode on an
  RTX 3070 Laptop GPU with FP16. It encoded all 1,673 chunks in 11.47 seconds;
  the atomic generation completed in 12.35 seconds and occupied 5,564,052
  bytes. Warm model load was 5.82 seconds.

#### Raw Dense result

| Channel / scope | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Wrong source@5 |
|---|---:|---:|---:|---:|---:|---:|
| Raw BM25 / scope-qualified | 0.6136 | 0.8409 | 0.9091 | 0.5229 | 0.5861 | 0.0000 |
| Product-query BM25 / scope-qualified | 0.6136 | 0.8182 | 0.8636 | 0.4686 | 0.5330 | 0.0000 |
| E5 Dense / scope-qualified | 0.6818 | 0.8636 | 0.8864 | 0.5793 | 0.6375 | 0.0000 |
| Raw BM25 / full library | 0.5227 | 0.7045 | 0.8409 | 0.5092 | 0.5310 | 0.4273 |
| Product-query BM25 / full library | 0.5682 | 0.7273 | 0.8182 | 0.4285 | 0.4875 | 0.0818 |
| E5 Dense / full library | 0.6818 | 0.8182 | 0.8409 | 0.5747 | 0.6243 | 0.0727 |

Interpretation:

- Raw Dense improves early ranking over both BM25 channels. Against raw BM25,
  scope Recall@5 rises 6.82 points, MRR 5.64 points, and nDCG@10 5.13 points.
  Full-library Recall@10 rises 11.36 points while wrong-source@5 falls from
  42.73% to 7.27%.
- Chinese scope Recall@5 rises from raw BM25's 30% to 45%, but Recall@10 remains
  75%; this is a useful early-rank gain, not the planned 15-point Recall@10
  semantic promotion threshold. English Recall@10 rises from 91.67% to 95.83%.
- Dense recovers `sam2-streaming-memory` at rank 1 after both BM25 channels miss
  it in top 20. It also moves `deepseek-r1-zero-boundary` from raw/product ranks
  7/13 to rank 2 and `kan-pruning-boundary` from rank 6 to rank 2.
- Dense loses `sam2-data-engine-scale` from top 20, weakens the long-context
  synthesis label coverage, and does not improve cross-paper Recall@10 beyond
  50%. These failures prevent Dense-only activation and are exactly why BM25
  must remain a fused channel.
- 135 of 1,673 chunks exceed the candidate's 512-token limit; the maximum
  observed input is 4,054 tokens. This is a visible chunking/model-fit risk.
  Do not raise max length beyond the model contract. Analyze labelled truncated
  chunks and, if needed, evaluate evidence-preserving subchunk pooling as a
  separate indexed representation rather than silently truncating more text.
- Warm query plus exact search measured p50/p95 22.38/38.77 ms in scope and
  95.29/113.77 ms over all 1,673 chunks. These are local-machine candidate
  measurements, not general production latency claims.

Phase 2B.2 will now measure the already-implemented Phase-0 RRF defaults
(`sparse_weight=2`, `dense_weight=1`, `rrf_k=20`, candidate multiplier 4).
Because those values predate the real qrels, this is an honest locked
measurement. If they fail promotion gates, parameter search must use a new
development fixture; the unseen-v2 labels will not be used as a tuning loop.

### 2026-09-02 — Phase 2B.2 preregistered BM25+dense RRF

Measured the Phase-0 defaults without changing them after observing raw Dense:
`sparse_weight=2`, `dense_weight=1`, `rrf_k=20`, and candidate multiplier 4.
The committed candidate report now contains both raw Dense and RRF channels,
and its offline validator requires all 22 case IDs in both scoped and
full-library modes.

| Channel / scope | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Wrong source@5 |
|---|---:|---:|---:|---:|---:|---:|
| Raw BM25 / scope-qualified | 0.6136 | 0.8409 | 0.9091 | 0.5229 | 0.5861 | 0.0000 |
| E5 Dense / scope-qualified | 0.6818 | 0.8636 | 0.8864 | 0.5793 | 0.6375 | 0.0000 |
| Preregistered RRF / scope-qualified | 0.7955 | 0.8864 | 0.9545 | 0.6660 | 0.7018 | 0.0000 |
| Product-query BM25 / full library | 0.5682 | 0.7273 | 0.8182 | 0.4285 | 0.4875 | 0.0818 |
| E5 Dense / full library | 0.6818 | 0.8182 | 0.8409 | 0.5747 | 0.6243 | 0.0727 |
| Preregistered RRF / full library | 0.7045 | 0.8409 | 0.8864 | 0.6649 | 0.6855 | 0.2636 |

The fusion result establishes real complementarity: against scope-qualified
raw BM25, Recall@5 improves 18.18 points, Recall@20 4.55 points, MRR 14.31
points, and nDCG@10 11.57 points. It also improves every displayed
full-library ranking metric over both single channels.

It is nevertheless rejected for product promotion. Full-library wrong-source
rate at five is 26.36%, better than raw BM25's 42.73% but much worse than the
current product-query BM25 path's 8.18%. The product safety gate says wrong
source and locator accuracy may not regress, so aggregate recall cannot waive
this failure.

Do not try more weights, RRF constants, candidate depths, or post-filters on
unseen-v2. The next step is to project an independent development qrel set from
the already established 41 real source-grounded library cases. Source-safe
fusion should combine the current product-query sparse ranking with Dense,
preserve exact structured routes, and be tuned only on that development set.
After the algorithm and parameters are frozen, unseen-v2 gets one new locked
acceptance run. Cross-encoder work remains behind this gate because reranking a
source-contaminated candidate pool would hide rather than solve the retrieval
boundary defect.

### 2026-09-02 — Phase 2B.3 independent development qrels and source-safe fusion

- Added `docs/retrieval_eval_development_v1.json`, projected from the 41
  established source-grounded library cases rather than from unseen-v2. It
  freezes 46 evidence contracts and 46 chunk labels over the complete current
  31-paper, 2,251-chunk corpus; unlabelled papers stay in the candidate corpus
  as hard background negatives.
- Added `docs/retrieval_bm25_baseline_development_v1.json` and regression checks
  that the original unseen-v2 projection remains unchanged. The development
  fixture is labelled as historical/development-only in both schema and copy.
- Extended the benchmark boundary so scoped and full-library modes may build
  mode-aware candidate retrievers. Existing BM25 defaults and reports are
  unchanged.
- Added a source-safe RRF policy: the current deterministic product-query BM25
  route supplies the sparse order, its first five results are copied unchanged,
  and Dense may reorder only the tail using sources already discovered in the
  sparse top N. Dense failure still returns the sparse ranking.
- Evaluated a fixed 3x3 grid on the development fixture only: sparse weights
  `2/4/8`, Dense weight `1`, RRF constant `20`, protected sparse prefix `5`,
  and sparse source-gate depths `5/10/20`. The unseen-v2 qrels were not touched
  during selection.
- An initial adapter let the product query route derive a larger internal
  candidate window from the hybrid pool. That changed the nominally protected
  top five. The adapter now pins the product route to the fixture's rank-20
  evaluation depth; a regression test enforces this comparison boundary, and
  the complete grid was rerun.

#### Development selection result

| Channel / scope | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Wrong source@5 |
|---|---:|---:|---:|---:|---:|---:|
| Product-query BM25 / scope-qualified | 0.5976 | 0.6951 | 0.7317 | 0.3913 | 0.4606 | 0.0000 |
| Selected safe RRF / scope-qualified | 0.5976 | 0.6951 | 0.8171 | 0.3995 | 0.4618 | 0.0000 |
| Product-query BM25 / full library | 0.5000 | 0.6220 | 0.6829 | 0.2832 | 0.3553 | 0.2976 |
| Selected safe RRF / full library | 0.5000 | 0.6341 | 0.7317 | 0.2881 | 0.3594 | 0.2976 |

The frozen selection is `sparse_weight=2`, `dense_weight=1`, `rrf_k=20`,
candidate multiplier `4`, protected sparse prefix `5`, and Dense source gate
depth `5`. It preserves top-five recall and source error exactly, improves
scope Recall@20 by 8.54 points, full-library Recall@10 by 1.22 points, and
full-library Recall@20 by 4.88 points. Full-library wrong-source@10 also falls
from 33.41% to 28.78% and wrong-source@20 from 38.41% to 35.12%.

Weight `4` produces a small scope Recall@10 increase but gives up the selected
configuration's scope/full Recall@20 and tail source precision. Weight `8`
mostly reproduces product BM25. Gate depth `5` is preferred over `10/20`
because it is the narrowest trust boundary and has the strongest measured
full-library Recall@20/source precision tradeoff.

This development result selects a candidate; it does not authorize product
activation. The next step is exactly one cache-only unseen-v2 run with the
frozen configuration. No further parameter adjustment will be made from that
result. If it fails the pre-existing promotion gates, Phase 2B stops and the
next retrieval experiment must be a separately declared representation change
(for example evidence-preserving subchunk pooling), not another weight search.

### Priority assessment after Phase 2B.3

| Priority | Work | User value | Dependency/risk | Decision |
|---|---|---:|---|---|
| P0.2d | Run the frozen safe-RRF configuration exactly once on unseen-v2 and record pass/fail against BM25 and source-safety gates | Very high | Must remain cache-only and must not tune from the result | Do next |
| P0.3 | Benchmark a bounded cross-encoder only if the frozen hybrid candidate pool passes unseen acceptance | High | Cannot repair missing or wrong-source candidates | Conditional on P0.2d |
| P1.1 | Add aggregate correctness, faithfulness, citation precision/completeness, contradiction, and automatic repair outcomes from existing Research QA contracts | Very high | Must cause retry/narrow/remove behavior, never another unresolved user label | Next product-quality track |
| P1.2 | Add explicit Off/Local/Cloud choice and rebuild/fallback UX | High | Blocked until retrieval candidate and end-to-end evidence gates pass | Do not activate yet |
| P1.3 | Add redacted OpenTelemetry-aligned local spans and optional Langfuse export | Medium-high | Raw questions, paper text, paths, and credentials remain local by default | After quality metrics contract |
| P2 | Continue small-module decomposition under the architecture ratchet | High long-term | Avoid broad refactors inside retrieval experiments | Incremental |

### 2026-09-02 — Phase 2B.4 locked unseen-v2 acceptance

The development-selected configuration was run once against unseen-v2 from a
fresh cache-only FP16 index. A first command pointed at the 31-paper development
database; corpus validation rejected all chunks before model loading, indexing,
or ranking because unseen-v2 requires its isolated 10-paper/1,673-chunk corpus.
That preflight failure is not a candidate measurement and did not overwrite the
report. The corrected isolated corpus then produced the only ranking result.

| Channel / scope | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Wrong source@5 |
|---|---:|---:|---:|---:|---:|---:|
| Product-query BM25 / scope-qualified | 0.6136 | 0.8182 | 0.8636 | 0.4686 | 0.5330 | 0.0000 |
| Frozen safe RRF / scope-qualified | 0.6136 | 0.8636 | 0.9318 | 0.4801 | 0.5578 | 0.0000 |
| Product-query BM25 / full library | 0.5682 | 0.7273 | 0.8182 | 0.4285 | 0.4875 | 0.0818 |
| Frozen safe RRF / full library | 0.5682 | 0.8182 | 0.8409 | 0.4389 | 0.5245 | 0.0818 |

The locked candidate preserves both top-five recall and wrong-source@5 exactly.
It improves scoped Recall@10/20 by 4.55/6.82 points and full-library
Recall@10/20 by 9.09/2.27 points; MRR and nDCG@10 also improve in both modes.
Full-library Chinese Recall@10 rises from 0.60 to 0.80, while scoped Chinese
Recall@10 rises from 0.70 to 0.80. English top-10 recall is unchanged while
English scoped Recall@20 reaches 1.0.

This passes the candidate-pool gate for a bounded top-20 cross-encoder study:
there are more relevant candidates to rerank without contaminating the first
five sources. It is not yet a product-promotion pass. The original threshold
called for a separately auditable semantic/paraphrase slice with at least a
15-point Recall@10 gain; unseen-v2 did not pre-register such a group label.
Chinese full-library queries clear 15 points, but scoped Chinese queries do not,
so neither is relabelled post hoc as proof. Product activation also remains
behind answer/citation end-to-end metrics, explicit Off/Local/Cloud consent,
package-size and latency decisions, and fallback UX.

The committed report validator now requires the selected safe channel, exact
case coverage, pinned cache-only provider identity, ready corpus/index state,
and cross-report preservation of product BM25 Recall@5 and wrong-source@5. It
also rejects regressions in Recall@10/20, MRR, or nDCG@10. CI and release gates
now validate the separate development BM25 fixture/report as well.

#### Phase 2B.4 runtime result

- Fresh index: 10 papers, 1,673 chunks, 5,564,052 bytes.
- Cache-only document embedding: 12.56 seconds; atomic build: 13.51 seconds.
- Model load: 6.69 seconds; no download and `trust_remote_code=false`.
- 135 chunks exceeded the pinned 512-token input boundary; the longest observed
  input was 4,054 tokens. This remains a representation-quality risk, not a
  reason to silently raise the model limit.
- Safe-RRF measured p50/p95: 59.58/117.48 ms scoped and 222.15/267.59 ms full
  library on the benchmark machine. These are comparison measurements, not
  general latency guarantees.

### Priority assessment after Phase 2B.4

| Priority | Work | User value | Dependency/risk | Decision |
|---|---|---:|---|---|
| P0.3a | Select and pin one multilingual cross-encoder under an explicit license/runtime contract | High | Must run locally without `trust_remote_code` or product dependency changes | Do next |
| P0.3b | Rerank only positions 6-20 of the frozen source-safe pool; preserve product BM25 top five and measure development then one locked unseen run | Very high | Reranker must fail open and cannot introduce a new source | After P0.3a |
| P1.1 | Aggregate correctness, faithfulness, citation precision/completeness, contradiction, locator, and automatic repair outcomes from existing Research QA contracts | Very high | Required before product activation; failures repair/retry/narrow automatically | After candidate ranking is frozen |
| P1.2 | Add a new pre-registered semantic/paraphrase acceptance slice | High | Needed to audit the original 15-point gate without post-hoc relabelling | Before product activation |
| P1.3 | Add explicit Off/Local/Cloud choice, index status/rebuild, privacy disclosure, and BM25 fallback UX | High | Only after model/package/latency and end-to-end quality are accepted | Not yet |
| P1.4 | Add redacted OpenTelemetry-aligned local spans and optional Langfuse export | Medium-high | No raw question, paper text, path, or credential export by default | After quality contracts |
| P2 | Continue contract-preserving decomposition under the architecture ratchet | High long-term | Do not mix broad refactors into model experiments | Incremental |

### 2026-09-02 — Phase 2C bounded multilingual cross-encoder benchmark

- Preselected `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` at immutable
  revision `8008f0154ca2387468013ed439df09d70c122412` before quality measurement.
  Its official model card identifies an Apache-2.0, approximately 0.1B-parameter
  multilingual retrieval cross-encoder trained on mMARCO, with 15-language
  metadata and direct Transformers sequence-classification loading:
  <https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1>.
- Rejected `Alibaba-NLP/gte-multilingual-reranker-base` for this first study
  because its official usage requires `trust_remote_code=True`, contrary to the
  benchmark's local execution boundary.
- Added a cache-first benchmark adapter with pinned model/revision, scalar-logit
  validation, FP16/CUDA execution, batch and 512-token limits, truncation and
  timing counters, and `trust_remote_code=false`. It preserves the full query
  and truncates only the passage.
- The current Transformers 4.29.2 environment required `sentencepiece 0.1.99`
  and `safetensors 0.4.5` to load this checkpoint safely on Windows. These were
  installed only in the benchmark environment and were not added to product or
  release requirements. SentencePiece 0.2.1 was explicitly rejected after a
  native tokenizer access violation.
- Added tail protection to the model-agnostic cross-encoder adapter. The
  development benchmark sent only positions 6-20 of the accepted safe-RRF pool
  to the scorer. Positions 1-5 never entered the model and remained byte-order
  stable; exceptions still fail open to the upstream order.
- Added `docs/retrieval_cross_encoder_mmarco_development_v1.json` and an offline
  validator for fixture/corpus hashes, pinned cache-only model contract, all 41
  scoped/full case IDs, protected top-five metrics, and invariant Recall@20
  candidate coverage. CI and release workflows run this validator without
  loading either model.

#### Development result and rejection

| Channel / scope | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 | Wrong source@5 |
|---|---:|---:|---:|---:|---:|---:|
| Safe RRF / scope-qualified | 0.5976 | 0.6951 | 0.8171 | 0.3995 | 0.4618 | 0.0000 |
| mMARCO tail rerank / scope-qualified | 0.5976 | 0.7073 | 0.8171 | 0.4021 | 0.4661 | 0.0000 |
| Safe RRF / full library | 0.5000 | 0.6341 | 0.7317 | 0.2881 | 0.3594 | 0.2976 |
| mMARCO tail rerank / full library | 0.5000 | 0.6341 | 0.7317 | 0.2926 | 0.3617 | 0.2976 |

The reranker preserves the intended safety invariants, but scope Recall@10
improves only 1.22 points and full-library Recall@10 does not improve. Contract
Recall@10 rises 2.44 points in each mode, while full-library wrong-source@10
worsens from 28.78% to 29.51%. The cache snapshot occupies 492,747,628 bytes;
137 of 1,200 scored pairs truncate, with a maximum observed 2,430 tokens.
Compared with safe RRF, p50 end-to-end latency rises from 37.16 to 106.69 ms
scoped and from 211.28 to 276.43 ms full-library on the benchmark machine.

This candidate is rejected before an unseen-v2 run. A roughly 493 MB local
cache plus extra tokenizer/runtime requirements and 65-70 ms p50 cost are not
justified by the small development gain. It is not added to product
requirements, settings, packages, or runtime. Cross-encoder work now pauses;
future candidates need an explicitly better package/latency hypothesis rather
than another unbounded model search.

### Priority assessment after Phase 2C

| Priority | Work | User value | Dependency/risk | Decision |
|---|---|---:|---|---|
| P0.4 | Build one aggregate, deterministic answer/evidence outcome report from existing Research QA contracts | Very high | Must measure correctness, faithfulness, citation precision/completeness, contradiction, locator, and whether failures were automatically repaired | Do next |
| P0.5 | Turn remaining verifier failures into internal retrieve/retry/narrow/remove actions and expose only actionable evidence boundaries | Very high | Directly addresses unresolved “待核验/核验失败” states shown to users | After P0.4 identifies actual residual classes |
| P1.1 | Add a pre-registered semantic/paraphrase slice independent of current model selection | High | Required before Dense product promotion; no post-hoc relabelling | In parallel with end-to-end quality work |
| P1.2 | Decide whether evidence-preserving subchunk pooling reduces the measured 512-token truncation risk | High | Must compare retrieval gains, index growth, and locator fidelity on development data first | After outcome report |
| P1.3 | Explicit Off/Local/Cloud choice and index/fallback UX | High | Dense remains off until quality, package, privacy, and recovery gates pass | Blocked |
| P1.4 | Redacted OpenTelemetry-aligned local spans and optional Langfuse export | Medium-high | Instrument the stable outcome contract, not transient experiment internals | After P0.4 |
| P2 | Contract-preserving decomposition under the architecture ratchet | High long-term | Keep model experiments out of large runtime files | Incremental |

### 2026-09-02 — P0.4 progress checkpoint 1

- Revalidated the committed development BM25, locked unseen-v2 Dense, and
  rejected development cross-encoder reports through their cache-free offline
  validators before starting answer-quality aggregation. Coverage remains 41,
  22, and 41 cases respectively.
- The outcome report will distinguish observed final user-facing quality from
  unobserved internal repair attempts. Historical all-green replays may prove
  the delivered answer/card state, but cannot be used to invent a repair rate
  when the old traces did not persist repair-attempt telemetry.
- Current next action: implement a small standalone aggregator with unit-tested
  metric mappings, visible unresolved-state scanning, source hashes, and a
  validate-only CI contract. Existing large evaluation and API modules will not
  absorb this reporting logic.

### 2026-09-02 — P0.4 complete: answer and evidence outcome contract

- Added `tools/research_qa/build_quality_outcome_report.py` and the committed
  `docs/research_qa_quality_outcomes_v1.json`. Generation is anchored to SHA-256
  hashes of the authoritative 56-case main and 22-case unseen-v2 live reports;
  CI/release can validate the committed result without those large/private raw
  artifacts being present.
- Across 78 live cases, all 78 final outcomes pass. Applied checks pass at
  223/223 for answer correctness, 78/78 for faithfulness, 134/134 for citation
  completeness, 202/202 for citation precision/routing, 71/71 for locators,
  and 156/156 for user-visible answer/card integrity.
- The contradiction slice is deliberately labelled as a narrow boundary:
  4/4 forbidden-claim/evidence checks pass, but this is not semantic entailment
  or NLI verification.
- Historical claim-evidence metadata exists for all 78 cases. It records 327
  finalization actions across 57 cases, including 134 citation repairs, 26
  citation rebindings, 38 renderer citation rejections, and 88 unsupported
  unplanned-claim removals. One mismatch was detected in an intermediate audit,
  while the final contract passed; zero uncited high-risk claims remained.
- Final rendered answer/card scanning found zero user-visible occurrences of
  “待核验”, “核验失败”, or their English provisional equivalents. The report
  explicitly keeps `attempt_count=null`: total repair attempts and therefore a
  repair-rate denominator were not persisted by these historical live traces.
- Fresh deterministic gates also pass: agent answer quality 5/5, grounded replay
  6/6, reviewed replay 5/5, and note-evidence automatic resolution 5/5. The note
  contract covers exact source refresh, stable locator alignment, exact quote
  relocation, saved-context disambiguation, and rejection of unsafe fuzzy
  auto-resolution.
- Added three unit tests for aggregation, visible-state scope, and fail-closed
  validation. Focused tests and Ruff pass; the committed report's validate-only
  gate passes.

### Priority assessment after P0.4

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P0.5a | Audit and remove non-actionable provisional verifier copy from the live pending-card path while preserving automatic refresh/retry | Very high | Historical final state is clean, but transient API/UI builders still contain “待核验” copy | Do next |
| P0.5b | Persist repair-attempt/outcome telemetry in the stable answer contract | High | Needed for a real repair success rate; must be local/redacted and not expose internal failure labels | After transient UX fix |
| P1.1 | Add a pre-registered semantic/paraphrase retrieval slice and a contradiction/NLI evaluation slice | High | Current contradiction coverage is only four deterministic boundary checks | After P0.5 |
| P1.2 | Evaluate evidence-preserving subchunk pooling | High | 512-token truncation remains measured; must preserve exact locators and source safety | After evaluation slices freeze |
| P1.3 | Add explicit Dense Off/Local/Cloud consent, status, rebuild, and BM25 fallback UX | High | Safe RRF is promising but still not product-approved | Blocked on P1.1/P1.2 |
| P1.4 | Add redacted OpenTelemetry-aligned local spans and optional Langfuse export | Medium-high | Instrument stable outcome/repair fields only | After P0.5b |
| P2 | Continue small-module decomposition under the architecture ratchet | High long-term | Avoid mixing broad refactors into quality-path fixes | Incremental |

### 2026-09-02 — P0.5a complete: pending evidence is internal work

- Changed the conversation-reference pending path from provisional evidence
  cards to a progress-only contract. Until answer/citation binding is durable,
  the public payload contains no candidate hit text, source identity, primary
  evidence, reader target, or provisional “待核验” copy. Existing polling,
  completed-answer overlay, stale-pending recovery, and background full render
  still replace the placeholder automatically.
- Added frontend defense in depth: a stale cache or mixed-version backend that
  still includes hits inside a pending pack cannot render those hits or expose
  Reader/Cite actions. The panel shows one explicit system-owned progress line
  and updates when a grounded payload arrives.
- Removed the non-actionable bibliography “自动校验中” chip from the literature
  basket. Actual missing/conflicting metadata issues and automatic repair paths
  remain; only the status that gave the user no decision or remedy was removed.
- Deleted obsolete pending-card shaping/filtering helpers from the 7k-line
  reference router, reducing the large module by roughly 200 lines instead of
  adding another compatibility layer.
- Verification: 124 reference-router unit tests plus 3 outcome-report tests
  pass; Ruff and diff checks pass; frontend lint/build pass functionally; and
  60 Playwright cases across the reference panel and message/citation flow pass.
  The only lint feedback was a React hook dependency warning introduced by the
  defensive hit filter; it has been corrected and will be rechecked with the
  next frontend gate.

### Priority assessment after P0.5a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P0.5b | Persist a stable local answer-finalization outcome so repair rates have a real denominator | High | Historical traces expose actions but not attempt/gate status; do not leak internal diagnostics into UI | Do next |
| P0.5c | Re-run the full research-note recovery/sanity suite and scan all ordinary-user copy for non-actionable verifier states | Very high | Pending refs are fixed; note terminal boundaries must remain honest and system-owned | After telemetry contract |
| P1.1 | Freeze semantic/paraphrase retrieval and contradiction/NLI evaluation slices | High | Current contradiction coverage is narrow and Dense promotion remains blocked | After P0.5 |
| P1.2 | Evaluate evidence-preserving subchunk pooling | High | Must reduce truncation without breaking exact locators | After P1.1 |
| P1.3 | Add Dense Off/Local/Cloud consent and rebuild/fallback UX | High | No activation before new evaluation slices pass | Blocked |
| P1.4 | Add redacted OpenTelemetry-aligned spans and optional Langfuse export | Medium-high | Reuse P0.5b's stable counters | After P0.5b |
| P2 | Continue ratcheted decomposition | High long-term | This step already removed dead router logic | Incremental |

### 2026-09-02 — P0.5b complete: measurable system-owned repair outcomes

- Added the small `kb/answer_finalization_outcome.py` contract instead of
  expanding the 10k-line finalizer. Every standard or fast-exact answer that
  reaches the evidence final gate now records a versioned, compact outcome in
  `message.meta.answer_quality.finalization_outcome` through the existing local
  answer-quality persistence path.
- The outcome records gate pass count, final pass/fail, whether automatic work
  was required and succeeded, grouped repair/narrow action counts, and issue
  detection counts. It explicitly excludes prompts, answer/evidence text,
  source paths, claim rows, and verifier detail strings, so it is suitable for
  later redacted tracing but is not rendered to ordinary users.
- Future fully covered runs can measure repair success as successful repaired
  cases divided by repair-required cases. The outcome aggregator understands
  complete and partial v1 coverage; it retains a null attempt count for the 78
  historical cases rather than inferring a denominator from action totals.
- Updated `docs/EVAL_DASHBOARD.md` with the v1 contract and reproducible report
  commands. Added four report tests and three outcome-contract tests, plus
  assertions on both finalizer paths.
- Verification: 233 focused outcome/finalizer tests pass (one intentional skip),
  Ruff and diff checks pass, and frontend lint/build pass without warnings.

### Priority assessment after P0.5b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P0.5c | Audit note-evidence terminal/recovery behavior and all ordinary-user verifier copy; run its full API/browser recovery suite | Very high | Must distinguish a true source-loss boundary from an internal retry without assigning system work to the user | Do next |
| P1.1 | Freeze a semantic/paraphrase retrieval slice plus contradiction/NLI slice | High | Required to judge Dense and semantic verification honestly | After P0.5c |
| P1.2 | Evaluate evidence-preserving subchunk pooling | High | 512-token truncation remains the largest measured Dense representation risk | After P1.1 |
| P1.3 | Add Dense Off/Local/Cloud consent and rebuild/fallback UX | High | Safe candidate remains disabled until promotion gates pass | Blocked |
| P1.4 | Add redacted OpenTelemetry-aligned spans and optional Langfuse export | Medium-high | Stable outcome counters now exist; privacy/export policy remains | After P1.1 design freeze |
| P2 | Ratcheted decomposition | High long-term | New telemetry lives in a 128-line module; router pending logic shrank | Incremental |

### 2026-09-03 — P0.5c complete: verifier work has an explicit owner

- Audited the complete research-note evidence lifecycle rather than changing
  labels alone. Deterministic source revisions, stable-locator moves, exact
  quote relocation, and saved-context disambiguation remain automatic;
  transient missing/unreadable/unstable outcomes retain bounded backoff;
  repeated failures become an honest source boundary; and a source-recovery
  event automatically revalidates terminal records without rewriting note text.
- Added the persisted `resolution_mode` contract with exactly three public-safe
  classes: `system_auto`, `user_decision`, and `source_boundary`. Unknown future
  reasons fail safe to `system_auto`; only the explicit candidate/source-change/
  location-change allowlist can create a user decision. Existing records remain
  compatible through the same fail-safe derivation.
- The note directory, issue filter, baseline panel, and evidence cards now rely
  on that ownership contract. Internal retries never enter the user's decision
  list. Terminal source loss/unreadability preserves the saved excerpt and
  audit history, explains that recovery is system-owned, and exposes no fake
  repair button.
- The ordinary research-agent source panel had one residual leak: it showed
  verifier-only “needs review” status and unsupported claims that the finalizer
  had already removed. Ordinary mode now keeps only the useful evidence map,
  source list, Reader action, and literature-basket action. Evaluation status,
  rejected claims, quality gates, routing, and tool diagnostics remain available
  only behind the explicit internal-debug build switch.
- Verification: 51 focused research-note unit/API tests pass; 9/9 note workspace
  and recovery Playwright scenarios pass; 5/5 affected research-agent source
  panel scenarios pass; Ruff, frontend lint, production TypeScript/Vite build,
  and `git diff --check` pass. The browser scenarios cover automatic recovery,
  real candidate decisions, terminal source preservation, missing fake actions,
  debug-only diagnostics, and source open/basket interactions.

### Priority assessment after P0.5c

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.1a | Freeze a pre-registered semantic/paraphrase retrieval slice before further Dense tuning | Very high | The original 15-point semantic gate cannot be audited from post-hoc language groups | Do next |
| P1.1b | Add a contradiction/entailment evaluation slice with lexical-overlap adversaries | Very high | Current 4/4 check is a narrow forbidden-claim boundary, not semantic verification | Immediately after slice schema |
| P1.2 | Evaluate locator-preserving subchunk pooling on the frozen slices | High | 512-token truncation is the largest measured Dense representation risk | After P1.1 fixtures freeze |
| P1.3 | Decide Dense promotion and only then build explicit Off/Local/Cloud consent, status, rebuild, and BM25 fallback UX | High | Safe RRF remains disabled; no user-facing switch for an unaccepted retriever | Blocked on P1.1/P1.2 |
| P1.4 | Add redacted OpenTelemetry-aligned local spans and optional Langfuse export | Medium-high | Instrument stable outcome counters; keep raw query, paper text, paths, and credentials local by default | After quality slice schema is stable |
| P2 | Ratcheted decomposition | High long-term | Keep evaluation logic in small modules and continue shrinking touched large files | Incremental |

### 2026-09-03 — P1.1a complete: semantic retrieval slice frozen before measurement

- Added `docs/retrieval_semantic_paraphrase_cases_v1.json`: 12 queries over
  the already-frozen 10-paper/1,673-chunk unseen-v2 corpus, balanced at six
  Chinese and six English queries. Eight are cross-language rewrites and all
  twelve are conceptual paraphrases. Two multi-paper comparisons remain in the
  slice so a model cannot pass only on single-paper scoping.
- Query authorship is independent of model output. Each case names its existing
  base contract, transformation type, rationale, and at least two evidence
  surface phrases deliberately withheld from the query. The registry pins the
  base qrels SHA-256 and a maximum 0.45 token-Jaccard overlap with the original
  product query.
- Added `kb/retrieval_semantic_eval_fixture.py`, which projects those queries
  in memory while inheriting each base case's relevance labels, claim-evidence
  contracts, source scope, chunk text hashes, and corpus fingerprint exactly.
  It rejects case reselection, base-fixture drift, locale drift, withheld-term
  leakage, excessive query overlap, and loss of the cross-language composition.
- The acceptance contract is now fixed at a 15-point Dense-over-product-BM25
  Recall@10 gain, with Recall@5 and wrong-source@5 reported as safety metrics.
  No BM25 or Dense result was run or inspected before this registry and its
  validation tests were frozen.
- Added an offline validator to CI and Windows release gates. Three focused
  tests, the validator, and Ruff pass. The validator reports 12 cases, 6/6
  locale balance, 8 cross-language cases, and the pinned 1,673-chunk corpus.

### Priority assessment after P1.1a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.1b | Freeze a balanced entailment/contradiction/insufficient-evidence slice using real local-paper excerpts and high-overlap adversaries | Very high | Prevents lexical overlap from being reported as semantic citation verification | Do next, before choosing an NLI model |
| P1.1c | Run product BM25 and the pinned Dense/safe-RRF candidate once on the now-frozen semantic slice | Very high | The 15-point gate is binding; a miss means no promotion | After P1.1b fixture freeze |
| P1.2 | Evaluate locator-preserving subchunk pooling on failures shared by the semantic and truncation slices | High | Avoid changing chunking without evidence of which long chunks fail | After P1.1c |
| P1.3 | Dense Off/Local/Cloud product choice and recovery UX | High | Only build after the retriever earns promotion | Blocked |
| P1.4 | Redacted OpenTelemetry-aligned spans and optional Langfuse export | Medium-high | Stable answer outcome and slice IDs are now available | After P1.1 quality contracts |
| P2 | Ratcheted decomposition | High long-term | New projection logic is isolated in a small module | Incremental |

### 2026-09-03 — P1.1b complete: semantic claim-verification gold set

- Added `docs/claim_entailment_eval_v1.json` before selecting or running any
  NLI model. It contains 24 claim/evidence cases over eight frozen real-paper
  evidence groups: eight entailments, eight direct contradictions, and eight
  insufficient-evidence claims, balanced at twelve English and twelve Chinese
  claims against traceable English paper excerpts.
- Four contradiction cases deliberately preserve high lexical overlap while
  reversing negation, component roles, numeric assignments, or storage
  semantics. The other four contradictions are cross-language relation/scope
  reversals. Every evidence item has one case of each label, so aggregate
  accuracy cannot hide a missing class.
- Each excerpt is bound to the existing unseen-v2 source case, labelled chunk
  ID, and pinned chunk-text hash. The validator rejects qrels drift, label or
  locale imbalance, missing evidence classes, weak lexical adversaries, and
  stale overlap measurements. No model name, prediction, or score is part of
  the gold set.
- Added a deterministic diagnostic runner for the current production citation
  verifier and committed `docs/claim_entailment_lexical_baseline_v1.json`.
  With valid inline citation syntax, the two-term-overlap verifier reaches only
  54.17% binary support accuracy: entailment recall 50%, non-entailment
  specificity 56.25%, contradiction false-support 50%, and insufficient-
  evidence false-support 37.5%. It falsely accepts 7 unsupported claims and
  rejects 4 supported cross-language claims.
- The diagnostic report explicitly records `semantic_entailment=false`; it is
  not a failed product gate because the product never claimed to run NLI, but
  it is now the binding baseline. Any future semantic candidate must reduce
  false support without silently replacing these labels or merely renaming the
  lexical score.
- Both fixture and diagnostic validators are in CI and Windows release gates.
  Four focused tests, both validators, Ruff, and diff checks pass.

### Priority assessment after P1.1b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.1c | Run product BM25, pinned Dense, and frozen safe RRF once on the pre-registered semantic slice | Very high | Determines whether Dense earns the original 15-point Recall@10 gate | Do next |
| P1.1d | Preselect one multilingual NLI candidate under license/package/privacy constraints, then measure once on the frozen 24 cases | Very high | Must prioritize contradiction false-support and cross-language entailment recall, not headline accuracy | After retrieval measurement |
| P1.2 | Evaluate locator-preserving subchunk pooling only on measured retrieval failures and truncated relevant chunks | High | Avoid broad rechunking without failure evidence | After P1.1c |
| P1.3 | Dense Off/Local/Cloud product choice and recovery UX | High | Blocked until semantic promotion and truncation decisions | Blocked |
| P1.4 | Redacted OpenTelemetry-aligned spans and optional Langfuse export | Medium-high | Quality IDs and stable outcome counters are ready; raw content remains local | After P1.1d |
| P2 | Ratcheted decomposition | High long-term | New semantic evaluator and validator remain small standalone modules | Incremental |

### 2026-09-03 — P1.1c complete: frozen semantic retrieval measurement

- Ran product BM25 and the pinned `intfloat/multilingual-e5-base` candidate
  exactly once on the pre-registered 12-case semantic/paraphrase slice. The
  Dense run was cache-only with `trust_remote_code=false`, reused all 1,673
  vectors from the accepted immutable index, and remained product-disabled.
- Product BM25 Recall@10 is 37.5% in scoped mode and 29.17% over the full
  library. Raw Dense reaches 50% in both modes: gains of 12.5 and 20.83
  percentage points respectively. It demonstrates real semantic value, but
  misses the binding 15-point threshold in scoped mode.
- The frozen source-safe RRF candidate preserves product Recall@5 and
  wrong-source@5 exactly, but reaches only 37.5% Recall@10 in both modes. Its
  gains are 0 and 8.33 points, so it fails the threshold in both required
  modes. It is rejected for product promotion; no user-facing Dense switch or
  hidden activation was added.
- Raw Dense rescues four semantic cases at top 10 (BitNet, SAM2, MedSAM, and
  ModernBERT), regresses on two (Gemma 3 and long-context synthesis), and shares
  three persistent misses with BM25 (Mamba, KAN, and TimesFM). These named
  failures, rather than aggregate score alone, now constrain the next design.
- Added `docs/retrieval_semantic_paraphrase_benchmark_v1.json`, which pins the
  fixture/corpus/provider/index identities, compact aggregate measurements,
  acceptance arithmetic, failure classes, and rejection decision. Ranked
  chunk lists remain local test artifacts; their normalized SHA-256 values are
  recorded in the report.
- Added a report validator to CI and Windows release gates. It rejects stale
  metric arithmetic, case/corpus drift, a changed provider encoding contract,
  downloads or remote code, a rebuilt index, accidental product activation,
  and any attempt to relabel the failed decision. The validator, two focused
  tests, Ruff, and diff checks pass.

### Priority assessment after P1.1c

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.2a | Diagnose token position, truncation, and retrievable evidence span for the five Dense misses/regressions | Very high | Rechunking is justified only where the relevant evidence is outside or diluted within the 512-token representation | Do next; analysis before implementation |
| P1.2b | Evaluate locator-preserving subchunk pooling on a separate development slice, then replay the frozen slice once | High | Must improve semantic recall without changing citation chunk IDs, page locators, or source safety | Only if P1.2a finds representation failures |
| P1.1d | Preselect and measure one multilingual NLI candidate on the frozen 24-case claim set | Very high | Current lexical verifier falsely supports 50% of contradictions; model/license/package/privacy gates remain binding | After P1.2a diagnosis; independent of Dense activation |
| P1.3 | Dense Off/Local/Cloud consent, index status, rebuild, and BM25 fallback UX | High | User choice is required, but exposing a rejected retriever would be misleading | Blocked until a candidate passes retrieval gates |
| P1.4 | Redacted OpenTelemetry-aligned spans and optional Langfuse export | Medium-high | Add after retrieval and verification outcome schemas stabilize; never export raw queries or paper text by default | After P1.1d/P1.2 decisions |
| P2 | Ratcheted decomposition of touched large modules | High long-term | Keep new diagnostics and evaluation logic small; reduce legacy files only at verified seams | Incremental |

### 2026-09-03 — P1.2a complete: truncation attribution rejected

- Added a tokenizer-only representation diagnostic using the exact pinned
  multilingual E5 revision, `passage: ` prefix, fast-tokenizer offsets, and
  512-token window. It creates no embeddings and does not rerun retrieval on
  the frozen acceptance slice.
- The full 1,673-chunk corpus contains 135 chunks above the encoder window and
  reaches 4,054 tokens, confirming that truncation exists as a general corpus
  condition. However, none of the 15 labelled semantic-retrieval evidence rows
  is truncated; the longest relevant row is 405 tokens.
- The five focus failures/regressions map to six labelled evidence rows. All
  six preserve every registered evidence term inside the encoder window and
  the longest is again 405 tokens. The observed semantic failures therefore
  cannot be attributed to evidence falling past token 512.
- Rejected broad storage rechunking: it would change stable citation units,
  increase index size, and risk page/anchor fidelity without addressing the
  measured cause. If a future development experiment tests passage focus, it
  must use multiple bounded child embeddings per unchanged parent, max-score
  the parent, and collapse to the original chunk ID/page locator before fusion
  or citation.
- Added `docs/retrieval_semantic_representation_diagnostic_v1.json`, a
  reproducible cache-only analyzer, a static drift validator, and three focused
  tests. The validator is included in CI and Windows release gates and rejects
  changed fixture/benchmark/model/window identities, unresolved evidence
  terms, false truncation attribution, product activation, or loss of the
  parent-locator contract. Analyzer, validator, tests, Ruff, and diff checks
  pass.

### Priority assessment after P1.2a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.1d | Preselect and measure one multilingual NLI candidate on the frozen 24-case entailment set | Very high | The current verifier falsely supports 50% of contradictions; this is a demonstrated evidence-trust failure | Do next |
| P1.2b | Pre-register parent-collapsed multi-vector passage-focus evaluation on a development slice | Medium-high | May recover semantic alignment, but it is not a truncation fix and must not tune on the frozen slice | After NLI, no product activation |
| P1.3 | Dense Off/Local/Cloud consent, index status, rebuild, and BM25 fallback UX | High | User choice remains required, but the current Dense/RRF candidate is rejected | Blocked until retrieval promotion |
| P1.4 | Redacted OpenTelemetry-aligned local spans and optional Langfuse export | Medium-high | Retrieval/verification outcome IDs are stabilizing; privacy redaction remains mandatory | After P1.1d candidate decision |
| P2 | Ratcheted decomposition of touched large modules | High long-term | Keep semantic verification and telemetry out of existing oversized runtime files | Incremental |

### 2026-09-03 — P1.1d-1 complete: multilingual NLI candidate frozen before measurement

- Preselected one candidate before inspecting the 24 frozen claim results:
  `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` at immutable revision
  `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`. It is MIT-licensed, uses
  safe weights, supports Chinese/English and cross-language NLI, and remains a
  benchmark-only local model with remote code and product activation disabled.
- Compared it with the smaller multilingual MiniLMv2-L12 alternative using
  primary model-card data. The selected model reports 80.8% average XNLI,
  88.3% English, and 81.16% Chinese accuracy versus 75.0%, 82.1%, and 75.1%
  for MiniLM. Its safetensors file is 557,652,046 bytes, only about 87 MB above
  the alternative, so the stronger safety-oriented baseline wins the tradeoff.
- Froze float32 execution, a 512-token pair window, evidence as premise, claim
  as hypothesis, `only_first` truncation so the claim is never dropped, and
  the model's entailment/neutral/contradiction label order.
- Froze all-pass thresholds before measurement: at least 75% three-way
  accuracy and entailment recall, at least 81.25% non-entailment specificity,
  contradiction false-support at most 12.5%, zero false support on the four
  high-overlap contradiction adversaries, insufficient-evidence false-support
  at most 25%, cross-language accuracy at least 70%, and zero truncated pairs.
  Passing permits only a shadow-integration design, never automatic promotion.
- Added `docs/claim_entailment_nli_candidate_v1.json`, an offline drift
  validator, and two focused tests. They reject fixture/model/revision/label,
  privacy, supply-chain, threshold, or single-run policy drift. Validator,
  tests, Ruff, and diff checks pass.

### Priority assessment after P1.1d-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.1d-2 | Implement the isolated local NLI benchmark adapter and run the one frozen measurement | Very high | Determines whether semantic verification is safe enough even for shadow use | Do next |
| P1.1d-3 | If and only if all gates pass, design lexical+NLI shadow arbitration and failure-safe fallback | High | A 24-case pass is not permission to replace production verification | Conditional |
| P1.2b | Parent-collapsed multi-vector retrieval development experiment | Medium-high | Retrieval value remains plausible but current failures are not truncation-caused | After NLI decision |
| P1.4 | Redacted OpenTelemetry-aligned spans and optional Langfuse export | Medium-high | Instrument stable verifier outcomes without raw claim/evidence export | After P1.1d-2 |
| P2 | Ratcheted decomposition | High long-term | Keep the NLI adapter/evaluator independent of legacy verifier/runtime files | Incremental |

### 2026-09-03 — P1.1d-2 complete: frozen multilingual NLI baseline measured once

- Added an isolated benchmark-only Transformers NLI adapter and evaluator. It
  loads only the pinned safetensors artifact, requires the exact three-label
  order, keeps the claim as the untruncated hypothesis, runs locally in
  float32, and never imports into the product verifier or runtime.
- Hardened candidate preparation after the legacy Hugging Face downloader and
  Windows curl path both discarded interrupted bytes. The benchmark preparer
  now persists 1 MB blocks, resumes with HTTP Range, and verifies the frozen
  557,652,046-byte size plus SHA-256 before exposing the weight file. This is
  explicit benchmark preparation; normal execution remains cache-only.
- Ran the preselected candidate exactly once over all 24 frozen cases. It
  achieved 79.17% three-way accuracy, 91.67% binary support accuracy, 87.5%
  entailment recall, 93.75% non-entailment specificity, and 91.67%
  cross-language accuracy. No pair was truncated.
- Contradiction false-support is 12.5% (one of eight), exactly at the frozen
  ceiling. All four high-overlap lexical adversaries were rejected and all
  eight insufficient-evidence claims were withheld from support. The remaining
  false support is the Chinese TimesFM cross-language role reversal; Gemma 3
  also exposes one false rejection plus neutral/contradiction class confusion.
- All pre-registered gates pass, so the candidate is
  `eligible_for_shadow_design` only. It is not product-enabled and cannot
  replace citation verification: the single false-supported contradiction is
  a concrete reason to retain conservative arbitration and product fallback.
- Added `docs/claim_entailment_nli_baseline_v1.json`, resumable preparation,
  report arithmetic/decision validation, and six focused candidate/evaluator/
  report tests. CI and Windows release gates now validate both preselection and
  measured report. Tests, validators, Ruff, and diff checks pass.

### Priority assessment after P1.1d-2

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.1d-3 | Define and evaluate a shadow-only structural-citation + semantic-NLI arbitration seam | Very high | NLI is much better than lexical overlap but still false-supports one contradiction; naïve AND loses supported evidence | Do next; no user-visible status or activation |
| P1.4a | Add redacted local spans for retrieval/verifier outcome IDs, duration, and fallback reason | High | Shadow comparison needs observability, but claims/evidence/paths must never be span attributes | After arbitration schema |
| P1.2b | Parent-collapsed multi-vector retrieval development experiment | Medium-high | Potential semantic gain remains, but current evidence rejects truncation as the cause | After verifier shadow contract |
| P1.3 | Dense Off/Local/Cloud choice and recovery UX | High but blocked | Current Dense/RRF candidate failed promotion; do not expose a misleading control | Blocked |
| P2 | Ratcheted decomposition | High long-term | Extract structural citation checks at a stable seam instead of adding logic to the large finalizer | Alongside P1.1d-3 |

### 2026-09-03 — P1.1d-3 complete: safe shadow arbitration seam selected

- Extracted numeric citation parsing and structural binding into the small
  `kb/agent/citation_structure.py` seam. It distinguishes missing, unbound,
  partial, unresolved, and structurally bound citations without pretending
  that citation syntax proves semantic support. The production verifier now
  consumes this seam while preserving its existing overlap decisions and
  response fields.
- Evaluated three deterministic combinations from the already-frozen lexical
  and NLI reports; no model was rerun. A naïve lexical-and-NLI intersection
  removes all false support but reduces entailment recall from 50% to 37.5%.
  Adding only an NLI contradiction veto to lexical overlap retains 50% recall
  and still false-supports two claims, so neither is a sound integration.
- Selected `structural citation binding + NLI argmax` for shadow observation:
  structurally invalid citations reject before semantics; entailment becomes
  shadow support; neutral is withheld; contradiction is a semantic conflict;
  unavailable/error is fail-closed in shadow and leaves production unchanged.
  On the frozen slice this retains the measured 91.67% binary accuracy, 87.5%
  entailment recall, and 93.75% non-entailment specificity, including the known
  TimesFM false-support failure rather than hiding it.
- The design is deliberately non-product: no answer is removed or retained by
  the NLI result, no ordinary-user “pending/failed verification” state is
  added, and the current deterministic verifier remains authoritative. Future
  telemetry may persist only outcome/fixture IDs, aggregate counts, duration,
  model revision, and fallback reason—never claims, evidence, source paths, or
  credentials.
- Added `docs/claim_verification_shadow_design_v1.json`, deterministic report
  generation/validation, arbitration helpers, and six focused structure/shadow
  tests. Twenty-seven verifier, presentation, benchmark, and shadow tests plus
  a broader 33-test Agent/trace regression set pass; CI/release validation,
  Ruff, and diff checks pass.

### Priority assessment after P1.1d-3

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.4a | Implement redacted local observability primitives for stable retrieval/finalization/verifier outcome IDs and durations | High | Needed to diagnose automatic recovery without handing users unresolved internal statuses; raw content must be impossible to record | Do next |
| P1.4b | Add optional OpenTelemetry-compatible exporter behind explicit configuration | Medium-high | Local spans must be proven redacted first; external export is opt-in only | After P1.4a |
| P1.2b | Pre-register a parent-collapsed multi-vector retrieval development experiment | Medium-high | Possible passage-focus gain, but not a truncation fix and no frozen-slice tuning | After observability primitive |
| P1.3 | Dense Off/Local/Cloud choice and recovery UX | High but blocked | Do not expose a retriever that failed its promotion gate | Blocked |
| P2 | Continue ratcheted decomposition at touched seams | High long-term | Citation structure is now separate; keep telemetry and providers isolated too | Incremental |

### 2026-09-03 — P1.4a-1 complete: redacted local span primitive and aggregate outcomes

- Added a default-off, thread-safe local JSONL span recorder with an
  OpenTelemetry-aligned completed-span shape. It writes only when
  `KB_LOCAL_OBSERVABILITY_PATH` is explicitly configured and observability
  failures can never fail the user task.
- Enforced event-specific attribute allowlists plus stable-ID, non-negative
  count, boolean, and duration validation. Queries, claims, evidence text,
  source paths, credentials, exception messages, and arbitrary attributes are
  rejected before persistence rather than merely omitted by convention.
- Connected answer finalization to one compact outcome span containing stable
  outcome/recheck IDs, repair and fallback counts, booleans, and measured
  duration. This complements the product's automatic repair and finalization
  path without surfacing internal pending/failed labels to ordinary users.
- Connected Agent completion to a redacted aggregate span containing final
  status, question-type and verification-outcome IDs, result/tool/error and
  supported/unsupported counts, and measured tool duration. No raw tool
  arguments, retrieved evidence, or trace source paths are copied.
- Added focused privacy, schema, default-off, write-failure, duration, and
  integration tests. The focused observability/finalization/Agent suite passes
  with 252 tests and one skip; Ruff and diff checks pass.

### Priority assessment after P1.4a-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.4a-2 | Record exact provider-reported model Token counts and model-call duration through the same redacted allowlist | High | Do not estimate absent usage or log model prompts/responses; streaming providers may omit usage | Do next |
| P1.4a-3 | Add top-level retrieval duration, hit/fallback counts, and stable strategy IDs | High | Instrument the user-visible retrieval operation, not every internal query variant; never record the query | After model-call spans |
| P1.4b | Add an optional OpenTelemetry exporter behind explicit configuration | Medium | External export stays blocked until local spans and privacy tests cover every event | After P1.4a |
| P1.2b | Pre-register parent-collapsed multi-vector passage-focus development evaluation | Medium-high | Retrieval quality remains the larger product gap, but instrumentation should exist before the next experiment | After P1.4a |
| P1.3 | Dense Off/Local/Cloud choice and recovery UX | High but blocked | Current Dense/RRF candidate failed promotion; no misleading switch | Blocked |
| P2 | Continue ratcheted decomposition | High long-term | Keep provider usage extraction and retrieval telemetry outside oversized runtime modules where possible | Incremental |

### 2026-09-03 — P1.4a-2 complete: exact model usage without content capture

- Added a small provider-usage adapter that recognizes only explicit
  OpenAI-compatible prompt/input, completion/output, and total Token counters.
  It never tokenizes prompts locally or estimates missing usage; total is
  derived only when both exact input and output counters are present.
- Added a redacted `pi_zaya.model_call` span with a coarse route ID
  (`text_generation`, `vision_generation`, or `web_search_generation`), call
  outcome, stable error category, attempt number, streaming flag, duration,
  and `usage_reported`. Model/base URLs, configured model names, prompts,
  responses, exception messages, and credentials are not accepted attributes.
- Connected non-streaming, web-search, retry, and streaming provider calls.
  Streaming requests remain protocol-compatible: the application does not add
  provider-specific usage options. It records counts only when a provider
  actually sends them and otherwise records `usage_reported=false` with no
  invented counters.
- Preserved failover semantics and made terminal stream delivery wait until the
  compact observation is finalized, avoiding a race in local diagnostics.
  Observability remains default-off and failure-isolated.
- Twenty-two focused privacy, usage extraction, chat, stream, timeout, failover,
  and DeepSeek-routing tests pass; Ruff and diff checks pass.

### Priority assessment after P1.4a-2

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.4a-3 | Instrument the top-level evidence retrieval operation with redacted duration, strategy, scope, fallback, cache, and hit counters | High | Needed to distinguish “model slow” from “retrieval weak” without recording the question | Do next |
| P1.4a-4 | Project shadow-verifier aggregate outcomes into the local schema without enabling NLI in product | Medium-high | Do not rerun the 557 MB model per request or expose internal statuses | After retrieval spans; design-only or offline observation |
| P1.2b | Pre-register parent-collapsed multi-vector passage-focus development evaluation | High | Semantic retrieval remains the larger user-facing quality gap | After local span coverage |
| P1.4b | Optional OpenTelemetry exporter | Medium | Exact local contract is now viable, but external export remains explicit opt-in | After P1.4a privacy replay |
| P1.3 | Dense Off/Local/Cloud product choice | High but blocked | No switch until a retriever passes promotion gates | Blocked |
| P2 | Ratcheted decomposition | High long-term | New model-call logic is isolated from the already-large runtime | Incremental |

### 2026-09-03 — P1.4a-3 complete: top-level retrieval observations

- Added a dedicated retrieval-observation adapter with closed strategy and
  scope vocabularies. It persists only the final hit count, query-variant
  count, translation flag, stable fallback reason, outcome, and measured
  duration; it has no parameter through which a query, translated query,
  evidence snippet, source title, or path can be recorded.
- Connected ordinary chat retrieval after current-paper, research-basket, or
  full-library filtering, so its count reflects the evidence set passed to the
  next product stage rather than every internal BM25 candidate. Structured
  exact/section matches are distinguished from the BM25 product channel.
- Connected Agent evidence retrieval through a small wrapper that records both
  success and propagated error outcomes. Explicit selected-source scope is
  represented only by the stable `selected_sources` ID; selected paths and
  retrieved rows never enter telemetry.
- Added retrieval cancellation/error projection in the chat worker without
  copying exception text. Unknown scope/channel values fail into stable
  `unknown_*` IDs instead of becoming arbitrary attributes.
- Added focused persistence, scope, translation, Agent success/error, and
  content-redaction tests. The 186-test retrieval/Agent/task-runtime regression
  set passes; Ruff and diff checks pass.

### Priority assessment after P1.4a-3

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.4a-4 | Emit aggregate outcomes for the existing production citation verifier | High | Completes retrieval→generation→verification→finalization diagnosis; must not imply semantic entailment | Do next |
| P1.4a-5 | Run a privacy replay over every local span type and freeze the schema/retention contract | High | External export remains unsafe until negative-content tests cover the complete event surface | After verifier spans |
| P1.2b | Pre-register parent-collapsed multi-vector passage-focus development evaluation | High | Next direct retrieval-quality experiment; keep parent citation identity unchanged | After P1.4a privacy freeze |
| P1.4b | Optional OpenTelemetry exporter | Medium | Explicit opt-in only; lower value than improving retrieval | After P1.2b decision unless diagnostics demand it |
| P1.3 | Dense Off/Local/Cloud product choice | High but blocked | Current candidate remains rejected | Blocked |
| P2 | Ratcheted decomposition | High long-term | Retrieval observation is a new small module; avoid adding policy to task runtime | Incremental |

### 2026-09-03 — P1.4a-4 complete: honest production-verifier outcomes

- Added one aggregate claim-verification span for the existing production
  verifier. It records only evidence-status ID, total/supported/unsupported
  counts, stable step status, fallback reason, and measured verification-step
  duration.
- Identifies the active method explicitly as `lexical_structural_v1`. This
  prevents dashboards from misrepresenting the current structural-citation and
  lexical-overlap check as semantic entailment. The benchmark-only multilingual
  NLI model remains disconnected from user requests and is not loaded or rerun.
- General or external-background answers record verification as skipped and
  `not_applicable`; verification errors and cancellation receive stable redacted
  reasons. No claim, evidence, citation text, question, answer, source path, or
  exception message is an accepted attribute.
- Measured the synchronous existing-RAG verifier path and reused Agent tool-step
  timing for direct Agent runs. This completes a local diagnostic chain across
  retrieval, model calls, production verification, and answer finalization.
- Forty-six focused verifier, Agent, presentation, trace, NLI-shadow, and
  content-redaction tests pass; Ruff and diff checks pass.

### Priority assessment after P1.4a-4

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.4a-5 | Freeze and validate the complete local span/privacy/retention contract | High | Default-off is insufficient if files can grow forever or an allowlist silently broadens | Do next |
| P1.2b | Pre-register and run a development-only parent-collapsed multi-vector passage-focus experiment | Very high | Directly targets the remaining semantic retrieval gap while preserving citation identity | After P1.4a contract |
| P1.4b | Optional OpenTelemetry exporter | Medium | Useful operationally, but lower user value than retrieval improvement and must remain explicit opt-in | After P1.2b decision |
| P1.3 | Dense Off/Local/Cloud product choice | High but blocked | Current candidate still fails promotion | Blocked |
| P2 | Ratcheted decomposition | High long-term | Production-verifier observation lives in its own module | Incremental |

### 2026-09-03 — P1.4a-5 complete: privacy and retention contract frozen

- Froze the complete five-event local observation contract in
  `docs/local_observability_contract_v1.json`: answer finalization, Agent run,
  retrieval, production claim verification, and model call. Every allowed
  attribute now has exactly one approved stable-ID, boolean, non-negative
  count, or duration type; untyped attributes are rejected.
- Added an executable privacy replay that attempts to inject query, prompt,
  claim/evidence text, response, exception, path, URL, model name, credentials,
  and API-key canaries into every event. CI and Windows release validation fail
  if any prohibited field is admitted, the runtime/report contracts drift, or
  external export becomes enabled.
- Added bounded local retention. The explicit JSONL sink rotates at 8 MiB by
  default and retains one backup; both limits are configurable within guarded
  bounds. Default recording remains off, and the sink remains local-only.
- Added rotation, complete type coverage, allowlist drift, prohibited-content,
  and contract replay tests. The validator reports five redacted event types,
  bounded local retention, and external export off; 23 focused tests, Ruff,
  and diff checks pass.

### Priority assessment after P1.4a-5

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.2b-1 | Freeze a parent-collapsed multi-vector passage-focus development protocol before measuring | Very high | Must not tune on the 12-case frozen semantic acceptance slice or change citation identity | Do next |
| P1.2b-2 | Run the development measurement and promote to one frozen replay only if pre-registered development gates pass | Very high | May recover semantic misses caused by representation dilution; index cost/latency must be measured | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Contract is safe enough to design, but retrieval quality has greater user value | Defer behind P1.2b |
| P1.3 | Dense Off/Local/Cloud product choice | High but blocked | User choice remains required only after a retrieval candidate passes quality gates | Blocked |
| P2 | Ratcheted decomposition | High long-term | Local telemetry is split into small typed adapters and a frozen validator | Incremental |

### 2026-09-03 — P1.2b-1 complete: passage-focus protocol frozen before measurement

- Froze a development-only parent-collapsed multi-vector candidate before
  measuring it. It retains every existing full-parent E5 embedding and adds
  exact original-text child slices only for parents above 320 content tokens:
  256-token windows, 192-token stride, and 64-token overlap from fast-tokenizer
  offsets, with no heading injection and no hidden child cap.
- Parent score is the maximum cosine score across its full and child
  representations. Ranking always returns the original parent chunk ID, full
  parent text, metadata, and page locator; child IDs/text can never become
  citation targets and stored knowledge-base chunks are unchanged.
- Froze the existing source-safe RRF contract (2:1 sparse:dense, RRF k=20,
  protected BM25 top five, Dense source gate depth five) and explicitly excluded
  cross-encoder reranking from this experiment.
- Froze all development gates: raw multi-vector Recall@10 cannot regress parent
  Dense in either scope and must gain at least 0.02 in one; safe RRF must preserve
  product Recall@5/wrong-source@5, not regress product Recall@10/MRR/nDCG@10,
  and gain at least 0.02 Recall@10 in one scope; representation count is capped
  at 2× and p95 latency at max(2× parent Dense, parent Dense + 15 ms).
- The 12-case semantic slice remains forbidden for tuning and may be replayed
  at most once only after every development gate passes. Its original 15-point
  safe-RRF Recall@10 gain in both scopes remains the binding promotion gate.
  Added a frozen registry, validator, CI/release gates, and two drift tests;
  validator, tests, Ruff, and diff checks pass.

### Priority assessment after P1.2b-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.2b-2 | Implement parent-collapse retrieval/index tooling and run the one development measurement | Very high | Decide with measured recall, ranking, index multiplier, and latency | Do next |
| P1.2b-3 | If every development gate passes, run exactly one frozen semantic replay | Very high | No replay or UI work after a development failure | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Still explicit opt-in; lower priority than evidence retrieval quality | After P1.2b decision |
| P1.3 | Dense Off/Local/Cloud product choice | High but blocked | Requires frozen candidate promotion | Blocked |
| P2 | Ratcheted decomposition | High long-term | Keep representation splitting/retrieval in a small standalone module | Alongside P1.2b-2 |

### 2026-09-03 — P1.2b-2 complete: global MaxP multi-vector candidate rejected

- Ran the one pre-registered development measurement cache-only on CUDA. The
  independent child index contains 1,672 child windows for 2,251 unchanged
  parents: 3,923 representations total, or 1.742781×. No child ID was returned
  and no stored parent chunk or locator changed.
- Raw parent-collapsed multi-vector Dense improves scoped Recall@10 from
  0.609756 to 0.634146 (+0.024390), but full-library Recall@10 falls from
  0.414634 to 0.329268 (-0.085366). It rescues scoped foveated-imaging and
  heterodyne cases, but pushes five full-library relevant results below rank 10
  while rescuing only one.
- Source-safe RRF preserves product Recall@5 and wrong-source@5 exactly in both
  modes and does not regress product Recall@10/MRR/nDCG@10. However, its
  Recall@10 gain is 0 scoped and only 0.012195 full-library, below the frozen
  0.02 development requirement.
- Representation cost and latency are not the blocker: multiplier 1.742781 is
  below 2×; multi-vector p95 is 39.895 ms scoped and 246.614 ms full-library,
  below the pre-registered 55.830/309.136 ms limits. Ranking quality fails two
  mandatory gates.
- The measured failure is consistent with global MaxP representation-count
  bias: longer parents receive more child-vector similarity opportunities and
  can displace a shorter relevant parent in a large candidate pool. Tuning the
  window/stride on these cases would overfit the development set, so this
  candidate is rejected and the allowed frozen semantic replay count remains
  unused. Product BM25 and all user surfaces remain unchanged.
- Added the benchmark-only parent-collapse retriever, offset-exact child index,
  frozen report/decision validator, pinned metrics, and identity/cost/latency
  tests. The report validator, seven focused tests, Ruff, and diff checks pass.

### Priority assessment after P1.2b-2

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.5a | Compare primary-source evidence for one stronger multilingual single-vector embedding candidate, then freeze it before measurement | Very high | Current E5 demonstrates semantic value but misses promotion; avoid MaxP length bias and remote-code/license risk | Do next |
| P1.5b | Measure the frozen stronger candidate on development first, then at most once on semantic acceptance if it passes | Very high | Same source-safe RRF, citation identity, Recall/MRR/nDCG and latency gates | Conditional |
| P1.2c | Consider candidate-limited child reranking with representation-count calibration | Medium | Only after a stronger single-vector baseline; global MaxP is rejected | Deferred |
| P1.4b | Optional OpenTelemetry exporter | Medium | Privacy contract is ready, but this does not improve evidence finding | After retrieval candidate decision |
| P1.3 | Dense Off/Local/Cloud product choice | High but blocked | No user switch until a candidate passes frozen promotion | Blocked |
| P2 | Ratcheted decomposition | High long-term | New experiment code remains isolated; do not add rejected MaxP to product modules | Incremental |

### 2026-09-03 — P1.5a complete: stronger single-vector candidate selected and frozen

- Compared primary model cards/papers and immutable repository metadata before
  measurement. Selected `intfloat/multilingual-e5-large-instruct` at revision
  `274baa43b0e13e37fafa6428dbc7938e62e5c439`: MIT, 24-layer XLM-R large,
  1024-dimensional attention-masked mean-pooled embeddings, 100-language
  training coverage, and an official safetensors artifact pinned at
  1,119,825,680 bytes and SHA-256 `dd6b6e4f...390fa85`.
- Froze the model-required query instruction to the evidence-finding task:
  retrieve relevant passages from the user's paper library that provide
  evidence for the academic question. Documents receive no instruction. The
  exact instruction, mean pooling, L2 normalization, 512-token window, float16
  CUDA execution, and provider identity are immutable benchmark inputs.
- Rejected BGE-M3 for this candidate round despite its attractive 1024
  dimensions, 8,192-token window, 100+ languages, and dense/sparse/ColBERT
  support. Its pinned official revision exposes the 2,271,145,830-byte primary
  model as `pytorch_model.bin` without an equivalent primary safetensors file,
  and correct multi-function integration would add a model-specific runtime
  before a stronger standard single-vector baseline is known.
- The remaining 512-token E5 limit is acceptable for this decision because the
  prior exact-token diagnostic found zero truncation among all semantic qrels
  and focus failures. This candidate tests instruction/representation quality,
  not a disproven truncation hypothesis.
- Froze development gates against base E5 and product BM25, including source-
  safe RRF invariants, at least 0.05 raw Recall@10 gain in one scope, at least
  0.02 safe-RRF gain in both scopes, a 1.5× index-size ceiling, and bounded p95
  latency. Only an all-pass development result permits one frozen semantic
  replay; product loading and UI remain disabled. Added registry, validator,
  CI/release gates, and two drift tests; all pass.

### Priority assessment after P1.5a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.5b-1 | Prepare and SHA-verify the pinned safetensors model through explicit resumable download | High | Benchmark-only; no normal-runtime downloads or remote code | Do next |
| P1.5b-2 | Run one development measurement against base E5 and product BM25 | Very high | Determines whether model quality justifies a frozen replay and larger index | After preparation |
| P1.5b-3 | Run one frozen semantic replay only if every development gate passes | Very high | Original 15-point safe-RRF gate remains binding | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Defer until retrieval candidate decision | Deferred |
| P1.3 | Dense Off/Local/Cloud choice | High but blocked | No product/UI work before frozen promotion | Blocked |

### 2026-09-03 — P1.5b-1 complete: pinned model prepared and independently verified

- Prepared only the already-frozen `intfloat/multilingual-e5-large-instruct`
  revision `274baa43b0e13e37fafa6428dbc7938e62e5c439`. The resumable download is an
  explicit benchmark setup action; normal product startup and retrieval still
  perform no network model download.
- The benchmark preparer accepted the cached safe `model.safetensors`, then an
  independent filesystem check confirmed exactly 1,119,825,680 bytes and
  SHA-256 `dd6b6e4f52db0a7aff83a13d10e6c5342ef9f6ab799bad3221f4b35ef390fa85`.
  No pickle weight or remote model code is permitted by the loader.
- Model acquisition required one interrupted transfer and one resumable retry;
  the final immutable artifact, rather than download success alone, is the
  completion condition. This keeps the upcoming one-shot measurement
  reproducible without weakening the frozen candidate contract.

### Priority assessment after P1.5b-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.5b-2 | Run the single frozen development measurement against base E5 and product BM25 | Very high | All candidate inputs are now present and immutable; the report path has a one-run guard | Do next |
| P1.5b-3 | Run one frozen semantic replay only if every development gate passes | Very high | Prevents acceptance-set tuning; original 15-point promotion gate remains binding | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Local redacted telemetry already covers diagnosis; external export does not improve retrieval | Deferred |
| P1.3 | Dense Off/Local/Cloud choice | High but blocked | No product/UI work before a frozen candidate earns promotion | Blocked |
| P1.2c | Candidate-limited child reranking | Medium-low | Global MaxP failed; do not add representation complexity before the stronger single-vector result | Deferred |

### 2026-09-03 — P1.5b-1a complete: safe comparison baseline made reproducible

- A measurement preflight stopped before query encoding because the stricter
  loader required safetensors while the local base-E5 cache contained only its
  older pickle weight. No report was created, so the frozen one-run development
  measurement was not consumed.
- Added the comparison baseline itself to the pre-measurement contract:
  `intfloat/multilingual-e5-base` at revision
  `d128750597153bb5987e10b1c3493a34e5a4502a`, with the official
  1,112,201,288-byte `model.safetensors` and SHA-256
  `a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7`.
  Candidate selection, query prefixes, fusion, fixtures, and gates did not
  change.
- The first verification correctly stopped but exposed a verifier bug: the
  project fixture hash helper normalizes text newlines and must not hash binary
  weights. Added a streaming binary SHA-256 helper and regression test while
  retaining normalized hashes for Git-tracked JSON fixtures.
- The explicit preparer now downloads and independently verifies both pinned
  safetensors artifacts; both pass byte and binary-SHA checks. The benchmark
  loader remains cache-only with `trust_remote_code=False` and
  `use_safetensors=True`. Two focused benchmark tests pass.

### Priority assessment after P1.5b-1a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.5b-2 | Run the one guarded development measurement now that both providers are safe and reproducible | Very high | The prior preflight created no report or metric; all frozen inputs are now loadable cache-only | Do next |
| P1.5b-3 | Run one frozen semantic replay only after an all-pass development decision | Very high | Acceptance data remains unseen by this candidate; 15-point safe-RRF gate is unchanged | Conditional |
| P1.4b | Add optional external telemetry export | Medium | Does not address the active retrieval quality uncertainty | Deferred |
| P1.3 | Add Dense Off/Local/Cloud product controls | High but blocked | Exposing an unpromoted candidate would transfer unresolved quality risk to users | Blocked |
| P2 | Continue ratcheted decomposition around accepted seams | High long-term | Keep benchmark hardening isolated; avoid broad refactors during a one-shot experiment | Incremental |

### 2026-09-03 — P1.5b-2 complete: E5-large-instruct rejected on development

- Ran the single guarded development measurement over the frozen 41-case,
  2,251-chunk corpus. The report was written once to
  `docs/retrieval_multilingual_e5_large_instruct_development_v1.json`; the
  candidate remained cache-only, product-disabled, and source text remained
  local.
- Raw large-E5 Recall@10 regressed against base E5 by 0.048780 scoped and
  0.085366 full-library. Source-safe RRF regressed against product BM25 by
  0.024390 scoped and 0.012195 full-library, and therefore also failed the
  no-regression checks. It preserved product Recall@5 and wrong-source@5, but
  protection alone is not a quality gain.
- Resource gates passed: the 9,801,713-byte index is 1.307474× the base index;
  large p95 query latency was 56.383 ms scoped and 210.960 ms full-library,
  within their frozen limits. The index build took 32.430 seconds. These costs
  do not justify a candidate that loses retrieval quality.
- Development status is `rejected`; the required all-pass condition is false.
  The frozen semantic acceptance fixture must not be run, preserving it for a
  genuinely eligible future candidate. No product loader, setting, API, or UI
  was enabled.

### Priority assessment after P1.5b-2

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.5b-2a | Pin the measured report contract and explain development regressions without tuning | High | Prevents silent metric drift and identifies whether the next hypothesis is model, query, or corpus related | Do next |
| P1.6a | Freeze an offline candidate comparison protocol before selecting another model | Very high | Two intuitive architecture upgrades have failed; model-by-model guessing is now low-yield | Proposed next |
| P1.6b | Evaluate multiple safe local single-vector candidates on a new inner-development split or public task, then nominate one | Very high | Must not repeatedly consume the frozen semantic acceptance set or tune on the current 41 cases | After protocol |
| P1.4b | Optional OpenTelemetry exporter | Medium | Redacted local traces are sufficient for current offline diagnosis | Deferred |
| P1.3 | Dense Off/Local/Cloud product controls | High but blocked | Both measured promotion candidates failed; keep dense unavailable to ordinary users | Blocked |
| P1.2c | Candidate-limited child reranking | Low | Large single-vector and global MaxP both failed; another retrieval layer now risks compounding noise | Deprioritized |

### 2026-09-03 — P1.5b-2a complete: rejection pinned and regressions bounded

- Strengthened report validation to pin Recall@5, Recall@10, MRR, nDCG@10,
  and wrong-source@5 for base dense, large dense, and source-safe RRF in both
  scopes. It also pins the measured index count/size ratio, candidate and base
  identities, cache-only execution, and byte-for-byte equality of the embedded
  product channel with the independently frozen BM25 baseline.
- Recomputes every development gate from the pinned channels and rejects any
  report/decision mismatch. A mutation test proves that changing the measured
  large scoped Recall@10 is detected along with the resulting arithmetic drift.
  The validator, ten focused tests, Ruff, and `git diff --check` pass.
- Development-only case comparison found six losses versus three rescues for
  raw large dense in scoped Recall@10, and seven losses versus three rescues in
  full-library Recall@10. Safe RRF had one scoped rescue versus two losses and
  two full-library rescues versus two losses; fractional multi-label recall
  still made the aggregate regress.
- Coverage labels are too sparse and overlapping for a model-choice conclusion.
  The candidate helped the isolated formula/system examples but was unstable on
  comparison, mechanism, and scope-boundary cases. This supports a protocol-led
  next step rather than selecting another larger model from anecdotes.

### Priority assessment after P1.5b-2a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.5b-2b | Check whether exact large-tokenizer truncation intersects qrels and observed regressions | High | 167 corpus chunks were truncated; causal attribution must precede the next candidate protocol | Do next |
| P1.6a | Freeze candidate comparison and data-splitting protocol | Very high | Avoids serial model guessing and reuse of the acceptance set | After truncation diagnostic |
| P1.6b | Nominate candidates using public evidence, then compare on an inner-development split | Very high | Requires safe weights, standard local loading, and explicit compute/storage budget | After protocol |
| P1.3 | Product Dense Off/Local/Cloud controls | High but blocked | Rejected results remain invisible to ordinary users | Blocked |
| P1.4b | Optional OpenTelemetry exporter | Medium | No current diagnostic requires external telemetry | Deferred |

### 2026-09-03 — P1.5b-2b complete: truncation ruled out as the main regression cause

- Replayed only token-length accounting with the pinned large-model tokenizer;
  no retrieval query, embedding, fusion setting, or acceptance fixture was run.
  The count reproduces the measured 167 truncated chunks among 2,251 corpus
  chunks.
- The 41 development cases reference 39 unique qrel chunks; six exceed 512
  tokens. However, none of the six scoped raw-large regression cases and none
  of the seven full-library raw-large regression cases has a truncated relevant
  chunk. Their maximum relevant lengths were 169–420 and 221–463 tokens,
  respectively.
- Of the four scope-specific safe-RRF regression occurrences, only
  `qclfm-vs-iism-depth-signal` contains one 525-token relevant chunk; the two
  full-library safe regressions contain none. This cannot explain the broader
  raw or fused loss pattern.
- Therefore broad rechunking, larger context windows, and child-vector rescue
  remain rejected as the next intervention. The remaining hypothesis is
  representation/task alignment and candidate-selection variance, which needs
  comparative evidence rather than post-hoc tuning of this rejected model.

### Priority assessment after P1.5b-2b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6a | Freeze a multi-candidate comparison, split, budget, and nomination protocol before naming models | Very high | Stops one-model-at-a-time guessing and protects both current development and frozen acceptance evidence | Do next |
| P1.6b | Curate safe standard-loader candidates from primary evidence under the frozen protocol | Very high | Candidate diversity should target representation/task alignment, not parameter count | After protocol |
| P1.6c | Run an inner-development screen and nominate at most one candidate for the existing 41-case gate | Very high | Requires a non-overlapping tuning surface or public retrieval task | Conditional |
| P1.3 | Product Dense Off/Local/Cloud controls | High but blocked | No dense candidate has earned product exposure | Blocked |
| P1.4b | Optional OpenTelemetry exporter | Medium | Local offline evidence already identified the decision boundary | Deferred |
| P1.2c | More chunk representations | Low | Truncation does not explain the observed failures and MaxP already regressed globally | Rejected for now |

### 2026-09-03 — P1.6a complete: protocol frozen before candidate shortlisting

- Added `dense_candidate_comparison_protocol_v1` before naming any new model.
  A deterministic locale-stratified SHA-256 split assigns 29 cases to
  screening and 12 cases to internal confirmation; English cases are split 2/1.
  The validator recomputes membership from the frozen 41-case fixture rather
  than trusting copied lists.
- At most three new candidates and one variant per candidate may be screened.
  Each must have a declared license, immutable revision, SHA-pinned safetensors
  no larger than 1.5 GB, at most 1,024 dimensions, standard Transformers
  loading without remote code, and an official query contract frozen before
  results. Parent chunks, source-safe 2:1 RRF, and absence of cross-encoder
  reranking remain fixed.
- Every candidate receives one screening run on the 29-case partition only.
  Only candidates passing raw/control, safe-RRF/product, wrong-source, index,
  and latency checks can be ranked; the deterministic ranking nominates at most
  one candidate. Only that nominee may see the 12-case confirmation partition,
  exactly once.
- A confirmed nominee's 29+12 per-case results are combined under the existing
  full-development gates. Only an all-pass result permits one replay of the
  untouched 12-case semantic acceptance fixture, where the original 15-point
  safe-RRF Recall@10 gain remains binding. Rejected candidates cannot be
  retested under protocol v1, and no product/UI exposure is permitted.
- Added a protocol registry, validator, two mutation tests, and CI/Windows
  release validation. All focused tests, Ruff, and `git diff --check` pass.

### Priority assessment after P1.6a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6b | Curate and batch-freeze up to three representation-diverse candidates using only primary evidence | Very high | Must satisfy the protocol before any screening metric is observed | Do next |
| P1.6c | Implement one batch screening runner that enforces the 29-case boundary and deterministic nomination | Very high | Build only after the batch shortlist is frozen | After shortlist |
| P1.6d | Run screening once per frozen candidate and permit at most one confirmation | Very high | No ad-hoc reruns or per-model tuning | Conditional |
| P1.3 | Product Dense Off/Local/Cloud controls | High but blocked | Product remains BM25 until the full frozen promotion chain passes | Blocked |
| P1.4b | Optional OpenTelemetry exporter | Medium | Lower direct reading benefit than retrieval candidate selection | Deferred |

### 2026-09-03 — P1.6b complete: two-candidate batch frozen from primary evidence

- Selected two candidates without reading screening results. Qwen3-Embedding-
  0.6B represents an instruction-aware decoder/last-token route: Apache-2.0,
  100+ languages, 1,024 dimensions, an official 8,192-token usage setting, and
  a 1,191,586,416-byte safetensors artifact. Granite-Embedding-278M-
  Multilingual represents a compact encoder/CLS route: Apache-2.0, 12 languages
  including Chinese and English, 768 dimensions, 512 tokens, and a
  556,109,872-byte safetensors artifact.
- Froze each immutable repository revision, binary weight SHA-256, official
  pooling/padding/prefix contract, float16 CUDA execution, and minimum
  Transformers version. Qwen uses the repository's exact English query prompt
  and last-token pooling; Granite uses no prefix and CLS pooling. Both use
  standard `AutoModel`, `trust_remote_code=False`, and L2 normalization.
- Did not fill the third protocol slot merely to increase model count. Excluded
  Arctic, GTE, and Jina for remote-code requirements; Nomic for remote code and
  weight size; BGE-M3 and non-instruct E5-large for the 1.5 GB ceiling;
  EmbeddingGemma for manual gated-license access; and multilingual MPNet for
  its 128-token sentence-similarity rather than paragraph-retrieval contract.
- Added the batch registry and validator, two mutation tests, and CI/Windows
  release validation. The validator also verifies the parent protocol hash,
  recomputes each encoding-contract hash, and keeps all screening counters and
  product exposure at zero. Four focused tests, Ruff, and diff checks pass.

### Priority assessment after P1.6b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6c-1 | Generalize the benchmark-only provider for frozen mean/CLS/last-token contracts without changing product dependencies | Very high | Pooling and padding must match official model semantics exactly | Do next |
| P1.6c-2 | Build a batch runner that can see only the 29 screening IDs and produces one immutable report per candidate | Very high | Must enforce data isolation and deterministic nomination before measurement | After provider |
| P1.6c-3 | Prepare pinned weights and an isolated compatible benchmark runtime | High | Qwen requires Transformers >=4.51; avoid mutating product runtime assumptions | After runner validation |
| P1.6d | Run the two screening measurements, nominate at most one, then conditionally confirm | Very high | Exactly one variant/run per frozen candidate | Conditional |
| P1.3 | Product Dense Off/Local/Cloud controls | High but blocked | No ordinary-user switch before the entire promotion chain passes | Blocked |

### 2026-09-03 — P1.6c-1 complete: benchmark provider supports frozen pooling variants

- Generalized the benchmark-only Transformers provider to support attention-
  masked mean, CLS, and last-token pooling plus explicit left/right padding.
  Qwen's last-token selection follows the official left-padding behavior and
  Granite's CLS path selects token zero; all outputs remain float32-normalized
  vectors after the frozen float16 model forward pass.
- Pooling and explicit padding now participate in new provider fingerprints.
  The legacy mean-pooling path intentionally preserves its old fingerprint, so
  base E5, E5-large-instruct, and parent-multivector reports and indexes do not
  drift. Runtime metadata now states pooling, effective padding policy, and
  `trust_remote_code=False`.
- Frozen batch sizes at one for the 8,192-token Qwen candidate and four for the
  512-token Granite candidate before measurement. The batch validator now
  constructs each provider descriptor and proves its fingerprint matches the
  separately frozen encoding-contract hash.
- Old candidate/report validators, the new batch validator, 15 focused tests,
  Ruff, and diff checks pass. This remains isolated from product imports and
  dependencies.

### Priority assessment after P1.6c-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6c-2 | Implement the 29-case-only screening runner, report validation, and deterministic one-nominee reducer | Very high | Must make confirmation leakage and alternate report paths impossible | Do next |
| P1.6c-3 | Create isolated compatible benchmark runtime and prepare both SHA-pinned weights | High | Qwen dependency upgrade must not mutate the product Python environment | After runner tests |
| P1.6d | Execute one screening run per candidate | Very high | Reports become immutable and metrics are pinned immediately afterward | Conditional |
| P1.6e | Run one confirmation only if exactly one eligible nominee exists | Very high | Confirmation IDs remain inaccessible until nomination | Conditional |
| P1.3 | Product controls | High but blocked | No candidate has crossed screening, confirmation, development, and acceptance | Blocked |

### 2026-09-03 — P1.6c-2 complete: screening and nomination runner enforces evidence boundaries

- Added one batch runner with fixed candidate report paths. It derives an
  in-memory screening fixture containing exactly the frozen 29 IDs, verifies
  the corpus against it, and evaluates base E5, product BM25, candidate dense,
  and source-safe RRF. Every channel/mode report must contain all and only those
  29 IDs and have no intersection with the 12 confirmation IDs.
- Screening requires raw Recall@10 no-regression in both scopes; safe-RRF
  Recall@5 and wrong-source@5 preservation; safe Recall@10/MRR/nDCG@10 no-
  regression; the 1.5× index ceiling; and the frozen latency rule. The decision
  is always recomputed from channel metrics rather than trusted from the report.
- Each candidate can write only its canonical one-shot report. A separate
  reducer refuses to nominate until every frozen candidate report validates,
  ignores all ineligible candidates, applies the frozen ranking order, and
  emits at most one immutable nomination report. Confirmation and product flags
  remain false in both report types.
- Preparation is a separate explicit-download command; measurement rejects
  downloads. Added tests for exact partition exclusion, multi-condition gates,
  and eligible-only deterministic nomination. Eleven related tests, Ruff, and
  diff checks pass; no model screening has run.

### Priority assessment after P1.6c-2

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6c-3a | Create a benchmark-only virtual environment layered over local Torch and pin a Qwen-compatible Transformers stack | High | Isolates dependency changes from product and existing quality gates | Do next |
| P1.6c-3b | Run cache-only load/pooling smoke tests for both frozen candidates before downloading full weights where possible | High | Detects architecture/version failures before consuming screening runs | After environment |
| P1.6c-3c | Explicitly prepare and SHA-verify both candidate weights | High | About 1.75 GB total; resumable and benchmark-only | After smoke prerequisites |
| P1.6d | Execute Granite then Qwen screening exactly once and pin reports | Very high | Lower-risk Granite first; neither may see confirmation | Conditional |
| P1.3 | Product controls | High but blocked | Still no promoted dense backend | Blocked |

### 2026-09-03 — P1.6c-3a/3b complete: isolated runtime and architecture smoke verified

- Froze a benchmark-only requirements file and hash before installation. The
  first design inherited system Torch 1.12.1; a metadata-only Qwen preflight
  failed before weight loading because Transformers 4.51 invokes
  `torch.compiler`. No screening data or report was accessed or created.
- Replaced that failed runtime design, still before measurement, with an
  isolated Torch 2.3.1+cu121 inside the `test_results` virtual environment.
  The host RTX 3070 Laptop GPU has 8,192 MiB, compute capability 8.6, and a
  driver compatible with CUDA 12.1. Product/global Torch and Transformers were
  not uninstalled or modified.
- Verified actual imported versions and paths: Python 3.9.7, isolated Torch
  2.3.1+cu121 with CUDA available, Transformers 4.51.3, Tokenizers 0.21.4,
  Hugging Face Hub 0.30.2, and Safetensors 0.5.3. The requirements hash and
  exact runtime/GPU contract are enforced by the candidate-batch validator.
- Downloaded configuration/tokenizer files only, then resolved Qwen3Config to
  Qwen3Model and Granite's XLMRobertaConfig to XLMRobertaModel without remote
  code. Both tokenizers are fast. Tensor smoke checks proved left/right last-
  token selection, CLS selection, and masked mean pooling. A non-fatal inherited
  torchvision image-extension warning is irrelevant because the benchmark is
  text-only and both required model classes resolved successfully.

### Priority assessment after P1.6c-3a/3b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6c-3c | Explicitly download and binary-SHA verify the two pinned safetensors artifacts | High | 1.75 GB total; resumable, immutable, and outside product artifacts | Do next |
| P1.6d-1 | Cache-only full model load and one-text embedding smoke for Granite then Qwen | Very high | Confirms weights, dtype, GPU memory, pooling, and vector dimensions without reading screening cases | After preparation |
| P1.6d-2 | Run Granite's one allowed 29-case screening measurement and pin it | Very high | Smaller/safer candidate first | Conditional |
| P1.6d-3 | Run Qwen's one allowed screening measurement and pin it | Very high | 8GB GPU plus long chunks may expose bounded OOM risk; no fallback contract changes | Conditional |
| P1.6e | Reduce both reports to zero or one nominee; confirm only if allowed | Very high | Confirmation remains inaccessible | Conditional |

### 2026-09-03 — P1.6c-3c complete: both pinned candidate weights prepared

- Ran only the explicit `--prepare-models --allow-download` path inside the
  frozen virtual environment. Qwen's 1,191,586,416-byte safetensors and
  Granite's 556,109,872-byte safetensors were fetched from their immutable
  revisions and independently passed streaming binary SHA-256 verification.
- The downloads stayed under the shared Hugging Face cache and benchmark-only
  `test_results` environment; they are not repository artifacts, product
  dependencies, startup downloads, or external processing of paper text.
- Preparation never loaded the development fixture, built an index, encoded a
  question, or created a screening report. Both frozen candidates therefore
  retain their single allowed screening run, and all confirmation/acceptance
  evidence remains untouched.

### Priority assessment after P1.6c-3c

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6d-1 | Cache-only full-weight load plus synthetic one-query/one-document embedding smoke for each candidate | Very high | Proves safetensors/model/GPU/pooling/dimension compatibility without consuming evaluation evidence | Do next |
| P1.6d-2 | Run and immediately pin Granite's one 29-case screening report | Very high | Smaller model establishes the first independent screening result | After smoke |
| P1.6d-3 | Run and immediately pin Qwen's one screening report | Very high | Longest observed corpus chunk is below 8,192, but 8GB GPU remains the main operational risk | After Granite |
| P1.6e | Generate deterministic nomination after both reports validate | Very high | Confirmation only if the reducer emits exactly one eligible nominee | Conditional |
| P1.3 | Product controls | High but blocked | Preparation success is not retrieval-quality evidence | Blocked |

### 2026-09-03 — P1.6d-1 complete: both real-weight embedding smokes pass

- Loaded each frozen model from cache through its isolated standard AutoModel
  class with safetensors only, CUDA float16, and remote code disabled. Used one
  synthetic English query and document unrelated to any evaluation fixture, so
  neither the 29 screening cases nor 12 confirmation cases were accessed.
- Granite produced 768-dimensional CLS-pooled query/document vectors with both
  L2 norms equal to 1.0. Its observed provider fingerprint exactly matches the
  frozen contract; one-query and one-document inference completed on CUDA.
- Qwen produced 1,024-dimensional left-padded last-token query/document vectors
  with both L2 norms equal to 1.0. Its provider fingerprint also matches the
  frozen contract and the official query prefix was present. The 8GB GPU loaded
  and inferred the real model without OOM.
- Windows Torch reported that flash attention was not compiled, so both models
  use the supported SDPA fallback. This affects performance but not the frozen
  pooling or retrieval contract, and latency gates will measure the actual
  environment. Both candidates still have zero screening runs.

### Priority assessment after P1.6d-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6d-2 | Run Granite's single 29-case screening measurement, validate it, and pin exact metrics | Very high | All preflight gates pass; smaller candidate minimizes first-run operational risk | Do next |
| P1.6d-3 | Run Qwen's single screening measurement and pin exact metrics | Very high | Proceed independently regardless of Granite result; batch comparison requires both | After Granite report is locked |
| P1.6e | Deterministically nominate zero or one candidate | Very high | Confirmation remains blocked until both reports validate | After both reports |
| P1.3 | Product controls | High but blocked | Synthetic smoke is compatibility evidence, not promotion evidence | Blocked |

### 2026-09-03 — P1.6d-2 complete: Granite screened once and rejected

- Ran Granite exactly once on the frozen 29-case screening partition. Report
  validation confirms every base/product/candidate/safe channel in both scopes
  contains exactly those 29 IDs and no confirmation ID. The report is immutable,
  cache-only, remote-code-free, and product-disabled.
- Raw Granite Recall@10 regressed versus base E5 by 0.137932 scoped and 0.172414
  full-library. Source-safe RRF improved scoped Recall@10 by 0.017242 over BM25
  but regressed full-library by 0.068965, so raw no-regression and safe no-
  regression both fail. Recall@5 and wrong-source@5 protection passed.
- Resource checks passed: 7,496,695-byte index, 1.000002× the base index;
  26.929 ms scoped and 144.977 ms full-library candidate p95 were below frozen
  limits. The model is compact and operationally acceptable, but speed cannot
  offset material evidence-retrieval loss.
- Status is permanently `rejected` under protocol v1, nomination eligibility is
  false, and confirmation was not accessed. Pinned all five key metrics for the
  two control and two candidate channels, the index contract, identities, and
  gate arithmetic; the validator and 12 related tests pass.

### Priority assessment after P1.6d-2

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6d-3 | Run Qwen's one frozen 29-case screening and pin its report | Very high | Independent contract was frozen before Granite results; must not be adjusted now | Do next |
| P1.6e | Reduce both validated reports and nominate Qwen only if it independently passes every gate | Very high | Granite is ineligible; zero nominees remains a valid outcome | After Qwen |
| P1.7a | If no nominee, stop model search and improve eval/query failure taxonomy before any protocol v2 | High | Three model routes failing would indicate selection uncertainty rather than insufficient parameter count | Conditional |
| P1.3 | Product controls | High but blocked | Granite is not exposed and cannot be user-selected | Blocked |

### 2026-09-03 — P1.6d-3 complete: Qwen screened once; representation passes, fusion rejects

- Ran Qwen exactly once on the same frozen 29-case screening partition with its
  pre-Granite contract unchanged. All four channels in both scopes contain only
  screening IDs; no confirmation or semantic-acceptance case was accessed.
- Raw Qwen Recall@10 improved over base E5 by 0.172413 scoped and 0.206897
  full-library. This is the first tested dense representation to produce a
  material, two-scope semantic retrieval gain and disproves the broader idea
  that dense retrieval itself lacks value for this library.
- The frozen source-safe RRF still failed: versus product BM25, scoped
  Recall@10 gained 0.017242, while full-library Recall@10 lost 0.017241.
  Full-library MRR improved slightly, but nDCG@10 also regressed; the all-metric
  no-regression gate is therefore false. Recall@5 and wrong-source@5 protection,
  latency, and index-size gates all passed.
- The 9,801,699-byte index is 1.307472× the control and built in 130.718 seconds.
  Candidate p95 was 71.579 ms scoped and 233.723 ms full-library, within frozen
  limits. These resource results make Qwen operationally plausible but do not
  override the evidence-ranking failure.
- Status is `rejected` under protocol v1, nomination eligibility is false, and
  confirmation remains untouched. Pinned both candidates' five key metrics in
  all channels/scopes, index contracts, identities, and gate arithmetic. Both
  report validators, 12 related tests, Ruff, and diff checks pass.

### Priority assessment after P1.6d-3

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.6e | Run and pin the deterministic reducer; expect zero eligible nominees and prohibit confirmation | Very high | Formalizes the decision independently of narrative interpretation | Do next |
| P1.7a | Diagnose Qwen/BM25 complementary wins and RRF displacement on screening IDs only | Very high | The representation is good; the next uncertainty is safe fusion, not another model | After zero-nominee lock |
| P1.7b | Freeze a fusion-v2 experiment only with a new split/protocol; do not retest Qwen under v1 | High | Avoid post-hoc tuning on the 29 observed cases and preserve confirmation evidence | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Retrieval quality now has a more direct next action | Deferred |
| P1.3 | Product controls | High but blocked | Raw dense gains do not authorize exposing a fused mode that regresses BM25 | Blocked |

### 2026-09-03 — P1.6e complete: zero-nominee decision locked; confirmation prohibited

- Ran the deterministic reducer only after both screening reports independently
  validated. Neither candidate passed every screening gate, so the formal
  result contains no eligible candidate, no nominee, and
  `confirmation_allowed=false`.
- Strengthened nomination validation to pin protocol and batch hashes plus the
  normalized SHA-256 of both immutable screening reports. The reducer decision
  is recomputed from the reports, and confirmation/product flags must remain
  false. A mutation test proves an attempted confirmation flip is rejected.
- Added both screening validators and nomination validation to Linux CI and the
  Windows release workflow. All three validators, 15 related tests, Ruff, and
  diff checks pass. The 12 internal-confirmation cases and 12 frozen semantic
  cases remain unexecuted and retain value for a future protocol.

### Priority assessment after P1.6e

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7a | Produce a read-only Qwen/BM25 complementarity and top-10 displacement diagnostic on screening reports only | Very high | Determines whether fusion, source gating, or metric variance caused the near miss | Do next |
| P1.7b | Freeze a fusion-v2 hypothesis only if P1.7a identifies a deterministic mechanism and a new tuning surface | High | Qwen cannot be retested on v1 screening or confirmation after rejection | Conditional |
| P1.7c | If no safe new evidence exists, prioritize real-user eval-set expansion before more retrieval code | High | Prevents optimizing to 29 known cases | Conditional |
| P1.4b | Optional external telemetry export | Medium | Does not collect query text by design and cannot replace a labeled eval set | Deferred |
| P1.3 | Product controls | High but blocked | Zero nominees means no dense product option | Blocked |

### 2026-09-03 — P1.7a complete: source gate contributes, chunk-level consensus is the main fusion failure

- Used the already-built Qwen index and identical frozen retrieval parameters
  for a read-only trace over the 29 tuning-allowed screening cases. The trace
  emitted only case IDs, qrel chunk/doc IDs, ranks, gate membership, and counts;
  no new configuration, confirmation case, or acceptance case was evaluated.
- Scoped: BM25 and Qwen both hit 18 cases, BM25 alone hit five, Qwen alone hit
  four, and neither hit two. Existing safe RRF captured only one of four
  Qwen-only cases and lost one BM25-only case. No relevant dense chunk was
  blocked by the source gate in scoped mode, so source filtering cannot explain
  the primary scoped failure.
- The scoped misses show exact-chunk consensus bias: qrels at dense ranks 6, 10,
  and 1 with allowed sources were fused to ranks 19, 19, and 16 when their
  sparse ranks were 33, 33, and 66. The successful Qwen-only case had dense rank
  1, sparse rank 22, and fused rank 8. Thus a strong dense-only passage is not
  reliably admitted unless sparse rank/overlap is already moderately helpful.
- Full-library: BM25 and Qwen both hit 13 cases, BM25 alone six, Qwen alone five,
  and neither five. Safe RRF captured only one of five Qwen-only cases, lost one
  BM25-only case, and had two improvements versus two regressions. The top-five
  source gate blocked three relevant dense chunks and sometimes reduced 80
  dense candidates to seven or 19.
- The clearest gate failure is `three-d-video-real-time-mechanism`: its qrel was
  BM25 rank 6 and Qwen rank 3, but its source was absent from BM25's first five;
  the dense hit was removed and the fused qrel fell to rank 13. The clearest
  consensus failure is `hsi-fsi-binary-pattern-cost`: source allowed, dense rank
  1, sparse rank 53, fused rank 13. The clearest sparse-protection failure is
  `sph-throughput-mechanism`: BM25 rank 10, dense rank 39, fused rank 16.
- Conclusion: removing only the source gate is insufficient because scoped mode
  has no relevant-source blocks yet still misses three of four dense-only wins.
  More RRF weight guessing also retains the exact-chunk consensus bias. The next
  candidate should protect the trusted sparse prefix while explicitly reserving
  a small tail quota for high-ranked dense passages that add new chunk evidence.

### Priority assessment after P1.7a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7b-1 | Freeze fusion protocol v2 with Qwen unchanged and at most three explicit tail-selection candidates | Very high | Uses the 29-case tuning surface; 12 confirmation cases stay untouched until one candidate passes | Do next |
| P1.7b-2 | Implement generic protected-prefix/tail fusion with identity/source audit fields and unit tests | Very high | Must not special-case observed qrels or case IDs | After protocol |
| P1.7b-3 | Screen the frozen fusion candidates once, nominate at most one, then conditionally confirm once | Very high | No additional model variants, prompts, weights, or chunking changes | Conditional |
| P1.7c | Expand labeled evidence queries from autonomous user-journey simulations | High | Required before any further protocol if v2 has no nominee | Conditional |
| P1.3 | Product controls | High but blocked | A diagnostic mechanism is not yet validated retrieval | Blocked |

### 2026-09-03 — P1.7b-1 complete: guarded fusion v2 protocol frozen before implementation

- Froze `retrieval_qwen_fusion_protocol_v2.json` before writing a new fusion
  implementation or measuring any variant. It pins the predecessor protocol,
  Qwen report, zero-nominee decision, 29 tuning IDs, 12 hidden confirmation
  IDs, and frozen semantic acceptance fixture by normalized SHA-256.
- The representation is immutable: same Qwen revision, safe-weight hash,
  query prefix, last-token pooling, 1,024 dimensions, parent chunks, product
  query processing, and existing index. No model, prompt, chunk, reranker, or
  index-build variant is allowed.
- Froze exactly three fusion candidates: the predecessor 2:1/k20 RRF without
  its source gate; a protected BM25 top-five plus one dense-novelty tail slot;
  and the same mechanism with two dense-novelty slots. Protected-tail variants
  use dense rank to retain the most semantically supported BM25 ranks 6–10,
  then append genuinely new dense chunks. They cannot inspect case IDs or
  relevance labels.
- All three must run together exactly once on the already-observed 29-case
  tuning partition. Eligibility requires exact Recall@5 and wrong-source@5
  preservation, no Recall@10/MRR/nDCG regression in either mode, and a strict
  Recall@10 gain in at least one mode. At most one eligible candidate may access
  the 12-case confirmation partition once; semantic acceptance remains later.
- Added a validator with mutation tests and wired it into Linux CI and the
  Windows release gate. The failure policy explicitly keeps product exposure
  disabled and requires new real-user evaluation evidence before any v3.

### Priority assessment after P1.7b-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7b-2 | Implement the generic protected-tail retriever plus identity, rank, fallback, and prefix-invariance tests | Very high | Implementation must match the frozen protocol without seeing evaluation labels | Do next |
| P1.7b-3 | Build a single cache-only runner, validate a dry synthetic path, then measure all three tuning candidates once | Very high | Confirmation remains inaccessible until deterministic nomination | After implementation |
| P1.7c | Expand real-user evidence-location queries if v2 yields no nominee | High | Protocol forbids further fusion tuning on the same 29 cases | Conditional |
| P1.3 | Product controls | High but blocked | Protocol definition alone is not retrieval-quality evidence | Blocked |

### 2026-09-03 — P1.7b-2 complete: label-agnostic protected-tail fusion implemented

- Added a small standalone `ProtectedDenseTailRetriever` rather than extending
  an already-large runtime module. It consumes only sparse and dense hit lists,
  preserves the first five sparse identities and their order, deduplicates
  chunk identities, and never receives case IDs, qrels, source labels, or
  fixture data.
- For ranks 6–10, it retains the sparse candidates with the best dense-rank
  support while keeping their original sparse order, then appends the frozen
  one or two genuinely novel dense candidates. If dense retrieval fails, it
  returns the original sparse order instead of surfacing an error to the user.
- Each returned hit carries internal channel, sparse-rank, dense-rank, strategy,
  and slot audit fields. This makes later displacement failures diagnosable
  without logging query or paper text and without changing product behavior.
- Added synthetic tests for prefix invariance, sparse-rank-10 protection,
  one/two novelty quotas, identity deduplication, dense failure fallback, short
  sparse results, and invalid safety parameters. The protocol validator,
  protected-tail tests, existing hybrid tests, Ruff, formatting, and diff checks
  pass (19 tests).

### Priority assessment after P1.7b-2

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7b-3a | Implement an immutable one-run tuning runner with exact control hashes, candidate membership, gate arithmetic, and deterministic nomination | Very high | Must make accidental confirmation access and partial candidate runs impossible | Do next |
| P1.7b-3b | Run a synthetic no-fixture smoke, then execute the single 29-case cache-only tuning measurement | Very high | Existing Qwen index avoids a new build; all three candidates share identical rankings | After runner validation |
| P1.7b-4 | Confirm the sole nominee once, or lock zero nominee without touching confirmation | Very high | Controlled entirely by the frozen gate result | Conditional |
| P1.7c | Expand real-user evidence-location queries if no candidate survives | High | Prevents a third round of tuning on known cases | Conditional |
| P1.3 | Product controls | High but blocked | The new retriever is benchmark-only and not wired into user settings | Blocked |

### 2026-09-03 — P1.7b-3a complete: immutable one-run fusion tuning runner ready

- Added a dedicated tuning runner that has no confirmation or download mode and
  refuses to overwrite its canonical report. It loads the existing verified
  Qwen index without rebuilding it and executes all three frozen candidates in
  one invocation over only the 29 tuning IDs.
- A shared in-memory query-embedding cache makes every fusion candidate consume
  the same Qwen query vector. This removes repeat-encoding noise and limits the
  run to exactly 29 unique query embeddings while retaining scoped and full-
  library retrieval behavior.
- Report validation pins protocol, fixture, predecessor Qwen result, candidate
  order, control behavior, representation, existing index build identity,
  tuning membership, gate arithmetic, deterministic nomination, confirmation
  isolation, and product-disabled state. A result fingerprint is derived only
  from metrics and per-case outcomes, excluding timing noise.
- Synthetic tests prove that equality without improvement cannot qualify, any
  MRR regression rejects a candidate, rejected candidates cannot be nominated,
  and a performance tie selects the smaller dense novelty quota. The new runner
  and fusion/protocol tests pass (14 tests), as do Ruff, formatting, and diff
  checks.

### Priority assessment after P1.7b-3a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7b-3b | Execute the only allowed cache-only 29-case tuning run, pin its fingerprint/metrics, and add CI validation | Very high | Implementation and candidate set are now frozen; report cannot be overwritten | Do next |
| P1.7b-4 | If exactly one nominee exists, run confirmation once; otherwise lock zero nominee and leave the hidden set untouched | Very high | No narrative override of gate arithmetic | Conditional |
| P1.7c | Add autonomous user-journey-derived evidence queries if fusion v2 fails | High | This is the predeclared next failure action | Conditional |
| P1.3 | Product controls | High but blocked | No user-visible retrieval option before confirmation and semantic acceptance | Blocked |

### 2026-09-03 — P1.7b-3b stopped safely: v2 execution invalidated on top-K contract mismatch

- Executed the canonical one-run tuning command once. It used the existing
  Qwen index, encoded exactly 29 unique tuning queries, wrote all three channels
  atomically, and did not access confirmation or semantic acceptance evidence.
- Post-run contract audit found that the fixture includes K=20 metrics, so the
  generic evaluator called each retriever with `top_k=20`. The protocol froze a
  top-10 fusion layout: candidate depth should have been 40, one-slot novelty at
  rank 10, and two-slot novelty at ranks 9–10. Actual depth was 80 and protected-
  tail novelty landed at ranks 11 or 11–12. This is an implementation/protocol
  mismatch, not a quality result that may promote.
- The invalid report is preserved unchanged and pinned by normalized file hash
  plus outcome fingerprint. Diagnostic-only output showed ungated RRF with
  +0.000000 scoped and +0.051724 full-library Recall@10, while both tail quotas
  were equal to BM25 at Recall@10. None of these values authorize confirmation.
- Added an explicit request-depth invariant to report validation and a separate
  invalidation artifact/validator. Effective outcome is no nominee, no
  confirmation, no retry under v2, no semantic replay, and no product exposure.
  CI and release gates validate that this invalid run can never be promoted.
- The incident also improves the next design: retrieval ranking depth and metric
  reporting depth must be separate contracts. A future protocol must ask fusion
  for its frozen ranking depth and calculate diagnostic K=20 only from a
  separately specified continuation policy; it cannot silently pass max(K).

### Priority assessment after invalidating P1.7b-3b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7c-1 | Freeze an autonomous user-journey query expansion protocol and evidence-labeling rules before adding cases | Very high | Predeclared v2 failure action; creates genuinely new evidence instead of reusing the 29 observed cases | Do next |
| P1.7c-2 | Simulate library-reading questions, add source/chunk qrels with independent structural checks, and reserve a new holdout | Very high | Must remain evidence-location tasks, never writing assistance | After protocol |
| P1.7d | Only after new evidence exists, consider fusion v3 with separate ranking/evaluation-depth contracts | High | No v2 parameter or candidate may be replayed as if unseen | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Useful operationally but lower direct evidence-retrieval value | Deferred |
| P1.3 | Product controls | High but blocked | Invalid tuning evidence cannot unlock a user-facing dense option | Blocked |

### 2026-09-03 — P1.7c-1 complete: new user-journey evidence protocol frozen

- Froze an 18-case local-library retrieval expansion before authoring any new
  question or inspecting chunks for qrels. It references the previous 41-case
  development fixture, 12-case semantic fixture, and fusion-v2 invalidation by
  hash, while forbidding reuse of case IDs, exact queries, or in-place qrel
  extension.
- Coverage is fixed at 12 Chinese and six English user-style questions across
  four mechanism, four limitation/boundary, four experiment/metric, four cross-
  paper comparison, and two equation/figure/table location journeys. At least
  12 primary documents must be represented, with no more than two primary cases
  per document.
- The product boundary is explicit: every question must help read or locate
  evidence in papers already in the library. Drafting, rewriting, review-
  narrative generation, submission preparation, and invented external evidence
  are rejected. External web/model knowledge cannot be used for qrel labels.
- Labeling is chunk-level and rank-independent with source page, text hash,
  evidence role, scope, grade, and cross-paper coverage checks. Retrieval cannot
  run until all 18 questions and qrels pass the audit and are frozen.
- A predeclared hash split will reserve six cases (four Chinese, two English) as
  a one-run holdout and leave 12 for future calibration. Product BM25 must be
  measured first on the frozen fixture before any fusion-v3 protocol exists.
  Added protocol mutation tests and CI/release validation.

### Priority assessment after P1.7c-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7c-2a | Create the 18-case authoring manifest from paper-level reading journeys, then inspect local chunks and record qrels without running retrieval | Very high | Directly expands realistic evidence-location coverage | Do next |
| P1.7c-2b | Build a strict fixture validator for corpus identity, hashes, scope, distributions, previous-case leakage, and deterministic holdout | Very high | Prevents convenient or malformed labels from entering evaluation | Alongside labeling |
| P1.7c-3 | Freeze the audited fixture and run the required BM25 baseline before any new fusion design | Very high | Establishes the actual user problem before selecting an algorithm | After qrel audit |
| P1.7d | Consider fusion v3 only from the new calibration partition | High | Holdout remains unavailable until a candidate is frozen | Conditional |
| P1.3 | Product controls | High but blocked | New labels are evaluation evidence, not a shipped feature | Blocked |

### 2026-09-03 — P1.7c-2a complete: 18 questions frozen before qrel inspection

- Authored and froze all 18 user questions from paper titles and reading-journey
  slots before inspecting chunks or running any retriever. The manifest records
  `retriever_executed=false` and `qrel_chunk_inspection_started=false`; its hash
  will become the source identity for the labeled fixture.
- The fixed distribution is 12 Chinese/six English with exactly the five
  predeclared journey types. Sixteen distinct primary documents are represented,
  no document has more than two primary cases, and all four comparison questions
  contain two explicit library-paper scopes.
- Eight questions intentionally cover papers that were only hard negatives in
  the prior fixture (ISTA-Net, learned primal-dual, large-image dual-scale SPI,
  real-time SPI video, Fashion-MNIST, historical Japanese recognition,
  compensation, and computational hyperspectral imaging). This increases corpus
  breadth rather than relabeling known wins and losses.
- Every question asks the system to locate mechanisms, limits, experiment facts,
  comparisons, or a figure/equation. Writing, polishing, manuscript, and
  submission intents are structurally rejected. Manifest distribution,
  previous-query independence, scope, and anti-writing mutation tests pass.

### Priority assessment after P1.7c-2a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7c-2b | Inspect registered local chunks directly, attach text hashes/pages/evidence roles, and drop any question whose evidence is ambiguous | Very high | Labels must not depend on BM25 or dense ranks | Do next |
| P1.7c-2c | Validate all 18 qrels against the committed 31-document corpus and materialize the deterministic 12/6 split | Very high | Makes future calibration and holdout boundaries executable | After labeling |
| P1.7c-3 | Freeze and measure product BM25 baseline | Very high | First allowed retrieval run on the new fixture | After audit |
| P1.7d | Freeze fusion v3 only after the baseline failure taxonomy exists | High | Avoids designing another fusion from intuition alone | Conditional |
| P1.3 | Product controls | High but blocked | Evaluation expansion does not change current user behavior | Blocked |

### 2026-09-03 — P1.7c-2b/2c complete: rank-independent qrels audited and frozen

- Inspected only the registered local paper chunks after the question manifest
  was frozen; BM25, dense retrieval, fusion, reranking, and answer generation
  were not run during labeling. The resulting fixture contains 18 questions,
  35 evidence-ready chunk labels, 15 distinct primary documents, and the same
  committed 31-document/2,251-chunk corpus identity.
- Four questions with unsupported premises were dropped and transparently
  replaced before retrieval: per-factor compensation assumptions, learned
  sensing calibration in computational hyperspectral imaging, interferometric
  ports in the S2ISM comparison, and a nonexistent bandwidth-time equation.
  Replacement lineage and reasons remain in the qrel registry instead of
  converting incorrect assumptions into convenient relevance labels.
- Every label now pins its logical document, local chunk identity, page range,
  evidence role, grade, evidence contract, and exact text hash. Cross-paper
  cases must cover every scoped paper, and English questions are checked for
  excessive contiguous copying from their labeled evidence.
- Materialized the predeclared deterministic split: 12 calibration cases and
  six holdout cases (four Chinese/two English). The holdout is explicitly
  unavailable for tuning. Its normalized-newline SHA-256 is
  `6463cf63ba1853fc8704eead89de0ee36061481613621b1fa72ea4b7fe5429ca`.
- Added a deterministic source-to-fixture builder, exact projection validator,
  mutation tests, and Linux CI/Windows release checks. The 13 protocol,
  manifest, fixture, anti-rank-leakage, hash, scope, and split tests pass; both
  fixture regeneration and standalone validation pass.

### Priority assessment after P1.7c-2b/2c

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7c-3a | Run and freeze the current product BM25 baseline on all 18 cases, with calibration/holdout aggregates and failure taxonomy | Very high | This is the first permitted retrieval run and reveals whether dense retrieval is solving real reading failures | Do next |
| P1.7c-3b | Validate baseline identity/metric arithmetic and prevent later overwrites or use of holdout case details for candidate design | Very high | Baseline must be trustworthy before another fusion protocol | Alongside baseline |
| P1.7d-1 | If calibration shows semantic misses, freeze fusion v3 with ranking depth separate from reporting depth | High | Candidate design may use calibration aggregates/cases only; holdout remains sealed | Conditional |
| P1.4b | Add an optional OpenTelemetry exporter behind the existing local observability contract | Medium | Valuable for operations, but lower direct impact than locating missing paper evidence | Deferred |
| P1.3 | Product controls | High but blocked | No user-visible dense option until a candidate passes calibration, one-run holdout, and semantic acceptance | Blocked |

### 2026-09-03 — P1.7c-3 complete: product BM25 baseline measured and locked

- Ran the first permitted retrieval measurement on all 18 frozen questions.
  The report includes raw BM25 and the current product query pipeline in both
  paper-scoped and full-library modes. Its normalized SHA-256 is
  `28748a9d590a637dcd8e2847f2ba9efc437b069ee4769feb95f3821986576fca`;
  deterministic result fingerprint is
  `d693926a5fbe959aee66d0bd57f0743daf58b8c55c685f9ea822c36c5689933e`.
- Product BM25 reaches only 0.231481 scoped Recall@10 and 0.157407 full-library
  Recall@10. Even at K=20 it reaches 0.481482 scoped and 0.259259 full-library.
  The new fixture therefore exposes a retrieval-recall problem, not merely a
  citation display or answer-formatting problem.
- On the 12-case calibration partition, product BM25 has 0.222222 scoped and
  0.152778 full-library Recall@10. Five scoped cases and seven full-library
  cases have no relevant passage in the first 20. Product query processing
  improves Recall@10 for only three scoped and one full-library calibration
  cases; the rest are unchanged, with no regression.
- Holdout results are recorded only as six-case aggregates in the outcome
  artifact and are not used in the failure taxonomy or future candidate design.
  This preserves a one-run decision surface after a candidate is frozen.
- Added baseline identity, compact-payload, case-membership, reporting-depth,
  source-safety, and metric-arithmetic checks. A deterministic outcome builder
  separates BM25 ranking depth 20 from its reporting depths, and explicitly
  requires future fusion candidates to use separate ranking/evaluation-depth
  fields. Five mutation tests, Ruff, the baseline validator, and outcome
  regeneration check pass; CI and release gates now enforce all of them.

### Priority assessment after P1.7c-3

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-1 | Freeze a calibration-only Qwen dense diagnostic using the already selected model/index and no fusion parameters | Very high | Determines whether semantic retrieval supplies complementary evidence on realistic reading tasks before another fusion design | Do next |
| P1.7d-2 | Run the diagnostic once and classify dense-only wins, BM25-only protections, misses, source errors, and relevant-rank depth | Very high | Only 12 calibration cases may inform design; holdout remains inaccessible | After protocol |
| P1.7d-3 | If complementarity exists, freeze fusion v3 with explicit ranking depth, candidate depth, continuation policy, and one-run holdout gate | High | Prevents another top-K contract incident and label-driven parameter search | Conditional |
| P1.4b | Optional OpenTelemetry exporter behind the local privacy contract | Medium | Operational benefit is real, but it cannot repair current evidence recall | Deferred |
| P1.3 | Product controls | High but blocked | A user choice is useful only after the dense path proves safer and more effective | Blocked |

### 2026-09-03 — P1.7d-1 complete: calibration-only dense diagnostic frozen

- Froze the diagnostic protocol before implementing its runner or encoding any
  of the new questions. It pins the 18-case fixture, BM25 baseline/outcome,
  prior Qwen report, candidate batch, and invalidated fusion-v2 artifact by
  normalized SHA-256.
- Only the 12 predeclared calibration IDs may be loaded. The six holdout cases
  and the older semantic acceptance fixture are inaccessible to this run. No
  fusion, reranker, model/prompt variant, index build, or product behavior is
  permitted.
- Reuses the exact Qwen3-Embedding-0.6B revision, last-token pooling, 1,024-
  dimensional query contract, and existing 2,251-chunk index. Downloads and
  remote code remain disabled; the index must report the predecessor build ID,
  provider identity, corpus fingerprint, and byte size without rebuilding.
- Ranking depth is frozen independently at 20; reported depths are 5, 10, and
  20. The diagnostic will compare the frozen product BM25 control with raw
  dense retrieval in scoped/full-library modes and classify calibration cases
  as both-hit, BM25-only, dense-only, or neither at K=10.
- A fusion-v3 protocol may be designed only if dense produces at least two
  dense-only cases in either mode and a strict aggregate Recall@10 gain in at
  least one mode. This permits algorithm design, not product exposure. Five
  mutation tests, validator, formatting, and Ruff checks pass; CI/release gates
  enforce the frozen boundary.

### Priority assessment after P1.7d-1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-2a | Implement a cache-only diagnostic runner that refuses holdout IDs, reuses the existing index, and validates a synthetic execution path | Very high | Makes the one allowed measurement auditable before loading the model | Do next |
| P1.7d-2b | Execute the 12-query dense diagnostic once and lock its complementarity result | Very high | No candidate parameters exist, so the run cannot silently become tuning | After runner validation |
| P1.7d-3 | Freeze fusion v3 only if the predeclared complementarity condition passes | High | Must derive structure from calibration failure classes while keeping the holdout sealed | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Defer until direct retrieval-quality evidence is resolved | Deferred |
| P1.3 | Product controls | High but blocked | Diagnostic success alone is insufficient for a user-facing switch | Blocked |

### 2026-09-03 — P1.7d-2a complete: cache-only dense diagnostic runner validated

- Implemented a dedicated runner that derives exactly the 12 calibration cases
  from the frozen fixture and raises before model loading if membership or count
  differs. It has no holdout, fusion, reranker, index-build, or download mode.
- Product BM25 is not remeasured: its per-case calibration projection is
  deterministically derived from the pinned baseline. Dense retrieval requests
  exactly rank depth 20 and reports K=5/10/20, so evaluator reporting depth can
  no longer mutate a fusion layout.
- The existing Qwen index must be ready under the exact predecessor provider
  identity. Query embeddings are cached by question across scoped and full-
  library modes; a valid report must contain exactly 12 unique encodings and
  the unchanged index build, fingerprint, chunk count, and byte size.
- Result validation recomputes product control projection, case membership,
  complementarity classes, rank buckets, aggregate gains, design eligibility,
  and a latency-independent outcome fingerprint. The canonical report cannot
  be overwritten after its single run.
- Five runner tests cover unique-win eligibility, insufficient unique wins,
  holdout exclusion, partition-count failure, and a complete synthetic report
  validation path. Together with five protocol tests, formatting and Ruff all
  pass.

### Priority assessment after P1.7d-2a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-2b | Execute the sole 12-query Qwen diagnostic using the verified local model/index; pin and gate the report | Very high | Runner now prevents accidental parameter search or holdout access | Do next |
| P1.7d-3 | Freeze fusion v3 only if both predeclared complementarity checks pass | High | Failure means dense is not yet useful enough on realistic reading tasks | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Remains secondary to retrieval correctness | Deferred |
| P1.3 | Product controls | High but blocked | No direct exposure from diagnostic evidence | Blocked |

### 2026-09-03 — P1.7d-2b complete: Qwen dense complementarity confirmed, not product-ready

- Executed the single permitted diagnostic with the verified local Qwen model
  and existing index. Exactly 12 unique calibration questions were encoded and
  shared across scoped/full-library retrieval. No holdout, fusion, reranker,
  download, index build, or product path was accessed.
- Scoped dense Recall@10 is 0.222222, exactly equal to product BM25. Dense has
  one unique hit, BM25 has two unique hits, both hit four, and neither hits five.
  Dense Recall@5, MRR, nDCG@10, and Recall@20 are lower, so replacing BM25 with
  dense would make the reading experience worse despite the complementary hit.
- Full-library dense Recall@10 is 0.222222 versus product BM25 0.152778, a
  +0.069444 gain. There are two dense-only hits, one BM25-only hit, three shared
  hits, and six misses. Dense also lowers wrong-source@5 from 0.383333 to
  0.216667, but MRR is slightly lower (0.173016 versus 0.178513).
- The predeclared fusion-v3 design condition passes: at least two dense-only
  cases exist in one mode and aggregate Recall@10 strictly improves in one
  mode. This proves complementarity only; `product_eligibility=false` remains
  enforced.
- Pinned result fingerprint
  `bb67c5f33daf4ce17d5f9d92d8237be41453d435fd367ff51a90a81f70985cb6`
  and normalized report SHA-256
  `ca0be8934730ac8088af8666682ece2fe1db12b67070400ecbfadb48d1f7b65a`.
  The report validator and 20 related protocol/fixture/baseline/diagnostic tests
  pass; CI and release gates now prevent result drift.

### Priority assessment after P1.7d-2b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-3a | Freeze one fusion-v3 candidate: protect BM25 top five, retain dense-supported sparse tail, reserve two dense novelty slots at ranks 9–10 | Very high | Two slots are justified by the full-library relevant dense rank-2 case; one slot would systematically miss it | Do next |
| P1.7d-3b | Implement an evaluator with fusion ranking depth 10, dense candidate depth 40, reporting depths 5/10, and a separate diagnostic continuation policy | Very high | Directly fixes the invalid v2 top-K coupling | After protocol |
| P1.7d-3c | Run calibration once; access the six-case holdout once only if all recall, rank-quality, contract, and source-safety gates pass | Very high | One candidate and one conditional holdout prevent parameter fishing | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Retrieval correctness still has higher user impact | Deferred |
| P1.3 | Product controls | High but blocked | Requires successful calibration, holdout, and older semantic acceptance gates | Blocked |

### 2026-09-03 — P1.7d-3a complete: one-candidate fusion-v3 protocol frozen

- Froze exactly one candidate before writing the selection runner: protected
  BM25 ranks 1–5, three dense-supported survivors from BM25 ranks 6–10, and two
  genuinely novel dense passages at output ranks 9–10. Sparse/dense candidate
  depth is exactly 40; dense failure returns the original BM25 top 10.
- The two novelty slots follow the observed calibration structure: one full-
  library dense-only relevant chunk was at dense rank 2, while known BM25-only
  evidence was in the protected top five. No alternative quota, weight, model,
  prompt, source gate, or reranker is admitted into this protocol.
- Selection ranking depth is exactly 10 and selection metrics are only K=5/10.
  K=20 is explicitly forbidden during selection and would require a separate
  frozen continuation contract later. The evaluator may not derive the fusion
  request depth from an unrelated maximum metric K.
- Calibration requires exact Recall@5 and wrong-source@5 preservation plus no
  regression in Recall@10, MRR, nDCG@10, contract Recall@10, and wrong-source@10
  in both modes, with a strict Recall@10 gain in at least one. Holdout can run
  once only after calibration passes, using the identical candidate and gates.
- Combined 18-case evidence must remain non-regressing and improve Recall@10 by
  at least 0.02 in one mode before the older semantic acceptance set can run.
  Semantic acceptance still requires +0.15 Recall@10 in both modes and prefix/
  source-safety preservation. Product exposure remains disabled throughout.
- Added six protocol mutation tests and CI/release validation for candidate
  count, depth separation, holdout isolation, source safety, and product lock.

### Priority assessment after P1.7d-3a

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-3b | Implement one-command calibration with automatic conditional holdout, exact control projection, gate recomputation, and immutable report | Very high | Candidate cannot be changed between seeing calibration and opening holdout | Do next |
| P1.7d-3c | Run the single selection command and lock either rejection or combined eligibility | Very high | No manual decision point or retry | After runner validation |
| P1.7d-4 | Run semantic acceptance once only if combined eligibility passes | High | Last retrieval-quality gate before considering product shadow integration | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Still secondary to completing retrieval selection | Deferred |
| P1.3 | Product controls | High but blocked | Protocol alone changes no user behavior | Blocked |

### 2026-09-03 — P1.7d-3b complete: conditional-holdout fusion runner validated

- Implemented one command that runs calibration first and, without a manual or
  configuration boundary, opens the holdout only when every frozen calibration
  gate passes. A failing calibration writes no holdout or combined result.
- Product control is re-evaluated at the exact selection ranking depth 10, then
  checked case-by-case against the frozen BM25 baseline projection for Recall@5,
  Recall@10, MRR, nDCG@10, contract Recall@10, wrong-source@5, and top-10 first-
  relevant rank. This catches query-pipeline or depth-dependent control drift.
- Fusion receives only top-K=10 while its sparse and dense sub-retrievers receive
  exactly 40 candidates through the frozen multiplier. Reports include K=5/10
  only, including wrong-source@10; no unrelated K=20 can alter output layout.
- The report validator recomputes calibration/holdout/combined gates, exact
  partition order, conditional access, product projection, cache count, index
  identity, decision, and latency-independent fingerprint. A passing calibration
  must encode exactly 18 total unique questions; a rejection must encode 12.
- Seven selection-runner tests plus the six protocol and seven protected-tail
  tests pass. They cover metric/source gates, minimum combined gain, strict
  partition isolation, stop-before-holdout behavior, and a complete synthetic
  rejected-report validation path; Ruff and formatting pass.

### Priority assessment after P1.7d-3b

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-3c | Execute the only fusion-v3 selection command and pin its rejection or combined eligibility | Very high | All selection behavior is now frozen and mechanically checked | Do next |
| P1.7d-4 | If eligible, execute semantic acceptance once; otherwise diagnose without trying a second fusion | High | Prevents post-result candidate search | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Lower priority until fusion selection finishes | Deferred |
| P1.3 | Product controls | High but blocked | Still no user-visible behavior change | Blocked |

### 2026-09-03 — P1.7d-3c execution incident: v3 closed before any candidate result

- The first v3 command completed its calibration retrieval but stopped in the
  pre-gate product-control projection check. It wrote no canonical report,
  exposed no candidate metric/gate result, and never accessed holdout, semantic
  acceptance, or product behavior.
- Root cause: the frozen BM25 baseline ranked to depth 20, where two calibration
  qrels first appeared at ranks 13 and 17 and therefore contributed 1/13 and
  1/17 to MRR. The v3 selection correctly ranked only to depth 10, but its
  validator projected first-relevant rank to `null` without also projecting
  those per-case MRR values to zero.
- Preserved the failed command, attempted protocol/runner hashes, exact failure
  stage, 12-query boundary, missing report, and no-promotion decision in a
  separate invalidation artifact. v3 is closed: it cannot open holdout, claim a
  selection result, run semantic acceptance, or enable product behavior.
- Froze v3.1 as one infrastructure-replacement run. It hashes every inherited
  v3 partition, representation, candidate, depth, product-control, calibration,
  holdout, combined, semantic, and safety section. All are byte-for-byte
  unchanged; the sole correction is: when baseline first-relevant rank is null
  or greater than selection depth 10, expected selection MRR is zero.
- v3.1 permits exactly one replacement execution with zero candidate,
  representation, index, ranking/reporting depth, gate, or holdout-policy
  changes. Four invalidation tests, four replacement-protocol tests, seven
  runner tests, validators, Ruff, and formatting pass; CI/release enforce the
  historical invalidation and corrected boundary.

### Priority assessment after freezing v3.1

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-3d | Execute the single unchanged-candidate v3.1 replacement; automatically stop or run holdout | Very high | No fusion outcome was observed before the sole infrastructure correction | Do next |
| P1.7d-4 | Run semantic acceptance only if v3.1 combined gate passes | High | Remains the last retrieval-quality decision surface | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Defer until selection has a valid result | Deferred |
| P1.3 | Product controls | High but blocked | Neither an invalid execution nor a replacement protocol changes the product | Blocked |

### 2026-09-03 — P1.7d-3d complete: v3.1 passes calibration, holdout, and combined gates

- The single unchanged-candidate replacement executed successfully. Calibration
  passed every gate, which automatically opened the six-case holdout in the
  same process with the same candidate, model, index, depths, and cache. Exactly
  18 unique questions were encoded; semantic acceptance and product remained
  untouched.
- Calibration Recall@10 improved by +0.111111 in both scoped and full-library
  modes. Holdout independently improved by +0.083333 scoped and +0.166666 full-
  library. Both partitions exactly preserve Recall@5 and wrong-source@5 and do
  not regress MRR, nDCG@10, contract Recall@10, or wrong-source@10.
- Across all 18 cases, scoped Recall@10 rises from 0.231481 to 0.333333
  (+0.101852); full-library rises from 0.157407 to 0.287037 (+0.129630).
  Scoped MRR rises 0.202778→0.215123 and nDCG@10 0.145849→0.174677. Full-
  library MRR rises 0.126852→0.150926, nDCG@10 0.103993→0.142668, and wrong-
  source@10 falls 0.388889→0.327778.
- The combined gate passes and the sole decision is
  `eligible_for_semantic_acceptance`; this is not product eligibility. Pinned
  result fingerprint
  `55b89302358f0f4ea18a73b422f0abd554e75b01f3e5a68da84874d2593f1284`
  and normalized report SHA-256
  `559d0ced289dad76095631fd10fa01c4f7e9cc0d6e514e150eac76d90b1bb36c`.
- Report validation, Ruff, and 29 fusion/protocol/invalidation/retriever tests
  pass. CI and release gates pin the conditional holdout boundary, all metric
  arithmetic, the selection decision, and the complete outcome fingerprint.

### Priority assessment after P1.7d-3d

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-4a | Implement a one-run semantic acceptance evaluator for the unchanged v3.1 candidate against the frozen paraphrase fixture | Very high | Confirms the gain is semantic, not only tied to the new user-journey wording | Do next |
| P1.7d-4b | Run once and lock pass/reject; no parameter or candidate changes | Very high | Required before any product shadow integration | After evaluator validation |
| P1.7e | If semantic passes, add product-disabled shadow execution/observability before exposing a setting | High | Measures latency, fallback, and source safety in the real pipeline without changing results users see | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Can follow product shadow instrumentation | Deferred |
| P1.3 | Product controls | High but blocked | Selection evidence alone is not enough for ordinary users | Blocked |

### 2026-09-03 — P1.7d-4a protocol complete: semantic acceptance frozen

- Froze a child acceptance protocol after the combined selection passed and
  before implementing or running its evaluator. It pins the v3.1 replacement
  protocol, valid selection report, 12-case semantic registry, earlier semantic
  BM25/E5 benchmark, and protected-tail implementation by hash.
- The candidate is inherited unchanged: Qwen provider identity, existing index,
  BM25 top-five protection, three supported sparse-tail survivors, two dense
  novelty slots, output depth 10, and candidate depth 40. No model, prompt,
  quota, fusion, reranker, or index variant exists.
- The current product BM25 control must reproduce the predecessor semantic
  Recall@5, Recall@10, and wrong-source@5 in scoped and full-library modes.
  This prevents a product-query drift from being credited to the candidate.
- Acceptance remains the previously declared threshold: Recall@10 gain of at
  least 0.15 versus product in both modes, with exact Recall@5 and wrong-
  source@5 preservation. The run count is one; failure forbids retry, candidate
  changes, product shadow, and ordinary-user exposure.
- Four protocol mutation tests, validator, Ruff, and formatting pass; CI/release
  now enforce the semantic evidence, threshold, candidate, and failure policy.

### Priority assessment after semantic protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-4b | Implement and synthetically validate the one-run semantic evaluator | Very high | Must filter the existing 31-document index to the frozen 10-document/1,673-chunk corpus without rebuilding | Do next |
| P1.7d-4c | Execute once and lock pass/reject | Very high | Final offline retrieval-quality gate | After evaluator validation |
| P1.7e | Product-disabled shadow only on pass | High | Operational proof before any user option | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Can be combined with shadow instrumentation later | Deferred |
| P1.3 | Product controls | High but blocked | Semantic protocol alone is not eligibility | Blocked |

### 2026-09-03 — P1.7d-4b complete: semantic evaluator validated

- Implemented a cache-only semantic evaluator that projects the frozen 12
  paraphrases onto their inherited qrels, filters the existing 31-document Qwen
  index down to the exact predecessor 10-document/1,673-chunk corpus, and never
  rebuilds embeddings.
- Product BM25 and the unchanged protected-tail candidate both rank to exactly
  10 with K=5/10 reporting. Product Recall@5, Recall@10, and wrong-source@5 must
  match the independently committed predecessor semantic report before the
  candidate result is accepted.
- Validation recomputes case membership, corpus/index identity, query-cache
  count, product control, per-mode 0.15 gain, prefix/source preservation,
  latency-independent result fingerprint, and pass/reject shadow decision. The
  canonical report cannot be overwritten.
- Five evaluator tests cover exact threshold acceptance, prefix/source
  rejection, corpus filtering, missing frozen chunks, and a complete synthetic
  rejected-report validation path. Together with four protocol and existing
  semantic-fixture tests, 13 checks pass; Ruff and formatting pass.

### Priority assessment after semantic evaluator validation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-4c | Execute the sole semantic acceptance run and pin pass/reject | Very high | Candidate and acceptance surface are fully frozen | Do next |
| P1.7e | If passed, define product-disabled shadow protocol with latency/fallback/source telemetry | High | Real-pipeline proof without changing user-visible answers | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Shadow can first use the existing local privacy-preserving contract | Deferred |
| P1.3 | Product controls | High but blocked | Only a semantic pass can unblock shadow, not direct exposure | Blocked |

### 2026-09-03 — P1.7d-4c preflight incident corrected before measurement

- The first semantic command stopped before model loading: the requested 1,673
  chunks were absent from the current 31-document product database. It created
  zero document/query embeddings, observed no candidate result, wrote no report,
  and changed no product state.
- The earlier 12-case semantic slice intentionally belongs to an isolated
  10-document blind-transfer database outside the product corpus. Its chunks
  still exist locally and match the frozen qrel corpus, but the selected Qwen
  index was built only for the newer 31-document development corpus. Filtering
  that index can never produce the semantic corpus.
- Closed the incorrect child protocol with a preflight invalidation artifact.
  It forbids retry, quality conclusions, shadow access, and candidate/gate
  changes. The attempted protocol/runner hashes, zero-query boundary, and exact
  root cause are pinned.
- Froze semantic protocol v2 with the exact inherited candidate, eligibility,
  0.15-per-mode gate, and failure policy. Its sole infrastructure correction is
  one dedicated Qwen index build from the isolated 10-document/1,673-chunk
  database before any query encoding. The product Qwen index must remain byte-
  identical before/after.
- Updated the evaluator to require the isolated database argument, validate all
  qrel chunk IDs/text hashes/logical documents before loading the model, build
  only the dedicated index with downloads disabled, then encode exactly 12
  queries. It reports both the dedicated build identity and product-index tree
  hashes. Twelve v2/invalidation/evaluator tests, Ruff, formatting, and protocol
  validators pass; CI/release enforce both the invalid attempt and replacement.

### Priority assessment after semantic v2 correction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-4d | Run a read-only isolated-corpus preflight, then execute the sole build-plus-query semantic v2 command | Very high | No semantic query has yet been encoded; all candidate/gate values remain frozen | Do next |
| P1.7e | Product-disabled shadow only on a semantic pass | High | Still no ordinary-user exposure | Conditional |
| P1.4b | Optional OpenTelemetry exporter | Medium | Can follow a valid semantic decision | Deferred |
| P1.3 | Product controls | High but blocked | Preflight corrections are not product evidence | Blocked |

### 2026-09-03 — P1.7d-4d read-only preflight complete

- Loaded the frozen semantic projection against the isolated blind-transfer
  database and found exactly 1,673 expected chunks with zero corpus, qrel,
  logical-document, or text-hash validation errors.
- Confirmed that the dedicated semantic Qwen index path does not exist. The
  preflight therefore wrote no embeddings, loaded no model, encoded no query,
  and did not consume the protocol's sole build-plus-query execution.
- An initial read-only probe omitted the fixture loader's required `repo_root`
  argument and stopped at the call boundary. It made no filesystem changes;
  the corrected probe above is the predeclared eligibility result.

### Priority assessment after semantic corpus preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7d-4e | Execute the sole semantic v2 index build and 12-query acceptance run; lock its result without tuning | Very high | Corpus and clean index destination are now verified | Do now |
| P1.7e | If passed, define product-disabled shadow execution with source, fallback, and latency telemetry | High | Needed before any user-visible retrieval option | Conditional |
| P1.4b | Add optional OpenTelemetry export after local shadow fields are stable | Medium | Avoid premature telemetry schema coupling | Deferred |
| P1.3 | Product controls | High but blocked | Requires semantic pass plus operational shadow evidence | Blocked |

### 2026-09-03 — P1.7d-4e complete: semantic acceptance passes once and is locked

- The sole semantic-v2 process built the dedicated Qwen index from the frozen
  isolated corpus, encoded exactly 12 paraphrased queries, and completed with no
  retry or candidate/gate change. The index contains 10 documents and 1,673
  chunks, is `ready`, and records corpus fingerprint
  `19dd55b81535593219b0fdb8744b6c74cdfe51d68c3567dfd822780be927ec7a`.
- Scoped Recall@10 improves 0.375000→0.666667 (+0.291667); full-library
  Recall@10 improves 0.291667→0.541667 (+0.250000). Recall@5 is preserved at
  0.250000 scoped and 0.166667 full-library, while wrong-source@5 is preserved
  at 0.000000 and 0.183333 respectively. Full-library wrong-source@10 also
  falls from 0.200000 to 0.158333.
- MRR and nDCG@10 improve in both modes: scoped MRR 0.111442→0.146858 and
  nDCG@10 0.171936→0.260787; full-library MRR 0.187831→0.214914 and nDCG@10
  0.187177→0.261644. The declared semantic gate therefore passes without
  trading away the protected top-five evidence boundary.
- Product index tree SHA-256 is byte-identical before and after
  (`c75e16595f8d96950846721da0c27b5c530a1f06a653d52fc326b0df7c7feaae`).
  The decision is `passed_for_product_shadow`; product retrieval remains
  disabled and no ordinary-user surface is exposed.
- Pinned result fingerprint
  `a3c81c2805cd887011219b68a33e6bbf3ed8688f5209b8483263e1a41ce2847a`,
  dedicated build ID `21e9718aa59f483988f3e88571a3c01c`, and report
  SHA-256 `5862818541ddff4ac70169e6183404abb3d9f184ec75db5e08b908e5084fb432`.
  Report validation, Ruff/format, and 12 semantic protocol, invalidation, and
  evaluator tests pass. CI and release now validate the locked report.

### Priority assessment after semantic acceptance

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7e-1 | Freeze a product-disabled shadow contract and integrate fail-open comparison telemetry into the real retrieval pipeline | Very high | Offline quality now passes; runtime latency, availability, and source-boundary behavior remain unknown | Do next |
| P1.7e-2 | Exercise the shadow path end to end with controlled local queries and lock an operational outcome | Very high | Required before a user can opt into dense retrieval | After integration gates |
| P1.3 | Offer an explicit sparse/dense retrieval choice only if shadow reliability and latency pass | High | Honors user control without silently changing evidence | Conditional |
| P1.4b | Export the stable local observability schema through optional OpenTelemetry | Medium | Valuable after shadow event fields stop changing | Deferred |
| P2 architecture | Continue extracting large runtime/UI modules behind characterization tests | High | Important maintainability work, but does not replace the current evidence-quality proof | Resume after shadow decision |

### 2026-09-03 — P1.7e-1a complete: product-disabled shadow contract frozen

- Added a pre-implementation shadow protocol pinned to the accepted semantic
  report, v3.1 fusion protocol, result fingerprint, Qwen provider identity, and
  unchanged protected-tail implementation. Five mutation tests and the protocol
  validator pass; CI/release now enforce it.
- Activation requires all three explicit local conditions: the exact value
  `KB_DENSE_SHADOW_ENABLED=1`, an explicit existing
  `KB_DENSE_SHADOW_INDEX_DIR`, and `KB_LOCAL_OBSERVABILITY_PATH`. The default is
  off. No index build, network access, raw-content telemetry, user-facing
  surface, answer evidence change, or product reranking is allowed.
- Runtime behavior is frozen as a single-worker, skip-when-busy, asynchronous
  fail-open comparison over a deep copy of the already materialized product
  candidate ranking. The accepted 5/3/2 protected-tail composition and K=10,
  candidate-depth=40 boundary are unchanged.
- Before seeing operational measurements, the gate is fixed at at least 18
  completed controlled queries, 100% product-output immutability, 100% protected
  prefix preservation, zero scoped source escapes, warm shadow p95 ≤250 ms,
  foreground submission p95 ≤5 ms, plus explicit fail-open and privacy replay
  tests. Passing cannot enable the product; it only permits designing the
  opt-in control.

### Priority assessment after shadow protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7e-1b | Implement the bounded async runner, cache-only provider/index preflight, content-free event, and real chat retrieval hook | Very high | Must prove the contract without touching visible evidence | Do next |
| P1.7e-2 | Run the frozen 18-query operational replay once and lock its outcome | Very high | Runtime latency/reliability are still unknown | After implementation tests |
| P1.3 | Design an explicit user opt-in only after operational pass | High | No silent rollout | Conditional |
| P2 architecture | Resume high-risk module extraction if operational replay rejects the runtime | High | Prevent feature work from worsening large-file debt | Conditional |

### 2026-09-03 — P1.7e-1b complete: fail-open shadow runtime integrated

- Added a bounded daemon shadow runner and a hook immediately after the real
  main-chat BM25/translation/expansion/scope result is materialized. The worker
  receives a deep copy and its fused output has no return path into grouping,
  citations, prompts, answers, task state, or the UI.
- The runner is inert unless every frozen activation condition is present. It
  lazily loads the exact Qwen provider in cache-only mode and calls only
  `EmbeddingIndexStore.load`; it never builds or refreshes an index. A document
  or index-pointer signature change invalidates the in-process runtime cache so
  a stale snapshot fails closed inside the shadow while BM25 stays live.
- At most one shadow is active. Concurrent requests are skipped and recorded
  without queuing user work. Model/dependency/index/provider failures are
  reduced to stable redacted categories and never propagate to the foreground.
  Scoped execution filters dense hits to source identities already admitted by
  the product-scoped candidate set.
- Added a sixth local observation event containing only counts, booleans,
  stable IDs, and duration. It records K=5/K=10 overlap and positional changes,
  dense-tail count, source escapes, prefix preservation, cold/cache state, and
  product-input immutability; questions, evidence, paths, titles, errors, and
  model names remain prohibited. The frozen privacy artifact was regenerated
  and passes its canary replay.
- Eighteen focused runtime, privacy, failure, concurrency, scope, retrieval-
  observation, and contract tests pass; Ruff passes. Product-visible retrieval
  remains BM25/structured-only and the new shadow defaults off.

### Priority assessment after shadow runtime integration

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7e-2a | Freeze and implement the one-run 18-query operational replay/report using the matching 31-document Qwen index | Very high | Only runtime evidence is missing; the existing index is read-only and corpus-matched | Do next |
| P1.7e-2b | Lock pass/reject, then keep shadow off or permit opt-in design accordingly | Very high | No tuning or rerun after observing latency | After replay |
| P1.3 | User retrieval choice | High but blocked | Must wait for operational result | Conditional |
| P2 architecture | Module extraction | High | Resume immediately if replay rejects; otherwise follow opt-in boundary work | Conditional |

### 2026-09-03 — P1.7e-2a implementation and read-only preflight complete

- Implemented a one-run operational report over all 18 frozen journeys in both
  scoped and full-library modes (36 shadow submissions). The first cold start is
  reported separately; the latency gate uses the remaining 35 warm queries.
  Report validation recomputes membership, summaries, thresholds, decision, and
  a latency-independent result fingerprint.
- The same process first runs a deliberate injected runtime failure and requires
  a redacted error plus unchanged product input. It also fingerprints every file
  in the explicit index before and after the replay; a single byte of change
  rejects the outcome.
- Fifteen protocol/runtime/report tests pass before measurement. A final
  read-only preflight found exactly 2,251 chunks, zero fixture/corpus errors, a
  1,024-dimensional index with the accepted provider identity, and unchanged
  index tree SHA-256
  `c75e16595f8d96950846721da0c27b5c530a1f06a653d52fc326b0df7c7feaae`.
  Neither the canonical report nor telemetry output exists, and no query was
  encoded during preflight.

### Priority assessment before the sole operational replay

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7e-2b | Execute the one allowed 36-submission replay and lock its operational pass/reject | Very high | All code, gates, corpus, provider, index, and clean destinations are verified | Do now |
| P1.3 | If passed, design—not silently enable—an explicit local Dense opt-in and resource disclosure | High | User choice remains mandatory | Conditional |
| P2 architecture | If rejected, keep shadow off and resume module-boundary reduction | High | Avoid adding unreliable runtime complexity | Conditional |

### 2026-09-03 — P1.7e-2b complete: operational replay rejected on warm latency

- The sole run completed all 36 scoped/full-library submissions with one cold
  start and 35 warm queries. Completion, product-input immutability, and
  protected-prefix preservation are all 100%; scoped source escapes are zero.
  The shadow admitted 71 dense novelty-tail hits without changing any answer or
  user-visible ranking.
- Foreground submission p50/p95 is 2.780/4.559 ms, inside the fixed 5 ms bound.
  Warm shadow p50 is 213.081 ms, but p95 is 261.435 ms and therefore exceeds
  the predeclared 250 ms limit by 11.435 ms. The operational gate rejects the
  candidate; its status is `rejected_keep_shadow_disabled` and opt-in design is
  not yet allowed.
- The in-process injected failure, privacy replay, 37 content-free observation
  events, and full index immutability all pass. Index tree SHA-256 remains
  `c75e16595f8d96950846721da0c27b5c530a1f06a653d52fc326b0df7c7feaae`.
- Pinned latency-independent result fingerprint
  `48e54e620d73681f3fbaf8ad7d3e86982b51f5079d7ab0f1d66878113500461e`
  and report SHA-256
  `90f066084ccae2ddc8fea7c41649cd55c221f6e5dceb0a61ac85b5bd3ac26998`.
  CI/release validate the locked rejection. There is no retry, threshold change,
  product enablement, or ordinary-user setting from this result.

### Priority assessment after operational rejection

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7f-1 | Profile and replace the exact DenseRetriever Python scalar dot-product loop with an optional vectorized local backend, under rank-equivalence tests | Very high | The observed p50≈213 ms points to exact scoring cost; semantics and fusion must remain byte-for-byte equivalent | Do next |
| P2 architecture | Move the reusable local Transformers provider out of evaluation tooling into a bounded `kb` adapter with compatibility re-export | High | Removes a reversed product→tools dependency introduced by shadowing and improves module boundaries | Pair with P1.7f |
| P1.7f-2 | Freeze a new performance-only child protocol before any second operational measurement | High | Must not reinterpret or overwrite the rejected v1 run | After profiling/equivalence |
| P1.3 | User Dense opt-in | High but blocked | Remains blocked until a distinct performance candidate passes its own frozen operational gate | Blocked |
| P1.4b | OpenTelemetry export | Medium | Local schema is still changing during performance work | Deferred |

### 2026-09-03 — P1.7f-1a complete: exact-scoring bottleneck isolated

- Profiled only the already-loaded 2,251×1,024 embedding matrix using an
  existing normalized vector as the query; no quality question, model call,
  index build, or operational retry was performed.
- The current Python scalar dot-product pass took 140.904 ms. A NumPy float32
  matrix-vector product took 11.475 ms on the same data (12.3× faster), retained
  the exact top-20 order, and differed in score by at most
  1.4647e-7. This accounts for most of the observed ≈213 ms warm median and is a
  materially better target than changing the 250 ms gate or reducing evidence
  depth.
- NumPy is already pinned in the Windows release runtime; development/CI may
  lack it, so the implementation must remain an optional import with the scalar
  path as a behavior-preserving fallback.

### Priority assessment after scoring profile

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7f-1b | Add immutable float32 matrix scoring to DenseRetriever when NumPy is available, with exact tie/order, score-tolerance, and fallback tests | Very high | Expected to remove ~129 ms without changing candidate semantics | Do next |
| P2 provider boundary | Relocate the local Transformers adapter from `tools` to `kb` after scoring is isolated | High | Avoid mixing two architectural changes in one equivalence proof | Next after vectorization |
| P1.7f-2 | Freeze a performance-only child protocol before measuring the optimized runtime | High | The rejected v1 artifact remains immutable | Conditional on rank equivalence |

### 2026-09-03 — P1.7f-1b complete: vectorized exact shortlist is rank-equivalent

- DenseRetriever now optionally stores one immutable float32 NumPy matrix. It
  uses matrix-vector multiplication only to find a conservative boundary band,
  then recomputes that shortlist with the original Python double-precision dot
  product and original deterministic tie rule. If the exact shortlist cannot
  fill K, it falls back to the full scalar path.
- NumPy remains optional: import absence retains the previous implementation and
  score/rank behavior. Evidence-readiness filtering and minimum-score semantics
  are applied to the exact result, not trusted to approximate arithmetic.
- Forty seeded random queries over 256×64 vectors, exact ties, no-NumPy fallback,
  hybrid/fusion consumers, fixture evaluation, parent collapsing, and shadow
  execution pass in 28 tests. The first pure-float32 attempt was correctly
  rejected by the equivalence test after one near-score reorder and was not kept.
- On the real 2,251×1,024 matrix, 30 independent query vectors produce exactly
  identical top-40 IDs and scores. Scalar p50/p95 is 147.927/173.983 ms;
  vectorized-exact-shortlist p50/p95 is 5.794/8.141 ms, a 25.53× median speedup.
  This is implementation evidence only, not a rerun of the rejected operational
  gate.

### Priority assessment after vectorized equivalence

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2 provider boundary | Move the reusable cache-only Transformers adapter into `kb`, leaving a compatibility re-export for frozen eval tools | High | Removes the only product-runtime import from `tools` before a new protocol is frozen | Do next |
| P1.7f-2a | Freeze a child protocol pinned to the rejected v1 report plus vectorized/provider implementation hashes | Very high | Cleanly separates performance correction from the original outcome | After boundary cleanup |
| P1.7f-2b | Run one new operational replay with unchanged model, index, queries, fusion, and thresholds | Very high | Only implementation latency may differ | Conditional |

### 2026-09-03 — P2 provider boundary cleanup complete

- Moved the 283-line reusable cache-only Transformers pooling adapter from
  evaluation tooling into `kb/transformers_embedding_provider.py`. Product
  shadow runtime now imports only from `kb`; the old tool path is a 16-line
  compatibility re-export so all frozen benchmark CLIs and tests retain their
  public import.
- Model identity, pooling, prefixes, max length, dtype, device, download=false,
  and remote-code=false behavior are unchanged. Twenty-four provider, candidate,
  shadow, vectorization, and compatibility tests pass; Ruff passes.
- This removes the reversed product→evaluation dependency without duplicating
  provider code and reduces the risk that an internal shadow feature deepens the
  project's existing module-boundary debt.

### Priority assessment after provider extraction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7f-2a | Freeze the vectorized-runtime child protocol with unchanged candidate and original 250 ms gate | Very high | All allowed code changes are complete and rank-equivalent | Do next |
| P1.7f-2b | Implement a separate validator/report path and run it exactly once | Very high | Must preserve the rejected v1 artifact | After protocol tests |
| P1.3 | Explicit user opt-in design | High but blocked | Still depends on the new runtime measurement | Blocked |

### 2026-09-03 — P1.7f-2a complete: vectorized replacement protocol frozen

- Froze a distinct child protocol pinned to the original shadow protocol, its
  rejected report and fingerprint, and the exact optimized DenseRetriever,
  shadow runtime, and relocated provider file hashes. Four mutation tests and
  its validator pass; CI/release enforce the child boundary.
- The only allowed behavioral implementation change is float32 matrix coarse
  selection followed by the original exact score/tie calculation with scalar
  fallback. Model, provider identity, query/case order, 31-document index,
  5/3/2 fusion, 36 submissions, privacy, fail-open behavior, 250 ms warm p95,
  and 5 ms foreground p95 remain inherited unchanged.
- The replacement has one run and a new report/telemetry destination. It cannot
  overwrite or reinterpret the v1 rejection, tune after measurement, enable the
  product, or expose an ordinary-user setting.

### Priority assessment after vectorized protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7f-2b | Implement and synthetically validate the separate child report adapter | Very high | Must reuse the original gate arithmetic without duplicating policy | Do next |
| P1.7f-2c | Execute once and lock pass/reject | Very high | Rank equivalence already proven; only operational latency remains | After adapter validation |
| P1.3 | Explicit local Dense choice | High but blocked | Conditional on replacement pass and full product quality gates | Blocked |

### 2026-09-03 — P1.7f-2b complete: child report adapter and preflight validated

- Added a separate vectorized replay/report adapter that reuses the original
  operational runner and validator for case membership, arithmetic, privacy,
  fail-open, index immutability, and threshold enforcement. It only adds the
  child protocol/implementation identities and its distinct decision wording,
  avoiding duplicated gate policy.
- Eleven adapter/protocol/parent tests pass, including rejection of a changed
  implementation hash and an attempt to relabel a failed inherited gate. The
  broader focused set passes 20 tests; Ruff and both parent/child validators
  pass. The locked v1 rejection still validates unchanged.
- Final read-only preflight confirms the real runtime selects
  `numpy_float32_exact_shortlist`; the new canonical report and telemetry paths
  do not exist. No query was encoded and no replacement run was consumed.

### Priority assessment before vectorized replacement replay

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.7f-2c | Execute the one allowed vectorized replay and lock pass/reject | Very high | Every inherited gate and destination is verified | Do now |
| P1.3 | On pass, implement an explicit per-user local retrieval choice with honest resource/index readiness disclosure | High | Must remain opt-in and local-library-only | Conditional |
| P2 architecture | On reject, keep Dense disabled and return to large-module decomposition | High | Prevent unreliable complexity from accumulating | Conditional |

### 2026-09-03 — P1.7f-2c complete: vectorized operational replacement passes once

- The sole child run completed the unchanged 36 scoped/full-library shadow
  submissions. Warm shadow p50/p95 falls from the rejected v1
  213.081/261.435 ms to 59.925/128.983 ms. Foreground submission p50/p95 is
  3.017/4.465 ms, within the inherited 5 ms boundary.
- Completion, product-input immutability, and protected-prefix preservation are
  100%; source escapes remain zero and the same 71 dense novelty-tail hits are
  observed. This confirms the speedup did not alter the candidate composition.
- All inherited checks pass, including the in-process failure probe, content-
  free privacy replay, and byte-identical index tree. Pinned result fingerprint
  `b63ec97dc7b5562c1c85e4c532a442725d4b55842caa9a172343dd94992600d0`
  and report SHA-256
  `2fc943fa0ec430a81f52fc78bd2610d4e92a23e3eaafe9b577f6e21d13326f9f`.
- The decision is `passed_for_explicit_opt_in_design`, not product enablement.
  Shadow remains default-off, product ranking is still BM25/structured-only,
  and no ordinary-user setting is exposed. CI/release validate both the old
  rejection and new performance pass.

### Priority assessment after vectorized operational pass

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3a | Define the smallest explicit local retrieval choice: `BM25（默认）` vs `BM25 + 语义检索（本地）`, with readiness/resource disclosure and automatic BM25 fallback | Very high | Matches the user's requested control without expanding into external knowledge or writing | Do next |
| P1.3b | Add backend request contract and per-request candidate selection; never build/download implicitly | Very high | User choice must not depend on process-wide environment switches | After contract freeze |
| P1.3c | Add the settings/chat control and E2E fallback/readiness tests | High | The UI must explain unavailable state instead of exposing unresolvable warnings | After backend |
| P2 architecture | Continue large-file extraction around any touched chat/settings boundary | High | New option must not add more logic to oversized files | In parallel with implementation structure |
| P1.4b | Optional OpenTelemetry export | Medium | Local event schema is now stable enough, but user retrieval control has higher value | Deferred |

### 2026-09-03 — P1.3a complete: explicit local retrieval contract frozen

- Froze `retrieval_product_opt_in_v1` before product implementation. The only
  request values are `bm25` and `hybrid_local`; BM25 remains the default and
  unknown values normalize to BM25. This choice is per generation request and
  remains independent from the existing external webpage/model supplement
  choice.
- The hybrid candidate is pinned to the semantically accepted and operationally
  passing Qwen protected-tail implementation. It may only retrieve chunks from
  papers already in the user's library: the embedding model contributes no
  answer content, external webpage content, or uncited model knowledge.
- Readiness is deliberately strict. The control may expose the hybrid choice
  only when optional dependencies, the pinned locally cached model, a current
  corpus-matched product index, and the exact vectorized backend are all ready.
  An unavailable hybrid mode is not shown as a dead option the user cannot fix.
- Neither readiness nor generation may install dependencies, download a model,
  build an index, or depend on an evaluation artifact. A runtime race or failure
  returns the untouched BM25 ranking and does not block the answer; the product
  can state that BM25 completed the turn without asking the user to repair it.
- Five contract mutation tests, the validator, and Ruff pass. CI and the Windows
  release workflow now enforce this boundary.

### Priority assessment after product opt-in contract freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3b-1 | Extract a reusable cache-only local Dense runtime/readiness service from shadow code | Very high | Product and shadow must share one pinned provider/index contract without adding logic to `task_runtime.py` | Do next |
| P1.3b-2 | Add a small readiness router and per-request fail-open candidate selector with focused tests | Very high | Enables an honest backend contract while keeping BM25 authoritative | After service extraction |
| P1.3d | Design system-managed explicit preparation (dependency/model/index lifecycle) | Very high | Required before the ordinary-user hybrid control may ship | Before UI exposure |
| P1.3c | Add the per-conversation UI control and E2E tests | High but blocked | Contract forbids a dead option before preparation exists | Blocked on P1.3d |
| P2 architecture | Keep new readiness/selection logic in bounded modules and a new router | High | Avoid further growth of oversized chat/settings files | Enforced during P1.3b |

### 2026-09-03 — P1.3b-1 complete: cache-only product Dense runtime isolated

- Added a bounded `kb/local_dense_runtime.py` service rather than adding model,
  index, or readiness lifecycle code to `task_runtime.py`. It owns the closed
  retrieval-mode normalization, fixed product index location, accepted Qwen
  provider construction, redacted readiness result, and signature-keyed runtime
  cache.
- The provider is cache-only, CUDA/float16, revision-pinned, and identity-equal
  to the already measured shadow candidate. The product path always resolves to
  `<db_dir>/dense_index`; it cannot point at or silently reuse an ignored
  evaluation artifact.
- Readiness verifies optional dependencies, NumPy vectorization, CUDA, the
  locally cached pinned model snapshot, and a committed index that matches both
  the current corpus and provider. Public results contain only a stable reason
  ID and document/chunk counts—never paths or exception text.
- Loading is separate from readiness and never calls the index builder. The
  in-process retriever cache invalidates when `docs.json` or the committed index
  pointer changes. The previously frozen shadow runtime remains byte-identical;
  a parity test binds the new provider identity to it without invalidating its
  operational evidence.
- Thirty-seven focused runtime, provider, embedding-index, and shadow tests pass;
  Ruff passes.

### Priority assessment after product Dense runtime isolation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3b-2a | Add a small authenticated/read-only readiness endpoint over the product DB | Very high | Gives the frontend one truthful gate without leaking paths or loading/downloading a model | Do next |
| P1.3b-2b | Implement a pure fail-open candidate selector, then thread `retrieval_mode` through generation metadata/task | Very high | Must prove BM25 identity on every unavailable/error branch before integration | After endpoint |
| P1.3d | Implement explicit system-managed preparation and progress | Very high | Still required before ordinary-user UI exposure | After backend selection contract |
| P1.3c | Per-conversation control and E2E | High but blocked | Do not show a dead choice | Blocked on preparation |
| P2 architecture | New endpoint stays in its own router; product selector stays outside task runtime | High | Maintain module boundaries | Continue |

### 2026-09-03 — P1.3b-2a complete: truthful readiness API added

- Added the authenticated, read-only
  `GET /api/retrieval/local-readiness` endpoint in a new 24-line router. It
  reports BM25 as the default and only includes `hybrid_local` in
  `available_modes` when the complete local readiness service passes.
- An unavailable runtime returns `show_mode_control=false` and only BM25 in the
  selectable modes. A ready runtime exposes both actual choices plus current
  document/chunk counts. Neither response contains a filesystem path, model
  location, raw exception, download action, or repair task.
- Route tests exercise both ready and unavailable states through the real
  FastAPI application and prove the unavailable response does not echo the DB
  path. Together with runtime/contract tests, 24 focused tests pass; Ruff
  passes.

### Priority assessment after readiness API

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3b-2b-1 | Implement a pure product hybrid selector with exact BM25 fallback, scope filtering, protected-tail fusion, and top-K remainder rules | Very high | Highest-risk semantic mutation can be proven without touching generation | Do next |
| P1.3b-2b-2 | Thread normalized `retrieval_mode` through request, stored user metadata, task trace, and the single retrieval integration point | Very high | Keep integration narrow and auditable | After selector tests |
| P1.3d | System-managed preparation workflow | Very high | Required for actual ordinary-user availability | Next milestone after backend integration |
| P1.3c | UI + E2E | High but blocked | Readiness API intentionally hides unavailable choice | Blocked on P1.3d |

### 2026-09-03 — P1.3b-2b-1 complete: fail-open product selector proven in isolation

- Added `kb/local_retrieval_selector.py` as a pure boundary between the existing
  BM25 candidates and the accepted protected-tail Dense candidate. BM25/default
  and unknown modes return a deep value-identical ranking without opening the
  Dense runtime.
- In hybrid mode the selector always runs the frozen 10-result, candidate-depth
  40, protected 5 / sparse survivor 3 / Dense novelty 2 composition. The first
  five BM25 identities are checked again after fusion. For deeper internal
  candidate lists, original BM25 remainder items are appended without duplicate
  identities so downstream grouping does not lose its established evidence
  window.
- Current-paper and basket-style scopes wrap Dense retrieval in a source filter
  derived from the already admitted BM25 sources. A detected prefix violation,
  source escape, dependency/model/index failure, or Dense query failure returns
  undecorated BM25 values and a stable content-free reason ID; it never blocks
  answer generation or mutates the caller's list.
- The runtime now exposes `load_ready`, which checks all prerequisites before
  opening the product index. Forty-five focused selector/runtime/fusion/shadow
  tests pass; Ruff passes.

### Priority assessment after isolated product selector

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3b-2b-2a | Add `retrieval_mode` to `/api/generate`, user-message metadata, task state, and research trace | Very high | Contract flow must be visible to tests before ranking changes | Do next |
| P1.3b-2b-2b | Call the selector only after all existing scope/audit/basket filters and before grouping/citation seeds | Very high | Preserves existing structured/special-case retrieval semantics | Immediately after request plumbing |
| P1.3b-2b-2c | Run generation regression plus full focused backend gates | Very high | Catch hidden interactions in the large task runtime | After integration |
| P1.3d | System-managed preparation | Very high | Remains the UI blocker | Next milestone |

### 2026-09-03 — P1.3b-2b-2 complete: per-request backend integration is fail-open

- `/api/generate` now accepts and normalizes `retrieval_mode`, stores the
  normalized value with the user turn, and passes it into the generation task.
  Unknown/missing values become BM25. The web client explicitly sends and
  optimistically stores the current default `bm25`, but exposes no hybrid UI
  before preparation exists.
- The worker applies local hybrid selection only after existing answer-audit,
  current-paper, and basket filters, and before heading/document grouping and
  citation seeds. Exact-support and structured-section fast paths remain
  authoritative and record a resolved `structured_*_preserved` fallback when a
  hybrid request reaches them.
- Each task and internal research trace records only requested/used mode, stable
  fallback reason, prefix result, novelty count, and cache/cold booleans. Local
  observability distinguishes `hybrid_local_product` from `bm25_product`; no
  question, evidence, path, or exception is recorded. The existing shadow still
  receives the pre-hybrid BM25 copy, so its comparison semantics remain intact.
- The selector's output drives every downstream evidence/citation consumer in
  the normal path; its fallback output is the original BM25 value set. The
  frontend production build passes. Backend API/runtime/privacy gates pass 107
  focused tests, including 70 generate-route regressions. A real-browser chat
  send/race suite passes all 27 cases, covering default local mode, external
  supplement independence, conversation switching, cancel/retry, duplicate
  send protection, and SSE completion.

### Priority assessment after per-request backend integration

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1 | Define a system-managed preparation lifecycle with explicit user start, bounded download/build phases, progress, cancellation, and atomic publish | Very high | This is the missing path that makes hybrid availability solvable instead of a dead status | Do next |
| P1.3d-2 | Decide release-runtime delivery for Torch/Transformers and GPU/CPU compatibility from measured size/latency evidence | Very high | Current release cannot prepare or run the accepted candidate | Part of protocol freeze |
| P1.3c | Only after preparation passes, expose the per-conversation control and resolved fallback notice | High | Backend is ready; ordinary UI remains correctly hidden | Blocked on preparation |
| P1.3b-3 | Full backend suite plus existing release validators | High | Run after preparation design stabilizes or before any checkpoint commit | Pending |
| P2 architecture | Extract any preparation jobs into new modules/router, not `library.py` or `task_runtime.py` | High | Prevent large-file regression | Mandatory design constraint |

### 2026-09-03 — P1.3d-1a complete: accepted CUDA runtime is not deployable as the core app

- Re-read the current release runbook. The published beta.13 ZIP is
  94,876,265 bytes and its clean-profile contract requires a self-contained
  embedded Python runtime with exact dependencies, no system Python/Node, and
  no packaged user data. New retrieval support must preserve those gates and
  may not turn restart into implicit network/model work.
- Measured the exact accepted candidate environment, without downloading or
  changing it. The CUDA Torch package directory is 4,228.5 MiB;
  Transformers/tokenizers/Hugging Face Hub/safetensors add about 86.8 MiB; the
  pinned Qwen snapshot is 1,151.5 MiB (1,136.4 MiB model weights). Bundling this
  stack would add more than 5.3 GiB of logical files to a roughly 90.5 MiB ZIP.
- The accepted provider is specifically CUDA + float16. The release dependency
  lock contains NumPy but none of Torch, Transformers, tokenizers, Hub, or
  safetensors. Therefore adding a preparation button now would still leave most
  installed users unable to complete it, and adding the current runtime to the
  core package is rejected on deployability before implementation.
- The screening report shows why the CUDA candidate remains valuable research
  evidence: 2,251 document chunks encoded in 128.4 seconds and 116 queries in
  6.49 seconds. It proves semantic value and GPU feasibility, but not CPU or
  clean-package feasibility.

### Priority assessment after deployability measurement

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b | Freeze and run a CPU float32 query-only compatibility screen against the accepted index | Very high | Cheapest way to learn whether a separately installed CPU runtime is viable before any full rebuild/package work | Do next |
| P1.3d-1c | If CPU query screen passes, measure CPU full-index build and optional-runtime download footprint; if it fails, screen ONNX/quantized or smaller local candidates | Very high | Branch on evidence, not preference | Conditional |
| P1.3d-2 | Design explicit resumable sidecar preparation only after one deployable runtime passes | Very high | Avoid implementing an installer for a rejected runtime | Blocked on feasibility |
| P1.3c | Ordinary-user UI | High but blocked | Current CUDA-only candidate must not be shown as generally available | Blocked |
| Core release | Keep Torch/model out of `requirements-release.txt` | Very high | Preserves the current 90 MiB-class clean install until a separate add-on is proven | Locked for now |

### 2026-09-03 — P1.3d-1b protocol frozen before any CPU query

- Froze a one-run CPU float32 compatibility screen over the same 12 semantic
  paraphrase cases and scoped/full-library modes that accepted the CUDA
  candidate. It reuses the accepted 1,673-chunk index read-only and creates no
  document embeddings, product index, downloads, or UI.
- The CPU provider keeps the exact Qwen model revision, prompt, pooling,
  padding, and max length; only device and dtype change. Its distinct provider
  identity is pinned. The existing CUDA-built index tree is pinned at SHA-256
  `928d30b1aa31bcdfebe529bdefda4f424b96e77148e48f981794a77d37a45013`
  and must remain byte-identical.
- Quality must still pass the original +0.15 Recall@10 semantic gate in both
  modes, preserve Recall@5/wrong-source@5 safety, and not regress the accepted
  parent fusion on Recall@5/10, MRR, or nDCG@10. Operationally, all 12 unique
  queries must finish, model load must be ≤30 seconds, and warm query encoding
  p95 must be ≤1,000 ms on the recorded Ryzen 7 5800H reference host.
- Passing only authorizes a full CPU document-index feasibility study; it does
  not authorize product enablement or an ordinary-user control. Four mutation
  tests, the validator, CI/release wiring, and Ruff pass.

### Priority assessment after CPU screen freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b-2 | Implement a report runner that loads the accepted index with its parent descriptor, embeds only the 12 queries on CPU, and recomputes all gates | Very high | Must validate cross-runtime compatibility without weakening index identity in product code | Do next |
| P1.3d-1b-3 | Perform read-only preflight, execute once, pin outcome and report hash | Very high | No rerun or threshold change after measurement | After runner tests |
| P1.3d-1c | Full CPU index build/footprint or ONNX/smaller-model branch | Very high | Determined by the frozen result | Conditional |

### 2026-09-03 — P1.3d-1b-2 complete: one-run CPU query runner implemented

- Added a dedicated one-run runner and report validator. It opens the accepted
  semantic index only through the pinned CUDA-parent descriptor, then supplies
  CPU float32 vectors for the 12 unique queries. Product index validation is
  therefore not weakened merely to permit a cross-runtime experiment.
- The runner caches each unique query vector, records only ordinal and latency,
  and recomputes product control, protected-tail fusion, semantic acceptance,
  parent non-regression, latency, completion, and index-integrity gates. It
  never records query text, evidence text, paths, or exception bodies in the
  report.
- It refuses to run when the result report already exists, fingerprints the
  accepted index before and after, creates no document embeddings, and releases
  both model providers after the bounded measurement. `--validate-only` is a
  separate read-only path for CI/release use after the outcome is pinned.
- Six protocol/runner unit tests pass, including rejection on either quality or
  p95 latency regression; Python compilation and Ruff pass. No CPU model has
  been loaded and no query has been encoded while implementing or testing this
  step.

### Priority assessment after CPU runner implementation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b-3a | Run a zero-query preflight over frozen inputs, accepted index identity, semantic corpus, local model cache, and absent result report | Very high | Prevents spending the protocol's sole run on a setup or identity error | Do next |
| P1.3d-1b-3b | Execute the frozen CPU screen exactly once and atomically retain its report | Very high | Produces the deployability branch decision | Only after all preflight checks pass |
| P1.3d-1b-3c | Pin result fingerprint/report hash and wire read-only outcome validation into CI/release | High | Makes the measured decision reproducible and tamper-evident | Immediately after measurement |
| P1.3d-1c | Full CPU index build/footprint or ONNX/smaller-model branch | Very high | Must follow the measured decision | Conditional |

### 2026-09-03 — P1.3d-1b-3a checkpoint: v1 preflight invalidated safely; v2 frozen

- The zero-query preflight correctly stopped before model loading because the
  v1 protocol's pinned index-tree SHA did not describe the accepted semantic
  index. All other prerequisites passed: the result report was absent, the
  parent provider identity matched, all 1,673 fixture chunks validated, the
  exact model revision was cached, and zero queries/document embeddings/index
  writes occurred.
- Preserved v1 unchanged and recorded the failed preflight in
  `retrieval_cpu_query_compatibility_preflight_invalidation_v1.json` (SHA-256
  `a249e30220a0db3cfb7b989bcbf614122967d55aa6e09fdc74b09e8be278d2ff`).
  The observed index tree was identical across the failed preflight.
- Froze successor protocol v2 (SHA-256
  `e524378d1be22de93c3de4a77949c8dd80dc8e35c99786598a27367aed1eaf37`)
  without changing the candidate, fusion, thresholds, or one-run limit. It now
  pins the actual tree plus build id, corpus fingerprint, current pointer,
  manifest, vector-file hash, file count, and byte size. The validator also
  pins the v1/invalidation lineage and proves no query was spent before v2.
- Seven protocol/runner tests, the updated validator, and Ruff pass. This is a
  protocol input correction before measurement, not a post-result threshold or
  candidate change.

### Priority assessment after safe preflight invalidation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b-3a | Repeat zero-query preflight against v2 and all strengthened index identities | Very high | Last setup check before the sole measurement | Do next |
| P1.3d-1b-3b | Execute v2 exactly once | Very high | Must not proceed if any strengthened identity differs | Immediately after a clean preflight |
| P1.3d-1b-3c | Pin and validate the immutable outcome | High | Prevent silent rerun or report drift | After measurement |
| P1.3d-1c | Follow pass/fail feasibility branch | Very high | Evidence decides CPU full-index vs lighter runtime | Conditional |

### 2026-09-03 — P1.3d-1b-3a complete: v2 zero-query preflight passed

- The strengthened v2 preflight passes every input and integrity check: result
  absent; tree hash/file count/byte size; current pointer; manifest; raw binary
  vector SHA; build id; provider identity; corpus fingerprint; all 1,673 fixture
  chunks; and the exact locally cached model revision.
- A first diagnostic expression used the repository's text-fixture SHA helper
  on `vectors.f32`; that helper deliberately normalizes CR/LF bytes and is not a
  binary hash. Rechecking with streaming raw SHA-256 matched the frozen vector
  identity. No protocol field changed because v2 was correct; this was a
  preflight-tool usage correction.
- The CPU model is still unloaded, queries encoded remain zero, no document
  embeddings were created, and the accepted index remains byte-identical. The
  sole measurement run is therefore still available.

### Priority assessment after clean v2 preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b-3b | Execute the frozen v2 CPU query compatibility run exactly once | Very high | Every prerequisite is now proven before model load | Do next |
| P1.3d-1b-3c | Validate, fingerprint, hash, and CI-pin the outcome | High | Must happen before interpreting or branching | Immediately after run |
| P1.3d-1c | Measure full CPU index/footprint only on pass; otherwise pivot to a lighter runtime | Very high | Avoids sunk-cost implementation | Conditional |

### 2026-09-03 — P1.3d-1b-3b checkpoint: v2 execution invalidated by runner persistence bug

- The sole v2 process encoded the expected 12 CPU queries and reached
  post-measurement validation, but exited before persisting the report. The
  validator compared the entire newly measured BM25 control channel to the
  parent channel; nondeterministic `latency_ms` values made that equality
  impossible even though rankings and quality metrics were the intended
  control.
- No report exists and no quality metric was printed or inspected, so this
  execution provides zero usable outcomes and cannot be called either a model
  pass or failure. The accepted index remained unchanged, no document vectors
  were built, and the product stayed disabled.
- Recorded the event in
  `retrieval_cpu_query_compatibility_execution_invalidation_v2.json` (SHA-256
  `7b574e7f2233e926d261256e12e4dedae0eb773172adbf1a546f19d71ee5c3eb`),
  including one attempted run, 12 executed queries, zero usable results, and
  zero post-execution threshold/candidate changes. A v2 rerun is forbidden.

### Priority assessment after v2 execution invalidation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b-3b-r | Freeze v3 recovery with identical scientific contract, stable ranking-only control validation, and persist-before-validate semantics | Very high | Prevents both false rejection and another lost outcome | Do next |
| P1.3d-1b-3b-r2 | Unit-test report retention plus latency-insensitive/ranking-sensitive control comparison | Very high | Recovery run must fail closed with its raw evidence retained | Before recovery execution |
| P1.3d-1b-3c | Execute the single audited recovery run and pin it | Very high | No further execution allowed regardless of result | After clean v3 preflight |
| P1.3d-1c | Follow the measured pass/fail branch | Very high | Still blocked until one usable outcome exists | Blocked |

### 2026-09-03 — P1.3d-1b-3b-r complete: audited v3 recovery is ready

- Froze recovery protocol v3 at SHA-256
  `45b994bafaff8e0068847594af5af894365c43dc19e80b8dff1dddc580f94023`.
  It explicitly carries the one prior attempted run, zero usable outcomes, zero
  observed metrics, and permits one recovery run. The dataset, read-only index,
  CPU provider, 5/3/2 protected-tail fusion, and every quality/latency threshold
  remain identical to v2.
- Product-control validation now removes only `latency_ms` and compares the
  entire remaining channel: mode membership, aggregate metrics, K values,
  per-case membership, relevant ranks, and per-case metrics. A changed ranking
  still fails; merely measuring at a different wall-clock speed does not.
- Result fingerprinting uses the same stable ranking projection while the raw
  per-mode and per-query latency remains in the report. The runner now atomically
  persists the complete raw report before validation, and any validation error
  leaves that report in place and forbids another run.
- Nine protocol/runner tests pass, including latency-insensitive but
  ranking-sensitive comparison and proof that a synthetic validation failure
  occurs only after its raw report is readable on disk. Compilation, protocol
  validation, and Ruff pass.

### Priority assessment before the single v3 recovery

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b-3c | Repeat identity-only preflight, then execute the one v3 recovery run | Very high | Last allowed path to a usable CPU outcome; raw evidence cannot be lost | Do next |
| P1.3d-1b-3d | Pin result fingerprint/report SHA and add validate-only gates | High | Required before any interpretation | Immediately after report exists |
| P1.3d-1c | Branch to CPU full-index study or lighter runtime | Very high | Determined only by the pinned v3 outcome | Conditional |

### 2026-09-03 — P1.3d-1b-3c preflight complete: v3 recovery authorized

- The final identity-only v3 preflight passes: protocol lineage, absent report,
  immutable tree, pointer, manifest, raw vectors, provider, and 1,673-chunk
  corpus all match their pins. It encoded zero queries and confirms zero prior
  usable outcomes with exactly one recovery run authorized.
- No model load, download, index build, product write, threshold change, or UI
  exposure occurred. The next process is the final allowed CPU query execution;
  no subsequent retry is permitted because its raw report will be retained.

### Priority assessment after final v3 preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1b-3c-run | Execute and retain the one v3 recovery outcome | Very high | Fully preflighted and auditable | Do now |
| P1.3d-1b-3d | Pin/validate the retained outcome without re-execution | High | Must precede design branching | Next |
| P1.3d-1c | Continue only along the pinned pass/fail branch | Very high | Prevents feature-first bias | Conditional |

### 2026-09-03 — P1.3d-1b complete: CPU query compatibility passed and pinned

- The single audited v3 recovery produced and retained
  `retrieval_cpu_query_compatibility_v1.json`. Its stable result fingerprint is
  `6063af5f27991d74db0a044218b78385f5ff4318e9deb229e27d21b564730ab4`;
  the normalized report SHA-256 is
  `52ed6a51d22b0e6f0b9fc2533e15faf5c66398e1a46b0e0c358568c34015e0ad`.
  Both are pinned in the validator.
- CPU float32 loaded the cached pinned Qwen model in 24,871.821 ms. All 12
  unique queries completed; embedding p50 was 416.520 ms and p95 was 432.246
  ms, passing the frozen ≤30 s load and ≤1,000 ms warm-query gates.
- The mixed-runtime result exactly preserves the accepted semantic quality:
  scoped Recall@10 rises from 0.375000 to 0.666667 (+0.291667), and full-library
  Recall@10 rises from 0.291667 to 0.541667 (+0.250000). Recall@5 and
  wrong-source@5 remain equal to product BM25; MRR and nDCG@10 do not regress
  versus the accepted parent in either mode. The read-only index tree is
  byte-identical before/after.
- The decision is only `passed_for_cpu_full_index_feasibility`: it authorizes a
  CPU document-index/footprint study, not product enablement, model bundling, a
  preparation workflow, or user-facing hybrid control. CI and Windows release
  now run protocol plus read-only outcome validation. Ten focused tests,
  validate-only under the ordinary project Python, and Ruff pass.

### Priority assessment after CPU query compatibility

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1c-1 | Freeze a bounded CPU full-index feasibility protocol: isolated 1,673 chunks, new temp index, no product writes, build throughput/RSS/index-size gates | Very high | Query feasibility is proven; document indexing cost is now the decisive unknown | Do next |
| P1.3d-1c-2 | Measure a clean CPU-only optional runtime footprint, not the existing 4.2 GiB CUDA environment | Very high | Determines whether sidecar delivery is credible | Run alongside the same feasibility milestone |
| P1.3d-2 | Design resumable sidecar preparation | High but blocked | Requires acceptable full build and footprint evidence | Blocked on P1.3d-1c |
| P1.3c | Expose per-conversation hybrid control | High but blocked | No ordinary-user choice until system-managed preparation is proven | Blocked |
| Core release | Keep Torch/model out of the beta.13-class core ZIP | Very high | CPU pass does not make a 1.15 GiB model suitable for the core installer | Continue locked |

### 2026-09-03 — P1.3d-1c-1a complete: CPU full-index protocol frozen

- Froze `retrieval_cpu_full_index_feasibility_protocol_v1.json` at SHA-256
  `7d95366f197dff57855c75e97ce1992ebdb5aaaf251a37a84576654638b4a941`.
  Its authorization is the pinned CPU-query pass; no GPU result alone can
  authorize this build.
- The one-run study uses the same Qwen revision and semantic contract on CPU,
  with document batch size 8, and writes only a new dedicated ignored test
  index. It pins the source DB tree/corpus, requires an initially absent output,
  prohibits product-index reuse/writes, and requires atomic publish plus
  readback of all 10 documents and 1,673 chunks.
- Product-oriented operational gates are frozen before model load: ≤30 s model
  load, ≤600 s full build, ≥2.75 chunks/s, ≤6,144 MiB peak process RSS, ≤10 MiB
  index, zero truncated chunks, 100% query completion, and ≤1,000 ms query p95.
  Quality must retain the CPU-query pass's Recall@5/10, MRR, nDCG@10, and
  wrong-source safety while independently passing the +0.15 Recall@10 gate.
- The corpus contains 1,488,995 characters; p50/p95/max chunk lengths are
  735/1,908/6,723 characters. The dedicated index and result report are absent,
  and 353.97 GiB disk is free. Five mutation tests, protocol validation, CI and
  release wiring, and Ruff pass. No CPU model load or document embedding was
  performed in this step.

### Priority assessment after full-index protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1c-1b | Implement one-shot runner with start receipt, peak-RSS measurement, failure retention, atomic report, readback, and quality gates | Very high | A long build must never become an untracked or repeatable experiment | Do next |
| P1.3d-1c-1c | Zero-embedding preflight, then execute once | Very high | Measures the remaining runtime unknown | After runner tests |
| P1.3d-1c-2 | CPU-only sidecar dependency/model footprint | Very high | Authorized only if the full build passes | Conditional |

### 2026-09-03 — P1.3d-1c-1b complete: one-shot full-index runner implemented

- Added a dedicated CPU full-index runner that refuses any existing attempt
  receipt, result report, or destination index. Before model load it atomically
  writes a one-run receipt; success, validation failure, ordinary exception, or
  interruption therefore cannot silently turn into an untracked retry.
- The runner builds only the dedicated test index, reads it back through the
  same CPU provider, evaluates both semantic modes, verifies stable BM25
  control, and records model load, document encoding/build throughput, exact
  parameter dtype, query timings, index identity/size, source-tree integrity,
  truncation, and cross-platform peak process RSS. User questions and evidence
  text are not written to the report.
- Successful and failed execution reports are atomically retained before
  validation; errors expose only a stable exception category. Persisted-report
  validation is independent of ignored runtime artifacts and will pin the
  measured index identity after execution, so clean CI/release can reproduce
  the decision without shipping the test index.
- A Windows x64 handle-width defect in the first peak-RSS unit test was caught
  before any model work and fixed with explicit Win32 signatures. Ten
  protocol/runner tests now pass, including one-shot receipt enforcement,
  performance/truncation rejection, latency-insensitive fingerprinting, and
  live RSS availability. Compilation, validator, and Ruff pass. No attempt
  receipt, model load, document embedding, destination index, or report exists.

### Priority assessment after full-index runner implementation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1c-1c | Run final zero-embedding preflight, then the single bounded full-index process | Very high | All failure-retention mechanics are now proven | Do next |
| P1.3d-1c-1d | Pin report, stable outcome, and committed-index identity; wire validate-only on success | High | Required regardless of pass/reject | After run |
| P1.3d-1c-2 | Measure CPU-only optional runtime footprint | Very high | Only if full-index outcome authorizes it | Conditional |

### 2026-09-03 — P1.3d-1c-1c preflight complete: one CPU build authorized

- The final zero-embedding preflight passes the CPU-query authorization,
  protocol, 12-case fixture, immutable source DB, candidate identity, exact
  local model revision, batch size 8, RSS probe, and free-space checks.
- Attempt receipt, result report, and dedicated index are all absent. The model
  remains unloaded; provider document/query counters are both zero. Baseline
  process RSS is 76.473 MiB and 353.97 GiB disk remains free.
- The next process is the only authorized full-index build. Its start receipt
  will permanently consume the run even if the process fails; any report will
  be retained before interpretation.

### Priority assessment after full-index preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3d-1c-1c-run | Execute the single CPU build/readback/quality run | Very high | All identities and retention safeguards pass | Do now |
| P1.3d-1c-1d | Pin and interpret retained result | High | No rerun under this protocol | Next |
| P1.3d-1c-2 | Continue only if the report authorizes runtime-footprint work | Very high | Conditional | Blocked pending result |

### 2026-09-03 — P1.3d-1c complete: native CPU full-index runtime rejected

- The one authorized run completed, atomically published/read back the isolated
  index, retained its report, and closed the attempt receipt as `completed`.
  `retrieval_cpu_full_index_feasibility_v1.json` is pinned at normalized SHA-256
  `866e2abb58566cb0f569e5eb4c3dca20e6f7764d60793dd68c2ea4e8f85f282a`
  with stable result fingerprint
  `3085c4ed8cf73b2e68fa071b569610ef1812902c726f1af7b6e91e7ddaae00af`.
- Quality and correctness passed completely. CPU-built vectors reproduce the
  accepted protected-tail metrics exactly in both modes: scoped Recall@10
  0.666667 (+0.291667 over product) and full-library Recall@10 0.541667
  (+0.250000), with unchanged Recall@5/wrong-source@5 and no MRR/nDCG
  regression. All 1,673 chunks were embedded with zero truncation/reuse, the
  source corpus stayed unchanged, atomic readback succeeded, and the final
  index is only 7,277,202 bytes.
- Runtime failed by a decisive margin. Model load passed at 25.212 s and query
  p95 passed at 513.492 ms, but document build took 4,952.501 s (82.54 minutes)
  versus the 600 s gate; throughput was 0.337809 chunks/s versus 2.75; peak RSS
  was 15,751.070 MiB versus 6,144 MiB. The model really ran with float32
  parameters, batch size 8, and observed sequences up to 4,712 tokens.
- Therefore `cpu_optional_runtime_footprint_study_allowed=false`. Packaging the
  current PyTorch CPU stack, building a sidecar installer for it, exposing a
  preparation button, or enabling hybrid for ordinary users are all rejected.
  The algorithm/index format is retained; the execution layer must change.
- CI and Windows release now validate the frozen protocol and rejected outcome
  without requiring the ignored 7.28 MB runtime index. Eleven focused tests,
  protocol/outcome validators, and Ruff pass.

### Better solution and priority after native CPU rejection

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-1 | Deterministically measure token-padding amplification for current batch-8 document order vs length-bucket/token-budget batching | Very high | Max sequence 4,712 and 15.75 GiB peak suggest padding is a major avoidable cost; tokenizer-only diagnostic is cheap and creates no vectors | Do next |
| P1.3e-2 | Freeze a small candidate set: length-aware Qwen CPU only if the diagnostic shows enough headroom, plus quantized ONNX and/or a materially smaller multilingual embedding model | Very high | Need an order-of-magnitude runtime improvement without losing protected-tail quality | After diagnostic |
| P1.3e-3 | Evaluate candidates first on the same 12 semantic cases and a bounded document sample, then allow at most one full build for the winner | Very high | Avoid another 82-minute blind full run | After candidate protocol |
| P1.3d-2 | Resumable preparation with chunk/token progress, ETA, heartbeat, cancellation, and checkpoints | High but blocked | Current document-level progress hid 20+ minute long tails | Only after a deployable runtime passes |
| P1.3c | User retrieval-mode control | High but blocked | Do not expose an unavailable or user-unfixable state | Remains blocked |
| Cross-encoder | Reranker productization | Medium | Base Dense preparation is still the bottleneck | Defer |

### 2026-09-03 — P1.3e-2a complete: two-stage lighter-runtime screen frozen

- Froze `retrieval_lighter_runtime_screen_protocol_v1.json` before any new
  artifact download at normalized SHA-256
  `dffc76974fcdd1d288e5bc5e16f5590e4d09e140f26a7591b269f396a5817dea`.
  The authorization is the pinned tokenizer-only decision that closed native
  PyTorch Qwen CPU; it does not reopen that rejected path.
- The candidate set is capped at two and its order is binding. Candidate 1 is
  Qwen's official 600,127,788-byte AVX2 `quint8` ONNX artifact at immutable
  revision `bd58e9f...`, preserving the accepted 1,024-dimensional
  last-token/instruction contract while changing the execution engine.
  Candidate 2, reached only if candidate 1 fails, is the official
  470,268,510-byte `multilingual-e5-small` ONNX artifact at immutable revision
  `03415a4...`; it is a materially smaller ~0.1B multilingual fallback with a
  384-dimensional mean-pooled representation.
- Runtime scope is deliberately narrow: embedded-release-compatible Python
  3.10.11, ONNX Runtime 1.23.2 CPU provider, eight intra-op threads, AVX2,
  and no Torch, Transformers, Optimum, remote code, system site packages, or
  changes to core release requirements. The host reports AVX2 on the Ryzen 7
  5800H; the pinned ONNX Runtime release provides a Windows CPython 3.10 wheel.
- Stage 0 may verify downloads, graph structure, dependency footprint, and CPU
  provider only; model inference and paper access remain forbidden. Stage 1 is
  limited to 12 frozen queries plus 64 deterministically selected stress
  chunks, ≤10 minutes per candidate, ≤4 GiB RSS, ≤1 s warm query p95, and a
  projected ≤10-minute/≥2.75 chunks/s full build. Qwen must additionally
  preserve the accepted quality replay and show mean/worst document-vector
  cosine ≥0.98/0.95 against the CPU parent.
- Evaluation stops at the first Stage-1 pass. Across both candidates, at most
  one Stage-2 full build is allowed; it must retain the existing Recall/MRR/
  nDCG/wrong-source gates, finish in ≤10 minutes with ≤6 GiB RSS, publish and
  read back atomically, and leave the source corpus unchanged. Five mutation
  tests, direct protocol validation, CI/release validate-only wiring, Ruff, and
  diff checks pass. No artifact, inference, full index, product toggle, or
  ordinary-user control was created.

### Priority assessment after lighter-runtime protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2b-1 | Implement Stage-0 isolated Python 3.10.11 environment, exact artifact download/verification, ONNX checker, graph I/O receipt, and dependency footprint | Very high | Supply-chain and runtime compatibility must fail before inference | Do next for Qwen only |
| P1.3e-2b-2 | Freeze the 64-chunk stress manifest before loading Qwen and implement a subprocess-bounded ONNX provider | Very high | Makes the 10-minute cap enforceable and prevents sample drift | After Stage 0 |
| P1.3e-2b-3 | Run one Qwen Stage-1 screen; touch E5-small only if Qwen fails | Very high | Preserves semantic evidence while minimizing downloads and experiments | Conditional sequence |
| P1.3e-3 | Permit the single full build only for a Stage-1 winner | Very high | Final deployability proof | Blocked |
| P1.3d-2/P1.3c | Preparation service and user retrieval control | High but blocked | Still no full-build-qualified runtime | Remain blocked |

### 2026-09-03 — P1.3e-2b-1 complete: Qwen ONNX Stage 0 passed and pinned

- Built a dedicated Python 3.10.11 virtual environment under ignored test
  artifacts and froze all 24 non-bootstrap distributions. It has user-site
  packages disabled and contains no Torch, Transformers, Optimum, or
  sentence-transformers. The product and release requirement files remain
  unchanged.
- The first remote transfer broke before 12 MB and retained only a partial
  file. The preparation runner was hardened before retry: exact-root partials
  now resume with bounded retries, retain progress, and cannot publish until
  frozen size and raw SHA-256 both pass. This network incident did not create a
  report, load a session, access paper text, or consume a Stage-1 measurement.
- Downloaded and verified Qwen's official AVX2 quint8 ONNX artifact at exactly
  600,127,788 bytes and SHA-256
  `a7eda29cc374b01ae75eb1af1ab65c3c61a64c43f39f16e08c6e37c075dcbf4d`.
  The successful process resumed the retained 8,388,608 bytes and transferred
  the remaining 591,739,180 bytes.
- ONNX 1.18 checker passed in 4,513.131 ms. ONNX Runtime 1.23.2 created a
  CPUExecutionProvider-only session in 2,369.116 ms with two integer inputs
  (`input_ids`, `attention_mask`) and both token-level plus already pooled,
  float32 1,024-dimensional `sentence_embedding` outputs. Session run calls,
  queries, documents, and vectors all remain zero.
- The full evaluation environment's recorded distributions occupy 204,279,163
  bytes, including the development-only graph checker and its dependencies.
  This is already far smaller than the prior CUDA/PyTorch environment, but it
  is not yet a product footprint claim; Stage 1 must determine the minimal
  runtime subset and actual working-set peak.
- The retained `retrieval_lighter_runtime_stage0_qwen_v1.json` is pinned at
  normalized SHA-256
  `1c7b2eed6469e002b229337ce5e90bd4770f202818002b4c3192bf8488f75fb9`.
  Fifteen focused protocol/preparation/report tests, direct validators,
  CI/release read-only wiring, Ruff, and diff checks pass. Stage 1 is allowed;
  product preparation and user-facing Hybrid remain blocked.

### Priority assessment after Qwen ONNX Stage 0

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2b-2a | Freeze the exact 64-parent stress manifest with qrel-first and length-stratified coverage before the first session run | Very high | Prevents choosing an easy throughput sample after seeing model speed | Do next |
| P1.3e-2b-2b | Implement ONNX provider plus hard subprocess timeout, query replay, parent-vector parity, RSS, and token-weighted full-build projection | Very high | Stage-0 graph compatibility is not runtime or semantic compatibility | After manifest |
| P1.3e-2b-3 | Execute Qwen Stage 1 once and retain pass/reject | Very high | Decides whether E5 download is needed | After runner tests |
| P1.3e-3 | One full build for the Stage-1 winner | Very high | Required before preparation design | Blocked |
| P1.3d-2/P1.3c | Resumable preparation and explicit user choice | High but blocked | No full-build-qualified runtime yet | Remain blocked |

### 2026-09-03 — P1.3e-2b-2a complete: 64-parent stress sample frozen

- Added a deterministic manifest builder and froze
  `retrieval_lighter_runtime_stress_manifest_v1.json` before the first ONNX
  session run at normalized SHA-256
  `1e697579abcdf241d943d9caffa160ed78a301d713ad7ab48e40ef02640fbd12`.
  It validates the exact 10-document/1,673-chunk source tree and accepted CPU
  parent-index tree before selection.
- All 13 unique parent chunks relevant to the 12 protected semantic cases are
  included. The remaining 51 are selected with a fixed seed and SHA-256 tie
  break from five global character-length rank strata with targets
  10/10/10/10/11. The resulting 64 parents cover all 10 papers.
- The sample contains 58,834 characters; min/p50/p95/max are
  37/880.5/2,112.8/3,563 versus corpus p50/p95/max
  735/1,924.8/6,723. Its p95 is intentionally a little heavier than the
  corpus, while its strata still preserve short and median work. Each row pins
  chunk id, text hash, source position, and CPU-parent vector position; paper
  text is not stored in the tracked manifest.
- Four selection-helper tests pass. Tokenizer/model/session-run counters remain
  zero and no vector exists. During validation, the umbrella protocol's Stage-1
  parent path was found to say `.../index` while the retained audited index is
  `.../index_v1`. This does not affect Stage 0, which never read the index, but
  Stage 1 under that pointer is prohibited. The error will be retained rather
  than editing a frozen predecessor.

### Priority assessment after stress-manifest freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2b-2a-fix | Pin the stress manifest, retain a pointer-invalidation receipt, and freeze a corrected Stage-1-only protocol | Very high | Prevents measuring against an unintended/missing parent | Do next before inference |
| P1.3e-2b-2b | Implement the subprocess-bounded ONNX provider and Stage-1 runner against the corrected protocol | Very high | Runtime and semantic compatibility remain unknown | After correction |
| P1.3e-2b-3 | Execute exactly one Qwen Stage-1 run | Very high | Decides whether E5 is needed | Blocked on runner |
| P1.3e-3/P1.3d-2/P1.3c | Full build, preparation, and UI choice | High but blocked | No Stage-1 winner yet | Remain blocked |

### 2026-09-03 — P1.3e-2b-2a-fix complete: parent pointer corrected before inference

- Retained `retrieval_lighter_runtime_stage1_pointer_invalidation_v1.json` at
  SHA-256
  `277ffe5f46d499df0c6b8cc895284dcb708814751549ec11a0b5f2005865583a`.
  It records the frozen missing `index` path, the actual `index_v1` path and
  tree identity, prohibits Stage-1 execution under the umbrella pointer, and
  explicitly preserves Stage 0 because that step never accessed the parent
  index. The historical protocol was not rewritten.
- Pinned the 64-parent manifest with a standalone validator that enforces its
  exact source identities, group counts, privacy schema, and zero-inference
  counters. Clean CI/release validation does not require the ignored source DB
  or index.
- Froze the corrected Stage-1-only protocol
  `retrieval_lighter_runtime_stage1_qwen_protocol_v1.json` at normalized
  SHA-256
  `62a6c5edb9f243dfd631f13a43c776bfbe0b221a4599ef5374c0f4b04d0b3e04`.
  It pins the Stage-0 report, invalidation, stress manifest, exact `index_v1`
  pointer/manifest/vector identities, Qwen tokenizer hashes and pad semantics,
  official ONNX output, isolated dependency lock, and the original source tree.
- The measurement is fixed to one public warm-up, 12 timed semantic queries,
  64 embedded stress parents, and tokenizer-only scheduling over the complete
  1,673-parent corpus. Full-build time is projected by the ratio of actual
  padded attention-work proxies, not a naive average chunks/second. Document
  parity must reach mean/worst cosine ≥0.98/0.95 against the audited float32
  CPU index.
- A hard 600-second child-process cap, ≤30-second load, ≤1-second query p95,
  ≤4 GiB peak RSS, projected ≤10-minute full build, ≥2.75 chunks/s, zero
  truncation, exact source/index/artifact preservation, and all accepted
  semantic gates remain mandatory. Sixteen manifest/invalidation/protocol
  tests and all direct validators/Ruff pass. No tokenizer was loaded and ONNX
  session-run count remains zero.

### Priority assessment after corrected Stage-1 freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2b-2b-1 | Implement the isolated ONNX worker: exact tokenizer, length-aware packing, session output validation, vector parity, and peak RSS | Very high | First actual inference must be bounded and independently checkable | Do next |
| P1.3e-2b-2b-2 | Implement parent orchestrator: one-shot receipt, 600 s termination, source/index/artifact before-after hashes, semantic replay, atomic retained report | Very high | Worker success alone cannot authorize quality or replay | Then |
| P1.3e-2b-3 | Zero-inference preflight followed by the single Stage-1 execution | Very high | Decides Qwen vs E5 branch | After runner tests |
| P1.3e-3/P1.3d-2/P1.3c | Full build, preparation, and explicit user choice | High but blocked | No measured Stage-1 pass | Remain blocked |

### 2026-09-03 — P1.3e-2b-2b complete: bounded Stage-1 worker/orchestrator implemented

- Added an isolated ONNX worker that revalidates the exact source tree,
  artifact, tokenizer files, Python environment, and package lock before model
  work. It uses only ONNX Runtime CPU plus the Rust tokenizer, records peak RSS,
  and rejects wrong output shape, non-finite/zero vectors, provider drift, and
  any source/artifact mutation.
- The worker performs one public warm-up, then one timed run for each of the 12
  frozen semantic queries. It tokenizes the complete corpus only to derive the
  actual length-aware batch schedule, embeds exactly the 64 preselected parent
  chunks, and projects full-build time from the full/stress padded quadratic
  attention-work ratio. It never embeds the complete corpus or publishes an
  index.
- Added a parent orchestrator with a one-shot attempt receipt and a real
  600-second subprocess timeout. It independently hashes the source, audited
  CPU index, artifact, tokenizer, and environment before launch, reloads the
  ignored worker result, computes 64-vector cosine parity against exact parent
  positions, and replays the accepted protected-tail fusion against the
  unchanged product BM25 control.
- The tracked report intentionally excludes query strings, paper text, and raw
  vectors. It retains only case ids, timing/token counts, work summaries,
  per-parent cosine results, retrieval metrics, gate arithmetic, and immutable
  identities. It is atomically persisted before validation on success; timeout,
  worker failure, or parent evaluation failure retains a categorical failure
  report and closes the one-shot receipt.
- Eight worker/orchestrator tests pass, including token-budget/oversize packing,
  quadratic-work arithmetic, vector shape/normalization, operational/parity
  rejection, and latency-insensitive stable fingerprinting. Compilation and
  Ruff pass. No worker output, attempt receipt, model inference, or result
  report exists yet.

### Priority assessment after Stage-1 implementation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2b-2c | Run final zero-inference preflight: exact artifact/tokenizer/source/index/environment plus absent receipt/output/report | Very high | Last safe point before consuming the only run | Do next |
| P1.3e-2b-3 | Execute and retain the single Qwen Stage-1 outcome | Very high | Resolves runtime and semantic compatibility | Immediately after preflight |
| P1.3e-2c | Pin the outcome and branch to either one full build or E5 Stage 0 | Very high | Branch must follow the retained result | Conditional |
| Preparation/UI | Remain blocked | High | No Stage-1/full-build pass yet | No change |

### 2026-09-03 — P1.3e-2b-2c complete: final Stage-1 preflight passed

- The corrected protocol, pointer invalidation, Stage-0 report, and 64-parent
  manifest all validate. The official 600,127,788-byte artifact, two pinned
  tokenizer files, exact 10-document source tree, and audited `index_v1` tree
  match their frozen identities.
- The dedicated Python 3.10.11 environment again reports all 24 exact pins,
  disabled user-site packages, and no heavy framework. Attempt receipt, worker
  result, and tracked Stage-1 report are absent.
- The preflight did not load the tokenizer, create an ONNX inference session,
  encode a query/document, or create a vector. The next invocation is the only
  authorized Stage-1 measurement and will permanently consume its receipt.

### Priority assessment after final Stage-1 preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2b-3-run | Execute the bounded Qwen ONNX worker and semantic replay once | Very high | All preconditions pass | Do now |
| P1.3e-2c | Pin and interpret pass/reject without rerun | Very high | Required immediately after retention | Next |
| P1.3e-3 or E5 fallback | Follow only the report-authorized branch | Very high | No speculative second download/build | Conditional |

### 2026-09-03 — P1.3e-1a complete: tokenizer-only padding diagnostic frozen

- Froze `retrieval_cpu_padding_diagnostic_protocol_v1.json` at SHA-256
  `7cb758235728c03efeadf48a6d2d66ec5bab403788ae1d819b10c030f9a303dd`
  and implemented its deterministic analyzer. It is authorized by the pinned
  82.54-minute rejection and cannot load a model, encode a query/document, or
  create a vector.
- The diagnostic compares the actual document-order batch-8 schedule to
  within-document length bucketing and to a length-bucketed 4,096-padded-token
  budget with oversize singletons. It reports real/effective/padded tokens,
  padding amplification, an attention-work proxy, peak batch tokens, and an
  estimate anchored to the measured full-build duration.
- Before seeing token results, a bounded Qwen follow-up requires all three:
  ≥4x attention-work proxy improvement, ≥50% peak padded-token reduction, and
  projected build ≤15 minutes. Passing authorizes only a bounded sample, never
  another full build or product enablement; failing sends priority directly to
  a lighter runtime.
- Seven protocol/analyzer tests pass, including oversize singleton packing,
  padding/attention arithmetic, and immutable branch thresholds. Compilation,
  protocol validation, CI/release wiring, and Ruff pass. The report is absent
  and no tokenizer/model work occurred during implementation.

### Priority assessment before padding diagnostic

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-1b | Run the cached tokenizer once, validate arithmetic, and pin outcome | Very high | Cheaply decides whether native Qwen deserves even a bounded follow-up | Do next |
| P1.3e-2 | Freeze lighter-runtime candidate screen if the diagnostic rejects; otherwise first validate length-aware Qwen on a bounded stress sample | Very high | Branch is predeclared | Conditional |
| P1.3d-2/P1.3c | Preparation/UI | High but blocked | No runtime has passed full deployability | Remain blocked |

### 2026-09-03 — P1.3e-1 complete: padding optimization is insufficient; native Qwen closed

- Ran the tokenizer-only diagnostic once and pinned
  `retrieval_cpu_padding_diagnostic_v1.json` at normalized SHA-256
  `af1a6fc67929af03ba3c1afaf3fe3a3c37330796343f8d1d251df8c57596973a`
  with result fingerprint
  `58ecfe98fc10f3e628320e870879d6b091f741a1e4c2069153be264467b78658`.
  The model was never loaded and zero queries/documents/vectors were encoded.
- The real corpus contains 489,070 effective tokens: p50 214, p95 792, p99
  1,989, max 4,712, with no item above 8,192. Current document-order batch 8
  nearly doubles work to 964,929 padded tokens and reaches 37,696 padded tokens
  in one batch.
- Length bucketing plus a 4,096-token budget reduces padding to 514,961
  (1.053x actual tokens) and peak batch tokens by 87.5%, so it would materially
  improve memory. But the quadratic attention-work proxy improves only 3.008x,
  below the frozen 4x threshold, and projects 1,646.716 s (27.45 minutes), well
  above the 15-minute bounded-follow-up threshold and 10-minute product goal.
- The predeclared decision is
  `skip_native_qwen_and_screen_lighter_runtime`; no bounded sample or second
  Qwen full build is authorized. Eight protocol/analyzer tests, both validators,
  CI/release outcome wiring, and Ruff pass.

### Priority assessment after closing native Qwen CPU

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2a | Define a two-stage lighter-runtime screen from authoritative model/runtime evidence: first query + bounded stress documents, then full semantic index only for one winner | Very high | Avoids another 82-minute blind run and unnecessary downloads | Do next |
| P1.3e-2b | Prefer an INT8/ONNX candidate that preserves Qwen semantics and one materially smaller multilingual model; cap candidates before measurement | Very high | Separates execution-engine gain from model-quality tradeoff | Freeze before downloads/embeddings |
| P1.3e-3 | Require absolute ≤10-minute projected/full build, ≤6 GiB RSS, ≤1 s query p95, and the same Recall/MRR/nDCG/wrong-source gates | Very high | Relative improvement alone is insufficient | Mandatory |
| P1.3d-2/P1.3c | Preparation and user control | High but blocked | Still no deployable Dense runtime | Remain blocked |
| P2 progress architecture | Add token/chunk heartbeat, ETA, cancellation, and checkpoint semantics to future preparation service | High | 20+ minute document-level silent spans observed | Design with the winning runtime, not current benchmark |

### 2026-09-03 — P1.3e-2b-3 complete: official Qwen quint8 ONNX rejected at Stage 1

- Consumed the sole authorized Qwen Stage-1 attempt after the zero-inference
  preflight and retained `retrieval_lighter_runtime_stage1_qwen_v1.json` at
  normalized SHA-256
  `d19e9576d0e2b272b27d155d76d7840c95d64374dd5af8f0a4f6df7400e0c9bb`.
  Its stable result fingerprint is
  `9bb2553b02b0b2d59faabb352cd19dcae719f01ab71ea9edd398d70cc1ba8beb`;
  validate-only mode, eight worker/orchestrator tests, and Ruff pass without
  performing inference.
- Operationally, provider load (2,570.809 ms), query p95 (110.332 ms), and
  peak RSS (1,890.293 MiB) pass. The 64-parent stress encode took
  50,134.572 ms, however, and the token/attention-work projection is
  1,178,381.641 ms (19.64 minutes) or 1.419744 chunks/s for the frozen
  1,673-parent corpus. Both fail the absolute 10-minute and 2.75 chunks/s
  gates, so a Qwen ONNX full build is prohibited.
- Compatibility also fails decisively: document-vector cosine against the
  accepted float32 CPU index is 0.795373 mean and 0.466472 minimum, below the
  frozen 0.98/0.95 requirements. Protected semantic Recall@10 gain is only
  0.083333 in each scope versus the required 0.15, while Recall@5 and wrong-
  source protections remain intact. The retained decision is
  `rejected_stage_1`; no product setting, preparation action, or ordinary-user
  surface was enabled.
- A read-only failure analysis over retained values finds that cosine declines
  strongly with token count (Pearson r = -0.775502). The 13 qrel parents average
  0.825578; the longest selection stratum averages 0.724687 and contains the
  three worst rows at 1,012, 2,094, and 1,590 tokens. The complete corpus token
  count still reproduces the prior exact 489,070 total with zero truncation, so
  sample ordering or tokenizer drift is not a credible explanation.
- Static tracing of the official ONNX graph shows that `sentence_embedding`
  gathers the last valid `token_embeddings` position from `attention_mask` and
  performs L2 normalization. The accepted model's pooling configuration is
  also last-token. Therefore changing application-side pooling would duplicate
  the graph and is not an evidence-backed rescue; the failure is consistent
  with sequence-length-sensitive quantized/exported representation drift.

### Priority assessment after Qwen ONNX rejection

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-1 | Compare the authorized E5-small fallback against retained multilingual-E5-base and parent-multivector evidence before downloading or embedding | Very high | A smaller member of a previously weak model family may be dominated; this check costs zero inference | Do next |
| P1.3e-2c-2 | If not dominated, freeze E5-small Stage-0/Stage-1 representation and truncation rules before its artifact is touched | Very high | Its 512-token limit cannot silently discard evidence tails | Conditional |
| P1.3e-3 | Reserve the batch's single full build only for a Stage-1 candidate that passes every operational gate | Very high | Qwen consumed no full build, so the budget remains available | Blocked pending fallback screen |
| P1.3d-2/P1.3c | Preparation service and explicit retrieval choice | High but blocked | No local dense runtime has passed deployability | Remain blocked; BM25 stays default |
| P1.4/P2 | Retrieval evaluation and architecture decomposition | High | Can progress without exposing a failed dense path to users | Resume if the bounded fallback is dominated |

### 2026-09-03 — P1.3e-2c-1 complete: E5-small fallback retained, but only with explicit tail coverage

- Compared the authorized fallback with retained E5-base evidence without
  loading a tokenizer or model. On the frozen 12-case semantic-paraphrase set,
  E5-base raw Dense reached Recall@10 0.500 in both modes versus product BM25
  0.375 scoped and 0.291667 full-library. It therefore contains a real semantic
  signal, but safe RRF reached only 0.375 in both modes and missed the required
  gain. This is a high-risk candidate family, not proof that E5-small is
  dominated before measurement.
- The prior E5-base parent-collapsed multivector development experiment is also
  negative evidence: full-library raw Recall@10 regressed by 0.085366 and safe
  fusion gained only 0.012195, while representation count grew 1.742781x. That
  exact route remains excluded and will not be silently revived.
- E5-small's 512-token limit makes a simple prefix-truncated parent index
  product-invalid even if truncation is counted: the user asks to find evidence
  anywhere in a library paper, so unembedded tails would create unexplained
  blind spots. The fallback remains eligible only under an E5-specific frozen
  representation that token-windows every parent with deterministic overlap,
  accounts for every window in runtime/RSS projections, collapses retrieval
  scores back to the immutable parent id, and returns the complete source chunk.
  This is a new candidate-specific hypothesis, not a claim that the prior E5-
  base multivector result passed.

### Priority assessment after E5 evidence review

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-2a | Run E5-small zero-inference Stage 0 with exact artifact/hash, ONNX checker, graph I/O, tokenizer metadata, and CPU-only provider receipt | Very high | Cheaply rejects incompatible supply/runtime before paper text or embeddings | Do next |
| P1.3e-2c-2b | Freeze E5 Stage-1 full-coverage window geometry, parent collapse, workload projection, and one-shot receipt | Very high | Prevents post-hoc window tuning and hidden 512-token tail loss | Only if Stage 0 passes |
| P1.3e-2c-2c | Run the sole bounded E5 Stage 1 | Very high | Must pass operational gates before consuming the one full-build budget | Conditional |
| P1.3e-3 | Build/evaluate one complete E5 index | Very high | Only route to model-quality evidence; no parent index can be reused | Blocked on Stage 1 |
| Preparation/UI | Remain blocked | High | Candidate risk is still unresolved | BM25 remains default and only ordinary-user path |

### 2026-09-03 — P1.3e-2c-2a complete: E5-small ONNX Stage 0 passed and pinned

- Generalized the Stage-0 preparation path so a candidate's tokenizer/config
  dependencies are fetched from the same immutable model revision and hashed
  before any measurement. The Qwen report remains backward-compatible and its
  validator unchanged.
- Downloaded and verified the official E5-small ONNX artifact at exactly
  470,268,510 bytes and SHA-256
  `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665`.
  The single transfer completed without retry. The five official ONNX-folder
  tokenizer/config files total 22,153,044 bytes and now have individual size
  and SHA-256 identities in the retained report.
- ONNX checker passed in 3,436.988 ms and CPUExecutionProvider-only session
  creation took 496.298 ms. The graph accepts `input_ids`, `attention_mask`,
  and `token_type_ids`, and exposes only float32 `last_hidden_state` with shape
  `[batch, sequence, 384]`; Stage 1 must therefore implement and test masked-
  mean pooling and normalization explicitly.
- Retained `retrieval_lighter_runtime_stage0_e5_small_v1.json` at normalized
  SHA-256
  `5d901b01af8201004adde29119321b60fb82a781b54e93688b22f25d3fde963d`.
  It records zero tokenizer loads, zero session runs, zero paper-text access,
  zero embeddings/vectors, and no product exposure. Its standalone validator
  is wired into CI and Windows release validation; 15 focused tests, Ruff, and
  diff checks pass.

### Priority assessment after E5-small Stage 0

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-2b-1 | Measure tokenizer-only full-coverage window counts from the frozen source, then freeze E5 Stage-1 geometry and runtime arithmetic before inference | Very high | Prevents choosing overlap/batch geometry after observing quality or speed | Do next |
| P1.3e-2c-2b-2 | Implement isolated masked-mean E5 worker, parent-max collapse, hard timeout, source/artifact integrity, and one-shot report | Very high | Graph has no sentence embedding and no reusable parent index | After protocol freeze |
| P1.3e-2c-2c | Execute one bounded E5 Stage 1 | Very high | Only an operational pass may unlock the batch's single full build | Conditional |
| P1.3e-3 | Full-build semantic evaluation | Very high | Still blocked; Stage 0 is compatibility evidence only | No change |
| Product preparation/UI | Remain blocked | High | No quality-qualified local runtime | BM25 remains default |

### 2026-09-03 — P1.3e-2c-2b-1a complete: E5 full-coverage window census frozen before tokenizer load

- Froze `retrieval_e5_small_window_census_protocol_v1.json` at normalized
  SHA-256
  `ca464188ca3b6d8ec89e61f07fd7d0b9b4c08f79b17481951fe883f5030d3fe0`
  before loading the tokenizer or source corpus. It binds the Qwen rejection,
  E5 Stage-0 result, 64-parent stress manifest, tokenizer hash, and exact
  10-document/1,673-parent source tree.
- The precommitted geometry uses 448 raw content tokens, nominal stride 384,
  and at least 64-token overlap. Prefix and two special tokens are added outside
  that budget; content token ids are sliced directly and never decode/re-
  encoded. The final start is pulled back to the tail when needed so every raw
  content token is represented at least once.
- The census may load only the Rust tokenizer and read the frozen source. It
  cannot create an ONNX session, run a model, create a vector, or retain paper/
  query text. It rejects any uncovered token, truncation, input above 512,
  window-count multiplier above 2.0x, or raw vector projection above 10 MB.
  Four mutation tests, direct validation, and Ruff pass.

### Priority assessment after E5 window geometry freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-2b-1b | Implement and run the tokenizer-only census once; pin workload, coverage, and decision | Very high | Determines whether the representation is operationally bounded before model inference | Do next |
| P1.3e-2c-2b-2 | Freeze/implement E5 Stage 1 only if census passes | Very high | Prevents measurement under an unbounded representation | Conditional |
| P1.3e-3 and product controls | Remain blocked | High | Still no runtime or quality evidence | No change |

### 2026-09-03 — P1.3e-2c-2b-1b complete: tokenizer-only E5 window census passed

- Ran the frozen census once in the isolated Python 3.10.11 environment. The
  1,673 parents contain 473,464 E5 content tokens (p50 220, p95 730, p99
  1,954, max 4,050) and deterministically expand to 2,016 windows, a 1.205021x
  multiplier. The 64-parent stress sample expands to 74 windows.
- All 473,464 content tokens are covered, minimum observed overlap is 64, no
  parent or query is truncated, and the largest model input is 453 tokens. The
  12 queries have p50/p95/max input lengths 46/57/57. Returned evidence remains
  the complete immutable parent; no child text is exposed or retained.
- The projected raw float32 384-dimensional vector payload is 3,096,576 bytes,
  below the 10 MB census limit. This is not an actual index-size or quality
  claim; those remain Stage-2 gates.
- Retained `retrieval_e5_small_window_census_v1.json` at normalized SHA-256
  `bd32ae3d12bad8d67203f9493da33f2ab0b57ac8ae4616992efdf4fb9745ee25`
  and result fingerprint
  `9ff2c8f4dc9f81ade0cbae5286ade071a1d0e8fcfeaecc9cae6ba6f33b33c3cf`.
  It records one tokenizer load and zero model/session/inference/vector calls.
  Twelve helper/protocol/report tests, validate-only, CI/release wiring, Ruff,
  and diff checks pass.

### Priority assessment after E5 window census

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-2b-2a | Freeze E5 Stage-1 one-shot protocol using the exact 2,016/74-window workload, masked-mean pooling, parent-max collapse, hard timeout, and full integrity checks | Very high | Last point to bind semantics before model inference | Do next |
| P1.3e-2c-2b-2b | Implement isolated worker/orchestrator and synthetic tests | Very high | Must prove pooling, coverage, projection, and one-shot behavior independently | After protocol |
| P1.3e-2c-2c | Execute one bounded Stage-1 measurement | Very high | Runtime pass is required before the single full build | Conditional |
| P1.3e-3 | Full E5 index plus semantic evaluation | Very high | No reusable index; only allowed after Stage 1 | Blocked |
| Product preparation/UI | Remain blocked | High | Census is workload evidence, not deployability | BM25 remains default |

### 2026-09-03 — P1.3e-2c-2b-2a complete: E5-small Stage-1 protocol frozen

- Froze `retrieval_lighter_runtime_stage1_e5_small_protocol_v1.json` at
  normalized SHA-256
  `01309083feb769ef99cfe61a0dd94b44b193c772e540bc5740c6f7ac31f5ad39`.
  It binds Stage 0, the pre-inference census and fingerprint, stress manifest,
  tokenizer/artifact/source identities, exact 2,016/74-window workloads, and
  the 12-query fixture.
- The worker must create only a CPUExecutionProvider session, explicitly apply
  attention-masked mean pooling plus float32 L2 normalization to
  `last_hidden_state`, and use right padding with zero token-type ids. Full-
  corpus tokenization verifies workload/coverage, but only the 74 stress
  windows and 12 timed queries may be embedded.
- Runtime projection is bound to full/stress padded-attention-work arithmetic,
  with hard 600-second termination, load ≤30 seconds, query p95 ≤1 second,
  peak RSS ≤4 GiB, projected complete build ≤10 minutes, and ≥2.75 parent
  chunks/s. Exact window counts, zero truncation/uncovered tokens, finite unit
  vectors, and unchanged source/artifact/tokenizer are mandatory.
- A Stage-1 pass authorizes only one atomic full build. Stage 2 must report raw
  Dense and unchanged protected-tail fusion, gain Recall@10 by at least 0.15 in
  both modes, preserve product Recall@5 and wrong-source@5, and avoid MRR/nDCG@10
  regressions versus product. It still cannot enable the product. Five protocol
  mutation tests, CI/release validation, Ruff, and diff checks pass.

### Priority assessment after E5 Stage-1 freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-2b-2b | Implement isolated E5 worker and parent orchestrator with pooling, batching, projection, RSS, integrity, timeout, and one-shot tests | Very high | Required before consuming the sole Stage-1 run | Do next |
| P1.3e-2c-2c | Preflight then execute one Stage-1 run | Very high | Decides whether the single full build is allowed | After implementation |
| P1.3e-3 | Complete index plus quality replay | Very high | Still blocked | Conditional |
| Product/UI | Remain blocked | High | No runtime/quality pass | No change |

### 2026-09-03 — P1.3e-2c-2b-2b complete: E5 Stage-1 worker/orchestrator implemented

- Added an isolated E5 worker that verifies the environment, source, artifact,
  and all five tokenizer files before and after execution. It constructs
  windows directly from token ids, adds prefix/BOS/EOS without decode/re-
  encoding, right-pads batches, supplies zero token-type ids, and performs
  attention-masked mean pooling plus float32 L2 normalization itself.
- The worker tokenizes all parents to reproduce the exact 2,016-window census,
  but embeds only 74 frozen stress windows. It performs one public warm-up and
  12 timed query runs, records unit-vector summaries/hashes rather than raw
  vectors or text, measures peak RSS, and projects the complete build using
  exact full/stress padded quadratic attention-work.
- Added a parent orchestrator with an isolated-environment preflight, atomic
  one-shot receipt, real 600-second subprocess termination, independent post-
  execution integrity hashes, gate recomputation, atomic tracked report, and a
  latency-insensitive outcome fingerprint. Failure after receipt permanently
  closes the attempt and cannot silently authorize Stage 2.
- Six new worker/orchestrator tests pass, covering tail coverage, masked pooling,
  shape/zero-vector rejection, complete/slow gate branches, and stable
  fingerprint behavior. Compilation and Ruff pass. No attempt receipt, worker
  result, Stage-1 report, session run, or vector exists yet.

### Priority assessment after E5 Stage-1 implementation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-2b-2c | Run final zero-inference preflight and verify execution artifacts are absent | Very high | Last safe point before the sole measurement | Do next |
| P1.3e-2c-2c | Consume and retain the single bounded Stage-1 run | Very high | Decides full-build authorization | Immediately after preflight |
| P1.3e-3 | Implement/run one full index and quality replay only on Stage-1 pass | Very high | Still conditional | Blocked |
| Product/UI | Remain blocked | High | No quality-qualified runtime | No change |

### 2026-09-03 — P1.3e-2c-2b-2c complete: final E5 Stage-1 preflight passed

- Revalidated the frozen protocol, Stage-0 report, census report/fingerprint,
  stress manifest, exact source tree, 470,268,510-byte ONNX artifact, all five
  tokenizer files, and the isolated 24-package Python 3.10.11 environment.
- Confirmed that the E5 attempt receipt, ignored worker result, and tracked
  Stage-1 report are all absent. The preflight did not create an ONNX session,
  run a model, access a vector, or consume the one-shot attempt.

### Priority assessment after final E5 preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-2c-2c | Execute exactly one bounded E5 Stage-1 measurement and persist pass/reject | Very high | All preconditions pass | Do now |
| P1.3e-3 | Follow only the retained Stage-1 authorization | Very high | No speculative full build | Conditional |
| Product/UI | Remain blocked | High | Stage 1 is not quality evidence | No change |

### 2026-09-03 — P1.3e-2c-2c complete: E5-small Stage 1 passed; one full build authorized

- Consumed the sole E5 Stage-1 attempt and retained
  `retrieval_lighter_runtime_stage1_e5_small_v1.json` at normalized SHA-256
  `19a42c13a779e4eabb37438d45749485fa6864f85d3c924c6f5a6163be1502cc`
  with result fingerprint
  `0750c32c42c09f8f9613734662ff1e6c3c7bee1169c9341b51e5afed778cd645`.
- Provider/tokenizer load took 998.062 ms, query p50/p95 were 9.517/12.104
  ms, the 74 stress windows encoded in 4,040.255 ms, and peak process RSS was
  1,168.598 MiB. The exact attention-work projection for all 2,016 windows is
  96,977.377 ms (1.62 minutes), equivalent to 17.251446 parent chunks/s.
- All 22 operational/integrity checks pass: exact parent/window counts, 453-
  token maximum input, zero truncation and uncovered content, finite normalized
  384-dimensional query/stress vectors, CPU-only provider, no full-corpus
  embeddings or quality evaluation, and unchanged source/artifact/tokenizer.
- The retained decision is `passed_stage_1_authorize_single_full_build`. It
  authorizes exactly one complete build and quality replay; it does not authorize
  product preparation, a user setting, or Hybrid retrieval. Validate-only, ten
  worker/runner/report tests, CI/release wiring, Ruff, and diff checks pass.

### Priority assessment after E5 Stage-1 pass

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-3a | Freeze the single full-build protocol/index format and exact raw-Dense + protected-tail semantic gates before execution | Very high | Runtime passed, but model quality remains the decisive risk | Do next |
| P1.3e-3b | Implement atomic window index, parent-max retrieval adapter, full-build worker, replay, and failure retention | Very high | Must return complete parent chunks and preserve BM25 protections | After protocol |
| P1.3e-3c | Consume the one full build and retain accept/reject | Very high | Final local Dense deployability decision | Conditional |
| P1.3d-2/P1.3c | Preparation service and explicit user choice | High but blocked | Stage 1 has no Recall/MRR/nDCG evidence | Remain blocked |
| P1.4/P2 | Broader eval and architecture decomposition | High | Resume after the candidate decision, regardless of pass/fail | Next program increment |

### 2026-09-03 — P1.3e-3a complete: the single E5 full-build/quality protocol is frozen

- Reused the existing `ParentCollapsedDenseRetriever` contract: one window is
  supplied as the parent representation and remaining windows as children;
  max cosine is collapsed to the unchanged complete parent chunk. No new
  product retriever or child-text result surface is required.
- Froze `retrieval_lighter_runtime_stage2_e5_small_protocol_v1.json` at
  normalized SHA-256
  `84647d00ef87a3aae2f4bce1208491a813c9d08f6e1baff527804a9aa05a013b`.
  Its first validation caught a pre-execution source-hash field mix-up; before
  any receipt/index/inference it was corrected to bind the parent-collapse and
  protected-tail implementations independently. Five mutation tests and the
  final validator pass.
- The benchmark-only index stores exactly 2,016 normalized float32 rows plus
  parent id/position/window ordinal metadata, never paper/query text. A staging
  generation must be completely fsynced and atomically renamed before an
  atomic `current.json` pointer; readback must verify shape, hashes, and parent
  mapping.
- The sole build must finish in ≤10 minutes at ≥2.75 parent chunks/s, query p95
  ≤1 second, peak RSS ≤6 GiB, and total index ≤10 MB with zero truncation or
  uncovered content. Quality is evaluated on the 12 frozen semantic cases for
  raw Dense (diagnostic) and unchanged protected-tail fusion. Promotion
  requires ≥0.15 Recall@10 gain in both modes, exact product Recall@5 and wrong-
  source@5 preservation, plus no MRR or nDCG@10 regression versus product.
- Even a pass authorizes only optional preparation design; it cannot enable a
  product mode or ordinary-user UI.

### Priority assessment after Stage-2 protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-3b | Implement atomic full-build worker, readback, fixed-query parent-collapse replay, one-shot report, and tests | Very high | Required before consuming the batch's only full build | Do next |
| P1.3e-3c | Final preflight and single execution | Very high | Decides local Dense feasibility | After implementation |
| P1.3d-2/P1.3c | Preparation and explicit user choice | High but blocked | Still no quality result | No change |

### 2026-09-03 — P1.3e-3b complete: atomic full-build worker and semantic replay implemented

- Added a dedicated full-build worker that embeds all 2,016 windows exactly
  once, places results back into immutable parent/window order regardless of
  batch execution order, and writes a text-free little-endian float32 matrix
  plus parent mapping and manifest into a staging generation.
- Publication atomically renames the complete generation and then atomically
  replaces `current.json`. A fresh readback verifies pointer/manifest/mapping/
  vector hashes, exact row shape, sequential window ordinals, unit-vector
  summary, and total on-disk bytes before success is returned.
- Added the parent orchestrator that reconstructs first-window parent vectors
  plus remaining child vectors, reuses `ParentCollapsedDenseRetriever`, and
  evaluates frozen product BM25, raw Dense, and unchanged protected-tail
  channels. It rejects any drift in the accepted BM25 control before scoring
  the candidate gates.
- Runtime, integrity, semantic quality, one-shot failure retention, and final
  decision arithmetic are independently recomputed outside the worker. Raw
  query and document vectors stay only in ignored artifacts/index and never
  enter the tracked report. Five atomic-index and quality-gate tests,
  compilation, and Ruff pass. No full-build attempt or index exists yet.

### Priority assessment after Stage-2 implementation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-3b-preflight | Run zero-inference final preflight for all contracts, environment, source/artifact/tokenizer, and absent execution artifacts | Very high | Last safe point before the batch's only full build | Do next |
| P1.3e-3c | Consume the one full build and pin runtime + Recall/MRR/nDCG outcome | Very high | Terminal candidate decision | Immediately after preflight |
| Product preparation/UI | Remain blocked | High | Await final outcome | No change |

### 2026-09-03 — P1.3e-3b-preflight complete: final full-build preflight passed

- Revalidated the Stage-2 protocol, Stage-1 authorization/fingerprint, Stage-0
  dependencies, accepted BM25 control, exact source/artifact/tokenizer, and the
  isolated Python environment.
- Confirmed the Stage-2 attempt receipt, worker output, tracked report, and
  atomic index root are all absent. No tokenizer/model/session/vector/index was
  created by preflight, so the single full-build budget remains unconsumed.

### Priority assessment after final full-build preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3e-3c | Execute and retain the single complete E5 build plus frozen semantic replay | Very high | All preconditions pass | Do now |
| P1.3d-2/P1.3c or P1.4/P2 | Branch only on the retained pass/fail | Very high | No post-hoc rescue or UI exposure | Conditional |

### 2026-09-03 — P1.3e-3c complete: E5-small full build and protected retrieval pass

- Consumed the candidate batch's single full build. The atomic generation
  `7ff6db9e9b40429d9aa55e2feca002fa` contains exactly 2,016 normalized
  384-dimensional window vectors for 1,673 parents. It was read back with
  matching manifest/window/vector hashes and occupies 3,254,053 bytes; no paper
  or query text is stored in the index.
- Provider load took 1,003.998 ms. Tokenization, complete embedding, atomic
  publication, and readback took 99,840.956 ms (1.66 minutes), or 16.756651
  parent chunks/s; peak process RSS was 1,240.867 MiB. All runtime, exact-count,
  coverage, unit-vector, atomicity, and source/artifact/tokenizer integrity gates
  pass.
- Raw Dense is useful but unsafe alone: scoped/full Recall@10 are
  0.583333/0.500000, while full-library wrong-source@5 rises to 0.283333 from
  product BM25's 0.183333. This confirms the product must not expose unprotected
  Dense as a normal mode.
- The unchanged protected-tail fusion passes every frozen quality gate. Scoped
  Recall@10 rises from 0.375000 to 0.625000 (+0.250000); full-library rises
  from 0.291667 to 0.458333 (+0.166666). Recall@5 remains exactly
  0.250000/0.166667 and wrong-source@5 exactly 0.000000/0.183333. MRR improves
  from 0.111442/0.187831 to 0.141435/0.208565, and nDCG@10 from
  0.171936/0.187177 to 0.248568/0.238724.
- Retained `retrieval_lighter_runtime_stage2_e5_small_v1.json` at normalized
  SHA-256
  `9cbccb2413da6ff66725120b529cae0b111adf7fe9947a67cb1e876ee357c73e`
  and result fingerprint
  `18fdba85d35503d2f5f60e38191df412482ec7fc1783341b4354fa0f37848a05`.
  The decision is
  `runtime_candidate_eligible_for_optional_preparation_design`; product remains
  disabled. Validate-only, nine full-build/quality/report tests, CI/release
  wiring, Ruff, and diff checks pass.

### Priority assessment after the deployable-runtime decision

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-1 | Build a product preparation service around the passed E5 runtime: optional download, exact hashes, progress/ETA, cancellation, atomic per-library index, stale detection, and repair | Very high | Converts the proven runtime into a reliable library operation without silent waiting | Do next |
| P1.3f-2 | Add an explicit retrieval choice only after readiness: BM25 default; “本地语义增强” uses protected-tail only; external model/web remains separately consented | Very high | Gives the user control without presenting unavailable or unsafe raw Dense | After preparation service and shadow replay |
| P1.3f-3 | Run product-shadow operational replay against the production adapter and expand judged cases/failure slices | Very high | The frozen 12-case pass is necessary but small; production wiring must not change rankings or library behavior | Before ordinary-user enablement |
| P1.3f-4 | Keep the ~470 MB model and ~22 MB tokenizer optional, verified, and out of the base Windows bundle; measure minimal ORT dependency footprint | High | Avoids inflating every installation for an opt-in feature | With preparation design |
| P1.5 | Cross-encoder reranker | Medium, blocked | First-stage retrieval now passes, but reranking should wait for production shadow evidence and an independent eval slice | Later |
| P1.4 | Retrieval/answer/citation evaluation and local OpenTelemetry-compatible observability | High | Continue Recall/MRR/nDCG plus faithfulness/citation and latency traces without external telemetry by default | Parallel next increment |
| P2 architecture | Decompose large API/runtime files along stable service boundaries | High | New dense preparation must land as isolated services, not add more code to oversized routers | Mandatory during productization |

### 2026-09-03 — P2 guardrail repair complete: retrieval observation extracted without raising budgets

- Extracted retrieval selection, trace publication, observability, shadow
  capture, cancellation, and failure recording from `kb/task_runtime.py` into
  the focused `kb/task_retrieval_observation.py` service. Structured exact and
  structured section fallback reasons remain unchanged at their public
  boundary.
- Moved the final claim/evidence audit orchestration into
  `kb/answer_finalization_outcome.py` and kept
  `kb/generation_answer_finalize_runtime.py` within its existing line budget.
  No architecture budget was increased: `task_runtime.py` is 10,504/10,529
  lines and `generation_answer_finalize_runtime.py` is 10,721/10,721 lines.
- Restored the small frontend budget overages through type/format compaction
  only, with no user-visible behavior change. The architecture budget, frontend
  lint/build, 229 answer-finalization tests, and diff checks pass.
- After extraction, the focused retrieval selector/observability/shadow suite
  passes 23/23 and the task answer-contract, provenance, citation grounding,
  sanitization, display-merge, source-binding, and background-task suite passes
  471 with 1 skipped. This explicitly checks the product path before starting
  the preparation service.

### Priority assessment after architecture guardrail repair

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-1a | Freeze the optional local-semantic preparation lifecycle and failure contract before adding product code | Very high | Prevents a 470 MB download or stale index from becoming an unexplained user-visible state | Do next |
| P1.3f-1b | Implement the preparation service as new bounded modules with atomic per-library generations, progress/ETA, cancel/resume, stale detection, and repair | Very high | Enables dependable local preparation while respecting architecture budgets | After contract freeze |
| P1.3f-3a | Replay production selection/shadow behavior against the prepared-index adapter and expand judged failure slices | Very high | Twelve frozen cases are too small for ordinary-user enablement | Before any UI mode |
| P1.3f-2 | Expose BM25 default versus ready-only “本地语义增强”; keep external web/model consent separate | High, blocked | A setting without a ready/repairable index would repeat the unresolved-status problem | Only after preparation and shadow pass |
| P1.4 | Extend retrieval/citation evaluation and local telemetry | High | Gives the next promotion decision broader evidence | Parallel after the lifecycle contract |

### 2026-09-03 — P1.3f-1a complete: optional preparation lifecycle frozen

- The integration audit found that the existing hidden product adapter still
  targets the rejected Qwen/CUDA/single-vector candidate. It must not be reused
  for the passed E5-small CPU/multi-window candidate. The accepted path now has
  an isolated model root and `<db_dir>/local_semantic_index/e5_small_v1`; it is
  explicitly separate from the legacy `dense_index` format.
- Froze `retrieval_local_semantic_preparation_protocol_v1.json` at raw-file
  SHA-256
  `75416104e64b461106be8161630adf471dd1eda8524229135ea9bedb54775a06`.
  It pins the immutable model revision, all six file sizes and hashes, the
  492,421,554-byte cap, CPU runtime versions, full-coverage representation,
  atomic index rules, restart reconciliation, and promotion gates.
- Model download requires an explicit preparation action and can never begin
  during a query. A partial transfer is resumable only after a valid Range
  response; every file is size- and SHA-256-verified before atomic publication.
  Cancellation keeps verified files and a valid partial transfer while
  preserving the previous ready index.
- Defined eleven public lifecycle states. Every non-ready state has one
  deterministic remediation (`prepare`, `cancel`, `wait`, `resume`, `rebuild`,
  `retry`, or `update_app`); paths and raw exceptions are never public. Only
  `ready` is available, and missing ORT/tokenizer dependencies hide the mode
  control instead of presenting an action the current build cannot complete.
- Added the executable public-status contract, protocol validator, six mutation
  cases, CI/release validation, and 19 focused tests. All pass. Product and UI
  remain disabled.

### Priority assessment after lifecycle freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-1b-1 | Implement a durable per-library preparation ledger plus verified artifact transport with progress/ETA, cancel, valid resume, and restart reconciliation | Very high | Turns every displayed preparation state into a system-owned operation | Do next, with fake local transport first |
| P1.3f-1b-2 | Implement the E5 CPU multi-window provider and atomic parent-window product index under the new format | Very high | Required to replace, not extend, the obsolete Qwen product adapter | After transport passes |
| P1.3f-1b-3 | Add management API and automatic stale/repair orchestration; never expose raw errors | High | Makes the lifecycle usable without asking the user to diagnose internals | After ledger/index |
| P1.3f-3a | Production shadow replay and larger judged failure slices | Very high | Still required before a user-visible retrieval choice | Blocked by prepared runtime |
| P1.3f-4 | Verify portable Windows ORT/tokenizers footprint and packaging | High | Missing runtime must be solved by the release, not by the user | Before UI promotion |

### 2026-09-03 — P1.3f-1b-1 complete: durable ledger and verified transport implemented

- Added a per-library SQLite preparation ledger with a single-active-task
  transaction, duplicate-start reuse, categorical/path-free public state,
  monotonic progress fields, cancellation that wins over late callbacks, and
  restart reconciliation to a resumable `interrupted` state. Unknown future or
  corrupt states fail closed to `repairable_error` with a `retry` action.
- Added the production HTTPS transport boundary for the six immutable E5 files.
  It sends bounded connect/read timeouts, disables content encoding, and uses
  Range only for an existing partial file. A resume is accepted only for an
  exact `206 Content-Range`; otherwise the partial is safely restarted from
  zero.
- Downloads remain `<file>.part` until exact length and SHA-256 both pass, then
  use atomic replacement. Cancellation fsyncs and retains the partial; network
  details are reduced to categorical reason ids; a corrupt response cannot
  replace an existing file or create the ready manifest. Repeated preparation
  reuses verified files without a network call.
- All tests use local fake transports and tiny artifacts; no model download was
  performed. Thirteen ledger/download/resume/cancel/restart/hash/oversize/
  transport tests and focused Ruff pass.

### Priority assessment after verified transport

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-1b-2a | Extract the already-measured E5 tokenizer/window/ONNX executor into bounded product modules and prove vector parity with the retained Stage-2 result | Very high | Prevents benchmark/product representation drift | Do next |
| P1.3f-1b-2b | Build/read an atomic, text-free parent-window index with cancel and corpus-change protection | Very high | Produces the actual repairable library asset | Immediately after executor parity |
| P1.3f-1b-3 | Orchestrate artifact + index lifecycle behind management APIs | High | Converts internal operations into deterministic user actions | After index tests |
| P1.3f-3a | Production shadow replay and expanded slices | Very high | Required before showing retrieval mode | Still blocked |

### 2026-09-03 — P1.3f-1b-2a complete: product E5 executor has exact benchmark parity

- Added a bounded CPU-only E5 runtime module with the frozen 448-token content
  windows, 384-token stride, tail-pulled final window, repeated passage prefix,
  explicit BOS/EOS, length-sorted 8-row/4,096-padded-token batches, attention-
  masked mean pooling, L2 normalization, and parent-position-preserving output.
  Query and document overlength fail categorically; no silent truncation exists.
- The first parity run correctly failed because it compared the current
  2,251-parent working library with the historical 1,673-parent Stage-2 corpus.
  The query vectors were already exact. The gate was corrected to compare the
  product and frozen benchmark implementations on the same current corpus while
  retaining the historical Stage-2 mapping hash as provenance.
- The final real-model parity run used Python 3.10.11, ONNX Runtime 1.23.2,
  tokenizers 0.21.4, the exact pinned artifact/tokenizer, and CPU only. All 12
  product query vectors have the identical float32 matrix SHA-256
  `6374e52e225e65333843be7aedfa5683414a60f3465f63b43434e5e54c245eed`
  and maximum absolute difference 0 from Stage 2.
- On the current 2,251-parent library, both implementations independently
  produced 2,663 windows, maximum input length 453, and identical input-id/
  mapping digest
  `3d013e00fabea7460af59734390e3798bffc9ad5ff748bd3457ca97211bd3b6c`.
  No document vectors or index generation were built in this parity step.
- Retained `retrieval_product_e5_runtime_parity_v1.json` at SHA-256
  `f900750d20f5fbe01a169efab6883feeb8743be08b5334b025be21c9817e16a6`.
  Validate-only is wired into CI/release; twelve executor/report tests and Ruff
  pass. Product/UI remain disabled.

### Priority assessment after executor parity

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-1b-2b | Implement/read/validate the text-free atomic E5 parent-window index, including cancellation and corpus-change rejection | Very high | Last data-plane component before reliable preparation orchestration | Do next |
| P1.3f-1b-3 | Join artifact, executor, index, and ledger behind management service/API | Very high | Gives each status a working system action | After index pass |
| P1.3f-3a | Replay protected-tail production selection and broaden judged cases | Very high | Required before mode visibility | After service wiring |
| P1.3f-4 | Add and measure locked ORT/tokenizers in portable Windows build | High | Removes `update_app` blocker for packaged users | Before user UI |

### 2026-09-03 — P1.3f-1b-2b complete: atomic incremental parent-window index implemented

- Added a dedicated E5 product index at
  `<db_dir>/local_semantic_index/e5_small_v1`; it never reads or mutates the
  legacy Qwen `dense_index`. It reuses the existing canonical corpus snapshot
  logic so stale detection has one definition across vector indexes.
- The manifest stores only chunk ids, text/metadata digests, counts, window
  ordinals, candidate/representation identity, and corpus fingerprints. Paper
  and query text are absent. Vectors are normalized float32 little-endian rows.
- Rebuilds tokenize the current complete parents, reuse every unchanged
  parent's window vectors by text digest, embed only changed parents, restore
  deterministic parent/window order, and recheck the committed corpus before
  publication. A corpus change rejects the generation rather than publishing a
  stale result.
- A fully fsynced staging generation is renamed before an atomic `current.json`
  replacement. Provider/mapping/hash/vector validation is mandatory on read.
  Cancellation is checked during tokenization, before every inference batch,
  before and immediately after the final commit hook; cancellation, inference,
  corpus-change, or pointer failures remove the orphan and preserve the prior
  ready pointer.
- Loading reconstructs `ParentCollapsedDenseRetriever` and returns only the
  unchanged complete parent chunks. Fifteen focused new-index plus existing
  embedding-index regressions and Ruff pass. No real current-library index was
  built and product/UI remain disabled.

### Priority assessment after atomic index implementation

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-1b-3a | Build the thread-safe preparation orchestrator and replace readiness with actionable lifecycle state | Very high | Makes preparation self-healing rather than exposing dead-end statuses | Do next |
| P1.3f-1b-3b | Add internal management endpoints for prepare/status/cancel/resume/rebuild; do not add the mode control | Very high | Enables autonomous E2E exercise without premature feature exposure | After service tests |
| P1.3f-3a | Product shadow replay through the new loaded index and protected-tail selector | Very high | Final ranking/wiring guard before UI | After orchestration |
| P1.3f-4 | Portable Windows dependency gate | High | Packaged runtime still lacks pinned ORT/tokenizers | Parallel before any UI |

### 2026-09-03 — P1.3f-1b-3 complete: actionable preparation service and management API implemented

- Added one cached, thread-safe service per library. It reconciles an abandoned
  task once per process session, verifies artifacts before model execution,
  advances the durable ledger through download/index phases, calculates byte or
  window ETA, and maps every artifact/runtime/index failure to a categorical
  repair action. Raw exceptions and paths never enter the response.
- `prepare`, `resume`, `rebuild`, and `retry` are accepted only when they match
  the current action. Duplicate starts return the active task. Cancellation is
  task-scoped and wins at download, tokenization, inference-batch, corpus-check,
  and commit checkpoints; a completed or unknown task cannot cancel a newer
  operation.
- Added management-authenticated start and cancel endpoints plus actionable GET
  readiness. The GET surface includes status/progress/remediation, but the
  promotion constant remains false: `available_modes` contains only BM25 and
  `show_mode_control` stays false even for a prepared index. Direct query-time
  download remains impossible.
- The architecture budget initially rejected a six-line wrapper added to the
  already frozen `embedding_index.py`. The wrapper was removed without raising
  the budget; the new index explicitly aliases the shared canonical snapshot
  until the planned P2 extraction. The file is now 838/840 and the complete
  architecture budget passes.
- Service success/failure/retry/stale/action validation, deferred cancellation,
  authenticated prepare/cancel API, readiness hiding, index preservation, and
  related regressions pass (17 focused tests in the final combined run). Ruff
  passes. Product retrieval still uses BM25 because the selection adapter has
  not yet been migrated or shadow-approved.

### Priority assessment after preparation orchestration

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3a-1 | Implement the read-only E5 runtime cache and migrate the protected-tail product selector off the obsolete Qwen adapter | Very high | A prepared index is useless until queries use the exact accepted executor, but migration must remain hidden | Do next |
| P1.3f-3a-2 | Build a disposable product-format index and run production selector/shadow replay, including fallback/cold-cache/stale/cancel slices | Very high | Verifies real product wiring rather than benchmark-only components | Immediately after runtime tests |
| P1.3f-4 | Lock ORT/tokenizers into the portable Windows runtime and measure size/startup | High | Current released build cannot prepare E5 | Before any user-visible choice |
| P1.3f-3b | Expand judged semantic and wrong-source slices beyond twelve cases | Very high | Current quality evidence is still too small for ordinary-user promotion | Before UI |

### 2026-09-03 — P1.3f-3a-1 complete: product selector migrated to verified E5 runtime, hidden bypass closed

- Added a read-only E5 runtime cache that requires exact packaged dependencies,
  full artifact hashes, the committed product-format index, and a matching
  current corpus before constructing a retriever. It uses the exact parity-
  tested executor and `ParentCollapsedDenseRetriever`; it never downloads,
  installs, or builds during a query.
- Cache identity includes the committed docs file, E5 index pointer, ready
  artifact manifest, model, and tokenizer signatures. A changed index/model/
  corpus invalidates the cache. Failures are reduced to stable local-semantic
  reason ids and return exact BM25 through the existing fail-open selector.
- Migrated `local_retrieval_selector.py` away from the obsolete Qwen/CUDA
  runtime. The old module is retained only for frozen historical evaluation
  contracts and tests; it is no longer the product selector default.
- Closed the hidden request bypass: while the shared promotion constant is
  false, a caller manually sending `hybrid_local` receives byte-for-byte BM25
  with `local_semantic_not_promoted`. Tests must explicitly enable the candidate
  to exercise protected-tail behavior. The readiness endpoint likewise keeps
  only BM25 visible.
- Runtime dependency/artifact/cache invalidation, protected prefix, scoped
  source isolation, Dense failure, not-promoted fallback, task answer contract,
  and related tests pass (93 in the combined run). Ruff, architecture budgets,
  and diff checks pass.

### Priority assessment after selector migration

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3a-2 | Build one disposable current-corpus product index with the parity-tested executor, then replay the actual selector in candidate-enabled shadow | Very high | Measures product format, cold/warm query, safety fallbacks, and current 2,251-parent behavior | Do next |
| P1.3f-3b | Add judged cases for current-library semantic misses and wrong-source confounders | Very high | Shadow operational success alone is not relevance proof | After runtime replay |
| P1.3f-4 | Portable Windows locked dependency build | High | Still blocks user preparation in release artifacts | Before UI |

### 2026-09-03 — P1.3f-3a-2a complete: actual-product selector shadow protocol frozen

- Confirmed that the mutable working library is no longer the historical test
  corpus: it now contains 31 documents and 2,251 complete parents. The frozen
  ten-document/1,673-parent source still exists separately and matches the
  Stage-2 tree SHA-256 exactly. The protocol therefore forbids using or writing
  the working library and requires an isolated disposable copy of the frozen
  source.
- Froze one and only one completed product-format full build, exact E5 model and
  tokenizer identities, all five product executor/index/runtime/selector/fusion
  source hashes, the 12-case semantic fixture, 24 actual selector calls, cold/
  warm cache behavior, protected-prefix and scoped-source invariants, atomic
  readback, runtime limits, and the Stage-2 relevance gates before execution.
- The candidate may be enabled only inside this offline replay. Passing cannot
  enable the product or expose a mode control; it authorizes only a larger
  judged-case phase. The source evaluation DB, current user DB, external web,
  model download, and user answer evidence remain outside the write boundary.
- Added a validator, six mutation tests, and CI/release validation. The first
  validator run caught that the qrels identity had been recorded as a raw
  Windows byte hash instead of the repository's cross-platform normalized-text
  hash; the frozen value was corrected before any model/index execution. The
  validator, six tests, Ruff, and diff checks pass.

### Priority assessment after selector-shadow protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3a-2b | Implement and dry-test the one-shot isolated worker/report validator, then consume the single product-format build | Very high | Prevents an avoidable 100-second build from being spent on orchestration defects | Do next |
| P1.3f-3b | Expand judged current-library semantic and wrong-source cases | Very high | Required because 12 cases cannot justify ordinary-user exposure | Only if actual selector keeps all frozen gates |
| P1.3f-4 | Lock and measure ORT/tokenizers in the portable Windows runtime | High | Released package still cannot run preparation | Before UI, after relevance remains eligible |
| P2.1 | Extract canonical corpus snapshot from the near-budget legacy index module | Medium | Lowers structural risk without changing retrieval | After the P1 decision point |

### 2026-09-03 — P1.3f-3a-2b preparation complete: one-shot product replay is dry-validated

- Added a one-shot controller plus an isolated Python 3.10.11 worker. The
  controller rejects existing outputs, changed source/qrels/model/tokenizer/
  product source, or dependency drift before it creates an attempt receipt. It
  copies the frozen DB, launches the exact ORT 1.23.2/tokenizers 0.21.4 runtime
  with networking unused, and preserves both source and current user DB.
- The worker uses the production `E5SmallOnnxExecutor`,
  `LocalSemanticWindowIndexStore`, `LocalSemanticRuntimeService`, and
  `select_product_retrieval` rather than evaluation-only vector adapters. It
  performs an atomic product-format build/readback, then sends all 12 questions
  through scoped and full-library product BM25 and the actual selector.
- The content-free trace checks exactly 24 calls, one cold load, at least 23 warm
  cache hits, zero fallback and source escape, protected top-five preservation,
  no mutation of the materialized sparse ranking, and product/UI remaining off.
  The report evaluator independently recomputes runtime and relevance decisions.
- Eleven protocol/runner tests, Ruff, both system and isolated-runtime syntax
  compilation, diff checks, and the complete read-only preflight pass. The
  preflight reconfirmed the frozen source tree and verified local artifacts; no
  full index has yet been consumed in this step.

### Priority assessment before the single full replay

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3a-2c | Consume the single isolated product build, freeze the report, and diagnose every failed gate before choosing the next branch | Very high | This is the first end-to-end test of the exact product data/query path | Do now |
| P1.3f-3b | Expand judged cases only if runtime and 12-case relevance both pass | Very high | Avoids scaling a selector whose product behavior already regressed | Conditional next |
| P1.3f-4 | Package ORT/tokenizers only if the relevance branch remains viable | High | Avoids increasing installer size for a rejected ranking path | Conditional after quality |
| P2.1 | Extract the canonical snapshot boundary | Medium | Worth doing regardless, but it does not answer the current retrieval decision | Hold until P1 result |

### 2026-09-03 — P1.3f-3a-2c pre-execution invalidation resolved without consuming the build

- The first worker process stopped in 3.1 seconds while importing the real
  product BM25 path: the retained model-only environment intentionally lacked
  `rank-bm25`. It failed before model load, tokenization, index directory
  creation, or worker/report output. Completed full builds remain zero; source,
  user library, product behavior, and tracked results were unchanged.
- Retained a tracked pre-execution invalidation record with the original
  protocol/attempt identities and archived the disposable copy under an ignored
  invalidation directory. No evidence was deleted or overwritten.
- The corrected design composes two existing read-only Python 3.10.11
  environments: exact ORT/tokenizers from the accepted model runtime and exact
  `rank-bm25==0.2.2` from the product CI runtime. Both provide the same pinned
  NumPy 2.2.6. No package installation, network call, model/selector tuning, or
  gate relaxation is permitted.
- Refroze the protocol with the invalidation evidence, environment paths,
  versions, and rank-BM25 source SHA-256. Added a drift mutation test. Twelve
  tests, validator, Ruff, diff check, exact overlay import check, and a second
  complete read-only preflight pass. The single completed-build allowance is
  still available.

### Priority assessment after pre-execution correction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3a-2c | Retry the now-complete actual product replay and accept either pass or evidence-backed rejection | Very high | Import boundary is fixed and all expensive-work preconditions now pass | Do now |
| P1.3f-3b | Expand judged cases only after a passing report | Very high | Still the next relevance uncertainty if the selector is viable | Conditional |
| P1.3f-4 | Portable runtime integration | High | Combined runtime proves dependencies but is not yet a distributable package | Conditional after relevance |
| P2.1 | Canonical snapshot extraction | Medium | Independent maintainability gain | Remains behind retrieval decision |

### 2026-09-03 — P1.3f-3a-2c complete: actual product-format E5 selector shadow passed

- Consumed exactly one completed full build over the isolated ten-document
  corpus. The product index atomically committed 1,673 parents and 2,016
  windows; its float32 vector SHA-256 is exactly the accepted Stage-2 matrix
  SHA-256. The conservative attempt-to-pointer build upper bound is 104.691
  seconds and the index occupies 3,681,208 bytes.
- The original worker reached post-build readback and all selector queries but
  its final Windows peak-RSS call lacked explicit 64-bit ctypes signatures. The
  index stayed ready. After fixing only that measurement adapter, a no-op
  recovery reused every vector and repeated the same 24 selector calls; no
  second embedding build occurred. A dedicated recovery record and three
  anti-relaxation tests preserve this chain of evidence.
- Direct recovery measurement of the same two-session index/read/query runtime
  graph peaked at 1,942.508 MiB. The actual selector recorded one cold load, 23
  warm cache hits, zero fallbacks, zero scoped-source escapes, 100% protected
  top-five preservation, 100% unchanged sparse inputs, and 65.36 ms warm p95.
- Scoped Recall@10 improved from 0.375 to 0.625 (+0.25); full-library Recall@10
  improved from 0.291667 to 0.458333 (+0.166666). Recall@5 and wrong-source@5
  stayed equal in both modes; MRR and nDCG@10 did not regress. Every frozen
  runtime and quality check passed.
- Pinned report result fingerprint
  `2baee02e77afd8e58a23313b01d89e119a1c7e2a42c7390eea48ef1fa58463b7`
  and report SHA-256
  `0f99ed8cd4c4a81a1c729b3503c09416cc2f1805a1c0403fdca6ab74f0e1fdcc`.
  CI/release validate both protocol and report. Seventeen focused tests, Ruff,
  and diff checks pass. Product mode and UI control remain disabled.

### Priority assessment after actual product replay

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3b-1 | Freeze a larger, current-library judged-case expansion with explicit semantic-miss and wrong-source-confounder strata before running the candidate | Very high | Twelve curated cases prove wiring, not broad relevance; this is now the dominant promotion risk | Do next |
| P1.3f-3b-2 | Build only the incremental vectors required by the expanded frozen corpus and compare the exact selector | Very high | Measures present user data without spending another blind full build | After fixture validation |
| P1.3f-4 | Produce a portable dependency/size/startup experiment | High | Runtime viability is proven locally, but packaging cost is still unknown | After expanded relevance stays eligible |
| P2.1 | Extract canonical snapshot and BM25 product-query boundaries | Medium-high | The near-budget legacy module and private import remain maintainability debt | Can follow the promotion decision |

### 2026-09-03 — P1.3f-3b-1a complete: larger judged suite audit found a stronger existing asset

- Audited the current 31-document/2,251-parent library against existing frozen
  evaluation assets. `retrieval_user_journey_expansion_v1.json` already matches
  it exactly: 18 traceable evidence-seeking journeys, 12 Chinese and six
  English, spanning four mechanisms, four limitations/boundaries, four
  experimental configurations, four cross-paper comparisons, and two figure
  locations. Four cases require evidence from two papers.
- The suite's product BM25 baseline is deliberately difficult: scoped
  Recall@10 is 0.222222 on calibration and 0.25 on the historical holdout;
  full-library Recall@10 is 0.152778 and 0.166667, with wrong-source@5 of
  0.383333 and 0.333333. It therefore covers the semantic-miss and source-
  confounder risks that a newly synthesized fixture was meant to add.
- Reusing these pre-registered, source-backed labels is stronger than asking a
  model to generate and grade new questions. The six-case holdout was already
  accessed by the earlier Qwen candidate, so the next report will explicitly
  treat all 18 as a locked regression/generalization suite, not claim a new
  untouched holdout and not permit tuning from failures.
- The current DB validates with zero qrels errors and corpus fingerprint
  `20ad35d505d2936dbd80e90df6199e4b3d51fdf4cf8872a36c9211c5d6c99d9f`.
  No model or candidate query was run in this audit.

### Priority assessment after larger-suite audit

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3b-1b | Freeze a one-run, no-tuning E5 product-selector protocol over the existing 18-journey/31-paper suite | Very high | Gains direct current-library evidence without weak self-labelled cases | Do next |
| P1.3f-3b-2 | Build one isolated current-corpus product index and run all 36 scoped/full selector calls | Very high | Decides whether the accepted E5 path generalizes to the actual reading domain | After protocol tests |
| P1.3f-4 | Portable runtime size/startup experiment | High | Justified only if current-domain quality stays safe | Conditional next |
| P2.1 | Retrieval snapshot/query-boundary extraction | Medium-high | Maintainability remains important but no longer blocks this evidence run | After current-domain result |

### 2026-09-03 — P1.3f-3b-1b complete: current-domain E5 protocol frozen

- Froze a single isolated full build for the unchanged E5 executor, product
  index/runtime, protected-tail selector, and exact 31-document/2,251-parent
  corpus. Expected representation size is 2,663 windows; all 18 journeys run in
  scoped and full-library mode for exactly 36 selector calls.
- The quality contract follows the stronger reading-journey regression rule:
  Recall@5, Recall@10, MRR, nDCG@10, and contract Recall@10 may not regress in
  either mode; wrong-source rates at 5 and 10 may not increase; at least one
  mode must gain 0.02 Recall@10. Failures cannot trigger tuning on this suite.
- Runtime limits retain the proven six-GiB/ten-minute/one-second/ten-MB bounds,
  plus exact counts, one cold load, 35 warm cache hits, zero fallback/source
  escape, protected-prefix preservation, immutable sparse input, atomic
  readback, and unchanged source/artifacts.
- The protocol explicitly states that historical Qwen already accessed the old
  holdout, so no new unseen-holdout claim is made. Passing authorizes only a
  packaging experiment, never product/UI enablement. Validator, six mutation
  tests, Ruff, and diff checks pass.

### Priority assessment before current-domain execution

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3b-2 | Dry-test the generalized one-shot runner, then build and replay the 31-paper suite once | Very high | Directly tests the user's actual evidence-reading domain and source confusion | Do next |
| P1.3f-4 | Portable dependency/size/startup experiment | High | Authorized only if this fixed candidate passes current-domain gates | Conditional |
| P2.1 | Extract snapshot and product-query seams | Medium-high | Structural risk remains, but changing them before the evidence run would invalidate source hashes | After report |

### 2026-09-03 — P1.3f-3b-2a complete: current-domain runner dry-validated

- The initial source hash covered unrelated mutable DB content such as reader
  sessions, chat SQLite files, converted Markdown directories, and reference
  caches. Before execution, the protocol was corrected and refrozen to hash and
  copy only committed `docs.json` plus the 31 indexed chunk JSONL files. This
  narrows the experiment to retrieval data and avoids copying or interpreting
  user interaction history.
- The narrowed snapshot is 3,577,031 bytes with SHA-256
  `09d20706edb877e5534f115f92b3252c0917caf23f3e91af55d37d7209557535`;
  the canonical product snapshot independently reports committed-docs SHA-256
  `12677731b639c272dff36c61fb8dbe3e64b69802c9bb19452ec995d049d5ec98`
  and corpus fingerprint
  `7e3261179b44da0e2baa430ea14243ef6885a49ee73d3e921b48bfc57f35bdf5`.
- Added the generalized one-shot controller and stricter case-level harm gates:
  no individual journey may lose Recall@10, contract Recall@10, MRR, or gain
  wrong-source@10 even when aggregate metrics improve. This closes the common
  failure mode where aggregate gain hides a harmed reading question.
- Protocol and runner mutation tests (12), Ruff, system/isolated compilation,
  diff checks, exact corpus/qrels/artifact/dependency validation, and the full
  read-only preflight pass. No model or current-domain index has run yet.

### Priority assessment before consuming the current-domain build

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3b-2b | Consume the one current-domain build and accept the pre-registered case-level verdict without tuning | Very high | Highest remaining relevance evidence before packaging | Do now |
| P1.3f-4 | Portable dependency/size/startup experiment | High | Only valuable if the fixed selector passes all current-domain cases | Conditional |
| P2.1 | Snapshot/query-boundary extraction | Medium-high | Next independent architecture repair | After verdict |

### 2026-09-03 — P1.3f-3b-2b complete: current-domain suite rejects full-library E5

- Completed the single isolated 31-document build in 121.589 seconds. The
  product-format index contains the exact expected 2,251 parents and 2,663
  windows, occupies 4,870,347 bytes, atomically reads back, and the full worker
  wall time was 134.448 seconds. Peak RSS was 2,383.582 MiB.
- All 36 selector operations were operationally clean: one cold load, 35 warm
  cache hits, 83.959 ms warm p95, zero fallback/source-scope escape, 100%
  protected top-five and sparse-input preservation. Every runtime gate passed.
- Scoped retrieval improved Recall@10 from 0.231481 to 0.259259 (+0.027778),
  contract Recall@10 from 0.25 to 0.277778, MRR from 0.202778 to 0.208333, and
  nDCG@10 from 0.145849 to 0.155695 with no case-level harm.
- Full-library retrieval produced no Recall@10, contract recall, MRR, or nDCG
  gain, while wrong-source@10 increased from 0.388889 to 0.411111. Five reading
  journeys received more off-source tail results: CASSI vs Fourier encoding,
  dual-scale mechanism, NAFNet vs classical denoising, perovskite device figure,
  and PILN vs PIDL supervision. The pre-registered case/aggregate gates
  therefore correctly rejected the candidate for full-library use.
- Pinned rejection result fingerprint
  `4bedbf5ae2b76f800474aee13ab190c2268dd3948a2c369eddf9d6feb39d7fcf`
  and report SHA-256
  `5965f9ed52d1674068f37d9df6e170c380708deacc673530e2316acdff62ef6e`.
  CI/release validate this rejection. Fourteen focused tests, validators, Ruff,
  and diff checks pass. Product remains BM25 and no packaging is authorized.

### Priority assessment after current-domain rejection

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3c-1 | Encode a fail-closed product boundary: local semantic retrieval may never replace full-library BM25 under the current evidence | Very high | Prevents future config/UI work from accidentally exposing the five-case wrong-source regression | Do next |
| P1.3f-3c-2 | Design a new untouched scoped-only evaluation set, then reconsider opt-in only for explicit paper selections | High | Scoped E5 helped safely, but existing suite is no longer untouched and gain is modest | After safety boundary; no tuning on current suite |
| P2.1 | Extract retrieval snapshot and product-query seams from near-budget modules | High | Architecture improvement no longer needs to wait for global dense promotion | After scoped safety guard |
| P1.3f-4 | Portable ORT/tokenizers packaging | Deferred | Adding ~492 MB is unjustified while global retrieval is rejected | Do not do until scoped-only evidence passes |

### 2026-09-03 — P1.3f-3c-1 complete: full-library local semantic path is fail-closed

- Added the evidence decision at the production task boundary without changing
  the frozen evaluated selector implementation. A `hybrid_local` request can
  reach the hidden candidate only for an explicit `current_paper`, basket, or
  selected-source scope. `library`, unknown, and other non-explicit scopes
  return the original sparse ranking with categorical reason
  `local_semantic_scope_not_eligible` before loading artifacts or the index.
- Ordinary BM25 requests still use the unchanged path. Structured/translated/
  expanded non-BM25 channels continue to preserve their existing rankings.
  Product promotion remains false, so this is a future-proof safety guard, not
  a user-visible mode change.
- Added direct tests proving full-library requests cannot call the semantic
  runtime, eligible current-paper scope can reach the hidden adapter, BM25 is
  unaffected, and scope normalization is explicit. Sixteen selector/task tests,
  Ruff, all frozen E5 protocol/report validators, diff checks, and the full
  architecture budget pass.

### Priority assessment after full-library fail-closed guard

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3c-2a | Freeze authoring rules for a genuinely new scoped-only evidence set, disjoint in evidence chunks from the 18 already observed journeys | High | Only path that can justify the modest scoped gain without reusing seen outcomes | Do next if labels can remain source-only and auditable |
| P2.1a | Repair historical evaluation provenance so source hashes describe executed snapshots rather than forbidding all future refactors | High | Current live-file hash checks make safe architecture evolution appear as evidence corruption | Parallel design, no source behavior change |
| P2.1b | Extract canonical snapshot/query seams after provenance is durable | High | Reduces private imports and oversized-module pressure | After provenance fix |
| P1.3f-4 | Portable dependency packaging | Deferred | Global mode is rejected and scoped mode lacks fresh acceptance evidence | Keep deferred |

### 2026-09-03 — P1.3f-3c-2a complete: fresh scoped authoring protocol frozen

- The previous 18 journeys used only 19 of the 31 logical documents and 35 of
  2,251 chunks. Identified the exact twelve never-used documents and froze one
  future scoped case per paper before inspecting their chunks. This yields a
  genuinely disjoint evidence source set without adding papers or external
  knowledge.
- Authoring may read only committed chunks in document order and page metadata;
  BM25, dense rankings/scores, external web, and external model knowledge are
  forbidden. Queries must paraphrase rather than copy evidence sentences, and
  every label requires page span and text SHA-256. Six Chinese and six English
  questions are required.
- Future evaluation is scoped-only with the candidate unchanged, at least 0.02
  Recall@10 gain, no aggregate Recall@5/MRR/nDCG/contract-recall regression,
  and zero individual case regression for Recall@10, MRR, or contract Recall.
  Passing still cannot enable the product.
- Validator, five mutation tests, Ruff, and diff checks pass. No source chunk
  from the twelve documents and no product/candidate retriever has been opened
  or executed before this freeze.

### Priority assessment after scoped authoring freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3c-2b | Inspect the twelve sources in committed order, author/validate disjoint qrels, then seal their hash before any retrieval | High | Produces the only honest new acceptance evidence for scoped E5 | Do next |
| P1.3f-3c-2c | Reuse the already-built current-domain index for one scoped-only replay; no additional embedding build | High | Cheaply tests the fixed candidate on unseen documents/evidence | After fixture seal |
| P2.1a | Historical source-snapshot provenance repair | High | Needed before deeper refactor | After scoped verdict |

### 2026-09-03 — P1.3f-3c-2b complete: twelve-paper fresh scoped fixture sealed

- Read only the committed chunk streams and page metadata for the twelve
  documents named by the pre-frozen protocol. Authored exactly one practical
  evidence-location question per paper: six Chinese and six English, covering
  mechanisms, limitations, parameter tradeoffs, experimental choices, concept
  distinctions, and application evidence. No BM25/dense query, prior ranking,
  external web source, or answer generation was used.
- Registered 21 direct evidence chunks across 12 single-paper scopes. Every
  label includes its logical document, exact page span, grade, evidence role,
  contract binding, and source-text SHA-256. All 21 are disjoint from the 35
  labels in the already observed 18-journey suite.
- Sealed `retrieval_scoped_fresh_cases_v1.json` before candidate execution with
  normalized SHA-256
  `3472826090e032739903db91dd8c359adbc68bff6fcbe40ec9f136bb945b1d27`.
  Its 12-document/702-chunk corpus fingerprint is
  `5d65f3a033d62e15c3b90b7c32314452cb49334652d1db0f0e74a55a2b125317`.
- The validator checks the sealed file hash, authoring lineage, locale quota,
  one-case-per-document order, old-suite disjointness, source pages/text
  hashes, contracts, absence of ranking results, and the non-product boundary.
  CI and Windows release now execute it. Twelve protocol/fixture mutation
  tests, Ruff, direct validation, and scoped diff checks pass.

### Priority assessment after fresh fixture sealing

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3c-2c | Replay product BM25 and the unchanged E5 protected-tail selector on the 12 sealed scoped cases using the existing full-library index | Very high | Only remaining honest evidence for a selected-paper semantic option; fixture can no longer move | Do next, one scoped-only run |
| P2.1a | Replace live-source provenance checks with immutable executed-snapshot identities | High | Needed regardless of the scoped verdict and unlocks safe architecture repair | After verdict |
| P1.3f-4 | Measure portable dependency size/startup for selected-paper use only | Conditional | ~492 MB remains unjustified unless the fresh scoped gate passes | Only after pass |
| P1.3f-3d | Design explicit user consent and index lifecycle without exposing full-library dense search | Conditional | Product boundary and opt-in UX matter, but implementation before fresh evidence would be premature | Only after pass and packaging evidence |

### 2026-09-03 — P1.3f-3c-2c-1 complete: scoped replay protocol frozen

- Froze the sealed 12-case/21-label fixture, the unchanged E5 model/tokenizer,
  five exact product source hashes, and the already built 31-document index
  (`a66067c83cd0440abc8381b31a7b154f`) before executing any candidate query.
- The replay may perform exactly 12 selected-paper selector calls and zero
  full-library calls, embedding builds, index writes, DB writes, downloads, or
  network requests. It must reuse the existing 2,251-parent/2,663-window index
  byte-for-byte.
- Pre-registered runtime gates require one cold load, at least 11 warm hits,
  zero fallback/source escape, exact sparse-input immutability, full protected-
  prefix preservation, warm p95 below one second, peak RSS below 6 GiB, and
  unchanged DB/index/artifacts.
- Pre-registered relevance gates require at least +0.02 Recall@10, no aggregate
  regression in Recall@5, Recall@10, MRR, nDCG@10, or contract Recall@10, and
  zero per-case regression in Recall@10, MRR, or contract Recall@10. A pass can
  authorize only a packaging experiment; it cannot enable the product or UI.
- Protocol validator, six anti-relaxation tests, Ruff, and diff checks pass;
  CI and Windows release now validate the frozen protocol.

### Priority assessment before scoped blind replay

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3c-2c-2 | Implement/dry-test the read-only scoped worker, then consume the sealed 12-query replay once | Very high | Direct decision evidence; no expensive rebuild and no opportunity to relabel | Do now |
| P2.1a | Repair historical evaluation provenance around immutable snapshots | High | Independent architecture value and next default if the candidate is rejected | After verdict |
| P1.3f-4 | Portable runtime size/startup measurement for selected-paper retrieval | Conditional | Relevant only if fresh scoped quality passes | After pass only |

### 2026-09-03 — P1.3f-3c-2c-2a complete: read-only scoped worker dry-validated

- Implemented a dedicated scoped-only worker rather than reusing the earlier
  two-mode evaluator. For each case it builds the current product-query BM25
  retriever from only that paper, computes one immutable sparse list, records
  the BM25 control, then supplies a copy of the same list to the frozen product
  selector. This removes full-library calls and avoids control/candidate input
  drift.
- The worker calls `status()`/`load()` only; no `build()` path is reachable.
  It records the exact top ten, per-case metrics, selector cold/warm behavior,
  fallbacks, source escapes, protected-prefix checks, sparse-input identity,
  process memory, and existing index identity.
- The controller verifies the sealed fixture, source-backed qrels, exact
  Python/package overlay, DB snapshot, index tree, E5 artifacts, and five
  product source hashes before starting. Post-run gates independently rehash
  DB/index/artifacts and persist the report before interpreting pass/reject.
- Eighteen runner/protocol/fixture tests, Ruff, compilation, diff checks, and
  the full read-only execution preflight pass. The candidate has not run.

### Priority assessment after scoped worker preflight

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-3c-2c-2b | Consume the one sealed 12-query scoped replay and pin its verdict | Very high | All inputs and gates are now immutable; delaying adds no evidence | Do now |
| P2.1a | Historical executed-snapshot provenance repair | High | Next unconditional maintainability task | After verdict |
| P1.3f-4 | Selected-scope portable runtime experiment | Conditional | Only authorized by a fresh scoped pass | After pass only |

### 2026-09-03 — P1.3f-3c-2c-2b complete: fresh selected-paper E5 replay passed

- Consumed the sealed replay exactly once. The worker completed 12 scoped calls
  and zero full-library calls, index builds, fallbacks, source escapes, network
  calls, or product/UI changes. It reused build
  `a66067c83cd0440abc8381b31a7b154f`; the working DB, index tree, model, and
  tokenizer hashes remained byte-identical.
- Fresh scoped Recall@10 improved from 0.541667 to 0.666667 (+0.125), and
  contract Recall@10 improved by the same amount. MRR rose from 0.407738 to
  0.425331 and nDCG@10 from 0.425857 to 0.471390. Recall@5 stayed exactly
  0.500000. No individual case lost Recall@10, contract Recall@10, or MRR.
- The new relevant evidence entered the top ten for the FDM integration-time
  boundary and SCINeRF overlap-rate questions, while the sequential compressed
  sensing case gained its second required evidence contract. Two hard cases
  (SCIGS filtering and unstained-tissue phase evidence) remain misses for both
  channels; the gain is therefore real but not evidence of complete coverage.
- Operational gates all pass: one cold load, 11 warm cache hits, 90.573 ms warm
  selector p95, 1,176.938 MiB peak RSS, 8.159 s worker wall time, 20 dense-tail
  insertions, and 100% protected-prefix/sparse-input preservation.
- Pinned result fingerprint
  `23a3a7262c8cde3a95ed2cbfbe6ea251318f05327cbf149438e9331fcbea7c76`
  and report SHA-256
  `90db9c97eef1196ab1e93f5531f662735daeef53c28adbb8a2a54cf5d0bb7fd8`.
  CI/release validate the report. Twenty-eight focused protocol, fixture,
  runner, and report tests plus Ruff pass. Product remains BM25-only.

### Priority assessment after fresh scoped pass

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4a | Freeze and run a Windows portable size/cold-start experiment for an optional selected-paper semantic component | Very high | Quality now justifies measuring the remaining ~487 MB deployment cost; bundling it blindly could bloat every install | Do next |
| P2.1a | Replace historical live-source hashes with immutable executed-snapshot provenance | High | Prevents valid refactors from invalidating evidence and is independent of product promotion | After packaging verdict |
| P1.3f-4b | Design explicit opt-in/setup/index lifecycle for selected-paper retrieval only | High, conditional | Must keep full-library BM25 and distinguish local semantic retrieval from external web/model knowledge | Only if optional packaging is viable |
| P1.3f-5 | Cross-encoder reranker screening | Deferred | Adding another model/runtime before a deployable dense path would compound size and latency | After selected-paper dense path is deployable |

### 2026-09-03 — P1.3f-4a-1 complete: optional Windows component experiment frozen and dry-validated

- Froze the published beta.13 portable archive (`9c78a7635896`), embedded
  CPython 3.10.11, accepted E5 artifacts, minimal ORT/tokenizers overlay,
  existing read-only 31-paper index, two previously successful selected-paper
  queries, source hashes, output paths, and pass/fail gates before assembling
  or executing the component.
- The experiment must leave the 94,876,265-byte base archive untouched. It may
  create only a separate optional archive with no user-library data and may not
  enable the product or expose a mode switch. External web/model knowledge is
  not involved and remains a separate future authorization choice.
- Pre-registered limits are 550 MB for raw and compressed component size,
  650 MB combined download, 30 seconds cold load, 1 second warm query p95,
  2 GiB peak RSS, zero index builds/fallbacks/source escapes, and byte-identical
  base/model/runtime sources after the run.
- Added a clean-checkout protocol validator, stricter local-file preflight,
  embedded-runtime worker, controller, and anti-relaxation/gate tests. Eleven
  focused tests, Ruff, compilation, direct protocol validation, and the
  no-write preflight pass. A Python compatibility issue in the new validator
  was caught before output creation and corrected; frozen runner/worker hashes
  did not change. No package has been assembled and no candidate query has run.

### Priority assessment before the portable experiment

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4a-2 | Assemble the separate component once, execute it under beta.13 embedded Python, and pin the verdict | Very high | Directly decides whether selected-paper E5 can be offered without bloating every base install | Do now |
| P2.1a | Replace historical live-source hashes with immutable executed-snapshot provenance | High | Required for maintainable future refactors regardless of packaging outcome | After verdict |
| P1.3f-4b | Specify opt-in/download/preparation/removal and failure recovery for selected-paper semantic retrieval | High, conditional | Only responsible after deployment cost and embedded-runtime behavior pass | After a packaging pass |
| P1.3f-5 | Cross-encoder reranker screening | Deferred | A second model is premature before the first optional component has a viable lifecycle | Keep deferred |

### 2026-09-03 — P1.3f-4a-2a complete: first assembly safely exposed a Windows path constraint

- The first real run stopped while copying the frozen ORT overlay. No component
  archive or report was written, the embedded worker/model never started, and
  zero retrieval queries ran. The beta.13 archive remains exactly 94,876,265
  bytes with SHA-256
  `ef9247b8af3856deee6b9c8ea5b2605cd0af58f77c41aa7acb571f98a75f9968`;
  the strict local preflight also reverified every model/runtime source.
- Root cause is Windows path length: the long experiment root plus the
  human-readable component directory plus ORT's deep optional tooling paths
  exceeded the filesystem limit during `copytree`. This is not a candidate
  relevance/runtime result and therefore must not be recorded as either an E5
  pass or rejection.
- Design improvement: keep the human-readable archive/manifest identity but
  use a short internal installation directory. This is also the safer eventual
  product design because component setup should not depend on the user's parent
  directory length. The aborted output contains only extracted/copy-staged
  release files under the dedicated ignored experiment directory and can be
  removed before the refrozen retry.

### Priority assessment after the path-length pre-execution stop

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4a-2b | Refreeze only the experiment output/internal component paths to a short deterministic layout, rerun dry validation, then retry once | Very high | Resolves an observed Windows deployment constraint without tuning relevance/runtime gates | Do now |
| P2.1a | Immutable executed-snapshot provenance | High | Still the next unconditional architecture repair | After packaging verdict |
| P1.3f-4b | Explicit selected-paper opt-in and component lifecycle | Conditional | Path-independent installation/removal must be part of this lifecycle if packaging passes | After packaging pass |

### 2026-09-03 — P1.3f-4a-2b complete: short internal layout refrozen and dry-validated

- Preserved the readable archive/component ID while shortening only the ignored
  experiment root to `test_results/e5p1` and the archive's internal component
  directory to `c`. No candidate, source hash, quality/runtime/size gate, query,
  product boundary, or beta.13 input changed.
- Recorded the first attempt inside the protocol as an aborted pre-execution
  path-length event with zero worker starts and zero queries. The refreeze is
  therefore auditable rather than silently overwriting a failed measurement.
- JSON validation, protocol validation, eleven focused mutation/gate tests,
  Ruff, and strict local-file no-write preflight pass. The short result root is
  absent, so the retry still has fresh-output semantics.

### Priority assessment before the path-safe retry

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4a-2c | Run the path-safe assembly/embedded-runtime experiment once and pin pass or rejection | Very high | The only remaining evidence needed before lifecycle design | Do now |
| P2.1a | Immutable executed-snapshot provenance | High | Next unconditional maintainability repair | After verdict |
| P1.3f-4b | Selected-paper opt-in/setup/removal/recovery design | Conditional | Must absorb both package size and path-safe install constraints | Only after pass |

### 2026-09-03 — P1.3f-4a-2c complete: optional E5 component rejected on startup gate

- Completed the path-safe frozen run under the unmodified beta.13 embedded
  CPython 3.10.11. The base archive, E5 artifacts, and ORT/tokenizers sources
  remained byte-identical; the component contains no user-library data and
  performed zero index builds, network requests, fallbacks, source escapes, or
  product/UI enablement.
- Packaging itself is feasible but expensive: 541,333,859 bytes unpacked,
  296,365,646 bytes compressed, and 391,241,911 bytes combined with the
  unchanged 94,876,265-byte base download. Assembly took 6.157 seconds and
  maximum-compression ZIP creation took 149.877 seconds.
- Embedded execution reproduced both expected selected-paper gains at relevant
  ranks 9 and 10. Model/index cold load was 2.445 seconds, warm query p95 was
  85.006 ms, and peak RSS was 1,168.672 MiB. Every frozen functional, scope,
  size, memory, and query-latency check passed.
- The single failure was the pre-registered import gate: 15,182.311 ms versus
  the 5,000 ms maximum. The measurement includes ORT/tokenizers and all worker
  product/evaluation imports, so the verdict is a valid rejection of this v1
  delivery composition, not proof that ONNX import alone is slow. The gate is
  not relaxed post hoc and the product remains BM25-only.
- Pinned rejection fingerprint
  `2120f98ec523d98c0e890a66be008d71a65a67f04e468b3edf61ada48625df7d`
  and report SHA-256
  `a938b1130c600aa8d9b1ccb11630848bba54ecef204d593348fde10764f42a89`.
  CI/release validate both protocol and report. Sixteen focused tests, Ruff,
  architecture budgets, validators, and scoped diff checks pass.

### Priority assessment after the v1 packaging rejection

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4c-1 | Attribute embedded startup time by module group without rerunning relevance or changing the rejected verdict | Very high | Determines whether to shrink the runtime payload, split lazy imports, or abandon this candidate; avoids guessing from one aggregate timer | Do next |
| P2.1a | Replace historical live-source hashes with immutable executed-snapshot provenance | High | Unconditional architecture repair and needed before broad refactors | After the bounded startup diagnosis |
| P1.3f-4c-2 | Freeze a v2 delivery candidate only if the diagnosis identifies a concrete pre-query startup reduction | Conditional | Must be a new candidate with unchanged relevance ranking and fresh deployment gates, not a retroactive v1 pass | After diagnosis |
| P1.3f-4b | User opt-in/setup/removal UI and lifecycle | Deferred | Exposing a choice for a rejected delivery would hand the unresolved problem to the user | Do not implement yet |
| P1.3f-5 | Cross-encoder reranker | Deferred | Adds more runtime cost before dense deployment is viable | Keep deferred |

### 2026-09-03 — P1.3f-4c-1 complete: startup time attributed to the experiment boundary

- Ran no retrieval and wrote no library/index data. Using fresh embedded-Python
  processes against the exact staged component, ORT plus tokenizers imported in
  379.067 ms; dependencies plus the real local-semantic product modules imported
  in 494.337 ms.
- Reconstructing the v1 worker's full import list took 1,832.198 ms with warm
  filesystem cache. Importing
  `tools.retrieval_eval.run_dense_research_qa_benchmark` alone contributed
  1,339.196 ms, roughly 73% of that full warm import. The benchmark module
  brings candidate/evaluation machinery that is not part of an ordinary
  selected-paper product query.
- The diagnostic cannot retrospectively turn the 15,182.311 ms cold v1 result
  into a pass, and warm timings are not a substitute for a new clean-path cold
  measurement. It does rule out immediately replacing the E5 model/runtime on
  the evidence available: the measured product dependency boundary is small,
  while the v1 timer was materially contaminated by evaluation-only imports.
- Improved next design: create a separately frozen v2 execution worker that
  stops the startup timer after importing only the dependencies and product
  retrieval modules. Fixture/evaluation helpers may load only after that
  timestamp. Re-extract the already pinned component archive to a new short
  path, keep all 5-second/size/memory/query gates unchanged, and forbid ranking
  or model changes. This tests the user-relevant startup boundary without
  rewriting the rejected v1 record.

### Priority assessment after startup attribution

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4c-2a | Implement and freeze the v2 product-only startup worker/protocol; verify it cannot change ranking or reuse the v1 output path | Very high | Directly tests the diagnosed issue with no relevance tuning | Do next |
| P1.3f-4c-2b | Execute one clean-path v2 startup/query replay and pin its independent verdict | Very high, conditional | Can authorize lifecycle design only if the original 5-second gate passes | After dry validation |
| P2.1a | Immutable executed-snapshot provenance | High | Proceed after the bounded v2 verdict, pass or reject | Next unconditional item |
| P1.3f-4b | Selected-paper user opt-in and component lifecycle | Deferred | User-facing choice remains blocked until v2 passes | Do not expose yet |

### 2026-09-03 — P1.3f-4c-2a complete: v2 product-startup boundary frozen

- Added a dedicated embedded worker whose startup timer contains only ORT,
  tokenizers, and the actual product modules used for BM25 candidate retrieval,
  local-semantic loading, and protected-tail selection. It directly composes
  existing `BM25Retriever` and `retrieval_engine`; it does not import the dense
  benchmark/evaluation stack.
- Froze the rejected v1 report and verdict, exact 296,365,646-byte component
  archive/tree, unchanged beta.13 archive, worker/controller hashes, fresh
  `test_results/e5p2` extraction path, and both v1 top-ten rankings. V2 cannot
  relabel relevance, alter fusion/model/index, or reinterpret v1 as passing.
- Kept the original 5-second import, 30-second cold-load, 1-second query p95,
  2-GiB RSS, 550/650-MB package limits and all scope/integrity gates. Passing
  may authorize component-lifecycle design only; it still cannot bundle the
  component by default, expose a mode switch, or enable the product.
- Twelve protocol/runner anti-relaxation tests, Ruff, compilation, direct
  validation, and strict no-extraction preflight pass. The v2 output path and
  report are absent; no v2 worker or retrieval query has run.

### Priority assessment before v2 execution

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4c-2b | Execute the single fresh-path product-startup replay and pin its independent verdict | Very high | Resolves whether v1 failed because of its harness boundary while preserving every threshold | Do now |
| P2.1a | Immutable executed-snapshot provenance | High | Start immediately after this bounded verdict | Next unconditional item |
| P1.3f-4b | Component lifecycle contract, not UI | Conditional | May be designed after a pass; user-visible activation still requires implementation and end-to-end recovery tests | Only after pass |

### 2026-09-03 — P1.3f-4c-2b-1 complete: v2 first execution caught a worker wiring defect before measurement

- The fresh-path v2 controller extracted both archives, but the worker stopped
  before model/index construction completed and before any query or report.
  `LocalSemanticRuntimeService` correctly returned
  `local_semantic_runtime_unavailable`.
- Root cause is isolated to the new v2 worker: its artifact verifier correctly
  pointed at the optional component, but its executor factory mistakenly used
  the service's default DB-relative model/tokenizer arguments instead of the
  verified component paths. The v1 worker used the correct ignored-argument
  wiring and had already proven the component files themselves work.
- This is neither a startup pass/rejection nor a candidate failure. The repair
  is constrained to the executor path wiring; the next retry must use another
  absent short result root, preserve all gates and rankings, record this failed
  attempt in the v2 protocol, and refreeze the changed worker hash before
  execution.

### Priority assessment after the v2 pre-query wiring stop

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4c-2b-2 | Correct only component artifact wiring, record/refreeze, dry-validate, then retry at a fresh path | Very high | Fixes a deterministic harness defect without changing the tested candidate | Do now |
| P2.1a | Immutable executed-snapshot provenance | High | Begins after the bounded v2 verdict | Next unconditional |
| P1.3f-4b | Component lifecycle contract | Deferred | Still blocked on a valid v2 outcome | Wait |

### 2026-09-03 — P1.3f-4c-2b-2 complete: component-path wiring corrected and refrozen

- Changed only the v2 executor factory to ignore DB-relative placeholder paths
  and open the already verified component's `model.onnx` and
  `tokenizer.json`, matching the successful v1 wiring. No candidate artifact,
  index, query, expected ranking, threshold, or product boundary changed.
- Recorded the first v2 attempt as a pre-query failure with zero candidate
  queries and no report. Refroze the corrected worker SHA-256 and moved the
  retry to the absent short path `test_results/e5p3`.
- JSON/protocol validation, twelve anti-relaxation tests, Ruff, compilation,
  strict local archive/source verification, and the no-extraction preflight
  pass. This is the final bounded v2 retry before moving to the unconditional
  architecture work.

### Priority assessment before the final v2 retry

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4c-2c | Execute and pin the valid product-startup measurement | Very high | One corrected fresh-path measurement resolves the delivery branch | Do now, no unbounded retries |
| P2.1a | Immutable executed-snapshot provenance | High | Begin next regardless of v2 outcome | Next |
| P1.3f-4b | Component lifecycle contract | Conditional | Authorized only by a valid pass; product/UI still remain off | After pass only |

### 2026-09-03 — P1.3f-4c-2c complete: product-startup v2 rejected and pinned

- Completed the final fresh-path run under beta.13's embedded CPython. Product-
  only cold import took 9,046.063 ms, independently exceeding the unchanged
  5,000 ms gate. Model/index cold load was 2,256.822 ms, warm query p95 was
  84.380 ms, and peak RSS was 1,167.973 MiB. Both selected-paper top-ten lists
  and relevant ranks 9/10 exactly matched v1, with no build, network, fallback,
  source escape, user data, or product/UI enablement.
- A second gate also failed because v1 recorded its component tree after worker
  imports had generated/updated bytecode, while the archive necessarily
  reproduces the pre-worker tree. This reveals a provenance-definition defect;
  it does not rescue the candidate because the independent product-import gate
  still fails by 4.046 seconds.
- No third retry or post-hoc threshold change is allowed. Keep the base product
  on BM25, do not show a semantic-mode control, and defer lifecycle UI. A future
  dense attempt must be a genuinely smaller/lazier delivery candidate with its
  own frozen protocol, not another reinterpretation of E5 v1.
- Pinned v2 rejection fingerprint
  `7789569cae9397c475472d0d0a63d312b8789f06de7531b357e4521cd5d48d85`
  and report SHA-256
  `5f8216155254b9b5fe61d46cc8375ed3b023ea68cbc81e27bdf338c9cf21c5ca`.
  CI/release validate protocol and report. Seventeen focused tests, Ruff, and
  architecture budgets pass.

### Priority assessment after the portable delivery rejection

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.1a | Separate pre-execution live-source verification from post-execution immutable provenance for the accepted/rejected E5 chain | Very high | Current validators can make historical evidence fail merely because implementation files are safely refactored; the v2 tree mismatch demonstrates the same identity-category problem | Do next |
| P2.1b | Extract canonical retrieval snapshot/query seams from oversized modules | High | Reduces private imports and makes future evaluation runners smaller | After provenance repair |
| P1.3f-4d | Screen a genuinely smaller/lazier local-semantic delivery at Stage 0 only | Medium-high | Dense relevance is promising, but current 296-MB/9-second delivery is not acceptable | After architecture repair; no UI |
| P1.3f-4b | User opt-in/setup/removal lifecycle | Deferred | A user choice cannot solve a rejected engineering dependency | Remains blocked |
| P1.3f-5 | Cross-encoder reranker | Deferred | Would worsen package/startup cost | Remains blocked |

### 2026-09-03 — P2.1a complete: execution-time source identity separated from live refactor checks

- Updated the E5 Stage-2, product selector shadow, 31-paper user-journey,
  fresh scoped replay, portable v1, and portable product-startup v2 validators.
  Their default post-execution mode now validates the frozen hash values,
  protocols, fixtures, reports, result fingerprints, gates, and decisions
  without requiring today's implementation file at the old path.
- Real preflight remains strict. Stage-2 actual execution, selector/user-
  journey runners, scoped replay, and both portable controllers explicitly use
  the live-file inspection mode before any model/build/query work. Missing or
  changed implementation files return categorized validation errors rather
  than uncaught file exceptions.
- Added six tests that simulate moved/refactored implementations. Historical
  evidence remains valid in default mode, while explicit preflight mode rejects
  the same mismatch. This removes the false choice between preserving evidence
  and safely decomposing oversized files; it does not weaken qrels/report or
  execution-time source checks.
- Forty-three focused provenance/protocol tests pass. Stage-2 and all five
  downstream E5 report validators pass, strict current-file preflights pass,
  Ruff/diff checks pass, and the full architecture budget remains green.

### Priority assessment after provenance repair

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.1b-1 | Extract the product-query BM25 candidate adapter from the oversized dense benchmark/private import surface into a small public module | Very high | Directly removes the 1.339-second warm benchmark import found in deployment diagnosis and establishes a stable retrieval boundary | Do next |
| P2.1b-2 | Migrate evaluation/product-startup callers and prove ranking parity without rerunning frozen acceptance outcomes | High | Shrinks coupling and makes future delivery probes representative | With extraction |
| P1.3f-4d | Stage-0 screen a smaller/lazier semantic delivery candidate | Medium-high | Only justified after the query boundary is clean | After extraction |
| P1.3f-4b | User-facing opt-in lifecycle | Deferred | Current E5 delivery remains rejected | Blocked |

### 2026-09-03 — P2.1b-1/2 complete: product-query evaluation adapter extracted and migrated

- Moved `ProductQuerySparseRetriever` out of
  `run_dense_research_qa_benchmark.py` into a dedicated small evaluation
  module. The adapter still uses the exact product BM25 query path, fixed
  rank-20 comparison depth, translation/expansion flags, and scoped/library
  semantics; no ranking algorithm changed.
- Migrated all 17 dense/cross-encoder/Qwen/E5/CPU/scoped/portable evaluation
  callers to the public adapter name. The dense benchmark shrank by the removed
  private class, and the product-startup worker no longer maintains a duplicate
  local implementation.
- Added direct tests for fixed-depth argument forwarding, empty-corpus behavior,
  and isolated import behavior. An independent process proves importing the
  adapter does not load `run_dense_research_qa_benchmark`. Forty-seven focused
  tests, every pinned E5 report, Ruff, diff checks, and architecture budgets
  pass; no frozen result was rerun or reinterpreted.
- Follow-up measurement prevents overstating the win: a standalone adapter
  import was about 1.521 seconds and a standalone dense benchmark import about
  1.501 seconds on the warm embedded runtime. The adapter removed improper
  benchmark coupling, but `kb/retrieval_engine.py` remains the dominant import
  because the product-query policy still lives in its 5,099-line monolith.

### Priority assessment after adapter extraction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.1b-3 | Extract the query translation/deterministic expansion/RRF/fallback policy into a small product-query module while preserving exact outputs | Very high | Shrinks a protected 5,099-line file and attacks the remaining adapter import boundary | Do next, guarded by parity tests |
| P2.1b-4 | Switch task/runtime and evaluation callers to the public query API; retain no private compatibility shim if all tests pass | High | Removes private cross-module imports and clarifies ownership | With extraction |
| P1.3f-4d | Screen a smaller/lazier semantic delivery | Medium-high | Should use the cleaned query boundary | After architecture work |
| P1.3f-4b | User opt-in lifecycle | Deferred | Delivery still rejected | Blocked |

### 2026-09-03 — full quality-gate closure after portable E5 and query-adapter work

- The complete backend suite passes: 5,413 passed and 43 skipped in 313.98
  seconds. The first complete run had one unrelated wall-clock assertion failure
  in `test_chat_stream_routes_share_one_total_deadline`; the unchanged test
  passed on isolated repetition, all 12 tests in its file passed, and the
  unchanged complete suite then passed. No timeout threshold or LLM behavior was
  weakened to obtain the green result.
- Repository Ruff, architecture budgets, and whitespace checks pass. In
  particular, the protected `kb/retrieval_engine.py` remains at its 5,099-line
  ceiling and every other protected oversized file stayed within budget.
- Frontend lint and production build pass. Playwright passes the 160-test smoke
  suite with three skips, the 127-test core suite, all four public-surface tests,
  and the evidence-workspace-hidden test. These runs exercise the product as a
  user without requiring manual participation.
- Both portable E5 rejection reports and all earlier pinned E5 reports validate.
  The decision therefore remains fail-closed: the released/product path is
  BM25-only, no semantic switch is exposed, and no rejected dependency setup or
  recovery work is handed to the user.
- No release version, tag, commit, or published artifact was changed in this
  work. The worktree still contains the broader in-progress project changes and
  they were preserved.

### Refined priority assessment at the green-gate boundary

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.1b-3 | Extract the exact product query policy from `kb/retrieval_engine.py` into a focused module, with output-parity tests before caller migration | Very high | Static dependency analysis identifies eight functions and about 962 lines, including translation, deterministic expansion, result merge/RRF, fallback search, cache access, and temporary-source filtering. Shared stop-token constants and caches must be injected or moved explicitly; this is the next meaningful architectural cut, not a mechanical rename | Next implementation step |
| P2.1b-4 | Move product and evaluation callers to that public query-policy API, then measure isolated cold import again | High | Tests whether the 1.5-second warm adapter import actually falls and removes private access to the monolith | Immediately after parity is proven |
| P1.3f-4d | Freeze and run Stage 0 for one genuinely smaller or lazier local-semantic delivery candidate | Medium-high | E5-small improved selected-paper Recall@10 by 0.125, but its 296-MB optional archive and 9.046-second product cold import failed delivery. A new candidate must improve deployment, not reinterpret these results | After the query boundary is smaller |
| P1.3f-4b | Design and implement optional setup/removal plus explicit user consent | Deferred | The user should be offered a choice only after there is an engineering-accepted component with automatic setup, health, fallback, and removal | Blocked until a delivery candidate passes |
| P1.3f-5 | Cross-encoder reranking | Deferred | It adds package, startup, and latency cost before dense retrieval is viable | Keep deferred |

The improved sequence is therefore architecture boundary first, cold-import
remeasurement second, and a new delivery candidate third. This avoids spending
more evaluation effort on a component that users still cannot reliably install
or start, while preserving the demonstrated selected-paper relevance signal for
the next candidate.

### 2026-09-03 — P2.1b-3 baseline frozen before query-policy extraction

- Confirmed the current product/evaluation boundary still imports five private
  query functions from `kb/retrieval_engine.py`. The move scope is the complete
  query-policy chain: translation, optional LLM expansion, deterministic domain
  variants, RRF merge, and fallback orchestration. Shared source filtering and
  stop-token data need one canonical owner rather than being copied.
- Before modification, `kb/retrieval_engine.py` is 5,099 lines with SHA-256
  `95166ccbda1dc0fac1d308ccef51ce99fbc30a2ae9f61783cc805bfb6db2537a`.
  The focused query/translation/temp-filter/dense-adapter baseline is 76/76
  passing. A fresh-process adapter import measured 3,227.995 ms on this run;
  this measurement is diagnostic, not a release gate and not directly
  comparable across filesystem-cache states.
- Extraction rule: introduce public query-policy names, migrate every production
  and evaluation caller, preserve exact search depth/weights/result metadata,
  and remove the old private query functions instead of leaving a second
  implementation. Existing result reports remain immutable; their strict
  pre-execution source checks may correctly reject replay after this refactor.

### Priority assessment at the extraction boundary

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.1b-3a | Move canonical stop-token data and temporary-source filtering to small shared owners | High | Prevents duplicated policy while allowing both document grouping and query orchestration to use the same definitions | Do first |
| P2.1b-3b | Move translation, expansion, deterministic variants, RRF, and fallback orchestration to `kb/product_query_policy.py` | Very high | Removes the evaluation/product query path from the 5,099-line monolith | After shared seams |
| P2.1b-4 | Migrate all callers and tests to public names, then run parity and import measurements | Very high | Proves this is an ownership-only refactor before any new semantic candidate work | Immediately after extraction |

### 2026-09-03 — P2.1b-3a complete: shared query terms and source filtering have canonical owners

- Added `kb/retrieval_query_terms.py` as the single owner of document-hint and
  direct-prompt stop-token sets. `kb/retrieval_engine.py` imports those exact
  immutable sets, so document matching and the coming query-policy module cannot
  silently diverge through copied constants.
- Moved temporary/conversion-artifact source detection into the existing
  `kb/source_filters.py` boundary as `is_temporary_source_path`. The retrieval
  engine retains only an imported internal alias for its unrelated grouping
  paths; the implementation and environment-root exception are no longer owned
  by the monolith.
- `kb/retrieval_engine.py` is now 4,976 lines, down 123 lines. Seventy-nine
  focused source-filter, document-anchor, query-expansion, and query-variant
  tests pass; Ruff and the full architecture budget pass.

### Priority assessment after shared-seam extraction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.1b-3b | Move the five query operations to `kb/product_query_policy.py`, with lazy LLM construction and unchanged runtime cache semantics | Very high | Removes the remaining query path from the monolith and avoids loading answer-generation code merely to import BM25 policy | Do now |
| P2.1b-4 | Migrate production/evaluation callers and tests to the public API; prohibit private imports from `retrieval_engine` | Very high | Enforces the new module boundary rather than leaving a compatibility dependency | Next |
| P1.3f-4d | Re-measure cold imports, then decide whether a smaller semantic candidate is justified | Medium-high | Must follow parity proof | Later |

### 2026-09-03 — P2.1b-3b/P2.1b-4 implementation complete: public product query policy is independent

- Added `kb/product_query_policy.py` with public translation, optional LLM
  expansion, deterministic variants, RRF merge, and fallback-search entrypoints.
  Candidate depths, translation weights, expansion weights, metadata fields,
  whole-library behavior, cache buckets, and fallback depths are unchanged.
- LLM client construction is now lazy inside the two operations that can
  actually call it. Importing deterministic BM25 policy therefore does not load
  answer-generation/LLM code. The product query adapter imports neither
  `kb/retrieval_engine` nor `kb.llm`.
- Migrated task execution, agent retrieval, research-QA evaluation, paper-guide
  tools, product-shadow replay, BM25 baselines, dense/hybrid tests, and the
  evaluation adapter to public query-policy names. Added an AST boundary test
  that rejects any future production/evaluation import of the five removed
  private engine names; no compatibility implementation remains.
- The engine is now 4,051 lines, down 1,048 lines from the frozen 5,099-line
  baseline; the focused policy and caller suite is 147/147 passing. Ruff,
  whitespace checks, and the full architecture budget pass.
- Three independent fresh-process measurements put direct policy import at
  65.215–75.959 ms and the complete BM25 product-query adapter at
  504.320–525.492 ms. The same-process-class engine import was
  3,851.511–3,943.752 ms. Compared with the pre-extraction adapter diagnostic of
  3,227.995 ms, this removes roughly 84% of the observed import cost. These are
  architectural diagnostics, not a release-performance gate.

### Refined priority assessment after query-policy extraction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.1b-5 | Run broad backend/integration and frozen-report validation for the ownership refactor | Very high | The focused parity suite is green, but task execution and evaluation imports are broad surfaces | Do next before semantic experiments |
| P1.3f-4d-1 | Define a deployment-first Stage-0 protocol for one smaller/lazier semantic candidate, including package/startup ceilings before relevance work | High | The query boundary is now fast enough; the next candidate must first prove it can be delivered, installed, removed, and started automatically | After broad gates |
| P1.3f-4d-2 | Run only Stage 0, then stop on any packaging/startup/memory failure | Conditional | Prevents another long relevance campaign for an unusable component | After protocol freeze |
| P1.3f-4b | User consent and optional component lifecycle | Deferred | Still requires an accepted delivery candidate; no unresolved setup state should be exposed | Blocked |

### 2026-09-03 — P2.1b-5 complete: broad gates close the query-policy refactor

- The complete backend suite passes with 5,416 passed and 43 skipped in 367.97
  seconds. This includes the three new module-boundary tests; there were no
  failures or changed timeout thresholds.
- The complete CI retrieval/semantic validation block passes: raw/product BM25,
  dense and cross-encoder reports, candidate screening, user-journey fusion,
  product shadow, CPU feasibility, E5 Stage 0/1/2, product/scoped E5 reports,
  both portable rejection reports, NLI/claim verification, and local
  observability all retain their pinned decisions.
- Full-repository Ruff, architecture budgets, and whitespace checks pass.
  Frontend lint and production build pass; the build transformed 3,908 modules.
- Unattended Playwright user simulation passes: smoke 160 passed/3 skipped,
  core 127 passed, public surface 4 passed, and the ordinary-build hidden
  workspace gate 1 passed. This covers first use, local upload, cited answers,
  evidence locating, automatic repair, research notes, project evidence views,
  and public/debug separation without manual participation.
- P2.1b is closed. The ownership refactor changes no product retrieval mode and
  exposes no semantic controls. `AGENTS.md` now records the new boundary:
  `product_query_policy.py` owns query construction/fusion/fallback, while
  `retrieval_engine.py` owns evidence grouping/deep-read/reference packing.

### Priority assessment after P2.1b closure

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-1 | Freeze a new architecture-qualified E5 delivery preflight using the extracted query boundary, a new clean result path, and all original size/startup/memory/query thresholds | Very high | The adapter import fell to about 0.5 seconds, so the previous 9.046-second startup result no longer represents the current architecture. The v2 rejection must remain immutable; this is a separately identified delivery candidate, not a third retry or threshold rewrite | Next |
| P1.3f-4d-2 | Run startup/package/scope gates only, reusing the already pinned model/index and rankings | High | Answers the deployment question cheaply before any relevance rerun or new model download | Only after protocol freeze |
| P1.3f-4d-3 | If startup still fails, profile the remaining import groups and screen a smaller/quantized runtime; if it passes, specify automatic install/health/fallback/removal | Conditional | Keeps failures inside engineering and does not ask users to solve dependencies | Branch on the measured verdict |
| P1.3f-4b | Implement explicit user consent and optional component lifecycle UI | Deferred | Becomes eligible only after the new delivery candidate passes all unattended recovery tests | Still blocked |
| P1.3f-5 | Cross-encoder reranking | Deferred | Adds cost before dense delivery is accepted | Keep deferred |

This sequence is better than immediately selecting another embedding model:
the relevance signal and artifacts already exist, while the principal measured
startup dependency has just been removed. A bounded startup-only candidate can
now determine whether that architectural change is sufficient. It must preserve
the v1/v2 rejection records and cannot enable product behavior by itself.

### 2026-09-03 — P1.3f-4d-1 complete: architecture-qualified v3 startup protocol frozen

- Added the independent candidate
  `multilingual_e5_small_extracted_query_policy_v3`. It reuses the exact
  beta.13 archive, optional E5 component, existing index, selected-paper cases,
  and v2 top-ten rankings. Relevance reruns and product enablement are forbidden;
  this candidate measures only whether the newly extracted query boundary makes
  the existing delivery viable.
- Preserved every original gate: 550-MB raw/archive component ceilings,
  650-MB combined download ceiling, 120-second extraction, 60-second worker,
  5-second product import, 30-second cold load, 1-second query p95, 2-GiB peak
  RSS, zero builds/network/fallbacks/source escapes, and exact ranking parity.
  V2's rejection report, fingerprint, decision, and both failed gates are
  immutable inputs.
- Repaired the provenance definition prospectively rather than rewriting v2.
  V3 fingerprints all 680 archived files/541,333,859 bytes and verifies each
  extracted file against the ZIP before the worker starts. Runtime bytecode is
  disabled. This protects the real payload without treating normal post-import
  cache changes as archive corruption.
- The worker now exercises the actual `BM25Retriever` plus public
  `search_hits_with_fallback` product path, not the evaluation adapter. It
  records and rejects any import of `kb.retrieval_engine` or `kb.llm` during the
  product timing boundary.
- Nineteen protocol/runner/boundary tests, Ruff, standalone protocol validation,
  and strict `--validate-only` preflight pass. The v3 report and short
  `test_results/e5p4` root are confirmed absent; no extraction, model load,
  retrieval query, network access, or product/UI change has occurred.

### Priority assessment after v3 protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-2 | Execute the single clean-path v3 package/startup replay and pin its verdict | Very high | Directly decides whether the architecture work cleared the original deployment blocker, with no tuning opportunity after measurement | Do now |
| P1.3f-4d-3a | On pass, freeze an automatic component lifecycle contract covering consent, install, health, fallback, retry, and removal before UI work | Conditional | Converts an engineering pass into a user-resolvable experience rather than exposing raw setup states | Only after pass |
| P1.3f-4d-3b | On failure, attribute remaining cold imports and screen a smaller/quantized runtime at Stage 0 | Conditional | Keeps another unusable component away from users | Only after failure |

### 2026-09-03 — P1.3f-4d-2 measured: v3 passes every frozen deployment gate

- The single authorized clean-path execution completed without a retry. Product
  import fell from v2's 9,046.063 ms to 2,056.288 ms, a 77.3% reduction and
  2,943.712 ms inside the unchanged 5-second ceiling. This confirms the earlier
  rejection was an architecture/import-boundary problem, not evidence that the
  E5 model itself needed immediate replacement.
- Component extraction was 3,666.763 ms, embedded worker wall time 5,413.957
  ms, cold model/index load 2,261.870 ms, warm query p95 90.245 ms, and peak RSS
  1,137.008 MiB. The 296,365,646-byte component archive and 391,241,911-byte
  combined download remain unchanged and within their pre-registered ceilings.
- All 680 component files/541,333,859 extracted bytes matched the ZIP before
  execution. Base and component archive hashes were unchanged. The worker loaded
  neither `kb.retrieval_engine` nor `kb.llm` during the product import boundary.
- Both selected-paper rankings and first-relevant ranks 9/10 exactly match v2,
  with zero index builds, network use, fallbacks, source escapes, product
  enablement, or ordinary-user surface exposure.
- Every v3 gate passed. The result fingerprint is
  `07b64dd0541c3c6bf3205f95aeee96c8bd70dd68f55760f13f4a4cabc3f91384`;
  the report SHA-256 is
  `1843fe12db7290cbcb07e892187f0882cee7f24c791790354b6a79eee1c6bc4e`.
  These values must be pinned before any lifecycle design starts.

### Priority assessment after the v3 pass

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-2a | Add the pinned report validator, mutation tests, and CI/release validation | Very high | Prevents the passing result from drifting or being relabeled | Do immediately |
| P1.3f-4d-3a-1 | Freeze an automatic optional-component lifecycle contract before implementation | Very high | The remaining user risk is no longer retrieval quality but a 296-MB download, integrity, disk usage, cancellation, retry, fallback, and removal | Next after report pinning |
| P1.3f-4d-3a-2 | Implement backend lifecycle/state machine and unattended failure recovery with BM25 always available | Very high | Ensures setup failures are solved by the application rather than reported as user tasks | After contract freeze |
| P1.3f-4b | Expose explicit selected-paper semantic consent and removal UI | High, conditional | UI is justified only after lifecycle API and end-to-end recovery pass | Still not yet |
| P1.3f-5 | Cross-encoder reranking | Deferred | Dense delivery and lifecycle come first | Keep deferred |

### 2026-09-03 — P1.3f-4d-2a complete: passing report pinned without erasing the rejection

- Added a dedicated v3 report validator that pins the protocol SHA-256, report
  SHA-256, result fingerprint, passing decision, and every frozen gate. It also
  recomputes the result fingerprint instead of trusting the stored value.
- Added mutation coverage for product-import timing, exact ranking parity,
  forbidden engine import, component payload identity, and the result
  fingerprint. The focused protocol/runner/report/query-boundary suite passes
  with 25 tests.
- CI and Windows release validation now check both the immutable v2 rejection
  and the independent v3 pass. This prevents a future build from presenting the
  architecture-qualified result while silently losing the original failure
  lineage or weakening its thresholds.
- Both v2 and v3 standalone protocol/report validators pass. Ruff, architecture
  budgets, and `git diff --check` also pass; no protected oversized file grew.
- This stage changed validation and release gates only. The product continues to
  use BM25, exposes no semantic retrieval control, and downloads or installs no
  optional component.

### Priority assessment after v3 report pinning

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-1 | Reconcile the existing preparation service with a frozen automatic component lifecycle contract | Very high | Existing lifecycle code may already solve part of consent, repair, cancellation, and removal; the contract must identify gaps rather than duplicate it | Do now |
| P1.3f-4d-3a-2 | Implement only the missing backend transitions and unattended recovery actions, keeping BM25 immediately available | Very high | Users must never receive a raw dependency/checksum/index failure that they are expected to diagnose | After the gap audit and contract freeze |
| P1.3f-4d-3a-3 | Add restart, low-disk, interrupted-download, checksum, stale-index, cancellation, and removal end-to-end recovery tests | Very high | Delivery is not user-ready until every failure reaches a safe automatic outcome | After backend implementation |
| P1.3f-4b | Expose explicit selected-paper semantic consent and removal UI | High, conditional | The choice is meaningful only when the application can complete or undo it unattended | Still blocked on recovery tests |
| P1.3f-5 | Cross-encoder reranking | Deferred | A second optional model would multiply lifecycle cost before the first is operational | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-1 complete: portable component lifecycle v2 frozen

- Audited the existing preparation implementation before designing a replacement.
  The range-resume transport, cooperative cancellation, verified-file atomic
  publish, atomic index generations, single-task ledger, and path-free public
  payload remain useful and must be reused.
- Identified the delivery mismatch precisely. The v1 path downloads six loose
  model files totalling 492,421,554 bytes and assumes runtime dependencies in the
  base build; the accepted v3 delivery is one 296,365,646-byte optional archive
  with 680 files/541,333,859 extracted bytes and its own pinned payload identity.
- Frozen `retrieval_local_semantic_component_lifecycle_v2`. It requires explicit
  consent before any network or disk mutation, separates that consent from
  external-web/model-knowledge consent, and keeps every query on BM25 until a
  healthy component and current selected-paper index are both ready.
- Capacity checks now precede download, extraction, and index build. The first
  install requires 971,917,233 free bytes (archive + payload + 128 MiB safety
  margin); cleanup may reclaim only Pi_zaya-owned staging/archive/old-generation
  data and may never delete PDFs, Markdown, notes, chats, or the BM25 index.
- Integrity, unsafe archive entries, install health failures, interrupted
  processes, stale/corrupt indices, and locked removal files now have automatic,
  bounded recovery outcomes. Checksum and index failures may not be handed to the
  user as a retry task. BM25 remains available throughout.
- Safe removal is idempotent and disables semantic queries before deleting the
  optional component and regenerable semantic index. Product/UI flags remain
  false until backend and unattended recovery tests pass.
- Added a pinned validator, nine contract/mutation tests, and CI plus Windows
  release validation. Together with the v1/v3 lineage tests, 21 focused tests,
  Ruff, JSON parsing, and whitespace checks pass.

### Priority assessment after lifecycle v2 freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-2a | Implement the versioned component manifest, capacity preflight, safe archive verifier/extractor, and atomic install generation | Very high | This replaces the obsolete loose-file/assumed-dependency path and contains the highest integrity and disk risks | Do next |
| P1.3f-4d-3a-2b | Extend the ledger/orchestrator with bounded automatic retry, quarantine, restart recovery, health rollback, and safe removal | Very high | Eliminates dead-end failure states and keeps all failures behind BM25 | After package core |
| P1.3f-4d-3a-3 | Exercise interrupted download, low disk, checksum/payload mismatch, unsafe archive, restart, stale index, cancellation, rollback, and locked removal end to end | Very high | A passing happy path alone is insufficient for ordinary users | After orchestration |
| P1.3f-4b | Add explicit selected-paper semantic enable/remove UI | High, conditional | Must consume only the simplified public states, never pipeline internals | Blocked on recovery E2E |
| P1.3f-5 | Cross-encoder reranking | Deferred | Still lower value than making the accepted dense component operable | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-2a complete: safe component delivery core implemented

- Added a typed, versioned release-component manifest boundary. It accepts only
  HTTPS URLs on an explicit allowlist, requires the immutable release id in the
  URL path, pins the E5 candidate/revision and both archive and extracted-payload
  identities, and rejects path-shaped archive names.
- Added capacity calculation before download and extraction. It includes the
  compressed archive, extracted generation, 128-MiB safety margin, and the
  current generation during upgrades; shortage errors contain only the exact
  additional-byte count, not a path or exception.
- Added a separate resumable archive downloader. It validates redirect hosts and
  exact `Content-Range`, restarts automatically when a server ignores a range,
  bounds bytes while streaming, preserves cancellable partials, removes corrupt
  partials, hashes before atomic publication, and reuses only a fully verified
  archive.
- Added a safe ZIP installer. It rejects absolute/backslash/traversal paths,
  encrypted entries, links/reparse/special entries, and case-insensitive path
  collisions. It verifies file count, raw bytes, every file hash through the
  canonical payload fingerprint, publishes a unique generation atomically, and
  switches `current.json` only after the health check passes. A failed upgrade
  leaves the previous healthy generation current.
- Twenty-five focused contract/delivery tests pass. They cover consent boundary,
  capacity, resume, invalid Range, oversize/hash failure, redirect escape,
  traversal, symlink, Windows case collision, payload drift, atomic publish, and
  rollback. Ruff, the frozen lifecycle validator, architecture budget, and
  whitespace checks pass.
- The real v3 archive metadata is compatible with the production verifier: all
  680 files and 541,333,859 raw bytes pass the new path/type/collision census.
  No files were extracted or installed by this compatibility check.

### Priority assessment after component delivery core

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-2b | Add the v2 lifecycle ledger and orchestrator for consent, bounded automatic recovery, manifest quarantine, restart reconciliation, and removal | Very high | The primitives are safe, but without orchestration failures could still escape as manual retry tasks | Do now |
| P1.3f-4d-3a-2c | Bind the installed component to the actual E5 runtime and selected-paper index, then run health/rollback and stale-index tests | Very high | Installation must not imply readiness; both runtime and current scoped index must pass | After orchestrator core |
| P1.3f-4d-3a-3 | Run unattended failure-path API/E2E simulation | Very high | Confirms ordinary users see BM25 continuity and concrete choices, not internal states | After runtime binding |
| P1.3f-4b | Add explicit enable/remove UI | High, conditional | Still blocked until backend recovery is proven end to end | Not yet |

### 2026-09-03 — P1.3f-4d-3a-2b complete: v2 lifecycle ledger and recovery orchestrator

- Added a separate durable v2 SQLite ledger instead of mutating the frozen v1
  preparation history. It serializes one active component task, persists consent
  and manifest identity, fences quarantine to the exact release/archive pair,
  and distinguishes user cancellation from process interruption.
- Internal download, verification, install, health, index, recovery, removal,
  and degraded states collapse to eight public states. The public payload never
  emits the technical state or internal failure id; all states declare BM25
  available, and integrity/index failures expose no retry action.
- Added the unattended orchestrator. Network and transient internal failures get
  at most two attempts with frozen backoff; repeated integrity failures quarantine
  that manifest version; low disk stops before transport creation; a later app
  start resumes non-user interruption automatically without asking for consent
  again. A deliberate cancel remains paused across restart.
- Installation is still not readiness. The orchestrator reaches `ready` only
  after component health and index readiness both pass. A stale ready record
  triggers background repair while BM25 remains live, and a verified installed
  component is reused instead of downloaded again.
- Removal can supersede an active preparation in the same durable task. It
  disables the component pointer first, releases runtime handles, removes the
  regenerable semantic index, clears consent, and remains idempotent. A simulated
  Windows locked generation is recorded as an owned cleanup target and removed
  automatically on the next start without re-enabling semantic retrieval.
- Forty-three focused v2 contract, package, download, ledger, recovery, and
  removal tests pass in 2.14 seconds. Ruff, architecture budgets, and whitespace
  checks pass. The new orchestrator remains disconnected from API and product
  retrieval, so BM25 behavior and ordinary-user UI are unchanged.

### Priority assessment after recovery orchestration

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-2c | Bind the portable component generation to the real E5 executor and `LocalSemanticWindowIndexStore` behind a small adapter | Very high | Replaces test doubles with the accepted product/runtime path while keeping orchestration isolated | Do next |
| P1.3f-4d-3a-2d | Run the exact 296-MB archive through download-free install, runtime health, selected-paper query, index stale repair, rollback, and removal simulation | Very high | This is the first full-size unattended proof of the lifecycle implementation | After adapter tests |
| P1.3f-4d-3a-3 | Add management API and E2E failure-path simulation using only simplified public states | Very high | Needed before any UI exposure | After full-size lifecycle proof |
| P1.3f-4b | Add explicit enable/remove UI | High, conditional | Still blocked on backend/API recovery E2E | Not yet |
| P1.3f-5 | Cross-encoder reranking | Deferred | Does not reduce the current operational risk | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-2c complete and 2d protocol frozen

- Added the portable runtime adapter that binds an installed generation to the
  exact inner component manifest, full model-artifact verification, component
  site-packages, ONNX Runtime 1.23.2 CPU provider, Tokenizers 0.21.4, NumPy
  2.2.6, the real E5 executor, and `LocalSemanticWindowIndexStore`.
- Health is cached only for an immutable generation signature. Runtime readiness
  requires both that health result and a current corpus-matching index; a
  different database scope, missing/stale index, or component mismatch fails
  closed to BM25. Runtime/index caches are explicitly released before removal.
- Twenty-one adapter, lifecycle, and real-index-boundary tests pass. They cover
  inner-manifest rejection, health caching, actual index build/load interfaces,
  database-scope fencing, stale/missing index rejection, cached runtime reuse,
  and pointer-first index removal.
- Frozen the independent full-size lifecycle v1 protocol before execution. It
  pins the new implementation files, lifecycle v2 contract, v3 pass report,
  beta.13 portable archive, exact 296-MB component, copied 31-document/2,251-
  chunk corpus and index, two v3 rankings, and a new clean `test_results/e5p5`
  path.
- The single allowed run must use embedded Python 3.10.11 with no network,
  install and health-check the exact 680-file component, preserve exact scoped
  rankings, mutate only a copied unrelated chunk, repair that stale index once,
  disable component/index pointers, and finish delayed Windows cleanup after the
  worker exits. Product and UI remain disabled regardless of outcome.
- Six protocol mutation tests, Ruff, standalone validation, and strict
  `--validate-only` preflight pass. Both the result root and report remain absent;
  the measurement has not yet run.

### Priority assessment after full-size protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-2d | Execute the one clean full-size lifecycle run and pin pass or rejection | Very high | Decides whether the new backend survives real package/runtime/index/removal costs | Do now, no retry or tuning |
| P1.3f-4d-3a-3 | On pass, add management API plus failure-path E2E; on failure, keep BM25 and repair the measured backend defect | Conditional | No UI should be designed against an unproven lifecycle | Branch on verdict |
| P1.3f-4b | Add explicit selected-paper semantic UI | Deferred | Remains blocked even if the full-size backend passes; API recovery E2E is still required | Not yet |

### 2026-09-03 — P1.3f-4d-3a-2d rejected: Windows atomic-install path exceeds MAX_PATH

- The one authorized full-size execution ran once and was not retried. The
  lifecycle consumed explicit test consent, used the local archive with zero
  network, attempted its bounded recovery twice, then ended in internal
  `degraded_bm25`. The public projection remained `bm25_available` with no retry
  action and no user-library or product/UI mutation.
- The failure occurred during atomic component extraction before runtime load,
  query, stale repair, or removal. The component ZIP itself passes complete CRC
  validation and retains the pinned 296,365,646-byte archive identity, so the
  evidence does not support an archive-integrity failure.
- Root-cause attribution is a Windows path-layout defect. The longest archive
  member is 148 characters; the v1 atomic staging root was 128 characters, making
  the longest target path 275 characters versus the traditional 260-character
  Windows boundary. The successful v3 experiment used a materially shorter
  extraction root and therefore did not expose this installer defect.
- Pinned the rejection report with fingerprint
  `362c52a8dc745fd62f4ca7739614f95314989cccb771f1fae2fd4b01ef136225`
  and report SHA-256
  `8d8f3fdc5414d523dd408ff5d45dc0e3255e50c89a758cf04e244dad6ef60abb`.
  Six report mutation tests plus six protocol tests pass, and CI/Windows release
  validation now preserve this failure. It cannot later be relabelled as a pass.
- This is a useful rejection: automatic recovery contained the engineering
  failure exactly as designed instead of showing a checksum error or repair task
  to the user. BM25 remains the only product retrieval path.

### Priority assessment after full-size v1 rejection

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-2e | Shorten component generation/staging names, choose a short app-owned Windows component root, and preflight the worst archive-member path before extraction | Very high | Directly repairs the measured 275-character failure and prevents partial extraction | Do next |
| P1.3f-4d-3a-2f | Freeze a distinct v2 full-size lifecycle protocol preserving the v1 rejection, then run once | Very high | Confirms the path repair without rewriting or retrying v1 | After focused path tests |
| P1.3f-4d-3a-3 | Management API and E2E failure simulation | Deferred | The backend is not yet full-size viable | Blocked on v2 pass |
| P1.3f-4b | User semantic enable/remove UI | Deferred | A UI would expose a rejected installer | Blocked |

### 2026-09-03 — P1.3f-4d-3a-2e complete: Windows path layout repaired before extraction

- New component generations use a 16-character immutable id and the atomic
  staging directory uses the compact `.s-<id>` form. Existing 32-character
  generation ids remain readable, so this repair does not strand an earlier
  app-owned installation.
- The default component root is now the short app-owned `<db-parent>/_c/e5`
  location. On Windows, an already-long database parent automatically falls
  back to the short per-user `%LOCALAPPDATA%/Pi_zaya/c/e5` root; it never moves
  or deletes the user's PDFs, Markdown, notes, chat database, or BM25 index.
- Before creating a staging directory, the installer scans every verified ZIP
  member against the actual destination root and the frozen 259-character
  compatibility budget. An overlong layout stops before extraction with the
  stable internal reason `component_storage_path_too_long`, keeps BM25 active,
  and does not quarantine a valid component archive.
- Twenty-nine focused component, lifecycle, and ledger tests pass, including
  compatibility with legacy ids and rejection before staging creation. Ruff,
  architecture budgets, whitespace checks, and the immutable v1 rejection
  validator pass.
- Against the exact accepted 680-file archive, the failed v1 layout would still
  reach 253 characters after only shortening ids, while the new clean v2 root
  reaches 239 characters. The repair therefore depends on both compact names
  and a compact app-owned root, not on weakening archive verification.

### Priority assessment after Windows path repair

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-2f | Freeze a distinct v2 full-size lifecycle protocol with the 239-character layout and immutable v1 rejection lineage, then execute it once | Very high | This is the decisive real-package proof for the measured repair | Do now, no retry or tuning |
| P1.3f-4d-3a-3 | On v2 pass, add a read/manage API over the simplified lifecycle states and test failure recovery end to end | Very high, conditional | Product exposure remains unsafe until the public contract is exercised through HTTP | Next only on pass |
| P1.3f-4b | Add explicit selected-paper semantic enable/remove UI | High, conditional | UI remains blocked on management API and unattended API/E2E recovery evidence | Not yet |
| P1.3f-5 | Cross-encoder reranking | Deferred | It does not reduce the current component-operability risk | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-2f protocol frozen: independent short-path candidate

- Frozen `retrieval_portable_e5_full_size_lifecycle_v2` as a new candidate; it
  does not edit or rerun the v1 rejection. The protocol pins the v1 protocol
  hash, rejection-report hash, rejection fingerprint, 275-character measured
  layout, root cause, decision, and immutable-verdict flag.
- V2 preserves the exact beta.13 embedded runtime, 296,365,646-byte component,
  680-file payload, copied 31-paper/2,251-chunk corpus, existing semantic index,
  two accepted v3 rankings, timing/memory gates, no-network rule, and disabled
  product/UI boundary. Its only functional candidate change is the compact
  app-owned `test_results/e5p6/c` component store and repaired implementation.
- The protocol requires the actual full-size layout to equal 239 characters and
  remain within the 259-character compatibility budget. This is checked before
  extraction as well as in the final measured gate.
- Seven v2 mutation tests plus the twelve v1 protocol/rejection tests pass.
  Ruff and all three standalone validators pass. Strict `--validate-only`
  preflight also passes and leaves both the v2 result root and report absent.

### Priority assessment after v2 protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-2f-run | Execute the one authorized clean v2 lifecycle measurement and pin its verdict | Very high | It proves or rejects the repaired installer with the real package and runtime | Do now, exactly once |
| P1.3f-4d-3a-3 | If v2 passes, expose only the simplified lifecycle management API and run automatic recovery simulations | Very high, conditional | Required before any ordinary-user control can exist | Next on pass |
| P1.3f-4b | Add explicit selected-paper semantic enable/remove UI | High, conditional | Still blocked on API-level recovery evidence | Not yet |
| P1.3f-5 | Cross-encoder reranking | Deferred | Operability and failure containment remain ahead of ranking complexity | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-2f complete: full-size short-path lifecycle passed

- The single authorized v2 execution passed every frozen gate on its first and
  only run. The report fingerprint is
  `7440961c0be3e681e4c1ac1bb6a9de8dc7a4d78d5c9e7719438893924907709d`;
  the report SHA-256 is
  `1fed1e34dde206b1f51d4d5f6aaae6538fd5ca25f68652f08a22e701459c069b`.
- Real timings were 7.88 s for base extraction, 17.60 s from consent to installed
  component plus ready index, 1.17 s cold runtime load, 581 ms warm-query p95,
  10.86 s for one-document stale-index repair, and 106 ms for pointer-first
  removal. Peak worker RSS was 1,130 MiB, within the frozen 2,048-MiB gate.
- Both selected-paper queries reproduced the exact v3 rankings in
  `hybrid_local` mode with no fallback and zero source escapes. The copied-index
  mutation became `stale`, rebuilt exactly once, and returned to `ready`.
- The measured longest destination path was exactly 239 characters. Removal
  disabled both component and index pointers before deletion; simulated locked
  cleanup completed automatically after worker exit with zero generations left.
  BM25 was available in every public state, no network was used, and product/UI
  remained disabled.
- Added a pinned v2 report validator, seven report-mutation tests, and CI plus
  Windows release validation. Twenty-six v1/v2 protocol/report tests, Ruff,
  standalone validators, and whitespace checks pass. The v1 rejection remains
  separately immutable and is not superseded or relabelled.

### Priority assessment after full-size v2 pass

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-3 | Add a narrow lifecycle management API with only simplified public states, explicit install consent, cancel, and safe remove; simulate failures through HTTP | Very high | Converts proven internals into an auditable boundary without exposing repair work to users | Do next |
| P1.3f-4d-3a-3b | Prove restart recovery, low disk, integrity quarantine, stale index, cancellation, concurrent requests, and removal through automatic API tests | Very high | Required before any UI or product query selection can rely on the API | Same stage |
| P1.3f-4b | Add explicit selected-paper enable/remove UI and ordinary-user E2E | High, conditional | Allowed only after the management API failure matrix passes | Blocked on API E2E |
| P1.3f-5 | Cross-encoder reranking | Deferred | Lifecycle safety still has higher user value | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-3 complete: lifecycle management API and failure E2E

- Added a separate v2 management surface at `/api/retrieval/local-component`
  with explicit endpoints for status, consented enable, cancel, resume, and
  idempotent remove. The legacy preparation endpoints remain untouched for
  compatibility; neither route family enables product retrieval.
- Responses expose only the eight lifecycle public states, a stable action id,
  bounded progress, exact additional bytes for low disk, and the invariant that
  BM25 remains available. Internal states, paths, checksums, exceptions, and
  integrity/index retry controls are not returned.
- Consent disclosure reports the 296,365,646-byte network download,
  541,333,859-byte installed footprint, zero library-content upload, and zero
  external content/model-knowledge addition. A declined consent performs no
  network or disk work. Even a configured manifest's first status read is lazy
  and does not create the lifecycle database before consent.
- The service accepts only an explicitly configured, model-pinned HTTPS manifest
  and allowlisted GitHub release redirect hosts. Without a valid delivery
  manifest it returns `unavailable`/`update_app` rather than offering a dead
  install action. Safe removal remains available and preserves `docs.json`,
  chunks, library content, notes, chat, and BM25 data.
- Automatic HTTP simulations cover all eight public states, management auth,
  no-consent immutability, duplicate requests, low disk, two-attempt integrity
  quarantine, pause/resume, process restart recovery, stale-index rebuild, and
  repeated removal. They verify no path or raw integrity reason escapes.
- Seventy-six focused component/runtime/API tests pass. Ruff, architecture
  budgets, both immutable v1 artifacts, both v2 artifacts, and whitespace checks
  pass. Product mode and ordinary-user UI remain disabled.

### Priority assessment after management API E2E

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-4 | Define and validate the release-owned component manifest artifact and Windows packaging handoff without publishing a release | Very high | API testing found that no formal shipped manifest exists; UI would otherwise expose an install that cannot resolve its package | Do next |
| P1.3f-4d-3a-4b | Exercise the packaged manifest → exact archive identity → lazy API availability chain offline | Very high | Prevents installer metadata drift and keeps first status read side-effect free | Same stage |
| P1.3f-4b | Add explicit selected-paper semantic enable/remove UI and browser E2E | High, conditional | Blocked until the packaged delivery manifest is reproducible | After manifest handoff passes |
| P1.3f-4c | Bind ready v2 service/runtime into the product query selector behind the opt-in state | High, conditional | Must follow UI consent and preserve selected-paper scoping/fail-open BM25 | After UI contract |
| P1.3f-5 | Cross-encoder reranking | Deferred | Still below safe sparse+dense product enablement | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-4 complete: release manifest and packaging handoff

- Added a deterministic Windows component builder. It accepts only the pinned
  E5 artifact tree and exact ONNX Runtime 1.23.2/Tokenizers 0.21.4 entries,
  rejects links/reparse points, case-insensitive collisions, changed package
  versions, overlong relative paths, invalid release tags, and existing outputs.
- The builder excludes `__pycache__`, `.pyc`, and `.pyo`, fixes ZIP timestamps
  and entry ordering, streams the 470-MB model instead of holding it in memory,
  and emits an immutable same-tag GitHub URL, archive checksum file, external
  manifest, and canonical extracted-payload identity. Installation alone is
  explicitly declared not to enable product retrieval.
- Added a separate release validator that rechecks the HTTPS/host/release-id
  binding, archive size/hash, every safe ZIP path, deterministic metadata, CRC
  through complete streaming reads, payload file count/bytes/hash, and the inner
  runtime/component manifest. Local source paths and user-library data are absent.
- Windows release automation now builds the isolated pinned component, validates
  it, embeds only its small manifest into the base app, verifies its checksum,
  retains it as a separate artifact, and adds the archive/checksum/manifest to
  the same-tag prerelease assets. The base portable builder rejects a manifest
  for another version or component.
- Runtime discovery now reads that bundled manifest, with the environment JSON
  retained only as an explicit development/operations override. Both configured
  status and declined consent remain lazy and create no database or component
  files. Disclosure sizes come from the exact bundled manifest.
- The first real offline release-format build produced a 299,558,739-byte ZIP
  with 360 files and a 538,888,249-byte payload. It removed all 320 cached bytecode
  files, passed the independent release validator, and reduced the measured
  e5p8 lifecycle layout maximum to 214 characters. It used no network, did not
  alter source inputs, and did not publish a release.
- Twenty-two release/factory tests, Ruff, workflow YAML parsing, portable-builder
  PowerShell parsing, independent real-archive validation, and whitespace checks
  pass.

### Priority assessment after release handoff

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-4c | Freeze a v3 full-size lifecycle protocol for the deterministic 360-file release candidate and execute once | Very high | The payload changed by removing bytecode and the container hash is new; UI cannot rely only on builder tests | Do next, no retry or tuning |
| P1.3f-4b | Add explicit selected-paper semantic enable/remove UI with browser recovery journeys | High, conditional | Allowed only if the release-format candidate passes real runtime/index lifecycle | Next on v3 pass |
| P1.3f-4c | Bind the ready lifecycle/runtime service into product query selection | High, conditional | Keep BM25 default and selected-paper scope; activate only after UI consent contract passes | After UI |
| P1.3f-5 | Cross-encoder reranking | Deferred | Still lower priority than completing safe dense opt-in | Keep deferred |

### 2026-09-03 — P1.3f-4d-3a-4c protocol frozen: release-format lifecycle v3

- Frozen an independent v3 lifecycle candidate around the deterministic
  360-file release archive. It preserves the full-size v2 protocol/report hashes,
  v2 result fingerprint and pass decision, plus the release builder and validator
  identities; neither earlier verdict can be rewritten by this run.
- The candidate pins the new 299,558,739-byte archive, 538,888,249-byte payload,
  360-file count, component manifest, copied evaluation corpus/index, exact two
  v3 rankings, existing timing/memory limits, and a 214-character exact path gate.
  Product/UI, external network, model download, and user-library writes remain
  forbidden regardless of the result.
- Seven mutation tests, Ruff, standalone validation, and strict runtime/archive/
  corpus `--validate-only` preflight pass. The clean `test_results/e5p8` root and
  v3 report are both absent, so no measurement has started yet.

### Priority assessment after release-format protocol freeze

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4d-3a-4c-run | Execute the one clean release-format lifecycle run and pin pass or rejection | Very high | Final backend evidence before designing ordinary-user consent interaction | Do now, exactly once |
| P1.3f-4b | On pass, implement the selected-paper semantic opt-in/remove UI and browser failure journeys | High, conditional | Must keep BM25 default and never expose internal repair states | Next on pass |
| P1.3f-4c | Bind ready semantic runtime into the selected-paper query path | High, conditional | Only after the user can explicitly control the component lifecycle | After UI contract |

### 2026-09-03 — P1.3f-4d-3a-4c complete: release-format lifecycle v3 passed

- The single authorized v3 execution passed every frozen gate on its first and
  only run. The report fingerprint is
  `b1bde7acb30be4c2b67b23bbd71df10c61e49ed2e66a39d92c3da87dd09db80d`;
  the report SHA-256 is
  `a23a0135c88d5f676eaacc2eb2166fa301ee10fc6d5f08745dc5aec564b739a0`.
- The deterministic 360-file release archive installed, loaded, queried,
  repaired one stale document index, removed its pointers, and completed
  delayed cleanup with no network access or user-library mutation. The two
  selected-paper queries reproduced the exact accepted rankings in
  `hybrid_local` mode with no fallback and zero source escapes.
- Measured timings were 7.37 s for base extraction, 27.53 s for the worker,
  12.03 s from consent to component plus ready index, 1.08 s cold runtime load,
  510 ms warm-query p95, 9.52 s for one-document stale-index repair, and 110 ms
  for pointer-first removal. Peak worker RSS was 1,130.551 MiB, within the
  frozen 2,048-MiB gate.
- The longest installed path was exactly 214 characters, below the frozen
  259-character compatibility budget. BM25 remained available throughout;
  product/UI and external sources remained disabled during the measurement.
- Added a pinned v3 report validator and mutation tests, and wired both v3
  protocol and report validation into CI and Windows release validation.
  Fourteen v3 protocol/report tests, Ruff, standalone validators, and
  whitespace checks pass.
- The runner retained its older generic `next_action` text even though the
  management API is already complete. That descriptive field is non-gating and
  is preserved rather than rewriting measured evidence; the frozen decision
  status explicitly authorizes the opt-in product/UI stage.

### Priority assessment after release-format lifecycle v3 pass

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4c | Bind the ready lifecycle runtime into the selected-paper product query selector behind an explicit `hybrid_local` request, with fail-open BM25 | Very high | Prevents an installed-but-unused UI; default queries and every non-ready state must remain BM25 | Do next, hidden until UI E2E |
| P1.3f-4b | Expose one explicit local-semantic enable/use/remove control and run unattended browser journeys for consent, progress, recovery, and removal | Very high | Makes the proven path usable without exposing checksums, retry chores, or external-source ambiguity | Immediately after selector tests |
| P1.3f-4e | Run full backend/frontend/architecture gates and a real selected-paper end-to-end retrieval comparison before promotion | Very high | Product enablement requires evidence across API, browser, scope fencing, citations, and BM25 fallback | Same promotion stage |
| P1.3f-5 | Add an optional cross-encoder reranker only after retrieval evaluation shows a residual ranking problem | Medium, conditional | Adds package, latency, and lifecycle cost; dense retrieval must first prove product value | Deferred |

### 2026-09-03 — P1.3f-4b/4c complete: explicit local-semantic product path

- Replaced the old experimental runtime default in the product selector with
  the release-owned portable component runtime. A query can reach it only when
  the user explicitly requests `hybrid_local`, the lifecycle is currently
  `ready`, and the effective scope is `current_paper`, `basket`, or another
  explicit selected-source scope. Default, full-library, non-ready, stale, and
  exception paths return the original BM25 objects unchanged.
- Promoted the guarded path in current `main`. The published beta.13 remains
  unchanged. A source build without a valid bundled component manifest shows no
  dead control; the next Windows build embeds the same-tag manifest and exposes
  the control while keeping BM25 as the default.
- Added a separate retrieval control next to, but independent from, the existing
  external-supplement control. First use discloses the exact ~286-MiB download
  and ~514-MiB installed footprint, local-only paper/query handling, and that it
  does not enable web or model knowledge. The preference becomes effective only
  after readiness and only inside an eligible selected-paper/basket scope.
- The ordinary-user lifecycle now closes every actionable state: preparation
  can be paused, paused work can continue, low disk reports the exact additional
  MiB and offers “continue after freeing space,” and removal explains and
  preserves PDFs, notes, chats, and BM25. Integrity/index failure is expressed
  as automatic BM25 continuity with no checksum, verification, or retry chore.
- Three unattended Chromium journeys pass. They exercise consent, real size
  copy, progress, pause/resume, selected-paper send payload, low-space recovery,
  internal-failure containment, full-library fallback, component removal, and
  separate external-source consent. The generated request used
  `retrieval_mode=hybrid_local`, `query_scope=current_paper`, and
  `external_supplement_mode=disabled`; full-library UI remained BM25.
- Added a release-time product-path validator and wired it after creation of the
  final staged embedded runtime. The local real-package execution used the
  299,558,739-byte archive and actual ONNX runtime/model, built a fresh index,
  then traversed the lifecycle-owned runtime through `TaskRetrievalObservation`.
  It completed in 9,001.417 ms with two dense novelty hits, protected the first
  five sparse hits, admitted zero out-of-scope sources, kept full-library and
  default requests byte-for-byte on BM25, and removed the component without
  changing the fixture library hashes or using network/external knowledge.
- The real product-path result SHA-256 is
  `a86846d63286af02c86fb20744376af087c3d5107d0f4a0e49cee1665b77585f`;
  the reusable validator SHA-256 is
  `843d354b0239f0c14a8fcef6d63b814dce469f2eb7ced08e7668c2944143e8db`.
  Two pre-execution harness mistakes (the embedded-runtime subdirectory and a
  store loader name) were corrected before the clean work root was created; the
  actual component/product execution then ran once and passed.
- Two hundred focused component/release/protocol tests, all v1-v3 frozen
  validators, Ruff, workflow YAML parsing, architecture budgets, and whitespace
  checks pass. Frontend build/lint and the broader regression suites remain the
  final promotion gate below.

### Priority assessment after product-path and browser pass

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4e | Run full unit/sanity suites, frontend build/lint, affected chat/browser regressions, and release contract checks | Very high | The path is now product-enabled on `main`; unrelated evidence and citation behavior must not regress | Do now before any further feature work |
| P1.3f-4f | Add a compact user-visible retrieval outcome indicator only when a requested semantic turn fell back to BM25, without exposing repair work | High, conditional | Users should know which retrieval actually ran, but ordinary BM25 turns need no extra noise | Assess from existing trace after gates |
| P1.3f-4g | Measure opt-in success, preparation failure containment, actual-mode rate, latency, and fallback reasons through the existing local observability contract | High | Determines whether dense retrieval helps real reading behavior and whether delivery needs optimization | Next evidence stage |
| P1.3f-5 | Evaluate cross-encoder reranking against observed residual ranking failures; do not add it by checklist alone | Medium, conditional | A second model would add download, memory, latency, consent, and lifecycle burden | Deferred pending product evidence |

### 2026-09-03 — P1.3f-4e/4f complete: full gates and quiet per-turn fallback outcome

- Completed the product-promotion regression gate. The full backend suites pass:
  5,234 unit tests passed with 41 expected skips, and 329 sanity tests passed
  with 2 expected skips. The earlier 200-test focused component/release/protocol
  set and every frozen v1-v3 validator also remain green.
- Frontend lint and the production build pass. The final browser smoke run,
  including five local-semantic journeys, passed 165 tests with 3
  build-conditional skips. Core citation/reader regression passed 127/127,
  public-surface isolation 4/4, hidden evidence-workspace isolation 1/1, and
  generation race coverage 27/27.
- The first full smoke attempt exposed one real compatibility defect: an older
  API or broad test mock could return an incomplete local-component object,
  and the new hook read the missing nested lifecycle state before falling back.
  The boundary now treats every incomplete payload as unavailable and continues
  on BM25; the affected first-run journey then passed 2/2 and the full smoke
  rerun passed cleanly.
- Added a compact per-answer outcome only when an explicit `hybrid_local`
  request actually falls back to BM25 because the semantic runtime could not
  be used. It appears for both the live answer and persisted history as
  “已自动使用 BM25,” contains no internal reason ID, and assigns no verification
  or retry work to the user.
- Ordinary BM25, full-library scope fencing, exact/structured retrieval, and
  successful local semantic turns add no fallback notice. The projection uses
  the already persisted retrieval trace, so it required no database migration
  or growth in the oversized backend runtimes.
- Five focused browser journeys verify positive fallback disclosure, absence of
  `dense_query_failed`, absence of a retry action, persistence through message
  hydration, ordinary-mode silence, full-library silence, and structured-mode
  silence. The full smoke rerun includes all five cases.
- Architecture budgets and whitespace checks pass. A first check correctly
  rejected a one-line increase in `MessageList.tsx`; the rendering seam was
  kept in a new small `localRetrievalOutcome.ts` module and the protected file
  remains exactly at its frozen 3,007-line budget. No budget was raised.
- No release, tag, or published artifact changed. `v0.1.0-beta.13` remains the
  existing BM25-only release; this product path belongs to a future Windows
  build after its normal release decision.

### Priority assessment after promotion gates and fallback outcome

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4g-1 | Add redacted local lifecycle observations for consent, preparation completion, pause, disk containment, degraded BM25 continuity, and removal; join them with existing requested/actual retrieval-mode observations | Very high | Measures whether the optional component works without recording queries, paper text, paths, keys, or external knowledge; exporter failure must remain invisible | Next implementation step |
| P1.3f-4g-2 | Build an aggregate local acceptance report for opt-in completion rate, actual hybrid-use rate, warm latency, fallback rate/reasons, and automatic-containment rate | High | A report, not more user-facing diagnostics, should decide whether delivery/startup needs optimization | After lifecycle observations |
| P2.2 | Extract the assistant notice/streaming-status render seam from the now full-budget `MessageList.tsx` | High architectural | The file is exactly at 3,007/3,007; the next relevant change must shrink it rather than raise the ratchet | Parallel small contract-preserving slice |
| P1.3f-5 | Reconsider a cross-encoder only if measured selected-paper failures remain ranking failures after hybrid retrieval | Medium, conditional | No evidence yet justifies a second model's package, memory, latency, and lifecycle burden | Continue to defer |

### 2026-09-03 — P1.3f-4g-1/4g-2 complete: privacy-safe lifecycle evidence and aggregate reporter

- Added local observation contract v2 without changing the frozen v1 file. The
  new contract adds one `pi_zaya.local_semantic_component` event and extends
  product retrieval observations with requested mode, actual mode, automatic
  fallback, cold-start, and index-cache flags. The sink remains disabled unless
  `KB_LOCAL_OBSERVABILITY_PATH` is explicitly configured, size-rotated locally,
  fail-open, and has no external exporter.
- Lifecycle orchestration now records one redacted start and one terminal result
  for enable/resume/recover/remove work. Terminal outcomes distinguish ready,
  user pause, required disk space, automatic BM25 containment, removal, and
  unavailable builds. Only closed operation/state/failure categories, attempt
  count, consent state, duration, and BM25 availability can enter the sink;
  task/session/manifest IDs, download URLs, paths, paper/query/evidence content,
  model names, exception text, and required byte counts cannot.
- Joined each product retrieval observation to its requested and actual mode.
  An explicit semantic request that used BM25 is now measurable as an automatic
  fallback, while ordinary Agent/BM25 observations remain compatible and do not
  acquire invented mode fields. Warm latency is identifiable only when
  `cold_start=false` and `index_cache_hit=true`.
- Added `kb/local_semantic_acceptance.py` and
  `tools/observability/build_local_semantic_acceptance_report.py`. The report
  contains only aggregate counts/rates, closed fallback categories, and warm
  hybrid P50/P95/max latency. Empty samples produce `null` plus explicit sample-
  sufficiency flags; malformed lines and pre-v2 events are counted as ignored,
  not treated as zero-valued success. Window-boundary imbalance is surfaced so
  rotated logs cannot silently overstate opt-in completion.
- The aggregate tool was exercised end to end against a JSONL privacy canary.
  No trace/span ID, internal failure ID, query, paper content, or path is copied
  into its report. The implementation intentionally does not fabricate a real-
  user acceptance result: no released build has accumulated a v2 observation
  window yet.
- Thirty-nine focused privacy/lifecycle/retrieval/reporter tests pass. The full
  backend gate passes with 5,242 unit tests and 41 expected skips; sanity plus
  Agent end-to-end passes 334 tests with 2 expected skips. Ruff, the frozen v2
  contract validator, all architecture budgets, whitespace checks, frontend
  lint and production build pass. The affected unattended Chromium suite passes
  5/5 local-semantic journeys. The initial frontend commands were denied by the
  filesystem sandbox at the Node user-directory lookup and passed unchanged
  outside that sandbox; this was an execution-environment restriction, not a
  product failure.
- No release, tag, or published artifact changed. `v0.1.0-beta.13` remains the
  existing published build.

### Priority assessment after local acceptance reporting is measurement-ready

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P1.3f-4g-3 | Make the existing real-package product-path validator emit a temporary v2 observation window and validate the aggregate report invariants | Very high | Proves the packaged ONNX lifecycle, actual-mode join, warm/cold split, removal, privacy, and reporter as one unattended chain; synthetic unit data alone is insufficient | Do next |
| P2.2 | Extract assistant notice and streaming-status rendering from `MessageList.tsx` | High architectural | The protected file remains exactly 3,007/3,007; the next relevant UI change must shrink it without changing behavior | Immediately after the release-path observation rehearsal |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window after a future release, if local collection was explicitly enabled | High, release-dependent | Only actual opt-in reading sessions can decide delivery/startup optimization; unavailable samples must remain unknown rather than zero | Wait for legitimate data; do not block current BM25 use |
| P1.3f-5 | Add a cross-encoder only if aggregate and case-level evidence isolates residual ranking failures after successful hybrid retrieval | Medium, conditional | It adds a second model, package size, memory, latency, consent, and lifecycle complexity | Continue to defer |

### 2026-09-03 — P1.3f-4g-3 complete: real packaged-runtime observation rehearsal

- Upgraded the existing release product-path validator from a direct ready-state
  fixture to the actual lifecycle service. It now exercises explicit enable,
  archive installation, runtime health, index construction, one cold and one
  warm selected-paper hybrid query, full-library scope fencing, default BM25,
  and component removal. Each retrieval calls the same
  `TaskRetrievalObservation.record` path used by product tasks.
- The validator owns a temporary observation sink and `LOCALAPPDATA` root inside
  its newly created work directory, restores the caller environment, clears
  runtime caches, deletes the raw JSONL in a `finally` block, and persists only
  the aggregate report. This closes a validator-isolation gap found during the
  first attempt: a long test path selected the Windows short-path fallback under
  the real user profile, which the sandbox correctly refused. No product path or
  path-length threshold was weakened.
- A second diagnostic attempt used the system Anaconda Python 3.9 and was
  contained as `component_install_health_failed`; the component is pinned to
  Python 3.10.11. The authoritative replay therefore used the existing packaged
  Python 3.10.11 runtime, matching the Windows release workflow, and passed.
  Both failed disposable work roots were removed after their ledgers were read;
  they contained only generated fixtures and are not recoverable.
- The passing real replay used the pinned 299,558,739-byte component archive and
  completed in 15,098.844 ms. It produced eight v2 events, accepted all eight,
  completed one opt-in lifecycle, removed one component, preserved the fixture
  library hashes, and retained zero raw observation files. Both selected-paper
  runs used hybrid retrieval with two dense novelty results, an unchanged top-
  five sparse prefix, and zero source escapes. The one warm sample was 45.263 ms.
- The aggregate actual-hybrid rate is 2/3 and automatic-fallback rate is 1/3
  only because the release validator deliberately issues one otherwise-invalid
  full-library hybrid request to prove the scope fence; that event is correctly
  categorized as `source_safety_boundary`. These controlled ratios are not
  presented as real-user adoption or reliability measurements.
- The aggregate result SHA-256 is
  `c97d49c295d46eaaca7ace02ddb07a4b373b92bc932ff6e5e81f4c6969a15b23`;
  the upgraded validator SHA-256 is
  `a5103a995c1bf44ff99d958bbc08ced80375b84118f8ea860db59a4f2f3152a0`.
  Fifty-one focused release/lifecycle/reporter tests, Ruff, contract
  validation, architecture budgets, and whitespace checks pass after the
  change. The earlier full backend, frontend, and 5/5 browser gates remain
  applicable because this final slice changes only the offline aggregate parser
  and release validator, not product API, frontend, or retrieval selection.

### Priority assessment after the packaged observation chain passes

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.2 | Extract assistant notice and streaming-status rendering from `MessageList.tsx` into a bounded component | Very high architectural | The file is frozen at 3,007/3,007; the next UI work otherwise has no safe growth room | Do next, behavior-preserving |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | The rehearsal proves plumbing, not adoption or field reliability; missing data remains unknown | Wait for legitimate released data |
| P1.4c | Use aggregate retrieval/verification outcomes to prioritize concrete evidence-location failures rather than add more user diagnostics | High | Directly improves reading and evidence finding while keeping internal repair work invisible | After the render extraction or when valid field data exists |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | Current evidence does not justify its second model and lifecycle burden | Continue to defer |

### 2026-09-03 — P2.2 complete: assistant streaming presentation extracted and ratcheted

- Added the bounded 106-line `AssistantStreamingMessage.tsx` component and
  moved assistant-avatar ownership plus all in-progress answer presentation out
  of `MessageList.tsx`. The parent now supplies generation state and evidence
  callbacks instead of deciding how partial text, activity state, notices, and
  traces are arranged.
- Preserved the existing reading-assistant behavior: partial-answer cleanup,
  leading source-notice separation, answer-contract source-summary precedence,
  empty-stream typing feedback, local semantic BM25 fallback disclosure, and
  Agent/Research trace panels. This is a render-boundary change only; it adds no
  writing, drafting, or submission workflow.
- Reduced protected `MessageList.tsx` from 3,007 to 2,950 lines (-57, 1.9%) and
  lowered its architecture ratchet to 2,950. No architecture budget was raised.
  Total production lines do not fall by the same amount because the extracted
  component makes the former inline behavior an independently testable module;
  the measurable gain is a narrower ownership boundary and lower change blast
  radius, not a claim of aggregate code deletion.
- Added two user-level regression fixtures and a dedicated Chromium spec. They
  prove that an empty assistant stream retains activity feedback and three
  typing dots, while a visible partial answer presents the answer-contract
  source summary and automatic hybrid-to-BM25 fallback outside the answer body.
  The new spec is part of the maintained frontend smoke gate.
- Frontend lint and production build pass. Focused streaming regressions pass
  3/3; the smoke suite passes 167 tests with 3 expected build-conditional skips
  (170 total); core browser journeys pass 127/127; public-surface checks pass
  4/4; hidden-evidence-workspace checks pass 1/1; and generation-race checks pass
  27/27. Ruff for the changed architecture checker, architecture budgets, and
  whitespace checks also pass. Backend product code did not change, so the
  previous full backend gate remains applicable.
- One operational test-harness constraint was made explicit by the run: the
  hidden-evidence and public-surface build variants cannot safely mutate the
  same frontend build workspace concurrently. Their first parallel run caused
  the hidden fixture not to load; the unchanged hidden suite passed 1/1 when
  rerun alone. These build-mutating gates must run serially. This is test-runner
  interference rather than a product behavior failure, but it must not be
  hidden or mistaken for a flaky user journey.
- No release, tag, or published artifact changed. `v0.1.0-beta.13` remains the
  published build.

### Priority assessment after the assistant streaming seam extraction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3a | Extract the completed-answer action footer (copy, save as evidence note, retry) from `MessageList.tsx`, with focused browser contracts | Very high architectural | It is the next small stable leaf in the remaining 2,950-line component and directly covers common reading actions without moving citation-location policy | Do next, behavior-preserving |
| P2.3b | Separate the completed assistant-message presentation after a stable locator view-model seam exists | High architectural | The remaining row still mixes citation location, evidence actions, answer notices, and render policy; moving it wholesale now would create a large high-risk component | Follow P2.3a in bounded slices |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | The rehearsal proves plumbing, not adoption or field reliability; missing data remains unknown | Wait for legitimate released data |
| P1.4c | Use valid field outcomes to prioritize concrete evidence-location failures instead of exposing unresolved diagnostics to users | High, data-dependent | This improves reading and evidence finding at the cause; current unreleased local rehearsal is not representative user evidence | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | Current retrieval evidence still does not justify a second model, package, consent, latency, and lifecycle burden | Continue to defer |

### 2026-09-03 — P2.3a complete: completed-answer actions extracted and failure notes fenced

- Added the bounded 63-line `AssistantMessageActions.tsx` component for the
  existing completed-answer actions: copy text, copy Markdown, add an eligible
  local answer to a research note, and retry the matching failed turn. Evidence
  eligibility, retry classification, prompt recovery, and generation state stay
  in their existing policy/orchestration layers; the component only presents
  the resulting contract.
- Reduced protected `MessageList.tsx` from 2,950 to 2,933 lines and lowered its
  architecture ratchet to 2,933. This is a further 17-line reduction in the
  orchestration surface and a cumulative 74-line reduction across P2.2-P2.3a.
  No architecture budget was raised.
- The first production build found that historical messages may legitimately
  omit `copy_markdown`. The extracted component initially narrowed that field
  to a required string; its public prop was corrected to remain optional, the
  same as `CopyBar`, rather than fabricating an empty value in the parent.
- End-to-end action testing exposed a separate user-facing quality defect:
  terminal generation-failure messages without a modern quality contract were
  still considered eligible for a research note by the legacy fallback. The
  research-note eligibility source now rejects recognized generation-failure
  answers. Failed turns remain copyable and retryable, but cannot be organized
  as evidence. This prevents an error string from entering the user's evidence
  library without changing valid local-answer behavior.
- Added a dedicated browser contract that verifies exact text/Markdown copy
  payloads, the local evidence-note handoff, retry of the matching prior user
  question, retry disablement during another active generation, and absence of
  the research-note action for a failed answer. The existing research-note
  helper contract now also proves that failure answers are excluded from bulk
  note construction. The new action spec is part of the maintained smoke gate.
- Frontend lint and the 3,914-module production build pass. Focused action and
  eligibility checks pass 3/3; the expanded smoke suite passes 169 tests with 3
  expected build-conditional skips (172 total); the core browser suite passes
  127/127; the full generation-race suite passes 27/27; public-surface checks
  pass 4/4; and the hidden evidence-workspace check passes 1/1 when run
  serially. Ruff for the changed budget checker, architecture budgets, and
  whitespace checks pass. Backend product code did not change, so the prior
  full backend gate remains applicable.
- The initial in-sandbox Node invocation again failed at the restricted user-
  directory `lstat`; the exact lint/build commands passed outside that sandbox.
  This is an execution-environment restriction, not a product failure. The two
  build-mutating public-surface variants were intentionally run serially as
  required by the P2.2 test finding, and both passed on their first run.
- No release, tag, or published artifact changed. `v0.1.0-beta.13` remains the
  published build.

### Priority assessment after the completed-answer action seam

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3b-1 | Extract completed-answer source notices plus primary Markdown rendering behind the existing typed locate-props seam | Very high architectural | This is the next cohesive render leaf in the still full-budget 2,933-line component; moving references, traces, and actions at the same time would create an unsafe broad prop contract | Do next as a behavior-preserving bounded slice |
| P2.3b-2 | Separate reference candidates, structured provenance, and trace panels only after P2.3b-1 establishes a compact interaction contract | High architectural | These panels share reader/shelf callbacks but not primary answer rendering; keeping the second slice separate limits citation-location blast radius | Follow P2.3b-1 if its focused tests remain stable |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | Packaged rehearsal data is not representative field evidence | Wait for legitimate released data |
| P1.4c | Use valid field outcomes to prioritize concrete evidence-location failures instead of exposing unresolved diagnostics to users | High, data-dependent | Current local rehearsal cannot identify real-user failure prevalence | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | No current measurement justifies the second model and lifecycle burden | Continue to defer |

### 2026-09-03 — P2.3b-1 complete: completed-answer primary content extracted

- Added the bounded 59-line `AssistantCompletedMessageContent.tsx` component
  for completed-answer source notices and primary Markdown rendering. It accepts
  the existing typed `MessageMarkdownLocateProps` seam, so citation clicks,
  hover behavior, strict evidence-location controls, and linkification policy
  remain owned by the parent orchestration instead of being reimplemented.
- Moved completed-answer low-confidence presentation preparation into the
  separate 33-line `assistantCompletedContentState.ts` module. The state builder
  resolves the existing low-confidence contract and removes only its recognized
  leading notice from the answer body; downstream citation and locate
  preparation continue to consume the same cleaned body.
- Reduced protected `MessageList.tsx` from 2,933 to 2,923 lines and lowered its
  architecture ratchet to 2,923. P2.2 through P2.3b-1 have now removed 84 lines
  from this orchestration surface. As with earlier extraction slices, the goal
  is narrower ownership and independent contracts rather than claiming an
  equivalent reduction in total repository lines. No budget was raised.
- The first lint run correctly rejected the initial version because a React
  component file also exported a non-component state builder, which breaks the
  fast-refresh module contract. The pure builder was moved to its own `.ts`
  module; the component file now exports only the React component, and lint
  passes without disabling or weakening the rule.
- Added three focused browser contracts: a completed local answer keeps its
  source notice outside Markdown while citations remain interactive; strict
  locate attributes and reader payload survive the component boundary; and a
  diagnosed low-confidence notice is separated from the stored body while its
  deduplicated candidate references remain available. The new spec is included
  in the maintained smoke gate.
- Frontend lint and the 3,916-module production build pass. Focused new and
  pre-existing content/locate contracts pass 5/5; the expanded smoke suite
  passes 172 tests with 3 expected build-conditional skips (175 total); the
  core citation/reader suite passes 127/127; the generation-lifecycle suite
  passes 27/27; public-surface checks pass 4/4; and the hidden evidence-
  workspace check passes 1/1 when run serially. Ruff for the changed budget
  checker, architecture budgets, and whitespace checks pass. Backend product
  code did not change, so the prior full backend gate remains applicable.
- The in-sandbox Node attempt remained blocked by the known restricted user-
  directory lookup; the exact commands passed outside that sandbox. The public
  and hidden build variants were again run serially and passed on their first
  run. No product failure or new test flake was observed.
- No release, tag, or published artifact changed. `v0.1.0-beta.13` remains the
  published build.

### Priority assessment after the primary completed-answer content seam

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3b-2a | Extract research-context receipt, unlinked reference candidates, and structured provenance chips behind one typed evidence-panel interaction contract | Very high architectural | These are all traceable-evidence navigation surfaces and share reader/shelf interactions; keeping Agent/Research traces out prevents the prop contract from becoming broad | Do next as one bounded render slice |
| P2.3b-2b | Extract Agent and Research trace presentation after P2.3b-2a, using a separate audit interaction contract | High architectural | Audit loading and reference actions have different lifecycle semantics from evidence navigation | Follow only after P2.3b-2a remains stable |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | Packaged rehearsal data is not representative field evidence | Wait for legitimate released data |
| P1.4c | Use valid field outcomes to prioritize concrete evidence-location failures instead of exposing unresolved diagnostics to users | High, data-dependent | Current local rehearsal cannot identify real-user failure prevalence | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | No current measurement justifies the second model and lifecycle burden | Continue to defer |

### 2026-09-03 — P2.3b-2a complete: evidence-navigation panels extracted

- Added the bounded 49-line `AssistantEvidenceNavigationPanels.tsx` component
  for the existing research-context receipt, unlinked reference candidates, and
  structured provenance locate chips. Their order and visibility inputs remain
  unchanged; the new component performs no source, policy, or locator
  inference.
- Added the 12-line `assistantEvidenceNavigationTypes.ts` interaction contract.
  It intentionally keeps three payload families distinct: citation details for
  reference actions, selected research-context packs/items for follow-up, and
  structured provenance entries plus snippets for exact evidence navigation.
  The raw reader-open callback remains optional and is the sole source of the
  candidate panel's open capability.
- Reduced protected `MessageList.tsx` from 2,923 to 2,916 lines and lowered its
  architecture ratchet to 2,916. P2.2 through P2.3b-2a have now removed 91
  lines from this orchestration surface. No architecture budget was raised.
- Added a browser contract proving that an unlinked bibliography candidate
  opens the bound local source and that its separate basket action preserves
  the same paper identity. The test is now part of the maintained smoke gate.
  Focused execution also reran the existing selected-basket context receipt and
  required structured-locate contracts; all three passed.
- Frontend lint and the 3,917-module production build pass. The expanded smoke
  suite passes 173 tests with 3 expected build-conditional skips (176 total);
  core citation/reader journeys pass 127/127; generation-race checks pass
  27/27; public-surface checks pass 4/4; and the hidden evidence-workspace gate
  passes 1/1 when run serially. Ruff for the changed budget checker,
  architecture budgets, and whitespace checks pass. Backend product code did
  not change, so the prior full backend gate remains applicable.
- The latest release runbook was rechecked before closing the slice. No build,
  version, release, or tag workflow was started: the working tree is not a
  distributable artifact, and the immutable `v0.1.0-beta.13` publication at
  tag `9c78a763` remains unchanged.

### Priority assessment after the evidence-navigation seam

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3b-2b | Extract Agent and Research trace presentation behind a separate typed audit interaction contract | Very high architectural | The remaining completed-answer row still owns audit loading and trace-reference actions; mixing them into evidence navigation would blur lifecycle semantics | Do next as a bounded render slice |
| P2.3c | Reassess `MessageList.tsx` state ownership after the audit seam rather than continue mechanical JSX extraction | High architectural | The file will still be large; the next useful boundary may be a stateful domain hook, but that choice should follow measured post-extraction shape | Decide after P2.3b-2b gates pass |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | Packaged rehearsal data is not representative field evidence | Wait for legitimate released data |
| P1.4c | Use valid field outcomes to prioritize concrete evidence-location failures instead of exposing unresolved diagnostics to users | High, data-dependent | No representative opt-in field sample exists yet | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | Current evidence still does not justify a second model, package, consent, memory, and latency burden | Continue to defer |

### 2026-09-03 — P2.3b-2b complete: audit-trace panels extracted

- Added the bounded 35-line `AssistantAuditTracePanels.tsx` component and the
  6-line `assistantAuditTraceTypes.ts` contract. Completed assistant messages
  now hand one typed audit interaction bundle to that boundary instead of
  wiring Agent and Research trace panels independently in `MessageList.tsx`.
- Kept the two audit lifecycles distinct inside the boundary. Agent traces
  still derive their saved-trace hint from the message, load the exact message
  audit lazily through `chatApi.getMessageAgentTrace`, and preserve separate
  open-reference and add-to-basket actions. Research traces still pass through
  their existing internal-build plus browser-flag guard and remain absent from
  ordinary user builds.
- Reduced protected `MessageList.tsx` from 2,916 to 2,908 lines and lowered its
  architecture ratchet to 2,908. P2.2 through P2.3b-2b have now removed 99
  lines from this orchestration surface. No architecture budget was raised.
- Promoted the existing two-case `research-trace-debug.spec.ts` into the core
  browser gate instead of adding a duplicate test file. With the internal
  debug build flag enabled, it proves both ordinary-user suppression and the
  deliberate two-flag debug reveal. Focused trace loading, reference opening,
  basket insertion, and visibility checks pass 4/4.
- Frontend lint and the 3,918-module production build pass. The smoke suite
  remains 173 passed with 3 expected build-conditional skips (176 total); the
  expanded core suite passes 129/129; generation-race checks pass 27/27;
  public-surface checks pass 4/4; and the hidden evidence-workspace gate passes
  1/1 when run serially. Backend product code did not change, so the prior full
  backend gate remains applicable.
- No release, version, or tag workflow ran. The published
  `v0.1.0-beta.13` tag `9c78a763` remains unchanged.

### Priority assessment after the audit-trace seam

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3c-1 | Extract research-note collection loading, cross-window refresh, count reporting, upsert, and delete into a focused hook | Very high architectural | This is the next cohesive state domain in `MessageList.tsx`; it can move without touching reader navigation or the citation shelf's much larger persistence lifecycle | Do next, leaving editor and cross-conversation navigation in the parent |
| P2.3c-2 | Freeze a typed citation-shelf lifecycle boundary before moving any shelf state | Very high architectural, higher risk | Shelf ownership spans local/backend revision guards, async scope tokens, repair, hydration, summary backfill, reader events, and snapshots; moving it wholesale would risk evidence loss | Inventory and split only after P2.3c-1 passes broad gates |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | Packaged rehearsal data is not representative field evidence | Wait for legitimate released data |
| P1.4c | Use valid field outcomes to prioritize concrete evidence-location failures instead of exposing unresolved diagnostics to users | High, data-dependent | No representative opt-in field sample exists yet | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | Current evidence still does not justify its second model and lifecycle cost | Continue to defer |

### 2026-09-03 — P2.3c-1 complete: research-note collection lifecycle extracted and stale mutations fenced

- Added the focused 109-line `useResearchNoteCollection.ts` hook. It now owns
  project-scoped note loading, request-sequence guards, window and
  `BroadcastChannel` refresh subscriptions, count reporting, sorted upsert, and
  deletion feedback. `MessageList.tsx` no longer carries those collection
  states, refs, effects, or mutation callbacks.
- Strengthened the lifecycle while extracting it. A project switch clears the
  previous project's visible collection immediately; a late response from the
  old scope is rejected; successful saves and deletes invalidate older loads
  and request a canonical post-mutation snapshot; and a saved record from an
  already-left project is not inserted into the active project. A delayed
  pre-delete refresh therefore cannot resurrect the deleted note or restore a
  stale count.
- Kept product scope narrow. The research-note editor, answer-to-note evidence
  eligibility, exact reader navigation, and cross-conversation source opening
  stay in their existing owners. This is traceable evidence organization for
  papers in the library, not manuscript drafting or submission preparation.
- Reduced protected `MessageList.tsx` from 2,908 to 2,852 lines (-56) and
  lowered its architecture ratchet to 2,852. The cumulative reduction across
  P2.2 through P2.3c-1 is 155 lines. No architecture budget was raised.
- Extended the internal regression page with an opt-in research-note portal and
  count probe, and added a maintained smoke test. The browser contract starts a
  delayed refresh in project A, switches to project B, proves the old A response
  cannot overwrite B, refreshes B through the public change event, then starts
  another delayed B refresh and proves deletion plus count zero survive that
  stale response.
- The new focused regression passes 1/1. Frontend lint and the 3,919-module
  production build pass. On the final code, the smoke suite passes 174 tests
  with 3 expected build-conditional skips (177 total); the internal-debug core
  suite passes 129/129; generation-lifecycle checks pass 27/27; public-surface
  checks pass 4/4; and the hidden evidence-workspace check passes 1/1 when run
  serially. Backend product code did not change, so the prior full backend gate
  remains applicable.
- One earlier smoke run timed out while an unrelated research-gap test waited
  eight seconds for its audit-ready message. The unchanged test passed alone,
  and both subsequent complete smoke runs passed it under four-worker load.
  This was a recorded one-off parallel-load delay, not concealed as a passing
  run and not addressed by weakening its assertion.
- No release, version, or tag workflow ran. The published
  `v0.1.0-beta.13` tag `9c78a763` remains unchanged.

### Priority assessment after the research-note collection seam

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3c-2a | Extract the reader-to-literature-basket window/`BroadcastChannel` subscriptions behind a typed event hook | Very high architectural | It is the smallest independently testable slice of the remaining shelf lifecycle and directly protects figure, equation, table, and selected-text evidence handoff without moving persistence | Do next with stale-scope and cleanup browser contracts |
| P2.3c-2b | Inventory and freeze the local/backend shelf scope, revision, and persistence contract before moving any shelf state | Very high architectural, higher risk | The remaining lifecycle still spans restoration, migration, revision guards, async repair, hydration, summary backfill, and snapshots; moving it wholesale could lose evidence | Follow P2.3c-2a; split only along measured ownership seams |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | Packaged rehearsal data is not representative field evidence | Wait for legitimate released data |
| P1.4c | Use valid field outcomes to prioritize concrete evidence-location failures instead of exposing unresolved diagnostics to users | High, data-dependent | No representative opt-in field sample exists yet | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | Current evidence still does not justify a second model, package, consent, memory, and latency lifecycle | Continue to defer |

### 2026-09-04 — P2.3c-2a complete: reader-to-shelf event boundary and immediate-switch persistence

- Re-read `AGENTS.md`, the release runbook, and the latest plan before changing
  the existing dirty working tree. No existing retrieval, evidence, or release
  gate was removed, and no manuscript-writing workflow was added.
- Added the bounded 86-line `useReaderShelfEvents.ts` hook. It owns both window
  events and both `BroadcastChannel` subscriptions, normalizes their existing
  payloads, and routes them through typed citation/selection callbacks. The
  callbacks and project identity follow the latest committed render without
  repeatedly creating subscriptions. Missing-project legacy payload behavior
  remains compatible; explicitly bound payloads cannot enter another project.
- Fixed acknowledgement semantics: malformed, wrong-type, wrong-project, and
  callback-declined events no longer receive a success acknowledgement. A
  positive acknowledgement means local acceptance, not proof of completed
  backend persistence. Cleanup detaches the handlers, closes both channels,
  removes window listeners, and rejects already-queued callbacks after unmount.
- The new browser journey found a real adjacent persistence race. Adding
  evidence and immediately switching projects could cancel the debounced
  backend save while a layout effect relabeled the old items with the new
  project. The old project retained its local snapshot but missed its backend
  update. A narrow storage-key guard now keeps the last snapshot bound to its
  old project until the existing scope effect flushes and restores state.
  Reader callbacks also reject a not-yet-aligned active snapshot. The larger
  persistence implementation remains in place rather than being moved wholesale.
- Reduced protected `MessageList.tsx` from 2,852 to 2,788 lines (-64), and
  ratcheted its budget to 2,788. The cumulative reduction since P2.2 is 219
  lines. This measures reduced orchestration ownership, not an equivalent
  reduction in aggregate repository lines; no pre-step budget was raised.
- Added two maintained smoke contracts in `reader-shelf-events.spec.ts`. The
  lifecycle harness proves current callbacks, scope filtering, truthful
  acknowledgements, no subscription churn on rerender, late-callback cleanup,
  and window-only operation without `BroadcastChannel`. The real browser
  channel journey proves project-isolated basket insertion, exact source and
  locator preservation, and a final project-A addition followed by a project-B
  switch in the same task before the save debounce can fire. Both projects'
  backend snapshots retain only their own accepted evidence.
- The first new test incorrectly expected raw table Markdown in the basket's
  compact display excerpt. Its assertion was corrected to the pre-existing
  normalized-text contract while retaining exact source, block, anchor, kind,
  and conversation assertions. The separate original-Markdown note-capture
  regression still passes; no product content-normalization rule was changed.
  A test helper's TypeScript parameter-property syntax was also replaced with
  explicit field assignment to respect `erasableSyntaxOnly`, without changing
  compiler settings.
- Final validation: focused new browser checks 2/2; frontend lint and the
  3,920-module production build pass; smoke 176 passed with 3 expected
  build-conditional skips (179 total); internal-debug core 129/129; generation
  lifecycle 27/27; public-surface 4/4; hidden evidence workspace 1/1, run
  serially. Ruff for the changed checker, architecture budgets, and whitespace
  checks pass. Backend product code did not change, so the prior full backend
  gate remains applicable. These are isolated browser/API-fixture regressions,
  not a new claim of packaged-release or real-user field acceptance.
- No commit, version, release, or tag workflow ran. The published
  `v0.1.0-beta.13` tag `9c78a763` remains unchanged.

### Priority assessment after reader-to-shelf event extraction

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3c-2b-1 | Freeze the typed shelf scope/snapshot/pending-save contract and cover rapid A-to-B-to-A switching, delayed hydration, and unmount/pagehide | Very high data safety | This slice found an actual lost-backend-save race in the mixed layout/effect ownership; correctness of transitions now outranks further mechanical extraction | Do next, reusing existing gates and adding only missing deterministic cases |
| P2.3c-2b-2 | Extract the bounded persistence scheduler only after the transition contract is stable | High architectural, higher risk | Local revisions, backend hydration, explicit empty intent, pending timers, and stale callbacks must remain aligned; repair and summary work have separate lifecycles | Follow the frozen contract; do not move all shelf state at once |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | Packaged rehearsal data is not representative field evidence | Wait for legitimate released data |
| P1.4c | Use valid field outcomes to prioritize concrete evidence-location failures instead of exposing unresolved diagnostics to users | High, data-dependent | No representative opt-in field sample exists yet | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures after successful hybrid retrieval | Medium, conditional | Current evidence does not justify an additional model and lifecycle burden | Continue to defer |

### 2026-09-04 — P2.3c-2b-1 complete: typed shelf transition contract and delayed-hydration repair

- Re-read the repository instructions, release runbook, and latest progress.
  Preserved the existing dirty tree, all quality gates, and the library-reading
  product boundary. No release or version change was requested or performed.
- First added controlled-response browser journeys against the real
  `MessageList` and shelf components. A-to-B-to-A switching and immediate
  pagehide/unmount flushes passed. Both deletion cases failed before the fix:
  removing the last visible item or clearing the basket while its initial GET
  was pending allowed that old GET to reinsert the removed paper.
- Added the 93-line `citeShelfPersistence.ts` domain boundary with typed
  `ShelfScopeSnapshot`, `ShelfHydrationRequest`, and `ShelfPendingSave` contracts.
  Production hydration and save eligibility now use it; it is not an unused
  interface-only extraction. Timers, storage migration, repair, metadata
  hydration, summary backfill, and saved collections retain their existing owners.
- Replaced the current-request ten-millisecond wall-clock mutation heuristic
  with a request-start snapshot comparison. A delayed response excludes papers
  removed since that request, preserves other backend-only papers, honors a
  distinct clear revision, and keeps additions made after clear. An untouched
  empty local cache still restores backend evidence even if its timestamp is
  newer; it cannot impersonate an explicit deletion.
- PATCH and DELETE acknowledgements no longer grant hydration authority to a
  newly entered project scope. The ordinary hydration/save/delete paths keep
  backend revisions monotonic, and an old empty-save response cannot consume a
  different pending empty intent. This does not claim to serialize all writes
  or change the separate, hidden evidence-workspace save paths.
- `MessageList.tsx` shrank from 2,788 to 2,744 lines (-44), with its budget
  ratcheted to 2,744. The cumulative reduction since P2.2 is 263 lines. The
  domain module and tests add repository lines; this is reduced orchestration
  ownership, not a claim that total source size fell by that amount.
- Added six maintained smoke tests covering repeated scope entry, deliberately
  reordered GETs, delayed deletion/clear, new evidence during hydration,
  immediate pagehide/unmount, late save acknowledgements, partial deletion with
  unseen backend evidence, clear-then-add, explicit-empty protection, and typed
  scope ownership. The internal harness can now switch back and unmount; no
  public product controls were added.
- Final validation: the focused six-test suite, frontend lint, the
  3,921-module production build, Ruff for the budget checker, architecture
  budgets, and whitespace checks pass. Serial full-suite reruns pass: smoke
  182 passed with 3 expected build-conditional skips (185 total), internal-debug
  core 129/129, generation lifecycle 27/27, public surface 4/4, and hidden
  evidence workspace 1/1. The public variants were also run serially. Backend
  product code did not change, so the prior full backend gate remains applicable
  and was not rerun in this slice. These are isolated browser/API-fixture checks,
  not packaged-release or representative real-user field acceptance.
- Initial overlapping smoke/core runs are recorded as failures, not acceptance:
  smoke 180 passed, 2 failed, 3 skipped;
  core 126 passed, 3 failed. Traces show startup/initial-conversation timeouts,
  one browser-page setup timeout, and a shared Playwright artifact-directory
  `ENOENT`. The unchanged complete suites passed when run serially, including
  all five previously failed cases. Neither suite's four workers, timeouts,
  nor assertions were weakened. Future local gate runs sharing `test-results`
  must not overlap; separate whole-suite execution also avoids excess browser
  load while retaining each gate's existing internal parallelism.
- No commit, version, release, or tag workflow ran. Published
  `v0.1.0-beta.13` and its immutable tag `9c78a763` remain unchanged.

#### Frozen ownership and persistence rules

| Boundary | Rule |
|---|---|
| Scope snapshot | Items, open state, project, and conversation travel together; a later render cannot relabel old evidence |
| Hydration response | Both request sequence and current project must match; A-to-B-to-A cannot revive the first A request |
| In-visit mutation | Reconcile against the request-start snapshot; removed identities stay removed and clear is distinct from removing one known item |
| Empty cache | A restored/default empty state is not permission to overwrite a nonempty backend basket |
| Pending save | Capture a typed project-bound request body only after the current scope has hydrated; an explicit empty write needs intent |
| Save acknowledgement | Update only the bound project's revision/acknowledged intent; do not hydrate a later visit |
| Leaving the component | Flush local accepted evidence and eligible backend work, cancel owned timers, and detach event listeners; browser shutdown itself is not a durable HTTP acknowledgement |

### Priority assessment after the shelf transition contract

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3c-2b-2a | Specify and test retained per-project mutation intent plus ordered save/delete handling across failed requests, leaving, and re-entry | Very high data safety | This slice fixes reconciliation within a live hydration request; the existing five-second empty intent and uncoordinated network writes are not a durable outbox, and current tests do not prove deletion recovery across browser restart | Do next before further lifecycle extraction; use controlled failures and response order, including remove-before-hydration followed by re-entry |
| P2.3c-2b-2b | Extract a bounded persistence scheduler around the proven scope and mutation contract | High architectural | Moving timers alone would preserve ambiguous pending-write ownership; repair and summaries have separate lifecycles | Follow ordered-save evidence; do not move all shelf state wholesale |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window only after a future release and explicit local collection choice | High, release-dependent | Packaged rehearsal data is not representative field evidence | Wait for legitimate released data |
| P1.4c | Prioritize concrete evidence-location failures from valid field outcomes | High, data-dependent | No representative opt-in field sample exists yet | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures | Medium, conditional | No new ranking evidence justifies another model, consent, memory, and latency lifecycle | Continue to defer |

### 2026-09-04 — P2.3c-2b-2a complete: retained shelf mutations and ordered retry

- Re-read the repository instructions, release runbook, and latest progress.
  Preserved the existing dirty tree, product boundary, and all quality gates.
  No commit, version, release, or tag workflow ran.
- Added failing user journeys before changing production behavior: removal
  before hydration followed by reload, a failed save requiring recovery without
  another edit, and clear/add while an older save is still in flight. These
  failures confirmed that current-visit reconciliation alone was insufficient.
- Added `citeShelfMutationJournal.ts` (121 lines): per-project retained additions,
  edits, removal identities, explicit clear, order, and open-state intent. It
  reuses the existing local/session/memory storage fallback; ordinary empty
  caches and malformed or wrong-project records never authorize a clear.
  Removal and clear record intent synchronously before a user can leave.
- Added `citeShelfSaveQueue.ts` (163 lines): one ordered write lane per project
  within a browser document, surviving component unmount/remount. Clear now
  uses an explicit empty PATCH in that lane rather than a separate racing DELETE.
  New eligible work is coalesced behind the active request. A retry fetches the
  latest backend basket before replay, retaining backend-only evidence.
- Both hydration and retained saves have bounded retries after 1, 2, 4, 8, and
  16 seconds. Exhaustion retains the intent without indefinite polling; network
  reconnection or window focus can resume. Re-entry replays the journal. Each
  queued job captures its own eligible intent rather than adopting mutations
  from a later, not-yet-hydrated project visit.
- HTTP success alone is not a save acknowledgement: returned content must
  contain the intended changes, and an old acknowledgement cannot discard a
  newer journal ID. The comparison follows backend whitespace compaction and
  omission of empty optional JSON fields without changing local note formatting.
  Backend truncation/cap behavior is not claimed to be completely mirrored.
- Testing exposed an intermediate React restoration render that could be
  mistaken for a fresh removal during StrictMode replay. A pending-restoration
  marker now prevents that stale state from being journaled. The existing
  request-sequence and project guards remain in place.
- `MessageList.tsx` shrank from 2,744 to 2,732 lines (-12), with its architecture
  budget ratcheted to 2,732; cumulative reduction since P2.2 is 275 lines.
  Local debounce/flush ownership, storage migration, metadata repair, and
  summary lifecycles remain separate. New modules add repository lines; this
  is a reduction of mixed ownership, not a claim of net source shrinkage.
- The maintained persistence smoke suite grew from 6 to 15 tests. Coverage
  includes failed initial GET, failed PATCH, pending deletion across reload,
  failed clear followed by closing and reopening a page, ordered clear/add,
  stale acknowledgements, backend-only evidence, re-add/order/blank annotations,
  wrong scope, false-positive HTTP success, storage normalization, and a virtual
  clock proving retry exhaustion and reconnection without real-time sleeps.
  The reopened-page helper binds the new page's GET rather than assuming an old
  fixed request index: beforeunload can legitimately initiate another request.
- Final validation: focused persistence 15/15; smoke 191 passed with 3 expected
  build-conditional skips (194 total); internal-debug core 129/129; generation
  lifecycle 27/27; public surface 4/4; hidden evidence workspace 1/1. Whole browser
  suites ran serially. Frontend lint, the 3,923-module build, Ruff for the changed
  budget checker, architecture budgets, and whitespace checks pass. Existing
  backend shelf store tests pass 11/11; backend product code did not change, so
  the prior full backend gate was not rerun in this slice. Browser cases use
  isolated API fixtures, not a packaged-release or real-user field acceptance.
- Initial core validation is not counted as acceptance: 128 passed and one
  forced-checkbox interaction failed. Trace evidence shows the click landed on
  the card rather than its checkbox as the layout changed. Five repeated runs
  produced one pass, one further forced-click failure, and three whole-journey
  timeouts. The test now uses normal actionability checks and explicit checked
  assertions; it does not force a click through layout changes. The next full
  core run again passed 128 tests but its multi-reload/follow-up journey exceeded
  the existing 30-second total timeout. No timeouts, worker limits, or assertions
  have been relaxed. The final complete core rerun passes 129/129, including
  that journey at 27.1 seconds. This establishes the final gate result, not a
  claim of ample timing headroom: fixture startup/reload profiling remains
  useful follow-up work, without inflating timeouts or dropping assertions.

#### Scope limits and next priority

This is a same-document queue plus a browser-storage journal, not a distributed
lock or server-side compare-and-swap protocol. Closing/reopening after a known
failed write is covered; an unknown in-flight server commit during process
termination, simultaneous writers in different windows, and the separate hidden
evidence-workspace write paths are not newly certified. Browser storage cleared
by the user, or complete storage failure, cannot provide restart durability.
These boundaries need automated investigation, not new unresolved user-facing
diagnostics or manual acceptance steps.

| Priority | Work | User value | Evidence/risk | Decision |
|---|---|---:|---|---|
| P2.3c-2b-2b | Exercise real FastAPI/SQLite storage with simultaneous windows and an unknown in-flight write across restart; assess revision/idempotency protection from reproduced failures | Very high data safety | The new lane orders one document only; backend writes have no compare-and-swap precondition, and hidden writers bypass the lane | Do next, before further mechanical extraction; use isolated data and automated tests |
| P2.3c-QA | Profile startup and repeated reloads in the long context-pack journey | High validation reliability | The final pass used 27.1 of the existing 30 seconds; removing forced clicks repairs actionability but does not establish timing margin under load | Include in the next test-maintenance slice, retaining interaction coverage and existing limits |
| P2.3c-2b-2c | Extract the remaining bounded local debounce/flush and storage-event ownership after conflict semantics are proven | High architectural | The queue and journal now have narrow boundaries; moving the rest before resolving writer ownership could preserve data races | Follow the concurrency findings; do not move repair/summary state wholesale |
| P1.3f-4g-4 | Evaluate a bounded real-user v2 observation window after a future release and explicit local collection choice | High, release-dependent | Fixture and packaged rehearsal data are not representative field evidence | Wait for legitimate released data |
| P1.4c | Prioritize concrete evidence-location failures from valid field outcomes | High, data-dependent | No representative opt-in field sample exists yet | Start when legitimate samples exist |
| P1.3f-5 | Reconsider a cross-encoder only for measured residual ranking failures | Medium, conditional | Persistence testing supplies no new ranking evidence or justification for an additional model lifecycle | Continue to defer |

### 2026-09-07 — P0 shelf multi-writer implementation after project review

Decision: implement the data-loss repair identified in
`PROJECT_PRIORITY_REVIEW_20260907.md` before additional architectural extraction.
`CURRENT_OPTIMIZATION_ROADMAP.md` is now the short, current priority queue; the
older tables above remain historical records, not competing next-step lists.

#### Implemented

- Shelf PATCH accepts a read revision. SQLite checks it inside the same
  `BEGIN IMMEDIATE` transaction as the replacement/operation replay. A stale
  writer receives HTTP 409, refreshes the shelf and retries retained intent.
- Operation receipts are durable and scoped by project/conversation. An already
  committed operation returns the current shelf without reapplying it, including
  when a response was lost and a later clear or re-add has already committed.
  A batch skips individually acknowledged operations, not just its final ID.
- Browser pending operations occupy separate storage entries so independent
  writes do not compete for one mutable journal value. Acknowledging an old
  batch removes only its captured operations, retaining later local work.
- Long offline journals drain in bounded batches; a 125-operation regression
  exercises the real API without raising the 120-operation/request limit.
- The first versioned write guards that scope against subsequent unversioned
  whole-snapshot replacement or DELETE. Untouched legacy scopes retain their
  existing compatibility behavior. Atomic append remains supported. Receipt
  rows are retained across shelf clears and removed when their scope is deleted.
- Hidden evidence-workspace save entrances use the operation/revision protocol
  as well. A guarded shelf's server revision overrides cache timestamps on
  re-entry; successful merges update the active UI without discarding pending
  local operations. Older acknowledgements cannot regress a newer known revision.
- `MessageList.tsx` is 2,726 lines versus the previous slice's 2,732; the 2,732
  ceiling was not raised. New logic lives in bounded shelf modules rather than
  expanding the protected component.

#### Verification and investigation record

- Existing shelf store/API tests: 26 passed. New concurrency unit/API tests:
  14 passed (8 unit, 6 API), including competing SQLite connections, stale
  snapshots, restart-persistent receipts, delete/re-add, clear/retry, scope
  isolation, legacy guard behavior and invalid operations.
- Existing persistence browser suite: 15 passed. Three real-API browser tests
  cover independent writers, a committed clear with a dropped response followed
  by a later append, and 125 offline operations. Three repetitions each pass:
  9/9. The shell/reference services are mocked; shelf requests go through the
  production FastAPI application and a disposable real SQLite database.
- Initial fixture attempts are not acceptance: the isolated test server first
  lacked required `KB_TEST_DATA_ROOT`/auxiliary paths, and a synthetic UI
  conversation needed binding to its real fixture conversation. Those setup
  defects were corrected; isolation enforcement stayed enabled.
- The first complete smoke attempt passed 192, skipped 3 and failed the new
  clear-loss test. Trace inspection proved it dropped a pre-clear normalization
  response, so its supposedly later append actually preceded the clear commit.
  Injection now matches an explicit clear operation. The competing-writer test
  uses independent caches, waits for both initial reads and the specific held
  mutations, and waits for the first pair of operation acknowledgements before
  injecting the next failure. These changes define the intended event sequence;
  no timeout, worker count or outcome assertion was relaxed.
- First full backend run: 5,583 passed, 43 skipped, before the last seven tests
  were added. The later run passed 5,589, skipped 43, and failed one converter
  subprocess startup assertion (`ready.wait(8.0)`) while heavy browser/build
  work ran alongside it. It is not recorded as a full pass. Final verification
  now serializes the heavy suites without changing the test's 8-second limit.
- Frontend lint/3,924-module production build, repository-wide Ruff,
  architecture budgets and `git diff --check` pass. Final complete browser and
  backend reruns passed; see the acceptance table below.
- The real shelf browser cases are included in the maintained smoke command.
  Frontend CI now sets up the existing backend Python dependency action for
  these cases; Windows release CI already provides it. Remote CI and packaged
  release acceptance have not been run or claimed in this slice.

#### Explicit semantics and remaining boundaries

Clear takes effect in successful transaction order; an addition committed after
that clear survives a duplicate clear retry. A new, genuinely later clear can
still clear current contents. Competing modifications to one item follow commit
order, not a CRDT/per-field conflict editor. Browser storage destruction or
complete storage unavailability cannot guarantee restart durability. Existing
shelf item/JSON limits remain; this slice does not certify arbitrary oversized
content, multi-device offline sync, packaged crash recovery or representative
field outcomes. Receipts must not be age-pruned without a protocol that keeps
old offline retries safe.

#### Updated next step

After this P0's final gates pass, do **P1-1 scoped dense source-boundary repair**:
carry the user-selected sources independently of BM25 hits, filter within the
allowed corpus before top K, and cover sparse-zero-hit and outside-global-top-40
examples. Preserve quality exclusions, explicit local opt-in, BM25 default and
the full-library no-go decision. Run paired retrieval and final-answer outcome
evaluation (P1-2) before considering another ranker. Citation direction/negation
failures (P1-3) follow measured evidence failures. Further component extraction,
cross-encoder productization, additional tracing infrastructure and expansion
of hidden workspaces do not displace these user-visible correctness tasks.

#### Final local acceptance — complete

The final fail-fast sequence ran the following whole suites serially, retaining
their configured internal workers, assertions and time limits:

| Gate | Final result |
|---|---|
| Browser smoke, including the 3 new real-API shelf cases | 194 passed, 3 existing build-conditional skips; 197 total |
| Internal-debug core citation/library/reader regressions | 129 passed |
| Generation lifecycle and supplementation-choice regressions | 27 passed |
| Ordinary-user surface isolation | 4 passed |
| Hidden evidence-workspace release isolation | 1 passed |
| Full backend unit + sanity suites, run after the browser sequence | 5,590 passed, 43 skipped; 350.02 seconds |
| New real-API browser cases repeated three times before final full sequence | 9 passed |
| Frontend lint and production build | Passed; 3,924 modules |
| Repository-wide Ruff, architecture budgets, whitespace | Passed |

The converter startup assertion that failed under concurrent heavy validation
passes in this final full backend run without code or timeout changes. The 17
Python warnings are dependency deprecations. This is local source-tree
acceptance, not a claim of remote CI, packaged release or field acceptance.
No release version/tag was changed, no commit/push was performed, and existing
user worktree changes were preserved. P0 is complete for the documented scope;
P1-1 is now the immediate next implementation task.
