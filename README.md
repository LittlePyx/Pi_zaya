# Pi_zaya

Pi_zaya is a local-first, evidence-grounded research agent for academic PDFs. It
helps users read papers, retrieve evidence, trace citations, compare papers, and
generate verifiable answers with grounded references.

This is a production-oriented AI agent / RAG engineering portfolio project, not
a toy PDF chatbot. It connects PDF conversion, structured indexing, hybrid
retrieval, agent planning, tool execution, claim verification, citation cards,
and a React trace UI into one end-to-end research workflow.

The product entry is FastAPI + React. The legacy Streamlit entry has been
removed; do not use `app.py`, `streamlit run`, or port `8501` as the product
entry.

## Problem

Academic PDF QA demos often fail in the places researchers care about most:

- They answer without showing which paper passage supports the claim.
- They blur paper-specific findings with general model knowledge.
- They do not expose retrieval scope, tool steps, or citation confidence.
- They treat PDF conversion, reference metadata, and answer grounding as
  separate demos instead of one workflow.

Pi_zaya is designed around the opposite constraint: if an answer claims
something about a paper, the user should be able to trace that claim back to
local evidence or see a clear disclosure that the answer used external model or
web context.

## Key Features

| Capability | What it provides |
|---|---|
| Anchored PDF conversion | Converts PDFs to Markdown while preserving page markers, sections, figures, formulas, and source anchors. |
| Evidence-based QA | Searches the indexed library, builds a RAG prompt from retrieved snippets, and returns grounded answers. |
| Citation tracing | Surfaces answer evidence, source cards, reference context, and reader locate targets. |
| Literature basket | Lets users collect papers and excerpts, keep local research context, and export citations. |
| Research Agent Mode | Adds explicit planning, source policy, evidence matrix, tool-use trace, and sentence-level citation support checks on top of the existing RAG flow. |
| Quality tooling | Scans conversion quality, runs repair flows, rebuilds indexes, and tracks metadata/reference sync. |

## Architecture

```mermaid
flowchart LR
  A["PDF Library"] --> B["PDF-to-Markdown Converter"]
  B --> C["Chunking + Structured Index"]
  C --> D["Hybrid Retrieval"]
  D --> E["Research Agent Runtime"]
  E --> P["Planner + Intent Router"]
  E --> R["Evidence Retriever"]
  E --> X["Reference Resolver"]
  E --> G["Reading Guide Tool"]
  E --> M["Paper Comparison Tool"]
  E --> Q["Research Run + Evidence Matrix"]
  E --> V["Claim Verifier"]
  P --> O["Grounded Answer + Citation Trace UI"]
  R --> O
  X --> O
  G --> O
  M --> O
  Q --> O
  V --> O
```

At a high level:

1. PDFs are converted into anchored Markdown with page markers and source
   locations.
2. Markdown is chunked and indexed for retrieval, references, figures, and
   reader navigation.
3. Retrieval returns candidate evidence under the current-paper, basket, or full
   library scope.
4. Research Agent Mode plans tool calls, checks evidence sufficiency, generates
   an answer, and verifies citation support.
5. The frontend keeps the answer clean while citations, reference cards, reader
   locate targets, and agent traces remain inspectable on demand.

Key backend entry points:

- `api/main.py`: FastAPI application
- `api/routers/generate.py`: streaming chat generation
- `api/routers/chat.py`: conversations, messages, uploads, and direct research-agent endpoint
- `api/routers/library.py`: library, conversion, quality, metadata, and indexing APIs
- `kb/task_runtime.py`: background generation/conversion runtime
- `kb/agent/`: lightweight Research Agent layer

Key frontend entry points:

- `web/src/main.tsx`: React entry
- `web/src/pages/ChatPage.tsx`: main chat workspace
- `web/src/pages/LibraryPage.tsx`: PDF/library workspace
- `web/src/components/chat/AgentTracePanel.tsx`: Research Agent trace UI
- `web/src/components/chat/CiteShelf.tsx`: literature basket UI
- `web/src/api/*.ts`: typed API contracts

## Research Agent Mode

Research Agent Mode is an incremental layer over the existing RAG system. It does
not replace retrieval, prompt building, citation cards, or the default chat flow.
When enabled, a response includes an `agent_trace` object with:

- `question_type`: `single_paper_qa`, `multi_paper_comparison`, `reading_guide`, `reference_followup`, or `unknown`
- `context.planner_intent`: typed planner output with task type, required tools,
  evidence need, target-paper hints, and planner confidence
- `context`: effective query scope, requested scope, current-paper lock, and selected-basket count when available
- `summary`: compact audit fields for claim support, scope, tool-call count, and error presence
- `research_run`: compact run metadata with source policy, subtask summary, and
  evidence-matrix rows for paper, method, result, limitation, quote, and support status
- `plan`: planned steps with goal, tool, and status
- `steps`: executed tool calls, observations, compact outputs, and errors
- `verification`: sentence-level citation/evidence support counts, `evidence_status`
  (`grounded`, `needs_review`, `insufficient`, or `not_applicable`), unsupported-claim reasons, and
  compact matched evidence sources

For reference-followup answers, resolved upstream references in the trace can be
opened in the reader or added to the literature basket from the chat UI.
For comparison and evidence-heavy answers, the trace can include a compact
evidence matrix so reviewers can scan which source supports which method,
result, or limitation without exposing raw tool logs in the main answer. The
same matrix is also attached to the pre-answer structured notes, so answer
generation can synthesize from the paper/method/result/limitation/evidence cells
instead of treating the matrix as UI-only audit metadata.
Historical agent traces can also be read through the compact audit endpoint
`GET /api/messages/{message_id}/agent-trace`. The React UI keeps this as a
collapsed, on-demand trace panel so ordinary answers are not crowded with
planning or tool-log details.
When opened, the panel shows the compact summary first; plan and tool-call logs
stay behind a second "Execution Details" disclosure.
When collapsed, the panel only shows compact evidence-check status and scope, not
raw question-type labels or tool execution logs.
Agent Mode also applies a lightweight evidence gate: answers with retrieved
local evidence keep local snippets as the authority, while the model may add
compact external academic background to improve framing. If OpenAI web search is
configured, that background can use API web search; otherwise it comes from the
normal text model. No-hit academic questions can also fall back to an external
model answer. External fallback and hybrid answers are visibly marked so users
can tell which claims are knowledge-base grounded and which parts are model/web
background; no-hit fallback uses `not_applicable` for local citation
verification.
The source blend is explicit in the trace: `local_grounded`,
`hybrid_local_external`, `external_academic`, or `general_llm`. General
non-library questions can use the normal text API without adding a knowledge-base
miss notice to the answer.
After generation, a lightweight answer-quality gate checks citation presence,
source disclosure, evidence overlap, and trace/debug leakage. If the answer
fails, the runtime makes one repair attempt; if it still cannot pass, it returns
a conservative evidence-only summary instead of exposing unsupported claims.
The main answer body is kept focused on the response and necessary citations in
rendered UI, API streaming, and stored chat messages; trace JSON, plan steps,
tool calls, and verification details remain behind the trace panel.

The planner uses simple heuristics first:

- comparison keywords -> `multi_paper_comparison`
- "how to read" / reading-guide language -> `reading_guide`
- reference, citation, upstream, prior-work language -> `reference_followup`
- otherwise -> `single_paper_qa`

The runtime then applies an evidence sufficiency gate. Low-confidence retrieval
hits are preserved in the trace for diagnosis, but only usable hits are treated
as local evidence. If the local library is insufficient, the answer is qualified
or routed to an external academic fallback when configured; it is not presented
as a knowledge-base-grounded result.

The tool layer wraps existing modules instead of adding external services:

- `retrieve_evidence`
- `retrieve_references`, which reads the local reference index when available and returns compact upstream-reference fields such as `ref_num`, `title`, `authors`, `year`, `doi`, `source_paper`, and `why_relevant`
- `build_reading_guide`
- `compare_papers`, which returns source-specific `paper`, `method`, `evidence`, `limitation`, and `relation_to_question` fields
- `generate_grounded_answer`
- `verify_answer_citations`

If no text LLM API key is configured, the agent still runs in degraded mode and
returns retrieved evidence notes plus a trace instead of crashing the app.

### Enable Agent Mode

In the React chat UI, toggle the `Normal` / `Agent` button in the composer before
sending a question. The setting is persisted per conversation and only affects
newly sent turns, so ordinary chat stays unchanged unless the user explicitly
enables Agent Mode. For explicit test or deep-link entry, open the app with
`/?agent_mode=1`.

API options:

```http
POST /api/generate
Content-Type: application/json

{
  "conv_id": "conversation-id",
  "prompt": "Compare these papers",
  "agent_mode": true,
  "query_scope": "basket",
  "prompt_context": {
    "items": [
      {"title": "Paper A", "sourcePath": "paper-a.md"}
    ]
  }
}
```

Direct non-conversation endpoint:

```http
POST /api/chat/research-agent
Content-Type: application/json

{
  "query": "How should I read this paper?",
  "top_k": 6,
  "query_scope": "current_paper",
  "source_lock_path": "converted-paper.md"
}
```

Supported `query_scope` values are `current_paper`, `basket`, and `library`.
Default chat behavior is unchanged when `agent_mode` is omitted or false.

## Quick Start

Requirements:

- Python `3.10.11` as declared in `.python-version`
- Node.js `24.13.0` as declared in `.nvmrc`
- At least one text LLM key for full answers: `QWEN_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENAI_API_KEY`

Install:

```powershell
Copy-Item .env.example .env

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd web
npm ci
cd ..
```

Run the development UI and API:

```powershell
.\run_new.ps1 -StopExisting
```

Equivalent wrapper:

```powershell
.\run.ps1 -StopExisting
```

Primary local URLs:

- React app: `http://127.0.0.1:5173/`
- FastAPI backend: `http://127.0.0.1:8000/`

Run backend only:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Run frontend only:

```powershell
cd web
npm run dev -- --host 127.0.0.1 --port 5173
```

Build frontend:

```powershell
cd web
npm run build
```

Single-service local production mode:

```powershell
cd web
npm run build
cd ..
python server.py
```

## Production Deployment

For public deployments, keep token gates disabled by default:

Set `KB_PRIVATE_INSTANCE_AUTH=0`, `KB_ENABLE_AUTH_GATE=0`, and
`KB_REQUIRE_AUTH=0` so ordinary users can open the app without an access token.

Chinese deployment note: 面向普通用户的公开部署保持 `KB_PRIVATE_INSTANCE_AUTH=0`、`KB_ENABLE_AUTH_GATE=0` 和 `KB_REQUIRE_AUTH=0`，用户打开应用不需要访问令牌。

Use private/internal auth only for controlled instances, and configure
`KB_ACCESS_TOKEN` or `KB_ACCESS_TOKEN_SHA256` when auth is enabled.

Run the production readiness check after startup:

```powershell
python tools\check_production_readiness.py --base-url http://127.0.0.1:8000
```

## Configuration

Common environment variables:

- `QWEN_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENAI_API_KEY`: text model access
- `QWEN_BASE_URL`, `DEEPSEEK_BASE_URL`, `OPENAI_BASE_URL`: optional provider base URLs
- `QWEN_MODEL`, `DEEPSEEK_MODEL`, `OPENAI_MODEL`: optional model names
- `KB_AGENT_WEB_SEARCH_ENABLED`: enable or disable no-hit academic web fallback
- `KB_AGENT_WEB_SEARCH_API_KEY`: optional OpenAI-compatible web-search key; falls back to `OPENAI_API_KEY`
- `KB_AGENT_WEB_SEARCH_MODEL`: web-search model, default `gpt-5-search-api`
- `KB_PDF_DIR`: source PDF directory
- `KB_MD_DIR`: converted Markdown directory
- `KB_DB_DIR`: retrieval/index directory
- `KB_CHAT_DB`: chat SQLite path
- `KB_LIBRARY_DB`: library SQLite path
- `KB_CROSSREF_BUDGET_S`: Crossref sync time budget

For local development, copy the development environment template:

```powershell
Copy-Item .env.example .env
```

Copy the production environment template only for production-style deployment:

```powershell
Copy-Item .env.production.example .env
```

The settings UI can also store local API preferences in `user_prefs.json`.
Environment variables and `.env` values take precedence.

## Typical Workflow

1. Open the Library page.
2. Upload or select PDFs.
3. Convert PDFs to Markdown.
4. Review conversion quality and run repair/reconversion when needed.
5. Rebuild the knowledge base indexes.
6. Ask questions from the current paper, selected literature basket, or full library.
7. Open answer evidence/citation cards in the Reader.
8. Add important papers or excerpts to the literature basket and export citations.

## Demo

A polished demo GIF/video is not checked into the repository yet. Suggested demo
path for portfolio use:

1. Upload one or two academic PDFs.
2. Convert and index the papers.
3. Ask a paper-specific question in Agent Mode.
4. Open the citation card and reader locate target.
5. Ask a comparison or reading-guide question.
6. Open the Research Agent Trace panel to show planner intent, tool calls,
   evidence matrix, evidence status, and claim verification.

## Evaluation

See [docs/EVAL_DASHBOARD.md](docs/EVAL_DASHBOARD.md) for metric categories,
manual/semi-automated evaluation tables, commands, current limitations, and
future work. The document intentionally does not include fabricated numbers.

Core evaluation dimensions:

- Retrieval Recall@k and retrieval relevance
- Citation precision and evidence locate success
- Claim support rate and unsupported claim rate
- No-evidence refusal accuracy
- Agent trace completeness, evidence-matrix coverage, and planner accuracy
- P50/P95 latency and cost per query when instrumentation is available

Metrics remain `TBD` until produced by a reproducible run or documented manual
review. The lightweight agent trace eval can write a JSON report with measured
fixture-regression fields and `null` placeholders for unmeasured live metrics
such as latency and cost.

The recorded fixture `docs/research_agent_eval_v1.jsonl` checks a small set of
answer-quality guardrails: local-evidence support, hybrid/external source
disclosure, expected answer-point coverage, and keeping trace/tool/debug content
out of the main answer. Treat these as reproducible regression checks, not as a
claimed live benchmark.

Useful commands:

```powershell
python -m pytest tests/unit -q
python tools\research_qa\validate_research_agent_golden.py
python tools\research_qa\run_agent_trace_eval.py --json-out test_results\agent_trace_eval.json
python tools\research_qa\export_research_agent_samples.py --db chat.sqlite3 --out test_results\research_agent_answer_samples.jsonl --limit 50
python tools\research_qa\review_research_agent_samples.py prepare --samples test_results\research_agent_answer_samples.jsonl --labels test_results\research_agent_answer_labels.jsonl
python tools\research_qa\review_research_agent_samples.py merge --samples test_results\research_agent_answer_samples.jsonl --labels test_results\research_agent_answer_labels.jsonl --out test_results\research_agent_answer_reviewed.jsonl
python tools\research_qa\run_agent_trace_eval.py --real-samples test_results\research_agent_answer_samples.jsonl --json-out test_results\agent_trace_real_replay_eval.json
python tools\research_qa\run_reviewed_replay_eval.py
python tools\research_qa\run_research_qa_eval.py --dry-run
python tools\converter_quality\run_converter_quality_eval.py --dry-run

cd web
npm run lint
npm run build
npm run test:e2e:smoke
```

## Data Files

Local runtime data is not intended for Git commits:

| Path | Purpose |
|---|---|
| `chat.sqlite3` | Conversations, messages, message refs, and chat metadata |
| `library.sqlite3` | Library metadata and source tracking |
| `db/` | Docs, chunks, reference index, and Crossref cache |
| `backups/` | Manual and automatic backups |

## Development Notes

- Preserve Markdown page markers like `<!-- kb_page: N -->`.
- Keep React API contracts typed in `web/src/api`.
- Prefer fixing quality issues in converter/retrieval/data flow code instead of only hiding UI states.
- Keep background and shared state thread-safe.
- Before high-risk operations, create or verify a backup.

## Roadmap

- Add a small labeled QA benchmark with expected evidence and manual rubric notes.
- Track retrieval/citation/claim metrics over time in JSON reports.
- Improve intent routing beyond keyword heuristics while keeping degraded mode.
- Add optional answer repair passes for unsupported claims.
- Add a demo video and curated portfolio walkthrough.
- Explore a future MCP-compatible local tool server for paper search, chunk
  reading, reference resolution, and citation-grounded exports.

## Portfolio / Interview Talking Points

- Local-first architecture: papers, indexes, references, and chat state are
  stored locally instead of requiring a hosted document service.
- Evidence grounding: answers are designed around citations, reader locate
  targets, and claim-level support checks.
- Agent runtime: explicit planner intent, tool calls, evidence sufficiency, and
  trace UI sit on top of the RAG pipeline.
- Production concerns: FastAPI + React architecture, typed frontend API
  contracts, background queues, SQLite stores, CI, unit/sanity/e2e tests, and
  degraded-mode behavior when LLM keys are missing.
- Evaluation honesty: docs and scripts expose evaluation dimensions without
  inventing benchmark numbers.

## Suggested GitHub Metadata

Repository description:

> Local-first evidence-grounded research agent for academic PDFs with RAG,
> citation tracing, agent planning, and verifiable answers.

Suggested topics:

`ai-agent`, `rag`, `llm`, `research-agent`, `fastapi`, `react`, `typescript`,
`pdf-processing`, `citation-tracing`, `agent-observability`, `llm-evaluation`
