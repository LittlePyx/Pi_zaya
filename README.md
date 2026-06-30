# Pi_zaya

Pi_zaya is a local-first, evidence-grounded research agent for academic PDFs.
It converts papers into anchored Markdown, builds structured retrieval indexes,
answers questions with traceable evidence, and helps manage a working literature
basket for reading, synthesis, and citation workflows.

The product entry is FastAPI + React. The legacy Streamlit entry has been
removed; do not use `app.py`, `streamlit run`, or port `8501` as the product
entry.

## What It Does

| Capability | What it provides |
|---|---|
| Anchored PDF conversion | Converts PDFs to Markdown while preserving page markers, sections, figures, formulas, and source anchors. |
| Evidence-based QA | Searches the indexed library, builds a RAG prompt from retrieved snippets, and returns grounded answers. |
| Citation tracing | Surfaces answer evidence, source cards, reference context, and reader locate targets. |
| Literature basket | Lets users collect papers and excerpts, keep local research context, and export citations. |
| Research Agent Mode | Adds explicit planning, tool-use trace, and sentence-level citation support checks on top of the existing RAG flow. |
| Quality tooling | Scans conversion quality, runs repair flows, rebuilds indexes, and tracks metadata/reference sync. |

## Architecture

```text
PDF upload
  -> api/routers/library.py
  -> kb/converter/pipeline.py
  -> anchored Markdown
  -> kb/chunking.py + structured indexes
  -> kb/store.py
  -> kb/retrieval_engine.py / kb/retriever.py
  -> kb/rag.py + kb/llm.py
  -> FastAPI chat/generate endpoints
  -> React chat, reader, citation cards, literature basket

Research Agent Mode
  -> kb/agent/planner.py
  -> kb/agent/tools.py wrappers around retrieval/RAG/reference logic
  -> kb/agent/verifier.py
  -> agent_trace returned with the assistant response
```

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
- `context`: effective query scope, requested scope, current-paper lock, and selected-basket count when available
- `summary`: compact audit fields for claim support, scope, tool-call count, and error presence
- `plan`: planned steps with goal, tool, and status
- `steps`: executed tool calls, observations, compact outputs, and errors
- `verification`: sentence-level citation/evidence support counts, unsupported-claim reasons, and compact matched evidence sources

For reference-followup answers, resolved upstream references in the trace can be
opened in the reader or added to the literature basket from the chat UI.
Historical agent traces can also be read through the compact audit endpoint
`GET /api/messages/{message_id}/agent-trace`. The React UI keeps this as a
collapsed, on-demand trace panel so ordinary answers are not crowded with
planning or tool-log details.
When opened, the panel shows the compact summary first; plan and tool-call logs
stay behind a second "Execution Details" disclosure.
The main answer body is kept focused on the response and necessary citations in
rendered UI, API streaming, and stored chat messages; trace JSON, plan steps,
tool calls, and verification details remain behind the trace panel.

The planner uses simple heuristics first:

- comparison keywords -> `multi_paper_comparison`
- "how to read" / reading-guide language -> `reading_guide`
- reference, citation, upstream, prior-work language -> `reference_followup`
- otherwise -> `single_paper_qa`

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

In the React chat UI, toggle the `Agent` button in the composer before sending a
question. The setting is persisted locally and only affects newly sent turns.

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

面向普通用户的公开部署保持 `KB_PRIVATE_INSTANCE_AUTH=0`、`KB_ENABLE_AUTH_GATE=0` 和 `KB_REQUIRE_AUTH=0`，用户打开应用不需要访问令牌。

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
- `KB_PDF_DIR`: source PDF directory
- `KB_MD_DIR`: converted Markdown directory
- `KB_DB_DIR`: retrieval/index directory
- `KB_CHAT_DB`: chat SQLite path
- `KB_LIBRARY_DB`: library SQLite path
- `KB_CROSSREF_BUDGET_S`: Crossref sync time budget

Copy the production environment template when needed:

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

## Evaluation

See [docs/EVAL_DASHBOARD.md](docs/EVAL_DASHBOARD.md) for metric categories,
manual/semi-automated evaluation tables, commands, current limitations, and
future work. The document intentionally does not include fabricated numbers.

Useful commands:

```powershell
python -m pytest tests/unit -q
python tools\research_qa\validate_research_agent_golden.py
python tools\research_qa\run_agent_trace_eval.py
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
