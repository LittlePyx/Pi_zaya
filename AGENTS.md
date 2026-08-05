# AGENTS.md

This file guides Codex when working in this repository.

## Project Overview

Pi_zaya is a FastAPI + React academic PDF knowledge base with RAG-powered Q&A, traceable citations, a literature basket, and conversion quality tooling. PDFs are converted to Markdown, chunked, indexed with BM25 plus structured indices, and queried through the backend API.

The legacy Streamlit entry has been removed. Do not use `app.py`, `streamlit run`, or port `8501` as the product entry.

## Commands

```powershell
# Development UI and API
.\run_new.ps1 -StopExisting
# Equivalent convenience wrapper
.\run.ps1 -StopExisting
```

```bash
# Backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Frontend
cd web
npm run dev -- --host 127.0.0.1 --port 5173

# Frontend build
cd web
npm run build

# CLI ingest markdown into knowledge base
python ingest.py <md_dir> <db_dir>
```

Primary local URLs:

- React app: `http://127.0.0.1:5173/`
- FastAPI backend: `http://127.0.0.1:8000/`

## Architecture

**Data flow:** PDF upload -> `api/routers/library.py` -> `kb/converter/pipeline.py` -> Markdown -> `kb/chunking.py` / structured indices -> `kb/store.py` -> retrieval (`kb/retriever.py`, `kb/retrieval_engine.py`) -> RAG prompt (`kb/rag.py`) -> LLM (`kb/llm.py`) -> React citation and literature-basket surfaces.

**Backend entry:** `api/main.py`, with local production/static serving via `server.py`.

**Frontend entry:** `web/src/main.tsx`, routed React application under `web/src/pages` and `web/src/components`.

### Key Backend Modules

| Module | Role |
|---|---|
| `api/routers/library.py` | Library, conversion, quality, repair, metadata, and indexing endpoints |
| `api/routers/chat.py` | Chat API and streaming integration |
| `api/routers/evidence_matrices.py` | Project evidence-matrix generation, versioning, cell audit, and export |
| `api/routers/research_briefs.py` | Project research-brief generation, versioning, evidence audit, and export |
| `api/reference_ui.py` | Backend citation/reference-card payload shaping for React |
| `kb/converter/pipeline.py` | PDF to Markdown orchestrator |
| `kb/converter/quality_center.py` | Conversion source-quality scan/repair summaries |
| `kb/converter/quality_repair.py` | Markdown quality analysis, repair planning, and safe repair |
| `kb/task_runtime.py` + `kb/bg_queue_state.py` | Thread-safe background queue for PDF conversion and chat tasks |
| `kb/retrieval_engine.py` | Query translation, heuristic filtering, result caching |
| `kb/reference_index.py` | Reference extraction and Crossref enrichment |
| `kb/reference_sync.py` | Non-blocking reference metadata sync |
| `kb/chat_store.py` | SQLite conversations, messages, and retrieval refs |
| `kb/evidence_matrix.py` | Source-balanced evidence-cell extraction, comparison boundaries, audit, and exporters |
| `kb/research_brief.py` | Research-brief source selection, evidence quality contract, bibliography, and exporters |
| `kb/library_store.py` | Library metadata and source tracking |

### Key Frontend Modules

| Module | Role |
|---|---|
| `web/src/pages/ChatPage.tsx` | Main chat experience |
| `web/src/pages/LibraryPage.tsx` | Library management, conversion, quality center, metadata |
| `web/src/components/chat/CiteShelf.tsx` | Literature basket / citation shelf UI |
| `web/src/components/chat/EvidenceMatrixWorkspace.tsx` | Project evidence-matrix editor, audit, revisions, and export UI |
| `web/src/components/chat/ResearchBriefWorkspace.tsx` | Project research-brief editor, audit, revisions, and export UI |
| `web/src/api/*.ts` | API clients and response types |
| `web/src/stores/*.ts` | Zustand stores for chat, library, and settings |
| `web/src/styles/index.css` | Application styling |

## Configuration

All via environment variables. At minimum set one API key:

- `QWEN_API_KEY` (primary), `DEEPSEEK_API_KEY` (fallback), or `OPENAI_API_KEY`
- Path overrides: `KB_PDF_DIR`, `KB_MD_DIR`, `KB_DB_DIR`, `KB_CHAT_DB`, `KB_LIBRARY_DB`
- `KB_CROSSREF_BUDGET_S` controls Crossref sync time budget

## Conventions

- Python uses `from __future__ import annotations` and `str | None` style type hints.
- Internal Python helpers generally use `_` prefixes.
- Background and shared state must remain thread-safe.
- Markdown page markers use `<!-- kb_page: N -->` and must be preserved through conversion repairs.
- React API contracts should be typed in `web/src/api`.
- Quality fixes should flow from source diagnostics to converter/repair code, not only hide bad UI states.

## Database Files

- `chat.sqlite3` - conversations, messages, message refs
- `library.sqlite3` - library metadata
- `db/` - `docs.json`, `chunks/*.jsonl`, `references_index.json`, `crossref_cache.json`
