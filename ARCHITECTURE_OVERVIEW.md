# kb_chat Architecture Overview

`kb_chat` is a local academic-paper knowledge base built on FastAPI + React.

The legacy Streamlit entry has been removed. Product work should use the React app on port `5173` and the FastAPI backend on port `8000`.

## Runtime Entries

- Backend API: `api/main.py`
- Local production/static wrapper: `server.py`
- React app: `web/src/main.tsx`
- Development launcher: `run_new.ps1`
- Convenience launcher: `run.ps1`

## Main Data Flow

1. PDF upload and library actions enter through `api/routers/library.py`.
2. PDF conversion is orchestrated by `kb/converter/pipeline.py`.
3. Markdown is chunked and indexed by `kb/chunking.py`, `kb/store.py`, and structured-index helpers.
4. Retrieval uses `kb/retriever.py` and `kb/retrieval_engine.py`.
5. RAG messages are assembled in `kb/rag.py`.
6. LLM calls are handled by `kb/llm.py`.
7. Citation/reference payloads are shaped by backend reference modules and rendered by React components.

## Backend Layers

- `api/routers/`: FastAPI route modules for chat, library, references, generation, and settings.
- `api/reference_ui.py`: citation card and reference payload enrichment for the React UI.
- `api/reference_card_quality.py`: citation-card quality checks and diagnostics.
- `kb/converter/`: PDF-to-Markdown conversion, quality analysis, repair, and source diagnostics.
- `kb/task_runtime.py` and `kb/bg_queue_state.py`: background task queue and shared runtime state.
- `kb/chat_store.py`: conversation and message persistence.
- `kb/library_store.py`: library metadata persistence.
- `kb/reference_index.py` and `kb/reference_sync.py`: reference extraction and Crossref metadata sync.

## Frontend Layers

- `web/src/pages/ChatPage.tsx`: chat page.
- `web/src/pages/LibraryPage.tsx`: library management, conversion, metadata, quality center, and repair workflows.
- `web/src/components/chat/CiteShelf.tsx`: literature basket / citation shelf.
- `web/src/api/`: typed API clients.
- `web/src/stores/`: Zustand stores.
- `web/src/styles/index.css`: shared styling.

## Quality Center Direction

Quality diagnostics should not stop at UI labels. When bad conversion, source alignment, citation cards, or literature-basket metadata are detected, the preferred fix path is:

1. Preserve or recover the missing signal in converter output.
2. Record source-quality diagnostics in conversion quality sidecars.
3. Expose actionable summaries through FastAPI.
4. Render clear React states and actions.
5. Rebuild indices after safe Markdown repairs.
6. Add focused tests that prevent the same issue from returning.

Current quality-center integration points:

- `kb/converter/quality_center.py`
- `kb/converter/quality_repair.py`
- `api/routers/library.py`
- `web/src/api/library.ts`
- `web/src/pages/LibraryPage.tsx`
- `web/src/components/chat/CiteShelf.tsx`

## Local Commands

```powershell
.\run_new.ps1 -StopExisting
.\run.ps1 -StopExisting
```

```bash
python -m pytest tests/unit/test_converter_quality_center.py tests/unit/test_converter_quality_repair.py tests/unit/test_converter_quality_gate.py tests/sanity/test_library_phase1_api.py -q
cd web && npm run build
```
