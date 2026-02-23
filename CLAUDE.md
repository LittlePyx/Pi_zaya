# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pi_zaya is a Streamlit-based academic PDF knowledge base with RAG-powered Q&A and traceable citations. PDFs are converted to Markdown, chunked, indexed with BM25, and queried via LLM with citation tracking.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
# Access at http://127.0.0.1:8501

# CLI ingest markdown into knowledge base
python ingest.py <md_dir> <db_dir>
```

No formal test suite or linter is configured. Existing test files (`test_simple.py`, `test2.py`) are ad-hoc.

## Architecture

**Data flow:** PDF Upload → `kb/converter/pipeline.py` → Markdown → `kb/chunking.py` → BM25 Index (`kb/store.py`) → Retrieval (`kb/retriever.py`) → RAG prompt (`kb/rag.py`) → LLM (`kb/llm.py`) → Citation rendering (`ui/refs_renderer.py`)

**Entry point:** `app.py` (~2200 lines) — two Streamlit pages: Chat and Library Management.

### Key modules in `kb/`

| Module | Role |
|---|---|
| `config.py` | `Settings` frozen dataclass, loads env vars with Qwen→DeepSeek→OpenAI fallback |
| `llm.py` | `DeepSeekChat` — OpenAI-compatible client with retry + streaming |
| `retriever.py` | `BM25Retriever` — BM25Okapi search with CJK-aware tokenization |
| `retrieval_engine.py` | Query translation (CJK→EN), heuristic filtering, result caching |
| `rag.py` | `build_messages()` — formats retrieval hits + history into LLM prompt |
| `chat_store.py` | SQLite (WAL mode) for conversations, messages, and retrieval refs |
| `chunking.py` | Splits markdown into overlapping chunks preserving heading context |
| `store.py` | JSON/JSONL persistence for document index and chunks |
| `reference_index.py` | Extracts references from markdown, builds index, Crossref enrichment |
| `reference_sync.py` | Daemon thread for non-blocking Crossref metadata sync |
| `citation_meta.py` | DOI extraction, title similarity matching, Crossref API calls |
| `task_runtime.py` + `bg_queue_state.py` | Thread-safe background queue for PDF conversion |
| `converter/pipeline.py` | PDF→MD orchestrator (normal/ultra_fast/no_llm modes) |
| `converter/llm_worker.py` | Multi-modal LLM calls (Qwen VL) for page screenshots |

### UI layer (`ui/`)

- `chat_widgets.py` — chat rendering, math normalization
- `refs_renderer.py` — citation popups, DOI links, citation shelf
- `runtime_patches.py` — CSS/JS injection for Streamlit customization
- `strings.py` — all UI string constants (Chinese)

## Configuration

All via environment variables (no config files). At minimum set one API key:

- `QWEN_API_KEY` (primary), `DEEPSEEK_API_KEY` (fallback), or `OPENAI_API_KEY`
- Path overrides: `KB_PDF_DIR`, `KB_MD_DIR`, `KB_DB_DIR`, `KB_CHAT_DB`, `KB_LIBRARY_DB`
- `KB_CROSSREF_BUDGET_S` — time budget for reference sync (default 45s)

## Conventions

- Python 3.12, `from __future__ import annotations` everywhere
- Type hints use `str | None` style (not `Optional`)
- Internal functions prefixed with `_`
- Thread-safe state via `threading.Lock` for all background operations
- Defensive error handling: broad `except Exception` with fallback values
- Markdown page markers: `<!-- kb_page: N -->` for page tracking through the pipeline
- `kb/runtime_state.py` holds shared singleton state across Streamlit reruns

## Database Files

- `chat.sqlite3` — conversations, messages, message_refs tables
- `library.sqlite3` — reference library metadata
- `db/` directory — `docs.json`, `chunks/*.jsonl`, `references_index.json`, `crossref_cache.json`
