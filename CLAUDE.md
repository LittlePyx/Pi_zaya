# CLAUDE.md

This file guides coding assistants working in this repository.

Pi_zaya is now a FastAPI + React academic PDF knowledge base. The legacy Streamlit entry has been removed; do not launch or modify `app.py`.

## Run

```powershell
.\run_new.ps1 -StopExisting
```

Equivalent wrapper:

```powershell
.\run.ps1 -StopExisting
```

Manual development servers:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
cd web && npm run dev -- --host 127.0.0.1 --port 5173
```

## Entry Points

- Backend API: `api/main.py`
- Local static server: `server.py`
- Frontend app: `web/src/main.tsx`
- Library page: `web/src/pages/LibraryPage.tsx`
- Chat page: `web/src/pages/ChatPage.tsx`

## Quality Direction

Conversion quality issues should be fixed at the source when possible:

- converter pipeline: `kb/converter/pipeline.py`
- quality analysis and safe repair: `kb/converter/quality_repair.py`
- quality center summaries and batch scan/repair: `kb/converter/quality_center.py`
- library quality endpoints: `api/routers/library.py`
- React quality UI: `web/src/pages/LibraryPage.tsx`

Frontend API contracts live in `web/src/api`.
