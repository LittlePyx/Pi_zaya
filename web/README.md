# Pi_zaya Web

This is the React + TypeScript frontend for Pi_zaya, a local-first,
evidence-grounded research agent for academic PDFs.

## Main Screens

- Chat workspace: RAG answers, Research Agent Mode toggle, evidence/citation surfaces, and literature basket.
- Library workspace: PDF upload, conversion, quality checks, metadata, and indexing.
- Reader surfaces: source locate targets, evidence anchors, and citation-card context.

## Development

Install dependencies from the repository root:

```powershell
cd web
npm ci
```

Run the Vite dev server:

```powershell
npm run dev -- --host 127.0.0.1 --port 5173
```

Build:

```powershell
npm run build
```

Lint:

```powershell
npm run lint
```

Smoke tests:

```powershell
npm run test:e2e:smoke
```

The app expects the FastAPI backend at `http://127.0.0.1:8000/` during local
development. From the repository root, `.\run_new.ps1 -StopExisting` starts both
the backend and frontend with the expected ports.

## Research Agent Mode

The composer has a compact `Normal` / `Agent` toggle. When enabled for a turn,
the frontend sends `agent_mode: true` to `/api/generate` and renders the returned
`agent_trace` inside a collapsible Research Agent Trace panel.

The toggle is remembered per conversation, and the default chat flow is
unchanged when it is off. For explicit test or deep-link entry, use
`/?agent_mode=1`.
