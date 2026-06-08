# Pi-zaya Production Deployment Runbook

This runbook is for the FastAPI + React product path only. Do not use `app.py`, `streamlit run`, or port `8501` for production checks.

## 1. Preflight

Run these checks before a release:

```powershell
git status --short
python --version
node --version
```

Recommended version anchors are `.python-version` and `.nvmrc`.

Install dependencies:

```powershell
pip install -r requirements.txt
cd web
npm ci
npm run build
cd ..
```

`web/dist/index.html` must exist before using the single-service production entry.

## 2. Configure `.env`

Copy the template and edit secrets/paths:

```powershell
Copy-Item .env.production.example .env
notepad .env
```

Required production settings:

- `KB_ENV=production`
- `KB_ACCESS_TOKEN` or `KB_ACCESS_TOKEN_SHA256`
- `KB_REQUIRE_AUTH=1`
- `KB_API_ALLOW_ORIGINS` with explicit origins, not `*`
- `KB_STARTUP_PREFLIGHT=1`
- `KB_DB_DIR`, `KB_CHAT_DB`, `KB_LIBRARY_DB`
- `KB_BACKUP_DIR`, `KB_DIAGNOSTICS_DIR`
- `KB_AUTO_BACKUP=1`
- At least one text model key: `DEEPSEEK_API_KEY`, `QWEN_API_KEY`, or `OPENAI_API_KEY`
- A dedicated vision key, usually `QWEN_API_KEY`, for image/table/figure-heavy workflows

For local HTTP testing, keep `KB_AUTH_COOKIE_SECURE=0`. Behind HTTPS, set `KB_AUTH_COOKIE_SECURE=1`.

Set `KB_STARTUP_STRICT=1` in deployment scripts when startup should stop immediately on blocking readiness errors.

## 3. Start

Single-service local production mode:

```powershell
python server.py
```

In production mode, `server.py` prints a startup preflight summary. It does not stop the server unless `KB_STARTUP_STRICT=1`, so first-run operators can still open the UI and fix API keys or paths.

Or run the ASGI app directly:

```powershell
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/`, enter the access token, then open Settings -> Connection -> Release readiness.

## 4. Readiness Checks

Public liveness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Protected production readiness:

```powershell
python tools\check_production_readiness.py --base-url http://127.0.0.1:8000 --token $env:KB_ACCESS_TOKEN
```

Use JSON output when wiring this into deployment scripts:

```powershell
python tools\check_production_readiness.py --json --base-url http://127.0.0.1:8000 --token $env:KB_ACCESS_TOKEN
```

Exit codes:

- `0`: ready
- `1`: warning, unless `--allow-warning` is set
- `2`: blocking error or readiness request failed

## 5. Smoke Test

Before announcing a release:

1. Open the app and confirm the login gate appears when auth is enabled.
2. Log in with `KB_ACCESS_TOKEN`.
3. Open Settings -> Connection and confirm release readiness is `OK`.
4. Upload or select a small known PDF.
5. Run conversion and update the knowledge base.
6. Ask one short question and verify `src` citation chips persist after refresh.
7. Open Reader from the answer and verify references, figures, tables, and basket actions still work.

## 6. Backup and Restore

In the UI, open Settings -> Maintenance to create a manual backup or export a diagnostic package.

Manual backup API:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/maintenance/backups `
  -Headers @{ "X-KB-Access-Token" = $env:KB_ACCESS_TOKEN } `
  -Body '{"label":"before-upgrade"}' `
  -ContentType 'application/json'
```

Diagnostic package:

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:8000/api/maintenance/diagnostics/export `
  -Headers @{ "X-KB-Access-Token" = $env:KB_ACCESS_TOKEN } `
  -OutFile diagnostics.zip
```

Backups are written to `KB_BACKUP_DIR`. Diagnostics are written to `KB_DIAGNOSTICS_DIR`.
The diagnostic package is designed for support: it includes readiness, counts, environment summary and redacted log tails, but not chat rows, chunks, PDFs, or API keys.

Automatic snapshots:

- In production, high-risk operations create a backup before running by default.
- Covered operations include library file deletion, replace conversion, quality repair, figure-asset refresh, manual or repair-flow reindex, batch metadata updates, and chat/project/message/citation-shelf deletion.
- If `KB_AUTO_BACKUP` is not explicitly set, the local React Settings -> Advanced switch can control automatic snapshots.
- If `KB_AUTO_BACKUP` is set, deployment configuration is authoritative and the UI switch is locked.
- `KB_AUTO_BACKUP=0` disables automatic snapshots.
- `KB_AUTO_BACKUP_MIN_INTERVAL_S=30` rate-limits repeated snapshots for the same operation type.
- `KB_AUTO_BACKUP_STRICT=1` blocks the operation when snapshot creation fails.
- `KB_BACKUP_KEEP_N=30` is the default retention count used by backup cleanup.

Use the maintenance API to verify a backup archive before relying on it, and to clean up old backup archives after confirming the newest backups verify successfully.
Before restoring data, run the restore dry-run through the maintenance API. It extracts the backup to a temporary directory, checks SQLite integrity, lists the configured restore targets, and reports blocking errors without overwriting current files.
The actual restore action requires typing `RESTORE <backup-file-name>`. The server creates a pre-restore backup first, writes an audit record to `KB_RESTORE_AUDIT_PATH` or `.runtime/restore_audit.jsonl`, restores the selected data files, and then requires a server restart before continuing.

Back up these paths before upgrades:

- `.env`
- `chat.sqlite3` or the configured `KB_CHAT_DB`
- `library.sqlite3` or the configured `KB_LIBRARY_DB`
- `db/` or the configured `KB_DB_DIR`
- `KB_BACKUP_DIR`
- `user_prefs.json` when users configured model/path settings in the UI

Restore by stopping the server, copying the saved files back, then running the readiness check again.

## 7. Upgrade Flow

```powershell
git pull
pip install -r requirements.txt
cd web
npm ci
npm run build
cd ..
python -m pytest tests\unit\test_check_production_readiness.py tests\sanity\test_api_auth_guard.py -q
python server.py
```

After restart, run:

```powershell
python tools\check_production_readiness.py --base-url http://127.0.0.1:8000 --token $env:KB_ACCESS_TOKEN
```

## 8. Troubleshooting

`401 Unauthorized` on `/api/readiness`: pass `--token`, set `KB_ACCESS_TOKEN`, or log in through the UI.

`api_auth` blocking readiness: production mode requires `KB_ACCESS_TOKEN` or `KB_ACCESS_TOKEN_SHA256`.

`frontend_build` blocking readiness: run `cd web; npm run build; cd ..`.

`cors` blocking readiness: replace `KB_API_ALLOW_ORIGINS=*` with explicit app origins.

Login works on localhost but not HTTPS: set `KB_AUTH_COOKIE_SECURE=1` behind HTTPS and make sure the proxy preserves cookies.

Login fails on local HTTP: set `KB_AUTH_COOKIE_SECURE=0` while testing without HTTPS.

Model readiness is missing or failed: configure the key/base URL/model in `.env` or in Settings, then use the Settings connection test before rerunning readiness.
